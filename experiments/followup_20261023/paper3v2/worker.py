"""One independent fit per Slurm array task. All progress lives on shared storage."""
import argparse
from dataclasses import asdict
import fcntl
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
import traceback
import numpy as np
import torch
from .algorithm import Config, init_gate, fit, logits, outer_update
from .data import Catalog, load_source
from .io import read, put, atomic, digest, sha, artifact, emit
from p3work.adapt import balanced_bce
from p3work.spatial_geometry import initialize_dictionary

class Pause(Exception):
    pass

def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return x

def save_checkpoint(folder, state):
    """Two generations, pointer written last; a torn new file leaves the old one."""
    pointer = Path(folder) / "latest.json"
    old = read(pointer) if pointer.exists() else None
    generation = (old.get("generation", 0) if old else 0) + 1
    filename = f"state-{generation % 2}.pt"
    path = Path(folder) / filename
    payload = cpu_tree(state)
    payload["rng_cpu"] = torch.get_rng_state()
    payload["rng_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload["rng_python"] = random.getstate()
    n = np.random.get_state()
    payload["rng_numpy"] = (n[0], n[1].tolist(), n[2], n[3], n[4])
    atomic(path, lambda temporary: torch.save(payload, temporary))
    put(pointer, {"path": filename, "generation": generation, "sha256": sha(path),
                  "outer_update": state["update"], "previous": {k: v for k, v in (old or {}).items() if k != "previous"}})

def load_checkpoint(folder):
    pointer = read(Path(folder) / "latest.json")
    for entry in (pointer, pointer.get("previous", {})):
        path = Path(folder) / entry.get("path", "missing")
        if path.is_file() and sha(path) == entry.get("sha256"):
            state = torch.load(path, map_location="cpu", weights_only=True)
            torch.set_rng_state(state["rng_cpu"])
            if torch.cuda.is_available() and state["rng_cuda"]:
                torch.cuda.set_rng_state_all(state["rng_cuda"])
            random.setstate(state["rng_python"])
            n = state["rng_numpy"]
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
            if entry is not pointer:
                # Keep the recovered generation intact while the next save overwrites
                # the corrupt slot, rather than overwriting our only good checkpoint.
                put(Path(folder) / "latest.json", entry)
            return state
    raise IOError("Both checkpoint generations unreadable; original files retained")

def summarize(rows):
    people = {}
    for row in rows:
        people.setdefault(row["participant_id"], []).append(row)
    p = [{"participant_id": person,
          "ba": float(np.mean([r["ba"] for r in sessions])),
          "bce": float(np.mean([r["bce"] for r in sessions]))} for person, sessions in sorted(people.items())]
    return {"participant_mean_ba": float(np.mean([r["ba"] for r in p])),
            "participant_mean_bce": float(np.mean([r["bce"] for r in p])), "participants": p,
            "sessions": len(rows)}

def evaluate(source, geometry, b, gate, catalog, config, folder, stop, progress, *, budget=20):
    rows = []
    parameter_id = digest({"B": b.detach().cpu().tolist(), "gate": gate.detach().cpu().tolist(),
                           "data": catalog.identity, "config": asdict(config), "budget": budget})
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for number, plan in enumerate(catalog.plans):
        if stop.is_set():
            raise Pause()
        path = folder / f"{plan.participant_id}-{plan.session_id}.json"
        row = read(path) if path.exists() else None
        if row is not None and row.get("parameter_id") != parameter_id:
            row = None
        if row is None:
            episode = catalog.materialize(plan, b.device, budget=budget)
            start = time.monotonic()
            weights, alpha, fallback = fit(source, geometry, b, gate, episode.support, config)
            if b.device.type == "cuda":
                torch.cuda.synchronize()
            calibration = time.monotonic() - start
            with torch.no_grad():
                pred = source.source_logits(episode.query_x) if fallback else logits(source, geometry, b, weights, alpha, episode.query_x)
                prediction = pred.argmax(1)
                # Prediction is computed before labels enter the metric calculation.
                y = episode.query_y
                ba = .5 * sum(float((prediction[y == c] == c).double().mean()) for c in (0, 1))
                base = source.source_logits(episode.query_x).argmax(1)
                run_ba = []
                for rid in sorted(set(episode.run_ids)):
                    mask = torch.tensor([r == rid for r in episode.run_ids], device=y.device)
                    if all(bool((y[mask] == c).any()) for c in (0, 1)):
                        run_ba.append(.5 * sum(float((prediction[mask & (y == c)] == c).double().mean()) for c in (0, 1)))
                flips = lambda mask: .5 * sum(float(mask[y == c].double().mean()) for c in (0, 1))
                row = {"parameter_id": parameter_id, "participant_id": plan.participant_id,
                    "session_id": plan.session_id, "budget": budget, "ba": ba,
                    "bce": float(balanced_bce(pred, y)), "gate": alpha.detach().cpu().tolist(),
                    "calibration_seconds": calibration, "fallback": fallback,
                    "future_runs_with_both_classes": len(run_ba),
                    "run_ba_p10": float(np.quantile(run_ba, .1)) if run_ba else None,
                    "negative_flip_rate": flips((base == y) & (prediction != y)),
                    "corrected_error_rate": flips((base != y) & (prediction == y)),
                    "f0_ba": flips(base == y),
                    "event_uids": list(episode.query_ids), "run_ids": list(episode.run_ids),
                    "predictions": prediction.cpu().tolist(), "labels": y.cpu().tolist(),
                    "f0_predictions": base.cpu().tolist()}
            put(path, row)
        rows.append(row)
        progress.update(stage=catalog.role, session=number + 1, sessions_total=len(catalog.plans), budget=budget)
    result = summarize(rows)
    put(folder / "summary.json", result)
    return result

def try_reuse(root, config, seed, inputs, catalog_identity):
    if config.variant not in ("P", "PCT"):
        return None
    method = "P+PCT" if config.variant == "PCT" else "P"
    for path in sorted((Path(root) / "runs/meta").glob("*/receipt.json")):
        try:
            receipt = read(path)
            job = receipt.get("job", {})
            if (receipt.get("status") != "META_JOB_COMPLETE" or job.get("method") != method
                or job.get("seed") != seed or job.get("dimension") != config.rank
                or job.get("radius") != config.radius or receipt.get("outer_updates_completed") != 1000
                or any(receipt.get(k) != inputs[k] for k in ("source_receipt_sha256", "reference_receipt_sha256", "signal_contract_sha256"))
                or receipt.get("stieger_manifest_sha256") != catalog_identity["manifest_sha256"]
                or receipt.get("split_sha256") != catalog_identity["split_sha256"]):
                continue
            selected = receipt["selected_parameter_checkpoint"]
            cp = torch.load(artifact(root, selected), map_location="cpu", weights_only=True)
            b = cp["state"]["parameter"]
            if b.shape != (99, config.rank) or not torch.isfinite(b).all():
                continue
            emit("COMPLETED_BASELINE_REUSED", receipt=str(path), checkpoint=selected["path"])
            return b, receipt["selected_checkpoint"], str(path)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    return None

def train_loop(*, state, step, validate, save, stop, total_updates, checkpoint_every=10):
    """Pure orchestration seam used for exact-resume and interrupted-validation tests."""
    while True:
        update = state["update"]
        if stop.is_set():
            save(state); raise Pause()
        if update and update % 50 == 0 and state.get("validated_update", 0) < update:
            save(state)  # do not lose optimization if validation is interrupted
            score = validate(state)
            score["outer_update"] = update
            state["history"].append(score)
            key = lambda s: (-s["participant_mean_ba"], s["participant_mean_bce"], s["outer_update"])
            if state.get("best_score") is None or key(score) < key(state["best_score"]):
                state["best_score"] = score
                state["best_b"], state["best_gate"] = cpu_tree(state["b"]), cpu_tree(state["gate"])
            state["validated_update"] = update
            save(state)
        if update >= total_updates:
            return state
        # step is functional: an exception cannot partially mutate the committed state.
        b, gate, metrics = step(state)
        state.update(b=b, gate=gate, update=update + 1, metrics=metrics)
        if state["update"] % checkpoint_every == 0:
            save(state)

def run(args):
    emit("V2_PROCESS_STARTED", job=os.environ.get("SLURM_JOB_ID"), task=args.task, node=os.uname().nodename)
    plan = read(args.plan); task = plan["tasks"][args.task]
    config = Config(**task["config"]); seed = task["seed"]
    output = Path(plan["output"]) / task["key"]; output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        emit("ALREADY_RUNNING", task=task["key"]); return 0
    if (output / "result.json").exists() and read(output / "result.json").get("status") == "COMPLETE":
        emit("COMPLETED_TASK_REUSED", task=task["key"]); return 0
    stop, done = threading.Event(), threading.Event()
    if os.environ.get("PAPER3V2_STOP_REQUESTED") == "1":
        stop.set()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    progress = {"status": "STARTING", "task": task["key"], "outer_update": 0, "stage": "loading"}
    def heartbeat():
        while not done.is_set():
            snapshot = {**progress, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            emit("TRAIN_PROGRESS", **snapshot)
            try:
                put(output / "progress.json", snapshot)
            except OSError as e:
                emit("PROGRESS_WRITE_RETRY", error=str(e))
            done.wait(30)
    threading.Thread(target=heartbeat, daemon=True).start()
    state = None
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA device unavailable")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        source, geometry, inputs = load_source(plan["root"], seed, device)
        train = Catalog(plan["root"], "meta_train")
        validation = Catalog(plan["root"], "meta_validation")
        if inputs["signal_contract_sha256"] != train.signal_sha or train.identity != validation.identity:
            raise ValueError("Source/cache/partition identity mismatch")
        scientific_id = digest({"task": task, "data": train.identity, "source": inputs,
                                "recipe": plan["recipe_sha256"]})
        if (output / "latest.json").exists():
            state = load_checkpoint(output)
            if state["identity"] != scientific_id:
                raise ValueError("Model/data configuration changed; use a new experiment output")
            state["b"] = state["b"].to(device).requires_grad_(True)
            state["gate"] = state["gate"].to(device).requires_grad_(True)
            emit("CHECKPOINT_RESUMED", update=state["update"])
        else:
            state = {"identity": scientific_id, "update": 0, "validated_update": 0, "history": [],
                "b": initialize_dictionary(config.rank, seed, device=device).requires_grad_(True),
                "gate": init_gate(config, device), "elapsed_seconds": 0.}
            reused = try_reuse(plan["root"], config, seed, inputs, train.identity)
            if reused:
                b, score, receipt = reused
                state.update(b=b.to(device).requires_grad_(True), update=1000, validated_update=1000,
                    best_b=b, best_gate=cpu_tree(state["gate"]), best_score=score, reused_receipt=receipt)
            save_checkpoint(output, state)
        start = time.monotonic(); prior_elapsed = state.get("elapsed_seconds", 0.)
        def save(s):
            s["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
            save_checkpoint(output, s)
        def step(s):
            progress.update(stage="meta_train", status="TRAINING", outer_update=s["update"], total_updates=1000)
            episodes = [train.materialize(train.draw(seed, s["update"] * 4 + k), device) for k in range(4)]
            answer = outer_update(source, geometry, s["b"], s["gate"], episodes, config)
            progress.update(outer_update=s["update"] + 1, **answer[2])
            return answer
        def validate(s):
            return evaluate(source, geometry, s["b"], s["gate"], validation, config,
                            output / f"validation/{s['update']:04d}", stop, progress)
        train_loop(state=state, step=step, validate=validate, save=save, stop=stop, total_updates=1000)
        # Configuration and chosen checkpoint are saved BEFORE opening the holdout catalog.
        selected = {"status": "SELECTED_ON_VALIDATION", "identity": scientific_id,
                    "task": task, "selection": state["best_score"], "holdout_used_for_selection": False}
        put(output / "selected.json", selected)
        best_b = state["best_b"].to(device).requires_grad_(True)
        best_gate = state["best_gate"].to(device).requires_grad_(True)
        holdout = Catalog(plan["root"], "internal_holdout")
        for budget in plan["budgets"]:
            score = evaluate(source, geometry, best_b, best_gate, holdout, config,
                             output / f"holdout/B{budget}", stop, progress, budget=budget)
            emit("HOLDOUT_RESULT", task=task["key"], budget=budget, **score)
        save(state)
        result = {**selected, "status": "COMPLETE", "elapsed_seconds": state["elapsed_seconds"],
                  "outer_updates": state["update"], "reused_receipt": state.get("reused_receipt"),
                  "internal_check_not_external_validation": True}
        put(output / "result.json", result)
        progress.update(status="COMPLETE", stage="done")
        emit("TASK_COMPLETE", task=task["key"])
        return 0
    except Pause:
        if state is not None:
            save_checkpoint(output, state)
        progress.update(status="PAUSED_CHECKPOINT_SAVED")
        emit("PAUSED_CHECKPOINT_SAVED", update=state["update"] if state else 0)
        return 75
    except Exception as e:
        # Save the last functional, committed update. Failed step is replayed once on restart.
        if state is not None:
            try:
                save_checkpoint(output, state)
            except Exception as checkpoint_error:
                emit("CHECKPOINT_WRITE_FAILED", reason=str(checkpoint_error))
        failure = {"status": "FAILED", "task": task["key"], "error_type": type(e).__name__,
                   "reason": str(e), "traceback": traceback.format_exc()}
        print(failure["traceback"], flush=True)
        try:
            put(output / "error.json", failure)
        except OSError:
            pass
        progress.update(status="FAILED", reason=str(e)); return 1
    finally:
        done.set()
        try:
            put(output / "progress.json", {**progress, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        except OSError:
            pass
        lock.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--task", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
