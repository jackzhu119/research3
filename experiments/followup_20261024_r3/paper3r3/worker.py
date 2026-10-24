"""One independent Round 3 task, with durable fit and session boundaries.

Legacy models are read-only inputs. Dictionary training uses four sequential
support fits and actual nonlinear future-query losses per outer update.
"""
import argparse
from dataclasses import asdict, fields
import fcntl
import math
import os
from pathlib import Path
import platform
import pickle
import random
import signal
import threading
import time
import traceback

import numpy as np
import torch

from .algorithm import Config, Fitted, fit, predict, support_tangent, information, _class_weights
from .baseline import fit_reference, reference_logits
from .data import Catalog, load_source
from .diagnostics import perturb_episode
from .io import artifact, atomic, digest, emit, put, read, sha
from p3work.adapt import balanced_bce, zero_fast_weights
from p3work.spatial_geometry import initialize_dictionary, stiefel_step


class Pause(Exception):
    """A cooperative stop at a complete session or committed meta update."""


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def save_checkpoint(folder, state):
    """Commit an alternating generation, then its hash-verified pointer."""
    folder = Path(folder)
    pointer = folder / "latest.json"
    old = read(pointer) if pointer.exists() else {}
    generation = old.get("generation", 0) + 1
    path = folder / f"state-{generation % 2}.pt"
    payload = cpu_tree(state)
    payload["rng_cpu"] = torch.get_rng_state()
    payload["rng_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload["rng_python"] = random.getstate()
    n = np.random.get_state()
    payload["rng_numpy"] = (n[0], n[1].tolist(), n[2], n[3], n[4])
    atomic(path, lambda temporary: torch.save(payload, temporary))
    put(pointer, {"path": path.name, "generation": generation, "sha256": sha(path),
                  "outer_update": state["update"],
                  "previous": {key: value for key, value in old.items() if key != "previous"}})


def read_checkpoint(folder, *, expected_identity=None):
    """Read a valid generation without changing files or any RNG state."""
    folder = Path(folder)
    pointer = read(folder / "latest.json")
    for entry in (pointer, pointer.get("previous", {})):
        name = entry.get("path", "")
        if not name or Path(name).name != name:
            continue
        path = folder / name
        if not path.is_file() or sha(path) != entry.get("sha256"):
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            continue
        if not isinstance(state, dict):
            continue
        if expected_identity is not None and state.get("identity") != expected_identity:
            raise ValueError("Checkpoint belongs to a different numerical experiment")
        return state, entry, entry is not pointer
    raise IOError("Both checkpoint generations unreadable; original files retained")


def load_checkpoint(folder, *, expected_identity=None):
    state, entry, recovered = read_checkpoint(folder, expected_identity=expected_identity)
    torch.set_rng_state(state["rng_cpu"])
    if torch.cuda.is_available() and state["rng_cuda"]:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    random.setstate(state["rng_python"])
    n = state["rng_numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if recovered:
        # The next alternating save must overwrite the corrupt slot.
        put(Path(folder) / "latest.json", entry)
    return state


def summarize(rows):
    if not rows:
        raise ValueError("An evaluation must contain at least one eligible session")
    people = {}
    for row in rows:
        people.setdefault(row["participant_id"], []).append(row)
    participants = [{"participant_id": person,
        "ba": float(np.mean([row["ba"] for row in sessions])),
        "bce": float(np.mean([row["bce"] for row in sessions]))}
        for person, sessions in sorted(people.items())]
    return {"participant_mean_ba": float(np.mean([row["ba"] for row in participants])),
            "participant_mean_bce": float(np.mean([row["bce"] for row in participants])),
            "participants": participants, "sessions": len(rows)}


def _batched(function, x, batch_size=64):
    return torch.cat([function(x[start:start + batch_size]) for start in range(0, len(x), batch_size)])


def _support_diagnostics(source, geometry, b, support, fitted, config):
    """Support-only audit of the surrogate against the actual exponential model."""
    result = dict(fitted.diagnostics)
    result["theta_norm"] = float(fitted.weights.theta.detach().norm())
    if fitted.fallback_reason or not len(support.y):
        return result
    m0, coordinates, features = support_tangent(source, geometry, support)
    with torch.no_grad():
        actual = _batched(lambda x: predict(source, geometry, b, fitted, x), support.x)
        margin = (actual[:, 1] - actual[:, 0]).double()
        head = torch.cat((fitted.weights.delta_w.double(), fitted.weights.delta_b.double()[None]))
        augmented = torch.cat((features.double(), features.new_ones((len(features), 1)).double()), 1)
        surrogate = m0 + coordinates @ b @ fitted.weights.theta + augmented @ head
        residual = margin - surrogate
        _, effective = information(coordinates @ b, augmented, m0, support.y, config)
        p = m0.sigmoid()
        kl = p * (torch.nn.functional.logsigmoid(m0) - torch.nn.functional.logsigmoid(margin))
        kl += (1 - p) * (torch.nn.functional.logsigmoid(-m0) - torch.nn.functional.logsigmoid(-margin))
        weights = _class_weights(support.y)
        loss = balanced_bce(actual, support.y)
        result.update(actual_support_bce=float(loss),
            actual_support_penalized_loss=float(loss + config.theta_l2 * fitted.weights.theta.square().sum()
                + config.head_l2 * head.square().sum() / len(head)),
            actual_nonlinear_surrogate_margin_rmse=float(residual.square().mean().sqrt()),
            actual_nonlinear_surrogate_margin_max_abs=float(residual.abs().max()),
            effective_dimension=float(effective), actual_source_support_kl=float((weights * kl).sum()),
            protection_fraction=float(fitted.protection_fraction))
    return result


def evaluate(source, geometry, b, catalog, config, folder, stop, progress, *, budget=20,
             scenario="clean", source_identity=None, variant=None, partition=None):
    """Persist completed session rows; adaptation never receives query labels."""
    variant = variant or config.variant
    partition = partition or ("holdout" if catalog.role == "internal_holdout" else catalog.role)
    parameter_id = digest({"B": b.detach().cpu().tolist(), "source": source_identity,
        "data": catalog.identity, "config": asdict(config), "variant": variant,
        "partition": partition, "budget": budget, "scenario": scenario, "evaluation_schema": 3})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_sessions = [{"participant_id": session.participant_id, "session_id": session.session_id,
        "query_signature": digest({"uids": list(session.audit["query_uids"]),
            "labels": [session.rows[uid]["label"] for uid in session.audit["query_uids"]]})}
        for session in catalog.plans]
    for number, session in enumerate(catalog.plans):
        if stop.is_set():
            raise Pause()
        path = folder / f"{session.participant_id}-{session.session_id}.json"
        row = read(path) if path.exists() else None
        if row is not None:
            signature = digest({"uids": row.get("event_uids"), "labels": row.get("labels")})
            expected = expected_sessions[number]
            if (row.get("parameter_id") != parameter_id or row.get("participant_id") != expected["participant_id"]
                    or row.get("session_id") != expected["session_id"]
                    or row.get("query_signature") != expected["query_signature"] or signature != expected["query_signature"]):
                row = None
        if row is None:
            episode = catalog.materialize(session, b.device, budget=budget)
            episode, perturbation = perturb_episode(episode, geometry, session, scenario)
            if b.device.type == "cuda":
                torch.cuda.synchronize(b.device)
            start = time.monotonic()
            reference_metadata = None
            if variant == "support_EA":
                reference_state, reference_metadata = fit_reference(source, geometry, episode.support, variant)
                fallback_reason = reference_metadata["fallback_reason"]
                diagnostics = dict(reference_metadata)
            else:
                try:
                    fitted = fit(source, geometry, b, episode.support, config)
                except (FloatingPointError, torch.linalg.LinAlgError) as error:
                    reason = f"numeric_session_failure:{type(error).__name__}:{error}"
                    fitted = Fitted(zero_fast_weights(source, b.shape[1], device=b.device),
                                    {"numeric_fallback": True}, reason)
                fallback_reason = fitted.fallback_reason
            if b.device.type == "cuda":
                torch.cuda.synchronize(b.device)
            calibration = time.monotonic() - start
            audit_start = time.monotonic()
            if variant != "support_EA":
                diagnostics = _support_diagnostics(source, geometry, b, episode.support, fitted, config)
            if b.device.type == "cuda":
                torch.cuda.synchronize(b.device)
            diagnostics_seconds = time.monotonic() - audit_start
            with torch.no_grad():
                function = (lambda x: reference_logits(source, geometry, reference_state, x)) if variant == "support_EA" else (
                    lambda x: predict(source, geometry, b, fitted, x))
                pred = _batched(function, episode.query_x)
                base_logits = _batched(source.source_logits, episode.query_x)
                if not bool(torch.isfinite(pred).all() and torch.isfinite(base_logits).all()):
                    raise FloatingPointError("Nonfinite query prediction")
                prediction, base = pred.argmax(1), base_logits.argmax(1)
                # Query labels first enter computation after all prediction calls.
                y = episode.query_y
                if not all(bool((y == c).any()) for c in (0, 1)):
                    raise ValueError("Both future query classes are required")
                balanced = lambda mask: .5 * sum(float(mask[y == c].double().mean()) for c in (0, 1))
                run_scores, run_metrics = [], []
                physical_runs = [] if session.audit.get("temporal_risk_eligible") is False else sorted(set(episode.run_ids))
                for rid in physical_runs:
                    mask = torch.tensor([value == rid for value in episode.run_ids], device=y.device)
                    if all(bool((y[mask] == c).any()) for c in (0, 1)):
                        score = .5 * sum(float((prediction[mask & (y == c)] == c).double().mean()) for c in (0, 1))
                        run_scores.append(score)
                        run_metrics.append({"run_id": rid, "n": int(mask.sum()),
                            "class_counts": [int((mask & (y == c)).sum()) for c in (0, 1)], "ba": score})
                row = {"parameter_id": parameter_id, "data_identity": catalog.identity,
                    "source_identity": source_identity, "partition": partition,
                    "participant_id": session.participant_id, "session_id": session.session_id,
                    "budget": budget, "scenario": scenario, "variant": variant,
                    "ba": balanced(prediction == y), "bce": float(balanced_bce(pred, y)),
                    "calibration_seconds": calibration, "diagnostics_seconds": diagnostics_seconds,
                    "fallback": fallback_reason is not None,
                    "fallback_reason": fallback_reason, "diagnostics": diagnostics,
                    "future_runs_with_both_classes": len(run_scores), "per_run_metrics": run_metrics,
                    "run_ba_min": min(run_scores) if run_scores else None,
                    "run_ba_p10": float(np.quantile(run_scores, .1)) if run_scores else None,
                    "temporal_risk_eligible": session.audit.get("temporal_risk_eligible", True),
                    "order_scope": session.audit.get("order_scope"),
                    "negative_flip_rate": balanced((base == y) & (prediction != y)),
                    "corrected_error_rate": balanced((base != y) & (prediction == y)),
                    "f0_ba": balanced(base == y), "event_uids": list(episode.query_ids),
                    "run_ids": list(episode.run_ids), "predictions": prediction.cpu().tolist(),
                    "labels": y.cpu().tolist(), "probabilities": pred.double().softmax(1).cpu().tolist(),
                    "f0_probabilities": base_logits.double().softmax(1).cpu().tolist(),
                    "f0_predictions": base.cpu().tolist(), "perturbation": perturbation,
                    "query_signature": digest({"uids": list(episode.query_ids), "labels": y.cpu().tolist()})}
                if reference_metadata is not None:
                    row["reference_fit"] = reference_metadata
            put(path, row)
        rows.append(row)
        progress.update(stage=partition if scenario == "clean" else "diagnostics", status="EVALUATING",
                        session=number + 1, sessions_total=len(catalog.plans), budget=budget, scenario=scenario)
    if stop.is_set():
        raise Pause()
    result = summarize(rows)
    result.update(expected_sessions=expected_sessions, data_identity=catalog.identity,
        parameter_id=parameter_id, partition=partition, budget=budget, scenario=scenario)
    put(folder / "summary.json", result)
    return result


def outer_update(source, geometry, b, catalog, seed, cursor, config, device):
    gradient = torch.zeros_like(b)
    losses = []
    for index in range(4):
        episode = catalog.materialize(catalog.draw(seed, cursor * 4 + index), device)
        fitted = fit(source, geometry, b, episode.support, config, create_graph=True)
        prediction = _batched(lambda x: predict(source, geometry, b, fitted, x), episode.query_x)
        loss = balanced_bce(prediction, episode.query_y) + b.sum() * 0
        gradient += torch.autograd.grad(loss / 4, b)[0].detach()
        losses.append(float(loss.detach()))
    updated = stiefel_step(b, gradient, learning_rate=.01).detach().requires_grad_(True)
    return updated, {"outer_query_bce": float(np.mean(losses)), "meta_gradient_norm": float(gradient.norm())}


def train_loop(*, state, step, validate, save, stop, total_updates, checkpoint_every=10, validate_every=50):
    """Functional commits allow interrupted validation to resume without replaying updates."""
    if min(total_updates, checkpoint_every, validate_every) < 1:
        raise ValueError("Positive training, checkpoint and validation budgets required")
    while True:
        update = state["update"]
        if stop.is_set():
            save(state)
            raise Pause()
        if update and (update % validate_every == 0 or update == total_updates) and state.get("validated_update", 0) < update:
            save(state)
            score = dict(validate(state), outer_update=update)
            state["history"].append(score)
            key = lambda row: (-row["participant_mean_ba"], row["participant_mean_bce"], row["outer_update"])
            if state.get("best_score") is None or key(score) < key(state["best_score"]):
                state.update(best_score=score, best_b=cpu_tree(state["b"]))
            state["validated_update"] = update
            save(state)
        if update >= total_updates:
            return state
        b, metrics = step(state)
        state.update(b=b, metrics=metrics, update=update + 1)
        if state["update"] % checkpoint_every == 0:
            save(state)


def _legacy_matches(config):
    return config.variant == "sgd10" and all(getattr(config, key) == value for key, value in {
        "theta_lr": .05, "head_lr": .01, "theta_l2": .01, "head_l2": .001}.items())


def _valid_dictionary(b, rank):
    return (isinstance(b, torch.Tensor) and b.shape == (99, rank) and b.dtype == torch.float64
        and bool(torch.isfinite(b).all()) and torch.allclose(b.T @ b,
            torch.eye(rank, dtype=b.dtype, device=b.device), atol=1e-10, rtol=1e-10))


def _valid_selection(score):
    return (isinstance(score, dict) and all(isinstance(score.get(key), (float, int))
        and math.isfinite(score[key]) for key in ("participant_mean_ba", "participant_mean_bce"))
        and 0 <= score["participant_mean_ba"] <= 1 and score["participant_mean_bce"] >= 0
        and isinstance(score.get("outer_update"), int) and 0 < score["outer_update"] <= 1000
        and score["outer_update"] % 50 == 0)


def try_reuse(root, config, seed, inputs, catalog_identity):
    """Reuse only completed, numerically matching original SGD dictionaries."""
    if not _legacy_matches(config):
        return None
    root = Path(root)
    old_output = root / "runs/followup_20261023_v2"
    suffix = "" if (config.rank, config.radius) == (4, .15) else f"-r{config.rank}-rho{str(config.radius).replace('.', 'p')}"
    folder = old_output / f"P{suffix}-seed{seed}"
    try:
        result, selected, old_plan = read(folder / "result.json"), read(folder / "selected.json"), read(old_output / "plan.json")
        old_task = selected["task"]
        old_config = old_task["config"]
        expected = digest({"task": old_task, "data": catalog_identity, "source": inputs,
                           "recipe": old_plan["recipe_sha256"]})
        if (result.get("status") == "COMPLETE" and result.get("outer_updates") == 1000
                and selected.get("status") == "SELECTED_ON_VALIDATION"
                and selected.get("holdout_used_for_selection") is False
                and old_task in old_plan["tasks"] and old_task["seed"] == seed
                and old_config.get("variant") == "P" and old_config.get("rank", 4) == config.rank
                and old_config.get("radius", .15) == config.radius
                and all(old_config.get(key, value) == value for key, value in {
                    "inner_steps": 10, "theta_lr": .05, "head_lr": .01, "theta_l2": .01, "head_l2": .001}.items())
                and result.get("identity") == selected.get("identity") == expected):
            state, _, _ = read_checkpoint(folder, expected_identity=expected)
            b = state["best_b"]
            if (state.get("update") == state.get("validated_update") == 1000
                    and selected.get("selection") == state.get("best_score")
                    and _valid_selection(state.get("best_score")) and _valid_dictionary(b, config.rank)):
                return b, state["best_score"], str(folder / "result.json")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass
    for receipt_path in sorted((root / "runs/meta").glob("*/receipt.json")):
        try:
            receipt = read(receipt_path)
            job = receipt.get("job", {})
            if (receipt.get("status") != "META_JOB_COMPLETE" or job.get("method") != "P"
                    or job.get("seed") != seed or job.get("dimension") != config.rank
                    or job.get("radius") != config.radius or receipt.get("outer_updates_completed") != 1000
                    or any(receipt.get(key) != inputs[key] for key in (
                        "source_receipt_sha256", "reference_receipt_sha256", "signal_contract_sha256"))
                    or receipt.get("stieger_manifest_sha256") != catalog_identity["manifest_sha256"]
                    or receipt.get("split_sha256") != catalog_identity["split_sha256"]):
                continue
            checkpoint = torch.load(artifact(root, receipt["selected_parameter_checkpoint"]), map_location="cpu", weights_only=True)
            b = checkpoint["state"]["parameter"]
            if _valid_dictionary(b, config.rank) and _valid_selection(receipt.get("selected_checkpoint")):
                return b, receipt["selected_checkpoint"], str(receipt_path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            continue
    return None


def _identity(plan, task, source_identity, data_identity, dependency_identity=None):
    training = None
    if task.get("family") == "basis":
        training = {"updates": task.get("updates", task.get("config", {}).get("updates",
            task.get("config", {}).get("outer_updates", plan.get("updates", 1000)))),
            "validate_every": task.get("validate_every", plan.get("validate_every", 50))}
    return digest({"task": task, "data": data_identity, "source": source_identity,
        "recipe": plan["recipe_sha256"], "basis": dependency_identity, "training": training})


def load_basis(plan, task, device, *, source_identity=None, data_identity=None):
    """Load a completed dependency's selected B without restoring its RNG."""
    key = task.get("basis_key")
    if not key or Path(key).name != key:
        raise ValueError("Evaluation requires a basis_key in the current run")
    basis_output = Path(task.get("basis_output", plan["output"]))
    basis_plan = read(basis_output / "plan.json") if task.get("basis_output") else plan
    if Path(basis_plan["output"]).resolve() != basis_output.resolve():
        raise ValueError("Dictionary source plan belongs to a different output directory")
    folder = basis_output / key
    result = read(folder / "result.json")
    if result.get("status") != "COMPLETE" or result.get("task", {}).get("family") != "basis":
        raise ValueError("Dictionary dependency is not complete")
    declared = next((candidate for candidate in basis_plan["tasks"] if candidate["key"] == key), None)
    if declared is None or result["task"] != declared:
        raise ValueError("Dictionary dependency belongs to a different task plan")
    if source_identity is not None:
        expected_data = data_identity if data_identity is not None else result.get("data_identity")
        expected = _identity(basis_plan, declared, source_identity, expected_data)
        if (result.get("source_identity") != source_identity or result.get("data_identity") != expected_data
                or result.get("identity") != expected):
            raise ValueError("Dictionary dependency belongs to different source, data, or recipe")
    locator = result["selected_checkpoint"]
    path = Path(locator["path"])
    if not path.is_absolute():
        path = basis_output / path
    path = path.resolve()
    if not path.is_relative_to(folder.resolve()) or sha(path) != locator["sha256"]:
        raise ValueError("Selected dictionary locator or content differs")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("identity") != result["identity"]:
        raise ValueError("Selected dictionary and dependency identities differ")
    if result["task"]["seed"] != task["seed"]:
        raise ValueError("Selected dictionary belongs to a different pipeline seed")
    b = checkpoint["b"].to(device)
    return b, {"key": key, "identity": result["identity"], "sha256": locator["sha256"],
               "basis_output": str(basis_output.resolve())}


def _configuration(task):
    values = dict(task.get("config", {}))
    variant = values.setdefault("variant", task.get("variant", "tangent_profiled"))
    for key in ("updates", "outer_updates", "validate_every", "checkpoint_every"):
        values.pop(key, None)
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError("Unknown algorithm configuration fields: " + ", ".join(sorted(unknown)))
    if variant == "support_EA":
        values["variant"] = "head_newton"
    return Config(**values), variant


def run(args):
    plan = read(args.plan)
    task = plan["tasks"][args.task]
    key = task["key"]
    if Path(key).name != key or key in (".", ".."):
        raise ValueError("Task key must name a single task directory")
    output = Path(plan["output"]) / key
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        emit("ALREADY_RUNNING", task=key)
        return 0
    stop, done = threading.Event(), threading.Event()
    if os.environ.get("PAPER3R3_STOP_REQUESTED") == "1":
        stop.set()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    progress = {"status": "STARTING", "task": key, "stage": "loading", "outer_update": 0}
    def heartbeat():
        while not done.is_set():
            snapshot = {**progress, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            emit("TRAIN_PROGRESS", **snapshot)
            try:
                put(output / "progress.json", snapshot)
            except OSError as error:
                emit("PROGRESS_WRITE_RETRY", reason=str(error))
            done.wait(30)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    state = None
    started = time.monotonic()
    prior_elapsed = 0.
    def save(value):
        value["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        save_checkpoint(output, value)
    try:
        if stop.is_set():
            raise Pause()
        config, variant = _configuration(task)
        seed = task["seed"]
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA device unavailable")
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        source, geometry, inputs = load_source(plan["root"], seed, device)
        source.eval()
        family = task.get("family", "core")
        if family not in ("basis", "core", "sensitivity", "reference", "transfer"):
            raise ValueError("Unknown task family: " + str(family))
        for dependency in task.get("requires", []):
            if read(Path(plan["output"]) / dependency / "result.json").get("status") != "COMPLETE":
                raise ValueError("Required task is incomplete: " + dependency)
        if family == "basis":
            if variant not in ("sgd10", "tangent_profiled", "tangent_information", "information_protected"):
                raise ValueError("Unsupported independently trained dictionary learner")
            catalog = Catalog(plan["root"], "meta_train")
            validation = Catalog(plan["root"], "meta_validation")
            if catalog.identity != validation.identity:
                raise ValueError("Training and validation data identities differ")
            dependency_identity = None
        else:
            if family == "transfer":
                from .external import ExternalCatalog
                manifest = task.get("external_manifest", plan.get("external_manifest"))
                if not manifest:
                    raise ValueError("Transfer task requires an explicit external_manifest")
                catalog = ExternalCatalog(plan["root"], manifest)
            else:
                catalog = Catalog(plan["root"], "internal_holdout")
            if task.get("basis_key"):
                basis_data = Catalog(plan["root"], "meta_train").identity if family == "transfer" else catalog.identity
                b, dependency_identity = load_basis(plan, task, device, source_identity=inputs,
                    data_identity=basis_data)
            elif variant == "support_EA" or variant in ("head_sgd", "head_newton"):
                b = initialize_dictionary(config.rank, seed, device=device)
                dependency_identity = None
            else:
                raise ValueError("Evaluation task is missing its trained dictionary dependency")
        if inputs["signal_contract_sha256"] != catalog.signal_sha:
            raise ValueError("Source and evaluation signal contracts differ")
        identity = _identity(plan, task, inputs, catalog.identity, dependency_identity)
        completed = read(output / "result.json") if (output / "result.json").exists() else None
        if completed and completed.get("status") == "COMPLETE":
            if completed.get("identity") != identity:
                raise ValueError("Completed task belongs to a different numerical experiment")
            progress.update(status="COMPLETE", stage="done", outer_update=completed.get("outer_updates", 0))
            emit("COMPLETED_TASK_REUSED", task=key)
            return 0
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        put(output / "runtime.json", runtime)
        if family == "basis":
            total = int(task.get("updates", task.get("config", {}).get("updates", task.get("config", {}).get("outer_updates", plan.get("updates", 1000)))))
            if (output / "latest.json").exists():
                state = load_checkpoint(output, expected_identity=identity)
                state["b"] = state["b"].to(device).requires_grad_(True)
                emit("CHECKPOINT_RESUMED", task=key, update=state["update"])
            else:
                state = {"identity": identity, "update": 0, "validated_update": 0, "history": [],
                    "b": initialize_dictionary(config.rank, seed, device=device).requires_grad_(True),
                    "elapsed_seconds": 0.}
                reused = try_reuse(plan["root"], config, seed, inputs, catalog.identity) if total == 1000 else None
                if reused is not None:
                    previous_b, score, receipt = reused
                    geometry.validate_dictionary(previous_b.to(device))
                    state.update(b=previous_b.to(device).requires_grad_(True), best_b=cpu_tree(previous_b),
                        best_score=score, update=1000, validated_update=1000, reused_receipt=receipt)
                    emit("COMPLETED_BASELINE_REUSED", task=key, receipt=receipt)
                save_checkpoint(output, state)
            prior_elapsed = state.get("elapsed_seconds", 0.)
            started = time.monotonic()
            if state.get("runtime") is not None and state["runtime"] != runtime:
                emit("RESUME_RUNTIME_CHANGED", task=key, bitwise_equivalence_claimed=False)
            state["runtime"] = runtime
            def step(value):
                progress.update(stage="meta_train", status="TRAINING", outer_update=value["update"], total_updates=total)
                answer = outer_update(source, geometry, value["b"], catalog, seed, value["update"], config, device)
                progress.update(outer_update=value["update"] + 1, **answer[1])
                return answer
            def validate(value):
                return evaluate(source, geometry, value["b"], validation, config,
                    output / f"validation/{value['update']:04d}", stop, progress, source_identity=inputs)
            train_loop(state=state, step=step, validate=validate, save=save, stop=stop,
                total_updates=total, checkpoint_every=int(task.get("checkpoint_every", plan.get("checkpoint_every", 10))),
                validate_every=int(task.get("validate_every", plan.get("validate_every", 50))))
            b = state["best_b"].to(device)
            geometry.validate_dictionary(b)
            path = output / "selected-basis.pt"
            atomic(path, lambda temporary: torch.save({"identity": identity, "b": cpu_tree(b)}, temporary))
            locator = {"path": str(path.resolve()), "sha256": sha(path)}
            selected = {"status": "SELECTED_ON_VALIDATION", "identity": identity, "task": task,
                "selection": state["best_score"], "selected_checkpoint": locator,
                "holdout_used_for_selection": False}
            put(output / "selected.json", selected)
            evaluate(source, geometry, b, validation, config, output / "evaluations/meta_validation/B20/clean",
                     stop, progress, source_identity=inputs)
            save(state)
            result = {**selected, "status": "COMPLETE", "outer_updates": state["update"],
                "reused_receipt": state.get("reused_receipt"), "elapsed_seconds": state["elapsed_seconds"]}
        else:
            if b.shape != (99, config.rank):
                raise ValueError("Selected dictionary rank differs from evaluation configuration")
            geometry.validate_dictionary(b)
            partition = task.get("partition", "holdout")
            selected = {"status": "FIXED_REFERENCE_NO_MODEL_SELECTION" if variant == "support_EA" else "FIXED_BASIS_EVALUATION",
                "identity": identity, "task": task, "selection": None, "basis": dependency_identity,
                "holdout_used_for_selection": False}
            put(output / "selected.json", selected)
            budgets = task.get("budgets", plan.get("budgets", [20, 4, 8, 12, 16]))
            scenarios = task.get("scenarios", ["clean"])
            for budget in budgets:
                for scenario in scenarios:
                    if scenario != "clean" and budget != 20:
                        continue
                    evaluate(source, geometry, b, catalog, config,
                        output / f"evaluations/{partition}/B{budget}/{scenario}", stop, progress,
                        budget=budget, scenario=scenario, source_identity=inputs, variant=variant, partition=partition)
            result = {**selected, "status": "COMPLETE", "outer_updates": 0,
                      "elapsed_seconds": time.monotonic() - started}
        result.update(source_identity=inputs, data_identity=catalog.identity,
                      internal_check_not_external_validation=family != "transfer")
        put(output / "result.json", result)
        (output / "error.json").unlink(missing_ok=True)
        progress.update(status="COMPLETE", stage="done", outer_update=result["outer_updates"])
        emit("TASK_COMPLETE", task=key)
        return 0
    except Pause:
        if state is not None:
            save(state)
        progress.update(status="PAUSED_CHECKPOINT_SAVED", stage=progress.get("stage", "loading"))
        emit("PAUSED_CHECKPOINT_SAVED", task=key, update=state["update"] if state else 0)
        return 75
    except Exception as error:
        if state is not None:
            try:
                save(state)
            except Exception as checkpoint_error:
                emit("CHECKPOINT_WRITE_FAILED", task=key, reason=str(checkpoint_error))
        failure = {"status": "FAILED", "task": key, "error_type": type(error).__name__,
                   "reason": str(error), "traceback": traceback.format_exc()}
        print(failure["traceback"], flush=True)
        try:
            put(output / "error.json", failure)
        except OSError:
            pass
        progress.update(status="FAILED", reason=str(error))
        return 1
    finally:
        done.set()
        thread.join(timeout=2)
        try:
            put(output / "progress.json", {**progress,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
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
