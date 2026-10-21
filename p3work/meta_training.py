"""Bounded, resumable development meta training. No implicit execution.

Only Stieger meta_train supplies gradients; meta_validation supplies the fixed
50-update checkpoint selection. Internal/external datasets are not opened here.
The execution path needs frozen real data, a verified real source model, measured
engineering receipts, CUDA, and an explicit user-started-GPU attestation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch

from .adapt import InnerConfig, adapted_logits, balanced_bce, inner_unroll
from .contracts import digest, file_digest, validate_signal_contract
from .f0_bridge import load_verified_source
from .methods.meta_train import outer_step
from .selection import choose_checkpoint, participant_equal_metrics
from .recovery_budget import PersistentWallBudget
from .spatial_geometry import SpatialGeometry, initialize_dictionary
from .training_state import atomic_json, capture_rng, load_checkpoint, local_path, restore_rng, save_checkpoint

META_CODE_FILES = ("p3work/meta_training.py", "p3work/meta_artifacts.py", "p3work/training_state.py",
    "p3work/recovery_budget.py", "p3work/meta_engineering.py",
    "p3work/selection.py", "p3work/episodes.py", "p3work/contracts.py", "p3work/splits.py",
    "p3work/adapt.py", "p3work/spatial_geometry.py", "p3work/methods/meta_train.py",
    "p3work/methods/lowrank.py", "p3work/source_reference_io.py", "p3work/f0_bridge.py", "tools/train_meta.py",
    "p3work/window_acceptance.py", "p3work/source_cache_acceptance.py",
    "p3work/source_training.py", "p3work/runtime_clock.py", "tools/run_controlled_pilot.py")

META_RUNTIME_KEYS = ("python", "torch", "cuda", "cudnn", "driver_versions", "device_name",
    "device_uuid", "device_total_memory_bytes", "device_compute_capability", "autocast", "tf32", "float32_matmul_precision",
    "deterministic_algorithms", "cublas_workspace_config", "code")


@dataclass(frozen=True)
class MetaJob:
    method: str
    dimension: int
    radius: float
    seed: int = 20261018
    phase: str = "screening"

    def __post_init__(self):
        if self.method not in {"P", "P+PCT", "B8", "B10", "B11"}:
            raise ValueError("An independently meta-trained registered method required")
        grid = {1, 2, 4} if self.method in {"B10", "B11"} else {2, 4, 8}
        if self.dimension not in grid or self.radius not in {.05, .15, .35}:
            raise ValueError("Registered nine-configuration grid required")
        if self.phase not in {"screening", "confirmation", "engineering"}:
            raise ValueError("Unknown meta phase")
        if self.phase in {"screening", "engineering"} and self.seed != 20261018:
            raise ValueError("The predeclared screening/engineering seed is 20261018")
        if self.phase == "confirmation" and self.seed not in {20261019, 20261020, 20261021, 20261022}:
            raise ValueError("Do not refit the already completed selected screening seed")

    @property
    def updates(self):
        return 100 if self.phase == "engineering" else 1000

    @property
    def identity(self):
        return digest(asdict(self))


def search_plan(method="P"):
    dimensions = (1, 2, 4) if method in {"B10", "B11"} else (2, 4, 8)
    jobs = [MetaJob(method, d, r) for d in dimensions for r in (.05, .15, .35)]
    return {"schema_version": "paper3-meta-search-v1", "method": method, "execute": False,
            "engineering_profile": {"method": method, "dimension": max(dimensions), "radius": .35,
                                    "seed": 20261018, "outer_updates": 100, "scientific_fit": False},
            "jobs": [{**asdict(j), "job_sha256": j.identity, "outer_updates": 1000,
                      "episodes_per_outer": 4, "inner_updates_per_episode": 10} for j in jobs],
            "screening_jobs": 9, "confirmation_additional_seeds": [20261019, 20261020],
            "final_extension_additional_seeds": [20261021, 20261022],
            "selected_screening_seed_reused": True, "full_five_seed_max_jobs": 13,
            "full_five_seed_outer_updates": 13000, "full_five_seed_inner_updates": 520000,
            "training_started": False, "external_access": False}


def _snapshot(path, identity, parameter, cursor, sampler, history, updates, wall, *, cuda,
              observed_updates, attempts, replayed):
    state = {"parameter": parameter.detach().cpu(), "outer_cursor": cursor,
             "sampler": sampler.state_dict(), "validation": history, "update_receipts": updates,
             "active_wall_seconds": wall, "confirmed_outer_updates_observed": observed_updates,
             "outer_attempts_dispatched": attempts, "replayed_observed_outer_updates": replayed}
    sha = save_checkpoint(path, identity=identity, state=state, cuda=cuda)
    return {"path": path.name, "relative_to_job_output": True, "sha256": sha, "outer_cursor": cursor}


def run_outer_loop(*, parameter, sampler, step, evaluate, identity, output,
                   total_updates=1000, max_wall_seconds, resume=None, cuda=False, monotonic=time.monotonic,
                   wall_amendment_note=None, utc_now=None, boot_id=None):
    """State-machine core; fixtures use synthetic callbacks, never real EEG fits.

    The actual runner alone authorizes data/GPU and writes scientific receipts.
    Every completed update and snapshot is counted; paused runs remain paused.
    """
    if total_updates not in {100, 1000} or type(max_wall_seconds) not in {int, float} or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Fixed registered update count and finite positive wall budget required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = parameter.device
    cursor, history, updates, elapsed_before = 0, [], [], 0.0
    observed_updates = attempts = replayed = 0
    if resume is not None:
        state = load_checkpoint(resume, identity=identity, cuda=cuda)
        cursor, history, updates = state["outer_cursor"], state["validation"], state["update_receipts"]
        elapsed_before = state["active_wall_seconds"]
        observed_updates = state.get("confirmed_outer_updates_observed", cursor)
        attempts = state.get("outer_attempts_dispatched", observed_updates)
        replayed = state.get("replayed_observed_outer_updates", 0)
        if type(cursor) is not int or not 0 <= cursor <= total_updates or len(updates) != cursor:
            raise ValueError("Invalid persisted meta cursor/update history")
        sampler.load_state_dict(state["sampler"])
        parameter = state["parameter"].to(device).detach().requires_grad_(True)
        prior_progress_path = output / "progress.json"
        if prior_progress_path.exists():
            prior = json.loads(prior_progress_path.read_text())
            if prior.get("identity_sha256") != digest(identity):
                raise ValueError("Durable meta progress belongs to a different job identity")
            # SIGKILL can leave observed updates after the latest atomic model
            # snapshot. Their compute/time is retained when those steps replay.
            elapsed_before = max(elapsed_before, prior.get("active_wall_seconds", 0))
            observed_updates = max(observed_updates, prior.get("confirmed_outer_updates_observed", 0))
            attempts = max(attempts, prior.get("outer_attempts_dispatched", 0))
            replayed = max(replayed, prior.get("replayed_observed_outer_updates", 0)) + max(0, prior.get("outer_updates_completed", 0) - cursor)
    budget_arguments = {"monotonic": monotonic, "amendment_note": wall_amendment_note}
    if utc_now is not None:
        budget_arguments["utc_now"] = utc_now
    if boot_id is not None:
        budget_arguments["boot_id"] = boot_id
    wall_budget = PersistentWallBudget(output / "wall_budget.json", digest(identity), max_wall_seconds,
                                       checkpoint_wall_seconds=elapsed_before, **budget_arguments)
    progress_path = output / "progress.json"
    latest = None
    step_in_progress = False
    sampler_before = rng_before = None
    parameter_before = None
    cursor_before = ledger_before = 0

    def snapshot():
        nonlocal latest
        path = output / f"outer-{cursor:04d}.pt"
        latest = _snapshot(path, identity, parameter, cursor, sampler, history, updates,
                           wall_budget.wall_seconds, cuda=cuda,
                           observed_updates=observed_updates, attempts=attempts, replayed=replayed)
        atomic_json(output / "latest_checkpoint.json", latest)
        return latest

    def progress(status):
        row = {"status": status, "identity_sha256": digest(identity), "outer_updates_completed": cursor,
               "episodes_completed": cursor * 4, "inner_updates_upper_bound": cursor * 40,
               "inner_updates_actual": sum(r["inner_updates_actual"] for r in updates),
               "confirmed_outer_updates_observed": observed_updates, "outer_attempts_dispatched": attempts,
               "replayed_observed_outer_updates": replayed,
               "compute_count_basis": "logical trajectory plus separately retained physical attempts/replay; unreturned interrupted work is a lower bound",
               "validation_checkpoints": len(history), "active_wall_seconds": wall_budget.wall_seconds,
               "latest_checkpoint": latest, "final_query_accessed": False}
        atomic_json(progress_path, row)
        return row

    def validation_if_due():
        if cursor and cursor % 50 == 0 and (not history or history[-1]["outer_update"] != cursor):
            score = {**evaluate(parameter), "outer_update": cursor}
            choose_checkpoint([score])
            history.append(score)
            snapshot()

    try:
        # Publish the initial parameter/RNG/sampler checkpoint before the first
        # dispatch. SIGKILL before the first scheduled validation must still
        # leave an exact recovery state, rather than an orphaned job identity.
        if resume is None:
            snapshot()
        if wall_budget.exhausted:
            snapshot()
            return parameter, {**progress("PAUSED_WALL_BUDGET"), "validation": history}
        wall_budget.start_interval()
        progress("RESUMED" if resume else "RUNNING")
        # An interruption during validation never skips that scheduled score.
        validation_if_due()
        while cursor < total_updates:
            if wall_budget.exhausted:
                snapshot()
                return parameter, {**progress("PAUSED_WALL_BUDGET"), "validation": history}
            sampler_before, rng_before = sampler.state_dict(), capture_rng(cuda=cuda)
            parameter_before, cursor_before, ledger_before = parameter, cursor, len(updates)
            step_in_progress = True
            episodes = tuple(sampler.next() for _ in range(4))
            attempts += 1
            progress("OUTER_ATTEMPT_DISPATCHED")
            next_parameter, receipt = step(parameter, episodes)
            if not torch.isfinite(next_parameter).all() or receipt.get("full_unroll") is not True:
                raise ValueError("Nonfinite or detached/approximate meta step forbidden")
            actual = receipt.get("inner_updates_actual")
            if type(actual) is not int or not 0 <= actual <= 40:
                raise ValueError("Step must report actual non-fallback inner updates separately")
            parameter = next_parameter
            cursor += 1
            updates.append({**receipt, "outer_update": cursor})
            observed_updates += 1
            step_in_progress = False
            if wall_budget.exhausted:
                snapshot()
                return parameter, {**progress("PAUSED_WALL_BUDGET"), "validation": history}
            validation_if_due()
            progress("RUNNING")
            wall_budget.checkpoint()
        latest = snapshot()
        return parameter, {**progress("UPDATE_BUDGET_COMPLETED"), "validation": history,
                           "selected_checkpoint": choose_checkpoint(history), "checkpoint": latest}
    except BaseException:
        if step_in_progress:
            parameter, cursor = parameter_before, cursor_before
            del updates[ledger_before:]
            sampler.load_state_dict(sampler_before)
            restore_rng(rng_before, cuda=cuda)
        snapshot()
        progress("INTERRUPTED_NOT_COMPLETE")
        raise
    finally:
        wall_budget.close()


def _configuration_receipt(root, path, job, *, signal_sha, split_sha, contract):
    if not path:
        raise ValueError("Confirmation requires a frozen nine-grid development selection")
    selection_path = local_path(root, path)
    receipt = json.loads(selection_path.read_text())
    if (receipt.get("status") != "META_CONFIGURATION_FROZEN" or receipt.get("method") != job.method
            or receipt.get("dimension") != job.dimension or receipt.get("radius") != job.radius
            or receipt.get("selection_role") != "meta_validation" or receipt.get("screening_seed") != 20261018
            or len(receipt.get("screening_receipts", [])) != 9):
        raise ValueError("Confirmation cannot retune or omit registered configurations")
    seen = set()
    for artifact in receipt["screening_receipts"]:
        path = local_path(root, artifact["path"])
        if file_digest(path) != artifact["sha256"]:
            raise ValueError("Screening receipt SHA mismatch")
        value = json.loads(path.read_text())
        config = MetaJob(**value["job"])
        if (config.method != job.method or config.phase != "screening" or value.get("status") != "META_JOB_COMPLETE"
                or value.get("outer_updates_completed") != 1000 or value.get("validation_checkpoints") != 20):
            raise ValueError("Nine real complete independent screening jobs required")
        seen.add(config.identity)
    if seen != {j["job_sha256"] for j in search_plan(job.method)["jobs"]}:
        raise ValueError("Screening search set differs from the registered nine configurations")
    from .meta_artifacts import freeze_development_configuration
    recomputed = freeze_development_configuration(root, [r["path"] for r in receipt["screening_receipts"]],
        method=job.method, output=selection_path.relative_to(root).as_posix(), execute=False,
        source_receipt=receipt.get("screening_source_receipt"),
        reference_receipt=receipt.get("screening_reference_receipt"),
        contract=contract, split_sha256=split_sha)
    recomputed["status"] = "META_CONFIGURATION_FROZEN"
    if receipt != recomputed or receipt.get("signal_contract_sha256") != signal_sha or receipt.get("split_sha256") != split_sha:
        raise ValueError("Configuration freeze differs from actual nine-grid validation evidence")
    return file_digest(selection_path)


def verify_meta_inputs(root, *, source_receipt, reference_receipt, split_path, job, selection_receipt=None):
    """Read-only real-data gates; imports do not initialize CUDA or fit models."""
    root = Path(root).resolve(strict=True)
    tasks = json.loads((root / "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE" or tasks.get("W3", {}).get("status") != "COMPLETE":
        raise ValueError("Real complete W1 and W3 are required for development episodes")
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    signal_sha = validate_signal_contract(contract)
    source, source_info = load_verified_source(root, source_receipt, contract)
    if source_info.get("pipeline_seed") != job.seed or source_info.get("seed") != job.seed:
        raise ValueError("The complete pipeline must use the matching same-seed source f0")
    from .source_reference_io import load_verified_reference
    reference, reference_info = load_verified_reference(root, reference_receipt, source_receipt, contract)
    from .episodes import load_episode_catalog
    train = load_episode_catalog(root, None, split_path, role="meta_train", purpose="meta_gradient")
    validation = load_episode_catalog(root, None, split_path, role="meta_validation", purpose="model_selection")
    if train.input_contract_sha256 != signal_sha or validation.input_contract_sha256 != signal_sha or train.split_sha256 != validation.split_sha256:
        raise ValueError("Meta train/validation signal or split identity mismatch")
    selection_sha = _configuration_receipt(root, selection_receipt, job, signal_sha=signal_sha,
                                          split_sha=train.split_sha256, contract=contract) if job.phase == "confirmation" else None
    return source, reference, train, validation, {"signal_contract_sha256": signal_sha,
        "source_architecture": source_info.get("architecture"), "source_feature_dim": source.feature_dim,
        "source_model_state_sha256": source.frozen_state_sha256,
        "source_training_event_uids_sha256": reference.events_sha256,
        "geometry_sha256": SpatialGeometry.from_reference(reference.matrix).sha256(),
        "source_receipt_sha256": file_digest(local_path(root, source_receipt)),
        "reference_receipt_sha256": file_digest(local_path(root, reference_receipt)),
        "stieger_manifest_sha256": train.manifest_sha256, "split_sha256": train.split_sha256,
        "selection_receipt_sha256": selection_sha}


def evaluate_development(source, geometry, parameter, catalog, job):
    if catalog.role != "meta_validation":
        raise PermissionError("Checkpoint selection accepts meta_validation only")
    rows, excluded = [], []
    for plan in catalog.plans:
        if not plan.query_eligible:
            excluded.append({"participant_id": plan.participant_id, "session_id": plan.session_id,
                             "reason": "predeclared_future_query_ineligible"})
            continue
        episode = catalog.materialize(plan, device=parameter.device)
        if job.method in {"B10", "B11"}:
            from .methods.lowrank import LowRankConfig, inner_unroll_lowrank, lowrank_logits
            config = LowRankConfig(rank=job.dimension, radius=job.radius, pct_weight=.1 if job.method == "B11" else 0)
            weights, _, fallback = inner_unroll_lowrank(source, geometry, parameter, episode.support, config=config)
            with torch.no_grad():
                logits = source.source_logits(episode.prediction_inputs.x) if fallback else lowrank_logits(source, geometry, episode.prediction_inputs.x, weights)
        else:
            config = InnerConfig(radius=job.radius, pct_weight=.1 if job.method == "P+PCT" else 0)
            family = "additive" if job.method == "B8" else "exp"
            weights, _, fallback = inner_unroll(source, geometry, parameter, episode.support, config=config, family=family)
            with torch.no_grad():
                logits = source.source_logits(episode.prediction_inputs.x) if fallback else adapted_logits(source, geometry, parameter, episode.prediction_inputs.x, weights, family=family)
        labels = episode.evaluation_labels.labels().to(logits.device)
        predicted = logits.argmax(1)
        ba = sum(float((predicted[labels == c] == c).double().mean()) for c in (0, 1)) / 2
        rows.append({"participant_id": plan.participant_id, "session_id": plan.session_id,
                     "ba": ba, "balanced_bce": float(balanced_bce(logits, labels)), "fallback": bool(fallback)})
    source.assert_unchanged()
    return {**participant_equal_metrics(rows, expected_participants=catalog.partition_people),
            "sessions": rows, "ineligible_sessions": excluded}


def execute_meta_job(root, *, job, source_receipt, reference_receipt, split_path,
                     budget_path, user_started_gpu=False, resume=None, selection_receipt=None):
    if not user_started_gpu:
        raise PermissionError("Explicit later user GPU boot is required; this tool never starts servers")
    root = Path(root).resolve(strict=True)
    source, reference, train, validation, inputs = verify_meta_inputs(root, source_receipt=source_receipt,
        reference_receipt=reference_receipt, split_path=split_path, job=job, selection_receipt=selection_receipt)
    budget_file = local_path(root, budget_path)
    budget = json.loads(budget_file.read_text())
    wall = budget.get("max_wall_seconds")
    if (budget.get("stage") != "meta_training" or budget.get("job_sha256") != job.identity
            or type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
            or budget.get("max_outer_updates") != job.updates or budget.get("max_jobs") != 1):
        raise ValueError("Exact bounded per-job execution budget required")
    if job.phase != "engineering":
        from .meta_engineering import verify_meta_engineering
        proof = verify_meta_engineering(root, budget.get("measured_meta_engineering_receipt"),
                                        job=job, inputs=inputs)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; real meta fitting has no CPU fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ValueError("Unsupported deterministic CUBLAS workspace")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True, timeout=5)
        driver_versions = query.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        driver_versions = None
    budget_constraints = {k: budget.get(k) for k in ("stage", "job_sha256", "max_outer_updates", "max_jobs")}
    budget_constraints["engineering_receipt"] = budget.get("measured_meta_engineering_receipt")
    identity = {**inputs, "job": asdict(job), "budget_constraints": budget_constraints,
                "code": {p: file_digest(root / p) for p in META_CODE_FILES}, "python": platform.python_version(),
                "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "device_name": properties.name, "device_uuid": str(getattr(properties, "uuid", "unavailable")),
                "device_total_memory_bytes": properties.total_memory,
                "cudnn": torch.backends.cudnn.version(), "driver_versions": driver_versions,
                "device_compute_capability": [properties.major, properties.minor],
                "full_unroll": True, "autocast": False, "tf32": False,
                "float32_matmul_precision": "highest", "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    runtime_profile = {k: identity[k] for k in META_RUNTIME_KEYS}
    if job.phase != "engineering" and proof.get("runtime_profile") != runtime_profile:
        raise ValueError("Engineering timing/runtime/code profile differs from the current GPU execution")
    output = local_path(root, "runs/meta/" + job.identity)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("The same meta job is already running") from None
        episode_plan = {"schema_version": "paper3-frozen-meta-episode-plans-v1",
                        "input_contract_sha256": inputs["signal_contract_sha256"],
                        "stieger_manifest_sha256": inputs["stieger_manifest_sha256"],
                        "split_sha256": inputs["split_sha256"], "budget_acquired_trials": 20,
                        "query_rule": "all_eligible_trials_in_future_complete_physical_runs_after_B20_boundary",
                        "meta_train": [p.to_dict() for p in train.plans],
                        "meta_validation": [p.to_dict() for p in validation.plans],
                        "final_or_internal_query_included": False}
        episode_path = output / "episode_plans.json"
        if episode_path.exists() and json.loads(episode_path.read_text()) != episode_plan:
            raise ValueError("Pre-generated frozen real episode manifest changed")
        if not episode_path.exists():
            atomic_json(episode_path, episode_plan)
        identity["episode_plan_sha256"] = file_digest(episode_path)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Existing job requires an exact code/data/update identity")
            prior_receipt_path = output / "receipt.json"
            if prior_receipt_path.exists() and json.loads(prior_receipt_path.read_text()).get("status") in {"META_JOB_COMPLETE", "META_ENGINEERING_COMPLETE"}:
                raise ValueError("Completed meta jobs are immutable and must not be refitted")
            latest_path = output / "latest_checkpoint.json"
            if latest_path.exists():
                if resume is None:
                    raise ValueError("Existing job requires an explicit latest-checkpoint resume")
                latest = json.loads(latest_path.read_text())
                requested = local_path(root, resume)
                if (latest.get("relative_to_job_output") is not True or Path(latest["path"]).name != latest["path"]
                        or requested != (output / latest["path"]).resolve() or file_digest(requested) != latest["sha256"]):
                    raise ValueError("Resume must use the latest verified checkpoint; older replay requires a separate recorded recovery")
            else:
                # A hard interruption between identity publication and the
                # initial checkpoint is repairable only before any dispatch.
                # Every actual attempt publishes progress before its kernel.
                progress_path = output / "progress.json"
                prior = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                if (resume is not None or prior.get("outer_attempts_dispatched", 0) != 0
                        or prior.get("outer_updates_completed", 0) != 0):
                    raise ValueError("Missing initial checkpoint after observed work cannot be silently repaired")
        elif resume is not None:
            raise ValueError("Resume has no prior same-job identity")
        atomic_json(identity_path, identity)
        # A wall-only operational amendment can extend a paused run without
        # changing its model/data/update identity. Every supplied budget stays
        # separately hashed and recorded; scientific limits remain exact.
        budget_log_path = output / "operational_budgets.json"
        budget_log = json.loads(budget_log_path.read_text()) if budget_log_path.exists() else []
        budget_record = {"path": budget_file.relative_to(root).as_posix(), "sha256": file_digest(budget_file),
                         "max_cumulative_wall_seconds": wall}
        if budget_record not in budget_log:
            budget_log.append(budget_record)
            atomic_json(budget_log_path, budget_log)
        random.seed(job.seed)
        np.random.seed(job.seed)
        torch.manual_seed(job.seed)
        torch.cuda.manual_seed_all(job.seed)
        source = source.to("cuda:0")
        source.assert_unchanged()
        geometry = SpatialGeometry.from_reference(reference.matrix.to("cuda:0"))
        from .episodes import DeterministicEpisodeSampler
        sampler = DeterministicEpisodeSampler(train, job.seed, device="cuda:0")
        if job.method in {"B10", "B11"}:
            from .methods.lowrank import initialize_lowrank, lowrank_outer_step, LowRankConfig
            parameter = initialize_lowrank(job.dimension, job.seed, device="cuda:0").requires_grad_(True)
            config = LowRankConfig(rank=job.dimension, radius=job.radius, pct_weight=.1 if job.method == "B11" else 0)
            kernel = lambda b, episodes: lowrank_outer_step(source, geometry, b, episodes, config=config, method=job.method)
        else:
            parameter = initialize_dictionary(job.dimension, job.seed, device="cuda:0").requires_grad_(True)
            config = InnerConfig(radius=job.radius, pct_weight=.1 if job.method == "P+PCT" else 0)
            kernel = lambda b, episodes: outer_step(source, geometry, b, episodes, config=config,
                                                     family="additive" if job.method == "B8" else "exp")

        def measured_step(value, episodes):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(0)
            start.record()
            receipt["training_kernel_dispatch_attempted"] = True
            value, info = kernel(value, episodes)
            end.record()
            end.synchronize()
            receipt["training_started"] = True
            actual = sum(10 for e in episodes if bool((e.support.y == 0).any() and (e.support.y == 1).any()))
            return value, {**info, "inner_updates_actual": actual,
                           "gpu_stream_elapsed_ms": start.elapsed_time(end),
                           "end_to_end_wall_seconds": time.monotonic() - started,
                           "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
                           "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(0)}

        receipt = {"status": "RUNNING", "synthetic": False, "job": asdict(job), **inputs,
                   "source_receipt": {"path": local_path(root, source_receipt).relative_to(root).as_posix(),
                                      "sha256": inputs["source_receipt_sha256"]},
                   "reference_receipt": {"path": local_path(root, reference_receipt).relative_to(root).as_posix(),
                                         "sha256": inputs["reference_receipt_sha256"]},
                   "identity_sha256": digest(identity),
                   "episode_plan": {"path": episode_path.relative_to(root).as_posix(),
                                    "sha256": file_digest(episode_path)},
                   "runtime_profile": runtime_profile,
                   "memory_measurement_scope": "per_outer_full_unroll_including_live_tensors_and_allocator_reservations",
                   "operational_budget": budget_record,
                   "started_at_utc": datetime.now(timezone.utc).isoformat(), "training_started": False,
                   "training_kernel_dispatch_attempted": False,
                   "remote_actions": False, "external_access": False, "full_unroll": True}
        atomic_json(output / "receipt.json", receipt)
        try:
            _, result = run_outer_loop(parameter=parameter, sampler=sampler, step=measured_step,
                evaluate=lambda b: evaluate_development(source, geometry, b, validation, job),
                identity=identity, output=output, total_updates=job.updates, max_wall_seconds=wall,
                resume=local_path(root, resume) if resume else None, cuda=True,
                wall_amendment_note=budget.get("operational_amendment_note"))
            source.assert_unchanged()
            status = ("META_ENGINEERING_COMPLETE" if job.phase == "engineering" else "META_JOB_COMPLETE") if result["status"] == "UPDATE_BUDGET_COMPLETED" else result["status"]
            receipt.update(result, status=status, source_state_unchanged=True,
                           training_started=result["confirmed_outer_updates_observed"] > 0,
                           scientific_fit_completed=status == "META_JOB_COMPLETE")
            if status in {"META_ENGINEERING_COMPLETE", "META_JOB_COMPLETE"}:
                selected = output / f"outer-{result['selected_checkpoint']['outer_update']:04d}.pt"
                receipt["selected_parameter_checkpoint"] = {"path": selected.relative_to(root).as_posix(),
                                                             "sha256": file_digest(selected)}
        except BaseException as exc:
            receipt.update(status="INTERRUPTED_NOT_COMPLETE", error_type=type(exc).__name__, scientific_fit_completed=False)
            atomic_json(output / "receipt.json", receipt)
            raise
        atomic_json(output / "receipt.json", receipt)
        return receipt
