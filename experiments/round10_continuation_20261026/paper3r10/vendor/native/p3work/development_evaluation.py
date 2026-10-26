"""Bounded W6 development evaluation, never external evaluation or GPU startup.

Importing/planning does not fit or read signal data. Production execution needs
verified source/dev artifacts, an actually completed calibration-only profile,
explicit budgets and a later user-started GPU. This module never boots a server.
CPU callback fixtures have a separate, permanently synthetic entry and cannot
supply scientific selection evidence. No network or credential API is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import time

import numpy as np
import torch

from . import method_registry as registry
from .adapt import CalibrationPrefix, PredictionInputs, PredictionResult, balanced_bce
from .contracts import digest, file_digest, validate_signal_contract
from .episodes import load_episode_catalog
from .prediction_store import (verify_prediction_receipt,
                               _local, _reference, _write_identical_or_new)
from .recovery_budget import PersistentWallBudget
from .selection import choose_strongest_control, participant_equal_metrics
from .spatial_geometry import SpatialGeometry
from .training_state import atomic_json

SCHEMA = "paper3-development-evaluation-v1"
SEEDS = (20261018, 20261019, 20261020)
METHODS = registry.METHOD_IDS
CONTROLS = tuple(f"B{i}" for i in range(1, 12))
TIMING_PROTOCOL = "CPU_staged_valid_support_to_synchronized_fitted_state_v1"
TIMING_SCOPE = ("Includes support transfer, method-specific artifact validation, source copy, "
                "target covariance/features and optimization; excludes catalog verification, "
                "source/reference loading, query inference, prediction I/O and acquisition waiting.")
GPU_MEMORY_SCOPE = "absolute_process_allocator_peak_after_reset_including_resident_source_not_delta"
WARMUP_FORWARDS = 2
CONFIGURATION_TIE_RULE = "participant_BA_desc/balanced_BCE_asc/selected_meta_checkpoint_asc/registered_grid_order"
CODE_FILES = ("p3work/development_evaluation.py", "tools/evaluate_development.py",
    "p3work/method_registry.py", "p3work/episodes.py", "p3work/prediction_store.py",
    "p3work/selection.py", "p3work/adapt.py", "p3work/spatial_geometry.py",
    "p3work/meta_artifacts.py", "p3work/methods/lowrank.py",
    "p3work/methods/classical.py", "p3work/methods/neural_controls.py",
    "p3work/f0_bridge.py", "p3work/source_reference_io.py", "p3work/classical_training.py",
    "p3work/contracts.py", "p3work/training_state.py", "p3work/recovery_budget.py", "p3work/runtime_clock.py",
    "p3work/fit_artifacts.py", "p3work/window_acceptance.py", "p3work/source_cache_acceptance.py",
    "p3work/source_training.py", "tools/run_controlled_pilot.py", "tools/prepare_source_cache.py",
    "tools/source_preparation_batch.py", "tools/r2_restore.py", "tools/r2_backup.py",
    "p3work/preprocess.py", "p3work/adapters/source_native.py", "p3work/adapters/stieger_native.py")
POLICY_FILES = ("protocol/BASELINE_SPEC.md", "protocol/EXPERIMENT_PROTOCOL.md", "protocol/METHOD_SPEC.md",
                "protocol/EXPERIMENT_MATRIX.json", "protocol/WORK_TASKS.md")


class DevelopmentGateError(ValueError):
    """Fixed non-sensitive gate failure; no EEG, paths or secret values."""


def _json(path):
    return json.loads(path.read_text())


def _descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def _put(path, value):
    _write_identical_or_new(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class DevelopmentJob:
    method: str
    configuration_index: int = 0
    seed: int = SEEDS[0]
    phase: str = "screening"

    def __post_init__(self):
        if self.method not in METHODS or type(self.configuration_index) is not int:
            raise DevelopmentGateError("registered_development_method_and_grid_index_required")
        if not 0 <= self.configuration_index < len(registry.registered_grid(self.method)):
            raise DevelopmentGateError("configuration_outside_registered_grid")
        if self.phase not in {"screening", "confirmation"} or type(self.seed) is not int:
            raise DevelopmentGateError("registered_development_phase_required")
        if self.seed not in ((SEEDS[0],) if self.phase == "screening" else SEEDS[1:]):
            raise DevelopmentGateError("screening_seed_reused_not_refitted_in_confirmation")

    @property
    def configuration(self):
        return registry.registered_grid(self.method)[self.configuration_index]

    @property
    def identity(self):
        return digest({**asdict(self), "configuration": self.configuration, "schema_version": SCHEMA})


def build_development_plan(root=None):
    """Exact registered search, with no data/artifact/GPU access or writes."""
    result = {"schema_version": SCHEMA, "status": "PLANNED_NOT_EXECUTED",
        "role": "meta_validation", "budget_acquired_trials": 20,
        "screening_seed": SEEDS[0], "confirmation_seeds": list(SEEDS),
        "methods": {m: [{"configuration_index": i, "configuration": c,
            "job_sha256": DevelopmentJob(m, i).identity} for i, c in enumerate(registry.registered_grid(m))]
            for m in METHODS}, "screening_jobs": sum(len(registry.registered_grid(m)) for m in METHODS),
        "additional_confirmation_jobs": 2*len(METHODS), "screening_selected_seed_reused": True,
        "B_star_candidates": list(CONTROLS), "B0_separately_reported": True,
        "configuration_tie_rule": CONFIGURATION_TIE_RULE, "B_star_BA_tolerance": .005,
        "B_star_tie_rule": "same_device_median_calibration_seconds_then_numeric_method_ID",
        "timing_protocol": TIMING_PROTOCOL, "timing_scope": TIMING_SCOPE,
        "training_started": False, "external_access": False, "server_actions": False,
        "production_execution_status": "GATED_NOT_EXECUTED_NO_REAL_PROFILE_OR_EVALUATION_CLAIM",
        "profile_plan": build_calibration_profile_plan(),
        "minimum_production_gap": ["real_W1_W2_W3_and_same_seed_artifacts",
            "later_user_started_CUDA_and_explicit_budgets", "actually_executed_full_SHA_verified_calibration_profile"],
        "required_gates": ["W1/W2/W3_COMPLETE", "verified_same_seed_f0_Rref",
            "verified_independent_learned_or_classical_artifacts", "later_user_started_CUDA",
            "explicit_wall_fit_and_per_calibration_limits", "same_device_measured_timing_profile"]}
    if root is not None:
        root = Path(root).resolve(strict=True)
        executed_root = Path(__file__).resolve().parents[1]
        result["code"] = {}
        for name in CODE_FILES:
            path = _local(root, name)
            if path.read_bytes() != (executed_root / name).read_bytes():
                raise DevelopmentGateError("root_code_artifact_differs_from_actual_execution_package")
            result["code"][name] = file_digest(path)
        result["policy"] = {p: file_digest(_local(root, p)) for p in POLICY_FILES}
    result["plan_sha256"] = digest(result)
    return result


def _catalog_binding(catalog):
    if (catalog.role != "meta_validation" or catalog.purpose != "model_selection"
            or len(catalog.partition_people) != 11 or len(set(catalog.partition_people)) != 11):
        raise DevelopmentGateError("only_complete_frozen_eleven_person_meta_validation_supported")
    plans = list(catalog.plans)
    keys = [(p.participant_id, p.session_id) for p in plans]
    if len(set(keys)) != len(keys) or not plans or any(p.role != "meta_validation" for p in plans):
        raise DevelopmentGateError("unique_validation_catalog_sessions_required")
    if ({p.participant_id for p in plans} != set(catalog.partition_people)
            or {p.participant_id for p in plans if p.query_eligible} != set(catalog.partition_people)
            or any(p.manifest_sha256 != catalog.manifest_sha256 or p.split_sha256 != catalog.split_sha256
                or p.input_contract_sha256 != catalog.input_contract_sha256 for p in plans)):
        raise DevelopmentGateError("every_frozen_validation_person_needs_evaluable_accounted_sessions")
    return {"catalog_identity_sha256": catalog.identity_sha256,
        "manifest_sha256": catalog.manifest_sha256, "split_sha256": catalog.split_sha256,
        "input_contract_sha256": catalog.input_contract_sha256,
        "participants": sorted(catalog.partition_people),
        "plans": [{"participant_id": p.participant_id, "session_id": p.session_id,
                   "temporal_audit_sha256": p.temporal_audit_sha256, "query_eligible": p.query_eligible,
                   "query_event_uids": list(p.query_event_uids)} for p in plans]}


def _dependencies(root, job, artifact_catalog):
    value = _json(_reference(root, artifact_catalog, suffix=".json"))
    if value.get("schema_version") != "paper3-development-artifact-catalog-v1":
        raise DevelopmentGateError("explicit_development_artifact_catalog_required")
    seeds = value.get("seeds", {})
    selected = seeds.get(str(job.seed), {})
    source, reference = selected.get("source_f0"), selected.get("source_reference")
    _reference(root, source, suffix=".json"); _reference(root, reference, suffix=".json")
    learned, bank = None, None
    if job.method in registry.LEARNED_METHODS:
        rows = [r for r in value.get("learned_artifacts", []) if r.get("method") == job.method
                and r.get("seed") == job.seed and r.get("configuration") == job.configuration]
        if len(rows) != 1:
            raise DevelopmentGateError("one_independent_same_method_seed_configuration_meta_artifact_required")
        learned = rows[0]["receipt"]
        _reference(root, learned, suffix=".json")
    if job.method in registry.CLASSICAL_METHODS:
        key = str(job.configuration["m"] if job.method == "B4" else job.configuration["shrinkage"])
        bank = value.get("source_banks", {}).get(job.method, {}).get(key)
        _reference(root, bank, suffix=".json")
    return {"source_f0": source, "source_reference": reference, "learned": learned, "source_bank": bank}


def verify_development_inputs(root, *, job, artifact_catalog, configuration_receipt=None):
    """Read-only formal gates. Does not initialize CUDA, fit, predict or label-score."""
    root = Path(root).resolve(strict=True)
    if type(job) is not DevelopmentJob:
        raise DevelopmentGateError("typed_registered_development_job_required")
    tasks = _json(_local(root, "status/tasks.json"))
    if any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3")):
        raise DevelopmentGateError("real_complete_W1_W2_W3_required")
    contract_path = _local(root, "contracts/input_contract.json")
    contract = _json(contract_path)
    signal_sha = validate_signal_contract(contract)
    deps = _dependencies(root, job, artifact_catalog)
    context = registry.Context.from_verified(root, deps["source_f0"]["path"], deps["source_reference"]["path"], contract)
    if context.pipeline_seed != job.seed or context.signal_contract_sha256 != signal_sha:
        raise DevelopmentGateError("verified_source_pipeline_seed_or_signal_mismatch")
    catalog = load_episode_catalog(root, role="meta_validation", purpose="model_selection")
    binding = _catalog_binding(catalog)
    if context.split_sha256 != catalog.split_sha256 or context.signal_contract_sha256 != catalog.input_contract_sha256:
        raise DevelopmentGateError("source_reference_validation_split_or_input_mismatch")
    w2 = tasks["W2"].get("cache_manifest")
    if not isinstance(w2, dict) or w2.get("sha256") != context.source_cache_sha256:
        raise DevelopmentGateError("source_context_must_match_actual_W2_cache")
    _reference(root, w2, suffix=".json")
    selected_update = None
    if deps["learned"]:
        learned = context._learned(job.method, job.configuration, deps["learned"]["path"])
        selected_update = learned.selected_outer_update
    if deps["source_bank"]:
        registry.load_verified_source_bank(root, deps["source_bank"], method=job.method,
            expected_m=job.configuration.get("m"), expected_shrink=job.configuration.get("shrinkage"),
            input_sha=signal_sha, cache_sha=context.source_cache_sha256)
    selection = None
    if job.phase == "confirmation":
        selection = verify_configuration_receipt(root, configuration_receipt)
        if selection["method"] != job.method or selection["configuration_index"] != job.configuration_index:
            raise DevelopmentGateError("confirmation_requires_exact_frozen_screening_configuration")
        if (selection["catalog"] != binding or selection["code"] != build_development_plan(root)["code"]):
            raise DevelopmentGateError("confirmation_catalog_or_code_differs_from_screening")
    plan = build_development_plan(root)
    inputs = {"job": asdict(job), "configuration": job.configuration, "dependencies": deps,
        "catalog": binding, "source_context_identity_sha256": context.identity_sha256,
        "selected_meta_outer_update": selected_update, "code": plan["code"], "policy": plan["policy"],
        "configuration_receipt": configuration_receipt, "configuration_tie_rule": CONFIGURATION_TIE_RULE}
    return context, catalog, inputs


def _score(root, plan, episode, descriptor, job=None):
    receipt = verify_prediction_receipt(root, descriptor)
    if (receipt["prediction_role"] != "meta_validation" or receipt["episode_plan_sha256"] != plan.temporal_audit_sha256
            or tuple(receipt["query_event_uids"]) != plan.query_event_uids):
        raise DevelopmentGateError("verified_prediction_must_precede_matching_validation_labels")
    if job is not None:
        configuration = _json(_reference(root, receipt["references"]["configuration"]))
        if (receipt["method"] != job.method or receipt["pipeline_seed"] != job.seed
                or configuration.get("method") != job.method or configuration.get("configuration") != job.configuration):
            raise DevelopmentGateError("prediction_method_seed_or_configuration_differs_from_job")
    # The label capability is touched only after the complete store transaction
    # and all output arrays were verified, including on crash recovery.
    labels = episode.evaluation_labels.labels(purpose="development_evaluation", prediction_receipt=descriptor).cpu()
    directory = _reference(root, descriptor).parent
    logits = torch.from_numpy(np.load(directory / "logits.npy", allow_pickle=False))
    if labels.shape != (len(plan.query_event_uids),) or any(int((labels == c).sum()) < 5 for c in (0, 1)):
        raise DevelopmentGateError("query_label_qualification_cannot_be_changed_per_method")
    predicted = logits.argmax(1)
    return {"participant_id": plan.participant_id, "session_id": plan.session_id,
        "ba": sum(float((predicted[labels == c] == c).double().mean()) for c in (0, 1))/2,
        "balanced_bce": float(balanced_bce(logits, labels)), "prediction_receipt": descriptor}


def _summarize(root, catalog, sessions, job=None):
    expected = {(p.participant_id, p.session_id): p for p in catalog.plans if p.query_eligible}
    if set(sessions) != {digest([p, s]) for p, s in expected}:
        raise DevelopmentGateError("all_eligible_validation_sessions_required_no_subset_completion")
    rows, times = [], []
    for person, session in expected:
        item = sessions[digest([person, session])]
        plan = expected[person, session]
        episode = catalog.materialize(plan, device="cpu")
        row = _score(root, plan, episode, item["prediction_receipt"], job)
        if row != item:
            raise DevelopmentGateError("saved_session_score_or_receipt_changed")
        rows.append(row)
        cost = _json(_reference(root, row["prediction_receipt"]).with_name("cost.json"))
        if not _positive(cost["calibration_wall_seconds"]):
            raise DevelopmentGateError("positive_measured_calibration_timing_required")
        times.append(cost["calibration_wall_seconds"])
    return {**participant_equal_metrics(rows, expected_participants=catalog.partition_people),
        "sessions": rows, "calibration_median_seconds": statistics.median(times),
        "calibration_seconds": times,
        "ineligible_sessions": [{"participant_id": p.participant_id, "session_id": p.session_id,
            "reason": "predeclared_future_query_ineligible", "temporal_audit_sha256": p.temporal_audit_sha256}
            for p in catalog.plans if not p.query_eligible]}


def _run_core(root, *, job, catalog, inputs, output, predict_one, max_fit_attempts,
              max_wall_seconds, synthetic, monotonic=time.monotonic, utc_now=None,
              boot_id=None, wall_amendment_note=None):
    """Session transactions: an already committed prediction never triggers refit.

    predict_one receives a plan and an attempt-specific directory; it must return
    a real prediction-store descriptor. Attempts and open wall intents are
    durable before dispatch. Failed/uncommitted attempts remain inspectable.
    """
    if synthetic is not True:
        raise DevelopmentGateError("BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR")
    if type(job) is not DevelopmentJob:
        raise DevelopmentGateError("typed_registered_development_job_required")
    root = Path(root).resolve(strict=True)
    destination = _local(root, output, exists=False)
    if not output.startswith("runs/") or type(max_fit_attempts) is not int or max_fit_attempts < 1 or not _positive(max_wall_seconds):
        raise DevelopmentGateError("explicit_root_safe_output_and_operational_limits_required")
    destination.mkdir(parents=True, exist_ok=True)
    identity = {"schema_version": SCHEMA, "job": asdict(job), "configuration": job.configuration,
        "catalog": _catalog_binding(catalog), "inputs": inputs, "synthetic": synthetic,
        "scientific_fit_claim": False}
    identity_sha = digest(identity)
    with (_local(root, output+"/.evaluation.lock", exists=False)).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DevelopmentGateError("same_development_job_already_running") from None
        _put(destination / "identity.json", identity)
        progress_path = _local(root, output+"/progress.json", exists=False)
        if not progress_path.exists() and any((destination / name).exists() for name in ("sessions", "receipt.json", "wall_budget.json")):
            raise DevelopmentGateError("missing_progress_after_observed_work_cannot_be_repaired_by_refit")
        state = _json(progress_path) if progress_path.exists() else {
            "identity_sha256": identity_sha, "fit_attempts_dispatched": 0, "sessions": {}, "active": None}
        if (state.get("identity_sha256") != identity_sha or type(state.get("fit_attempts_dispatched")) is not int
                or state["fit_attempts_dispatched"] < 0 or not isinstance(state.get("sessions"), dict)):
            raise DevelopmentGateError("invalid_or_changed_resumable_evaluation_identity")
        expected_keys = {digest([p.participant_id, p.session_id]) for p in catalog.plans if p.query_eligible}
        if set(state["sessions"]) - expected_keys:
            raise DevelopmentGateError("unexpected_validation_session_checkpoint")
        if len(state["sessions"]) > state["fit_attempts_dispatched"]:
            raise DevelopmentGateError("completed_sessions_exceed_durable_dispatch_count")
        atomic_json(progress_path, state)
        _local(root, output+"/wall_budget.json", exists=False)
        _local(root, output+"/wall_budget.json.lock", exists=False)
        options = {"monotonic": monotonic, "boot_id": boot_id, "amendment_note": wall_amendment_note}
        if utc_now is not None:
            options["utc_now"] = utc_now
        journal = PersistentWallBudget(destination / "wall_budget.json", identity_sha, max_wall_seconds, **options)
        try:
            if journal.blocked:
                raise DevelopmentGateError("uncertain_clock_history_blocks_evaluation")
            journal.start_interval()
            for plan in catalog.plans:
                if not plan.query_eligible:
                    continue
                key = digest([plan.participant_id, plan.session_id])
                if key in state["sessions"]:
                    _score(root, plan, catalog.materialize(plan, device="cpu"), state["sessions"][key]["prediction_receipt"], job)
                    continue
                active = state.get("active")
                committed = None
                if active is not None:
                    if (active.get("session_key") != key or type(active.get("attempt")) is not int
                            or active["attempt"] != state["fit_attempts_dispatched"]
                            or active.get("output") != output+f"/sessions/{key}/attempt-{active['attempt']:06d}"):
                        raise DevelopmentGateError("invalid_active_attempt_checkpoint")
                    receipt_path = _local(root, active["output"]+"/predictions/receipt.json", exists=False)
                    if receipt_path.exists():
                        committed = _descriptor(root, receipt_path)
                if committed is None:
                    if journal.exhausted or state["fit_attempts_dispatched"] >= max_fit_attempts:
                        atomic_json(progress_path, state)
                        return {"status": "PAUSED_OPERATIONAL_BUDGET", "synthetic": synthetic,
                            "sessions_completed": len(state["sessions"]), "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                            "budget": journal.report(), "scientific_fit_completed": False}
                    state["fit_attempts_dispatched"] += 1
                    attempt = state["fit_attempts_dispatched"]
                    attempt_output = output+f"/sessions/{key}/attempt-{attempt:06d}"
                    state["active"] = {"session_key": key, "attempt": attempt, "output": attempt_output}
                    atomic_json(progress_path, state)
                    committed = predict_one(plan, attempt_output)
                if committed.get("path") != state["active"]["output"]+"/predictions/receipt.json":
                    raise DevelopmentGateError("prediction_transaction_outside_durable_session_attempt")
                row = _score(root, plan, catalog.materialize(plan, device="cpu"), committed, job)
                state["sessions"][key] = row
                state["active"] = None
                atomic_json(progress_path, state)
                journal.checkpoint()
            summary = _summarize(root, catalog, state["sessions"], job)
            result = {"schema_version": SCHEMA, "status": "SYNTHETIC_VALIDATION_FIXTURE_COMPLETE",
                "synthetic": True, "scientific_fit_completed": False, "measured": False,
                "role": "meta_validation", "method": job.method, "seed": job.seed, "job": asdict(job),
                "configuration": job.configuration, "identity_sha256": identity_sha, "inputs": inputs,
                "catalog": identity["catalog"], "timing_protocol": TIMING_PROTOCOL,
                "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                "labels_after_verified_predictions": True, **summary}
            _put(destination / "receipt.json", result)
            return {**result, "receipt": _descriptor(root, destination / "receipt.json")}
        finally:
            # SIGKILL cannot run this block; the existing journal then recovers
            # the whole same-boot open interval, including downtime.
            journal.close()


def run_synthetic_validation(root, *, job, catalog, output, predict_one,
                             max_fit_attempts, max_wall_seconds, **clock_options):
    """CPU callback fixtures only. Its receipts always fail scientific selection."""
    return _run_core(root, job=job, catalog=catalog, output=output, predict_one=predict_one,
        inputs={"scope": "SYNTHETIC_SOFTWARE_FIXTURE_NOT_RESEARCH_EVIDENCE"}, synthetic=True,
        max_fit_attempts=max_fit_attempts, max_wall_seconds=max_wall_seconds, **clock_options)


def execute_validation_job(root, *, job, artifact_catalog, budget, user_started_gpu=False,
                           configuration_receipt=None, output=None):
    """Real, profile-proven local evaluation; never starts a GPU/server."""
    invoked_at = time.monotonic()
    if user_started_gpu is not True:
        raise PermissionError("later_explicit_user_GPU_boot_required")
    root = Path(root).resolve(strict=True)
    limits = _read_budget(root, budget, stage="development_evaluation", expected_identity=job.identity)
    profile_ref = limits.get("measured_timing_profile")
    if not isinstance(profile_ref, dict) or not {"path", "sha256"} <= set(profile_ref):
        raise DevelopmentGateError("complete_measured_profile_artifact_descriptor_required")
    header = _json(_reference(root, _ref(profile_ref), suffix=".json"))
    if (header.get("schema_version") != PROFILE_SCHEMA or header.get("synthetic") is not False
            or header.get("status") != "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE"
            or not {"identity", "records", "initial_budget", "completion_budget"} <= set(header)):
        raise DevelopmentGateError("complete_typed_calibration_profile_proof_required")
    runtime = _production_runtime()
    return _execute_validation_production(root, job=job, artifact_catalog=artifact_catalog, limits=limits,
        profile=_ref(profile_ref), runtime=runtime, configuration_receipt=configuration_receipt, output=output,
        invoked_at=invoked_at)


def verify_development_receipt(root, descriptor):
    """Recheck all real dependencies, profiles, parameters, predictions and scores."""
    root = Path(root).resolve(strict=True)
    value = _json(_reference(root, descriptor, suffix=".json"))
    if (value.get("schema_version") != SCHEMA or value.get("status") != "DEVELOPMENT_EVALUATION_COMPLETE"
            or value.get("synthetic") is not False or value.get("scientific_fit_completed") is not True
            or value.get("measured") is not True or value.get("role") != "meta_validation"):
        raise DevelopmentGateError("completed_real_development_evidence_required_no_fixture_or_subset")
    return _verify_development_production(root, descriptor, value)


def freeze_screening_configuration(root, *, method, screening_receipts, output=None, execute=False):
    """All registered screen settings, one seed, same catalog and GPU host."""
    if method not in METHODS or len(screening_receipts) != len(registry.registered_grid(method)):
        raise DevelopmentGateError("all_registered_screening_configurations_required")
    root = Path(root).resolve(strict=True)
    rows = [verify_development_receipt(root, r) for r in screening_receipts]
    jobs = [DevelopmentJob(**r["job"]) for r in rows]
    if ({j.configuration_index for j in jobs} != set(range(len(registry.registered_grid(method))))
            or any(j.method != method or j.phase != "screening" or j.seed != SEEDS[0] for j in jobs)
            or len({digest(r["catalog"]) for r in rows}) != 1
            or len({r["inputs"].get("device_identity_sha256") for r in rows}) != 1
            or any(re.fullmatch(r"[0-9a-f]{64}", str(r["inputs"].get("device_identity_sha256", ""))) is None for r in rows)):
        raise DevelopmentGateError("complete_same_source_query_device_single_seed_screening_required")
    if len({digest(r["inputs"]["dependencies"]["source_f0"]) for r in rows}) != 1:
        raise DevelopmentGateError("screening_settings_must_share_identical_source_f0")
    winner = min(rows, key=lambda r: (-r["participant_mean_ba"], r["participant_mean_bce"],
        r["inputs"].get("selected_meta_outer_update") or 0, r["job"]["configuration_index"]))
    result = {"schema_version": SCHEMA, "status": "DEVELOPMENT_CONFIGURATION_FROZEN", "method": method,
        "configuration_index": winner["job"]["configuration_index"], "configuration": winner["configuration"],
        "screening_seed": SEEDS[0], "screening_receipts": screening_receipts,
        "selected_screening_receipt": screening_receipts[rows.index(winner)],
        "catalog": winner["catalog"], "code": winner["inputs"]["code"],
        "configuration_tie_rule": CONFIGURATION_TIE_RULE, "synthetic": False, "external_access": False}
    if execute:
        if output is None or not output.startswith("runs/"):
            raise DevelopmentGateError("explicit_immutable_selection_output_required")
        path = _local(root, output, exists=False); path.parent.mkdir(parents=True, exist_ok=True)
        _put(path, result)
    return result


def verify_configuration_receipt(root, descriptor):
    root = Path(root).resolve(strict=True)
    value = _json(_reference(root, descriptor, suffix=".json"))
    if value.get("status") != "DEVELOPMENT_CONFIGURATION_FROZEN" or value.get("synthetic") is not False:
        raise DevelopmentGateError("real_immutable_screening_configuration_receipt_required")
    rebuilt = freeze_screening_configuration(root, method=value["method"], screening_receipts=value["screening_receipts"])
    if rebuilt != value:
        raise DevelopmentGateError("frozen_configuration_evidence_or_selection_changed")
    return value


def select_strongest_control(root, *, configuration_receipts, confirmation_receipts, output=None, execute=False):
    """All B1..B11, all three seeds. No internal/external results accepted."""
    root = Path(root).resolve(strict=True)
    if set(configuration_receipts) != set(CONTROLS) or len(confirmation_receipts) != 33:
        raise DevelopmentGateError("eleven_controls_times_three_complete_pipelines_required")
    selections = {m: verify_configuration_receipt(root, d) for m, d in configuration_receipts.items()}
    if any(value.get("method") != method for method, value in selections.items()):
        raise DevelopmentGateError("independent_same_method_screening_selection_required")
    values = [verify_development_receipt(root, d) for d in confirmation_receipts]
    if len({(r["method"], r["seed"]) for r in values}) != 33:
        raise DevelopmentGateError("unique_control_seed_confirmation_coverage_required")
    common, sources, score_rows = set(), {}, []
    for descriptor, row in zip(confirmation_receipts, values):
        method, seed = row["method"], row["seed"]
        if method not in CONTROLS or seed not in SEEDS:
            raise DevelopmentGateError("only_registered_control_confirmation_seeds_accepted")
        frozen = selections[method]
        if (row["configuration"] != frozen["configuration"] or row["job"]["configuration_index"] != frozen["configuration_index"]
                or (seed == SEEDS[0] and descriptor != frozen["selected_screening_receipt"])
                or (seed != SEEDS[0] and row["inputs"]["configuration_receipt"] != configuration_receipts[method])):
            raise DevelopmentGateError("confirmation_must_reuse_exact_frozen_configuration_and_screen_seed")
        common.add(digest({"catalog": row["catalog"], "device": row["inputs"].get("device_identity_sha256"),
                           "timing_protocol": row["timing_protocol"]}))
        sources.setdefault(seed, set()).add(digest(row["inputs"]["dependencies"]["source_f0"]))
        score_rows.append({"role": "meta_validation", "measured": True, "method": method, "seed": seed,
            "participant_mean_ba": row["participant_mean_ba"], "calibration_median_seconds": row["calibration_median_seconds"],
            "device_identity_sha256": row["inputs"].get("device_identity_sha256")})
    if len(common) != 1 or any(len(v) != 1 for v in sources.values()):
        raise DevelopmentGateError("all_controls_require_same_query_device_and_same_seed_source")
    selected = choose_strongest_control(score_rows)
    # Explicit implementation convention, not a verbatim protocol sentence:
    # the existing selector uses the median of three pipeline session medians.
    result = {"schema_version": SCHEMA, "status": "STRONGEST_DEVELOPMENT_CONTROL_FROZEN",
        "synthetic": False, "selected": selected, "configuration_receipts": configuration_receipts,
        "confirmation_receipts": confirmation_receipts, "evidence_rows": score_rows,
        "BA_tolerance": .005, "timing_statistic": "median_of_three_pipeline_session_medians",
        "B0_separately_reported": True, "primary_method": "P", "PCT_primary_weight": 0,
        "does_not_complete_internal_check_or_W6": True, "external_access": False}
    if execute:
        if output is None or not output.startswith("runs/"):
            raise DevelopmentGateError("explicit_immutable_control_selection_output_required")
        path = _local(root, output, exists=False); path.parent.mkdir(parents=True, exist_ok=True)
        _put(path, result)
    return result


PROFILE_SCHEMA = "paper3-development-calibration-profile-v1"
PROFILE_RULE = "first_frozen_catalog_order_query_eligible_B20_valid_support_both_classes"
_PROFILE_SEAL = object()


@dataclass(frozen=True)
class VerifiedCalibrationProfile:
    receipt: dict
    identity_sha256: str
    runtime_profile: dict
    methods: tuple
    maximum_calibration_seconds: dict
    synthetic: bool
    _seal: object


def _ref(value):
    return {key: value[key] for key in ("path", "sha256")}


def build_calibration_profile_plan(methods=None):
    """Finite resource measurement, separate from nine-grid scientific selection.

    A declared subset contains every setting of each requested method. No
    timing or scientific score is invented, and this plan opens no dataset.
    """
    methods = tuple(METHODS if methods is None else methods)
    if (not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods)
            or methods != tuple(m for m in METHODS if m in methods)):
        raise DevelopmentGateError("unique_registered_order_profile_methods_required")
    jobs = [DevelopmentJob(method, index) for method in methods
            for index in range(len(registry.registered_grid(method)))]
    value = {"schema_version": PROFILE_SCHEMA, "purpose": "calibration_engineering_profile",
        "methods": list(methods), "seed": SEEDS[0], "support_selection_rule": PROFILE_RULE,
        "jobs": [{"job": asdict(j), "configuration": j.configuration, "job_sha256": j.identity} for j in jobs],
        "registered_calibrations": len(jobs), "maximum_distinct_settings": 118,
        "query_predictions_computed": False, "query_labels_accessed": False,
        "scientific_selection_allowed": False, "timing_protocol": TIMING_PROTOCOL,
        "warmup_source_forward_calls_per_actual_attempt": WARMUP_FORWARDS,
        "warmup_target_optimizer_updates": 0, "measurement_samples_per_configuration": 1,
        "percentile_interpretation": "n=1 p50=p90 is the single observed calibration, not a repeated-fit distribution",
        "timing_scope": TIMING_SCOPE, "final_protocol_freeze_sha256": None,
        "requires_explicit_wall_GPU_per_calibration_and_fit_attempt_limits": True}
    value["plan_sha256"] = digest(value)
    return value


def _production_runtime():
    """One actual user-started GPU; never a CPU substitute or server action."""
    if not torch.cuda.is_available():
        raise DevelopmentGateError("real_development_requires_CUDA_no_CPU_fallback")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise DevelopmentGateError("registered_deterministic_CUBLAS_workspace_required")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", ""))
    if not uuid or uuid.lower() in {"none", "unavailable"}:
        raise DevelopmentGateError("physical_GPU_identity_required_for_timing")
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        raise DevelopmentGateError("actual_GPU_driver_identity_required") from None
    if not driver:
        raise DevelopmentGateError("actual_GPU_driver_identity_required")
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info
    pools = [{k: row.get(k) for k in ("internal_api", "prefix", "num_threads", "version", "architecture")}
        for row in threadpool_info()]
    pools.sort(key=lambda row: (str(row["prefix"]), str(row["internal_api"])))
    return {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver_versions": driver,
        "device_name": props.name, "device_uuid": uuid, "compute_capability": [props.major, props.minor],
        "device_total_memory_bytes": props.total_memory,
        "CPU_machine": platform.machine(), "CPU_processor": platform.processor(),
        "torch_CPU_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "CPU_BLAS_threadpools": pools,
        "neural_device": "cuda:0", "classical_device": "existing_scipy_sklearn_CPU_kernels_same_GPU_host",
        "tf32": False, "autocast": False, "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "timing_protocol": TIMING_PROTOCOL, "timing_scope": TIMING_SCOPE,
        "cuda_event_metric": "elapsed_stream_interval_including_host_wait_not_active_kernel_compute"}


def _on_cuda(context):
    context.source.to("cuda:0")
    geometry = context.geometry
    object.__setattr__(context, "geometry", SpatialGeometry(*(v.to("cuda:0")
        for v in (geometry.q, geometry.rref_sqrt, geometry.rref_invsqrt, geometry.e))))
    context._assert_current()
    if context.geometry.q.device.type != "cuda" or any(p.device.type != "cuda" for p in context.source.parameters()):
        raise DevelopmentGateError("real_neural_context_cannot_use_CPU_fallback")


def _input_summary(context):
    return {"signal_contract_sha256": context.signal_contract_sha256, "split_sha256": context.split_sha256,
        "source_cache_sha256": context.source_cache_sha256, "source_interface_sha256": context.source_interface_sha256,
        "source_training_UID_set_sha256": context.reference.events_sha256,
        "source_feature_dim": context.source.feature_dim, "reference_array_sha256": context.reference_array_sha256,
        "geometry_sha256": context.geometry_sha256, "final_protocol_freeze_sha256": None}


def _validate_dependency_parameters(root, context, job, dependencies):
    selected = None
    if job.method in registry.LEARNED_METHODS:
        if dependencies.get("learned") is None:
            raise DevelopmentGateError("independent_learned_artifact_required")
        selected = context._learned(job.method, job.configuration, dependencies["learned"]["path"]).selected_outer_update
    if job.method in registry.CLASSICAL_METHODS:
        registry.load_verified_source_bank(root, dependencies["source_bank"], method=job.method,
            expected_m=job.configuration.get("m"), expected_shrink=job.configuration.get("shrinkage"),
            input_sha=context.signal_contract_sha256, cache_sha=context.source_cache_sha256)
    return selected


def _profile_dependencies(root, plan, artifact_catalog):
    jobs = [DevelopmentJob(**row["job"]) for row in plan["jobs"]]
    dependencies = {j.identity: _dependencies(root, j, artifact_catalog) for j in jobs}
    if len({digest({k: row[k] for k in ("source_f0", "source_reference")}) for row in dependencies.values()}) != 1:
        raise DevelopmentGateError("profile_settings_must_share_same_screening_source_and_reference")
    return jobs, dependencies


def _profile_session(catalog):
    # W3's audited fallback records encode missing-class support. Selection uses
    # metadata order and support eligibility, never query scores or labels.
    for plan in catalog.plans:
        if (plan.query_eligible and len(plan.acquired_event_uids) == 20 and len(plan.support_event_uids) >= 2
                and plan.fallback_reason is None):
            if not hasattr(catalog, "materialize_support"):
                raise DevelopmentGateError("public_support_only_catalog_interface_required")
            support = catalog.materialize_support(plan, device="cpu")
            if (support.role != "meta_validation_support" or support.event_uids != plan.support_event_uids
                    or support.acquired_event_uids != plan.acquired_event_uids
                    or not all(bool((support.y == c).any()) for c in (0, 1))):
                raise DevelopmentGateError("profile_metadata_and_actual_B20_support_disagree")
            return plan, support
    raise DevelopmentGateError("fixed_validation_catalog_has_no_eligible_two_class_B20_profile_session")


def _read_budget(root, descriptor, *, stage, expected_identity):
    value = _json(_reference(root, descriptor, suffix=".json"))
    if (value.get("stage") != stage or value.get("job_sha256" if stage == "development_evaluation" else "plan_sha256") != expected_identity
            or value.get("max_jobs") != 1 or value.get("max_download_bytes") != 0
            or type(value.get("max_fit_attempts")) is not int or value["max_fit_attempts"] < 1
            or not all(_positive(value.get(k)) for k in ("max_wall_seconds", "max_gpu_seconds", "max_calibration_seconds"))
            or value["max_gpu_seconds"] < value["max_wall_seconds"]):
        raise DevelopmentGateError("explicit_exact_job_wall_GPU_attempt_and_calibration_budget_required")
    if stage == "development_calibration_profile" and value["max_fit_attempts"] > 118:
        raise DevelopmentGateError("calibration_profile_at_most_118_actual_attempts_per_declared_run")
    return value


def _initial_budget(root, output, limits):
    target = _local(root, output+"/initial_budget.json", exists=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        _put(target, limits)
    else:
        original = _json(target)
        for key in ("stage", "job_sha256", "plan_sha256", "max_jobs", "max_download_bytes", "max_calibration_seconds"):
            if original.get(key) != limits.get(key):
                raise DevelopmentGateError("execution_budget_constraints_changed_use_new_branch")
    return _descriptor(root, target)


@contextmanager
def _calibration_deadline(seconds):
    if not _positive(seconds) or not hasattr(signal, "setitimer"):
        raise DevelopmentGateError("positive_main_thread_calibration_deadline_required")
    previous = signal.getitimer(signal.ITIMER_REAL)
    if previous != (0.0, 0.0):
        raise DevelopmentGateError("nested_calibration_deadlines_forbidden")
    def expired(signum, frame):
        raise TimeoutError("calibration_wall_deadline_exceeded")
    old_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def candidate_identity(context, job, plan, *, purpose, execution_identity_sha256, initial_budget, attempt):
    """Fixed producer construction; no label or final W7 authority is present."""
    if purpose not in {"calibration_engineering_profile", "development_evaluation"} or type(attempt) is not int or attempt < 1:
        raise DevelopmentGateError("registered_candidate_purpose_and_durable_attempt_required")
    return {"schema_version": "paper3-development-candidate-v1", "purpose": purpose,
        "method": job.method, "pipeline_seed": job.seed, "phase": job.phase,
        "configuration_index": job.configuration_index, "configuration": job.configuration,
        "job_sha256": job.identity, "execution_identity_sha256": execution_identity_sha256,
        "participant_id": plan.participant_id, "session_id": plan.session_id,
        "episode_plan_sha256": plan.temporal_audit_sha256,
        "catalog_manifest_sha256": plan.manifest_sha256, "split_sha256": plan.split_sha256,
        "support_role": "meta_validation_support", "acquired_event_uids": list(plan.acquired_event_uids),
        "support_event_uids": list(plan.support_event_uids), "query_event_uids": list(plan.query_event_uids),
        "signal_contract_sha256": context.signal_contract_sha256,
        "source_context_identity_sha256": context.identity_sha256,
        "initial_budget": initial_budget, "attempt": attempt,
        "timing_protocol": TIMING_PROTOCOL, "final_protocol_freeze_sha256": None}


def _fit_statistics(fitted, plan):
    payload = fitted.payload
    if type(payload) is registry.classical.ClassicalModel:
        receipt = payload.receipt
        updates, parameters = receipt["target_optimizer_updates"], receipt["target_trainable_parameters"]
        fallback = receipt.get("fallback_reason")
        solves, evaluations = receipt.get("target_exact_solves", 0), receipt.get("function_evaluations", 0)
    else:
        updates, parameters, fallback = payload.updates, payload.trainable_parameters, payload.fallback_reason
        solves, evaluations = 0, 0
    if any(type(v) is not int or v < 0 for v in (updates, parameters, solves, evaluations)):
        raise DevelopmentGateError("actual_fit_statistics_required")
    if fitted.method == "B0":
        if updates or parameters:
            raise DevelopmentGateError("frozen_control_has_no_target_updates_or_parameters")
    elif fallback is None and fitted.method not in registry.CLASSICAL_METHODS and updates != 10:
        raise DevelopmentGateError("registered_neural_calibration_requires_ten_actual_updates")
    elif fitted.method == "B4" and fallback is None and (updates != 0 or solves != 1):
        raise DevelopmentGateError("CSP_exact_solve_not_optimizer_updates")
    elif fitted.method == "B5" and updates > 200:
        raise DevelopmentGateError("TS_calibration_exceeds_registered_iteration_budget")
    labels = [] if fitted.method == "B0" else list(plan.support_event_uids)
    return {"acquired_calibration_trials": 20, "valid_support_trials": len(plan.support_event_uids),
        "accessible_labels": len(labels), "label_access_event_uids": labels,
        "optimizer_updates": updates, "target_exact_solves": solves, "target_function_evaluations": evaluations,
        "fallback_reason": fallback, "target_trainable_parameters": parameters,
        "unused_acquired_trials": 20-len(plan.support_event_uids), "audit_trials": 0,
        "acquisition_elapsed_seconds": None}


def _candidate_load(root, directory, context, job, plan, candidate, runtime, *, software_test_double=False,
                    allow_pointer_recovery=False):
    """Verify and reissue existing parameters, without fitting or restoring RNG."""
    from .fit_artifacts import load_registered_fit
    base = _local(root, directory)
    state_path = _local(root, directory+"/candidate_state.json", exists=False)
    model_path = _local(root, directory+"/fitted/receipt.json")
    timing_path, cost_path = _local(root, directory+"/timing.json"), _local(root, directory+"/cost.json")
    warmup_path = _local(root, directory+"/warmup.json")
    model, timing_ref, cost_ref = _descriptor(root, model_path), _descriptor(root, timing_path), _descriptor(root, cost_path)
    expected = {"schema_version": SCHEMA, "status": "FIT_PARAMETERS_COMMITTED",
        "synthetic": software_test_double, "candidate_identity": candidate,
        "candidate_identity_sha256": digest(candidate), "parameters": model, "timing": timing_ref, "cost": cost_ref,
        "warmup": _descriptor(root, warmup_path),
        "query_labels_accessed": False, "final_protocol_freeze_sha256": None}
    if state_path.exists():
        if _json(state_path) != expected:
            raise DevelopmentGateError("candidate_checkpoint_state_or_dependencies_changed")
    elif not allow_pointer_recovery:
        raise DevelopmentGateError("missing_candidate_state_requires_explicit_producer_recovery")
    fitted = load_registered_fit(context, model, expected_method=job.method,
        expected_configuration=job.configuration, expected_support_role="meta_validation_support",
        expected_acquired_event_uids=plan.acquired_event_uids, expected_support_event_uids=plan.support_event_uids,
        candidate_identity=candidate, software_test_double=software_test_double)
    timing, cost = _json(timing_path), _json(cost_path)
    warmup = _json(warmup_path)
    if (warmup.get("status") != "SUPPORT_SOURCE_FORWARD_WARMUP_COMPLETE" or warmup.get("synthetic") is not software_test_double
            or warmup.get("candidate_identity_sha256") != digest(candidate) or warmup.get("runtime_profile") != runtime
            or warmup.get("source_context_identity_sha256") != context.identity_sha256
            or warmup.get("signal_contract_sha256") != context.signal_contract_sha256
            or warmup.get("support_event_uids") != list(plan.support_event_uids)
            or warmup.get("acquired_event_uids") != list(plan.acquired_event_uids)
            or warmup.get("source_forward_calls") != WARMUP_FORWARDS or warmup.get("target_optimizer_updates") != 0
            or warmup.get("query_labels_accessed") is not False or warmup.get("final_protocol_freeze_sha256") is not None
            or not _positive(warmup.get("wall_seconds"))):
        raise DevelopmentGateError("fixed_support_only_source_warmup_proof_required")
    statistics_row = _fit_statistics(fitted, plan)
    if any(cost.get(k) != v for k, v in statistics_row.items()):
        raise DevelopmentGateError("candidate_actual_update_paid_label_or_parameter_ledger_changed")
    if (timing.get("schema_version") != SCHEMA or timing.get("status") != "CALIBRATION_MEASURED"
            or timing.get("synthetic") is not software_test_double or timing.get("candidate_identity_sha256") != digest(candidate)
            or timing.get("runtime_profile") != runtime or timing.get("timing_protocol") != TIMING_PROTOCOL
            or timing.get("final_protocol_freeze_sha256") is not None
            or timing.get("query_labels_accessed") is not False or not _positive(timing.get("calibration_wall_seconds"))
            or cost.get("calibration_wall_seconds") != timing["calibration_wall_seconds"]
            or timing.get("measurement_samples") != 1
            or timing.get("p50_calibration_wall_seconds") != timing["calibration_wall_seconds"]
            or timing.get("p90_calibration_wall_seconds") != timing["calibration_wall_seconds"]
            or not _positive(timing.get("calibration_deadline_seconds"))
            or timing["calibration_wall_seconds"] > timing["calibration_deadline_seconds"]
            or type(timing.get("peak_GPU_allocated_bytes")) is not int or timing["peak_GPU_allocated_bytes"] < 0
            or type(timing.get("peak_GPU_reserved_bytes")) is not int
            or timing.get("GPU_memory_scope") != GPU_MEMORY_SCOPE
            or type(runtime.get("device_total_memory_bytes")) is not int or runtime["device_total_memory_bytes"] <= 0
            or not timing["peak_GPU_allocated_bytes"] <= timing["peak_GPU_reserved_bytes"] <= runtime["device_total_memory_bytes"]
            or type(timing.get("cuda_event_elapsed_seconds")) not in (int, float)
            or not math.isfinite(timing["cuda_event_elapsed_seconds"]) or timing["cuda_event_elapsed_seconds"] < 0
            or timing["cuda_event_elapsed_seconds"] > timing["calibration_wall_seconds"]+1):
        raise DevelopmentGateError("candidate_measured_timing_proof_invalid")
    try:
        started = datetime.fromisoformat(timing["started_at_utc"])
        ended = datetime.fromisoformat(timing["finished_at_utc"])
        if started.utcoffset() is None or ended.utcoffset() is None or abs((ended-started).total_seconds()-timing["calibration_wall_seconds"]) > 5:
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise DevelopmentGateError("candidate_timing_UTC_and_monotonic_interval_disagree") from None
    _reference(root, candidate["initial_budget"], suffix=".json")
    if not state_path.exists():
        # Only an explicit producer may repair a kill between publication of
        # complete parameter bytes and its pointer. Read-only proof validators
        # require the originally committed pointer and never recreate files.
        _put(state_path, expected)
    return fitted, _descriptor(root, state_path), timing, cost


def _warmup_source(root, destination, context, support, candidate, runtime, wall_budget, maximum):
    remaining = min(maximum, wall_budget.remaining_seconds)
    if not _positive(remaining):
        raise DevelopmentGateError("wall_budget_exhausted_before_fixed_source_warmup")
    began = time.monotonic()
    with _calibration_deadline(remaining), torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        values = support.x.to("cuda:0")
        torch.cuda.synchronize()
        for _ in range(WARMUP_FORWARDS):
            context.source.source_logits(values)
        torch.cuda.synchronize()
        del values
    duration = time.monotonic()-began
    if duration > remaining or wall_budget.exhausted:
        raise DevelopmentGateError("warmup_wall_budget_exceeded")
    _put(destination / "warmup.json", {"status": "SUPPORT_SOURCE_FORWARD_WARMUP_COMPLETE", "synthetic": False,
        "candidate_identity_sha256": digest(candidate), "runtime_profile": runtime,
        "source_context_identity_sha256": context.identity_sha256,
        "signal_contract_sha256": context.signal_contract_sha256,
        "support_event_uids": list(support.event_uids), "acquired_event_uids": list(support.acquired_event_uids),
        "source_forward_calls": WARMUP_FORWARDS, "target_optimizer_updates": 0,
        "wall_seconds": duration, "query_labels_accessed": False, "final_protocol_freeze_sha256": None})


def _candidate_exists(root, directory):
    return any(_local(root, directory+suffix, exists=False).exists()
               for suffix in ("/candidate_state.json", "/fitted/receipt.json"))


def _measure_candidate(root, directory, context, job, catalog, plan, candidate, runtime, *, deadline_seconds,
                       wall_budget, software_test_double=False):
    """Actual CUDA-host producer. No query tensor, label capability or score."""
    if _candidate_exists(root, directory):
        return _candidate_load(root, directory, context, job, plan, candidate, runtime,
            software_test_double=software_test_double, allow_pointer_recovery=True)
    if software_test_double:
        raise DevelopmentGateError("synthetic_math_fixture_must_supply_existing_parameters_no_production_clock_substitute")
    from .fit_artifacts import save_registered_fit
    if context.geometry.q.device.type != "cuda" or runtime.get("neural_device") != "cuda:0":
        raise DevelopmentGateError("real_calibration_producer_requires_actual_CUDA")
    support = catalog.materialize_support(plan, device="cpu")
    destination = _local(root, directory, exists=False)
    destination.mkdir(parents=True, exist_ok=True)
    _warmup_source(root, destination, context, support, candidate, runtime, wall_budget, deadline_seconds)
    remaining = wall_budget.remaining_seconds
    limit = min(deadline_seconds, remaining)
    if not _positive(limit) or wall_budget.exhausted:
        raise DevelopmentGateError("wall_budget_exhausted_before_calibration")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    try:
        with _calibration_deadline(limit), torch.autocast(device_type="cuda", enabled=False):
            begin.record()
            if job.method not in registry.CLASSICAL_METHODS:
                support = CalibrationPrefix(support.x.to("cuda:0"), support.y.to("cuda:0"), support.event_uids,
                    support.acquired_event_uids, support.protocol_freeze_sha, role=support.role)
            deps = candidate["dependencies"]
            fitted = context.fit_development(job.method, support, job.configuration,
                learned_receipt=deps["learned"]["path"] if deps["learned"] else None, classic_model=deps["source_bank"])
            end.record()
            torch.cuda.synchronize()
        elapsed = time.monotonic()-start
        ended_utc = datetime.now(timezone.utc).isoformat()
        event_elapsed = begin.elapsed_time(end)/1000
        # A synchronization/dependency call that crossed the deadline cannot
        # manufacture a successful profile or scientific candidate on return.
        if elapsed > limit or wall_budget.exhausted:
            raise DevelopmentGateError("measured_calibration_or_cumulative_wall_limit_exceeded")
        timing = {"schema_version": SCHEMA, "status": "CALIBRATION_MEASURED", "synthetic": False,
            "candidate_identity_sha256": digest(candidate), "runtime_profile": runtime,
            "timing_protocol": TIMING_PROTOCOL, "timing_scope": TIMING_SCOPE,
            "started_at_utc": started_utc, "finished_at_utc": ended_utc,
            "calibration_wall_seconds": elapsed, "calibration_deadline_seconds": limit,
            "measurement_samples": 1, "p50_calibration_wall_seconds": elapsed, "p90_calibration_wall_seconds": elapsed,
            "cuda_event_elapsed_seconds": event_elapsed, "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(),
            "GPU_memory_scope": GPU_MEMORY_SCOPE,
            "query_labels_accessed": False, "final_protocol_freeze_sha256": None}
        cost = {**_fit_statistics(fitted, plan), "calibration_wall_seconds": elapsed}
        _put(destination / "timing.json", timing); _put(destination / "cost.json", cost)
        save_registered_fit(context, fitted, directory+"/fitted", candidate_identity=candidate)
        if wall_budget.exhausted:
            raise DevelopmentGateError("wall_budget_exceeded_after_parameter_commit")
        return _candidate_load(root, directory, context, job, plan, candidate, runtime, allow_pointer_recovery=True)
    except Exception as error:
        _put(destination / "attempt_failure.json", {"status": "CALIBRATION_ATTEMPT_NOT_QUALIFIED",
            "candidate_identity_sha256": digest(candidate), "error_type": type(error).__name__,
            "observed_wall_seconds": max(0., time.monotonic()-start), "scientific_fit_completed": False})
        raise


def _profile_metadata(root, methods, artifact_catalog, runtime):
    plan = build_calibration_profile_plan(methods)
    jobs, dependencies = _profile_dependencies(root, plan, artifact_catalog)
    tasks = _json(_local(root, "status/tasks.json"))
    if any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3")):
        raise DevelopmentGateError("real_complete_W1_W2_W3_required")
    contract = _json(_local(root, "contracts/input_contract.json"))
    signal_sha = validate_signal_contract(contract)
    for stage in ("W2", "W3"):
        _reference(root, tasks[stage]["cache_manifest"], suffix=".json")
    software = build_development_plan(root)
    return {"schema_version": PROFILE_SCHEMA, "plan": plan, "dependencies": dependencies,
        "signal_contract_sha256": signal_sha, "W2_cache": tasks["W2"]["cache_manifest"],
        "W3_cache": tasks["W3"]["cache_manifest"], "runtime_profile": runtime,
        "code": software["code"], "policy": software["policy"],
        "final_protocol_freeze_sha256": None}, jobs


def verify_calibration_profile_inputs(root, *, artifact_catalog, methods=None):
    """Read-only real source/dev proof and fixed support choice; no CUDA or fit."""
    root = Path(root).resolve(strict=True)
    metadata, _ = _profile_metadata(root, methods, artifact_catalog, {"status": "GPU_NOT_INITIALIZED_READ_ONLY"})
    context, catalog, jobs, plan, support = _verified_profile_context(root, metadata)
    return {"status": "PROFILE_INPUTS_VERIFIED_NOT_EXECUTED", "stages_verified": True,
        "registered_calibrations": len(jobs), "input_summary": _input_summary(context),
        "catalog": _catalog_binding(catalog), "profile_session": plan.to_dict(),
        "valid_support_trials": len(support.event_uids), "acquired_calibration_trials": 20,
        "query_predictions_computed": False, "query_labels_accessed": False,
        "training_started": False, "CUDA_checked": False, "external_access": False,
        "server_actions": False, "final_protocol_freeze_sha256": None}


def _verified_profile_context(root, identity):
    """Full source/ref/learned/bank/corpus proof, with no fit or label scoring."""
    plan = build_calibration_profile_plan(identity["plan"]["methods"])
    if plan != identity["plan"]:
        raise DevelopmentGateError("profile_plan_is_not_the_exact_registered_finite_grid")
    software = build_development_plan(root)
    if identity["code"] != software["code"] or identity["policy"] != software["policy"]:
        raise DevelopmentGateError("profile_code_or_protocol_changed")
    tasks = _json(_local(root, "status/tasks.json"))
    if (any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3"))
            or tasks["W2"]["cache_manifest"] != identity["W2_cache"]
            or tasks["W3"]["cache_manifest"] != identity["W3_cache"]):
        raise DevelopmentGateError("profile_real_cache_stage_identity_changed")
    contract = _json(_local(root, "contracts/input_contract.json"))
    if validate_signal_contract(contract) != identity["signal_contract_sha256"] or identity.get("final_protocol_freeze_sha256") is not None:
        raise DevelopmentGateError("profile_signal_identity_is_separate_from_final_W7_freeze")
    jobs = [DevelopmentJob(**row["job"]) for row in plan["jobs"]]
    if set(identity["dependencies"]) != {job.identity for job in jobs}:
        raise DevelopmentGateError("profile_requires_every_declared_setting_dependency")
    first = identity["dependencies"][jobs[0].identity]
    context = registry.Context.from_verified(root, first["source_f0"]["path"], first["source_reference"]["path"], contract)
    if context.pipeline_seed != SEEDS[0] or context.source_cache_sha256 != identity["W2_cache"]["sha256"]:
        raise DevelopmentGateError("profile_requires_actual_screening_seed_and_complete_W2_source")
    for job in jobs:
        deps = identity["dependencies"][job.identity]
        if any(deps.get(key) != first[key] for key in ("source_f0", "source_reference")):
            raise DevelopmentGateError("all_profile_grids_require_one_same_seed_source_reference")
        for value in deps.values():
            if value is not None:
                _reference(root, value, suffix=".json")
        if (job.method in registry.LEARNED_METHODS) != (deps.get("learned") is not None):
            raise DevelopmentGateError("independent_learned_profile_artifact_coverage_mismatch")
        if (job.method in registry.CLASSICAL_METHODS) != (deps.get("source_bank") is not None):
            raise DevelopmentGateError("independent_classical_profile_bank_coverage_mismatch")
        _validate_dependency_parameters(root, context, job, deps)
    catalog = load_episode_catalog(root, role="meta_validation", purpose="model_selection")
    _catalog_binding(catalog)
    if context.split_sha256 != catalog.split_sha256 or context.signal_contract_sha256 != catalog.input_contract_sha256:
        raise DevelopmentGateError("profile_context_and_validation_permission_differ")
    profile_plan, support = _profile_session(catalog)
    return context, catalog, jobs, profile_plan, support


def _attempt_limit(state, limits, initial_limit=None):
    requested = limits["max_fit_attempts"]
    if "initial_fit_attempt_limit" not in state:
        initial = requested if initial_limit is None else initial_limit
        state.update(initial_fit_attempt_limit=initial, fit_attempt_limit=initial, fit_limit_amendments=[])
    limit = state["initial_fit_attempt_limit"]
    if type(limit) is not int or limit < 1 or not isinstance(state.get("fit_limit_amendments"), list):
        raise DevelopmentGateError("invalid_fit_attempt_budget_history")
    for amendment in state["fit_limit_amendments"]:
        if (amendment.get("previous") != limit or type(amendment.get("new")) is not int or amendment["new"] <= limit
                or not isinstance(amendment.get("note"), str) or not amendment["note"].strip()
                or amendment.get("scientific_identity_changed") is not False):
            raise DevelopmentGateError("invalid_fit_attempt_budget_amendment")
        limit = amendment["new"]
    if state.get("fit_attempt_limit") != limit:
        raise DevelopmentGateError("fit_attempt_limit_differs_from_explicit_amendments")
    if requested != limit:
        note = limits.get("operational_amendment_note")
        if requested <= limit or not isinstance(note, str) or not note.strip():
            raise DevelopmentGateError("larger_fit_attempt_limit_requires_explicit_operational_amendment")
        state["fit_limit_amendments"].append({"previous": limit, "new": requested, "note": note,
            "scientific_identity_changed": False})
        state["fit_attempt_limit"] = requested


def _gpu_limit_history(state, limits, initial_limit):
    if "initial_gpu_limit_seconds" not in state:
        state.update(initial_gpu_limit_seconds=initial_limit, gpu_limit_seconds=initial_limit, gpu_limit_amendments=[])
    previous = state["initial_gpu_limit_seconds"]
    if previous != initial_limit or not _positive(previous) or not isinstance(state.get("gpu_limit_amendments"), list):
        raise DevelopmentGateError("invalid_initial_GPU_limit_or_amendment_history")
    for amendment in state["gpu_limit_amendments"]:
        if (amendment.get("previous") != previous or not _positive(amendment.get("new")) or amendment["new"] <= previous
                or not isinstance(amendment.get("note"), str) or not amendment["note"].strip()
                or amendment.get("scientific_identity_changed") is not False):
            raise DevelopmentGateError("invalid_GPU_limit_operational_amendment")
        previous = amendment["new"]
    if state.get("gpu_limit_seconds") != previous:
        raise DevelopmentGateError("GPU_limit_differs_from_explicit_amendments")
    requested = limits["max_gpu_seconds"]
    if requested != previous:
        note = limits.get("operational_amendment_note")
        if requested <= previous or not isinstance(note, str) or not note.strip():
            raise DevelopmentGateError("larger_GPU_limit_requires_explicit_operational_amendment")
        state["gpu_limit_amendments"].append({"previous": previous, "new": requested, "note": note,
            "scientific_identity_changed": False})
        state["gpu_limit_seconds"] = requested


def _resource_history_gate(root, output, state, *, work_directory):
    wall = _local(root, output+"/wall_budget.json", exists=False)
    _local(root, output+"/wall_budget.json.lock", exists=False)
    observed = (state.get("wall_ledger_initialized") is True or state.get("fit_attempts_dispatched", 0) != 0
        or state.get("active") is not None or bool(state.get("completed"))
        or _local(root, output+"/"+work_directory, exists=False).exists()
        or _local(root, output+"/receipt.json", exists=False).exists())
    if observed and not wall.is_file():
        raise DevelopmentGateError("missing_wall_ledger_after_observed_work_no_budget_reset")
    return wall


def _open_wall_budget(path, identity_sha, limits, initial, *, started_at=None):
    floor = 0.
    if started_at is not None:
        prior = _json(path) if path.exists() else {}
        prior_wall = prior.get("durable_wall_seconds", 0.)
        if type(prior_wall) not in (int, float) or not math.isfinite(prior_wall) or prior_wall < 0:
            raise DevelopmentGateError("invalid_prior_wall_high_water_no_reset")
        floor = prior_wall+max(0., time.monotonic()-started_at)
    new = not path.exists()
    journal = PersistentWallBudget(path, identity_sha,
        initial["max_wall_seconds"] if new else limits["max_wall_seconds"],
        checkpoint_wall_seconds=floor, amendment_note=limits.get("operational_amendment_note"))
    if new and limits["max_wall_seconds"] != initial["max_wall_seconds"]:
        try:
            journal.amend_limit(limits["max_wall_seconds"], limits.get("operational_amendment_note"))
        except BaseException:
            journal.close()
            raise
    return journal


def _completion_record(root, output, journal, state, limits, initial_budget, *, schema):
    report = journal.report()
    snapshot = _json(journal.path)
    path = _local(root, output+"/wall_journal_snapshot-"+digest(snapshot)+".json", exists=False)
    _put(path, snapshot)
    return {"schema_version": schema, "identity_sha256": state["identity_sha256"],
        "initial_budget": initial_budget, "wall_budget": report, "wall_journal_snapshot": _descriptor(root, path),
        "max_gpu_seconds": limits["max_gpu_seconds"], "fit_attempts_dispatched": state["fit_attempts_dispatched"],
        **{k: state[k] for k in ("initial_fit_attempt_limit", "fit_attempt_limit", "fit_limit_amendments",
            "initial_gpu_limit_seconds", "gpu_limit_seconds", "gpu_limit_amendments")}}


def _artifact_manifest(root, output, descriptors):
    files = {}
    for item in descriptors:
        path = _reference(root, _ref(item))
        if not path.is_relative_to(_local(root, output)):
            raise DevelopmentGateError("output_manifest_member_outside_owned_execution_directory")
        files[item["path"]] = {"path": item["path"], "sha256": item["sha256"], "bytes": path.stat().st_size,
            "role": "result", "dataset": "Stieger2021"}
    return {"schema_version": 1, "files": [files[k] for k in sorted(files)]}


def _publish_completion(root, output, result, references, journal):
    """Hash payloads, check budget, publish manifest, then final receipt LAST.

    An interrupted, uncommitted manifest has a retained immutable version. It
    may be replaced on an explicitly resumed run; a committed receipt is never
    overwritten. Prediction/candidate artifacts themselves remain immutable.
    """
    directory = _local(root, output)
    receipt_path = _local(root, output+"/receipt.json", exists=False)
    if receipt_path.exists():
        raise DevelopmentGateError("completed_execution_receipt_cannot_be_overwritten")
    manifest = _artifact_manifest(root, output, references)
    journal.checkpoint()
    if journal.exhausted:
        raise DevelopmentGateError("wall_budget_exceeded_before_final_publication")
    result["publication_wall_seconds"] = journal.wall_seconds
    body = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    manifest["files"].append({"path": output+"/receipt.json", "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body), "role": "result", "dataset": "Stieger2021"})
    manifest["files"].sort(key=lambda row: row["path"])
    manifest_path = _local(root, output+"/artifact_manifest.json", exists=False)
    if manifest_path.exists():
        previous = _json(manifest_path)
        owned_version = _local(root, output+"/.publication/"+digest(previous)+".json", exists=False)
        if not owned_version.is_file() or _json(owned_version) != previous:
            raise DevelopmentGateError("unowned_uncommitted_output_manifest_preserved")
    version = _local(root, output+"/.publication/"+digest(manifest)+".json", exists=False)
    version.parent.mkdir(parents=True, exist_ok=True)
    _put(version, manifest)
    atomic_json(manifest_path, manifest)
    if journal.exhausted:
        raise DevelopmentGateError("wall_budget_exceeded_after_manifest_before_final_status_commit")
    _write_identical_or_new(receipt_path, body)
    return _descriptor(root, receipt_path)


def _candidate_artifacts(root, descriptor):
    from .fit_artifacts import artifact_files
    state = _json(_reference(root, descriptor))
    return [descriptor, state["cost"], state["timing"], state["warmup"], *artifact_files(root, state["parameters"])]


def execute_calibration_profile(root, *, artifact_catalog, budget, methods=None, user_started_gpu=False, output=None):
    """Finite real calibration-only profile, after later explicit user GPU boot.

    Every requested method's complete registered grid runs once on the same
    metadata-selected legal B20 support. No query prediction or query label is
    requested. All interrupted fit attempts remain charged in a durable ledger.
    """
    invoked_at = time.monotonic()
    if user_started_gpu is not True:
        raise PermissionError("later_explicit_user_GPU_boot_required")
    root = Path(root).resolve(strict=True)
    plan = build_calibration_profile_plan(methods)
    limits = _read_budget(root, budget, stage="development_calibration_profile", expected_identity=plan["plan_sha256"])
    runtime = _production_runtime()
    identity, jobs = _profile_metadata(root, plan["methods"], artifact_catalog, runtime)
    identity_sha = digest(identity)
    output = output or "runs/development_profiles/"+identity_sha
    if not output.startswith("runs/"):
        raise DevelopmentGateError("profile_output_must_be_owned_runs_directory")
    destination = _local(root, output, exists=False)
    destination.mkdir(parents=True, exist_ok=True)
    with _local(root, output+"/.profile.lock", exists=False).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DevelopmentGateError("same_calibration_profile_already_running") from None
        _put(destination / "identity.json", identity)
        initial_budget = _initial_budget(root, output, limits)
        _resource_history_gate(root, output, {}, work_directory="fits")
        if (destination / "receipt.json").exists():
            saved = _descriptor(root, destination / "receipt.json")
            verify_calibration_profile(root, saved, expected_runtime=runtime)
            return {**_json(destination / "receipt.json"), "receipt": saved}
        progress_path = _local(root, output+"/progress.json", exists=False)
        if not progress_path.exists() and any((destination / name).exists() for name in ("fits", "wall_budget.json")):
            raise DevelopmentGateError("missing_profile_progress_after_observed_fit_cannot_refit")
        state = _json(progress_path) if progress_path.exists() else {
            "identity_sha256": identity_sha, "fit_attempts_dispatched": 0, "completed": {}, "active": None}
        if (state.get("identity_sha256") != identity_sha or type(state.get("fit_attempts_dispatched")) is not int
                or state["fit_attempts_dispatched"] < 0 or not isinstance(state.get("completed"), dict)
                or set(state["completed"]) - {j.identity for j in jobs}
                or len(state["completed"]) > state["fit_attempts_dispatched"]):
            raise DevelopmentGateError("invalid_profile_dispatch_checkpoint")
        wall_path = _resource_history_gate(root, output, state, work_directory="fits")
        initial_values = _json(_reference(root, initial_budget))
        _attempt_limit(state, limits, initial_values["max_fit_attempts"])
        _gpu_limit_history(state, limits, initial_values["max_gpu_seconds"])
        atomic_json(progress_path, state)
        journal = _open_wall_budget(wall_path, identity_sha, limits, initial_values, started_at=invoked_at)
        try:
            journal.start_interval()
            state["wall_ledger_initialized"] = True; atomic_json(progress_path, state)
            context, catalog, jobs, profile_plan, _ = _verified_profile_context(root, identity)
            if journal.exhausted:
                raise DevelopmentGateError("profile_wall_budget_exhausted_during_input_validation")
            _on_cuda(context)
            for job in jobs:
                if job.identity in state["completed"]:
                    record = state["completed"][job.identity]
                else:
                    active = state.get("active")
                    reuse = False
                    if active is not None:
                        expected_path = output+f"/fits/{job.identity}/attempt-{active.get('attempt', 0):06d}"
                        if (active.get("job_sha256") != job.identity or active.get("directory") != expected_path
                                or active.get("attempt") != state["fit_attempts_dispatched"]):
                            raise DevelopmentGateError("invalid_profile_active_attempt_identity")
                        reuse = _candidate_exists(root, active["directory"])
                    if not reuse:
                        if journal.exhausted or state["fit_attempts_dispatched"] >= state["fit_attempt_limit"]:
                            return {"status": "CALIBRATION_PROFILE_PAUSED_BUDGET", "synthetic": False,
                                "actual_calibrations_completed": len(state["completed"]),
                                "fit_attempts_dispatched": state["fit_attempts_dispatched"], "scientific_selection_allowed": False}
                        state["fit_attempts_dispatched"] += 1
                        attempt = state["fit_attempts_dispatched"]
                        active = {"job_sha256": job.identity, "attempt": attempt,
                            "directory": output+f"/fits/{job.identity}/attempt-{attempt:06d}", "phase": "FIT_DISPATCHED"}
                        state["active"] = active; atomic_json(progress_path, state)
                    candidate = candidate_identity(context, job, profile_plan, purpose="calibration_engineering_profile",
                        execution_identity_sha256=identity_sha, initial_budget=initial_budget, attempt=active["attempt"])
                    candidate["dependencies"] = identity["dependencies"][job.identity]
                    fitted, saved, _, cost = _measure_candidate(root, active["directory"], context, job, catalog,
                        profile_plan, candidate, runtime, deadline_seconds=limits["max_calibration_seconds"], wall_budget=journal)
                    if cost["fallback_reason"] is not None:
                        raise DevelopmentGateError("profile_requires_successful_calibration_on_registered_two_class_support")
                    record = {"job": asdict(job), "attempt": active["attempt"], "candidate_state": saved}
                    journal.checkpoint()
                    if journal.exhausted:
                        raise DevelopmentGateError("profile_wall_budget_exceeded_before_fit_commit")
                    state["completed"][job.identity] = record
                    state["active"] = None; atomic_json(progress_path, state)
                expected_candidate = candidate_identity(context, job, profile_plan, purpose="calibration_engineering_profile",
                    execution_identity_sha256=identity_sha, initial_budget=initial_budget, attempt=record["attempt"])
                expected_candidate["dependencies"] = identity["dependencies"][job.identity]
                directory = str(Path(record["candidate_state"]["path"]).parent)
                _candidate_load(root, directory, context, job, profile_plan, expected_candidate, runtime)
            journal.checkpoint()
            if journal.exhausted:
                raise DevelopmentGateError("profile_wall_budget_exceeded_before_completion")
            completion = _completion_record(root, output, journal, state, limits, initial_budget, schema=PROFILE_SCHEMA)
            completion_path = destination / ("completion_budget-"+digest(completion)+".json")
            _put(completion_path, completion)
            result = {"schema_version": PROFILE_SCHEMA, "status": "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE",
                "synthetic": False, "identity_sha256": identity_sha, "identity": _descriptor(root, destination / "identity.json"),
                "initial_budget": initial_budget, "catalog": _catalog_binding(catalog), "profile_session": profile_plan.to_dict(),
                "input_summary": _input_summary(context), "runtime_profile": runtime,
                "methods": plan["methods"], "records": [state["completed"][j.identity] for j in jobs],
                "actual_calibrations_completed": len(jobs), "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                "query_predictions_computed": False, "query_labels_accessed": False, "scientific_selection_allowed": False,
                "completion_budget": _descriptor(root, completion_path),
                "timing_protocol": TIMING_PROTOCOL, "final_protocol_freeze_sha256": None}
            # Hash/check all parameter members before the final status commit.
            # Re-verification after publication is a read-only consumer action,
            # not additional calibration work with an unreported budget.
            references = [result["identity"], initial_budget, result["completion_budget"], completion["wall_journal_snapshot"]]
            for row in result["records"]:
                references.extend(_candidate_artifacts(root, row["candidate_state"]))
            if journal.exhausted:
                raise DevelopmentGateError("profile_wall_budget_exceeded_during_final_artifact_hashing")
            descriptor = _publish_completion(root, output, result, references, journal)
            return {**result, "receipt": descriptor}
        finally:
            journal.close()


def verify_calibration_profile(root, descriptor, *, expected_runtime=None, software_test_double=False):
    """Full proof: exact grid + live dependencies + typed parameter bytes + time.

    No fit, query prediction, label reveal or GPU initialization happens here.
    A status/runtime JSON without all actual parameter artifacts cannot pass.
    Explicit synthetic verification returns a non-production capability.
    """
    root = Path(root).resolve(strict=True)
    if not isinstance(descriptor, dict) or not {"path", "sha256"} <= set(descriptor):
        raise DevelopmentGateError("complete_measured_profile_artifact_descriptor_required")
    path = _reference(root, _ref(descriptor), suffix=".json")
    result = _json(path)
    if (type(software_test_double) is not bool or result.get("schema_version") != PROFILE_SCHEMA
            or result.get("status") != "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE"
            or result.get("synthetic") is not software_test_double
            or any(result.get(k) is not False for k in ("query_predictions_computed", "query_labels_accessed", "scientific_selection_allowed"))
            or result.get("timing_protocol") != TIMING_PROTOCOL or result.get("final_protocol_freeze_sha256") is not None):
        raise DevelopmentGateError("complete_typed_calibration_profile_proof_required")
    if not {"identity", "identity_sha256", "runtime_profile", "catalog", "profile_session", "records", "initial_budget", "completion_budget"} <= set(result):
        raise DevelopmentGateError("full_SHA_profile_parameter_timing_and_budget_chain_required")
    identity = _json(_reference(root, result["identity"], suffix=".json"))
    if digest(identity) != result.get("identity_sha256") or identity.get("runtime_profile") != result.get("runtime_profile"):
        raise DevelopmentGateError("profile_identity_or_runtime_bytes_changed")
    if expected_runtime is not None and expected_runtime != result["runtime_profile"]:
        raise DevelopmentGateError("measured_profile_device_or_runtime_differs_from_current_GPU")
    runtime = result["runtime_profile"]
    if not software_test_double and (runtime.get("neural_device") != "cuda:0" or not runtime.get("device_uuid")
            or runtime.get("tf32") is not False or runtime.get("autocast") is not False
            or runtime.get("deterministic_algorithms") is not True or not runtime.get("driver_versions")):
        raise DevelopmentGateError("profile_requires_measured_same_GPU_noAMP_noTF32_runtime")
    context, catalog, jobs, plan, _ = _verified_profile_context(root, identity)
    if (result["catalog"] != _catalog_binding(catalog) or result["profile_session"] != json.loads(json.dumps(plan.to_dict()))
            or result["input_summary"] != _input_summary(context)
            or result.get("methods") != identity["plan"]["methods"]
            or result.get("actual_calibrations_completed") != len(jobs)
            or type(result.get("fit_attempts_dispatched")) is not int or result["fit_attempts_dispatched"] < len(jobs)):
        raise DevelopmentGateError("profile_session_grid_count_or_context_changed")
    initial = _json(_reference(root, result["initial_budget"], suffix=".json"))
    if (initial.get("stage") != "development_calibration_profile" or initial.get("plan_sha256") != identity["plan"]["plan_sha256"]
            or not _positive(initial.get("max_calibration_seconds"))):
        raise DevelopmentGateError("profile_initial_budget_not_bound_to_fixed_plan")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != len(jobs):
        raise DevelopmentGateError("complete_registered_profile_grid_artifacts_required")
    maxima, seen_attempts = {}, set()
    _verify_completion_budget(root, result, identity_sha=digest(identity))
    references = [result["identity"], result["initial_budget"], result["completion_budget"], _ref(descriptor),
        _json(_reference(root, result["completion_budget"]))["wall_journal_snapshot"]]
    for record, job in zip(records, jobs):
        attempt = record.get("attempt")
        if (record.get("job") != asdict(job) or type(attempt) is not int or not 0 < attempt <= result["fit_attempts_dispatched"]
                or attempt in seen_attempts):
            raise DevelopmentGateError("profile_grid_or_actual_attempt_identity_mismatch")
        seen_attempts.add(attempt)
        candidate = candidate_identity(context, job, plan, purpose="calibration_engineering_profile",
            execution_identity_sha256=digest(identity), initial_budget=result["initial_budget"], attempt=attempt)
        candidate["dependencies"] = identity["dependencies"][job.identity]
        expected_directory = path.parent.relative_to(root).as_posix()+f"/fits/{job.identity}/attempt-{attempt:06d}"
        if record["candidate_state"]["path"] != expected_directory+"/candidate_state.json":
            raise DevelopmentGateError("profile_parameter_receipt_outside_canonical_attempt")
        fitted, current_ref, timing, cost = _candidate_load(root, expected_directory, context, job, plan, candidate, runtime,
            software_test_double=software_test_double)
        if current_ref != record["candidate_state"] or cost["fallback_reason"] is not None:
            raise DevelopmentGateError("profile_fit_not_successfully_verified")
        if timing["calibration_deadline_seconds"] > initial["max_calibration_seconds"]:
            raise DevelopmentGateError("profile_fit_exceeded_explicit_initial_calibration_limit")
        maxima[job.method] = max(maxima.get(job.method, 0.), timing["calibration_wall_seconds"])
        references.extend(_candidate_artifacts(root, current_ref))
    output = path.parent.relative_to(root).as_posix()
    expected_manifest = _artifact_manifest(root, output, references)
    if _json(_local(root, output+"/artifact_manifest.json")) != expected_manifest:
        raise DevelopmentGateError("profile_explicit_full_SHA_artifact_manifest_incomplete")
    return VerifiedCalibrationProfile(_ref(descriptor), digest(identity), runtime, tuple(result["methods"]), maxima,
        software_test_double, _PROFILE_SEAL)


def _verify_completion_budget(root, result, *, identity_sha):
    record = _json(_reference(root, result["completion_budget"], suffix=".json"))
    wall = record.get("wall_budget", {})
    if (record.get("identity_sha256") != identity_sha or record.get("initial_budget") != result["initial_budget"]
            or wall.get("identity_sha256") != identity_sha or not _positive(wall.get("max_wall_seconds"))
            or type(wall.get("wall_seconds")) not in (int, float) or not math.isfinite(wall["wall_seconds"])
            or not 0 <= wall["wall_seconds"] < wall["max_wall_seconds"] or wall.get("blocked") is not False
            or wall.get("exhausted") is not False or not _positive(record.get("max_gpu_seconds"))
            or record["max_gpu_seconds"] < wall["max_wall_seconds"]
            or record.get("fit_attempts_dispatched") != result["fit_attempts_dispatched"]):
        raise DevelopmentGateError("completed_calibration_or_evaluation_must_fit_explicit_operational_budget")
    published = result.get("publication_wall_seconds")
    if (type(published) not in (int, float) or not math.isfinite(published)
            or not wall["wall_seconds"] <= published < wall["max_wall_seconds"]):
        raise DevelopmentGateError("final_publication_must_precede_explicit_wall_limit")
    initial = _json(_reference(root, result["initial_budget"]))
    if "wall_journal_snapshot" not in record:
        raise DevelopmentGateError("complete_durable_wall_journal_snapshot_required_no_self_reported_cap")
    snapshot = _json(_reference(root, record["wall_journal_snapshot"], suffix=".json"))
    checker = object.__new__(PersistentWallBudget)
    checker._state = snapshot
    try:
        checker._validate_state()  # Pure validation: no constructor/locks/clock settlement.
    except ValueError:
        raise DevelopmentGateError("invalid_full_wall_budget_history") from None
    if (snapshot.get("identity_sha256") != identity_sha
            or snapshot.get("initial_max_wall_seconds") != initial.get("max_wall_seconds")
            or snapshot.get("max_wall_seconds") != wall["max_wall_seconds"]
            or snapshot.get("durable_wall_seconds") != wall["wall_seconds"]
            or snapshot.get("clock_block") is not None
            or snapshot.get("checkpoint_wall_high_water_seconds") != wall.get("checkpoint_wall_high_water_seconds")):
        raise DevelopmentGateError("completion_wall_summary_must_match_initial_budget_and_durable_amendment_chain")
    gpu_keys = ("initial_gpu_limit_seconds", "gpu_limit_seconds", "gpu_limit_amendments")
    if not set(gpu_keys) <= set(record) or record.get("initial_gpu_limit_seconds") != initial.get("max_gpu_seconds"):
        raise DevelopmentGateError("completion_GPU_limit_must_bind_initial_budget")
    gpu_state = {k: record[k] for k in gpu_keys}
    _gpu_limit_history(gpu_state, {"max_gpu_seconds": record["max_gpu_seconds"]}, initial["max_gpu_seconds"])
    if record.get("initial_fit_attempt_limit") != initial.get("max_fit_attempts"):
        raise DevelopmentGateError("fit_attempt_initial_limit_changed")
    state = {k: record[k] for k in ("initial_fit_attempt_limit", "fit_attempt_limit", "fit_limit_amendments")}
    _attempt_limit(state, {"max_fit_attempts": record["fit_attempt_limit"]})
    if record["fit_attempts_dispatched"] > state["fit_attempt_limit"]:
        raise DevelopmentGateError("completed_execution_exceeds_explicit_fit_attempt_limit")


def _evaluation_metadata(root, job, artifact_catalog, profile, runtime, selection, *, software_test_double=False):
    deps = _dependencies(root, job, artifact_catalog)
    tasks = _json(_local(root, "status/tasks.json"))
    if any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3")):
        raise DevelopmentGateError("real_complete_W1_W2_W3_required")
    for stage in ("W2", "W3"):
        _reference(root, tasks[stage]["cache_manifest"], suffix=".json")
    contract = _json(_local(root, "contracts/input_contract.json"))
    software = build_development_plan(root)
    return {"schema_version": SCHEMA, "job": asdict(job), "configuration": job.configuration,
        "dependencies": deps, "profile_receipt": _ref(profile) if isinstance(profile, dict) else profile.receipt, "runtime_profile": runtime,
        "W2_cache": tasks["W2"]["cache_manifest"], "W3_cache": tasks["W3"]["cache_manifest"],
        "signal_contract_sha256": validate_signal_contract(contract),
        "code": software["code"], "policy": software["policy"], "configuration_receipt": selection,
        "configuration_tie_rule": CONFIGURATION_TIE_RULE, "final_protocol_freeze_sha256": None,
        "synthetic": software_test_double}


def _evaluation_context(root, identity):
    job = DevelopmentJob(**identity["job"])
    software = build_development_plan(root)
    if (identity.get("configuration") != job.configuration or identity.get("code") != software["code"]
            or identity.get("policy") != software["policy"] or identity.get("final_protocol_freeze_sha256") is not None):
        raise DevelopmentGateError("evaluation_registered_configuration_code_or_policy_changed")
    tasks = _json(_local(root, "status/tasks.json"))
    if (any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3"))
            or tasks["W2"]["cache_manifest"] != identity["W2_cache"]
            or tasks["W3"]["cache_manifest"] != identity["W3_cache"]):
        raise DevelopmentGateError("evaluation_requires_same_real_source_and_validation_cache")
    contract = _json(_local(root, "contracts/input_contract.json"))
    if validate_signal_contract(contract) != identity["signal_contract_sha256"]:
        raise DevelopmentGateError("evaluation_signal_input_changed")
    deps = identity["dependencies"]
    for value in deps.values():
        if value is not None:
            _reference(root, value, suffix=".json")
    context = registry.Context.from_verified(root, deps["source_f0"]["path"], deps["source_reference"]["path"], contract)
    if context.pipeline_seed != job.seed or context.source_cache_sha256 != identity["W2_cache"]["sha256"]:
        raise DevelopmentGateError("evaluation_source_pipeline_or_W2_cache_mismatch")
    if (job.method in registry.LEARNED_METHODS) != (deps.get("learned") is not None):
        raise DevelopmentGateError("evaluation_requires_own_independent_learned_artifact")
    if (job.method in registry.CLASSICAL_METHODS) != (deps.get("source_bank") is not None):
        raise DevelopmentGateError("evaluation_requires_exact_matching_new_classical_source_bank")
    selected_update = _validate_dependency_parameters(root, context, job, deps)
    catalog = load_episode_catalog(root, role="meta_validation", purpose="model_selection")
    binding = _catalog_binding(catalog)
    if context.split_sha256 != catalog.split_sha256 or context.signal_contract_sha256 != catalog.input_contract_sha256:
        raise DevelopmentGateError("evaluation_context_and_query_catalog_mismatch")
    if job.phase == "confirmation":
        selected = verify_configuration_receipt(root, identity["configuration_receipt"])
        if (selected["method"] != job.method or selected["configuration_index"] != job.configuration_index
                or selected["catalog"] != binding or selected["code"] != identity["code"]):
            raise DevelopmentGateError("confirmation_requires_unchanged_frozen_screening_configuration")
    elif identity.get("configuration_receipt") is not None:
        raise DevelopmentGateError("screening_cannot_inherit_an_outcome_selected_configuration")
    return context, catalog, {"job": asdict(job), "configuration": job.configuration,
        "dependencies": deps, "catalog": binding, "source_context_identity_sha256": context.identity_sha256,
        "input_summary": _input_summary(context), "selected_meta_outer_update": selected_update,
        "code": identity["code"], "policy": identity["policy"],
        "configuration_receipt": identity["configuration_receipt"], "configuration_tie_rule": CONFIGURATION_TIE_RULE,
        "profile_receipt": identity["profile_receipt"], "runtime_profile": identity["runtime_profile"],
        "device_identity_sha256": digest(identity["runtime_profile"]), "signal_contract_sha256": context.signal_contract_sha256,
        "final_protocol_freeze_sha256": None}


def _profile_matches_input(root, profile, context, *, software_test_double=False):
    if (type(profile) is not VerifiedCalibrationProfile or profile._seal is not _PROFILE_SEAL
            or profile.synthetic is not software_test_double):
        raise DevelopmentGateError("verified_real_profile_capability_required")
    result = _json(_reference(root, profile.receipt))
    if result["input_summary"] != _input_summary(context):
        raise DevelopmentGateError("profile_source_architecture_geometry_input_or_training_UIDs_differ")


def _prediction_binding(root, descriptor, context, job, plan, cost, configuration, model):
    from .prediction_store import COST_OPTIONAL, COST_REQUIRED
    value = verify_prediction_receipt(root, descriptor)
    expected_source = {"path": context.source_receipt_path, "sha256": context.source_receipt_sha256}
    if (value["method"] != job.method or value["pipeline_seed"] != job.seed
            or value["episode_plan_sha256"] != plan.temporal_audit_sha256
            or value["prediction_role"] != "meta_validation" or tuple(value["query_event_uids"]) != plan.query_event_uids
            or value["references"]["model"] != model or value["references"]["configuration"] != configuration
            or value["references"]["source_f0"] != expected_source
            or value["references"]["method_code"] != _descriptor(root, _local(root, "p3work/method_registry.py"))):
        raise DevelopmentGateError("prediction_source_model_configuration_or_candidate_dependency_mismatch")
    measured = {key: val for key, val in cost.items() if key in COST_REQUIRED | COST_OPTIONAL}
    if _json(_reference(root, descriptor).with_name("cost.json")) != measured:
        raise DevelopmentGateError("prediction_and_actual_fitted_calibration_ledgers_disagree")
    return value


def _predict_candidate(root, directory, context, job, catalog, plan, fitted, cost, configuration, *, software_test_double=False):
    from .prediction_store import COST_OPTIONAL, COST_REQUIRED, commit_predictions
    receipt_path = _local(root, directory+"/predictions/receipt.json", exists=False)
    if receipt_path.exists():
        descriptor = _descriptor(root, receipt_path)
        _prediction_binding(root, descriptor, context, job, plan, cost, configuration,
            _descriptor(root, _local(root, directory+"/fitted/receipt.json")))
        return descriptor
    episode = catalog.materialize(plan, device="cpu")
    query = episode.prediction_inputs
    if job.method not in registry.CLASSICAL_METHODS and not software_test_double:
        query = PredictionInputs(query.x.to("cuda:0"), query.event_uids, query.protocol_freeze_sha)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        predicted = context.predict(fitted, query)
    prediction = PredictionResult(*(t.detach().cpu() for t in (predicted.logits, predicted.probabilities, predicted.labels)),
        predicted.event_uids, predicted.protocol_freeze_sha)
    manifest = _json(_reference(root, _json(_local(root, "status/tasks.json"))["W3"]["cache_manifest"]))
    refs = {"source_f0": {"path": context.source_receipt_path, "sha256": context.source_receipt_sha256},
        "model": _descriptor(root, _local(root, directory+"/fitted/receipt.json")),
        "method_code": _descriptor(root, _local(root, "p3work/method_registry.py")), "configuration": configuration,
        "input_contract": manifest["input_contract"], "participant_split": manifest["participant_split"],
        "episode_manifest": _json(_local(root, "status/tasks.json"))["W3"]["cache_manifest"]}
    measured_cost = {key: value for key, value in cost.items() if key in COST_REQUIRED | COST_OPTIONAL}
    saved = commit_predictions(root, directory+"/predictions", prediction, plan, method=job.method,
        pipeline_seed=job.seed, references=refs, cost=measured_cost, purpose="development_evaluation")["receipt"]
    _prediction_binding(root, saved, context, job, plan, cost, configuration, refs["model"])
    return saved


def _execute_validation_production(root, *, job, artifact_catalog, limits, profile, runtime,
                                   configuration_receipt, output, software_test_double=False, candidate_producer=None,
                                   prediction_producer=None, invoked_at=None):
    invoked_at = time.monotonic() if invoked_at is None else invoked_at
    if type(software_test_double) is not bool or not software_test_double and (candidate_producer is not None or prediction_producer is not None):
        raise DevelopmentGateError("production_evaluation_cannot_accept_caller_fit_or_prediction_callbacks")
    identity = _evaluation_metadata(root, job, artifact_catalog, profile, runtime, configuration_receipt,
        software_test_double=software_test_double)
    identity_sha = digest(identity)
    output = output or "runs/development/"+identity_sha
    if not output.startswith("runs/"):
        raise DevelopmentGateError("evaluation_output_must_be_owned_runs_directory")
    destination = _local(root, output, exists=False)
    destination.mkdir(parents=True, exist_ok=True)
    with _local(root, output+"/.evaluation.lock", exists=False).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DevelopmentGateError("same_development_job_already_running") from None
        _put(destination / "identity.json", identity)
        initial_budget = _initial_budget(root, output, limits)
        _resource_history_gate(root, output, {}, work_directory="sessions")
        if (destination / "receipt.json").exists():
            saved = _descriptor(root, destination / "receipt.json")
            result = (verify_synthetic_development_receipt if software_test_double else verify_development_receipt)(root, saved)
            return {**result, "receipt": saved}
        progress_path = _local(root, output+"/progress.json", exists=False)
        if not progress_path.exists() and any((destination / name).exists() for name in ("sessions", "wall_budget.json")):
            raise DevelopmentGateError("missing_evaluation_progress_after_observed_work_cannot_refit")
        state = _json(progress_path) if progress_path.exists() else {
            "identity_sha256": identity_sha, "fit_attempts_dispatched": 0, "completed": {}, "active": None}
        if (state.get("identity_sha256") != identity_sha or type(state.get("fit_attempts_dispatched")) is not int
                or state["fit_attempts_dispatched"] < 0 or not isinstance(state.get("completed"), dict)
                or len(state["completed"]) > state["fit_attempts_dispatched"]):
            raise DevelopmentGateError("invalid_evaluation_dispatch_checkpoint")
        wall_path = _resource_history_gate(root, output, state, work_directory="sessions")
        initial_values = _json(_reference(root, initial_budget))
        _attempt_limit(state, limits, initial_values["max_fit_attempts"])
        _gpu_limit_history(state, limits, initial_values["max_gpu_seconds"])
        atomic_json(progress_path, state)
        journal = _open_wall_budget(wall_path, identity_sha, limits, initial_values, started_at=invoked_at)
        try:
            journal.start_interval()
            state["wall_ledger_initialized"] = True; atomic_json(progress_path, state)
            if isinstance(profile, dict):
                profile = verify_calibration_profile(root, profile, expected_runtime=runtime)
            if (type(profile) is not VerifiedCalibrationProfile or profile._seal is not _PROFILE_SEAL
                    or profile.synthetic is not software_test_double or job.method not in profile.methods
                    or profile.runtime_profile != runtime
                    or limits["max_calibration_seconds"] <= profile.maximum_calibration_seconds[job.method]):
                raise DevelopmentGateError("actual_same_GPU_profile_and_explicit_calibration_allowance_required")
            context, catalog, inputs = _evaluation_context(root, identity)
            _profile_matches_input(root, profile, context, software_test_double=software_test_double)
            _put(destination / "inputs.json", inputs)
            configuration_path = destination / "configuration.json"
            _put(configuration_path, {"method": job.method, "configuration": job.configuration,
                "pipeline_seed": job.seed, "phase": job.phase, "signal_contract_sha256": context.signal_contract_sha256,
                "final_protocol_freeze_sha256": None})
            configuration = _descriptor(root, configuration_path)
            eligible = {digest([p.participant_id, p.session_id]): p for p in catalog.plans if p.query_eligible}
            if set(state["completed"]) - set(eligible):
                raise DevelopmentGateError("evaluation_checkpoint_contains_other_query_session")
            if journal.exhausted:
                raise DevelopmentGateError("evaluation_wall_budget_exhausted_during_input_validation")
            if not software_test_double:
                _on_cuda(context)
            for key, plan in eligible.items():
                if journal.exhausted:
                    raise DevelopmentGateError("evaluation_wall_budget_exhausted")
                if key in state["completed"]:
                    record = state["completed"][key]
                else:
                    active = state.get("active")
                    reuse = False
                    if active is not None:
                        expected_directory = output+f"/sessions/{key}/attempt-{active.get('attempt', 0):06d}"
                        if (active.get("session_key") != key or active.get("directory") != expected_directory
                                or active.get("attempt") != state["fit_attempts_dispatched"]):
                            raise DevelopmentGateError("invalid_evaluation_active_candidate_identity")
                        reuse = _candidate_exists(root, active["directory"])
                    if not reuse:
                        if state["fit_attempts_dispatched"] >= state["fit_attempt_limit"]:
                            return {"status": "PAUSED_OPERATIONAL_BUDGET", "synthetic": software_test_double,
                                "sessions_completed": len(state["completed"]), "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                                "scientific_fit_completed": False, "budget": journal.report()}
                        state["fit_attempts_dispatched"] += 1
                        attempt = state["fit_attempts_dispatched"]
                        active = {"session_key": key, "attempt": attempt, "phase": "FIT_DISPATCHED",
                            "directory": output+f"/sessions/{key}/attempt-{attempt:06d}"}
                        state["active"] = active; atomic_json(progress_path, state)
                    candidate = candidate_identity(context, job, plan, purpose="development_evaluation",
                        execution_identity_sha256=identity_sha, initial_budget=initial_budget, attempt=active["attempt"])
                    candidate["dependencies"] = identity["dependencies"]
                    if software_test_double:
                        if candidate_producer is None:
                            raise DevelopmentGateError("explicit_synthetic_candidate_fixture_required")
                        fitted, candidate_ref, _, cost = candidate_producer(root, active["directory"], context, job, catalog, plan, candidate, runtime)
                    else:
                        fitted, candidate_ref, _, cost = _measure_candidate(root, active["directory"], context, job, catalog,
                            plan, candidate, runtime, deadline_seconds=limits["max_calibration_seconds"], wall_budget=journal)
                    if journal.exhausted:
                        raise DevelopmentGateError("evaluation_wall_budget_exceeded_before_prediction")
                    state["active"].update(phase="FIT_PARAMETERS_COMMITTED", candidate_state=candidate_ref)
                    atomic_json(progress_path, state)
                    if software_test_double and prediction_producer is not None:
                        prediction_ref = prediction_producer(root, active["directory"], context, job, catalog, plan, fitted, cost, configuration)
                    else:
                        prediction_ref = _predict_candidate(root, active["directory"], context, job, catalog, plan, fitted, cost, configuration,
                            software_test_double=software_test_double)
                    if journal.exhausted:
                        raise DevelopmentGateError("evaluation_wall_budget_exceeded_before_label_release")
                    row = _score(root, plan, catalog.materialize(plan, device="cpu"), prediction_ref, job)
                    journal.checkpoint()
                    if journal.exhausted:
                        raise DevelopmentGateError("evaluation_wall_budget_exceeded_before_session_commit")
                    record = {"metrics": row, "attempt": active["attempt"], "candidate_state": candidate_ref,
                        "directory": active["directory"]}
                    state["completed"][key] = record; state["active"] = None
                    atomic_json(progress_path, state)
                candidate = candidate_identity(context, job, plan, purpose="development_evaluation",
                    execution_identity_sha256=identity_sha, initial_budget=initial_budget, attempt=record["attempt"])
                candidate["dependencies"] = identity["dependencies"]
                _, saved, _, actual_cost = _candidate_load(root, record["directory"], context, job, plan, candidate, runtime,
                    software_test_double=software_test_double)
                if saved != record["candidate_state"]:
                    raise DevelopmentGateError("completed_candidate_descriptor_changed")
                _prediction_binding(root, record["metrics"]["prediction_receipt"], context, job, plan, actual_cost,
                    configuration, _ref(_json(_reference(root, saved))["parameters"]))
            summary = _summarize(root, catalog, {k: v["metrics"] for k, v in state["completed"].items()}, job)
            journal.checkpoint()
            if journal.exhausted:
                raise DevelopmentGateError("evaluation_wall_budget_exceeded_before_completion")
            completion = _completion_record(root, output, journal, state, limits, initial_budget, schema=SCHEMA)
            completion_path = destination / ("completion_budget-"+digest(completion)+".json")
            _put(completion_path, completion)
            result = {"schema_version": SCHEMA,
                "status": "SYNTHETIC_REGISTERED_VALIDATION_FIXTURE_COMPLETE" if software_test_double else "DEVELOPMENT_EVALUATION_COMPLETE",
                "synthetic": software_test_double, "measured": not software_test_double,
                "scientific_fit_completed": not software_test_double, "role": "meta_validation", "method": job.method,
                "seed": job.seed, "job": asdict(job), "configuration": job.configuration, "identity_sha256": identity_sha,
                "identity": _descriptor(root, destination / "identity.json"), "inputs": inputs,
                "catalog": _catalog_binding(catalog), "initial_budget": initial_budget,
                "completion_budget": _descriptor(root, completion_path), "calibrations": [state["completed"][k] for k in eligible],
                "timing_protocol": TIMING_PROTOCOL, "labels_after_verified_predictions": True,
                "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                "signal_contract_sha256": context.signal_contract_sha256, "final_protocol_freeze_sha256": None, **summary}
            references = [result["identity"], initial_budget, result["completion_budget"], completion["wall_journal_snapshot"],
                _descriptor(root, destination / "inputs.json"), configuration]
            for row in result["calibrations"]:
                references.extend(_candidate_artifacts(root, row["candidate_state"]))
                receipt = _reference(root, row["metrics"]["prediction_receipt"])
                backup = _json(receipt.with_name("artifact_manifest.json"))
                references.extend(_ref(r) for r in backup["files"])
                references.append(_descriptor(root, receipt.with_name("artifact_manifest.json")))
            if journal.exhausted:
                raise DevelopmentGateError("evaluation_wall_budget_exceeded_during_final_artifact_hashing")
            saved = _publish_completion(root, output, result, references, journal)
            return {**result, "receipt": saved}
        finally:
            journal.close()


def _verify_development_production(root, descriptor, result, *, software_test_double=False):
    required = {"identity", "inputs", "catalog", "calibrations", "initial_budget", "completion_budget", "job"}
    if not required <= set(result):
        raise DevelopmentGateError("complete_production_parameter_prediction_and_budget_proof_required")
    identity = _json(_reference(root, result["identity"], suffix=".json"))
    identity_sha = digest(identity)
    if result.get("identity_sha256") != identity_sha:
        raise DevelopmentGateError("evaluation_identity_bytes_changed")
    if identity.get("synthetic") is not software_test_double:
        raise DevelopmentGateError("production_and_synthetic_execution_identities_cannot_be_mixed")
    profile = verify_calibration_profile(root, identity["profile_receipt"], expected_runtime=identity["runtime_profile"],
        software_test_double=software_test_double)
    context, catalog, inputs = _evaluation_context(root, identity)
    _profile_matches_input(root, profile, context, software_test_double=software_test_double)
    job = DevelopmentJob(**identity["job"])
    if (result.get("inputs") != inputs or result.get("catalog") != _catalog_binding(catalog)
            or result.get("method") != job.method or result.get("seed") != job.seed or result.get("job") != asdict(job)
            or result.get("configuration") != job.configuration or result.get("timing_protocol") != TIMING_PROTOCOL
            or result.get("labels_after_verified_predictions") is not True
            or result.get("signal_contract_sha256") != context.signal_contract_sha256
            or result.get("final_protocol_freeze_sha256") is not None):
        raise DevelopmentGateError("evaluation_source_method_configuration_query_or_W7_binding_changed")
    initial = _json(_reference(root, result["initial_budget"]))
    if initial.get("stage") != "development_evaluation" or initial.get("job_sha256") != job.identity:
        raise DevelopmentGateError("evaluation_initial_budget_not_bound_to_registered_job")
    _verify_completion_budget(root, result, identity_sha=identity_sha)
    eligible = [p for p in catalog.plans if p.query_eligible]
    calibrations = result["calibrations"]
    if not isinstance(calibrations, list) or len(calibrations) != len(eligible):
        raise DevelopmentGateError("complete_validation_parameter_coverage_required_no_subset")
    output = str(Path(descriptor["path"]).parent)
    references = [result["identity"], result["initial_budget"], result["completion_budget"], _ref(descriptor),
        _descriptor(root, _local(root, output+"/inputs.json")), _descriptor(root, _local(root, output+"/configuration.json")),
        _json(_reference(root, result["completion_budget"]))["wall_journal_snapshot"]]
    if _json(_local(root, output+"/inputs.json")) != inputs:
        raise DevelopmentGateError("evaluation_input_dependency_file_changed")
    sessions, attempts = {}, set()
    for record, plan in zip(calibrations, eligible):
        key = digest([plan.participant_id, plan.session_id])
        attempt = record.get("attempt")
        if (type(attempt) is not int or not 0 < attempt <= result["fit_attempts_dispatched"] or attempt in attempts
                or record.get("directory") != output+f"/sessions/{key}/attempt-{attempt:06d}"):
            raise DevelopmentGateError("evaluation_calibration_attempt_identity_changed")
        attempts.add(attempt)
        candidate = candidate_identity(context, job, plan, purpose="development_evaluation",
            execution_identity_sha256=identity_sha, initial_budget=result["initial_budget"], attempt=attempt)
        candidate["dependencies"] = identity["dependencies"]
        fitted, saved, timing, cost = _candidate_load(root, record["directory"], context, job, plan, candidate, identity["runtime_profile"],
            software_test_double=software_test_double)
        if saved != record["candidate_state"] or timing["calibration_deadline_seconds"] > initial["max_calibration_seconds"]:
            raise DevelopmentGateError("evaluation_parameter_or_calibration_limit_evidence_changed")
        _prediction_binding(root, record["metrics"]["prediction_receipt"], context, job, plan, cost,
            _descriptor(root, _local(root, output+"/configuration.json")), _ref(_json(_reference(root, saved))["parameters"]))
        row = _score(root, plan, catalog.materialize(plan, device="cpu"), record["metrics"]["prediction_receipt"], job)
        if row != record["metrics"]:
            raise DevelopmentGateError("evaluation_prediction_first_session_score_changed")
        sessions[key] = row
        references.extend(_candidate_artifacts(root, saved))
        receipt_path = _reference(root, record["metrics"]["prediction_receipt"])
        backup = _json(receipt_path.with_name("artifact_manifest.json"))
        references.extend(_ref(r) for r in backup["files"])
        references.append(_descriptor(root, receipt_path.with_name("artifact_manifest.json")))
    summary = _summarize(root, catalog, sessions, job)
    if any(result.get(k) != value for k, value in summary.items()):
        raise DevelopmentGateError("evaluation_participant_equal_metrics_or_full_query_coverage_changed")
    if _json(_local(root, output+"/artifact_manifest.json")) != _artifact_manifest(root, output, references):
        raise DevelopmentGateError("evaluation_complete_full_SHA_output_artifact_manifest_required")
    return result


def verify_synthetic_development_receipt(root, descriptor):
    """CPU transaction fixtures only; never accepted by scientific selection."""
    root = Path(root).resolve(strict=True)
    result = _json(_reference(root, descriptor))
    if (result.get("status") != "SYNTHETIC_REGISTERED_VALIDATION_FIXTURE_COMPLETE"
            or result.get("synthetic") is not True or result.get("scientific_fit_completed") is not False
            or result.get("measured") is not False):
        raise DevelopmentGateError("explicit_synthetic_validation_transaction_required")
    return _verify_development_production(root, descriptor, result, software_test_double=True)
