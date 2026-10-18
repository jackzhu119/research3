"""Bounded W6 development evaluation, never external evaluation or GPU startup.

Importing/planning does not fit or read signal data. Production input gates use
the verified method registry and Stieger meta_validation capability; execution
remains blocked until its measured-profile proof producer/validator exists.
CPU callback fixtures have a separate, permanently synthetic entry and cannot
supply scientific selection evidence. No network or credential API is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import time

import numpy as np
import torch

from . import method_registry as registry
from .adapt import balanced_bce
from .contracts import digest, file_digest, validate_signal_contract
from .episodes import load_episode_catalog
from .prediction_store import (verify_prediction_receipt,
                               _local, _reference, _write_identical_or_new)
from .recovery_budget import PersistentWallBudget
from .selection import choose_strongest_control, participant_equal_metrics
from .training_state import atomic_json

SCHEMA = "paper3-development-evaluation-v1"
SEEDS = (20261018, 20261019, 20261020)
METHODS = registry.METHOD_IDS
CONTROLS = tuple(f"B{i}" for i in range(1, 12))
TIMING_PROTOCOL = "CPU_staged_valid_support_to_synchronized_fitted_state_v1"
TIMING_SCOPE = ("Includes support transfer, method-specific artifact validation, source copy, "
                "target covariance/features and optimization; excludes catalog verification, "
                "source/reference loading, query inference, prediction I/O and acquisition waiting.")
CONFIGURATION_TIE_RULE = "participant_BA_desc/balanced_BCE_asc/selected_meta_checkpoint_asc/registered_grid_order"
CODE_FILES = ("p3work/development_evaluation.py", "tools/evaluate_development.py",
    "p3work/method_registry.py", "p3work/episodes.py", "p3work/prediction_store.py",
    "p3work/selection.py", "p3work/adapt.py", "p3work/spatial_geometry.py",
    "p3work/meta_artifacts.py", "p3work/methods/lowrank.py",
    "p3work/methods/classical.py", "p3work/methods/neural_controls.py",
    "p3work/f0_bridge.py", "p3work/source_reference_io.py", "p3work/classical_training.py",
    "p3work/contracts.py", "p3work/training_state.py", "p3work/recovery_budget.py")
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
        "production_execution_status": "BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR",
        "minimum_production_gap": ["bounded_real_calibration_profile_producer_and_full_SHA_proof_validator",
            "sealed_registry_runner_binding_fitted_checkpoints_timing_and_prediction_receipts"],
        "required_gates": ["W1/W2/W3_COMPLETE", "verified_same_seed_f0_Rref",
            "verified_independent_learned_or_classical_artifacts", "later_user_started_CUDA",
            "explicit_wall_fit_and_per_calibration_limits", "same_device_measured_timing_profile"]}
    if root is not None:
        root = Path(root).resolve(strict=True)
        result["code"] = {p: file_digest(_local(root, p)) for p in CODE_FILES}
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
    """Fail-closed production entry: the measured-profile validator is pending.

    Exact real prerequisites and bounded budgets are checked read-only. This
    module cannot yet establish a measured timing-profile chain, so no JSON
    status flag, caller callback, CPU fallback or CUDA presence authorizes a fit.
    """
    if user_started_gpu is not True:
        raise PermissionError("later_explicit_user_GPU_boot_required")
    root = Path(root).resolve(strict=True)
    verify_development_inputs(root, job=job, artifact_catalog=artifact_catalog,
        configuration_receipt=configuration_receipt)
    limits = _json(_reference(root, budget, suffix=".json"))
    if (limits.get("stage") != "development_evaluation" or limits.get("job_sha256") != job.identity
            or limits.get("max_jobs") != 1 or limits.get("max_download_bytes") != 0
            or type(limits.get("max_fit_attempts")) is not int or limits["max_fit_attempts"] < 1
            or not all(_positive(limits.get(k)) for k in ("max_wall_seconds", "max_gpu_seconds", "max_calibration_seconds"))
            or limits["max_gpu_seconds"] < limits["max_wall_seconds"]):
        raise DevelopmentGateError("explicit_exact_job_wall_GPU_attempt_and_calibration_budget_required")
    raise DevelopmentGateError("BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR")


def verify_development_receipt(root, descriptor):
    """No scientific evidence accepted before the production proof chain exists.

    The store verifier alone proves prediction-byte integrity, not execution,
    same-device timing, a real fit or a correctly measured pilot. In particular,
    a status/runtime JSON is never an alternative to that missing producer and
    validator. Synthetic receipts are explicitly rejected before this gate.
    """
    root = Path(root).resolve(strict=True)
    value = _json(_reference(root, descriptor, suffix=".json"))
    if (value.get("schema_version") != SCHEMA or value.get("status") != "DEVELOPMENT_EVALUATION_COMPLETE"
            or value.get("synthetic") is not False or value.get("scientific_fit_completed") is not True
            or value.get("measured") is not True or value.get("role") != "meta_validation"):
        raise DevelopmentGateError("completed_real_development_evidence_required_no_fixture_or_subset")
    raise DevelopmentGateError("BLOCKED_MISSING_MEASURED_PROFILE_VALIDATOR")


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
