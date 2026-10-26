"""One persistent W6 internal check after complete, immutable development selection.

No network or server operations. Planning/freezing reads only development
evidence and split metadata. Internal signals require the real GPU execution
gate, a root-wide durable claim and its exact registered access authority.
All frozen method/seed/session predictions commit before any label release.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from . import development_evaluation as dev
from . import method_registry as registry
from .adapt import CalibrationPrefix, PredictionInputs, PredictionResult, balanced_bce
from .contracts import digest, file_digest, validate_signal_contract
from .episodes import load_episode_catalog
from .fit_artifacts import load_registered_fit, save_registered_fit
from .prediction_store import (COST_OPTIONAL, COST_REQUIRED, _local, _reference,
    commit_predictions, verify_prediction_receipt)
from .recovery_budget import PersistentWallBudget
from .selection import participant_equal_metrics
from .training_state import atomic_json

SCHEMA = "paper3-internal-validation-v1"
FREEZE_SCHEMA = "paper3-pre-internal-development-freeze-v1"
OUTPUT = "runs/internal_validation_once"
CLAIM = "runs/internal_validation_once_claim.json"
ACCESS = OUTPUT + "/internal_access.json"
CODE = ("p3work/internal_validation.py", "tools/run_internal_validation.py")


class InternalValidationError(ValueError):
    pass


def _json(path):
    return json.loads(path.read_text())


def _desc(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def _put(path, value):
    dev._put(path, value)  # Atomic no-overwrite/content-identical publication.


def _software(root):
    software = dev.build_development_plan(root)
    executed_root = Path(__file__).resolve().parents[1]
    for relative in CODE:
        path = _local(root, relative)
        if path.read_bytes() != (executed_root / relative).read_bytes():
            raise InternalValidationError("recorded_internal_implementation_differs_from_executed_code")
        software["code"][relative] = file_digest(path)
    return software


def _shape(spec):
    if (not isinstance(spec, dict) or spec.get("schema_version") != "paper3-internal-freeze-spec-v1"
            or set(spec.get("configuration_receipts", {})) != set(dev.METHODS)
            or not isinstance(spec.get("confirmation_receipts"), list)
            or len(spec["confirmation_receipts"]) != len(dev.METHODS)*len(dev.SEEDS)
            or not isinstance(spec.get("strongest_control_receipt"), dict)
            or not isinstance(spec.get("artifact_catalog"), dict)):
        raise InternalValidationError("all_fourteen_method_three_seed_and_Bstar_development_evidence_required")


def freeze_development(root, spec, *, output=None, execute=False):
    """Recompute every dev choice before creating the whole-package freeze.

    Calls only meta-validation/source verifiers, never the internal loader.
    This freeze is the pre-internal W6 selection freeze, not final W7 authority.
    """
    root = Path(root).resolve(strict=True)
    _shape(spec)
    tasks = _json(_local(root, "status/tasks.json"))
    if any(tasks.get(stage, {}).get("status") != "COMPLETE" for stage in ("W1", "W2", "W3")):
        raise InternalValidationError("complete_real_W1_W2_W3_required_before_selection_freeze")
    contract = _json(_local(root, "contracts/input_contract.json"))
    input_sha = validate_signal_contract(contract)
    cache_path = _reference(root, tasks["W3"]["cache_manifest"])
    manifest = _json(cache_path)
    split_path = _reference(root, manifest["participant_split"])
    from .episodes import _split
    split = _split(split_path)
    _reference(root, tasks["W2"]["cache_manifest"])
    _reference(root, spec["artifact_catalog"])
    selections = {method: dev.verify_configuration_receipt(root, ref)
                  for method, ref in spec["configuration_receipts"].items()}
    confirmations, seen, common, sources = [], set(), set(), {}
    for ref in spec["confirmation_receipts"]:
        value = dev.verify_development_receipt(root, ref)
        method, seed = value["method"], value["seed"]
        if method not in dev.METHODS or seed not in dev.SEEDS or (method, seed) in seen:
            raise InternalValidationError("unique_all_method_seed_development_coverage_required")
        seen.add((method, seed))
        selected = selections[method]
        job = dev.DevelopmentJob(**value["job"])
        if (selected["method"] != method or value["configuration"] != selected["configuration"]
                or job.configuration_index != selected["configuration_index"]
                or (seed == dev.SEEDS[0] and ref != selected["selected_screening_receipt"])
                or (seed != dev.SEEDS[0] and value["inputs"]["configuration_receipt"] != spec["configuration_receipts"][method])):
            raise InternalValidationError("internal_configuration_must_be_selected_only_on_complete_development")
        dependencies = dev._dependencies(root, job, spec["artifact_catalog"])
        if dependencies != value["inputs"]["dependencies"]:
            raise InternalValidationError("artifact_catalog_not_exact_selected_method_seed_dependencies")
        common.add(digest({"catalog": value["catalog"], "device": value["inputs"]["device_identity_sha256"],
                          "timing_protocol": value["timing_protocol"]}))
        sources.setdefault(seed, set()).add(digest(dependencies["source_f0"]))
        confirmations.append({"method": method, "seed": seed, "job": asdict(job),
            "configuration": job.configuration, "development_receipt": ref, "dependencies": dependencies,
            "source_context_identity_sha256": value["inputs"]["source_context_identity_sha256"],
            "selected_meta_outer_update": value["inputs"]["selected_meta_outer_update"],
            "profile_receipt": value["inputs"]["profile_receipt"],
            "runtime_profile": value["inputs"]["runtime_profile"]})
    if (seen != {(m, s) for m in dev.METHODS for s in dev.SEEDS} or len(common) != 1
            or any(len(v) != 1 for v in sources.values())):
        raise InternalValidationError("whole_package_same_validation_device_and_same_seed_source_required")
    strongest = _json(_reference(root, spec["strongest_control_receipt"]))
    control_refs = [r["development_receipt"] for r in confirmations if r["method"] in dev.CONTROLS]
    if (strongest.get("configuration_receipts") != {m: spec["configuration_receipts"][m] for m in dev.CONTROLS}
            or not isinstance(strongest.get("confirmation_receipts"), list)
            or len(strongest["confirmation_receipts"]) != 33
            or {digest(r) for r in strongest["confirmation_receipts"]} != {digest(r) for r in control_refs}):
        raise InternalValidationError("Bstar_must_use_exact_same_selected_control_evidence")
    rebuilt = dev.select_strongest_control(root,
        configuration_receipts={m: spec["configuration_receipts"][m] for m in dev.CONTROLS},
        confirmation_receipts=strongest["confirmation_receipts"])
    if strongest != rebuilt:
        raise InternalValidationError("Bstar_must_recompute_from_all_eleven_controls_three_seeds")
    software = _software(root)
    body = {"schema_version": FREEZE_SCHEMA, "status": "IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE",
        "synthetic": False, "spec": spec, "method_order": list(dev.METHODS), "seeds": list(dev.SEEDS),
        "selected_jobs": sorted(confirmations, key=lambda r: (dev.METHODS.index(r["method"]), dev.SEEDS.index(r["seed"]))),
        "Bstar": strongest["selected"], "input_contract_sha256": input_sha,
        "input_contract": manifest["input_contract"], "participant_split": manifest["participant_split"],
        "split_sha256": split["split_sha256"], "W2_cache": tasks["W2"]["cache_manifest"],
        "W3_cache": tasks["W3"]["cache_manifest"], "internal_participants": split["partitions"]["internal_holdout"],
        "all_split_participants": split["partitions"], "code": software["code"], "policy": software["policy"],
        "internal_data_used_for_selection": False, "retuning_after_internal_check_allowed": False,
        "internal_check_repetitions": 1, "final_protocol_freeze_sha256": None}
    result = {**body, "freeze_sha256": digest(body)}
    if execute:
        if not isinstance(output, str) or not output.startswith("runs/") or not output.endswith(".json"):
            raise InternalValidationError("immutable_pre_internal_freeze_JSON_output_required")
        target = _local(root, output, exists=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        _put(target, result)
    return result


def verify_freeze(root, descriptor):
    value = _json(_reference(root, descriptor, suffix=".json"))
    body = dict(value); recorded = body.pop("freeze_sha256", None)
    if (value.get("schema_version") != FREEZE_SCHEMA or recorded != digest(body)
            or value.get("synthetic") is not False):
        raise InternalValidationError("whole_package_immutable_real_freeze_required")
    rebuilt = freeze_development(root, value["spec"])
    if rebuilt != value:
        raise InternalValidationError("frozen_configuration_source_reference_meta_or_development_evidence_changed")
    return value


def plan_internal(root, freeze=None):
    """No internal event/EEG/label read and no CUDA check, even with a freeze."""
    root = Path(root).resolve(strict=True)
    tasks = _json(_local(root, "status/tasks.json"))
    gaps = [stage+"_COMPLETE" for stage in ("W1", "W2", "W3")
            if tasks.get(stage, {}).get("status") != "COMPLETE"]
    if freeze is None:
        gaps.append("whole_package_real_three_seed_development_and_Bstar_freeze")
    else:
        verify_freeze(root, freeze)
    return {"schema_version": SCHEMA, "status": "PLANNED_NOT_EXECUTED", "minimum_artifact_gaps": gaps,
        "fixed_output": OUTPUT, "persistent_claim": CLAIM, "methods": list(dev.METHODS), "seeds": list(dev.SEEDS),
        "all_predictions_committed_before_any_label_release": True, "internal_payloads_accessed": False,
        "CUDA_checked": False, "actual_fits": 0, "W6_complete": False,
        "network_operations": 0, "server_actions": False}


def _claim(root, identity):
    """One root-wide claim; changed selection, budget, GPU or output is refused."""
    if identity.get("output") != OUTPUT:
        raise InternalValidationError("one_fixed_internal_output_required")
    path = _local(root, CLAIM, exists=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {"schema_version": SCHEMA, "status": "INTERNAL_CHECK_CLAIMED_IRREVERSIBLE_ATTEMPT",
        "identity": identity, "identity_sha256": digest(identity), "fixed_output": OUTPUT,
        "claim_deletion_or_new_output_retry_allowed": False}
    if path.exists() and _json(path) != expected:
        raise InternalValidationError("one_time_claim_already_bound_do_not_delete_or_change_output")
    _put(path, expected)
    return _desc(root, path)


def _register_access(root, frozen, claim):
    """Future execution only, after the durable claim; no authority from a flag."""
    value = {"schema_version": "paper3-internal-check-access-v1", "status": "AUTHORIZED_ONCE_DEVELOPMENT_CHECK",
        "role": "internal_holdout", "split_sha256": frozen["split_sha256"],
        "input_contract_sha256": frozen["input_contract_sha256"], "labels_after_predictions_only": True,
        "training_and_selection_allowed": False, "selected_configuration_sha256": frozen["freeze_sha256"],
        "one_time_claim": claim, "fixed_output": OUTPUT}
    path = _local(root, ACCESS, exists=False); path.parent.mkdir(parents=True, exist_ok=True)
    _put(path, value); ref = _desc(root, path)
    tasks_path = _local(root, "status/tasks.json")
    tasks = _json(tasks_path)
    if tasks["W3"].get("status") != "COMPLETE" or tasks["W3"].get("cache_manifest") != frozen["W3_cache"]:
        raise InternalValidationError("W3_authority_changed_before_access_registration")
    prior = tasks["W3"].get("internal_access_receipt")
    if prior is not None and prior != ref:
        raise InternalValidationError("existing_internal_authority_preserved_no_replacement")
    if prior is None:
        tasks["W3"]["internal_access_receipt"] = ref
        atomic_json(tasks_path, tasks)
    return ref


def _budget(root, descriptor, freeze_sha):
    value = _json(_reference(root, descriptor))
    if (value.get("stage") != "internal_development_check" or value.get("freeze_sha256") != freeze_sha
            or type(value.get("max_jobs")) is not int or value["max_jobs"] != 1
            or type(value.get("max_download_bytes")) is not int or value["max_download_bytes"] != 0
            or type(value.get("max_fit_attempts")) is not int or value["max_fit_attempts"] < 1
            or not all(dev._positive(value.get(k)) for k in ("max_wall_seconds", "max_gpu_seconds", "max_calibration_seconds"))
            or value["max_gpu_seconds"] < value["max_wall_seconds"]):
        raise InternalValidationError("explicit_frozen_one_check_wall_GPU_attempt_calibration_budget_required")
    return value


def _candidate(context, job, episode, frozen, claim, budget):
    return {"schema_version": SCHEMA, "purpose": "target_evaluation", "evaluation_role": "internal_development_check",
        "method": job.method, "pipeline_seed": job.seed, "configuration": job.configuration,
        "support_role": "calibration_prefix", "signal_contract_sha256": context.signal_contract_sha256,
        "source_context_identity_sha256": context.identity_sha256,
        "participant_id": episode.participant_id, "session_id": episode.session_id,
        "episode_plan_sha256": episode.temporal_audit_sha256, "freeze_sha256": frozen["freeze_sha256"],
        "one_time_claim": claim, "budget": budget, "final_protocol_freeze_sha256": None}


def _load_fit(root, directory, context, job, episode, candidate):
    if _json(_local(root, directory+"/dispatch.json")) != candidate:
        raise InternalValidationError("saved_once_dispatched_candidate_identity_changed")
    model = _desc(root, _local(root, directory+"/fitted/receipt.json"))
    fitted = load_registered_fit(context, model, expected_method=job.method,
        expected_configuration=job.configuration, expected_support_role="calibration_prefix",
        expected_acquired_event_uids=episode.acquired_event_uids,
        expected_support_event_uids=episode.support_event_uids, candidate_identity=candidate)
    timing = _json(_local(root, directory+"/timing.json"))
    cost = _json(_local(root, directory+"/cost.json"))
    claim = _json(_reference(root, candidate["one_time_claim"]))
    runtime = claim["identity"]["runtime"]
    limits = _json(_reference(root, candidate["budget"]))
    warmup = _json(_local(root, directory+"/warmup.json"))
    if (warmup.get("status") != "SUPPORT_SOURCE_FORWARD_WARMUP_COMPLETE"
            or warmup.get("synthetic") is not False or warmup.get("candidate_identity_sha256") != digest(candidate)
            or warmup.get("runtime_profile") != runtime
            or warmup.get("source_context_identity_sha256") != context.identity_sha256
            or warmup.get("signal_contract_sha256") != context.signal_contract_sha256
            or warmup.get("support_event_uids") != list(episode.support_event_uids)
            or warmup.get("acquired_event_uids") != list(episode.acquired_event_uids)
            or warmup.get("source_forward_calls") != dev.WARMUP_FORWARDS
            or warmup.get("target_optimizer_updates") != 0 or warmup.get("query_labels_accessed") is not False
            or not dev._positive(warmup.get("wall_seconds"))):
        raise InternalValidationError("real_support_only_same_GPU_source_warmup_required")
    if (timing.get("candidate_identity_sha256") != digest(candidate)
            or timing.get("runtime_profile") != runtime
            or timing.get("query_labels_accessed") is not False or not dev._positive(timing.get("calibration_wall_seconds"))
            or not dev._positive(timing.get("calibration_deadline_seconds"))
            or timing["calibration_deadline_seconds"] > limits["max_calibration_seconds"]
            or timing["calibration_wall_seconds"] > timing["calibration_deadline_seconds"]
            or cost.get("calibration_wall_seconds") != timing["calibration_wall_seconds"]
            or any(cost.get(k) != v for k, v in dev._fit_statistics(fitted, episode).items())):
        raise InternalValidationError("saved_internal_fit_timing_or_update_ledger_changed")
    try:
        start, end = (datetime.fromisoformat(timing[k]) for k in ("started_at_utc", "finished_at_utc"))
        if (start.utcoffset() is None or end.utcoffset() is None
                or abs((end-start).total_seconds()-timing["calibration_wall_seconds"]) > 5
                or type(timing.get("peak_GPU_allocated_bytes")) is not int
                or type(timing.get("peak_GPU_reserved_bytes")) is not int
                or not 0 <= timing["peak_GPU_allocated_bytes"] <= timing["peak_GPU_reserved_bytes"] <= runtime["device_total_memory_bytes"]):
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise InternalValidationError("internal_actual_GPU_memory_or_timing_interval_proof_invalid") from None
    return fitted, model, cost


def _fit_once(root, directory, context, job, catalog, episode, candidate, dependencies, runtime, journal, maximum):
    """Never refit an uncertain attempt; restore only complete typed parameters."""
    destination = _local(root, directory, exists=False)
    if destination.exists():
        if not _local(root, directory+"/fitted/receipt.json", exists=False).exists():
            raise InternalValidationError("uncertain_internal_attempt_preserved_no_refit_or_new_directory")
        return _load_fit(root, directory, context, job, episode, candidate)
    destination.mkdir(parents=True)
    _put(destination / "dispatch.json", candidate)
    support = catalog.materialize_support(episode, device="cpu")
    dev._warmup_source(root, destination, context, support, candidate, runtime, journal, maximum)
    limit = min(maximum, journal.remaining_seconds)
    if not dev._positive(limit) or journal.exhausted:
        raise InternalValidationError("cumulative_internal_budget_exhausted_before_fit")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    began = time.monotonic(); started = datetime.now(timezone.utc).isoformat()
    try:
        with dev._calibration_deadline(limit), torch.autocast(device_type="cuda", enabled=False):
            if job.method not in registry.CLASSICAL_METHODS:
                support = CalibrationPrefix(support.x.to("cuda:0"), support.y.to("cuda:0"), support.event_uids,
                    support.acquired_event_uids, support.protocol_freeze_sha, role=support.role)
            fitted = context.fit(job.method, support, job.configuration,
                learned_receipt=dependencies["learned"]["path"] if dependencies["learned"] else None,
                classic_model=dependencies["source_bank"])
            torch.cuda.synchronize()
        elapsed = time.monotonic()-began
        if elapsed > limit or journal.exhausted:
            raise InternalValidationError("internal_calibration_deadline_exceeded_preserve_failure")
        cost = {**dev._fit_statistics(fitted, episode), "calibration_wall_seconds": elapsed}
        timing = {"schema_version": SCHEMA, "candidate_identity_sha256": digest(candidate),
            "runtime_profile": runtime, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_wall_seconds": elapsed, "calibration_deadline_seconds": limit,
            "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_GPU_reserved_bytes": torch.cuda.max_memory_reserved(), "query_labels_accessed": False}
        _put(destination / "timing.json", timing); _put(destination / "cost.json", cost)
        save_registered_fit(context, fitted, directory+"/fitted", candidate_identity=candidate)
        return _load_fit(root, directory, context, job, episode, candidate)
    except BaseException as error:
        _put(destination / "attempt_failure.json", {"status": "FAILED_ATTEMPT_PRESERVED_NO_REFIT",
            "error_type": type(error).__name__, "candidate_identity_sha256": digest(candidate),
            "observed_wall_seconds": time.monotonic()-began})
        raise


def _prediction(root, directory, context, job, catalog, episode, fitted, model, cost, configuration, frozen, access):
    destination = directory+"/predictions"
    existing = _local(root, destination+"/receipt.json", exists=False)
    refs = {"source_f0": {"path": context.source_receipt_path, "sha256": context.source_receipt_sha256},
        "model": model, "method_code": _desc(root, _local(root, "p3work/method_registry.py")),
        "configuration": configuration, "input_contract": frozen["input_contract"],
        "participant_split": frozen["participant_split"], "episode_manifest": frozen["W3_cache"]}
    if existing.exists():
        ref = _desc(root, existing)
    else:
        query = catalog.materialize(episode, device="cpu").prediction_inputs
        if job.method not in registry.CLASSICAL_METHODS:
            query = PredictionInputs(query.x.to("cuda:0"), query.event_uids, query.protocol_freeze_sha)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            predicted = context.predict(fitted, query)
        result = PredictionResult(*(x.detach().cpu() for x in
            (predicted.logits, predicted.probabilities, predicted.labels)), predicted.event_uids, predicted.protocol_freeze_sha)
        ref = commit_predictions(root, destination, result, episode, method=job.method, pipeline_seed=job.seed,
            references=refs, cost={k: v for k, v in cost.items() if k in COST_REQUIRED | COST_OPTIONAL},
            purpose="internal_development_check", selected_configuration_sha256=frozen["freeze_sha256"],
            internal_access_receipt=access)["receipt"]
    value = verify_prediction_receipt(root, ref)
    if (value["references"] != refs or value["method"] != job.method or value["pipeline_seed"] != job.seed
            or value["selected_configuration_sha256"] != frozen["freeze_sha256"]
            or value["internal_access_receipt"] != access or value["prediction_role"] != "internal_holdout"
            or value["episode_plan_sha256"] != episode.temporal_audit_sha256):
        raise InternalValidationError("existing_internal_prediction_identity_or_frozen_parameters_changed")
    if _json(_reference(root, ref).with_name("cost.json")) != {k: v for k, v in cost.items() if k in COST_REQUIRED | COST_OPTIONAL}:
        raise InternalValidationError("internal_prediction_and_actual_fit_cost_ledgers_disagree")
    return ref


def _score_internal(root, episode, prediction, evaluation, freeze_sha):
    """Prediction verification necessarily precedes the label capability call."""
    value = verify_prediction_receipt(root, prediction)
    if (value["prediction_role"] != "internal_holdout" or value["purpose"] != "internal_development_check"
            or value["selected_configuration_sha256"] != freeze_sha
            or value["episode_plan_sha256"] != episode.temporal_audit_sha256
            or tuple(value["query_event_uids"]) != episode.query_event_uids):
        raise InternalValidationError("complete_matching_internal_prediction_required_before_labels")
    labels = evaluation.evaluation_labels.labels(purpose="internal_development_check", prediction_receipt=prediction).cpu()
    logits = torch.from_numpy(np.load(_reference(root, prediction).with_name("logits.npy"), allow_pickle=False))
    if labels.shape != (len(episode.query_event_uids),) or any(int((labels == c).sum()) < 5 for c in (0, 1)):
        raise InternalValidationError("fixed_internal_query_class_qualification_changed")
    predictions = logits.argmax(1)
    return {"participant_id": episode.participant_id, "session_id": episode.session_id,
        "ba": sum(float((predictions[labels == c] == c).double().mean()) for c in (0, 1))/2,
        "balanced_bce": float(balanced_bce(logits, labels)), "prediction_receipt": prediction}


def _phase_gate(state, required_keys):
    """Pure durable-transaction guard, including label-release crash recovery."""
    if (not isinstance(state.get("predictions"), dict) or not isinstance(state.get("scores"), dict)
            or type(state.get("fit_attempts_dispatched")) is not int or state["fit_attempts_dispatched"] < 0
            or not isinstance(state.get("dispatched_keys"), list)
            or len(state["dispatched_keys"]) != len(set(state["dispatched_keys"]))
            or state["fit_attempts_dispatched"] != len(state["dispatched_keys"])
            or set(state["dispatched_keys"]) - required_keys
            or set(state["predictions"]) - set(state["dispatched_keys"])
            or set(state["scores"]) - set(state["predictions"])
            or type(state.get("labels_revealed")) is not bool):
        raise InternalValidationError("persistent_once_check_counter_or_phase_ledger_invalid")
    if state["labels_revealed"] and set(state["predictions"]) != required_keys:
        raise InternalValidationError("labels_already_revealed_cannot_refit_or_add_predictions")
    if state["scores"] and not state["labels_revealed"]:
        raise InternalValidationError("scores_without_durable_label_release_barrier")


def _dispatch(state, key, candidate_exists, maximum):
    if state["labels_revealed"]:
        raise InternalValidationError("internal_labels_revealed_no_more_fits")
    if key in state["dispatched_keys"]:
        if not candidate_exists:
            raise InternalValidationError("prior_dispatched_attempt_missing_preserve_claim_no_refit")
        return
    if candidate_exists:
        raise InternalValidationError("unowned_internal_candidate_directory_preserved")
    if state["fit_attempts_dispatched"] >= maximum:
        raise InternalValidationError("internal_fit_attempt_budget_exhausted")
    state["fit_attempts_dispatched"] += 1
    state["dispatched_keys"].append(key)


def _release_barrier(root, destination, state, required_keys, frozen):
    _phase_gate(state, required_keys)
    if set(state["predictions"]) != required_keys:
        raise InternalValidationError("every_frozen_prediction_must_commit_before_any_label_release")
    for ref in state["predictions"].values():
        verify_prediction_receipt(root, ref)
    _put(destination / "all_predictions_committed.json", {"schema_version": SCHEMA,
        "freeze_sha256": frozen["freeze_sha256"], "predictions": state["predictions"],
        "labels_revealed_before_commit": False})
    state.update(phase="LABELS_AFTER_ALL_PREDICTIONS", labels_revealed=True)
    atomic_json(destination / "progress.json", state)


def _existing_run_gate(root, destination):
    prior = _local(root, CLAIM, exists=False).exists()
    if destination.exists() and not prior:
        raise InternalValidationError("unowned_internal_output_preserved_no_claim_adoption")
    if prior and (not destination.exists() or not (destination / "progress.json").is_file()
                  or not (destination / "wall_budget.json").is_file()):
        raise InternalValidationError("prior_one_time_claim_missing_ledger_preserved_no_reset")
    return prior


def verify_internal_completion(root, *, freeze):
    """Read-only replay of a sealed check, with no fit, CUDA, or new authority.

    Existing internal capability is required and is never created by this
    verifier. It may recompute scores for already committed predictions only.
    """
    root = Path(root).resolve(strict=True)
    frozen = verify_freeze(root, freeze)
    destination = _local(root, OUTPUT)
    _existing_run_gate(root, destination)
    result_path = _local(root, OUTPUT+"/receipt.json")
    result = _json(result_path)
    state = _json(_local(root, OUTPUT+"/progress.json"))
    claim = _desc(root, _local(root, CLAIM))
    identity = _json(_reference(root, claim))["identity"]
    if (result.get("status") != "ONE_TIME_INTERNAL_CHECK_COMPLETE" or result.get("synthetic") is not False
            or result.get("measured") is not True or result.get("whole_package_freeze") != freeze
            or result.get("freeze_sha256") != frozen["freeze_sha256"] or result.get("one_time_claim") != claim
            or identity["freeze"] != freeze or identity["freeze_sha256"] != frozen["freeze_sha256"]
            or identity["budget"] != result.get("budget")
            or state.get("identity_sha256") != digest(identity) or result.get("retuning_allowed") is not False):
        raise InternalValidationError("complete_sealed_same_freeze_internal_transaction_required")
    limits = _budget(root, result["budget"], frozen["freeze_sha256"])
    wall = _json(_local(root, OUTPUT+"/wall_budget.json"))
    report = result.get("cumulative_budget_report", {})
    if (wall.get("identity_sha256") != digest(identity) or wall.get("interval") is not None
            or wall.get("clock_block") is not None or report.get("interval_open") is not False
            or report.get("exhausted") is not False or report.get("blocked") is not False
            or not 0 <= report.get("wall_seconds", -1) <= wall.get("durable_wall_seconds", -1)
                <= min(limits["max_wall_seconds"], limits["max_gpu_seconds"])):
        raise InternalValidationError("complete_closed_cumulative_internal_budget_required")
    access = _desc(root, _local(root, ACCESS))
    if result.get("internal_access_receipt") != access:
        raise InternalValidationError("original_registered_internal_authority_required")
    catalog = load_episode_catalog(root, role="internal_holdout", purpose="internal_development_check",
        internal_access_receipt=access)
    eligible = [p for p in catalog.plans if p.query_eligible]
    if (list(catalog.partition_people) != frozen["internal_participants"]
            or {p.participant_id for p in eligible} != set(catalog.partition_people)):
        raise InternalValidationError("sealed_internal_people_cannot_be_dropped")
    required = {digest([row["method"], row["seed"], p.participant_id, p.session_id])
                for row in frozen["selected_jobs"] for p in eligible}
    _phase_gate(state, required)
    if (not state["labels_revealed"] or set(state["predictions"]) != required or set(state["scores"]) != required
            or result.get("predictions") != state["predictions"] or result.get("scores") != state["scores"]
            or result.get("fit_attempts_dispatched") != state["fit_attempts_dispatched"]):
        raise InternalValidationError("sealed_complete_all_method_seed_predictions_and_scores_required")
    barrier = _json(_reference(root, result["all_predictions_committed"]))
    if barrier != {"schema_version": SCHEMA, "freeze_sha256": frozen["freeze_sha256"],
                  "predictions": state["predictions"], "labels_revealed_before_commit": False}:
        raise InternalValidationError("original_global_prediction_before_label_barrier_required")
    summaries = []
    contract = _json(_reference(root, frozen["input_contract"]))
    for row in frozen["selected_jobs"]:
        metrics = []
        job = dev.DevelopmentJob(**row["job"])
        dependencies = row["dependencies"]
        context = registry.Context.from_verified(root, dependencies["source_f0"]["path"],
            dependencies["source_reference"]["path"], contract)
        if context.identity_sha256 != row["source_context_identity_sha256"]:
            raise InternalValidationError("sealed_same_verified_source_reference_context_required")
        dev._validate_dependency_parameters(root, context, job, dependencies)
        for episode in eligible:
            key = digest([row["method"], row["seed"], episode.participant_id, episode.session_id])
            reference = state["predictions"][key]
            saved = verify_prediction_receipt(root, reference)
            expected_directory = OUTPUT+"/candidates/"+key
            candidate = _candidate(context, job, episode, frozen, claim, result["budget"])
            fitted, model, cost = _load_fit(root, expected_directory, context, job, episode, candidate)
            if (reference["path"] != expected_directory+"/predictions/receipt.json"
                    or saved["method"] != row["method"] or saved["pipeline_seed"] != row["seed"]
                    or saved["references"]["source_f0"] != row["dependencies"]["source_f0"]
                    or saved["references"]["model"] != model
                    or _json(_reference(root, reference).with_name("cost.json")) !=
                        {k: v for k, v in cost.items() if k in COST_REQUIRED | COST_OPTIONAL}):
                raise InternalValidationError("sealed_prediction_method_seed_or_source_changed")
            score = _score_internal(root, episode, reference, catalog.materialize(episode, device="cpu"), frozen["freeze_sha256"])
            if score != state["scores"][key]:
                raise InternalValidationError("sealed_prediction_metrics_do_not_replay")
            metrics.append(score)
            del fitted
        summaries.append({"method": row["method"], "seed": row["seed"],
            **participant_equal_metrics(metrics, expected_participants=catalog.partition_people)})
        del context
    if summaries != result.get("method_seed_summaries"):
        raise InternalValidationError("sealed_participant_equal_internal_summary_changed")
    return {**result, "receipt": _desc(root, result_path), "idempotent": True}


def execute_internal(root, *, freeze, budget, user_started_gpu=False):
    """One immutable all-method/three-seed check; retries resume identical bytes."""
    if user_started_gpu is not True:
        raise PermissionError("explicit_later_user_started_GPU_required_before_internal_access")
    root = Path(root).resolve(strict=True)
    frozen = verify_freeze(root, freeze)
    limits = _budget(root, budget, frozen["freeze_sha256"])
    runtime = dev._production_runtime()
    for row in frozen["selected_jobs"]:
        profile = dev.verify_calibration_profile(root, row["profile_receipt"], expected_runtime=runtime)
        if (runtime != row["runtime_profile"] or limits["max_calibration_seconds"] <= profile.maximum_calibration_seconds[row["method"]]):
            raise InternalValidationError("same_actual_development_GPU_profile_and_calibration_allowance_required")
    identity = {"freeze": freeze, "freeze_sha256": frozen["freeze_sha256"], "budget": budget,
                "runtime": runtime, "output": OUTPUT}
    destination = _local(root, OUTPUT, exists=False)
    _existing_run_gate(root, destination)
    destination.mkdir(parents=True, exist_ok=True)
    with _local(root, OUTPUT+"/.once.lock", exists=False).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InternalValidationError("internal_check_already_running") from None
        claim = _claim(root, identity)
        if _local(root, OUTPUT+"/receipt.json", exists=False).exists():
            return verify_internal_completion(root, freeze=freeze)
        progress_path = destination / "progress.json"
        if progress_path.exists():
            state = _json(progress_path)
            if state.get("identity_sha256") != digest(identity):
                raise InternalValidationError("persistent_internal_progress_identity_changed")
        else:
            state = {"schema_version": SCHEMA, "identity_sha256": digest(identity), "phase": "PREDICTIONS_ONLY",
                "fit_attempts_dispatched": 0, "dispatched_keys": [], "predictions": {}, "scores": {}, "labels_revealed": False}
            atomic_json(progress_path, state)
        wall_path = destination / "wall_budget.json"
        if state["fit_attempts_dispatched"] and not wall_path.exists():
            raise InternalValidationError("missing_prior_cumulative_wall_ledger_no_reset")
        journal = PersistentWallBudget(wall_path, digest(identity), min(limits["max_wall_seconds"], limits["max_gpu_seconds"]))
        try:
            journal.start_interval()
            access = _register_access(root, frozen, claim)
            catalog = load_episode_catalog(root, role="internal_holdout", purpose="internal_development_check",
                internal_access_receipt=access)
            if (list(catalog.partition_people) != frozen["internal_participants"] or len(catalog.partition_people) != 11
                    or {p.participant_id for p in catalog.plans if p.query_eligible} != set(catalog.partition_people)):
                raise InternalValidationError("all_frozen_eleven_internal_people_required_no_replacement_or_subset")
            eligible = [p for p in catalog.plans if p.query_eligible]
            required_keys = {digest([r["method"], r["seed"], p.participant_id, p.session_id])
                             for r in frozen["selected_jobs"] for p in eligible}
            _phase_gate(state, required_keys)
            contract = _json(_reference(root, frozen["input_contract"]))
            for row in frozen["selected_jobs"]:
                job = dev.DevelopmentJob(**row["job"]); dependencies = row["dependencies"]
                context = registry.Context.from_verified(root, dependencies["source_f0"]["path"],
                    dependencies["source_reference"]["path"], contract)
                if (context.identity_sha256 != row["source_context_identity_sha256"]
                        or context.pipeline_seed != job.seed or context.split_sha256 != frozen["split_sha256"]
                        or context.source_cache_sha256 != frozen["W2_cache"]["sha256"]):
                    raise InternalValidationError("same_frozen_real_source_reference_context_required")
                dev._validate_dependency_parameters(root, context, job, dependencies)
                dev._on_cuda(context)
                config_path = destination / f"configuration-{job.method}-{job.seed}.json"
                _put(config_path, {"method": job.method, "pipeline_seed": job.seed, "configuration": job.configuration,
                    "whole_package_freeze_sha256": frozen["freeze_sha256"], "selection_uses_internal_data": False})
                configuration = _desc(root, config_path)
                for episode in eligible:
                    key = digest([job.method, job.seed, episode.participant_id, episode.session_id])
                    directory = OUTPUT+"/candidates/"+key
                    candidate = _candidate(context, job, episode, frozen, claim, budget)
                    if key in state["predictions"]:
                        fitted, model, cost = _load_fit(root, directory, context, job, episode, candidate)
                        actual = _prediction(root, directory, context, job, catalog, episode, fitted, model,
                            cost, configuration, frozen, access)
                        if actual != state["predictions"][key]:
                            raise InternalValidationError("saved_prediction_pointer_not_exact_method_seed_session")
                        del fitted
                        continue
                    if journal.exhausted:
                        raise InternalValidationError("internal_cumulative_wall_budget_exhausted")
                    candidate_path = _local(root, directory, exists=False)
                    _dispatch(state, key, candidate_path.exists(), limits["max_fit_attempts"])
                    atomic_json(progress_path, state)  # Charge before any target fit.
                    fitted, model, cost = _fit_once(root, directory, context, job, catalog, episode,
                        candidate, dependencies, runtime, journal, limits["max_calibration_seconds"])
                    prediction = _prediction(root, directory, context, job, catalog, episode, fitted, model,
                        cost, configuration, frozen, access)
                    state["predictions"][key] = prediction
                    atomic_json(progress_path, state); journal.checkpoint()
                    del fitted
                del context
            if journal.exhausted:
                raise InternalValidationError("internal_wall_budget_exhausted_before_label_release")
            _release_barrier(root, destination, state, required_keys, frozen)
            summaries = []
            for row in frozen["selected_jobs"]:
                metrics = []
                for episode in eligible:
                    if journal.exhausted:
                        raise InternalValidationError("internal_wall_budget_exhausted_after_predictions")
                    key = digest([row["method"], row["seed"], episode.participant_id, episode.session_id])
                    score = _score_internal(root, episode, state["predictions"][key],
                        catalog.materialize(episode, device="cpu"), frozen["freeze_sha256"])
                    if key in state["scores"] and state["scores"][key] != score:
                        raise InternalValidationError("committed_internal_score_changed")
                    state["scores"][key] = score; atomic_json(progress_path, state)
                    metrics.append(score); journal.checkpoint()
                summaries.append({"method": row["method"], "seed": row["seed"],
                    **participant_equal_metrics(metrics, expected_participants=catalog.partition_people)})
            journal.graceful_close()
            if journal.exhausted:
                raise InternalValidationError("internal_wall_budget_exhausted_before_immutable_completion")
            previous_receipt = _local(root, OUTPUT+"/receipt.json", exists=False)
            if previous_receipt.exists():
                previous = _json(previous_receipt)
                if (previous.get("status") != "ONE_TIME_INTERNAL_CHECK_COMPLETE"
                        or previous.get("whole_package_freeze") != freeze or previous.get("one_time_claim") != claim
                        or previous.get("predictions") != state["predictions"] or previous.get("scores") != state["scores"]
                        or previous.get("method_seed_summaries") != summaries):
                    raise InternalValidationError("sealed_internal_completion_changed")
                state["phase"] = "COMPLETE_SEALED_NO_RETUNING"
                atomic_json(progress_path, state)
                return {**previous, "receipt": _desc(root, previous_receipt), "idempotent": True}
            result = {"schema_version": SCHEMA, "status": "ONE_TIME_INTERNAL_CHECK_COMPLETE",
                "synthetic": False, "measured": True, "whole_package_freeze": freeze,
                "freeze_sha256": frozen["freeze_sha256"], "one_time_claim": claim,
                "internal_access_receipt": access, "all_predictions_committed": _desc(root, destination / "all_predictions_committed.json"),
                "participants": list(catalog.partition_people), "method_seed_summaries": summaries,
                "eligible_sessions": len(eligible), "ineligible_sessions": [p.to_dict() for p in catalog.plans if not p.query_eligible],
                "predictions": state["predictions"], "scores": state["scores"],
                "fit_attempts_dispatched": state["fit_attempts_dispatched"], "budget": budget,
                "cumulative_budget_report": journal.report(), "all_labels_after_all_predictions": True,
                "retuning_allowed": False, "external_payloads_accessed": False,
                "final_protocol_freeze_sha256": None, "W6_status_mutated": False}
            _put(destination / "receipt.json", result)
            state["phase"] = "COMPLETE_SEALED_NO_RETUNING"
            atomic_json(progress_path, state)
            return {**result, "receipt": _desc(root, destination / "receipt.json")}
        except BaseException as error:
            failure = {"schema_version": SCHEMA,
                "status": "FAILED_OR_PAUSED_PRESERVE_ONE_TIME_CLAIM", "error_type": type(error).__name__,
                "phase": state["phase"], "fit_attempts_dispatched": state["fit_attempts_dispatched"],
                "labels_revealed": state["labels_revealed"], "new_output_or_refit_allowed": False,
                "failed_at_utc": datetime.now(timezone.utc).isoformat()}
            failure_path = destination / "failures" / (digest(failure)+".json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            _put(failure_path, failure)
            atomic_json(destination / "last_failure.json", _desc(root, failure_path))
            raise
        finally:
            journal.close()
