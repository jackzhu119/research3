"""Immutable, label-blind Stieger development prediction commits.

This module never fits, predicts, loads EEG, imports an evaluator, or accepts
EvaluationLabels. Callers supply existing label-free PredictionResult values,
an issued EpisodePlan, measured cost, and explicit local artifact references.
Source/model scientific validity remains the production artifact loader's job.
No external/clinical prediction or payload access is supported by this store.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import torch

from .adapt import PredictionResult
from .contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from .episodes import EpisodePlan, _internal_access, _split


REFERENCE_KEYS = frozenset({"source_f0", "model", "method_code", "configuration",
                            "input_contract", "episode_manifest", "participant_split"})
PAYLOAD_NAMES = frozenset({"logits.npy", "probabilities.npy", "predictions.npy",
                           "identity.json", "episode_plan.json", "cost.json"})
COST_REQUIRED = frozenset({"acquired_calibration_trials", "valid_support_trials", "accessible_labels",
                          "label_access_event_uids", "optimizer_updates", "calibration_wall_seconds"})
COST_OPTIONAL = frozenset({"target_exact_solves", "target_function_evaluations", "gpu_compute_seconds",
                          "acquisition_elapsed_seconds", "unused_acquired_trials", "audit_trials", "fallback_reason"})
SCHEMA = "paper3-development-prediction-store-v1"


class PredictionStoreError(ValueError):
    """Fixed non-sensitive failure codes; existing artifacts are never removed."""


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PredictionStoreError("concrete_sha256_required")
    return value


def _blocked(relative):
    tokens = re.sub(r"[^a-z0-9]+", "_", relative.lower()).split("_")
    return any(token == "hs" or re.fullmatch(r"hs\d+", token)
               or token.startswith(("hsstroke", "hsbci", "wang", "external", "clinical", "sealed"))
               for token in tokens)


def _local(root, relative, *, exists=True):
    if (not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative
            or any(p in {"", ".", ".."} for p in relative.split("/"))
            or any(ord(c) < 32 for c in relative) or _blocked(relative)
            or any(p.lower().startswith(".env") or p.lower() in {".aws", ".ssh", "credentials", "secrets", "legacy_baseline_v1"}
                   for p in relative.split("/"))
            or Path(relative).suffix.lower() in {".npz", ".pkl", ".pickle", ".pem", ".key"}):
        raise PredictionStoreError("unsafe_credential_legacy_or_external_artifact_path")
    cursor = root
    for part in relative.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise PredictionStoreError("symlink_artifact_forbidden")
    resolved = cursor.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise PredictionStoreError("artifact_outside_paper3_root")
    return resolved


def _reference(root, value, *, suffix=None):
    if (not isinstance(value, dict)
            or set(value) not in ({"path", "sha256"}, {"path", "sha256", "bytes"})):
        raise PredictionStoreError("explicit_path_sha256_reference_required")
    if "bytes" in value and (type(value["bytes"]) is not int or value["bytes"] < 0):
        raise PredictionStoreError("referenced_artifact_nonnegative_integer_bytes_required")
    _sha(value["sha256"])
    path = _local(root, value["path"])
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    if (not path.is_file() or (suffixes is not None and path.suffix not in suffixes)
            or ("bytes" in value and path.stat().st_size != value["bytes"])
            or file_digest(path) != value["sha256"]):
        raise PredictionStoreError("referenced_artifact_full_sha256_or_type_mismatch")
    return path


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode()


def _read_json(path):
    return json.loads(path.read_bytes())


def _uids(value, *, minimum=0):
    if (not isinstance(value, (tuple, list)) or len(value) < minimum
            or any(not isinstance(uid, str) or not uid or any(ord(c) < 32 for c in uid) for uid in value)
            or len(set(value)) != len(value)):
        raise PredictionStoreError("unique_original_event_uids_required")
    return tuple(value)


def _validate_plan(root, plan, references):
    if type(plan) is not EpisodePlan or plan.role not in {"meta_validation", "internal_holdout"}:
        raise PermissionError("only_stieger_validation_or_authorized_internal_predictions_supported")
    for sha in (plan.manifest_sha256, plan.split_sha256, plan.input_contract_sha256, plan.temporal_audit_sha256):
        _sha(sha)
    if re.fullmatch(r"S[1-9][0-9]*", plan.participant_id) is None or re.fullmatch(r"[1-9][0-9]*", plan.session_id) is None:
        raise PredictionStoreError("native_stieger_participant_session_identity_required")
    acquired = _uids(plan.acquired_event_uids)
    support, query = _uids(plan.support_event_uids), _uids(plan.query_event_uids, minimum=20)
    excluded = [_uids(getattr(plan, key)) for key in
                ("unused_boundary_run_uids", "incomplete_future_run_uids", "query_excluded_uids")]
    if (len(acquired) != 20 or plan.query_eligible is not True or set(query) & set(acquired)
            or tuple(uid for uid in acquired if uid in set(support)) != support
            or any(set(query) & set(values) for values in excluded)):
        raise PredictionStoreError("B20_prefix_future_query_or_support_identity_invalid")
    try:
        audit = json.loads(plan.temporal_audit_json)
    except (TypeError, ValueError):
        raise PredictionStoreError("actual_temporal_audit_json_required") from None
    if not isinstance(audit, dict) or digest(audit) != plan.temporal_audit_sha256:
        raise PredictionStoreError("temporal_audit_digest_mismatch")
    fields = {"dataset": "Stieger2021", "participant_id": plan.participant_id, "session_id": plan.session_id,
        "role": plan.role, "manifest_sha256": plan.manifest_sha256, "split_sha256": plan.split_sha256,
        "input_contract_sha256": plan.input_contract_sha256, "query_eligible": True, "budget": 20,
        "acquired_calibration_trials": 20, "query_rule": plan.query_rule,
        "support_fallback_reason": plan.fallback_reason}
    if any(audit.get(key) != value for key, value in fields.items()):
        raise PredictionStoreError("episode_plan_temporal_audit_binding_mismatch")
    for key, expected in (("acquired_prefix_uids", acquired), ("support_uids", support), ("query_uids", query),
                          ("unused_boundary_run_uids", excluded[0]),
                          ("incomplete_or_unverified_future_run_uids", excluded[1]), ("query_excluded_uids", excluded[2])):
        if tuple(audit.get(key, [])) != expected:
            raise PredictionStoreError("episode_plan_original_uid_audit_mismatch")
    if plan.query_rule not in {"future_complete_physical_runs", "verified_continuous_original_trial_block"}:
        raise PredictionStoreError("registered_future_query_rule_required")
    if any(not isinstance(ref, dict)
           or set(ref) not in ({"path", "sha256"}, {"path", "sha256", "bytes"})
           or not isinstance(ref["path"], str)
           for ref in references.values()):
        raise PredictionStoreError("explicit_path_sha256_reference_required")
    if any("bytes" in ref and (type(ref["bytes"]) is not int or ref["bytes"] < 0)
           for ref in references.values()):
        raise PredictionStoreError("referenced_artifact_nonnegative_integer_bytes_required")
    for ref in references.values():
        _local(root, ref["path"], exists=False)  # Path-only checks never read a body.
    if not references["method_code"]["path"].startswith("p3work/"):
        raise PredictionStoreError("new_p3work_method_code_reference_required")
    if not references["model"]["path"].startswith(("runs/", "checkpoints/")):
        raise PredictionStoreError("explicit_source_or_candidate_model_artifact_required")
    if not references["source_f0"]["path"].startswith(("runs/", "manifests/")):
        raise PredictionStoreError("explicit_source_f0_receipt_metadata_required")
    # Pin the metadata manifest BEFORE opening any supplied manifest path: a
    # mistaken events/label descriptor must not be read as episode metadata.
    tasks = _read_json(_local(root, "status/tasks.json"))
    if (tasks.get("W3", {}).get("status") != "COMPLETE"
            or tasks["W3"].get("cache_manifest") != references["episode_manifest"]
            or references["episode_manifest"]["sha256"] != plan.manifest_sha256):
        raise PredictionStoreError("W3_complete_receipt_bound_episode_metadata_manifest_required")
    paths = {key: _reference(root, references[key], suffix=".py" if key == "method_code" else
                            (".pt", ".json", ".npy") if key == "model" else ".json") for key in REFERENCE_KEYS}
    contract = _read_json(paths["input_contract"])
    if validate_signal_contract(contract) != plan.input_contract_sha256:
        raise PredictionStoreError("prediction_input_contract_differs_from_episode")
    split = _split(paths["participant_split"])
    if split["split_sha256"] != plan.split_sha256 or plan.participant_id not in split["partitions"][plan.role]:
        raise PredictionStoreError("prediction_participant_not_in_frozen_partition")
    manifest = _read_json(paths["episode_manifest"])
    if (file_digest(paths["episode_manifest"]) != plan.manifest_sha256
            or manifest.get("schema_version") != "paper3-stieger-cache-v2"
            or manifest.get("status") != "STIEGER_CACHE_COMPLETE" or manifest.get("dataset") != "Stieger2021"
            or manifest.get("synthetic") is not False or manifest.get("input_contract_sha256") != plan.input_contract_sha256
            or manifest.get("input_contract") != references["input_contract"]
            or manifest.get("participant_split") != references["participant_split"]):
        raise PredictionStoreError("actual_stieger_cache_manifest_and_contract_references_required")
    sessions = [s for s in manifest.get("sessions", []) if s.get("participant_id") == plan.participant_id
                and str(s.get("session_id")) == plan.session_id]
    if len(sessions) != 1:
        raise PredictionStoreError("unique_registered_native_session_required")
    session = sessions[0]
    if (session.get("partition") != plan.role or audit.get("source_record_id") != session.get("source_record_id")
            or audit.get("source_file_sha256") != session.get("source_file_sha256")
            or audit.get("events_sha256") != session.get("events", {}).get("sha256")):
        raise PredictionStoreError("prediction_temporal_audit_differs_from_registered_session")
    originals = audit.get("original_event_audit")
    if not isinstance(originals, list) or not originals:
        raise PredictionStoreError("label_blind_original_event_coordinates_required")
    original_uids = _uids([row.get("event_uid") for row in originals])
    by_uid = dict(zip(original_uids, originals))
    original_indices = [row.get("original_trial_index") for row in originals]
    if (original_uids[:20] != acquired
            or any(type(index) is not int or index < 0 for index in original_indices)
            or any(a >= b for a, b in zip(original_indices, original_indices[1:]))
            or any(row["event_uid"] != f"Stieger2021:v1:{plan.participant_id}:session{plan.session_id}:{audit['source_file_sha256']}:trial{row['original_trial_index']}"
                   for row in originals)
            or not set(acquired + query) <= set(original_uids)
            or any(by_uid[uid].get("valid") is not True for uid in support + query)
            or tuple(uid for uid in acquired if by_uid[uid].get("valid") is True) != support
            or any(any(key in row for key in ("label", "labels", "y", "query_y")) for row in originals)):
        raise PredictionStoreError("original_trial_metadata_missing_invalid_or_not_label_blind")
    runs = [row.get("physical_run_id") for row in originals]
    future, discarded, incomplete = originals[20:], [], []
    if all(run is None for run in runs):
        if plan.query_rule != "verified_continuous_original_trial_block":
            raise PredictionStoreError("future_query_rule_differs_from_native_run_evidence")
    elif any(run is None for run in runs) or plan.query_rule != "future_complete_physical_runs":
        raise PredictionStoreError("partial_or_inconsistent_physical_run_identity")
    else:
        seen, previous = set(), object()
        for run in runs:
            if run != previous:
                if run in seen:
                    raise PredictionStoreError("noncontiguous_physical_run_identity")
                seen.add(run); previous = run
        boundary = runs[19]
        discarded = [row for row in future if row["physical_run_id"] == boundary]
        future = [row for row in future if row["physical_run_id"] != boundary]
        complete = {run for run in {row["physical_run_id"] for row in future}
                    if all(row.get("physical_run_complete_verified") is True for row in future if row["physical_run_id"] == run)}
        incomplete = [row for row in future if row["physical_run_id"] not in complete]
        future = [row for row in future if row["physical_run_id"] in complete]
    expected = (tuple(row["event_uid"] for row in future if row.get("valid") is True),
                tuple(row["event_uid"] for row in discarded), tuple(row["event_uid"] for row in incomplete),
                tuple(row["event_uid"] for row in future if row.get("valid") is not True))
    if expected != (query, *excluded):
        raise PredictionStoreError("future_complete_original_trial_query_does_not_match_temporal_audit")
    source = _read_json(paths["source_f0"])
    if (source.get("status") != "SOURCE_F0_COMPLETE" or source.get("synthetic") is not False
            or set(source.get("training_dataset_ids", [])) != SOURCE_IDS
            or source.get("signal_contract_sha256") != plan.input_contract_sha256):
        raise PredictionStoreError("real_four_source_f0_receipt_reference_required")
    return audit


def _validate_access(root, plan, purpose, selected_sha, internal_access_receipt):
    if plan.role == "meta_validation":
        if purpose not in {"model_selection", "development_evaluation"} or internal_access_receipt is not None:
            raise PermissionError("validation_prediction_purpose_invalid")
        return None
    if purpose != "internal_development_check" or internal_access_receipt is None or selected_sha is None:
        raise PermissionError("internal_predictions_require_explicit_selected_development_check_access")
    _reference(root, internal_access_receipt, suffix=".json")
    tasks = _read_json(_local(root, "status/tasks.json"))
    access, _ = _internal_access(root, tasks, internal_access_receipt, plan.split_sha256, plan.input_contract_sha256)
    if access.get("selected_configuration_sha256") != selected_sha:
        raise PermissionError("internal_selected_configuration_differs_from_access_receipt")
    return dict(internal_access_receipt)


def _cost(plan, method, cost):
    if not isinstance(cost, dict) or not COST_REQUIRED <= set(cost) or not set(cost) <= COST_REQUIRED | COST_OPTIONAL:
        raise PredictionStoreError("explicit_three_ledger_calibration_cost_required")
    result = json.loads(_json_bytes(cost))
    count_fields = {"acquired_calibration_trials", "valid_support_trials", "accessible_labels", "optimizer_updates",
                    "target_exact_solves", "target_function_evaluations", "unused_acquired_trials", "audit_trials"}
    for key in count_fields & set(result):
        if type(result[key]) is not int or result[key] < 0:
            raise PredictionStoreError("actual_calibration_counts_must_be_nonnegative_integers")
    labels = _uids(result["label_access_event_uids"])
    if (result["acquired_calibration_trials"] != 20 or result["valid_support_trials"] != len(plan.support_event_uids)
            or result["accessible_labels"] != len(labels)
            or not set(labels) <= set(plan.support_event_uids)
            or tuple(uid for uid in plan.support_event_uids if uid in set(labels)) != labels
            or result["optimizer_updates"] > 200
            or (method == "B0" and (labels or result["optimizer_updates"]))):
        raise PredictionStoreError("calibration_acquisition_label_or_actual_update_ledger_mismatch")
    for key in ("calibration_wall_seconds", "gpu_compute_seconds", "acquisition_elapsed_seconds"):
        if key not in result or (key == "acquisition_elapsed_seconds" and result[key] is None):
            continue
        if type(result[key]) not in {int, float} or not math.isfinite(result[key]) or result[key] < 0:
            raise PredictionStoreError("measured_calibration_duration_must_be_finite_nonnegative")
    if result.get("audit_trials", 0) != 0:
        raise PredictionStoreError("current_fixed_B20_store_does_not_authorize_sequential_audit")
    if result.get("fallback_reason") is not None and (not isinstance(result["fallback_reason"], str)
            or re.fullmatch(r"[A-Za-z0-9_]+", result["fallback_reason"]) is None):
        raise PredictionStoreError("fixed_non_sensitive_fallback_code_required")
    return result


def _arrays(result, plan):
    if type(result) is not PredictionResult:
        raise TypeError("existing_label_free_PredictionResult_required")
    if (result.protocol_freeze_sha != plan.input_contract_sha256 or tuple(result.event_uids) != plan.query_event_uids
            or len(set(result.event_uids)) != len(result.event_uids)):
        raise PredictionStoreError("prediction_uid_order_or_input_contract_mismatch")
    logits, probabilities, predictions = result.logits.detach().cpu(), result.probabilities.detach().cpu(), result.labels.detach().cpu()
    if (logits.shape != (len(plan.query_event_uids), 2) or probabilities.shape != logits.shape
            or predictions.shape != (len(logits),) or logits.dtype not in {torch.float32, torch.float64}
            or probabilities.dtype != logits.dtype or predictions.dtype not in {torch.int32, torch.int64}
            or not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all()
            or bool((probabilities < 0).any()) or bool((probabilities > 1).any())
            or not torch.allclose(probabilities, logits.softmax(1), atol=2e-7 if logits.dtype == torch.float32 else 1e-12,
                                  rtol=2e-6 if logits.dtype == torch.float32 else 1e-12)
            or not torch.equal(predictions, logits.argmax(1))):
        raise PredictionStoreError("same_logit_softmax_and_argmax_predictions_required")
    return {"logits.npy": logits.contiguous().numpy(), "probabilities.npy": probabilities.contiguous().numpy(),
            "predictions.npy": predictions.to(torch.int64).contiguous().numpy()}


def _npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _row(root, path, body):
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body), "role": "result", "dataset": "Stieger2021"}


def _write_identical_or_new(path, body):
    if path.is_symlink():
        raise PredictionStoreError("symlink_artifact_forbidden")
    if path.exists():
        if not path.is_file() or path.read_bytes() != body:
            raise PredictionStoreError("immutable_prediction_artifact_differs_no_overwrite")
        return
    fd, temporary = tempfile.mkstemp(prefix=".prediction-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Atomic no-overwrite commit, even on races.
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != body:
                raise PredictionStoreError("immutable_prediction_artifact_differs_no_overwrite") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


def commit_predictions(root, output, result, plan, *, method, pipeline_seed, references, cost,
                       purpose="model_selection", selected_configuration_sha256=None,
                       internal_access_receipt=None):
    """Commit already computed predictions before any internal label access.

    Content-identical retries are idempotent. Existing different or corrupted
    artifacts are preserved and rejected. A durable pending identity permits a
    retry after interruption; receipt.json is atomically created LAST.
    """
    root = Path(root).resolve(strict=True)
    if isinstance(output, Path):
        if output.is_absolute():
            try:
                output = output.relative_to(root)
            except ValueError:
                raise PredictionStoreError("artifact_outside_paper3_root") from None
        output = output.as_posix()
    if (not isinstance(output, str) or not output.startswith(("runs/", "predictions/"))
            or not isinstance(method, str) or re.fullmatch(r"[A-Za-z0-9_+-]{1,64}", method) is None
            or type(pipeline_seed) is not int or pipeline_seed not in range(20261018, 20261023)
            or not isinstance(references, dict) or set(references) != REFERENCE_KEYS):
        raise PredictionStoreError("explicit_safe_output_method_pipeline_seed_and_artifact_references_required")
    destination = _local(root, output, exists=False)
    audit = _validate_plan(root, plan, references)
    source = _read_json(_reference(root, references["source_f0"], suffix=".json"))
    if source.get("pipeline_seed", source.get("seed")) != pipeline_seed:
        raise PredictionStoreError("source_f0_and_prediction_pipeline_seed_mismatch")
    configuration = _read_json(_reference(root, references["configuration"], suffix=".json"))
    declared_method = configuration.get("method", configuration.get("job", {}).get("method"))
    if declared_method is not None and declared_method != method:
        raise PredictionStoreError("configuration_method_differs_from_prediction_identity")
    selected_sha = _sha(selected_configuration_sha256) if selected_configuration_sha256 is not None else references["configuration"]["sha256"]
    access = _validate_access(root, plan, purpose, selected_sha, internal_access_receipt)
    measured_cost, arrays = _cost(plan, method, cost), _arrays(result, plan)
    identity = {"schema_version": SCHEMA, "dataset": "Stieger2021", "participant_id": plan.participant_id,
        "session_id": plan.session_id, "prediction_role": plan.role, "purpose": purpose,
        "method": method, "pipeline_seed": pipeline_seed, "references": json.loads(_json_bytes(references)),
        "selected_configuration_sha256": selected_sha, "internal_access_receipt": access,
        "episode_plan_sha256": plan.temporal_audit_sha256, "full_episode_plan_sha256": digest(plan.to_dict()),
        "input_contract_sha256": plan.input_contract_sha256, "split_sha256": plan.split_sha256,
        "labels_accessed_before_predictions": False}
    bodies = {name: _npy_bytes(value) for name, value in arrays.items()}
    bodies.update({"identity.json": _json_bytes(identity), "episode_plan.json": _json_bytes(plan.to_dict()),
                   "cost.json": _json_bytes(measured_cost)})
    content_sha = digest({"identity": identity, "payload_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in bodies.items()}})
    destination.mkdir(parents=True, exist_ok=True)
    _local(root, output)  # Recheck components after creating directories.
    lock_path = _local(root, output+"/.prediction_store.lock", exists=False)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt_path = _local(root, output+"/receipt.json", exists=False)
        manifest_path = destination / "artifact_manifest.json"
        if receipt_path.exists():
            descriptor = {"path": receipt_path.relative_to(root).as_posix(), "sha256": file_digest(receipt_path)}
            previous = verify_prediction_receipt(root, descriptor)
            if previous.get("content_identity_sha256") != content_sha:
                raise PredictionStoreError("different_prediction_identity_refuses_existing_destination")
            return {"receipt": descriptor, "artifact_manifest": {"path": manifest_path.relative_to(root).as_posix(),
                    "sha256": file_digest(manifest_path)}, "idempotent": True}
        pending = _local(root, output+"/.pending_identity.json", exists=False)
        if pending.exists():
            transaction = _read_json(pending)
            if transaction.get("content_identity_sha256") != content_sha:
                raise PredictionStoreError("different_prediction_identity_refuses_pending_destination")
        else:
            if any(path.name != ".prediction_store.lock" for path in destination.iterdir()):
                raise PredictionStoreError("unowned_existing_prediction_destination_preserved")
            transaction = {"schema_version": SCHEMA, "content_identity_sha256": content_sha,
                           "transaction_started_at_utc": datetime.now(timezone.utc).isoformat()}
            _write_identical_or_new(pending, _json_bytes(transaction))
        rows = [_row(root, destination / name, bodies[name]) for name in sorted(bodies)]
        payload = _json_bytes({"schema_version": 1, "identity_sha256": digest(identity), "files": rows})
        receipt = {**identity, "status": "PREDICTIONS_COMMITTED", "content_identity_sha256": content_sha,
            "query_event_uids": list(plan.query_event_uids), "acquired_event_uids": list(plan.acquired_event_uids),
            "support_event_uids": list(plan.support_event_uids), "valid_query_trials": len(plan.query_event_uids),
            "source_record_id": audit["source_record_id"], "source_file_sha256": audit["source_file_sha256"],
            "payload_manifest": {"path": (destination / "payload_manifest.json").relative_to(root).as_posix(),
                                 "sha256": hashlib.sha256(payload).hexdigest()},
            "transaction_started_at_utc": transaction["transaction_started_at_utc"],
            "query_labels_in_store": False, "prediction_values_supplied_not_computed_here": True}
        receipt_body = _json_bytes(receipt)
        backup_rows = rows + [_row(root, destination / "payload_manifest.json", payload), _row(root, receipt_path, receipt_body)]
        artifact_body = _json_bytes({"schema_version": 1, "files": sorted(backup_rows, key=lambda row: row["path"])})
        for name, body in sorted(bodies.items()):
            _write_identical_or_new(destination / name, body)
        _write_identical_or_new(destination / "payload_manifest.json", payload)
        _write_identical_or_new(manifest_path, artifact_body)
        _write_identical_or_new(receipt_path, receipt_body)
        descriptor = {"path": receipt_path.relative_to(root).as_posix(), "sha256": file_digest(receipt_path)}
        verify_prediction_receipt(root, descriptor)
        return {"receipt": descriptor, "artifact_manifest": {"path": manifest_path.relative_to(root).as_posix(),
                "sha256": file_digest(manifest_path)}, "idempotent": False}


def verify_prediction_receipt(root, descriptor):
    """Read-only full SHA, identity, cost, and same-logit prediction verification."""
    root = Path(root).resolve(strict=True)
    path = _reference(root, descriptor, suffix=".json")
    receipt = _read_json(path)
    if (receipt.get("schema_version") != SCHEMA or receipt.get("status") != "PREDICTIONS_COMMITTED"
            or receipt.get("dataset") != "Stieger2021" or receipt.get("prediction_role") not in {"meta_validation", "internal_holdout"}
            or receipt.get("labels_accessed_before_predictions") is not False
            or receipt.get("query_labels_in_store") is not False):
        raise PredictionStoreError("committed_label_blind_development_prediction_receipt_required")
    payload_path = _reference(root, receipt.get("payload_manifest"), suffix=".json")
    if payload_path.parent != path.parent or payload_path.name != "payload_manifest.json":
        raise PredictionStoreError("prediction_payload_manifest_directory_mismatch")
    manifest = _read_json(payload_path)
    rows = manifest.get("files")
    if not isinstance(rows, list) or {Path(r.get("path", "")).name for r in rows} != PAYLOAD_NAMES or len(rows) != len(PAYLOAD_NAMES):
        raise PredictionStoreError("exact_prediction_payload_coverage_required")
    files = {}
    for row in rows:
        target = _reference(root, {k: row[k] for k in ("path", "sha256")})
        if (target.parent != path.parent or target.stat().st_size != row.get("bytes")
                or row.get("role") != "result" or row.get("dataset") != "Stieger2021"):
            raise PredictionStoreError("prediction_payload_exact_path_bytes_or_role_mismatch")
        files[target.name] = target
    identity = _read_json(files["identity.json"])
    if manifest.get("identity_sha256") != digest(identity) or any(receipt.get(k) != v for k, v in identity.items()):
        raise PredictionStoreError("prediction_identity_binding_mismatch")
    content_sha = digest({"identity": identity, "payload_sha256": {r["path"].split("/")[-1]: r["sha256"] for r in rows}})
    if receipt.get("content_identity_sha256") != content_sha:
        raise PredictionStoreError("prediction_content_identity_digest_mismatch")
    plan_dict = _read_json(files["episode_plan.json"])
    temporal = plan_dict.pop("temporal_audit")
    plan_dict["temporal_audit_json"] = json.dumps(temporal, sort_keys=True, separators=(",", ":"), allow_nan=False)
    for key in ("acquired_event_uids", "support_event_uids", "query_event_uids", "unused_boundary_run_uids",
                "incomplete_future_run_uids", "query_excluded_uids"):
        plan_dict[key] = tuple(plan_dict[key])
    plan = EpisodePlan(**plan_dict)
    _validate_plan(root, plan, identity["references"])
    source = _read_json(_reference(root, identity["references"]["source_f0"], suffix=".json"))
    if source.get("pipeline_seed", source.get("seed")) != identity["pipeline_seed"]:
        raise PredictionStoreError("source_f0_and_prediction_pipeline_seed_mismatch")
    configuration = _read_json(_reference(root, identity["references"]["configuration"], suffix=".json"))
    declared_method = configuration.get("method", configuration.get("job", {}).get("method"))
    if declared_method is not None and declared_method != identity["method"]:
        raise PredictionStoreError("configuration_method_differs_from_prediction_identity")
    _validate_access(root, plan, identity["purpose"], identity["selected_configuration_sha256"], identity["internal_access_receipt"])
    _cost(plan, identity["method"], _read_json(files["cost.json"]))
    if (receipt.get("episode_plan_sha256") != plan.temporal_audit_sha256
            or receipt.get("full_episode_plan_sha256") != digest(plan.to_dict())
            or tuple(receipt.get("query_event_uids", [])) != plan.query_event_uids
            or tuple(receipt.get("acquired_event_uids", [])) != plan.acquired_event_uids
            or tuple(receipt.get("support_event_uids", [])) != plan.support_event_uids):
        raise PredictionStoreError("prediction_receipt_episode_uid_or_digest_mismatch")
    prediction = PredictionResult(torch.from_numpy(np.load(files["logits.npy"], allow_pickle=False)),
        torch.from_numpy(np.load(files["probabilities.npy"], allow_pickle=False)),
        torch.from_numpy(np.load(files["predictions.npy"], allow_pickle=False)),
        plan.query_event_uids, plan.input_contract_sha256)
    _arrays(prediction, plan)
    backup_path = path.with_name("artifact_manifest.json")
    backup = _read_json(_local(root, backup_path.relative_to(root).as_posix()))
    expected = rows + [_row(root, payload_path, payload_path.read_bytes()), _row(root, path, path.read_bytes())]
    if backup != {"schema_version": 1, "files": sorted(expected, key=lambda r: r["path"])}:
        raise PredictionStoreError("prediction_backup_manifest_exact_file_set_mismatch")
    return receipt
