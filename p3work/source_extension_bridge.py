"""Independent five-source capabilities; never accepted by the primary bridge.

Native metadata and complete raw backup proofs remain mandatory. Loading an
extension does not fit a model, contact a provider, open a query, or use CUDA.
Unknown absolute units are accepted only with recorded common-gain evidence;
the immutable source kernel applies float64 joint RMS before float32 EEGNet.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np
import torch

from . import source_training as core
from .contracts import CHANNELS, SOURCE_IDS, digest, file_digest, validate_signal_contract
from .f0_bridge import CompactEEGNet, load_verified_source
from .adapt import FrozenSourceModel
from .training_state import local_path

BASE_MANIFEST = "cache/source_v2/source_cache.json"
BASE_SHA = "bfc204577dfe51c1fc798c1241391ab63509fc2ad9ebfb267ba09254e01ae644"
INPUT_SHA = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
ARMS = {"BASE4_PLUS_YANG": "Yang2025", "BASE4_PLUS_DREYER": "Dreyer2023"}
SEEDS = tuple(range(20261018, 20261023))
STAGE = "pre_rms_car11_temporal_demeaned"
PARALLEL_DREYER_PLAN_SHA = "afc8dd0d2d873f455e901c6f702fc2d5649fde15f76873c40a24424c10d48e5a"
PARALLEL_DREYER_PLAN = "audit/source_extensions_20261019/Dreyer2023/plan_after_all520_native_review_v5.json"
PARALLEL_DREYER_PLAN_FILE_SHA = "440c78ca7d1d2a3be685041924bdfe1fbcdd5ad9683592f97c39d50184ad2967"
PARALLEL_YANG_PLAN_SHA = "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
YANG_MEMBER_PLAN = f"audit/source_extensions_20261019/Yang2025/{PARALLEL_YANG_PLAN_SHA}/frozen_plan.json"
YANG_MEMBER_PLAN_FILE_SHA = "55c7a36d339e2923d0c3fec45072d660127ff84f9a65dfbc3805f9259d3b3fe6"
YANG_MEMBER_TRANSPORT = "tools/resume_yang_source_transport_v2.py"
YANG_MEMBER_TRANSPORT_SHA = "2ae1e75e47a75032155da44c1cb45afa78a1ec82da3ecd11d6f85ec60e1c0f71"
YANG_MEMBER_HISTORY = "audit/data_plan_20261019/yang/single_session_download_receipt.json"
YANG_MEMBER_HISTORY_SHA = "2c90692b775985ab5fb68d2e92ea31513ab9e88b3318dfc9f6daa8b1edd08c9f"
YANG_MEMBER_ORIGINAL_ROOT = "/workspace/research3"
PARALLEL_WRAPPER = "tools/prepare_dreyer_source_cpu_v5.py"
PARALLEL_WRAPPER_SHA = "c604a0189993ec6e77dd4af0dfb6812b38d69791e691f066b83278eff06b7126"
PARALLEL_PARENT_WRAPPER = "tools/prepare_dreyer_source_cpu_v4.py"
PARALLEL_PARENT_WRAPPER_SHA = "3bdfd19d9d1596f92b689121f8231d8b2403449ecbc26dd0b27c0f7ddee59a1a"
DREYER_NATIVE = "p3work/source_extensions/dreyer2023.py"
DREYER_NATIVE_SHA = "5c31e969eeabd17634864f6783d8473a0a79f1888faf274d758cd9c5ba4cef77"
DREYER_QUALIFICATION = "audit/data_plan_20261020/dreyer_structure_case_review/all520_closed_qualification_proof.json"
DREYER_QUALIFICATION_SHA = "585ade4a3d6101c3d922466fbcacbc89ddac907ff6978a9a0f608c834737d35e"
DREYER_ADMISSION = "audit/data_plan_20261020/dreyer_structure_case_review/v5_native_admission.json"
DREYER_ADMISSION_SHA = "560ec9bdd0ddba53f5b926beaebdd153c61fa0ca8de1e2033273f9d8d022eed9"
_ISSUER = object()


class ExtensionGateError(ValueError):
    """Non-sensitive, closed source-extension provenance failure."""


def local(root, relative, *, exists=True):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or any(ord(c) < 32 for c in relative)
            or any(p.lower().startswith(".env") or p.lower() in
                   {".aws", ".ssh", "secrets", "credentials", "legacy_baseline_v1",
                    "external", "sealed", "query", "internal_holdout"} for p in Path(relative).parts)
            or Path(relative).suffix.lower() in {".npz", ".pkl", ".pickle", ".pem", ".key"}):
        raise ExtensionGateError("unsafe_or_non_source_extension_path")
    path = local_path(root, relative)
    if exists and not path.is_file():
        raise ExtensionGateError("source_extension_artifact_missing")
    return path


def descriptor(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path),
            "bytes": path.stat().st_size}


def artifact(root, item, closure=None):
    if (not isinstance(item, dict) or not {"path", "sha256"} <= set(item)
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256"))) is None):
        raise ExtensionGateError("source_extension_artifact_descriptor_required")
    path = local(root, item["path"])
    actual = descriptor(root, path)
    if actual["sha256"] != item["sha256"] or ("bytes" in item and actual["bytes"] != item["bytes"]):
        raise ExtensionGateError("source_extension_artifact_SHA_or_size_mismatch")
    if closure is not None:
        previous = closure.setdefault(actual["path"], actual["sha256"])
        if previous != actual["sha256"]:
            raise ExtensionGateError("conflicting_source_extension_artifact")
    return path


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, UnicodeError) as error:
        raise ExtensionGateError("invalid_source_extension_JSON") from error


def baseline_epoch(root, receipt_path, seed):
    """Read the REAL baseline-four fit and all four LODO proofs, preserving RNG."""
    if type(seed) is not int or seed not in SEEDS:
        raise ExtensionGateError("registered_pipeline_seed_required")
    contract = read_json(local(root, "contracts/input_contract.json"))
    if validate_signal_contract(contract) != INPUT_SHA or file_digest(local(root, BASE_MANIFEST)) != BASE_SHA:
        raise ExtensionGateError("immutable_baseline_four_input_required")
    receipt_file = local(root, receipt_path)
    # This loader verifies actual checkpoints, every LODO receipt, training UIDs,
    # epoch selection, and baseline cache SHA. It is NOT weakened or patched.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        source, receipt = load_verified_source(root, receipt_path, contract)
    source.assert_unchanged()
    if (receipt.get("pipeline_seed") != seed or receipt.get("seed") != seed
            or receipt.get("source_cache_manifest_sha256") != BASE_SHA):
        raise ExtensionGateError("same_seed_real_baseline_final_f0_required")
    lodo_file = artifact(root, receipt["source_lodo_receipt"])
    lodo = read_json(lodo_file)
    epochs = lodo.get("final_epochs")
    selected = lodo.get("fold_selected_epochs")
    if (not isinstance(selected, list) or len(selected) != 4
            or type(epochs) is not int or epochs != core.final_epoch_from_lodo(selected)
            or receipt.get("configuration", {}).get("effective_epochs") != epochs):
        raise ExtensionGateError("actual_baseline_four_LODO_epoch_evidence_required")
    return {"seed": seed, "epochs": epochs, "baseline_f0": descriptor(root, receipt_file),
            "baseline_LODO": descriptor(root, lodo_file), "fold_selected_epochs": selected,
            "baseline_checkpoint": receipt["checkpoint"],
            "policy": "reuse_actual_baseline_four_same_seed_final_epochs_no_new_LODO_search"}


def _unit_qc(root, row, closure, *, native_metadata=None, native_profile=None):
    qc = row.get("preprocessing_qc")
    if (not isinstance(qc, dict) or qc.get("normalization_applied") is not False
            or qc.get("reference") != "CAR_exact_clinic11"):
        raise ExtensionGateError("pre_RMS_CAR11_extension_QC_required")
    evidence = read_json(artifact(root, row.get("unit_evidence"), closure))
    if not isinstance(evidence, dict):
        raise ExtensionGateError("actual_native_unit_evidence_object_required")
    if (native_metadata is not None and (evidence.get("dataset") != row.get("dataset")
            or evidence.get("native_metadata") != native_metadata
            or evidence.get("native_profile") != native_profile)):
        raise ExtensionGateError("unit_evidence_must_bind_this_native_record_and_profile")
    unit = qc.get("stored_unit")
    if unit == "V":
        native_units = evidence.get("native_units", evidence.get("native_signal_units", {}))
        if (qc.get("si_units_verified") is not True or evidence.get("si_units_verified") is not True
                or any(native_units.get(name) not in {"V", "uV", "µV", "mV"} for name in CHANNELS)):
            raise ExtensionGateError("SI_volt_evidence_required")
    elif unit == "calibrated_numeric_unknown_SI":
        calibration = evidence.get("calibration", evidence.get("common_gain_evidence", {}))
        try:
            scale = Fraction(calibration["common_scale_rational"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            raise ExtensionGateError("actual_positive_common_native_calibration_required") from None
        if (qc.get("common_positive_gain_verified") is not True
                or evidence.get("common_positive_gain_verified") is not True
                or calibration.get("common_positive_gain_verified") is not True or scale <= 0
                or calibration.get("SI_unit_verified") is not False):
            raise ExtensionGateError("unknown_SI_requires_verified_common_positive_gain")
    else:
        raise ExtensionGateError("unsupported_extension_stored_unit")
    if qc.get("per_channel_gain_unknown") is True:
        raise ExtensionGateError("joint_RMS_does_not_remove_unknown_relative_channel_gain")


def _receipt_entries(root, item, closure):
    """Offline validation only: the complete PUT/GET proof, under the new profile."""
    from . import source_extension_io
    # The original public restore gate deliberately rejects these new sources.
    # Reuse the new document-bound private profile, never modify its globals.
    for relative, module_file in (("p3work/source_extension_io.py", source_extension_io.__file__),):
        if Path(module_file).read_bytes() != local(root, relative).read_bytes():
            raise ExtensionGateError("executed_extension_IO_code_differs_from_root")
        closure[relative] = file_digest(local(root, relative))
    _, restore = source_extension_io._modules()
    for relative in ("tools/r2_backup.py", "tools/r2_restore.py"):
        closure[relative] = file_digest(local(root, relative))
    entries, _ = restore.load_verified_receipt(artifact(root, item, closure))
    return entries


def _native_rows(root, shard, dataset, closure):
    """Bind every cache event to original native metadata and complete raw GET."""
    metadata = read_json(artifact(root, shard["native_metadata"], closure))
    records = metadata.get("trials", metadata.get("events")) if isinstance(metadata, dict) else metadata
    if not isinstance(records, list) or not records:
        raise ExtensionGateError("native_extension_original_trials_required")
    if isinstance(metadata, dict) and not (metadata.get("waveform_QC_applied") is False
            or metadata.get("derived_before_waveform_preprocessing") is True):
        raise ExtensionGateError("waveform_QC_independent_native_metadata_required")
    backed = _receipt_entries(root, shard["raw_backup_receipt"], closure)
    raw_files = shard.get("raw_files")
    if (not isinstance(raw_files, list) or not raw_files
            or len(backed) != len(raw_files)
            or {(r["path"], r["sha256"], r["bytes"], r["role"], r["dataset"]) for r in backed}
                != {(r["path"], r["sha256"], r["bytes"], "source_development_raw", dataset) for r in raw_files}):
        raise ExtensionGateError("complete_raw_GET_must_match_exact_frozen_record_components")
    raw_sha = shard.get("raw_sha256")
    raw = [r for r in backed if r.get("sha256") == raw_sha and r.get("dataset") == dataset]
    if len(raw) != 1 or raw[0].get("role") not in {"source_development_raw", "source_train_raw"}:
        raise ExtensionGateError("actual_full_GET_backed_extension_raw_required")
    raw_event_sha = None
    if dataset == "Yang2025":
        raw_files = metadata.get("raw_files", {})
        signal_file, event_file = raw_files.get("signal", {}), raw_files.get("event", {})
        raw_event_sha = event_file.get("sha256")
        event_proof = [r for r in backed if r.get("sha256") == raw_event_sha and r.get("dataset") == dataset]
        if (metadata.get("dataset_id") != "Yang2025_2C" or signal_file.get("sha256") != raw_sha
                or not isinstance(raw_event_sha, str) or len(event_proof) != 1
                or event_proof[0].get("role") not in {"source_development_raw", "source_train_raw"}):
            raise ExtensionGateError("Yang_actual_native_data_event_pair_full_GET_required")
        profile = read_json(artifact(root, shard.get("native_profile"), closure))
        if (profile.get("schema_version") != "paper3-yang-source-profile-v1"
                or profile.get("dataset_id") != "Yang2025_2C" or profile.get("source_only") is not True
                or profile.get("MI_onset_verified") is not False
                or profile.get("context_from_marker_s") != [2.25, 3.75]
                or profile.get("crop_from_marker_s") != [2.5, 3.5]
                or profile.get("input_contract_sha256") != INPUT_SHA
                or profile.get("native_unit_policy") != "common_gain_numeric_unresolved_no_absolute_QC"):
            raise ExtensionGateError("exact_source_only_Yang_analysis_profile_required")
    native = {}
    for row in records:
        if isinstance(row, dict) and isinstance(metadata, dict):
            row = {**{k: metadata[k] for k in ("dataset", "source_file_sha256", "native_sfreq_hz")
                      if k in metadata}, **row}
        if isinstance(row, dict) and dataset == "Yang2025":
            row = {**row, "dataset": dataset, "raw_sha256": raw_sha,
                "raw_event_sha256": raw_event_sha, "original_trial_index": row.get("original_event_index"),
                "canonical_label": {0: "left_hand", 1: "right_hand"}.get(row.get("label")),
                "analysis_anchor_sample": row.get("context_start_sample"),
                "original_context_samples": [row.get("context_start_sample"), row.get("context_stop_sample_exclusive")]}
        uid = row.get("event_uid", row.get("trial_uid")) if isinstance(row, dict) else None
        if (not isinstance(uid, str) or not uid or uid in native or row.get("dataset") != dataset
                or row.get("raw_sha256", row.get("source_file_sha256")) != raw_sha):
            raise ExtensionGateError("native_extension_UID_or_raw_identity_mismatch")
        native[uid] = row
    return native


def _cache_backup(root, shard, dataset, closure):
    keys = ("X", "y", "events", "exclusions", "metadata", "native_metadata", "unit_evidence")
    expected = {(artifact(root, shard[k], closure).relative_to(root).as_posix(),
                 shard[k]["sha256"], shard[k]["bytes"],
                 "result" if k in keys[:5] else "audit", dataset) for k in keys}
    rows = _receipt_entries(root, shard.get("cache_backup_receipt"), closure)
    actual = [(r["path"], r["sha256"], r["bytes"], r["role"], r["dataset"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ExtensionGateError("complete_exact_cache_objects_full_GET_proof_required")


def _completed_preparation(root, manifest, manifest_file, closure):
    """A full cache needs the final GET receipt AND closed, in-budget CPU ledger.

    All checks are read-only; in particular, never instantiate the coordinator's
    mutable Ledger or invoke prepare(). An individual record is insufficient.
    """
    from . import source_extension_coordinator as coordinator
    dataset, plan_sha = manifest["dataset"], manifest.get("plan_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(plan_sha)) is None:
        raise ExtensionGateError("frozen_source_preparation_plan_SHA_required")
    audit = f"audit/source_extensions_20261019/{dataset}/{plan_sha}"
    expected_path = f"cache/source_extensions_20261019/{dataset}/{plan_sha}/source_cache.json"
    if manifest_file != root / expected_path:
        raise ExtensionGateError("canonical_complete_source_cache_path_required")
    plan_file = artifact(root, descriptor(root, local(root, audit+"/frozen_plan.json")), closure)
    plan = read_json(plan_file)
    coordinator.validate_plan(root, plan)
    if (plan.get("plan_sha256") != plan_sha or plan.get("dataset") != dataset
            or manifest.get("official_inventory") != plan.get("inventory")
            or manifest.get("native_profile") != plan.get("native_profile")
            or manifest.get("processing_code") != plan["code"].get("p3work/preprocess.py")
            or manifest.get("coordinator_code") != plan["code"].get("p3work/source_extension_coordinator.py")):
        raise ExtensionGateError("completed_cache_must_match_frozen_coordinator_plan")
    for item in [plan["inventory"], plan["native_profile"], *plan["code"].values()]:
        artifact(root, item, closure)
    status_file = local(root, audit+"/coordinator_status.json")
    status = read_json(artifact(root, descriptor(root, status_file), closure))
    if (status.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE" or status.get("dataset") != dataset
            or status.get("manifest") != descriptor(root, manifest_file)
            or status.get("record_count") != plan.get("record_count")
            or status.get("valid_trials") != manifest.get("valid_trials")
            or status.get("training_started") is not False or status.get("GPU_actions") != 0
            or status.get("external_payload_accessed") is not False):
        raise ExtensionGateError("actual_global_source_coordinator_completion_required")
    final_items = [(descriptor(root, manifest_file), "result"),
        (manifest.get("native_inventory"), "audit"), (manifest.get("unit_evidence"), "audit"),
        (descriptor(root, plan_file), "audit"), (manifest.get("role_assignment"), "audit")]
    final_items.extend(_parallel_evidence(root, manifest, status, audit, closure))
    expected = {(artifact(root, item, closure).relative_to(root).as_posix(), item["sha256"],
                 item["bytes"], role, dataset) for item, role in final_items}
    rows = _receipt_entries(root, status.get("backup_receipt"), closure)
    actual = [(r["path"], r["sha256"], r["bytes"], r["role"], r["dataset"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ExtensionGateError("actual_complete_final_source_manifest_full_GET_required")
    mandatory_r2_bytes = sum(item["bytes"] for item, _ in final_items)
    cache_keys = ("X", "y", "events", "exclusions", "metadata", "native_metadata", "unit_evidence")
    for shard in manifest["shards"]:
        mandatory_r2_bytes += sum(r["bytes"] for r in shard["raw_files"])
        mandatory_r2_bytes += sum(shard[k]["bytes"] for k in cache_keys)
    ledger_file = local(root, audit+"/budget_ledger.json")
    ledger = read_json(artifact(root, descriptor(root, ledger_file), closure))
    spent = ledger.get("elapsed_seconds")
    charged, history = ledger.get("charged"), ledger.get("history")
    if (ledger.get("schema_version") != "paper3-source-extension-budget-ledger-v1"
            or ledger.get("plan_sha256") != plan_sha or "wall_intent" in ledger or "active_transfer" in ledger
            or type(spent) not in (int, float) or not math.isfinite(spent)
            or not 0 <= spent <= plan["budgets"]["wall_seconds"]
            or not isinstance(charged, dict) or set(charged) != {"provider_body_bytes", "r2_get_body_bytes"}
            or any(type(v) is not int or not 0 <= v <= plan["budgets"][k] for k,v in charged.items())
            or not isinstance(history, list)):
        raise ExtensionGateError("closed_in_budget_source_coordinator_ledger_required")
    counted = {k: 0 for k in charged}
    for row in history:
        if (not isinstance(row, dict) or row.get("kind") not in charged
                or type(row.get("reserved_bytes")) is not int or row["reserved_bytes"] < 0):
            raise ExtensionGateError("source_coordinator_transfer_history_invalid")
        if row.get("status") == "SUCCESS_FULL_BODY_VERIFIED":
            amount = row.get("verified_bytes")
            if type(amount) is not int or not 0 <= amount <= row["reserved_bytes"]:
                raise ExtensionGateError("source_coordinator_verified_transfer_amount_invalid")
        elif row.get("status") in {"FAILURE_FULL_RESERVATION_CONSUMED", "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}:
            amount = row["reserved_bytes"]
        else:
            raise ExtensionGateError("source_coordinator_transfer_not_closed")
        counted[row["kind"]] += amount
    if counted != charged:
        raise ExtensionGateError("source_coordinator_consumed_bytes_do_not_match_history")
    if charged["r2_get_body_bytes"] < mandatory_r2_bytes:
        raise ExtensionGateError("source_coordinator_R2_ledger_below_required_full_GET_bytes")


def _v5_wrapper(root, closure):
    """Load only the frozen V5 read-only verifier; never its live scheduler."""
    import importlib.util
    references = ((PARALLEL_WRAPPER, PARALLEL_WRAPPER_SHA),
        (PARALLEL_PARENT_WRAPPER, PARALLEL_PARENT_WRAPPER_SHA),
        (PARALLEL_DREYER_PLAN, PARALLEL_DREYER_PLAN_FILE_SHA),
        (DREYER_NATIVE, DREYER_NATIVE_SHA),
        (DREYER_QUALIFICATION, DREYER_QUALIFICATION_SHA),
        (DREYER_ADMISSION, DREYER_ADMISSION_SHA))
    for name, sha in references:
        artifact(root, {"path": name, "sha256": sha}, closure)
    path = local(root, PARALLEL_WRAPPER)
    spec = importlib.util.spec_from_file_location("_p3_extension_V5_offline_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {"DREYER_PLAN_SHA": PARALLEL_DREYER_PLAN_SHA,
        "NATIVE_ADAPTER_SHA": DREYER_NATIVE_SHA,
        "QUALIFICATION_COMPLETION_SHA": DREYER_QUALIFICATION_SHA,
        "ADMISSION_REVIEW_SHA": DREYER_ADMISSION_SHA,
        "YANG_PLAN_SHA": PARALLEL_YANG_PLAN_SHA,
        "ENTRY": PARALLEL_WRAPPER, "PLAN_PATH": PARALLEL_DREYER_PLAN,
        "INTERSECT_QUALIFICATION_STAGE_CAP": False}
    if any(digest(getattr(module, key, None)) != digest(value) for key, value in expected.items()):
        raise ExtensionGateError("exact_frozen_Dreyer_V5_readonly_verifier_required")
    return module


def _yang_member_catalog(root, completed, closure):
    """Authorize archive identities from the frozen plan and real closed proof.

    ZIP member names describe provider bytes, not files to restore beside this
    program. The original offline verifier still proves every raw/cache GET,
    all five final GETs, and the eight issued calls before this classification.
    """
    import importlib

    plan_file = artifact(root, {"path": YANG_MEMBER_PLAN,
        "sha256": YANG_MEMBER_PLAN_FILE_SHA}, closure)
    plan = read_json(plan_file)
    if (plan.get("plan_sha256") != PARALLEL_YANG_PLAN_SHA
            or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != PARALLEL_YANG_PLAN_SHA
            or plan.get("dataset") != "Yang2025" or plan.get("role") != "source_train"
            or plan.get("source_only") is not True or plan.get("task_subset") != "2C"
            or plan.get("record_count") != 153 or len(plan.get("records", [])) != 153):
        raise ExtensionGateError("exact_frozen_Yang153_member_plan_required")
    wrapper_file = artifact(root, {"path": YANG_MEMBER_TRANSPORT,
        "sha256": YANG_MEMBER_TRANSPORT_SHA}, closure)
    transport = importlib.import_module("tools.resume_yang_source_transport_v2")
    if Path(transport.__file__).read_bytes() != wrapper_file.read_bytes():
        raise ExtensionGateError("frozen_Yang_member_transport_verifier_required")
    actual = transport.verify_complete_transport_history(root, plan)
    proof = completed.get("transport_completion_proof") if isinstance(completed, dict) else None
    history = actual.get("transport_history") if isinstance(actual, dict) else None
    if (not isinstance(actual, dict) or actual.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE"
            or actual.get("already_complete_offline_verified") is not True
            or type(actual.get("verified_records")) is not int or actual["verified_records"] != 153
            or not isinstance(history, dict) or type(history.get("calls_used")) is not int
            or history["calls_used"] != 8 or digest(proof) != digest(actual)
            or completed.get("mode") != "already_complete"
            or completed.get("plan_sha256") != PARALLEL_YANG_PLAN_SHA
            or digest(completed.get("transport_history")) != digest(history)
            or completed.get("manifest") != actual.get("manifest")
            or completed.get("backup_receipt") != actual.get("backup_receipt")):
        raise ExtensionGateError("actual_Yang153_five_GET_eight_closed_calls_required_for_members")
    if len(_receipt_entries(root, actual["backup_receipt"], closure)) != 5:
        raise ExtensionGateError("actual_Yang_five_final_full_GET_objects_required_for_members")
    manifest = read_json(artifact(root, actual["manifest"], closure))
    if (actual["manifest"]["path"] != f"cache/source_extensions_20261019/Yang2025/{PARALLEL_YANG_PLAN_SHA}/source_cache.json"
            or manifest.get("dataset") != "Yang2025" or manifest.get("role") != "source_train"
            or manifest.get("source_only") is not True
            or not isinstance(manifest.get("shards"), list) or len(manifest["shards"]) != 153):
        raise ExtensionGateError("canonical_completed_Yang_member_manifest_required")
    shards = {s.get("record_id"):s for s in manifest["shards"]}
    records, members = {}, set()
    for record in plan["records"]:
        rid, subject, session = (record.get(k) for k in ("record_id", "subject_id", "session_id"))
        if (not isinstance(rid, str) or re.fullmatch(r"sub-\d{3}_ses-\d{2}", rid) is None
                or rid != f"{subject}_{session}" or rid not in shards
                or not isinstance(record.get("raw_files"), list) or len(record["raw_files"]) != 2):
            raise ExtensionGateError("exact_Yang_member_record_roster_required")
        shard = shards[rid]
        native_path = f"audit/source_extensions_20261019/Yang2025/{PARALLEL_YANG_PLAN_SHA}/{rid}/native_metadata.json"
        if (native_path in records or shard.get("subject_id") != subject or shard.get("session_id") != session
                or shard.get("raw_files") != record["raw_files"]
                or not isinstance(shard.get("native_metadata"), dict)
                or shard["native_metadata"].get("path") != native_path):
            raise ExtensionGateError("canonical_Yang_member_record_metadata_required")
        for raw, filename in zip(record["raw_files"], ("data.bdf", "evt.bdf")):
            member = raw.get("download", {}).get("member")
            expected = f"WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset/{subject}/{session}/eeg/{filename}"
            if (member != expected or raw.get("download", {}).get("kind") != "zip_range"
                    or type(raw.get("bytes")) is not int or raw["bytes"] <= 0
                    or re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is None):
                raise ExtensionGateError("exact_official_Yang_member_identity_required")
            identity = (member, raw["sha256"], raw["bytes"])
            if identity in members:
                raise ExtensionGateError("unique_306_Yang_archive_members_required")
            members.add(identity)
        records[native_path] = (record, shard)
    if len(records) != 153 or len(shards) != 153 or len(members) != 306:
        raise ExtensionGateError("unique_153_records_and_306_Yang_members_required")
    historical = {"path": YANG_MEMBER_HISTORY, "sha256": YANG_MEMBER_HISTORY_SHA, "bytes": 8929}
    if plan.get("historical_get_evidence") != [historical]:
        raise ExtensionGateError("exact_Yang_historical_download_receipt_required")
    return {"records": records, "historical": historical}


def _yang_raw_member_get(root, record, shard, closure):
    rows = _receipt_entries(root, shard["raw_backup_receipt"], closure)
    fields = ("path", "sha256", "bytes", "role", "dataset")
    expected = [{"path":r["path"], "sha256":r["sha256"], "bytes":r["bytes"],
        "role":"source_development_raw", "dataset":"Yang2025"} for r in record["raw_files"]]
    if (len(rows) != 2 or {tuple(r.get(k) for k in fields) for r in rows}
            != {tuple(r[k] for k in fields) for r in expected}):
        raise ExtensionGateError("both_Yang_native_members_require_exact_raw_full_GET_proofs")


def _yang_native_members(root, item, metadata, catalog, closure):
    """Classify only the two bound raw fields in canonical native metadata."""
    pair = catalog["records"].get(item["path"])
    if pair is None:
        raise ExtensionGateError("canonical_Yang_native_member_metadata_required")
    record, shard = pair
    if (any(item.get(k) != shard["native_metadata"].get(k) for k in ("path", "sha256", "bytes"))
            or not isinstance(metadata, dict)
            or metadata.get("schema_version") != "paper3-yang-native-inventory-record-v1"
            or metadata.get("dataset_id") != "Yang2025_2C"
            or type(metadata.get("dataset_version")) is not int or metadata["dataset_version"] != 5
            or type(metadata.get("subject")) is not int
            or metadata["subject"] != int(record["subject_id"].split("-")[1])
            or type(metadata.get("session")) is not int
            or metadata["session"] != int(record["session_id"].split("-")[1])
            or metadata.get("participant_id") != "Yang2025_2C:"+record["subject_id"]
            or ("record_id" in metadata and metadata["record_id"] != record["record_id"])
            or metadata.get("source_record_id") != record["raw_files"][0]["download"]["member"]
            or not isinstance(metadata.get("raw_files"), dict)
            or set(metadata["raw_files"]) != {"signal", "event"}):
        raise ExtensionGateError("exact_Yang_native_schema_record_and_member_pair_required")
    references = []
    for name, raw in zip(("signal", "event"), record["raw_files"]):
        value = metadata["raw_files"][name]
        if (not isinstance(value, dict) or value.get("path") != raw["download"]["member"]
                or value.get("sha256") != raw["sha256"] or type(value.get("bytes")) is not int
                or value["bytes"] != raw["bytes"]):
            raise ExtensionGateError("exact_Yang_native_member_SHA_size_and_record_required")
        references.append(value)
    _yang_raw_member_get(root, record, shard, closure)
    return {id(value) for value in references}


def _yang_historical_members(root, item, receipt, catalog, closure):
    """Classify the pinned producer's two historical absolute raw identities.

    The recorded producer root is immutable historical context. It is compared
    literally, never resolved, rewritten to a restore root, or opened.
    """
    if (any(item.get(k) != catalog["historical"].get(k) for k in ("path", "sha256", "bytes"))
            or not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1 or receipt.get("status") != "ONE_SESSION_NATIVE_VERIFIED"
            or type(receipt.get("subject")) is not int or receipt["subject"] != 1
            or type(receipt.get("session")) is not int or receipt["session"] != 1
            or not isinstance(receipt.get("objects"), list) or len(receipt["objects"]) != 2):
        raise ExtensionGateError("exact_pinned_Yang_historical_member_receipt_required")
    native_path = f"audit/source_extensions_20261019/Yang2025/{PARALLEL_YANG_PLAN_SHA}/sub-001_ses-01/native_metadata.json"
    record, shard = catalog["records"][native_path]
    download = record["raw_files"][0]["download"]
    target = f"{YANG_MEMBER_ORIGINAL_ROOT}/data/source_extensions_20261019/Yang2025/sub-001/ses-01"
    if (receipt.get("source_url") != download["url"] or receipt.get("archive_bytes") != download["archive_bytes"]
            or receipt.get("raw_target_directory") != target):
        raise ExtensionGateError("exact_original_Yang_historical_producer_required")
    references = []
    for value, raw in zip(receipt["objects"], record["raw_files"]):
        if (not isinstance(value, dict) or value.get("path") != f"{YANG_MEMBER_ORIGINAL_ROOT}/{raw['path']}"
                or value.get("archive_member") != raw["download"]["member"]
                or value.get("sha256") != raw["sha256"] or type(value.get("bytes")) is not int
                or value["bytes"] != raw["bytes"] or value.get("crc32") != raw["crc32"]
                or value.get("compressed_bytes") != raw["download"]["compressed_bytes"]
                or value.get("status") != "FRESH_NATIVE_FULL_SHA_CRC_VERIFIED"):
            raise ExtensionGateError("exact_Yang_historical_member_path_SHA_size_and_record_required")
        references.append(value)
    _yang_raw_member_get(root, record, shard, closure)
    return {id(value) for value in references}


def _provenance_closure(root, values, closure, *, aliases=None, yang_completion=None):
    """Track scientific proof dependencies, preserving historical code aliases.

    Provider EDF payloads need full-GET receipts, not another local raw copy.
    Historical plan code is covered by the explicit qualification code aliases;
    following its original mutable canonical paths would confuse V4 with V5.
    A recorded Yang process is audit context, never a live-ledger dependency.
    """
    aliases = aliases or {}
    seen = set()
    member_catalog = None

    def visit(value, field=None, provider_references=frozenset()):
        nonlocal member_catalog
        if isinstance(value, dict):
            if id(value) in provider_references:
                # Preserve every other dependency nested in this descriptor.
                for name, item in value.items():
                    if name not in {"path", "sha256", "bytes"}:
                        visit(item, name, provider_references)
                return
            if {"path", "sha256", "bytes"} <= set(value):
                relative = value["path"]
                if not isinstance(relative, str):
                    raise ExtensionGateError("source_extension_artifact_descriptor_required")
                if relative.startswith("data/"):
                    return
                candidate = value
                alias = aliases.get(relative)
                if alias is not None and all(alias.get(k) == value.get(k) for k in ("sha256", "bytes")):
                    candidate = alias
                key = (candidate["path"], candidate["sha256"])
                if key in seen:
                    return
                path = artifact(root, candidate, closure)
                seen.add(key)
                if path.suffix == ".json":
                    body = read_json(path)
                    references = frozenset()
                    if (yang_completion is not None and re.fullmatch(
                            rf"audit/source_extensions_20261019/Yang2025/{PARALLEL_YANG_PLAN_SHA}/sub-\d{{3}}_ses-\d{{2}}/native_metadata\.json",
                            candidate["path"])):
                        if member_catalog is None:
                            member_catalog = _yang_member_catalog(root, yang_completion, closure)
                        references = _yang_native_members(root, candidate, body, member_catalog, closure)
                    elif yang_completion is not None and candidate["path"] == YANG_MEMBER_HISTORY:
                        if member_catalog is None:
                            member_catalog = _yang_member_catalog(root, yang_completion, closure)
                        references = _yang_historical_members(root, candidate, body, member_catalog, closure)
                    visit(body, provider_references=references)
                return
            for name, item in value.items():
                if name not in {"code", "initial_yang_process"}:
                    visit(item, name, provider_references)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, field, provider_references)

    visit(values)


def _parallel_prior_budget(root, value, current_plan, closure, *, wrapper=None, aliases=None):
    """All six original closed journals, verified by the immutable V5 checker."""
    wrapper = wrapper or _v5_wrapper(root, closure)
    try:
        expected = wrapper.prior_budget_evidence(root, current_plan)
    except wrapper.WrapperError as exc:
        raise ExtensionGateError("Dreyer_six_actual_conserved_closed_histories_required") from exc
    if (not isinstance(value, dict) or digest(value) != digest(expected)
            or value.get("accounting_schema") != "paper3-dreyer-six-ledger-budget-v1"
            or [r.get("source") for r in value.get("closed_history_sources", [])]
                != ["V1", "V2", "auxiliary_schema_review", "V3", "V4", "native_qualification"]):
        raise ExtensionGateError("Dreyer_six_actual_conserved_closed_histories_required")
    _provenance_closure(root, expected, closure, aliases=aliases)


def _peer_coordination_closure(root, declaration, coordination, closure):
    """Bind completed Yang proofs only after the V5 verifier proves takeover.

    A live fingerprint is an observation. Its nested objects never authorize
    reading a live Yang ledger, acquiring its lock, or opening a payload.
    """
    keys = {"schema_version", "initial_yang_process", "last_verified_live_peer",
            "global_lock_takeover", "completed_yang_proof"}
    initial = declaration.get("initial_yang_process")
    if (not isinstance(coordination, dict) or set(coordination) != keys
            or coordination.get("schema_version") != "paper3-dreyer-yang-peer-coordination-v1"
            or type(coordination.get("global_lock_takeover")) is not bool
            or not isinstance(initial, dict)
            or digest(coordination.get("initial_yang_process")) != digest(initial)):
        raise ExtensionGateError("Dreyer_V5_recorded_Yang_peer_coordination_required")
    mode = initial.get("mode")
    last_live = initial if mode == "concurrent_no_cache_wrapper_v2" else None
    if (mode not in {"concurrent_no_cache_wrapper_v2", "already_complete"}
            or digest(coordination["last_verified_live_peer"]) != digest(last_live)):
        raise ExtensionGateError("Dreyer_V5_last_verified_Yang_fingerprint_changed")
    completed = coordination["completed_yang_proof"]
    if not coordination["global_lock_takeover"]:
        if mode != "concurrent_no_cache_wrapper_v2" or completed is not None:
            raise ExtensionGateError("Dreyer_V5_live_Yang_must_not_claim_closed_completion")
        return
    if (not isinstance(completed, dict) or completed.get("mode") != "already_complete"
            or completed.get("plan_sha256") != PARALLEL_YANG_PLAN_SHA
            or mode == "already_complete" and digest(completed) != digest(initial)):
        raise ExtensionGateError("Dreyer_V5_takeover_requires_complete_Yang_proof")
    history = completed.get("transport_history")
    proof = completed.get("transport_completion_proof")
    if (not isinstance(history, dict) or not isinstance(proof, dict)
            or proof.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE"
            or proof.get("already_complete_offline_verified") is not True
            or type(proof.get("verified_records")) is not int or proof["verified_records"] != 153):
        raise ExtensionGateError("Dreyer_V5_full_Yang153_transport_proof_required")
    for value in (history.get("provenance_closure"), proof.get("provenance_closure")):
        if not isinstance(value, dict) or not value:
            raise ExtensionGateError("Dreyer_V5_completed_Yang_provenance_closure_required")
        for relative, sha in value.items():
            artifact(root, {"path": relative, "sha256": sha}, closure)
    _provenance_closure(root, completed, closure, yang_completion=completed)


def _parallel_evidence(root, manifest, status, audit, closure):
    """Require real V5 numeric recomputation and seven final complete GETs.

    Only fixed, offline wrapper methods run here. In particular fixed_plans,
    current_yang, build_declaration, PeerGuard and execute are never called.
    """
    item = manifest.get("operational_evidence")
    if item is None:
        if status.get("operational_evidence") is not None:
            raise ExtensionGateError("source_parallel_evidence_manifest_status_mismatch")
        if manifest.get("dataset") == "Dreyer2023":
            raise ExtensionGateError("Dreyer_v5_operational_proof_and_seven_GET_objects_required")
        return []
    if (not isinstance(item, dict) or manifest.get("dataset") != "Dreyer2023"
            or manifest.get("plan_sha256") != PARALLEL_DREYER_PLAN_SHA
            or status.get("operational_evidence") != item
            or item.get("path") != audit+"/parallel_execution_receipt.json"):
        raise ExtensionGateError("exact_declared_Dreyer_V5_CPU_schedule_required")
    wrapper = _v5_wrapper(root, closure)
    evidence = read_json(artifact(root, item, closure))
    implementation = {"path": PARALLEL_WRAPPER, "sha256": PARALLEL_WRAPPER_SHA}
    if (not isinstance(evidence.get("wrapper_implementation"), dict)
            or any(evidence["wrapper_implementation"].get(k) != v for k,v in implementation.items())):
        raise ExtensionGateError("declared_Dreyer_V5_CPU_wrapper_required")
    declaration = evidence.get("root_declaration")
    if not isinstance(declaration, dict) or declaration.get("path") != audit+"/parallel_root_declaration.json":
        raise ExtensionGateError("Dreyer_V5_parallel_root_declaration_required")
    root_value = read_json(artifact(root, declaration, closure))
    plan_file = artifact(root, {"path": PARALLEL_DREYER_PLAN,
        "sha256": PARALLEL_DREYER_PLAN_FILE_SHA}, closure)
    plan = read_json(plan_file)
    frozen = read_json(artifact(root, descriptor(root, local(root, audit+"/frozen_plan.json")), closure))
    if (frozen != plan or plan.get("plan_sha256") != PARALLEL_DREYER_PLAN_SHA
            or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != PARALLEL_DREYER_PLAN_SHA):
        raise ExtensionGateError("Dreyer_exact_current_V5_frozen_plan_required")
    # Bind the direct-read state/report files before the expensive verifier.
    # A concurrent edit cannot become a newly sealed, unverified dependency.
    states, reports = [], []
    for record in plan["records"]:
        state_file = local(root, audit+"/"+record["record_id"]+"/state.json")
        state_ref = descriptor(root, state_file)
        artifact(root, state_ref, closure)
        states.append(state_ref)
        if record["record_id"] in root_value["qualification_evidence"]["reference_states"]:
            regression_file = local(root, audit+"/"+record["record_id"]+"/v4_numeric_regression.json")
            report_ref = descriptor(root, regression_file)
            artifact(root, report_ref, closure)
            reports.append(report_ref)
    try:
        verified_root, verified_ref = wrapper.verify_declaration(root, declaration["path"], plan)
        complete = wrapper.verified_completion(root, plan, declaration)
    except wrapper.WrapperError as exc:
        raise ExtensionGateError("Dreyer_V5_actual_qualification_numeric_and_complete_GET_proofs_required") from exc
    if verified_root != root_value or verified_ref != declaration or complete is None:
        raise ExtensionGateError("Dreyer_V5_actual_qualification_numeric_and_complete_GET_proofs_required")
    if any(digest(complete.get(k)) != digest(value) for k,value in status.items()):
        raise ExtensionGateError("Dreyer_V5_verified_completion_must_match_exact_global_status")
    expected = {"schema_version": "paper3-dreyer-parallel-cpu-execution-v5",
        "status": "RECORDS_VERIFIED_FINAL_BACKUP_PENDING", "dataset": "Dreyer2023",
        "plan_sha256": PARALLEL_DREYER_PLAN_SHA,
        "concurrent_yang_plan_sha256": PARALLEL_YANG_PLAN_SHA, "record_count": 520,
        "record_ids_sha256": digest([r["record_id"] for r in manifest["shards"]]),
        "wrapper_implementation": descriptor(root, local(root, PARALLEL_WRAPPER)),
        "root_declaration": declaration, "scientific_kernel_unchanged": True,
        "qualification_evidence": root_value.get("qualification_evidence"),
        "old128_numerical_regression": evidence.get("old128_numerical_regression"),
        "yang_peer_coordination": evidence.get("yang_peer_coordination"),
        "GPU_actions": 0, "actual_fits": 0, "external_payload_accessed": False,
        "carried_over_verified_R2_body_bytes": 0, "server_actions": False, "limits": wrapper.LIMITS}
    if digest(evidence) != digest(expected):
        raise ExtensionGateError("Dreyer_V5_parallel_CPU_evidence_scope_or_budget_changed")
    _peer_coordination_closure(root, root_value, evidence.get("yang_peer_coordination"), closure)
    regressions = evidence.get("old128_numerical_regression")
    if (not isinstance(regressions, dict)
            or type(regressions.get("reference_records")) is not int or regressions["reference_records"] != 128
            or type(regressions.get("verified_records")) is not int or regressions["verified_records"] != 128
            or not isinstance(regressions.get("comparisons"), list) or len(regressions["comparisons"]) != 128
            or regressions.get("embedded_in_final_GET_backed_operational_evidence") is not True):
        raise ExtensionGateError("Dreyer_V5_all128_actual_numeric_regressions_required")
    # The frozen declaration verifier above authenticated these snapshots.
    aliases = root_value["qualification_evidence"]["historical_code_aliases"]
    _parallel_prior_budget(root, root_value.get("prior_budget_evidence"), plan, closure,
        wrapper=wrapper, aliases=aliases)
    _provenance_closure(root, [root_value["qualification_evidence"], regressions], closure, aliases=aliases)
    # Recheck snapshots after numeric verification, then bind each state's
    # artifacts and the full actual old/new numeric reports to freshness.
    for report_ref in reports:
        artifact(root, report_ref, closure)
    _provenance_closure(root, states, closure, aliases=aliases)
    return [(item, "audit"), (declaration, "audit")]


def _native_window(dataset, event, original):
    """Do not conflate Yang's source analysis bound with physical MI onset/end."""
    context = original["original_context_samples"]
    sfreq = original.get("native_sfreq_hz", original.get("native_sfreq"))
    if (type(sfreq) not in (int, float) or not np.isfinite(sfreq) or sfreq <= 60
            or not isinstance(context, list) or len(context) != 2
            or any(type(v) is not int for v in context)
            or context[0] != original["analysis_anchor_sample"] or context[0] < 0
            or context[1] - context[0] != round(1.5*sfreq)):
        raise ExtensionGateError("exact_original_1p5_second_native_context_required")
    if dataset == "Yang2025":
        marker = original.get("original_marker_sample")
        if (sfreq != 1000 or type(marker) is not int
                or context != [marker+2250, marker+3750]
                or original.get("crop_start_sample") != marker+2500
                or original.get("crop_stop_sample_exclusive") != marker+3500
                or original.get("label_code") not in {"1", "2"}
                or original["label"] != {"1": 0, "2": 1}[original["label_code"]]
                or event.get("raw_event_sha256") != original.get("raw_event_sha256")
                or event.get("MI_onset_verified") is not False
                or event.get("physical_MI_end_verified") is not False):
            raise ExtensionGateError("Yang_source_analysis_intersection_not_physical_MI_required")
        return
    verified_end = original.get("task_end_verified") is True
    if not verified_end:
        try:
            cue = Fraction(original["native_cue_onset_token"]) * Fraction(str(sfreq))
            end = Fraction(original["native_end_onset_token"]) * Fraction(str(sfreq))
            verified_end = (original.get("mi_onset_verified") is True
                and original["analysis_anchor_sample"] == -(-cue.numerator // cue.denominator)
                and original["task_end_sample"] == end.numerator // end.denominator)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            verified_end = False
    if (not verified_end or original.get("acquisition_order_verified") is not True
            or context[1] > original["task_end_sample"]):
        raise ExtensionGateError("Dreyer_real_cue_paired_end_and_order_required")


def _frozen_roster(root, manifest, closure):
    """Exact provider records; a one-session pilot cannot become the full arm."""
    from . import source_extension_coordinator as coordinator
    path = artifact(root, manifest.get("official_inventory"), closure)
    relative = "p3work/source_extension_coordinator.py"
    if Path(coordinator.__file__).read_bytes() != local(root, relative).read_bytes():
        raise ExtensionGateError("source_roster_verifier_code_differs_from_root")
    closure[relative] = file_digest(local(root, relative))
    records = coordinator.normalize_inventory(manifest["dataset"], read_json(path))
    expected = {r["record_id"]: r for r in records}
    shards = manifest["shards"]
    if (manifest.get("record_count") != len(records) or len(shards) != len(records)
            or {s.get("record_id") for s in shards} != set(expected)):
        raise ExtensionGateError("full_official_extension_record_roster_required")
    for shard in shards:
        record = expected[shard["record_id"]]
        if (shard.get("subject_id") != record["subject_id"] or shard.get("session_id") != record["session_id"]
                or shard.get("raw_files") != record["raw_files"]
                or shard.get("raw_sha256") != record["raw_files"][0]["sha256"]):
            raise ExtensionGateError("extension_shard_provider_record_identity_mismatch")


@dataclass(frozen=True)
class ExtensionCache:
    """New composite source-only capability, never a primary VerifiedCache."""
    root: Path
    arm: str
    n_times: int
    input_contract_sha256: str
    manifest_path: str
    manifest_sha256: str
    arrays: tuple
    labels: np.ndarray
    rows: tuple
    locations: tuple
    artifact_sha256: dict
    upstream_acceptance_sha256: dict
    manifest_payload: dict
    baseline: object
    extension_manifest: dict
    _issuer: object

    def batch(self, indices):
        if self._issuer is not _ISSUER:
            raise ExtensionGateError("unissued_extension_cache")
        values = np.asarray(indices)
        if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= len(self.rows)):
            raise ExtensionGateError("invalid_extension_batch_indices")
        return np.stack([self.arrays[self.locations[int(i)][0]][self.locations[int(i)][1]] for i in values])

    def assert_fresh(self, *, composite_required=False):
        if self._issuer is not _ISSUER:
            raise ExtensionGateError("unissued_extension_cache")
        for relative, sha in self.artifact_sha256.items():
            artifact(self.root, {"path": relative, "sha256": sha})
        if composite_required:
            artifact(self.root, {"path": self.manifest_path, "sha256": self.manifest_sha256})


def load_extension_cache(root, *, arm, manifest_path, manifest_sha256, output):
    """Issue a bounded-memory, SHA-bound combination of old four + ONE new source."""
    from tools.run_controlled_pilot import verify_inputs
    root = Path(root).resolve(strict=True)
    if arm not in ARMS:
        raise ExtensionGateError("registered_source_extension_arm_required")
    baseline = verify_inputs(root)
    if baseline.manifest_sha256 != BASE_SHA or baseline.input_contract_sha256 != INPUT_SHA:
        raise ExtensionGateError("exact_completed_baseline_four_cache_required")
    closure = {k: v for k, v in baseline.artifact_sha256.items() if k != "status/tasks.json"}
    path = artifact(root, {"path": manifest_path, "sha256": manifest_sha256}, closure)
    manifest = read_json(path)
    dataset = ARMS[arm]
    expected = {"schema_version": "paper3-source-extension-cache-v1", "status": "SOURCE_EXTENSION_CACHE_COMPLETE",
        "dataset": dataset, "role": "source_train", "source_only": True, "unique_role_within_branch": True,
        "channels": list(CHANNELS), "stage": STAGE, "input_contract_sha256": INPUT_SHA,
        "synthetic": False, "old_npz_reused": False, "normalization_applied": False}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ExtensionGateError("complete_source_only_extension_manifest_required")
    if dataset == "Yang2025" and manifest.get("task_subset") != "2C":
        raise ExtensionGateError("Yang_explicit_2C_subset_required")
    assignment = read_json(artifact(root, manifest.get("role_assignment"), closure))
    if (not isinstance(assignment, dict) or assignment.get("dataset") != dataset
            or assignment.get("role") != "source_train" or assignment.get("source_only") is not True
            or assignment.get("unique_role_within_branch") is not True):
        raise ExtensionGateError("frozen_unique_source_role_assignment_required")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ExtensionGateError("extension_source_shards_required")
    _frozen_roster(root, manifest, closure)
    _completed_preparation(root, manifest, path, closure)
    arrays = list(baseline.arrays)
    labels = list(np.asarray(baseline.labels).tolist())
    rows, locations = list(baseline.rows), list(baseline.locations)
    seen = {r["event_uid"] for r in rows}
    originals, shard_ids = set(), set()
    for shard in shards:
        person, session = str(shard.get("subject_id", "")), str(shard.get("session_id", ""))
        if not person or not session or (person, session, shard.get("raw_sha256")) in shard_ids:
            raise ExtensionGateError("unique_subject_session_raw_shard_required")
        shard_ids.add((person, session, shard.get("raw_sha256")))
        paths = {key: artifact(root, shard.get(key), closure) for key in ("X", "y", "events", "exclusions")}
        artifact(root, shard.get("adapter_code"), closure)
        artifact(root, shard.get("metadata"), closure)
        _cache_backup(root, shard, dataset, closure)
        native = _native_rows(root, shard, dataset, closure)
        x = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
        y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
        all_events, exclusions = read_json(paths["events"]), read_json(paths["exclusions"])
        if (not isinstance(all_events, list) or any(not isinstance(row, dict) for row in all_events)
                or any("valid" in row and type(row["valid"]) is not bool for row in all_events)):
            raise ExtensionGateError("extension_event_objects_and_boolean_validity_required")
        events = [row for row in all_events if row.get("valid", True)]
        if (x.dtype != np.dtype("float32") or x.ndim != 3 or x.shape[1:] != (11, 128)
                or y.dtype != np.dtype("int64") or y.shape != (len(x),) or not np.isin(y, [0, 1]).all()
                or not isinstance(events, list) or len(events) != len(x) or not isinstance(exclusions, list)):
            raise ExtensionGateError("extension_X_y_event_schema_mismatch")
        if shard.get("valid_trials") != len(x) or shard.get("original_events") != len(all_events):
            raise ExtensionGateError("source_shard_original_and_valid_counts_must_match_actual_events")
        accounted = set()
        for index, event in enumerate(events):
            uid = event.get("event_uid") if isinstance(event, dict) else None
            if (uid not in native or uid in seen or uid in accounted or event.get("role") != "source_train"
                    or event.get("dataset") != dataset or str(event.get("subject_id")) != person
                    or str(event.get("session_id")) != session or event.get("input_contract_sha256") != INPUT_SHA
                    or type(event.get("label")) is not int or event["label"] != int(y[index])
                    or event.get("unit_evidence") != shard["unit_evidence"]
                    or event.get("raw_sha256") != shard["raw_sha256"]):
                raise ExtensionGateError("extension_event_role_UID_label_identity_mismatch")
            original = native[uid]
            compare = ("label", "original_trial_index", "analysis_anchor_sample", "original_context_samples")
            if dataset == "Dreyer2023":
                compare += ("task_end_sample",)
            for key in compare:
                if key not in original or event.get(key) != original[key]:
                    raise ExtensionGateError("extension_cache_differs_from_original_native_event")
            if original.get("canonical_label") != {0: "left_hand", 1: "right_hand"}[int(y[index])]:
                raise ExtensionGateError("verified_native_LR_anchor_and_order_required")
            _native_window(dataset, event, original)
            if event.get("cache_index", index) != index:
                raise ExtensionGateError("original_native_order_must_match_cache_rows")
            signal = np.asarray(x[index], dtype=np.float64)
            scale = float(np.max(np.abs(signal)))
            if (not np.isfinite(signal).all() or scale <= 0
                    or np.max(np.abs(signal.mean(0))) > scale*2e-6
                    or np.max(np.abs(signal.mean(1))) > scale*2e-6
                    or hashlib.sha256(np.asarray(x[index], dtype="<f4").tobytes(order="C")).hexdigest()
                        != event.get("processed_signal_sha256")):
                raise ExtensionGateError("finite_nonzero_CAR_temporal_mean_SHA_QC_required")
            _unit_qc(root, event, closure, native_metadata=shard["native_metadata"],
                     native_profile=shard.get("native_profile"))
            original_key = (dataset, event["raw_sha256"], original["original_trial_index"])
            if original_key in originals:
                raise ExtensionGateError("duplicated_original_extension_trial")
            originals.add(original_key)
            seen.add(uid); accounted.add(uid)
            rows.append(MappingProxyType(event)); labels.append(int(y[index]))
            locations.append((len(arrays), index))
        for event in exclusions:
            uid = event.get("event_uid") if isinstance(event, dict) else None
            if uid not in native or uid in accounted or not event.get("reason", event.get("exclusion_reason")):
                raise ExtensionGateError("explicit_native_exclusion_required")
            accounted.add(uid)
        if accounted != set(native):
            raise ExtensionGateError("every_native_trial_must_be_valid_or_excluded")
        arrays.append(x)
    if manifest.get("valid_trials") != len(rows) - len(baseline.rows):
        raise ExtensionGateError("completed_extension_valid_count_must_match_actual_arrays")
    if {r["dataset"] for r in rows} != set(SOURCE_IDS) | {dataset}:
        raise ExtensionGateError("exact_five_source_extension_required")
    # Exact cross-source duplicate signals are not independent LODO/sample
    # support; refuse rather than silently count/reassign the duplicate.
    signal_sources = {}
    for row in rows:
        signal_sources.setdefault(row["processed_signal_sha256"], set()).add(row["dataset"])
    if any(len(value) > 1 for value in signal_sources.values()):
        raise ExtensionGateError("exact_signals_cross_source_dataset_boundary")
    values = np.asarray(labels, dtype=np.int64); values.flags.writeable = False
    payload = {"schema_version": "paper3-combined-source-extension-input-v1", "arm": arm,
        "role": "source_train", "source_only": True, "primary_H1_changed": False,
        "baseline_manifest": descriptor(root, local(root, BASE_MANIFEST)),
        "extension_manifest": descriptor(root, path), "input_contract_sha256": INPUT_SHA,
        "training_dataset_ids": sorted(set(SOURCE_IDS) | {dataset}), "original_trials": len(rows),
        "event_uids_sha256": digest([r["event_uid"] for r in rows]), "old_NPZ_reused": False}
    encoded = (json.dumps(payload, indent=2, allow_nan=False)+"\n").encode()
    combined_path = (Path(output) / "combined_source_manifest.json").as_posix()
    local(root, combined_path, exists=False)
    return ExtensionCache(root, arm, 128, INPUT_SHA, combined_path, hashlib.sha256(encoded).hexdigest(),
        tuple(arrays), values, tuple(rows), tuple(locations), closure,
        {**baseline.upstream_acceptance_sha256, "extension_manifest": manifest_sha256},
        payload, baseline, descriptor(root, path), _ISSUER)


def load_verified_extension_source(root, receipt_path):
    """Separate downstream interface; old primary loaders continue to refuse it."""
    from .source_extension_training import verify_completed_extension
    root = Path(root).resolve(strict=True)
    receipt = verify_completed_extension(root, receipt_path)
    checkpoint = core.verified_checkpoint(artifact(root, receipt["checkpoint"]))
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = CompactEEGNet(11, 128, core.SOURCE_CONFIG["dropout"])
        model.load_state_dict(checkpoint["model"], strict=True)
        encoder = torch.nn.Sequential(model.features, torch.nn.Flatten(start_dim=1))
        frozen = FrozenSourceModel(encoder, model.classifier, feature_dim=64, input_layout="N1CT")
    frozen.assert_unchanged()
    return frozen, receipt
