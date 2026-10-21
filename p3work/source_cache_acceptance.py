"""Read-only full-cohort source acceptance using retained metadata/R2 proofs.

No raw EEG is opened or restored. The 505 official source identities must have
native trial/audit metadata and completed raw+metadata full-GET receipts. Every
original LR trial must occur exactly once in valid cache events or exclusions.
This prevents partial/schema caches from becoming formal inputs by status alone;
it is not a cryptographic attestation against a party forging all local records.
Native-context SHA fields are checked as metadata bindings, not independently
recomputed from raw EEG. Stored float32 RMS is checked against pre-cast QC RMS.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .contracts import CHANNELS, SOURCE_IDS, digest, file_digest, validate_signal_contract
from .training_state import local_path
from tools.prepare_source_cache import (SourceCacheGateError, expected_source_objects,
    verify_native_reader_provenance)
from tools.r2_restore import load_verified_receipt
from tools.r2_restore import backup as _backup
from tools.source_preparation_batch import VERSIONS


class SourceCacheAcceptanceError(ValueError):
    pass


_UID_KEYS = ("dataset", "source_version", "provider_subject", "session_id", "source_file_sha256",
             "source_record_id", "physical_run_id", "original_trial_index", "raw_label_code")
_LABELS = {"BNCI2014_001": {1: "left_hand", 2: "right_hand"},
    "Cho2017": {"imagery_left": "left_hand", "imagery_right": "right_hand"},
    "Lee2019_MI": {1: "right_hand", 2: "left_hand"}, "PhysionetMI": {"T1": "left_hand", "T2": "right_hand"}}
_EXCLUSIONS = {"insufficient_original_task_context", "Nonfinite context", "Zero/nonfinite trial energy"}
_SI_FACTORS = {"V": 1.0, "uV": 1e-6, "µV": 1e-6, "mV": 1e-3}
_READER_FIELDS = ("native_reader_revision", "physionet_tal_reader_sha256", "source_dispatch_sha256")


def _validate_context_and_qc(row, original, contract, signal_sha, stored_signal):
    """Bind bounded-context/unit QC to metadata and the actual float32 cache.

    The writer passes an already sliced context to prepare_trial, with local
    task_sample=0. Its QC stop is therefore the context length; the event's
    original_context_samples retain the recording-global sample coordinates.
    """
    sfreq, anchor = original.get("native_sfreq"), original["analysis_anchor_sample"]
    bounds = original.get("real_trial_start_end")
    if (type(sfreq) not in (int, float) or not np.isfinite(sfreq) or sfreq <= 60
            or not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] <= anchor < bounds[1]):
        raise SourceCacheAcceptanceError("native_context_sampling_and_original_trial_bounds_required")
    stop = anchor + round(contract["context_s"][1] * sfreq)
    context = row.get("original_context_samples")
    if (not isinstance(context, list) or any(type(value) is not int for value in context)
            or context != [anchor, stop] or stop > original["task_end_sample"] or stop > bounds[1]
            or not isinstance(row.get("native_context_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", row["native_context_sha256"]) is None):
        raise SourceCacheAcceptanceError("native_context_SHA_binding_and_exact_bounded_samples_required")
    if original["dataset"] == "PhysionetMI":
        unit, evidence_kind = "V", "native_EDF_header_MNE_SI"
        header_unit = original.get("native_unit_string")
        evidence_valid = (isinstance(header_unit, str) and header_unit in _SI_FACTORS
            and original.get("unit_verified_from_native_header") is True
            and original.get("data_returned_by_mne_unit") == "V"
            and type(original.get("physical_scale_to_si_volts")) in (int, float)
            and original.get("physical_scale_to_si_volts") == _SI_FACTORS.get(header_unit))
    else:
        unit, evidence_kind = "uV", "fixed_adapter_numeric_unit_declaration_not_native_header"
        evidence_valid = (original.get("numeric_unit_declared_by_fixed_adapter") == unit
            and original.get("native_unit_string") is None
            and original.get("unit_verified_from_native_header") is False
            and original.get("absolute_amplitude_qc_authorized") is False)
    evidence = original.get("physical_scale_evidence")
    if (not evidence_valid or not isinstance(evidence, str) or not evidence
            or row.get("effective_numeric_unit") != unit or row.get("unit_evidence_kind") != evidence_kind):
        raise SourceCacheAcceptanceError("effective_numeric_unit_must_match_fixed_adapter_or_MNE_SI_evidence")
    qc = row.get("preprocessing_qc")
    if (not isinstance(qc, dict) or qc.get("signal_contract_sha256") != signal_sha
            or qc.get("reference") != "CAR_exact_clinic11" or qc.get("normalization_applied") is not False
            or qc.get("data_after_context_accessed") is not False
            or type(qc.get("filter_context_ends_at_native_sample")) is not int
            or qc["filter_context_ends_at_native_sample"] != stop - anchor
            or qc.get("native_unit") != unit or qc.get("stored_unit") != "V"
            or type(qc.get("si_conversion_factor")) not in (int, float)
            or qc["si_conversion_factor"] != _SI_FACTORS[unit]
            or type(qc.get("joint_rms")) not in (int, float)
            or not np.isfinite(qc["joint_rms"]) or qc["joint_rms"] <= 0):
        raise SourceCacheAcceptanceError("preprocessing_QC_must_bind_signal_context_CAR_unit_and_pre_RMS_state")
    actual_rms = float(np.sqrt(np.mean(np.square(np.asarray(stored_signal, dtype=np.float64)), dtype=np.float64)))
    # Float32 storage rounds the float64 pre-RMS crop. This numerical tolerance
    # bounds that rounding, including subnormals; it is not an amplitude gate.
    if (not np.isfinite(actual_rms) or actual_rms <= 0
            or not np.isclose(actual_rms, qc["joint_rms"], rtol=2 * np.finfo(np.float32).eps,
                              atol=float(np.nextafter(np.float32(0), np.float32(1))))):
        raise SourceCacheAcceptanceError("stored_source_RMS_does_not_match_pre_cast_pre_RMS_QC")


def verify_source_cache_acceptance(cache):
    """Validate a freshly read VerifiedCache; return all retained artifact SHAs.

    The caller first validates X/y/events and signal processing. This additional
    gate requires complete native coverage and historical completed GET proofs,
    without requiring raw files on the GPU restore host or making any request.
    """
    root = Path(cache.root).resolve(strict=True)
    artifacts, required_cache_backup = {}, {}

    def checked(item, *, cache_backup=False):
        if type(item) is not dict or not {"path", "sha256"} <= set(item):
            raise SourceCacheAcceptanceError("explicit_source_artifact_SHA_descriptor_required")
        path = local_path(root, item["path"])
        if (not path.is_file() or re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is None
                or file_digest(path) != item["sha256"]
                or "bytes" in item and item["bytes"] != path.stat().st_size):
            raise SourceCacheAcceptanceError("source_artifact_SHA_or_size_mismatch")
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = item["sha256"]
        if cache_backup:
            required_cache_backup[relative] = (item["sha256"], path.stat().st_size)
        return path

    def descriptor(path):
        return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}

    def verified_backup(path):
        try:
            return load_verified_receipt(path)
        except (_backup.BackupError, OSError, ValueError, KeyError, TypeError) as exc:
            raise SourceCacheAcceptanceError("source_R2_receipt_not_complete_GET_verified") from exc

    tasks_path = local_path(root, "status/tasks.json")
    tasks = json.loads(tasks_path.read_text())
    w2 = tasks.get("W2", {})
    if (tasks.get("W1", {}).get("status") != "COMPLETE" or w2.get("status") != "COMPLETE"
            or w2.get("input_contract_sha256") != cache.input_contract_sha256):
        raise SourceCacheAcceptanceError("complete_same_input_W1_W2_required")
    manifest_path = checked(w2.get("cache_manifest"), cache_backup=True)
    if (manifest_path != local_path(root, cache.manifest_path) or file_digest(manifest_path) != cache.manifest_sha256):
        raise SourceCacheAcceptanceError("cache_capability_not_bound_to_actual_W2_manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != "paper3-source-cache-v2" or manifest.get("status") != "SOURCE_CACHE_COMPLETE"
            or manifest.get("synthetic") is not False or manifest.get("old_npz_reused") is not False
            or manifest.get("full_source_inventory_verified") is not True or manifest.get("normalization_applied") is not False
            or manifest.get("training_started") is not False or manifest.get("role") != "source_train"
            or manifest.get("stage") != "pre_rms_car11_temporal_demeaned"
            or manifest.get("input_contract_sha256") != cache.input_contract_sha256
            or any(manifest.get(k) for k in ("fixture_scope", "test_double", "test_double_not_research_evidence"))):
        raise SourceCacheAcceptanceError("complete_real_full_source_inventory_cache_required_no_schema_subset")
    contract_path = local_path(root, "contracts/input_contract.json")
    checked(descriptor(contract_path), cache_backup=True)
    contract = json.loads(contract_path.read_text())
    if validate_signal_contract(contract) != cache.input_contract_sha256:
        raise SourceCacheAcceptanceError("frozen_signal_contract_must_match_source_cache_capability")
    inventory_path = checked(manifest.get("raw_inventory"), cache_backup=True)
    inventory = json.loads(inventory_path.read_text())
    expected = expected_source_objects()
    files = inventory.get("files", [])
    identities = [(r.get("dataset"), str(r.get("provider_subject")), str(r.get("session_id")), r.get("file")) for r in files]
    if (len(expected) != 505 or inventory.get("schema_version") != "paper3-source-native-inventory-v2"
            or inventory.get("status") != "SOURCE_RAW_INVENTORY_COMPLETE" or inventory.get("role") != "source_train"
            or inventory.get("synthetic") is not False or inventory.get("old_npz_reused") is not False
            or inventory.get("provider_roster_verified") is not True or len(identities) != 505
            or len(set(identities)) != 505 or set(identities) != set(expected)):
        raise SourceCacheAcceptanceError("exact_505_official_native_source_objects_required")
    terms_path = checked(inventory.get("provider_terms_evidence"), cache_backup=True)
    terms = json.loads(terms_path.read_text())
    if (set(terms.get("licences", {})) != SOURCE_IDS or not terms.get("files")
            or any(not isinstance(row.get("value"), str) or not row["value"] or not isinstance(row.get("evidence"), str)
                   or not row["evidence"] for row in terms["licences"].values())):
        raise SourceCacheAcceptanceError("retained_four_source_terms_and_license_evidence_required")
    for evidence in terms["files"]:
        checked(evidence, cache_backup=True)
    adapter_path = local_path(root, "p3work/adapters/source_native.py")
    adapter_sha = file_digest(adapter_path)
    artifacts[adapter_path.relative_to(root).as_posix()] = adapter_sha
    native, people, reader_revisions, person_reader_revisions = {}, set(), {}, {}
    raw_receipt_descriptors, raw_catalog_objects = {}, {}
    for item, ident in zip(files, identities):
        dataset, subject, session, name = ident
        if (item.get("url") != expected[ident] or item.get("version") != VERSIONS[dataset]
                or item.get("complete_download") is not True or item.get("http_status") != 200
                or item.get("native_schema_audited") is not True or type(item.get("bytes")) is not int or item["bytes"] <= 0
                or str(item.get("response_content_length")) != str(item["bytes"])
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
                or Path(item.get("path", "")).name != name):
            raise SourceCacheAcceptanceError("official_source_complete_download_native_audit_identity_required")
        people.add((dataset, subject))
        metadata_path = checked(item.get("native_trials"))
        audit_path = checked(item.get("adapter_audit"))
        receipt_path = checked(item.get("backup_receipt"), cache_backup=True)
        receipt_rows, _ = verified_backup(receipt_path)
        raw = {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
               "role": "source_development_raw", "dataset": dataset}
        expected_objects = {(raw["path"], raw["sha256"], raw["bytes"], raw["role"], dataset)}
        for path in (metadata_path, audit_path):
            expected_objects.add((path.relative_to(root).as_posix(), file_digest(path), path.stat().st_size, "audit", dataset))
        if len(receipt_rows) != 3 or {(r["path"], r["sha256"], r["bytes"], r["role"], r["dataset"]) for r in receipt_rows} != expected_objects:
            raise SourceCacheAcceptanceError("raw_native_and_audit_need_their_exact_completed_three_object_R2_GET_receipt")
        raw_receipt_descriptors[receipt_path.relative_to(root).as_posix()] = file_digest(receipt_path)
        raw_catalog_objects[ident] = (next(r for r in receipt_rows if r["role"] == "source_development_raw"), item["backup_receipt"])
        rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        audit = json.loads(audit_path.read_text())
        if (not rows or audit.get("schema_audit_status") != "actual_native_schema_read"
                or audit.get("source_file_sha256") != item["sha256"] or audit.get("source_file_bytes") != item["bytes"]
                or audit.get("n_native_lr_trials") != len(rows) or audit.get("selected_channel_order") != list(CHANNELS)
                or audit.get("adapter_file_sha256") != adapter_sha
                or audit.get("label_counts") != dict(Counter(r.get("canonical_label") for r in rows))):
            raise SourceCacheAcceptanceError("native_metadata_and_fixed_adapter_audit_count_identity_mismatch")
        try:
            revision = verify_native_reader_provenance(root, dataset, rows, audit)
        except (SourceCacheGateError, OSError, ValueError, KeyError, TypeError) as error:
            raise SourceCacheAcceptanceError("native_registered_reader_revision_or_row_audit_binding_mismatch") from error
        if revision is not None:
            reader_revisions[revision["revision_sha256"]] = revision
            person_reader_revisions.setdefault((dataset, subject), {})[revision["revision_sha256"]] = revision
            for key in ("base_adapter", "variant_reader", "dispatcher"):
                checked(revision[key])
        for row in rows:
            uid = row.get("trial_uid")
            if (row.get("dataset") != dataset or row.get("role") != "source"
                    or str(row.get("provider_subject")) != subject or str(row.get("session_id")) != session
                    or row.get("source_version") != item["version"] or row.get("source_file_sha256") != item["sha256"]
                    or row.get("source_file_bytes") != item["bytes"] or not isinstance(row.get("source_record_id"), str)
                    or not (row["source_record_id"] == name or row["source_record_id"].startswith(name + "::"))
                    or type(row.get("original_trial_index")) is not int or row["original_trial_index"] < 1
                    or row.get("canonical_label") != _LABELS[dataset].get(row.get("raw_label_code"))
                    or row.get("canonical_label") not in {"left_hand", "right_hand"}
                    or row.get("task_end_verified") is not True or type(row.get("task_end_sample")) is not int
                    or type(row.get("analysis_anchor_sample")) is not int or row["task_end_sample"] <= row["analysis_anchor_sample"]
                    or any(k not in row for k in _UID_KEYS) or uid != digest({k: row[k] for k in _UID_KEYS}) or uid in native):
                raise SourceCacheAcceptanceError("native_original_trial_UID_source_class_or_provenance_mismatch")
            native[uid] = row
    expected_revisions = [reader_revisions[key] for key in sorted(reader_revisions)]
    if expected_revisions or any(key in manifest for key in ("native_reader_revisions", "reader_dispatcher_code")):
        if manifest.get("native_reader_revisions") != expected_revisions:
            raise SourceCacheAcceptanceError("cache_manifest_reader_revisions_must_match_all_native_objects")
        dispatcher = manifest.get("reader_dispatcher_code")
        if not isinstance(dispatcher, dict) or dispatcher.get("path") != "p3work/adapters/source_dispatch.py":
            raise SourceCacheAcceptanceError("cache_manifest_actual_reader_dispatcher_code_required")
        dispatcher_path = checked(dispatcher)
        from .adapters import source_dispatch
        if (dispatcher.get("bytes") != dispatcher_path.stat().st_size
                or dispatcher["sha256"] != file_digest(Path(source_dispatch.__file__))):
            raise SourceCacheAcceptanceError("packaged_reader_dispatcher_differs_from_actual_executed_code")
    catalog_path = checked({"path": w2.get("raw_r2_receipts_catalog"), "sha256": w2.get("raw_r2_receipts_catalog_sha256")}, cache_backup=True)
    catalog = json.loads(catalog_path.read_text())
    objects = catalog.get("objects", [])
    if (catalog.get("status") != "ALL_505_RAW_OBJECTS_FULL_GET_SHA256_VERIFIED"
            or checked(catalog.get("raw_inventory")) != inventory_path
            or {r.get("path"): r.get("sha256") for r in catalog.get("receipts", [])} != raw_receipt_descriptors
            or len(catalog.get("receipts", [])) != 505 or len(objects) != 505):
        raise SourceCacheAcceptanceError("full_505_source_R2_receipts_catalog_required")
    catalog_seen = set()
    for obj in objects:
        ident = tuple(obj.get("identity", []))
        if ident not in raw_catalog_objects or ident in catalog_seen or (obj.get("raw"), obj.get("backup_receipt")) != raw_catalog_objects[ident]:
            raise SourceCacheAcceptanceError("source_R2_catalog_raw_receipt_identity_mismatch")
        catalog_seen.add(ident)
    seen_people, accounted, valid_count, excluded_count = set(), set(), 0, 0
    cache_rows_by_uid = {row["event_uid"]: (row, location) for row, location in zip(cache.rows, cache.locations)}
    if len(cache_rows_by_uid) != len(cache.rows):
        raise SourceCacheAcceptanceError("duplicated_valid_source_trial")
    for shard in manifest.get("shards", []) + manifest.get("participants_without_usable_trials", []):
        person = (shard.get("dataset"), str(shard.get("subject_id")))
        if person not in people or person in seen_people:
            raise SourceCacheAcceptanceError("source_participant_shards_must_cover_exact_official_roster_once")
        seen_people.add(person)
        expected_person_revisions = person_reader_revisions.get(person, {})
        expected_person_revisions = [expected_person_revisions[key] for key in sorted(expected_person_revisions)]
        if expected_person_revisions or "native_reader_revisions" in shard:
            if shard.get("native_reader_revisions") != expected_person_revisions:
                raise SourceCacheAcceptanceError("source_shard_reader_revisions_must_match_its_original_metadata")
        for key, expected_path in (("adapter_code", "p3work/adapters/source_native.py"),
                                   ("processing_code", "p3work/preprocess.py")):
            if key in shard:
                if not isinstance(shard[key], dict) or shard[key].get("path") != expected_path:
                    raise SourceCacheAcceptanceError("source_shard_exact_adapter_or_processing_code_path_required")
                checked(shard[key])
        exclusions = json.loads(checked(shard.get("exclusions"), cache_backup=True).read_text())
        if not isinstance(exclusions, list):
            raise SourceCacheAcceptanceError("explicit_original_source_exclusions_required")
        events = []
        if "X" in shard:
            for name in ("X", "y", "events"):
                checked(shard[name], cache_backup=True)
            events = json.loads(local_path(root, shard["events"]["path"]).read_text())
            if (shard.get("original_trials_processed") != len(events) + len(exclusions)
                    or shard.get("valid_trials") != len(events) or shard.get("excluded_trials") != len(exclusions)):
                raise SourceCacheAcceptanceError("source_shard_original_valid_excluded_counts_mismatch")
        elif (shard.get("reason") != "no_valid_trials_under_frozen_contract" or not exclusions
              or shard.get("excluded_original_trials") != len(exclusions)):
            raise SourceCacheAcceptanceError("no_valid_source_participant_requires_full_original_exclusion_audit")
        for row, excluded in [(row, False) for row in events] + [(row, True) for row in exclusions]:
            uid = row.get("event_uid")
            original = native.get(uid)
            if (original is None or uid in accounted or (original["dataset"], str(original["provider_subject"])) != person):
                raise SourceCacheAcceptanceError("source_trial_missing_duplicate_or_wrong_participant")
            if excluded:
                if row.get("reason") not in _EXCLUSIONS or row.get("original_metadata") != original:
                    raise SourceCacheAcceptanceError("excluded_source_trial_must_retain_exact_native_metadata_and_QC_reason")
                excluded_count += 1
            else:
                if (uid not in cache_rows_by_uid or row != cache_rows_by_uid[uid][0]
                        or any(row.get(k) != v for k, v in original.items() if k != "role")
                        or any((key in row) != (key in original) or row.get(key) != original.get(key)
                               for key in _READER_FIELDS)
                        or row.get("role") != "source_train" or row.get("raw_sha256") != original["source_file_sha256"]
                        or row.get("original_trial_id") != original["original_trial_index"]
                        or row.get("label") != {"left_hand": 0, "right_hand": 1}[original["canonical_label"]]):
                    raise SourceCacheAcceptanceError("valid_source_event_native_trial_label_or_cache_identity_mismatch")
                shard_index, trial_index = cache_rows_by_uid[uid][1]
                array = cache.arrays[shard_index]
                if array.dtype != np.dtype("float32"):
                    raise SourceCacheAcceptanceError("new_source_cache_builder_float32_storage_required")
                actual = hashlib.sha256(np.asarray(array[trial_index], dtype="<f4").tobytes(order="C")).hexdigest()
                if actual != row.get("processed_signal_sha256"):
                    raise SourceCacheAcceptanceError("processed_source_signal_SHA_does_not_match_actual_float32_cache_bytes")
                _validate_context_and_qc(row, original, contract, cache.input_contract_sha256, array[trial_index])
                valid_count += 1
            accounted.add(uid)
    if (seen_people != people or accounted != set(native) or valid_count != len(cache.rows)
            or manifest.get("original_trials_processed") != len(native)):
        raise SourceCacheAcceptanceError("all_source_people_and_all_original_LR_trials_must_be_valid_or_explicitly_excluded")
    backup_path = checked({"path": w2.get("cache_r2_receipt"), "sha256": w2.get("cache_r2_receipt_sha256")})
    backed, _ = verified_backup(backup_path)
    if any(row["role"] == "source_development_raw" for row in backed):
        raise SourceCacheAcceptanceError("GPU_cache_backup_must_not_package_original_raw_payloads")
    backed_files = {row["path"]: (row["sha256"], row["bytes"]) for row in backed}
    if any(backed_files.get(relative) != value for relative, value in required_cache_backup.items()):
        raise SourceCacheAcceptanceError("cache_R2_complete_GET_proof_does_not_cover_all_required_cache_and_metadata_artifacts")
    return {"status": "FULL_SOURCE_CACHE_ACCEPTED", "raw_objects": 505, "participants": len(people),
        "original_trials": len(native), "valid_trials": valid_count, "excluded_trials": excluded_count,
        "raw_local_required": False, "network_or_training_started": False,
        "native_context_verification": "metadata_binding_and_SHA_format_not_independent_raw_reprocessing",
        "artifact_sha256": artifacts}
