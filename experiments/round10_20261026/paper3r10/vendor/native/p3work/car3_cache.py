"""CPU-only CAR3 sensitivity preparation from the exact accepted source cache.

No network, model, fit, GPU or external payload access. This module deliberately
does not implement an input accepted by the existing clinic11 training gate.
The inherited float32 cache cannot recover its pre-cast float64 parent signal.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

BASELINE_MANIFEST = "cache/source_v2/source_cache.json"
BASELINE_SHA = "bfc204577dfe51c1fc798c1241391ab63509fc2ad9ebfb267ba09254e01ae644"
BASELINE_CONTRACT_SHA = "5c8ea3f01634c060546af29b9330d314c4f03c265677a74b875ac7c302e31872"
CONTRACT_PATH = "contracts/sensitivity_car3_v1.json"
OUTPUT = "cache/source_c3czc4_v1"
AUDIT = "audit/external_expansion_20261019/round2/car3_preparation"
SOURCE_COUNTS = {"BNCI2014_001": (9, 2592), "Cho2017": (52, 10520),
                 "Lee2019_MI": (54, 10800), "PhysionetMI": (109, 4918)}
CHANNELS11 = ("FC3", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP4")
SELECT = (3, 5, 7)
ROLE = "source_sensitivity_preparation_only"
STAGE11 = "pre_rms_car11_temporal_demeaned"
STAGE3 = "pre_rms_car3_temporal_demeaned_from_verified_float32_car11"
UID_KEYS = ("dataset", "source_version", "provider_subject", "session_id", "source_file_sha256",
            "source_record_id", "physical_run_id", "original_trial_index", "raw_label_code")
LABELS = {"BNCI2014_001": {1: "left_hand", 2: "right_hand"},
          "Cho2017": {"imagery_left": "left_hand", "imagery_right": "right_hand"},
          "Lee2019_MI": {1: "right_hand", 2: "left_hand"},
          "PhysionetMI": {"T1": "left_hand", "T2": "right_hand"}}


class Car3PreparationError(ValueError):
    """Fixed local preparation gate failure, without credential or EEG output."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def local(root, relative, *, exists=True):
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
            or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))
            or any(ord(c) < 32 for c in relative)
            or any(p.lower().startswith(".env") or p.lower() in
                   {".aws", ".ssh", "credentials", "secrets", "legacy_baseline_v1"}
                   for p in relative.split("/"))
            or Path(relative).suffix.lower() in {".npz", ".pkl", ".pickle", ".key", ".pem"}):
        raise Car3PreparationError("unsafe_or_legacy_path")
    cursor = root
    for part in relative.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise Car3PreparationError("symlink_forbidden")
    path = cursor.resolve(strict=exists)
    if not path.is_relative_to(root):
        raise Car3PreparationError("path_outside_root")
    return path


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha(path),
            "bytes": path.stat().st_size}


def checked(root, item, artifacts, *, required_prefix=None):
    if (not isinstance(item, dict) or not {"path", "sha256"} <= set(item)
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None):
        raise Car3PreparationError("explicit_artifact_SHA_required")
    path = local(root, item["path"])
    if required_prefix is not None and not path.is_relative_to(root / required_prefix):
        raise Car3PreparationError("input_not_in_frozen_source_cache")
    actual = descriptor(root, path)
    if (actual["sha256"] != item["sha256"] or
            "bytes" in item and item["bytes"] != actual["bytes"]):
        raise Car3PreparationError("input_SHA_or_bytes_mismatch")
    artifacts[actual["path"]] = actual
    return path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Car3PreparationError("malformed_JSON") from error


def validate_contract(contract):
    expected = {"schema_version": "paper3-source-car3-sensitivity-preparation-v1",
        "status": "CPU_PREPARATION_ONLY_NOT_SCIENTIFIC_FREEZE", "role": ROLE,
        "channels": ["C3", "Cz", "C4"], "reference": "CAR_exact_C3_Cz_C4",
        "native_reference_requirement": "monopolar_common_reference_only",
        "bipolar_derivations_accepted": False, "BCIIV2b_accepted": False,
        "sfreq_hz": 128, "context_s": [0, 1.5], "crop_s": [.25, 1.25],
        "n_times": 128, "band_hz": [8, 30], "unit": "V", "stage": STAGE3,
        "normalization_applied": False, "baseline_input_contract_semantic_sha256": BASELINE_CONTRACT_SHA,
        "baseline_source_manifest_sha256": BASELINE_SHA,
        "source_datasets": sorted(SOURCE_COUNTS), "same_source_subjects_and_trials": True,
        "new_independent_participants": 0, "training_ready": False, "actual_fits": 0,
        "primary_11channel_compatible": False, "GPU_training_entry_implemented": False,
        "scientific_configuration_frozen": False}
    if (not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items())
            or any(type(contract.get(k)) is not bool for k in
                ("normalization_applied", "training_ready", "primary_11channel_compatible"))):
        raise Car3PreparationError("separate_preparation_only_CAR3_contract_required")
    geometry = contract.get("geometry_constraints", {})
    if any(geometry.get(k) != v for k, v in {"C": 3, "centered_dimension_d": 2,
            "full_traceless_dimension": 3, "symmetric_traceless_dimension": 2,
            "skew_symmetric_dimension": 1, "original_r4_or_r8_config_accepted": False}.items()):
        raise Car3PreparationError("CAR3_geometry_constraint_mismatch")
    return digest(contract)


def project_car3(signal):
    """P3 S on bounded channels-by-time or trials-by-channels-by-time arrays."""
    x = np.asarray(signal)
    if x.ndim not in (2, 3) or x.shape[-2:] != (11, 128) or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise Car3PreparationError("finite_11_by_128_float_signal_required")
    selected = np.asarray(x[..., SELECT, :], dtype=np.float64)
    selected -= selected.mean(axis=-2, keepdims=True)
    return selected


def rounding_bound(signal):
    return float(6 * float(np.finfo(np.float32).eps) * float(np.max(np.abs(signal)))
                 + 4 * float(np.nextafter(np.float32(0), np.float32(1))))


def signal_sha(signal):
    return hashlib.sha256(np.asarray(signal, dtype="<f4").tobytes(order="C")).hexdigest()


def _validate_row(row, signal, label, dataset, subject, seen):
    qc = row.get("preprocessing_qc", {})
    uid = row.get("event_uid")
    if (not isinstance(row, dict) or row.get("dataset") != dataset or row.get("role") != "source_train"
            or row.get("cohort") != "source" or str(row.get("subject_id")) != subject
            or str(row.get("provider_subject")) != subject or row.get("input_contract_sha256") != BASELINE_CONTRACT_SHA
            or any(k not in row for k in UID_KEYS) or uid != digest({k: row[k] for k in UID_KEYS})
            or uid != row.get("trial_uid") or uid in seen
            or type(row.get("label")) is not int or row["label"] != int(label)
            or row.get("canonical_label") != {0: "left_hand", 1: "right_hand"}.get(int(label))
            or row.get("canonical_label") != LABELS[dataset].get(row.get("raw_label_code"))
            or row.get("processed_signal_sha256") != signal_sha(signal)
            or row.get("raw_sha256") != row.get("source_file_sha256")
            or row.get("task_end_verified") is not True):
        raise Car3PreparationError("source_UID_label_signal_or_role_mismatch")
    sfreq, anchor = row.get("native_sfreq"), row.get("analysis_anchor_sample")
    bounds, context = row.get("real_trial_start_end"), row.get("original_context_samples")
    if (type(sfreq) not in (int, float) or not np.isfinite(sfreq) or sfreq <= 60
            or type(anchor) is not int or not isinstance(bounds, list) or len(bounds) != 2
            or any(type(v) is not int for v in bounds) or not 0 <= bounds[0] <= anchor < bounds[1]
            or context != [anchor, anchor + round(1.5 * sfreq)] or context[1] > bounds[1]
            or type(row.get("task_end_sample")) is not int or context[1] > row["task_end_sample"]):
        raise Car3PreparationError("original_exact_1p5_context_and_task_bounds_required")
    unit = "V" if dataset == "PhysionetMI" else "uV"
    factor = 1. if unit == "V" else 1e-6
    if (not isinstance(qc, dict) or qc.get("signal_contract_sha256") != BASELINE_CONTRACT_SHA
            or qc.get("normalization_applied") is not False or qc.get("reference") != "CAR_exact_clinic11"
            or qc.get("data_after_context_accessed") is not False or qc.get("native_unit") != unit
            or qc.get("stored_unit") != "V" or qc.get("si_conversion_factor") != factor
            or qc.get("filter_context_ends_at_native_sample") != context[1] - anchor
            or row.get("effective_numeric_unit") != unit):
        raise Car3PreparationError("verified_pre_RMS_unit_and_original_QC_required")
    if dataset == "PhysionetMI":
        valid_units = {"V": 1., "uV": 1e-6, "µV": 1e-6, "mV": 1e-3}
        if (row.get("unit_verified_from_native_header") is not True
                or row.get("data_returned_by_mne_unit") != "V"
                or row.get("physical_scale_to_si_volts") != valid_units.get(row.get("native_unit_string"))
                or row.get("unit_evidence_kind") != "native_EDF_header_MNE_SI"):
            raise Car3PreparationError("Physio_native_SI_unit_evidence_required")
    elif (row.get("numeric_unit_declared_by_fixed_adapter") != "uV"
            or row.get("native_unit_string") is not None or row.get("unit_verified_from_native_header") is not False
            or row.get("absolute_amplitude_qc_authorized") is not False
            or row.get("unit_evidence_kind") != "fixed_adapter_numeric_unit_declaration_not_native_header"):
        raise Car3PreparationError("MAT_fixed_adapter_unit_evidence_required")
    rms = float(np.sqrt(np.mean(np.square(np.asarray(signal, dtype=np.float64)))))
    if (type(qc.get("joint_rms")) not in (int, float) or not np.isfinite(qc["joint_rms"])
            or rms <= 0 or not np.isclose(rms, qc["joint_rms"], rtol=2*np.finfo(np.float32).eps,
                atol=float(np.nextafter(np.float32(0), np.float32(1))))):
        raise Car3PreparationError("stored_pre_RMS_energy_QC_mismatch")
    tol = rounding_bound(signal)
    if (np.max(np.abs(np.asarray(signal, dtype=np.float64).mean(axis=0))) > tol
            or np.max(np.abs(np.asarray(signal, dtype=np.float64).mean(axis=-1))) > tol):
        raise Car3PreparationError("expected_CAR11_and_temporal_demean_required")
    result = project_car3(signal)
    if not np.isfinite(result).all() or float(np.sum(result*result)) <= 0:
        raise Car3PreparationError("CAR3_nonfinite_or_zero_energy")
    seen.add(uid)


def validate_shard(root, shard, artifacts, seen):
    dataset, subject = shard.get("dataset"), str(shard.get("subject_id"))
    if (dataset not in SOURCE_COUNTS or re.fullmatch(r"[1-9][0-9]*", subject) is None
            or shard.get("role") != "source_train" or shard.get("stage") != STAGE11
            or shard.get("input_contract_sha256") != BASELINE_CONTRACT_SHA):
        raise Car3PreparationError("only_four_source_pre_RMS_shards_allowed")
    paths = {name: checked(root, shard.get(name), artifacts, required_prefix="cache/source_v2")
             for name in ("X", "y", "events", "exclusions")}
    if any(paths[name].as_posix() != (root / "cache/source_v2" / dataset / subject /
            {"X": "X.npy", "y": "y.npy", "events": "events.json", "exclusions": "exclusions.json"}[name]).as_posix()
           for name in paths):
        raise Car3PreparationError("exact_frozen_per_participant_paths_required")
    for name, expected in (("adapter_code", "p3work/adapters/source_native.py"),
                           ("processing_code", "p3work/preprocess.py")):
        if shard.get(name, {}).get("path") != expected:
            raise Car3PreparationError("source_provenance_code_path_mismatch")
        checked(root, shard[name], artifacts)
    x = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
    y = np.load(paths["y"], mmap_mode="r", allow_pickle=False)
    events, exclusions = read_json(paths["events"]), read_json(paths["exclusions"])
    n = shard.get("valid_trials")
    if (type(n) is not int or n <= 0 or x.dtype != np.dtype("float32") or x.shape != (n, 11, 128)
            or y.dtype != np.dtype("int64") or y.shape != (n,) or not np.isin(y, [0, 1]).all()
            or not isinstance(events, list) or len(events) != n or exclusions != []
            or shard.get("excluded_trials") != 0 or shard.get("original_trials_processed") != n):
        raise Car3PreparationError("frozen_X_y_events_exclusions_schema_mismatch")
    for start in range(0, n, 64):
        batch = x[start:start+64]
        if not np.isfinite(batch).all():
            raise Car3PreparationError("nonfinite_source_array")
        for offset, signal in enumerate(batch):
            row = events[start+offset]
            if not isinstance(row, dict):
                raise Car3PreparationError("source_event_object_required")
            _validate_row(row, signal, y[start+offset], dataset, subject, seen)
    return {"shard": shard, "paths": paths, "n": n}


def validate_inputs(root):
    """Full local SHA and bounded-memory schema audit. Writes nothing."""
    root = Path(root).resolve(strict=True)
    artifacts = {}
    cache_path = checked(root, {"path": BASELINE_MANIFEST, "sha256": BASELINE_SHA}, artifacts)
    manifest = read_json(cache_path)
    expected = {"schema_version": "paper3-source-cache-v2", "status": "SOURCE_CACHE_COMPLETE",
        "synthetic": False, "old_npz_reused": False, "role": "source_train",
        "stage": STAGE11, "input_contract_sha256": BASELINE_CONTRACT_SHA,
        "full_source_inventory_verified": True, "exact_window_acceptance_scope": "complete_frozen_W1_verified",
        "normalization_applied": False, "training_started": False,
        "participants_without_usable_trials": []}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise Car3PreparationError("exact_complete_pre_RMS_four_source_manifest_required")
    baseline_path = local(root, "contracts/input_contract.json")
    baseline = read_json(baseline_path)
    if (digest(baseline) != BASELINE_CONTRACT_SHA or baseline.get("channels") != list(CHANNELS11)
            or baseline.get("reference") != "CAR_exact_clinic11" or baseline.get("sfreq_hz") != 128
            or baseline.get("context_s") != [0, 1.5] or baseline.get("crop_s") != [.25, 1.25]
            or baseline.get("n_times") != 128 or baseline.get("status") != "window_frozen"):
        raise Car3PreparationError("frozen_baseline_11channel_contract_required")
    artifacts["contracts/input_contract.json"] = descriptor(root, baseline_path)
    checked(root, manifest["raw_inventory"], artifacts)
    if "reader_dispatcher_code" in manifest:
        checked(root, manifest["reader_dispatcher_code"], artifacts)
    for revision in manifest.get("native_reader_revisions", []):
        for field in ("base_adapter", "dispatcher", "variant_reader"):
            checked(root, revision[field], artifacts)
    contract_path = local(root, CONTRACT_PATH)
    contract = read_json(contract_path)
    contract_sha = validate_contract(contract)
    artifacts[CONTRACT_PATH] = descriptor(root, contract_path)
    if not isinstance(manifest.get("shards"), list):
        raise Car3PreparationError("source_shards_required")
    seen, people, records, trials = set(), set(), [], Counter()
    counts = Counter()
    for shard in manifest["shards"]:
        if not isinstance(shard, dict):
            raise Car3PreparationError("source_shard_object_required")
        record = validate_shard(root, shard, artifacts, seen)
        person = shard["dataset"], str(shard["subject_id"])
        if person in people:
            raise Car3PreparationError("duplicate_source_participant")
        people.add(person); records.append(record)
        counts[person[0]] += 1; trials[person[0]] += record["n"]
    actual = {dataset: (counts[dataset], trials[dataset]) for dataset in counts}
    if (actual != SOURCE_COUNTS or len(seen) != sum(v[1] for v in SOURCE_COUNTS.values())
            or manifest.get("original_trials_processed") != len(seen)):
        raise Car3PreparationError("exact_224_participant_28830_trial_coverage_required")
    return {"root": root, "manifest": manifest, "records": records, "artifacts": artifacts,
            "contract": contract, "contract_sha": contract_sha, "contract_path": contract_path,
            "dataset_counts": {d: {"provider_participant_ids": counts[d], "valid_trials": trials[d]}
                               for d in sorted(counts)}}


def write_json_new(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _unchanged(root, artifacts):
    for relative, original in artifacts.items():
        if descriptor(root, local(root, relative)) != original:
            raise Car3PreparationError("input_changed_during_preparation")


def prepare(root, *, execute=False):
    """Default full local validation only; execute creates immutable new outputs.

    A pre-existing output or audit directory is refused. Interrupted outputs
    remain unclaimed and cannot be resumed, overwritten or accepted as complete.
    """
    validated = validate_inputs(root)
    root = validated["root"]
    destination, audit = local(root, OUTPUT, exists=False), local(root, AUDIT, exists=False)
    if destination.exists() or audit.exists():
        raise Car3PreparationError("existing_output_refused_no_implicit_resume")
    common = {"schema_version": "paper3-car3-preparation-receipt-v1", "role": ROLE,
        "training_ready": False, "actual_fits": 0, "training_started": False,
        "GPU_actions": 0, "network_operations": 0, "external_payloads_accessed": False,
        "primary_11channel_compatible": False, "new_independent_participants": 0,
        "baseline_manifest": validated["artifacts"][BASELINE_MANIFEST],
        "baseline_input_contract_semantic_sha256": BASELINE_CONTRACT_SHA,
        "sensitivity_contract": descriptor(root, validated["contract_path"]),
        "sensitivity_contract_semantic_sha256": validated["contract_sha"],
        "source_dataset_counts": validated["dataset_counts"],
        "source_provider_participant_ids": len(validated["records"]),
        "source_valid_trials": sum(r["n"] for r in validated["records"]),
        "source_excluded_trials": 0, "source_QC": "inherited original accepted 11-channel QC; no independent 3-channel native QC",
        "numerical_scope": validated["contract"]["numerical_scope"],
        "input_artifacts_SHA_verified": len(validated["artifacts"])}
    if not execute:
        return {**common, "status": "LOCAL_VALIDATION_ONLY_PASSED", "numerical_cache_created": False}
    codes = {p: descriptor(root, local(root, p)) for p in
             ("p3work/car3_cache.py", "tools/prepare_car3_cache.py")}
    if Path(__file__).read_bytes() != (root / "p3work/car3_cache.py").read_bytes():
        raise Car3PreparationError("executed_and_recorded_conversion_code_differ")
    destination.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    write_json_new(audit / "preparation_started.json", {**common, "status": "IN_PROGRESS_NOT_CONSUMABLE",
        "conversion_code": codes, "started_at_utc": datetime.now(timezone.utc).isoformat()})
    shards, produced = [], []
    max_cast_error, max_tolerance = 0., 0.
    for record in validated["records"]:
        parent, paths, n = record["shard"], record["paths"], record["n"]
        dataset, subject = parent["dataset"], str(parent["subject_id"])
        folder = destination / dataset / subject
        folder.mkdir(parents=True, exist_ok=False)
        x = np.load(paths["X"], mmap_mode="r", allow_pickle=False)
        output = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float32, shape=(n, 3, 128))
        events = read_json(paths["events"])
        for start in range(0, n, 64):
            exact_from_stored = project_car3(x[start:start+64])
            stored = exact_from_stored.astype(np.float32)
            if not np.isfinite(stored).all():
                raise Car3PreparationError("derived_float32_not_finite")
            tolerance = rounding_bound(x[start:start+64])
            error = float(np.max(np.abs(stored.astype(np.float64)-exact_from_stored)))
            if error > tolerance:
                raise Car3PreparationError("derived_float32_rounding_bound_exceeded")
            max_cast_error = max(max_cast_error, error); max_tolerance = max(max_tolerance, tolerance)
            output[start:start+len(stored)] = stored
            for offset, signal in enumerate(stored):
                row = events[start+offset]
                row["parent_preprocessing_qc"] = row.pop("preprocessing_qc")
                row["parent_processed_signal_sha256"] = row.pop("processed_signal_sha256")
                row["parent_processing_sha256"] = row.pop("processing_sha256")
                row["parent_selected_indices"] = row.pop("selected_indices")
                row.update(role=ROLE, input_contract_sha256=validated["contract_sha"],
                    baseline_input_contract_sha256=BASELINE_CONTRACT_SHA,
                    processing_sha256=codes["p3work/car3_cache.py"]["sha256"],
                    processed_signal_sha256=signal_sha(signal), stage=STAGE3,
                    preprocessing_qc={"reference": "CAR_exact_C3_Cz_C4", "normalization_applied": False,
                        "stored_unit": "V", "inherited_original_11channel_QC": True,
                        "independent_3channel_native_QC_performed": False})
        output.flush(); del output
        shutil.copyfile(paths["y"], folder / "y.npy")
        shutil.copyfile(paths["exclusions"], folder / "exclusions.json")
        write_json_new(folder / "events.json", events)
        write_json_new(folder / "uids.json", [e["event_uid"] for e in events])
        header = {"schema_version": "paper3-car3-shard-header-v1", "role": ROLE,
            "dataset": dataset, "subject_id": subject, "shape": [n, 3, 128], "dtype": "float32",
            "unit": "V", "channels": ["C3", "Cz", "C4"], "reference": "CAR_exact_C3_Cz_C4",
            "stage": STAGE3, "context_s": [0, 1.5], "crop_s": [.25, 1.25], "sfreq_hz": 128,
            "normalization_applied": False, "training_ready": False, "actual_fits": 0,
            "primary_11channel_compatible": False, "source_QC": common["source_QC"],
            "parent_artifacts": {k: validated["artifacts"][parent[k]["path"]]
                                 for k in ("X", "y", "events", "exclusions")},
            "sensitivity_contract_semantic_sha256": validated["contract_sha"], "conversion_code": codes}
        write_json_new(folder / "header.json", header)
        files = {name: descriptor(root, folder / filename) for name, filename in
                 {"X": "X.npy", "y": "y.npy", "events": "events.json", "uids": "uids.json",
                  "exclusions": "exclusions.json", "header": "header.json"}.items()}
        if files["y"]["sha256"] != parent["y"]["sha256"]:
            raise Car3PreparationError("derived_labels_not_byte_identical")
        shards.append({"dataset": dataset, "subject_id": subject, "valid_trials": n,
            "excluded_trials": 0, "role": ROLE, "stage": STAGE3, **files})
        produced.extend({**item, "role": "result", "dataset": dataset} for item in files.values())
    _unchanged(root, validated["artifacts"])
    for relative, original in codes.items():
        if descriptor(root, local(root, relative)) != original:
            raise Car3PreparationError("conversion_code_changed_during_preparation")
    manifest = {**common, "schema_version": "paper3-source-car3-cache-preparation-v1",
        "status": "SOURCE_CAR3_SENSITIVITY_PREPARED_NOT_TRAINING_READY", "channels": ["C3", "Cz", "C4"],
        "stage": STAGE3, "normalization_applied": False, "synthetic": False, "old_npz_reused": False,
        "shards": shards, "conversion_code": codes, "no_source_UID_or_label_changed": True}
    write_json_new(destination / "source_cache3.json", manifest)
    catalog = {"schema_version": "paper3-car3-input-artifact-SHA-review-v1", "status": "ALL_INPUT_SHA_MATCHED_BEFORE_AND_AFTER",
        "all_input_files_SHA_verified_before": True, "all_input_files_SHA_reverified_after": True,
        "original_inputs_modified": False, "files": list(validated["artifacts"].values())}
    write_json_new(audit / "input_unchanged_proof.json", catalog)
    receipt = {**common, "status": manifest["status"], "numerical_cache_created": True,
        "conversion_code": codes, "manifest": descriptor(root, destination / "source_cache3.json"),
        "input_unchanged_proof": descriptor(root, audit / "input_unchanged_proof.json"),
        "no_source_UID_or_label_changed": True, "all_input_files_SHA_verified_before_and_after": True,
        "maximum_output_cast_absolute_error_V": max_cast_error,
        "maximum_input_rounding_tolerance_V": max_tolerance,
        "float64_native_parent_reconstructed": False,
        "R2_backup_performed_by_this_tool": False, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json_new(audit / "conversion_receipt.json", receipt)
    for path in (destination / "source_cache3.json", audit / "input_unchanged_proof.json",
                 audit / "conversion_receipt.json", audit / "preparation_started.json", validated["contract_path"],
                 *(root / p for p in codes)):
        produced.append({**descriptor(root, path), "role": "archive", "dataset": "Paper3SourceSensitivityPreparation"})
    backup_manifest = {"schema_version": 1, "files": produced}
    write_json_new(audit / "r2_backup_manifest.json", backup_manifest)
    return {**receipt, "backup_manifest": descriptor(root, audit / "r2_backup_manifest.json")}
