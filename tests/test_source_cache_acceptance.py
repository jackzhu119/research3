"""505-object SOFTWARE_TEST_DOUBLE metadata proofs, never real EEG/R2 evidence.

All raw hashes/GET receipts below are fabricated in tmp_path, explicitly labelled
software doubles. No original raw file exists, no network client is created, and
no fit or GPU action occurs. The production gate itself has no test bypass.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from p3work import source_cache_acceptance as acceptance
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest
from tools.r2_restore import ENTRY_FIELDS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def proof_receipt(entries, prefix):
    """Fabricated completed envelope used solely to test local validation."""
    return {"schema_version": 1, "operation": "paper3_explicit_manifest_backup", "execute": True,
        "status": "verified", "verification": "full_get_body_sha256", "prefix": prefix,
        "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_A_REAL_UPLOAD_OR_GET", "objects": [
            {**entry, "key": prefix + entry["path"], "status": "verified",
             "put": {"status": "created", "http_status": 200, "request_body_bytes": entry["bytes"],
                     "request_body_sha256": entry["sha256"]},
             "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                     "bytes_read": entry["bytes"], "sha256": entry["sha256"], "declared_content_length": entry["bytes"]}}
            for entry in entries]}


@pytest.fixture
def factory(tmp_path):
    root = tmp_path
    adapter = root / "p3work/adapters/source_native.py"
    adapter.parent.mkdir(parents=True); adapter.write_bytes(b"# SOFTWARE_TEST_DOUBLE_ADAPTER_NO_EEG\n")
    contract = {"schema_version": "paper3-signal-v2-preparation", "status": "window_frozen",
        "channels": list(CHANNELS), "sfreq_hz": 128, "selected_profile": "short",
        "context_s": [0, 2], "crop_s": [0.5, 1.5], "n_times": 128,
        "protocol_freeze_sha256": None,
        "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "b" * 64,
            "raw_metadata_manifest_sha256": "c" * 64,
            "long": {"participant_mean_coverage": 0.75, "participant_mean_abs_class_exclusion_gap": 0.01},
            "short": {"participant_mean_coverage": 0.9, "participant_mean_abs_class_exclusion_gap": 0.01}},
        "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_ACTUAL_WINDOW_EVIDENCE"}
    signal_sha = digest(contract)
    contract_path = root / "contracts/input_contract.json"
    write(contract_path, contract)
    def desc(path): return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}
    def entry(path, role, dataset): return {**desc(path), "role": role, "dataset": dataset}
    terms = {"licences": {ds: {"value": "SOFTWARE_TEST_DOUBLE_NOT_LICENSE_ADVICE", "evidence": "retained fixture text"} for ds in SOURCE_IDS}, "files": []}
    for i in range(22):
        path = root / f"audit/terms/evidence-{i}.txt"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SOFTWARE_TEST_DOUBLE_PUBLIC_TERMS_PLACEHOLDER")
        terms["files"].append(desc(path))
    terms_path = root / "audit/terms/manifest.json"; write(terms_path, terms)
    inventory = {"schema_version": "paper3-source-native-inventory-v2", "status": "SOURCE_RAW_INVENTORY_COMPLETE",
        "role": "source_train", "synthetic": False, "old_npz_reused": False, "provider_roster_verified": True,
        "provider_terms_evidence": desc(terms_path), "files": []}
    by_person = defaultdict(list)
    for index, (ident, url) in enumerate(sorted(acceptance.expected_source_objects().items())):
        dataset, subject, session, name = ident
        raw_sha = hashlib.sha256(("SOFTWARE_TEST_DOUBLE_RAW/" + repr(ident)).encode()).hexdigest()
        raw_path = f"data/SOFTWARE_TEST_DOUBLE_MISSING_RAW/{index}/{name}"
        rows = []
        codes = ({"BNCI2014_001": [1, 2], "Cho2017": ["imagery_left", "imagery_right"],
                  "Lee2019_MI": [2, 1], "PhysionetMI": ["T1", "T2"]})[dataset]
        for trial, code in enumerate(codes, 1):
            sfreq = {"BNCI2014_001": 250.0, "Cho2017": 512.0, "Lee2019_MI": 1000.0, "PhysionetMI": 160.0}[dataset]
            anchor = 5000 + trial * 1000
            row = {"dataset": dataset, "role": "source", "provider_subject": subject, "session_id": session,
                "source_version": acceptance.VERSIONS[dataset], "source_file_sha256": raw_sha, "source_file_bytes": 317,
                "source_record_id": name + "::SOFTWARE_TEST_DOUBLE", "physical_run_id": "1", "original_trial_index": trial,
                "raw_label_code": code, "canonical_label": "left_hand" if trial == 1 else "right_hand",
                "task_end_verified": True, "task_end_sample": anchor + round(4 * sfreq),
                "analysis_anchor_sample": anchor, "native_sfreq": sfreq,
                "real_trial_start_end": [anchor - 250, anchor + round(4 * sfreq)],
                "physical_scale_evidence": "SOFTWARE_TEST_DOUBLE_NOT_A_NATIVE_UNIT_READ"}
            if dataset == "PhysionetMI":
                row.update(native_unit_string="uV", unit_verified_from_native_header=True,
                    physical_scale_to_si_volts=1e-6, data_returned_by_mne_unit="V")
            else:
                row.update(native_unit_string=None, numeric_unit_declared_by_fixed_adapter="uV",
                    unit_verified_from_native_header=False, absolute_amplitude_qc_authorized=False)
            row["trial_uid"] = digest({key: row[key] for key in acceptance._UID_KEYS})
            rows.append(row); by_person[dataset, subject].append(row)
        folder = root / f"audit/raw_objects/{index}"
        folder.mkdir(parents=True)
        native_path, audit_path, backup_path = folder / "trial_metadata.jsonl", folder / "adapter_audit.json", folder / "backup_receipt.json"
        native_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        write(audit_path, {"schema_audit_status": "actual_native_schema_read", "source_file_sha256": raw_sha,
            "source_file_bytes": 317, "n_native_lr_trials": len(rows), "selected_channel_order": list(CHANNELS),
            "adapter_file_sha256": file_digest(adapter), "label_counts": dict(Counter(r["canonical_label"] for r in rows)),
            "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_A_NATIVE_READ"})
        raw = {"path": raw_path, "sha256": raw_sha, "bytes": 317, "role": "source_development_raw", "dataset": dataset}
        write(backup_path, proof_receipt([raw, entry(native_path, "audit", dataset), entry(audit_path, "audit", dataset)], f"paper3/software-test-double/raw/{index}/"))
        inventory["files"].append({"dataset": dataset, "provider_subject": subject, "session_id": session, "file": name,
            "url": url, "version": acceptance.VERSIONS[dataset], "path": raw_path, "sha256": raw_sha, "bytes": 317,
            "complete_download": True, "http_status": 200, "response_content_length": "317", "native_schema_audited": True,
            "native_trials": desc(native_path), "adapter_audit": desc(audit_path), "backup_receipt": desc(backup_path)})
    inventory_path, manifest_path, catalog_path = root / "audit/inventory.json", root / "cache/source/source_cache.json", root / "audit/raw_catalog.json"
    manifest = {"schema_version": "paper3-source-cache-v2", "status": "SOURCE_CACHE_COMPLETE", "synthetic": False,
        "old_npz_reused": False, "role": "source_train", "stage": "pre_rms_car11_temporal_demeaned",
        "full_source_inventory_verified": True, "normalization_applied": False, "training_started": False,
        "input_contract_sha256": signal_sha, "original_trials_processed": sum(map(len, by_person.values())),
        "shards": [], "participants_without_usable_trials": []}
    rng = np.random.default_rng(63)
    for (dataset, subject), rows in sorted(by_person.items()):
        folder = root / "cache/source" / dataset / subject; folder.mkdir(parents=True)
        pre_cast = rng.normal(scale=1e-6, size=(len(rows), 11, 128))
        pre_cast -= pre_cast.mean(1, keepdims=True); pre_cast -= pre_cast.mean(2, keepdims=True)
        x = pre_cast.astype(np.float32)
        y = np.asarray([0 if r["canonical_label"] == "left_hand" else 1 for r in rows], dtype=np.int64)
        np.save(folder / "X.npy", x, allow_pickle=False); np.save(folder / "y.npy", y, allow_pickle=False)
        events = []
        for i, r in enumerate(rows):
            unit = "V" if dataset == "PhysionetMI" else "uV"
            ncontext = round(contract["context_s"][1] * r["native_sfreq"])
            events.append({**r, "role": "source_train", "event_uid": r["trial_uid"], "subject_id": subject,
                "original_trial_id": r["original_trial_index"], "label": int(y[i]), "input_contract_sha256": signal_sha,
                "raw_sha256": r["source_file_sha256"], "processed_signal_sha256": hashlib.sha256(x[i].astype("<f4").tobytes()).hexdigest(),
                "native_context_sha256": hashlib.sha256(("SOFTWARE_TEST_DOUBLE_CONTEXT/" + r["trial_uid"]).encode()).hexdigest(),
                "original_context_samples": [r["analysis_anchor_sample"], r["analysis_anchor_sample"] + ncontext],
                "effective_numeric_unit": unit,
                "unit_evidence_kind": "native_EDF_header_MNE_SI" if dataset == "PhysionetMI" else "fixed_adapter_numeric_unit_declaration_not_native_header",
                "preprocessing_qc": {"signal_contract_sha256": signal_sha, "native_unit": unit, "stored_unit": "V",
                    "si_conversion_factor": 1.0 if unit == "V" else 1e-6, "reference": "CAR_exact_clinic11",
                    "joint_rms": float(np.sqrt(np.mean(pre_cast[i] ** 2, dtype=np.float64))),
                    "normalization_applied": False, "filter_context_ends_at_native_sample": ncontext,
                    "data_after_context_accessed": False}})
        write(folder / "events.json", events); write(folder / "exclusions.json", [])
        manifest["shards"].append({"dataset": dataset, "subject_id": subject,
            "original_trials_processed": len(rows), "valid_trials": len(rows), "excluded_trials": 0,
            **{name: desc(folder / name_with_suffix) for name, name_with_suffix in
               (("X", "X.npy"), ("y", "y.npy"), ("events", "events.json"), ("exclusions", "exclusions.json"))}})
    tasks = {"W1": {"status": "COMPLETE"}, "W2": {"status": "COMPLETE", "input_contract_sha256": signal_sha}}
    cache = SimpleNamespace(root=root, n_times=128, input_contract_sha256=signal_sha)

    def seal():
        write(terms_path, terms); inventory["provider_terms_evidence"] = desc(terms_path)
        for item in inventory["files"]:
            for key in ("native_trials", "adapter_audit", "backup_receipt"): item[key] = desc(root / item[key]["path"])
        write(inventory_path, inventory); manifest["raw_inventory"] = desc(inventory_path)
        receipts, objects = [], []
        for item in inventory["files"]:
            receipt = json.loads((root / item["backup_receipt"]["path"]).read_text())
            raw = next(row for row in receipt["objects"] if row["role"] == "source_development_raw")
            raw = {**{k: raw[k] for k in ENTRY_FIELDS}, "key": raw["key"]}
            receipts.append(item["backup_receipt"])
            ident = [item["dataset"], str(item["provider_subject"]), str(item["session_id"]), item["file"]]
            objects.append({"identity": ident, "raw": raw, "backup_receipt": item["backup_receipt"]})
        write(catalog_path, {"status": "ALL_505_RAW_OBJECTS_FULL_GET_SHA256_VERIFIED", "raw_inventory": desc(inventory_path),
            "receipts": receipts, "objects": objects})
        arrays, labels, cache_rows, locations = [], [], [], []
        cache_entries = {}
        def add(path, role="manifest", dataset="Paper3"):
            cache_entries[path.relative_to(root).as_posix()] = entry(path, role, dataset)
        for shard in manifest["shards"] + manifest["participants_without_usable_trials"]:
            for key in (("X", "y", "events", "exclusions") if "X" in shard else ("exclusions",)):
                shard[key] = desc(root / shard[key]["path"]); add(root / shard[key]["path"], "result", shard["dataset"])
            if "X" in shard:
                x = np.load(root / shard["X"]["path"], mmap_mode="r", allow_pickle=False)
                events = json.loads((root / shard["events"]["path"]).read_text())
                locations.extend((len(arrays), i) for i in range(len(events)))
                arrays.append(x); labels.extend(np.load(root / shard["y"]["path"], allow_pickle=False)); cache_rows.extend(events)
        write(manifest_path, manifest)
        for path in (manifest_path, inventory_path, catalog_path, contract_path, terms_path): add(path)
        for row in terms["files"]: add(root / row["path"], "audit")
        for row in inventory["files"]: add(root / row["backup_receipt"]["path"], "audit")
        backup_path = root / "audit/cache_backup.json"
        write(backup_path, proof_receipt(list(cache_entries.values()), "paper3/software-test-double/cache/"))
        tasks["W2"].update(cache_manifest=desc(manifest_path), raw_r2_receipts_catalog=catalog_path.relative_to(root).as_posix(),
            raw_r2_receipts_catalog_sha256=file_digest(catalog_path), cache_r2_receipt=backup_path.relative_to(root).as_posix(),
            cache_r2_receipt_sha256=file_digest(backup_path))
        write(root / "status/tasks.json", tasks)
        cache.manifest_path, cache.manifest_sha256 = manifest_path.relative_to(root).as_posix(), file_digest(manifest_path)
        cache.arrays, cache.labels, cache.rows, cache.locations = arrays, np.asarray(labels), cache_rows, locations
    seal()
    return SimpleNamespace(root=root, cache=cache, manifest=manifest, inventory=inventory, terms=terms, tasks=tasks,
        seal=seal, desc=desc, by_person=by_person, contract=contract)


def test_exact_505_objects_all_people_all_original_trials_accept_without_any_local_raw_or_network(factory, monkeypatch):
    f = factory
    def forbidden(*args, **kwargs): raise AssertionError("Source acceptance attempted raw/network/GPU/fit")
    import socket
    monkeypatch.setattr(socket, "create_connection", forbidden)
    proof = acceptance.verify_source_cache_acceptance(f.cache)
    assert proof["raw_objects"] == 505 and proof["participants"] == 224
    assert proof["original_trials"] == proof["valid_trials"] == 1010
    assert proof["raw_local_required"] is proof["network_or_training_started"] is False
    assert all(not (f.root / r["path"]).exists() for r in f.inventory["files"])
    assert len(proof["artifact_sha256"]) > 2000


@pytest.mark.parametrize("violation", ["partial_inventory", "schema_status", "full_flag", "missing_terms", "normalization", "missing_person", "missing_trial", "wrong_label", "wrong_native_coordinate", "wrong_float_hash", "no_exclusions"])
def test_status_relabel_cannot_replace_full_roster_native_trial_and_byte_integrity(factory, violation):
    f = factory
    shard = f.manifest["shards"][0]
    if violation == "partial_inventory": f.inventory["files"].pop()
    elif violation == "schema_status": f.manifest["status"] = "PILOT_CACHE_PARTIAL"
    elif violation == "full_flag": f.manifest["full_source_inventory_verified"] = False
    elif violation == "missing_terms": f.terms["licences"].pop("Cho2017")
    elif violation == "normalization": f.manifest["normalization_applied"] = True
    elif violation == "missing_person": f.manifest["shards"].pop()
    elif violation == "no_exclusions": shard.pop("exclusions")
    else:
        path = f.root / shard["events"]["path"]
        rows = json.loads(path.read_text())
        if violation == "missing_trial": rows.pop()
        elif violation == "wrong_label": rows[0]["label"] = 1 - rows[0]["label"]
        elif violation == "wrong_native_coordinate": rows[0]["source_record_id"] = "other_record.mat"
        elif violation == "wrong_float_hash": rows[0]["processed_signal_sha256"] = "f" * 64
        write(path, rows)
    if violation == "no_exclusions":
        # Keep the prior sealed manifest but remove its necessary local member.
        f.manifest["shards"][0] = shard
        write(f.root / f.cache.manifest_path, f.manifest)
        f.cache.manifest_sha256 = file_digest(f.root / f.cache.manifest_path)
        f.tasks["W2"]["cache_manifest"] = f.desc(f.root / f.cache.manifest_path)
        write(f.root / "status/tasks.json", f.tasks)
    else: f.seal()
    with pytest.raises((ValueError, KeyError, TypeError)):
        acceptance.verify_source_cache_acceptance(f.cache)


@pytest.mark.parametrize("violation", ["raw_get_incomplete", "native_missing", "native_bytes", "audit_bytes", "cache_R2_subset", "cache_R2_wrong_body"])
def test_historical_full_GET_and_retained_native_proofs_are_required_without_fetching_raw(factory, violation):
    f = factory
    item = f.inventory["files"][0]
    if violation == "raw_get_incomplete":
        path = f.root / item["backup_receipt"]["path"]; receipt = json.loads(path.read_text())
        receipt["objects"][0]["get"]["complete_body_read"] = False; write(path, receipt); f.seal()
    elif violation == "native_missing": (f.root / item["native_trials"]["path"]).unlink()
    elif violation in {"native_bytes", "audit_bytes"}:
        key = "native_trials" if violation == "native_bytes" else "adapter_audit"
        with (f.root / item[key]["path"]).open("ab") as stream: stream.write(b"CHANGED_SOFTWARE_TEST_DOUBLE")
    else:
        path = f.root / f.tasks["W2"]["cache_r2_receipt"]; receipt = json.loads(path.read_text())
        if violation == "cache_R2_subset": receipt["objects"].pop()
        else: receipt["objects"][0]["get"]["sha256"] = "f" * 64
        write(path, receipt); f.tasks["W2"]["cache_r2_receipt_sha256"] = file_digest(path)
        write(f.root / "status/tasks.json", f.tasks)
    with pytest.raises((ValueError, FileNotFoundError)):
        acceptance.verify_source_cache_acceptance(f.cache)


def test_no_valid_participant_is_accounted_by_all_native_QC_exclusions(factory):
    f = factory
    shard = next(row for row in f.manifest["shards"] if row["dataset"] == "Cho2017" and row["subject_id"] == "1")
    originals = f.by_person["Cho2017", "1"]
    write(f.root / shard["exclusions"]["path"], [{"event_uid": r["trial_uid"], "reason": "insufficient_original_task_context", "original_metadata": r} for r in originals])
    f.manifest["shards"].remove(shard)
    f.manifest["participants_without_usable_trials"].append({"dataset": "Cho2017", "subject_id": "1",
        "reason": "no_valid_trials_under_frozen_contract", "excluded_original_trials": len(originals), "exclusions": shard["exclusions"]})
    f.seal()
    proof = acceptance.verify_source_cache_acceptance(f.cache)
    assert proof["participants"] == 224 and proof["excluded_trials"] == 2 and proof["valid_trials"] == 1008


@pytest.mark.parametrize("violation", ["native_context_sha_missing", "native_context_sha_invalid",
    "context_missing", "context_start", "context_stop", "event_unit", "unit_evidence", "qc_missing"])
def test_valid_source_event_requires_real_writer_context_unit_and_QC_fields(factory, violation):
    f = factory
    path = f.root / f.manifest["shards"][0]["events"]["path"]
    rows = json.loads(path.read_text()); row = rows[0]
    if violation == "native_context_sha_missing": row.pop("native_context_sha256")
    elif violation == "native_context_sha_invalid": row["native_context_sha256"] = "unverified"
    elif violation == "context_missing": row.pop("original_context_samples")
    elif violation == "context_start": row["original_context_samples"][0] -= 1
    elif violation == "context_stop": row["original_context_samples"][1] += 1
    elif violation == "event_unit": row["effective_numeric_unit"] = "V"
    elif violation == "unit_evidence": row["unit_evidence_kind"] = "native_EDF_header_MNE_SI"
    elif violation == "qc_missing": row.pop("preprocessing_qc")
    write(path, rows); f.seal()
    with pytest.raises(acceptance.SourceCacheAcceptanceError):
        acceptance.verify_source_cache_acceptance(f.cache)


@pytest.fixture
def context_QC_software_fixture():
    """Pure writer-shaped metadata/tensors; not EEG or preprocessing evidence."""
    signal_sha = "a" * 64
    original = {"dataset": "BNCI2014_001", "analysis_anchor_sample": 6000,
        "native_sfreq": 250.0, "real_trial_start_end": [5750, 7000], "task_end_sample": 7000,
        "native_unit_string": None, "numeric_unit_declared_by_fixed_adapter": "uV",
        "unit_verified_from_native_header": False, "absolute_amplitude_qc_authorized": False,
        "physical_scale_evidence": "SOFTWARE_TEST_DOUBLE_NOT_A_NATIVE_UNIT_READ"}
    pre_cast = np.random.default_rng(712).normal(scale=1e-6, size=(11, 128))
    pre_cast -= pre_cast.mean(0, keepdims=True); pre_cast -= pre_cast.mean(1, keepdims=True)
    stored = pre_cast.astype(np.float32)
    row = {"native_context_sha256": "d" * 64, "original_context_samples": [6000, 6500],
        "effective_numeric_unit": "uV", "unit_evidence_kind": "fixed_adapter_numeric_unit_declaration_not_native_header",
        "preprocessing_qc": {"signal_contract_sha256": signal_sha, "reference": "CAR_exact_clinic11",
            "normalization_applied": False, "data_after_context_accessed": False,
            "filter_context_ends_at_native_sample": 500, "native_unit": "uV", "stored_unit": "V",
            "si_conversion_factor": 1e-6, "joint_rms": float(np.sqrt(np.mean(pre_cast ** 2)))}}
    return row, original, {"context_s": [0, 2]}, signal_sha, stored


@pytest.mark.parametrize("field,value", [("signal_contract_sha256", "f" * 64),
    ("reference", "CAR_all_provider_channels"), ("normalization_applied", True),
    ("data_after_context_accessed", True), ("filter_context_ends_at_native_sample", 6500),
    ("native_unit", "V"), ("stored_unit", "uV"), ("si_conversion_factor", 1.0),
    ("joint_rms", 0), ("joint_rms", float("nan")), ("joint_rms", 1.0)])
def test_preprocessing_QC_must_bind_actual_pre_RMS_state_and_local_context(context_QC_software_fixture, field, value):
    row, original, contract, signal_sha, stored = context_QC_software_fixture
    row["preprocessing_qc"][field] = value
    with pytest.raises(acceptance.SourceCacheAcceptanceError):
        acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)


@pytest.mark.parametrize("violation", ["trial_stop", "task_stop", "anchor_before_trial", "wrong_fixed_unit",
    "header_claim_for_MAT", "missing_unit_evidence"])
def test_context_cannot_exceed_original_trial_or_invent_native_unit_evidence(context_QC_software_fixture, violation):
    row, original, contract, signal_sha, stored = context_QC_software_fixture
    if violation == "trial_stop": original["real_trial_start_end"][1] = 6499
    elif violation == "task_stop": original["task_end_sample"] = 6499
    elif violation == "anchor_before_trial": original["real_trial_start_end"][0] = 6001
    elif violation == "wrong_fixed_unit": original["numeric_unit_declared_by_fixed_adapter"] = "V"
    elif violation == "header_claim_for_MAT": original["unit_verified_from_native_header"] = True
    elif violation == "missing_unit_evidence": original.pop("physical_scale_evidence")
    with pytest.raises(acceptance.SourceCacheAcceptanceError):
        acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)


def test_Physio_MNE_outputs_volts_already_independent_of_original_EDF_scale(context_QC_software_fixture):
    row, original, contract, signal_sha, stored = context_QC_software_fixture
    original.update(dataset="PhysionetMI", native_unit_string="uV", unit_verified_from_native_header=True,
        physical_scale_to_si_volts=1e-6, data_returned_by_mne_unit="V")
    row.update(effective_numeric_unit="V", unit_evidence_kind="native_EDF_header_MNE_SI")
    row["preprocessing_qc"].update(native_unit="V", si_conversion_factor=1.0)
    acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)
    # Applying the original EDF factor a second time is an invalid receipt.
    row["preprocessing_qc"]["si_conversion_factor"] = 1e-6
    with pytest.raises(acceptance.SourceCacheAcceptanceError):
        acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)


def test_stored_float32_rounding_is_allowed_without_an_amplitude_threshold(context_QC_software_fixture):
    row, original, contract, signal_sha, stored = context_QC_software_fixture
    acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)
    # Very small finite values remain legal: only numerical consistency matters.
    stored = (stored.astype(np.float64) * 1e-25).astype(np.float32)
    row["preprocessing_qc"]["joint_rms"] = float(np.sqrt(np.mean(stored.astype(np.float64) ** 2)))
    acceptance._validate_context_and_qc(row, original, contract, signal_sha, stored)


def test_normalized_cache_with_resealed_hashes_cannot_claim_original_pre_RMS_QC(factory):
    f = factory
    shard = f.manifest["shards"][0]
    array_path, event_path = f.root / shard["X"]["path"], f.root / shard["events"]["path"]
    x = np.load(array_path, allow_pickle=False)
    x[0] = x[0] / np.sqrt(np.mean(x[0].astype(np.float64) ** 2))
    np.save(array_path, x, allow_pickle=False)
    rows = json.loads(event_path.read_text())
    rows[0]["processed_signal_sha256"] = hashlib.sha256(x[0].astype("<f4").tobytes()).hexdigest()
    write(event_path, rows); f.seal()
    with pytest.raises(acceptance.SourceCacheAcceptanceError, match="RMS"):
        acceptance.verify_source_cache_acceptance(f.cache)
