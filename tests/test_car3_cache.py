"""Targeted mathematical and fail-closed tests; synthetic fixtures are not fits."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, resample_poly, sosfiltfilt

from p3work import car3_cache as car3

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def synthetic_signal(seed=1, n=2):
    x = np.random.default_rng(seed).normal(size=(n, 11, 128)) * 1e-5
    x -= x.mean(axis=1, keepdims=True)
    x -= x.mean(axis=-1, keepdims=True)
    return x.astype(np.float32)


@pytest.fixture
def sample(tmp_path, monkeypatch):
    """Small software-only cache with explicit monkeypatched pinned identity."""
    x = synthetic_signal()
    y = np.asarray([0, 1], dtype=np.int64)
    folder = tmp_path / "cache/source_v2/BNCI2014_001/1"
    folder.mkdir(parents=True)
    np.save(folder / "X.npy", x, allow_pickle=False)
    np.save(folder / "y.npy", y, allow_pickle=False)
    events = []
    for index, signal in enumerate(x):
        row = {"dataset": "BNCI2014_001", "role": "source_train", "cohort": "source",
            "subject_id": "1", "provider_subject": "1", "source_version": "synthetic_unit_test_only",
            "session_id": "T", "source_file_sha256": "a"*64, "source_record_id": "fixture.mat::run1",
            "physical_run_id": "run1", "original_trial_index": index+1, "raw_label_code": index+1,
            "input_contract_sha256": car3.BASELINE_CONTRACT_SHA, "label": index,
            "canonical_label": ("left_hand", "right_hand")[index], "raw_sha256": "a"*64,
            "task_end_verified": True, "native_sfreq": 256., "analysis_anchor_sample": 256,
            "real_trial_start_end": [0, 1024], "original_context_samples": [256, 640], "task_end_sample": 1024,
            "effective_numeric_unit": "uV", "numeric_unit_declared_by_fixed_adapter": "uV",
            "native_unit_string": None, "unit_verified_from_native_header": False,
            "absolute_amplitude_qc_authorized": False,
            "unit_evidence_kind": "fixed_adapter_numeric_unit_declaration_not_native_header",
            "processed_signal_sha256": car3.signal_sha(signal), "selected_indices": list(range(11)),
            "processing_sha256": "b"*64,
            "preprocessing_qc": {"signal_contract_sha256": car3.BASELINE_CONTRACT_SHA,
                "normalization_applied": False, "reference": "CAR_exact_clinic11", "data_after_context_accessed": False,
                "native_unit": "uV", "stored_unit": "V", "si_conversion_factor": 1e-6,
                "filter_context_ends_at_native_sample": 384,
                "joint_rms": float(np.sqrt(np.mean(signal.astype(np.float64)**2)))}}
        row["trial_uid"] = row["event_uid"] = car3.digest({k: row[k] for k in car3.UID_KEYS})
        events.append(row)
    write_json(folder / "events.json", events)
    write_json(folder / "exclusions.json", [])
    provenance = {}
    for key, name in (("adapter_code", "p3work/adapters/source_native.py"), ("processing_code", "p3work/preprocess.py")):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic unit-test provenance only\n")
        provenance[key] = car3.descriptor(tmp_path, path)
    inventory = tmp_path / "audit/source_inventory.json"
    write_json(inventory, {"synthetic_unit_test_only": True})
    shard = {"dataset": "BNCI2014_001", "subject_id": "1", "role": "source_train", "stage": car3.STAGE11,
        "input_contract_sha256": car3.BASELINE_CONTRACT_SHA, "valid_trials": 2, "excluded_trials": 0,
        "original_trials_processed": 2, **provenance,
        **{key: car3.descriptor(tmp_path, folder / name) for key, name in
           {"X": "X.npy", "y": "y.npy", "events": "events.json", "exclusions": "exclusions.json"}.items()}}
    manifest = {"schema_version": "paper3-source-cache-v2", "status": "SOURCE_CACHE_COMPLETE", "synthetic": False,
        "old_npz_reused": False, "role": "source_train", "stage": car3.STAGE11,
        "input_contract_sha256": car3.BASELINE_CONTRACT_SHA, "full_source_inventory_verified": True,
        "exact_window_acceptance_scope": "complete_frozen_W1_verified", "normalization_applied": False,
        "training_started": False, "participants_without_usable_trials": [], "original_trials_processed": 2,
        "raw_inventory": car3.descriptor(tmp_path, inventory), "shards": [shard]}
    write_json(tmp_path / car3.BASELINE_MANIFEST, manifest)
    monkeypatch.setattr(car3, "BASELINE_SHA", car3.file_sha(tmp_path / car3.BASELINE_MANIFEST))
    monkeypatch.setattr(car3, "SOURCE_COUNTS", {"BNCI2014_001": (1, 2)})
    write_json(tmp_path / "contracts/input_contract.json", json.loads((ROOT / "contracts/input_contract.json").read_text()))
    contract = json.loads((ROOT / car3.CONTRACT_PATH).read_text())
    contract["baseline_source_manifest_sha256"] = car3.BASELINE_SHA
    contract["source_datasets"] = ["BNCI2014_001"]
    write_json(tmp_path / car3.CONTRACT_PATH, contract)
    for name in ("p3work/car3_cache.py", "tools/prepare_car3_cache.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    return tmp_path, manifest, events, x, y


def test_projection_reference_invariance_and_P3SP11_identity():
    rng = np.random.default_rng(472)
    x = rng.normal(size=(11, 128))
    reference = rng.normal(size=(1, 128))
    np.testing.assert_allclose(car3.project_car3(x + reference), car3.project_car3(x), atol=2e-15, rtol=2e-14)
    p11x = x - x.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(car3.project_car3(p11x), car3.project_car3(x), atol=1e-15, rtol=2e-14)


def test_identical_linear_temporal_pipeline_commutes_before_crop():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(11, 384))
    sos = butter(4, [8, 30], btype="bandpass", fs=256, output="sos")
    def temporal(a):
        a = resample_poly(sosfiltfilt(sos, a, axis=-1), 1, 2, axis=-1, padtype="line")[:, 32:160]
        return a - a.mean(axis=-1, keepdims=True)
    def spatial(a):
        a = a[list(car3.SELECT), :].copy()
        return a - a.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(spatial(temporal(x)), temporal(spatial(x)), atol=2e-14, rtol=4e-12)


def test_float32_equivalence_has_explicit_rounding_bound():
    original = np.random.default_rng(29).normal(size=(11, 128)) * 1e-5
    original -= original.mean(axis=0, keepdims=True)
    original -= original.mean(axis=-1, keepdims=True)
    inherited = original.astype(np.float32)
    exact = car3.project_car3(original)
    derived = car3.project_car3(inherited).astype(np.float32)
    assert np.max(np.abs(derived.astype(np.float64)-exact)) < car3.rounding_bound(original)
    assert not np.array_equal(exact, derived)


def test_plan_writes_nothing(sample):
    root, _, _, _, _ = sample
    before = {str(p): car3.file_sha(p) for p in root.rglob("*") if p.is_file()}
    result = car3.prepare(root)
    assert result["status"] == "LOCAL_VALIDATION_ONLY_PASSED"
    assert not result["training_ready"] and result["actual_fits"] == 0
    assert not (root / car3.OUTPUT).exists() and not (root / car3.AUDIT).exists()
    assert before == {str(p): car3.file_sha(p) for p in root.rglob("*") if p.is_file()}


def test_execute_preserves_UID_labels_and_input_bytes(sample):
    root, _, events, x, y = sample
    before = {str(p): car3.file_sha(p) for p in (root / "cache/source_v2").rglob("*") if p.is_file()}
    receipt = car3.prepare(root, execute=True)
    assert not receipt["training_ready"] and receipt["actual_fits"] == 0
    assert not receipt["primary_11channel_compatible"] and receipt["new_independent_participants"] == 0
    manifest = car3.read_json(root / car3.OUTPUT / "source_cache3.json")
    shard = manifest["shards"][0]
    np.testing.assert_array_equal(np.load(root / shard["X"]["path"]), car3.project_car3(x).astype(np.float32))
    np.testing.assert_array_equal(np.load(root / shard["y"]["path"]), y)
    derived_events = car3.read_json(root / shard["events"]["path"])
    assert [e["event_uid"] for e in derived_events] == [e["event_uid"] for e in events]
    assert all(e["role"] == car3.ROLE and e["parent_preprocessing_qc"]["normalization_applied"] is False for e in derived_events)
    assert before == {str(p): car3.file_sha(p) for p in (root / "cache/source_v2").rglob("*") if p.is_file()}
    backup = car3.read_json(root / receipt["backup_manifest"]["path"])
    assert all(e["role"] in {"result", "archive"} for e in backup["files"])
    for e in backup["files"]:
        assert car3.file_sha(root / e["path"]) == e["sha256"]
    with pytest.raises(car3.Car3PreparationError, match="existing_output"):
        car3.prepare(root, execute=True)


@pytest.mark.parametrize("relative", ["../secret", "/tmp/a", "cache/../a", "cache//a", "cache/./a",
    "cache\\a", "cache/a.npz", "legacy_baseline_v1/a.npy", ".env", "cache/secrets/a", "cache/a\n"])
def test_unsafe_paths_rejected(tmp_path, relative):
    with pytest.raises(car3.Car3PreparationError, match="unsafe"):
        car3.local(tmp_path, relative, exists=False)


def test_inside_root_symlink_rejected(tmp_path):
    (tmp_path / "original").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "original", target_is_directory=True)
    with pytest.raises(car3.Car3PreparationError, match="symlink"):
        car3.local(tmp_path, "link/file.npy", exists=False)


@pytest.mark.parametrize("artifact", ["X", "y", "events", "exclusions"])
def test_input_hash_tamper_refused(sample, artifact):
    root, manifest, _, _, _ = sample
    path = root / manifest["shards"][0][artifact]["path"]
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(car3.Car3PreparationError, match="SHA"):
        car3.prepare(root, execute=True)
    assert not (root / car3.OUTPUT).exists()


@pytest.mark.parametrize("change", ["RMS", "unit", "window", "UID", "label", "external", "signal", "nonfinite"])
def test_row_scientific_and_permission_gates(sample, change):
    _, manifest, events, x, _ = sample
    row, signal = copy.deepcopy(events[0]), x[0].copy()
    if change == "RMS": row["preprocessing_qc"]["normalization_applied"] = True
    if change == "unit": row["preprocessing_qc"]["stored_unit"] = "unknown"
    if change == "window": row["original_context_samples"] = [256, 768]
    if change == "UID": row["event_uid"] = "c"*64
    if change == "label": row["label"] = 1
    if change == "external": row["role"] = "sealed_query"
    if change == "signal": signal[0, 0] += 1e-7
    if change == "nonfinite": signal[0, 0] = np.nan
    with pytest.raises(car3.Car3PreparationError):
        car3._validate_row(row, signal, 0, "BNCI2014_001", "1", set())


@pytest.mark.parametrize("field,value", [("training_ready", True), ("BCIIV2b_accepted", True),
    ("native_reference_requirement", "bipolar"), ("context_s", [0, 2]), ("crop_s", [.5, 1.5]),
    ("channels", ["C3", "C4"]), ("primary_11channel_compatible", True)])
def test_contract_cannot_promote_training_or_bipolar(field, value):
    contract = car3.read_json(ROOT / car3.CONTRACT_PATH)
    contract[field] = value
    with pytest.raises(car3.Car3PreparationError, match="contract"):
        car3.validate_contract(contract)


def test_geometry_dimensions_are_separate_and_r4_rejected():
    contract = car3.read_json(ROOT / car3.CONTRACT_PATH)
    assert car3.validate_contract(contract)
    contract["geometry_constraints"]["original_r4_or_r8_config_accepted"] = True
    with pytest.raises(car3.Car3PreparationError, match="geometry"):
        car3.validate_contract(contract)


@pytest.mark.parametrize("shape", [(3, 128), (11, 256), (1, 11, 128, 1)])
def test_projection_shape_cannot_reinterpret_existing_input(shape):
    with pytest.raises(car3.Car3PreparationError):
        car3.project_car3(np.ones(shape))
