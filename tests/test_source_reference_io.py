"""Synthetic CPU reference-I/O SOFTWARE fixtures, not real EEG/reference receipts.

The cache and f0 readers are explicitly mocked where artifact math/I/O is under
test. This does not complete W1/W2/W4 or substantiate any real source fit.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from p3work.adapt import FrozenSourceModel
from p3work.contracts import CHANNELS, SOURCE_IDS, file_digest, validate_signal_contract
from p3work.f0_bridge import load_verified_source as actual_source_loader
from p3work import source_reference_io as reference_io
from p3work.source_reference_io import load_verified_reference, prepare_source_reference
from p3work.spatial_geometry import helmert_rows, tensor_sha256
from tools import run_controlled_pilot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


@pytest.fixture
def software_fixture(tmp_path, monkeypatch):
    root = tmp_path
    contract = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
                "selected_profile": "long", "context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256,
                "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
                "raw_metadata_manifest_sha256": "b" * 64,
                "long": {"participant_mean_coverage": .9, "participant_mean_abs_class_exclusion_gap": .02},
                "short": {"participant_mean_coverage": .95, "participant_mean_abs_class_exclusion_gap": .01}}}
    write(root / "contracts/input_contract.json", contract)
    signal_sha = validate_signal_contract(contract)
    q = helmert_rows()
    arrays, rows, locations, allowed = [], [], [], []
    for dataset in sorted(SOURCE_IDS):
        values = []
        for subject, trial, direction, keep in (("personA", 0, 0, True), ("personA", 1, 2, False), ("personB", 0, 1, True)):
            z = torch.zeros(10, 256, dtype=torch.float64)
            z[direction, :2] = torch.tensor([1., -1.], dtype=torch.float64)
            values.append((q.T @ z).numpy())
            uid = f"SOFTWARE_FIXTURE_ONLY/{dataset}/{subject}/{trial}"
            rows.append({"event_uid": uid, "dataset": dataset, "subject_id": subject, "role": "source_train"})
            locations.append((len(arrays), len(values) - 1))
            if keep:
                allowed.append(uid)
        arrays.append(np.stack(values))
    manifest_path = root / "manifests/source_cache_fixture.json"
    write(manifest_path, {"scope": "SOFTWARE_TEST_DOUBLE_NOT_SOURCE_CACHE"})
    cache = SimpleNamespace(input_contract_sha256=signal_sha, manifest_path=manifest_path.relative_to(root).as_posix(),
                            manifest_sha256=file_digest(manifest_path), arrays=arrays, rows=rows, locations=locations)
    uid_path = root / "fixtures/training_uid_allowlist.json"
    training = {"schema_version": "paper3-source-training-uids-v2", "role": "source_pretrain_train",
                "synthetic": False, "signal_contract_sha256": signal_sha, "event_uids": allowed,
                "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_REAL_TRAINING"}
    write(uid_path, training)
    source_path = root / "fixtures/f0_receipt.json"
    source_receipt = {"status": "SOFTWARE_TEST_DOUBLE_NOT_SOURCE_F0_COMPLETE",
                      "training_uid_manifest": descriptor(root, uid_path),
                      "source_cache_manifest_sha256": cache.manifest_sha256}
    write(source_path, source_receipt)
    source = FrozenSourceModel(nn.Sequential(nn.Flatten(), nn.Linear(11 * 256, 3)), nn.Linear(3, 2), feature_dim=3)
    calls = []
    def verify(_root):
        calls.append("cache_gate")
        return cache
    def mock_source(_root, relative, _contract):
        calls.append("f0_gate_test_double")
        torch.rand(7)  # Caller must preserve RNG even when bridge initializes CPU modules.
        return source, json.loads((root / relative).read_text())
    monkeypatch.setattr(run_controlled_pilot, "verify_inputs", verify)
    monkeypatch.setattr(reference_io, "load_verified_source", mock_source)
    def reseal():
        write(uid_path, training)
        source_receipt["training_uid_manifest"] = descriptor(root, uid_path)
        write(source_path, source_receipt)
    return SimpleNamespace(root=root, contract=contract, cache=cache, training=training, source_receipt=source_receipt,
                           source_path=source_path.relative_to(root).as_posix(), uid_path=uid_path,
                           calls=calls, reseal=reseal)


def produce(f):
    return prepare_source_reference(f.root, f.source_path, f.contract, execute=True, output="software_fixture_reference")


def load(f):
    return load_verified_reference(f.root, "software_fixture_reference/receipt.json", f.source_path, f.contract)


def test_default_dry_run_runs_real_gate_interfaces_without_computation_writes_or_cuda(software_fixture, monkeypatch):
    f = software_fixture
    def forbidden(*args, **kwargs):
        pytest.fail("A dry-run software reference fixture attempted covariance computation or CUDA")
    monkeypatch.setattr(reference_io, "source_reference", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    before = torch.get_rng_state().clone()
    result = prepare_source_reference(f.root, f.source_path, f.contract)
    assert result["status"] == "PLANNED_SOURCE_REFERENCE" and result["execute"] is False
    assert result["training_events"] == 8
    assert f.calls == ["cache_gate", "f0_gate_test_double"]
    assert torch.equal(before, torch.get_rng_state())
    assert not (f.root / "runs").exists()


def test_synthetic_io_roundtrip_uses_only_f0_training_allowlist_with_hierarchical_weighting(software_fixture):
    f = software_fixture
    result = produce(f)
    reference, receipt = load(f)
    expected = torch.eye(10, dtype=torch.float64) * .005
    expected[0, 0] += .475
    expected[1, 1] += .475
    # Extra cache trial along coordinate 2 must NOT enter f0's training reference.
    torch.testing.assert_close(reference.matrix, expected, atol=1e-14, rtol=0)
    assert len(reference.event_uids) == 8
    assert set(reference.event_uids) == set(f.training["event_uids"])
    assert receipt["source_f0_receipt_sha256"] == file_digest(f.root / f.source_path)
    assert receipt["source_training_uid_manifest_sha256"] == file_digest(f.uid_path)
    assert receipt["training_uid_set_sha256"] == reference.events_sha256
    assert receipt["training_dataset_ids"] == sorted(SOURCE_IDS)
    assert receipt["weighting"] == "dataset_equal/subject_equal/trial_equal" and receipt["shrinkage"] == .05
    torch.testing.assert_close(reference.sqrt @ reference.invsqrt, torch.eye(10, dtype=torch.float64), atol=1e-13, rtol=0)
    assert result["training_started"] is False and result["gpu_execution"] is False
    assert len(f.calls) == 4  # Full source gates are rerun for loading.


def test_a_pure_receipt_claim_does_not_replace_actual_verified_source_loader(software_fixture, monkeypatch):
    f = software_fixture
    monkeypatch.setattr(reference_io, "load_verified_source", actual_source_loader)
    with pytest.raises(ValueError, match="verified real four-source"):
        produce(f)
    assert not (f.root / "software_fixture_reference").exists()


@pytest.mark.parametrize("violation", ["duplicate_uid", "unknown_uid", "wrong_role", "missing_dataset", "synthetic", "protocol", "validation_event"])
def test_training_uid_or_source_role_violations_stop_before_reference_output(software_fixture, violation):
    f = software_fixture
    if violation == "duplicate_uid":
        f.training["event_uids"].append(f.training["event_uids"][0])
    elif violation == "unknown_uid":
        f.training["event_uids"].append("META_OR_EXTERNAL_NOT_IN_SOURCE_CACHE")
    elif violation == "wrong_role":
        f.training["role"] = "meta_train"
    elif violation == "missing_dataset":
        f.training["event_uids"] = [u for u in f.training["event_uids"] if "PhysionetMI" not in u]
    elif violation == "synthetic":
        f.training["synthetic"] = True
    elif violation == "protocol":
        f.training["signal_contract_sha256"] = "e" * 64
    else:
        f.cache.rows[0]["role"] = "source_validation"
    f.reseal()
    with pytest.raises((ValueError, PermissionError)):
        produce(f)
    assert not (f.root / "software_fixture_reference").exists()


@pytest.mark.parametrize("target", ["reference", "Q", "training_event_uids"])
def test_artifact_byte_changes_are_rejected(software_fixture, target):
    f = software_fixture
    receipt = produce(f)
    item = receipt["training_event_uids"] if target == "training_event_uids" else receipt["arrays"][target]
    with (f.root / item["path"]).open("ab") as stream:
        stream.write(b"SOFTWARE_FIXTURE_CHANGED_BYTES")
    with pytest.raises(ValueError, match="integrity mismatch"):
        load(f)


@pytest.mark.parametrize("target", ["reference", "reference_sqrt", "Q"])
def test_resealed_but_mathematically_invalid_array_is_rejected(software_fixture, target):
    f = software_fixture
    receipt = produce(f)
    item = receipt["arrays"][target]
    p = f.root / item["path"]
    values = np.load(p, allow_pickle=False)
    values *= 2
    np.save(p, values, allow_pickle=False)
    receipt["arrays"][target] = {**descriptor(f.root, p), "array_sha256": tensor_sha256(torch.from_numpy(values))}
    write(f.root / "software_fixture_reference/receipt.json", receipt)
    with pytest.raises(ValueError, match="Helmert|SPD-root"):
        load(f)


def test_f0_receipt_or_training_event_set_change_invalidates_saved_reference(software_fixture):
    f = software_fixture
    produce(f)
    f.source_receipt["changed_after_reference"] = True
    f.reseal()
    with pytest.raises(ValueError, match="input/f0/UID-bound"):
        load(f)


def test_existing_output_is_never_overwritten(software_fixture):
    f = software_fixture
    produce(f)
    before = (f.root / "software_fixture_reference/receipt.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        produce(f)
    assert (f.root / "software_fixture_reference/receipt.json").read_bytes() == before
