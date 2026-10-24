"""Optional original-MAT preparation/order contracts; synthetic numerical tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper3r5.external import (CHANNELS, INDICES, ORDER_SCOPE, ExternalCatalog,
                              native_schema, prepare, prepare_context, validate_contract)
from paper3r5.io import digest, put, sha


def contract():
    return {"channels": CHANNELS, "context_s": [0, 1.5], "crop_s": [.25, 1.25],
            "n_times": 128, "sfreq_hz": 128, "band_hz": [8, 30], "reference": "CAR_exact_clinic11"}


def catalog_fixture(tmp_path, *, invalid_index=1):
    root, cache = tmp_path / "project", tmp_path / "new-data"
    put(root / "contracts/input_contract.json", contract())
    rng = np.random.default_rng(174)
    x = rng.normal(size=(50, 11, 128)) * 1e-6
    x -= x.mean(1, keepdims=True); x -= x.mean(2, keepdims=True)
    x[invalid_index] = 0
    cache.mkdir()
    np.save(cache / "X.npy", x.astype(np.float32), allow_pickle=False)
    rows = [{"event_uid": f"w-{i}", "trial_uid": f"w-{i}", "label": i % 2,
             "cache_index": i, "original_trial_index": i * 3 + 2,
             "valid": i != invalid_index, "physical_run_id": None} for i in range(50)]
    put(cache / "events.json", rows)
    manifest = {"status": "WEIBO_EXTERNAL_CACHE_COMPLETE", "order_scope": ORDER_SCOPE,
                "signal_contract_sha256": digest(contract()), "temporal_risk_eligible": False,
                "query_policy": "fixed_valid_LR_positions_after_B20_for_all_shorter_budgets",
                "sessions": [{"participant_id": "W01", "session_id": "0",
                    "X": {"path": "X.npy", "sha256": sha(cache / "X.npy")},
                    "events": {"path": "events.json", "sha256": sha(cache / "events.json")}}]}
    put(cache / "manifest.json", manifest)
    return root, cache, rows, manifest


def test_context_is_bounded_and_pre_rms_scaling_is_preserved():
    rng = np.random.default_rng(713)
    signal = rng.normal(size=(64, 1600)) * 15
    expected, qc = prepare_context(signal, contract())
    changed = signal.copy()
    changed[:, :600] = np.nan
    changed[:, 900:] = np.inf
    actual, _ = prepare_context(changed, contract())
    np.testing.assert_array_equal(actual, expected)
    scaled, _ = prepare_context(signal * 3, contract())
    np.testing.assert_allclose(scaled, 3 * expected, atol=1e-19, rtol=1e-12)
    assert expected.shape == (11, 128)
    np.testing.assert_allclose(expected.mean(0), 0, atol=1e-20)
    np.testing.assert_allclose(expected.mean(1), 0, atol=1e-20)
    assert qc["normalization_applied"] is False and qc["data_after_context_accessed"] is False
    assert qc["filter_context_ends_at_native_sample"] == 900
    assert 0 < np.sqrt(np.mean(expected ** 2)) < .001


def test_bad_selected_context_is_rejected_but_unselected_channels_are_irrelevant():
    rng = np.random.default_rng(124)
    trial = rng.normal(size=(64, 1600))
    baseline, _ = prepare_context(trial, contract())
    trial[0] = np.nan  # Fp1 is outside the fixed measured clinic11.
    np.testing.assert_array_equal(prepare_context(trial, contract())[0], baseline)
    trial[INDICES[0], 601] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare_context(trial, contract())
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        prepare_context(np.zeros((64, 1600)), contract())


def test_native_schema_rejects_axis_and_class_guessing():
    labels = np.tile([1, 2, 3, 4, 5, 6, 7], 15)[:, None]
    data = np.zeros((64, 1600, len(labels)), dtype=np.float64)
    schema = native_schema(data, labels)
    assert schema["sampling_rate_hz"] == 200 and schema["mi_onset_trial_sample"] == 600
    assert schema["acquisition_order_verified"] is False
    assert schema["unit_independently_calibrated_from_MAT"] is False
    with pytest.raises(ValueError, match="64 x 1600"):
        native_schema(data.transpose(2, 0, 1), labels)
    bad = labels.astype(float); bad[0] = 1.5
    with pytest.raises(ValueError, match="integer"):
        native_schema(data, bad)


def test_paid_invalid_prefix_and_fixed_b20_query_for_all_budgets(tmp_path):
    root, cache, rows, _ = catalog_fixture(tmp_path)
    catalog = ExternalCatalog(root, cache / "manifest.json")
    plan = catalog.plans[0]
    four = catalog.materialize(plan, "cpu", budget=4)
    twenty = catalog.materialize(plan, "cpu", budget=20)
    assert four.support.acquired_event_uids == ("w-0", "w-1", "w-2", "w-3")
    assert four.support.event_uids == ("w-0", "w-2", "w-3")
    assert len(twenty.support.acquired_event_uids) == 20 and len(twenty.support.y) == 19
    assert four.query_ids == twenty.query_ids == tuple(f"w-{i}" for i in range(20, 50))
    assert not set(twenty.support.acquired_event_uids) & set(twenty.query_ids)
    torch.testing.assert_close(four.query_x, twenty.query_x)
    assert set(twenty.run_ids) == {"unknown"} and twenty.temporal_risk_eligible is False
    assert plan.audit["acquisition_order_verified"] is False and catalog.people == ["W01"]
    assert catalog.identity["signal_contract_sha256"] == digest(contract())


def test_cache_hash_contract_and_path_tampering_fail_closed(tmp_path):
    root, cache, _, manifest = catalog_fixture(tmp_path)
    changed_contract = contract(); changed_contract["crop_s"] = [.5, 1.5]
    put(root / "contracts/input_contract.json", changed_contract)
    with pytest.raises(ValueError, match="exact frozen"):
        ExternalCatalog(root, cache / "manifest.json")
    put(root / "contracts/input_contract.json", contract())
    (cache / "events.json").write_text("[]")
    with pytest.raises(ValueError, match="path/hash"):
        ExternalCatalog(root, cache / "manifest.json")
    manifest["sessions"][0]["events"] = {"path": "../outside.json", "sha256": "a" * 64}
    put(cache / "manifest.json", manifest)
    with pytest.raises(ValueError, match="path/hash"):
        ExternalCatalog(root, cache / "manifest.json")


def test_missing_optional_sources_emit_skip_manifest_without_network(tmp_path, monkeypatch):
    root = tmp_path / "project"
    put(root / "contracts/input_contract.json", contract())
    source = tmp_path / "source.json"
    put(source, {"dataset_id": "Weibo2014", "mirror_version": "v1.0.3",
                 "files": [{"path": "sourcedata/subject_1.mat", "size": 1, "checksum": "a" * 64}]})
    def no_network(*_, **__):
        raise AssertionError("Unexpected network in local-only optional preparation")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    result = prepare(root, tmp_path / "new-data", source, [1], tmp_path / "raw", download=False)
    assert result["status"] == "SKIPPED_OPTIONAL" and not result["sessions"]
    assert result["failures"][0]["status"] == "SKIPPED_OPTIONAL"


def test_sequential_preparation_commits_people_preserves_existing_raw_and_stops_network_failure(tmp_path, monkeypatch):
    root, output, raw = tmp_path / "project", tmp_path / "new-data", tmp_path / "raw"
    put(root / "contracts/input_contract.json", contract())
    raw.mkdir()
    payload = b"public"
    (raw / "subject_2.mat").write_bytes(payload)
    source = tmp_path / "source.json"
    put(source, {"dataset_id": "Weibo2014", "mirror_version": "v1.0.3", "files": [
        {"path": f"sourcedata/subject_{i}.mat", "size": len(payload), "checksum": "a" * 64}
        for i in range(1, 5)]})
    calls = []
    def download(entry, raw_dir):
        subject = int(Path(entry["path"]).stem.split("_")[1]); calls.append(subject)
        if subject == 3:
            raise TimeoutError("simulated optional network timeout")
        path = Path(raw_dir) / Path(entry["path"]).name
        path.write_bytes(payload)
        return path
    def preprocess(*_):
        x = np.full((42, 11, 128), 1e-6, dtype=np.float32)
        rows = [{"event_uid": f"trial-{i}", "label": i % 2, "valid": True} for i in range(42)]
        return x, rows, {"valid_lr_trials": 42}
    monkeypatch.setattr("paper3r5.external.download_source", download)
    monkeypatch.setattr("paper3r5.external.preprocess_weibo", preprocess)
    result = prepare(root, output, source, [1, 2, 3, 4], raw,
                     download=True, delete_raw_after_prepare=True)
    assert calls == [1, 2, 3]  # Do not fetch later people after a network failure.
    assert result["prepared_subjects"] == ["W01", "W02"]
    assert not (raw / "subject_1.mat").exists()
    assert (raw / "subject_2.mat").read_bytes() == payload
    assert not result["all_requested_subjects_prepared"]
    assert json.loads((output / "manifest.json").read_text())["prepared_subjects"] == ["W01", "W02"]
    # A later invocation adds a person without discarding already committed ones.
    result = prepare(root, output, source, [4], raw, download=True, delete_raw_after_prepare=True)
    assert result["prepared_subjects"] == ["W01", "W02", "W04"]
    assert not (raw / "subject_4.mat").exists()
