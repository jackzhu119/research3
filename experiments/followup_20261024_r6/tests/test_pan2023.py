"""Pan2023 multi-session/paid-prefix and bounded original preprocessing tests."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper3r6.pan2023 import (CHANNELS, INDICES, ORDER_SCOPE, PanCatalog,
                             native_schema, prepare_context)
from paper3r6.io import digest, put, sha


def contract():
    return {"channels": CHANNELS, "context_s": [0, 1.5], "crop_s": [.25, 1.25],
            "n_times": 128, "sfreq_hz": 128, "band_hz": [8, 30], "reference": "CAR_exact_clinic11"}


def fixture(tmp_path):
    root, cache = tmp_path / "project", tmp_path / "pan"
    put(root / "contracts/input_contract.json", contract())
    rng = np.random.default_rng(231)
    sessions = []
    for person, session in [("PAN01", "0"), ("PAN01", "1"), ("PAN02", "0")]:
        d = cache / person / session; d.mkdir(parents=True)
        x = rng.normal(size=(120, 11, 128)).astype(np.float32) * 1e-6
        x[1] = 0
        np.save(d / "X.npy", x, allow_pickle=False)
        rows = [{"event_uid": f"{person}-{session}-{i}", "trial_uid": f"{person}-{session}-{i}",
                 "label": i % 2, "cache_index": i, "original_trial_index": i,
                 "valid": i != 1, "physical_run_id": None} for i in range(120)]
        put(d / "events.json", rows)
        def a(name):return {"path": str((d / name).relative_to(cache)), "sha256": sha(d / name), "bytes": (d / name).stat().st_size}
        sessions.append({"participant_id": person, "session_id": session,
                         "X": a("X.npy"), "events": a("events.json")})
    manifest = {"dataset_id": "Pan2023", "status": "PAN2023_EXTERNAL_CACHE_COMPLETE",
                "order_scope": ORDER_SCOPE, "signal_contract_sha256": digest(contract()),
                "temporal_risk_eligible": False,
                "query_policy": "fixed_valid_original_session_trial_positions_after_B20_for_all_budgets",
                "sessions": sessions}
    put(cache / "manifest.json", manifest)
    return root, cache, manifest


def test_pan_context_reads_only_authorized_native_window_and_scales_linearly():
    rng = np.random.default_rng(172)
    trial = rng.normal(size=(28, 1750)) * 12
    expected, qc = prepare_context(trial, contract())
    changed = trial.copy(); changed[:, :750] = np.nan; changed[:, 1125:] = np.inf
    np.testing.assert_array_equal(prepare_context(changed, contract())[0], expected)
    scaled = prepare_context(trial * 3, contract())[0]
    np.testing.assert_allclose(scaled, expected * 3, rtol=1e-12, atol=1e-19)
    assert expected.shape == (11, 128)
    np.testing.assert_allclose(expected.mean(0), 0, atol=1e-18)
    np.testing.assert_allclose(expected.mean(1), 0, atol=1e-18)
    assert qc["normalization_applied"] is False
    assert qc["filter_context_ends_at_native_sample"] == 1125


def test_pan_selected_channel_qc_does_not_touch_unselected_electrodes():
    trial = np.random.default_rng(264).normal(size=(28, 1750))
    expected = prepare_context(trial, contract())[0]
    trial[0] = np.nan  # FC5 not in clinic11.
    np.testing.assert_array_equal(prepare_context(trial, contract())[0], expected)
    trial[INDICES[0], 760] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        prepare_context(trial, contract())
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        prepare_context(np.zeros((28, 1750)), contract())


def test_pan_native_layout_never_guesses_axes_or_labels():
    x = np.zeros((120, 1750, 28), dtype=np.float64)
    y = np.tile([1., 2.], 60)[None]
    v = native_schema(x, y, [[250.]])
    assert v["class_counts"] == {"1": 60, "2": 60}
    assert v["physical_run_id_available"] is False and v["mi_onset_trial_sample"] == 750
    for a, b, fs in [(x.transpose(2, 0, 1), y, [[250.]]), (x, y - 1, [[250.]]),
                      (x, y, [[200.]]), (x.astype(np.float32), y, [[250.]])]:
        with pytest.raises(ValueError):native_schema(a, b, fs)


def test_pan_keeps_two_real_sessions_without_counting_them_as_people(tmp_path):
    root, cache, _ = fixture(tmp_path)
    c = PanCatalog(root, cache / "manifest.json")
    assert c.people == ["PAN01", "PAN02"]
    assert len(c.plans) == 3 and len(c.by_person["PAN01"]) == 2
    assert [p.session_id for p in c.by_person["PAN01"]] == ["0", "1"]
    assert c.identity["dataset_id"] == "Pan2023"


def test_pan_short_budget_never_replaces_invalid_paid_position_and_keeps_query(tmp_path):
    root, cache, _ = fixture(tmp_path)
    c = PanCatalog(root, cache / "manifest.json");p = c.plans[0]
    query = None
    for b in (4, 8, 12, 16, 20):
        v = c.materialize(p, "cpu", budget=b)
        assert len(v.support.acquired_event_uids) == b
        assert len(v.support.event_uids) == b - 1
        assert v.support.x.shape == (b - 1, 11, 128)
        assert v.query_x.shape == (100, 11, 128)
        assert all(r == "unknown" for r in v.run_ids)
        assert v.temporal_risk_eligible is False
        if query is None:query = v.query_ids
        assert v.query_ids == query
        assert not set(v.support.acquired_event_uids) & set(v.query_ids)


def test_pan_duplicate_session_is_error_but_shared_person_is_expected(tmp_path):
    root, cache, m = fixture(tmp_path)
    m["sessions"].append(m["sessions"][0]);put(cache / "manifest.json", m)
    with pytest.raises(ValueError, match="Duplicate Pan2023 participant/session"):
        PanCatalog(root, cache / "manifest.json")


def test_pan_unknown_blocks_and_strict_source_contract_are_preserved(tmp_path):
    root, cache, m = fixture(tmp_path)
    rows_path = cache / m["sessions"][0]["events"]["path"]
    import json
    rows = json.loads(rows_path.read_text());rows[0]["physical_run_id"] = "invented-block-0"
    put(rows_path, rows);m["sessions"][0]["events"]["sha256"] = sha(rows_path)
    m["sessions"][0]["events"]["bytes"] = rows_path.stat().st_size
    put(cache / "manifest.json", m)
    with pytest.raises(ValueError, match="unknown-block"):
        PanCatalog(root, cache / "manifest.json")


def test_pan_cache_artifact_tampering_is_visible(tmp_path):
    root, cache, m = fixture(tmp_path)
    (cache / m["sessions"][0]["X"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="path/hash"):
        PanCatalog(root, cache / "manifest.json")


def test_pan_runtime_does_not_hash_signal_again_and_validates_only_first_cached_load(tmp_path, monkeypatch):
    import paper3r6.pan2023 as pan
    root, cache, _ = fixture(tmp_path)
    original_sha = pan.sha
    calls = []
    def tracked(path):
        calls.append(str(path));return original_sha(path)
    monkeypatch.setattr(pan, "sha", tracked)
    c = PanCatalog(root, cache / "manifest.json")
    assert not any(path.endswith("X.npy") for path in calls)
    checks = []
    original_finite = pan.np.isfinite
    def finite(x):
        checks.append(1);return original_finite(x)
    monkeypatch.setattr(pan.np, "isfinite", finite)
    first = c.array(c.plans[0]);second = c.array(c.plans[0])
    assert first is second and len(checks) == 1
