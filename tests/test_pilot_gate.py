"""CPU software fixtures test only gates/identities; NEVER fit a model or use CUDA.

The synthetic arrays below exercise cache schema validation. Their temporary
declarations are test doubles and are not actual source receipts or EEG caches.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p3work.contracts import CHANNELS, SOURCE_IDS, file_digest, validate_signal_contract

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper3_controlled_pilot", ROOT / "tools/run_controlled_pilot.py")
pilot = importlib.util.module_from_spec(spec)
# dataclass resolves its defining module during module execution.
import sys
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)
PRODUCTION_UPSTREAM_VERIFIER = pilot._verify_upstream_acceptance


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def gate_fixture(tmp_path, monkeypatch):
    # These tiny arrays test the cache/runner component, not real W1/W2.
    # Full upstream metadata verifiers have their own non-mocked gate tests.
    monkeypatch.setattr(pilot, "_verify_upstream_acceptance", lambda cache: None)
    root = tmp_path
    contract = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
                "selected_profile": "long", "context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256,
                "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
                  "raw_metadata_manifest_sha256": "b" * 64,
                  "long": {"participant_mean_coverage": .9, "participant_mean_abs_class_exclusion_gap": .02},
                  "short": {"participant_mean_coverage": .95, "participant_mean_abs_class_exclusion_gap": .01}}}
    write(root / "contracts/input_contract.json", contract)
    signal_sha = validate_signal_contract(contract)
    for relative in ("p3work/adapters/source_native.py", "p3work/preprocess.py"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    descriptor = lambda relative: {"path": relative, "sha256": file_digest(root / relative)}
    adapter = descriptor("p3work/adapters/source_native.py")
    processing = descriptor("p3work/preprocess.py")
    manifest = {"schema_version": "paper3-source-cache-v2", "status": "SOURCE_CACHE_COMPLETE",
                "synthetic": False, "old_npz_reused": False, "role": "source_train",
                "input_contract_sha256": signal_sha, "shards": []}
    cache_dir = root / "cache"
    cache_dir.mkdir()
    rng = np.random.default_rng(20261018)
    for dataset in sorted(SOURCE_IDS):
        x = rng.normal(size=(2, 11, 256))
        x -= x.mean(1, keepdims=True)
        x -= x.mean(2, keepdims=True)
        np.save(cache_dir / (dataset + "_X.npy"), x, allow_pickle=False)
        np.save(cache_dir / (dataset + "_y.npy"), np.array([0, 1], dtype=np.int64), allow_pickle=False)
        rows = [{"event_uid": f"TEST_DOUBLE_ONLY/{dataset}/{i}", "dataset": dataset,
                 "subject_id": "TEST_DOUBLE_ONLY", "source_record_id": dataset + ":TEST_DOUBLE_ONLY", "original_trial_id": i, "label": i,
                 "role": "source_train", "input_contract_sha256": signal_sha, "raw_sha256": "c" * 64,
                 "adapter_sha256": adapter["sha256"], "processing_sha256": processing["sha256"]}
                for i in (0, 1)]
        write(cache_dir / (dataset + "_events.json"), rows)
        manifest["shards"].append({"dataset": dataset, "role": "source_train", "input_contract_sha256": signal_sha,
              "stage": "pre_rms_car11_temporal_demeaned", "adapter_code": adapter, "processing_code": processing,
              **{key: descriptor(f"cache/{dataset}_{suffix}") for key, suffix in
                 (("X", "X.npy"), ("y", "y.npy"), ("events", "events.json"))}})
    tasks = {"W1": {"status": "COMPLETE"}, "W2": {"status": "COMPLETE"}}

    def seal():
        for shard in manifest["shards"]:
            for key in ("X", "y", "events"):
                p = root / shard[key]["path"]
                if p.is_file():
                    shard[key]["sha256"] = file_digest(p)
        write(root / "manifests/source_cache.json", manifest)
        tasks["W2"]["cache_manifest"] = descriptor("manifests/source_cache.json")
        write(root / "status/tasks.json", tasks)

    seal()
    return root, contract, manifest, tasks, seal


def test_missing_input_package_is_rejected_without_training(tmp_path, capsys):
    assert pilot.main(["--root", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED" and result["training_started"] is False


def test_fixture_cache_identity_and_equal_source_sampling_only(gate_fixture):
    root, _, _, _, _ = gate_fixture
    cache = pilot.verify_inputs(root)
    assert set(row["dataset"] for row in cache.rows) == SOURCE_IDS
    assert cache.batch(torch.tensor([0, 2, 4])).shape == (3, 11, 256)
    weights = pilot.sampling_weights(cache)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    for dataset in SOURCE_IDS:
        mask = torch.tensor([row["dataset"] == dataset for row in cache.rows])
        torch.testing.assert_close(weights[mask].sum(), torch.tensor(.25, dtype=torch.float64))


def test_upstream_evidence_failure_cannot_be_bypassed_by_complete_flags(gate_fixture, monkeypatch):
    root, _, _, _, _ = gate_fixture
    def missing_actual_evidence(cache):
        raise ValueError("Actual complete metadata evidence is missing")
    monkeypatch.setattr(pilot, "_verify_upstream_acceptance", missing_actual_evidence)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Must gate before CUDA"))
    with pytest.raises(ValueError, match="complete metadata"):
        pilot.verify_inputs(root)


def test_upstream_file_map_and_exact_proof_digest_are_bound_separately(gate_fixture, monkeypatch):
    from p3work import (window_acceptance, source_cache_acceptance, contracts,
                        f0_bridge, spatial_geometry, source_training, runtime_clock)
    root, _, _, _, _ = gate_fixture
    cache = pilot.verify_inputs(root)
    proof = {"artifact_sha256": "f" * 64}
    for name in ("selection_plan", "split", "inventory", "window_input_session_manifest", "window_coverage"):
        relative = "audit/TEST_DOUBLE_ONLY_" + name + ".json"
        write(root / relative, {"scope": "SOFTWARE_COMPONENT_TEST_ONLY"})
        proof[name] = {"path": relative, "sha256": file_digest(root / relative)}
    for module in (window_acceptance, source_cache_acceptance, contracts,
                   f0_bridge, spatial_geometry, source_training, runtime_clock):
        path = root / "p3work" / Path(module.__file__).name
        path.write_bytes(Path(module.__file__).read_bytes())
    runner = root / "tools/run_controlled_pilot.py"
    runner.parent.mkdir(parents=True)
    runner.write_bytes(Path(pilot.__file__).read_bytes())
    monkeypatch.setattr(window_acceptance, "verify_frozen_window", lambda r: proof)
    monkeypatch.setattr(source_cache_acceptance, "verify_source_cache_acceptance",
                        lambda c: {"artifact_sha256": {"TEST_DOUBLE_ONLY": "e" * 64}})
    PRODUCTION_UPSTREAM_VERIFIER(cache)
    assert cache.upstream_acceptance_sha256["exact_window"] == "f" * 64
    assert len(cache.upstream_acceptance_sha256["full_source_cache"]) == 64
    assert cache.artifact_sha256["TEST_DOUBLE_ONLY"] == "e" * 64
    assert proof["split"]["path"] in cache.artifact_sha256
    # A --root pointing to another code version cannot record that version's
    # SHA while actually running this process's model/training implementation.
    with (root / "p3work/source_training.py").open("ab") as stream:
        stream.write(b"\n# SOFTWARE_TEST_DOUBLE_DIFFERENT_PACKAGED_CODE\n")
    with pytest.raises(pilot.PilotGateError, match="differs from packaged code"):
        PRODUCTION_UPSTREAM_VERIFIER(cache)


def test_dry_run_never_checks_cuda_or_calls_executor(gate_fixture, monkeypatch, capsys):
    from p3work import runtime_clock
    root, _, _, _, _ = gate_fixture
    def forbidden(*args, **kwargs):
        pytest.fail("A CPU gate fixture attempted CUDA or model execution")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(pilot, "execute_pilot", forbidden)
    monkeypatch.setattr(runtime_clock, "collect_clock_preflight", forbidden)
    assert pilot.main(["--root", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training_started"] is False and result["gpu_checked"] is False
    assert result["planned_actual_updates"] == 200
    assert not (root / "runs").exists()


def test_execute_requires_both_flags_and_cuda_without_cpu_fallback(gate_fixture, monkeypatch, capsys):
    root, _, _, _, _ = gate_fixture
    assert pilot.main(["--root", str(root), "--execute"]) == 2
    assert "Both --execute" in capsys.readouterr().out
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pilot.main(["--root", str(root), "--execute", "--user-started-gpu"]) == 2
    assert "CUDA is unavailable" in capsys.readouterr().out
    assert not (root / "runs").exists()


@pytest.mark.parametrize("stage", ["W1", "W2"])
def test_incomplete_stages_are_rejected(gate_fixture, stage):
    root, _, _, tasks, seal = gate_fixture
    tasks[stage]["status"] = "IN_PROGRESS"
    seal()
    with pytest.raises(pilot.PilotGateError, match=stage):
        pilot.verify_inputs(root)


def test_unfrozen_contract_and_cache_sha_mismatch_fail_closed(gate_fixture):
    root, contract, manifest, _, seal = gate_fixture
    contract["status"] = "pending"
    write(root / "contracts/input_contract.json", contract)
    with pytest.raises(ValueError, match="not frozen"):
        pilot.verify_inputs(root)
    contract["status"] = "window_frozen"
    write(root / "contracts/input_contract.json", contract)
    with (root / manifest["shards"][0]["X"]["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(pilot.PilotGateError, match="SHA256"):
        pilot.verify_inputs(root)


@pytest.mark.parametrize("kind", ["external", "missing_source", "legacy_npz", "synthetic", "stale_contract", "bad_label", "event_role", "old_code"])
def test_invalid_cache_or_partition_identity_is_rejected(gate_fixture, kind):
    root, _, manifest, _, seal = gate_fixture
    shard = manifest["shards"][0]
    if kind == "external":
        shard["dataset"] = "HSStroke2026_v6"
    elif kind == "missing_source":
        manifest["shards"].pop()
    elif kind == "legacy_npz":
        shard["X"]["path"] = "cache/old_paper1.npz"
    elif kind == "synthetic":
        manifest["synthetic"] = True
    elif kind == "stale_contract":
        manifest["input_contract_sha256"] = "d" * 64
    elif kind in {"bad_label", "event_role", "old_code"}:
        path = root / shard["events"]["path"]
        rows = json.loads(path.read_text())
        key, value = {"bad_label": ("label", 7), "event_role": ("role", "external_query"),
                      "old_code": ("processing_sha256", "e" * 64)}[kind]
        rows[0][key] = value
        write(path, rows)
    seal()
    with pytest.raises((pilot.PilotGateError, ValueError)):
        pilot.verify_inputs(root)


@pytest.mark.parametrize("violation", ["car", "duplicate_uid", "duplicate_origin"])
def test_pre_rms_and_duplicate_original_events_are_checked(gate_fixture, violation):
    root, _, manifest, _, seal = gate_fixture
    shard = manifest["shards"][0]
    if violation == "car":
        path = root / shard["X"]["path"]
        x = np.load(path, allow_pickle=False)
        x[0] += 1.0  # Break both CAR and temporal demeaning, after valid file rehash.
        np.save(path, x, allow_pickle=False)
    else:
        path = root / shard["events"]["path"]
        rows = json.loads(path.read_text())
        key = "event_uid" if violation == "duplicate_uid" else "original_trial_id"
        rows[1][key] = rows[0][key]
        write(path, rows)
    seal()
    with pytest.raises(pilot.PilotGateError):
        pilot.verify_inputs(root)


def test_checkpoint_tree_identity_comparison_is_exact_and_no_training():
    a = {"model": {"weight": torch.ones(2)}, "update": 100, "sampler": [torch.tensor([1, 2])]}
    b = {"model": {"weight": torch.ones(2)}, "update": 100, "sampler": [torch.tensor([1, 2])]}
    assert pilot._same_tree(a, b)
    b["model"]["weight"][0] += 1e-5
    assert not pilot._same_tree(a, b)


def test_pure_timing_summary_excludes_fixed_warmup_without_extra_updates():
    # Fake timing scalars only: no CUDA events, model, optimizer or forward call.
    rows = [{"branch_update_zero_based": i, "cpu_sampling_normalization_seconds": .1,
             "gpu_h2d_seconds": .002, "gpu_forward_backward_step_seconds": (i + 1) / 1000}
            for i in range(100)]
    result = pilot._timing_summary(rows)
    assert len(result["per_update_timings"]) == 100
    assert result["warmup_updates_in_this_segment"] == 10
    assert result["measured_updates_in_this_segment"] == 90
    assert result["measured_forward_backward_step"]["p50_seconds"] == pytest.approx(.0555)
    assert result["measured_forward_backward_step"]["p90_seconds"] == pytest.approx(.0911)
    assert result["cpu_sampling_normalization_total_seconds"] == pytest.approx(10)
    assert result["billing_time"] is False
