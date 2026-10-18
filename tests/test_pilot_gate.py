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


def test_fixed_source_prediction_selection_is_ordered_and_source_only(gate_fixture):
    root, _, _, _, _ = gate_fixture
    cache = pilot.verify_inputs(root)
    indices, uids, labels = pilot.engineering_prediction_selection(cache)
    assert len(indices) == 8
    expected = [row["event_uid"] for dataset in sorted(SOURCE_IDS)
                for row in sorted((r for r in cache.rows if r["dataset"] == dataset), key=lambda r: r["event_uid"])[:16]]
    assert uids == expected
    assert labels == [int(cache.labels[i]) for i in indices]
    cache.rows[0]["role"] = "meta_validation_query"
    with pytest.raises(pilot.PilotGateError, match="source_train"):
        pilot.engineering_prediction_selection(cache)


def test_prediction_metrics_use_actual_logit_classes_and_explicit_missing_class():
    # SOFTWARE_TEST_DOUBLE: rounded probability tie can differ from logit argmax.
    result = pilot.engineering_prediction_metrics([1], [[.5, .5]], predictions=[1])
    assert result["accuracy"] == 1 and result["balanced_accuracy"] is None
    assert result["class_counts"] == {"0": 0, "1": 1}
    assert result["class_recall"] == {"0": None, "1": 1.0}
    assert result["loss_cross_entropy"] == pytest.approx(np.log(2))
    result = pilot.engineering_prediction_metrics([0, 1], [[1, 0], [1, 0]], predictions=[0, 0])
    assert result["balanced_accuracy"] == .5 and np.isfinite(result["loss_cross_entropy"])
    with pytest.raises(pilot.PilotGateError):
        pilot.engineering_prediction_metrics([0], [[.7, .7]])


def test_checkpoint_save_load_and_validation_have_separate_monotonic_timings(tmp_path):
    # CPU tensors only, no model, optimizer, fit, or CUDA call.
    path, events, identity = tmp_path / "TEST_DOUBLE.pt", [], {"scope": "SOFTWARE_TEST_DOUBLE"}
    state = {"scope": pilot.PREDICTION_SCOPE, "identity": identity,
             "identity_sha256": pilot.digest(identity), "update": 100, "tensor": torch.arange(3)}
    pilot._atomic_checkpoint(path, state, io_timings=events)
    loaded = pilot._load_checkpoint(path, expected_sha=file_digest(path), io_timings=events)
    pilot._validate_checkpoint(loaded, identity, 100, path=path, io_timings=events)
    assert pilot._same_tree(loaded, state)
    assert [e["operation"] for e in events] == ["checkpoint_save_fsync_atomic_commit", "checkpoint_sha256_before_load",
        "checkpoint_load_weights_only", "checkpoint_sha256_after_load", "checkpoint_loaded_identity_validation"]
    assert all(e["monotonic_seconds"] >= 0 and e["status"] == "complete" for e in events)
    with pytest.raises(pilot.PilotGateError, match="update"):
        pilot._validate_checkpoint(loaded, identity, 50, path=path, io_timings=events)
    assert events[-1]["status"] == "failed"


def test_checkpoint_hash_mismatch_blocks_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "TEST_DOUBLE.pt"
    path.write_bytes(b"SOFTWARE_TEST_DOUBLE_NOT_A_CHECKPOINT")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Mismatched checkpoint was loaded"))
    with pytest.raises(pilot.PilotGateError, match="SHA256"):
        pilot._load_checkpoint(path, expected_sha="a" * 64)


def dependency_evidence_fixture(root):
    # A deliberately minimal SOFTWARE_TEST_DOUBLE dependency profile, not a GPU lock.
    versions = {"torch": str(torch.__version__), "numpy": np.__version__}
    (root / "requirements-gpu.lock").write_text("\n".join(f"{key}=={value}" for key, value in versions.items()) + "\n")
    write(root / "audit/gpu_resolved_dependencies.json", [{"name": k, "version": v} for k, v in versions.items()])
    return pilot._resolved_dependencies(root)


@pytest.mark.parametrize("complete", [True, False])
def test_result_manifests_cover_only_actual_allowlisted_evidence_without_cycle(tmp_path, complete):
    root, output = tmp_path, tmp_path / "runs/TEST_DOUBLE"
    output.mkdir(parents=True)
    dependencies = dependency_evidence_fixture(root)
    write(root / "runs/controlled_pilot_job_claim.json", {"scope": "SOFTWARE_TEST_DOUBLE"})
    write(output / "identity.json", {"scope": "SOFTWARE_TEST_DOUBLE"})
    actual = pilot.CHECKPOINT_NAMES if complete else pilot.CHECKPOINT_NAMES[:1]
    for name in actual:
        (output / name).write_bytes(b"SOFTWARE_TEST_DOUBLE_NO_REAL_FIT")
    if complete:
        for name in (*pilot.PREDICTION_NAMES, "engineering_prediction_replay.json"):
            write(output / name, {"scope": "SOFTWARE_TEST_DOUBLE"})
    # Unrelated config/target/temporary files must never be discovered or copied.
    for name in (".env", "target_predictions.json", "unapproved.pt", "resume.pt.partial"):
        (output / name).write_bytes(b"MUST_NOT_BE_DISCOVERED")
    receipt = {"status": "ENGINEERING_PILOT_COMPLETE" if complete else "ENGINEERING_PILOT_FAILED",
               "scope": "SOFTWARE_TEST_DOUBLE", "resolved_dependencies": dependencies}
    pilot._finalize_result_manifests(root, output, receipt)
    artifact = json.loads((output / "engineering_artifact_manifest.json").read_text())
    backup = json.loads((output / "engineering_backup_manifest.json").read_text())
    receipt_disk = json.loads((output / "engineering_receipt.json").read_text())
    names = {Path(row["path"]).name for row in backup["files"]}
    assert not names.intersection({".env", "target_predictions.json", "unapproved.pt", "resume.pt.partial"})
    assert "engineering_receipt.json" not in {Path(row["path"]).name for row in artifact["files"]}
    assert "engineering_artifact_manifest.json" in names and "engineering_receipt.json" in names
    assert "engineering_backup_manifest.json" not in names
    assert set(actual) <= names
    assert not complete or set(pilot.CHECKPOINT_NAMES) | set(pilot.PREDICTION_NAMES) | {"engineering_prediction_replay.json"} <= names
    assert receipt_disk["artifact_manifest"]["sha256"] == file_digest(output / "engineering_artifact_manifest.json")
    from tools.r2_backup import load_manifest
    files, _ = load_manifest(output / "engineering_backup_manifest.json", root)
    assert len(files) == len(backup["files"])  # Local schema/SHA check only; no SDK/network.


def test_result_manifest_rejects_symlink_even_inside_root(tmp_path):
    root, output = tmp_path, tmp_path / "runs/TEST_DOUBLE"
    output.mkdir(parents=True)
    write(root / "runs/controlled_pilot_job_claim.json", {"scope": "SOFTWARE_TEST_DOUBLE"})
    (root / "private_config.json").write_text("DO_NOT_BACK_UP")
    (output / "identity.json").symlink_to(root / "private_config.json")
    with pytest.raises(pilot.PilotGateError, match="Symlink"):
        pilot._finalize_result_manifests(root, output, {"status": "ENGINEERING_PILOT_FAILED"})
    assert not (output / "engineering_backup_manifest.json").exists()


@pytest.mark.parametrize("destination", ["alias", "outside", "not_runs", "runs_root"])
def test_manifest_helper_rejects_bad_destination_before_any_write(tmp_path, destination):
    root = tmp_path / "paper3"
    (root / "runs").mkdir(parents=True)
    target = tmp_path / "SOFTWARE_TEST_DOUBLE_UNOWNED"
    target.mkdir()
    if destination == "alias":
        output = root / "runs/linked"
        output.symlink_to(target, target_is_directory=True)
    elif destination == "outside":
        output = target
    elif destination == "not_runs":
        output = root / "metadata"
        output.mkdir()
    else:
        output = root / "runs"
    with pytest.raises(pilot.PilotGateError):
        pilot._finalize_result_manifests(root, output, {"status": "ENGINEERING_PILOT_FAILED"})
    assert not list(target.iterdir())
    assert not list(root.rglob("engineering_*manifest.json"))


def test_resolved_dependencies_require_all_actual_pins_and_plain_package_fields(tmp_path):
    dependency_evidence_fixture(tmp_path)
    path = tmp_path / "audit/gpu_resolved_dependencies.json"
    write(path, [{"name": "torch", "version": str(torch.__version__)}])
    with pytest.raises(pilot.PilotGateError, match="Torch/NumPy"):
        pilot._resolved_dependencies(tmp_path)
    write(path, [{"name": "torch", "version": str(torch.__version__), "config": "DO_NOT_COPY"}])
    with pytest.raises(pilot.PilotGateError, match="only package"):
        pilot._resolved_dependencies(tmp_path)


def test_resolved_dependencies_reject_internal_symlink_before_reading_payload(tmp_path, monkeypatch):
    dependency_evidence_fixture(tmp_path)
    path = tmp_path / "audit/gpu_resolved_dependencies.json"
    target = tmp_path / "SOFTWARE_TEST_DOUBLE_metadata.json"
    path.replace(target)
    path.symlink_to(target)
    original_read = Path.read_text
    def reject_payload_read(self, *args, **kwargs):
        if self in (path, target):
            pytest.fail("Symlinked dependency payload was read")
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", reject_payload_read)
    with pytest.raises(pilot.PilotGateError, match="Symlink"):
        pilot._resolved_dependencies(tmp_path)


@pytest.mark.parametrize("alias", ["runs", "runs/linked"])
def test_output_symlink_is_rejected_before_claim_or_output_creation(gate_fixture, monkeypatch, alias):
    root, _, _, _, _ = gate_fixture
    cache = pilot.verify_inputs(root)
    target = root / "SOFTWARE_TEST_DOUBLE_UNOWNED"
    target.mkdir()
    link = root / alias
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(pilot.PilotGateError, match="Symlink"):
        pilot.execute_pilot(cache, alias + "/pilot", user_started_gpu=True)
    assert list(target.iterdir()) == []
    assert not (root / "runs/controlled_pilot_job_claim.json").exists()


@pytest.mark.parametrize("changed", [False, True])
def test_prediction_replay_reloads_both_checkpoints_without_optimizer_updates(gate_fixture, monkeypatch, changed):
    # SOFTWARE_TEST_DOUBLE: fake model/eval outputs, real CPU checkpoint JSON/SHA pipeline.
    root, _, _, _, _ = gate_fixture
    cache, identity = pilot.verify_inputs(root), {"scope": "SOFTWARE_TEST_DOUBLE"}
    output = root / "runs/TEST_DOUBLE"
    output.mkdir(parents=True)
    for index, name in enumerate(pilot.CHECKPOINT_NAMES[1::2]):
        state = {"scope": pilot.PREDICTION_SCOPE, "identity": identity,
                 "identity_sha256": pilot.digest(identity), "update": 100,
                 "model": {"value": torch.tensor(float(index if changed else 0))}, "optimizer": {}}
        pilot._atomic_checkpoint(output / name, state)

    models = []
    class FakeModel:
        def __init__(self):
            self.state = None
            self.training = True
            models.append(self)
        def state_dict(self):
            return self.state
        def eval(self):
            self.training = False
            return self
        def train(self, mode=True):
            self.training = mode
            return self
        def __call__(self, x):
            assert not torch.is_grad_enabled() and not self.training
            return torch.tensor([[float(self.state["value"]), 1.0]] * len(x))

    class NoStepOptimizer:
        def state_dict(self):
            return {}
        def step(self):
            pytest.fail("Prediction replay attempted an optimizer update")

    monkeypatch.setattr(pilot, "_new_state", lambda _: (FakeModel(), NoStepOptimizer(), None))
    monkeypatch.setattr(pilot, "_restore", lambda saved, model, optimizer, sampler, ident: setattr(model, "state", saved["model"]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    actual_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *a, **kw: tensor if a == ("cuda:0",) else actual_to(tensor, *a, **kw))
    timings = []
    if changed:
        with pytest.raises(pilot.PilotGateError, match="predictions or metrics differ"):
            pilot._prediction_replay(cache, output, identity, io_timings=timings, deadline=float("inf"))
    else:
        result = pilot._prediction_replay(cache, output, identity, io_timings=timings, deadline=float("inf"))
        assert result["sha256"] == file_digest(root / result["path"])
    proof = json.loads((output / "engineering_prediction_replay.json").read_text())
    assert proof["all_exact"] is (not changed)
    assert proof["optimizer_updates_for_replay"] == 0 and proof["role"] == "source_train"
    assert len([row for row in timings if row["operation"] == "checkpoint_load_weights_only"]) == 2
    assert all(model.training for model in models)
    for key, name in zip(("continuous", "resumed"), pilot.CHECKPOINT_NAMES[1::2]):
        assert proof[key]["checkpoint"]["sha256"] == file_digest(output / name)
        prediction = json.loads((root / proof[key]["predictions"]["path"]).read_text())
        assert prediction["eval_mode"] and prediction["optimizer_updates"] == 0
        assert prediction["metrics"] == pilot.engineering_prediction_metrics(prediction["labels"],
            prediction["probabilities"], predictions=prediction["predictions"])


def test_each_checkpoint_io_stage_checks_same_deadline_after_completion(tmp_path, monkeypatch):
    clock, calls, timings = [1.0], [], []
    monkeypatch.setattr(pilot.time, "monotonic", lambda: clock[0])
    def action():
        calls.append("SOFTWARE_TEST_DOUBLE")
        clock[0] = 6.0
    with pytest.raises(TimeoutError, match="wall-time"):
        pilot._io_event(timings, "checkpoint_save", tmp_path / "TEST_DOUBLE.pt", action, deadline=5)
    assert calls == ["SOFTWARE_TEST_DOUBLE"] and timings[0]["status"] == "failed"
    with pytest.raises(TimeoutError):
        pilot._io_event(timings, "checkpoint_load", tmp_path / "TEST_DOUBLE.pt", action, deadline=5)
    assert len(calls) == 1  # Expired budget never begins a new stage.


def test_prediction_selection_caps_each_source_without_rng_or_payload_access():
    from types import SimpleNamespace
    import random
    rows = [{"dataset": dataset, "event_uid": f"SOFTWARE_TEST_DOUBLE/{dataset}/{index:03}",
             "role": "source_train", "label": index % 2}
            for dataset in reversed(sorted(SOURCE_IDS)) for index in reversed(range(19))]
    cache = SimpleNamespace(rows=rows, labels=np.asarray([row["label"] for row in rows]))
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    indices, uids, labels = pilot.engineering_prediction_selection(cache)
    assert len(indices) == 64
    assert uids == [f"SOFTWARE_TEST_DOUBLE/{dataset}/{index:03}" for dataset in sorted(SOURCE_IDS) for index in range(16)]
    assert labels == [index % 2 for _ in SOURCE_IDS for index in range(16)]
    assert random.getstate() == python_state and torch.equal(torch_state, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and np.array_equal(numpy_state[1], after[1]) and numpy_state[2:] == after[2:]


def test_complete_result_manifest_requires_all_proof_files(tmp_path):
    output = tmp_path / "runs/TEST_DOUBLE"
    output.mkdir(parents=True)
    dependencies = dependency_evidence_fixture(tmp_path)
    write(tmp_path / "runs/controlled_pilot_job_claim.json", {"scope": "SOFTWARE_TEST_DOUBLE"})
    with pytest.raises(pilot.PilotGateError, match="every declared"):
        pilot._finalize_result_manifests(tmp_path, output,
            {"status": "ENGINEERING_PILOT_COMPLETE", "resolved_dependencies": dependencies})
    assert not (output / "engineering_backup_manifest.json").exists()


@pytest.mark.parametrize("fail_before_updates,fail_final_hash", [(False, False), (True, False), (False, True)])
def test_runner_stage_wiring_keeps_total_200_and_one_claim_with_mock_execution(gate_fixture, monkeypatch, fail_before_updates, fail_final_hash):
    """SOFTWARE_TEST_DOUBLE execution plumbing: no model fit, CUDA or command runs."""
    from types import SimpleNamespace
    from p3work import runtime_clock
    root, _, _, _, _ = gate_fixture
    cache = pilot.verify_inputs(root)
    dependency_evidence_fixture(root)
    for name in ("f0_bridge.py", "spatial_geometry.py", "contracts.py"):
        (root / "p3work" / name).write_text("# SOFTWARE_TEST_DOUBLE_NOT_PRODUCTION_CODE\n")

    class FakeModel:
        def __init__(self):
            self.state = {"value": torch.tensor(0.0)}
            self.training = True
        def state_dict(self):
            return self.state
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode
        def __call__(self, x):
            assert not self.training and not torch.is_grad_enabled()
            return torch.tensor([[0.0, 1.0]] * len(x))

    class NoStepOptimizer:
        def state_dict(self):
            return {}
        def step(self):
            pytest.fail("Mock plumbing invoked a real optimizer step")

    monkeypatch.setattr(pilot, "_new_state", lambda _: (FakeModel(), NoStepOptimizer(), None))
    def fake_checkpoint(model, optimizer, sampler, update, identity):
        return {"scope": pilot.PREDICTION_SCOPE, "identity": identity, "identity_sha256": pilot.digest(identity),
            "update": update, "model": model.state_dict(), "optimizer": {}, "sampler_rng": torch.tensor([0]),
            "torch_rng": torch.tensor([0]), "cuda_rng": torch.tensor([0]), "python_rng": (0,), "numpy_rng": [0]}
    monkeypatch.setattr(pilot, "_checkpoint", fake_checkpoint)
    monkeypatch.setattr(pilot, "_restore", lambda saved, model, optimizer, sampler, ident: setattr(model, "state", saved["model"]))
    segments = []
    def fake_updates(cache, model, optimizer, sampler, begin, stop, *, deadline, progress, phase):
        segments.append((begin, stop))
        if fail_before_updates:
            raise RuntimeError("SOFTWARE_TEST_DOUBLE_INJECTED_FAILURE_NO_REAL_FIT")
        progress["actual_completed_updates"] += stop - begin
        progress["optimizer_updates_dispatched"] += stop - begin
        progress["training_started"] = True  # Component mock, not real scientific training.
        return {"phase": phase, "actual_updates": stop - begin}
    monkeypatch.setattr(pilot, "_updates", fake_updates)
    monkeypatch.setattr(runtime_clock, "collect_clock_preflight", lambda: {"scope": "SOFTWARE_TEST_DOUBLE"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(
        name="SOFTWARE_TEST_DOUBLE_NOT_A_REAL_GPU", uuid="TEST_DOUBLE", total_memory=1024, major=0, minor=0))
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a: None)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda *a: None)
    for obj, name in ((torch.backends.cudnn, "benchmark"), (torch.backends.cudnn, "deterministic"),
                      (torch.backends.cudnn, "allow_tf32"), (torch.backends.cuda.matmul, "allow_tf32")):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="SOFTWARE_TEST_DOUBLE\n"))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    actual_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *a, **kw: tensor if a == ("cuda:0",) else actual_to(tensor, *a, **kw))
    if fail_final_hash:
        finalizing = [False]
        actual_replay, actual_digest = pilot._prediction_replay, pilot.file_digest
        def replay_then_inject_failure(*args, **kwargs):
            value = actual_replay(*args, **kwargs)
            finalizing[0] = True
            return value
        def fail_final_checkpoint_hash(path):
            if finalizing[0] and Path(path).name == "engineering_initial.pt":
                raise OSError("SOFTWARE_TEST_DOUBLE_FINAL_HASH_FAILURE")
            return actual_digest(path)
        monkeypatch.setattr(pilot, "_prediction_replay", replay_then_inject_failure)
        monkeypatch.setattr(pilot, "file_digest", fail_final_checkpoint_hash)
    result = pilot.execute_pilot(cache, "runs/TEST_DOUBLE", user_started_gpu=True)
    assert result["status"] == ("ENGINEERING_PILOT_FAILED" if fail_before_updates or fail_final_hash else "ENGINEERING_PILOT_COMPLETE")
    assert segments == ([(0, 100)] if fail_before_updates else [(0, 100), (0, 50), (50, 100)])
    assert result["actual_completed_updates"] == result["optimizer_updates_dispatched"] == (0 if fail_before_updates else 200)
    assert result["prediction_replay_optimizer_updates"] == 0
    assert result["exact_resume_verified"] is (not fail_before_updates)
    if fail_final_hash:
        assert result["result_manifest_status"] == "BLOCKED_NO_COMPLETE_BACKUP_MANIFEST"
        assert result["training_started"] is True and result["actual_completed_updates"] == 200
        saved_receipt = json.loads((root / "runs/TEST_DOUBLE/engineering_receipt.json").read_text())
        assert saved_receipt["status"] == "ENGINEERING_PILOT_FAILED" and saved_receipt["training_started"] is True
        assert not (root / "runs/TEST_DOUBLE/engineering_backup_manifest.json").exists()
    else:
        assert result["result_manifest_status"] == "LOCAL_MANIFESTS_COMPLETE_R2_NOT_ATTEMPTED"
        from tools.r2_backup import load_manifest
        files, _ = load_manifest(root / result["backup_manifest_path"], root)
        names = {Path(row["path"]).name for row in files}
        if fail_before_updates:
            assert "engineering_initial.pt" in names
            assert not names.intersection(set(pilot.CHECKPOINT_NAMES[1:]) | set(pilot.PREDICTION_NAMES))
        else:
            assert names >= set(pilot.CHECKPOINT_NAMES) | set(pilot.PREDICTION_NAMES)
    with pytest.raises(pilot.PilotGateError, match="already claimed"):
        pilot.execute_pilot(cache, "runs/DIFFERENT_OUTPUT", user_started_gpu=True)
    assert len(segments) == (1 if fail_before_updates else 3)  # Changing output cannot begin a second job.
