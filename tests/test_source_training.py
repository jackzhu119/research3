"""Mathematical CPU/mock software checks only, never genuine EEG/CUDA fits.

All temporary source identities and pilot artifacts below are explicit test
doubles. Completion-shaped workflow envelopes are marked SOFTWARE_TEST_DOUBLE;
their placeholder checkpoints cannot load as scientific source models.
"""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from p3work import source_training as training
from p3work.contracts import SOURCE_IDS, digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper3_train_source_test", ROOT / "tools/train_source.py")
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 11, 128)).astype(np.float32)
    x -= x.mean(1, keepdims=True); x -= x.mean(2, keepdims=True)
    rows, y = [], []
    for d, dataset in enumerate(sorted(SOURCE_IDS)):
        for i in range(4):
            index = d * 4 + i
            rows.append({"dataset": dataset, "subject_id": str(i // 2 + 1),
                "event_uid": f"MATHEMATICAL_FIXTURE/{dataset}/{i}", "role": "source_train",
                "processed_signal_sha256": __import__("hashlib").sha256(x[index].tobytes()).hexdigest()})
            y.append(i % 2)
    for relative in ("tools/run_controlled_pilot.py", "p3work/f0_bridge.py", "p3work/spatial_geometry.py", "p3work/contracts.py"):
        target = tmp_path / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    training.atomic_json(tmp_path / "cache_fixture.json", {"scope": "MATHEMATICAL_TEST_DOUBLE_ONLY"})
    monkeypatch.setattr(training, "_code_policy", lambda root: {"code": {"test_double": "c" * 64}, "policy": {"test_double": "p" * 64}})
    return SimpleNamespace(root=tmp_path, n_times=128, rows=rows, labels=np.asarray(y, dtype=np.int64),
        manifest_path="cache_fixture.json", manifest_sha256=file_digest(tmp_path / "cache_fixture.json"),
        input_contract_sha256="a" * 64, artifact_sha256={"cache_fixture.json": file_digest(tmp_path / "cache_fixture.json")},
        batch=lambda indices: x[np.asarray(indices, dtype=np.int64)])


def fixture_config(**updates):
    return {**training.SOURCE_CONFIG, "epochs": 2, "steps_per_epoch": 2,
            "batch_size": 4, "threads": 1, "checkpoint_every_updates": 2, **updates}


def first_job(cache):
    return training.build_training_plan(cache)["jobs"][0]


def test_registered_lodo_seed_and_fit_budgets_are_plans_not_completions(fixture_cache):
    cache = fixture_cache
    development = training.build_training_plan(cache)
    confirmation = training.build_training_plan(cache, profile="confirmation")
    assert development["planned_neural_fits"] == 15 and development["planned_core_total_fits"] == 25
    assert confirmation["planned_neural_fits"] == 25 and confirmation["planned_core_total_fits"] == 35
    assert development["max_neural_optimizer_updates"] == 76800
    assert confirmation["max_neural_optimizer_updates"] == 128000
    assert development["training_started"] is False
    assert confirmation["seeds"] == list(range(20261018, 20261023))
    assert all(item["actual_fits_completed"] == 0 for item in development["classical_dependencies"])
    for job in confirmation["jobs"]:
        train = [cache.rows[i] for i in job["train_indices"]]
        val = [cache.rows[i] for i in job["validation_indices"]]
        if job["phase"] == "lodo":
            assert {r["dataset"] for r in val} == {job["heldout_dataset"]}
            assert len({r["dataset"] for r in train}) == 3
            assert not ({(r["dataset"], r["subject_id"]) for r in train} &
                        {(r["dataset"], r["subject_id"]) for r in val})
        else:
            assert not val and job["final_epochs_status"] == "PENDING_FOUR_SAME_SEED_LODO"


def test_duplicate_signal_cannot_cross_source_lodo_boundary(fixture_cache):
    cache = fixture_cache
    cache.rows[4]["processed_signal_sha256"] = cache.rows[0]["processed_signal_sha256"]
    with pytest.raises(training.SourceTrainingError, match="duplicate audit"):
        training.build_training_plan(cache)


def test_complete_plan_is_rebuilt_before_execution_or_export(fixture_cache):
    cache = fixture_cache; plan = training.build_training_plan(cache)
    training.validate_training_plan(cache, plan)
    changed = copy.deepcopy(plan); changed.update(jobs=[], seeds=[])
    # Even a caller recomputing its own subset digest cannot register it.
    changed["plan_sha256"] = digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(training.SourceTrainingError, match="complete registered source plan"):
        training.execute_plan(cache, changed, cache.root / "no_execute", runtime={"device": "cuda:0"}, pilot_evidence={})
    with pytest.raises(training.SourceTrainingError, match="complete registered source plan"):
        training.export_f0_receipts(cache, changed, cache.root / "no_export", pilot_evidence={})
    assert not (cache.root / "no_execute").exists()


def test_returned_plan_cannot_mutate_the_registered_config(fixture_cache):
    cache = fixture_cache; original = copy.deepcopy(training.SOURCE_CONFIG)
    plan = training.build_training_plan(cache)
    plan["configuration"]["learning_rate"] = .8
    plan["configuration"]["betas"][0] = .1
    assert training.SOURCE_CONFIG == original
    with pytest.raises(training.SourceTrainingError, match="complete registered source plan"):
        training.validate_training_plan(cache, plan)


def test_confirmation_extension_reuses_only_all_verified_registered_development_fits(fixture_cache, monkeypatch):
    cache = fixture_cache; output = cache.root / "extension"; runtime = {"device": "cuda:0", "scope": "MOCK_ONLY"}
    development = training.build_training_plan(cache); confirmation = training.build_training_plan(cache, profile="confirmation")
    training.install_training_plan(cache, development, output, runtime=runtime)
    with pytest.raises(training.SourceTrainingError, match="explicit confirmation"):
        training.install_training_plan(cache, confirmation, output, runtime=runtime)
    with pytest.raises(FileNotFoundError):
        training.install_training_plan(cache, confirmation, output, runtime=runtime, extend_confirmation=True)
    assert json.loads((output / "training_plan.json").read_text()) == development
    calls = []
    # Only the validator is mocked here; this is not evidence of real fits.
    monkeypatch.setattr(training, "verify_complete_fit", lambda c, job, folder: calls.append(job["job_id"]))
    for job in development["jobs"]:
        training.atomic_json(output / job["job_id"] / "identity.json", {"runtime": runtime})
    receipt = training.install_training_plan(cache, confirmation, output, runtime=runtime, extend_confirmation=True)
    assert len(calls) == 15 and len(set(calls)) == 15
    extension = json.loads((cache.root / receipt["path"]).read_text())
    assert extension["verified_neural_fits_reused"] == 15 and extension["additional_neural_fits_planned"] == 10
    assert extension["new_fits_started_by_this_operation"] == 0
    assert json.loads((output / "training_plan.json").read_text()) == confirmation
    with pytest.raises(training.SourceTrainingError, match="explicit confirmation"):
        training.install_training_plan(cache, development, output, runtime=runtime, extend_confirmation=True)


def test_confirmation_preserves_per_seed_receipt_bytes_and_never_refits_first_15(fixture_cache, monkeypatch):
    cache = fixture_cache; output = cache.root / "SOFTWARE_TEST_DOUBLE_SOURCE_WORKFLOW"
    development = training.build_training_plan(cache)
    confirmation = training.build_training_plan(cache, profile="confirmation")
    scope = "SOFTWARE_TEST_DOUBLE_NO_REAL_FIT_OR_GPU"
    runtime = {"device": "cuda:0", "fixture_scope": scope}
    proof = {"receipt": {"path": "SOFTWARE_TEST_DOUBLE_PILOT_ONLY"}, "exact_resume_verified": True,
             "fixture_scope": scope}
    monkeypatch.setattr(training, "verify_engineering_pilot", lambda *a, **k: proof)
    original_immutable = training.immutable_json
    original_atomic = training.atomic_json
    def marked_fixture_receipt(path, value):
        original_immutable(path, {**value, "fixture_scope": scope})
    monkeypatch.setattr(training, "immutable_json", marked_fixture_receipt)
    def marked_fixture_progress(path, value):
        if Path(path).name == "source_training_progress.json":
            value = {**value, "fixture_scope": scope, "actual_scientific_fits_completed": 0,
                     "counts_above_are_mock_workflow_units": True}
        original_atomic(path, value)
    monkeypatch.setattr(training, "atomic_json", marked_fixture_progress)
    def selected_epoch(job):
        return (sorted(SOURCE_IDS).index(job["heldout_dataset"]) + 2) if job["phase"] == "lodo" else 3
    def mock_summary(c, job, folder):
        return {"complete": True, "selected_epoch": selected_epoch(job), "fixture_scope": scope}
    monkeypatch.setattr(training, "verify_complete_fit", mock_summary)
    def materialize_mock(job, epochs):
        folder = output / job["job_id"]
        training.atomic_json(folder / "identity.json", {"runtime": runtime,
            "configuration": {**training.SOURCE_CONFIG, "effective_epochs": epochs}, "fixture_scope": scope})
        training.atomic_json(folder / "status.json", {"status": "SOURCE_FIT_COMPLETE", "fixture_scope": scope})
        # Deliberately NOT torch serialization: this cannot load as real f0.
        (folder / "selected.pt").write_bytes(b"SOFTWARE_TEST_DOUBLE_NO_MODEL_WEIGHTS")
        training.atomic_json(folder / "training_uids.json", {"fixture_scope": scope})
    for job in development["jobs"]:
        materialize_mock(job, 80 if job["phase"] == "lodo" else 3)
    training.install_training_plan(cache, development, output, runtime=runtime)
    first_receipts = training.export_f0_receipts(cache, development, output, pilot_evidence=proof)
    paths = [output / f"{kind}_seed-{seed}.json" for seed in development["seeds"]
             for kind in ("f0_receipt", "source_lodo")]
    old_bytes = {path: path.read_bytes() for path in paths}
    old_hashes = {path: file_digest(path) for path in paths}
    actual_fit_kernel_calls = []
    def mock_new_fit(c, job, folder, **kwargs):
        actual_fit_kernel_calls.append(job["job_id"])
        assert job["seed"] in {20261021, 20261022}
        materialize_mock(job, kwargs["epochs"])
        return {"complete": True, "new_updates_this_invocation": 0, "fixture_scope": scope}
    monkeypatch.setattr(training, "run_fit", mock_new_fit)
    result = training.execute_plan(cache, confirmation, output, runtime=runtime,
        pilot_evidence=proof, extend_confirmation=True)
    assert len(actual_fit_kernel_calls) == 10
    assert not set(actual_fit_kernel_calls) & {job["job_id"] for job in development["jobs"]}
    assert {path: path.read_bytes() for path in paths} == old_bytes
    assert {path: file_digest(path) for path in paths} == old_hashes
    assert result["f0_receipts"][:3] == first_receipts
    profile3 = json.loads((output / "source_profile_development.json").read_text())
    profile5 = json.loads((output / "source_profile_confirmation.json").read_text())
    assert profile3["profile_neural_fits_verified_complete"] == 15
    assert profile5["profile_neural_fits_verified_complete"] == 25
    assert all("profile_neural_fits_verified_complete" not in json.loads(path.read_text())
               for path in paths if path.name.startswith("f0_receipt"))
    # Same evidence is idempotent; a resealed different pilot is forbidden and
    # cannot overwrite already bound per-seed evidence.
    training.export_f0_receipts(cache, confirmation, output, pilot_evidence=proof)
    changed = {**proof, "fixture_difference": "DIFFERENT_MOCK_EVIDENCE"}
    monkeypatch.setattr(training, "verify_engineering_pilot", lambda *a, **k: changed)
    with pytest.raises(training.SourceTrainingError, match="Immutable source receipt differs"):
        training.export_f0_receipts(cache, confirmation, output, pilot_evidence=changed)
    assert {path: path.read_bytes() for path in paths} == old_bytes


def test_hierarchical_sampling_equal_dataset_person_observed_class():
    rows, y = [], []
    for dataset, person, label, count in [("A", "1", 0, 3), ("A", "1", 1, 1),
                                        ("A", "2", 0, 1), ("A", "2", 1, 1),
                                        ("B", "1", 0, 3), ("B", "1", 1, 2)]:
        rows.extend([{"dataset": dataset, "subject_id": person}] * count); y.extend([label] * count)
    weights = training.hierarchical_weights(rows, y).numpy()
    assert np.isclose(weights.sum(), 1)
    for dataset in ("A", "B"):
        assert np.isclose(sum(w for r, w in zip(rows, weights) if r["dataset"] == dataset), .5)
    for person in ("1", "2"):
        for label in (0, 1):
            assert np.isclose(sum(w for r, c, w in zip(rows, y, weights)
                                 if r == {"dataset": "A", "subject_id": person} and c == label), .125)


def test_validation_is_participant_equal_ce_and_final_rule_is_per_seed_median():
    rows = [{"dataset": "A", "subject_id": "1"}] * 3 + [{"dataset": "A", "subject_id": "2"}]
    probabilities = np.asarray([[.9, .1]] * 3 + [[.1, .9]])
    value = training.participant_validation_loss(rows, np.zeros(4, dtype=int), probabilities)
    assert np.isclose(value, (-np.log(.9) - np.log(.1)) / 2)
    assert training.final_epoch_from_lodo([2, 3, 4, 5]) == 3
    with pytest.raises(training.SourceTrainingError):
        training.final_epoch_from_lodo([2, 3, 4])


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): same_tree(x, y)
    else:
        assert a == b


def test_mid_epoch_resume_replays_model_optimizer_all_rng_and_cursor_exactly(fixture_cache):
    cache, runtime = fixture_cache, {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}
    job, config = first_job(cache), fixture_config()
    full, resumed = cache.root / "full", cache.root / "resumed"
    training.run_fit(cache, job, full, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    paused = training.run_fit(cache, job, resumed, runtime=runtime, epochs=2, config=config,
                              max_new_updates=3, synthetic_fixture=True)
    assert not paused["complete"] and paused["checkpoint_cursor"] == {"epoch": 1, "update_in_epoch": 1}
    final = training.run_fit(cache, job, resumed, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert final["complete"] and final["committed_optimizer_updates"] == 4
    a = torch.load(full / "last_state.pt", weights_only=True)
    b = torch.load(resumed / "last_state.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "best_model", "history", "completed_updates", "update_in_epoch"):
        same_tree(a[key], b[key])
    assert json.loads((resumed / "status.json").read_text())["status"] == "SYNTHETIC_FIXTURE_COMPLETE"
    assert not list(cache.root.rglob("f0_receipt*.json"))
    uid = json.loads((resumed / "training_uids.json").read_text())
    assert uid["schema_version"] == "paper3-source-training-uids-v2"
    assert uid["role"] == "source_pretrain_train" and uid["synthetic"] is True


def test_source_checkpoint_identity_mismatch_preserves_prior_state(fixture_cache):
    cache = fixture_cache; folder = cache.root / "pause"; job = first_job(cache)
    runtime = {"device": "cpu", "scope": "FIXTURE"}
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=fixture_config(),
                     max_new_updates=1, synthetic_fixture=True)
    before = file_digest(folder / "last_state.pt")
    for changed in ({"runtime": {"device": "cpu", "scope": "DIFFERENT_RUNTIME"}},
                    {"config": fixture_config(learning_rate=.003)}):
        kwargs = {"runtime": runtime, "config": fixture_config(), **changed}
        with pytest.raises(training.SourceTrainingError, match="identity changed"):
            training.run_fit(cache, job, folder, epochs=2, synthetic_fixture=True, **kwargs)
        assert file_digest(folder / "last_state.pt") == before


def test_resume_recovers_epoch_archive_after_commit_archive_hard_interrupt(fixture_cache):
    cache = fixture_cache; job = first_job(cache); folder = cache.root / "archive_interrupted"
    runtime, config = {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}, fixture_config()
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config,
                     max_new_updates=2, synthetic_fixture=True)
    saved = torch.load(folder / "last_state.pt", weights_only=True)
    assert saved["completed_epochs"] == 1 and saved["update_in_epoch"] == 0
    (folder / "epoch_0001.pt").unlink()  # Exact state of kill after commit, before archive.
    training.checkpoint_sidecar(folder / "epoch_0001.pt").unlink()
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    same_tree(torch.load(folder / "epoch_0001.pt", weights_only=True), saved)
    assert (folder / "epoch_0002.pt").is_file()


def test_first_epoch_validation_failure_resumes_from_saved_initial_state_exactly(fixture_cache, monkeypatch):
    cache, job = fixture_cache, first_job(fixture_cache)
    runtime, config = {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}, fixture_config()
    continuous, interrupted = cache.root / "continuous_validation", cache.root / "failed_validation"
    training.run_fit(cache, job, continuous, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    real_predict = training.predict_indices

    def failed_validation(*args, **kwargs):
        initial = training.verified_checkpoint(interrupted / "last_state.pt")
        assert initial["completed_updates"] == initial["completed_epochs"] == initial["update_in_epoch"] == 0
        assert initial["history"] == [] and initial["best_model"] is None
        raise RuntimeError("MATHEMATICAL_FIXTURE_FIRST_VALIDATION_FAILURE")

    monkeypatch.setattr(training, "predict_indices", failed_validation)
    with pytest.raises(RuntimeError, match="MATHEMATICAL_FIXTURE_FIRST_VALIDATION_FAILURE"):
        training.run_fit(cache, job, interrupted, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert training.verified_checkpoint(interrupted / "last_state.pt")["completed_updates"] == 0
    journal = json.loads((interrupted / "execution_progress.json").read_text())
    assert journal["last_observed_trajectory_update"] == journal["observed_optimizer_calls_total"] == 2
    monkeypatch.setattr(training, "predict_indices", real_predict)
    resumed = training.run_fit(cache, job, interrupted, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert resumed["complete"] and resumed["committed_optimizer_updates"] == 4
    assert resumed["observed_optimizer_calls_total"] == 6 and resumed["replayed_observed_updates_this_invocation"] == 2
    reference = training.verified_checkpoint(continuous / "last_state.pt")
    actual = training.verified_checkpoint(interrupted / "last_state.pt")
    for key in ("model", "optimizer", "rng", "best_model", "history", "completed_updates", "completed_epochs",
                "update_in_epoch", "best_epoch", "best_loss", "bad_epochs"):
        same_tree(actual[key], reference[key])


@pytest.mark.parametrize("damage", ["new_payload_old_sidecar", "old_payload_new_sidecar", "missing_sidecar", "trailing_bytes"])
def test_checkpoint_torn_pairs_reject_before_deserialization_without_rehash(tmp_path, monkeypatch, damage):
    path = tmp_path / "checkpoint.pt"; sidecar = training.checkpoint_sidecar(path)
    training.atomic_checkpoint(path, {"mathematical_fixture": torch.tensor([1.0])})
    old_payload, old_sidecar = path.read_bytes(), sidecar.read_bytes()
    training.atomic_checkpoint(path, {"mathematical_fixture": torch.tensor([2.0])})
    if damage == "new_payload_old_sidecar":
        sidecar.write_bytes(old_sidecar)
    elif damage == "old_payload_new_sidecar":
        path.write_bytes(old_payload)
    elif damage == "missing_sidecar":
        sidecar.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"TORN_PAIR_TRAILING_BYTES")
    before = (path.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None)
    monkeypatch.setattr(training.torch, "load", lambda *a, **k: pytest.fail("Corrupt payload was deserialized before SHA verification"))
    with pytest.raises(training.SourceTrainingError, match="integrity|SHA256/bytes"):
        training.verified_checkpoint(path)
    assert (path.read_bytes(), sidecar.read_bytes() if sidecar.exists() else None) == before


@pytest.mark.parametrize("target", ["last_state.pt", "epoch_0002.pt", "selected.pt"])
def test_resume_does_not_reconstruct_missing_sidecar_of_existing_checkpoint(fixture_cache, target):
    cache, job = fixture_cache, first_job(fixture_cache)
    folder = cache.root / "missing_integrity"
    runtime, config = {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}, fixture_config()
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    path = folder / target; before = path.read_bytes()
    training.checkpoint_sidecar(path).unlink()
    with pytest.raises(training.SourceTrainingError, match="integrity pair is missing"):
        training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert path.read_bytes() == before and not training.checkpoint_sidecar(path).exists()


def test_missing_initial_checkpoint_after_observed_work_is_not_silently_restarted(fixture_cache):
    cache, job = fixture_cache, first_job(fixture_cache)
    folder = cache.root / "missing_initial"
    runtime, config = {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}, fixture_config()
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config,
                     max_new_updates=1, synthetic_fixture=True)
    path = folder / "last_state.pt"
    path.unlink(); training.checkpoint_sidecar(path).unlink()
    with pytest.raises(training.SourceTrainingError, match="missing after observed work"):
        training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert not path.exists() and not training.checkpoint_sidecar(path).exists()


def test_verified_completed_selection_retry_preserves_selected_payload_and_sidecar(fixture_cache):
    cache, job = fixture_cache, first_job(fixture_cache)
    folder = cache.root / "selection_retry"
    runtime, config = {"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"}, fixture_config()
    training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    selected = folder / "selected.pt"; sidecar = training.checkpoint_sidecar(selected)
    before = selected.read_bytes(), sidecar.read_bytes()
    result = training.run_fit(cache, job, folder, runtime=runtime, epochs=2, config=config, synthetic_fixture=True)
    assert result["complete"] and result["new_updates_this_invocation"] == 0
    assert (selected.read_bytes(), sidecar.read_bytes()) == before


def test_early_stop_retains_earliest_checkpoint_and_fixed_final_has_no_validation(fixture_cache, monkeypatch):
    cache = fixture_cache; job = first_job(cache)
    monkeypatch.setattr(training, "participant_validation_loss", lambda *a: 1.0)
    config = fixture_config(epochs=5, steps_per_epoch=1, patience=2)
    result = training.run_fit(cache, job, cache.root / "early", runtime={"device": "cpu"},
                              epochs=5, config=config, synthetic_fixture=True)
    assert result["epochs_completed"] == 3 and result["selected_epoch"] == 1
    assert result["stop_reason"] == "source_early_stopping"
    final = copy.deepcopy(job); final.update(phase="final", validation_indices=[])
    monkeypatch.setattr(training, "participant_validation_loss", lambda *a: pytest.fail("final used a validation set"))
    result = training.run_fit(cache, final, cache.root / "final", runtime={"device": "cpu"},
                              epochs=2, config=fixture_config(), synthetic_fixture=True)
    assert result["epochs_completed"] == result["selected_epoch"] == 2


def test_fixture_completion_and_missing_lodo_never_export_scientific_f0(fixture_cache):
    cache = fixture_cache; job = first_job(cache); plan = training.build_training_plan(cache)
    folder = cache.root / job["job_id"]
    training.run_fit(cache, job, folder, runtime={"device": "cpu"}, epochs=2,
                     config=fixture_config(), synthetic_fixture=True)
    with pytest.raises(training.SourceTrainingError, match="fixtures/paused"):
        training.verify_complete_fit(cache, job, folder)
    with pytest.raises(training.SourceTrainingError, match="engineering pilot"):
        training.export_f0_receipts(cache, plan, cache.root)
    assert not list(cache.root.rglob("f0_receipt*.json"))


@pytest.fixture
def complete_artifact_envelope_double(fixture_cache, monkeypatch):
    """Explicit mocked evidence envelope for verifier software tests only.

    The registered policy itself is replaced with a four-update math policy;
    these temporary flags exercise validation, never real SOURCE_F0 export.
    """
    cache = fixture_cache; config = fixture_config()
    monkeypatch.setattr(training, "SOURCE_CONFIG", copy.deepcopy(config))
    job = next(item for item in training.build_training_plan(cache)["jobs"] if item["phase"] == "final")
    folder = cache.root / "MOCK_COMPLETION_ENVELOPE_ONLY"
    training.run_fit(cache, job, folder, runtime={"device": "cpu", "scope": "MATHEMATICAL_FIXTURE_ONLY"},
                     epochs=2, config=config, synthetic_fixture=True)
    identity = json.loads((folder / "identity.json").read_text())
    identity.update(synthetic=False, device="cuda:0", runtime={"device": "cuda:0", "scope": "MOCK_ENVELOPE_NOT_REAL_GPU"})
    training.atomic_json(folder / "identity.json", identity)
    for name in ("last_state.pt", "selected.pt", "epoch_0001.pt", "epoch_0002.pt"):
        checkpoint = torch.load(folder / name, weights_only=True)
        checkpoint.update(identity=identity, identity_sha256=digest(identity))
        if name == "selected.pt": checkpoint["synthetic"] = False
        training.atomic_checkpoint(folder / name, checkpoint)
    for name in ("fit_summary.json", "training_uids.json", "source_validation_uids.json", "execution_progress.json"):
        value = json.loads((folder / name).read_text())
        if "synthetic" in value: value["synthetic"] = False
        if "identity_sha256" in value: value["identity_sha256"] = digest(identity)
        training.atomic_json(folder / name, value)
    def seal_mock_manifest():
        files = [path for path in folder.iterdir() if path.suffix in {".json", ".pt", ".npy"}
                 and path.name not in {"status.json", "artifact_manifest.json"}]
        training.atomic_json(folder / "artifact_manifest.json", {"identity_sha256": digest(identity),
            "files": [training.artifact(cache.root, path) for path in sorted(files)]})
        training.atomic_json(folder / "status.json", {"status": "SOURCE_FIT_COMPLETE", "synthetic": False,
            "identity_sha256": digest(identity), "artifact_manifest": training.artifact(cache.root, folder / "artifact_manifest.json")})
    seal_mock_manifest()
    return cache, job, folder, seal_mock_manifest


def test_completion_manifest_cannot_omit_predictions_or_epoch_archives(complete_artifact_envelope_double):
    cache, job, folder, seal = complete_artifact_envelope_double
    assert training.verify_complete_fit(cache, job, folder)["committed_optimizer_updates"] == 4
    (folder / "training_logits.npy").unlink(); seal()
    with pytest.raises(training.SourceTrainingError, match="artifact coverage"):
        training.verify_complete_fit(cache, job, folder)


def test_selected_weights_and_uid_cannot_drift_from_committed_evidence(complete_artifact_envelope_double):
    cache, job, folder, seal = complete_artifact_envelope_double
    selected = torch.load(folder / "selected.pt", weights_only=True)
    selected["model"][next(iter(selected["model"]))] += 1
    training.atomic_checkpoint(folder / "selected.pt", selected); seal()
    with pytest.raises(training.SourceTrainingError, match="selected-weight evidence"):
        training.verify_complete_fit(cache, job, folder)


def make_pilot_fixture(cache):
    folder = cache.root / "pilot_fixture"; folder.mkdir()
    runtime = {"device": "cuda:0", "device_name": "MATHEMATICAL_TEST_DOUBLE", "device_uuid": "FIXTURE_UUID",
        "device_total_memory": 0, "compute_capability": [0, 0], "torch": "FIXTURE_TORCH", "cuda_runtime": "FIXTURE_CUDA",
        "cudnn": 0, "driver_versions": ["FIXTURE_DRIVER"], "numpy": "FIXTURE_NUMPY", "python": "FIXTURE_PYTHON"}
    identity = {**runtime, "driver_versions_local_nvidia_smi": runtime["driver_versions"],
        "input_contract_sha256": cache.input_contract_sha256, "cache_manifest_sha256": cache.manifest_sha256,
        "tf32": False, "mixed_precision": False, "deterministic_algorithms": True}
    for key, relative in {"runner_sha256": "tools/run_controlled_pilot.py", "bridge_sha256": "p3work/f0_bridge.py",
                          "geometry_sha256": "p3work/spatial_geometry.py", "contracts_sha256": "p3work/contracts.py"}.items():
        identity[key] = file_digest(cache.root / relative)
    training.atomic_json(folder / "identity.json", identity)
    hashes = {}
    for name, update in (("engineering_initial.pt", 0), ("engineering_continuous100.pt", 100),
                         ("engineering_resume50.pt", 50), ("engineering_resumed100.pt", 100)):
        state = {"scope": "ENGINEERING_SOURCE_PILOT_ONLY", "identity": identity, "identity_sha256": digest(identity),
                 "update": update, "model": {"mathematical_fixture": torch.tensor([float(update)])},
                 "optimizer": {"step": update}, "sampler_rng": torch.tensor([1], dtype=torch.uint8),
                 "torch_rng": torch.tensor([2], dtype=torch.uint8), "cuda_rng": [], "python_rng": [1], "numpy_rng": [2]}
        training.atomic_checkpoint(folder / name, state); hashes[name] = file_digest(folder / name)
    receipt = {"status": "ENGINEERING_PILOT_COMPLETE", "scope": "ENGINEERING_SOURCE_PILOT_ONLY", "training_started": True,
        "actual_completed_updates": 200, "exact_resume_verified": True, "user_started_gpu_attested": True,
        "resume_comparison": {k: True for k in ("model", "optimizer", "sampler_rng", "torch_rng", "cuda_rng",
                                                   "python_rng", "numpy_rng", "update", "identity_sha256")}, "checkpoint_files": hashes}
    training.atomic_json(folder / "engineering_receipt.json", receipt)
    return "pilot_fixture/engineering_receipt.json", runtime


def test_pilot_requires_real_artifacts_200_updates_identity_precision_and_replay(fixture_cache):
    cache = fixture_cache; path, runtime = make_pilot_fixture(cache)
    proof = training.verify_engineering_pilot(cache, path, runtime=runtime)
    assert proof["actual_completed_updates"] == 200 and proof["exact_resume_verified"] is True
    with pytest.raises(training.SourceTrainingError, match="runtime differs"):
        training.verify_engineering_pilot(cache, path, runtime={**runtime, "device_uuid": "CHANGED"})
    receipt_path = cache.root / path; receipt = json.loads(receipt_path.read_text())
    receipt["actual_completed_updates"] = 199; training.atomic_json(receipt_path, receipt)
    with pytest.raises(training.SourceTrainingError, match="200-update"):
        training.verify_engineering_pilot(cache, path)


def test_pilot_rechecks_checkpoint_states_not_just_receipt_booleans(fixture_cache):
    cache = fixture_cache; path, _ = make_pilot_fixture(cache)
    receipt_path = cache.root / path; receipt = json.loads(receipt_path.read_text())
    resumed_path = receipt_path.with_name("engineering_resumed100.pt")
    resumed = torch.load(resumed_path, weights_only=True)
    resumed["model"]["mathematical_fixture"] += 1
    training.atomic_checkpoint(resumed_path, resumed)
    receipt["checkpoint_files"][resumed_path.name] = file_digest(resumed_path)
    training.atomic_json(receipt_path, receipt)
    with pytest.raises(training.SourceTrainingError, match="actually differ"):
        training.verify_engineering_pilot(cache, path)


def test_operational_wall_budget_amendment_preserves_fit_identity_and_resume(fixture_cache):
    cache = fixture_cache; output = cache.root / "operational"; job = first_job(cache)
    path, budget = training.operational_budget(output, initial_seconds=1)
    budget["consumed_active_seconds"] = 2; training.atomic_json(path, budget)
    before = training.fit_identity(cache, job, fixture_config(), {"device": "cpu"}, synthetic_fixture=True)
    with pytest.raises(training.SourceTrainingError, match="amendment note"):
        training.operational_budget(output, initial_seconds=1, amend_seconds=5)
    _, amended = training.operational_budget(output, initial_seconds=1, amend_seconds=5, amendment_note="Software fixture extension only")
    assert amended["revision"] == 2 and amended["consumed_active_seconds"] == 2
    assert before == training.fit_identity(cache, job, fixture_config(), {"device": "cpu"}, synthetic_fixture=True)
    paused = training.run_fit(cache, job, output / "fit", runtime={"device": "cpu"}, epochs=2,
        config=fixture_config(), deadline=time.monotonic() - 1, synthetic_fixture=True)
    assert paused["stop_reason"] == "operational_wall_budget_pause" and paused["committed_optimizer_updates"] == 0
    resumed = training.run_fit(cache, job, output / "fit", runtime={"device": "cpu"}, epochs=2,
        config=fixture_config(), synthetic_fixture=True)
    assert resumed["complete"] and resumed["committed_optimizer_updates"] == 4


def open_wall_fixture(tmp_path, *, consumed=3, high_water=None, initial_seconds=10,
                      last_utc=100.0, last_mono=50.0):
    """Pure metadata clock intent, never a model/data/GPU execution."""
    path, budget = training.operational_budget(tmp_path, initial_seconds=initial_seconds)
    budget.update(consumed_active_seconds=consumed, active_invocation=True, last_observed_unix=last_utc)
    budget["clock_state"].update(wall_high_water_seconds=consumed if high_water is None else high_water,
        interval={"base_wall_seconds": 3.0, "start_utc_seconds": 100.0, "start_monotonic_seconds": 50.0,
                  "last_observed_utc_seconds": last_utc, "last_observed_monotonic_seconds": last_mono,
                  "boot_id": "SYNTHETIC_BOOT", "intent": "persisted_before_work"})
    training.atomic_json(path, budget)
    return path, budget


def test_unclean_wall_restart_charges_unobserved_time_before_amendment(tmp_path, monkeypatch):
    path, budget = open_wall_fixture(tmp_path)
    monkeypatch.setattr(training, "_source_wall_clock", lambda: {
        "utc_seconds": 120.0, "monotonic_seconds": 70.0, "boot_id": "SYNTHETIC_BOOT"})
    _, recovered = training.operational_budget(tmp_path, initial_seconds=10)
    assert recovered["consumed_active_seconds"] == 23
    assert recovered["active_invocation"] is False
    assert recovered["clock_state"]["interval"] is None
    assert recovered["clock_state"]["wall_high_water_seconds"] == 23
    assert recovered["unclean_restart_charges"][0]["charged_seconds"] == 20
    _, amended = training.operational_budget(tmp_path, initial_seconds=10, amend_seconds=30,
        amendment_note="Explicit operational extension after interrupted mathematical fixture")
    assert amended["consumed_active_seconds"] == 23 and amended["authorized_total_seconds"] == 30


@pytest.mark.parametrize("clock,reason", [
    ({"utc_seconds": 90.0, "monotonic_seconds": 70.0, "boot_id": "SYNTHETIC_BOOT"}, "utc_clock_rolled_back"),
    ({"utc_seconds": 120.0, "monotonic_seconds": 40.0, "boot_id": "SYNTHETIC_BOOT"}, "monotonic_clock_rolled_back"),
    ({"utc_seconds": 120.0, "monotonic_seconds": 51.0, "boot_id": "SYNTHETIC_BOOT"}, "utc_monotonic_elapsed_disagreement_exceeds_5s"),
    ({"utc_seconds": 120.0, "monotonic_seconds": 70.0, "boot_id": "OTHER_BOOT"}, "boot_changed_or_unknown"),
])
def test_unclean_clock_rollback_disagreement_and_changed_boot_block_durably(tmp_path, monkeypatch, clock, reason):
    path, _ = open_wall_fixture(tmp_path)
    monkeypatch.setattr(training, "_source_wall_clock", lambda: clock)
    with pytest.raises(training.SourceTrainingError, match="work is blocked"):
        training.operational_budget(tmp_path, initial_seconds=10)
    blocked = json.loads(path.read_text())
    assert blocked["active_invocation"] is True
    assert blocked["consumed_active_seconds"] == blocked["clock_state"]["wall_high_water_seconds"] == 23
    assert reason in blocked["clock_state"]["clock_block"]["reasons"]
    before = path.read_bytes()
    monkeypatch.setattr(training, "_source_wall_clock", lambda: {
        "utc_seconds": 130.0, "monotonic_seconds": 80.0, "boot_id": "SYNTHETIC_BOOT"})
    with pytest.raises(training.SourceTrainingError, match="history is blocked"):
        training.operational_budget(tmp_path, initial_seconds=10, amend_seconds=100,
                                    amendment_note="Fixture amendment cannot excuse unknown time")
    assert path.read_bytes() == before


def test_clock_high_water_survives_rollback_above_start_and_lower_checkpoint_floor(tmp_path, monkeypatch):
    path, _ = open_wall_fixture(tmp_path, consumed=5, high_water=23, last_utc=120, last_mono=70)
    monkeypatch.setattr(training, "_source_wall_clock", lambda: {
        "utc_seconds": 115.0, "monotonic_seconds": 65.0, "boot_id": "SYNTHETIC_BOOT"})
    with pytest.raises(training.SourceTrainingError, match="work is blocked"):
        training.operational_budget(tmp_path, initial_seconds=10)
    saved = json.loads(path.read_text())
    assert saved["consumed_active_seconds"] == saved["clock_state"]["wall_high_water_seconds"] == 23
    assert set(saved["clock_state"]["clock_block"]["reasons"]) == {
        "utc_clock_rolled_back", "monotonic_clock_rolled_back"}


@pytest.mark.parametrize("legacy", [True, False])
def test_open_legacy_or_incomplete_clock_journal_does_not_get_zero_charge_reset(tmp_path, legacy):
    path, budget = training.operational_budget(tmp_path, initial_seconds=10)
    budget.update(consumed_active_seconds=3, active_invocation=True, last_observed_unix=100.0)
    if legacy:
        del budget["clock_state"]
    training.atomic_json(path, budget)
    with pytest.raises(training.SourceTrainingError, match="blocked"):
        training.operational_budget(tmp_path, initial_seconds=10)
    saved = json.loads(path.read_text())
    assert saved["active_invocation"] is True and saved["consumed_active_seconds"] == 3
    assert saved["clock_state"]["clock_block"] is not None


def test_rejected_amendment_preserves_recovered_clock_high_water(tmp_path, monkeypatch):
    path, _ = open_wall_fixture(tmp_path)
    monkeypatch.setattr(training, "_source_wall_clock", lambda: {
        "utc_seconds": 120.0, "monotonic_seconds": 70.0, "boot_id": "SYNTHETIC_BOOT"})
    with pytest.raises(training.SourceTrainingError, match="amendment note"):
        training.operational_budget(tmp_path, initial_seconds=10, amend_seconds=30)
    saved = json.loads(path.read_text())
    assert saved["consumed_active_seconds"] == saved["clock_state"]["wall_high_water_seconds"] == 23
    assert saved["active_invocation"] is False
    assert saved["authorized_total_seconds"] == 10 and len(saved["unclean_restart_charges"]) == 1
    _, recovered = training.operational_budget(tmp_path, initial_seconds=10)
    assert recovered["consumed_active_seconds"] == 23 and len(recovered["unclean_restart_charges"]) == 1


def test_invalid_operational_journal_fails_closed(tmp_path):
    path, budget = training.operational_budget(tmp_path, initial_seconds=10)
    for fields in ({"consumed_active_seconds": -1}, {"consumed_active_seconds": True},
                   {"authorized_total_seconds": 20}, {"revision": 4}, {"active_invocation": "true"},
                   {"clock_state": {**budget["clock_state"], "schema_version": "unknown"}}):
        training.atomic_json(path, {**budget, **fields})
        with pytest.raises(training.SourceTrainingError, match="Operational wall budget"):
            training.operational_budget(tmp_path, initial_seconds=10)


def test_ordinary_execution_error_closes_operational_timer_without_claiming_completion(fixture_cache, monkeypatch):
    cache = fixture_cache; plan = training.build_training_plan(cache)
    proof = {"receipt": {"path": "EXPLICIT_MOCK_PROOF_ONLY"}, "exact_resume_verified": True}
    monkeypatch.setattr(training, "verify_engineering_pilot", lambda *a, **k: proof)
    clock = {"utc": 100.0, "mono": 50.0}
    monkeypatch.setattr(training.time, "time", lambda: clock["utc"])
    monkeypatch.setattr(training.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setattr(training, "_boot_id", lambda: "SYNTHETIC_BOOT")
    def no_fit(*a, **k):
        clock.update(utc=107.0, mono=57.0)
        raise training.SourceTrainingError("Software test double: fail before a real fit")
    monkeypatch.setattr(training, "run_fit", no_fit)
    output = cache.root / "mock_error_path"
    with pytest.raises(training.SourceTrainingError, match="Software test double"):
        training.execute_plan(cache, plan, output, runtime={"device": "cuda:0", "scope": "MOCK_ONLY"},
                              pilot_evidence=proof, max_new_updates=1)
    wall = json.loads((output / "operational_budget.json").read_text())
    assert wall["active_invocation"] is False and wall["consumed_active_seconds"] == 7
    assert wall["clock_state"]["wall_high_water_seconds"] == 7 and wall["clock_state"]["interval"] is None
    assert not (output / "source_training_progress.json").exists()
    assert not list(output.rglob("f0_receipt*.json"))


def test_active_heartbeat_rollback_retains_high_water_and_finally_cannot_clear_block(fixture_cache, monkeypatch):
    cache = fixture_cache; plan = training.build_training_plan(cache)
    proof = {"receipt": {"path": "EXPLICIT_MOCK_PROOF_ONLY"}, "exact_resume_verified": True}
    monkeypatch.setattr(training, "verify_engineering_pilot", lambda *a, **k: proof)
    clock = {"utc": 100.0, "mono": 50.0}
    monkeypatch.setattr(training.time, "time", lambda: clock["utc"])
    monkeypatch.setattr(training.time, "monotonic", lambda: clock["mono"])
    monkeypatch.setattr(training, "_boot_id", lambda: "SYNTHETIC_BOOT")

    def no_fit(*args, **kwargs):
        clock.update(utc=120.0, mono=70.0); kwargs["heartbeat"]()
        clock.update(utc=115.0, mono=65.0); kwargs["heartbeat"]()
        pytest.fail("Rollback must block before returning a fit result")

    monkeypatch.setattr(training, "run_fit", no_fit)
    output = cache.root / "mock_clock_rollback"
    with pytest.raises(training.SourceTrainingError, match="history is blocked"):
        training.execute_plan(cache, plan, output, runtime={"device": "cuda:0", "scope": "MOCK_ONLY"},
                              pilot_evidence=proof, max_wall_seconds=100)
    wall = json.loads((output / "operational_budget.json").read_text())
    assert wall["active_invocation"] is True
    assert wall["consumed_active_seconds"] == wall["clock_state"]["wall_high_water_seconds"] == 20
    assert set(wall["clock_state"]["clock_block"]["reasons"]) == {
        "utc_clock_rolled_back", "monotonic_clock_rolled_back"}
    assert not (output / "source_training_progress.json").exists()


def test_dry_run_and_missing_double_flag_never_touch_cuda(fixture_cache, monkeypatch, capsys):
    cache = fixture_cache
    monkeypatch.setattr(cli, "verify_inputs", lambda root: cache)
    monkeypatch.setattr(cli, "cuda_runtime", lambda: pytest.fail("CUDA queried in dry run"))
    assert cli.main(["--root", str(cache.root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planned_core_total_fits"] == 25 and result["actual_fits_started_this_invocation"] == 0
    assert result["gpu_checked"] is False
    assert cli.main(["--root", str(cache.root), "--execute"]) == 2
    assert "Both --execute" in json.loads(capsys.readouterr().out)["reason"]


def test_real_source_entry_never_bypasses_w1_or_pilot_to_query_gpu(tmp_path, fixture_cache, monkeypatch, capsys):
    monkeypatch.setattr(cli, "cuda_runtime", lambda: pytest.fail("GPU queried before prerequisites"))
    assert cli.main(["--root", str(tmp_path), "--execute", "--user-started-gpu"]) == 2
    capsys.readouterr()
    monkeypatch.setattr(cli, "verify_inputs", lambda root: fixture_cache)
    assert cli.main(["--root", str(fixture_cache.root), "--execute", "--user-started-gpu"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["training_started_this_invocation"] is False and result["execution_attempted"] is False
