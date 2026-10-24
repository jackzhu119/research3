"""CPU orchestration checks on synthetic fixtures, never scientific results."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper3r5 import worker
from paper3r5.algorithm import Config, fit as fit_algorithm
from paper3r5.data import Catalog, load_source
from paper3r5.io import digest, put, read, sha
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import initialize_dictionary

SEEDS = (20261018, 20261019, 20261020)


@pytest.fixture(autouse=True)
def restore_process_settings(monkeypatch):
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.delenv("PAPER3R5_STOP_REQUESTED", raising=False)
    torch.set_num_threads(1)
    yield
    for sig, handler in handlers.items():
        signal.signal(sig, handler)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
     torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = cudnn


def write_project(root):
    """Three seed-specific frozen source receipts and disjoint participant roles."""
    contract = {"n_times": 128, "fixture": "SYNTHETIC_SOFTWARE_TEST_ONLY"}
    put(root / "contracts/input_contract.json", contract)
    contract_sha = digest(contract)
    def desc(path):
        return {"path": str(path.relative_to(root)), "sha256": sha(path)}
    np.save(root / "reference.npy", np.eye(10) / 10)
    for seed in SEEDS:
        torch.manual_seed(seed)
        path = root / f"source-{seed}.pt"
        torch.save({"model": CompactEEGNet(11, 128).state_dict(), "config": {"dropout": .25}}, path)
        receipt = root / f"runs/source_training_v2/f0_receipt_seed-{seed}.json"
        put(receipt, {"status": "SOURCE_F0_COMPLETE", "pipeline_seed": seed,
            "checkpoint": desc(path), "signal_contract_sha256": contract_sha})
        put(root / f"runs/training_campaign_v1/references/seed-{seed}/receipt.json", {
            "status": "SOURCE_REFERENCE_COMPLETE", "source_f0_receipt_sha256": sha(receipt),
            "signal_contract_sha256": contract_sha, "arrays": {"reference": desc(root / "reference.npy")}})
    roles = dict(zip(("meta_train", "meta_validation", "internal_holdout"), ("S1", "S2", "S3")))
    put(root / "split.json", {"partitions": {key: [person] for key, person in roles.items()}, "split_sha256": "b" * 64})
    sessions = []
    rng = np.random.default_rng(18)
    for role, person in roles.items():
        x = rng.normal(size=(100, 11, 128)).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        x -= x.mean(2, keepdims=True)
        np.save(root / f"{person}.npy", x)
        rows = [{"participant_id": person, "session_id": "1", "tasknumber": 1, "label": i % 2,
            "trial_uid": f"{person}-{i}", "event_uid": f"{person}-{i}", "original_trial_index": i,
            "physical_run_id": str(i // 20), "physical_run_complete_verified": True,
            "valid": i != 0, "cache_index": i if i else None} for i in range(100)]
        put(root / f"{person}.json", rows)
        sessions.append({"participant_id": person, "session_id": "1", "partition": role,
            "acquisition_order_verified": True, "X": desc(root / f"{person}.npy"),
            "events": desc(root / f"{person}.json")})
    put(root / "cache.json", {"status": "STIEGER_CACHE_COMPLETE", "participant_split": desc(root / "split.json"),
        "input_contract_sha256": contract_sha, "sessions": sessions})
    put(root / "status/tasks.json", {"W3": {"cache_manifest": desc(root / "cache.json")}})


def plan_fixture(tmp_path):
    write_project(tmp_path)
    basis = {"key": "basis-test", "family": "basis", "seed": SEEDS[0], "group": "test",
        "variant": "tangent_profiled", "config": {"variant": "tangent_profiled", "rank": 4,
        "tangent_steps": 1, "head_l2": .1}, "requires": [], "budgets": [20], "scenarios": ["clean"]}
    evaluation = {"key": "eval-test", "family": "core", "seed": SEEDS[0], "group": "test",
        "variant": "tangent_information", "basis_key": basis["key"], "requires": [basis["key"]],
        "config": {"variant": "tangent_information", "rank": 4, "tangent_steps": 1, "head_l2": .1},
        "budgets": [20, 4], "scenarios": ["clean", "gain10", "label_noise10"]}
    plan = {"root": str(tmp_path), "output": str(tmp_path / "out"), "recipe_sha256": "c" * 64,
        "updates": 1, "validate_every": 50, "checkpoint_every": 1, "budgets": [20, 4],
        "tasks": [basis, evaluation]}
    put(tmp_path / "plan.json", plan)
    return plan, Namespace(plan=str(tmp_path / "plan.json"), task=0, device="cpu")


def test_basis_then_evaluation_real_fit_metrics_batching_and_reuse(tmp_path, monkeypatch):
    plan, args = plan_fixture(tmp_path)
    assert worker.run(args) == 0
    basis = Path(plan["output"]) / "basis-test"
    result = read(basis / "result.json")
    assert result["status"] == "COMPLETE" and result["outer_updates"] == 1
    assert result["holdout_used_for_selection"] is False
    assert result["selection"]["outer_update"] == 1
    assert not (basis / "evaluations/holdout").exists()
    assert read(basis / "latest.json")["outer_update"] == 1
    original_predict = worker.predict
    batches = []
    def tracked_predict(source, geometry, b, fitted, query_x):
        batches.append(len(query_x))
        return original_predict(source, geometry, b, fitted, query_x)
    monkeypatch.setattr(worker, "predict", tracked_predict)
    args.task = 1
    assert worker.run(args) == 0
    assert max(batches) == 64 and 16 in batches
    output = Path(plan["output"]) / "eval-test"
    rows = [read(output / f"evaluations/holdout/B{budget}/clean/S3-1.json") for budget in (20, 4)]
    assert rows[0]["query_signature"] == rows[1]["query_signature"]
    assert len(rows[0]["predictions"]) == 80
    for row in rows:
        assert row["ba"] - row["f0_ba"] == pytest.approx(row["corrected_error_rate"] - row["negative_flip_rate"])
        assert row["fallback_reason"] is None
        assert row["diagnostics"]["effective_dimension"] >= 0
        assert row["diagnostics"]["actual_nonlinear_surrogate_margin_rmse"] >= 0
        assert row["partition"] == "holdout"
        summary = read(output / f"evaluations/holdout/B{row['budget']}/clean/summary.json")
        assert summary["expected_sessions"] == [{"participant_id": "S3", "session_id": "1",
                                                 "query_signature": row["query_signature"]}]
    assert (output / "evaluations/holdout/B20/gain10/summary.json").exists()
    assert not (output / "evaluations/holdout/B4/gain10").exists()
    saved = {path: path.read_bytes() for path in basis.glob("*.pt")}
    assert worker.run(args) == 0
    assert saved == {path: path.read_bytes() for path in saved}


def test_reference_never_enters_outer_training(tmp_path, monkeypatch):
    plan, args = plan_fixture(tmp_path)
    task = plan["tasks"][1]
    task.update(family="reference", variant="support_EA", requires=[])
    task.pop("basis_key")
    task["config"]["variant"] = "support_EA"
    task["scenarios"] = ["clean"]
    plan["tasks"] = [task]
    put(args.plan, plan)
    def forbidden(*args, **kwargs):
        pytest.fail("Fixed reference entered meta training")
    monkeypatch.setattr(worker, "train_loop", forbidden)
    monkeypatch.setattr(worker, "fit", forbidden)
    assert worker.run(args) == 0
    output = Path(plan["output"]) / task["key"]
    assert read(output / "selected.json")["status"] == "FIXED_REFERENCE_NO_MODEL_SELECTION"
    row = read(output / "evaluations/holdout/B4/clean/S3-1.json")
    assert row["reference_fit"]["uses_support_labels"] is False
    assert not (output / "latest.json").exists()


def test_stop_before_loading_returns75_and_does_not_touch_dependencies(tmp_path, monkeypatch):
    plan, args = plan_fixture(tmp_path)
    monkeypatch.setenv("PAPER3R5_STOP_REQUESTED", "1")
    monkeypatch.setattr(worker, "load_source", lambda *args: pytest.fail("stop request ignored"))
    assert worker.run(args) == 75
    output = Path(plan["output"]) / plan["tasks"][0]["key"]
    assert read(output / "progress.json")["status"] == "PAUSED_CHECKPOINT_SAVED"
    assert not (output / "result.json").exists()


@pytest.mark.parametrize("corrupt_latest", [False, True])
def test_checkpoint_identity_rejection_preserves_rng_and_files(tmp_path, corrupt_latest):
    for update in (1, 2):
        worker.save_checkpoint(tmp_path, {"identity": "original", "update": update})
    if corrupt_latest:
        (tmp_path / read(tmp_path / "latest.json")["path"]).write_bytes(b"torn")
    saved = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(ValueError, match="different numerical experiment"):
        worker.load_checkpoint(tmp_path, expected_identity="different")
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1]) and after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert saved == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_torn_generation_recovery_preserves_good_generation_on_next_failed_save(tmp_path, monkeypatch):
    for update in (1, 2):
        worker.save_checkpoint(tmp_path, {"identity": "same", "update": update})
    (tmp_path / read(tmp_path / "latest.json")["path"]).write_bytes(b"torn")
    state = worker.load_checkpoint(tmp_path, expected_identity="same")
    assert state["update"] == 1
    original = worker.atomic
    def torn(path, writer):
        if str(path).endswith(".pt"):
            Path(path).write_bytes(b"torn again")
            raise IOError("injected write interruption")
        return original(path, writer)
    monkeypatch.setattr(worker, "atomic", torn)
    with pytest.raises(IOError):
        worker.save_checkpoint(tmp_path, state)
    assert worker.load_checkpoint(tmp_path)["update"] == 1


def test_actual_four_episode_update_exact_resume_and_interrupted_validation(tmp_path):
    write_project(tmp_path)
    source, geometry, _ = load_source(tmp_path, SEEDS[0], "cpu")
    source.eval()
    catalog = Catalog(tmp_path, "meta_train", cache_gb=.01)
    cfg = Config(variant="tangent_profiled", tangent_steps=1, head_l2=.1)
    initial = initialize_dictionary(4, SEEDS[0])
    def fresh():
        return {"identity": "resume", "b": initial.clone().requires_grad_(True), "update": 0,
                "validated_update": 0, "history": []}
    def step(state):
        return worker.outer_update(source, geometry, state["b"], catalog, SEEDS[0], state["update"], cfg, "cpu")
    def score(state):
        return {"participant_mean_ba": .6, "participant_mean_bce": .5}
    continuous = worker.train_loop(state=fresh(), step=step, validate=score, save=lambda _: None,
        stop=threading.Event(), total_updates=2, checkpoint_every=1, validate_every=2)
    state = fresh()
    def interrupted(state):
        raise worker.Pause()
    with pytest.raises(worker.Pause):
        worker.train_loop(state=state, step=step, validate=interrupted,
            save=lambda value: worker.save_checkpoint(tmp_path / "cp", value), stop=threading.Event(),
            total_updates=2, checkpoint_every=1, validate_every=2)
    restored = worker.load_checkpoint(tmp_path / "cp", expected_identity="resume")
    restored["b"].requires_grad_(True)
    def forbidden(state):
        pytest.fail("Committed update replayed after interrupted validation")
    resumed = worker.train_loop(state=restored, step=forbidden, validate=score, save=lambda _: None,
        stop=threading.Event(), total_updates=2, checkpoint_every=1, validate_every=2)
    assert torch.equal(resumed["b"], continuous["b"])
    assert resumed["validated_update"] == 2 and [row["outer_update"] for row in resumed["history"]] == [2]


def test_evaluation_row_resume_keeps_finished_session_and_programming_errors_visible(tmp_path, monkeypatch):
    write_project(tmp_path)
    source, geometry, inputs = load_source(tmp_path, SEEDS[0], "cpu")
    source.eval()
    catalog = Catalog(tmp_path, "internal_holdout")
    b = initialize_dictionary(4, SEEDS[0])
    cfg = Config(variant="tangent_profiled", tangent_steps=1, head_l2=.1)
    output = tmp_path / "evaluation"
    worker.evaluate(source, geometry, b, catalog, cfg, output, threading.Event(), {}, source_identity=inputs)
    before = (output / "S3-1.json").read_bytes()
    monkeypatch.setattr(worker, "fit", lambda *a, **kw: pytest.fail("Completed session was recalibrated"))
    worker.evaluate(source, geometry, b, catalog, cfg, output, threading.Event(), {}, source_identity=inputs)
    assert before == (output / "S3-1.json").read_bytes()
    corrupt = read(output / "S3-1.json")
    corrupt["event_uids"][0] = "wrong-query-event"
    corrupt["query_signature"] = digest({"uids": corrupt["event_uids"], "labels": corrupt["labels"]})
    put(output / "S3-1.json", corrupt)
    monkeypatch.setattr(worker, "fit", fit_algorithm)
    worker.evaluate(source, geometry, b, catalog, cfg, output, threading.Event(), {}, source_identity=inputs)
    repaired = read(output / "S3-1.json")
    assert repaired["query_signature"] == read(output / "summary.json")["expected_sessions"][0]["query_signature"]
    assert repaired["event_uids"][0] != "wrong-query-event"
    def bug(*a, **kw):
        raise TypeError("injected programming error")
    monkeypatch.setattr(worker, "fit", bug)
    with pytest.raises(TypeError, match="programming error"):
        worker.evaluate(source, geometry, b, catalog, cfg, output, threading.Event(), {},
                        source_identity={**inputs, "changed": True})


def test_completed_task_identity_mismatch_is_failure_without_checkpoint_overwrite(tmp_path):
    plan, args = plan_fixture(tmp_path)
    assert worker.run(args) == 0
    output = Path(plan["output"]) / "basis-test"
    saved = {path.name: path.read_bytes() for path in output.iterdir() if path.suffix == ".pt" or path.name == "latest.json"}
    plan["recipe_sha256"] = "d" * 64
    put(args.plan, plan)
    assert worker.run(args) == 1
    assert "different numerical experiment" in read(output / "error.json")["reason"]
    assert saved == {name: (output / name).read_bytes() for name in saved}


def test_legacy_reuse_matches_inputs_and_reads_best_only_without_rng_or_file_changes(tmp_path):
    write_project(tmp_path)
    _, _, inputs = load_source(tmp_path, SEEDS[0], "cpu")
    catalog = Catalog(tmp_path, "meta_train")
    old_output = tmp_path / "runs/followup_20261023_v2"
    old_task = {"key": f"P-seed{SEEDS[0]}", "seed": SEEDS[0], "arm": "core",
                "config": {"variant": "P", "rank": 4, "radius": .15}}
    old_plan = {"recipe_sha256": "e" * 64, "tasks": [old_task]}
    folder = old_output / old_task["key"]
    identity = digest({"task": old_task, "data": catalog.identity, "source": inputs,
                       "recipe": old_plan["recipe_sha256"]})
    score = {"participant_mean_ba": .61, "participant_mean_bce": .7, "outer_update": 500}
    selected = {"status": "SELECTED_ON_VALIDATION", "identity": identity, "task": old_task,
                "holdout_used_for_selection": False, "selection": score}
    b = initialize_dictionary(4, 3)
    put(old_output / "plan.json", old_plan)
    put(folder / "selected.json", selected)
    put(folder / "result.json", {**selected, "status": "COMPLETE", "outer_updates": 1000})
    worker.save_checkpoint(folder, {"identity": identity, "update": 1000, "validated_update": 1000,
        "b": initialize_dictionary(4, 4), "best_b": b, "best_score": score})
    saved = {path.name: path.read_bytes() for path in folder.iterdir()}
    random.seed(291)
    np.random.seed(292)
    torch.manual_seed(293)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    reused = worker.try_reuse(tmp_path, Config(variant="sgd10"), SEEDS[0], inputs, catalog.identity)
    assert reused is not None and torch.equal(reused[0], b) and reused[1] == score
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert saved == {path.name: path.read_bytes() for path in folder.iterdir()}
    assert worker.try_reuse(tmp_path, Config(variant="sgd10", head_l2=.1), SEEDS[0], inputs, catalog.identity) is None
    assert worker.try_reuse(tmp_path, Config(variant="sgd10"), SEEDS[0], {**inputs, "signal_contract_sha256": "wrong"}, catalog.identity) is None


def test_dependency_rejects_stale_recipe_and_does_not_restore_rng(tmp_path):
    plan, args = plan_fixture(tmp_path)
    assert worker.run(args) == 0
    _, _, inputs = load_source(tmp_path, SEEDS[0], "cpu")
    catalog = Catalog(tmp_path, "meta_train")
    torch.manual_seed(861)
    before = torch.get_rng_state().clone()
    b, _ = worker.load_basis(plan, plan["tasks"][1], "cpu", source_identity=inputs, data_identity=catalog.identity)
    assert b.shape == (99, 4) and torch.equal(torch.get_rng_state(), before)
    plan["recipe_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different source, data, or recipe"):
        worker.load_basis(plan, plan["tasks"][1], "cpu", source_identity=inputs, data_identity=catalog.identity)
    assert torch.equal(torch.get_rng_state(), before)


def test_single_class_support_produces_explicit_fallback_and_zero_meta_gradient(tmp_path, monkeypatch):
    write_project(tmp_path)
    source, geometry, _ = load_source(tmp_path, SEEDS[0], "cpu")
    source.eval()
    catalog = Catalog(tmp_path, "meta_train")
    original = catalog.materialize
    def single_class(*args, **kwargs):
        episode = original(*args, **kwargs)
        episode.support = replace(episode.support, y=torch.zeros_like(episode.support.y))
        return episode
    monkeypatch.setattr(catalog, "materialize", single_class)
    b = initialize_dictionary(4, 1).requires_grad_(True)
    updated, metrics = worker.outer_update(source, geometry, b, catalog, SEEDS[0], 0, Config(), "cpu")
    assert metrics["meta_gradient_norm"] == 0
    torch.testing.assert_close(updated, b)


def test_standalone_external_plan_loads_original_basis_without_rng_restore(tmp_path):
    plan, args = plan_fixture(tmp_path)
    put(Path(plan["output"]) / "plan.json", plan)
    assert worker.run(args) == 0
    _, _, inputs = load_source(tmp_path, SEEDS[0], "cpu")
    data_identity = Catalog(tmp_path, "meta_train").identity
    external_task = {**plan["tasks"][1], "family": "transfer", "basis_output": plan["output"], "requires": []}
    external_plan = {**plan, "output": str(tmp_path / "external-out"), "recipe_sha256": "f" * 64,
                     "tasks": [external_task]}
    torch.manual_seed(231)
    before = torch.get_rng_state().clone()
    b, dependency = worker.load_basis(external_plan, external_task, "cpu",
        source_identity=inputs, data_identity=data_identity)
    assert b.shape == (99, 4) and torch.equal(before, torch.get_rng_state())
    assert dependency["basis_output"] == str(Path(plan["output"]).resolve())


def test_signal_at_session_boundary_preserves_row_for_resume(tmp_path, monkeypatch):
    write_project(tmp_path)
    source, geometry, inputs = load_source(tmp_path, SEEDS[0], "cpu")
    source.eval()
    catalog = Catalog(tmp_path, "internal_holdout")
    b = initialize_dictionary(4, 1)
    cfg = Config(tangent_steps=1, head_l2=.1)
    stop = threading.Event()
    original = worker.predict
    def request_stop(*args, **kwargs):
        stop.set()
        return original(*args, **kwargs)
    monkeypatch.setattr(worker, "predict", request_stop)
    folder = tmp_path / "row-pause"
    with pytest.raises(worker.Pause):
        worker.evaluate(source, geometry, b, catalog, cfg, folder, stop, {}, source_identity=inputs)
    assert (folder / "S3-1.json").exists() and not (folder / "summary.json").exists()
    def forbidden(*args, **kwargs):
        pytest.fail("Finished interrupted session was recalibrated")
    monkeypatch.setattr(worker, "fit", forbidden)
    assert worker.evaluate(source, geometry, b, catalog, cfg, folder,
        threading.Event(), {}, source_identity=inputs)["sessions"] == 1


def test_production_isolated_entry_executes_worker(tmp_path):
    plan, args = plan_fixture(tmp_path)
    entry = Path(__file__).resolve().parents[1] / "paper3r5/entry.py"
    completed = subprocess.run([sys.executable, "-I", "-S", "-B", "-u", str(entry), "worker",
        "--plan", args.plan, "--task", "0", "--device", "cpu"], cwd=tmp_path,
        text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = read(Path(plan["output"]) / "basis-test/result.json")
    assert result["status"] == "COMPLETE" and result["outer_updates"] == 1
