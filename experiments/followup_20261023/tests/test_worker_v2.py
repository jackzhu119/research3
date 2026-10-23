"""Actual CPU orchestration tests; synthetic fixtures are not scientific results."""
from argparse import Namespace
import copy
from pathlib import Path
import random
import signal
import threading

import numpy as np
import pytest
import torch

import START
from paper3v2 import worker
from paper3v2.algorithm import Config, init_gate, outer_update
from paper3v2.io import put, read
from test_algorithm import fixture
from test_pipeline import write_project


@pytest.fixture(autouse=True)
def preserve_process_settings():
    """run() installs production handlers; never leave them in the pytest process."""
    handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.set_num_threads(2)
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = cudnn


def short_reference_plan(tmp_path, variant):
    write_project(tmp_path)
    plan = START.make_plan(tmp_path, tmp_path / "followup")
    task = next(t for t in plan["tasks"] if t["config"]["variant"] == variant and t["seed"] == 20261018)
    task["budgets"] = [20, 4]
    task["diagnostics"] = []
    plan["tasks"] = [task]
    plan["budgets"] = [20, 4]
    put(tmp_path / "plan.json", plan)
    return plan, Namespace(plan=str(tmp_path / "plan.json"), task=0, device="cpu")


@pytest.mark.parametrize("variant", ["head_only", "support_EA"])
def test_reference_worker_completes_real_cpu_evaluation_without_meta_fit(tmp_path, monkeypatch, variant):
    plan, args = short_reference_plan(tmp_path, variant)
    monkeypatch.delenv("PAPER3V2_STOP_REQUESTED", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Fixed reference must never enter outer optimization or validation selection")

    monkeypatch.setattr(worker, "outer_update", forbidden)
    monkeypatch.setattr(worker, "train_loop", forbidden)
    assert worker.run(args) == 0
    out = Path(plan["output"]) / plan["tasks"][0]["key"]
    result = read(out / "result.json")
    selected = read(out / "selected.json")
    assert result["status"] == "COMPLETE" and result["outer_updates"] == 0
    assert selected["status"] == "FIXED_REFERENCE_NO_MODEL_SELECTION"
    assert selected["selection"] is None and selected["holdout_used_for_selection"] is False
    assert not (out / "validation").exists()
    rows = [read(out / f"holdout/B{budget}/S3-1.json") for budget in (20, 4)]
    assert rows[0]["event_uids"] == rows[1]["event_uids"]
    assert rows[0]["query_signature"] == rows[1]["query_signature"]
    for budget, row in zip((20, 4), rows):
        assert row["budget"] == budget and len(row["predictions"]) == 40
        assert np.asarray(row["probabilities"]).shape == (40, 2)
        assert np.asarray(row["f0_probabilities"]).shape == (40, 2)
        assert row["reference_fit"]["fallback_reason"] is None
        assert row["reference_fit"]["uses_support_labels"] is (variant == "head_only")
        assert row["reference_fit"]["optimizer_updates"] == (10 if variant == "head_only" else 0)
        summary = read(out / f"holdout/B{budget}/summary.json")
        assert summary["expected_sessions"] == [{"participant_id": "S3", "session_id": "1",
                                                  "query_signature": row["query_signature"]}]
    saved = {p.name: p.read_bytes() for p in out.iterdir() if p.name == "latest.json" or p.suffix == ".pt"}
    assert worker.run(args) == 0
    assert saved == {name: (out / name).read_bytes() for name in saved}


@pytest.mark.parametrize("corrupt_latest", [False, True])
def test_checkpoint_identity_rejection_preserves_rng_and_recovery_pointer(tmp_path, corrupt_latest):
    for update in (1, 2):
        worker.save_checkpoint(tmp_path, {"identity": "original", "update": update})
    if corrupt_latest:
        (tmp_path / read(tmp_path / "latest.json")["path"]).write_bytes(b"torn new checkpoint")
    saved = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    random.seed(873); np.random.seed(874); torch.manual_seed(875)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(ValueError, match="different numerical experiment"):
        worker.load_checkpoint(tmp_path, expected_identity="incompatible")
    assert random.getstate() == before[0]
    after = np.random.get_state()
    assert after[0] == before[1][0] and np.array_equal(after[1], before[1][1])
    assert after[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert saved == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_worker_mismatched_identity_does_not_overwrite_either_generation(tmp_path, monkeypatch):
    plan, args = short_reference_plan(tmp_path, "head_only")
    monkeypatch.delenv("PAPER3V2_STOP_REQUESTED", raising=False)
    out = Path(plan["output"]) / plan["tasks"][0]["key"]
    out.mkdir(parents=True)
    for update in (3, 4):
        worker.save_checkpoint(out, {"identity": "different-experiment", "update": update})
    saved = {p.name: p.read_bytes() for p in out.iterdir()}
    assert worker.run(args) == 1
    assert read(out / "error.json")["error_type"] == "ValueError"
    assert "different numerical experiment" in read(out / "error.json")["reason"]
    assert saved == {name: (out / name).read_bytes() for name in saved}
    assert not (out / "result.json").exists()


def test_real_dual_functional_updates_pause_resume_preserves_adam_exactly(tmp_path):
    _, source, geometry, episode, initial_b = fixture()
    config = Config(variant="dual", inner_steps=2)
    initial_gate = init_gate(config, "cpu")

    def fresh_state():
        return {"identity": "synthetic-dual-resume", "update": 0, "validated_update": 0,
                "history": [], "b": initial_b.detach().clone().requires_grad_(True),
                "gate": initial_gate.detach().clone().requires_grad_(True)}

    def step(state):
        # The interrupted training loop must keep the previous committed moments
        # intact, including when the next functional optimizer call is evaluated.
        prior = copy.deepcopy(state.get("gate_optimizer"))
        answer = outer_update(source, geometry, state["b"], state["gate"], [episode],
                              config, gate_state=state.get("gate_optimizer"))
        if prior is not None:
            assert state["gate_optimizer"]["step"] == prior["step"]
            for key in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(state["gate_optimizer"][key], prior[key])
        return answer

    def no_validation(*args):
        pytest.fail("This short exact-resume test never reaches the 50-update validation boundary")

    continuous = worker.train_loop(state=fresh_state(), step=step, validate=no_validation,
        save=lambda _: None, stop=threading.Event(), total_updates=4, checkpoint_every=1)
    stop = threading.Event()

    def stop_after_second_update(state):
        answer = step(state)
        if state["update"] == 1:
            stop.set()
        return answer

    interrupted = fresh_state()
    with pytest.raises(worker.Pause):
        worker.train_loop(state=interrupted, step=stop_after_second_update,
            validate=no_validation, save=lambda state: worker.save_checkpoint(tmp_path, state),
            stop=stop, total_updates=4, checkpoint_every=1)
    assert interrupted["update"] == 2
    restored = worker.load_checkpoint(tmp_path, expected_identity="synthetic-dual-resume")
    assert restored["update"] == 2 and restored["gate_optimizer"]["step"] == 2
    for key in ("b", "gate"):
        restored[key].requires_grad_(True)
    resumed = worker.train_loop(state=restored, step=step, validate=no_validation,
        save=lambda state: worker.save_checkpoint(tmp_path, state), stop=threading.Event(),
        total_updates=4, checkpoint_every=1)
    assert resumed["update"] == continuous["update"] == 4
    for key in ("b", "gate"):
        assert torch.equal(resumed[key], continuous[key])
    assert resumed["gate_optimizer"]["step"] == continuous["gate_optimizer"]["step"] == 4
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(resumed["gate_optimizer"][key], continuous["gate_optimizer"][key])
    assert torch.count_nonzero(resumed["gate_optimizer"]["exp_avg"]) > 0
