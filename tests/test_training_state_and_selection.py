"""Software-only interruption fixtures; no real data, CUDA or scientific fit."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from p3work.meta_training import MetaJob, execute_meta_job, run_outer_loop, search_plan, verify_meta_inputs
from p3work.selection import choose_checkpoint, choose_strongest_control, participant_equal_metrics
from p3work.training_state import capture_rng, load_checkpoint, restore_rng, save_checkpoint


IDENTITY = {"scope": "SOFTWARE_TEST_DOUBLE", "synthetic": True, "code_sha256": "a" * 64}


class ToySampler:
    def __init__(self):
        self.cursor = 0

    def next(self):
        self.cursor += 1
        return self.cursor

    def state_dict(self):
        return {"cursor": self.cursor, "scope": "SOFTWARE_TEST_DOUBLE"}

    def load_state_dict(self, state):
        assert state["scope"] == "SOFTWARE_TEST_DOUBLE"
        self.cursor = state["cursor"]


def seed():
    random.seed(20261018)
    np.random.seed(20261018)
    torch.manual_seed(20261018)


def toy_step(value, episodes):
    noise = torch.randn_like(value) + random.random() + float(np.random.random())
    return (value + noise * 1e-4 + sum(episodes) * 1e-8).detach().requires_grad_(True), {
        "full_unroll": True, "inner_updates_actual": 30, "scope": "SOFTWARE_TEST_DOUBLE"}


def score(_):
    return {"participant_mean_ba": .5, "participant_mean_bce": 1.0, "scope": "SOFTWARE_TEST_DOUBLE"}


def test_exact_rng_checkpoint_roundtrip_and_corruption_rejected(tmp_path):
    seed()
    p = tmp_path / "s.pt"
    save_checkpoint(p, identity=IDENTITY, state={"cursor": 3})
    expected = (random.random(), np.random.random(), torch.rand(3))
    state = load_checkpoint(p, identity=IDENTITY)
    assert state["cursor"] == 3
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    with pytest.raises(ValueError):
        load_checkpoint(p, identity={**IDENTITY, "code_sha256": "b" * 64})
    p.write_bytes(p.read_bytes() + b"torn write")
    with pytest.raises(ValueError):
        load_checkpoint(p, identity=IDENTITY)


def test_hundred_update_resume_exact_all_rng_and_sampler_cursor(tmp_path):
    seed()
    initial = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    uninterrupted, full = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step,
        evaluate=score, identity=IDENTITY, output=tmp_path / "full", total_updates=100, max_wall_seconds=10000)
    seed()
    clock = [0.0]

    def measured(value, episodes):
        clock[0] += 1
        return toy_step(value, episodes)

    sampler = ToySampler()
    _, partial = run_outer_loop(parameter=initial, sampler=sampler, step=measured,
        evaluate=score, identity=IDENTITY, output=tmp_path / "resume", total_updates=100,
        max_wall_seconds=50, monotonic=lambda: clock[0], utc_now=lambda: 1000000000 + clock[0], boot_id="SOFTWARE_TEST_DOUBLE")
    assert partial["status"] == "PAUSED_WALL_BUDGET" and partial["outer_updates_completed"] == 50
    assert sampler.cursor == 200 and partial["inner_updates_actual"] == 1500
    resumed, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step,
        evaluate=score, identity=IDENTITY, output=tmp_path / "resume", total_updates=100,
        max_wall_seconds=10000, resume=tmp_path / "resume/outer-0050.pt",
        wall_amendment_note="SOFTWARE_TEST_DOUBLE explicit operational wall extension")
    assert torch.equal(uninterrupted, resumed)
    assert result["outer_updates_completed"] == 100 and result["validation_checkpoints"] == 2
    assert result["selected_checkpoint"]["outer_update"] == 50
    assert result["episodes_completed"] == 400 and result["inner_updates_actual"] == 3000


def test_failed_outer_step_rolls_back_sampler_and_rng_before_snapshot(tmp_path):
    seed()
    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    baseline, _ = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "baseline", total_updates=100, max_wall_seconds=10000)
    seed()
    sampler = ToySampler()
    calls = [0]

    def interrupted(value, episodes):
        calls[0] += 1
        candidate = toy_step(value, episodes)
        if calls[0] == 7:
            raise RuntimeError("SOFTWARE_TEST_DOUBLE interrupted during outer step")
        return candidate

    with pytest.raises(RuntimeError):
        run_outer_loop(parameter=initial, sampler=sampler, step=interrupted, evaluate=score,
            identity=IDENTITY, output=tmp_path / "interrupted", total_updates=100, max_wall_seconds=10000)
    assert sampler.cursor == 24
    resumed, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "interrupted", total_updates=100, max_wall_seconds=10000,
        resume=tmp_path / "interrupted/outer-0006.pt")
    assert torch.equal(resumed, baseline) and result["outer_updates_completed"] == 100


def test_interrupted_validation_is_completed_before_next_outer_update(tmp_path):
    calls = [0]

    def interrupted_score(value):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("SOFTWARE_TEST_DOUBLE interrupted validation")
        return score(value)

    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError):
        run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=interrupted_score,
            identity=IDENTITY, output=tmp_path, total_updates=100, max_wall_seconds=10000)
    _, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path, total_updates=100, max_wall_seconds=10000,
        resume=tmp_path / "outer-0050.pt")
    assert [r["outer_update"] for r in result["validation"]] == [50, 100]


def test_registered_search_and_no_duplicate_selected_seed():
    for method in ("P", "B8", "B10", "B11"):
        plan = search_plan(method)
        assert len(plan["jobs"]) == 9 and len({j["job_sha256"] for j in plan["jobs"]}) == 9
        assert plan["full_five_seed_inner_updates"] == 520000
        assert plan["selected_screening_seed_reused"] is True
    with pytest.raises(ValueError):
        MetaJob("B10", 8, .15)
    with pytest.raises(ValueError):
        MetaJob("P", 2, .15, phase="confirmation", seed=20261018)
    with pytest.raises(ValueError):
        MetaJob("B9", 2, .15)


def test_selection_exact_ba_bce_early_ties_and_participant_equal_sessions():
    rows = [{"outer_update": 50, "participant_mean_ba": .8, "participant_mean_bce": .4},
            {"outer_update": 100, "participant_mean_ba": .8, "participant_mean_bce": .3},
            {"outer_update": 150, "participant_mean_ba": .8, "participant_mean_bce": .3}]
    assert choose_checkpoint(rows)["outer_update"] == 100
    sessions = [{"participant_id": "synthetic-a", "session_id": "1", "ba": 1., "balanced_bce": .2},
                {"participant_id": "synthetic-a", "session_id": "2", "ba": 0., "balanced_bce": .8},
                {"participant_id": "synthetic-b", "session_id": "1", "ba": 1., "balanced_bce": .2}]
    result = participant_equal_metrics(sessions, expected_participants={"synthetic-a", "synthetic-b"})
    assert result["participant_mean_ba"] == .75
    with pytest.raises(ValueError):
        participant_equal_metrics(sessions, expected_participants={"synthetic-a", "synthetic-b", "missing-synthetic"})


def test_strongest_control_requires_all_methods_seeds_and_common_measured_device():
    rows = [{"role": "meta_validation", "measured": True, "method": f"B{i}", "seed": seed,
             "participant_mean_ba": .8 if i == 11 else .798, "calibration_median_seconds": .1 if i in {1, 2} else .5,
             "device_identity_sha256": "a" * 64}
            for i in range(1, 12) for seed in (20261018, 20261019, 20261020)]
    assert choose_strongest_control(rows)["method"] == "B1"
    with pytest.raises(ValueError):
        choose_strongest_control(rows[:-1])
    changed = deepcopy(rows)
    changed[0]["device_identity_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        choose_strongest_control(changed)
    changed[0]["role"] = "internal_holdout"
    with pytest.raises(PermissionError):
        choose_strongest_control(changed)


def test_real_execution_gate_before_any_cuda_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA gate should not be reached"))
    with pytest.raises(PermissionError):
        execute_meta_job(tmp_path, job=MetaJob("P", 2, .15), source_receipt="x", reference_receipt="x",
                         split_path="x", budget_path="x")


def test_durable_observed_work_and_wall_high_water_survive_model_snapshot_replay(tmp_path):
    import json
    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    seed()
    expected, _ = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "baseline", total_updates=100, max_wall_seconds=10000)
    seed()
    clock = [0.0]

    def step(value, episodes):
        clock[0] += 1
        return toy_step(value, episodes)

    output = tmp_path / "resume"
    run_outer_loop(parameter=initial, sampler=ToySampler(), step=step, evaluate=score, identity=IDENTITY,
        output=output, total_updates=100, max_wall_seconds=50, monotonic=lambda: clock[0],
        utc_now=lambda: 1000000000 + clock[0], boot_id="SOFTWARE_TEST_DOUBLE")
    # Simulate durable observations from three successful updates after a last
    # atomic model snapshot; the real SIGKILL wall journal is tested separately.
    p = output / "progress.json"
    progress = json.loads(p.read_text())
    progress.update(outer_updates_completed=53, confirmed_outer_updates_observed=53,
                    outer_attempts_dispatched=53, active_wall_seconds=60)
    p.write_text(json.dumps(progress))
    actual, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=output, total_updates=100, max_wall_seconds=10000,
        resume=output / "outer-0050.pt", wall_amendment_note="SOFTWARE_TEST_DOUBLE recovery wall amendment")
    assert torch.equal(actual, expected)
    assert result["outer_updates_completed"] == 100 and result["confirmed_outer_updates_observed"] == 103
    assert result["outer_attempts_dispatched"] == 103 and result["replayed_observed_outer_updates"] == 3
    assert result["active_wall_seconds"] >= 60


def test_interrupt_between_commit_assignments_rolls_back_whole_logical_transaction(tmp_path):
    import linecache
    import sys
    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    seed()
    expected, _ = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "baseline", total_updates=100, max_wall_seconds=10000)
    seed()
    sampler = ToySampler()

    def interrupt(frame, event, arg):
        if (frame.f_code is run_outer_loop.__code__ and event == "line" and frame.f_locals.get("cursor") == 7
                and frame.f_locals.get("step_in_progress") is True
                and linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() == "step_in_progress = False"):
            raise KeyboardInterrupt("SOFTWARE_TEST_DOUBLE interrupt in logical commit window")
        return interrupt

    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_outer_loop(parameter=initial, sampler=sampler, step=toy_step, evaluate=score,
                identity=IDENTITY, output=tmp_path / "interrupted", total_updates=100, max_wall_seconds=10000)
    finally:
        sys.settrace(None)
    assert sampler.cursor == 24
    actual, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "interrupted", total_updates=100, max_wall_seconds=10000,
        resume=tmp_path / "interrupted/outer-0006.pt")
    assert torch.equal(actual, expected)
    assert result["outer_updates_completed"] == 100 and result["confirmed_outer_updates_observed"] == 101


def test_real_sigkill_before_first_validation_recovers_initial_snapshot(tmp_path):
    """Kill only a local synthetic child; no EEG, CUDA, server or network."""
    import json
    import os
    import signal
    import subprocess
    import sys
    output = tmp_path / "interrupted"
    program = """
import os, runpy, signal, sys
from pathlib import Path
import torch
ns = runpy.run_path(sys.argv[2])
ns['seed']()
calls = [0]
def interrupted(value, episodes):
    calls[0] += 1
    if calls[0] == 7:
        os.kill(os.getpid(), signal.SIGKILL)
    return ns['toy_step'](value, episodes)
ns['run_outer_loop'](parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
    sampler=ns['ToySampler'](), step=interrupted, evaluate=ns['score'],
    identity=ns['IDENTITY'], output=Path(sys.argv[1]), total_updates=100,
    max_wall_seconds=10000)
"""
    child = subprocess.run([sys.executable, "-c", program, str(output), str(Path(__file__).resolve())],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
    assert child.returncode == -signal.SIGKILL, child.stderr.decode()
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    assert latest["outer_cursor"] == 0 and latest["path"] == "outer-0000.pt"
    progress = json.loads((output / "progress.json").read_text())
    assert progress["outer_updates_completed"] == 6 and progress["outer_attempts_dispatched"] == 7
    seed()
    initial = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    expected, _ = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=tmp_path / "baseline", total_updates=100, max_wall_seconds=10000)
    actual, result = run_outer_loop(parameter=initial, sampler=ToySampler(), step=toy_step, evaluate=score,
        identity=IDENTITY, output=output, total_updates=100, max_wall_seconds=10000,
        resume=output / "outer-0000.pt")
    assert torch.equal(actual, expected)
    assert result["confirmed_outer_updates_observed"] == 106
    assert result["outer_attempts_dispatched"] == 107 and result["replayed_observed_outer_updates"] == 6
    assert result["active_wall_seconds"] >= progress["active_wall_seconds"]
