"""CPU-only native checkpoint/clock fixtures, no EEG, GPU or scheduler actions."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "/workspace/paper3_hpc")
import p3work.meta_training as meta
from p3work.recovery_budget import PersistentWallBudget, WallBudgetBlocked
import r1_clock_recovery as repair


class Clock:
    def __init__(self, utc=1792979000.0, mono=2128831.4086003,
                 boot="6d28dc8b-0dd2-468a-a2ad-2346f0fb38d5"):
        self.utc, self.mono, self.boot = utc, mono, boot
    def advance(self, seconds):
        self.utc += seconds
        self.mono += seconds
    def arguments(self):
        return {"monotonic": lambda: self.mono, "utc_now": lambda: self.utc,
                "boot_id": lambda: self.boot}


def actual_cross_node_state(identity="7cbaf14c5b3cf8b826569c4a5654c978ca81dfc28f416c4b187f7d8573be6be1"):
    """Actual school clock values; amendment note is explicit fixture metadata."""
    interval = {"sequence": 596, "identity_sha256": identity,
        "base_wall_seconds": 6648.613922640216,
        "start_utc_seconds": 1792976738.360695,
        "start_monotonic_seconds": 2129165.092619733,
        "boot_id": "5e2dff02-30a5-468d-b605-8e61bf353afb",
        "last_observed_utc_seconds": 1792976738.387979,
        "last_observed_monotonic_seconds": 2129165.119900163,
        "intent": "persisted_before_work"}
    block = {"clock": {"utc_seconds": 1792978897.978574,
                       "monotonic_seconds": 2128729.387174312,
                       "boot_id": "6d28dc8b-0dd2-468a-a2ad-2346f0fb38d5"},
             "utc_elapsed_seconds": 2159.617879152298,
             "monotonic_elapsed_seconds": -435.7054454209283,
             "charged_wall_seconds": 8808.231801792514,
             "reasons": ["boot_changed_or_unknown"], "interval": deepcopy(interval),
             "resolution": "blocked_no_automatic_reset_or_clock_override"}
    return {"schema_version": "paper3-operational-wall-budget-v1",
        "identity_sha256": identity, "scope": "operational_only_no_scientific_identity_change",
        "initial_max_wall_seconds": 443400, "max_wall_seconds": 1330200,
        "durable_wall_seconds": 8808.231801792514,
        "checkpoint_wall_high_water_seconds": 6648.600285200402,
        "interval": interval, "next_interval_sequence": 597,
        "clock_block": block, "closed_intervals": 595,
        "recovery_events": [], "amendments": [{"previous_max_wall_seconds": 443400,
            "new_max_wall_seconds": 1330200, "note": "SOFTWARE_FIXTURE existing authorized wall extension",
            "scientific_identity_changed": False}]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_actual_school_clock_fixture_original_blocks_repair_preserves_every_observation(tmp_path):
    output = tmp_path / "runs/meta" / ("a" * 64)
    path = output / "wall_budget.json"
    state = actual_cross_node_state()
    write(path, state)
    clock = Clock()
    budget = PersistentWallBudget(path, state["identity_sha256"], 1330200, **clock.arguments())
    with pytest.raises(WallBudgetBlocked):
        budget.start_interval()
    proof = {"output": output, "identity_sha256": state["identity_sha256"],
             "job": {"method": "P+PCT", "dimension": 2, "radius": .05, "seed": 20261018, "phase": "screening"},
             "checkpoint_verified": {"active_wall_seconds": state["checkpoint_wall_high_water_seconds"]},
             "resolutions": []}
    assert repair._recover_cross_node_clock(budget, proof)
    assert budget.minimum_charged_seconds == state["durable_wall_seconds"]
    event = budget._state["authorized_clock_recoveries"][0]
    assert event["original_clock_block"] == state["clock_block"]
    assert event["original_unsettled_interval"] == state["interval"]
    assert event["unknown_additional_elapsed_seconds"] is None
    assert budget._state["closed_intervals"] == 595
    assert budget._state["clock_history_qualification"]["exact_cumulative_elapsed_claimed"] is False
    budget.start_interval()
    assert budget._state["interval"]["boot_id"] == clock.boot
    clock.advance(5)
    budget.checkpoint()
    assert budget.minimum_charged_seconds == state["durable_wall_seconds"] + 5
    budget.close()
    saved = json.loads(path.read_text())
    assert saved["authorized_clock_recoveries"][0] == event
    assert saved["durable_wall_seconds"] >= state["durable_wall_seconds"]


@pytest.mark.parametrize("condition", ["utc_rollback", "same_boot_skew", "unknown_boot", "no_checkpoint", "changed_identity"])
def test_other_clock_failures_or_missing_identity_proof_stay_blocked(tmp_path, condition):
    output = tmp_path / "runs/meta" / ("a" * 64)
    path = output / "wall_budget.json"
    state = actual_cross_node_state()
    if condition == "utc_rollback":
        state["clock_block"]["reasons"].append("utc_clock_rolled_back")
    if condition == "same_boot_skew":
        state["clock_block"]["reasons"] = ["utc_monotonic_elapsed_disagreement_exceeds_5s"]
    if condition == "unknown_boot":
        state["clock_block"]["clock"]["boot_id"] = None
    write(path, state)
    budget = PersistentWallBudget(path, state["identity_sha256"], 1330200, **Clock().arguments())
    proof = {"output": output, "identity_sha256": state["identity_sha256"], "job": {},
             "checkpoint_verified": None if condition == "no_checkpoint" else {"active_wall_seconds": 6648.6},
             "resolutions": []}
    if condition == "changed_identity":
        proof["identity_sha256"] = "b" * 64
    assert repair._recover_cross_node_clock(budget, proof) is False
    assert budget.blocked
    assert budget._state["clock_block"] == state["clock_block"]
    assert "authorized_clock_recoveries" not in budget._state
    budget.close()


class Sampler:
    def __init__(self): self.cursor = 0
    def next(self):
        self.cursor += 1
        return self.cursor
    def state_dict(self): return {"cursor": self.cursor}
    def load_state_dict(self, state): self.cursor = state["cursor"]


def seed():
    random.seed(20261018)
    np.random.seed(20261018)
    torch.manual_seed(20261018)


def step(value, episodes):
    noise = torch.rand_like(value) + random.random() + float(np.random.random())
    return (value + noise * 1e-4 + sum(episodes) * 1e-8).detach().requires_grad_(True), {
        "full_unroll": True, "inner_updates_actual": 30, "synthetic": True}


def score(value):
    return {"participant_mean_ba": .5, "participant_mean_bce": 1., "synthetic": True}


@pytest.fixture
def native_module(monkeypatch):
    originals = {name: getattr(meta, name) for name in ("run_outer_loop", "load_checkpoint", "PersistentWallBudget")}
    monkeypatch.setenv("SLURM_JOB_ID", "772906")
    yield meta
    for name, value in originals.items():
        setattr(meta, name, value)
    if hasattr(meta, "_paper3_r1_cross_node_clock_recovery"):
        del meta._paper3_r1_cross_node_clock_recovery


def interrupted_fixture(root, module):
    job = module.MetaJob("P", 8, .35, phase="engineering")
    identity = {"job": asdict(job), "scope": "SOFTWARE_TEST_DOUBLE", "code_sha256": "a" * 64}
    output = root / "runs/meta" / job.identity
    write(output / "identity.json", identity)
    seed()
    attempts = []
    def interrupted(value, episodes):
        attempts.append(1)
        if len(attempts) == 8:
            raise KeyboardInterrupt("SOFTWARE_FIXTURE interrupted original loop")
        return step(value, episodes)
    with pytest.raises(KeyboardInterrupt):
        module.run_outer_loop(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
            sampler=Sampler(), step=interrupted, evaluate=score, identity=identity,
            output=output, total_updates=100, max_wall_seconds=1330200,
            **Clock().arguments())
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    resume = output / latest["path"]
    # This models the actual retained cross-node clock journal independently
    # of the native model/RNG checkpoint generated above.
    write(output / "wall_budget.json", actual_cross_node_state(repair._digest(identity)))
    return identity, output, resume


def test_native_same_checkpoint_resume_original_fails_repaired_exact_rng_trajectory_passes(tmp_path, native_module):
    module = native_module
    identity, output, resume = interrupted_fixture(tmp_path, module)
    original_checkpoint = resume.read_bytes()
    call = dict(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
        sampler=Sampler(), step=step, evaluate=score, identity=identity, output=output,
        total_updates=100, max_wall_seconds=1330200, resume=resume, **Clock().arguments())
    with pytest.raises(WallBudgetBlocked):
        module.run_outer_loop(**call)
    assert resume.read_bytes() == original_checkpoint
    original_loop = module.run_outer_loop
    baseline = tmp_path / "separate-software-baseline"
    seed()
    expected, _ = original_loop(parameter=call["parameter"], sampler=Sampler(), step=step,
        evaluate=score, identity=identity, output=baseline, total_updates=100,
        max_wall_seconds=1330200, **Clock().arguments())
    assert repair.install(tmp_path)["status"] == "INSTALLED"
    assert repair.install(tmp_path)["status"] == "ALREADY_INSTALLED"
    call["sampler"] = Sampler()
    observed, result = module.run_outer_loop(**call)
    assert torch.equal(observed, expected)
    assert result["status"] == "UPDATE_BUDGET_COMPLETED"
    assert result["outer_updates_completed"] == 100
    assert result["validation_checkpoints"] == 2
    qualification = result["operational_clock_history_qualification"]
    assert qualification["exact_cumulative_elapsed_claimed"] is False
    assert qualification["unknown_elapsed_seconds"] is None
    assert qualification["audit_path"] == str((output / "wall_budget.json").relative_to(tmp_path))
    assert call["sampler"].cursor == 400
    assert resume.read_bytes() == original_checkpoint
    saved = json.loads((output / "wall_budget.json").read_text())
    assert len(saved["authorized_clock_recoveries"]) == 1
    assert saved["authorized_clock_recoveries"][0]["verified_checkpoint"]["outer_cursor"] == 7
    assert saved["clock_history_qualification"]["exact_cumulative_elapsed_claimed"] is False


@pytest.mark.parametrize("invalid", ["missing_allocation", "changed_identity", "damaged_checkpoint"])
def test_no_bypass_of_actual_allocation_or_native_full_checkpoint_loader(tmp_path, native_module, monkeypatch, invalid):
    module = native_module
    identity, output, resume = interrupted_fixture(tmp_path, module)
    if invalid == "missing_allocation":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif invalid == "changed_identity":
        identity = {**identity, "code_sha256": "b" * 64}
    elif invalid == "damaged_checkpoint":
        resume.write_bytes(resume.read_bytes() + b"damaged")
    repair.install(tmp_path)
    with pytest.raises((WallBudgetBlocked, ValueError)):
        module.run_outer_loop(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
            sampler=Sampler(), step=step, evaluate=score, identity=identity, output=output,
            total_updates=100, max_wall_seconds=1330200, resume=resume, **Clock().arguments())
    saved = json.loads((output / "wall_budget.json").read_text())
    assert saved["clock_block"] is not None
    assert "authorized_clock_recoveries" not in saved


def test_existing_unknown_interval_count_is_never_reduced(tmp_path):
    output = tmp_path / "runs/meta" / ("a" * 64)
    state = actual_cross_node_state()
    state["clock_history_qualification"] = {"unknown_interval_count": 5,
                                            "exact_cumulative_elapsed_claimed": False}
    write(output / "wall_budget.json", state)
    budget = PersistentWallBudget(output / "wall_budget.json", state["identity_sha256"],
                                  1330200, **Clock().arguments())
    proof = {"output": output, "identity_sha256": state["identity_sha256"], "job": {},
             "checkpoint_verified": {"active_wall_seconds": 6648.6}, "resolutions": []}
    assert repair._recover_cross_node_clock(budget, proof)
    assert budget._state["clock_history_qualification"]["unknown_interval_count"] == 6
    budget.close()


def test_clock_observation_failure_retains_original_block(tmp_path):
    output = tmp_path / "runs/meta" / ("a" * 64)
    state = actual_cross_node_state()
    write(output / "wall_budget.json", state)
    budget = PersistentWallBudget(output / "wall_budget.json", state["identity_sha256"],
                                  1330200, **Clock().arguments())
    def unavailable():
        raise RuntimeError("clock temporarily unavailable")
    budget._clock = unavailable
    proof = {"output": output, "identity_sha256": state["identity_sha256"], "job": {},
             "checkpoint_verified": {"active_wall_seconds": 6648.6}, "resolutions": []}
    assert repair._recover_cross_node_clock(budget, proof) is False
    assert budget._state["clock_block"] == state["clock_block"]
    budget.close()


def test_returning_to_historical_node_does_not_ignore_known_monotonic_rollback(tmp_path):
    output = tmp_path / "runs/meta" / ("a" * 64)
    state = actual_cross_node_state()
    write(output / "wall_budget.json", state)
    clock = Clock(mono=2129165.10, boot=state["interval"]["boot_id"])
    budget = PersistentWallBudget(output / "wall_budget.json", state["identity_sha256"],
                                  1330200, **clock.arguments())
    proof = {"output": output, "identity_sha256": state["identity_sha256"], "job": {},
             "checkpoint_verified": {"active_wall_seconds": 6648.6}, "resolutions": []}
    assert repair._recover_cross_node_clock(budget, proof) is False
    assert budget._state["clock_block"] == state["clock_block"]
    budget.close()


def test_corrupt_metadata_is_read_with_an_actual_byte_bound(tmp_path, monkeypatch):
    requested_sizes = []
    class BoundedReader:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, maximum):
            requested_sizes.append(maximum)
            assert maximum > 0
            return b" " * maximum
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: BoundedReader())
    assert repair._read(tmp_path / "oversized-corrupt-metadata.json") is None
    assert requested_sizes == [8 * 1024 * 1024 + 1]


def test_no_resume_or_wrong_latest_descriptor_cannot_authorize_recovery(tmp_path, native_module):
    module = native_module
    identity, output, resume = interrupted_fixture(tmp_path, module)
    values = {"identity": identity, "output": output, "total_updates": 100, "resume": None}
    assert repair._registered_resume(module, tmp_path, values) is None
    values["resume"] = resume
    assert repair._registered_resume(module, tmp_path, values) is not None
    latest = json.loads((output / "latest_checkpoint.json").read_text())
    latest["sha256"] = "b" * 64
    write(output / "latest_checkpoint.json", latest)
    assert repair._registered_resume(module, tmp_path, values) is None


def test_installation_rejects_another_project_root(tmp_path, native_module):
    repair.install(tmp_path)
    with pytest.raises(ValueError, match="different project root"):
        repair.install(tmp_path / "another-project")


def test_amendment_failure_releases_original_exclusive_journal_lock(tmp_path, native_module, monkeypatch):
    import fcntl
    identity, output, resume = interrupted_fixture(tmp_path, native_module)
    repair.install(tmp_path)
    def failure(*args):
        raise RuntimeError("SOFTWARE_FIXTURE bookkeeping amendment failed")
    monkeypatch.setattr(repair, "_recover_cross_node_clock", failure)
    with pytest.raises(WallBudgetBlocked, match="clock block retained"):
        native_module.run_outer_loop(parameter=torch.zeros(2, dtype=torch.float64, requires_grad=True),
            sampler=Sampler(), step=step, evaluate=score, identity=identity, output=output,
            total_updates=100, max_wall_seconds=1330200, resume=resume, **Clock().arguments())
    with (output / "wall_budget.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((output / "wall_budget.json").read_text())["clock_block"] is not None


def test_result_annotation_preserves_all_native_fields_even_without_new_resolution(tmp_path):
    output = tmp_path / "runs/meta" / ("a" * 64)
    state = actual_cross_node_state()
    state["clock_history_qualification"] = {"exact_cumulative_elapsed_claimed": False,
        "unknown_elapsed_seconds": None, "unknown_interval_count": 2}
    write(output / "wall_budget.json", state)
    proof = {"root": tmp_path, "output": output, "identity_sha256": state["identity_sha256"],
             "resolutions": []}
    parameter = object()
    native = {"status": "PAUSED_WALL_BUDGET", "outer_updates_completed": 600,
              "confirmed_outer_updates_observed": 626, "active_wall_seconds": 8808.2318,
              "checkpoint": {"path": "outer-0600.pt", "sha256": "a" * 64},
              "validation": [{"participant_mean_ba": .7, "participant_mean_bce": .6}]}
    before = deepcopy(native)
    actual_parameter, annotated = repair._qualify_result(proof, (parameter, native))
    assert actual_parameter is parameter
    assert native == before
    assert {key: value for key, value in annotated.items()
            if key != "operational_clock_history_qualification"} == before
    assert annotated["operational_clock_history_qualification"]["unknown_interval_count"] == 2
