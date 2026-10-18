"""Injected clock/local-file software fixtures; no sleep, EEG, GPU or network."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from p3work.recovery_budget import (PersistentWallBudget, WallBudgetBlocked,
                                   WallBudgetError, WallBudgetExhausted)


IDENTITY = "a" * 64


class Clock:
    def __init__(self, utc=1000.0, mono=100.0, boot="SOFTWARE_TEST_BOOT"):
        self.utc, self.mono, self.boot = utc, mono, boot

    def advance(self, seconds):
        self.utc += seconds; self.mono += seconds

    def budget(self, path, maximum=100, **kwargs):
        return PersistentWallBudget(path, IDENTITY, maximum, monotonic=lambda: self.mono,
            utc_now=lambda: datetime.fromtimestamp(self.utc, timezone.utc), boot_id=lambda: self.boot, **kwargs)


def hard_kill_double(budget):
    """Kernel lock release without graceful-close: exact persisted-kill state."""
    budget._lock.close(); budget._closed = True


def test_intent_is_durable_before_work_and_normal_pause_excludes_downtime(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path); b.start_interval()
    saved = json.loads(path.read_text())
    assert saved["interval"]["intent"] == "persisted_before_work"
    assert saved["interval"]["start_utc_seconds"] == 1000
    c.advance(12); assert b.wall_seconds == 12
    b.graceful_close(); b.close()
    c.advance(200)
    resumed = c.budget(path)
    assert resumed.wall_seconds == 12 and resumed.remaining_seconds == 88
    resumed.start_interval(); c.advance(3); resumed.close()
    assert json.loads(path.read_text())["durable_wall_seconds"] == 15


def test_hardkill_recovers_entire_unclosed_interval_including_downtime(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path); b.start_interval(); c.advance(7)
    b.checkpoint(5)
    assert b.minimum_charged_seconds == 7
    c.advance(4); hard_kill_double(b)
    c.advance(9)
    resumed = c.budget(path, checkpoint_wall_seconds=3)
    assert resumed.wall_seconds == 20 and not resumed.blocked
    assert resumed.report()["recovered_interval_count"] == 1
    resumed.close()


def test_real_sigkill_releases_lock_and_retains_durable_intent(tmp_path):
    path = tmp_path / "actual_child_sigkill_fixture.json"
    script = """
import os, signal, sys
from p3work.recovery_budget import PersistentWallBudget
budget = PersistentWallBudget(sys.argv[1], 'a' * 64, 100,
    monotonic=lambda: 100., utc_now=lambda: 1000., boot_id='SOFTWARE_TEST_BOOT')
budget.start_interval()
os.kill(os.getpid(), signal.SIGKILL)
"""
    child = subprocess.run([sys.executable, "-c", script, str(path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=10)
    assert child.returncode == -9
    assert json.loads(path.read_text())["interval"] is not None
    c = Clock(utc=1010, mono=110)
    recovered = c.budget(path)
    assert not recovered.blocked and recovered.wall_seconds == 10
    assert recovered.report()["recovered_interval_count"] == 1
    recovered.close()


def test_high_water_combines_checkpoint_and_durable_progress_without_reset(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path, checkpoint_wall_seconds=18, minimum_charged_seconds=21)
    assert b.minimum_charged_seconds == 21
    b.start_interval(); c.advance(4); b.checkpoint(30)
    assert b.minimum_charged_seconds == 30
    c.advance(2); b.close(checkpoint_wall_seconds=1)
    resumed = c.budget(path, checkpoint_wall_seconds=10)
    assert resumed.minimum_charged_seconds == 32
    resumed.close()


def test_budget_exhaustion_requires_explicit_operational_only_extension(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path, maximum=5); b.start_interval(); c.advance(6); b.close()
    with pytest.raises(WallBudgetError, match="explicit amendment"):
        c.budget(path, maximum=10)
    b = c.budget(path, maximum=5)
    with pytest.raises(WallBudgetExhausted): b.start_interval()
    b.close()
    b = c.budget(path, maximum=10, amendment_note="Explicit software-fixture operational extension")
    assert b.identity_sha256 == IDENTITY and b.minimum_charged_seconds == 6
    assert b.max_wall_seconds == 10 and b.report()["amendment_count"] == 1
    b.start_interval(); c.advance(1); b.close()
    journal = json.loads(path.read_text())
    assert journal["durable_wall_seconds"] == 7
    assert journal["amendments"][0]["scientific_identity_changed"] is False


@pytest.mark.parametrize("anomaly", ["utc_backwards", "mono_backwards", "boot_change", "clock_skew", "boot_unknown"])
def test_uncertain_clock_history_permanently_blocks_without_reset(tmp_path, anomaly):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path, checkpoint_wall_seconds=4); b.start_interval(); hard_kill_double(b)
    c.advance(10)
    if anomaly == "utc_backwards": c.utc -= 11
    elif anomaly == "mono_backwards": c.mono -= 11
    elif anomaly == "boot_change": c.boot = "OTHER_SOFTWARE_BOOT"
    elif anomaly == "boot_unknown": c.boot = None
    else: c.utc += 5.001
    blocked = c.budget(path)
    assert blocked.blocked and blocked.minimum_charged_seconds >= 4
    assert blocked.report()["clock_block"]["reasons"]
    assert blocked.remaining_seconds == 0
    with pytest.raises(WallBudgetBlocked):
        _ = blocked.wall_seconds
    with pytest.raises(WallBudgetBlocked): blocked.start_interval()
    blocked.close()
    # Neither repaired clocks nor a new limit erase the unresolved interval.
    c.boot = "SOFTWARE_TEST_BOOT"; c.utc = 1020; c.mono = 120
    again = c.budget(path, maximum=200, amendment_note="Budget-only extension cannot clear a clock block")
    assert again.blocked
    with pytest.raises(WallBudgetBlocked): again.checkpoint(50)
    assert again.minimum_charged_seconds >= 50
    again.close()


def test_at_most_five_seconds_clock_drift_charges_larger_elapsed(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path); b.start_interval(); hard_kill_double(b)
    c.advance(10); c.utc += 5
    resumed = c.budget(path)
    assert not resumed.blocked and resumed.wall_seconds == 15
    resumed.close()


def test_live_clock_anomaly_blocks_before_further_checkpoint_work(tmp_path):
    c = Clock(); b = c.budget(tmp_path / "wall.json"); b.start_interval()
    c.advance(2); c.utc += 6
    with pytest.raises(WallBudgetBlocked): b.checkpoint()
    assert b.blocked and b.minimum_charged_seconds == 8
    b.close()


@pytest.mark.parametrize("rollback", ["utc", "mono", "both"])
def test_observed_wall_high_water_and_clock_rollback_above_interval_start(tmp_path, rollback):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path); b.start_interval(); c.advance(20)
    assert b.wall_seconds == 20
    assert json.loads(path.read_text())["durable_wall_seconds"] == 20
    # Both readings still exceed the start; compare the latest observation too.
    if rollback in {"utc", "both"}: c.utc -= 1
    if rollback in {"mono", "both"}: c.mono -= 1
    with pytest.raises(WallBudgetBlocked):
        _ = b.wall_seconds
    assert b.minimum_charged_seconds >= 20 and b.report()["wall_seconds"] >= 20
    b.close()


def test_scientific_identity_and_duplicate_writer_are_rejected(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path)
    with pytest.raises(WallBudgetError, match="Another process"):
        c.budget(path)
    b.close()
    before = path.read_bytes()
    with pytest.raises(WallBudgetError, match="identity changed"):
        PersistentWallBudget(path, "b" * 64, 100)
    assert path.read_bytes() == before


def test_rejected_budget_amendment_still_preserves_larger_checkpoint_high_water(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path); b.close()
    with pytest.raises(WallBudgetError, match="explicit amendment"):
        c.budget(path, maximum=200, checkpoint_wall_seconds=30)
    resumed = c.budget(path)
    assert resumed.minimum_charged_seconds == 30
    resumed.close()


@pytest.mark.parametrize("field,value", [("durable_wall_seconds", -1), ("durable_wall_seconds", 0),
    ("max_wall_seconds", 200), ("next_interval_sequence", 0)])
def test_corrupt_or_rolled_back_journal_is_never_silently_reinitialized(tmp_path, field, value):
    c = Clock(); path = tmp_path / "wall.json"
    b = c.budget(path, checkpoint_wall_seconds=10); b.close()
    journal = json.loads(path.read_text()); journal[field] = value
    path.write_text(json.dumps(journal)); before = path.read_bytes()
    with pytest.raises(WallBudgetError): c.budget(path)
    assert path.read_bytes() == before


def test_context_manager_closes_normal_exceptions_but_not_research_identity(tmp_path):
    c = Clock(); path = tmp_path / "wall.json"
    with pytest.raises(RuntimeError, match="software fixture"):
        with c.budget(path) as b:
            c.advance(3)
            raise RuntimeError("software fixture")
    resumed = c.budget(path)
    assert resumed.wall_seconds == 3 and resumed.report()["recovered_interval_count"] == 0
    resumed.close()
