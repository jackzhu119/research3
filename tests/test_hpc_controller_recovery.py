"""Deployment control tests; no data, CUDA, source fitting or network."""
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from p3work import training_campaign as campaign
from p3work import source_extension_downstream as downstream


@pytest.mark.parametrize("action", [None, {}, {"kind": "UNREGISTERED_SOFTWARE_FIXTURE"},
                                   {"kind": []}, {"kind": 3}])
@pytest.mark.parametrize("seconds", [0, 30])
def test_invalid_action_is_rejected_before_spawn_or_timeout(tmp_path, monkeypatch, action, seconds):
    monkeypatch.setattr(campaign.multiprocessing, "get_context",
                        lambda *args: pytest.fail("invalid action spawned a child"))
    with pytest.raises(campaign.CampaignError, match="^unregistered_campaign_action$"):
        campaign.dispatch_bounded(tmp_path, 30, action, remaining_seconds=seconds)
    assert list(tmp_path.iterdir()) == []


def test_invalid_production_action_is_rejected_before_recovery_or_import(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "guard_child_recovery",
                        lambda *args, **kwargs: pytest.fail("invalid action entered recovery"))
    with pytest.raises(campaign.CampaignError, match="^unregistered_campaign_action$"):
        campaign.Production(tmp_path, 30).perform({"kind": "not_registered"})
    assert list(tmp_path.iterdir()) == []


def test_scheduler_signal_handlers_only_request_stop_and_restore_handlers(tmp_path):
    previous = {number: signal.getsignal(number) for number in (signal.SIGUSR1, signal.SIGTERM)}
    with pytest.raises(RuntimeError, match="software fixture"):
        with campaign.graceful_stop_signals(True) as stop:
            assert not stop.is_set()
            os.kill(os.getpid(), signal.SIGUSR1)
            assert stop.is_set()
            stop.clear()
            os.kill(os.getpid(), signal.SIGTERM)
            assert stop.is_set()
            assert list(tmp_path.iterdir()) == []
            raise RuntimeError("software fixture")
    assert {number: signal.getsignal(number) for number in previous} == previous


def test_default_entry_does_not_replace_runpod_signal_handlers():
    previous = {number: signal.getsignal(number) for number in (signal.SIGUSR1, signal.SIGTERM)}
    with campaign.graceful_stop_signals() as stop:
        assert not stop.is_set()
        assert {number: signal.getsignal(number) for number in previous} == previous


@pytest.mark.parametrize("pgid", [None, 123, os.getpgrp()])
def test_scheduler_cannot_interrupt_unacknowledged_or_foreign_process_group(monkeypatch, pgid):
    monkeypatch.setattr(campaign.os, "killpg", lambda *args: pytest.fail("foreign process group signalled"))
    with pytest.raises(campaign.CampaignError, match="acknowledged_owned_stage_group"):
        campaign.interrupt_owned_stage(SimpleNamespace(pid=987654321), pgid)


def test_scheduler_pause_before_spawn_does_not_authorize_a_fit_or_write_files(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign.multiprocessing, "get_context", lambda *args: pytest.fail("paused child spawned"))
    assert campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=30,
                                     stop_requested=lambda: True) is None
    assert list(tmp_path.iterdir()) == []


def _transport(monkeypatch, *, extension=False, error=None, stop_on_receive=False, no_result=False):
    """Only transport messages and a fake clock; no OS child or scientific fit."""
    stop = threading.Event()
    clock = {"now": 0.}
    worker = SimpleNamespace(pid=987654321, alive=False)
    worker.start = lambda: setattr(worker, "alive", True)
    worker.is_alive = lambda: worker.alive
    worker.join = lambda seconds: clock.update(now=clock["now"] + seconds)
    worker.terminate = worker.kill = lambda: setattr(worker, "alive", False)
    sent, signals = [], []
    group = {"type": "owned_group_ready" if extension else "group_ready", "pid": worker.pid, "pgid": worker.pid}
    interrupted = {"type": "result", "ok": False, "error_type": error or "KeyboardInterrupt",
                   "reason": "owned_stage_failed_" + (error or "KeyboardInterrupt")}
    messages = [group] + ([] if no_result else [interrupted])

    def poll(seconds):
        if sent:
            stop.set()
        if not messages:
            clock["now"] += seconds
        return bool(messages)

    def recv():
        value = messages.pop(0)
        if stop_on_receive:
            stop.set()
        return value

    def send(value):
        sent.append(value)
        if value == "dispatch_owned_stage":
            stop.set()

    parent = SimpleNamespace(poll=poll, recv=recv, send=send, close=lambda: None)
    child = SimpleNamespace(close=lambda: None)
    context = SimpleNamespace(Pipe=lambda **kwargs: (parent, child), Process=lambda **kwargs: worker)
    monkeypatch.setattr(campaign.multiprocessing, "get_context", lambda *args: context)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(campaign.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(campaign, "live_group_members", lambda *args: ())
    return stop, clock, worker, sent, signals


def test_pause_before_ack_never_dispatches_or_interrupts_a_kernel(tmp_path, monkeypatch):
    stop, clock, worker, sent, signals = _transport(monkeypatch, stop_on_receive=True)
    assert campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=30,
                                     stop_requested=stop.is_set) is None
    assert sent == []
    assert signals == [(worker.pid, signal.SIGTERM)]
    assert not worker.alive


def test_stop_during_ownership_publication_does_not_send_a_dispatch_ack(tmp_path, monkeypatch):
    stop, clock, worker, sent, signals = _transport(monkeypatch)
    assert campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=30,
        stop_requested=stop.is_set, on_group_ready=lambda *args: stop.set()) is None
    assert sent == [] and signals == [(worker.pid, signal.SIGTERM)]
    assert not worker.alive


def test_acknowledged_stage_unwinds_on_sigint_and_returns_no_completion(tmp_path, monkeypatch):
    stop, clock, worker, sent, signals = _transport(monkeypatch)
    stopped = []
    assert campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=30,
        stop_requested=stop.is_set, on_group_stopped=lambda: stopped.append(True)) is None
    assert sent == ["dispatch_owned_stage"]
    assert signals == [(worker.pid, signal.SIGINT), (worker.pid, signal.SIGTERM)]
    assert stopped == [True] and not worker.alive
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("remaining,grace,expected_deadline", [(30, 2, 2), (12, 120, 7)])
def test_scheduler_cleanup_never_extends_original_absolute_deadline(tmp_path, monkeypatch,
                                                                  remaining, grace, expected_deadline):
    stop, clock, worker, sent, signals = _transport(monkeypatch, no_result=True)
    assert campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=remaining,
        stop_requested=stop.is_set, stop_grace_seconds=grace) is None
    assert expected_deadline <= clock["now"] <= expected_deadline + .21
    assert signals == [(worker.pid, signal.SIGINT), (worker.pid, signal.SIGTERM)]
    assert not worker.alive and sent == ["dispatch_owned_stage"]


def test_scheduler_stop_does_not_hide_unrelated_child_control_error(tmp_path, monkeypatch):
    stop, clock, worker, sent, signals = _transport(monkeypatch, error="ValueError")
    with pytest.raises(campaign.CampaignError, match="stage_blocked:ValueError"):
        campaign.dispatch_bounded(tmp_path, 30, {"kind": "pilot"}, remaining_seconds=30,
                                  stop_requested=stop.is_set)
    assert not worker.alive


def test_extension_pause_before_spawn_retains_permission_gate_and_no_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(downstream.multiprocessing, "get_context", lambda *args: pytest.fail("paused child spawned"))
    monkeypatch.setattr(downstream, "_latest_catalog", lambda *args: pytest.fail("paused catalog loaded"))
    with pytest.raises(PermissionError):
        downstream.execute_stage(tmp_path, {}, "profile", execute=True, stop_requested=lambda: True)
    result = downstream.execute_stage(tmp_path, {}, "profile", execute=True, user_started_gpu=True,
                                      stop_requested=lambda: True)
    assert result["status"] == "PAUSED_NOT_COMPLETE"
    assert result["scientific_fit_completion_claim"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [None, "ValueError"])
def test_extension_interrupt_closes_existing_wall_without_changing_scientific_cap(tmp_path, monkeypatch, error):
    stop, clock, worker, sent, signals = _transport(monkeypatch, extension=True, error=error)
    catalog = {"arm": "BASE4_PLUS_YANG"}
    budget = {"max_total_wall_seconds": 100, "caps": {"calibration_profile": 30}}
    monkeypatch.setattr(downstream, "_latest_catalog", lambda *args: catalog)
    monkeypatch.setattr(downstream, "_budget", lambda *args: (budget, {"path": "budget.json", "sha256": "a" * 64}))
    monkeypatch.setattr(downstream, "_primary", lambda *args: None)
    if error is None:
        result = downstream.execute_stage(tmp_path, catalog, "profile", execute=True, user_started_gpu=True,
                                          stop_requested=stop.is_set)
        assert result["status"] == "PAUSED_NOT_COMPLETE"
        assert result["scientific_fit_completion_claim"] is False and result["server_actions"] is False
        assert (tmp_path / result["controller_closed_wall"]["path"]).is_file()
    else:
        with pytest.raises(downstream.DownstreamError, match="owned_stage_failed_ValueError"):
            downstream.execute_stage(tmp_path, catalog, "profile", execute=True, user_started_gpu=True,
                                      stop_requested=stop.is_set)
    import json
    resources = tmp_path / "runs/source_extension_downstream/BASE4_PLUS_YANG/controller"
    wall = json.loads((resources / "wall.json").read_text())
    assert wall["max_wall_seconds"] == 100
    assert wall["interval"] is None and wall["closed_intervals"] == 1
    assert (resources / "claim.json").is_file()
    assert signals == [(worker.pid, signal.SIGINT), (worker.pid, signal.SIGTERM)]
    assert sent == ["dispatch_owned_stage"] and not worker.alive


def test_mainline_cli_passes_optional_signal_event_and_reports_pause(tmp_path, monkeypatch, capsys):
    from tools import run_training_campaign as cli
    calls = []

    def paused(root, **kwargs):
        assert root == tmp_path and kwargs["user_started_gpu"] is True
        assert kwargs["max_wall_seconds"] == 43200  # Existing cap is retained.
        assert kwargs["stop_grace_seconds"] == 17
        assert not kwargs["stop_requested"]()
        os.kill(os.getpid(), signal.SIGUSR1)
        assert kwargs["stop_requested"]()
        calls.append(True)
        return {"status": "PAUSED_SCHEDULER_SIGNAL_SAME_JOB", "completion_claimed": False}

    monkeypatch.setattr(cli, "execute_campaign", paused)
    assert cli.main(["--root", str(tmp_path), "--execute", "--user-started-gpu",
                     "--handle-slurm-signals", "--stop-grace-seconds", "17"]) == 3
    assert calls == [True] and '"completion_claimed": false' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_extension_cli_passes_optional_signal_event_and_reports_pause(tmp_path, monkeypatch, capsys):
    from tools import run_source_extension_downstream as cli
    calls = []

    def paused(root, catalog, **kwargs):
        assert root == tmp_path and kwargs["user_started_gpu"] is True
        assert kwargs["execute"] is True and kwargs["stop_grace_seconds"] == 19
        os.kill(os.getpid(), signal.SIGTERM)
        assert kwargs["stop_requested"]()
        calls.append(True)
        return {"status": "PAUSED_NOT_COMPLETE", "scientific_fit_completion_claim": False}

    monkeypatch.setattr(cli.run, "auto_development", paused)
    assert cli.main(["--root", str(tmp_path), "--arm", "BASE4_PLUS_YANG", "--stage", "auto-development",
                     "--execute", "--user-started-gpu", "--handle-slurm-signals",
                     "--stop-grace-seconds", "19"]) == 3
    assert calls == [True] and '"scientific_fit_completion_claim": false' in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_real_scheduler_signal_unwinds_owned_pure_metadata_child_without_any_fit(tmp_path, monkeypatch):
    """Actual OS signal/group/pipe cleanup, with an explicitly non-scientific child.

    The 120s cap belongs only to this metadata transport fixture. It neither
    invokes a scientific kernel nor changes a production startup/fit deadline.
    """
    context = campaign.multiprocessing.get_context("spawn")
    program = r'''
import json, os, sys, time
from pathlib import Path
from multiprocessing.connection import Connection
os.setsid()
connection = Connection(int(sys.argv[1]))
root = Path(sys.argv[2])
connection.send({'type':'group_ready','pid':os.getpid(),'pgid':os.getpgrp()})
if connection.recv() != 'dispatch_owned_stage':
    raise SystemExit(2)
try:
    (root/'SOFTWARE_NO_FIT_started').write_text('pure metadata transport fixture')
    while True: time.sleep(.01)
except KeyboardInterrupt:
    (root/'SOFTWARE_NO_FIT_checkpoint.json').write_text(json.dumps({
        'synthetic':True,'scientific_fit_completion_claim':False,
        'numeric_libraries_loaded':[name for name in ('numpy','scipy','sklearn','torch') if name in sys.modules]}))
    connection.send({'type':'result','ok':False,'error_type':'KeyboardInterrupt','reason':'software_scheduler_pause'})
    connection.recv()
'''
    workers = []

    class MetadataProcess:
        def __init__(self, **kwargs):
            self.connection = kwargs["args"][3]
            self.child = None
            workers.append(self)

        @property
        def pid(self):
            return self.child.pid

        def start(self):
            self.child = subprocess.Popen([sys.executable, "-B", "-c", program,
                str(self.connection.fileno()), str(tmp_path)], pass_fds=(self.connection.fileno(),),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def is_alive(self):
            return self.child.poll() is None

        def join(self, seconds):
            try: self.child.wait(timeout=seconds)
            except subprocess.TimeoutExpired: pass

        def terminate(self):
            self.child.terminate()

        def kill(self):
            self.child.kill()

    monkeypatch.setattr(campaign.multiprocessing, "get_context", lambda *args:
        SimpleNamespace(Pipe=context.Pipe, Process=MetadataProcess))
    ownership = []
    with campaign.graceful_stop_signals(True) as stop:
        def scheduler_stop():
            if (tmp_path / "SOFTWARE_NO_FIT_started").exists() and not stop.is_set():
                os.kill(os.getpid(), signal.SIGUSR1)
            return stop.is_set()

        def acknowledged(pid, pgid, deadline):
            assert os.getpgid(pid) == pgid == pid and pgid != os.getpgrp()
            ownership.append(pgid)

        assert campaign.dispatch_bounded(tmp_path, 120, {"kind": "pilot"}, remaining_seconds=120,
            stop_requested=scheduler_stop, stop_grace_seconds=3, on_group_ready=acknowledged) is None
        assert stop.is_set()
    import json
    value = json.loads((tmp_path / "SOFTWARE_NO_FIT_checkpoint.json").read_text())
    assert value == {"synthetic": True, "scientific_fit_completion_claim": False, "numeric_libraries_loaded": []}
    assert len(workers) == len(ownership) == 1 and not workers[0].is_alive()
    assert campaign.live_group_members(ownership[0]) == ()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["SOFTWARE_NO_FIT_checkpoint.json", "SOFTWARE_NO_FIT_started"]
