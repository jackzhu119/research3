"""Local status/clock test doubles only; no real command, network or GPU."""
from datetime import datetime, timedelta, timezone
import importlib
import json
from types import SimpleNamespace

import pytest

from p3work import runtime_clock as clock


@pytest.fixture
def observation(monkeypatch):
    utc = datetime(2026, 10, 18, tzinfo=timezone.utc)
    wall = iter((utc, utc + timedelta(seconds=3)))
    monotonic = iter((100_000_000_000, 103_000_000_000))
    monkeypatch.setattr(clock, "_utc_now", lambda: next(wall))
    monkeypatch.setattr(clock.time, "monotonic_ns", lambda: next(monotonic))
    monkeypatch.setattr(clock.shutil, "which", lambda name, *, path: "/usr/bin/" + name)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = b"yes\n" if command[0].endswith("timedatectl") else b"Last offset : +0.123 seconds\nLeap status : Normal\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr=b"")

    monkeypatch.setattr(clock.subprocess, "run", run)
    return calls


def test_import_does_not_observe_or_run_a_command(monkeypatch):
    monkeypatch.setattr(clock.subprocess, "run", lambda *a, **k: pytest.fail("import performed a command"))
    monkeypatch.setattr(clock.shutil, "which", lambda *a, **k: pytest.fail("import inspected binaries"))
    importlib.reload(clock)


def test_local_observations_have_bracketed_clocks_without_invented_UTC_accuracy(observation):
    result = clock.collect_clock_preflight()
    assert result["scope"] == "observation_only_not_external_UTC_measurement"
    assert result["request_start_utc"] == "2026-10-18T00:00:00+00:00"
    assert result["request_end_utc"] == "2026-10-18T00:00:03+00:00"
    assert result["observation_duration_monotonic_ns"] == 3_000_000_000
    assert result["monotonic_ns_start"] == 100_000_000_000 and result["monotonic_ns_end"] == 103_000_000_000
    assert result["ntp_synchronized"] is True
    assert result["utc_offset_seconds"] is result["utc_uncertainty_seconds"] is result["independent_utc_reference"] is None
    assert result["utc_accuracy_status"] == "not_measured_no_independent_reference"
    assert result["observations"]["chronyc_tracking"]["stdout"].startswith("Last offset : +0.123")
    assert result["clock_or_service_modified"] is result["external_time_request_performed"] is result["GPU_inspected"] is False
    assert len(observation) == 2
    json.dumps(result, allow_nan=False)


def test_only_fixed_local_read_only_commands_and_sanitized_environment_are_used(observation):
    clock.collect_clock_preflight()
    commands = [row[0] for row in observation]
    assert commands == [["/usr/bin/timedatectl", "show", "--property=NTPSynchronized", "--value"],
                        ["/usr/bin/chronyc", "-n", "-h", "/run/chrony/chronyd.sock", "tracking"]]
    for _, kwargs in observation:
        assert kwargs["timeout"] == 2.0 and kwargs["shell"] is False and kwargs["check"] is False
        assert kwargs["stdin"] is clock.subprocess.DEVNULL
        assert kwargs["env"] == {"PATH": clock.SYSTEM_PATH, "LC_ALL": "C", "LANG": "C"}


def test_missing_binaries_return_not_available_without_starting_a_process(observation, monkeypatch):
    monkeypatch.setattr(clock.shutil, "which", lambda name, *, path: None)
    result = clock.collect_clock_preflight()
    assert not observation and result["ntp_synchronized"] is None
    assert all(value["status"] == "not_available" and value["reason"] == "binary_not_found"
               for value in result["observations"].values())


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "oserror"])
def test_unavailable_local_services_keep_explicit_bounded_failure_evidence(observation, monkeypatch, failure):
    def run(command, **kwargs):
        if failure == "timeout":
            raise clock.subprocess.TimeoutExpired(command, 2, output=b"partial local reply", stderr=b"timed out")
        if failure == "oserror":
            raise PermissionError("NOT_CLOCK_EVIDENCE_DO_NOT_COPY")
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"System bus is unavailable")
    monkeypatch.setattr(clock.subprocess, "run", run)
    result = clock.collect_clock_preflight()
    expected = {"timeout": "command_timeout", "nonzero": "local_command_failed", "oserror": "local_command_unavailable"}[failure]
    assert result["ntp_synchronized"] is None
    assert all(value["status"] == "not_available" and value["reason"] == expected
               for value in result["observations"].values())
    assert "NOT_CLOCK_EVIDENCE" not in json.dumps(result)
    if failure == "timeout":
        assert result["observations"]["chronyc_tracking"]["stdout"] == "partial local reply"


def test_raw_output_is_finite_and_truncated_ntp_output_is_not_interpreted(observation, monkeypatch):
    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=b"yes\n" + b"x" * 10000, stderr=b"\xff" * 10000)
    monkeypatch.setattr(clock.subprocess, "run", run)
    result = clock.collect_clock_preflight()
    assert result["ntp_synchronized"] is None
    for value in result["observations"].values():
        assert len(value["stdout"]) <= 4096 and len(value["stderr"]) <= 4096
        assert value["stdout_truncated"] is value["stderr_truncated"] is True


@pytest.mark.parametrize("reply,expected", [(b"no\n", False), (b"unknown\n", None), (b"true\n", None)])
def test_NTP_boolean_is_parsed_conservatively_and_is_not_an_offset(observation, monkeypatch, reply, expected):
    monkeypatch.setattr(clock.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=reply, stderr=b""))
    result = clock.collect_clock_preflight()
    assert result["ntp_synchronized"] is expected
    assert result["utc_offset_seconds"] is result["utc_uncertainty_seconds"] is None


def test_UTC_rollback_does_not_replace_the_monotonic_duration(observation, monkeypatch):
    utc = datetime(2026, 10, 18, tzinfo=timezone.utc)
    wall = iter((utc, utc - timedelta(seconds=30)))
    monkeypatch.setattr(clock, "_utc_now", lambda: next(wall))
    result = clock.collect_clock_preflight()
    assert result["local_UTC_request_ordered"] is False
    assert result["observation_duration_monotonic_ns"] == 3_000_000_000


def test_nonmonotonic_observation_stays_explicit_and_does_not_invent_a_duration(observation, monkeypatch):
    monotonic = iter((100, 99))
    monkeypatch.setattr(clock.time, "monotonic_ns", lambda: next(monotonic))
    result = clock.collect_clock_preflight()
    assert result["monotonic_observation_status"] == "nonmonotonic_observation"
    assert result["observation_duration_monotonic_ns"] is None
