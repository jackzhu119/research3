"""Bounded local clock observations; no external UTC measurement or clock edit.

Importing this module performs no preflight. The caller decides when an actual
execution may collect these observations; timestamps are operational metadata,
not a numerical runtime identity or evidence of an independently measured UTC
offset. chronyc is restricted to a local Unix socket, with no DNS/UDP fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone
import shutil
import subprocess
import time


SCHEMA_VERSION = "paper3-runtime-clock-preflight-v1"
SCOPE = "observation_only_not_external_UTC_measurement"
COMMAND_TIMEOUT_SECONDS = 2.0
MAX_OUTPUT_BYTES = 4096
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
CHRONY_SOCKET = "/run/chrony/chronyd.sock"
_COMMAND_ENVIRONMENT = {"PATH": SYSTEM_PATH, "LC_ALL": "C", "LANG": "C"}


def _utc_now():
    return datetime.now(timezone.utc)


def _bounded_output(value):
    value = b"" if value is None else value
    raw = value.encode("utf-8", errors="replace") if isinstance(value, str) else bytes(value)
    return raw[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace"), len(raw) > MAX_OUTPUT_BYTES


def _observe(command):
    executable = shutil.which(command[0], path=SYSTEM_PATH)
    result = {"command": list(command), "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
              "status": "not_available", "reason": "binary_not_found", "returncode": None,
              "stdout": "", "stderr": "", "stdout_truncated": False, "stderr_truncated": False,
              "output_encoding": "utf-8_with_replacement", "output_limit_bytes_per_stream": MAX_OUTPUT_BYTES}
    if executable is None:
        return result
    result["command"][0] = executable
    try:
        completed = subprocess.run(result["command"], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=COMMAND_TIMEOUT_SECONDS,
            check=False, shell=False, env=dict(_COMMAND_ENVIRONMENT))
        stdout, stderr = completed.stdout, completed.stderr
        result.update(returncode=completed.returncode,
                      status="observed" if completed.returncode == 0 else "not_available",
                      reason=None if completed.returncode == 0 else "local_command_failed")
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout, error.stderr
        result["reason"] = "command_timeout"
    except OSError:
        # Do not copy exception strings: they are not clock evidence and may
        # contain incidental host information. Ordinary stderr stays bounded.
        result["reason"] = "local_command_unavailable"
        return result
    result["stdout"], result["stdout_truncated"] = _bounded_output(stdout)
    result["stderr"], result["stderr_truncated"] = _bounded_output(stderr)
    return result


def collect_clock_preflight() -> dict:
    """Observe this host only; never contact a time server or alter a service.

    An NTP synchronization boolean is reported separately from UTC accuracy.
    Without an independent reference, offset and uncertainty remain unknown.
    Durations use monotonic_ns, even if local UTC jumps during the observation.
    """
    utc_start = _utc_now()
    monotonic_start = time.monotonic_ns()
    ntp = _observe(("timedatectl", "show", "--property=NTPSynchronized", "--value"))
    chrony = _observe(("chronyc", "-n", "-h", CHRONY_SOCKET, "tracking"))
    monotonic_end = time.monotonic_ns()
    utc_end = _utc_now()
    synchronized = None
    if ntp["status"] == "observed" and not ntp["stdout_truncated"]:
        synchronized = {"yes": True, "no": False}.get(ntp["stdout"].strip())
    monotonic_valid = monotonic_end >= monotonic_start
    return {"schema_version": SCHEMA_VERSION, "scope": SCOPE, "status": "LOCAL_OBSERVATION_RECORDED",
        "request_start_utc": utc_start.isoformat(), "request_end_utc": utc_end.isoformat(),
        "monotonic_ns_start": monotonic_start, "monotonic_ns_end": monotonic_end,
        "observation_duration_monotonic_ns": monotonic_end - monotonic_start if monotonic_valid else None,
        "monotonic_observation_status": "ordered" if monotonic_valid else "nonmonotonic_observation",
        "local_UTC_request_ordered": utc_end >= utc_start,
        "ntp_synchronized": synchronized, "utc_offset_seconds": None, "utc_uncertainty_seconds": None,
        "independent_utc_reference": None, "utc_accuracy_status": "not_measured_no_independent_reference",
        "observations": {"timedatectl_ntp_synchronized": ntp, "chronyc_tracking": chrony},
        "clock_or_service_modified": False, "external_time_request_performed": False,
        "GPU_inspected": False, "credential_environment_copied": False}
