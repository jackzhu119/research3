"""Detached report/archive helpers, independent of scientific job control."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

from .common import best_effort_json, load_plan, lock, safe_read, utc


def _identity(pid, command):
    try:
        if int(pid) <= 1:
            return None
        folder = Path('/proc') / str(int(pid))
        if folder.stat().st_uid != os.getuid():
            return None
        actual = folder.joinpath('cmdline').read_bytes().split(b'\0')
        if actual and actual[-1] == b'':
            actual.pop()
        if actual != [os.fsencode(value) for value in command]:
            return None
        fields = folder.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        return dict(pid=int(pid), start_ticks=fields[19])
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _find_existing(command):
    """Recover an exact same-user helper after a lost telemetry receipt.

    Only inspect process metadata. No process is ever signalled or modified;
    workers and helpers for another plan/role do not match the complete argv.
    """
    try:
        folders = Path('/proc').iterdir()
        for folder in folders:
            if folder.name.isdigit():
                identity = _identity(int(folder.name), command)
                if identity is not None:
                    return identity
    except OSError:
        pass
    return None


def ensure_sidecars(plan_path, *, popen=subprocess.Popen):
    """Adopt exact existing helpers or start missing ones; never stop any process.

    Every helper has its own loop and lock. A missing credential, HTTP error or
    report error remains inside that helper and has no scientific stop channel.
    """
    plan_path, plan = load_plan(plan_path)
    runtime = Path(plan['runtime'])
    runtime.mkdir(parents=True, exist_ok=True)
    commands = {
        'reporter': ['report-loop'],
        'paper-upload': ['upload', '--channel', 'paper'],
        'full-upload': ['upload', '--channel', 'full'],
    }
    result = {}
    with lock(runtime / 'sidecar-launch.lock'):
        for name, tail in commands.items():
            command = [plan.get('python', os.sys.executable), '-I', '-S', '-B', '-u',
                       plan['worker_entry'], *tail, '--plan', str(plan_path)]
            record_path = runtime / (name + '-process.json')
            prior = safe_read(record_path, {}) or {}
            identity = _identity(prior.get('pid'), command)
            if identity and (identity['start_ticks'] == prior.get('start_ticks')
                             or (prior.get('status') == 'HELPER_STARTUP_PENDING'
                                 and prior.get('command') == command
                                 and prior.get('start_ticks') is None)):
                value = dict(status='EXISTING_HELPER_REUSED', **identity,
                             updated_utc=utc(), command=command, training_controlled=False)
                best_effort_json(record_path, value)
                result[name] = value
                continue
            # Process records are operational telemetry. A failed atomic write
            # must not force a duplicate uploader next time start is invoked.
            identity = _find_existing(command)
            if identity is not None:
                value = dict(status='EXISTING_HELPER_DISCOVERED', **identity,
                             updated_utc=utc(), command=command, training_controlled=False)
                best_effort_json(record_path, value)
                result[name] = value
                continue
            try:
                with (runtime / (name + '.log')).open('ab', buffering=0) as log:
                    child = popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                  start_new_session=True, close_fds=True, cwd=plan['root'])
                # Delayed /proc visibility is pending, not a failed scientific
                # task and not authorization to launch another copy.
                identity = None
                for _ in range(10):
                    identity = _identity(child.pid, command)
                    if identity is not None or child.poll() is not None:
                        break
                    time.sleep(.01)
                phase = ('HELPER_STARTED' if identity else 'HELPER_STARTUP_PENDING'
                         if child.poll() is None else 'HELPER_EXITED_EARLY')
                value = dict(status=phase,
                             pid=child.pid, start_ticks=(identity or {}).get('start_ticks'),
                             updated_utc=utc(), command=command, training_controlled=False)
                best_effort_json(record_path, value)
                result[name] = value
            except Exception as error:
                result[name] = dict(status='HELPER_START_RETRY_REQUIRED',
                                    error_type=type(error).__name__, training_controlled=False)
                best_effort_json(runtime / (name + '-start-error.json'), result[name])
    return result
