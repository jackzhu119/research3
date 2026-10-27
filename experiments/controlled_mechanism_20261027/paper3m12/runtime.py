"""Four persistent M12 workers; queue observation and telemetry are independent.

This module imports only the standard library. It never cancels an allocation,
signals a scientific process, changes resource limits, or imports Torch in the
CPU controller. Scientific completion belongs to worker-written receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

from .common import (atomic_json, best_effort_json, digest, emit, load_plan, lock,
                     read_json, safe_read, sha_file, task_folder, utc, verify_result)

SLOTS = 4
TERMINAL = {'COMPLETE', 'BLOCKED', 'BLOCKED_DEPENDENCY', 'SKIPPED_DECISION'}
LIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}
_RECEIPTS = {}
_INVALID_GATE = object()


def runtime_dir(plan):
    return Path(plan.get('runtime', Path(plan['output']) / 'runtime'))


def marker(plan):
    return 'paper3m12-' + plan['recipe_sha256'][:16]


def _state_path(plan, task):
    return runtime_dir(plan) / 'tasks' / (task['key'] + '.json')


def _receipt(plan, task):
    path = task_folder(plan, task) / 'result.json'
    try:
        info = path.stat()
        stamp = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    except OSError:
        return None
    cache_key = (str(path), digest(task), plan.get('recipe_sha256'))
    previous = _RECEIPTS.get(cache_key)
    if previous is not None and previous[0] == stamp:
        return previous[1]
    result = verify_result(plan, task)
    _RECEIPTS[cache_key] = (stamp, result)
    if len(_RECEIPTS) > 4096:
        _RECEIPTS.pop(next(iter(_RECEIPTS)))
    return result


def task_states(plan):
    """Small records only: no source tensors, full predictions or archive reads."""
    result = {}
    for task in plan['tasks']:
        path = _state_path(plan, task)
        try:
            row = read_json(path, {}) or {}
        except (OSError, ValueError, TypeError) as error:
            row = dict(phase='STATE_OBSERVATION_RETRY', current_error=dict(error_type=type(error).__name__,
                       reason='Operational task record unreadable; retry observation without changing scientific state'))
        if not isinstance(row, dict):
            row = dict(phase='STATE_OBSERVATION_RETRY', current_error=dict(error_type='InvalidTaskRecord'))
        if _receipt(plan, task) is not None:
            row = dict(row, phase='COMPLETE', current_error=None)
        elif row.get('phase') == 'COMPLETE':
            row = dict(row, phase='COMPLETION_OBSERVATION_RETRY',
                       current_error=dict(error_type='CompletionReceiptMissingOrChanged'))
        row.setdefault('phase', 'NEW')
        result[task['key']] = row
    return result


def _dependencies(task):
    return task.get('dependencies', task.get('depends_on', []))


def _gate_result(plan, task, tasks):
    gate = task.get('gate')
    if not gate:
        return None
    parent = tasks.get(gate.get('task'))
    if parent is None:
        raise ValueError('Gate task is not in this M12 plan')
    receipt = _receipt(plan, parent)
    if receipt is None:
        return None
    value = receipt
    for part in str(gate.get('field', 'decision')).split('.'):
        if not isinstance(value, dict) or part not in value:
            return _INVALID_GATE
        value = value[part]
    return value == gate.get('equals', 'GO')


def _resolve(plan, task, states, tasks):
    dependencies = [states[key]['phase'] for key in _dependencies(task)]
    if any(value == 'SKIPPED_DECISION' for value in dependencies) and not task.get('allow_skipped_dependencies'):
        return 'SKIPPED_DECISION'
    if any(value.startswith('BLOCKED') for value in dependencies) and not task.get('allow_blocked_dependencies'):
        return 'BLOCKED_DEPENDENCY'
    allowed = {'COMPLETE'}
    if task.get('allow_skipped_dependencies'):
        allowed.add('SKIPPED_DECISION')
    if task.get('allow_blocked_dependencies'):
        allowed.update({'BLOCKED', 'BLOCKED_DEPENDENCY'})
    if any(value not in allowed for value in dependencies):
        return 'WAITING_DEPENDENCY'
    gate = _gate_result(plan, task, tasks)
    if gate is _INVALID_GATE:
        return 'BLOCKED_DEPENDENCY'
    if gate is False:
        return 'SKIPPED_DECISION'
    if task.get('gate') and gate is None:
        return 'WAITING_DEPENDENCY'
    return 'READY'


def _refresh_terminal_dependencies(plan, states):
    tasks = {t['key']: t for t in plan['tasks']}
    changed = True
    while changed:
        changed = False
        for task in plan['tasks']:
            row = states[task['key']]
            if row['phase'] in TERMINAL or row['phase'] == 'RUNNING':
                continue
            resolution = _resolve(plan, task, states, tasks)
            if resolution in {'SKIPPED_DECISION', 'BLOCKED_DEPENDENCY'}:
                value = dict(row, phase=resolution, task_key=task['key'],
                             task_identity=digest(task), updated_utc=utc(),
                             reason='Predeclared research decision' if resolution.startswith('SKIPPED')
                             else 'An upstream task is scientifically blocked')
                atomic_json(_state_path(plan, task), value)
                states[task['key']] = value
                changed = True
    return states


def claim_next(plan, slot, job_id, *, now=time.time):
    runtime = runtime_dir(plan)
    with lock(runtime / 'claim.lock'):
        states = _refresh_terminal_dependencies(plan, task_states(plan))
        tasks = {task['key']: task for task in plan['tasks']}
        for task in plan['tasks']:
            row = states[task['key']]
            if (row['phase'] in TERMINAL or row['phase'].endswith('OBSERVATION_RETRY')
                    or row.get('retry_after', 0) > now()):
                continue
            if _resolve(plan, task, states, tasks) != 'READY':
                continue
            path = runtime / 'task-locks' / (hashlib.sha256(task['key'].encode()).hexdigest() + '.lock')
            path.parent.mkdir(parents=True, exist_ok=True)
            lease = path.open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            # Completion may be published after the initial task-state snapshot
            # but before its lease becomes available. Recheck *under the lease*
            # before treating that finished worker as a lost process.
            if _receipt(plan, task) is not None:
                fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
                continue
            try:
                latest = read_json(_state_path(plan, task), {}) or {}
            except (OSError, ValueError, TypeError):
                fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
                continue
            if latest.get('phase') in TERMINAL or latest.get('retry_after', 0) > now():
                fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
                continue
            row = latest or row
            # A free lock with an old RUNNING record is a crashed scientific
            # process, not a second active fit. Count it before resuming.
            failures = int(row.get('failures', 0))
            if row['phase'] == 'RUNNING':
                failures += 1
            if failures >= int(plan.get('max_task_failures', 3)):
                value = dict(row, phase='BLOCKED', failures=failures,
                             current_error=dict(error_type='RepeatedWorkerLoss'), updated_utc=utc())
                atomic_json(_state_path(plan, task), value)
                fcntl.flock(lease, fcntl.LOCK_UN); lease.close()
                continue
            token = hashlib.sha256(os.urandom(32)).hexdigest()
            value = dict(row, phase='RUNNING', task_key=task['key'], task_identity=digest(task),
                         slot=slot, job_id=str(job_id), pid=os.getpid(), attempt_token=token,
                         attempt=int(row.get('attempt', 0)) + 1, failures=failures,
                         current_error=None, updated_utc=utc(), started_unix=now())
            atomic_json(_state_path(plan, task), value)
            return task, value, lease
    return None


def _run(command, runner=None, timeout=20):
    return (runner or subprocess.run)(command, capture_output=True, text=True,
                                      timeout=timeout, check=False)


def observe_queue(plan, runner=None):
    command = ['squeue', '--noheader', '--user=' + pwd.getpwuid(os.getuid()).pw_name,
               '--format=%i|%j|%T|%N|%R']
    response = _run(command, runner)
    if response.returncode:
        raise RuntimeError('SlurmQueueObservationFailed')
    pattern = re.compile(r'^' + re.escape(marker(plan)) + r'-s([0-3])-a([0-9]+)$')
    rows = []
    for line in response.stdout.splitlines():
        parts = line.strip().split('|', 4)
        if len(parts) != 5:
            continue
        match = pattern.fullmatch(parts[1])
        if not match or not re.fullmatch(r'[0-9]+', parts[0]):
            continue
        rows.append(dict(job_id=parts[0], name=parts[1], slot=int(match[1]),
                         attempt=int(match[2]), state=parts[2], node=parts[3], reason=parts[4]))
    return rows


def _account(job_id, runner=None):
    response = _run(['sacct', '--noheader', '--parsable2', '--jobs=' + str(job_id),
                     '--format=JobIDRaw,State,ExitCode'], runner)
    if response.returncode:
        return None
    for line in response.stdout.splitlines():
        pieces = line.strip().split('|')
        if len(pieces) >= 3 and pieces[0] == str(job_id):
            state = pieces[1].split()[0].rstrip('+')
            return dict(state=state, exit_code=pieces[2])
    return None


def _batch(plan_path, plan, slot, attempt):
    runtime = runtime_dir(plan)
    path = runtime / 'scripts' / ('slot-%d-attempt-%d.sh' % (slot, attempt))
    python = plan.get('python', sys.executable)
    entry = Path(__file__).with_name('entry.py')
    command = [python, '-I', '-S', '-B', '-u', str(entry), 'worker', '--plan', str(plan_path),
               '--slot', str(slot), '--device', 'cuda:0']
    lines = ['#!/bin/bash', 'set -e', 'export PYTHONDONTWRITEBYTECODE=1',
             'export CUBLAS_WORKSPACE_CONFIG=:4096:8',
             'export OMP_NUM_THREADS=16', 'export MKL_NUM_THREADS=16',
             'exec ' + ' '.join(shlex.quote(str(value)) for value in command), '']
    from .common import atomic_bytes
    atomic_bytes(path, '\n'.join(lines).encode())
    return path


def scheduler_step(plan_path, plan, *, runner=None, submit=False, observe_only=False, now=time.time):
    runtime = runtime_dir(plan)
    runtime.mkdir(parents=True, exist_ok=True)
    prior = read_json(runtime / 'scheduler-state.json', {}) or {}
    if prior and prior.get('marker') != marker(plan):
        raise ValueError('Another experiment owns this scheduler history')
    live = observe_queue(plan, runner)
    slots = prior.get('slots', {str(i): {'attempts': []} for i in range(SLOTS)})
    for slot in range(SLOTS):
        slots.setdefault(str(slot), {'attempts': []})
        attempts = slots[str(slot)].setdefault('attempts', [])
        observed = [row for row in live if row['slot'] == slot]
        for row in observed:
            item = next((a for a in attempts if a.get('job_id') == row['job_id']), None)
            if item is None:
                item = dict(attempt=row['attempt'], name=row['name'], job_id=row['job_id'],
                            status='ADOPTED', submitted_unix=now())
                attempts.append(item)
            item.update(slurm_state=row['state'], node=row['node'], reason=row['reason'],
                        status='ACKNOWLEDGED', observed_utc=utc())
        live_ids = {row['job_id'] for row in observed}
        for item in attempts:
            if item.get('job_id') and item['job_id'] not in live_ids and item.get('slurm_state') in LIVE:
                accounted = _account(item['job_id'], runner)
                if accounted and accounted['state'] not in LIVE:
                    item.update(slurm_state=accounted['state'], exit_code=accounted['exit_code'], status='FINISHED')
                else:
                    item['status'] = 'AWAITING_ACCOUNTING'
    states = _refresh_terminal_dependencies(plan, task_states(plan))
    remaining = [key for key, row in states.items() if row['phase'] not in TERMINAL]
    actual_live = len(live)
    if submit and not observe_only and remaining:
        for slot in range(SLOTS):
            attempts = slots[str(slot)]['attempts']
            observed = [row for row in live if row['slot'] == slot]
            if observed or actual_live >= SLOTS:
                continue
            if attempts and attempts[-1].get('status') in {
                    'SUBMITTING', 'SUBMISSION_UNKNOWN', 'AWAITING_ACCOUNTING'}:
                continue  # Reconcile an ambiguous RPC; never submit a duplicate.
            if attempts and attempts[-1].get('retry_after', 0) > now():
                continue
            if attempts and attempts[-1].get('job_id') and attempts[-1].get('status') != 'FINISHED':
                continue
            maximum = int(plan.get('max_worker_attempts', 12))
            if len(attempts) >= maximum:
                continue
            attempt = max([a.get('attempt', 0) for a in attempts] + [0]) + 1
            name = marker(plan) + '-s%d-a%d' % (slot, attempt)
            script = _batch(plan_path, plan, slot, attempt)
            item = dict(attempt=attempt, name=name, status='SUBMITTING', submitted_unix=now(),
                        slurm_state='SUBMISSION_PENDING')
            attempts.append(item)
            state = dict(marker=marker(plan), slots=slots, updated_utc=utc())
            atomic_json(runtime / 'scheduler-state.json', state)  # Persist before RPC.
            command = ['sbatch', '--parsable', '--job-name=' + name, '--partition=hpc_gpu',
                       '--nodes=1', '--ntasks=1', '--cpus-per-task=16', '--mem=64G',
                       '--gres=gpu:1', '--time=5-03:00:00', '--signal=B:USR1@120',
                       '--output=' + str(runtime / 'logs' / ('slot-%d-%%j.out' % slot)),
                       '--error=' + str(runtime / 'logs' / ('slot-%d-%%j.err' % slot)), str(script)]
            (runtime / 'logs').mkdir(exist_ok=True)
            try:
                response = _run(command, runner, timeout=40)
                candidate = response.stdout.strip().split(';', 1)[0]
                if response.returncode == 0 and re.fullmatch(r'[0-9]+', candidate):
                    item.update(job_id=candidate, slurm_state='PENDING', status='ACKNOWLEDGED')
                    actual_live += 1
                elif response.returncode != 0:
                    item.update(status='FINISHED', slurm_state='REJECTED',
                                error_type='SlurmSubmissionRejected', retry_after=now() + 60)
                else:
                    item.update(status='SUBMISSION_UNKNOWN', error_type='UnparseableSubmissionAcknowledgement')
            except (OSError, subprocess.TimeoutExpired) as error:
                item.update(status='SUBMISSION_UNKNOWN', error_type=type(error).__name__)
            atomic_json(runtime / 'scheduler-state.json', dict(marker=marker(plan), slots=slots,
                                                               updated_utc=utc()))
    jobs = []
    for slot in range(SLOTS):
        attempts = slots[str(slot)]['attempts']
        if attempts:
            jobs.append(dict(slot=slot, **attempts[-1]))
    value = dict(marker=marker(plan), slots=slots, jobs=jobs, inflight_jobs=actual_live,
                 jobs_in_successful_queue_observation=len(live),
                 controller_status='RUNNING', observed_utc=utc(),
                 queue_observation_successful=True, remaining_tasks=len(remaining),
                 resources=dict(jobs=4, gpus_per_job=1, cpus_per_job=16, memory_per_job='64G'),
                 duplicate_slot_allocations=[slot for slot in range(SLOTS)
                                            if sum(row['slot'] == slot for row in live) > 1])
    atomic_json(runtime / 'scheduler-state.json', value)
    return value


def _task_progress(plan, task, row):
    progress = safe_read(task_folder(plan, task) / 'progress.json', {}) or {}
    current = progress.get('attempt_token') in (None, row.get('attempt_token'))
    return dict(key=task['key'], kind=task.get('kind'), phase=row['phase'],
                slot=row.get('slot'), job_id=row.get('job_id'),
                attempt=row.get('attempt', 0), current_error=row.get('current_error'),
                progress=progress if current else {}, progress_is_current_attempt=current)


def status(plan_or_path):
    plan = plan_or_path if isinstance(plan_or_path, dict) else load_plan(plan_or_path)[1]
    runtime = runtime_dir(plan)
    states = task_states(plan)
    counts = Counter(row['phase'] for row in states.values())
    scheduler = safe_read(runtime / 'scheduler-state.json', {}) or {}
    running = [_task_progress(plan, t, states[t['key']]) for t in plan['tasks']
               if states[t['key']]['phase'] == 'RUNNING']
    all_terminal = all(row['phase'] in TERMINAL for row in states.values())
    if counts['BLOCKED'] or counts['BLOCKED_DEPENDENCY']:
        overall = 'PARTIAL_BLOCKED' if all_terminal else 'RUNNING_WITH_BLOCKED_TASKS'
    elif all_terminal and counts['SKIPPED_DECISION']:
        overall = 'STOPPED_BY_PREDECLARED_RESEARCH_RULE'
    elif all_terminal:
        overall = 'COMPLETE'
    else:
        overall = 'RUNNING_OR_READY'
    value = dict(schema_version='paper3-m12-operational-status-v1', recipe_sha256=plan['recipe_sha256'],
                 status=overall, updated_utc=utc(), total_tasks=len(plan['tasks']),
                 complete_tasks=counts['COMPLETE'], skipped_by_decision=counts['SKIPPED_DECISION'],
                 blocked_tasks=counts['BLOCKED'] + counts['BLOCKED_DEPENDENCY'],
                 task_phase_counts=dict(counts), current_tasks=running,
                 scheduler={k: scheduler.get(k) for k in ['marker', 'jobs', 'inflight_jobs',
                            'controller_status', 'observed_utc', 'queue_observation_successful']},
                 workers=[safe_read(runtime / 'workers' / ('slot-%d.json' % i), {}) for i in range(SLOTS)],
                 operational_observation_errors=[dict(task_key=key, **row.get('current_error', {}))
                     for key, row in states.items() if row['phase'].endswith('OBSERVATION_RETRY')],
                 automatic_child_refresh=False, scientific_completion_from_receipts=True,
                 monitoring_full_scientific_payloads_rehashed=False)
    best_effort_json(runtime / 'status.json', value)
    return value


def completed_task_rows(plan):
    return [(task, _receipt(plan, task)) for task in plan['tasks'] if _receipt(plan, task) is not None]


def _controller_identity(pid, plan_path, token):
    try:
        pid = int(pid)
        if pid <= 1:
            return None
        folder = Path('/proc') / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in {'Z', 'X', 'x'}:
            return None
        argv = [part.decode() for part in (folder / 'cmdline').read_bytes().split(b'\0') if part]
        entry = str(Path(__file__).with_name('entry.py'))
        if (entry not in argv or 'controller' not in argv or '--plan' not in argv
                or argv[argv.index('--plan') + 1] != str(plan_path)
                or '--token' not in argv or argv[argv.index('--token') + 1] != token):
            return None
        return dict(pid=pid, start_ticks=fields[19])
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def _start_sidecars(plan_path, plan):
    enabled = plan.get('start_sidecars', not plan.get('synthetic_fixture', False))
    if not enabled:
        return dict(status='SIDECARS_DISABLED_SOFTWARE_FIXTURE')
    try:
        return importlib.import_module('paper3m12.background').ensure_sidecars(plan_path)
    except Exception as error:
        value = dict(status='SIDECAR_START_RETRY_NEEDED', error_type=type(error).__name__,
                     updated_utc=utc(), training_controlled=False)
        best_effort_json(runtime_dir(plan) / 'sidecars-start-error.json', value)
        emit('M12_SIDECAR_START_RETRY', error_type=type(error).__name__, training_controlled=False)
        return value


def start(plan_path, *, submit=False, popen=subprocess.Popen, now=time.monotonic, sleep=time.sleep):
    if not submit:
        raise ValueError('Explicit --submit is required to start M12 Slurm management')
    plan_path, plan = load_plan(plan_path)
    runtime = runtime_dir(plan); runtime.mkdir(parents=True, exist_ok=True)
    with lock(runtime / 'start.lock'):
        prior = safe_read(runtime / 'controller-process.json', {}) or {}
        identity = _controller_identity(prior.get('pid', 0), plan_path, prior.get('launch_token'))
        if identity is not None:
            ready = safe_read(runtime / 'controller-ready.json', {}) or {}
            matched = (ready.get('pid') == identity['pid'] and ready.get('launch_token') == prior.get('launch_token')
                       and ready.get('first_slurm_observation_completed') is True)
            return dict(status='EXISTING_M12_CONTROLLER_READY' if matched else 'EXISTING_M12_STARTUP_PENDING',
                        pid=identity['pid'], runtime=str(runtime), scientific_jobs_restarted=0,
                        sidecars=_start_sidecars(plan_path, plan))
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        command = [plan.get('python', sys.executable), '-I', '-S', '-B', '-u',
                   str(Path(__file__).with_name('entry.py')), 'controller', '--plan', str(plan_path),
                   '--token', token, '--submit']
        with (runtime / 'controller.log').open('ab', buffering=0) as log:
            child = popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                          start_new_session=True, close_fds=True)
        atomic_json(runtime / 'controller-process.json', dict(pid=child.pid, launch_token=token,
                    started_utc=utc(), plan=str(plan_path), recipe_sha256=plan['recipe_sha256']))
        until = now() + 8
        result = 'M12_STARTUP_PENDING'
        while now() < until:
            if child.poll() is not None:
                result = 'M12_CONTROLLER_START_FAILED'; break
            ready = safe_read(runtime / 'controller-ready.json', {}) or {}
            if (ready.get('pid') == child.pid and ready.get('launch_token') == token
                    and ready.get('first_slurm_observation_completed') is True):
                result = 'M12_CONTROLLER_READY'; break
            sleep(.1)
        value = dict(status=result, pid=child.pid, launch_token=token, runtime=str(runtime),
                     resources=dict(jobs=4, gpus_per_job=1, cpus_per_job=16, memory_per_job='64G'),
                     existing_rounds_cancelled=0, automatic_child_refresh=False,
                     terminal_can_close=True, training_progress_claimed=False)
        value['sidecars'] = (_start_sidecars(plan_path, plan) if result != 'M12_CONTROLLER_START_FAILED'
                             else dict(status='NOT_STARTED_CONTROLLER_FAILED'))
        atomic_json(runtime / 'STARTED.json', value)
        return value


def controller_loop(plan_path, *, submit=False, token=None, runner=None, stop=None, once=False):
    plan_path, plan = load_plan(plan_path)
    runtime = runtime_dir(plan); stop = stop or threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:  # Software fixtures may run in another thread.
            pass
    first = True
    with lock(runtime / 'controller.lock', blocking=False):
        while not stop.is_set():
            try:
                snapshot = scheduler_step(plan_path, plan, runner=runner, submit=submit, observe_only=first)
                if first:
                    atomic_json(runtime / 'controller-ready.json', dict(status='READY', pid=os.getpid(),
                        launch_token=token, first_slurm_observation_completed=True,
                        queue_observation_successful=True, updated_utc=utc(),
                        training_progress_claimed=False, automatic_child_refresh=False))
                    first = False
                current = status(plan)
                best_effort_json(runtime / 'controller-status.json', dict(snapshot, task_status=current,
                                                                         updated_utc=utc()))
                if once:
                    return current
                if current['status'] in {'COMPLETE', 'STOPPED_BY_PREDECLARED_RESEARCH_RULE', 'PARTIAL_BLOCKED'}:
                    return current
            except Exception as error:
                best_effort_json(runtime / 'controller-error.json', dict(error_type=type(error).__name__,
                    reason=str(error)[:500], updated_utc=utc(), scientific_workers_signalled=0))
                emit('M12_CONTROLLER_OPERATION_RETRY', error_type=type(error).__name__)
                if once:
                    return dict(status='M12_QUEUE_OBSERVATION_PENDING', error_type=type(error).__name__)
            stop.wait(2 if first else 15)
    return dict(status='M12_MANAGEMENT_STOPPED', scientific_workers_signalled=0)


def worker_loop(plan_path, slot, *, device='cuda:0', run_task=None, stop=None, once=False, sleep=time.sleep):
    if slot not in range(SLOTS):
        raise ValueError('M12 worker slot must be 0..3')
    _, plan = load_plan(plan_path)
    runtime = runtime_dir(plan); stop = stop or threading.Event()
    job_id = os.environ.get('SLURM_JOB_ID', 'local-software-fixture')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        try:
            signal.signal(sig, lambda *_: stop.set())
        except ValueError:
            pass
    telemetry = dict(slot=slot, job_id=job_id, pid=os.getpid(), phase='IMPORTING',
                     updated_utc=utc(), numerical_progress={})
    done = threading.Event()

    def heartbeat():
        while not done.wait(20):
            best_effort_json(runtime / 'workers' / ('slot-%d.json' % slot),
                             dict(telemetry, heartbeat_utc=utc()))

    try:
        with lock(runtime / 'worker-slots' / ('slot-%d.lock' % slot), blocking=False):
            threading.Thread(target=heartbeat, daemon=True).start()
            best_effort_json(runtime / 'workers' / ('slot-%d.json' % slot), telemetry)
            scientific_run = run_task
            initial = status(plan)
            if initial['status'] in {'COMPLETE', 'STOPPED_BY_PREDECLARED_RESEARCH_RULE', 'PARTIAL_BLOCKED'}:
                return 0 if initial['status'] != 'PARTIAL_BLOCKED' else 2
            if scientific_run is None:
                scientific_run = importlib.import_module('paper3m12.worker').run_task
            while not stop.is_set():
                claimed = claim_next(plan, slot, job_id)
                if claimed is None:
                    current = status(plan)
                    telemetry.update(phase='WAITING_DEPENDENCY_OR_SLOT', updated_utc=utc())
                    best_effort_json(runtime / 'workers' / ('slot-%d.json' % slot), telemetry)
                    if current['status'] in {'COMPLETE', 'STOPPED_BY_PREDECLARED_RESEARCH_RULE', 'PARTIAL_BLOCKED'}:
                        return 0 if current['status'] != 'PARTIAL_BLOCKED' else 2
                    if once:
                        return 0
                    stop.wait(3); continue
                task, row, lease = claimed
                telemetry.update(phase='RUNNING', task_key=task['key'], task_kind=task.get('kind'),
                                 attempt_token=row['attempt_token'], updated_utc=utc(), numerical_progress={})

                def progress(**values):
                    values = dict(values, attempt_token=row['attempt_token'], task_key=task['key'], updated_utc=utc())
                    telemetry.update(numerical_progress=values, last_activity_utc=utc())
                    if (values.get('numerical_advance') is True
                            or any(key in values for key in ('step', 'iteration', 'outer_update', 'session',
                                                            'completed_units', 'updates'))):
                        telemetry['last_numerical_advance_utc'] = utc()
                    best_effort_json(task_folder(plan, task) / 'progress.json', values)
                    best_effort_json(runtime / 'workers' / ('slot-%d.json' % slot), telemetry)

                progress(stage='STARTING', scientific_fit_started=False)
                emit('M12_TASK_STARTED', task_key=task['key'], slot=slot, job_id=job_id,
                     attempt=row['attempt'])
                paused = False
                try:
                    scientific_run(plan, task, device=device, progress=progress, stop=stop)
                    result = verify_result(plan, task)
                    if result is None:
                        raise ValueError('Scientific worker did not seal the required completion receipt')
                    row.update(phase='COMPLETE', current_error=None, updated_utc=utc(),
                               result_sha256=sha_file(task_folder(plan, task) / 'result.json'))
                    atomic_json(_state_path(plan, task), row)
                    emit('M12_TASK_COMPLETE', task_key=task['key'], slot=slot, job_id=job_id)
                except BaseException as error:
                    if isinstance(error, (SystemExit, GeneratorExit)):
                        raise
                    paused = (type(error).__name__ in {'PauseRequested', 'Paused'}
                              or isinstance(error, KeyboardInterrupt) or stop.is_set())
                    failures = row.get('failures', 0) + (0 if paused else 1)
                    permanent = isinstance(error, (ValueError, TypeError, KeyError, AssertionError))
                    phase = ('PAUSED' if paused else 'BLOCKED' if permanent or failures >= int(plan.get('max_task_failures', 3))
                             else 'RETRY_WAIT')
                    row.update(phase=phase, failures=failures, updated_utc=utc(),
                        retry_after=time.time() + (0 if paused else min(30 * failures, 120)),
                        current_error=dict(error_type=type(error).__name__, reason=str(error)[:700]))
                    atomic_json(_state_path(plan, task), row)
                    try:
                        error_path = task_folder(plan, task) / ('error-attempt-%d.log' % row['attempt'])
                        error_path.parent.mkdir(parents=True, exist_ok=True)
                        error_path.write_text(''.join(traceback.format_exception(type(error), error, error.__traceback__)))
                    except OSError:
                        pass
                    emit('M12_TASK_PAUSED' if paused else 'M12_TASK_ERROR', task_key=task['key'],
                         slot=slot, error_type=type(error).__name__, phase=phase)
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                    lease.close()
                    telemetry.update(phase='IDLE', task_key=None, updated_utc=utc())
                    best_effort_json(runtime / 'workers' / ('slot-%d.json' % slot), telemetry)
                if paused:
                    return 75
                if once:
                    return 0
            return 75
    except BlockingIOError:
        emit('M12_EXISTING_SLOT_WORKER_REUSED', slot=slot, scientific_workers_signalled=0)
        return 0
    finally:
        done.set()


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('command', choices=('start', 'controller', 'worker', 'status'))
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--token')
    parser.add_argument('--slot', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.command == 'worker':
        if args.slot is None:
            parser.error('worker requires --slot')
        return worker_loop(args.plan, args.slot, device=args.device)
    if args.command == 'start':
        value = start(args.plan, submit=args.submit)
    elif args.command == 'controller':
        value = controller_loop(args.plan, submit=args.submit, token=args.token)
    else:
        value = status(args.plan)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 2 if value.get('status') == 'M12_CONTROLLER_START_FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
