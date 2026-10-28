"""One durable R1 GPU allocation; native science and output identities retained.

The detached manager invokes this module once in each fresh metadata process.
It must own the original controller.lock before invoking a cycle. This module
never signals or cancels an existing scientific process. Old allocations drain
their current actions before a replacement allocation can be submitted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
OPERATIONAL = 'runs/training_campaign_parallel_v1'
REPAIR_ID = 'paper3-r1-single-gpu-4cpu-64g-v1'
ACTIVE = frozenset(('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
    'REQUEUED', 'REQUEUE_FED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'))
RECORD_LIMIT = 8 * 1024 * 1024
BATCH = '''#!/bin/bash
set -u
printf 'R1_SINGLE_GPU_PROCESS_STARTED job=%s node=%s time=%s\\n' "${SLURM_JOB_ID:-}" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"$@" &
worker_pid=$!
interrupted=0
forward_usr1() { interrupted=1; kill -USR1 "$worker_pid" 2>/dev/null || true; }
forward_term() { interrupted=1; kill -TERM "$worker_pid" 2>/dev/null || true; }
trap forward_usr1 USR1
trap forward_term TERM INT
while :; do
  interrupted=0
  wait "$worker_pid"
  code=$?
  if [ "$interrupted" -eq 0 ]; then break; fi
done
exit "$code"
'''


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=utc(), **fields), ensure_ascii=False), flush=True)


def read(path, default=None):
    try:
        with Path(path).open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > RECORD_LIMIT:
                raise ValueError('R1_OPERATIONAL_RECORD_TOO_LARGE: ' + str(path))
            raw = handle.read(RECORD_LIMIT + 1)
        if len(raw) > RECORD_LIMIT:
            raise ValueError('R1_OPERATIONAL_RECORD_TOO_LARGE: ' + str(path))
        return json.loads(raw)
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.r1-single-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def lock(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield handle


def resources(partition='hpc_gpu'):
    return dict(partition=partition, jobs=1, nodes_per_job=1, gpus_per_job=1,
        gpus_total=1, cpus_per_job=4, cpus_total=4, memories=['64G'],
        memory_per_job='64G', memory_total='64G')


def route_frontier(frontier):
    """Retain every scientific field and barrier; route all work to slot zero."""
    value = dict(frontier)
    ready = [dict(action, affinity_slot=0) for action in frontier.get('ready_actions', [])]
    value['ready'] = value['ready_actions'] = ready
    value['resource_policy'] = dict(resources(), repair_id=REPAIR_ID,
        native_action_identities_changed=False, phase_barriers_preserved=True,
        extension_arms_run_sequentially=True, timing_observations_relabelled=False)
    return value


def _normal_state(value):
    return value.strip().split()[0].rstrip('+').upper() if value.strip() else 'UNKNOWN'


class SingleScheduler:
    """Durable intent-before-submit, exact-name reconciliation, one live intent."""
    def __init__(self, root, work, launch, worker_command, *, runner=None, clock=None):
        self.root, self.work = Path(root).resolve(), Path(work).resolve()
        self.runtime = self.root / OPERATIONAL
        self.work.mkdir(parents=True, exist_ok=True)
        (self.runtime / 'logs').mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / 'allocation-state.json'
        self.batch_path = self.work / 'single-gpu-worker.sbatch'
        self.launch = dict(launch)
        if self.launch.get('partition') != 'hpc_gpu':
            raise ValueError('RETAINED_R1_PARTITION_MUST_BE_HPC_GPU')
        if not isinstance(worker_command, list) or not worker_command or any(
            not isinstance(v, str) for v in worker_command):
            raise ValueError('COMPLETE_SINGLE_WORKER_COMMAND_REQUIRED')
        self.worker_command = [v.replace('{slot}', '0') for v in worker_command]
        identity = [str(self.root), str(self.work), REPAIR_ID]
        self.marker = 'paper3r1s-' + hashlib.sha256(
            json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:16]
        self.pattern = re.compile(re.escape(self.marker) + r'-s0-a([1-9]\d*)$')
        self.runner, self.clock = runner or subprocess.run, clock or time.time
        self.state = read(self.state_path)
        if self.state is None:
            self.state = dict(schema_version=REPAIR_ID, marker=self.marker,
                root=str(self.root), created_at=self.clock(), attempts=[],
                next_submit_at=0, rejections=0, controller_status='NOT_STARTED')
        if not isinstance(self.state, dict) or self.state.get('marker') != self.marker or not isinstance(
            self.state.get('attempts'), list):
            raise ValueError('SINGLE_ALLOCATION_HISTORY_DIFFERS_OR_CORRUPT')
        legacy = read(self.runtime/'scheduler-state.json', {}) or {}
        legacy_created = legacy.get('created_at')
        self.query_created_at = min(self.state['created_at'], legacy_created) if type(
            legacy_created) in (int, float) else self.state['created_at']
        self.queue = []
        self.accounting = []
        if not self.batch_path.exists() or self.batch_path.read_text() != BATCH:
            fd, temporary = tempfile.mkstemp(prefix='.r1-single-batch-', dir=self.work)
            try:
                with os.fdopen(fd, 'w') as handle:
                    handle.write(BATCH)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o700)
                os.replace(temporary, self.batch_path)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def save(self):
        self.state['updated_at'] = self.clock()
        self.state['inflight_jobs'] = len(self.unresolved())
        atomic(self.state_path, self.state)

    def unresolved(self):
        return [a for a in self.state['attempts'] if a.get('status') not in ('FINISHED', 'REJECTED')]

    def _run(self, command, timeout=30):
        return self.runner(command, capture_output=True, text=True, timeout=timeout)

    def _owned(self, name, comment, job_id):
        match = self.pattern.fullmatch(name)
        if not match:
            return False
        if comment == self.marker:
            return True
        return comment in ('', '(null)', 'None', 'Unknown') and any(
            a.get('name') == name and a.get('job_id') in (None, job_id)
            for a in self.state['attempts'])

    def _adopt(self, row):
        sequence = int(self.pattern.fullmatch(row['name']).group(1))
        found = next((a for a in self.state['attempts'] if a['sequence'] == sequence), None)
        if found is None:
            found = dict(sequence=sequence, name=row['name'], marker=self.marker,
                job_id=None, status='INTENT', intent_at=self.clock(), origin='SLURM_RECOVERED')
            self.state['attempts'].append(found)
        if found.get('job_id') not in (None, row['job_id']):
            raise ValueError('ONE_ATTEMPT_NAME_HAS_MULTIPLE_JOBS')
        found['job_id'] = row['job_id']
        return found

    def observe(self):
        user = self.launch.get('user') or self.launch.get('account') or getpass.getuser()
        response = self._run(['squeue', '--noheader', '--user', user,
            '--format=%i|%k|%T|%j|%R|%N'])
        if response.returncode:
            raise RuntimeError('SQUEUE_QUERY_FAILED: ' + (response.stderr or '')[-2000:])
        queue = []
        for line in response.stdout.splitlines():
            fields = [v.strip() for v in line.split('|')]
            if len(fields) >= 4 and fields[0].isdigit():
                queue.append(dict(job_id=fields[0], comment=fields[1], state=_normal_state(fields[2]),
                    name=fields[3], queue_reason=fields[4] if len(fields)>4 else None,
                    node=fields[5] if len(fields)>5 else None))
        start = datetime.fromtimestamp(self.query_created_at - 86400, timezone.utc).strftime('%Y-%m-%d')
        response = self._run(['sacct', '--noheader', '--parsable2', '--allocations', '--user', user,
            '--starttime', start, '--format=JobID%80,State%40,ExitCode,Comment%100,JobName%150'])
        if response.returncode:
            raise RuntimeError('SACCT_QUERY_FAILED: ' + (response.stderr or '')[-2000:])
        accounting = []
        for line in response.stdout.splitlines():
            fields = [v.strip() for v in line.split('|')]
            if len(fields) >= 5 and fields[0].isdigit():
                code = fields[2].split(':')
                accounting.append(dict(job_id=fields[0], state=_normal_state(fields[1]), comment=fields[3],
                    name=fields[4], exit_code=int(code[0]) if code[0].isdigit() else None,
                    exit_signal=int(code[1]) if len(code)>1 and code[1].isdigit() else None))
        self.queue, self.accounting = queue, accounting
        mine = {}
        for row in accounting:
            if self._owned(row['name'], row['comment'], row['job_id']):
                attempt = self._adopt(row)
                mine[row['job_id']] = row
                if row['state'].startswith('CANCELLED') and not attempt.get('cancelled_observed'):
                    attempt['cancelled_observed'] = True
                    (self.runtime/'STOP').touch()
                    atomic(self.work/'cancelled-by-user.json', dict(job_id=row['job_id'], name=attempt['name'],
                        state=row['state'], updated_utc=utc(), automatic_replacement_disabled=True))
                    emit('R1_SINGLE_GPU_CANCELLED_STOP_RETAINED', job_id=row['job_id'])
        current = {}
        for row in queue:
            if self._owned(row['name'], row['comment'], row['job_id']):
                self._adopt(row)
                current[row['job_id']] = row
        for attempt in self.unresolved():
            job = attempt.get('job_id')
            if job in current:
                row = current[job]
                attempt.update(status='ACKNOWLEDGED', slurm_state=row['state'],
                    node=row.get('node'), queue_reason=row.get('queue_reason'), observed_at=self.clock())
                continue
            row = mine.get(job)
            if row is None or row['state'] in ACTIVE:
                local = read(self.runtime / 'scheduler-outcomes' / (str(job) + '.json'), {}) if job else {}
                if local.get('marker') == self.marker and local.get('name') == attempt['name'] and (
                    str(local.get('job_id')) == str(job)) and local.get('exit_code') == 0:
                    row = dict(job_id=job, state='COMPLETED', exit_code=0, exit_signal=0)
                else:
                    attempt.update(status='UNCERTAIN', error_type='AWAITING_QUEUE_OR_ACCOUNTING_EVIDENCE')
                    continue
            attempt.update(status='FINISHED', slurm_state=row['state'], exit_code=row.get('exit_code'),
                exit_signal=row.get('exit_signal'), finished_at=self.clock())
            self.state['next_submit_at'] = self.clock() + (0 if row.get('exit_code') == 0 else 30)
        if len(current) > 1:
            self.state['controller_status'] = 'MULTIPLE_OWNED_JOBS_OBSERVED_NO_NEW_SUBMISSION'
        elif (self.runtime/'STOP').exists():
            self.state['controller_status'] = 'STOPPED_BY_USER'
        elif self.unresolved():
            self.state['controller_status'] = ('AWAITING_SUBMISSION_EVIDENCE' if any(
                a['status']=='UNCERTAIN' for a in self.unresolved()) else 'RUNNING_OR_QUEUED')
        else:
            self.state['controller_status'] = 'OBSERVED_NO_ALLOCATION'
        self.save()
        return self.summary()

    def command(self, attempt):
        command = ['sbatch', '--parsable', '--job-name='+attempt['name'], '--comment='+self.marker,
            '--partition=hpc_gpu', '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=64G',
            '--gres=gpu:1', '--time='+str(self.launch.get('time_limit', '5-03:00:00')),
            '--signal=B:USR1@300', '--no-requeue', '--chdir=/tmp', '--open-mode=append',
            '--output='+str(self.runtime/'logs'/'single-gpu-%j.out'),
            '--error='+str(self.runtime/'logs'/'single-gpu-%j.err'),
            '--export=ALL,BASH_ENV=,ENV=,PYTHONHOME=,PYTHONPATH=,PYTHONUNBUFFERED=1,'
            'PAPER3_R1_SLOT=0,PAPER3_R1_MARKER='+self.marker+',PAPER3_R1_ATTEMPT_NAME='+attempt['name']]
        for key, flag in (('account','--account='), ('exclude','--exclude=')):
            if self.launch.get(key):
                command.append(flag+str(self.launch[key]))
        return command + [str(self.batch_path)] + self.worker_command

    def maybe_submit(self, *, wanted, allowed):
        if (not wanted or not allowed or self.unresolved() or (self.runtime/'STOP').exists()
                or self.clock() < self.state.get('next_submit_at', 0)):
            return self.summary()
        sequence = max([a['sequence'] for a in self.state['attempts']] or [0]) + 1
        attempt = dict(sequence=sequence, slot=0, marker=self.marker,
            name=self.marker+'-s0-a'+str(sequence), job_id=None, status='INTENT', intent_at=self.clock())
        attempt['command'] = self.command(attempt)
        self.state['attempts'].append(attempt)
        # Durable reservation survives a crash or lost sbatch acknowledgement.
        self.save()
        emit('R1_SINGLE_GPU_SUBMITTING', attempt=sequence, resources=resources())
        try:
            response = self._run(attempt['command'], timeout=45)
        except FileNotFoundError as error:
            attempt.update(status='REJECTED', error_type=type(error).__name__, message=str(error))
            self.state['next_submit_at'] = self.clock()+60
        except (OSError, subprocess.SubprocessError) as error:
            attempt.update(status='UNCERTAIN', error_type=type(error).__name__, message=str(error)[-2000:])
        else:
            match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?', response.stdout.strip()) if response.returncode==0 else None
            if match:
                attempt.update(status='ACKNOWLEDGED', job_id=match.group(1), acknowledged_at=self.clock(),
                    slurm_state='SUBMITTED')
                self.state['rejections'] = 0
                emit('R1_SINGLE_GPU_SUBMITTED', job_id=attempt['job_id'], resources=resources())
            else:
                reason = (response.stderr or '')[-4000:]
                explicit = response.returncode and re.search(
                    r'QOSMaxSubmitJobPerUserLimit|Job violates accounting/QOS policy|'
                    r'Invalid (?:generic resource|partition|account|qos)|'
                    r'Requested (?:node configuration is not available|time limit is invalid)|'
                    r'AssociationJobLimit|QOSMaxJobsPerUserLimit', reason, re.I)
                if explicit:
                    self.state['rejections'] += 1
                    delay = min(300, 15 * 2**min(self.state['rejections']-1, 5))
                    self.state['next_submit_at'] = self.clock()+delay
                    attempt.update(status='REJECTED', error_type='SLURM_SUBMISSION_REJECTED', message=reason)
                else:
                    attempt.update(status='UNCERTAIN', error_type='UNCLEAR_SBATCH_ACKNOWLEDGEMENT',
                        returncode=response.returncode, message=reason)
        self.state['controller_status'] = 'SUBMITTED_OR_AWAITING_ACKNOWLEDGEMENT'
        self.save()
        return self.summary()

    def summary(self):
        attempts = self.state['attempts']
        latest = attempts[-1] if attempts else {}
        unresolved = self.unresolved()
        jobs = []
        for attempt in unresolved or ([latest] if latest else []):
            jobs.append(dict(slot=0, job_id=attempt.get('job_id'), status=attempt.get('status'),
                slurm_state=attempt.get('slurm_state'), attempt=attempt.get('sequence'),
                node=attempt.get('node'), queue_reason=attempt.get('queue_reason'), cpus=4,
                gpus=1, memory='64G', partition='hpc_gpu', name=attempt.get('name')))
        return dict(controller_status=self.state.get('controller_status'), marker=self.marker,
            inflight_jobs=len(unresolved), jobs=jobs,
            active_job_ids=sorted({a['job_id'] for a in unresolved if a.get('job_id')}),
            terminal_job_ids=sorted({a['job_id'] for a in attempts if a.get('job_id') and a['status']=='FINISHED'}),
            resources=resources())


def legacy_rows(env, scheduler):
    """Exact original marker only; other papers and rounds never match."""
    marker = env['marker']
    pattern = re.compile(re.escape(marker)+r'-s[012]-a[1-9]\d*$')
    history = read(env['runtime']/'scheduler-state.json', {})
    owned = {a.get('name'): a.get('job_id') for row in history.get('slots', {}).values()
             for a in row.get('attempts', [])}
    def matches(row):
        return pattern.fullmatch(row['name']) is not None and (
            row['comment'] == marker or (row['comment'] in ('', '(null)', 'Unknown', 'None')
                and owned.get(row['name']) == row['job_id']))
    live = [row for row in scheduler.queue if matches(row)]
    terminal = [row['job_id'] for row in scheduler.accounting if matches(row) and row['state'] not in ACTIVE]
    return live, terminal


def cancel_old_pending(env, scheduler, rows):
    """Hold then recheck exact old pending jobs; never cancel a live allocation."""
    history = read(env['runtime']/'scheduler-state.json', {})
    known = {str(a.get('job_id')): a.get('name') for slot in history.get('slots', {}).values()
        for a in slot.get('attempts', []) if a.get('job_id')}
    results = []
    def pending_identity(job, name):
        response = scheduler._run(['scontrol', 'show', 'job', '-o', job], timeout=15)
        if response.returncode:
            return False, 'IDENTITY_QUERY_FAILED'
        def field(key):
            match = re.search(r'(?:^|\s)'+re.escape(key)+r'=(\S+)', response.stdout)
            return match.group(1) if match else None
        owner = field('UserId')
        expected_owner = scheduler.launch.get('user') or scheduler.launch.get('account') or getpass.getuser()
        if (field('JobId') != job or field('JobName') != name or owner is None
                or owner.split('(', 1)[0] != expected_owner):
            return False, 'IDENTITY_CHANGED'
        if field('Comment') not in (None, '', '(null)', env['marker']):
            return False, 'MARKER_CHANGED'
        if field('JobState') != 'PENDING':
            return False, 'ALREADY_ALLOCATED_OR_TERMINAL'
        if field('NodeList') not in ('(null)', 'None', '', None):
            return False, 'NODE_ALREADY_ALLOCATED'
        return True, 'EXACT_PENDING_UNALLOCATED'
    for row in rows:
        job, name = row['job_id'], row['name']
        if row['state'] != 'PENDING' or known.get(job) != name:
            continue
        valid, reason = pending_identity(job, name)
        if not valid:
            results.append(dict(job_id=job, status=reason))
            continue
        response = scheduler._run(['scontrol', 'hold', job], timeout=15)
        if response.returncode:
            results.append(dict(job_id=job, status='HOLD_FAILED_NO_CANCELLATION'))
            continue
        valid, reason = pending_identity(job, name)
        if not valid:
            results.append(dict(job_id=job, status='AFTER_HOLD_'+reason))
            continue
        response = scheduler._run(['scancel', '--state=PENDING', job], timeout=15)
        value = dict(job_id=job, name=name, status='EXACT_HELD_PENDING_CANCEL_REQUESTED' if
            response.returncode == 0 else 'CANCEL_REQUEST_FAILED', running_jobs_cancelled=0,
            updated_utc=utc())
        results.append(value)
        emit('R1_OLD_PENDING_HANDOVER', **value)
    if results:
        atomic(scheduler.work/'old-pending-handover.json', dict(results=results, updated_utc=utc()))
    return results


def slot_locks_free(runtime):
    held = []
    try:
        for slot in range(3):
            handle = (Path(runtime)/('worker-slot-%d.lock'%slot)).open('a')
            held.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        return True
    finally:
        for handle in held:
            handle.close()


def load_environment(args):
    """Load deployed native DAG, with one bounded receipt projection cache."""
    source = Path(args.legacy_dir).resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_r1_single_original_parallel', source/'parallel_runtime.py')
    pr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pr
    spec.loader.exec_module(pr)
    env = pr.load_operations(Path(args.root), Path(args.native_package))
    sys.path.insert(0, str(HERE))
    import receipt_projection
    receipt_projection.install(env['bridge'])
    original = env['bridge']._read
    missing = object()
    cache = {}
    def per_cycle(path, default=None):
        key = str(Path(path))
        if key not in cache:
            cache[key] = original(path, missing)
        return default if cache[key] is missing else cache[key]
    env['bridge']._read = per_cycle
    env['parallel_module'] = pr
    return env


def cycle(args, *, runner=None, clock=time.time, env=None):
    """Observe, drain old work, publish native frontier, submit at most one job."""
    started = time.monotonic()
    emit('R1_SINGLE_GPU_CYCLE_START', management_only=True, scientific_processes_signalled=0)
    env = env or load_environment(args)
    root, runtime, module, bridge = (env[k] for k in ('root', 'runtime', 'original', 'bridge'))
    command = read(args.worker_command_json)
    scheduler = SingleScheduler(root, args.work, env['launch'], command, runner=runner, clock=clock)
    snapshot = scheduler.observe()
    old_live, old_terminal = legacy_rows(env, scheduler)
    pending_changes = cancel_old_pending(env, scheduler, old_live)
    # Use original per-action lock reconciliation, never erase a live child lease.
    module.reconcile_claims(runtime, dict(terminal_job_ids=old_terminal+snapshot['terminal_job_ids']))
    claims = read(runtime/'claims.json', {}) or {}
    if not isinstance(claims, dict):
        raise ValueError('ORIGINAL_CLAIM_RECORD_INVALID')
    own_live_ids = set(snapshot['active_job_ids'])
    old_claims = {key: row for key, row in claims.items() if str(row.get('job_id')) not in own_live_ids}
    must_drain = bool(old_live or old_claims or (not own_live_ids and not slot_locks_free(runtime)))
    if must_drain:
        with lock(runtime/'.queue.lock'):
            atomic(runtime/'queue.json', dict(status='RUNNING', phase='RESOURCE_CHANGE_DRAIN',
                ready_actions=[], retired_slots=[0,1,2], updated_utc=clock(),
                resource_change='1 GPU / 4 CPU / 64G', current_actions_finish_without_signal=True))
        result = dict(status='DRAINING_EXISTING_SCIENCE', phase='RESOURCE_CHANGE_DRAIN',
            scheduler=snapshot, old_live_jobs=old_live, retained_claims=list(old_claims),
            all_original_complete=False, all_complete=False, jobs_submitted=0, scientific_jobs_signalled=0,
            old_pending_handover=pending_changes,
            updated_utc=utc(), elapsed_seconds=round(time.monotonic()-started, 3))
        atomic(args.result, result)
        emit('R1_SINGLE_GPU_CYCLE_FINISHED', **result)
        return result
    if (runtime/'STOP').exists():
        result = dict(status='STOPPED_BY_USER', phase='USER_STOP', scheduler=snapshot,
            all_original_complete=False, all_complete=False, jobs_submitted=0, scientific_jobs_signalled=0, updated_utc=utc())
        atomic(args.result, result)
        emit('R1_SINGLE_GPU_CYCLE_FINISHED', **result)
        return result
    emit('R1_SINGLE_GPU_NATIVE_PLANNING_STARTED', management_only=True)
    frontier = bridge.plan_ready(root)
    routed = route_frontier(frontier)
    queue = dict(status='COMPLETE' if frontier.get('all_complete') else 'RUNNING',
        phase=frontier['phase'], ready_actions=routed['ready_actions'], retired_slots=[],
        updated_utc=clock(), resource_policy=routed['resource_policy'])
    with lock(runtime/'.queue.lock'):
        atomic(runtime/'queue.json', queue)
    before = len(scheduler.state['attempts'])
    snapshot = scheduler.maybe_submit(wanted=bool(routed['ready_actions']) and not frontier.get('all_complete'),
        allowed=bool(args.submit))
    state = read(runtime/'state.json', {}) or {}
    pr = env.get('parallel_module')
    if pr is not None:
        # The per-cycle projected cache reuses planning observations here. A
        # completed old task must not remain a phantom RUNNING progress row.
        pr.repair_progress_state(env, state)
    # Native publisher still owns task identities, counters and scientific flags.
    # Intercept its two writes so no observer sees the legacy 3-GPU resource text.
    original_put = module.put
    def corrected_put(path, value):
        if Path(path).name == 'progress.json' and isinstance(value, dict):
            value['requested_resources'] = resources()
            value['resource_policy'] = routed['resource_policy']
        return atomic(path, value)
    module.put = corrected_put
    try:
        progress = module.publish_progress(root, runtime, routed, snapshot, state)
    finally:
        module.put = original_put
    complete = bool(frontier.get('all_complete')) and not snapshot['inflight_jobs']
    result = dict(status='ALL_ORIGINAL_WORK_COMPLETE' if complete else 'RUNNING',
        repair_id=REPAIR_ID, phase=frontier['phase'], scheduler=snapshot,
        all_original_complete=complete, all_complete=complete, science_complete=bool(frontier.get('all_complete')),
        ready_actions=len(routed['ready_actions']), mainline_complete=frontier.get('mainline_complete', False),
        counts={status: sum(row.get('status')==status for row in progress.get('tasks', []))
            for status in ('COMPLETE','RUNNING','WAITING_DEPENDENCY_OR_SLOT','RETRY_WAIT')},
        jobs_submitted=sum(a.get('job_id') is not None for a in scheduler.state['attempts'][before:]),
        scientific_jobs_signalled=0, original_scientific_files_changed=False,
        updated_utc=utc(), elapsed_seconds=round(time.monotonic()-started, 3))
    atomic(args.result, result)
    emit('R1_SINGLE_GPU_CYCLE_FINISHED', **result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--cycle', action='store_true', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--legacy-dir', type=Path, required=True)
    parser.add_argument('--native-package', type=Path, required=True)
    parser.add_argument('--worker-command-json', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    cycle(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
