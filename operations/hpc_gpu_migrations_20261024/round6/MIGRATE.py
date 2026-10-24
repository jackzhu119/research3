#!/usr/bin/env python3
"""Checkpoint and migrate this existing round to one hpc_gpu allocation."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROUND = 6
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
RUNTIME = importlib.import_module('paper3r%d.runtime' % ROUND)
SUBMIT = importlib.import_module('paper3r%d.submit' % ROUND)


def read(path, default=None):
    value = RUNTIME.read_json(path, default)
    return default if value is None else value


def emit(event, **values):
    print(json.dumps({'event': event, 'round': ROUND, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}, ensure_ascii=False), flush=True)


def plan_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_count(plan):
    return sum(read(Path(plan['output']) / task['key'] / 'result.json', {}).get('status') == 'COMPLETE'
               for task in plan['tasks'])


def target_resources(launch):
    memory = launch.get('memory_total')
    if memory is None and launch.get('memory'):
        match = re.fullmatch(r'([1-9]\d*)([KMGT])', str(launch['memory']).upper())
        if match:
            memory = str(int(match[1]) * int(launch.get('slots', 3))) + match[2]
    return (launch.get('partition') == 'hpc_gpu' and int(launch.get('slots', 0)) == 3
            and int(launch.get('cpus', 0)) == 12 and str(memory).upper() == '128G'
            and launch.get('gres', 'gpu:3') == 'gpu:3')


def verify_existing_plan(path, root, output):
    plan = read(path)
    if not isinstance(plan, dict) or not isinstance(plan.get('tasks'), list):
        raise RuntimeError('Existing round plan is unavailable: ' + str(path))
    if Path(plan['root']).resolve() != root or Path(plan['output']).resolve() != output:
        raise RuntimeError('Existing plan root/output differs from the requested round')
    module = importlib.import_module('paper3r%d.plan' % ROUND)
    expected = module.make_plan(root, output, gpus=3, cpus=12)
    if plan.get('recipe_sha256') != expected.get('recipe_sha256'):
        raise RuntimeError('Numerical recipe differs; existing results were left untouched')
    for task, candidate in zip(plan['tasks'], expected['tasks'], strict=True):
        for key in ('external_manifest', 'basis_output'):
            if (key in task) != (key in candidate):
                raise RuntimeError('An original input locator is missing from the task declaration')
        if task.get('external_manifest'):
            manifest = Path(task['external_manifest'])
            required_sha = plan.get('pan_manifest_sha256') if task.get('external_dataset') == 'Pan2023' else plan.get('external_manifest_sha256')
            if not manifest.is_file() or required_sha and plan_sha(manifest) != required_sha:
                raise RuntimeError('Original small dataset manifest is unavailable or differs: ' + str(manifest))
        if task.get('basis_output'):
            basis = read(Path(task['basis_output']) / 'plan.json', {})
            if basis.get('recipe_sha256') != plan.get('basis_recipe_sha256'):
                raise RuntimeError('Original basis-run recipe differs')
    def numerical(tasks):
        return [{k: v for k, v in task.items() if k not in ('external_manifest', 'basis_output')}
                for task in tasks]
    if numerical(plan['tasks']) != numerical(expected['tasks']):
        raise RuntimeError('Numerical task declarations differ; existing results were left untouched')
    return plan


class Handoff:
    def __init__(self, plan_path, *, runner=None, sleeper=None, clock=None):
        self.path = Path(plan_path).resolve()
        self.plan = read(self.path)
        self.output = Path(self.plan['output']).resolve()
        self.runtime = self.output / 'runtime'
        self.directory = self.runtime / 'hpc_gpu_migration'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.marker = RUNTIME.job_marker(self.plan)
        self.runner = runner or subprocess.run
        self.sleep = sleeper or time.sleep
        self.clock = clock or time.time
        self.original_plan_sha = plan_sha(self.path)
        self.sent = set()

    def progress(self, phase, **values):
        state = {'phase': phase, 'round': ROUND, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'marker': self.marker, 'complete_tasks': complete_count(self.plan), 'total_tasks': len(self.plan['tasks']),
                 'target': {'partition': 'hpc_gpu', 'nodes': 1, 'slurm_jobs': 1, 'gpus': 3,
                            'cpus_total': 36, 'memory_total': '128G'}, **values}
        RUNTIME.atomic_json(self.directory / 'progress.json', state)
        emit(phase, **{k: v for k, v in state.items() if k not in ('phase', 'round', 'updated_utc', 'target')})
        return state

    def call(self, command, timeout=30):
        result = self.runner(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(command[0] + ' returned ' + str(result.returncode) + ': ' + result.stderr[-500:])
        return result

    def queue(self):
        result = self.call(['squeue', '--noheader', '--user', os.environ.get('USER', 'zhuziyuan'),
                            '--format=%i|%k|%T|%j|%P'])
        rows = []
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) < 5 or fields[1].strip() != self.marker:
                continue
            if not re.fullmatch(r'\d+', fields[0].strip()):
                raise RuntimeError('Owned allocation has an unexpected non-regular job ID')
            rows.append({'job_id': fields[0].strip(), 'state': fields[2].strip().split()[0].rstrip('+').upper(),
                         'name': fields[3].strip(), 'partition': fields[4].strip()})
        return rows

    def accounting(self, attempts):
        created = float(self.plan.get('prepared_at', self.clock())) - 86400
        start = time.strftime('%Y-%m-%d', time.gmtime(created))
        result = self.call(['sacct', '--noheader', '--parsable2', '--allocations', '--user',
                            os.environ.get('USER', 'zhuziyuan'), '--starttime', start,
                            '--format=JobID%80,State%40,ExitCode,Comment%64,JobName%100'])
        return RUNTIME.parse_accounting(result.stdout, self.marker, {a.get('name') for a in attempts})

    @contextmanager
    def controller_lock(self):
        with (self.runtime / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def quiesce(self):
        # Both old controller and allocation already watch this cooperative flag.
        # It prevents an old controller from submitting another gpu1 allocation.
        (self.runtime / 'STOP').touch()
        while True:
            try:
                rows = self.queue()
                for row in rows:
                    job = row['job_id']
                    action = (job, 'cancel_pending' if row['state'] == 'PENDING' else 'checkpoint')
                    if action in self.sent:
                        continue
                    if row['state'] == 'PENDING':
                        # Slurm checks state at cancellation time: a job that
                        # starts between observation and request is not killed.
                        self.call(['scancel', '--state=PENDING', job])
                        self.progress('OLD_PENDING_ALLOCATION_CANCEL_REQUESTED', job_id=job)
                    elif row['state'] in ('RUNNING', 'CONFIGURING', 'SUSPENDED', 'RESIZING'):
                        self.call(['scancel', '--batch', '--signal=USR1', job])
                        self.progress('OLD_ALLOCATION_CHECKPOINT_REQUESTED', job_id=job)
                    self.sent.add(action)
                if rows:
                    self.progress('WAITING_FOR_OLD_ALLOCATION_CHECKPOINT', old_allocations=rows)
                else:
                    try:
                        with self.controller_lock():
                            # Recheck under the old controller's lock; a last
                            # submission may have been in flight when STOP arose.
                            if self.queue():
                                continue
                            return self.archive_epoch()
                    except BlockingIOError:
                        self.progress('WAITING_FOR_OLD_CONTROLLER_TO_STOP')
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                self.progress('MIGRATION_OBSERVATION_RETRY', error_type=type(error).__name__, reason=str(error)[:500])
            self.sleep(5)

    def archive_epoch(self):
        if plan_sha(self.path) != self.original_plan_sha:
            raise RuntimeError('Original plan changed during resource handoff')
        old = read(self.runtime / 'state.json', {})
        attempts = {int(a['sequence']): dict(a) for a in old.get('attempts', [])}
        for path in sorted((self.runtime / 'receipts').glob('*.json')):
            row = read(path, {})
            if row.get('marker') == self.marker and 'sequence' in row:
                attempts.setdefault(int(row['sequence']), {}).update(row)
        observed = self.accounting(list(attempts.values()))
        active = [row for row in observed if row['state'] in RUNTIME.ACTIVE_STATES]
        if active:
            self.progress('WAITING_FOR_OLD_ALLOCATION_ACCOUNTING', old_allocations=active)
            raise RuntimeError('Slurm accounting still reports an active old allocation')
        by_job = {a.get('job_id'): a for a in attempts.values() if a.get('job_id')}
        for row in observed:
            if row['job_id'] not in by_job:
                pending = next((a for a in reversed(list(attempts.values()))
                                if not a.get('job_id') and a.get('status') in ('INTENT', 'UNCERTAIN')
                                and a.get('name') == row.get('name')), None)
                if pending is not None:
                    pending['job_id'] = row['job_id']
                    by_job[row['job_id']] = pending
                else:
                    seq = max(attempts, default=0) + 1
                    attempts[seq] = {'sequence': seq, 'marker': self.marker, 'job_id': row['job_id'],
                                     'name': row.get('name', self.marker + '-a' + str(seq)),
                                     'intent_at': self.clock(), 'status': 'ACKNOWLEDGED', 'origin': 'MIGRATION_ACCOUNTING_RECONCILIATION'}
                    by_job[row['job_id']] = attempts[seq]
            by_job[row['job_id']].update(slurm_state=row['state'])
        observed_ids = {row['job_id'] for row in observed}
        unresolved = [a for a in attempts.values()
                      if a.get('status') not in ('FINISHED', 'REJECTED')
                      and (not a.get('job_id') or a['job_id'] not in observed_ids
                           and not read(self.runtime / 'outcomes' / (a['job_id'] + '.json'), {}).get('exit_code') in (0, 2, 75))]
        if unresolved:
            self.progress('WAITING_FOR_OLD_SUBMISSION_RECONCILIATION', unresolved_names=[a.get('name') for a in unresolved])
            raise RuntimeError('An old submission has no confirmed terminal Slurm or batch receipt yet')
        archive = self.directory / ('history-%d-%d' % (int(self.clock() * 1000000), os.getpid()))
        archive.mkdir(parents=True, exist_ok=False)
        for name in ('state.json', 'launch.json', 'controller.pid.json', 'allocation_state.json'):
            source = self.runtime / name
            if source.is_file():
                shutil.copy2(source, archive / name)
        for name in ('receipts', 'outcomes'):
            source = self.runtime / name
            if source.is_dir():
                shutil.copytree(source, archive / name)
        # Keep every historical job ID and monotone sequence. Deleting receipts
        # would let accounting re-adopt old CANCELLED jobs as new failures.
        for row in attempts.values():
            row.update(previous_outcome=row.get('outcome'), status='FINISHED',
                       outcome='RESOURCE_MIGRATION_HANDOFF', migration_finished_at=self.clock())
            RUNTIME.atomic_json(self.runtime / 'receipts' / ('allocation-a%03d.json' % row['sequence']), row)
        state = {'version': 2, 'mode': 'SINGLE_NODE_THREE_GPUS', 'marker': self.marker,
                 'attempts': sorted(attempts.values(), key=lambda a: a['sequence']),
                 'created_at': old.get('created_at', self.clock()), 'submit_rejections': 0,
                 'next_submit_at': 0, 'controller_status': 'RESOURCE_MIGRATION_READY',
                 'migration_history': str(archive)}
        RUNTIME.atomic_json(self.runtime / 'state.json', state)
        allocation = read(self.runtime / 'allocation_state.json')
        if isinstance(allocation, dict):
            for index, operation in allocation.get('tasks', {}).items():
                task = self.plan['tasks'][int(index)]
                if read(self.output / task['key'] / 'result.json', {}).get('status') != 'COMPLETE':
                    operation.update(phase='NEW', status='NEW', migration_retry_requested=True)
            RUNTIME.atomic_json(self.runtime / 'allocation_state.json', allocation)
        for source in (self.runtime / 'STOP', self.output / 'STOP'):
            if source.exists():
                shutil.move(str(source), str(archive / ('runtime_STOP' if source.parent == self.runtime else 'output_STOP')))
        self.progress('CHECKPOINT_HANDOFF_READY', history=str(archive), original_plan_sha256=self.original_plan_sha)
        return archive


def startup_arguments(args, extra):
    result = ['--root', str(args.root), '--output', str(args.output), '--python', str(args.python),
              '--partition', 'hpc_gpu', '--gpus', '3', '--cpus', '12', '--memory-total', '128G']
    return result + extra


def helper_alive(receipt, plan_path):
    try:
        pid = int(receipt['pid'])
        process = Path('/proc') / str(pid)
        if process.stat().st_uid != os.getuid():
            return False
        command = (process / 'cmdline').read_bytes().split(b'\0')
        tokens = [v.decode(errors='replace') for v in command if v]
        return '--handoff' in tokens and '--output' in tokens and str(plan_path.parent) in tokens
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--output')
    parser.add_argument('--python', default='/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python')
    parser.add_argument('--partition', default='hpc_gpu')
    parser.add_argument('--gpus', type=int, default=3)
    parser.add_argument('--cpus', type=int, default=12)
    parser.add_argument('--memory-total', default='128G')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--handoff', action='store_true', help=argparse.SUPPRESS)
    args, extra = parser.parse_known_args(argv)
    if (args.partition, args.gpus, args.cpus, args.memory_total.upper()) != ('hpc_gpu', 3, 12, '128G'):
        parser.error('This resource migration requests hpc_gpu, 3 GPUs, 12 CPUs per slot and 128G total')
    args.root = Path(args.root).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve() if args.output else args.root / ('runs/followup_20261024_r%d' % ROUND)
    args.python = Path(args.python).expanduser().absolute()
    if not args.root.is_dir() or not args.python.is_file():
        parser.error('Existing project root and interpreter are required')
    path = args.output / 'plan.json'
    forwarded = startup_arguments(args, extra)
    if not path.is_file():
        subprocess.run([str(args.python), '-I', '-S', '-B', '-u', str(PACKAGE / 'START.py'), *forwarded], check=True)
    plan = verify_existing_plan(path, args.root, args.output)
    before = plan_sha(path)
    status = {'status': 'MIGRATION_PREPARED', 'round': ROUND, 'tasks': len(plan['tasks']),
              'complete_tasks_reused': complete_count(plan), 'remaining_tasks': len(plan['tasks']) - complete_count(plan),
              'partition': 'hpc_gpu', 'gpu_model_requested_by_group': 'A100-PCIe-40GB', 'nodes': 1,
              'slurm_jobs': 1, 'gpus': 3, 'cpus_per_job': 36, 'host_memory_per_job': '128G',
              'plan': str(path), 'plan_sha256': before, 'scientific_recipe_sha256': plan['recipe_sha256']}
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if args.prepare_only or not args.submit and not args.handoff:
        return 0
    migration = Handoff(path)
    if not args.handoff:
        with (migration.directory / 'startup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt = read(migration.directory / 'startup.json', {})
            if helper_alive(receipt, path):
                emit('EXISTING_MIGRATION_PROCESS_REUSED', pid=receipt['pid'])
                return 0
            command = [str(args.python), '-I', '-S', '-B', '-u', str(PACKAGE / 'MIGRATE.py'), '--handoff', *forwarded]
            with (migration.directory / 'migration.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, close_fds=True)
            receipt = {'pid': process.pid, 'started_at': time.time(), 'plan': str(path), 'log': str(migration.directory / 'migration.log')}
            RUNTIME.atomic_json(migration.directory / 'startup.json', receipt)
            emit('MIGRATION_PROCESS_STARTED', **receipt)
        return 0
    with (migration.directory / 'handoff.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('EXISTING_MIGRATION_PROCESS_REUSED')
            return 0
        launch = read(migration.runtime / 'launch.json', {})
        if not target_resources(launch) or (migration.runtime / 'STOP').exists() or (migration.output / 'STOP').exists():
            migration.progress('CHECKPOINT_HANDOFF_STARTING')
            migration.quiesce()
        # The original START preserves plan bytes and adopts the same marker.
        result = subprocess.run([str(args.python), '-I', '-S', '-B', '-u', str(PACKAGE / 'START.py'), *forwarded, '--submit'])
        if plan_sha(path) != before:
            raise RuntimeError('Original plan changed during resource migration')
        migration.progress('HPC_GPU_CONTROLLER_STARTED' if result.returncode == 0 else 'NEW_CONTROLLER_START_RETRY_REQUIRED', returncode=result.returncode)
        return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        emit('MIGRATION_RETRY_REQUIRED', error_type=type(error).__name__, reason=str(error)[:1000])
        raise SystemExit(1)
