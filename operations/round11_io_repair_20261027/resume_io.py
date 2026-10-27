#!/usr/bin/env python3
"""Operational R11 I/O recovery; the original scientific recipe stays immutable.

The four original allocation slots and their durable Slurm intents are reused.
Only subsequently submitted workers use this entry. Healthy existing jobs are
never signalled or cancelled by this module.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import legacy_controller as legacy
from atomic_io import install_atomic_patch

REPAIR_ID = 'paper3-r11-io-resume-v1'
DEFAULT_PLAN = Path('/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json')


def emit(event, **fields):
    record = json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             **fields), ensure_ascii=False)
    try:
        print(record, flush=True)
    except OSError:
        try:
            os.write(2, (record+'\n').encode())
        except OSError:
            pass


def is_telemetry_path(path, runtime):
    """A deliberately small list: none of these files authorizes scientific reuse."""
    path, runtime = Path(path).absolute(), Path(runtime).absolute()
    if path.parent == runtime:
        return path.name in ('status.json', 'controller-status.json') or bool(
            re.fullmatch(r'slot-[0-3]\.json', path.name))
    return path.parent == runtime / 'numerical-observations' and path.name.endswith('.json')


def install_telemetry_patch(common, runtime):
    """Defer exhausted monitoring writes visibly; retain strict scientific writes."""
    previous = common.atomic
    if getattr(previous, '_r11_io_telemetry_runtime', None) == str(Path(runtime).absolute()):
        return previous._r11_io_telemetry_state
    state = {'deferred_writes': 0, 'last_error': None}
    last_emitted = {}

    def atomic(path, value):
        try:
            return previous(path, value)
        except OSError as error:
            if not is_telemetry_path(path, runtime):
                raise
            state['deferred_writes'] += 1
            state['last_error'] = dict(error_type=type(error).__name__, errno=error.errno,
                                      path=str(Path(path).absolute().relative_to(Path(runtime).absolute())))
            # Report the real loss of freshness. Never claim this write was durable.
            key = str(path); now = time.monotonic()
            if now - last_emitted.get(key, -1e9) >= 30:
                last_emitted[key] = now
                record = dict(event='R11_TELEMETRY_WRITE_DEFERRED', repair_id=REPAIR_ID,
                              scientific_write_skipped=False, **state['last_error'])
                try:
                    os.write(2, (json.dumps(record, ensure_ascii=False) + '\n').encode())
                except OSError:
                    # A failed log sink cannot promote a telemetry-only failure
                    # into a scientific interruption. The counter remains set.
                    pass
            return None

    atomic._r11_io_telemetry_runtime = str(Path(runtime).absolute())
    atomic._r11_io_telemetry_state = state
    common.atomic = atomic
    for name, module in list(sys.modules.items()):
        if name.startswith('paper3r11.') and module is not None and getattr(module, 'atomic', None) is previous:
            module.atomic = atomic
    return state


def load_env(package, plan_path, *, expected_recipe=legacy.DEFAULT_RECIPE):
    env = legacy.load_original(package, plan_path, expected_recipe=expected_recipe)
    env['atomic_patch'] = install_atomic_patch(env['common'])
    env['telemetry'] = install_telemetry_patch(env['common'], env['runtime'])
    return env


def repaired_worker_command(env, python=None):
    return [str(Path(python or env['header']['python']).expanduser().absolute()),
            '-I', '-S', '-B', '-u', str(HERE / 'resume_io.py'), 'worker',
            '--plan', str(env['path']), '--package', str(env['package']),
            '--slot', '{slot}', '--device', 'cuda:0']


def original_launch(env):
    launch = legacy.bounded_read(env['runtime'] / 'launch.json', {})
    if (launch.get('plan_sha256') != env['header']['plan_sha256']
            or launch.get('campaign_identity') != env['header']['recipe_sha256']
            or Path(launch.get('runtime', '')).resolve() != env['runtime'].resolve()):
        raise ValueError('ORIGINAL_LAUNCH_HISTORY_DIFFERS')
    return launch


def monitor_failure_only(error, runtime, key):
    """Recognize the exact temporary-file-to-monitor destination error shape."""
    if not isinstance(error, dict) or error.get('error_type') != 'FileNotFoundError':
        return False
    match = re.search(r"'([^']+)'\s*->\s*'([^']+)'", str(error.get('reason', '')))
    if not match:
        return False
    source, target = (Path(value) for value in match.groups())
    runtime = Path(runtime)
    allowed = {runtime / 'status.json', runtime / 'numerical-observations' / (key + '.json')}
    return (target in allowed and source.parent == target.parent
            and source.name.startswith('.' + target.name + '.') and source.name != target.name)


def recover_monitor_failures(env):
    """Seal exact completed results or retry only known monitor-write failures.

    The task lease and independent scientific lock must both be free. This also
    protects a surviving numerical child after its former supervisor exited.
    Full declared artifact hashes are checked before a COMPLETE is recovered.
    """
    rt, common, runtime = env['runtime_module'], env['common'], env['runtime']
    _, plan = common.load_plan(env['path']); tasks, _ = rt.plan_index(plan)
    state = rt._state(plan); recovered = []; skipped_live = []; deferred = []
    candidates = [key for key, row in state['tasks'].items()
                  if key in tasks and row.get('phase') in rt.BLOCKED
                  and monitor_failure_only(row.get('current_error'), runtime, key)]
    for key in candidates:
        task = tasks[key]; folder = common.task_folder(plan, task)
        with common.locked(runtime / 'tasks.lock'):
            current = rt._state(plan)['tasks'].get(key, {})
            if current.get('phase') not in rt.BLOCKED or not monitor_failure_only(
                    current.get('current_error'), runtime, key):
                continue
            (runtime / 'task-locks').mkdir(parents=True, exist_ok=True)
            folder.mkdir(parents=True, exist_ok=True)
            lease = (runtime / 'task-locks' / (key + '.lock')).open('a')
            science = (folder / '.scientific.lock').open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(science, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close(); science.close(); skipped_live.append(key); continue
        try:
            # A new scientific result may have finished after the supervisor's
            # telemetry exception. No training/prediction is executed here.
            valid = common.verify_result(plan, task, sealed=False, verify_artifacts=True)
            with common.locked(runtime / 'tasks.lock'):
                state = rt._state(plan); row = state['tasks'].get(key, {})
                if row.get('phase') not in rt.BLOCKED or not monitor_failure_only(
                        row.get('current_error'), runtime, key):
                    continue
                history = list(row.get('io_recoveries', []))
                event = dict(repair_id=REPAIR_ID, prior_phase=row['phase'],
                             prior_error=row.get('current_error'), utc=common.utc(),
                             full_artifacts_verified=valid, scientific_computation_executed=False)
                if valid:
                    if not common.verify_result(plan, task, sealed=True, verify_artifacts=False):
                        common.seal_result(plan, task)
                    row.update(phase='COMPLETE', task_identity=common.task_identity(plan, task),
                               result_sha256=common.sha(folder / 'result.json'),
                               completion_sha256=common.sha(folder / 'completion.json'),
                               current_error=None, consecutive_failures=0, retry_at=0,
                               finished_at=time.time(), exit_code=0,
                               last_numerical_progress=common.read(folder / 'progress.json', {}))
                    event['action'] = 'EXACT_COMPLETED_RESULT_SEALED_WITHOUT_RETRAINING'
                    recovered.append(key)
                else:
                    row.update(phase='PAUSED', current_error=None, retry_at=0,
                               consecutive_failures=0)
                    event['action'] = 'MONITOR_FAILURE_RETRY_FROM_EXISTING_CHECKPOINTS'
                    deferred.append(key)
                history.append(event); row['io_recoveries'] = history
                state['tasks'][key] = row; common.atomic(runtime / 'tasks.json', state)
        finally:
            science.close(); lease.close()
    return dict(recovered_complete_tasks=recovered, monitor_failures_requeued=deferred,
                live_tasks_left_untouched=skipped_live, unrelated_blocked_tasks_reset=False)


def compact_task_status(env):
    """Read durable state once without writing observations for thousands of tasks."""
    rt, common = env['runtime_module'], env['common']
    _, plan = common.load_plan(env['path']); state = rt._state(plan)
    completed = rt.completed_task_rows(plan, state); blocked = rt.blocked_task_keys(plan, state, set(completed))
    total = len(plan['tasks'])
    return dict(complete_tasks=len(completed), total_tasks=total, blocked_tasks=len(blocked),
                runnable_remaining_tasks=total-len(completed)-len(blocked),
                updated_utc=common.utc(), completion_counts_from_durable_verified_cache=True,
                telemetry=dict(env['telemetry']))


def controller_loop(plan_path, work, *, package=None, submit=False, runner=None,
                    stop=None, once=False, expected_recipe=legacy.DEFAULT_RECIPE,
                    launch_token=None, sleep=None, clock=time.time):
    env = load_env(package, plan_path, expected_recipe=expected_recipe)
    history = legacy.validate_operational_history(env)
    rt, common, runtime = env['runtime_module'], env['common'], env['runtime']
    work = Path(work); work.mkdir(parents=True, exist_ok=True); stop = stop or threading.Event()
    launch = original_launch(env)
    repaired = dict(launch, worker_command=repaired_worker_command(env, launch['python']))
    with common.locked(runtime / 'controller.lock', nonblocking=True):
        common.atomic(work / 'replacement-command.json', dict(
            repair_id=REPAIR_ID, original_launch_sha256=common.sha(runtime / 'launch.json'),
            original_worker_command=launch['worker_command'], worker_command=repaired['worker_command'],
            resource_configuration_changed=False, scientific_recipe_changed=False,
            healthy_scientific_jobs_signalled=0, updated_utc=common.utc()))
        common.atomic(runtime / 'controller-process.json', dict(kind='controller', pid=os.getpid(),
            started_utc=common.utc(), operational_repair=True, repair_id=REPAIR_ID,
            source_files_changed=False, submit_reaffirmed=bool(submit)))
        scheduler = rt.FourJobScheduler(runtime, repaired, runner=runner, emit=emit, clock=clock)
        # Reconcile persisted submissions and current queue first. No submission
        # is possible in this initial observation, including on repeated apply.
        first = None
        while not stop.is_set():
            try:
                observed = scheduler.step(keep_workers=False)
            except OSError as error:
                observed = dict(controller_status='OBSERVATION_RETRY',
                                observation_error=type(error).__name__)
            if (observed.get('controller_status') not in
                    ('EXISTING_SCHEDULER_REUSED', 'OBSERVATION_RETRY')
                    and not observed.get('observation_error')):
                first = observed
                break
            emit('R11_FIRST_QUEUE_OBSERVATION_PENDING',
                 controller_status=observed.get('controller_status'),
                 observation_error=observed.get('observation_error'), submission_performed=False)
            if once:
                return dict(status='FIRST_QUEUE_OBSERVATION_PENDING',
                            first_slurm_observation_completed=False,
                            queue_observation_successful=False, repair_id=REPAIR_ID,
                            summary=observed)
            if sleep is not None:
                sleep(2.)
            else:
                stop.wait(2.)
        if first is None:
            return dict(status='STOPPED_BEFORE_FIRST_QUEUE_OBSERVATION', repair_id=REPAIR_ID)
        common.atomic(work / 'first-observation.json', dict(summary=first, history=history,
            first_cycle_submit_allowed=False, updated_utc=common.utc()))
        ready = dict(status='READY', pid=os.getpid(),
            recipe_sha256=env['header']['recipe_sha256'], repair_id=REPAIR_ID,
            launch_token=launch_token, memory_patch_loaded=True, atomic_patch_loaded=True,
            first_slurm_observation_completed=True, queue_observation_successful=True,
            first_slurm_observation_performed=True, training_progress_claimed=False,
            updated_utc=common.utc())
        common.atomic(work / 'controller-ready.json', ready)
        common.atomic(runtime / 'controller-ready.json', ready)
        common.atomic(runtime / 'controller-error.json', {})
        previous_stamp = rt.file_signature(env['path'])
        while not stop.is_set():
            try:
                stamp = rt.file_signature(env['path'])
                if stamp != previous_stamp:
                    if common.sha(env['path']) != launch['plan_sha256']:
                        raise ValueError('AUTHORIZED_PLAN_CHANGED')
                    previous_stamp = stamp
                recovery = recover_monitor_failures(env)
                snapshot = compact_task_status(env)
                summary = first if first is not None else scheduler.step(
                    keep_workers=bool(snapshot['runnable_remaining_tasks']) and bool(submit))
                first = None
                value = dict(summary, task_status=snapshot, repair_id=REPAIR_ID,
                    recovery=recovery, operational_repair=True, updated_utc=common.utc(),
                    automatic_replacement_authorized=bool(submit), scientific_sources_changed=False)
                common.atomic(runtime / 'controller-status.json', value)
                common.atomic(work / 'controller-status.json', value)
                if once:
                    return value
                if not snapshot['runnable_remaining_tasks'] and summary.get('controller_status') == 'FINISHED':
                    return value
                if summary.get('controller_status', '').startswith('BLOCKED_') and not summary.get('inflight_jobs'):
                    return value
            except OSError as error:
                # Exact intents remain on disk. On retry the original scheduler
                # reconciles queue/accounting before any further submission.
                emit('R11_CONTROLLER_IO_RETRY', error_type=type(error).__name__, errno=error.errno,
                     durable_task_state_reset=False, scientific_workers_signalled=0)
                if once:
                    raise
            delay = min(30., max(1., float(launch.get('poll_seconds', 30))))
            if sleep is not None:
                sleep(delay)
            else:
                stop.wait(delay)
    return {'status': 'STOPPED', 'repair_id': REPAIR_ID}


def science_popen(env, original=None):
    """Change only the isolated entrypoint, retaining every scientific argument."""
    original = original or subprocess.Popen
    entry = str(env['package'] / 'paper3r11/platform/entry.py')
    def spawn(command, *args, **kwargs):
        command = list(command)
        if entry in command:
            index = command.index(entry)
            if index+1 < len(command) and command[index+1] in ('science', 'science-daemon'):
                command[index] = str(HERE / 'resume_io.py')
                command += ['--package', str(env['package'])]
        return original(command, *args, **kwargs)
    return spawn


def worker_loop(plan_path, slot, *, package=None, device='cuda:0', popen=None,
                sleep=time.sleep, stop=None, once=False, expected_recipe=legacy.DEFAULT_RECIPE):
    env = load_env(package, plan_path, expected_recipe=expected_recipe)
    emit('R11_WORKER_IO_REPAIR_ACTIVE', repair_id=REPAIR_ID, slot=slot,
         job_id=os.environ.get('SLURM_JOB_ID'), original_source_files_changed=False,
         scientific_recipe_sha256=env['header']['recipe_sha256'])
    # Every newly started worker can safely finish an exact old unsealed result.
    recover_monitor_failures(env)
    return env['runtime_module'].worker_loop(env['path'], slot, device=device,
        popen=science_popen(env, popen), sleep=sleep, stop=stop, once=once)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('mode', choices=('controller', 'worker', 'science', 'science-daemon'))
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--package', type=Path)
    parser.add_argument('--work', type=Path); parser.add_argument('--launch-token')
    parser.add_argument('--submit', action='store_true'); parser.add_argument('--slot', type=int, choices=range(4))
    parser.add_argument('--device', default='cuda:0'); parser.add_argument('--response-dir')
    parser.add_argument('--task-key')
    parsed = parser.parse_args(args)
    if parsed.mode in ('science', 'science-daemon'):
        env = load_env(parsed.package, parsed.plan)
        # Original entry restores existing venv packages and dispatches original
        # scientific modules. No algorithm, seed, loss or identity is altered.
        # Preserve scientific flags and their ordering byte-for-byte. Only our
        # operational package-location hint is removed before original dispatch.
        forwarded = list(args)
        if '--package' in forwarded:
            index = forwarded.index('--package'); del forwarded[index:index+2]
        return importlib.import_module('paper3r11.platform.entry').main(forwarded)
    if parsed.mode == 'worker':
        if parsed.slot is None:
            parser.error('worker requires --slot')
        return worker_loop(parsed.plan, parsed.slot, package=parsed.package, device=parsed.device)
    if parsed.work is None:
        parser.error('controller requires --work')
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        result = controller_loop(parsed.plan, parsed.work, package=parsed.package,
            submit=parsed.submit, stop=stop, launch_token=parsed.launch_token)
    except Exception as error:
        emit('R11_IO_REPAIR_CONTROLLER_FAILED', error_type=type(error).__name__)
        try:
            env = load_env(parsed.package, parsed.plan)
            env['common'].atomic(env['runtime'] / 'controller-error.json', dict(
                status='IO_REPAIR_CONTROLLER_FAILED', error_type=type(error).__name__,
                reason=str(error)[:500], updated_utc=env['common'].utc()))
        except Exception:
            pass
        raise
    emit('R11_IO_REPAIR_CONTROLLER_RETURNED', **result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
