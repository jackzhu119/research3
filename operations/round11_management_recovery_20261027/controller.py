"""Recover queue supervision without touching any existing scientific worker."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import threading
import time

from environment import DEFAULT_RECIPE, HERE, REPAIR_ID, load_env, validate_history, load_launch
import legacy_controller as legacy
import resume_io
from state_io import StateUnavailable, StateIdentityError


def emit(event, **fields):
    resume_io.emit(event, **fields)


def _observe_ok(summary):
    return (summary.get('controller_status') not in
            {'EXISTING_SCHEDULER_REUSED', 'OBSERVATION_RETRY'}
            and not summary.get('observation_error'))


def _publish(env, path, value):
    """Management telemetry has no authority to interrupt scientific work."""
    try:
        env['common'].atomic(path, value)
        return True
    except Exception as error:
        emit('R11_MANAGEMENT_TELEMETRY_DEFERRED', error_type=type(error).__name__,
             scientific_workers_signalled=0)
        return False


def repaired_worker_command(env, python=None):
    return [str(Path(python or env['header']['python']).expanduser().absolute()),
            '-I', '-S', '-B', '-u', str(HERE/'entry.py'), 'worker',
            '--plan', str(env['path']), '--package', str(env['package']),
            '--slot', '{slot}', '--device', 'cuda:0']


def controller_loop(plan_path, work, *, package=None, submit=False, runner=None,
                    stop=None, once=False, expected_recipe=DEFAULT_RECIPE,
                    launch_token=None, sleep=None, clock=time.time):
    env = load_env(package, plan_path, expected_recipe=expected_recipe)
    rt, common, runtime = env['runtime_module'], env['common'], env['runtime']
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    # Required startup observations use the same retry/no-reset rule as polls.
    # Keep an unavailable history visible and do not create a scheduler or READY.
    while not stop.is_set():
        try:
            history = validate_history(env)
            launch = load_launch(env)
            break
        except StateUnavailable as error:
            failure = dict(status='OBSERVATION_RETRY', repair_id=REPAIR_ID,
                startup_stage='RETAINED_AUTHORIZATION_AND_LAUNCH', error_type=type(error).__name__,
                unreadable_state_file=Path(error.path).name, updated_utc=common.utc(),
                submission_performed=False, durable_task_state_reset=False,
                scientific_workers_signalled=0, first_slurm_observation_completed=False)
            _publish(env, work/'controller-status.json', failure)
            _publish(env, runtime/'controller-error.json', failure)
            if once:
                return failure
            if sleep is not None:
                sleep(2.)
            else:
                stop.wait(2.)
    else:
        return dict(status='STOPPED_BEFORE_HISTORY_OBSERVATION',scientific_workers_signalled=0)
    repaired = dict(launch, worker_command=repaired_worker_command(env, launch['python']))
    plan = common.load_plan(env['path'])[1]
    first = True
    last_good = None
    started = common.utc()

    def wait(seconds):
        if sleep is not None:
            sleep(seconds)
        else:
            stop.wait(seconds)

    with common.locked(runtime/'controller.lock', nonblocking=True):
        previous_error = legacy.bounded_read(runtime/'controller-error.json', {})
        if previous_error and not (work/'original-controller-error.json').exists():
            _publish(env, work/'original-controller-error.json', previous_error)
        common.atomic(work/'replacement-command.json', dict(
            repair_id=REPAIR_ID, original_launch_sha256=common.sha(runtime/'launch.json'),
            original_worker_command=launch['worker_command'], worker_command=repaired['worker_command'],
            resource_configuration_changed=False, scientific_recipe_changed=False,
            healthy_scientific_jobs_signalled=0, updated_utc=common.utc()))
        common.atomic(runtime/'controller-process.json', dict(
            kind='controller', pid=os.getpid(), launch_token=launch_token,
            recipe_sha256=plan['recipe_sha256'], started_utc=started,
            operational_repair=True, repair_id=REPAIR_ID, source_files_changed=False,
            submit_reaffirmed=bool(submit)))
        scheduler = rt.FourJobScheduler(runtime, repaired, runner=runner, emit=emit, clock=clock)
        plan_stamp = rt.file_signature(env['path'])
        while not stop.is_set():
            try:
                current = rt.file_signature(env['path'])
                if current != plan_stamp:
                    if common.sha(env['path']) != launch['plan_sha256']:
                        raise StateIdentityError('Original authorized R11 plan changed')
                    plan_stamp = current
                # A valid task observation is required before any submission.
                # In particular, missing/malformed tasks.json never becomes {}.
                state = rt._state(plan)
                complete = rt.completed_task_rows(plan, state)
                blocked = rt.blocked_task_keys(plan, state, set(complete))
                snapshot = dict(complete_tasks=len(complete), total_tasks=len(plan['tasks']),
                    blocked_tasks=len(blocked),
                    runnable_remaining_tasks=len(plan['tasks'])-len(complete)-len(blocked),
                    updated_utc=common.utc(), observation_successful=True,
                    completion_counts_from_durable_verified_cache=True,
                    complete_state_history_preserved=True)
                summary = scheduler.step(keep_workers=False if first else
                    bool(submit and snapshot['runnable_remaining_tasks']))
                if not _observe_ok(summary):
                    raise StateUnavailable(runtime/'scheduler-state.json',
                                           OSError('Fresh queue observation unavailable'), 1)
                value = dict(summary, task_status=snapshot, repair_id=REPAIR_ID,
                    operational_repair=True, updated_utc=common.utc(),
                    queue_observation_successful=True, task_state_observation_successful=True,
                    first_cycle_submit_allowed=False if first else bool(submit),
                    original_operational_history=history,
                    automatic_replacement_authorized=bool(submit),
                    scientific_sources_changed=False, scientific_workers_signalled=0)
                if first:
                    common.atomic(work/'first-observation.json', value)
                    ready = dict(status='READY', pid=os.getpid(), launch_token=launch_token,
                        recipe_sha256=plan['recipe_sha256'], repair_id=REPAIR_ID,
                        first_slurm_observation_completed=True,
                        first_slurm_observation_performed=True, queue_observation_successful=True,
                        task_state_observation_successful=True, memory_patch_loaded=True,
                        atomic_patch_loaded=True, strict_state_patch_loaded=True,
                        training_progress_claimed=False, updated_utc=common.utc())
                    common.atomic(work/'controller-ready.json', ready)
                    common.atomic(runtime/'controller-ready.json', ready)
                    first = False
                # Clear an old error only after a genuinely successful cycle.
                _publish(env, runtime/'controller-error.json', {})
                _publish(env, runtime/'controller-status.json', value)
                _publish(env, work/'controller-status.json', value)
                _publish(env, work/'controller-heartbeat.json', dict(
                    status='HEALTHY', pid=os.getpid(), launch_token=launch_token,
                    updated_utc=common.utc(), last_good_observation_utc=snapshot['updated_utc'],
                    training_progress_claimed=False))
                last_good = snapshot
                if once:
                    return value
                if not snapshot['runnable_remaining_tasks'] and summary.get('controller_status') == 'FINISHED':
                    return value
            except (StateUnavailable, OSError, MemoryError) as error:
                # Keep the same controller alive. Queue/task state cannot be
                # inferred from stale data, and no worker is signalled here.
                failure = dict(status='OBSERVATION_RETRY', repair_id=REPAIR_ID,
                    error_type=type(error).__name__, reason=str(error)[:300],
                    updated_utc=common.utc(), last_good_task_observation=last_good,
                    first_slurm_observation_completed=not first,
                    task_state_observation_successful=False,
                    queue_observation_successful=False, submission_performed=False,
                    durable_task_state_reset=False, scientific_workers_signalled=0)
                if isinstance(error, StateUnavailable):
                    failure.update(unreadable_state_file=Path(error.path).name,
                                   observation_error_type=error.cause,
                                   bounded_read_attempts=error.attempts)
                _publish(env, runtime/'controller-error.json', failure)
                _publish(env, work/'controller-status.json', failure)
                _publish(env, work/'controller-heartbeat.json', dict(
                    failure, pid=os.getpid(), launch_token=launch_token,
                    training_progress_claimed=False))
                emit('R11_MANAGEMENT_OBSERVATION_RETRY', error_type=type(error).__name__,
                     submission_performed=False, scientific_workers_signalled=0)
                if once:
                    return failure
            except StateIdentityError as error:
                failure = dict(status='BLOCKED_IDENTITY_MISMATCH', repair_id=REPAIR_ID,
                    error_type=type(error).__name__, reason=str(error)[:300],
                    updated_utc=common.utc(), submission_performed=False,
                    scientific_workers_signalled=0, durable_task_state_reset=False)
                _publish(env, runtime/'controller-error.json', failure)
                _publish(env, work/'controller-status.json', failure)
                return failure
            wait(2. if first else min(30., max(1., float(launch.get('poll_seconds', 30.)))))
    return dict(status='STOPPED', repair_id=REPAIR_ID, scientific_workers_signalled=0)


def _science_popen(env, original=None):
    original = original or subprocess.Popen
    old_entry = str(env['package']/'paper3r11/platform/entry.py')
    def spawn(command, *args, **kwargs):
        command = list(command)
        if old_entry in command:
            index = command.index(old_entry)
            if index+1 < len(command) and command[index+1] in ('science', 'science-daemon'):
                command[index] = str(HERE/'entry.py')
                command += ['--package', str(env['package'])]
        return original(command, *args, **kwargs)
    return spawn


def worker_loop(plan_path, slot, *, package=None, device='cuda:0', stop=None,
                once=False, popen=None, sleep=time.sleep, expected_recipe=DEFAULT_RECIPE):
    env = load_env(package, plan_path, expected_recipe=expected_recipe)
    rt, common, runtime = env['runtime_module'], env['common'], env['runtime']
    stop = stop or threading.Event()

    class ExternalOperationalPause(BaseException):
        pass

    # Retry at operation boundaries, not inside _state: claim/finish hold
    # tasks.lock, so their exception must first unwind and release that lock.
    # A healthy writer/operator can then repair visibility while this same
    # worker retains its persistent, currently idle scientific child.
    originals = {name:getattr(rt,name) for name in ('status','claim_task','finish_task')}
    wrappers = {}
    def retry_operation(name, function):
        def run(*args, **kwargs):
            while True:
                try:
                    return function(*args, **kwargs)
                except StateUnavailable as error:
                    if stop.is_set():
                        raise ExternalOperationalPause() from None
                    _publish(env, runtime/('slot-%d.json' % slot), dict(
                        status='WAITING_FOR_OPERATIONAL_STATE', operation=name,
                        error_type=type(error).__name__, unreadable_state_file=Path(error.path).name,
                        slot=slot, job_id=os.environ.get('SLURM_JOB_ID','CPU_TEST'),
                        updated_utc=common.utc(), scientific_workers_signalled=0,
                        task_failure_count_incremented=False))
                    emit('R11_WORKER_OPERATIONAL_STATE_RETRY', slot=slot, operation=name,
                         scientific_workers_signalled=0, task_failure_count_incremented=False)
                    if sleep is time.sleep:
                        stop.wait(1.)
                    else:
                        sleep(1.)
                    if stop.is_set():
                        raise ExternalOperationalPause() from None
        return run
    try:
        for name,function in originals.items():
            wrappers[name]=retry_operation(name,function)
            setattr(rt,name,wrappers[name])
        return rt.worker_loop(env['path'], slot, device=device,
            popen=_science_popen(env, popen), stop=stop, once=once, sleep=sleep)
    except ExternalOperationalPause:
        _publish(env, runtime/('slot-%d.json' % slot), dict(status='PAUSED',slot=slot,
            job_id=os.environ.get('SLURM_JOB_ID','CPU_TEST'),exit_code=75,
            updated_utc=common.utc(), task_failure_count_incremented=False))
        job = os.environ.get('SLURM_JOB_ID','')
        if job.isdigit():
            rt.write_worker_outcome(runtime,slot,job,75,round=11,
                                    operational_state_pause=True)
        return 75
    finally:
        for name,function in originals.items():
            if getattr(rt,name) is wrappers.get(name):
                setattr(rt,name,function)


def science_main(argv):
    args = list(argv)
    def argument(name):
        index = args.index(name)
        return args[index+1]
    package, path = argument('--package'), argument('--plan')
    env = load_env(package, path)
    index = args.index('--package'); del args[index:index+2]
    return importlib.import_module('paper3r11.platform.entry').main(args)
