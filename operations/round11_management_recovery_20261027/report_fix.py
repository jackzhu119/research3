"""Independent, observable R11 CPU reporting and compact monitoring loops.

No scientific task state, result, Slurm allocation or scientific process is
changed here. Reporting is an owned CPU child; network and telemetry failures
remain in their own management role and preserve the last successful report.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import importlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPAIR_ID = 'paper3-r11-observable-report-monitor-v1'
HERE = Path(__file__).resolve().parent


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _write(env, path, value):
    try:
        env['common'].atomic(Path(path), value)
        return True
    except Exception:
        return False


def _read(env, path, default=None):
    try:
        value = env['common'].read(Path(path), default)
        return value if value is not None else default
    except Exception:
        return default


def _uploads(env):
    return env.get('uploads_module') or importlib.import_module('paper3r11.platform.uploads')


def _plan(env):
    return env['common'].load_plan(env['path'], validate_science_stage=False)[1]


@contextmanager
def _role_lock(env, name):
    path = Path(env['runtime'])/(name+'-upload.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _ready(env, work, role, token):
    value = dict(status='READY', role=role, pid=os.getpid(), launch_token=token,
                 recipe_sha256=env['header']['recipe_sha256'], lock_acquired=True,
                 repair_id=REPAIR_ID, updated_utc=_utc(),
                 scientific_progress_claimed=False, training_controlled=False)
    # A readiness receipt is operational evidence only. The supervisor will
    # retain STARTUP_PENDING if its destination is temporarily unavailable.
    _write(env, Path(work)/(role+'-ready.json'), value)
    return value


def _error(error):
    # Do not copy exception text, request URLs, headers or credentials.
    return dict(error_type=type(error).__name__, http_status=getattr(error, 'status', None))


def _heartbeat(env, work, role, status, stop, interval):
    while not stop.wait(interval):
        value = dict(status, heartbeat_utc=_utc(), repair_id=REPAIR_ID,
                     pid=os.getpid(), training_controlled=False)
        _write(env, Path(work)/(role+'-heartbeat.json'), value)
        target = (Path(env['runtime'])/'reporter-status.json' if role == 'report'
                  else Path(env['runtime'])/'upload-monitor'/'heartbeat.json')
        _write(env, target, value)


def _start_heartbeat(env, work, role, status, interval):
    done = threading.Event()
    thread = threading.Thread(target=_heartbeat,
        args=(env, work, role, status, done, max(.01, interval)), daemon=True)
    thread.start()
    return done, thread


def _report_child(env, plan, work, stop, *, runner=None, popen=None,
                  timeout=1800., terminate_wait=10.):
    """Bound only a newly owned CPU report process, never a Slurm worker.

    Production uses Popen so the management role can react to TERM promptly.
    The injectable runner is a software-test seam and retains the same exact
    CPU-only command/environment. No process-group or discovered PID is killed.
    """
    command = [str(plan.get('python') or env['header'].get('python') or sys.executable),
               '-I', '-S', '-B', '-u', str(HERE/'report_child.py'),
               '--plan', str(env['path']), '--package', str(env['package']),
               '--expected-recipe', str(env['header']['recipe_sha256'])]
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    for key in ('PYTHONHOME', 'PYTHONPATH', 'BASH_ENV', 'ENV'):
        child_env.pop(key, None)
    with (Path(work)/'report-child.log').open('ab', buffering=0) as log:
        if runner is not None:
            result = runner(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            env=child_env, timeout=timeout, check=False)
            if result.returncode:
                raise RuntimeError('OwnedCPUReportFailed')
            return
        child = (popen or subprocess.Popen)(command, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, env=child_env, start_new_session=True,
                    close_fds=True)
        deadline = time.monotonic()+timeout
        while child.poll() is None:
            if stop.wait(.1) or time.monotonic() >= deadline:
                # This object was created immediately above with CUDA hidden.
                # It is never a scientific PID found through task metadata.
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=terminate_wait)
                except subprocess.TimeoutExpired:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=terminate_wait)
                raise InterruptedError('OwnedCPUReportStopped')
        if child.returncode:
            raise RuntimeError('OwnedCPUReportFailed')


def _verified_report_coverage(env, plan, observed_keys):
    """Verify only the compact report/coverage, never prediction archives."""
    common = env['common']; folder = Path(plan['output'])/'reports'
    receipt = common.read(folder/'report-status.json', {})
    count = receipt.get('sealed_complete_tasks')
    expected = {task['key'] for task in plan['tasks']}
    if (receipt.get('recipe_sha256') != plan['recipe_sha256']
            or type(count) is not int or not len(observed_keys) <= count <= len(expected)
            or receipt.get('report_sha256') != common.sha(folder/'report.json')):
        raise ValueError('ReportCoverageNotSealed')
    report = common.read(folder/'report.json', {}, max_bytes=256*1024*1024)
    coverage = report.get('coverage', {})
    keys = coverage.get('complete_task_keys')
    if (report.get('recipe_sha256') != plan['recipe_sha256']
            or not isinstance(keys, list) or any(not isinstance(key, str) for key in keys)
            or len(keys) != count or len(set(keys)) != count
            or not set(observed_keys) <= set(keys) <= expected
            or coverage.get('sealed_complete_tasks') != count
            or receipt.get('complete_task_keys_sha256') != common.digest(sorted(keys))
            or coverage.get('complete_task_keys_sha256') != receipt['complete_task_keys_sha256']):
        raise ValueError('ReportCoverageNotSealed')
    return count, keys


def run_report(env, work, *, stop=None, launch_token=None, once=False,
               runner=None, popen=None, interval=60., heartbeat_interval=5.,
               report_timeout=1800., **_):
    """Retry the complete observation/report cycle, including state reads."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    runtime = Path(env['runtime']); native_work = runtime/'reporter'
    native_work.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    status = dict(status='STARTING', role='report', repair_id=REPAIR_ID,
                  recipe_sha256=env['header']['recipe_sha256'], training_controlled=False)
    with _role_lock(env, 'report') as acquired:
        if not acquired:
            return dict(status='EXISTING_REPORTER_REUSED', training_controlled=False)
        _ready(env, work, 'report', launch_token)
        _write(env, runtime/'report-upload-process.json',
               dict(kind='report', pid=os.getpid(), repair_id=REPAIR_ID,
                    launch_token=launch_token, started_utc=_utc()))
        done, thread = _start_heartbeat(env, work, 'report', status, heartbeat_interval)
        try:
            while not stop.is_set():
                try:
                    plan = _plan(env); u = _uploads(env)
                    signatures = {key: row['completion_sha256']
                                  for key, row in env['runtime_module'].completed_task_rows(plan).items()}
                    expected = [task['key'] for task in plan['tasks']]
                    signature = env['common'].digest(signatures)
                    prior = _read(env, native_work/'last-generated.json', {})
                    last = prior.get('signature') if prior.get('recipe_sha256') == plan['recipe_sha256'] else None
                    previous = _read(env, work/'report-last-success.json', {})
                    status.update(status='WAITING_FOR_NEW_COMPLETION', role='report',
                        recipe_sha256=plan['recipe_sha256'], observed_complete_tasks=len(signatures),
                        complete_tasks=previous.get('report_complete_tasks', prior.get('complete_tasks')),
                        report_complete_tasks=previous.get('report_complete_tasks', prior.get('complete_tasks')),
                        total_tasks=len(expected), observation_utc=_utc(),
                        last_success_utc=previous.get('last_success_utc'),
                        training_controlled=False, repair_id=REPAIR_ID,
                        current_error=None, updated_utc=_utc())
                    complete = (len(signatures) == len(expected)
                                and u.report_completion(plan['output'], plan['recipe_sha256'], expected))
                    if complete:
                        status.update(status='COMPLETE', complete_tasks=len(signatures),
                                      report_complete_tasks=len(signatures))
                    elif signature != last and signatures:
                        status.update(status='RUNNING_REPORT', report_started_utc=_utc(),
                                      complete_tasks_at_report_start=len(signatures))
                        _write(env, runtime/'reporter-status.json', dict(status))
                        _report_child(env, plan, work, stop, runner=runner, popen=popen,
                                      timeout=report_timeout)
                        # The child may observe a later completion frontier.
                        # Store its actual sealed coverage, not a guessed count.
                        report_count, reported_keys = _verified_report_coverage(env, plan, signatures)
                        env['common'].atomic(native_work/'last-generated.json',
                            dict(recipe_sha256=plan['recipe_sha256'], signature=signature,
                                 complete_tasks=report_count, updated_utc=_utc()))
                        success = dict(recipe_sha256=plan['recipe_sha256'],
                                       report_complete_tasks=report_count,
                                       last_success_utc=_utc(), signature=signature)
                        env['common'].atomic(work/'report-last-success.json', success)
                        done_report = (report_count == len(expected)
                            and u.report_completion(plan['output'], plan['recipe_sha256'], expected))
                        status.update(status='COMPLETE' if done_report else 'PARTIAL_REPORT_GENERATED',
                                      complete_tasks=report_count, report_complete_tasks=report_count,
                                      last_success_utc=success['last_success_utc'])
                except Exception as error:
                    status.update(status='STOPPING' if stop.is_set() else 'RETRYING',
                                  current_error=_error(error), training_controlled=False,
                                  repair_id=REPAIR_ID, updated_utc=_utc())
                    _write(env, work/'report-last-error.json', dict(status))
                else:
                    status.update(current_error=None, updated_utc=_utc())
                _write(env, work/'report-status.json', dict(status))
                _write(env, runtime/'reporter-status.json', dict(status))
                if once or status.get('status') == 'COMPLETE':
                    return dict(status)
                stop.wait(max(.01, interval))
        finally:
            done.set(); thread.join(timeout=max(.1, heartbeat_interval+.1))
    return dict(status, status='STOPPED', training_controlled=False)


_CHANNEL_FIELDS = frozenset(('status', 'channel', 'recipe_sha256', 'complete_tasks',
    'total_tasks', 'updated_utc', 'error_type', 'http_status', 'commit', 'manifest',
    'path', 'bytes', 'files', 'collection_status', 'full_raw_results_uploaded',
    'remote_complete_blob_GET_SHA256_verified', 'remote_full_GET_SHA256_verified',
    'training_controlled', 'current_operation', 'heartbeat_utc', 'last_success_utc'))


def _monitor_snapshot(env, plan, clock=time.time):
    snapshot = env['runtime_module'].status(env['path'], plan=plan, clock=clock)
    now = clock(); scheduler = snapshot.get('scheduler', {})
    observed = scheduler.get('updated_at')
    age = max(0., now-float(observed)) if isinstance(observed, (int, float)) else None
    snapshot['scheduler_observation'] = dict(
        queue_polled_by_monitor=False, source='durable_controller_queue_observation',
        observed_unix=observed, age_seconds=round(age, 1) if age is not None else None,
        fresh=age is not None and age <= 90,
        cached_slurm_state_is_not_a_new_queue_observation=True)
    snapshot['uploads'] = {channel: {key: value for key, value in state.items()
                           if key in _CHANNEL_FIELDS and not isinstance(value, (dict, list))}
                          for channel, state in snapshot.get('uploads', {}).items()
                          if isinstance(state, dict)}
    for channel, state in snapshot['uploads'].items():
        state['progress_scope'] = 'channel receipt only; not scientific completion'
        state['observation_age_seconds'] = _age(state.get('updated_utc'), now)
        observer = _read(env, Path(env['runtime'])/('upload-'+channel)/'observer-status.json', {})
        if isinstance(observer, dict) and observer.get('recipe_sha256') == plan['recipe_sha256']:
            # Whole-cycle legacy status can be old during a healthy long
            # archive. Preserve that receipt while exposing fresh, bounded
            # observer evidence separately; a heartbeat proves no new upload.
            fields = _CHANNEL_FIELDS | frozenset(('transport_activity', 'legacy_verified_event_utc',
                'verified_groups', 'verified_task_groups', 'observer_heartbeat_utc',
                'legacy_cycle_status', 'legacy_cycle_updated_utc', 'status_scope',
                'latest_verified_event', 'latest_verified_event_utc', 'observed_utc',
                'cycle_complete', 'full_completion_claimed', 'lock_scope',
                'current_transport_activity', 'observed_verified_groups',
                'upload_metadata_observation_errors', 'verification_scope',
                'legacy_event_age_seconds', 'last_verified_part_utc',
                'last_index_publish_utc', 'last_blob_verified_utc',
                'original_upload_lock_owned', 'existing_uploader_preserved',
                'legacy_status_is_not_current_transport_proof'))
            state['observer'] = {key: value for key, value in observer.items()
                                 if key in fields and not isinstance(value, (dict, list))}
            state['observer_heartbeat_not_a_new_remote_verification'] = True
            if observer.get('status') == 'OBSERVING_EXISTING_UPLOADER':
                _effective_observer_status(state, observer, now)
    snapshot['management_repair'] = dict(repair_id=REPAIR_ID,
        monitor_observation_utc=_utc(), monitor_observation_unix=now,
        reporting_and_upload_do_not_control_training=True,
        monitor_error_history_preserved=True)
    return snapshot


def _effective_observer_status(state, observer, now):
    """Supersede an old error only with newer, recent remote progress proof."""
    state['legacy_channel_receipt'] = {key: value for key, value in state.items()
                                       if key in _CHANNEL_FIELDS and not isinstance(value, (dict, list))}
    last_error = observer.get('last_cycle_error')
    failure_utc = last_error.get('updated_utc') if isinstance(last_error, dict) else None
    if failure_utc is None and state.get('status') == 'RETRYING':
        failure_utc = state.get('updated_utc')
    proof_times = [observer.get(key) for key in ('last_verified_part_utc',
                   'last_blob_verified_utc', 'last_index_publish_utc')]
    proof_times = [value for value in proof_times if _age(value, now) is not None]
    newest = max(proof_times, key=lambda value: _epoch(value)) if proof_times else None
    heartbeat_age = _age(observer.get('observer_heartbeat_utc'), now)
    proof_age = _age(newest, now)
    proof_newer_than_error = (newest is not None and
        (failure_utc is None or (_epoch(failure_utc) is not None
                               and _epoch(newest) > _epoch(failure_utc))))
    fresh = (proof_age is not None and proof_age <= 1200
             and heartbeat_age is not None and heartbeat_age <= 90
             and _epoch(newest) <= now+60)
    effective = ('LEGACY_UPLOADER_ACTIVE_VERIFIED_PROGRESS'
                 if fresh and proof_newer_than_error else 'UPLOAD_PROGRESS_UNCONFIRMED')
    state['legacy_channel_observation_age_seconds'] = state.get('observation_age_seconds')
    state.update(status=effective, effective_status=effective,
                 status_scope='Current observer evidence; whole-cycle receipt preserved separately',
                 effective_last_remote_proof_utc=newest,
                 effective_remote_proof_age_seconds=proof_age,
                 observer_heartbeat_age_seconds=heartbeat_age,
                 last_cycle_error_superseded_by_newer_remote_proof=bool(fresh and proof_newer_than_error),
                 full_completion_claimed_by_observer=False)
    if heartbeat_age is not None:
        state.update(updated_utc=observer['observer_heartbeat_utc'],
                     observation_age_seconds=heartbeat_age)
    if fresh and proof_newer_than_error:
        # Keep the old exception only in its dated original receipt. It is not
        # the current headline once a newer remote verification proves progress.
        state.pop('error_type', None); state.pop('http_status', None)
        state['current_error'] = None
    else:
        state['current_error'] = dict(error_type='UploadProgressUnconfirmed',
            last_known_channel_error_type=state.get('error_type'),
            last_known_channel_error_utc=failure_utc)


def _epoch(value):
    try:
        import calendar
        return calendar.timegm(time.strptime(value, '%Y-%m-%dT%H:%M:%SZ'))
    except (TypeError, ValueError):
        return None


def _age(value, now):
    stamp = _epoch(value)
    return round(max(0., now-stamp), 1) if stamp is not None else None


def run_monitor(env, work, *, stop=None, credentials=None, launch_token=None,
                once=False, client_factory=None, interval=60., heartbeat_interval=5.,
                clock=time.time, **_):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    runtime = Path(env['runtime']); native_work = runtime/'upload-monitor'
    native_work.mkdir(parents=True, exist_ok=True); stop = stop or threading.Event()
    status = dict(status='STARTING', role='monitor', repair_id=REPAIR_ID,
                  recipe_sha256=env['header']['recipe_sha256'], training_controlled=False)
    with _role_lock(env, 'monitor') as acquired:
        if not acquired:
            return dict(status='EXISTING_MONITOR_REUSED', training_controlled=False)
        _ready(env, work, 'monitor', launch_token)
        _write(env, runtime/'monitor-upload-process.json',
               dict(kind='monitor', pid=os.getpid(), repair_id=REPAIR_ID,
                    launch_token=launch_token, started_utc=_utc()))
        done, thread = _start_heartbeat(env, work, 'monitor', status, heartbeat_interval)
        try:
            while not stop.is_set():
                try:
                    plan = _plan(env); u = _uploads(env)
                    snapshot = _monitor_snapshot(env, plan, clock=clock)
                    status.update(status='PUBLISHING', snapshot_utc=snapshot['updated_utc'],
                                  observed_complete_tasks=snapshot.get('complete_tasks'))
                    factory = client_factory
                    if factory is None:
                        from upload_fix import make_client
                        factory = lambda: make_client(env, 'monitor', native_work,
                            credentials=credentials, stop=stop)
                    client = factory()
                    target = ('monitoring/paper3-round11/'+plan['stage']+'/'+plan['recipe_sha256']+'/status.json')
                    commit = client.git_publish({target: env['common'].encoded(snapshot)},
                                                'Update R11 independent operational observation')
                    u.publish_stage_index(client, plan, 'monitor', dict(path=target, commit=commit),
                                          monitor_snapshot=snapshot)
                    env['common'].atomic(work/'monitor-last-success.json',
                        dict(commit=commit, snapshot_utc=snapshot['updated_utc'],
                             published_utc=_utc(), complete_tasks=snapshot.get('complete_tasks')))
                    status.update(status='PUBLISHED', commit=commit,
                                  last_success_utc=_utc(), current_error=None,
                                  updated_utc=_utc(), training_controlled=False)
                except Exception as error:
                    status.update(status='STOPPING' if stop.is_set() else 'RETRYING',
                                  current_error=_error(error), updated_utc=_utc(),
                                  training_controlled=False)
                    _write(env, work/'monitor-last-error.json', dict(status))
                _write(env, work/'monitor-status.json', dict(status))
                _write(env, native_work/'status.json', dict(status))
                if once:
                    return dict(status)
                stop.wait(max(.01, interval))
        finally:
            done.set(); thread.join(timeout=max(.1, heartbeat_interval+.1))
    return dict(status, status='STOPPED', training_controlled=False)
