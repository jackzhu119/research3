"""Safe detached repair of R11 management, reports and upload observers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from legacy_controller import bounded_read
import processes

DEFAULT_PLAN = Path('/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json')
REPAIR_ID = 'paper3-r11-management-recovery-v2'
ROLES = processes.ROLES
PRESERVE_WRITERS = frozenset(('paper', 'full'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def emit(**value):
    try:
        print(json.dumps(dict(event='R11_MANAGEMENT_REPAIR', **value), ensure_ascii=False), flush=True)
    except (OSError, ValueError, TypeError):
        pass


def load_env(args):
    from environment import load_env as load
    return load(args.package, args.plan)


def validate_operational_history(env):
    from environment import validate_history
    return validate_history(env)


def _role_lock(env, role):
    return Path(env['runtime']) / ('controller.lock' if role == 'controller' else role+'-upload.lock')


def _read(path, default=None):
    return bounded_read(path, default, max_bytes=64*1024*1024)


def _write(env, path, value):
    """Management telemetry has no stop channel to current scientific workers."""
    try:
        env['common'].atomic(path, value)
        return True
    except Exception:
        return False


def _default_work(env):
    return Path.home()/'paper3-r11-management-recovery/work'/env['header']['recipe_sha256'][:16]


def _known_pids(env, work, role):
    runtime = Path(env['runtime'])
    names = [work/(role+'-process.json'),
             runtime/('controller-process.json' if role == 'controller' else role+'-upload-process.json')]
    if role != 'controller':
        names += [runtime/('upload-'+role)/'process.json']
    if role in PRESERVE_WRITERS:
        names += [runtime/(role+'-observer-process.json')]
    for base in ('paper3-r11-operational-repair', 'paper3-r11-io-repair'):
        names.append(Path.home()/base/'work'/env['header']['recipe_sha256'][:16]/(role+'-process.json'))
    return [row.get('pid') for path in names if isinstance(row := _read(path, {}), dict)]


def _preserve_history(env, work):
    """Copy management error/status bytes without touching scientific receipts."""
    runtime = Path(env['runtime'])
    names = ('controller-error.json', 'controller-status.json', 'controller-process.json',
             'controller-ready.json', 'reporter-status.json', 'scheduler-state.json')
    paths = [runtime/name for name in names]
    paths += [runtime/('upload-'+role)/'status.json' for role in ('paper','full','monitor')]
    saved = []
    for source in paths:
        try:
            size = source.stat().st_size
            if size > 64*1024*1024:
                continue
            with source.open('rb') as handle:
                data = handle.read(size+1)
            if len(data) != size:
                continue
            checksum = hashlib.sha256(data).hexdigest()
            target = work/'history'/source.relative_to(runtime).parent/source.stem/(checksum+'.json')
            if not target.exists():
                from atomic_io import atomic_bytes
                atomic_bytes(target, data)
            saved.append(dict(path=str(source.relative_to(runtime)), sha256=checksum, bytes=size))
        except (OSError, ValueError):
            # A transient status read must not prevent repairing another role.
            saved.append(dict(path=str(source.relative_to(runtime)), status='OBSERVATION_UNAVAILABLE'))
    return saved


def ready_for(identity, env, role):
    if not identity.get('current_patch') or not identity.get('launch_token') or not identity.get('work'):
        return None
    work = Path(identity['work']).expanduser().resolve()
    ready = _read(work/(role+'-ready.json'), {})
    if not isinstance(ready, dict) or not all((
            ready.get('status') == 'READY', ready.get('pid') == identity['pid'],
            ready.get('role', role) == role,
            ready.get('launch_token') == identity['launch_token'],
            ready.get('recipe_sha256') == env['header']['recipe_sha256'])):
        return None
    if role == 'controller':
        valid = (ready.get('first_slurm_observation_completed') is True
                 and ready.get('queue_observation_successful') is True
                 and processes.lock_live(_role_lock(env, role)))
    elif role in PRESERVE_WRITERS and ready.get('lock_scope') == 'repair_observer':
        valid = (ready.get('lock_acquired') is True
                 and ready.get('existing_uploader_preserved') is True
                 and ready.get('original_upload_lock_owned') is False
                 and processes.lock_live(work/(role+'-observer.lock')))
    else:
        valid = (ready.get('lock_acquired') is True
                 and processes.lock_live(_role_lock(env, role)))
    return ready if valid else None


def _role_result(identity, ready, role, **fields):
    observer = bool(ready and ready.get('lock_scope') == 'repair_observer')
    state = (role.upper()+'_OBSERVER_READY_LEGACY_UPLOADER_PRESERVED' if observer else
             'EXISTING_REPAIRED_HELPER_READY' if ready else 'EXISTING_REPAIRED_HELPER_STARTUP_PENDING')
    return dict(role=role, status=state, pid=identity['pid'], work=identity.get('work'),
                helper_ready=bool(ready), original_upload_lock_owned=(ready or {}).get('original_upload_lock_owned'),
                transfer_completion_claimed=False, **fields)


def check(args):
    env = load_env(args); history = validate_operational_history(env)
    runtime = Path(env['runtime'])
    work = Path(args.work or _default_work(env)).expanduser().resolve()
    roles = {}
    for role in ROLES:
        owned = processes.find_owned(env, role, _known_pids(env, work, role))
        roles[role] = dict(lock_live=processes.lock_live(_role_lock(env, role)),
                           owned_processes=[dict(pid=row['pid'], current_patch=row['current_patch'],
                                                 ready=bool(ready_for(row, env, role))) for row in owned])
    return dict(status='READ_ONLY_MANAGEMENT_CHECK', repair_id=REPAIR_ID, history=history,
                recipe_sha256=env['header']['recipe_sha256'], roles=roles,
                controller_error=_read(runtime/'controller-error.json', {}),
                scientific_sources_changed=False, scientific_workers_signalled=0,
                slurm_calls=0, gpu_used=False)


def _start_role(env, args, work, role, *, popen, sleep, clock, finder, stopper, ready_timeout):
    identities = finder(env, role, _known_pids(env, work, role))
    current = next((row for row in identities if row.get('current_patch')), None)
    if current is not None:
        return _role_result(current, ready_for(current, env, role), role,
                            duplicate_helpers_started=0, old_helpers_signalled=0)
    live_lock = processes.lock_live(_role_lock(env, role))
    legacy = [row for row in identities if not row.get('current_patch')]
    preserved = []; stopped = []
    if role in PRESERVE_WRITERS:
        # Current school logs prove these writers are making real verified
        # transfers. An old RETRYING JSON is not authorization to interrupt one.
        preserved = [dict(pid=row['pid'], start_ticks=row['start_ticks']) for row in legacy]
    elif live_lock and not legacy:
        return dict(role=role, status='LOCK_OWNER_UNCONFIRMED_LEFT_UNTOUCHED', helper_ready=False,
                    old_helpers_signalled=0, scientific_workers_signalled=0)
    else:
        for row in legacy:
            event = stopper(row, env, grace=2., sleep=sleep, clock=clock)
            stopped.append(event)
    token = hashlib.sha256(os.urandom(32)).hexdigest()
    python = str(Path(args.python or env['header']['python']).expanduser().absolute())
    command = [python, '-I', '-S', '-B', '-u', str(HERE/'entry.py'), role,
               '--plan', str(env['path']), '--package', str(env['package']),
               '--work', str(work), '--launch-token', token]
    if role == 'controller':
        command.append('--submit')
    if args.credentials is not None and role in ('paper','full','monitor'):
        command += ['--credentials', str(Path(args.credentials).expanduser())]
    with (work/(role+'.log')).open('ab', buffering=0) as log:
        child = popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                      start_new_session=True, close_fds=True)
    process = dict(role=role, pid=child.pid, launch_token=token, recipe_sha256=env['header']['recipe_sha256'],
                   started_utc=utc(), script=str(HERE/'entry.py'), work=str(work),
                   healthy_legacy_uploaders_preserved=preserved)
    _write(env, work/(role+'-process.json'), process)
    if role != 'controller':
        receipt = role+('-observer-process.json' if role in PRESERVE_WRITERS else '-upload-process.json')
        _write(env, Path(env['runtime'])/receipt, dict(process, operational_repair=True))
    until = clock()+ready_timeout
    observed = None; ready = None
    while True:
        code = child.poll()
        if code is not None:
            return dict(role=role, pid=child.pid, status='HELPER_START_FAILED', helper_ready=False,
                        exit_code=code, legacy_uploaders_preserved=preserved, handover=stopped)
        if finder is processes.find_owned:
            item = processes.identify(child.pid, env, role)
            observed = item if item and item.get('launch_token') == token else None
        else:
            owned = finder(env, role, (child.pid,))
            observed = next((row for row in owned if row['pid'] == child.pid and row.get('launch_token') == token), None)
        ready = ready_for(observed, env, role) if observed else None
        if ready or clock() >= until:
            break
        sleep(.1)
    if observed:
        result = _role_result(observed, ready, role)
    else:
        result = dict(role=role, pid=child.pid, status='HELPER_STARTUP_IDENTITY_PENDING', helper_ready=False)
    result.update(legacy_uploaders_preserved=preserved, handover=stopped,
                  old_helpers_signalled=sum(bool(row.get('signals')) for row in stopped),
                  original_worker_allocations_changed=False)
    return result


def start(args, *, popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic,
          finder=processes.find_owned, stopper=processes.stop_owned, ready_timeout=4.):
    if not args.submit:
        raise ValueError('Repair start requires explicit --submit for existing-slot replacement management')
    env = load_env(args)
    try:
        history = validate_operational_history(env)
    except Exception as error:
        # The controller revalidates this before any observation/submission.
        # A temporary management read must not block unrelated observers.
        history = dict(status='REQUIRED_HISTORY_OBSERVATION_UNAVAILABLE',
                       error_type=type(error).__name__, prior_authorization_verified=False,
                       task_history_reset=False, controller_must_revalidate=True)
    work = Path(args.work or _default_work(env)).expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
    results = []
    with (work/'start.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        saved = _preserve_history(env, work)
        _write(env, work/'INPUT.json', dict(history=history, header=env['header'],
                                          prior_management_observations=saved, updated_utc=utc()))
        for role in ROLES:
            try:
                row = _start_role(env, args, work, role, popen=popen, sleep=sleep, clock=clock,
                                  finder=finder, stopper=stopper, ready_timeout=ready_timeout)
            except Exception as error:
                row = dict(role=role, status='HELPER_START_RETRY_REQUIRED', helper_ready=False,
                           error_type=type(error).__name__, scientific_workers_signalled=0)
            results.append(row)
            _write(env, work/(role+'-handover.json'), dict(row, updated_utc=utc()))
        ready_count = sum(bool(row.get('helper_ready')) for row in results)
        failed = any(row['status'] in ('HELPER_START_FAILED','HELPER_START_RETRY_REQUIRED',
                                      'LOCK_OWNER_UNCONFIRMED_LEFT_UNTOUCHED') for row in results)
        result = dict(status='MANAGEMENT_REPAIR_READY' if ready_count == len(ROLES) else
                      'MANAGEMENT_REPAIR_PARTIAL' if failed else 'MANAGEMENT_REPAIR_STARTUP_PENDING',
                      repair_id=REPAIR_ID, recipe_sha256=env['header']['recipe_sha256'], work=str(work),
                      processes=results, ready_helpers=ready_count, total_helpers=len(ROLES),
                      scientific_workers_signalled=0, slurm_jobs_cancelled=0,
                      original_scientific_files_changed=False, original_plan_changed=False,
                      original_four_slots_preserved=True, healthy_existing_uploaders_preserved=True,
                      readiness_scope='Management helpers/observers started; no transfer or scientific completion claim',
                      full_transfer_completion_claimed=False, network_recovery_claimed=False,
                      supervisor_slurm_calls=0, first_queue_observation_submits=False,
                      actual_gpu_allocation_claimed=False,
                      terminal_can_close=True, updated_utc=utc())
        _write(env, work/'STARTED.json', result)
        return result


def _wait_management_lock(env, work, role, stop):
    while processes.lock_live(_role_lock(env, role)) and not stop.is_set():
        _write(env, work/(role+'-waiting.json'), dict(status='WAITING_ORIGINAL_MANAGEMENT_LOCK',
            role=role, pid=os.getpid(), updated_utc=utc(), scientific_workers_signalled=0))
        stop.wait(.5)
    return not stop.is_set()


def parser():
    value = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    value.add_argument('mode', choices=('check','start',*ROLES,'worker','science','science-daemon'))
    value.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    value.add_argument('--package', type=Path); value.add_argument('--work', type=Path)
    value.add_argument('--python'); value.add_argument('--credentials', type=Path)
    value.add_argument('--launch-token'); value.add_argument('--submit', action='store_true')
    value.add_argument('--once', action='store_true')
    value.add_argument('--slot', type=int, choices=range(4)); value.add_argument('--device', default='cuda:0')
    value.add_argument('--response-dir'); value.add_argument('--task-key')
    return value


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in ('science','science-daemon'):
        # The immutable scientific entry owns scientific argument validation.
        # Preserve even future scientific flags byte-for-byte.
        from controller import science_main
        return science_main(raw)
    args = parser().parse_args(raw)
    if args.mode == 'worker':
        if args.slot is None:
            raise ValueError('worker requires --slot')
        from controller import worker_loop
        return worker_loop(args.plan, args.slot, package=args.package, device=args.device, once=args.once)
    if args.mode in ('check','start'):
        result = check(args) if args.mode == 'check' else start(args)
    else:
        env = load_env(args)
        work = Path(args.work or _default_work(env)).expanduser().resolve(); work.mkdir(parents=True, exist_ok=True)
        stop = threading.Event()
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: stop.set())
        if args.mode not in PRESERVE_WRITERS and not _wait_management_lock(env, work, args.mode, stop):
            result = dict(status='STOPPED_BEFORE_MANAGEMENT_LOCK', role=args.mode)
        elif args.mode == 'controller':
            from controller import controller_loop
            result = controller_loop(args.plan, work, package=args.package, submit=args.submit,
                                     stop=stop, launch_token=args.launch_token)
        elif args.mode in PRESERVE_WRITERS:
            from upload_fix import run_channel
            result = run_channel(env, args.mode, work, stop=stop, credentials=args.credentials,
                                 launch_token=args.launch_token)
        elif args.mode == 'report':
            from report_fix import run_report
            result = run_report(env, work, stop=stop, launch_token=args.launch_token)
        else:
            from report_fix import run_monitor
            result = run_monitor(env, work, stop=stop, credentials=args.credentials,
                                 launch_token=args.launch_token)
    if result is not None:
        emit(**result)
    return 2 if isinstance(result, dict) and result.get('status') == 'MANAGEMENT_REPAIR_PARTIAL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
