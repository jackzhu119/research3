"""Install-independent supervisor: replace only the owned R11 controller."""
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
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from legacy_controller import bounded_read, validate_operational_history

DEFAULT_PLAN = Path('/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json')
OLD_SUPERVISOR_SHA = '3d63152c39ccaa50f99823e6159848fa7ba637128635c08b033726871fcd85e9'


def proc_identity(pid, proc_root='/proc'):
    try:
        pid = int(pid)
        if pid <= 1:
            return None
        base = Path(proc_root)/str(pid)
        if base.stat().st_uid != os.getuid():
            return None
        args = [x.decode('utf-8') for x in (base/'cmdline').read_bytes().split(b'\0') if x]
        raw = (base/'stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=pid, uid=base.stat().st_uid, start_ticks=fields[19], args=args)
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def argument(args, name):
    if args.count(name) != 1:
        return None
    at = args.index(name)
    return args[at+1] if at+1 < len(args) else None


def controller_identity(pid, env, *, proc_root='/proc'):
    identity = proc_identity(pid, proc_root)
    if identity is None:
        return None
    args = identity['args']
    plan = argument(args, '--plan')
    if not plan:
        return None
    launch = bounded_read(env['runtime']/'launch.json', {})
    allowed = {env['path'].resolve()}
    if (launch.get('plan_sha256') == env['header']['plan_sha256']
            and launch.get('campaign_identity') == env['header']['recipe_sha256']):
        allowed.add(Path(launch.get('plan', '')).expanduser().resolve())
    candidate = Path(plan).expanduser().resolve()
    if candidate not in allowed or not candidate.is_file():
        return None
    if env['common'].sha(candidate) != env['header']['plan_sha256']:
        return None
    python = {Path(sys.executable).resolve(), Path(env['header'].get('python', sys.executable)).resolve()}
    if not args or Path(args[0]).resolve() not in python:
        return None
    at = 1
    while at < len(args) and args[at] in ('-I', '-S', '-B', '-u', '-E', '-s', '-O', '-OO', '-q'):
        at += 1
    if at+1 >= len(args) or args[at+1] != 'controller':
        return None
    script = Path(args[at]).resolve()
    own = script == HERE/'resume_io.py'
    original = script == env['package']/'paper3r11/platform/entry.py'
    previous = (script.name == 'repair.py'
        and script.parent.parent == Path.home()/'paper3-r11-operational-repair'
        and script.is_file() and hashlib.sha256(script.read_bytes()).hexdigest() == OLD_SUPERVISOR_SHA)
    if not (own or original or previous):
        return None
    return dict(identity, script=str(script), current_patch=own)


def lock_live(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open('r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def stop_controller(pid, env, *, identify=controller_identity, send=os.kill,
                    sleep=time.sleep, clock=time.monotonic, grace=15):
    first = identify(pid, env)
    if first is None:
        return dict(status='CONTROLLER_OWNERSHIP_UNCONFIRMED', signals=[])
    def same():
        now = identify(pid, env)
        return now is not None and all(now[k] == first[k] for k in ('pid', 'uid', 'start_ticks', 'args'))
    if not same():
        return dict(status='PID_IDENTITY_CHANGED', signals=[])
    try:
        send(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return dict(status='CONTROLLER_ALREADY_ENDED', signals=[])
    signals = ['SIGTERM']; until = clock()+grace
    while same() and clock() < until:
        sleep(.1)
    if same():
        # An operational controller may be stuck in a filesystem syscall.
        # Never use a process group, worker PID or Slurm cancellation.
        if not same():
            return dict(status='PID_IDENTITY_CHANGED', signals=signals)
        try:
            send(int(pid), signal.SIGKILL); signals.append('SIGKILL')
        except ProcessLookupError:
            pass
    return dict(status='OWNED_CONTROLLER_STOP_REQUESTED', pid=int(pid), signals=signals,
                scientific_workers_signalled=0)


def load_env(args):
    from resume_io import load_env as load
    return load(args.package, args.plan)


def check(args):
    env = load_env(args); history = validate_operational_history(env)
    return dict(status='READ_ONLY_IO_REPAIR_CHECK', **history,
        controller_lock_live=lock_live(env['runtime']/'controller.lock'),
        controller_error=bounded_read(env['runtime']/'controller-error.json', {}),
        scheduler=bounded_read(env['runtime']/'scheduler-state.json', {}),
        scientific_sources_changed=False, slurm_calls=0)


def start(args, *, popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic,
          identify=controller_identity, stop=stop_controller):
    if not args.submit:
        raise ValueError('Resume requires explicit --submit; installation alone never submits')
    env = load_env(args); history = validate_operational_history(env)
    work = Path(args.work or Path.home()/'paper3-r11-io-repair/work'/env['header']['recipe_sha256'][:16]).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    atomic = env['common'].atomic
    with (work/'start.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic(work/'INPUT.json', dict(history=history, header=env['header'],
            original_scientific_files_changed=False, live_workers_preserved=True))
        candidates = [bounded_read(work/'controller-process.json', {}).get('pid'),
                      bounded_read(env['runtime']/'controller-process.json', {}).get('pid')]
        owned = next((x for pid in dict.fromkeys(candidates) if pid
                      and (x := identify(pid, env)) is not None), None)
        # A detached child can still be verifying the original plan before it
        # acquires controller.lock. Repeating the command must not spawn a
        # second controller during that interval.
        pending = bounded_read(work/'controller-process.json', {})
        nonce = argument(owned['args'], '--launch-token') if owned else None
        if (owned and owned.get('current_patch') and nonce
                and pending.get('pid') == owned['pid'] and pending.get('launch_token') == nonce
                and pending.get('recipe_sha256') == env['header']['recipe_sha256']
                and not lock_live(env['runtime']/'controller.lock')):
            return dict(status='EXISTING_IO_CONTROLLER_STARTUP_PENDING', pid=owned['pid'], work=str(work),
                terminal_can_close=True, scientific_workers_signalled=0, slurm_jobs_cancelled=0)
        if lock_live(env['runtime']/'controller.lock'):
            if owned is None:
                result = dict(status='CONTROLLER_OWNERSHIP_UNCONFIRMED_NO_ACTION', work=str(work),
                    scientific_workers_signalled=0, slurm_jobs_cancelled=0)
                atomic(work/'STARTED.json', result); return result
            ready = bounded_read(work/'controller-ready.json', {})
            nonce = argument(owned['args'], '--launch-token')
            if (owned.get('current_patch') and nonce and ready.get('pid') == owned['pid']
                    and ready.get('launch_token') == nonce
                    and ready.get('atomic_patch_loaded') is True
                    and ready.get('queue_observation_successful') is True
                    and ready.get('first_slurm_observation_completed') is True
                    and ready.get('recipe_sha256') == env['header']['recipe_sha256']):
                return dict(status='EXISTING_IO_CONTROLLER_REUSED', pid=owned['pid'], work=str(work),
                    terminal_can_close=True, scientific_workers_signalled=0, slurm_jobs_cancelled=0)
            pending = bounded_read(work/'controller-process.json', {})
            if (owned.get('current_patch') and nonce and pending.get('pid') == owned['pid']
                    and pending.get('launch_token') == nonce
                    and pending.get('recipe_sha256') == env['header']['recipe_sha256']):
                return dict(status='EXISTING_IO_CONTROLLER_STARTUP_PENDING', pid=owned['pid'], work=str(work),
                    terminal_can_close=True, scientific_workers_signalled=0, slurm_jobs_cancelled=0)
            stopped = stop(owned['pid'], env)
            atomic(work/'OLD_CONTROLLER.json', stopped)
            until = clock()+10
            while lock_live(env['runtime']/'controller.lock') and clock() < until:
                sleep(.1)
            if lock_live(env['runtime']/'controller.lock'):
                result = dict(status='OLD_CONTROLLER_LOCK_NOT_RELEASED', work=str(work),
                    scientific_workers_signalled=0, slurm_jobs_cancelled=0)
                atomic(work/'STARTED.json', result); return result
        python = str(Path(args.python or env['header']['python']).expanduser().absolute())
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        command = [python, '-I', '-S', '-B', '-u', str(HERE/'resume_io.py'), 'controller',
            '--plan', str(env['path']), '--package', str(env['package']), '--work', str(work),
            '--launch-token', token, '--submit']
        with (work/'controller.log').open('ab', buffering=0) as log:
            child = popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                          start_new_session=True, close_fds=True)
        process = dict(pid=child.pid, launch_token=token, recipe_sha256=env['header']['recipe_sha256'],
                       kind='controller', script=str(HERE/'resume_io.py'), started_utc=env['common'].utc())
        atomic(work/'controller-process.json', process)
        until = clock()+8; status = 'STARTUP_PENDING'
        while True:
            ready = bounded_read(work/'controller-ready.json', {})
            code = child.poll()
            if code is not None:
                status = 'STARTUP_FAILED'; break
            if (ready.get('pid') == child.pid and ready.get('launch_token') == token
                    and ready.get('recipe_sha256') == env['header']['recipe_sha256']
                    and ready.get('atomic_patch_loaded') is True
                    and ready.get('queue_observation_successful') is True
                    and ready.get('first_slurm_observation_completed') is True):
                status = 'IO_REPAIR_CONTROLLER_READY'; break
            if clock() >= until:
                break
            sleep(.1)
        result = dict(status=status, pid=child.pid, work=str(work), log=str(work/'controller.log'),
            terminal_can_close=True, scientific_workers_signalled=0, slurm_jobs_cancelled=0,
            original_scientific_files_changed=False, original_plan_changed=False,
            worker_policy='adopt live allocations; replace only ended slots, maximum four',
            upload_processes_modified=0, actual_gpu_allocation_claimed=False)
        atomic(work/'STARTED.json', result); return result


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode', choices=('check', 'start')); p.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    p.add_argument('--package', type=Path); p.add_argument('--work', type=Path); p.add_argument('--python')
    p.add_argument('--submit', action='store_true'); args = p.parse_args(argv)
    result = start(args) if args.mode == 'start' else check(args)
    print(json.dumps(dict(event='R11_IO_REPAIR', **result), ensure_ascii=False), flush=True)
    return 2 if result['status'] in ('STARTUP_FAILED', 'OLD_CONTROLLER_LOCK_NOT_RELEASED',
        'CONTROLLER_OWNERSHIP_UNCONFIRMED_NO_ACTION') else 0


if __name__ == '__main__':
    raise SystemExit(main())
