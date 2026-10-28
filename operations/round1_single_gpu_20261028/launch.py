"""Detached R1 single allocation manager; numerical work remains in Slurm."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from single_gpu import atomic, read, emit, OPERATIONAL, resources


@contextmanager
def lock(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield handle


def identity(pid):
    try:
        if type(pid) is not int or pid <= 1:
            return None
        folder = Path('/proc') / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        return dict(pid=pid, uid=os.getuid(), start_ticks=fields[19],
            argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v])
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def option(argv, key):
    if not isinstance(argv, list) or argv.count(key) != 1:
        return None
    at = argv.index(key)
    return argv[at+1] if at+1 < len(argv) else None


def script(argv):
    if not argv or Path(argv[0]).resolve() != Path(sys.executable).resolve():
        return None
    at = 1
    while at < len(argv) and argv[at] in ('-I', '-S', '-B', '-u', '-E', '-s', '-O', '-OO'):
        at += 1
    return Path(argv[at]).resolve() if at < len(argv) else None


def same(record, current):
    return (isinstance(record, dict) and current is not None
        and current.get('argv') == record.get('argv')
        and (record.get('start_ticks') is None or record['start_ticks'] == current['start_ticks']))


def stop_exact(current, *, process=identity, send=os.kill, sleep=time.sleep, clock=time.monotonic):
    """Caller validates role; recheck PID/start/argv before each CPU-only signal."""
    signals = []
    for number, seconds in ((signal.SIGTERM, 12), (signal.SIGKILL, 5)):
        if process(current['pid']) != current:
            return dict(status='MANAGEMENT_STOPPED', pid=current['pid'], signals=signals)
        try:
            send(current['pid'], number)
        except ProcessLookupError:
            break
        signals.append(signal.Signals(number).name)
        emit('R1_PRIOR_CPU_MANAGER_STOP_REQUESTED', pid=current['pid'], signal=signals[-1],
            scientific_jobs_signalled=0)
        until = clock() + seconds
        while process(current['pid']) == current and clock() < until:
            sleep(.1)
    if process(current['pid']) == current:
        raise RuntimeError('PRIOR_CPU_MANAGER_EXIT_PENDING')
    return dict(status='MANAGEMENT_STOPPED', pid=current['pid'], signals=signals)


def context(root, home, work):
    root, home, work = (Path(p).expanduser().resolve() for p in (root, home, work))
    runtime = root / OPERATIONAL
    launch = read(runtime/'launch.json')
    if not isinstance(launch, dict) or Path(launch.get('root', '')).resolve() != root:
        raise ValueError('ORIGINAL_R1_LAUNCH_REQUIRED')
    command = launch.get('worker_command')
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise ValueError('ORIGINAL_R1_WORKER_COMMAND_REQUIRED')
    native = Path(launch.get('package') or option(command, '--package') or '').resolve()
    for name in ('worker.py', 'kernel_bridge.py', 'action_startup.py', 'START.py'):
        if not (native/name).is_file():
            raise FileNotFoundError('EXISTING_NATIVE_FILE_REQUIRED: ' + str(native/name))
    base = home/'paper3-r1-full-repair'
    candidates = [base/'refresh-fixed-43aeea5f1714c8171352', base/'2bfd664251ee5d1bcaf5']
    legacy = next((p for p in candidates if (p/'parallel_runtime.py').is_file()), None)
    if legacy is None:
        raise FileNotFoundError('EXISTING_R1_MANAGEMENT_ADAPTER_REQUIRED')
    old_cache = base/'2bfd664251ee5d1bcaf5/verified_context_cache.py'
    if not old_cache.is_file():
        raise FileNotFoundError('EXISTING_R1_ACTION_CACHE_REQUIRED')
    at = command.index('--root') if '--root' in command else -1
    if at < 2 or Path(command[at-1]).resolve() != native/'worker.py':
        raise ValueError('ORIGINAL_NATIVE_WORKER_COMMAND_LAYOUT_DIFFERS')
    worker = [launch.get('python', sys.executable), '-I', '-S', '-B', '-u',
        str(HERE/'worker_runner.py'), *command[at:]]
    return dict(root=str(root), home=str(home), work=str(work), runtime=str(runtime),
        legacy_dir=str(legacy), native_package=str(native), cache_path=str(old_cache),
        python=launch.get('python', sys.executable), worker_command=worker,
        resources=resources(), schema_version='paper3-r1-single-gpu-launch-v1')


def stop_prior_managers(config, *, process=identity, stop=stop_exact):
    root, home = Path(config['root']), Path(config['home'])
    old_work = home/'paper3-r1-full-repair/work'
    stopped = []
    for name in ('memory-recovery-process.json', 'controller-process.json'):
        record = read(old_work/name, {}) or {}
        current = process(record.get('pid'))
        if current is None:
            continue
        if not same(record, current) or option(current['argv'], '--root') != str(root):
            # A reused PID must not be signalled; controller.lock remains the fence.
            continue
        entry = script(current['argv'])
        if entry is None or '--child' in current['argv']:
            continue
        memory = (entry.name == 'recovery.py' and entry.is_relative_to(home/'paper3-r1-controller-recovery')
            and '--guard' in current['argv'])
        legacy = (entry.name == 'supervisor.py' and entry.is_relative_to(home/'paper3-r1-full-repair')
            and 'manage' in current['argv'])
        if memory or legacy:
            stopped.append(stop(current))
    return stopped


def retire_old_observer(config):
    import observer
    results = []
    for current in observer.legacy_monitor_candidates(config['root'], config['home'], config['work']):
        if current.get('pid') != os.getpid():
            results.append(stop_exact(current))
    return results


def run_cycle(config, token, stop):
    work = Path(config['work'])
    result = work/('cycle-'+token+'.json')
    result.unlink(missing_ok=True)
    command = [config['python'], '-I', '-S', '-B', '-u', str(HERE/'single_gpu.py'), '--cycle',
        '--root', config['root'], '--work', config['work'], '--legacy-dir', config['legacy_dir'],
        '--native-package', config['native_package'], '--worker-command-json', str(work/'worker-command.json'),
        '--result', str(result), '--submit']
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    started = time.monotonic(); last = 0
    try:
        while child.poll() is None:
            elapsed = time.monotonic()-started
            if elapsed-last >= 30:
                emit('R1_SINGLE_GPU_MANAGEMENT_CYCLE_ACTIVE', elapsed_seconds=round(elapsed),
                    scientific_jobs_signalled=0)
                last = elapsed
            if stop.wait(.2):
                raise InterruptedError('MANAGEMENT_STOP_REQUESTED')
            if elapsed >= 900:
                raise TimeoutError('MANAGEMENT_METADATA_CYCLE_EXCEEDED_900_SECONDS_SCIENCE_UNTOUCHED')
        if child.returncode:
            raise RuntimeError('MANAGEMENT_CYCLE_EXIT_'+str(child.returncode))
        value = read(result)
        if not isinstance(value, dict) or 'status' not in value:
            raise RuntimeError('MANAGEMENT_CYCLE_RESULT_MISSING')
        return value
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=4)


def guard(args):
    config = read(Path(args.work)/'config.json')
    if config.get('root') != str(Path(args.root).resolve()) or config.get('home') != str(Path(args.home).resolve()):
        raise ValueError('MANAGEMENT_CONFIG_ROOT_DIFFERS')
    work, runtime = Path(config['work']), Path(config['runtime'])
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with lock(work/'guardian.lock'):
        emit('R1_SINGLE_GPU_GUARDIAN_STARTED', pid=os.getpid(), resources=resources(),
            dependency_installations=0, scientific_jobs_signalled=0)
        prior = stop_prior_managers(config)
        atomic(work/'prior-management-stop.json', dict(stopped=prior, scientific_jobs_signalled=0))
        # Controller handoff is fenced even when an old metadata child is exiting.
        while not stop.is_set():
            try:
                controller_lock = (runtime/'controller.lock').open('a')
                fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                controller_lock.close()
                emit('R1_WAITING_FOR_PRIOR_METADATA_CYCLE', scientific_jobs_signalled=0)
                stop.wait(5)
        else:
            return 0
        try:
            try:
                retired = retire_old_observer(config)
                atomic(work/'prior-observer-stop.json', dict(stopped=retired))
                import observer
                atomic(work/'observer-start.json', observer.start(config['root'], config['home'], work))
            except Exception as error:
                emit('R1_OBSERVER_START_RETRY', error_type=type(error).__name__)
            failures = sequence = 0
            while not stop.is_set():
                if (work/'STOP').exists() or (runtime/'STOP').exists():
                    atomic(work/'controller-status.json', dict(status='STOPPED_BY_USER', resources=resources(),
                        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                    break
                sequence += 1
                try:
                    value = run_cycle(config, args.token, stop)
                    failures = 0
                    value.update(guardian_pid=os.getpid(), token=args.token, cycle=sequence,
                        resources=resources(), updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    atomic(work/'controller-status.json', value)
                    atomic(work/'controller-ready.json', dict(pid=os.getpid(), token=args.token,
                        first_slurm_observation_completed=True, status=value['status']))
                    emit('R1_SINGLE_GPU_PROGRESS', **value)
                    if value.get('all_complete') or value.get('all_original_complete') or value['status']=='STOPPED_BY_USER':
                        break
                    delay = 30
                except InterruptedError:
                    break
                except Exception as error:
                    failures += 1; delay = min(60, 5 * 2**min(failures-1, 4))
                    value = dict(status='RETRYING', guardian_pid=os.getpid(), token=args.token,
                        cycle=sequence, error_type=type(error).__name__, reason=str(error)[:1000],
                        resources=resources(), scientific_jobs_signalled=0, retry_seconds=delay,
                        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                    atomic(work/'controller-status.json', value)
                    emit('R1_SINGLE_GPU_MANAGEMENT_RETRY', **value)
                # Keep observer startup failures independent of scientific dispatch.
                try:
                    import observer
                    observer.start(config['root'], config['home'], work)
                except Exception as error:
                    emit('R1_OBSERVER_START_RETRY', error_type=type(error).__name__)
                stop.wait(delay)
        finally:
            controller_lock.close()
    return 0


def start(args):
    root, home = Path(args.root).expanduser().resolve(), Path(args.home).expanduser().resolve()
    work = Path(args.work or home/'paper3-r1-single-gpu/work').expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with lock(work/'start.lock'):
        previous = read(work/'controller-process.json', {}) or {}
        current = identity(previous.get('pid'))
        if same(previous, current) and script(current['argv']) == HERE/'launch.py' and '--guard' in current['argv']:
            return dict(status='R1_SINGLE_GPU_ALREADY_RUNNING', pid=current['pid'], work=str(work),
                resources=resources(), terminal_can_close=True)
        config = context(root, home, work)
        atomic(work/'config.json', config)
        atomic(work/'worker-command.json', config['worker_command'])
        # --apply is explicit authorization to resume; later STOP remains durable.
        for path in (work/'STOP', root/OPERATIONAL/'STOP'):
            if path.exists():
                atomic(work/('resume-'+hashlib.sha256(str(path).encode()).hexdigest()[:12]+'.json'),
                    dict(path=str(path), explicit_user_resume=True, saved_utc=time.time()))
                path.unlink()
        token = hashlib.sha256(os.urandom(32)).hexdigest()
        command = [config['python'], '-I', '-S', '-B', '-u', str(HERE/'launch.py'), '--guard',
            '--root', str(root), '--home', str(home), '--work', str(work), '--token', token]
        with (work/'controller.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                close_fds=True, start_new_session=True)
        record = dict(pid=child.pid, argv=command, root=str(root), token=token,
            start_ticks=None, started_utc=time.time())
        atomic(work/'controller-process.json', record)
        until = time.monotonic()+2
        while time.monotonic() < until and child.poll() is None:
            current = identity(child.pid)
            if current is not None and current['argv'] == command:
                record['start_ticks'] = current['start_ticks']; atomic(work/'controller-process.json', record)
                break
            time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError('R1_SINGLE_GPU_GUARDIAN_START_FAILED_READ_CONTROLLER_LOG')
        value = dict(status='R1_SINGLE_GPU_BACKGROUND_STARTED', pid=child.pid, work=str(work),
            resources=resources(), terminal_can_close=True, scientific_jobs_signalled=0,
            note='后台核对原作业并续跑；此回执只证明管理程序启动，实际GPU运行以队列和任务日志为准。')
        atomic(work/'STARTED.json', value)
        return value


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--apply', action='store_true'); modes.add_argument('--guard', action='store_true')
    modes.add_argument('--check', action='store_true')
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--home', default=str(Path.home())); parser.add_argument('--work'); parser.add_argument('--token')
    args = parser.parse_args(argv)
    if args.work is None:
        args.work = str(Path(args.home).expanduser().resolve()/'paper3-r1-single-gpu/work')
    if args.guard:
        return guard(args)
    if args.check:
        work = Path(args.work)
        value = dict(controller=read(work/'controller-status.json', {}), observer=read(work/'observer-status.json', {}),
            allocation=read(work/'allocation-state.json', {}), process=read(work/'controller-process.json', {}))
    else:
        value = start(args)
    emit('R1_SINGLE_GPU_RESULT', **value)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
