#!/usr/bin/env python3
"""Replace only login-side wrappers after verifying the existing held workers."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

DEFAULT_ROOT = '/public/home/zhuziyuan/projects/paper3'


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, sort_keys=True), flush=True)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def held_lock(path):
    try:
        stream = Path(path).open('r')
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def proc_snapshot(pid, proc_root=Path('/proc')):
    """UID, full argv, boot and start ticks bind a signal to one local process."""
    try:
        folder = Path(proc_root) / str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        tokens = [x.decode(errors='replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
        return {'pid': int(pid), 'uid': folder.stat().st_uid, 'argv': tokens,
                'start_ticks': int(fields[19]),
                'boot_id': (Path(proc_root)/'sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, ValueError, IndexError):
        return None


def scan_processes():
    values = []
    for folder in Path('/proc').iterdir():
        if folder.name.isdigit():
            row = proc_snapshot(int(folder.name))
            if row:
                values.append(row)
    return values


def option(tokens, name):
    try:
        return tokens[tokens.index(name) + 1]
    except (ValueError, IndexError):
        return None


def matches(row, script, root, round_number, kind, runtime):
    argv = row.get('argv', [])
    if row.get('uid') != os.getuid() or str(script) not in argv:
        return False
    if option(argv, '--root') != str(root):
        return False
    if round_number != 1 and option(argv, '--round') != str(round_number):
        return False
    if kind == 'controller':
        return '--controller' in argv if round_number == 1 else option(argv, '--runtime') == str(runtime)
    return option(argv, '--operational') == str(runtime)


def memory_mib(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)(?:[cn])?', str(value), re.I)
    if not match:
        return None
    return round(float(match[1]) * {'': 1, 'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2}[match[2].upper()])


def call(argv, runner):
    result = runner(argv, text=True, capture_output=True, timeout=25)
    if result.returncode:
        raise RuntimeError('scheduler_read_failed:' + argv[0])
    return result.stdout


def verify_held(runtime, launch, round_number, runner):
    state = read(runtime/'scheduler-state.json', {}) or {}
    identity = [str(runtime), launch.get('campaign_identity', 'original-campaign-v1' if round_number == 1 else 'round5-6-original-recipe')]
    marker = 'paper3r%dp-' % round_number + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:16]
    if state.get('marker') != marker:
        raise RuntimeError('scheduler_marker_not_original_runtime_identity')
    wanted = {}
    for slot in range(3):
        attempts = state.get('slots', {}).get(str(slot), {}).get('attempts', [])
        if not attempts or not str(attempts[-1].get('job_id', '')).isdigit():
            raise RuntimeError('three_existing_job_acknowledgements_required')
        wanted[str(attempts[-1]['job_id'])] = slot
    if len(wanted) != 3:
        raise RuntimeError('three_distinct_existing_jobs_required')
    account = launch.get('account') or getpass.getuser()
    text = call(['squeue', '--array', '--noheader', '--user', account,
                 '--format=%i|%u|%k|%T|%j|%P|%R'], runner)
    rows = []
    for line in text.splitlines():
        fields = [v.strip() for v in line.split('|')]
        if len(fields) >= 7 and fields[2] == marker:
            rows.append(fields)
    if len(rows) != 3 or {v[0] for v in rows} != set(wanted):
        raise RuntimeError('exact_three_original_workers_not_visible')
    partition = 'hpc_gpu' if round_number == 1 else 'hpc_gpu1'
    memories = ['43G', '43G', '42G'] if round_number == 1 else ['64G'] * 3
    proof = []
    for fields in rows:
        job, user, comment, status, name, actual_partition, reason = fields[:7]
        slot = wanted[job]
        if (user != account or status != 'PENDING' or actual_partition != partition or
                not re.fullmatch(re.escape(marker) + '-s' + str(slot) + r'-a[1-9]\d*', name)):
            raise RuntimeError('existing_worker_is_not_owned_held_pending')
        detail = call(['scontrol', 'show', 'job', '-o', job], runner)
        values = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)', detail))
        tres = dict(x.split('=', 1) for x in values.get('ReqTRES', values.get('TRES', '')).split(',') if '=' in x)
        if (values.get('JobId') != job or values.get('JobName') != name or values.get('Comment') != marker or
                values.get('UserId', '').split('(', 1)[0] != account or values.get('Partition') != partition or
                values.get('JobState') != 'PENDING' or values.get('Priority') != '0' or
                values.get('Reason') not in ('JobHeldUser', 'JobHeldAdmin') or
                values.get('NumNodes') not in ('1', '1-1') or values.get('NumCPUs') != '12'):
            raise RuntimeError('existing_worker_identity_or_resource_mismatch:' + job)
        actual_memory = memory_mib(values.get('MinMemoryNode', tres.get('mem', '')))
        gpu_text = values.get('TresPerNode', values.get('Gres', ''))
        one_gpu = tres.get('gres/gpu') == '1' or bool(re.search(r'gpu(?::[^,=]+)*:1(?:,|$)', gpu_text))
        if actual_memory != memory_mib(memories[slot]) or not one_gpu:
            raise RuntimeError('existing_worker_memory_or_gpu_mismatch:' + job)
        proof.append({'job_id': job, 'slot': slot, 'partition': partition, 'cpus': 12,
                      'memory': memories[slot], 'gpus': 1, 'observed_NumNodes': values['NumNodes']})
    return {'marker': marker, 'jobs': sorted(proof, key=lambda r: r['slot'])}


def old_folder(launch, round_number):
    filename = 'worker.py' if round_number == 1 else 'three_jobs.py'
    candidates = [Path(value).parent for value in launch.get('worker_command', [])
                  if isinstance(value, str) and Path(value).name == filename]
    if len(candidates) != 1:
        raise RuntimeError('existing_operational_package_not_unique')
    return candidates[0].resolve()


def _repair_round_unlocked(root, round_number, package, *, execute=False, runner=subprocess.run,
                 killer=os.kill, processes=scan_processes, snapshot=proc_snapshot,
                 sleeper=time.sleep, clock=time.monotonic, lock_check=held_lock):
    root, package = Path(root).resolve(), Path(package).resolve()
    runtime = root / ('runs/training_campaign_parallel_v1' if round_number == 1 else
                      'runs/followup_20261024_r%d/runtime/three_jobs' % round_number)
    launch = read(runtime/'launch.json')
    if not isinstance(launch, dict):
        raise RuntimeError('existing_launch_receipt_required')
    new_folder = package/('round1' if round_number == 1 else 'round56')
    starter = new_folder/'START.py'
    if not starter.is_file():
        raise RuntimeError('fixed_package_entry_missing')
    controller_name = 'START.py' if round_number == 1 else 'coordinator.py'
    current = processes()
    if any(matches(p, new_folder/controller_name, root, round_number, 'controller', runtime) for p in current):
        return {'round': round_number, 'status': 'ALREADY_FIXED_CONTROLLER_RUNNING', 'runtime': str(runtime)}
    gate = read(runtime/'HANDOFF_READY.json', {}) or {}
    if gate.get('status') == 'READY':
        return {'round': round_number, 'status': 'OBSERVE_ONLY_HANDOFF_ALREADY_READY', 'runtime': str(runtime)}
    proof = verify_held(runtime, launch, round_number, runner)
    prior_folder = old_folder(launch, round_number)
    selected = []
    for kind, name in [('controller', controller_name), ('monitor', 'monitor.py')]:
        for row in current:
            if matches(row, prior_folder/name, root, round_number, kind, runtime):
                selected.append({**row, 'kind': kind})
    if sum(row['kind'] == 'controller' for row in selected) > 1:
        raise RuntimeError('multiple_exact_old_controllers_require_observation')
    result = {'round': round_number, 'status': 'REPAIR_PREPARED', 'runtime': str(runtime),
              'held_queue_proof': proof, 'old_package': str(prior_folder),
              'fixed_package': str(new_folder), 'processes': selected}
    if not execute:
        return result
    history = runtime/'repair-history'/str(time.time_ns())
    history.mkdir(parents=True, exist_ok=False)
    atomic(history/'before.json', {**result, 'launch': launch,
                                   'scheduler_state': read(runtime/'scheduler-state.json')})
    credentials = next((option(row['argv'], '--credentials') for row in selected
                        if row['kind'] == 'monitor' and option(row['argv'], '--credentials')), None)
    for row in selected:
        actual = snapshot(row['pid'])
        expected = {key: row[key] for key in ('pid', 'uid', 'argv', 'start_ticks', 'boot_id')}
        if actual is None:
            continue
        if actual != expected:
            raise RuntimeError('local_process_identity_changed_before_signal')
        if (read(runtime/'HANDOFF_READY.json', {}) or {}).get('status') == 'READY':
            raise RuntimeError('handoff_completed_while_repair_prepared')
        signum = signal.SIGKILL if row['kind'] == 'controller' else signal.SIGTERM
        killer(row['pid'], signum)
        atomic(history/('signal-%s.json' % row['pid']), {'process': row, 'signal': int(signum),
                                                      'scope': 'login_controller_or_monitor_only'})
    deadline = clock() + 15
    while lock_check(runtime/'controller.lock') or lock_check(runtime/'monitor.lock'):
        if clock() >= deadline:
            raise RuntimeError('old_login_wrapper_lock_release_pending')
        sleeper(.2)
    # The native START frontends regenerate only operational launch metadata.
    # Runtime path + campaign identity remain unchanged, hence the nine job IDs
    # and scheduler receipts are adopted without a new sbatch call here.
    command = [sys.executable, '-I', '-S', '-B', '-u', str(starter), '--root', str(root),
               '--python', sys.executable, '--execute']
    if launch.get('old_job'):
        command += ['--old-job', str(launch['old_job'])]
    if round_number == 1:
        command += ['--direct-package', launch['direct_package'], '--payload-manifest', launch['payload_manifest']]
    else:
        command += ['--round', str(round_number), '--runtime', str(runtime), '--scientific-package', launch['package'],
                    '--account', launch.get('account') or getpass.getuser()]
    if credentials:
        command += ['--credentials', credentials]
    started = runner(command, text=True, capture_output=True, timeout=90)
    atomic(history/'restart.json', {'command': command, 'returncode': started.returncode,
                                  'stdout': started.stdout, 'stderr': started.stderr})
    if started.returncode:
        raise RuntimeError('fixed_login_wrapper_start_failed:see_repair_history')
    result.update(status='FIXED_CONTROLLERS_STARTED_EXISTING_JOBS_PRESERVED',
                  history=str(history), starter_stdout=started.stdout,
                  jobs=[row['job_id'] for row in proof['jobs']])
    atomic(history/'complete.json', result)
    return result



def repair_round(root, round_number, package, **options):
    if not options.get('execute', False):
        return _repair_round_unlocked(root, round_number, package, **options)
    runtime = Path(root).resolve() / ('runs/training_campaign_parallel_v1' if round_number == 1 else
                    'runs/followup_20261024_r%d/runtime/three_jobs' % round_number)
    # Independent from the controller/monitor locks which must be released
    # before the new START frontend is allowed to acquire them.
    with (runtime/'repair.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'round': round_number, 'status': 'REPAIR_ALREADY_RUNNING', 'runtime': str(runtime)}
        return _repair_round_unlocked(root, round_number, package, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--rounds', nargs='+', type=int, choices=(1, 5, 6), default=[1, 5, 6])
    parser.add_argument('--package', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    failures = 0
    for round_number in dict.fromkeys(args.rounds):
        try:
            value = repair_round(args.root, round_number, args.package, execute=args.execute)
            emit('SLURM_NODE_RANGE_REPAIR', **value)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            failures += 1
            emit('SLURM_NODE_RANGE_REPAIR_RETRY', round=round_number, error_type=type(error).__name__,
                 reason=str(error), scientific_outputs_unchanged=True,
                 note='No repair command submits, releases, or cancels Slurm jobs; existing native controllers own handoff.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
