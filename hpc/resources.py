#!/usr/bin/env python3
"""Read-only resource observations around an owned Slurm child process.

This wrapper never changes scientific thread limits, submits a job, restores a
checkpoint, or claims an upload. It records actual exit status and measurements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

STAGES = ('test', 'prepare', 'pilot', 'train', 'backup')
PHYSICAL = {'cyhpc_1': {'max_cpus': 56, 'max_mem_gib': 503, 'max_gpus': 0},
            'hpc_gpu1': {'max_cpus': 76, 'max_mem_gib': 1007, 'max_gpus': 8}}


class ResourceError(ValueError):
    pass


def _need(value, code):
    if not value:
        raise ResourceError(code)


def _number(value):
    try:
        return int(value) if re.fullmatch(r'[0-9]+', str(value)) else None
    except (TypeError, ValueError):
        return None


def _time_seconds(value):
    """Slurm numeric TimeLimit is minutes; D-HH:MM:SS/H:MM:SS are explicit."""
    if not isinstance(value, str):
        return None
    if re.fullmatch(r'[0-9]+', value):
        return int(value) * 60
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', value)
    if match and int(match[3]) < 60 and int(match[4]) < 60:
        return int(match[1] or 0) * 86400 + int(match[2]) * 3600 + int(match[3]) * 60 + int(match[4])
    return None


def _device_count(value):
    if not isinstance(value, str) or not value or value in {'-1', 'NoDevFiles', 'none', 'void'}:
        return 0 if isinstance(value, str) and value in {'-1', 'NoDevFiles', 'none', 'void', ''} else None
    devices = set()
    for part in value.split(','):
        if re.fullmatch(r'\d+', part):
            devices.add(part)
        elif re.fullmatch(r'\d+-\d+', part):
            low, high = map(int, part.split('-'))
            if high < low or high - low > 1024:
                return None
            devices.update(map(str, range(low, high + 1)))
        elif re.fullmatch(r'(?:GPU|MIG)-[A-Za-z0-9/-]+', part):
            devices.add(part)
        else:
            return None
    return len(devices)


def _safe_path(path):
    path = Path(os.path.abspath(path))
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'RESOURCE_SYMLINK_PATH_REFUSED')
    return path


def write_json(path, value):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.resource-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def allocation(environ=None, hostname=None):
    env = os.environ if environ is None else environ
    job = env.get('SLURM_JOB_ID', '')
    _need(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job) is not None, 'SLURM_JOB_REQUIRED_NO_LOGIN_TRAINING')
    host = socket.gethostname() if hostname is None else hostname
    node = env.get('SLURMD_NODENAME', host)
    _need(not any(str(item).split('.')[0].lower().startswith('login') for item in (host, node)),
          'LOGIN_NODE_EXECUTION_REFUSED')
    cpus = _number(env.get('SLURM_CPUS_PER_TASK'))
    memory = _number(env.get('SLURM_MEM_PER_NODE'))
    if memory is None:
        per_cpu = _number(env.get('SLURM_MEM_PER_CPU'))
        memory = per_cpu * cpus if per_cpu is not None and cpus is not None else None
    gpus = _number(env.get('SLURM_GPUS_ON_NODE'))
    if gpus is None:
        gpus = _device_count(env.get('SLURM_JOB_GPUS'))
    if gpus is None:
        gpus = _device_count(env.get('CUDA_VISIBLE_DEVICES'))
    partition = env.get('SLURM_JOB_PARTITION')
    if partition not in PHYSICAL:
        partition = None
    return {'job_id': job, 'partition': partition, 'cpus': cpus,
            'memory_gib': memory / 1024 if memory is not None else None,
            'gpus': gpus, 'time_limit_seconds': _time_seconds(env.get('SLURM_TIMELIMIT', env.get('PAPER3_HPC_TIME_LIMIT'))),
            'requested_wall_seconds': _number(env.get('PAPER3_HPC_WALL_TIME_SECONDS')),
            'account_quota_verified': False}


def process_tree(pid, proc_root=Path('/proc')):
    """Read only PPid and VmRSS; never cmdline, environment or task payload."""
    values = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parent = rss = None
            for line in (entry / 'status').read_text().splitlines():
                if line.startswith('PPid:'):
                    parent = int(line.split()[1])
                elif line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) * 1024
            if parent is not None:
                values[int(entry.name)] = (parent, rss or 0)
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        added = {child for child, (parent, _) in values.items() if parent in owned}
        if added <= owned:
            break
        owned |= added
    return owned, sum(values.get(child, (0, 0))[1] for child in owned)


def query_gpu(owned_pids, runner=subprocess.run):
    """Measure only compute PIDs descended from this task, not neighbor jobs."""
    try:
        response = runner(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=5, check=False)
        if response.returncode:
            return {'available': False, 'owned_process_memory_mib': 0, 'owned_gpus': []}
        memory = {}
        for line in response.stdout.splitlines():
            fields = [v.strip() for v in line.split(',')]
            if len(fields) == 3 and re.fullmatch(r'GPU-[A-Za-z0-9-]+', fields[0]):
                pid, amount = _number(fields[1]), _number(fields[2])
                if pid in owned_pids and amount is not None:
                    memory[fields[0]] = memory.get(fields[0], 0) + amount
        devices = []
        if memory:
            response = runner(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.total',
                               '--format=csv,noheader,nounits'], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=5, check=False)
            if response.returncode == 0:
                for line in response.stdout.splitlines():
                    row = [v.strip() for v in line.split(',')]
                    if len(row) == 3 and row[0] in memory:
                        utilization, total = _number(row[1]), _number(row[2])
                        if utilization is not None and total is not None:
                            devices.append({'gpu_uuid': row[0], 'device_utilization_percent': utilization,
                                            'total_memory_mib': total, 'owned_process_memory_mib': memory[row[0]]})
        return {'available': True, 'owned_process_memory_mib': sum(memory.values()), 'owned_gpus': devices}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {'available': False, 'owned_process_memory_mib': 0, 'owned_gpus': []}


def recommendations(report, limits=None):
    allocation_value = report['allocation']
    measured = report['measurements']
    cpus, memory = allocation_value.get('cpus'), allocation_value.get('memory_gib')
    wall = measured['wall_seconds']
    fraction = measured['cpu_seconds'] / wall / cpus if wall > 0 and cpus else None
    memory_fraction = measured['peak_rss_bytes'] / 1024**3 / memory if memory else None
    physical = PHYSICAL.get(allocation_value.get('partition'), {})
    limits = limits or {}
    max_cpu = limits.get('max_cpus')
    max_memory = limits.get('max_mem_gib')
    cpu_evidence = wall >= 60 and measured.get('samples', 0) >= 3 and fraction is not None
    cpu = {'allocated': cpus, 'measured_allocation_utilization_percent': fraction * 100 if fraction is not None else None,
           'suggested': cpus, 'action': 'keep_until_measured', 'automatic_apply': False}
    mem = {'allocated_gib': memory, 'measured_peak_fraction': memory_fraction,
           'suggested_gib': memory, 'action': 'keep_until_measured', 'automatic_apply': False}
    if cpu_evidence:
        if fraction < .35:
            cpu['action'] = 'review_lower_request_or_independent_IO_work; do_not_raise_scientific_threads'
        elif fraction >= .85:
            cpu['action'] = 'manual_scaling_benchmark_before_request_increase'
            if isinstance(max_cpu, int) and max_cpu > cpus:
                cpu['candidate_for_manual_benchmark'] = min(max_cpu, physical.get('max_cpus', max_cpu), math.ceil(cpus * 1.25))
    if memory_fraction is not None and wall >= 60:
        if memory_fraction >= .85:
            mem['action'] = 'memory_pressure_review_quota_before_increase'
            if isinstance(max_memory, (int, float)) and max_memory > memory:
                candidate = math.ceil(max(memory * 1.25, measured['peak_rss_bytes'] / 1024**3 * 1.3))
                mem['suggested_gib'] = min(candidate, max_memory, physical.get('max_mem_gib', max_memory))
                mem['action'] = 'manual_request_increase_within_user_confirmed_limit'
        elif memory_fraction < .35:
            mem['action'] = 'review_lower_request_after_all_stages_measured'
    time_limit = allocation_value.get('time_limit_seconds')
    request_time = allocation_value.get('requested_wall_seconds')
    timing = {'measured_wall_seconds': wall, 'allocated_limit_seconds': time_limit,
              'requested_wall_seconds': request_time,
              'action': 'record_only_limit_not_observed', 'automatic_apply': False}
    if time_limit:
        timing['measured_limit_fraction'] = wall / time_limit
        timing['action'] = ('review_longer_Slurm_request_within_quota_without_changing_scientific_budget'
                            if wall / time_limit >= .85 else 'keep_until_all_stages_measured')
    normal_return = report.get('actual_exit_code') == 0 and not report.get('signals_forwarded')
    if not normal_return:
        timing['action'] = 'preserve_request_after_pause_or_failure; do_not_reduce_from_incomplete_run'
    elif wall >= 60:
        candidate = max(600, math.ceil(wall * 1.5 / 60) * 60)
        current = time_limit or request_time
        max_hours = limits.get('max_time_hours')
        maximum = max_hours * 3600 if isinstance(max_hours, (int, float)) and max_hours > 0 else None
        if current and candidate <= current:
            timing['candidate_for_manual_walltime_seconds'] = candidate
            timing['action'] = 'manual_future_request_reference_for_this_completed_stage_only'
        elif maximum and candidate <= maximum:
            timing['candidate_for_manual_walltime_seconds'] = candidate
            timing['action'] = 'manual_future_request_within_user_confirmed_time_limit'
    return {'schema_version': 'paper3-hpc-resource-recommendations-v1', 'cpu': cpu, 'memory': mem, 'time': timing,
            'gpu': {'allocated': allocation_value.get('gpus'), 'suggested': allocation_value.get('gpus'),
                    'measured_owned_peak_memory_mib': measured.get('owned_process_gpu_peak_memory_mib'),
                    'action': 'keep; multi_gpu_scaling_not_verified', 'automatic_apply': False},
            'quota_limits_supplied': bool(limits and any(value is not None for value in limits.values())),
            'physical_limits_are_not_account_quotas': True, 'scientific_thread_profile_changed': False,
            'notes': ['Recommendations do not submit or modify jobs.',
                      'Memory is max(wait4 peak RSS, sampled live descendant RSS sum), not an exact continuous cgroup peak.',
                      'No CPU increase is applied without a separate scaling check and school quota approval.']}


def _limits(root):
    try:
        value = json.loads(_safe_path(root / 'hpc/resources.json').read_bytes())
        row = value.get('recommendation_limits', {})
        return row if isinstance(row, dict) else {}
    except (OSError, ValueError):
        return {}


def run_stage(command, root, stage, output, *, environ=None, hostname=None,
              popen=subprocess.Popen, waiter=os.wait4, gpu_query=query_gpu,
              process_query=process_tree, clock=time.monotonic, sleep=time.sleep, interval=10):
    """Run one owned command. Unit fixtures inject wait4/Popen, never a fit."""
    _need(stage in STAGES and command, 'RESOURCE_STAGE_AND_COMMAND_REQUIRED')
    root, output = _safe_path(root), _safe_path(output)
    assigned = allocation(environ, hostname)
    env = dict(os.environ if environ is None else environ)
    report = {'schema_version': 'paper3-hpc-resource-observation-v1', 'stage': stage,
              'observation_id': uuid.uuid4().hex,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'allocation': assigned,
              'command_entry': Path(command[0]).name, 'signals_forwarded': [],
              'checkpoint_saved_claimed': False, 'upload_success_claimed': False}
    started = clock()
    child = popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    old_handlers = {}
    def forward(signum, frame):
        sent = False
        try:
            # Only the direct stage controller handles checkpoint/pause policy.
            os.kill(child.pid, signum)
            sent = True
        except ProcessLookupError:
            pass
        report['signals_forwarded'].append({'signal': signal.Signals(signum).name, 'direct_owned_child': sent})
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, forward)
    maximum_rss = maximum_gpu = samples = 0
    measurement_errors = {'process_tree': 0, 'gpu_sample': 0, 'live_receipt_write': 0}
    gpu_available = False
    device_peaks = {}
    next_sample = started
    usage = None
    try:
        while True:
            now = clock()
            if now >= next_sample:
                try:
                    pids, rss = process_query(child.pid)
                except (OSError, ValueError):
                    measurement_errors['process_tree'] += 1
                    pids, rss = {child.pid}, 0
                maximum_rss = max(maximum_rss, rss)
                try:
                    gpu = gpu_query(pids)
                except (OSError, ValueError):
                    measurement_errors['gpu_sample'] += 1
                    gpu = {'available': False}
                gpu_available |= gpu.get('available') is True
                maximum_gpu = max(maximum_gpu, gpu.get('owned_process_memory_mib', 0))
                for row in gpu.get('owned_gpus', []):
                    key = row['gpu_uuid']
                    before = device_peaks.get(key, {'device_utilization_percent': 0, 'owned_process_memory_mib': 0})
                    device_peaks[key] = dict(row,
                        device_utilization_percent=max(before['device_utilization_percent'], row['device_utilization_percent']),
                        owned_process_memory_mib=max(before['owned_process_memory_mib'], row['owned_process_memory_mib']))
                samples += 1
                next_sample = now + interval
                try:
                    write_json(output / 'resources.live.json', {
                        'schema_version': 'paper3-hpc-live-resource-observation-v1',
                        'state': 'RUNNING_OBSERVATION_NOT_A_COMPLETION_RECEIPT',
                        'observation_id': report['observation_id'], 'stage': stage, 'allocation': assigned,
                        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                        'wall_seconds': max(0, now - started), 'sampled_tree_peak_rss_bytes': maximum_rss,
                        'owned_process_gpu_peak_memory_mib': maximum_gpu, 'samples': samples,
                        'cpu_seconds': None, 'actual_exit_code': None,
                        'signals_forwarded': report['signals_forwarded'], 'measurement_errors': measurement_errors,
                        'checkpoint_saved_claimed': False, 'upload_success_claimed': False})
                except (OSError, ValueError):
                    measurement_errors['live_receipt_write'] += 1
            try:
                pid, status, observed = waiter(child.pid, os.WNOHANG)
            except InterruptedError:
                continue
            if pid:
                usage = observed
                child.returncode = os.waitstatus_to_exitcode(status)
                break
            sleep(min(.25, max(.01, next_sample - clock())))
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    elapsed = max(0, clock() - started)
    peak = max(maximum_rss, int(usage.ru_maxrss * 1024))
    report.update(returned_at_utc=datetime.now(timezone.utc).isoformat(), actual_exit_code=child.returncode,
                  measurements={'wall_seconds': elapsed, 'cpu_user_seconds': usage.ru_utime,
                    'cpu_system_seconds': usage.ru_stime, 'cpu_seconds': usage.ru_utime + usage.ru_stime,
                    'wait4_peak_rss_bytes': int(usage.ru_maxrss * 1024), 'sampled_tree_peak_rss_bytes': maximum_rss,
                    'peak_rss_bytes': peak, 'samples': samples, 'gpu_observation_available': gpu_available,
                    'owned_process_gpu_peak_memory_mib': maximum_gpu, 'owned_gpu_peaks': list(device_peaks.values()),
                    'measurement_errors': measurement_errors})
    report['recommendations'] = recommendations(report, _limits(root))
    # Slurm requeue may preserve its job ID. Keep each attempt immutable while
    # retaining a convenient latest report at resources.json.
    write_json(output / ('resources-' + report['observation_id'] + '.json'), report)
    write_json(output / 'resources.json', report)
    write_json(output / 'resource_recommendations.json', report['recommendations'])
    summary = ('# Paper3 Slurm resource observations\n\n'
               f"Stage: {stage}; actual exit: {child.returncode}; wall: {elapsed:.2f} s.\n\n"
               f"CPU: {report['measurements']['cpu_seconds']:.2f} CPU-s; peak observed RSS: {peak / 1024**3:.3f} GiB; "
               f"owned GPU process memory peak: {maximum_gpu} MiB.\n\n"
               'See resource_recommendations.json. School account quotas are not inferred from node sizes; '
               'scientific thread settings and future job requests were not changed. '
               'A successful wrapper exit is not evidence of scientific completion or an R2 upload.\n')
    path = _safe_path(output / 'resource_recommendations.md')
    path.write_text(summary)
    return report


def collect_sacct(job_id, runner=subprocess.run):
    _need(re.fullmatch(r'[0-9]+(?:_[0-9]+)?', job_id) is not None, 'INVALID_SLURM_JOB_ID')
    fields = ['JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'TotalCPU', 'AllocCPUS', 'MaxRSS', 'ReqMem']
    try:
        result = runner(['sacct', '-j', job_id, '--parsable2', '--noheader', '--units=K',
                         '--format=' + ','.join(fields)], stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=15, check=False)
        rows = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                values = line.rstrip('|').split('|')
                if len(values) == len(fields) and re.fullmatch(r'[A-Za-z0-9_.+-]+', values[0]):
                    rows.append(dict(zip(fields, values)))
        return {'schema_version': 'paper3-slurm-accounting-v1', 'job_id': job_id,
                'status': 'SACCT_ROWS_OBSERVED' if rows else 'SACCT_NOT_YET_AVAILABLE', 'rows': rows,
                'scientific_completion_claimed': False}
    except (OSError, subprocess.TimeoutExpired):
        return {'schema_version': 'paper3-slurm-accounting-v1', 'job_id': job_id,
                'status': 'SACCT_UNAVAILABLE', 'rows': [], 'scientific_completion_claimed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--root', type=Path, required=True)
    launch.add_argument('--stage', choices=STAGES, required=True)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('command', nargs=argparse.REMAINDER)
    summary = sub.add_parser('summarize')
    summary.add_argument('--input', type=Path, required=True)
    accounting = sub.add_parser('accounting')
    accounting.add_argument('--job-id', required=True)
    accounting.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == 'run':
            command = args.command[1:] if args.command[:1] == ['--'] else args.command
            report = run_stage(command, args.root, args.stage, args.output)
            return report['actual_exit_code'] if report['actual_exit_code'] >= 0 else 128 - report['actual_exit_code']
        if args.mode == 'accounting':
            report = collect_sacct(args.job_id)
            write_json(args.output, report)
        else:
            report = recommendations(json.loads(_safe_path(args.input).read_bytes()))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print('Resource observation unavailable: ' + (str(exc) if isinstance(exc, ResourceError) else type(exc).__name__))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
