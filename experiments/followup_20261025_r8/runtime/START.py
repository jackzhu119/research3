#!/usr/bin/env python3
"""Round 8: prepare three single-GPU workers and detached progress synchronization."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from coordinator import prepare_context
from three_jobs import atomic, read, emit


def start_process(context, kind, credentials=None, popen=subprocess.Popen):
    runtime, launch = context['runtime'], context['launch']
    script = PACKAGE / ('coordinator.py' if kind == 'controller' else 'monitor.py')
    receipt_path = runtime / (kind+'-process.json')
    prior = read(receipt_path, {}) or {}
    with (runtime/(kind+'.lock')).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'kind': kind, 'reused': True, 'pid': prior.get('pid'),
                    'log': str(runtime/(kind+'.log'))}
    command = [launch['python'], '-I', '-S', '-B', '-u', str(script)]
    if kind == 'controller':
        command += ['--plan', str(context['plan_path']), '--runtime', str(runtime),
                    '--package', launch['package'], '--python', launch['python'],
                    '--account', launch['account']]
    else:
        command += ['--round', '8', '--root', str(context['root']), '--operational', str(runtime)]
        if credentials:
            command += ['--credentials', str(Path(credentials).expanduser())]
    log = runtime / (kind+'.log')
    with log.open('ab', buffering=0) as handle:
        child = popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=handle,
                      start_new_session=True, close_fds=True)
    receipt = {'pid': child.pid, 'kind': kind, 'script': str(script),
               'runtime': str(runtime), 'started_at': time.time()}
    atomic(receipt_path, receipt)
    return {**receipt, 'reused': False, 'log': str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--runtime')
    parser.add_argument('--package', default=str(PACKAGE.parent))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--account')
    parser.add_argument('--credentials')
    parser.add_argument('--submit', '--execute', dest='execute', action='store_true')
    args = parser.parse_args(argv)
    context = prepare_context(args.plan, args.runtime, python=args.python,
                              package=args.package, account=args.account)
    runtime = context['runtime']
    emit('ROUND_8_THREE_JOBS_PREPARED', round=8, jobs=3, gpus_total=3,
         partition='hpc_gpu', cpus_per_job=16, cpus_total=48,
         memory_per_job='64G', memory_total='192G',
         tasks=context['total_tasks'], complete_tasks=context['complete_tasks'],
         output=str(context['output']), plan=str(context['plan_path']),
         full_test=False, full_data_rescan=False, package_install=False,
         github_branch='paper3-r8-progress',
         github_status='monitoring/paper3-round8/three-jobs-status.json')
    atomic(runtime/'launch.json', context['launch'])
    if not args.execute:
        return 0
    with (runtime/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        own_stop = runtime/'STOP'
        if own_stop.exists():
            history = runtime/'startup-history'
            history.mkdir(exist_ok=True)
            os.replace(own_stop, history/('STOP-'+str(time.time_ns())))
        processes = [start_process(context, 'controller'),
                     start_process(context, 'monitor', args.credentials)]
    emit('ROUND_8_BACKGROUND_PROCESSES_STARTED', round=8, processes=processes,
         submission_receipts=str(runtime/'scheduler-submissions'),
         progress=str(runtime/'progress.json'),
         note='Slurm job IDs appear in submission receipts; node stdout confirms execution.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
