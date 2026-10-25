"""Round 8 scheduler only; three independent single-GPU jobs, no migration gate."""
import argparse
import fcntl
import getpass
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from three_jobs import plan_context, result_complete, controller_step, atomic, emit


def prepare_context(plan_path, runtime=None, *, python=None, package=None, account=None):
    plan_path, plan, output, runtime, plan_sha = plan_context(plan_path, runtime)
    root = Path(plan['root']).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Existing project root not found: '+str(root))
    package = Path(package or PACKAGE.parent).expanduser().resolve()
    if not (package/'paper3r8/entry.py').is_file():
        raise ValueError('Round 8 scientific entry.py missing from installed package')
    python = Path(python or sys.executable).expanduser().absolute()
    if not python.is_file():
        raise ValueError('Existing Python interpreter missing: '+str(python))
    command = [str(python), '-I', '-S', '-B', '-u', str(PACKAGE/'three_jobs.py'), 'worker',
               '--plan', str(plan_path), '--runtime', str(runtime), '--round', '8',
               '--slot', '{slot}', '--package', str(package), '--python', str(python)]
    launch = {'round': 8, 'worker_command': command, 'partition': 'hpc_gpu', 'cpus': 16,
              'memories': ['64G']*3, 'gres': 'gpu:1', 'submit_hold': False, 'exclude': '',
              'account': account or getpass.getuser(), 'user': getpass.getuser(),
              'campaign_identity': plan['recipe_sha256'], 'python': str(python),
              'package': str(package), 'time_limit': '5-03:00:00', 'poll_seconds': 30,
              'root': str(root), 'plan': str(plan_path), 'runtime': str(runtime),
              'original_plan_sha256': plan_sha}
    complete = sum(result_complete(output, task, plan['recipe_sha256']) for task in plan['tasks'])
    return {'root': root, 'output': output, 'runtime': runtime, 'plan_path': plan_path,
            'plan': plan, 'launch': launch, 'total_tasks': len(plan['tasks']),
            'complete_tasks': complete, 'round': 8}


def controller_loop(context, *, runner=None, clock=None, stop=None, once=False):
    runtime = context['runtime']
    stop = stop or threading.Event()
    runner = runner or subprocess.run
    with (runtime/'controller.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('ROUND_8_EXISTING_CONTROLLER_REUSED')
            return 0
        atomic(runtime/'launch.json', context['launch'])
        emit('ROUND_8_CONTROLLER_STARTED', runtime=str(runtime),
             output=str(context['output']), jobs=3, partition='hpc_gpu')
        while not stop.is_set():
            try:
                summary = controller_step(context['plan_path'], context['launch'], runtime,
                                          runner=runner, clock=clock)
                progress = summary.get('progress', {})
                complete = (progress.get('total_tasks', 0)>0 and
                            progress.get('complete_tasks') == progress.get('total_tasks'))
                summary['status'] = 'COMPLETE' if complete else summary.get('controller_status', 'RUNNING')
                atomic(runtime/'controller-status.json', summary)
                emit('ROUND_8_CONTROLLER_PROGRESS', status=summary['status'],
                     complete_tasks=progress.get('complete_tasks'), total_tasks=progress.get('total_tasks'),
                     jobs=summary.get('jobs', []))
                if complete:
                    # Running workers observe complete receipts and exit themselves.
                    # Remove only our exact owned pending jobs so no finished work stays queued.
                    for job in summary.get('jobs', []):
                        if (job.get('status') == 'ACKNOWLEDGED' and job.get('slurm_state') == 'PENDING'
                                and str(job.get('name', '')).startswith(summary['marker']+'-s')):
                            answer = runner(['scancel', '--state=PENDING', str(job['job_id'])],
                                            capture_output=True, text=True, timeout=25)
                            emit('ROUND_8_UNUSED_PENDING_WORKER_RELEASED', job_id=job['job_id'],
                                 returncode=answer.returncode)
                    if summary.get('controller_status') == 'FINISHED':
                        return 0
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                emit('ROUND_8_CONTROLLER_RETRY', error_type=type(error).__name__, reason=str(error)[:600])
            if once:
                return 0
            stop.wait(float(context['launch']['poll_seconds']))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--runtime')
    parser.add_argument('--package', default=str(PACKAGE.parent))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--account')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    context = prepare_context(args.plan, args.runtime, python=args.python,
                              package=args.package, account=args.account)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    return controller_loop(context, stop=stop, once=args.once)


if __name__ == '__main__':
    raise SystemExit(main())
