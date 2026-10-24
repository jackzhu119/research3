"""Production isolated entry across real synthetic fit, evaluation, report and monitor."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from paper3r5.io import put, read
from paper3r5.monitor import cycle
from test_worker import plan_fixture


def test_isolated_fit_evaluation_reporting_and_nonblocking_sync(tmp_path):
    plan, args = plan_fixture(tmp_path)
    plan['seeds'] = [20261018]
    first = plan['tasks'][1]
    first.update(group='candidate', diagnostics_budget=20)
    second = copy.deepcopy(first)
    second.update(key='eval-baseline', group='baseline', variant='head_newton')
    second['config']['variant'] = 'head_newton'
    plan['tasks'].append(second)
    plan['comparisons'] = [['candidate', 'baseline']]
    put(args.plan, plan)
    entry = Path(__file__).resolve().parents[1] / 'paper3r5/entry.py'
    command = [sys.executable, '-I', '-S', '-B', '-u', str(entry)]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('PAPER3R5_STOP_REQUESTED', None)
    for index in range(3):
        result = subprocess.run(command + ['worker', '--plan', args.plan, '--task', str(index), '--device', 'cpu'],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert read(Path(plan['output']) / plan['tasks'][index]['key'] / 'result.json')['status'] == 'COMPLETE'
    saved = {p: p.read_bytes() for p in Path(plan['output']).rglob('*.pt')}
    reused = subprocess.run(command + ['worker', '--plan', args.plan, '--task', '0', '--device', 'cpu'],
                            env=env, capture_output=True, text=True, timeout=90)
    assert reused.returncode == 0 and 'COMPLETED_TASK_REUSED' in reused.stdout
    assert all(p.read_bytes() == value for p, value in saved.items())
    report = subprocess.run(command + ['report', '--plan', args.plan],
                            env=env, capture_output=True, text=True, timeout=90)
    assert report.returncode == 0, report.stdout + report.stderr
    summary = read(Path(plan['output']) / 'reports/report.json')
    assert summary['status'] == 'COMPLETE', summary['issues']
    assert summary['complete_tasks'] == 3
    # One synthetic holdout person is enough to exercise the pipeline, but
    # must never yield a significance claim or a completed inference family.
    assert not summary['primary_family_complete']
    assert summary['comparisons'][0]['n_participants'] == 1
    assert summary['comparisons'][0]['p'] is None
    class FailedSync:
        def ensure_branch(self):
            raise OSError('synthetic GitHub connection timeout')
    _, _, finished = cycle(plan, None, client=FailedSync())
    assert not finished
    local = read(Path(plan['output']) / 'status.json')
    assert local['complete_tasks'] == 3 and local['report_status'] == 'COMPLETE'
    assert all(p.read_bytes() == value for p, value in saved.items())
