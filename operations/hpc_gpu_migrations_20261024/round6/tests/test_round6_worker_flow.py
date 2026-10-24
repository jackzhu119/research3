"""Real R6 workers through one allocation, using synthetic EEG only."""
import copy
import os
from pathlib import Path
import subprocess
import sys

from paper3r6.io import digest, put, read, sha
from test_worker import plan_fixture, SEEDS
from test_pan2023 import contract, fixture as pan_fixture


def test_real_uq_allocation_pan_multisession_report_and_resume(tmp_path):
    project = tmp_path / 'project'
    plan, args = plan_fixture(project)
    real_contract = contract()
    signal_sha = digest(real_contract)
    put(project / 'contracts/input_contract.json', real_contract)
    for seed in SEEDS:
        receipt_path = project / f'runs/source_training_v2/f0_receipt_seed-{seed}.json'
        receipt = read(receipt_path)
        receipt['signal_contract_sha256'] = signal_sha
        put(receipt_path, receipt)
        reference_path = project / f'runs/training_campaign_v1/references/seed-{seed}/receipt.json'
        reference = read(reference_path)
        reference.update(signal_contract_sha256=signal_sha, source_f0_receipt_sha256=sha(receipt_path))
        put(reference_path, reference)
    cache = read(project / 'cache.json')
    cache['input_contract_sha256'] = signal_sha
    put(project / 'cache.json', cache)
    tasks = read(project / 'status/tasks.json')
    tasks['W3']['cache_manifest']['sha256'] = sha(project / 'cache.json')
    put(project / 'status/tasks.json', tasks)
    _, pan, _ = pan_fixture(tmp_path / 'external')

    basis, template = plan['tasks']
    basis.update(variant='uq_joint')
    basis['config'].update(variant='uq_joint', head_l2=.1)
    candidates = []
    for variant in ('uq_joint', 'uq_map', 'uq_mean_head', 'uq_linear', 'uq_head'):
        task = copy.deepcopy(template)
        task.update(key=variant, variant=variant, group=variant, budgets=[20], scenarios=['clean'])
        task['config'].update(variant=variant, head_l2=.1)
        candidates.append(task)
    external = copy.deepcopy(candidates[0])
    external.update(key='pan-joint', family='transfer', external_manifest=str(pan / 'manifest.json'),
                    external_dataset='Pan2023', partition='pan2023_published_order', group='pan:joint')
    ea = copy.deepcopy(external)
    ea.update(key='pan-ea', variant='support_EA', group='pan:ea', requires=[])
    ea.pop('basis_key')
    ea['config']['variant'] = 'head_newton'
    plan.update(tasks=[basis, *candidates, external, ea], seeds=[SEEDS[0]],
                comparisons=[['pan:joint', 'pan:ea']], primary_metrics=['nll_balanced', 'ba'],
                primary_partition='pan2023_published_order', primary_expected_participants=14,
                primary_expected_sessions=28, primary_budget=20, primary_scenario='clean')
    put(args.plan, plan)
    entry = Path(__file__).resolve().parents[1] / 'paper3r6/entry.py'
    command = [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'allocation',
               '--plan', args.plan, '--python', sys.executable, '--device', 'cpu',
               '--slots', '3', '--cpus-per-slot', '1', '--poll-seconds', '.02']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PAPER3R6_STOP_REQUESTED', None)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    output = Path(plan['output'])
    errors = lambda: ''.join(p.read_text() for p in (output / 'logs').glob('*.err'))
    assert result.returncode == 0, result.stdout + result.stderr + errors()
    status = read(output / 'allocation_progress.json')
    assert status['complete_tasks'] == 8 and status['allocation_status'] == 'COMPLETE'
    row_path = lambda variant: output / variant / 'evaluations/holdout/B20/clean/S3-1.json'
    point, linear, head = [read(row_path(v)) for v in ('uq_map', 'uq_linear', 'uq_head')]
    assert point['predictions'] == linear['predictions'] == head['predictions']
    mixture = read(row_path('uq_joint'))
    assert mixture['diagnostics']['prediction_family'] == 'bounded_curvature_probability_average'
    assert not mixture['diagnostics']['single_spatial_transform']
    assert mixture['prediction_seconds'] > 0 and mixture['prediction_seconds_per_trial'] > 0
    assert mixture['supportdiagnostics_seconds'] >= 0
    pan_row = read(output / 'pan-joint/evaluations/pan2023_published_order/B20/clean/PAN01-0.json')
    pan_row2 = read(output / 'pan-joint/evaluations/pan2023_published_order/B20/clean/PAN01-1.json')
    assert pan_row['participant_id'] == pan_row2['participant_id']
    assert pan_row['session_id'] != pan_row2['session_id']
    assert not pan_row['temporal_risk_eligible'] and not pan_row['per_run_metrics']
    ea_row = read(output / 'pan-ea/evaluations/pan2023_published_order/B20/clean/PAN01-0.json')
    assert ea_row['variant'] == 'support_EA'
    assert ea_row['reference_fit']['uses_support_labels'] is False
    persisted = {p: p.read_bytes() for p in output.rglob('*') if p.suffix == '.pt' or p.name == 'PAN01-0.json'}
    again = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0, again.stdout + again.stderr
    assert all(p.read_bytes() == b for p, b in persisted.items())
    report = subprocess.run([sys.executable, '-I', '-S', '-B', '-u', str(entry), 'report', '--plan', args.plan],
                            env=env, capture_output=True, text=True, timeout=60)
    assert report.returncode == 0, report.stdout + report.stderr
    summary = read(output / 'reports/report.json')
    assert summary['status'] == 'COMPLETE' and summary['complete_tasks'] == 8
    assert not summary['primary_family_complete']
    assert all(row['p'] is None for row in summary['primary_multimetric_comparisons'])
