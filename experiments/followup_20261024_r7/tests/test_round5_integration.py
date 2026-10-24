"""Actual new-geometry workers in one allocation, on synthetic EEG only."""
from pathlib import Path
import copy
import os
import subprocess
import sys

from paper3r7.io import put, read
from test_worker import plan_fixture


def test_single_allocation_nonlinear_fit_coupled_shift_probe_report_and_resume(tmp_path):
    plan,args=plan_fixture(tmp_path)
    basis,evaluation=plan['tasks']
    basis.update(variant='geometry_refined')
    basis['config'].update(variant='geometry_refined',geometry='single',inner_steps=1)
    evaluation.update(variant='geometry_refined',group='candidate',diagnostics_budget=20,
                      scenarios=['clean','coupled_ks15','coupled_null15'])
    evaluation['config'].update(variant='geometry_refined',geometry='symmetric',inner_steps=1)
    baseline=copy.deepcopy(evaluation)
    baseline.update(key='eval-single',group='baseline')
    baseline['config']['geometry']='single'
    probe=copy.deepcopy(evaluation)
    probe.update(key='probe-bch2',group='probe:bch2',variant='geometry_tangent',scenarios=['clean'],budgets=[20])
    probe['config'].update(geometry='bch2',variant='geometry_tangent')
    plan.update(tasks=[basis,evaluation,baseline,probe],seeds=[20261018],comparisons=[['candidate','baseline']])
    put(args.plan,plan)
    entry=Path(__file__).resolve().parents[1]/'paper3r7/entry.py'
    command=[sys.executable,'-I','-S','-B','-u',str(entry),'allocation','--plan',args.plan,
             '--python',sys.executable,'--device','cpu','--slots','3','--cpus-per-slot','1','--poll-seconds','.02']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('PAPER3R7_STOP_REQUESTED',None)
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    output=Path(plan['output'])
    errors=lambda: ''.join(p.read_text() for p in (output/'logs').glob('*.err'))
    assert result.returncode==0,result.stdout+result.stderr+errors()
    status=read(output/'allocation_progress.json')
    assert status['allocation_status']=='COMPLETE' and status['complete_tasks']==4
    assert status['requested_slots']==3
    row=read(output/'eval-test/evaluations/holdout/B20/clean/S3-1.json')
    diagnostics=row['diagnostics']
    assert diagnostics['prediction_family']=='nonlinear_geometry'
    assert diagnostics['geometry']=='symmetric' and diagnostics['actual_fit_steps']==1
    assert diagnostics['inverse_consistency_defect']<1e-10
    assert diagnostics['final_actual_support_loss']<=diagnostics['initial_actual_support_loss']+1e-12
    assert 'actual_nonlinear_surrogate_margin_rmse' not in diagnostics
    shifted=read(output/'eval-test/evaluations/holdout/B20/coupled_ks15/S3-1.json')
    assert shifted['perturbation']['support_changed'] and shifted['perturbation']['query_changed']
    assert shifted['query_signature']==row['query_signature']
    assert not (output/'eval-test/evaluations/holdout/B4/coupled_ks15').exists()
    probe_row=read(output/'probe-bch2/evaluations/holdout/B20/clean/S3-1.json')
    assert probe_row['diagnostics']['optimizer_scope']=='common_identity_tangent_probe'
    assert not probe_row['diagnostics']['actual_nonlinear_optimization_performed']
    saved={p:p.read_bytes() for p in output.rglob('*') if p.suffix=='.pt' or p.name=='S3-1.json'}
    again=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert again.returncode==0,again.stdout+again.stderr
    assert all(p.read_bytes()==value for p,value in saved.items())
    report=subprocess.run([sys.executable,'-I','-S','-B','-u',str(entry),'report','--plan',args.plan],
                          env=env,capture_output=True,text=True,timeout=60)
    assert report.returncode==0,report.stdout+report.stderr
    summary=read(output/'reports/report.json')
    assert summary['status']=='COMPLETE' and summary['complete_tasks']==4
    assert summary['comparisons'][0]['p'] is None  # one synthetic participant
    assert any(row['diagnostic']=='inverse_consistency_defect' for row in summary['diagnostics'])
