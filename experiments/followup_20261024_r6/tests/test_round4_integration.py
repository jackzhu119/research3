"""Production allocation/worker/report integration on synthetic CPU EEG only."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys

from paper3r6.io import put, read
from test_worker import plan_fixture
from test_worker import write_project


def test_one_allocation_runs_crossfit_dag_and_reuses_completed_outputs(tmp_path):
    plan,args = plan_fixture(tmp_path)
    basis,evaluation = plan['tasks']
    basis.update(variant='xfit_stack')
    basis['config'].update(variant='xfit_stack',cv_folds=2)
    evaluation.update(variant='xfit_stack',group='stack',diagnostics_budget=20)
    evaluation['config'].update(variant='xfit_stack',cv_folds=2)
    selector = copy.deepcopy(evaluation)
    selector.update(key='eval-selector',variant='xfit_select_stack',group='selector')
    selector['config'].update(variant='xfit_select_stack',cv_head_grid=[.01,.1],cv_radius_grid=[.05,.15])
    gce = copy.deepcopy(evaluation)
    gce.update(key='eval-gce',variant='gce_profiled',group='gce')
    gce['config'].update(variant='gce_profiled',gce_q=.7)
    plan.update(tasks=[basis,evaluation,selector,gce],seeds=[20261018],comparisons=[['selector','stack']])
    put(args.plan,plan)
    entry=Path(__file__).resolve().parents[1]/'paper3r6/entry.py'
    command=[sys.executable,'-I','-S','-B','-u',str(entry),'allocation','--plan',args.plan,
             '--python',sys.executable,'--device','cpu','--slots','3','--cpus-per-slot','1','--poll-seconds','.02']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('PAPER3R6_STOP_REQUESTED',None)
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
    output=Path(plan['output'])
    assert result.returncode==0,result.stdout+result.stderr+'\n'+''.join(p.read_text() for p in (output/'logs').glob('*.err'))
    status=read(output/'allocation_progress.json')
    assert status['allocation_status']=='COMPLETE' and status['complete_tasks']==4
    assert status['requested_slots']==3 and len(status['slots'])==3
    row=read(output/'eval-test/evaluations/holdout/B20/clean/S3-1.json')
    assert row['diagnostics']['prediction_family']=='convex_probability_pool'
    assert row['diagnostics']['cv']['n_candidate_fits']>0
    assert 'actual_nonlinear_surrogate_margin_rmse' not in row['diagnostics']
    selected=read(output/'eval-selector/evaluations/holdout/B20/clean/S3-1.json')
    assert selected['query_signature']==row['query_signature']
    saved={p:p.read_bytes() for p in output.rglob('*') if p.suffix=='.pt' or p.name=='S3-1.json'}
    again=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert again.returncode==0,again.stdout+again.stderr
    assert all(p.read_bytes()==data for p,data in saved.items())
    report=subprocess.run([sys.executable,'-I','-S','-B','-u',str(entry),'report','--plan',args.plan],
                          env=env,capture_output=True,text=True,timeout=60)
    assert report.returncode==0,report.stdout+report.stderr
    summary=read(output/'reports/report.json')
    assert summary['status']=='COMPLETE' and summary['complete_tasks']==4
    assert summary['cv_selection_frequency']
    assert summary['comparisons'][0]['p'] is None  # one synthetic person


def test_completed_matching_round3_dictionary_is_readonly_and_does_not_restore_rng(tmp_path):
    import torch
    from paper3r6 import worker, core
    from paper3r6.algorithm import Config
    from paper3r6.data import Catalog, load_source
    from paper3r6.io import sha
    from p3work.spatial_geometry import initialize_dictionary
    write_project(tmp_path)
    _,_,inputs=load_source(tmp_path,20261018,'cpu')
    data=Catalog(tmp_path,'meta_train').identity
    old_output=tmp_path/'runs/followup_20261024_r3'
    task={'key':'basis-matched_sgd-r4-seed20261018','family':'basis','seed':20261018,
          'config':{'variant':'sgd10','rank':4,'radius':.15,'head_l2':.1},'updates':1000}
    old={'output':str(old_output),'recipe_sha256':'d'*64,'tasks':[task],'updates':1000,'validate_every':50}
    identity=worker._identity(old,task,inputs,data)
    folder=old_output/task['key'];folder.mkdir(parents=True)
    selected=folder/'selected-basis.pt'
    b=initialize_dictionary(4,73)
    torch.save({'identity':identity,'b':b},selected)
    put(old_output/'plan.json',old)
    put(folder/'result.json',{'status':'COMPLETE','outer_updates':1000,'task':task,'identity':identity,
        'source_identity':inputs,'data_identity':data,'selection':{'participant_mean_ba':.5,
        'participant_mean_bce':.7,'outer_update':1000},'selected_checkpoint':{'path':str(selected),'sha256':sha(selected)}})
    saved={p:p.read_bytes() for p in old_output.rglob('*') if p.is_file()}
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    answer=worker.try_reuse_round3(tmp_path,Config(variant='sgd10',head_l2=.1),20261018,inputs,data)
    assert answer is not None
    torch.testing.assert_close(answer[0],b)
    assert torch.equal(before,torch.get_rng_state())
    assert all(p.read_bytes()==value for p,value in saved.items())
    assert worker.try_reuse_round3(tmp_path,Config(variant='sgd10',head_l2=.001),20261018,inputs,data) is None
