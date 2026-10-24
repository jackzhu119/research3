"""Production R7 allocation on synthetic EEG; not scientific accuracy data."""
import copy
import os
from pathlib import Path
import subprocess
import sys

from paper3r7.io import digest, put, read, sha
from test_worker import plan_fixture, SEEDS
from test_pan2023 import contract, fixture as pan_fixture


def test_actual_spatiotemporal_risk_head_pan_covariance_and_resume(tmp_path):
    root=tmp_path/'project';plan,args=plan_fixture(root)
    signal_sha=digest(contract());put(root/'contracts/input_contract.json',contract())
    for seed in SEEDS:
        p=root/f'runs/source_training_v2/f0_receipt_seed-{seed}.json'
        receipt=read(p);receipt['signal_contract_sha256']=signal_sha;put(p,receipt)
        q=root/f'runs/training_campaign_v1/references/seed-{seed}/receipt.json'
        reference=read(q);reference.update(signal_contract_sha256=signal_sha,source_f0_receipt_sha256=sha(p));put(q,reference)
    cache=read(root/'cache.json');cache['input_contract_sha256']=signal_sha;put(root/'cache.json',cache)
    tasks=read(root/'status/tasks.json');tasks['W3']['cache_manifest']['sha256']=sha(root/'cache.json');put(root/'status/tasks.json',tasks)
    _,pan,_=pan_fixture(tmp_path/'external')
    basis,template=plan['tasks']
    config={'variant':'r7_refined','mode':'spatiotemporal','objective':'relative_dro',
            'head_policy':'budget','head_l2':.1,'rank':4,'tangent_steps':1,'inner_steps':1}
    basis.update(variant='r7_refined',config=config)
    st=copy.deepcopy(template);st.update(key='st',variant='r7_refined',config=config,
        group='candidate',budgets=[20],scenarios=['clean','time_cov_plus35','label_noise10'])
    static=copy.deepcopy(st);static.update(key='static',group='baseline',scenarios=['clean'])
    static['config']=dict(config,mode='static',objective='ce',head_policy='full')
    cov=copy.deepcopy(st);cov.update(key='cov',family='reference',group='cov',variant='covariance_only',requires=[],scenarios=['clean','time_cov_plus35'])
    cov.pop('basis_key');cov['config']={'variant':'head_newton','rank':4,'head_l2':.1}
    global_time=copy.deepcopy(cov);global_time.update(key='global',group='global',variant='r7_refined')
    global_time['config']=dict(config,mode='global_time',objective='ce',head_policy='full')
    transfer=copy.deepcopy(st);transfer.update(key='pan-st',group='pan:st',family='transfer',
        external_manifest=str(pan/'manifest.json'),external_dataset='Pan2023',partition='pan2023_published_order',scenarios=['clean'])
    cov_external=copy.deepcopy(cov);cov_external.update(key='pan-cov',group='pan:cov',family='transfer',
        external_manifest=str(pan/'manifest.json'),external_dataset='Pan2023',partition='pan2023_published_order')
    plan.update(tasks=[basis,st,static,cov,global_time,transfer,cov_external],
        seeds=[SEEDS[0]],comparisons=[['pan:st','pan:cov']],primary_metrics=['nll_balanced','ba'],
        primary_partition='pan2023_published_order',primary_expected_participants=14,primary_expected_sessions=28)
    put(args.plan,plan)
    entry=Path(__file__).resolve().parents[1]/'paper3r7/entry.py'
    command=[sys.executable,'-I','-S','-B','-u',str(entry),'allocation','--plan',args.plan,
        '--python',sys.executable,'--device','cpu','--slots','3','--cpus-per-slot','1','--poll-seconds','.02']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('PAPER3R7_STOP_REQUESTED',None)
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
    output=Path(plan['output'])
    errors=lambda:''.join(p.read_text() for p in (output/'logs').glob('*.err'))
    assert result.returncode==0,result.stdout+result.stderr+errors()
    assert read(output/'allocation_progress.json')['complete_tasks']==7
    row=read(output/'st/evaluations/holdout/B20/clean/S3-1.json')
    d=row['diagnostics']
    assert d['prediction_family']=='finite_window_spatiotemporal'
    assert abs(d['support_white_energy_relative_change'])<1e-10
    assert d['effective_head_rank']<=14  # one paid position in the fixture is invalid.
    assert 'support_band_outside_8_30_after' in d and 'head_absorbed_spatial_fraction' in d
    assert d['risk_scope']=='same_finite_labeled_support_reweighting_only'
    changed=read(output/'cov/evaluations/holdout/B20/time_cov_plus35/S3-1.json')
    assert changed['variant']=='covariance_only'
    assert changed['diagnostics']['time_stress_exact_covariance_prediction_null_expected']
    assert changed['diagnostics']['time_stress_prediction_max_abs_probability_change']<1e-8
    assert changed['perturbation']['query_covariance_relative_error']<1e-12
    global_row=read(output/'global/evaluations/holdout/B20/clean/S3-1.json')
    assert global_row['diagnostics']['global_time_active_parameter_count']==1
    first=read(output/'pan-st/evaluations/pan2023_published_order/B20/clean/PAN01-0.json')
    second=read(output/'pan-st/evaluations/pan2023_published_order/B20/clean/PAN01-1.json')
    assert first['participant_id']==second['participant_id'] and first['session_id']!=second['session_id']
    saved={p:p.read_bytes() for p in output.rglob('*') if p.suffix=='.pt' or p.name=='PAN01-0.json'}
    again=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert again.returncode==0,again.stdout+again.stderr
    assert all(p.read_bytes()==v for p,v in saved.items())
    report=subprocess.run([sys.executable,'-I','-S','-B','-u',str(entry),'report','--plan',args.plan],
                           env=env,capture_output=True,text=True,timeout=60)
    assert report.returncode==0,report.stdout+report.stderr
    summary=read(output/'reports/report.json')
    assert summary['complete_tasks']==7 and summary['status']=='COMPLETE'
    assert not summary['primary_family_complete']


def test_worker_identity_uses_input_contents_instead_of_installation_path():
    from paper3r7.worker import _identity
    plan={'recipe_sha256':'unchanged'}
    first={'family':'transfer','external_manifest':'/old/install/data.json','basis_output':'/old/basis'}
    second=dict(first,external_manifest='/new/install/data.json',basis_output='/new/basis')
    data={'manifest_sha256':'same-content'}
    dep={'key':'basis','identity':'same-producer','sha256':'same-basis','basis_output':'/old/basis'}
    moved=dict(dep,basis_output='/new/basis')
    assert _identity(plan,first,{},data,dep)==_identity(plan,second,{},data,moved)
    assert _identity(plan,first,{},data,dep)!=_identity(plan,second,{},dict(data,manifest_sha256='changed'),moved)
