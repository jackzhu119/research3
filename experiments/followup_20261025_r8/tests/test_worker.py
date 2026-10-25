"""CPU integration checks using synthetic EEG, not scientific accuracy results."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import os
import signal
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch

from paper3r8 import worker
from paper3r8.transport import TransportConfig, init_meta_parameters
from paper3r7.io import digest,put,read,sha
from paper3r7.data import Catalog,load_source
from p3work.f0_bridge import CompactEEGNet

SEEDS=(20261018,20261019,20261020)

@pytest.fixture(autouse=True)
def process_settings(monkeypatch):
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    threads=torch.get_num_threads()
    deterministic=torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    monkeypatch.delenv('PAPER3R8_STOP_REQUESTED',raising=False)
    torch.set_num_threads(1)
    yield
    for s,h in handlers.items():signal.signal(s,h)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)

def write_project(root):
    """Three seed-specific frozen source receipts and disjoint participant roles."""
    contract = {"n_times": 128, "fixture": "SYNTHETIC_SOFTWARE_TEST_ONLY"}
    put(root / "contracts/input_contract.json", contract)
    contract_sha = digest(contract)
    def desc(path):
        return {"path": str(path.relative_to(root)), "sha256": sha(path)}
    np.save(root / "reference.npy", np.eye(10) / 10)
    for seed in SEEDS:
        torch.manual_seed(seed)
        path = root / f"source-{seed}.pt"
        torch.save({"model": CompactEEGNet(11, 128).state_dict(), "config": {"dropout": .25}}, path)
        receipt = root / f"runs/source_training_v2/f0_receipt_seed-{seed}.json"
        put(receipt, {"status": "SOURCE_F0_COMPLETE", "pipeline_seed": seed,
            "checkpoint": desc(path), "signal_contract_sha256": contract_sha})
        put(root / f"runs/training_campaign_v1/references/seed-{seed}/receipt.json", {
            "status": "SOURCE_REFERENCE_COMPLETE", "source_f0_receipt_sha256": sha(receipt),
            "signal_contract_sha256": contract_sha, "arrays": {"reference": desc(root / "reference.npy")}})
    roles = dict(zip(("meta_train", "meta_validation", "internal_holdout"), ("S1", "S2", "S3")))
    put(root / "split.json", {"partitions": {key: [person] for key, person in roles.items()}, "split_sha256": "b" * 64})
    sessions = []
    rng = np.random.default_rng(18)
    for role, person in roles.items():
        x = rng.normal(size=(100, 11, 128)).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        x -= x.mean(2, keepdims=True)
        np.save(root / f"{person}.npy", x)
        rows = [{"participant_id": person, "session_id": "1", "tasknumber": 1, "label": i % 2,
            "trial_uid": f"{person}-{i}", "event_uid": f"{person}-{i}", "original_trial_index": i,
            "physical_run_id": str(i // 20), "physical_run_complete_verified": True,
            "valid": i != 0, "cache_index": i if i else None} for i in range(100)]
        put(root / f"{person}.json", rows)
        sessions.append({"participant_id": person, "session_id": "1", "partition": role,
            "acquisition_order_verified": True, "X": desc(root / f"{person}.npy"),
            "events": desc(root / f"{person}.json")})
    put(root / "cache.json", {"status": "STIEGER_CACHE_COMPLETE", "participant_split": desc(root / "split.json"),
        "input_contract_sha256": contract_sha, "sessions": sessions})
    put(root / "status/tasks.json", {"W3": {"cache_manifest": desc(root / "cache.json")}})


def fixture(tmp_path, mode='global'):
    root=tmp_path/'project';root.mkdir();write_project(root)
    config={'kind':'bures','parameterization':mode,'alpha':.5,'shrinkage':.05}
    train={'key':'train-test','family':'train','group':'train','seed':SEEDS[0],
           'config':config,'requires':[],'updates':2,'validate_every':1,
           'checkpoint_every':1,'episodes_per_update':1,'query_batch_size':17}
    evaluation={'key':'eval-test','family':'core','group':'candidate','seed':SEEDS[0],
        'config':config,'meta_key':'train-test','requires':['train-test'],
        'budgets':[20,4],'scenarios':['clean','label_noise10'],'partition':'holdout'}
    plan={'root':str(root),'output':str(root/'runs/r8'),'recipe_sha256':'test-r8-v1',
          'tasks':[train,evaluation]}
    put(root/'plan-r8.json',plan)
    return plan,Namespace(plan=str(root/'plan-r8.json'),task=0,device='cpu')


def test_real_train_evaluation_and_idempotent_resume(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    assert worker.run(args)==0
    folder=Path(plan['output'])/'train-test'
    result=read(folder/'result.json')
    assert result['outer_updates']==2
    assert result['holdout_used_for_selection'] is False
    assert result['selection_metric']=='participant_mean_nll'
    assert not (folder/'evaluations/holdout').exists()
    sizes=[];original=worker.predict_transport
    def track(source,geometry,fitted,x):
        sizes.append(len(x));return original(source,geometry,fitted,x)
    monkeypatch.setattr(worker,'predict_transport',track)
    args.task=1;assert worker.run(args)==0
    assert max(sizes)<=64
    output=Path(plan['output'])/'eval-test'
    a=read(output/'evaluations/holdout/B20/clean/S3-1.json')
    b=read(output/'evaluations/holdout/B4/clean/S3-1.json')
    assert a['query_signature']==b['query_signature']
    assert len(a['labels'])==80 and a['support_acquired_trials']==20
    assert a['support_valid_trials']==19
    assert a['ba']-a['f0_ba']==pytest.approx(a['corrected_error_rate']-a['negative_flip_rate'])
    assert a['fallback_reason'] is None
    assert read(output/'evaluations/holdout/B20/clean/summary.json')['expected_sessions'][0]['query_signature']==a['query_signature']
    files={p:p.read_bytes() for p in output.rglob('*.json') if p.name!='progress.json'}
    assert worker.run(args)==0
    assert all(p.read_bytes()==data for p,data in files.items())


def test_evidence_meta_training_gradient_and_chunk_invariance(tmp_path):
    plan,args=fixture(tmp_path,'evidence')
    root=plan['root'];source,geometry,_=load_source(root,SEEDS[0],'cpu')
    catalog=Catalog(root,'meta_train')
    cfg=TransportConfig(**plan['tasks'][0]['config'])
    original=init_meta_parameters(cfg,'cpu')
    outputs=[]
    for batch in (17,80):
        params=original.detach().clone().requires_grad_(True)
        opt=torch.optim.Adam([params],lr=.01)
        metrics=worker.train_step(source,geometry,catalog,cfg,params,opt,
            seed=SEEDS[0],update=0,episodes_per_update=1,batch_size=batch)
        assert metrics['meta_gradient_norm']>0
        outputs.append((params.detach(),metrics))
    assert torch.allclose(outputs[0][0],outputs[1][0],atol=2e-6,rtol=2e-5)
    assert outputs[0][1]['outer_query_nll']==pytest.approx(outputs[1][1]['outer_query_nll'],abs=1e-7)


def test_pause_checkpoint_and_exact_continuation(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    original=worker.train_step
    calls=[]
    def pause_after_step(*a,**kw):
        answer=original(*a,**kw)
        calls.append(kw['update'])
        os.kill(os.getpid(),signal.SIGUSR1)
        return answer
    monkeypatch.setattr(worker,'train_step',pause_after_step)
    assert worker.run(args)==75
    folder=Path(plan['output'])/'train-test'
    assert read(folder/'latest.json')['outer_update']==1
    assert not (folder/'result.json').exists()
    monkeypatch.setattr(worker,'train_step',original)
    assert worker.run(args)==0
    assert read(folder/'result.json')['outer_updates']==2
    assert calls==[0]
    state=worker.load_checkpoint(folder,read(folder/'result.json')['identity'],restore_rng=False)
    assert int(next(iter(state['optimizer']['state'].values()))['step'])==2
    original_parameters=state['parameters'].clone()
    plan['output']=str(Path(plan['output']).with_name('uninterrupted'))
    put(args.plan,plan)
    assert worker.run(args)==0
    complete_folder=Path(plan['output'])/'train-test'
    complete_result=read(complete_folder/'result.json')
    complete=worker.load_checkpoint(complete_folder,complete_result['identity'],restore_rng=False)
    assert torch.equal(original_parameters,complete['parameters'])


def test_corrupt_latest_recovers_previous_generation(tmp_path):
    state={'identity':'same','update':1,'parameters':torch.ones(2)}
    worker.save_checkpoint(tmp_path,state)
    worker.save_checkpoint(tmp_path,dict(state,update=2))
    latest=read(tmp_path/'latest.json')
    (tmp_path/latest['path']).write_bytes(b'corrupt')
    recovered=worker.load_checkpoint(tmp_path,'same',restore_rng=False)
    assert recovered['update']==1
    assert read(tmp_path/'latest.json')['outer_update']==1
    with pytest.raises(ValueError):worker.load_checkpoint(tmp_path,'other',restore_rng=False)


def test_class_reference_only_train_prefix_and_fiber(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    ref={'key':'classref','family':'class_reference','group':'classref','seed':SEEDS[0],
         'requires':[],'config':{'kind':'identity'}}
    task={'key':'fiber','family':'core','group':'fiber','seed':SEEDS[0],
        'requires':['classref'],'reference_key':'classref','config':{'kind':'fiber','alpha':1.},
        'budgets':[20],'scenarios':['clean']}
    plan['tasks']=[ref,task];put(args.plan,plan)
    original=Catalog.materialize
    def forbidden(*a,**kw):raise AssertionError('Class reference must read only acquired prefix')
    monkeypatch.setattr(Catalog,'materialize',forbidden)
    assert worker.run(args)==0
    result=read(Path(plan['output'])/'classref/result.json')
    assert result['reference_participants']==1 and result['reference_sessions']==1
    assert result['reference_scope']=='meta_train_first20_acquired_prefix_only'
    monkeypatch.setattr(Catalog,'materialize',original)
    args.task=1;assert worker.run(args)==0
    row=read(Path(plan['output'])/'fiber/evaluations/holdout/B20/clean/S3-1.json')
    assert row['fallback_reason'] is None


def test_head_step_controls_and_reference_never_train(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    plan['tasks']=[{'key':v,'family':'reference','group':v,'seed':SEEDS[0],
         'variant':v,'config':{'kind':'identity'},'requires':[],
         'budgets':[20],'scenarios':['clean']} for v in ('source_frozen','support_EA','head_sgd10','head_sgd100')]
    put(args.plan,plan)
    monkeypatch.setattr(worker,'train_step',lambda *a,**kw:pytest.fail('Reference entered meta-training'))
    for i,t in enumerate(plan['tasks']):
        args.task=i;assert worker.run(args)==0
        row=read(Path(plan['output'])/t['key']/'evaluations/holdout/B20/clean/S3-1.json')
        if t['variant'].startswith('head_sgd'):
            assert row['diagnostics']['optimizer_updates']==int(t['variant'][8:])


def test_isolated_entry_executes_existing_environment(tmp_path):
    plan,args=fixture(tmp_path)
    plan['tasks']=[{'key':'f0','family':'reference','group':'f0','seed':SEEDS[0],
        'variant':'source_frozen','config':{'kind':'identity'},'requires':[],
        'budgets':[20],'scenarios':['clean']}]
    put(args.plan,plan)
    entry=Path(__file__).resolve().parents[1]/'paper3r8/entry.py'
    # Match the school layout: executable and explicit venv site-packages.
    # This executor keeps torch in user-site, which isolated Python excludes.
    venv=tmp_path/'venv';(venv/'bin').mkdir(parents=True)
    interpreter=venv/'bin/python';interpreter.symlink_to(sys.executable)
    lib=venv/'lib'/('python%d.%d'%sys.version_info[:2]);lib.mkdir(parents=True)
    (lib/'site-packages').symlink_to(Path(torch.__file__).resolve().parent.parent,target_is_directory=True)
    result=subprocess.run([str(interpreter),'-I','-S','-B','-u',str(entry),'worker',
        '--plan',args.plan,'--task','0','--device','cpu'],capture_output=True,text=True,timeout=60,
        env=dict(os.environ,SLURM_CPUS_PER_TASK='1'))
    assert result.returncode==0,result.stdout+result.stderr
    assert 'R8_SCIENCE_PROCESS_STARTED' in result.stdout
    assert read(Path(plan['output'])/'f0/result.json')['status']=='COMPLETE'


def test_class_reference_equal_people_not_session_counts(tmp_path):
    from types import SimpleNamespace
    q=torch.zeros(10,11,dtype=torch.float64);q[:,:10]=torch.eye(10,dtype=torch.float64)
    geometry=SimpleNamespace(q=q,rref_invsqrt=torch.eye(10,dtype=torch.float64))
    rows={};arrays={};plans=[]
    def add(person,session,axis):
        name=person+'-'+session
        x=np.zeros((20,11,8),dtype=np.float64)
        for i in range(20):x[i,axis if i%2 else 0,:]=np.arange(1,9)
        ids=[name+'-'+str(i) for i in range(20)]
        events={u:{'label':i%2,'valid':True,'cache_index':i} for i,u in enumerate(ids)}
        p=SimpleNamespace(participant_id=person,session_id=session,rows=events,
                          audit={'acquired_prefix_uids':ids})
        plans.append(p);arrays[name]=x
    add('A','1',1);add('A','2',1);add('B','1',2)
    cat=SimpleNamespace(plans=plans,signal_sha='f'*64,array=lambda p:arrays[p.participant_id+'-'+p.session_id])
    result=worker.class_reference(None,geometry,cat,tmp_path,threading.Event(),{},'identity')
    value=torch.load(result['selected_checkpoint']['path'],weights_only=True)['class_reference']
    expected=torch.diag(torch.tensor([-1.,.5,.5,0,0,0,0,0,0,0],dtype=torch.float64))
    assert torch.allclose(value,expected)
    assert result['reference_participants']==2 and result['reference_sessions']==3


def test_label_permutation_preserves_signal_count_and_query(tmp_path):
    plan,args=fixture(tmp_path)
    source,geometry,_=load_source(plan['root'],SEEDS[0],'cpu')
    cat=Catalog(plan['root'],'internal_holdout');session=cat.plans[0]
    ep=cat.materialize(session,'cpu',budget=20)
    changed,meta=worker.perturb(ep,geometry,session,'support_label_permutation')
    repeated,_=worker.perturb(ep,geometry,session,'support_label_permutation')
    assert torch.equal(torch.bincount(changed.support.y),torch.bincount(ep.support.y))
    assert changed.support.x is ep.support.x
    assert changed.query_x is ep.query_x and changed.query_y is ep.query_y
    assert changed.support.event_uids==ep.support.event_uids
    assert changed.support.acquired_event_uids==ep.support.acquired_event_uids
    assert torch.equal(changed.support.y,repeated.support.y)
    assert not torch.equal(changed.support.y,ep.support.y)
    assert meta['class_counts_preserved'] is True and meta['query_changed'] is False
    cfg=TransportConfig(kind='recolor',alpha=1)
    a=worker.fit_transport(source,geometry,ep.support,cfg)
    b=worker.fit_transport(source,geometry,changed.support,cfg)
    assert torch.equal(a.transform,b.transform)


def test_bundled_pan_and_weibo_caches_execute_without_network(tmp_path,monkeypatch):
    """Real packaged cache layout + synthetic source weights, software scope only."""
    import urllib.request
    plan,args=fixture(tmp_path)
    root=Path(plan['root']);package=Path(__file__).resolve().parents[1]
    contract=read(Path(__file__).parent/'fixtures/input_contract.json')
    signal_sha=digest(contract);put(root/'contracts/input_contract.json',contract)
    for seed in SEEDS:
        rp=root/f'runs/source_training_v2/f0_receipt_seed-{seed}.json'
        row=read(rp);row['signal_contract_sha256']=signal_sha;put(rp,row)
        ref=root/f'runs/training_campaign_v1/references/seed-{seed}/receipt.json'
        row=read(ref);row.update(signal_contract_sha256=signal_sha,source_f0_receipt_sha256=sha(rp));put(ref,row)
    tasks=[]
    for dataset,expected in [('pan2023',28),('weibo',10)]:
        manifest=package/'data'/dataset/'manifest.json'
        assert read(manifest)['signal_contract_sha256']==signal_sha
        tasks.append({'key':dataset,'family':'transfer','group':dataset,'seed':SEEDS[0],
            'config':{'kind':'bures','alpha':.5},'requires':[], 'budgets':[20],
            'scenarios':['clean'],'external_manifest':str(manifest),'partition':dataset,
            'expected_sessions':expected})
    plan['tasks']=tasks;put(args.plan,plan)
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:pytest.fail('Worker attempted network download'))
    for i,t in enumerate(tasks):
        args.task=i;assert worker.run(args)==0
        summary=read(Path(plan['output'])/t['key']/f'evaluations/{t["partition"]}/B20/clean/summary.json')
        assert summary['sessions']==t['expected_sessions']
        assert summary['fallback_sessions']==0


def test_validation_best_state_is_a_snapshot_not_parameter_alias():
    params=torch.tensor([1.,2.],requires_grad=True)
    snapshot=worker.cpu_tree({'best_parameters':params})
    with torch.no_grad():params.add_(10)
    assert snapshot['best_parameters'].tolist()==[1.,2.]
