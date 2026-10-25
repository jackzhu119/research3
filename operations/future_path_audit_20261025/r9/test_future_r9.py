"""Exercise the unchanged planned R9 algorithms through its published correction."""
from argparse import Namespace
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading

import pytest

PACKAGE=Path('/workspace/paper3-round9')
sys.path.insert(0,'/workspace/paper3-r5-r8-repair')
from r58_runtime_hook import install
install('paper3r9')
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PACKAGE/'tests'))
import torch
from paper3r9 import worker
from paper3r9.plan import build,SEEDS,SCENARIOS
from paper3r9.gp import GPFit
from paper3r7.data import Catalog,load_source
from paper3r7.io import read,put,sha,digest
from synthetic_project import write_project


class StrictCUDAIndexTypes(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        if func in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
            assert args[0].dtype==args[2].dtype,'CUDA indexed assignment dtype mismatch'
        return func(*args,**(kwargs or {}))


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    prior=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    monkeypatch.delenv('PAPER3R9_STOP_REQUESTED',raising=False)
    yield
    torch.set_num_threads(prior)
    for s,h in handlers.items():signal.signal(s,h)


def project(tmp_path):
    root=tmp_path/'project';root.mkdir();write_project(root)
    actual_contract(root)
    full=build(root,root/'runs/r9')
    # A fixture subset uses original planned config dictionaries. No submitted
    # source file, server plan or result path is modified by these tests.
    refs=[t for t in full['tasks'] if t['family']=='feature_reference' and t['seed']==SEEDS[0]]
    plan={**full,'tasks':refs}
    path=root/'software-plan.json';put(path,plan)
    with StrictCUDAIndexTypes():
        for i in range(len(refs)):
            assert worker.run(Namespace(plan=str(path),task=i,device='cpu'))==0
    params={t['config']['preprocessing']:torch.load(
        read(Path(plan['output'])/t['key']/'result.json')['selected_checkpoint']['path'],
        map_location='cpu',weights_only=True) for t in refs}
    source,geometry,source_id=load_source(root,SEEDS[0],'cpu')
    catalog=Catalog(root,'internal_holdout')
    return full,params,source,geometry,source_id,catalog


def actual_contract(root):
    """Give synthetic source files the existing frozen contract for external IO."""
    c=read('/workspace/research3/contracts/input_contract.json');csha=digest(c)
    put(root/'contracts/input_contract.json',c)
    for seed in SEEDS:
        source_path=root/f'runs/source_training_v2/f0_receipt_seed-{seed}.json'
        source=read(source_path);source['signal_contract_sha256']=csha;put(source_path,source)
        p=root/f'runs/training_campaign_v1/references/seed-{seed}/receipt.json'
        ref=read(p);ref.update(signal_contract_sha256=csha,source_f0_receipt_sha256=sha(source_path));put(p,ref)
    p=root/'cache.json';m=read(p);m['input_contract_sha256']=csha;put(p,m)
    p=root/'status/tasks.json';m=read(p);m['W3']['cache_manifest']['sha256']=sha(root/'cache.json');put(p,m)


def test_every_planned_config_predicts_and_gp_state_roundtrips(tmp_path):
    full,params,source,geometry,_,catalog=project(tmp_path)
    unique={json.dumps(t['config'],sort_keys=True):t['config'] for t in full['tasks']}
    assert len(unique)==104
    with StrictCUDAIndexTypes():
        for raw in unique.values():
            cfg=worker.Config(**raw)
            for budget in (4,20):
                episode=catalog.materialize(catalog.plans[0],'cpu',budget=budget)
                fitted,diagnostics,reason=worker._fitter(source,geometry,episode.support,cfg,params[cfg.preprocessing],'gp')
                logits=worker._predictor(source,geometry,fitted,'gp')(episode.query_x)
                assert logits.shape==(80,2)
                assert bool(torch.isfinite(logits).all())
                assert bool(torch.isfinite(worker.balanced_loss(logits,episode.query_y)))
                restored=GPFit.from_state_dict(fitted['gp'].state_dict())
                features,base=worker.feature_logits(source,geometry,episode.query_x,fitted['ea'])
                white=fitted['reference'].transform(features)
                torch.testing.assert_close(restored.predict_proba(white,base),fitted['gp'].predict_proba(white,base))
                assert fitted['gp'].diagnostics['support_kernel_rank']<=budget


def test_actual_all_core_selection_tasks_and_stress_null_eval_resume(tmp_path):
    root=tmp_path/'project';root.mkdir();write_project(root)
    actual_contract(root)
    full=build(root,root/'runs/r9')
    chosen=[]
    for original in full['tasks']:
        if original['seed']!=SEEDS[0]:continue
        family=original['family']
        if family in ('feature_reference','select') or (family=='core' and original['partition']=='holdout'):
            item=dict(original)
            if family=='core':item['budgets']=[20,4]
            # Keep the entire actual hyperparameter candidate grid (5linear or15RBF).
            chosen.append(item)
    # All predeclared stress types + label-null through real worker.run,
    # including the EA-preprocessed GP and constant-mixture paths.
    for family,group,scenario in [('stress','ea_rbf_anchored',s) for s in SCENARIOS]+[
            ('label_null','rbf_anchored','support_label_permutation'),
            ('stress','rbf_global_mixture','gain25')]:
        chosen.append(dict(next(t for t in full['tasks'] if t['seed']==SEEDS[0]
            and t['family']==family and t['group']==group and t['partition']=='holdout'
            and t['scenarios']==[scenario])))
    plan={**full,'tasks':chosen};path=root/'software-plan.json';put(path,plan)
    before=(PACKAGE/'paper3r9/worker.py').read_bytes()
    with StrictCUDAIndexTypes():
        for i,t in enumerate(chosen):
            assert worker.run(Namespace(plan=str(path),task=i,device='cpu'))==0,t['key']
            result=read(Path(plan['output'])/t['key']/'result.json')
            assert result['status']=='COMPLETE'
            assert result['runtime_corrections']['native_source_file_changed'] is False
    assert (PACKAGE/'paper3r9/worker.py').read_bytes()==before
    assert len(chosen)==40
    # Completed task replay reuses predictions and does not run another GP fit.
    task=chosen[-1];folder=Path(plan['output'])/task['key']
    native=(folder/'result.json').read_bytes()
    assert worker.run(Namespace(plan=str(path),task=len(chosen)-1,device='cpu'))==0
    assert (folder/'result.json').read_bytes()==native
    from paper3r9.report import build_report
    report=build_report(plan)
    assert report['status']=='COMPLETE',report['issues']
    assert report['complete_tasks']==40
    assert report['csv_tables']['diagnostics']
    assert report['primary_family_complete'] is False # No synthetic cohort inference.


@pytest.mark.parametrize('dataset',['Pan2023','Weibo2014'])
def test_actual_external_gp_ea_and_label_null(tmp_path,dataset):
    full,params,source,geometry,source_id,_=project(tmp_path)
    external='pan2023' if dataset=='Pan2023' else 'weibo'
    manifest=PACKAGE/'data'/external/'manifest.json'
    if dataset=='Pan2023':
        from paper3r7.pan2023 import PanCatalog as cls
    else:
        from paper3r7.external import ExternalCatalog as cls
    catalog=cls(full['root'],manifest,cache_gb=.1)
    catalog.plans=catalog.plans[:1]
    cfg=worker.Config(kernel='rbf',preprocessing='ea',predictive='constant_mixture',tau=3.,lengthscale=.5)
    with StrictCUDAIndexTypes():
        for scenario in ('clean','support_label_permutation','time_cov_plus35'):
            summary=worker.evaluate(source,geometry,catalog,cfg,params['ea'],
                tmp_path/dataset/scenario,threading.Event(),{},budget=4,scenario=scenario,
                variant='gp',partition=dataset,source_identity=source_id,batch_size=16)
            assert summary['sessions']==1
            assert summary['participant_mean_bce']>=0
            assert 0<=summary['participant_mean_ba']<=1
