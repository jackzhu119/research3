"""Actual frozen CompactEEGNet CPU integration; synthetic data, no scientific claim."""
from argparse import Namespace
import csv
from dataclasses import replace
from pathlib import Path
import signal
import threading

import numpy as np
import pytest
import torch

from paper3r9 import worker
from paper3r9.plan import build,core_methods
from paper3r9.report import build_report
from paper3r7.data import Catalog,load_source
from paper3r7.io import put,read,sha
from synthetic_project import write_project,SEEDS

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    threads=torch.get_num_threads();deterministic=torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1');monkeypatch.delenv('PAPER3R9_STOP_REQUESTED',raising=False)
    torch.set_num_threads(1)
    yield
    for s,h in handlers.items():signal.signal(s,h)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(deterministic)


def fixture(tmp_path,preprocessing='raw'):
    root=tmp_path/'project';root.mkdir();write_project(root)
    cfg={'kernel':'rbf','preprocessing':preprocessing,'tau':.3,'lengthscale':1.,'predictive':'anchored_laplace','anchor_cap':.75}
    ref={'key':'feature-reference','family':'feature_reference','group':'reference','seed':SEEDS[0],'config':cfg,'requires':[]}
    select={'key':'select','family':'select','group':'select','seed':SEEDS[0],'config':cfg,
        'requires':['feature-reference'],'reference_key':'feature-reference','candidates':[{'tau':.1,'lengthscale':1.},{'tau':.3,'lengthscale':1.}]}
    evaluate={'key':'evaluate','family':'core','group':'rbf_anchored','seed':SEEDS[0],'config':cfg,
        'requires':['feature-reference','select'],'reference_key':'feature-reference','selection_key':'select',
        'variant':'gp','partition':'holdout','budgets':[20,4],'scenarios':['clean']}
    plan={'root':str(root),'output':str(root/'runs/r9'),'recipe_sha256':'synthetic-v1','tasks':[ref,select,evaluate],
        'seeds':[SEEDS[0]],'comparisons':[],'budgets':[20,4]}
    path=root/'plan.json';put(path,plan)
    return plan,Namespace(plan=str(path),task=0,device='cpu')


def run_all(plan,args):
    for i in range(len(plan['tasks'])):args.task=i;assert worker.run(args)==0


def test_plan_resources_dependencies_counts_and_fixed_family(tmp_path):
    plan=build(tmp_path)
    assert len(plan['tasks'])==1548
    assert plan['counts']=={'feature_reference':6,'select':12,'core':234,'stress':432,'label_null':54,'prior_sensitivity':702,'anchor_sensitivity':108}
    assert len({t['key'] for t in plan['tasks']})==len(plan['tasks'])
    tasks={t['key']:t for t in plan['tasks']}
    for t in tasks.values():assert all(dep in tasks for dep in t.get('requires',[]))
    assert plan['resources']['jobs']==3 and plan['resources']['cpus_per_job']==16 and plan['resources']['memory_per_job']=='64G'
    assert plan['full_test_at_start'] is False and plan['compute_node_downloads'] is False
    assert len(plan['comparisons'])*len(plan['primary_partitions'])*len(plan['primary_metrics'])==24


@pytest.mark.parametrize('preprocessing',['raw','ea'])
def test_actual_source_reference_selection_evaluation_resume_and_paper_metrics(tmp_path,preprocessing):
    plan,args=fixture(tmp_path,preprocessing);run_all(plan,args)
    root=Path(plan['output']);ref=read(root/'feature-reference/result.json');selected=read(root/'select/result.json')
    assert ref['reference_scope']['future_queries_read'] is False
    assert ref['reference_scope']['participants']==1
    assert selected['matched_MAP_PP_hyperparameters'] is True and selected['holdout_used_for_selection'] is False
    a=read(root/'evaluate/evaluations/holdout/B20/clean/S3-1.json')
    b=read(root/'evaluate/evaluations/holdout/B4/clean/S3-1.json')
    assert a['query_signature']==b['query_signature'] and a['support_acquired_trials']==20 and a['support_valid_trials']==19
    assert b['support_acquired_trials']==4 and b['support_valid_trials']==3
    assert a['ba']-a['f0_ba']==pytest.approx(a['corrected_error_rate']-a['negative_flip_rate'])
    assert len(a['prediction_diagnostics']['information_fraction'])==len(a['labels'])
    assert max(a['prediction_diagnostics']['applied_gp_mixture_weight'])<=.75+1e-12
    assert (root/'evaluate/evaluations/holdout/B20/clean/S3-1.gp.pt').is_file()
    old=(root/'evaluate/evaluations/holdout/B20/clean/S3-1.json').read_bytes()
    assert worker.run(args)==0
    assert (root/'evaluate/evaluations/holdout/B20/clean/S3-1.json').read_bytes()==old
    report=build_report(plan)
    assert report['complete_tasks']==3 and report['status']=='COMPLETE'
    tables=report['compact_metrics'];assert tables['session_metrics'] and tables['participant_metrics'] and tables['query_mechanism_bins']
    with (root/'reports'/tables['session_metrics'][0]).open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert all(r['query_signature']==a['query_signature'] for r in rows)
    assert {r['budget'] for r in rows}=={'20','4'}
    with (root/'reports'/tables['query_mechanism_bins'][0]).open() as f:bins=list(csv.DictReader(f))
    assert sum(int(r['query_trials']) for r in bins)==160
    assert {r['query_signature'] for r in bins}=={a['query_signature']}


def test_reference_does_not_materialize_future_query(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    monkeypatch.setattr(Catalog,'materialize',lambda *a,**k:pytest.fail('Reference loaded future query'))
    assert worker.run(args)==0


def test_selection_does_not_read_holdout_catalog(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    original=worker.Catalog
    roles=[]
    def tracked(root,role,**kw):
        roles.append(role)
        assert role!='internal_holdout'
        return original(root,role,**kw)
    monkeypatch.setattr(worker,'Catalog',tracked)
    assert worker.run(args)==0
    args.task=1;assert worker.run(args)==0
    assert roles==['meta_train','meta_validation']


def test_pause_after_session_reuses_completed_session_on_resume(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path)
    assert worker.run(args)==0;args.task=1;assert worker.run(args)==0
    args.task=2
    original=worker.put;paused=[]
    def checkpoint_then_pause(path,value):
        original(path,value)
        if str(path).endswith('/B20/clean/S3-1.json') and not paused:
            paused.append(True);signal.raise_signal(signal.SIGUSR1)
    monkeypatch.setattr(worker,'put',checkpoint_then_pause)
    assert worker.run(args)==75
    path=Path(plan['output'])/'evaluate/evaluations/holdout/B20/clean/S3-1.json';old=path.read_bytes()
    monkeypatch.setattr(worker,'put',original)
    assert worker.run(args)==0 and path.read_bytes()==old


def test_empty_or_missingclass_support_uses_exact_source(tmp_path):
    plan,args=fixture(tmp_path);assert worker.run(args)==0
    source,geometry,_=load_source(plan['root'],SEEDS[0],'cpu');catalog=Catalog(plan['root'],'internal_holdout')
    episode=catalog.materialize(catalog.plans[0],'cpu',budget=4)
    checkpoint=read(Path(plan['output'])/'feature-reference/result.json')['selected_checkpoint']['path']
    params=torch.load(checkpoint,map_location='cpu',weights_only=True)
    for support in (replace(episode.support,y=torch.zeros_like(episode.support.y)),
                    replace(episode.support,x=episode.support.x[:0],y=episode.support.y[:0],event_uids=())):
        fitted,diag,reason=worker._fitter(source,geometry,support,worker.Config(),params,'gp')
        prediction=worker._predictor(source,geometry,fitted,'gp')(episode.query_x).softmax(1)
        expected=source.source_logits(episode.query_x).double().softmax(1)
        assert reason is not None
        assert torch.allclose(prediction,expected,atol=1e-12,rtol=1e-12)


def test_constant_mixture_is_actual_constant_control(tmp_path):
    plan,args=fixture(tmp_path);assert worker.run(args)==0
    source,geometry,_=load_source(plan['root'],SEEDS[0],'cpu');catalog=Catalog(plan['root'],'internal_holdout')
    episode=catalog.materialize(catalog.plans[0],'cpu',budget=20)
    path=read(Path(plan['output'])/'feature-reference/result.json')['selected_checkpoint']['path']
    params=torch.load(path,map_location='cpu',weights_only=True)
    laplace,_,_=worker._fitter(source,geometry,episode.support,worker.Config(predictive='laplace'),params,'gp')
    mixture,_,_=worker._fitter(source,geometry,episode.support,worker.Config(predictive='constant_mixture'),params,'gp')
    raw=worker._predictor(source,geometry,laplace,'gp')(episode.query_x).softmax(1)
    mixed=worker._predictor(source,geometry,mixture,'gp')(episode.query_x).softmax(1)
    base=source.source_logits(episode.query_x).double().softmax(1)
    assert torch.allclose(mixed,.25*base+.75*raw,atol=1e-12,rtol=1e-12)
    assert set(mixture['prediction_diagnostics']['applied_gp_mixture_weight'])=={.75}


def test_temperature_controls_cannot_change_accuracy(tmp_path):
    plan,args=fixture(tmp_path)
    source,geometry,_=load_source(plan['root'],SEEDS[0],'cpu');catalog=Catalog(plan['root'],'internal_holdout')
    episode=catalog.materialize(catalog.plans[0],'cpu',budget=20)
    for temperature in (.5,2.):
        cfg=worker.Config(temperature=temperature)
        raw,_,_=worker._fitter(source,geometry,episode.support,cfg,None,'source_temperature')
        raw_logits=worker._predictor(source,geometry,raw,'source_temperature')(episode.query_x)
        assert torch.equal(raw_logits.argmax(1),source.source_logits(episode.query_x).argmax(1))
        ea,_,_=worker._fitter(source,geometry,episode.support,cfg,None,'ea_temperature')
        ea_logits=worker._predictor(source,geometry,ea,'ea_temperature')(episode.query_x)
        original=worker.reference_logits(source,geometry,ea,episode.query_x)
        assert torch.equal(ea_logits.argmax(1),original.argmax(1))


def test_frozen_feature_cache_reuses_across_hyperparameters(tmp_path,monkeypatch):
    plan,args=fixture(tmp_path);assert worker.run(args)==0
    args.task=1;assert worker.run(args)==0
    cache=Path(plan['output'])/'frozen-feature-cache';files=list(cache.rglob('*.pt'))
    assert files
    original={p:p.read_bytes() for p in files}
    # Re-run validation after removing just completion marker: all existing
    # candidate session JSON and features are reused, not regenerated.
    (Path(plan['output'])/'select/result.json').unlink()
    monkeypatch.setattr(worker,'_fitter',lambda *a,**kw:pytest.fail('Completed candidate refitted'))
    assert worker.run(args)==0
    assert all(p.read_bytes()==data for p,data in original.items())


def test_new_prior_candidate_reuses_cached_encoder_features(tmp_path,monkeypatch):
    from paper3r7.data import FrozenSource
    plan,args=fixture(tmp_path);assert worker.run(args)==0
    args.task=1;assert worker.run(args)==0
    (Path(plan['output'])/'select/result.json').unlink()
    plan['tasks'][1]['candidates'].append({'tau':1.,'lengthscale':1.})
    put(args.plan,plan)
    monkeypatch.setattr(FrozenSource,'features_and_logits',lambda *a,**kw:pytest.fail('Prior sweep repeated encoder forward'))
    assert worker.run(args)==0
    chosen=read(Path(plan['output'])/'select/result.json')
    assert len(chosen['selection_history'])==3


def test_isolated_school_entry_runs_actual_source_pipeline(tmp_path):
    import subprocess
    import sys
    plan,args=fixture(tmp_path)
    plan['tasks']=[{'key':'source','family':'core','group':'source','seed':SEEDS[0],
        'config':{'kernel':'rbf','preprocessing':'raw'},'requires':[],'variant':'source_frozen',
        'partition':'holdout','budgets':[20],'scenarios':['clean']}]
    put(args.plan,plan)
    entry=Path(__file__).resolve().parents[1]/'paper3r9/entry.py'
    venv=tmp_path/'venv';(venv/'bin').mkdir(parents=True)
    interpreter=venv/'bin/python';interpreter.symlink_to(sys.executable)
    lib=venv/'lib'/('python%d.%d'%sys.version_info[:2]);lib.mkdir(parents=True)
    (lib/'site-packages').symlink_to(Path(torch.__file__).resolve().parent.parent,target_is_directory=True)
    process=subprocess.run([str(interpreter),'-I','-S','-B','-u',str(entry),'worker',
        '--plan',args.plan,'--task','0','--device','cpu'],capture_output=True,text=True,timeout=45)
    assert process.returncode==0,process.stdout+process.stderr
    assert 'R9_SCIENCE_PROCESS_STARTED' in process.stdout
    assert read(Path(plan['output'])/'source/result.json')['status']=='COMPLETE'


def test_decision_certificate_uses_no_query_label_and_preserves_anchor_class(tmp_path):
    plan,args=fixture(tmp_path);assert worker.run(args)==0
    source,geometry,_=load_source(plan['root'],SEEDS[0],'cpu');catalog=Catalog(plan['root'],'internal_holdout')
    episode=catalog.materialize(catalog.plans[0],'cpu',budget=20)
    with torch.no_grad():source.model.classifier.bias.copy_(torch.tensor([-4.,4.]))
    path=read(Path(plan['output'])/'feature-reference/result.json')['selected_checkpoint']['path']
    params=torch.load(path,map_location='cpu',weights_only=True)
    fitted,_,_=worker._fitter(source,geometry,episode.support,worker.Config(),params,'gp')
    prediction=worker._predictor(source,geometry,fitted,'gp')(episode.query_x)
    certificate=torch.tensor(fitted['prediction_diagnostics']['anchor_decision_preservation_certificate'])
    assert bool(certificate.any())
    assert torch.equal(prediction.argmax(1)[certificate],source.source_logits(episode.query_x).argmax(1)[certificate])
    assert fitted['diagnostics']['query_anchor_certificate_violations']==0
