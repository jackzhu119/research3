"""Audit actual R5 declared task dispatch and source-only reference execution."""
from pathlib import Path
import signal
import sys
sys.path.insert(0, '/workspace/paper3-round5-hpcgpu')
sys.path.insert(0, '/workspace/paper3-round5-hpcgpu/tests')
import pytest
import torch
from paper3r5 import worker
from paper3r5.plan import make_plan
from paper3r5.io import read, put
from test_worker import plan_fixture

ORIGINAL=worker._configuration

def corrected_configuration(task):
    config,variant=ORIGINAL(task)
    if task.get('variant')=='support_EA':variant='support_EA'
    return config,variant

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1);monkeypatch.delenv('PAPER3R5_STOP_REQUESTED',raising=False)
    yield
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(det)
    for sig,h in handlers.items():signal.signal(sig,h)

def test_original_reference_bug_matches_actual_plan(tmp_path):
    plan=make_plan(tmp_path,tmp_path/'out')
    actual=[t for t in plan['tasks'] if t['variant']=='support_EA']
    assert len(actual)==3
    assert all(ORIGINAL(t)[1]=='head_newton' for t in actual)

def test_corrected_dispatch_matches_all_105_declared_variants(tmp_path):
    plan=make_plan(tmp_path,tmp_path/'out')
    assert len(plan['tasks'])==105
    for task in plan['tasks']:
        config,variant=corrected_configuration(task)
        assert variant==task['variant']
        if variant=='support_EA':assert config.variant=='head_newton'

@pytest.mark.parametrize('seed',[20261018,20261019,20261020])
def test_real_actual_plan_ea_reference_is_label_free(tmp_path,monkeypatch,seed):
    plan,args=plan_fixture(tmp_path)
    task=next(t for t in make_plan(tmp_path,tmp_path/'other')['tasks']
              if t['key']==f'reference-support_EA-seed{seed}')
    task['budgets']=[20,4]
    plan['tasks']=[task];put(args.plan,plan)
    monkeypatch.setattr(worker,'_configuration',corrected_configuration)
    def forbidden(*args,**kwargs):pytest.fail('Support EA entered the head/spatial optimizer')
    monkeypatch.setattr(worker,'fit',forbidden)
    monkeypatch.setattr(worker,'train_loop',forbidden)
    assert worker.run(args)==0
    folder=Path(plan['output'])/task['key']
    assert read(folder/'selected.json')['status']=='FIXED_REFERENCE_NO_MODEL_SELECTION'
    for budget in (20,4):
        row=read(folder/f'evaluations/holdout/B{budget}/clean/S3-1.json')
        assert row['variant']=='support_EA'
        assert row['reference_fit']['uses_support_labels'] is False
        assert row['reference_fit']['optimizer_updates']==0
        assert row['diagnostics']['variant']=='support_EA'
