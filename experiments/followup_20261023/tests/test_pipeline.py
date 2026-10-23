"""Software-only synthetic fixtures; never reported as EEG scientific results."""
from argparse import Namespace
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import threading
import numpy as np
import pytest
import torch
from paper3v2 import worker
from paper3v2.data import Catalog, load_source
from paper3v2.io import put, read, sha, digest
from paper3v2.algorithm import Config, init_gate, outer_update
from p3work.f0_bridge import CompactEEGNet
from test_algorithm import fixture
import START


def write_project(root):
    contract = {"n_times": 128, "fixture": "SYNTHETIC_SOFTWARE_TEST_ONLY"}
    put(root / 'contracts/input_contract.json', contract)
    signal_sha = digest(contract)
    def desc(p): return {"path": str(p.relative_to(root)), "sha256": sha(p)}
    cp = root / 'source.pt'
    torch.save({"model": CompactEEGNet(11,128).state_dict(), "config":{"dropout":.25}},cp)
    src = root / 'runs/source_training_v2/f0_receipt_seed-20261018.json'
    put(src, {"status":"SOURCE_F0_COMPLETE", "pipeline_seed":20261018,
              "checkpoint":desc(cp), "signal_contract_sha256":signal_sha})
    np.save(root/'reference.npy', np.eye(10)/10)
    put(root/'runs/training_campaign_v1/references/seed-20261018/receipt.json',
        {"status":"SOURCE_REFERENCE_COMPLETE", "source_f0_receipt_sha256":sha(src),
         "signal_contract_sha256":signal_sha, "arrays":{"reference":desc(root/'reference.npy')}})
    roles = dict(zip(['meta_train','meta_validation','internal_holdout'],['S1','S2','S3']))
    split = {"partitions":{k:[v] for k,v in roles.items()}, "split_sha256":"b"*64}
    put(root/'split.json',split)
    sessions=[]; rng=np.random.default_rng(18)
    for role,person in roles.items():
        x=rng.normal(size=(60,11,128)).astype(np.float32)
        x-=x.mean(1,keepdims=True); x-=x.mean(2,keepdims=True)
        np.save(root/(person+'.npy'),x)
        rows=[{"participant_id":person,"session_id":"1", "tasknumber":1,"label":i%2,
            "trial_uid":person+str(i),"event_uid":person+str(i),"original_trial_index":i,
            "physical_run_id":str(i//20), "physical_run_complete_verified":True,
            "valid":i!=0,"cache_index":i if i else None} for i in range(60)]
        put(root/(person+'.json'),rows)
        sessions.append({"participant_id":person,"session_id":"1","partition":role,
            "acquisition_order_verified":True,"X":desc(root/(person+'.npy')),
            "events":desc(root/(person+'.json'))})
    put(root/'cache.json',{"status":"STIEGER_CACHE_COMPLETE", "participant_split":desc(root/'split.json'),
                          "input_contract_sha256":signal_sha,"sessions":sessions})
    put(root/'status/tasks.json',{"W3":{"cache_manifest":desc(root/'cache.json')}})


def test_prefix_budget_never_replaces_invalid_or_changes_future_query(tmp_path):
    write_project(tmp_path)
    cat=Catalog(tmp_path,'meta_train',cache_gb=.001)
    a,b=cat.materialize(cat.plans[0],'cpu',budget=4),cat.materialize(cat.plans[0],'cpu',budget=20)
    assert len(a.support.acquired_event_uids)==4 and len(a.support.y)==3
    assert len(b.support.y)==19
    assert a.query_ids==b.query_ids and len(a.query_ids)==40
    torch.testing.assert_close(a.query_x,b.query_x,atol=0,rtol=0)
    assert set(a.support.acquired_event_uids).isdisjoint(a.query_ids)
    source,geometry,_=load_source(tmp_path,20261018,'cpu')
    assert source.source_logits(a.query_x).shape==(40,2)


def test_real_cpu_update_exact_resume(tmp_path):
    _,source,geometry,episode,b=fixture();cfg=Config(variant='dual',inner_steps=2)
    gate=init_gate(cfg,'cpu')
    nb,ng,_=outer_update(source,geometry,b,gate,[episode],cfg)
    worker.save_checkpoint(tmp_path,{"update":1,"b":nb,"gate":ng})
    continuous=outer_update(source,geometry,nb,ng,[episode],cfg)
    state=worker.load_checkpoint(tmp_path)
    restored=outer_update(source,geometry,state['b'].requires_grad_(True),state['gate'].requires_grad_(True),[episode],cfg)
    assert torch.equal(continuous[0],restored[0]) and torch.equal(continuous[1],restored[1])


def test_end_to_end_worker_load_train_select_holdout_and_reuse(tmp_path,monkeypatch):
    write_project(tmp_path)
    plan=START.make_plan(tmp_path,tmp_path/'followup')
    plan['tasks']=plan['tasks'][:1]
    plan['tasks'][0]['config']['inner_steps']=1
    plan['budgets']=[20,4]
    put(tmp_path/'plan.json',plan)
    actual_loop=worker.train_loop
    def short_loop(**kw):
        kw['total_updates']=50
        return actual_loop(**kw)
    monkeypatch.setattr(worker,'train_loop',short_loop)
    args=Namespace(plan=str(tmp_path/'plan.json'),task=0,device='cpu')
    assert worker.run(args)==0
    out=Path(plan['output'])/plan['tasks'][0]['key']
    assert read(out/'result.json')['outer_updates']==50
    assert read(out/'selected.json')['holdout_used_for_selection'] is False
    row=read(out/'holdout/B4/S3-1.json')
    assert len(row['predictions'])==40 and row['future_runs_with_both_classes']==2
    previous=(out/'latest.json').read_bytes()
    assert worker.run(args)==0 and (out/'latest.json').read_bytes()==previous


def test_bash42_signal_checkpoint_and_requeue(tmp_path):
    bash=Path('/workspace/paper3-bash42-portability/bash-4.2/bash')
    if not bash.exists(): pytest.skip('Bash 4.2 fixture not installed')
    (tmp_path/'entry.py').write_text('import signal,time,sys\nsignal.signal(signal.SIGUSR1,lambda *_:sys.exit(75))\nprint("READY",flush=True)\nwhile True: time.sleep(.1)\n')
    control=tmp_path/'scontrol';control.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$PAPER3_TEST_REQUEUE"\n');control.chmod(0o700)
    env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],
         'SLURM_ARRAY_TASK_ID':'0','SLURM_JOB_ID':'123','PAPER3_TEST_REQUEUE':str(tmp_path/'requeue')}
    p=subprocess.Popen([str(bash),str(START.PACKAGE/'train.sbatch'),str(tmp_path),sys.executable,str(tmp_path),'unused'],
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    try:
        assert p.stdout.readline().startswith('TRAIN_PROCESS_STARTED')
        assert p.stdout.readline().strip()=='READY'
        p.send_signal(signal.SIGUSR1)
        output=p.communicate(timeout=10)[0]
        assert p.returncode==0 and 'CHECKPOINT_SAVED' in output
        assert (tmp_path/'requeue').read_text().strip()=='requeue 123'
    finally:
        if p.poll() is None: p.kill();p.wait()
