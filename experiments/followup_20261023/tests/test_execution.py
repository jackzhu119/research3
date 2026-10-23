import base64
import json
from pathlib import Path
import threading
import urllib.error
import numpy as np
import pytest
import torch
from paper3v2.worker import save_checkpoint, load_checkpoint, train_loop, Pause
from paper3v2.io import read, put
from paper3v2.report import paired_effect, holm
from paper3v2.monitor import GitHub, collect
import START

def state():
    return {"identity":"test", "update":0, "validated_update":0, "history":[],
            "b":torch.zeros(2, dtype=torch.float64), "gate":torch.zeros(2, dtype=torch.float64)}

def test_checkpoint_generation_fallback(tmp_path):
    s=state();s["update"]=10;save_checkpoint(tmp_path,s)
    s["update"]=20;save_checkpoint(tmp_path,s)
    (tmp_path/read(tmp_path/"latest.json")["path"]).write_bytes(b"torn")
    assert load_checkpoint(tmp_path)["update"] == 10

def test_interrupted_resume_and_validation_do_not_repeat_optimization(tmp_path):
    seen=[]; stop=threading.Event()
    def step(s):
        seen.append(s["update"])
        return s["b"] + .01, s["gate"] + .02, {}
    def save(s): save_checkpoint(tmp_path,s)
    def interrupted(s): raise Pause()
    s=state()
    with pytest.raises(Pause):
        train_loop(state=s,step=step,validate=interrupted,save=save,stop=stop,total_updates=100)
    assert seen == list(range(50))
    s=load_checkpoint(tmp_path)
    train_loop(state=s,step=step,validate=lambda s:{"participant_mean_ba":.6,"participant_mean_bce":.5},
               save=save,stop=stop,total_updates=100)
    assert seen == list(range(100))
    assert [r["outer_update"] for r in s["history"]] == [50,100]
    torch.testing.assert_close(s["b"],torch.ones(2,dtype=torch.float64))

def test_failed_update_does_not_commit_half_state(tmp_path):
    s=state(); save_checkpoint(tmp_path,s)
    def step(s): raise RuntimeError("injected kernel failure")
    with pytest.raises(RuntimeError):
        train_loop(state=s,step=step,validate=lambda s:None,save=lambda s:None,
                   stop=threading.Event(),total_updates=100)
    assert s["update"] == load_checkpoint(tmp_path)["update"] == 0

def test_three_gpu_24cpu_queue(tmp_path):
    plan=START.make_plan(tmp_path,tmp_path/"out")
    assert len(plan["tasks"]) == 24 and len({t["key"] for t in plan["tasks"]}) == 24
    command=START.submit_command(plan,"/path with space/python",[0,2,23])
    assert "--array=0,2,23%3" in command
    assert "--cpus-per-task=24" in command and "--mem=128G" in command
    assert "--gres=gpu:NVIDIAA100-SXM4-80GB:1" in command
    assert "--exclude=gpu3" in command
    assert not any("test" in item for item in command[:17])

def test_participant_statistics_not_pseudoreplicated():
    result=paired_effect([.03]*11)
    assert result["n_participants"] == 11
    assert result["p"] == 2/2048
    assert result["delta_ba_pp"] == pytest.approx(3.)
    assert holm([.01,.03,.04]) == pytest.approx([.03,.06,.06])

def test_github_push_updates_same_file_and_branch_without_credentials_in_payload():
    gh=GitHub({"token":"unit-test-token", "repo":"a/b", "branch":"progress"})
    calls=[]
    def api(path, body=None, method=None):
        calls.append((path,body,method))
        return {"sha":"oldsha"} if body is None else {"ok":True}
    gh.api=api
    gh.publish({"complete_tasks":1})
    body=calls[-1][1]
    assert body["sha"] == "oldsha" and body["branch"] == "progress"
    assert json.loads(base64.b64decode(body["content"])) == {"complete_tasks":1}
    assert "unit-test-token" not in json.dumps(calls)

def test_github_retries_conflict_from_new_sha():
    gh=GitHub({"token":"test", "repo":"a/b", "branch":"progress"}); attempts=[]
    def api(path,body=None,method=None):
        if body is None: return {"sha":str(len(attempts))}
        attempts.append(body["sha"])
        if len(attempts)==1: raise urllib.error.HTTPError(path,409,"conflict",{},None)
        return {}
    gh.api=api; gh.publish({})
    assert attempts == ["0","1"]

def test_error_is_visible_without_marking_other_tasks_failed(tmp_path):
    plan=START.make_plan(tmp_path,tmp_path)
    folder=tmp_path/plan["tasks"][0]["key"]
    put(folder/"progress.json", {"status":"FAILED","stage":"meta_train","outer_update":123})
    result=collect(plan)
    assert result["tasks"][0]["outer_update"] == 123
    assert result["tasks"][1]["status"] == "QUEUED_OR_NOT_STARTED"

def test_recovered_checkpoint_preserved_if_next_write_is_torn(tmp_path,monkeypatch):
    from paper3v2 import worker
    s=state();s['update']=10;save_checkpoint(tmp_path,s)
    s['update']=20;save_checkpoint(tmp_path,s)
    (tmp_path/read(tmp_path/'latest.json')['path']).write_bytes(b'corrupt')
    recovered=load_checkpoint(tmp_path)
    assert recovered['update']==10
    original=worker.atomic
    def interrupt(path,writer):
        if str(path).endswith('.pt'):
            Path(path).write_bytes(b'torn again')
            raise IOError('injected write failure')
        original(path,writer)
    monkeypatch.setattr(worker,'atomic',interrupt)
    with pytest.raises(IOError):save_checkpoint(tmp_path,recovered)
    assert load_checkpoint(tmp_path)['update']==10


def test_report_uses_same_11_people_across_seeds_and_curves(tmp_path):
    from paper3v2.report import aggregate
    plan=START.make_plan(tmp_path,tmp_path)
    plan['budgets']=[20,4]
    for task in plan['tasks']:
        folder=tmp_path/task['key'];put(folder/'result.json',{'status':'COMPLETE'})
        delta=.03 if task['config']['variant']=='dual' else 0.
        for budget in plan['budgets']:
            for i in range(1,12):
                put(folder/f'holdout/B{budget}/S{i}-1.json',{
                    'participant_id':f'S{i}','session_id':'1','ba':.6+delta,'bce':.5,
                    'f0_ba':.5,'negative_flip_rate':.1,'corrected_error_rate':.2+delta,
                    'run_ba_p10':.55,'calibration_seconds':.12,'future_runs_with_both_classes':2,
                    'event_uids':['a','b'],'labels':[0,1],'fallback':None})
    report=aggregate(plan)
    row=report['comparisons'][0]
    assert row['n_participants']==11 and row['delta_ba_pp']==pytest.approx(3.)
    assert len(report['budget_curves'])==16
    assert (tmp_path/'comparisons.csv').is_file() and (tmp_path/'budget_curves.csv').is_file()
    bad=tmp_path/'P-seed20261018/holdout/B20/S1-1.json';v=read(bad);v['event_uids']=['other','b'];put(bad,v)
    with pytest.raises(ValueError,match='Query trial'):aggregate(plan)
