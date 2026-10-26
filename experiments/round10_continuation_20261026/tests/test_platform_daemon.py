"""Real CPU subprocess dispatch tests; no real EEG, CUDA or Slurm."""
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from paper3r10.platform import common,runtime
from test_platform_contract import plan_fixture,write_plan,complete

FIXTURE='''import os,threading,time
if os.environ.get('R10_FIXTURE_FAIL_IMPORT')=='1':raise ModuleNotFoundError('synthetic import failure')
from pathlib import Path
import sys
sys.path.append(__TORCH_SITE__)
import torch
from paper3r10.platform.common import atomic,sha,task_identity,locked
counter=Path(os.environ['R10_FIXTURE_IMPORT_COUNTER'])
counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')
assert not torch.cuda.is_initialized()
_STOP_REQUESTED=threading.Event();_PROGRESS_LOCK=threading.RLock()
class PauseRequested(Exception):pass
def run_task(plan,task,device):
 folder=Path(plan['output'])/task['key'];folder.mkdir(parents=True,exist_ok=True)
 with locked(folder/'.scientific.lock'):
  print('TASK_STDOUT_'+task['key'],flush=True)
  mode=task.get('mode')
  (folder/'checkpoint.bin').write_bytes(b'atomic-checkpoint-fixture')
  if mode=='crash':os._exit(9)
  if mode=='error':raise RuntimeError('one-task-fixture-failure')
  if mode=='cuda_fatal':raise RuntimeError('CUDA error: device-side assert triggered (synthetic CPU replay)')
  if mode=='pause':
   while not _STOP_REQUESTED.wait(.01):pass
   raise PauseRequested('checkpoint retained')
  csv=folder/'participants.csv';csv.write_text('participant_id,ba\\nsoftware_fixture,.5\\n')
  item={'path':csv.name,'sha256':sha(csv),'bytes':csv.stat().st_size}
  atomic(folder/'result.json',{'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],
   'task_key':task['key'],'task_identity':task_identity(plan,task),'scientific_scope':'SOFTWARE_TEST_ONLY',
   'artifacts':[item],'paper_artifacts':[item]})
'''


def fixture_plan(tmp_path,monkeypatch,tasks):
    path,plan=plan_fixture(tmp_path,tasks)
    original=Path(plan['package']);package=tmp_path/'package';(package/'paper3r10').mkdir(parents=True)
    (package/'paper3r10/__init__.py').write_text('')
    shutil.copytree(original/'paper3r10/platform',package/'paper3r10/platform',ignore=shutil.ignore_patterns('__pycache__'))
    import torch
    fixture=package/'fixture_science.py';fixture.write_text(FIXTURE.replace('__TORCH_SITE__',repr(str(Path(torch.__file__).resolve().parents[1]))))
    counter=tmp_path/'numerical-imports.txt';monkeypatch.setenv('R10_FIXTURE_IMPORT_COUNTER',str(counter))
    plan.update(package=str(package),stage='mechanism',persistent_science=True,science_module='fixture_science',
                code_manifest={'fixture_science.py':common.sha(fixture)})
    write_plan(path,plan);runtime.prepare(path)
    return path,plan,counter


def task(key,**kwargs):return dict({'key':key,'kind':'fixture','family':'fixture','role':'meta_validation','requires':[]},**kwargs)


def claim_and_run(science,plan,clock):
    claimed=runtime.claim_task(plan,0,'CPU_FIXTURE',clock=lambda:clock)
    assert claimed;work,lease,row=claimed
    attempt=runtime.runtime_dir(plan)/'attempts'/work['key']/str(row['attempt'])
    try:
        code=science.run(work,row,attempt,threading.Event())
        phase=runtime.finish_task(plan,work,code,clock=lambda:clock)
        return work,code,phase,attempt
    finally:lease.close()


def test_three_tasks_one_numerical_import_and_python_failure_does_not_restart(tmp_path,monkeypatch):
    path,plan,counter=fixture_plan(tmp_path,monkeypatch,[task('first'),task('bad',mode='error'),task('last')])
    science=runtime.PersistentScience(plan,path,0,'cpu')
    try:
        first=claim_and_run(science,plan,1000);bad=claim_and_run(science,plan,1000);last=claim_and_run(science,plan,1000)
        assert (first[1],bad[1],last[1])==(0,1,0)
        assert counter.read_text()=='1' and science.incarnation==1
        assert 'TASK_STDOUT_last' in (last[3]/'stdout.log').read_text()
        assert runtime.completed_task_rows(plan).keys()=={'first','last'}
    finally:science.close()


def test_fatal_child_exit_restarts_only_scientific_child_for_next_task(tmp_path,monkeypatch):
    path,plan,counter=fixture_plan(tmp_path,monkeypatch,[task('crash',mode='crash'),task('good')])
    science=runtime.PersistentScience(plan,path,0,'cpu')
    try:
        crashed=claim_and_run(science,plan,1000);good=claim_and_run(science,plan,1000)
        assert crashed[1]==9 and good[1]==0
        assert science.incarnation==2 and counter.read_text()=='2'
    finally:science.close()


@pytest.mark.parametrize('trigger',['signal_event','runtime_stop'])
def test_usr1_checkpoint_pause_keeps_orphan_scientific_lock_until_response(tmp_path,monkeypatch,trigger):
    path,plan,counter=fixture_plan(tmp_path,monkeypatch,[task('pausing',mode='pause')])
    science=runtime.PersistentScience(plan,path,0,'cpu');claimed=runtime.claim_task(plan,0,'CPU_FIXTURE')
    work,lease,row=claimed;stop=threading.Event();answer=[]
    thread=threading.Thread(target=lambda:answer.append(science.run(work,row,
        runtime.runtime_dir(plan)/'attempts/pausing/1',stop)),daemon=True)
    thread.start();checkpoint=common.task_folder(plan,work)/'checkpoint.bin';deadline=time.monotonic()+15
    while not checkpoint.exists() and time.monotonic()<deadline:time.sleep(.05)
    assert checkpoint.exists()
    lease.close()  # Simulate loss of supervisor lease while the scientific child survives.
    assert runtime.claim_task(plan,1,'OTHER_ALLOCATION') is None
    stop_file=runtime.runtime_dir(plan)/'STOP'
    if trigger=='runtime_stop':stop_file.write_text('software fixture checkpoint request')
    else:stop.set()
    thread.join(timeout=10)
    try:
        assert answer==[75] and checkpoint.read_bytes()==b'atomic-checkpoint-fixture'
        assert runtime.finish_task(plan,work,75)=='PAUSED'
        stop_file.unlink(missing_ok=True)
        resumed=runtime.claim_task(plan,1,'OTHER_ALLOCATION');assert resumed;resumed[1].close()
    finally:science.close()


def test_large_campaign_claims_and_compact_status_do_not_scan_task_directories(tmp_path,monkeypatch):
    tasks=[task('candidate-%05d'%i) for i in range(8000)]
    path,plan=plan_fixture(tmp_path,tasks);plan['stage']='mechanism';write_plan(path,plan);runtime.prepare(path)
    monkeypatch.setattr(runtime,'verify_result',lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No per-claim directory scan')))
    claimed=runtime.claim_task(plan,0,'CPU_FIXTURE');assert claimed;claimed[1].close()
    snapshot=runtime.status(path,plan=plan)
    assert snapshot['total_tasks']==8000 and snapshot['task_rows_included']<=64 and snapshot['task_rows_truncated']
    assert snapshot['family_summary']['fixture']['total']==8000


def test_terminal_dependency_selects_after_failed_candidates_without_global_block(tmp_path):
    tasks=[task('bad'),task('eval',requires=['bad']),
           dict(key='select',family='selection',kind='selection',role='meta_validation',requires=[],requires_terminal=['eval'])]
    path,plan=plan_fixture(tmp_path,tasks);plan['stage']='mechanism';write_plan(path,plan);runtime.prepare(path)
    work,lease,row=runtime.claim_task(plan,0,'1');lease.close()
    common.atomic(common.task_folder(plan,work)/'scientific_error.json',{'error_type':'ValueError','reason':'invalid candidate',
                  'attempt_token':row['attempt_token']})
    assert runtime.finish_task(plan,work,1)=='BLOCKED_CONFIGURATION'
    selected=runtime.claim_task(plan,0,'1');assert selected and selected[0]['key']=='select';selected[1].close()


def test_shared_import_failure_exits_allocation_before_attempting_entire_campaign(tmp_path,monkeypatch):
    path,plan,counter=fixture_plan(tmp_path,monkeypatch,[task('first'),task('untouched')])
    monkeypatch.setenv('R10_FIXTURE_FAIL_IMPORT','1')
    assert runtime.persistent_worker_loop(path,0,device='cpu')==2
    rows=common.read(runtime.runtime_dir(plan)/'tasks.json')['tasks']
    assert rows['first']['attempt']==1 and 'untouched' not in rows
    assert rows['first']['current_error']['error_type']=='ScientificWorkerImportFailed'


def test_corrupted_cuda_context_response_restarts_child_without_failing_next_task(tmp_path,monkeypatch):
    path,plan,counter=fixture_plan(tmp_path,monkeypatch,[task('fatal-context',mode='cuda_fatal'),task('good')])
    science=runtime.PersistentScience(plan,path,0,'cpu')
    try:
        failed=claim_and_run(science,plan,1000);good=claim_and_run(science,plan,1000)
        assert failed[1]==1 and good[1]==0
        assert counter.read_text()=='2'
    finally:science.close()
