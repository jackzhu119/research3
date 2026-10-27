"""Actual legacy operational modules and fake Slurm, no GPU/network calls."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('/workspace/paper3-round11')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PACKAGE))
import controller as ctl
import environment
import state_io
from paper3r11.platform import common, runtime, scheduler


@pytest.fixture(autouse=True)
def original_aliases():
    names = ('atomic','atomic_bytes','read','load_plan','_state','_read')
    before = [(module,key,getattr(module,key)) for name,module in list(sys.modules.items())
              if name.startswith('paper3r11.') and module is not None
              for key in names if hasattr(module,key)]
    load, atomic = scheduler.FourJobScheduler._load, scheduler._atomic
    query = scheduler.FourJobScheduler._query
    yield
    for module,key,value in before: setattr(module,key,value)
    scheduler.FourJobScheduler._load, scheduler._atomic = load, atomic
    scheduler.FourJobScheduler._query = query


def fixture(tmp_path):
    root = tmp_path/'project'; root.mkdir()
    output = root/'runs/r11-fixture'; run = output/'runtime'; run.mkdir(parents=True)
    tasks = [dict(key='a',kind='fixture',role='meta_validation',requires=[])]
    plan = dict(schema_version=common.PLAN_SCHEMA,round=11,stage='development',
        synthetic_fixture=True,root=str(root),output=str(output),package=str(PACKAGE),
        python=sys.executable,tasks=tasks,secondary_ablation_family=[],reference_strategy='fixture',
        resources=dict(jobs=4,gpus_per_job=1,cpus_per_job=16,memory_per_job='64G'),
        code_manifest={'paper3r11/platform/common.py':common.sha(PACKAGE/'paper3r11/platform/common.py')})
    plan=common.sign_plan(plan);path=root/'plan.json';common.atomic(path,plan)
    launch=runtime.context(path)['launch'];common.atomic(run/'launch.json',launch)
    common.atomic(run/'SUBMIT_AUTHORIZATION.json',dict(explicit_submit=True,
        plan_sha256=common.sha(path),recipe_sha256=plan['recipe_sha256']))
    common.atomic(run/'tasks.json',dict(recipe_sha256=plan['recipe_sha256'],tasks={
        'a':dict(phase='RUNNING',attempt=8,failure_history=[dict(error_type='old-native-crash')])}))
    marker=ctl.legacy.scheduler_identity(run,plan['recipe_sha256'])
    slots={str(i):dict(attempts=[dict(slot=i,sequence=1,marker=marker,
        name=f'{marker}-s{i}-a1',status='ACKNOWLEDGED',job_id=str(100+i),intent_at=1.)],
        next_submit_at=0.,rejections=0,worker_failures=0) for i in range(4)}
    common.atomic(run/'scheduler-state.json',dict(version=1,marker=marker,created_at=1.,slots=slots))
    env=environment.load_env(PACKAGE,path,expected_recipe=None)
    return path,plan,env


def runner_for(env, calls, live=(0,1,2,3), failed=()):
    marker=ctl.legacy.scheduler_identity(env['runtime'],env['header']['recipe_sha256'])
    def runner(command, **kwargs):
        calls.append(command)
        if command[0]=='squeue':
            output='\n'.join(f'{100+i}|{marker}|RUNNING|{marker}-s{i}-a1|gpu{i}|gpu{i}' for i in live)
        elif command[0]=='sacct':
            output='\n'.join(f'{100+i}|FAILED|2:0|{marker}|{marker}-s{i}-a1' for i in failed)
        elif command[0]=='sbatch': output=str(900+sum(row[0]=='sbatch' for row in calls))
        else: raise AssertionError('No scientific/cancellation command authorized in test')
        return SimpleNamespace(returncode=0,stdout=output,stderr='')
    return runner


def test_first_good_observation_adopts_four_jobs_and_preserves_attempt_history(tmp_path):
    path,plan,env=fixture(tmp_path);calls=[];before=(env['runtime']/'tasks.json').read_bytes()
    result=ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,launch_token='nonce',runner=runner_for(env,calls))
    assert result['task_state_observation_successful'] and result['queue_observation_successful']
    assert result['inflight_jobs']==4 and not any(row[0]=='sbatch' for row in calls)
    assert (env['runtime']/'tasks.json').read_bytes()==before
    ready=common.read(tmp_path/'work/controller-ready.json')
    assert ready['status']=='READY' and ready['strict_state_patch_loaded'] and ready['launch_token']=='nonce'
    assert common.read(env['runtime']/'controller-error.json')=={}


def test_exhausted_task_read_is_retry_not_exit_reset_or_false_ready(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path);calls=[];before=(env['runtime']/'tasks.json').read_bytes()
    monkeypatch.setattr(ctl,'load_env',lambda *a,**k:env)
    def missing(plan): raise state_io.StateUnavailable(env['runtime']/'tasks.json',OSError('visibility'),6)
    monkeypatch.setattr(runtime,'_state',missing)
    common.atomic(env['runtime']/'controller-error.json',dict(status='OLD_ERROR'))
    result=ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,runner=runner_for(env,calls))
    assert result['status']=='OBSERVATION_RETRY' and not result['submission_performed']
    assert not calls and not (tmp_path/'work/controller-ready.json').exists()
    assert (env['runtime']/'tasks.json').read_bytes()==before
    assert common.read(env['runtime']/'controller-error.json')['status']=='OBSERVATION_RETRY'


def test_same_controller_recovers_after_read_error_then_clears_error(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path);calls=[];original=runtime._state;reads=[0];stop=threading.Event()
    monkeypatch.setattr(ctl,'load_env',lambda *a,**k:env)
    def unstable(plan):
        reads[0]+=1
        if reads[0]==1:raise state_io.StateUnavailable(env['runtime']/'tasks.json',ValueError('short JSON'),6)
        return original(plan)
    monkeypatch.setattr(runtime,'_state',unstable)
    waits=[]
    def wait(seconds):
        waits.append(seconds)
        if len(waits)==2:stop.set()
    ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,
        expected_recipe=None,runner=runner_for(env,calls),stop=stop,sleep=wait)
    assert reads[0]==2 and common.read(env['runtime']/'controller-error.json')=={}
    assert common.read(tmp_path/'work/controller-ready.json')['status']=='READY'
    assert not any(row[0]=='sbatch' for row in calls)


def test_scheduler_false_exists_never_resets_retained_slots(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path)
    launch=ctl.resume_io.original_launch(env);instance=scheduler.FourJobScheduler(env['runtime'],launch)
    original=Path.exists
    monkeypatch.setattr(Path,'exists',lambda p:False if p==instance.state_path else original(p))
    instance._load()
    assert [instance.state['slots'][str(i)]['attempts'][0]['job_id'] for i in range(4)]==['100','101','102','103']


def test_missing_scheduler_state_cannot_create_new_allocation_history(tmp_path):
    path,plan,env=fixture(tmp_path);launch=ctl.resume_io.original_launch(env)
    instance=scheduler.FourJobScheduler(env['runtime'],launch)
    before=(env['runtime']/'scheduler-state.json').read_bytes();instance.state_path.unlink()
    with pytest.raises(state_io.StateUnavailable):instance._load()
    assert instance.state is None and not instance.state_path.exists()
    instance.state_path.write_bytes(before)
    instance._load();assert len(instance.state['slots']['0']['attempts'])==1


@pytest.mark.parametrize('mutation',[lambda v:v['slots']['0']['attempts'][0].pop('sequence'),
    lambda v:v['slots']['0'].update(next_submit_at='bad-clock'),
    lambda v:v['slots']['0']['attempts'][0].update(status='UNKNOWN_BROKEN'),
    lambda v:v.update(created_at=None)])
def test_readable_broken_scheduler_is_deferred_without_submission(tmp_path,mutation):
    path,plan,env=fixture(tmp_path);launch=ctl.resume_io.original_launch(env)
    value=common.read(env['runtime']/'scheduler-state.json');mutation(value)
    common.atomic(env['runtime']/'scheduler-state.json',value);calls=[]
    instance=scheduler.FourJobScheduler(env['runtime'],launch,runner=runner_for(env,calls))
    with pytest.raises(state_io.StateUnavailable):instance.step(keep_workers=True)
    assert not calls


def test_foreign_readable_recipe_is_permanent_identity_block(tmp_path):
    path,plan,env=fixture(tmp_path);value=common.read(env['runtime']/'tasks.json')
    value['recipe_sha256']='f'*64;common.atomic(env['runtime']/'tasks.json',value);calls=[]
    result=ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,runner=runner_for(env,calls))
    assert result['status']=='BLOCKED_IDENTITY_MISMATCH' and not calls
    assert common.read(env['runtime']/'tasks.json')['recipe_sha256']=='f'*64


def test_failed_queue_observation_cannot_issue_ready_or_submit(tmp_path):
    path,plan,env=fixture(tmp_path);calls=[]
    def unavailable(command,**kwargs):
        calls.append(command)
        assert command[0]=='squeue'
        return SimpleNamespace(returncode=1,stdout='',stderr='temporary scheduler unavailable')
    result=ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,runner=unavailable)
    assert result['status']=='OBSERVATION_RETRY'
    assert not (tmp_path/'work/controller-ready.json').exists()
    assert not result['submission_performed'] and len(calls)==1


def test_failed_slots_replaced_after_observation_without_touching_live_slots(tmp_path):
    path,plan,env=fixture(tmp_path);calls=[];clock=[1000.];stop=threading.Event();waits=[]
    before=(env['runtime']/'tasks.json').read_bytes()
    launch=(env['runtime']/'launch.json').read_bytes()
    def wait(seconds):
        waits.append(seconds);clock[0]+=31
        if len(waits)>=2:stop.set()
    ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,stop=stop,
        expected_recipe=None,runner=runner_for(env,calls,live=(2,3),failed=(0,1)),
        clock=lambda:clock[0],sleep=wait)
    submissions=[row for row in calls if row[0]=='sbatch']
    assert len(submissions)==2
    assert all({'--mem=64G','--cpus-per-task=16','--gres=gpu:1'}<=set(row) for row in submissions)
    assert all(str(ROOT/'entry.py') in row for row in submissions)
    assert (env['runtime']/'tasks.json').read_bytes()==before
    assert (env['runtime']/'launch.json').read_bytes()==launch
    state=common.read(env['runtime']/'scheduler-state.json')
    assert len(state['slots']['0']['attempts'])==len(state['slots']['1']['attempts'])==2
    assert len(state['slots']['2']['attempts'])==len(state['slots']['3']['attempts'])==1
    assert all(row[0] in ('squeue','sacct','sbatch') for row in calls)


def test_required_bootstrap_history_deferred_until_available_without_false_ready(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path);calls=[]
    monkeypatch.setattr(ctl,'validate_history',lambda env: (_ for _ in ()).throw(
        state_io.StateUnavailable(env['runtime']/'SUBMIT_AUTHORIZATION.json',OSError('stale'),6)))
    result=ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,runner=runner_for(env,calls))
    assert result['startup_stage']=='RETAINED_AUTHORIZATION_AND_LAUNCH'
    assert result['status']=='OBSERVATION_RETRY' and not calls
    assert not (tmp_path/'work/controller-ready.json').exists()


def test_invalid_completed_cache_cannot_be_silently_retrained(tmp_path):
    path,plan,env=fixture(tmp_path)
    value=common.read(env['runtime']/'tasks.json');value['tasks']['a'].update(phase='COMPLETE')
    common.atomic(env['runtime']/'tasks.json',value)
    with pytest.raises(state_io.StateUnavailable):runtime._state(plan)
    assert common.read(env['runtime']/'tasks.json')['tasks']['a']['phase']=='COMPLETE'


def test_successful_fresh_queue_query_clears_only_stale_observation_error(tmp_path):
    path,plan,env=fixture(tmp_path);value=common.read(env['runtime']/'scheduler-state.json')
    value['observation_error']='SQUEUE_RETRY';common.atomic(env['runtime']/'scheduler-state.json',value)
    calls=[]
    ctl.controller_loop(path,tmp_path/'work',package=PACKAGE,submit=True,once=True,
        expected_recipe=None,runner=runner_for(env,calls))
    state=common.read(env['runtime']/'scheduler-state.json')
    assert state.get('observation_error') is None
    assert state['last_successful_queue_observation_utc']


def test_future_worker_retries_state_after_releasing_task_lock_without_restarting_child(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path);run=env['runtime'];original=(run/'tasks.json').read_bytes()
    (run/'tasks.json').write_bytes(b'{')
    monkeypatch.setattr(ctl,'load_env',lambda *a,**k:env)
    events=[];stop=threading.Event()
    def fake_native_worker(path,slot,**kwargs):
        events.append('persistent-child-created-once')
        task,lease,row=runtime.claim_task(plan,slot,'100')
        lease.close();events.append('claim-after-real-state-recovered')
        assert row['attempt']==9 and row['failure_history']==[dict(error_type='old-native-crash')]
        return 0
    monkeypatch.setattr(runtime,'worker_loop',fake_native_worker)
    def repair_after_unwind(seconds):
        def writer():
            with common.locked(run/'tasks.lock'):
                events.append('other-writer-acquired-task-lock')
                (run/'tasks.json').write_bytes(original)
        thread=threading.Thread(target=writer);thread.start();thread.join(timeout=1)
        assert not thread.is_alive(), 'worker kept tasks.lock while waiting for repair'
    result=ctl.worker_loop(path,0,package=PACKAGE,device='cpu',stop=stop,
        expected_recipe=None,sleep=repair_after_unwind)
    assert result==0 and events==['persistent-child-created-once','other-writer-acquired-task-lock',
                                 'claim-after-real-state-recovered']


def test_future_worker_only_external_stop_pauses_state_wait_without_task_failure(tmp_path,monkeypatch):
    path,plan,env=fixture(tmp_path);before=(env['runtime']/'tasks.json').read_bytes()
    monkeypatch.setattr(ctl,'load_env',lambda *a,**k:env);stop=threading.Event()
    def unavailable(*a,**k): raise state_io.StateUnavailable(env['runtime']/'tasks.json',OSError('stale'),6)
    monkeypatch.setattr(runtime,'status',unavailable)
    def fake_native(path,slot,**kwargs):
        try:runtime.status(path,plan=plan)
        except Exception:pytest.fail('state wait was converted into a scientific task error')
    monkeypatch.setattr(runtime,'worker_loop',fake_native)
    result=ctl.worker_loop(path,0,package=PACKAGE,device='cpu',expected_recipe=None,
        stop=stop,sleep=lambda _:stop.set())
    assert result==75 and (env['runtime']/'tasks.json').read_bytes()==before
    assert common.read(env['runtime']/'slot-0.json')['status']=='PAUSED'
