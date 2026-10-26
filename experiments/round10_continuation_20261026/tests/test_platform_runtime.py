import copy
import json
from pathlib import Path
import sys
import signal
import threading
from types import SimpleNamespace

import pytest

from paper3r10.platform import common,runtime
from paper3r10.platform.scheduler import FourJobScheduler
from test_platform_contract import complete,plan_fixture,write_plan


def test_claim_leases_prevent_duplicate_task_across_four_slots(tmp_path):
    _,plan=plan_fixture(tmp_path)
    first=runtime.claim_task(plan,0,'100',clock=lambda:1000)
    assert first
    assert runtime.claim_task(plan,1,'101',clock=lambda:1000) is None
    assert runtime.claim_task(plan,2,'102',clock=lambda:1000) is None
    assert runtime.claim_task(plan,3,'103',clock=lambda:1000) is None
    first[1].close()
    next_claim=runtime.claim_task(plan,1,'101',clock=lambda:1001)
    assert next_claim and next_claim[2]['attempt']==2
    next_claim[1].close()


def test_surviving_scientific_child_lock_prevents_duplicate_after_supervisor_loss(tmp_path):
    _,plan=plan_fixture(tmp_path);folder=common.task_folder(plan,plan['tasks'][0]);folder.mkdir(parents=True)
    with common.locked(folder/'.scientific.lock'):
        assert runtime.claim_task(plan,0,'100') is None
    claim=runtime.claim_task(plan,0,'100');assert claim;claim[1].close()


def test_three_persistent_failures_quarantine_and_do_not_poison_independent_tasks(tmp_path):
    tasks=[dict(key='bad',kind='fixture',role='meta_validation',requires=[]),
           dict(key='dependent',kind='fixture',role='meta_validation',requires=['bad']),
           dict(key='good',kind='fixture',role='meta_train',requires=[])]
    path,plan=plan_fixture(tmp_path,tasks);clock=[1000.]
    for attempt in range(3):
        task,lease,row=runtime.claim_task(plan,0,'1',clock=lambda:clock[0])
        assert task['key']=='bad';lease.close()
        phase=runtime.finish_task(plan,task,1,clock=lambda:clock[0])
        assert phase==('QUARANTINED' if attempt==2 else 'RETRYING')
        clock[0]+=1000
    task,lease,row=runtime.claim_task(plan,1,'2',clock=lambda:clock[0]);lease.close()
    assert task['key']=='good'
    snap=runtime.status(path,clock=lambda:clock[0])
    rows={r['key']:r for r in snap['tasks']}
    assert rows['dependent']['phase']=='BLOCKED_DEPENDENCY'
    assert snap['blocked_tasks']==2


def test_old_error_is_not_current_after_success_or_new_attempt(tmp_path):
    path,plan=plan_fixture(tmp_path);task,lease,row=runtime.claim_task(plan,0,'1',clock=lambda:1)
    lease.close();folder=common.task_folder(plan,task)
    common.atomic(folder/'error.json',dict(error_type='OldFailure',reason='original'))
    assert runtime.finish_task(plan,task,1,clock=lambda:2)=='RETRYING'
    task,lease,row=runtime.claim_task(plan,1,'2',clock=lambda:100);lease.close()
    assert runtime.status(path)['tasks'][0]['current_error'] is None
    complete(plan,task);runtime.finish_task(plan,task,0,clock=lambda:110)
    assert runtime.status(path)['tasks'][0]['phase']=='COMPLETE'
    assert runtime.status(path)['tasks'][0]['current_error'] is None


def test_structural_native_error_blocks_once_and_explicit_retry_preserves_history(tmp_path):
    path,plan=plan_fixture(tmp_path);task,lease,row=runtime.claim_task(plan,0,'1');lease.close()
    folder=common.task_folder(plan,task)
    common.atomic(folder/'scientific_error.json',dict(error_type='FileNotFoundError',reason='Pinned cache unavailable',
                                                    attempt_token=row['attempt_token']))
    assert runtime.finish_task(plan,task,1)=='BLOCKED_CONFIGURATION'
    assert runtime.claim_task(plan,1,'2') is None
    before=common.read(runtime.runtime_dir(plan)/'tasks.json')['tasks'][task['key']]
    assert runtime.retry_blocked(path)==[task['key']]
    after=common.read(runtime.runtime_dir(plan)/'tasks.json')['tasks'][task['key']]
    assert after['failure_history']==before['failure_history']
    assert after['phase']=='PAUSED' and after['explicit_retries']
    with pytest.raises(ValueError,match='explicit resume --submit'):
        runtime.main(['resume','--plan',str(path),'--retry-blocked'])


def test_usr1_pause_preserves_checkpoint_without_counting_failure(tmp_path):
    path,plan=plan_fixture(tmp_path);stop=threading.Event();signals=[]
    folder=common.task_folder(plan,plan['tasks'][0]);folder.mkdir(parents=True)
    checkpoint=folder/'checkpoint.bin';checkpoint.write_bytes(b'synthetic original checkpoint')
    class Child:
        pid=321
        code=None
        def poll(self):return self.code
        def send_signal(self,sig):signals.append(sig);self.code=75
        def wait(self):return self.code
    child=Child()
    def sleep(_):stop.set()
    code=runtime.worker_loop(path,0,device='cpu',popen=lambda *a,**k:child,sleep=sleep,stop=stop)
    assert code==75 and signals==[signal.SIGUSR1]
    assert checkpoint.read_bytes()==b'synthetic original checkpoint'
    row=common.read(runtime.runtime_dir(plan)/'tasks.json')['tasks'][plan['tasks'][0]['key']]
    assert row['phase']=='PAUSED' and row['consecutive_failures']==0 and row['current_error'] is None


def test_fresh_heartbeat_does_not_imply_fresh_numerical_update(tmp_path):
    path,plan=plan_fixture(tmp_path);task,lease,row=runtime.claim_task(plan,0,'1',clock=lambda:1000);lease.close()
    p=common.task_folder(plan,task)/'progress.json'
    common.atomic(p,dict(attempt_token=row['attempt_token'],outer_updates_completed=3,status='TRAINING'))
    first=runtime.status(path,clock=lambda:1010)['tasks'][0]
    common.atomic(p,dict(attempt_token=row['attempt_token'],outer_updates_completed=3,status='TRAINING',updated_utc='new-heartbeat'))
    second=runtime.status(path,clock=lambda:1040)['tasks'][0]
    assert first['numerical_advance_age_seconds']==0
    assert second['numerical_advance_age_seconds']==30
    common.atomic(p,dict(attempt_token=row['attempt_token'],outer_updates_completed=4,status='TRAINING'))
    third=runtime.status(path,clock=lambda:1050)['tasks'][0]
    assert third['numerical_advance_age_seconds']==0


def test_old_claim_progress_marked_stale(tmp_path):
    path,plan=plan_fixture(tmp_path);task,lease,row=runtime.claim_task(plan,0,'1');lease.close()
    p=common.task_folder(plan,task)/'progress.json'
    common.atomic(p,dict(attempt_token='different-old-claim',outer_update=500))
    snap=runtime.status(path)['tasks'][0]
    assert not snap['progress_is_current_attempt'] and snap['numerical_progress']=={}


def test_scheduler_mock_only_four_single_gpu_16cpu68g_and_no_cancellation(tmp_path):
    path,_=plan_fixture(tmp_path);ctx=runtime.context(path)
    commands=[];next_job=[100]
    def run(command,**options):
        commands.append(command)
        if command[0]=='sbatch':
            next_job[0]+=1
            return SimpleNamespace(returncode=0,stdout=str(next_job[0])+'\n',stderr='')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    scheduler=FourJobScheduler(ctx['runtime'],ctx['launch'],runner=run)
    scheduler.step()
    submitted=[c for c in commands if c[0]=='sbatch']
    assert len(submitted)==4
    assert all('--cpus-per-task=16' in c and '--mem=64G' in c and '--gres=gpu:1' in c and '--partition=hpc_gpu' in c for c in submitted)
    assert not any(c[0]=='scancel' for c in commands)
    assert all('paper3r10p-' in next(x for x in c if x.startswith('--job-name=')) for c in submitted)


def test_uncertain_sbatch_ack_is_not_resubmitted(tmp_path):
    path,_=plan_fixture(tmp_path);ctx=runtime.context(path);calls=[]
    def runner(command,**options):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='lost-ack' if command[0]=='sbatch' else '',stderr='')
    scheduler=FourJobScheduler(ctx['runtime'],ctx['launch'],runner=runner)
    scheduler.step();scheduler.step()
    assert len([c for c in calls if c[0]=='sbatch'])==4


def test_real_cpu_dummy_science_subprocess_is_sealed_reused_without_rerun(tmp_path):
    path,plan=plan_fixture(tmp_path)
    stub=tmp_path/'dummy.py'
    stub.write_text('''import argparse,json,os,sys\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--task-key');p.add_argument('--device');a=p.parse_args()\nv=json.loads(Path(a.plan).read_text());sys.path.insert(0,v['package'])\nfrom paper3r10.platform.common import atomic,sha,task_identity\nt=next(t for t in v['tasks'] if t['key']==a.task_key);f=Path(v['output'])/t['key'];f.mkdir(parents=True,exist_ok=True)\nc=f/'invocations.txt';c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\nx=f/'participants.csv';x.write_text('participant_id,ba\\nS01,0.5\\n')\nitem={'path':x.name,'sha256':sha(x),'bytes':x.stat().st_size}\natomic(f/'progress.json',{'status':'COMPLETE','session':1,'attempt_token':os.environ['PAPER3_R10_ATTEMPT_TOKEN']})\natomic(f/'result.json',{'status':'COMPLETE','recipe_sha256':v['recipe_sha256'],'task_key':t['key'],'task_identity':task_identity(v,t),'artifacts':[item],'paper_artifacts':[item],'synthetic':True,'scientific_scope':'SOFTWARE_TEST_ONLY'})\n''')
    plan['worker_entry']=str(stub);write_plan(path,plan)
    assert runtime.worker_loop(path,0,device='cpu')==0
    assert common.verify_result(plan,plan['tasks'][0])
    assert runtime.worker_loop(path,1,device='cpu')==0
    assert (common.task_folder(plan,plan['tasks'][0])/'invocations.txt').read_text()=='1'


def test_unreadable_existing_scheduler_state_never_resets_submission_history(tmp_path):
    scheduler=FourJobScheduler(tmp_path/'runtime',{'round':10,'worker_command':['python','fixture.py']},
        runner=lambda *_a,**_k:(_ for _ in ()).throw(AssertionError('No RPC after corrupt durable state')))
    scheduler.state_path.write_text('incomplete-json')
    with pytest.raises(ValueError,match='unreadable'):
        scheduler.step()


def test_optional_missing_baseline_isolated_without_blocking_core_and_rechecks_once(tmp_path):
    tasks=[dict(key='core',kind='evaluate',role='meta_validation',requires=[]),
           dict(key='historical',kind='evaluate',role='meta_validation',requires=[],
                optional_baseline=True,requires_baseline_artifacts=True,baseline_artifacts=[])]
    path,plan=plan_fixture(tmp_path,tasks)
    prepared=runtime.prepare(path)
    assert prepared['optional_baselines_unavailable']==['historical']
    task,lease,_=runtime.claim_task(plan,0,'100');lease.close()
    assert task['key']=='core'
    rows={t['key']:t for t in runtime.status(path)['tasks']}
    assert rows['historical']['phase']=='BLOCKED_CONFIGURATION'


def test_status_exposes_queued_jobs_and_controller_failure_without_scheduler_rpc(tmp_path):
    path,plan=plan_fixture(tmp_path);runtime.prepare(path);folder=runtime.runtime_dir(plan)
    common.atomic(folder/'scheduler-state.json',{'controller_status':'RUNNING','slots':{'0':{'attempts':[
        {'job_id':'123','status':'ACKNOWLEDGED','slurm_state':'PENDING','queue_reason':'Resources','sequence':1}]}}})
    common.atomic(folder/'controller-error.json',{'error_type':'FixtureError','status':'FAILED'})
    snapshot=runtime.status(path)
    assert snapshot['scheduler']['jobs'][0]['job_id']=='123'
    assert snapshot['scheduler']['jobs'][0]['queue_reason']=='Resources'
    assert snapshot['scheduler']['controller_error']['error_type']=='FixtureError'
    assert snapshot['scheduler']['queue_observation_performed_by_status'] is False


def test_wall_limit_or_preemption_preserves_checkpoint_replacement_budget(tmp_path):
    scheduler=FourJobScheduler(tmp_path/'runtime',{'round':10,'worker_command':['python','fixture.py']})
    with scheduler.acquire_lock():
        scheduler._load()
        for state in ('TIMEOUT','PREEMPTED'):
            for _ in range(5):
                attempt=scheduler._new_attempt(0)
                scheduler._finish(attempt,{'state':state,'exit_code':0,'exit_signal':15})
                assert attempt['outcome']=='CHECKPOINT'
        assert scheduler.state['slots']['0']['worker_failures']==0
        assert not scheduler.state['slots']['0'].get('quarantine')


def test_launch_distinguishes_failed_controller_spawn_from_ready_before_closing_terminal(tmp_path):
    path,plan=plan_fixture(tmp_path)
    class Failed:
        pid=123
        def poll(self):return 2
    failed=runtime.start(path,submit=True,uploads=False,popen=lambda *a,**k:Failed())
    assert failed['status']=='CONTROLLER_FAILED_TO_START'
    assert failed['controller_bootstrap']['exit_code']==2
    class Ready:
        pid=124
        def poll(self):return None
    def acknowledged(*a,**k):
        common.atomic(runtime.runtime_dir(plan)/'controller-ready.json',dict(pid=124,
                     recipe_sha256=plan['recipe_sha256'],status='BOOTSTRAP_READY'))
        return Ready()
    ready=runtime.start(path,submit=True,uploads=False,popen=acknowledged)
    assert ready['status']=='DETACHED_CONTROLLER_STARTED'
    assert ready['controller_bootstrap']['acknowledged'] is True
