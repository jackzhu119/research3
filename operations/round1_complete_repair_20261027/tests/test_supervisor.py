from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import supervisor as s


def proc(tmp_path,pid,args,*,state='S',start='12345'):
    folder=tmp_path/str(pid);folder.mkdir()
    # fields[0] is original /proc state (field 3), starttime is field 22.
    fields=[state]+['0']*18+[start]+['0']*5
    (folder/'stat').write_text(str(pid)+' (python with spaces) '+' '.join(fields))
    (folder/'cmdline').write_bytes(b'\0'.join(os.fsencode(v) for v in args)+b'\0')
    return folder


def argv(root,original,*extra):
    return [sys.executable,'-I','-S','-B','-u',str(original/'START.py'),'--controller','--root',str(root),*extra]


def test_owned_original_controller_matches_actual_script_position(tmp_path):
    root=tmp_path/'project';original=tmp_path/'original';p=tmp_path/'proc';p.mkdir()
    proc(p,101,argv(root,original))
    value=s.owned_controller(101,root,original,proc_root=p)
    assert value['pid']==101 and value['start_ticks']=='12345' and value['current_patch'] is False


@pytest.mark.parametrize('change',[
    lambda a:a[:5]+['/tmp/worker.py']+a[6:],
    lambda a:a+['--child'],
    lambda a:a+['--root','/different'],
    lambda a:[sys.executable,'-c','pass',str(Path('/tmp/original')/'START.py'),'--controller','--root','/wrong'],
    lambda a:a[:6]+['--child','--root',a[-1]],
])
def test_no_worker_or_arbitrary_argv_token_is_controller(tmp_path,change):
    root=tmp_path/'project';original=tmp_path/'original';p=tmp_path/'proc';p.mkdir()
    proc(p,102,change(argv(root,original)))
    assert s.owned_controller(102,root,original,proc_root=p) is None


def test_root_python_and_zombie_mismatch_refused(tmp_path):
    root=tmp_path/'project';original=tmp_path/'original';p=tmp_path/'proc';p.mkdir()
    proc(p,103,argv(root,original),state='Z')
    proc(p,104,argv(root/'wrong',original))
    proc(p,105,['/foreign/python',*argv(root,original)[1:]])
    assert all(s.owned_controller(i,root,original,proc_root=p) is None for i in (103,104,105))


def test_current_manager_is_recognized_without_recognizing_action_child(tmp_path):
    root=tmp_path/'project';original=tmp_path/'original';p=tmp_path/'proc';p.mkdir()
    a=[sys.executable,'-I','-S','-B','-u',str(HERE/'supervisor.py'),'manage','--root',str(root),'--launch-token','fresh']
    proc(p,106,a)
    value=s.owned_controller(106,root,original,proc_root=p)
    assert value['current_patch'] is True
    proc(p,107,[*a[:6],'start',*a[7:]])
    assert s.owned_controller(107,root,original,proc_root=p) is None


def test_foreign_uid_never_read_or_signalled(tmp_path,monkeypatch):
    root=tmp_path/'project';original=tmp_path/'original';p=tmp_path/'proc';p.mkdir()
    proc(p,108,argv(root,original))
    monkeypatch.setattr(s.os,'getuid',lambda:os.stat(p/'108').st_uid+1)
    assert s.owned_controller(108,root,original,proc_root=p) is None


def test_pid_reuse_between_lookup_and_signal_never_signalled(monkeypatch):
    row=dict(pid=123,uid=5,start_ticks='old',argv=['original'])
    monkeypatch.setattr(s,'owned_controller',lambda *a,**k:dict(row,start_ticks='new'))
    calls=[]
    result=s.stop_old_controller(row,'root','package',send=lambda *a:calls.append(a),sleep=lambda _:None)
    assert result['status']=='CONTROLLER_IDENTITY_CHANGED' and calls==[]


def test_only_verified_management_pid_gets_term_and_optional_kill(monkeypatch):
    row=dict(pid=123,uid=5,start_ticks='old',argv=['original'])
    monkeypatch.setattr(s,'owned_controller',lambda *a,**k:dict(row))
    now=[0]
    def sleep(n):now[0]+=n
    calls=[]
    result=s.stop_old_controller(row,'root','package',send=lambda *a:calls.append(a),sleep=sleep,clock=lambda:now[0])
    assert calls==[(123,signal.SIGTERM),(123,signal.SIGKILL)]
    assert result['scientific_children_signalled']==0 and result['slurm_allocations_cancelled']==0


def test_term_ended_controller_never_escalates_to_kill(monkeypatch):
    row=dict(pid=123,uid=5,start_ticks='old',argv=['original']);alive=[True]
    monkeypatch.setattr(s,'owned_controller',lambda *a,**k:dict(row) if alive[0] else None)
    calls=[]
    def send(*a):calls.append(a);alive[0]=False
    s.stop_old_controller(row,'root','package',send=send,sleep=lambda _:pytest.fail('ended controller waited'))
    assert calls==[(123,signal.SIGTERM)]


def args(tmp_path,**updates):
    value=dict(root=tmp_path/'root',home=tmp_path/'home',package=HERE,work=tmp_path/'work',
        python=sys.executable,submit=True,launch_token=None,refresh_pretraining=False)
    value.update(updates)
    return types.SimpleNamespace(**value)


def stub_start(tmp_path,monkeypatch,*,controllers=None):
    value=args(tmp_path);value.work.mkdir()
    monkeypatch.setattr(s.install_hooks,'apply',lambda *a,**k:dict(status='APPLIED_SOFTWARE_TEST_ONLY'))
    monkeypatch.setattr(s,'original_package',lambda *a:tmp_path/'original')
    monkeypatch.setattr(s,'find_controllers',lambda *a,**k:controllers or [])
    return value


def test_start_without_explicit_submit_never_mutates_or_spawns(tmp_path,monkeypatch):
    monkeypatch.setattr(s.install_hooks,'apply',lambda *a,**k:pytest.fail('hook mutated'))
    with pytest.raises(ValueError,match='EXPLICIT_SUBMIT_REQUIRED'):
        s.start(args(tmp_path,submit=False),popen=lambda *a,**k:pytest.fail('spawned'))


def test_detached_controller_ready_requires_exact_nonce_and_successful_live_child(tmp_path,monkeypatch):
    value=stub_start(tmp_path,monkeypatch);seen=[]
    class Child:
        pid=321
        def poll(self):return None
    def spawn(command,**kw):
        seen.append((command,kw))
        token=s.option(command,'--launch-token')
        s.pr.put(value.work/'controller-ready.json',dict(pid=321,launch_token=token,
            first_slurm_observation_completed=True,queue_observation_successful=True))
        return Child()
    result=s.start(value,popen=spawn)
    assert result['status']=='R1_REPAIR_CONTROLLER_READY'
    command,options=seen[0]
    assert command[5]==str(HERE/'supervisor.py') and command[6]=='manage'
    assert options['start_new_session'] is True and options['close_fds'] is True
    assert 'sbatch' not in command and 'scancel' not in command
    assert result['allocation_cancellations']==0 and result['completed_results_reset']==0


@pytest.mark.parametrize('ready_update',[dict(launch_token='stale'),dict(pid=999),dict(first_slurm_observation_completed=False)])
def test_stale_startup_receipt_does_not_report_ready(tmp_path,monkeypatch,ready_update):
    value=stub_start(tmp_path,monkeypatch);now=[0]
    class Child:
        pid=321
        def poll(self):return None
    def spawn(command,**kw):
        row=dict(pid=321,launch_token=s.option(command,'--launch-token'),first_slurm_observation_completed=True,
                 queue_observation_successful=True);row.update(ready_update)
        s.pr.put(value.work/'controller-ready.json',row);return Child()
    def sleep(n):now[0]+=1
    result=s.start(value,popen=spawn,sleep=sleep,clock=lambda:now[0])
    assert result['status']=='STARTUP_PENDING'


def test_exited_child_with_ready_receipt_is_not_reported_ready(tmp_path,monkeypatch):
    value=stub_start(tmp_path,monkeypatch)
    class Child:
        pid=321
        def poll(self):return 2
    def spawn(command,**kw):
        s.pr.put(value.work/'controller-ready.json',dict(pid=321,launch_token=s.option(command,'--launch-token'),
            first_slurm_observation_completed=True,queue_observation_successful=True));return Child()
    assert s.start(value,popen=spawn)['status']=='STARTUP_FAILED'


def test_repeat_current_manager_nonce_prevents_duplicate_spawn(tmp_path,monkeypatch):
    row=dict(pid=321,current_patch=True,argv=[sys.executable,str(HERE/'supervisor.py'),'manage','--launch-token','nonce'])
    value=stub_start(tmp_path,monkeypatch,controllers=[row])
    s.pr.put(value.work/'controller-process.json',dict(pid=321,launch_token='nonce'))
    result=s.start(value,popen=lambda *a,**k:pytest.fail('duplicate manager'))
    assert result['status'].startswith('EXISTING_R1_REPAIR_') and result['status'].endswith(('REUSED','STARTUP_PENDING'))


def test_current_manager_without_ready_is_pending_not_false_ready(tmp_path,monkeypatch):
    row=dict(pid=321,current_patch=True,argv=[sys.executable,str(HERE/'supervisor.py'),'manage','--launch-token','nonce'])
    value=stub_start(tmp_path,monkeypatch,controllers=[row])
    s.pr.put(value.work/'controller-process.json',dict(pid=321,launch_token='nonce'))
    result=s.start(value,popen=lambda *a,**k:pytest.fail('duplicate manager'))
    assert result['status'].endswith('_STARTUP_PENDING') and 'REUSED' not in result['status']


def test_unsuccessful_queue_observation_never_reports_ready(tmp_path,monkeypatch):
    value=stub_start(tmp_path,monkeypatch);now=[0]
    class Child:
        pid=321
        def poll(self):return None
    def spawn(command,**kw):
        s.pr.put(value.work/'controller-ready.json',dict(pid=321,launch_token=s.option(command,'--launch-token'),
            first_slurm_observation_completed=True,queue_observation_successful=False))
        return Child()
    def sleep(n):now[0]+=1
    result=s.start(value,popen=spawn,sleep=sleep,clock=lambda:now[0])
    assert result['status']=='STARTUP_PENDING'


def test_original_package_prefers_exact_current_launch_package(tmp_path,monkeypatch):
    root=tmp_path/'root';home=tmp_path/'home';first=tmp_path/'old';current=tmp_path/'current'
    s.pr.put(root/s.pr.OPERATIONAL/'launch.json',dict(package=str(current)))
    monkeypatch.setattr(s.install_hooks,'discover_packages',lambda *a:[first,current])
    assert s.original_package(root,home)==current


def test_current_jobs_excludes_finished_rejected_or_pending_attempts(tmp_path):
    state=dict(slots={'0':dict(attempts=[dict(job_id='1',status='ACKNOWLEDGED',slurm_state='RUNNING')]),
                      '1':dict(attempts=[dict(job_id='2',status='FINISHED',slurm_state='RUNNING')]),
                      '2':dict(attempts=[dict(job_id='3',status='ACKNOWLEDGED',slurm_state='PENDING')])})
    s.pr.put(tmp_path/'scheduler-state.json',state)
    assert s.current_jobs(tmp_path)==[(0,'1')]


def test_reference_runtime_uses_exact_recorded_class_not_arbitrary_first_observation(tmp_path,monkeypatch):
    def normalized(value):
        return {k:v for k,v in value.items() if k!='device_uuid'}
    canonical=types.SimpleNamespace(_profile=normalized,_raw_profile=normalized)
    monkeypatch.setitem(sys.modules,'hardware_comparison',canonical)
    right=dict(device_name='A100-PCIE-40GB',timing_protocol='original',python='3.12.3')
    wrong=dict(right,device_name='A100-SXM4-80GB')
    s.pr.put(tmp_path/s.pr.OPERATIONAL/'runtime_transfer_comparison.json',dict(
        hardware_class_profile=right,observations={
            'first_historical':dict(runtime_profile=dict(wrong,device_uuid='wrong-uuid')),
            'actual_primary':dict(runtime_profile=dict(right,device_uuid='actual-uuid'))}))
    selected,source=s.reference_runtime(tmp_path)
    assert selected==dict(right,device_uuid='actual-uuid')
    assert source['source']=='existing_primary_runtime_observation'


def test_ambiguous_unmatched_journal_does_not_fabricate_reference(tmp_path,monkeypatch):
    canonical=types.SimpleNamespace(_profile=lambda p:{k:v for k,v in p.items() if k!='device_uuid'},
        _raw_profile=lambda p:{k:v for k,v in p.items() if k!='device_uuid'})
    monkeypatch.setitem(sys.modules,'hardware_comparison',canonical)
    s.pr.put(tmp_path/s.pr.OPERATIONAL/'runtime_transfer_comparison.json',dict(
        hardware_class_profile={'device_name':'required'},observations={
            'other':dict(runtime_profile=dict(device_name='other',timing_protocol='original',device_uuid='physical'))}))
    selected,source=s.reference_runtime(tmp_path)
    assert selected is None and source['source']=='PRIMARY_RUNTIME_NOT_YET_AVAILABLE'


def test_probe_call_only_existing_allocation_step_and_identity_bound_result(tmp_path):
    seen=[]
    def run(command,**kw):
        seen.append((command,kw))
        return types.SimpleNamespace(returncode=0,stdout='noise\n'+json.dumps(dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT',job_id='123',slot=2,runtime_profile={'observed':True}))+'\n',stderr='')
    result=s.node_call('probe',tmp_path,tmp_path/'runtime',2,'123',sys.executable,runner=run)
    command,options=seen[0]
    assert command[0]=='srun' and '--jobid=123' in command and '--overlap' in command
    assert '--gres=gpu:1' in command and 'sbatch' not in command and 'scancel' not in command
    assert result['step_returncode']==0 and result['job_id']=='123'


def test_refresh_call_uses_no_gpu_and_no_new_allocation(tmp_path):
    seen=[]
    def run(command,**kw):
        seen.append(command)
        return types.SimpleNamespace(returncode=0,stdout=json.dumps(dict(status='CHECKPOINT_REFRESH_REQUESTED',job_id='123',slot=0)),stderr='')
    result=s.node_call('refresh',tmp_path,tmp_path/'runtime',0,'123',sys.executable,runner=run)
    assert '--gres=none' in seen[0] and '--jobid=123' in seen[0]
    assert result['status']=='CHECKPOINT_REFRESH_REQUESTED'


def test_wrong_job_stdout_or_timeout_cannot_authorize_runtime_observation(tmp_path):
    def wrong(command,**kw):
        return types.SimpleNamespace(returncode=0,stdout=json.dumps(dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT',job_id='foreign',slot=0)),stderr='')
    assert s.node_call('probe',tmp_path,tmp_path/'runtime',0,'123',sys.executable,runner=wrong)['status']=='ALLOCATION_STEP_RETRY'
    def timeout(*a,**kw):raise subprocess.TimeoutExpired(['software-test-srun'],1)
    result=s.node_call('probe',tmp_path,tmp_path/'runtime',0,'123',sys.executable,runner=timeout)
    assert result['status']=='ALLOCATION_STEP_RETRY' and result['error_type']=='TimeoutExpired'


def test_allocation_watch_observes_existing_job_without_child_refresh(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';work=tmp_path/'work'
    s.pr.put(runtime/'worker-status/slot-0.json',dict(claim_token='claim',child_pid=222))
    monkeypatch.setattr(s,'current_jobs',lambda p:[(0,'123')])
    monkeypatch.setattr(s,'reference_runtime',lambda p:({'device_name':'actual'},{'source':'software-fixture'}))
    calls=[]
    def call(mode,root,runtime,slot,job,python,**kw):
        calls.append(mode)
        return (dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT',job_id=job,slot=slot,
                     step_returncode=0,runtime_profile={'device_name':'actual'}) if mode=='probe'
                else dict(status='CHECKPOINT_REFRESH_REQUESTED',job_id=job,slot=slot))
    monkeypatch.setattr(s,'node_call',call)
    class Stop:
        n=0
        def is_set(self):return self.n>=2
        def wait(self,n):self.n+=1
    s.watch_allocations(tmp_path,runtime,work,sys.executable,Stop())
    assert calls==['probe']
    value=s.pr.read_json(work/'hardware-observations.json')
    assert value['slots']['0']['verified'] is True
    assert value['slots']['0']['job_id']=='123'
    assert value['new_allocations_created_by_probe'] is False


@pytest.mark.parametrize('probe_available',[True,False])
def test_watch_never_interrupts_importing_or_replaced_children_on_three_allocations(tmp_path,monkeypatch,probe_available):
    runtime=tmp_path/'runtime';work=tmp_path/'work'
    jobs=[(0,'772906'),(1,'772904'),(2,'772905')]
    for slot,job in jobs:
        s.pr.put(runtime/'worker-status'/('slot-%d.json'%slot),dict(
            job_id=job,slot=slot,pid=100+slot,child_pid=200+slot,
            claim_token='before-import-'+str(slot),action_key='development-before-'+str(slot),
            phase='RUNNING',kernel_ready=False))
    monkeypatch.setattr(s,'current_jobs',lambda p:jobs)
    monkeypatch.setattr(s,'reference_runtime',lambda p:({'device_name':'actual'},{'source':'software-fixture'}))
    calls=[]
    def call(mode,root,runtime,slot,job,python,**kw):
        calls.append((mode,slot,job))
        assert mode=='probe', 'Observing an absent import-time marker must never signal a child'
        # Simulate an import completing and the worker immediately switching to
        # another action while the hardware-only allocation step is still live.
        s.pr.put(runtime/'worker-status'/('slot-%d.json'%slot),dict(
            job_id=job,slot=slot,pid=100+slot,child_pid=300+slot,
            claim_token='during-probe-'+str(slot),action_key='development-new-'+str(slot),
            phase='RUNNING',kernel_ready=False))
        return (dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT',job_id=job,slot=slot,
                     step_returncode=0,runtime_profile={'device_name':'actual'}) if probe_available
                else dict(status='ALLOCATION_STEP_RETRY',job_id=job,slot=slot,returncode=1))
    monkeypatch.setattr(s,'node_call',call)
    notices=[]
    monkeypatch.setattr(s.pr,'emit',lambda event,**fields:notices.append((event,fields)))
    class Stop:
        n=0
        def is_set(self):return self.n>=4
        def wait(self,n):
            self.n+=1
            # No repair-active marker is ever written, covering newly imported
            # tasks and an old package whose optional marker write is delayed.
            for slot,job in jobs:
                s.pr.put(runtime/'worker-status'/('slot-%d.json'%slot),dict(
                    job_id=job,slot=slot,pid=100+slot,child_pid=400+slot+self.n*10,
                    claim_token='new-claim-'+str(self.n),action_key='development-'+str(self.n),
                    phase='RUNNING',kernel_ready=False))
    s.watch_allocations(tmp_path,runtime,work,sys.executable,Stop())
    assert sorted(calls)==[('probe',slot,job) for slot,job in jobs]
    assert not (work/'refresh-results').exists()
    value=s.pr.read_json(work/'hardware-observations.json')
    assert value['refresh_policy']=='DISABLED_NEW_ACTIONS_NEVER_SIGNALLED'
    assert value['scientific_children_signalled']==0
    assert value['new_allocations_created_by_probe'] is False
    assert set(value['slots'])=={'0','1','2'}
    assert notices[0][0]=='R1_AUTOMATIC_CHILD_REFRESH_DISABLED'


def test_new_allocation_is_only_probed_never_refreshed(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';work=tmp_path/'work';current=['772906']
    monkeypatch.setattr(s,'current_jobs',lambda p:[(0,current[0])])
    monkeypatch.setattr(s,'reference_runtime',lambda p:({'device_name':'actual'},{}))
    calls=[]
    def call(mode,root,runtime,slot,job,python,**kw):
        calls.append((mode,job))
        assert mode=='probe'
        return dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT',job_id=job,slot=slot,
                    step_returncode=0,runtime_profile={'device_name':'actual'})
    monkeypatch.setattr(s,'node_call',call)
    class Stop:
        n=0
        def is_set(self):return self.n>=2
        def wait(self,n):self.n+=1;current[0]='replacement'
    s.watch_allocations(tmp_path,runtime,work,sys.executable,Stop())
    assert calls==[('probe','772906'),('probe','replacement')]
    assert not (work/'refresh-results').exists()
