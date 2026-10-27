from __future__ import annotations
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

import pytest

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r1_refresh_hotfix_template',HERE/'refresh_hotfix_template.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    home=tmp_path/'home';root=tmp_path/'project';base=home/'paper3-r1-full-repair'
    old=base/h.OLD_PACKAGE;work=base/'work';old.mkdir(parents=True);work.mkdir();root.mkdir()
    with zipfile.ZipFile(HERE/'delivery/paper3-r1-complete-repair.zip') as z:
        for name in ('MANIFEST.json','supervisor.py',*h.DEPENDENCIES):(old/name).write_bytes(z.read(name))
    fixed=(HERE/'supervisor.py').read_bytes();digest=h.sha(fixed)
    monkeypatch.setattr(h,'SUPERVISOR_B85',base64.b85encode(fixed).decode())
    monkeypatch.setattr(h,'SUPERVISOR_SHA256',digest)
    overlay=base/('refresh-fixed-'+digest[:20])
    args=[sys.executable,'-I','-S','-B','-u',str(old/'supervisor.py'),'manage','--root',str(root),
        '--home',str(home),'--package',str(old),'--work',str(work),'--submit',
        '--launch-token','old-nonce','--python',sys.executable]
    row=dict(pid=123,uid=os.getuid(),start_ticks='1000',argv=args)
    records={123:row};signals=[];spawns=[];times=[0.0]
    h.atomic(work/'controller-process.json',h.encoded(dict(pid=123,launch_token='old-nonce',root=str(root))))
    def process(pid):return copy.deepcopy(records.get(pid))
    def send(pid,sig):signals.append((pid,sig));records.pop(pid,None)
    def popen(command,**kwargs):
        spawns.append((command,kwargs));records[456]=dict(pid=456,uid=os.getuid(),start_ticks='2000',argv=command)
        token=h.option(command,'--launch-token')
        h.atomic(work/'controller-ready.json',h.encoded(dict(pid=456,launch_token=token,
            first_slurm_observation_completed=True,queue_observation_successful=True)))
        class Child:
            pid=456
            def poll(self):return None
        return Child()
    def sleep(seconds):times[0]+=seconds
    kwargs=dict(process=process,send=send,popen=popen,clock=lambda:times[0],sleep=sleep,random_bytes=lambda n:b'x'*n)
    return dict(home=home,root=root,old=old,work=work,overlay=overlay,records=records,
        signals=signals,spawns=spawns,kwargs=kwargs)


def test_default_is_readonly_and_never_imports_training_or_spawns(fixture):
    x=fixture;before={str(p):p.read_bytes() for p in x['home'].rglob('*') if p.is_file()}
    value=h.apply(x['home'],x['root'],**x['kwargs'])
    assert value['status']=='READ_ONLY_HOTFIX_READY'
    assert x['signals']==x['spawns']==[]
    assert before=={str(p):p.read_bytes() for p in x['home'].rglob('*') if p.is_file()}
    assert not x['overlay'].exists()


def test_only_exact_manager_is_replaced_existing_package_and_children_untouched(fixture):
    x=fixture;before={p.name:p.read_bytes() for p in x['old'].iterdir()}
    x['records'][999]=dict(pid=999,uid=os.getuid(),start_ticks='3000',argv=['worker.py','--child'])
    value=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert value['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert x['signals']==[(123,signal.SIGTERM)]
    assert 999 in x['records'] and len(x['spawns'])==1
    command,kwargs=x['spawns'][0]
    assert command[5:7]==[str(x['overlay']/'supervisor.py'),'manage']
    assert h.option(command,'--package')==str(x['old'])
    assert h.option(command,'--work')==str(x['work'])
    assert h.option(command,'--launch-token')!='old-nonce'
    assert kwargs['start_new_session'] is True
    assert before=={p.name:p.read_bytes() for p in x['old'].iterdir()}
    assert value['scientific_children_signalled']==value['allocations_cancelled']==0
    assert value['automatic_child_refresh'] is False
    assert set(p.name for p in x['overlay'].iterdir())=={'supervisor.py',*h.DEPENDENCIES}


def test_repeated_apply_keeps_same_manager_and_no_more_signals(fixture):
    x=fixture;h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    second=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert second['status']=='R1_REFRESH_HOTFIX_ALREADY_RUNNING'
    assert second['manager_pid']==456 and second['controller_ready'] is True
    assert x['signals']==[(123,signal.SIGTERM)] and len(x['spawns'])==1


@pytest.mark.parametrize('edit',[
    lambda a:a[:5]+['/other/supervisor.py']+a[6:],
    lambda a:a[:6]+['start']+a[7:],
    lambda a:a+['--child'],
    lambda a:a+['--root','/other'],
    lambda a:[a[0],'-c','pass',*a[3:]],
    lambda a:[*a[:-2],'--launch-token','another','--python',a[0]],
])
def test_wrong_process_or_claim_never_mutates_or_signals(fixture,edit):
    x=fixture;x['records'][123]['argv']=edit(x['records'][123]['argv'])
    with pytest.raises(ValueError):h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==x['spawns']==[] and not x['overlay'].exists()


def test_pid_identity_change_before_signal_is_refused(fixture):
    x=fixture;original=x['kwargs']['process'];calls=[0]
    def changed(pid):
        calls[0]+=1;value=original(pid)
        if value is not None and calls[0]>=5:value['start_ticks']='reused'
        return value
    x['kwargs']['process']=changed
    with pytest.raises(ValueError,match='CHANGED_BEFORE_SIGNAL'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==x['spawns']==[]


def test_manifest_corruption_and_foreign_dependency_refused_before_signal(fixture):
    x=fixture;(x['old']/'parallel_runtime.py').write_text('corrupt')
    with pytest.raises(ValueError,match='ORIGINAL_REPAIR_FILE_DIFFERS'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==x['spawns']==[]


def test_overlay_symlink_refused_without_signal(fixture,tmp_path):
    x=fixture;other=tmp_path/'other';other.mkdir();x['overlay'].symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError,match='OVERLAY_IS_SYMLINK'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==x['spawns']==[]


def test_pending_term_never_escalates_or_starts_duplicate(fixture):
    x=fixture
    x['kwargs']['send']=lambda pid,sig:x['signals'].append((pid,sig))
    with pytest.raises(RuntimeError,match='TERM_PENDING'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==[(123,signal.SIGTERM)] and x['spawns']==[]


def test_retry_after_recorded_management_term_resumes_without_another_signal(fixture):
    x=fixture;normal=x['kwargs']['send']
    x['kwargs']['send']=lambda pid,sig:x['signals'].append((pid,sig))
    with pytest.raises(RuntimeError,match='TERM_PENDING'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    x['records'].pop(123);x['kwargs']['send']=normal
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert x['signals']==[(123,signal.SIGTERM)] and len(x['spawns'])==1


def test_retry_after_popen_failure_uses_recorded_stop_only(fixture):
    x=fixture;normal=x['kwargs']['popen']
    def failed(*a,**k):raise OSError('simulated exec resource failure')
    x['kwargs']['popen']=failed
    with pytest.raises(OSError):h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert h.read(x['work']/'controller-process.json')['pid']==123
    x['kwargs']['popen']=normal
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert x['signals']==[(123,signal.SIGTERM)] and len(x['spawns'])==1


def test_nonempty_fork_argv_waits_for_exact_executed_controller_identity(fixture):
    x=fixture;normal=x['kwargs']['process'];reads=[0]
    def transitioning(pid):
        if pid==456:
            reads[0]+=1
            if reads[0]<=2:
                return dict(pid=456,uid=os.getuid(),start_ticks='2000',argv=[sys.executable,'-c','fork-before-exec'])
        return normal(pid)
    x['kwargs']['process']=transitioning
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY' and reads[0]>=3
    assert h.read(x['work']/'controller-process.json')['pid']==456
    assert len(x['spawns'])==1 and x['signals']==[(123,signal.SIGTERM)]


def test_failed_new_process_has_durable_launch_record_and_can_restart(fixture):
    x=fixture;normal=x['kwargs']['popen']
    def failed(command,**kwargs):
        x['spawns'].append((command,kwargs))
        class Child:
            pid=456
            def poll(self):return 2
        return Child()
    x['kwargs']['popen']=failed
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED'
    assert h.read(x['work']/'controller-process.json')['pid']==456
    x['kwargs']['popen']=normal
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert len(x['spawns'])==2 and x['signals']==[(123,signal.SIGTERM)]


def test_unrecorded_dead_manager_never_starts_or_signals(fixture):
    x=fixture;x['records'].clear()
    with pytest.raises(ValueError,match='NO_RECORDED_HOTFIX_STOP'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['signals']==x['spawns']==[] and not x['overlay'].exists()


def test_ready_must_match_new_pid_and_nonce(fixture):
    x=fixture
    h.atomic(x['work']/'controller-ready.json',h.encoded(dict(pid=456,launch_token='stale',
        first_slurm_observation_completed=True,queue_observation_successful=True)))
    assert not h.ready_for(x['work'],456,'fresh')


def test_inflight_and_just_orphaned_refresh_steps_are_drained_before_ready(fixture):
    x=fixture;normal=x['kwargs']['send'];scans=[];orphan_seen=[False]
    initial=dict(pid=901,uid=os.getuid(),start_ticks='1500',argv=['srun','exact old refresh'],ppid=123)
    orphan=dict(pid=902,uid=os.getuid(),start_ticks='1600',argv=['srun','exact orphan refresh'],ppid=1)
    x['records'][901]=initial
    def scan(manager,root,old,**kwargs):
        scans.append(kwargs['orphans'])
        if kwargs['orphans'] and 123 not in x['records'] and not orphan_seen[0]:
            orphan_seen[0]=True
            x['records'][902]=copy.deepcopy(orphan)
        if 902 in x['records']:return [copy.deepcopy(orphan)]
        return [copy.deepcopy(initial)] if 901 in x['records'] else []
    x['kwargs']['scan_steps']=scan
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert x['signals']==[(901,signal.SIGTERM),(123,signal.SIGTERM),(902,signal.SIGTERM)]
    assert result['old_refresh_step_pids']==[901,902]
    assert result['old_refresh_steps_remaining']==0
    assert result['scientific_children_signalled']==0
    assert True in scans and len(x['spawns'])==1


def test_refresh_launcher_exec_delayed_past_first_empty_orphan_scan_is_drained(fixture):
    x=fixture;orphan_scans=[0];created=[False]
    step=dict(pid=903,uid=os.getuid(),start_ticks='1500',argv=['srun','delayed old refresh'],ppid=1)
    def scan(manager,root,old,**kwargs):
        if kwargs['orphans']:
            orphan_scans[0]+=1
            if orphan_scans[0]==3 and not created[0]:
                created[0]=True;x['records'][903]=copy.deepcopy(step)
        return [copy.deepcopy(step)] if 903 in x['records'] else []
    x['kwargs']['scan_steps']=scan
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert orphan_scans[0]>=3
    assert x['signals']==[(123,signal.SIGTERM),(903,signal.SIGTERM)]
    assert result['old_refresh_step_pids']==[903] and result['old_refresh_steps_remaining']==0


def test_undrained_refresh_step_refuses_ready_or_new_controller(fixture):
    x=fixture;normal=x['kwargs']['send']
    step=dict(pid=901,uid=os.getuid(),start_ticks='1500',argv=['srun','fixture refresh'],ppid=123)
    x['records'][901]=step
    x['kwargs']['scan_steps']=lambda *a,**k:[copy.deepcopy(step)]
    def send(pid,sig):
        if pid==901:x['signals'].append((pid,sig))
        else:normal(pid,sig)
    x['kwargs']['send']=send
    with pytest.raises(RuntimeError,match='OLD_REFRESH_STEP_PENDING'):
        h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert x['spawns']==[] and x['signals']==[(901,signal.SIGTERM),(123,signal.SIGTERM)]


def test_captured_step_pid_reuse_is_never_signalled(fixture):
    x=fixture
    old=dict(pid=901,uid=os.getuid(),start_ticks='1500',argv=['srun','old'],ppid=123)
    x['records'][901]=dict(old,start_ticks='different-new-process')
    x['kwargs']['scan_steps']=lambda *a,**k:[copy.deepcopy(old)]
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY'
    assert x['signals']==[(123,signal.SIGTERM)] and 901 in x['records']


def test_step_disappearing_before_term_does_not_block_handover(fixture):
    x=fixture;normal=x['kwargs']['send']
    step=dict(pid=901,uid=os.getuid(),start_ticks='1500',argv=['srun','old'],ppid=123)
    x['records'][901]=step
    x['kwargs']['scan_steps']=lambda *a,**k:[copy.deepcopy(step)] if 901 in x['records'] else []
    def send(pid,sig):
        if pid==901:x['records'].pop(901);raise ProcessLookupError('finished meanwhile')
        return normal(pid,sig)
    x['kwargs']['send']=send
    result=h.apply(x['home'],x['root'],activate=True,**x['kwargs'])
    assert result['status']=='R1_REFRESH_HOTFIX_CONTROLLER_READY' and len(x['spawns'])==1


def test_refresh_step_parser_rejects_probe_wrong_job_and_other_workers(tmp_path):
    root=tmp_path/'root';old=tmp_path/'package';proc=tmp_path/'proc';proc.mkdir()
    h.atomic(root/h.RUNTIME/'scheduler-state.json',h.encoded(dict(slots={
        str(slot):dict(attempts=[dict(job_id=job)]) for slot,job in enumerate(['772906','772904','772905'])})))
    command=['/usr/bin/srun','--jobid=772906','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',
        '--mem=256M','--gres=none','--time=00:04:00',sys.executable,'-I','-S','-B',
        str(old/'newnode_tool.py'),'refresh','--root',str(root),'--runtime',str(root/h.RUNTIME),
        '--job-id','772906','--slot','0']
    rows={}
    variants=[command,[v.replace('256M','2G').replace('--gres=none','--gres=gpu:1') for v in command],
        [v.replace('772906','unknownjob') for v in command],command+['--child'],
        [v.replace(str(root),'/other') for v in command],command]
    for index,args in enumerate(variants,10):
        (proc/str(index)).mkdir();rows[index]=dict(pid=index,uid=os.getuid(),ppid=123,
            start_ticks='1500' if index!=15 else '500',argv=args)
    value=h.refresh_steps(dict(pid=123,start_ticks='1000'),root,old,
        proc_root=proc,process=lambda pid:rows.get(pid))
    assert [row['pid'] for row in value]==[10]
    rows[10]['ppid']=1
    assert h.refresh_steps(dict(pid=123,start_ticks='1000'),root,old,
        proc_root=proc,process=lambda pid:rows.get(pid))==[]
    assert [row['pid'] for row in h.refresh_steps(dict(pid=123,start_ticks='1000'),root,old,
        proc_root=proc,process=lambda pid:rows.get(pid),orphans=True)]==[10]


def test_actual_entry_build_is_pure_and_payload_hash_matches(tmp_path):
    spec=importlib.util.spec_from_file_location('builder',HERE/'build_refresh_hotfix.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    (tmp_path/'supervisor.py').write_bytes((HERE/'supervisor.py').read_bytes())
    (tmp_path/'refresh_hotfix_template.py').write_bytes((HERE/'refresh_hotfix_template.py').read_bytes())
    result=builder.build(tmp_path);data=(tmp_path/'delivery'/result['entry']).read_bytes()
    assert h.sha(data)==result['entry_sha256'] and len(data)==result['entry_bytes']
    module={};exec(compile(data,'hotfix.py','exec'),module)
    assert h.sha(base64.b85decode(module['SUPERVISOR_B85']))==module['SUPERVISOR_SHA256']
    assert result['default']=='read_only_inspection'
