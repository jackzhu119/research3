from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import single_gpu as sg


def response(output='', code=0, error=''):
    return types.SimpleNamespace(stdout=output, stderr=error, returncode=code)


class Slurm:
    def __init__(self):
        self.queue = []
        self.accounts = []
        self.calls = []
        self.submit = response('98001\n')

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        assert kwargs['timeout'] <= 45
        if command[0] == 'squeue':
            return response('\n'.join('|'.join(row) for row in self.queue))
        if command[0] == 'sacct':
            return response('\n'.join('|'.join(row) for row in self.accounts))
        if command[0] == 'sbatch':
            if isinstance(self.submit, Exception):
                raise self.submit
            return self.submit
        raise AssertionError(command)


@pytest.fixture
def scheduler(tmp_path):
    root = tmp_path/'project'; runtime=root/sg.OPERATIONAL; runtime.mkdir(parents=True)
    fake=Slurm(); now=[1000000.]
    scheduler=sg.SingleScheduler(root,tmp_path/'work',{'partition':'hpc_gpu','account':'unituser'},
        ['/school/venv/bin/python','-I','-S','-B','/immutable/worker_runner.py','--slot','{slot}'],
        runner=fake,clock=lambda:now[0])
    return scheduler,fake,now


def acknowledge(scheduler, fake):
    scheduler.observe()
    scheduler.maybe_submit(wanted=True,allowed=True)
    attempt=scheduler.state['attempts'][-1]
    fake.queue=[['98001',scheduler.marker,'RUNNING',attempt['name'],'gpu1','gpu1']]
    return attempt


def test_exact_single_resources_and_retained_thread_policy(scheduler):
    s,fake,_=scheduler
    acknowledge(s,fake)
    commands=[cmd for cmd in fake.calls if cmd[0]=='sbatch']
    assert len(commands)==1
    assert '--cpus-per-task=4' in commands[0] and '--mem=64G' in commands[0]
    assert '--gres=gpu:1' in commands[0] and '--nodes=1' in commands[0]
    assert '--partition=hpc_gpu' in commands[0]
    assert s.worker_command[-1]=='0'
    assert 'OMP_NUM_THREADS=1' in s.batch_path.read_text()
    assert s.summary()['resources']['cpus_total']==4
    assert s.summary()['resources']['gpus_total']==1
    assert s.summary()['resources']['memory_total']=='64G'


def test_repeat_cycles_do_not_duplicate_live_allocation(scheduler):
    s,fake,_=scheduler
    acknowledge(s,fake)
    for _ in range(4):
        s.observe();s.maybe_submit(wanted=True,allowed=True)
    assert sum(c[0]=='sbatch' for c in fake.calls)==1


@pytest.mark.parametrize('ack', [response('not-a-job-id'),response('',1,'connection lost'),
    subprocess.TimeoutExpired(['sbatch'],45),OSError('pipe broke')])
def test_uncertain_ack_reserves_allocation_and_never_blindly_retries(scheduler,ack):
    s,fake,now=scheduler;fake.submit=ack
    s.observe();s.maybe_submit(wanted=True,allowed=True)
    now[0]+=10000
    for _ in range(3):
        s.observe();s.maybe_submit(wanted=True,allowed=True)
    assert sum(c[0]=='sbatch' for c in fake.calls)==1
    assert s.state['attempts'][0]['status']=='UNCERTAIN'


def test_durable_intent_precedes_sbatch(scheduler):
    s,fake,_=scheduler
    original=s.runner
    def runner(cmd,**kw):
        if cmd[0]=='sbatch':
            stored=sg.read(s.state_path)
            assert stored['attempts'][-1]['status']=='INTENT'
            assert stored['attempts'][-1]['name'] in cmd[2]
        return original(cmd,**kw)
    s.runner=runner;s.observe();s.maybe_submit(wanted=True,allowed=True)


def test_lost_ack_adopted_from_exact_persisted_name(scheduler):
    s,fake,_=scheduler;fake.submit=response('lost')
    s.observe();s.maybe_submit(wanted=True,allowed=True)
    name=s.state['attempts'][0]['name']
    fake.queue=[['98022','(null)','PENDING',name,'AssocGrpGRES','']]
    s.observe();s.maybe_submit(wanted=True,allowed=True)
    assert s.summary()['active_job_ids']==['98022']
    assert sum(c[0]=='sbatch' for c in fake.calls)==1


def test_foreign_jobs_not_adopted_or_mutated(scheduler):
    s,fake,_=scheduler
    fake.queue=[['778994','other','RUNNING','p1-composition','gpu2','gpu2'],
        ['779010','r11','PENDING','paper3r11p-cd7-s0-a9','AssocGrpGRES','']]
    s.observe()
    assert s.summary()['active_job_ids']==[]
    assert all(cmd[0] in ('squeue','sacct') for cmd in fake.calls)


@pytest.mark.parametrize('state',['CANCELLED','CANCELLED by 2461'])
def test_manual_cancellation_sets_persistent_stop(scheduler,state):
    s,fake,now=scheduler;a=acknowledge(s,fake)
    fake.queue=[];fake.accounts=[['98001',state,'0:15',s.marker,a['name']]]
    s.observe();now[0]+=100;s.maybe_submit(wanted=True,allowed=True)
    assert (s.runtime/'STOP').exists()
    assert sum(cmd[0]=='sbatch' for cmd in fake.calls)==1
    assert s.summary()['controller_status']=='STOPPED_BY_USER'


def test_paused_local_outcome_waits_for_accounting_before_retry(scheduler):
    s,fake,now=scheduler;a=acknowledge(s,fake);fake.queue=[]
    sg.atomic(s.runtime/'scheduler-outcomes'/'98001.json',dict(marker=s.marker,name=a['name'],
        job_id='98001',exit_code=75))
    s.observe();now[0]+=200;s.maybe_submit(wanted=True,allowed=True)
    assert s.state['attempts'][0]['status']=='UNCERTAIN'
    assert sum(cmd[0]=='sbatch' for cmd in fake.calls)==1
    fake.accounts=[['98001','CANCELLED','0:15',s.marker,a['name']]]
    s.observe();s.maybe_submit(wanted=True,allowed=True)
    assert (s.runtime/'STOP').exists()
    assert sum(cmd[0]=='sbatch' for cmd in fake.calls)==1


@pytest.mark.parametrize('endstate,exitcode',[('TIMEOUT','0:0'),('FAILED','2:0'),('NODE_FAIL','0:9')])
def test_recoverable_terminal_allocation_continues_without_fit_reset(scheduler,endstate,exitcode):
    s,fake,now=scheduler;a=acknowledge(s,fake)
    checkpoint=s.root/'runs'/'checkpoint.pt';checkpoint.parent.mkdir(exist_ok=True);checkpoint.write_bytes(b'kept')
    fake.queue=[];fake.accounts=[['98001',endstate,exitcode,s.marker,a['name']]]
    s.observe();now[0]+=31;fake.submit=response('98002\n');s.maybe_submit(wanted=True,allowed=True)
    assert sum(cmd[0]=='sbatch' for cmd in fake.calls)==2
    assert checkpoint.read_bytes()==b'kept'


def test_stop_exists_prevents_submit_and_is_not_removed(scheduler):
    s,fake,_=scheduler;(s.runtime/'STOP').touch()
    s.observe();s.maybe_submit(wanted=True,allowed=True)
    assert not any(c[0]=='sbatch' for c in fake.calls)
    assert (s.runtime/'STOP').exists()


def test_cancelled_history_can_be_explicitly_resumed_without_old_cancel_retrigger(scheduler):
    s,fake,now=scheduler;a=acknowledge(s,fake)
    fake.queue=[];fake.accounts=[['98001','CANCELLED','0:15',s.marker,a['name']]]
    s.observe();(s.runtime/'STOP').unlink();now[0]+=40
    s.observe();fake.submit=response('98002\n');s.maybe_submit(wanted=True,allowed=True)
    assert not (s.runtime/'STOP').exists()
    assert sum(c[0]=='sbatch' for c in fake.calls)==2


def test_bad_state_is_not_reset(scheduler):
    s,fake,_=scheduler;s.state_path.write_text('{broken')
    with pytest.raises(ValueError):
        sg.SingleScheduler(s.root,s.work,s.launch,s.worker_command,runner=fake)


def test_extension_and_serial_routing_preserves_scientific_fields():
    original={'phase':'EXISTING_SOURCE_EXTENSIONS','mainline_complete':True,'all_complete':False,
        'ready_actions':[{'kind':'extension_development','key':'YANG','affinity_slot':1,'serial':False,
            'expected_receipt':'runs/YANG/freeze.json','arm':'BASE4_PLUS_YANG'},
            {'kind':'extension_development','key':'DREYER','affinity_slot':2,'serial':False,
            'expected_receipt':'runs/DREYER/freeze.json','arm':'BASE4_PLUS_DREYER'}]}
    routed=sg.route_frontier(original)
    assert [a['affinity_slot'] for a in routed['ready_actions']]==[0,0]
    assert [a['affinity_slot'] for a in original['ready_actions']]==[1,2]
    for before,after in zip(original['ready_actions'],routed['ready_actions']):
        assert {k:v for k,v in before.items() if k!='affinity_slot'}=={
            k:v for k,v in after.items() if k!='affinity_slot'}


def test_native_slot_locks_guard_live_process(scheduler):
    s,_,_=scheduler
    with (s.runtime/'worker-slot-1.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        assert sg.slot_locks_free(s.runtime) is False
    assert sg.slot_locks_free(s.runtime) is True


@pytest.fixture
def native(tmp_path,monkeypatch):
    original=Path('/workspace/paper3-r1-full-repair')
    memory=Path('/workspace/paper3-r1-controller-memory-recovery-20261028')
    root=tmp_path/'project';runtime=root/sg.OPERATIONAL;package=tmp_path/'native';package.mkdir()
    legacy=tmp_path/'legacy';legacy.mkdir()
    for name in ('parallel_runtime.py','atomic_io.py'):
        shutil.copyfile(original/name,legacy/name)
    for name in ('START.py','worker.py','scheduler.py','handoff.py','retire_legacy.py','kernel_bridge.py','audited_runtime.py'):
        shutil.copyfile(original/'vendor'/name,package/name)
    for name in ('kernel_bridge','scheduler','handoff','retire_legacy','audited_runtime'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    monkeypatch.syspath_prepend(str(memory))
    (root/'p3work').mkdir(parents=True)
    shutil.copyfile('/workspace/paper3-r11-repo/p3work/method_registry.py',root/'p3work/method_registry.py')
    marker='paper3r1p-'+hashlib.sha256(json.dumps([str(runtime),'original-campaign-v1'],separators=(',',':')).encode()).hexdigest()[:16]
    launch=dict(root=str(root),package=str(package),python=sys.executable,partition='hpc_gpu',account='unituser',
        worker_command=[sys.executable,str(package/'worker.py'),'--root',str(root)],campaign_identity='original-campaign-v1')
    state=dict(marker=marker,created_at=time.time()-3600,slots={str(i):dict(attempts=[dict(slot=i,sequence=1,
        marker=marker,job_id=str(9000+i),name=marker+'-s%d-a1'%i,status='ACKNOWLEDGED')],
        next_submit_at=0,rejections=0) for i in range(3)})
    for filename,value in [('launch.json',launch),('scheduler-state.json',state),('state.json',{'tasks':{}}),
        ('claims.json',{}),('HANDOFF_READY.json',{'status':'READY'})]:
        sg.atomic(runtime/filename,value)
    work=tmp_path/'work';work.mkdir();command=work/'command.json';sg.atomic(command,[sys.executable,'/new/worker_runner.py','--slot','{slot}'])
    args=argparse.Namespace(root=root,work=work,legacy_dir=legacy,native_package=package,
        worker_command_json=command,result=work/'result.json',submit=False)
    env=sg.load_environment(args);bridge=env['bridge']
    for path,status in [('runs/controlled_source_pilot_seed20261018/engineering_receipt.json','ENGINEERING_PILOT_COMPLETE'),
        ('runs/source_training_v2/source_profile_development.json','SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING'),
        ('runs/source_classical_v2/receipt.json','SOURCE_CLASSICAL_CORE_COMPLETE')]+[(bridge.OUTPUT+'/references/seed-%s/receipt.json'%seed,
        'SOURCE_REFERENCE_COMPLETE') for seed in bridge.SEEDS]:
        sg.atomic(root/path,{'status':status})
    yield args,env,Slurm()
    for name in ('kernel_bridge','scheduler','handoff','retire_legacy','audited_runtime'):
        sys.modules.pop(name,None)


def finish(env,action):
    value=dict(status=env['bridge'].STATUS[action['kind']])
    if action['kind'] in ('meta_selection','development_selection'):
        setting=env['bridge']._grid_function(env['root'])(action['method'])[0]
        if action['kind']=='meta_selection':value.update(dimension=setting.get('dimension',setting.get('rank')),radius=setting['radius'])
        else:value.update(configuration_index=0,configuration=setting)
    if action['kind']=='extension_initialize':value.update(arm=action['arm'],schema_version='test')
    sg.atomic(env['root']/action['expected_receipt'],value)


def test_original_scientific_dag_all_phases_extensions_complete_on_one_slot(native):
    args,env,fake=native
    scientific=[];original_launch=(env['runtime']/'launch.json').read_bytes()
    original_history=(env['runtime']/'scheduler-state.json').read_bytes()
    phases=[]
    for _ in range(30):
        # A production cycle is a new interpreter. Reload for a fresh bounded
        # receipt cache while this test preserves original module identities.
        env=sg.load_environment(args)
        result=sg.cycle(args,runner=fake,env=env)
        queue=sg.read(env['runtime']/'queue.json');phases.append(result['phase'])
        assert queue['retired_slots']==[]
        assert all(row['affinity_slot']==0 for row in queue['ready_actions'])
        progress=sg.read(env['runtime']/'progress.json')
        assert progress['requested_resources']==sg.resources()
        if result['all_complete']:
            break
        assert queue['ready_actions']
        for action in queue['ready_actions']:
            finish(env,action)
            scientific.append(action['key'])
    else:raise AssertionError(phases)
    assert phases[-1]=='ALL_ORIGINAL_WORK_COMPLETE'
    assert 'MAINLINE_FINALIZATION' in phases and 'EXISTING_SOURCE_EXTENSIONS' in phases
    assert 'internal' in scientific
    assert 'extension-development-BASE4_PLUS_YANG' in scientific
    assert 'extension-development-BASE4_PLUS_DREYER' in scientific
    assert (env['runtime']/'launch.json').read_bytes()==original_launch
    assert (env['runtime']/'scheduler-state.json').read_bytes()==original_history
    assert all(command[0] in ('squeue','sacct') for command in fake.calls)


def test_live_legacy_action_is_drained_without_signal_cancel_or_new_submit(native):
    args,env,fake=native;args.submit=True
    fake.queue=[['9000',env['marker'],'RUNNING',env['marker']+'-s0-a1','gpu1','gpu1']]
    sg.atomic(env['runtime']/'claims.json',{'native-key':dict(job_id='9000',slot=0,token='retained')})
    result=sg.cycle(args,runner=fake,env=env)
    assert result['status']=='DRAINING_EXISTING_SCIENCE'
    assert sg.read(env['runtime']/'claims.json')['native-key']['token']=='retained'
    queue=sg.read(env['runtime']/'queue.json')
    assert queue['ready_actions']==[] and queue['retired_slots']==[0,1,2]
    assert not (env['runtime']/'STOP').exists()
    assert all(command[0] in ('squeue','sacct') for command in fake.calls)


@pytest.mark.parametrize('after_hold,expect_cancel',[('PENDING',True),('RUNNING',False)])
def test_exact_pending_is_held_rechecked_but_running_never_cancelled(native,after_hold,expect_cancel):
    args,env,fake=native;args.submit=True
    job='9000';name=env['marker']+'-s0-a1'
    fake.queue=[[job,env['marker'],'PENDING',name,'AssocGrpGRES','']]
    original=fake.__call__;held=[False]
    def runner(command,**kw):
        if command[0]=='scontrol':
            fake.calls.append(command)
            if command[1]=='hold':held[0]=True;return response()
            state=after_hold if held[0] else 'PENDING'
            return response('JobId=9000 JobName='+name+' UserId=unituser(123) Comment='+env['marker']+
                ' JobState='+state+' NodeList='+('(null)' if state=='PENDING' else 'gpu1'))
        if command[0]=='scancel':
            fake.calls.append(command);assert held[0];return response()
        return original(command,**kw)
    result=sg.cycle(args,runner=runner,env=env)
    assert result['status']=='DRAINING_EXISTING_SCIENCE'
    cancellations=[command for command in fake.calls if command[0]=='scancel']
    assert bool(cancellations) is expect_cancel
    if cancellations:assert cancellations==[['scancel','--state=PENDING','9000']]


def test_actual_isolated_cycle_cli_uses_only_existing_stdlib_and_native_metadata(native,tmp_path):
    args,env,_=native
    bindir=tmp_path/'bin';bindir.mkdir();commands=tmp_path/'commands.log'
    for name in ('squeue','sacct','sbatch','scancel','scontrol'):
        path=bindir/name
        path.write_text('#!/bin/sh\nprintf "%s\\n" '+name+' >> "$SINGLE_TEST_COMMAND_LOG"\n'+
            ('exit 0\n' if name in ('squeue','sacct') else 'exit 77\n'))
        path.chmod(0o700)
    command=[sys.executable,'-I','-S','-B','-u',str(HERE/'single_gpu.py'),'--cycle',
        '--root',str(args.root),'--work',str(args.work),'--legacy-dir',str(args.legacy_dir),
        '--native-package',str(args.native_package),'--worker-command-json',str(args.worker_command_json),
        '--result',str(args.result)]
    child=subprocess.run(command,env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],
        SINGLE_TEST_COMMAND_LOG=str(commands)),text=True,capture_output=True,timeout=15)
    assert child.returncode==0,child.stdout+'\n'+child.stderr
    result=sg.read(args.result)
    assert result['phase']=='META_SCREENING' and result['ready_actions']==5
    assert result['scheduler']['resources']['cpus_total']==4
    assert commands.read_text().splitlines()==['squeue','sacct']
    assert 'R1_SINGLE_GPU_CYCLE_START' in child.stdout and 'R1_SINGLE_GPU_CYCLE_FINISHED' in child.stdout
