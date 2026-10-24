"""R4 checkpoint handoff: fake Slurm only; original science files stay intact."""
from collections import deque
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from paper3r4 import runtime


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location('tested_r4_handoff', PACKAGE/'MIGRATE.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSlurm:
    def __init__(self, queues=(), accounting=()):
        self.queues = deque(queues)
        self.accounts = deque(accounting)
        self.last_accounting = ''
        self.commands = []
        self.signal_failures = deque()
        self.submission_results = deque()
        self.sleeps = []
        self.now = 1793000000.

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        name = Path(command[0]).name
        assert name in ('squeue','sacct','scancel','sbatch'), 'No real subprocess is allowed'
        if name == 'squeue':
            value = self.queues.popleft() if self.queues else ''
        elif name == 'sacct':
            if self.accounts:
                self.last_accounting = self.accounts.popleft()
            value = self.last_accounting
        elif name == 'scancel':
            value = self.signal_failures.popleft() if self.signal_failures else ''
        else:
            value = self.submission_results.popleft() if self.submission_results else '900001\n'
        if isinstance(value, Exception):
            raise value
        if isinstance(value, tuple):
            return SimpleNamespace(returncode=value[0],stdout='',stderr=value[1])
        return SimpleNamespace(returncode=0,stdout=value,stderr='')

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        assert len(self.sleeps)<20, 'Handoff did not converge in the fake scheduler'

    def of(self, name):
        return [command for command in self.commands if command[0]==name]


@pytest.fixture
def old_run(tmp_path):
    output = tmp_path/'project/runs/followup_20261024_r4'
    root = output.parent.parent
    rt = output/'runtime'
    rt.mkdir(parents=True)
    plan = {'root':str(root),'output':str(output),'recipe_sha256':'synthetic-original-r4-recipe',
            'prepared_at':1792837924.4,'cpus':16,'gpus':3,
            'tasks':[{'key':'complete','requires':[]},{'key':'partial','requires':[]}]}
    path = output/'plan.json'
    path.write_text(json.dumps(plan,indent=7)+'\n')
    marker = runtime.job_marker(plan)
    attempt = {'sequence':4,'marker':marker,'name':marker+'-a4','job_id':'7001',
               'status':'ACKNOWLEDGED','intent_at':1792838000.}
    def write(relative,value):
        item=output/relative
        item.parent.mkdir(parents=True,exist_ok=True)
        item.write_text(json.dumps(value,indent=3)+'\n')
        return item
    write('runtime/state.json',{'marker':marker,'attempts':[attempt],
          'created_at':1792837924.4,'controller_status':'ALLOCATION_RUNNING'})
    write('runtime/receipts/allocation-a004.json',attempt)
    write('runtime/outcomes/7001.json',{'exit_code':75,'stop_reason':'time_limit'})
    write('runtime/launch.json',{'partition':'hpc_gpu1','slots':3,'cpus':16,'memory':'64G'})
    write('runtime/allocation_state.json',{'recipe_sha256':plan['recipe_sha256'],'tasks':{
          '0':{'key':'complete','phase':'COMPLETE','status':'COMPLETE','attempt':1},
          '1':{'key':'partial','phase':'PAUSED','status':'PAUSED','attempt':1}}})
    write('complete/result.json',{'status':'COMPLETE','identity':'original-scientific-id'})
    write('partial/latest.json',{'slot':0,'sha256':'original-checkpoint-digest'})
    checkpoint=output/'partial/state-0.pt'
    checkpoint.write_bytes(b'original-opaque-numerical-checkpoint-and-rng')
    write('partial/holdout/B20/person-session.json',{'parameter_id':'original','probabilities':[[.2,.8]]})
    (output/'STOP').touch()
    protected=[path,output/'complete/result.json',output/'partial/latest.json',checkpoint,
               output/'partial/holdout/B20/person-session.json']
    before={item:(item.read_bytes(),item.stat().st_mtime_ns) for item in protected}
    return plan,path,rt,marker,before


def queue(marker,state='RUNNING',job='7001',name=None,partition='hpc_gpu1'):
    return f'{job}|{marker}|{state}|{name or marker+"-a4"}|{partition}\n'


def account(marker,state='CANCELLED',job='7001',name=None,comment=None):
    return f'{job}|{state}|75:0|{marker if comment is None else comment}|{name or marker+"-a4"}\n'


def perform(migration,path,fake):
    handoff=migration.Handoff(path,runner=fake,sleeper=fake.sleep,clock=lambda:fake.now)
    archive=handoff.quiesce()
    return handoff,archive


@pytest.mark.parametrize('state,cancel_command',[
    ('PENDING',['scancel','--state=PENDING','7001']),
    ('RUNNING',['scancel','--batch','--signal=USR1','7001'])])
def test_exact_owned_job_requests_checkpoint_or_pending_cancel_and_preserves_science(migration,old_run,state,cancel_command):
    plan,path,rt,marker,before=old_run
    unrelated=queue('paper3r5-unrelated',state='RUNNING',job='8111')
    fake=FakeSlurm([queue(marker,state)+unrelated,unrelated,unrelated],
                   [account(marker)+account('paper3r5-unrelated',job='8111')])
    _,archive=perform(migration,path,fake)
    assert fake.of('scancel')==[cancel_command]
    assert all(item.read_bytes()==data and item.stat().st_mtime_ns==mtime
               for item,(data,mtime) in before.items())
    assert (archive/'state.json').is_file() and (archive/'launch.json').is_file()
    assert (archive/'allocation_state.json').is_file() and (archive/'outcomes/7001.json').is_file()
    assert (archive/'runtime_STOP').is_file() and (archive/'output_STOP').is_file()
    assert not (rt/'STOP').exists() and not (rt.parent/'STOP').exists()
    state=json.loads((rt/'state.json').read_text())
    assert state['attempts'][0]['job_id']=='7001'
    assert state['attempts'][0]['status']=='FINISHED'
    assert state['attempts'][0]['outcome']=='RESOURCE_MIGRATION_HANDOFF'
    operations=json.loads((rt/'allocation_state.json').read_text())['tasks']
    assert operations['0']['phase']=='COMPLETE' and operations['1']['phase']=='NEW'


def test_lost_ack_old_accounting_remains_finished_and_next_sequence_is_monotone(migration,old_run):
    plan,path,rt,marker,_=old_run
    old=json.loads((rt/'state.json').read_text())
    old['attempts'].append({'sequence':5,'marker':marker,'name':marker+'-a5','job_id':None,
                            'status':'UNCERTAIN','intent_at':1792838100.})
    (rt/'state.json').write_text(json.dumps(old))
    rows=account(marker)+account(marker,state='TIMEOUT',job='7002',name=marker+'-a5',comment='')
    fake=FakeSlurm(['',''],[rows,rows])
    perform(migration,path,fake)
    handed=json.loads((rt/'state.json').read_text())
    attempts=handed['attempts']
    assert {'7001','7002'} <= {attempt.get('job_id') for attempt in attempts}
    assert next(attempt for attempt in attempts if attempt.get('job_id')=='7002')['sequence']==5
    assert all(attempt['status']=='FINISHED' and attempt['outcome']=='RESOURCE_MIGRATION_HANDOFF'
               for attempt in attempts)
    maximum=max(attempt['sequence'] for attempt in attempts)
    controller=runtime.SingleAllocationController(path,runner=fake,
        launch={'python':sys.executable,'package':str(PACKAGE)})
    controller.step()
    assert len(fake.of('sbatch'))==1
    assert controller.state['attempts'][-1]['sequence']==maximum+1
    assert '--partition=hpc_gpu' in fake.of('sbatch')[0]
    assert '--cpus-per-task=36' in fake.of('sbatch')[0]
    assert '--mem=128G' in fake.of('sbatch')[0]
    assert controller.state['controller_status']!='CANCELLED'


def test_observation_signal_and_accounting_failures_retry_without_losing_checkpoint(migration,old_run):
    plan,path,rt,marker,before=old_run
    fake=FakeSlurm([subprocess.TimeoutExpired('squeue',30),queue(marker),queue(marker),'','','',''],
                   [subprocess.TimeoutExpired('sacct',30),account(marker)])
    fake.signal_failures.append((1,'synthetic signal transport temporarily unavailable'))
    perform(migration,path,fake)
    assert len(fake.of('scancel'))==2
    assert len(fake.of('sacct'))==2
    assert len(fake.sleeps)>=3
    assert not fake.of('sbatch')
    assert all(item.read_bytes()==data for item,(data,_) in before.items())


def test_new_controller_qos_rejection_after_handoff_waits_then_retries_one_job(migration,old_run):
    plan,path,rt,marker,_=old_run
    fake=FakeSlurm(['',''],[account(marker)])
    perform(migration,path,fake)
    fake.submission_results.extend([(1,'sbatch: error: QOSMaxSubmitJobPerUserLimit\n'
        'sbatch: error: Batch job submission failed: Job violates accounting/QOS policy'),'900002\n'])
    controller=runtime.SingleAllocationController(path,runner=fake,clock=lambda:fake.now,
        launch={'python':sys.executable,'package':str(PACKAGE)})
    controller.step()
    assert len(fake.of('sbatch'))==1
    assert controller.state['controller_status']=='WAITING_FOR_CAPACITY'
    controller.step()
    assert len(fake.of('sbatch'))==1
    fake.now+=31
    controller.step()
    assert len(fake.of('sbatch'))==2
    assert not any(value.startswith('--array') for command in fake.of('sbatch') for value in command)


def test_pending_to_running_race_sends_checkpoint_signal_after_state_filtered_cancel(migration,old_run):
    plan,path,rt,marker,_=old_run
    fake=FakeSlurm([queue(marker,'PENDING'),queue(marker,'RUNNING'),'',''],[account(marker)])
    perform(migration,path,fake)
    assert ['scancel','--state=PENDING','7001'] in fake.of('scancel')
    assert ['scancel','--batch','--signal=USR1','7001'] in fake.of('scancel')


def test_queue_absence_with_active_accounting_is_not_handoff_completion(migration,old_run):
    plan,path,rt,marker,_=old_run
    fake=FakeSlurm(['','','',''],[account(marker,'RUNNING'),account(marker,'PREEMPTED')])
    perform(migration,path,fake)
    assert len(fake.of('sacct'))==2, 'Active owned accounting must be waited out before clearing STOP'
    assert fake.sleeps
    handed=json.loads((rt/'state.json').read_text())
    assert handed['attempts'][0]['slurm_state']=='PREEMPTED'


def test_unacknowledged_submission_waits_for_exact_accounting_name_before_archiving(migration,old_run):
    plan,path,rt,marker,_=old_run
    state=json.loads((rt/'state.json').read_text())
    state['attempts'].append({'sequence':5,'marker':marker,'name':marker+'-a5','job_id':None,
                              'status':'UNCERTAIN','intent_at':1792838100.})
    (rt/'state.json').write_text(json.dumps(state))
    fake=FakeSlurm(['','','',''],[account(marker),
        account(marker)+account(marker,'CANCELLED',job='7002',name=marker+'-a5',comment='')])
    perform(migration,path,fake)
    assert len(fake.of('sacct'))==2 and fake.sleeps
    attempts=json.loads((rt/'state.json').read_text())['attempts']
    recovered=next(attempt for attempt in attempts if attempt['sequence']==5)
    assert recovered['job_id']=='7002' and recovered['status']=='FINISHED'
    assert recovered['outcome']=='RESOURCE_MIGRATION_HANDOFF'


def test_acknowledged_job_without_terminal_account_or_batch_receipt_keeps_stop_and_retries(migration,old_run):
    plan,path,rt,marker,_=old_run
    (rt/'outcomes/7001.json').unlink()
    fake=FakeSlurm(['','','',''],['',account(marker,'TIMEOUT')])
    observations=[]
    def wait(seconds):
        observations.append((rt/'STOP').exists())
        assert json.loads((rt/'state.json').read_text())['controller_status']=='ALLOCATION_RUNNING'
        fake.sleep(seconds)
    handoff=migration.Handoff(path,runner=fake,sleeper=wait,clock=lambda:fake.now)
    handoff.quiesce()
    assert observations==[True] and len(fake.of('sacct'))==2
    assert json.loads((rt/'state.json').read_text())['attempts'][0]['slurm_state']=='TIMEOUT'


def test_known_job_terminal_batch_receipt_can_resolve_temporary_accounting_absence(migration,old_run):
    _,path,rt,_,_=old_run
    fake=FakeSlurm(['',''],[''])
    perform(migration,path,fake)
    state=json.loads((rt/'state.json').read_text())
    assert state['attempts'][0]['job_id']=='7001'
    assert state['attempts'][0]['outcome']=='RESOURCE_MIGRATION_HANDOFF'


@pytest.mark.parametrize('prepare_only',[False,True])
def test_prepare_only_never_stops_or_cancels_existing_jobs(migration,tmp_path,monkeypatch,prepare_only):
    root=tmp_path/'project'
    output=root/'runs/followup_20261024_r4'
    output.mkdir(parents=True)
    module=importlib.import_module('paper3r4.plan')
    plan=module.make_plan(root,output,cpus=16)
    path=output/'plan.json'
    path.write_text(json.dumps(plan,indent=7)+'\n')
    before=path.read_bytes()
    def forbidden(*args,**kwargs):
        pytest.fail('Preparation must not start a subprocess or touch the scheduler')
    monkeypatch.setattr(migration.subprocess,'run',forbidden)
    monkeypatch.setattr(migration.subprocess,'Popen',forbidden)
    args=['--root',str(root),'--output',str(output),'--python',sys.executable]
    if prepare_only:
        args.append('--prepare-only')
    assert migration.main(args)==0
    assert path.read_bytes()==before
    assert not (output/'runtime/STOP').exists() and not (output/'STOP').exists()
