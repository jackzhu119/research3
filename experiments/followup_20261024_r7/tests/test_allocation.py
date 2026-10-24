"""Single-allocation scheduling tests, including actual isolated child processes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from paper3r7.allocation import Allocation, allocated_devices, cpu_groups, worker_command
from paper3r7.io import put, read
from paper3r7 import allocation as allocation_module


def test_gpu_devices_preserve_allocated_uuid_mig_and_numeric_tokens():
    tokens = ['GPU-abc', 'MIG-GPU-def/1/0', '4']
    actual, source = allocated_devices({'CUDA_VISIBLE_DEVICES': ','.join(tokens), 'SLURM_JOB_GPUS': '0,1,2,3'}, 3)
    assert actual == tokens and source == 'CUDA_VISIBLE_DEVICES'
    assert allocated_devices({'SLURM_STEP_GPUS': '2-4'}, 3) == (['2', '3', '4'], 'SLURM_STEP_GPUS')
    assert allocated_devices({'SLURM_JOB_GPUS': 'GPU-a,GPU-b,GPU-c'}, 3)[0] == ['GPU-a', 'GPU-b', 'GPU-c']


@pytest.mark.parametrize('environment', [{}, {'CUDA_VISIBLE_DEVICES': '-1'},
    {'CUDA_VISIBLE_DEVICES': '', 'SLURM_JOB_GPUS': '0,1,2'},
    {'CUDA_VISIBLE_DEVICES': '-1', 'SLURM_JOB_GPUS': '0,1,2'},
    {'CUDA_VISIBLE_DEVICES': '0,1'}, {'CUDA_VISIBLE_DEVICES': '0,0,1'},
    {'CUDA_VISIBLE_DEVICES': 'all'}, {'SLURM_JOB_GPUS': '9-2'}])
def test_no_node_enumeration_or_unallocated_gpu_fallback(environment):
    with pytest.raises(ValueError):
        allocated_devices(environment, 3)


def test_cpu_groups_nonoverlap_and_explicit_observed_shortfall():
    assert cpu_groups(range(8, 44), 3, 12) == [list(range(8, 20)), list(range(20, 32)), list(range(32, 44))]
    assert cpu_groups(range(11), 3, 12) == [list(range(3)), list(range(3, 6)), list(range(6, 9))]
    with pytest.raises(ValueError):
        cpu_groups([0, 1], 3, 12)


def test_worker_command_is_one_visible_gpu_and_isolated_entry():
    command = worker_command('/venv/bin/python', '/run/plan.json', 7)
    assert command[:5] == ['/venv/bin/python', '-I', '-S', '-B', '-u']
    assert command[-6:] == ['--plan', '/run/plan.json', '--task', '7', '--device', 'cuda:0']
    assert 'paper3r7/entry.py' in command[5]


DUMMY = r'''
import json,os,pathlib,signal,sys,time
plan=json.loads(pathlib.Path(sys.argv[1]).read_text());index=int(sys.argv[2]);task=plan['tasks'][index]
folder=pathlib.Path(plan['output'])/task['key'];folder.mkdir(parents=True,exist_ok=True)
def write(name,data):
    (folder/name).write_text(json.dumps(data))
def stop(*args):
    write('checkpoint.json',{'saved':True});sys.exit(75)
signal.signal(signal.SIGUSR1,stop)
write('observation.json',{'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
    'slot':os.environ.get('PAPER3R7_GPU_SLOT'),'affinity':sorted(os.sched_getaffinity(0)),
    'pid':os.getpid(),'started':time.time()})
write('progress.json',{'status':'TRAINING','stage':'synthetic','outer_update':1,'total_updates':3})
if task.get('action')=='wait':
    while True:time.sleep(.01)
time.sleep(task.get('delay',.03))
if task.get('action')=='fail':sys.exit(1)
if task.get('action')=='zero_without_result':sys.exit(0)
if task.get('action')=='pause_once' and not (folder/'checkpoint.json').exists():stop()
write('result.json',{'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],'finished':time.time()})
'''


@pytest.fixture
def project(tmp_path):
    script = tmp_path/'dummy.py';script.write_text(DUMMY)
    def create(tasks):
        output = tmp_path/'output';output.mkdir(exist_ok=True)
        plan={'recipe_sha256':'synthetic-allocation-recipe','output':str(output),'tasks':tasks}
        path=output/'plan.json';put(path,plan)
        affinity=sorted(os.sched_getaffinity(0))[:6]
        if len(affinity)<3:pytest.skip('Requires three CPU affinity IDs for real isolated process smoke')
        cpus=max(1,len(affinity)//3)
        def command(python,plan_path,index,**kwargs):
            return [python,'-I','-S','-B','-u',str(script),str(plan_path),str(index)]
        def make(**changes):
            return Allocation(path,python=sys.executable,slots=3,cpus_per_slot=cpus,
                environment={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-a,MIG-GPU-b/1/0,2'},
                affinity=affinity,command_factory=command,**changes)
        return plan,path,make
    return create


def test_real_three_slot_dependency_scheduling_isolation_and_scientific_failure(project):
    tasks=[{'key':'basis','delay':.2},{'key':'bad','action':'fail','delay':.05},
        {'key':'other','delay':.15},{'key':'consumer','requires':['basis']},
        {'key':'blocked','requires':['bad']},{'key':'sibling'}]
    plan,path,make=project(tasks);allocation=make()
    assert allocation.run(.01)==2
    state=read(Path(plan['output'])/'runtime/allocation_state.json')
    assert [state['tasks'][str(i)]['phase'] for i in range(6)] == [
        'COMPLETE','FAILED','COMPLETE','COMPLETE','WAITING_INPUT','COMPLETE']
    assert state['failed_tasks']==1 and state['waiting_tasks']==1
    output=Path(plan['output'])
    observations=[read(output/task['key']/'observation.json') for task in tasks if task['key']!='blocked']
    for observation in observations:
        slot=int(observation['slot'])
        assert observation['gpu']==['GPU-a','MIG-GPU-b/1/0','2'][slot]
        assert observation['affinity']==allocation.cpu_affinities[slot]
    assert all(not (set(a)&set(b)) for i,a in enumerate(allocation.cpu_affinities)
               for b in allocation.cpu_affinities[i+1:])
    initial=[read(output/key/'observation.json')['started'] for key in ['basis','bad','other']]
    assert max(initial)<read(output/'basis/result.json')['finished']
    assert read(output/'consumer/observation.json')['started']>=read(output/'basis/result.json')['finished']
    assert (output/'runtime/errors/task-1.json').is_file()
    assert len(list((output/'logs').glob('*.out')))==5
    # The scheduler public heartbeat exposes logical GPU slots, never UUIDs.
    public=(output/'allocation_progress.json').read_text()
    assert 'GPU-a' not in public and 'MIG-GPU-b' not in public
    assert read(output/'allocation_progress.json')['complete_tasks']==4
    stamps={key:(output/key/'result.json').read_bytes() for key in ['basis','other','consumer','sibling']}
    again=make();assert again.run(.01)==2
    assert all((output/key/'result.json').read_bytes()==data for key,data in stamps.items())
    assert read(output/'runtime/allocation_state.json')['tasks']['1']['attempt']==1


def test_checkpoint_exit_stops_other_active_workers_then_resumes_without_completed_replay(project):
    plan,path,make=project([{'key':'already'},{'key':'pause','action':'pause_once','delay':.15},
                            {'key':'slow','delay':1}])
    allocation=make();assert allocation.run(.01)==75
    output=Path(plan['output']);state=read(output/'runtime/allocation_state.json')
    assert state['allocation_status']=='PAUSED_CHECKPOINT_SAVED'
    assert read(output/'already/result.json')['status']=='COMPLETE'
    assert read(output/'pause/checkpoint.json')['saved']
    assert read(output/'slow/checkpoint.json')['saved']
    saved=(output/'already/result.json').read_bytes()
    resumed=make();assert resumed.run(.01)==0
    assert (output/'already/result.json').read_bytes()==saved
    final=read(output/'runtime/allocation_state.json')
    assert final['complete_tasks']==3
    assert final['tasks']['0']['attempt']==1
    assert final['tasks']['1']['attempt']==2 and final['tasks']['2']['attempt']==2


def test_signal_forwarding_waits_for_real_children_checkpoint(project):
    plan,path,make=project([{'key':str(i),'action':'wait'} for i in range(3)])
    allocation=make();allocation.acquire()
    try:
        assert allocation.cycle() is None
        output=Path(plan['output']);deadline=time.monotonic()+5
        while not all((output/str(i)/'observation.json').exists() for i in range(3)):
            assert time.monotonic()<deadline;time.sleep(.01)
        allocation.request_stop('SIGUSR1')
        while True:
            code=allocation.cycle()
            if code is not None:break
            assert time.monotonic()<deadline;time.sleep(.01)
        assert code==75
        assert all(read(output/str(i)/'checkpoint.json')['saved'] for i in range(3))
        assert not allocation.running
    finally:allocation.close()


def test_completed_marker_is_reused_without_worker_start(project):
    plan,path,make=project([{'key':'complete'}])
    output=Path(plan['output']);put(output/'complete/result.json',{'status':'COMPLETE'})
    allocation=make();assert allocation.run(.01)==0
    assert not (output/'complete/observation.json').exists()
    state=read(output/'runtime/allocation_state.json');assert state['tasks']['0']['attempt']==0


def test_exit_zero_without_complete_marker_is_not_scientific_completion(project):
    plan,path,make=project([{'key':'missing','action':'zero_without_result'}, {'key':'good'}])
    allocation=make();assert allocation.run(.01)==2
    state=read(Path(plan['output'])/'runtime/allocation_state.json')
    assert state['tasks']['0']['phase']=='FAILED'
    assert state['tasks']['1']['phase']=='COMPLETE'
    assert state['complete_tasks']==1


def test_idle_poll_does_not_rescan_all_result_files_or_fsync_state(project,monkeypatch):
    plan,path,make=project([{'key':'wait','action':'wait'}])
    allocation=make();allocation.acquire()
    try:
        assert allocation.cycle() is None
        def forbidden(*args,**kwargs):pytest.fail('Idle polling re-read completion markers')
        monkeypatch.setattr(allocation,'_result_complete',forbidden)
        stamp=allocation.state_path.stat().st_mtime_ns
        for _ in range(5):assert allocation.cycle() is None
        assert allocation.state_path.stat().st_mtime_ns==stamp
    finally:
        # Wait until the real dummy worker installs its cooperative handler.
        folder=Path(plan['output'])/'wait';deadline=time.monotonic()+5
        while not (folder/'observation.json').exists():
            assert time.monotonic()<deadline;time.sleep(.01)
        monkeypatch.undo()
        allocation.request_stop('test_cleanup')
        while allocation.running:
            allocation.cycle();time.sleep(.01)
        allocation.close()


def test_single_allocation_lock_rejects_second_owner(project):
    plan,path,make=project([{'key':'one'}])
    first=make();first.acquire()
    try:
        with pytest.raises(RuntimeError,match='already owns'):
            make().run(.01)
    finally:first.close()


def test_invalid_gpu_mapping_preserves_existing_results(project):
    plan,path,make=project([{'key':'done'}]);output=Path(plan['output'])
    put(output/'done/result.json',{'status':'COMPLETE'})
    before=(output/'done/result.json').read_bytes()
    with pytest.raises(ValueError,match='No allocation-specific'):
        Allocation(path,slots=3,environment={},affinity=range(36))
    assert (output/'done/result.json').read_bytes()==before


def test_early_import_signal_requests_checkpoint_before_starting_workers(project,monkeypatch):
    plan,path,make=project([{'key':'never-start'}])
    monkeypatch.setenv('PAPER3R7_STOP_REQUESTED','1')
    allocation=make();assert allocation.run(.01)==75
    assert not (Path(plan['output'])/'never-start/observation.json').exists()
    assert read(Path(plan['output'])/'allocation_progress.json')['stop_reason']=='early_signal'


def test_actual_isolated_allocation_entry_cpu_smoke_reuses_saved_results(project):
    plan,path,make=project([{'key':f'done-{index}'} for index in range(3)])
    output=Path(plan['output'])
    for task in plan['tasks']:put(output/task['key']/'result.json',{'status':'COMPLETE'})
    entry=Path(__file__).resolve().parents[1]/'paper3r7/entry.py'
    env=dict(os.environ);env.pop('PAPER3R7_STOP_REQUESTED',None)
    result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(entry),'allocation',
        '--plan',str(path),'--python',sys.executable,'--slots','3','--cpus-per-slot','1','--device','cpu'],
        env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ALLOCATION_PROGRESS' in result.stdout
    state=read(output/'runtime/allocation_state.json')
    assert state['allocation_status']=='COMPLETE' and state['complete_tasks']==3
    assert state['gpu_mapping_source']=='CPU_TEST_ONLY'
    assert all(row['attempt']==0 for row in state['tasks'].values())


def test_observation_write_failure_never_cancels_real_training_children(project,monkeypatch):
    plan,path,make=project([{'key':'one'},{'key':'two'},{'key':'three'}])
    original=allocation_module.put
    failures=[]
    def fail_public(path,value):
        if Path(path).name=='allocation_progress.json':
            failures.append(path);raise OSError('synthetic observation filesystem failure')
        return original(path,value)
    monkeypatch.setattr(allocation_module,'put',fail_public)
    allocation=make();assert allocation.run(.01)==0
    assert failures
    assert all(read(Path(plan['output'])/task['key']/'result.json')['status']=='COMPLETE' for task in plan['tasks'])
