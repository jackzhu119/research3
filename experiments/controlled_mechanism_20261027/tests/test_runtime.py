from __future__ import annotations
import errno
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper3m12 import common as c
from paper3m12 import runtime as r


def make_plan(tmp_path, tasks=None):
    value = dict(schema_version='paper3-m12-plan-v1', root=str(tmp_path),
                 output=str(tmp_path/'output'), tasks=tasks or [dict(key='first', kind='paired_block', dependencies=[])],
                 recipe_sha256='a'*64, synthetic_fixture=True)
    path = tmp_path/'plan.json'; c.atomic_json(path, value)
    return path, c.load_plan(path)[1]


def complete(plan, task, **values):
    folder = c.task_folder(plan, task); folder.mkdir(parents=True, exist_ok=True)
    payload = folder/'metrics.csv'; payload.write_text('method,BA\nanchor,0.5\n')
    relative = str(payload.relative_to(Path(plan['output'])))
    result = dict(status='COMPLETE', task_key=task['key'], task_identity=c.digest(task),
                  recipe_sha256=plan.get('recipe_sha256'),
                  files={relative:dict(path=relative, bytes=payload.stat().st_size, sha256=c.sha_file(payload))}, **values)
    c.atomic_json(folder/'result.json', result)
    return result


def test_json_reads_only_actual_size_not_large_limit(tmp_path, monkeypatch):
    p=tmp_path/'small.json';p.write_bytes(b'{"count":1}')
    reads=[]; original=Path.open
    class Traced:
        def __init__(self, real):self.real=real
        def __enter__(self):self.real.__enter__();return self
        def __exit__(self,*args):return self.real.__exit__(*args)
        def fileno(self):return self.real.fileno()
        def read(self,n):reads.append(n);return self.real.read(n)
    def traced(self,*args,**kwargs):
        value=original(self,*args,**kwargs)
        return Traced(value) if self==p and args and args[0]=='rb' else value
    monkeypatch.setattr(Path,'open',traced)
    assert c.read_json(p,max_bytes=64*1024*1024)=={'count':1}
    assert reads==[12]


def test_shared_filesystem_replace_loss_retries_private_unique_staging(tmp_path,monkeypatch):
    target=tmp_path/'state.json'; real=os.replace;paths=[]
    def replace(source,destination):
        paths.append(Path(source))
        if len(paths)==1:
            Path(source).unlink();raise FileNotFoundError(errno.ENOENT,'simulated staging loss')
        return real(source,destination)
    monkeypatch.setattr(c.os,'replace',replace)
    c.atomic_json(target,{'step':3},sleep=lambda _:None)
    assert c.read_json(target)=={'step':3}
    assert len(set(paths))==2 and all(p.parent.name=='.m12-atomic' for p in paths)
    assert (target.parent/'.m12-atomic').stat().st_mode&0o777==0o700


def test_ambiguous_replace_ack_accepts_only_exact_target(tmp_path,monkeypatch):
    target=tmp_path/'result.json';real=os.replace
    def replaced(source,destination):real(source,destination);raise OSError(errno.EIO,'lost acknowledgement')
    monkeypatch.setattr(c.os,'replace',replaced)
    c.atomic_json(target,{'complete':True},sleep=lambda _:None)
    assert c.read_json(target)=={'complete':True}


def test_permanent_disk_failure_propagates_but_telemetry_never_controls_fit(tmp_path,monkeypatch):
    def broken(*a,**k):raise OSError(errno.ENOSPC,'full disk')
    monkeypatch.setattr(c,'atomic_bytes',broken)
    with pytest.raises(OSError):c.atomic_json(tmp_path/'science.json',{'a':1})
    assert c.best_effort_json(tmp_path/'status.json',{'a':1}) is False


def test_start_requires_explicit_submit_and_has_no_side_effect(tmp_path):
    path,plan=make_plan(tmp_path)
    with pytest.raises(ValueError,match='--submit'):r.start(path)
    assert not r.runtime_dir(plan).exists()


def test_first_controller_observation_submits_zero_then_four_exact_resources(tmp_path):
    path,plan=make_plan(tmp_path);calls=[]
    def runner(command,**kw):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='' if command[0]=='squeue' else str(9000+len(calls)),stderr='')
    first=r.controller_loop(path,submit=True,token='fresh',runner=runner,once=True)
    assert [x[0] for x in calls]==['squeue']
    ready=c.read_json(r.runtime_dir(plan)/'controller-ready.json')
    assert ready['first_slurm_observation_completed'] is True and ready['launch_token']=='fresh'
    second=r.scheduler_step(path,plan,submit=True,runner=runner)
    submissions=[x for x in calls if x[0]=='sbatch']
    assert len(submissions)==4 and second['inflight_jobs']==4
    for command in submissions:
        assert '--gres=gpu:1' in command and '--cpus-per-task=16' in command
        assert '--mem=64G' in command and '--partition=hpc_gpu' in command
        script=Path(command[-1]).read_text()
        assert ' worker ' in script and '--slot' in script
        assert 'scancel' not in script and 'kill' not in script and 'ulimit' not in script


def test_queue_failure_never_claims_ready_or_submits(tmp_path):
    path,plan=make_plan(tmp_path);calls=[]
    def runner(command,**kw):calls.append(command);return SimpleNamespace(returncode=1,stdout='',stderr='temporary unavailable')
    result=r.controller_loop(path,submit=True,runner=runner,once=True)
    assert result['status']=='M12_QUEUE_OBSERVATION_PENDING'
    assert not (r.runtime_dir(plan)/'controller-ready.json').exists()
    assert all(command[0]=='squeue' for command in calls)


def test_restart_adopts_only_own_four_jobs_leaving_other_rounds_untouched(tmp_path):
    path,plan=make_plan(tmp_path);calls=[]
    lines=['772906|paper3r1p-old-s0-a2|RUNNING|gpu1|gpu1',
           '775288|paper3r11p-old-s2-a1|RUNNING|gpu2|gpu2']
    lines += ['%d|%s-s%d-a1|RUNNING|gpu3|gpu3'%(8000+i,r.marker(plan),i) for i in range(4)]
    def runner(command,**kw):calls.append(command);return SimpleNamespace(returncode=0,stdout='\n'.join(lines),stderr='')
    observed=r.scheduler_step(path,plan,submit=True,runner=runner)
    assert observed['inflight_jobs']==4 and len(observed['jobs'])==4
    assert {x['job_id'] for x in observed['jobs']}=={str(8000+i) for i in range(4)}
    assert all(command[0]=='squeue' for command in calls)


def test_ambiguous_submit_is_not_blindly_retried(tmp_path):
    path,plan=make_plan(tmp_path);calls=[]
    def runner(command,**kw):
        calls.append(command)
        if command[0]=='sbatch':raise subprocess.TimeoutExpired(command,1)
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    r.scheduler_step(path,plan,submit=True,runner=runner)
    assert sum(x[0]=='sbatch' for x in calls)==4
    r.scheduler_step(path,plan,submit=True,runner=runner)
    assert sum(x[0]=='sbatch' for x in calls)==4


def test_existing_scientific_completion_reused_without_running_fit(tmp_path):
    path,plan=make_plan(tmp_path);complete(plan,plan['tasks'][0])
    def forbidden(*a,**k):pytest.fail('completed task was refitted')
    assert r.worker_loop(path,0,run_task=forbidden,once=True,device='cpu')==0
    assert r.status(plan)['complete_tasks']==1


def test_worker_persists_science_despite_progress_write_failure(tmp_path,monkeypatch):
    path,plan=make_plan(tmp_path)
    monkeypatch.setattr(r,'best_effort_json',lambda *a,**k:False)
    def science(plan,task,**kwargs):
        kwargs['progress'](outer_update=3,stage='fitting')
        return complete(plan,task)
    assert r.worker_loop(path,0,run_task=science,once=True,device='cpu')==0
    assert r.status(plan)['complete_tasks']==1


def test_persistent_worker_completes_multiple_tasks_in_same_process(tmp_path):
    tasks=[dict(key='same-process-'+str(i),kind='paired_block',dependencies=[]) for i in range(4)]
    path,plan=make_plan(tmp_path,tasks);seen=[]
    def science(plan,task,**kwargs):
        seen.append((task['key'],os.getpid()))
        kwargs['progress'](session=len(seen),stage='support_fit',numerical_advance=True)
        return complete(plan,task)
    assert r.worker_loop(path,0,run_task=science,device='cpu')==0
    assert len(seen)==4 and len({pid for key,pid in seen})==1
    assert r.status(plan)['complete_tasks']==4


def test_failed_task_does_not_interrupt_independent_task(tmp_path):
    tasks=[dict(key='bad',kind='paired_block',dependencies=[]),dict(key='good',kind='paired_block',dependencies=[])]
    path,plan=make_plan(tmp_path,tasks)
    def science(plan,task,**kwargs):
        if task['key']=='bad':raise ValueError('fixed scientific identity mismatch')
        return complete(plan,task)
    r.worker_loop(path,0,run_task=science,once=True,device='cpu')
    r.worker_loop(path,1,run_task=science,once=True,device='cpu')
    state=r.status(plan)
    assert state['complete_tasks']==1 and state['blocked_tasks']==1
    assert c.read_json(r._state_path(plan,tasks[0]))['phase']=='BLOCKED'


def test_temporary_operational_read_problem_does_not_poison_dependency_or_other_work(tmp_path,monkeypatch):
    tasks=[dict(key='temporary',kind='paired_block',dependencies=[]),
           dict(key='dependent',kind='paired_block',dependencies=['temporary']),
           dict(key='unrelated',kind='paired_block',dependencies=[])]
    _,plan=make_plan(tmp_path,tasks)
    c.atomic_json(r._state_path(plan,tasks[0]),{'phase':'NEW'})
    real=r.read_json
    def stale(path,*args,**kwargs):
        if Path(path)==r._state_path(plan,tasks[0]):raise OSError(errno.ESTALE,'temporary shared filesystem read')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(r,'read_json',stale)
    claimed=r.claim_next(plan,0,'own-allocation')
    assert claimed[0]['key']=='unrelated'
    claimed[2].close()
    assert not r._state_path(plan,tasks[1]).exists()
    assert r.status(plan)['blocked_tasks']==0
    monkeypatch.setattr(r,'read_json',real)
    assert r.task_states(plan)['temporary']['phase']=='NEW'


def test_transient_fit_errors_bounded_after_three_attempts(tmp_path):
    path,plan=make_plan(tmp_path);task=plan['tasks'][0]
    def science(*a,**k):raise RuntimeError('temporary native resource failure')
    for _ in range(3):
        r.worker_loop(path,0,run_task=science,once=True,device='cpu')
        state=c.read_json(r._state_path(plan,task));state['retry_after']=0;c.atomic_json(r._state_path(plan,task),state)
    assert state['phase']=='BLOCKED' and state['failures']==3
    assert r.claim_next(plan,1,'fixture') is None


def test_pause_keeps_checkpoint_without_incrementing_failure_or_claiming_complete(tmp_path):
    path,plan=make_plan(tmp_path);task=plan['tasks'][0]
    class PauseRequested(Exception):pass
    def science(plan,task,**kwargs):
        c.atomic_json(c.task_folder(plan,task)/'checkpoint.json',{'step':10})
        raise PauseRequested('between durable units')
    assert r.worker_loop(path,0,run_task=science,once=True,device='cpu')==75
    row=c.read_json(r._state_path(plan,task))
    assert row['phase']=='PAUSED' and row['failures']==0
    assert r.status(plan)['complete_tasks']==0
    assert c.read_json(c.task_folder(plan,task)/'checkpoint.json')=={'step':10}


def test_negative_research_decision_skips_descendants_and_runs_honest_final_report(tmp_path):
    tasks=[dict(key='decision',kind='decision',dependencies=[]),
           dict(key='new-cohort',kind='paired_block',dependencies=['decision'],gate={'task':'decision','field':'decision','equals':'GO'}),
           dict(key='final',kind='final_report',dependencies=['decision','new-cohort'],allow_skipped_dependencies=True)]
    path,plan=make_plan(tmp_path,tasks);called=[]
    def science(plan,task,**kwargs):
        called.append(task['key']);return complete(plan,task,**({'decision':'STOP'} if task['key']=='decision' else {}))
    r.worker_loop(path,0,run_task=science,once=True,device='cpu')
    r.worker_loop(path,1,run_task=science,once=True,device='cpu')
    value=r.status(plan)
    assert called==['decision','final']
    assert value['status']=='STOPPED_BY_PREDECLARED_RESEARCH_RULE'
    assert value['complete_tasks']==2 and value['skipped_by_decision']==1


def test_missing_gate_decision_is_blocked_not_fabricated_research_stop(tmp_path):
    tasks=[dict(key='decision',kind='decision',dependencies=[]),
           dict(key='target',kind='paired_block',dependencies=['decision'],
                gate={'task':'decision','field':'decision','equals':'GO'})]
    _,plan=make_plan(tmp_path,tasks)
    complete(plan,tasks[0])  # Deliberately malformed control result lacks decision.
    assert r.claim_next(plan,0,'fixture') is None
    value=r.status(plan)
    assert value['skipped_by_decision']==0 and value['blocked_tasks']==1
    assert value['status']=='PARTIAL_BLOCKED'


def test_gate_cycles_are_rejected_before_any_allocation(tmp_path):
    tasks=[dict(key='a',kind='decision',dependencies=[],gate={'task':'b','field':'decision','equals':'GO'}),
           dict(key='b',kind='decision',dependencies=['a'])]
    value=dict(root=str(tmp_path),output=str(tmp_path/'out'),tasks=tasks)
    path=tmp_path/'plan.json';c.atomic_json(path,value)
    with pytest.raises(ValueError,match='cycle'):c.load_plan(path)


def test_original_three_claims_do_not_duplicate_and_stale_process_loss_is_bounded(tmp_path):
    tasks=[dict(key='t'+str(i),kind='paired_block',dependencies=[]) for i in range(4)]
    _,plan=make_plan(tmp_path,tasks);leases=[]
    try:
        for slot in range(4):leases.append(r.claim_next(plan,slot,str(8000+slot)))
        assert len({x[0]['key'] for x in leases})==4
        assert r.claim_next(plan,0,'new') is None
    finally:
        for _,_,lease in leases:lease.close()
    # No worker receipt exists: the old lock has gone, so this is a recovery.
    recovered=r.claim_next(plan,0,'replacement')
    assert recovered[1]['failures']==1 and recovered[1]['attempt']==2
    recovered[2].close()


def test_receipt_published_after_snapshot_is_rechecked_under_acquired_lease(tmp_path,monkeypatch):
    _,plan=make_plan(tmp_path);task=plan['tasks'][0]
    complete(plan,task)
    # Deterministic handover race: initial state was captured before the other
    # worker finished, while its actual receipt is visible once its lease opens.
    monkeypatch.setattr(r,'task_states',lambda plan:{task['key']:dict(phase='RUNNING',failures=0,attempt=1)})
    assert r.claim_next(plan,1,'second-live-allocation') is None
    assert not r._state_path(plan,task).exists()


def test_wrong_task_result_cannot_be_used_as_scientific_completion(tmp_path):
    _,plan=make_plan(tmp_path);task=plan['tasks'][0]
    complete(plan,task)
    result=c.read_json(c.task_folder(plan,task)/'result.json');result['task_identity']='wrong'
    c.atomic_json(c.task_folder(plan,task)/'result.json',result)
    assert c.verify_result(plan,task) is None and r.status(plan)['complete_tasks']==0


def test_entry_status_isolated_imports_no_torch_or_numpy(tmp_path):
    path,plan=make_plan(tmp_path)
    harness="import runpy,sys;sys.argv=[sys.argv[1], 'status','--plan',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')"
    run=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'paper3m12/entry.py'),'status','--plan',str(path)],
                       capture_output=True,text=True,timeout=10)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['total_tasks']==1
    script="import sys;sys.path.insert(0,sys.argv[1]);import paper3m12.runtime;assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules;print('CPU_CONTROLLER_IMPORT_OK')"
    checked=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(ROOT)],capture_output=True,text=True,timeout=10)
    assert checked.returncode==0 and checked.stdout.strip()=='CPU_CONTROLLER_IMPORT_OK'


def test_four_real_cpu_workers_claim_each_scientific_unit_exactly_once(tmp_path):
    tasks=[dict(key='unit-'+str(i),kind='paired_block',dependencies=[]) for i in range(12)]
    path,plan=make_plan(tmp_path,tasks);audit=tmp_path/'actual-executions.jsonl'
    def run_slot(slot):
        def science(plan,task,**kwargs):
            with c.lock(tmp_path/'audit.lock'):
                with audit.open('a') as handle:
                    handle.write(json.dumps({'key':task['key'],'pid':os.getpid()})+'\n')
            time.sleep(.01)
            complete(plan,task)
        raise SystemExit(r.worker_loop(path,slot,run_task=science,device='cpu'))
    ctx=multiprocessing.get_context('fork')
    processes=[ctx.Process(target=run_slot,args=(slot,)) for slot in range(4)]
    for process in processes:process.start()
    for process in processes:process.join(timeout=6)
    assert all(process.exitcode==0 for process in processes)
    rows=[json.loads(line) for line in audit.read_text().splitlines()]
    assert len(rows)==12 and len({row['key'] for row in rows})==12
    assert r.status(plan)['complete_tasks']==12


def test_real_slurm_pause_signal_waits_for_durable_unit_without_forced_unwind(tmp_path):
    path,plan=make_plan(tmp_path)
    script=r'''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from paper3m12 import common as c
from paper3m12 import runtime as r
class PauseRequested(Exception):pass
def science(plan,task,**kwargs):
 folder=c.task_folder(plan,task);folder.mkdir(parents=True,exist_ok=True)
 (folder/'started').write_text('owned CPU fixture')
 for step in range(1,6):
  time.sleep(.04)
  c.atomic_json(folder/'checkpoint.json',{'step':step})
 kwargs['progress'](step=5,stage='unit_committed')
 if kwargs['stop'].is_set():raise PauseRequested('durable unit committed before pause')
 raise AssertionError('parent fixture must have sent USR1')
raise SystemExit(r.worker_loop(sys.argv[2],0,run_task=science,device='cpu'))
'''
    process=subprocess.Popen([sys.executable,'-I','-S','-B','-c',script,str(ROOT),str(path)],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    folder=c.task_folder(plan,plan['tasks'][0]);until=time.monotonic()+4
    while not (folder/'started').exists() and time.monotonic()<until:time.sleep(.005)
    assert (folder/'started').exists()
    os.kill(process.pid, __import__('signal').SIGUSR1)
    stdout,stderr=process.communicate(timeout=5)
    assert process.returncode==75,stderr
    assert c.read_json(folder/'checkpoint.json')=={'step':5}
    row=c.read_json(r._state_path(plan,plan['tasks'][0]))
    assert row['phase']=='PAUSED' and row['failures']==0
    assert not (folder/'result.json').exists()


def test_real_controller_start_is_idempotent_with_local_slurm_command_fixtures(tmp_path,monkeypatch):
    path,plan=make_plan(tmp_path);binary=tmp_path/'software-only-slurm';binary.mkdir()
    for name,body in {
        'squeue':'#!/bin/sh\nexit 0\n',
        'sbatch':'#!/bin/sh\nprintf "8000\\n"\n',
        'sacct':'#!/bin/sh\nprintf "8000|PENDING|0:0\\n"\n',
    }.items():
        target=binary/name;target.write_text(body);target.chmod(0o700)
    monkeypatch.setenv('PATH',str(binary)+os.pathsep+os.environ['PATH'])
    launched=None
    try:
        first=r.start(path,submit=True)
        launched=first['pid']
        assert first['status']=='M12_CONTROLLER_READY'
        assert first['sidecars']['status']=='SIDECARS_DISABLED_SOFTWARE_FIXTURE'
        second=r.start(path,submit=True)
        assert second['status']=='EXISTING_M12_CONTROLLER_READY'
        assert second['pid']==first['pid']
        assert second['scientific_jobs_restarted']==0
        ready=c.read_json(r.runtime_dir(plan)/'controller-ready.json')
        assert ready['launch_token']==first['launch_token']
        assert ready['training_progress_claimed'] is False
    finally:
        if launched:
            # Only this test-owned CPU manager; no scientific or Slurm process.
            try:os.kill(launched,__import__('signal').SIGTERM)
            except ProcessLookupError:pass
            until=time.monotonic()+3
            while r._controller_identity(launched,path,first['launch_token']) and time.monotonic()<until:
                time.sleep(.01)


def test_sidecar_start_failure_is_reported_without_controlling_training(tmp_path,monkeypatch):
    path,plan=make_plan(tmp_path);plan['start_sidecars']=True
    def unavailable(name):raise OSError(errno.EIO,'sidecar launch temporarily unavailable')
    monkeypatch.setattr(r.importlib,'import_module',unavailable)
    result=r._start_sidecars(path,plan)
    assert result['status']=='SIDECAR_START_RETRY_NEEDED' and result['training_controlled'] is False
