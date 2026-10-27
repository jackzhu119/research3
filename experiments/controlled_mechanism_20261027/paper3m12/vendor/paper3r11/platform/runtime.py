"""R11-only durable task execution and explicit Slurm submission boundary.

No numerical library, network client or credential is imported by this module.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from .common import (atomic, digest, emit, load_plan, locked, read, seal_result, sha,
                     atomic_bytes, task_folder, task_identity, utc, verify_result)
from .common import verify_descriptor
from .scheduler import FourJobScheduler, write_worker_outcome

MAX_ATTEMPTS = 3
_PLAN_INDEX={}
_RECOVERED=set()
PROGRESS_FIELDS = ('status','stage','phase','outer_update','outer_updates_completed',
                   'total_updates','session','sessions_total','completed_sessions',
                   'budget','iteration','iterations','model_update','delta_norm',
                   'updated_utc','fallback_sessions','fit_count','predicted_queries',
                   'numerical_update_utc','attempt_token')
BLOCKED = ('QUARANTINED','BLOCKED_CONFIGURATION','BLOCKED_DEPENDENCY')


def runtime_dir(plan):
    return Path(plan['output'])/'runtime'


def context(plan_path, *, python=None, account=None):
    path, plan = load_plan(plan_path)
    runtime = runtime_dir(plan); runtime.mkdir(parents=True, exist_ok=True)
    package = Path(plan['package'])
    entry = package/'paper3r11/platform/entry.py'
    if not entry.is_file():
        raise ValueError('R11 isolated entry is absent from the installed package')
    python = str(Path(python or plan.get('python') or sys.executable).absolute())
    resources = dict(plan.get('resources', {}))
    if resources.get('jobs', 4)!=4 or resources.get('gpus_per_job', 1)!=1:
        raise ValueError('R11 uses exactly four single-GPU worker allocations')
    if resources.get('cpus_per_job',16)!=16 or resources.get('memory_per_job','64G')!='64G':
        raise ValueError('R11 workers require 16 CPUs and 64G memory each')
    if resources.get('partition','hpc_gpu')!='hpc_gpu':
        raise ValueError('R11 stage is configured for hpc_gpu')
    command = [python,'-I','-S','-B','-u',str(entry),'worker','--plan',str(path),
               '--slot','{slot}','--device','cuda:0']
    launch = {'round':11,'partition':'hpc_gpu','cpus':16,'memories':['64G']*4,
              'gres':'gpu:1','time_limit':resources.get('time_limit','5-03:00:00'),
              'poll_seconds':resources.get('poll_seconds',30),
              'account':account or resources.get('account') or getpass.getuser(),
              'user':getpass.getuser(),'campaign_identity':plan['recipe_sha256'],
              'worker_command':command,'python':python,'package':str(package),
              'root':plan['root'],'output':plan['output'],'plan':str(path),
              'plan_sha256':sha(path),'runtime':str(runtime),'exclude':''}
    return {'plan_path':path,'plan':plan,'runtime':runtime,'launch':launch}


def plan_index(plan):
    recipe=plan['recipe_sha256']
    if recipe not in _PLAN_INDEX:
        tasks={t['key']:t for t in plan['tasks']}
        _PLAN_INDEX[recipe]=(tasks,{key:task_identity(plan,t) for key,t in tasks.items()})
    return _PLAN_INDEX[recipe]


def completed_task_rows(plan,state=None):
    """Durable, previously sealed completion cache; no task-directory I/O."""
    state=_state(plan) if state is None else state;tasks,identities=plan_index(plan)
    return {key:row for key,row in state.get('tasks',{}).items()
            if key in tasks and row.get('phase')=='COMPLETE' and row.get('task_identity')==identities[key]
            and re.fullmatch('[0-9a-f]{64}',str(row.get('result_sha256','')))
            and re.fullmatch('[0-9a-f]{64}',str(row.get('completion_sha256','')))}


def blocked_task_keys(plan,state,complete):
    tasks,_=plan_index(plan)
    blocked={k for k,v in state['tasks'].items() if k in tasks and v.get('phase') in BLOCKED}
    while True:
        found={k for k,t in tasks.items() if set(t.get('requires',[]))&blocked and k not in complete}
        if found<=blocked:return blocked
        blocked|=found


def recover_completion_state(plan):
    """One metadata-only recovery on prepare/resume, never in each claim/poll."""
    identity=(str(Path(plan['output']).resolve()),plan['recipe_sha256'])
    if identity in _RECOVERED:return
    runtime=runtime_dir(plan);output=Path(plan['output']);tasks,identities=plan_index(plan)
    with locked(runtime/'tasks.lock'):
        state=_state(plan)
        existing={p.name:p for p in output.iterdir() if p.name in tasks} if output.is_dir() else {}
        candidates=set(existing)|{k for k,v in state['tasks'].items() if v.get('phase')=='COMPLETE' and k in tasks}
        recovered=0
        for key in candidates:
            task=tasks[key];folder=task_folder(plan,task);row=state['tasks'].get(key,{})
            if verify_result(plan,task,verify_artifacts=False):
                row.update(phase='COMPLETE',task_identity=identities[key],result_sha256=sha(folder/'result.json'),
                           completion_sha256=sha(folder/'completion.json'),current_error=None)
                state['tasks'][key]=row;recovered+=1
            elif row.get('phase')=='COMPLETE':
                row.update(phase='BLOCKED_CONFIGURATION',current_error={'error_type':'CompletedReceiptMissingOrChanged',
                    'reason':'Restore the exact sealed result; completed scientific work was not silently rerun'})
        state['completion_cache_recovered']=True
        atomic(runtime/'tasks.json',state)
        atomic(runtime/'recovery.json',dict(recipe_sha256=plan['recipe_sha256'],existing_task_directories=len(existing),
            verified_sealed_completions=recovered,full_artifact_bytes_rehashed=False,updated_utc=utc()))
    _RECOVERED.add(identity)


def file_signature(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def prepare(plan_path, *, python=None, account=None):
    value = context(plan_path, python=python, account=account)
    plan=value['plan'];root=Path(plan['root'])
    if plan.get('study_registry'):
        registry=verify_descriptor(root,plan['study_registry']);registry_sha=sha(registry)
        for task in plan['tasks']:
            expected=task.get('input_identities',{}).get('registry_sha256')
            if expected and expected!=registry_sha:
                raise ValueError('Task registry identity differs from the locked study registry')
    verified_baselines=set();optional_errors={}
    for task in plan['tasks']:
        artifacts=task.get('baseline_artifacts',{})
        if isinstance(artifacts,dict):artifacts=list(artifacts.values())
        try:
            if not isinstance(artifacts,list):raise ValueError('Baseline artifacts must be explicit SHA descriptors')
            if task.get('requires_baseline_artifacts') and not artifacts:
                raise ValueError('Historical baseline artifact has not been supplied')
            for artifact in artifacts:
                identity=digest(artifact)
                if identity not in verified_baselines:
                    verify_descriptor(root,artifact);verified_baselines.add(identity)
        except (OSError,ValueError) as error:
            if not task.get('optional_baseline'):raise
            optional_errors[task['key']]=dict(error_type=type(error).__name__,reason=str(error)[:500])
    # Preparing an sbatch text is harmless; it is not a submission.
    scheduler = FourJobScheduler(value['runtime'],value['launch'])
    source=Path(value['plan_path']);target=Path(value['plan']['output'])/'plan.json'
    if target.is_file() and sha(target)!=sha(source):
        raise ValueError('Existing R11 output belongs to a different exact plan')
    if not target.is_file():atomic_bytes(target,source.read_bytes())
    atomic(value['runtime']/'launch.json',value['launch'])
    recover_completion_state(plan)
    with locked(value['runtime']/'tasks.lock'):
        state=_state(plan)
        for task in plan['tasks']:
            row=state['tasks'].get(task['key'],{})
            if row.get('phase') in ('RUNNING','COMPLETE'):continue
            if task['key'] in optional_errors:
                row.update(phase='BLOCKED_CONFIGURATION',preflight_optional_baseline=True,
                           current_error=dict(optional_errors[task['key']],classification='OPTIONAL_BASELINE_UNAVAILABLE'),
                           retry_at=0,consecutive_failures=0)
                state['tasks'][task['key']]=row
            elif row.get('preflight_optional_baseline'):
                row.update(phase='PAUSED',preflight_optional_baseline=False,current_error=None,retry_at=0)
        atomic(value['runtime']/'tasks.json',state)
    atomic(value['runtime']/'preflight.json',dict(status='READY_WITH_OPTIONAL_BASELINES_UNAVAILABLE' if optional_errors else 'READY',
        optional_baseline_errors=optional_errors,updated_utc=utc(),source_registry_checked=True,
        baseline_artifacts_verified_once=len(verified_baselines),full_EEG_scan_performed=False))
    result = dict(status='PREPARED_NOT_SUBMITTED',round=11,stage=value['plan']['stage'],
                  plan=str(value['plan_path']),runtime=str(value['runtime']),
                  total_tasks=len(value['plan']['tasks']),jobs=4,gpus_per_job=1,
                  cpus_per_job=16,memory_per_job='64G',partition='hpc_gpu',
                  sbatch_file=str(scheduler.batch_path),submitted_jobs=0,
                  optional_baselines_unavailable=sorted(optional_errors))
    atomic(value['runtime']/'prepared.json',result)
    return result


def _authorization(ctx):
    receipt = read(ctx['runtime']/'SUBMIT_AUTHORIZATION.json',{})
    if (receipt.get('explicit_submit') is not True or receipt.get('plan_sha256')!=sha(ctx['plan_path'])
            or receipt.get('recipe_sha256')!=ctx['plan']['recipe_sha256']):
        raise ValueError('An exact plan requires explicit --submit before any Slurm action')


def await_controller(runtime,recipe,pid,child,timeout=3):
    deadline=time.monotonic()+timeout
    while True:
        ready=read(Path(runtime)/'controller-ready.json',{})
        if ready.get('pid')==pid and ready.get('recipe_sha256')==recipe:
            return dict(status='BOOTSTRAP_READY',acknowledged=True,receipt=ready)
        code=child.poll() if hasattr(child,'poll') else None
        if code is not None:return dict(status='CONTROLLER_FAILED_TO_START',acknowledged=False,exit_code=code)
        if time.monotonic()>=deadline:return dict(status='ACK_PENDING',acknowledged=False)
        time.sleep(.05)


def start(plan_path, *, submit=False, python=None, account=None, popen=None, uploads=True,
          credentials=None):
    prepared = prepare(plan_path,python=python,account=account)
    if not submit:
        return prepared
    ctx = context(plan_path,python=python,account=account)
    popen = popen or subprocess.Popen
    with locked(ctx['runtime']/'launch.lock'):
        atomic(ctx['runtime']/'SUBMIT_AUTHORIZATION.json',dict(explicit_submit=True,
            plan_sha256=sha(ctx['plan_path']),recipe_sha256=ctx['plan']['recipe_sha256'],
            authorized_utc=utc(),scope='R11 owned allocations only'))
        # Existing long-lived processes keep a flock; never trust a PID alone.
        with (ctx['runtime']/'controller.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                live = False
            except BlockingIOError:
                live = True
        processes = [];bootstrap=dict(status='EXISTING_CONTROLLER_REUSED',acknowledged=True)
        if live:
            processes.append(dict(kind='controller',reused=True,
                                  **read(ctx['runtime']/'controller-process.json',{})))
        else:
            command = [ctx['launch']['python'],'-I','-S','-B','-u',
                       str(Path(ctx['plan']['package'])/'paper3r11/platform/entry.py'),
                       'controller','--plan',str(ctx['plan_path'])]
            with (ctx['runtime']/'controller.log').open('ab',buffering=0) as log:
                child = popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                              start_new_session=True,close_fds=True)
            bootstrap=await_controller(ctx['runtime'],ctx['plan']['recipe_sha256'],child.pid,child)
            receipt = dict(kind='controller',pid=child.pid,started_utc=utc(),reused=False,bootstrap=bootstrap)
            atomic(ctx['runtime']/'controller-process.json',receipt);processes.append(receipt)
        if uploads:
            from .uploads import start_uploads
            processes += start_uploads(ctx['plan_path'],python=ctx['launch']['python'],
                                       credentials=credentials,popen=popen)
    overall='CONTROLLER_FAILED_TO_START' if bootstrap['status']=='CONTROLLER_FAILED_TO_START' else (
        'DETACHED_CONTROLLER_ACK_PENDING' if bootstrap['status']=='ACK_PENDING' else 'DETACHED_CONTROLLER_STARTED')
    return dict(prepared,status=overall,processes=processes,controller_bootstrap=bootstrap,
                terminal_can_close=True,submission_scope='Only R11 exact marker',
                actual_slurm_acknowledgements=str(ctx['runtime']/'scheduler-submissions'))


def _state(plan):
    runtime = runtime_dir(plan)
    path=runtime/'tasks.json'
    value = read(path,None,max_bytes=64*1024*1024) if path.is_file() else {}
    if not isinstance(value,dict):
        raise ValueError('R11 operational task state is unreadable; its attempt history was not reset')
    if value and value.get('recipe_sha256')!=plan['recipe_sha256']:
        raise ValueError('Operational task state belongs to another R11 recipe')
    return value or {'recipe_sha256':plan['recipe_sha256'],'tasks':{}}


def claim_task(plan, slot, job_id, *, clock=time.time):
    runtime = runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    (runtime/'task-locks').mkdir(exist_ok=True)
    with locked(runtime/'tasks.lock'):
        state = _state(plan)
        complete = set(completed_task_rows(plan,state))
        blocked=blocked_task_keys(plan,state,complete);terminal=complete|blocked
        for task in plan['tasks']:
            key = task['key'];row=state['tasks'].get(key,{})
            if (key in terminal or not set(task.get('requires',[]))<=complete
                    or not set(task.get('requires_terminal',[]))<=terminal
                    or row.get('phase') in BLOCKED
                    or row.get('retry_at',0)>clock()
                    or task.get('slot_affinity',slot)!=slot):
                continue
            lease = (runtime/'task-locks'/(key+'.lock')).open('a')
            try:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close();continue
            folder = task_folder(plan,task);folder.mkdir(parents=True,exist_ok=True)
            native = (folder/'.scientific.lock').open('a')
            try:
                fcntl.flock(native,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close();native.close();continue
            native.close()  # Scientific worker owns this second independent lock.
            attempt = int(row.get('attempt',0))+1
            token = digest([plan['recipe_sha256'],key,str(job_id),slot,attempt,clock(),os.getpid()])
            row = {**row,'key':key,'phase':'RUNNING','slot':slot,'job_id':str(job_id),
                   'attempt':attempt,'attempt_token':token,'started_at':clock(),
                   'retry_at':0,'current_error':None,'last_model_update':None}
            state['tasks'][key]=row
            atomic(runtime/'tasks.json',state)
            return task,lease,row
    return None


def finish_task(plan,task,code,*,stopped=False,clock=time.time):
    runtime=runtime_dir(plan)
    with locked(runtime/'tasks.lock'):
        state=_state(plan);row=state['tasks'].get(task['key'],{})
        if code==0:
            try:
                seal_result(plan,task)
            except (ValueError,OSError,TypeError,KeyError):
                code=2
        complete=verify_result(plan,task,verify_artifacts=False)
        folder=task_folder(plan,task)
        candidates=[p for p in (folder/'scientific_error.json',folder/'error.json') if p.is_file()]
        error_path=max(candidates,key=lambda p:p.stat().st_mtime) if candidates else folder/'scientific_error.json'
        error=read(error_path,{})
        if not isinstance(error,dict):error={}
        # Old retained errors are historical, not the new attempt's failure.
        if (error.get('attempt_token') and error['attempt_token']!=row.get('attempt_token')):
            error={}
        elif error_path.is_file() and error_path.stat().st_mtime<row.get('started_at',0):error={}
        signature=digest({'exit_code':code,'error_type':error.get('error_type'),
                          'reason':error.get('reason')})
        failures=int(row.get('consecutive_failures',0))
        if complete:
            phase='COMPLETE';failures=0;current=None
            row.update(task_identity=task_identity(plan,task),result_sha256=sha(folder/'result.json'),
                       completion_sha256=sha(folder/'completion.json'),last_numerical_progress=read(folder/'progress.json',{}))
        elif code==75 or stopped:
            phase='PAUSED';current=None
        else:
            failures+=1
            structural=(error.get('retryable') is False or error.get('classification')=='CONFIGURATION'
                        or error.get('error_type') in ('ValueError','FileNotFoundError','PermissionError'))
            phase='BLOCKED_CONFIGURATION' if structural else 'QUARANTINED' if failures>=MAX_ATTEMPTS else 'RETRYING'
            current={'error_type':error.get('error_type','WorkerExit'),
                     'reason':str(error.get('reason','No valid completion receipt'))[:800],
                     'exit_code':code,'signature':signature,'observed_utc':utc()}
        history=list(row.get('failure_history',[]))
        if current:history.append(dict(current,attempt=row.get('attempt')))
        row.update(phase=phase,exit_code=code,finished_at=clock(),current_error=current,
                   consecutive_failures=failures,failure_history=history,
                   retry_at=0 if complete else clock()+(0 if phase=='PAUSED' else min(120,10*2**min(failures,3))))
        state['tasks'][task['key']]=row;atomic(runtime/'tasks.json',state)
        return phase


def retry_blocked(plan_path,task_keys=None):
    """Explicit operational retry after repair; immutable science is never edited."""
    path,plan=load_plan(plan_path);runtime=runtime_dir(plan);changed=[]
    keys=set(task_keys) if task_keys is not None else {t['key'] for t in plan['tasks']}
    with locked(runtime/'tasks.lock'):
        state=_state(plan)
        for key,row in state['tasks'].items():
            if key not in keys or row.get('phase') not in BLOCKED:continue
            events=list(row.get('explicit_retries',[]))
            events.append(dict(reason='USER_EXPLICIT_RETRY_AFTER_REPAIR',utc=utc(),prior_phase=row['phase']))
            row.update(phase='PAUSED',consecutive_failures=0,retry_at=0,current_error=None,explicit_retries=events)
            changed.append(key)
        atomic(runtime/'tasks.json',state)
    if (runtime/'scheduler-state.json').is_file():
        ctx=context(path)
        FourJobScheduler(runtime,ctx['launch']).retry_blocked()
    return changed


# Root CLI checks explicit --submit before using this compatibility spelling.
reset_blocked=retry_blocked


def scheduler_snapshot(runtime):
    """Bounded local scheduler metadata; no squeue/sacct RPC from monitoring."""
    runtime=Path(runtime);state=read(runtime/'scheduler-state.json',{}) or {};jobs=[]
    for slot in range(4):
        value=state.get('slots',{}).get(str(slot),{});attempts=value.get('attempts',[])
        last=attempts[-1] if attempts else {}
        jobs.append(dict(slot=slot,job_id=last.get('job_id'),status=last.get('status','NOT_SUBMITTED'),
                         slurm_state=last.get('slurm_state'),queue_reason=last.get('queue_reason'),node=last.get('node'),
                         attempt=last.get('sequence',0),worker_failures=value.get('worker_failures',0),quarantine=value.get('quarantine')))
    error=read(runtime/'controller-error.json',{})
    return dict(controller_status=state.get('controller_status','NOT_STARTED'),marker=state.get('marker'),
                updated_at=state.get('updated_at'),observation_error=state.get('observation_error'),inflight_jobs=state.get('inflight_jobs',0),jobs=jobs,
                submission_block=state.get('submission_block'),controller_error=error,
                submission_authorized=(runtime/'SUBMIT_AUTHORIZATION.json').is_file(),
                process_start_receipt=read(runtime/'controller-process.json',{}),
                bootstrap_ready=read(runtime/'controller-ready.json',{}),
                queue_observation_performed_by_status=False)


def task_progress_item(plan,task,row,*,done=False,clock=time.time):
    runtime=runtime_dir(plan);folder=task_folder(plan,task)
    progress_path=folder/'progress.json';progress=row.get('last_numerical_progress',{}) if done else (
        read(progress_path,{}) if row.get('attempt',0) else {})
    if not isinstance(progress,dict):progress={}
    modified=progress_path.stat().st_mtime if not done and row.get('attempt',0) and progress_path.is_file() else None
    current=bool(modified is not None and (progress.get('attempt_token')==row.get('attempt_token')
        if progress.get('attempt_token') else modified>=row.get('started_at',float('inf'))))
    counter={k:progress[k] for k in ('outer_update','outer_updates_completed','completed_sessions','session',
             'iteration','model_update','fit_count','predicted_queries') if k in progress}
    positive=any(isinstance(v,(int,float)) and not isinstance(v,bool) and v>0 for v in counter.values())
    observation_path=runtime/'numerical-observations'/(task['key']+'.json')
    observation=read(observation_path,{}) if row.get('attempt',0) else {}
    if observation.get('attempt_token')!=row.get('attempt_token'):observation={}
    if current and positive and observation.get('counter')!=counter:
        observation=dict(attempt_token=row.get('attempt_token'),counter=counter,last_advance_unix=clock(),
                         source_progress_mtime=modified,observed_utc=utc())
        atomic(observation_path,observation)
    return {'key':task['key'],'kind':task.get('kind'),'phase':'COMPLETE' if done else row.get('phase','NEW'),
          'slot':row.get('slot'),'job_id':row.get('job_id'),'attempt':row.get('attempt',0),
          'attempt_token':row.get('attempt_token'),'current_error':None if done else row.get('current_error'),
          'historical_failures':len(row.get('failure_history',[])),
          'numerical_progress':{k:progress[k] for k in PROGRESS_FIELDS if k in progress} if current or done else {},
          'progress_is_current_attempt':current,'progress_age_seconds':round(max(0,clock()-modified),1) if modified else None,
          'last_numerical_advance_utc':observation.get('observed_utc'),
          'numerical_advance_age_seconds':round(max(0,clock()-observation['last_advance_unix']),1) if observation else None}


def status(plan_path,*,clock=time.time,plan=None,full=False):
    path=Path(plan_path).expanduser().resolve()
    if plan is None:_,plan=load_plan(path,validate_science_stage=False)
    runtime=runtime_dir(plan);state=_state(plan);tasks,identities=plan_index(plan)
    completed=completed_task_rows(plan,state);complete_keys=set(completed)
    blocked_keys=blocked_task_keys(plan,state,complete_keys)
    phases={k:'COMPLETE' if k in complete_keys else 'BLOCKED_DEPENDENCY' if k in blocked_keys
            and state['tasks'].get(k,{}).get('phase') not in BLOCKED else state['tasks'].get(k,{}).get('phase','NEW') for k in tasks}
    active=[k for k in tasks if phases[k]=='RUNNING'];blocked=[k for k in tasks if k in blocked_keys]
    pending=[k for k in tasks if k not in complete_keys and k not in blocked_keys and k not in active]
    recent=sorted(complete_keys,key=lambda k:state['tasks'][k].get('finished_at',0),reverse=True)
    included=list(tasks) if full else list(dict.fromkeys(active+blocked[:24]+recent[:8]+pending[:8]))[:64]
    rows=[];families={};phase_counts={}
    for key,task in tasks.items():
        phase=phases[key];phase_counts[phase]=phase_counts.get(phase,0)+1
        family=str(task.get('family',task.get('kind','unknown')))
        counts=families.setdefault(family,dict(total=0,complete=0,active=0,blocked=0))
        counts['total']+=1;counts['complete']+=int(key in complete_keys)
        counts['active']+=int(phase=='RUNNING');counts['blocked']+=int(key in blocked_keys)
    for key in included:
        item=task_progress_item(plan,tasks[key],state['tasks'].get(key,{}),done=key in complete_keys,clock=clock)
        item['phase']=phases[key];rows.append(item)
    complete=len(complete_keys);blocked=len(blocked_keys);remaining=len(tasks)-complete-blocked
    scheduler=scheduler_snapshot(runtime)
    acknowledged=[j for j in scheduler['jobs'] if j.get('job_id')]
    if complete==len(tasks):overall='COMPLETE'
    elif scheduler.get('controller_status','').startswith('BLOCKED_') and not scheduler.get('inflight_jobs'):
        overall='BLOCKED_OPERATIONAL'
    elif scheduler.get('controller_error',{}).get('error_type'):overall='CONTROLLER_ERROR'
    elif scheduler['controller_status']=='OBSERVATION_RETRY':overall='SCHEDULER_OBSERVATION_RETRY'
    elif not remaining:overall='BLOCKED'
    elif not scheduler['submission_authorized']:overall='PREPARED_NOT_SUBMITTED'
    elif acknowledged and all(j.get('slurm_state')=='PENDING' for j in acknowledged):overall='QUEUED'
    elif scheduler['controller_status']=='NOT_STARTED':overall='CONTROLLER_STARTING'
    else:overall='RUNNING_OR_READY'
    result={'schema_version':'paper3-r11-operational-status-v1','round':11,'stage':plan['stage'],
            'recipe_sha256':plan['recipe_sha256'],'updated_utc':utc(),'sampled_unix':clock(),
            'total_tasks':len(tasks),'complete_tasks':complete,'blocked_tasks':blocked,'runnable_remaining_tasks':remaining,
            'status':overall,
            'tasks':rows,'task_rows_included':len(rows),'task_rows_truncated':len(rows)<len(tasks),
            'task_phase_counts':phase_counts,'family_summary':families,
            'complete_task_keys_sha256':digest(sorted(complete_keys)),
            'science_completion_requires_verified_receipts':True,'completion_counts_from_durable_verified_cache':True,
            'monitoring_rehashes_full_payloads':False,'scheduler':scheduler,
            'reporter':read(runtime/'reporter-status.json',{}),
            'uploads':{c:read(runtime/('upload-'+c)/'status.json',{}) for c in ('paper','full')}}
    atomic(runtime/'status.json',result);return result


def child_command(plan,path,task,device):
    if plan.get('worker_entry'):
        if device!='cpu':raise ValueError('Alternate fixture entries are permitted only for explicit CPU tests')
        # Explicit local fixture/alternate entry is part of the pinned plan.
        return [plan.get('python',sys.executable),'-I','-S','-B','-u',plan['worker_entry'],
                '--plan',str(path),'--task-key',task['key'],'--device',device]
    return [plan.get('python',sys.executable),'-I','-S','-B','-u',
            str(Path(plan['package'])/'paper3r11/platform/entry.py'),'science',
            '--plan',str(path),'--task-key',task['key'],'--device',device]


def child_environment(slot,token,device):
    env=dict(os.environ)
    for key in ('BASH_ENV','ENV','PYTHONHOME','PYTHONPATH','PAPER3_R11_STOP_REQUESTED'):
        env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               PAPER3_R11_SLOT=str(slot),PAPER3_R11_ATTEMPT_TOKEN=token)
    if device!='cpu':
        visible=env.get('CUDA_VISIBLE_DEVICES') or env.get('SLURM_STEP_GPUS') or env.get('SLURM_JOB_GPUS')
        if not visible or visible in ('-1','none','NoDevFiles','void'):
            raise ValueError('This worker has no assigned single-GPU allocation')
        match=re.fullmatch(r'(\d+)-(\d+)',visible)
        if match:visible=match.group(1)
        # Each allocation uses its own logical CUDA device zero, never slot1/2.
        env['CUDA_VISIBLE_DEVICES']=visible.split(',')[0]
    return env


class PersistentScience:
    """One numerical process per allocation; Python task errors do not respawn it."""
    def __init__(self,plan,path,slot,device,*,popen=None,sleep=time.sleep):
        self.plan,self.path,self.slot,self.device=plan,path,slot,device
        self.popen=popen or subprocess.Popen;self.sleep=sleep;self.child=None
        self.incarnation=0;self.sequence=0;self.log=None;self.responses=None;self.bootstrap_failed=False

    def _start(self):
        if self.child is not None and self.child.poll() is None:return
        if self.child is not None and self.child.stdin:
            try:self.child.stdin.close()
            except OSError:pass
        if self.log:self.log.close()
        self.incarnation+=1;self.sequence=0;self.bootstrap_failed=False
        self.responses=runtime_dir(self.plan)/'science-processes'/('slot-%d'%self.slot)/(
            '%d-%d-%d'%(os.getpid(),self.incarnation,time.time_ns()))
        self.responses.mkdir(parents=True,exist_ok=True)
        command=[self.plan.get('python',sys.executable),'-I','-S','-B','-u',
            str(Path(self.plan['package'])/'paper3r11/platform/entry.py'),'science-daemon',
            '--plan',str(self.path),'--device',self.device,'--response-dir',str(self.responses)]
        self.log=(self.responses/'process.log').open('ab',buffering=0)
        self.child=self.popen(command,stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,
            text=True,bufsize=1,env=child_environment(self.slot,'0'*64,self.device),
            start_new_session=True,close_fds=True)
        atomic(self.responses/'process.json',dict(pid=self.child.pid,incarnation=self.incarnation,
            recipe_sha256=self.plan['recipe_sha256'],started_utc=utc(),worker_import_started=True,worker_imports_per_process=1))

    def run(self,task,claim,attempt,stop,heartbeat=None):
        self._start();self.sequence+=1;number=self.sequence
        request=dict(task_key=task['key'],attempt_token=claim['attempt_token'],sequence=number,attempt_dir=str(attempt))
        try:
            self.child.stdin.write(json.dumps(request)+'\n');self.child.stdin.flush()
        except (BrokenPipeError,OSError):
            return self.child.wait() if self.child.poll() is not None else 2
        sent=False;next_heartbeat=0;next_stop_check=0;response=self.responses/('%06d.json'%number)
        while True:
            value=read(response,{})
            if (value.get('attempt_token')==claim['attempt_token'] and value.get('task_key')==task['key']
                    and value.get('sequence')==number):
                if value.get('restart_required'):
                    try:self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.child.kill();self.child.wait(timeout=5)
                return int(value['exit_code'])
            code=self.child.poll()
            if code is not None:
                self.bootstrap_failed=code!=75 and not (self.responses/'READY.json').is_file()
                if self.bootstrap_failed:
                    atomic(self.responses/'bootstrap-error.json',dict(status='SCIENTIFIC_IMPORT_FAILED',
                        exit_code=code,process_log=str(self.responses/'process.log'),updated_utc=utc()))
                return code if code else 2
            if time.monotonic()>=next_stop_check:
                if (runtime_dir(self.plan)/'STOP').exists():stop.set()
                next_stop_check=time.monotonic()+1
            if stop.is_set() and not sent:
                try:self.child.send_signal(signal.SIGUSR1)
                except ProcessLookupError:pass
                sent=True
            if heartbeat and time.monotonic()>=next_heartbeat:
                heartbeat(self.child.pid,sent);next_heartbeat=time.monotonic()+10
            self.sleep(.1)

    def close(self):
        if self.child is not None and self.child.poll() is None:
            try:self.child.stdin.write('{"shutdown":true}\n');self.child.stdin.flush();self.child.stdin.close()
            except (BrokenPipeError,OSError):pass
            try:self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:self.child.send_signal(signal.SIGUSR1)
                except ProcessLookupError:pass
        if self.log:self.log.close();self.log=None


def persistent_worker_loop(plan_path,slot,*,device='cuda:0',popen=None,sleep=time.sleep,stop=None,once=False):
    path,plan=load_plan(plan_path);runtime=runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    stop=stop or threading.Event();job=os.environ.get('SLURM_JOB_ID','CPU_TEST');previous={};code=2
    original_sha=sha(path);stamp=file_signature(path)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):
        previous[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.set())
    science=PersistentScience(plan,path,slot,device,popen=popen,sleep=sleep)
    try:
        while not stop.is_set():
            current=file_signature(path)
            if current!=stamp:
                if sha(path)!=original_sha:raise ValueError('R11 plan changed during execution')
                stamp=current
            if (runtime/'STOP').exists():stop.set();break
            snapshot=status(path,plan=plan)
            if snapshot['complete_tasks']==snapshot['total_tasks']:code=0;break
            claim=claim_task(plan,slot,job)
            if claim is None:
                if not snapshot['runnable_remaining_tasks']:code=0;break
                if once:code=0;break
                sleep(2);continue
            task,lease,row=claim
            attempt=runtime/'attempts'/task['key']/('%04d'%row['attempt']);attempt.mkdir(parents=True,exist_ok=True)
            atomic(attempt/'claim.json',row)
            def heartbeat(pid,sent):
                item=task_progress_item(plan,task,row)
                atomic(runtime/('slot-%d.json'%slot),dict(item,status='CHECKPOINTING' if sent else 'RUNNING',
                    child_pid=pid,persistent_scientific_process=True,updated_utc=utc()))
            bootstrap_failed=False
            try:
                emit('R11_TASK_STARTED',task_key=task['key'],job_id=job,slot=slot,attempt=row['attempt'])
                returned=science.run(task,row,attempt,stop,heartbeat)
                bootstrap_failed=science.bootstrap_failed
                if bootstrap_failed:
                    atomic(task_folder(plan,task)/'scientific_error.json',dict(status='FAILED',
                        error_type='ScientificWorkerImportFailed',attempt_token=row['attempt_token'],retryable=True,
                        reason='See allocation-local scientific process.log; remaining tasks were not attempted in this broken process'))
                phase=finish_task(plan,task,returned,stopped=stop.is_set())
                emit('R11_TASK_RETURNED',task_key=task['key'],slot=slot,exit_code=returned,status=phase)
            except Exception as error:
                atomic(task_folder(plan,task)/'scientific_error.json',dict(status='FAILED',error_type=type(error).__name__,
                    reason=str(error)[:800],attempt_token=row['attempt_token'],updated_utc=utc()))
                finish_task(plan,task,2,stopped=stop.is_set())
            finally:lease.close()
            if bootstrap_failed:code=2;break
            if once:code=75 if stop.is_set() else 0;break
        if stop.is_set():code=75
        return code
    finally:
        science.close()
        atomic(runtime/('slot-%d.json'%slot),dict(status='PAUSED' if code==75 else 'COMPLETE' if code==0 else 'FAILED',
            slot=slot,job_id=job,exit_code=code,persistent_scientific_process=True,updated_utc=utc()))
        if str(job).isdigit():write_worker_outcome(runtime,slot,job,code,round=11)
        for sig,handler in previous.items():signal.signal(sig,handler)


def worker_loop(plan_path,slot,*,device='cuda:0',popen=None,sleep=time.sleep,stop=None,once=False):
    path,plan=load_plan(plan_path)
    if not plan.get('worker_entry') and (not plan.get('synthetic_fixture') or plan.get('persistent_science')):
        return persistent_worker_loop(path,slot,device=device,popen=popen,sleep=sleep,stop=stop,once=once)
    path,plan=load_plan(plan_path);runtime=runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    original_sha=sha(path);plan_stamp=file_signature(path);job=os.environ.get('SLURM_JOB_ID','CPU_TEST');stop=stop or threading.Event()
    popen=popen or subprocess.Popen;previous={};active=None;exit_code=2
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):
        previous[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.set())
    try:
        while not stop.is_set():
            stamp=file_signature(path)
            if stamp!=plan_stamp:
                if sha(path)!=original_sha:raise ValueError('R11 plan changed during execution')
                plan_stamp=stamp
            if (runtime/'STOP').exists():stop.set();break
            snapshot=status(path,plan=plan)
            if snapshot['complete_tasks']==snapshot['total_tasks']:exit_code=0;break
            claim=claim_task(plan,int(slot),job)
            if claim is None:
                if not snapshot['runnable_remaining_tasks']:exit_code=0;break
                atomic(runtime/('slot-%d.json'%slot),dict(status='WAITING_DEPENDENCIES',slot=slot,job_id=job,updated_utc=utc()))
                if once:exit_code=0;break
                sleep(2);continue
            task,lease,claim_row=claim
            attempt_dir=runtime/'attempts'/task['key']/('%04d'%claim_row['attempt']);attempt_dir.mkdir(parents=True,exist_ok=True)
            atomic(attempt_dir/'claim.json',claim_row)
            try:
                env=child_environment(slot,claim_row['attempt_token'],device)
                with (attempt_dir/'stdout.log').open('ab',buffering=0) as out,(attempt_dir/'stderr.log').open('ab',buffering=0) as err:
                    active=popen(child_command(plan,path,task,device),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                 env=env,start_new_session=True,close_fds=True)
                    emit('R11_TASK_STARTED',task_key=task['key'],job_id=job,slot=slot,attempt=claim_row['attempt'],pid=active.pid)
                    sent=False;heartbeat=0
                    while active.poll() is None:
                        if (runtime/'STOP').exists():stop.set()
                        if stop.is_set() and not sent:
                            try:active.send_signal(signal.SIGUSR1)
                            except ProcessLookupError:pass
                            sent=True
                        if time.monotonic()>=heartbeat:
                            item=task_progress_item(plan,task,claim_row)
                            atomic(runtime/('slot-%d.json'%slot),dict(item,status='CHECKPOINTING' if sent else 'RUNNING',
                                child_pid=active.pid,updated_utc=utc()))
                            heartbeat=time.monotonic()+10
                        sleep(.2)
                    code=active.wait();active=None
                phase=finish_task(plan,task,code,stopped=stop.is_set())
                emit('R11_TASK_RETURNED',task_key=task['key'],slot=slot,exit_code=code,status=phase)
            except Exception as error:
                if active is not None and active.poll() is None:
                    active.send_signal(signal.SIGUSR1);active.wait();active=None
                atomic(task_folder(plan,task)/'error.json',dict(status='FAILED',error_type=type(error).__name__,
                    reason=str(error)[:800],classification='CONFIGURATION' if isinstance(error,ValueError) else 'RUNTIME',
                    retryable=not isinstance(error,ValueError),updated_utc=utc()))
                finish_task(plan,task,2,stopped=stop.is_set())
            finally:
                lease.close()
            if once:exit_code=75 if stop.is_set() else 0;break
        if stop.is_set():exit_code=75
        return exit_code
    finally:
        atomic(runtime/('slot-%d.json'%slot),dict(status='PAUSED' if exit_code==75 else 'COMPLETE' if exit_code==0 else 'FAILED',
                                               slot=slot,job_id=job,exit_code=exit_code,updated_utc=utc()))
        if str(job).isdigit():write_worker_outcome(runtime,slot,job,exit_code,round=11)
        for sig,handler in previous.items():signal.signal(sig,handler)


def controller_loop(plan_path,*,runner=None,clock=time.time,stop=None,once=False):
    ctx=context(plan_path);_authorization(ctx);stop=stop or threading.Event()
    with locked(ctx['runtime']/'controller.lock',nonblocking=True):
        atomic(ctx['runtime']/'controller-ready.json',dict(status='BOOTSTRAP_READY',pid=os.getpid(),
            recipe_sha256=ctx['plan']['recipe_sha256'],updated_utc=utc(),
            first_slurm_observation_performed=False,training_progress_claimed=False))
        atomic(ctx['runtime']/'controller-error.json',{})
        scheduler=FourJobScheduler(ctx['runtime'],ctx['launch'],runner=runner,clock=clock,emit=emit)
        plan_stamp=file_signature(ctx['plan_path'])
        while not stop.is_set():
            stamp=file_signature(ctx['plan_path'])
            if stamp!=plan_stamp:
                if sha(ctx['plan_path'])!=ctx['launch']['plan_sha256']:
                    raise ValueError('Authorized R11 plan changed; automatic submission stopped')
                plan_stamp=stamp
            snapshot=status(ctx['plan_path'],clock=clock,plan=ctx['plan'])
            remaining=snapshot['runnable_remaining_tasks']
            keep=bool(remaining)
            summary=scheduler.step(keep_workers=keep)
            atomic(ctx['runtime']/'controller-status.json',dict(summary,task_status=snapshot,updated_utc=utc()))
            if once:return summary
            if summary.get('controller_status','').startswith('BLOCKED_') and not summary.get('inflight_jobs'):
                return summary
            if not remaining and summary.get('controller_status')=='FINISHED':return summary
            stop.wait(float(ctx['launch']['poll_seconds']))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','submit','resume','status','controller','worker'):
        p=sub.add_parser(name);p.add_argument('--plan',required=True)
        if name in ('submit','resume'):
            p.add_argument('--submit',action='store_true');p.add_argument('--credentials')
            if name=='resume':p.add_argument('--retry-blocked',action='store_true')
        if name=='worker':p.add_argument('--slot',type=int,choices=(0,1,2,3),required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare':result=prepare(args.plan)
    elif args.command in ('submit','resume'):
        if args.command=='resume' and args.retry_blocked:
            if not args.submit:raise ValueError('Retrying blocked work requires explicit resume --submit --retry-blocked')
            prepare(args.plan);retry_blocked(args.plan)
        result=start(args.plan,submit=args.submit,credentials=args.credentials)
    elif args.command=='status':result=status(args.plan)
    elif args.command=='controller':
        try:result=controller_loop(args.plan)
        except Exception as error:
            _,plan=load_plan(args.plan,validate_science_stage=False)
            atomic(runtime_dir(plan)/'controller-error.json',dict(status='CONTROLLER_START_OR_POLL_FAILED',
                error_type=type(error).__name__,reason=str(error)[:500],updated_utc=utc()))
            raise
    else:return worker_loop(args.plan,args.slot,device=args.device,once=args.once)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
