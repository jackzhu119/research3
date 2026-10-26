"""R10-only durable task execution and explicit Slurm submission boundary.

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
    entry = package/'paper3r10/platform/entry.py'
    if not entry.is_file():
        raise ValueError('R10 isolated entry is absent from the installed package')
    python = str(Path(python or plan.get('python') or sys.executable).absolute())
    resources = dict(plan.get('resources', {}))
    if resources.get('jobs', 4)!=4 or resources.get('gpus_per_job', 1)!=1:
        raise ValueError('R10 uses exactly four single-GPU worker allocations')
    if resources.get('cpus_per_job',16)!=16 or resources.get('memory_per_job','68G')!='68G':
        raise ValueError('R10 workers require 16 CPUs and 68G memory each')
    if resources.get('partition','hpc_gpu')!='hpc_gpu':
        raise ValueError('R10 stage is configured for hpc_gpu')
    command = [python,'-I','-S','-B','-u',str(entry),'worker','--plan',str(path),
               '--slot','{slot}','--device','cuda:0']
    launch = {'round':10,'partition':'hpc_gpu','cpus':16,'memories':['68G']*4,
              'gres':'gpu:1','time_limit':resources.get('time_limit','5-03:00:00'),
              'poll_seconds':resources.get('poll_seconds',30),
              'account':account or resources.get('account') or getpass.getuser(),
              'user':getpass.getuser(),'campaign_identity':plan['recipe_sha256'],
              'worker_command':command,'python':python,'package':str(package),
              'root':plan['root'],'output':plan['output'],'plan':str(path),
              'plan_sha256':sha(path),'runtime':str(runtime),'exclude':''}
    return {'plan_path':path,'plan':plan,'runtime':runtime,'launch':launch}


def prepare(plan_path, *, python=None, account=None):
    value = context(plan_path, python=python, account=account)
    plan=value['plan'];root=Path(plan['root'])
    if plan.get('study_registry'):
        registry=verify_descriptor(root,plan['study_registry'])
        for task in plan['tasks']:
            expected=task.get('input_identities',{}).get('registry_sha256')
            if expected and expected!=sha(registry):
                raise ValueError('Task registry identity differs from the locked study registry')
    for task in plan['tasks']:
        artifacts=task.get('baseline_artifacts',{})
        if isinstance(artifacts,dict):artifacts=list(artifacts.values())
        if not isinstance(artifacts,list):raise ValueError('Baseline artifacts must be explicit SHA descriptors')
        if task.get('requires_baseline_artifacts') and not artifacts:
            raise ValueError('A required strong baseline artifact has not been supplied')
        for artifact in artifacts:verify_descriptor(root,artifact)
    # Preparing an sbatch text is harmless; it is not a submission.
    scheduler = FourJobScheduler(value['runtime'],value['launch'])
    source=Path(value['plan_path']);target=Path(value['plan']['output'])/'plan.json'
    if target.is_file() and sha(target)!=sha(source):
        raise ValueError('Existing R10 output belongs to a different exact plan')
    if not target.is_file():atomic_bytes(target,source.read_bytes())
    atomic(value['runtime']/'launch.json',value['launch'])
    result = dict(status='PREPARED_NOT_SUBMITTED',round=10,stage=value['plan']['stage'],
                  plan=str(value['plan_path']),runtime=str(value['runtime']),
                  total_tasks=len(value['plan']['tasks']),jobs=4,gpus_per_job=1,
                  cpus_per_job=16,memory_per_job='68G',partition='hpc_gpu',
                  sbatch_file=str(scheduler.batch_path),submitted_jobs=0)
    atomic(value['runtime']/'prepared.json',result)
    return result


def _authorization(ctx):
    receipt = read(ctx['runtime']/'SUBMIT_AUTHORIZATION.json',{})
    if (receipt.get('explicit_submit') is not True or receipt.get('plan_sha256')!=sha(ctx['plan_path'])
            or receipt.get('recipe_sha256')!=ctx['plan']['recipe_sha256']):
        raise ValueError('An exact plan requires explicit --submit before any Slurm action')


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
            authorized_utc=utc(),scope='R10 owned allocations only'))
        # Existing long-lived processes keep a flock; never trust a PID alone.
        with (ctx['runtime']/'controller.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                live = False
            except BlockingIOError:
                live = True
        processes = []
        if live:
            processes.append(dict(kind='controller',reused=True,
                                  **read(ctx['runtime']/'controller-process.json',{})))
        else:
            command = [ctx['launch']['python'],'-I','-S','-B','-u',
                       str(Path(ctx['plan']['package'])/'paper3r10/platform/entry.py'),
                       'controller','--plan',str(ctx['plan_path'])]
            with (ctx['runtime']/'controller.log').open('ab',buffering=0) as log:
                child = popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                              start_new_session=True,close_fds=True)
            receipt = dict(kind='controller',pid=child.pid,started_utc=utc(),reused=False)
            atomic(ctx['runtime']/'controller-process.json',receipt);processes.append(receipt)
        if uploads:
            from .uploads import start_uploads
            processes += start_uploads(ctx['plan_path'],python=ctx['launch']['python'],
                                       credentials=credentials,popen=popen)
    return dict(prepared,status='DETACHED_CONTROLLER_STARTED',processes=processes,
                terminal_can_close=True,submission_scope='Only R10 exact marker',
                actual_slurm_acknowledgements=str(ctx['runtime']/'scheduler-submissions'))


def _state(plan):
    runtime = runtime_dir(plan)
    path=runtime/'tasks.json'
    value = read(path,None) if path.is_file() else {}
    if not isinstance(value,dict):
        raise ValueError('R10 operational task state is unreadable; its attempt history was not reset')
    if value and value.get('recipe_sha256')!=plan['recipe_sha256']:
        raise ValueError('Operational task state belongs to another R10 recipe')
    return value or {'recipe_sha256':plan['recipe_sha256'],'tasks':{}}


def claim_task(plan, slot, job_id, *, clock=time.time):
    runtime = runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    (runtime/'task-locks').mkdir(exist_ok=True)
    with locked(runtime/'tasks.lock'):
        state = _state(plan)
        complete = {t['key'] for t in plan['tasks'] if verify_result(plan,t,verify_artifacts=False)}
        for task in plan['tasks']:
            key = task['key'];row=state['tasks'].get(key,{})
            if (key in complete or not set(task.get('requires',[]))<=complete
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


def status(plan_path,*,clock=time.time):
    path,plan=load_plan(plan_path,validate_science_stage=False)
    runtime=runtime_dir(plan);state=_state(plan);rows=[];complete=0
    for task in plan['tasks']:
        row=state['tasks'].get(task['key'],{})
        # Completion artifacts were verified once before the completion seal.
        # Frequent monitoring verifies the small result/seal bytes, without
        # rereading every prediction/checkpoint payload on the shared filesystem.
        done=verify_result(plan,task,verify_artifacts=False);complete+=int(done)
        progress_path=task_folder(plan,task)/'progress.json';progress=read(progress_path,{})
        if not isinstance(progress,dict):progress={}
        modified=progress_path.stat().st_mtime if progress_path.is_file() else None
        current=bool(modified is not None and (progress.get('attempt_token')==row.get('attempt_token')
            if progress.get('attempt_token') else modified>=row.get('started_at',float('inf'))))
        counter={k:progress[k] for k in ('outer_update','outer_updates_completed','completed_sessions','session',
                 'iteration','model_update','fit_count','predicted_queries') if k in progress}
        observation_path=runtime/'numerical-observations'/(task['key']+'.json')
        observation=read(observation_path,{})
        if observation.get('attempt_token')!=row.get('attempt_token'):observation={}
        positive=any(isinstance(v,(int,float)) and not isinstance(v,bool) and v>0 for v in counter.values())
        if current and positive and observation.get('counter')!=counter:
            observation=dict(attempt_token=row.get('attempt_token'),counter=counter,last_advance_unix=clock(),
                             source_progress_mtime=modified,observed_utc=utc())
            atomic(observation_path,observation)
        item={'key':task['key'],'kind':task.get('kind'),'phase':'COMPLETE' if done else row.get('phase','NEW'),
              'slot':row.get('slot'),'job_id':row.get('job_id'),'attempt':row.get('attempt',0),
              'attempt_token':row.get('attempt_token'),'current_error':None if done else row.get('current_error'),
              'historical_failures':len(row.get('failure_history',[])),
              'numerical_progress':{k:progress[k] for k in PROGRESS_FIELDS if k in progress} if current or done else {},
              'progress_is_current_attempt':current,'progress_age_seconds':round(max(0,clock()-modified),1) if modified else None,
              'last_numerical_advance_utc':observation.get('observed_utc'),
              'numerical_advance_age_seconds':round(max(0,clock()-observation['last_advance_unix']),1) if observation else None}
        rows.append(item)
    blocked_keys={r['key'] for r in rows if r['phase'] in BLOCKED}
    while True:
        found={t['key'] for t in plan['tasks'] if set(t.get('requires',[]))&blocked_keys}
        if found<=blocked_keys:break
        blocked_keys|=found
    for item in rows:
        if item['key'] in blocked_keys and item['phase'] not in BLOCKED and item['phase']!='COMPLETE':
            item['phase']='BLOCKED_DEPENDENCY'
    blocked=sum(r['phase'] in BLOCKED for r in rows)
    result={'schema_version':'paper3-r10-operational-status-v1','round':10,'stage':plan['stage'],
            'recipe_sha256':plan['recipe_sha256'],'plan_sha256':sha(path),'updated_utc':utc(),
            'sampled_unix':clock(),'total_tasks':len(rows),'complete_tasks':complete,
            'blocked_tasks':blocked,'status':'COMPLETE' if complete==len(rows) else 'BLOCKED' if blocked else 'RUNNING_OR_READY',
            'tasks':rows,'science_completion_requires_verified_receipts':True,
            'monitoring_rehashes_full_payloads':False,
            'uploads':{c:read(runtime/('upload-'+c)/'status.json',{}) for c in ('paper','full')}}
    atomic(runtime/'status.json',result)
    return result


def child_command(plan,path,task,device):
    if plan.get('worker_entry'):
        if device!='cpu':raise ValueError('Alternate fixture entries are permitted only for explicit CPU tests')
        # Explicit local fixture/alternate entry is part of the pinned plan.
        return [plan.get('python',sys.executable),'-I','-S','-B','-u',plan['worker_entry'],
                '--plan',str(path),'--task-key',task['key'],'--device',device]
    return [plan.get('python',sys.executable),'-I','-S','-B','-u',
            str(Path(plan['package'])/'paper3r10/platform/entry.py'),'science',
            '--plan',str(path),'--task-key',task['key'],'--device',device]


def child_environment(slot,token,device):
    env=dict(os.environ)
    for key in ('BASH_ENV','ENV','PYTHONHOME','PYTHONPATH','PAPER3_R10_STOP_REQUESTED'):
        env.pop(key,None)
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               PAPER3_R10_SLOT=str(slot),PAPER3_R10_ATTEMPT_TOKEN=token)
    if device!='cpu':
        visible=env.get('CUDA_VISIBLE_DEVICES') or env.get('SLURM_STEP_GPUS') or env.get('SLURM_JOB_GPUS')
        if not visible or visible in ('-1','none','NoDevFiles','void'):
            raise ValueError('This worker has no assigned single-GPU allocation')
        match=re.fullmatch(r'(\d+)-(\d+)',visible)
        if match:visible=match.group(1)
        # Each allocation uses its own logical CUDA device zero, never slot1/2.
        env['CUDA_VISIBLE_DEVICES']=visible.split(',')[0]
    return env


def worker_loop(plan_path,slot,*,device='cuda:0',popen=None,sleep=time.sleep,stop=None,once=False):
    path,plan=load_plan(plan_path);runtime=runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    original_sha=sha(path);job=os.environ.get('SLURM_JOB_ID','CPU_TEST');stop=stop or threading.Event()
    popen=popen or subprocess.Popen;previous={};active=None;exit_code=2
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):
        previous[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.set())
    try:
        while not stop.is_set():
            if sha(path)!=original_sha:raise ValueError('R10 plan changed during execution')
            if (runtime/'STOP').exists():stop.set();break
            snapshot=status(path)
            if snapshot['complete_tasks']==snapshot['total_tasks']:exit_code=0;break
            claim=claim_task(plan,int(slot),job)
            if claim is None:
                pending=[r for r in snapshot['tasks'] if r['phase']!='COMPLETE' and r['phase'] not in BLOCKED]
                if not pending:exit_code=2;break
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
                    emit('R10_TASK_STARTED',task_key=task['key'],job_id=job,slot=slot,attempt=claim_row['attempt'],pid=active.pid)
                    sent=False;heartbeat=0
                    while active.poll() is None:
                        if (runtime/'STOP').exists():stop.set()
                        if stop.is_set() and not sent:
                            try:active.send_signal(signal.SIGUSR1)
                            except ProcessLookupError:pass
                            sent=True
                        if time.monotonic()>=heartbeat:
                            snap=status(path);item=next(r for r in snap['tasks'] if r['key']==task['key'])
                            atomic(runtime/('slot-%d.json'%slot),dict(item,status='CHECKPOINTING' if sent else 'RUNNING',
                                child_pid=active.pid,updated_utc=utc()))
                            heartbeat=time.monotonic()+10
                        sleep(.2)
                    code=active.wait();active=None
                phase=finish_task(plan,task,code,stopped=stop.is_set())
                emit('R10_TASK_RETURNED',task_key=task['key'],slot=slot,exit_code=code,status=phase)
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
        if str(job).isdigit():write_worker_outcome(runtime,slot,job,exit_code,round=10)
        for sig,handler in previous.items():signal.signal(sig,handler)


def controller_loop(plan_path,*,runner=None,clock=time.time,stop=None,once=False):
    ctx=context(plan_path);_authorization(ctx);stop=stop or threading.Event()
    with locked(ctx['runtime']/'controller.lock',nonblocking=True):
        scheduler=FourJobScheduler(ctx['runtime'],ctx['launch'],runner=runner,clock=clock,emit=emit)
        while not stop.is_set():
            if sha(ctx['plan_path'])!=ctx['launch']['plan_sha256']:
                raise ValueError('Authorized R10 plan changed; automatic submission stopped')
            snapshot=status(ctx['plan_path'],clock=clock)
            remaining=[t for t in snapshot['tasks'] if t['phase']!='COMPLETE' and t['phase'] not in BLOCKED]
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
    elif args.command=='controller':result=controller_loop(args.plan)
    else:return worker_loop(args.plan,args.slot,device=args.device,once=args.once)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
