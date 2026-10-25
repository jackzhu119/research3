"""Three independent single-GPU workers over an unchanged Round 9 task DAG.

Only this operational directory is new. The original plan, task result paths,
checkpoint format and numerical worker are retained verbatim.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from generic_scheduler import ThreeJobScheduler, write_worker_outcome


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, ensure_ascii=False, sort_keys=True), flush=True)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def plan_context(plan_path, runtime=None):
    plan_path = Path(plan_path).resolve()
    plan_data = plan_path.read_bytes()
    plan = json.loads(plan_data)
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('A nonempty original task plan is required')
    keys = [t.get('key') for t in tasks]
    if any(not isinstance(k, str) or Path(k).name != k or k in ('.','..') for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('Original task keys must be unique relative directory names')
    known = set(keys)
    if any(not set(t.get('requires',[])) <= known for t in tasks):
        raise ValueError('Original task dependency is absent')
    # Validate the small declared DAG once; never read training arrays here.
    remaining={t['key']:set(t.get('requires',[])) for t in tasks}
    done=set()
    while remaining:
        ready={key for key,deps in remaining.items() if deps <= done}
        if not ready:raise ValueError('Task plan contains cyclic dependencies')
        done.update(ready)
        for key in ready:remaining.pop(key)
    if not isinstance(plan.get('recipe_sha256'), str) or not plan['recipe_sha256']:
        raise ValueError('Task plan must declare a recipe identity')
    output = Path(plan['output']).resolve()
    runtime = Path(runtime).resolve() if runtime else output/'runtime/three_jobs'
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime/'task-locks').mkdir(exist_ok=True)
    return plan_path, plan, output, runtime, hashlib.sha256(plan_data).hexdigest()


def result_complete(output, task, recipe):
    value = read(Path(output)/task['key']/'result.json', {})
    return value.get('status') == 'COMPLETE' and value.get('recipe_sha256') == recipe


def _native_unlocked(folder):
    path = Path(folder)/'.lock'
    if not path.exists():
        return True
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def claim_task(plan, output, runtime, slot, job_id, clock=time.time):
    """Return (index, lease file handle, attempt), or None. Handles own leases.

    Flocks, not PID/time guesses, release ownership after a dead node/process.
    A surviving native child retains its own .lock and is never duplicated.
    """
    (runtime/'task-locks').mkdir(parents=True,exist_ok=True)
    with locked(runtime/'tasks.lock'):
        state = read(runtime/'tasks.json', {})
        if state and state.get('recipe_sha256') != plan.get('recipe_sha256'):
            raise ValueError('Operational task state belongs to a different recipe')
        state = state or {'recipe_sha256':plan.get('recipe_sha256'),'tasks':{}}
        completed = {t['key'] for t in plan['tasks'] if result_complete(output,t,plan.get('recipe_sha256'))}
        for index, task in enumerate(plan['tasks']):
            if task['key'] in completed or not set(task.get('requires',[])) <= completed:
                continue
            row = state['tasks'].get(str(index), {})
            if row.get('retry_at',0) > clock():
                continue
            lease = (runtime/'task-locks'/('%05d.lock'%index)).open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close(); continue
            if not _native_unlocked(output/task['key']):
                lease.close(); continue
            attempt = int(row.get('attempt',0))+1
            state['tasks'][str(index)] = {**row,'key':task['key'],'phase':'RUNNING',
                'slot':slot,'job_id':str(job_id),'pid':os.getpid(),'attempt':attempt,
                'started_at':clock(),'retry_at':0}
            atomic(runtime/'tasks.json',state)
            return index, lease, attempt
        return None


def finish_task(plan, output, runtime, index, code, stopped=False, clock=time.time):
    with locked(runtime/'tasks.lock'):
        state = read(runtime/'tasks.json', {'recipe_sha256':plan.get('recipe_sha256'),'tasks':{}})
        row = state['tasks'].get(str(index),{})
        complete = result_complete(output,plan['tasks'][index],plan.get('recipe_sha256'))
        phase = 'COMPLETE' if complete else 'PAUSED' if code == 75 or stopped else 'RETRYING'
        delay = 0 if complete else min(300,30*2**min(max(0,int(row.get('attempt',1))-1),4))
        row.update(phase=phase,exit_code=code,finished_at=clock(),retry_at=clock()+delay)
        state['tasks'][str(index)] = row
        atomic(runtime/'tasks.json',state)
        return phase


def worker_command(python, package, round_number, plan_path, index, device='cuda:0'):
    return [str(python),'-I','-S','-B','-u',str(Path(__file__).resolve().parent/'backend_entry.py'),
            '--original-entry',str(Path(package)/('paper3r%d/entry.py'%round_number)),
            'worker','--plan',str(plan_path),'--task',str(index),'--device',device]


def child_environment(round_number, slot, device):
    env = dict(os.environ)
    for key in ('BASH_ENV','ENV','PYTHONHOME','PYTHONPATH', 'PAPER3R%d_STOP_REQUESTED'%round_number):
        env.pop(key,None)
    if device != 'cpu':
        visible = env.get('CUDA_VISIBLE_DEVICES','').strip()
        if not visible:
            visible = env.get('SLURM_STEP_GPUS',env.get('SLURM_JOB_GPUS','')).strip()
            match=re.fullmatch(r'(\d+)-(\d+)',visible)
            if match and match.group(1)==match.group(2):visible=match.group(1)
            if visible:
                env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[0].strip()
        if not visible or visible in ('-1','NoDevFiles','none','void'):
            raise ValueError('No GPU assigned by this single-GPU Slurm job')
        if ',' in visible:
            env['CUDA_VISIBLE_DEVICES']=visible.split(',')[0].strip()
        # Do not map logical worker slot 1/2 onto physical GPU 1/2. Each Slurm
        # job already exposes its own device; the scientific worker uses cuda:0.
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='16',
        MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        **{'PAPER3R%d_GPU_SLOT'%round_number:str(slot)})
    return env


def publish_progress(plan_path, runtime=None, jobs=None, round_number=None, clock=time.time):
    _, plan, output, runtime, _ = plan_context(plan_path,runtime)
    state = read(runtime/'tasks.json',{})
    rows = []
    for index, task in enumerate(plan['tasks']):
        operation = state.get('tasks',{}).get(str(index),{})
        done = result_complete(output,task,plan.get('recipe_sha256'))
        progress_path = output/task['key']/'progress.json'
        progress = read(progress_path,{}) if not done else {}
        rows.append({'index':index,'key':task['key'],'family':task.get('family','other'),
            'status':'COMPLETE' if done else operation.get('phase','NEW'),
            'slot':operation.get('slot'),'job_id':operation.get('job_id'),
            'attempt':operation.get('attempt',0),'retry_at':operation.get('retry_at'),
            'progress_path':str(progress_path),
            'worker_progress':{k:progress[k] for k in ('status','stage','outer_update','total_updates',
                  'session','sessions_total','budget','updated_utc') if k in progress}})
    complete = sum(r['status']=='COMPLETE' for r in rows)
    slots = {str(slot):read(runtime/('slot-%d.json'%slot),{'status':'NOT_STARTED'}) for slot in range(3)}
    value = {'schema_version':'paper3-three-independent-gpu-jobs-v1','round':round_number,
        'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(clock())),
        'status':'COMPLETE' if complete==len(rows) else 'RUNNING',
        'phase':'COMPLETE' if complete==len(rows) else 'TASK_DAG',
        'recipe_sha256':plan.get('recipe_sha256'),'complete_tasks':complete,'total_tasks':len(rows),
        'tasks':rows,'jobs':jobs or [],'slots':slots,
        'resources':{'jobs':3,'gpus_total':3,'cpus_per_job':16,'memory_per_job':'64G',
                     'cpus_total':48,'memory_total':'192G','partition':'hpc_gpu',
                     'gpu_model_requested_by_group':'A100 40GB; actual name recorded by scientific worker','excluded_nodes':[]},
        'plan_identity_preserved':True,'native_checkpoints_reused':True}
    atomic(runtime/'progress.json',value)
    return value


def controller_step(plan_path, launch, runtime=None, *, runner=None, clock=None, emit_callback=emit):
    _, plan, _, runtime, _ = plan_context(plan_path,runtime)
    completed = all(result_complete(Path(plan['output']),t,plan.get('recipe_sha256')) for t in plan['tasks'])
    scheduler = ThreeJobScheduler(runtime,{**launch,'campaign_identity':plan.get('recipe_sha256'),
        'submit_hold':False},runner=runner,clock=clock,emit=emit_callback)
    summary = scheduler.step(keep_workers=not completed)
    progress = publish_progress(plan_path,runtime,summary['jobs'],launch['round'],clock or time.time)
    return {**summary,'progress':progress}


def worker_main(args, *, popen=None, sleep=time.sleep):
    plan_path, plan, output, runtime, original_plan_sha = plan_context(args.plan,args.runtime)
    slot, round_number = int(args.slot), int(args.round_number)
    job_id = os.environ.get('SLURM_JOB_ID','CPU_TEST')
    stop = threading.Event()
    previous = {}
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig,lambda *_:stop.set())
    popen = popen or subprocess.Popen
    code = 2
    active = None
    emit('THREE_JOBS_WORKER_STARTED',round=round_number,slot=slot,job_id=job_id,
         source_package=str(args.package),original_output=str(output))
    try:
        while not stop.is_set():
            if (runtime/'STOP').exists():
                stop.set();break
            if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=original_plan_sha:
                raise ValueError('Original scientific plan changed while worker was active')
            if all(result_complete(output,t,plan.get('recipe_sha256')) for t in plan['tasks']):
                break
            claim = claim_task(plan,output,runtime,slot,job_id)
            if claim is None:
                atomic(runtime/('slot-%d.json'%slot),{'status':'WAITING_DEPENDENCIES','slot':slot,'job_id':job_id})
                sleep(args.poll_seconds);continue
            index, lease, attempt = claim
            task = plan['tasks'][index]
            logs = runtime/'logs';logs.mkdir(exist_ok=True)
            stdout_path = logs/('task-%d-attempt-%d.out'%(index,attempt))
            stderr_path = logs/('task-%d-attempt-%d.err'%(index,attempt))
            try:
                command = worker_command(args.python,args.package,round_number,plan_path,index,args.device)
                env = child_environment(round_number,slot,args.device)
                with stdout_path.open('ab',buffering=0) as stdout,stderr_path.open('ab',buffering=0) as stderr:
                    active = popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                        env=env,start_new_session=True,close_fds=True)
                    emit('THREE_JOBS_TASK_STARTED',round=round_number,slot=slot,task=task['key'],
                         pid=active.pid,attempt=attempt,job_id=job_id)
                    sent = False; heartbeat = 0
                    while active.poll() is None:
                        if (runtime/'STOP').exists():stop.set()
                        if stop.is_set() and not sent:
                            try: active.send_signal(signal.SIGUSR1)
                            except ProcessLookupError: pass
                            sent = True
                        if time.monotonic()>=heartbeat:
                            progress=read(output/task['key']/'progress.json',{})
                            atomic(runtime/('slot-%d.json'%slot),{'status':'CHECKPOINTING' if sent else 'RUNNING',
                                'slot':slot,'job_id':job_id,'task_index':index,'task_key':task['key'],
                                'pid':active.pid,'attempt':attempt,'stdout':str(stdout_path),'stderr':str(stderr_path),
                                'updated_at':time.time(),'worker_progress':progress})
                            emit('THREE_JOBS_TASK_PROGRESS',round=round_number,slot=slot,task=task['key'],
                                progress={k:progress[k] for k in ('status','stage','outer_update','total_updates',
                                    'session','sessions_total','budget','updated_utc') if k in progress})
                            heartbeat=time.monotonic()+30
                        sleep(.5)
                    result_code=active.wait();active=None
                phase=finish_task(plan,output,runtime,index,result_code,stop.is_set())
                emit('THREE_JOBS_TASK_RETURNED',round=round_number,slot=slot,task=task['key'],
                     exit_code=result_code,status=phase)
                if result_code==75:stop.set()
            except Exception as error:
                if active is not None and active.poll() is None:
                    try: active.send_signal(signal.SIGUSR1)
                    except ProcessLookupError: pass
                    active.wait()
                    active=None
                finish_task(plan,output,runtime,index,2,stop.is_set())
                emit('THREE_JOBS_TASK_RETRY',round=round_number,slot=slot,task=task['key'],
                     error_type=type(error).__name__, reason=str(error)[:600], stderr=str(stderr_path))
            finally:
                lease.close()
        code=75 if stop.is_set() else 0
        return code
    finally:
        atomic(runtime/('slot-%d.json'%slot),{'status':'PAUSED' if code==75 else 'COMPLETE' if code==0 else 'FAILED',
            'slot':slot,'job_id':job_id,'exit_code':code,'updated_at':time.time()})
        if job_id.isdigit():write_worker_outcome(runtime,slot,job_id,code,round=round_number)
        for sig,handler in previous.items():signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('worker','status'))
    parser.add_argument('--plan',required=True)
    parser.add_argument('--runtime')
    parser.add_argument('--round',dest='round_number',type=int,required=True,choices=(9,))
    parser.add_argument('--slot',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--package')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    parser.add_argument('--poll-seconds',type=float,default=3)
    args=parser.parse_args(argv)
    if args.mode=='worker':
        if not args.package:parser.error('--package is required for worker')
        return worker_main(args)
    print(json.dumps(publish_progress(args.plan,args.runtime,round_number=args.round_number),ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
