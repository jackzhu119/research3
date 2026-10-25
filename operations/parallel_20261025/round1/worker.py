#!/usr/bin/env python3
"""One independent GPU supervisor; only its child imports scientific kernels."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid


def emit(event, **fields):
    try:
        print(json.dumps({'event':event,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields}, ensure_ascii=False), flush=True)
    except OSError:
        pass


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid())+'-'+threading.current_thread().name)
    with temporary.open('w') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def action_hash(action):
    return hashlib.sha256(str(action['key']).encode()).hexdigest()


@contextmanager
def queue_lock(runtime):
    with (Path(runtime)/'.queue.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def _bridge(package):
    package=str(Path(package).resolve())
    if package not in sys.path:
        sys.path.insert(0,package)
    import kernel_bridge
    return kernel_bridge


def load_direct(direct_package):
    path=Path(direct_package)/'direct_runner.py'
    spec=importlib.util.spec_from_file_location('_r1_original_direct_runner',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def claim_next(root,runtime,slot,job_id,bridge,clock=time.time):
    """Return a locked action and claim; never clear another worker's lease."""
    runtime=Path(runtime)
    with queue_lock(runtime):
        queue=read_json(runtime/'queue.json',{}) or {}
        if queue.get('status')=='COMPLETE':
            return None,'COMPLETE'
        if queue.get('status')!='RUNNING':
            return None,'WAITING_QUEUE'
        claims=read_json(runtime/'claims.json',{}) or {}
        backoff=read_json(runtime/'action-backoff.json',{}) or {}
        if slot in queue.get('retired_slots',[]) and not any(int(c.get('slot',-1))==slot for c in claims.values()):
            return None,'COMPLETE'
        if any(int(c.get('slot',-1))==slot for c in claims.values()):
            return None,'WAITING_OWN_LEASE_RECONCILIATION'
        if any(c.get('serial') for c in claims.values()):
            return None,'WAITING_SERIAL_ACTION'
        for action in queue.get('ready_actions',[]):
            if action.get('affinity_slot') not in (None,slot):
                continue
            key=action['key']
            if key in claims or clock()<backoff.get(key,{}).get('retry_after',0):
                continue
            if action.get('serial') and claims:
                continue
            if bridge.action_complete(root,action):
                continue
            lock_path=runtime/'action-locks'/(action_hash(action)+'.lock')
            lock_path.parent.mkdir(parents=True,exist_ok=True)
            action_lock=lock_path.open('a')
            try:
                fcntl.flock(action_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                action_lock.close();continue
            # The queue lock and this action lock remain held through durable claim.
            claim={'slot':slot,'job_id':str(job_id),'pid':os.getpid(),'token':uuid.uuid4().hex,
                   'claimed_utc':clock(),'serial':bool(action.get('serial')),
                   'action_kind':action.get('kind')}
            claims[key]=claim
            try:
                atomic_json(runtime/'claims.json',claims)
            except BaseException:
                action_lock.close();raise
            return (action,claim,action_lock),'CLAIMED'
        return None,'WAITING_READY_ACTION'


def release_claim(runtime,action,claim,*,retry=False,error=None,clock=time.time):
    runtime=Path(runtime)
    with queue_lock(runtime):
        claims=read_json(runtime/'claims.json',{}) or {}
        current=claims.get(action['key'],{})
        if current.get('token')==claim['token']:
            claims.pop(action['key'],None)
            atomic_json(runtime/'claims.json',claims)
        if retry:
            backoff=read_json(runtime/'action-backoff.json',{}) or {}
            backoff[action['key']]={'retry_after':clock()+60,'error':error,
                                    'job_id':claim['job_id'],'attempt_token':claim['token']}
            atomic_json(runtime/'action-backoff.json',backoff)


PROGRESS_KEYS=('status','stage','phase','outer_update','total_updates','session','sessions_total',
    'budget','outer_updates_completed','inner_updates_actual','episodes_completed',
    'confirmed_outer_updates_observed','last_observed_trajectory_update','observed_optimizer_calls_total')


def progress_for_action(root,action):
    folder=Path(root)/Path(action['expected_receipt']).parent
    rows=[]
    for name in ('progress.json','execution_progress.json','receipt.json'):
        path=folder/name
        value=read_json(path,{}) or {}
        if value:
            rows.append({'file':str(path.relative_to(root)),
                         'age_seconds':round(max(0,time.time()-path.stat().st_mtime),1),
                         'values':{k:value[k] for k in PROGRESS_KEYS if k in value}})
    return rows


def worker_status(args,phase,*,action=None,claim=None,child_pid=None,error=None):
    value={'slot':args.slot,'job_id':os.environ.get('SLURM_JOB_ID','local'),
           'pid':os.getpid(),'phase':phase,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'updated_at':time.time(),'child_pid':child_pid,'error':error}
    if action is not None:
        value.update(action_key=action['key'],action_kind=action.get('kind'),
                     expected_receipt=action['expected_receipt'],
                     action_progress=progress_for_action(Path(args.root),action))
    if claim is not None:
        value['claim_token']=claim['token']
    atomic_json(Path(args.runtime_dir)/'worker-status'/('slot-%d.json'%args.slot),value)
    return value


_observer_gate = {'thread': None}


def run_child(command,*,env,cwd,log,stop,observe,token,deadline):
    """Drain output independently of any slow shared-filesystem observation."""
    marker=('KERNEL_READY '+token).encode()
    ready=False;sent=False;recent=b'';last_observe=0;observer=None;last_notice=0
    with Path(log).open('ab',buffering=0) as output:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,close_fds=True)
        def observe_safely():
            try: observe(process.pid)
            except Exception as error:
                emit('R1_WORKER_OBSERVER_RETRY',error_type=type(error).__name__)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout,selectors.EVENT_READ)
                while selector.get_map() or process.poll() is None:
                    if time.monotonic()>=deadline:
                        stop.set()
                    if stop.is_set() and ready and not sent and process.poll() is None:
                        try:
                            process.send_signal(signal.SIGINT);sent=True
                            emit('R1_KERNEL_CHECKPOINT_REQUESTED',child_pid=process.pid)
                        except ProcessLookupError:
                            pass
                    if time.monotonic()-last_observe>=15:
                        observer=_observer_gate['thread']
                        if observer is None or not observer.is_alive():
                            observer=threading.Thread(target=observe_safely,daemon=True,name='progress-observer')
                            _observer_gate['thread']=observer
                            observer.start()
                        last_observe=time.monotonic()
                    if time.monotonic()-last_notice>=30:
                        emit('R1_KERNEL_RUNNING',child_pid=process.pid,kernel_ready=ready,
                             checkpoint_stop_requested=stop.is_set(),checkpoint_signal_sent=sent)
                        last_notice=time.monotonic()
                    for key,_ in selector.select(timeout=.5):
                        block=os.read(key.fileobj.fileno(),65536)
                        if not block:
                            selector.unregister(key.fileobj);key.fileobj.close();continue
                        try: output.write(block)
                        except OSError as error:
                            emit('R1_KERNEL_LOG_WRITE_RETRY',error_type=type(error).__name__)
                        if not ready:
                            recent=(recent+block)[-65536:]
                            if marker in recent:
                                ready=True
                                emit('R1_KERNEL_READY_OBSERVED',child_pid=process.pid)
                        try:
                            sys.stdout.write(block.decode('utf-8',errors='replace'));sys.stdout.flush()
                        except OSError:
                            pass
                if observer is not None:
                    observer.join(timeout=.2)
                return process.wait()
        except BaseException:
            # A supervision I/O fault must not release the numerical action lock
            # while its child still computes. Drain until ready, checkpoint, wait.
            stop.set()
            emit('R1_CHILD_DRAINING_AFTER_SUPERVISOR_ERROR',child_pid=process.pid)
            while process.poll() is None:
                if ready and not sent:
                    try:process.send_signal(signal.SIGINT);sent=True
                    except ProcessLookupError:pass
                try:block=os.read(process.stdout.fileno(),65536)
                except (OSError,ValueError):
                    time.sleep(.1);continue
                if block:
                    recent=(recent+block)[-65536:]
                    ready=ready or marker in recent
                    try:output.write(block)
                    except OSError:pass
                else:
                    time.sleep(.1)
            process.wait()
            raise


def child_main(args):
    record=read_json(args.action_file)
    action,token=record['action'],record['claim']['token']
    result_path=Path(args.action_file).with_name('returned.json')
    bridge=_bridge(args.package)
    requested={'value':False}
    def interrupted(*_):
        requested['value']=True
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,interrupted)
    def ready():
        print('KERNEL_READY '+token,flush=True)
    try:
        result=bridge.perform_action(Path(args.root),action,Path(args.payload_manifest),
            allocation_seconds=args.allocation_seconds,direct_package=args.direct_package,
            stop_requested=lambda:requested['value'],on_ready=ready)
        complete=bridge.action_complete(Path(args.root),action)
        row={'status':'COMPLETE' if complete else 'PAUSED','result':result,
             'action_key':action['key'],'claim_token':token,'receipt_complete':complete}
        atomic_json(result_path,row)
        emit('R1_ACTION_RETURNED',action_key=action['key'],status=row['status'])
        return 0 if complete else 75
    except KeyboardInterrupt:
        row={'status':'PAUSED_CHECKPOINT_UNWIND','action_key':action['key'],'claim_token':token,
             'error_type':'KeyboardInterrupt','receipt_complete':bridge.action_complete(Path(args.root),action)}
        atomic_json(result_path,row)
        emit('R1_ACTION_CHECKPOINT_UNWOUND',action_key=action['key'])
        return 75
    except BaseException as error:
        row={'status':'ERROR','action_key':action['key'],'claim_token':token,
             'error_type':type(error).__name__,'message':str(error)[:4000]}
        atomic_json(result_path,row)
        traceback.print_exc()
        emit('R1_ACTION_ERROR',action_key=action['key'],error_type=type(error).__name__,reason=str(error)[:2000])
        return 2


def supervisor_main(args):
    root,runtime=Path(args.root).resolve(),Path(args.runtime_dir).resolve()
    runtime.mkdir(parents=True,exist_ok=True)
    bridge=_bridge(args.package)
    from scheduler import write_worker_outcome
    job_id=os.environ.get('SLURM_JOB_ID','local')
    stop=threading.Event()
    for number in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
        signal.signal(number,lambda *_:stop.set())
    deadline=time.monotonic()+max(1,args.allocation_seconds-300)
    code=2
    with (runtime/('worker-slot-%d.lock'%args.slot)).open('a') as slot_lock:
        try:
            fcntl.flock(slot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            emit('R1_SLOT_ALREADY_RUNNING',slot=args.slot)
            return 75
        try:
            direct=load_direct(args.direct_package)
            cache=direct.reuse_node_cache(root,str(args.prior_job))
            env=direct.child_environment(root,Path(args.payload_manifest),args.allocation_seconds,cache_paths=cache)
            env['PYTHONPATH']=os.pathsep.join([str(Path(args.package).resolve()),env.get('PYTHONPATH','')])
            emit('R1_WORKER_STARTED',slot=args.slot,job_id=job_id,node=os.uname().nodename,
                 checkpoint_identity_preserved=True)
            while True:
                if time.monotonic()>=deadline or (runtime/'STOP').exists():stop.set()
                if stop.is_set():
                    code=75;worker_status(args,'CHECKPOINTED_OR_IDLE_STOP');break
                try:
                    claimed,reason=claim_next(root,runtime,args.slot,job_id,bridge)
                except Exception as error:
                    emit('R1_CLAIM_RETRY',slot=args.slot,error_type=type(error).__name__,reason=str(error)[:1000])
                    stop.wait(5);continue
                if claimed is None:
                    worker_status(args,reason)
                    if reason=='COMPLETE':code=0;break
                    stop.wait(5);continue
                action,claim,action_lock=claimed
                try:
                    folder=runtime/'actions'/action_hash(action)/'attempts'/claim['token']
                    folder.mkdir(parents=True,exist_ok=True)
                    action_file=folder/'action.json'
                    atomic_json(action_file,{'action':action,'claim':claim})
                    worker_status(args,'ACTION_STARTING',action=action,claim=claim)
                    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--child',
                        '--action-file',str(action_file),'--root',str(root),'--runtime-dir',str(runtime),
                        '--slot',str(args.slot),'--package',str(args.package),
                        '--payload-manifest',str(args.payload_manifest),'--direct-package',str(args.direct_package),
                        '--prior-job',str(args.prior_job),'--allocation-seconds',str(args.allocation_seconds)]
                    def observe(pid):
                        if (runtime/'STOP').exists():stop.set()
                        worker_status(args,'ACTION_RUNNING',action=action,claim=claim,child_pid=pid)
                    child_code=run_child(command,env=env,cwd=root,log=folder/'kernel.log',
                                         stop=stop,observe=observe,token=claim['token'],deadline=deadline)
                    returned=read_json(folder/'returned.json',{}) or {}
                    complete=bridge.action_complete(root,action)
                    outcome={'action_key':action['key'],'slot':args.slot,'job_id':str(job_id),
                        'claim_token':claim['token'],'returncode':child_code,'receipt_complete':complete,
                        'status':'COMPLETE' if complete else ('PAUSED' if child_code==75 else 'ERROR'),
                        'returned':returned,'log':str(folder/'kernel.log'),'finished_at':time.time()}
                    atomic_json(runtime/'action-outcomes'/(action_hash(action)+'-'+str(job_id)+'.json'),outcome)
                    atomic_json(runtime/'actions'/action_hash(action)/'latest.json',outcome)
                    release_claim(runtime,action,claim,retry=not complete and not stop.is_set(),
                                  error=returned.get('error_type','ExitCode'+str(child_code)))
                    emit('R1_ACTION_FINISHED',slot=args.slot,action_key=action['key'],
                         returncode=child_code,receipt_complete=complete,log=str(folder/'kernel.log'))
                    if stop.is_set():code=75;break
                except BaseException as error:
                    # run_child only returns after the child has ended; failures
                    # before its creation do not own a numerical kernel process.
                    release_claim(runtime,action,claim,retry=True,error=type(error).__name__)
                    emit('R1_ACTION_SUPERVISOR_RETRY',slot=args.slot,action_key=action['key'],
                         error_type=type(error).__name__,reason=str(error)[:2000])
                    if stop.is_set():code=75;break
                finally:
                    action_lock.close()
            return code
        finally:
            write_worker_outcome(runtime,args.slot,job_id,code)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--runtime-dir',required=True)
    parser.add_argument('--slot',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--payload-manifest',required=True)
    parser.add_argument('--direct-package',required=True)
    parser.add_argument('--package',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--prior-job',default='769863')
    parser.add_argument('--allocation-seconds',type=int,default=442800)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--action-file',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.allocation_seconds<301:parser.error('allocation-seconds must exceed checkpoint grace')
    return child_main(args) if args.child else supervisor_main(args)


if __name__=='__main__':
    raise SystemExit(main())
