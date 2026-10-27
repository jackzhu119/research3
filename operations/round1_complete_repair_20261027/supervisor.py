"""Detached R1 recovery: reuse allocations, refresh owned action children."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import parallel_runtime as pr
import install_hooks


def identity(pid,proc_root=Path('/proc')):
    try:
        pid=int(pid)
        if pid<=1:return None
        p=Path(proc_root)/str(pid)
        if p.stat().st_uid!=os.getuid():return None
        text=(p/'stat').read_text(); fields=text[text.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        argv=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        return dict(pid=pid,uid=p.stat().st_uid,start_ticks=fields[19],argv=argv)
    except (OSError,ValueError,IndexError,UnicodeError):return None


def option(argv,key):
    if argv.count(key)!=1:return None
    i=argv.index(key)
    return argv[i+1] if i+1<len(argv) else None


def lock_live(path):
    path=Path(path)
    if not path.exists():return False
    with path.open('r') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(handle,fcntl.LOCK_UN)
    return False


def owned_controller(pid,root,original,*,proc_root=Path('/proc')):
    value=identity(pid,proc_root)
    if value is None:return None
    a=value['argv']; requested=option(a,'--root')
    if not requested or Path(requested).resolve()!=Path(root).resolve():return None
    if not a or Path(a[0]).resolve()!=Path(sys.executable).resolve():return None
    at=1
    while at<len(a) and a[at] in ('-I','-S','-B','-u','-E','-s','-O','-OO'):at+=1
    if at>=len(a):return None
    script=Path(a[at]).resolve()
    old=script==Path(original).resolve()/'START.py' and '--controller' in a[at+1:] and '--child' not in a
    new=script==HERE/'supervisor.py' and len(a)>at+1 and a[at+1]=='manage'
    return dict(value,current_patch=new) if old or new else None


def find_controllers(root,original,proc_root=Path('/proc')):
    found=[]
    for p in Path(proc_root).iterdir():
        if p.name.isdigit():
            value=owned_controller(p.name,root,original,proc_root=proc_root)
            if value:found.append(value)
    return found


def stop_old_controller(row,root,original,*,send=os.kill,sleep=time.sleep,clock=time.monotonic):
    def unchanged():
        value=owned_controller(row['pid'],root,original)
        return value is not None and all(value[k]==row[k] for k in ('pid','uid','start_ticks','argv'))
    if not unchanged():return dict(status='CONTROLLER_IDENTITY_CHANGED',signals=[])
    try:send(row['pid'],signal.SIGTERM)
    except ProcessLookupError:return dict(status='CONTROLLER_ALREADY_ENDED',signals=[])
    signals=['SIGTERM'];until=clock()+10
    while unchanged() and clock()<until:sleep(.1)
    if unchanged():
        try:send(row['pid'],signal.SIGKILL);signals.append('SIGKILL')
        except ProcessLookupError:pass
    return dict(status='OWNED_MANAGEMENT_CONTROLLER_STOP_REQUESTED',pid=row['pid'],signals=signals,
                scientific_children_signalled=0,slurm_allocations_cancelled=0)


def original_package(root,home):
    paths=install_hooks.discover_packages(root,home)
    launch=pr.read_json(Path(root)/pr.OPERATIONAL/'launch.json',{},required=True)
    preferred=launch.get('package') or option(launch.get('worker_command',[]),'--package')
    for p in paths:
        if preferred and p.resolve()==Path(preferred).resolve():return p
    if len(paths)==1:return paths[0]
    raise ValueError('EXACT_CURRENT_R1_OPERATIONAL_PACKAGE_REQUIRED')


def reference_runtime(root):
    path=Path(root)/pr.OPERATIONAL/'runtime_transfer_comparison.json'
    journal=pr.read_json(path,{}) or {}
    recorded=journal.get('hardware_class_profile')
    if isinstance(recorded,dict):
        from hardware_comparison import _profile, _raw_profile
        for value in journal.get('observations',{}).values():
            profile=value.get('runtime_profile',{})
            try: matches=_profile(profile)==recorded or _raw_profile(profile)==recorded
            except (ValueError,KeyError,TypeError):matches=False
            if matches and profile.get('timing_protocol') and profile.get('device_uuid'):
                return profile,dict(path=str(path),source='existing_primary_runtime_observation')
    for p in sorted((Path(root)/'runs/training_campaign_v1/profiles').glob('*/receipt.json')):
        value=pr.read_json(p,{}) or {};profile=value.get('runtime_profile')
        if value.get('status')=='DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE' and isinstance(profile,dict):
            return profile,dict(path=str(p),source='completed_primary_profile')
    return None,dict(source='PRIMARY_RUNTIME_NOT_YET_AVAILABLE')


def current_jobs(runtime):
    state=pr.read_json(Path(runtime)/'scheduler-state.json',{},required=True)
    result=[]
    for slot in range(3):
        attempts=state.get('slots',{}).get(str(slot),{}).get('attempts',[])
        if not attempts:continue
        a=attempts[-1]
        if a.get('job_id') and a.get('slurm_state')=='RUNNING' and a.get('status') not in ('FINISHED','REJECTED'):
            result.append((slot,str(a['job_id'])))
    return result


def node_call(mode,root,runtime,slot,job,python,*,runner=subprocess.run):
    command=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',
        '--mem='+('2G' if mode=='probe' else '256M'),'--gres='+('gpu:1' if mode=='probe' else 'none'),
        '--time=00:04:00',python,'-I','-S','-B',str(HERE/'newnode_tool.py'),mode,
        '--root',str(root),'--runtime',str(runtime),'--job-id',job,'--slot',str(slot)]
    try:
        response=runner(command,capture_output=True,text=True,timeout=180 if mode=='probe' else 40)
        for line in reversed(response.stdout.splitlines()):
            try:value=json.loads(line)
            except ValueError:continue
            if isinstance(value,dict) and str(value.get('job_id'))==job and value.get('slot')==slot:
                return dict(value,step_returncode=response.returncode)
        return dict(status='ALLOCATION_STEP_RETRY',job_id=job,slot=slot,returncode=response.returncode)
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(status='ALLOCATION_STEP_RETRY',job_id=job,slot=slot,error_type=type(error).__name__)


def watch_allocations(root,runtime,work,python,stop,*,runner=subprocess.run):
    """Observe existing allocations; never interrupt an action child.

    Missing repair markers during imports do not authorize SIGINT. Startup
    hooks are loaded naturally by each new action; this watcher only observes
    hardware so slow imports, new claims, retries and restarts cannot create a
    repeated checkpoint-refresh loop.
    """
    observed={};last_attempt={}
    pr.emit('R1_AUTOMATIC_CHILD_REFRESH_DISABLED',
        refresh_policy='DISABLED_NEW_ACTIONS_NEVER_SIGNALLED',
        scientific_children_signalled=0,live_allocations_signalled=0)
    while not stop.is_set():
        try:
            jobs=current_jobs(runtime);reference,source=reference_runtime(root)
            def one(item):
                slot,job=item; key=(slot,job); profile=observed.get(str(slot),{})
                if profile.get('job_id')!=job or not profile.get('verified'):
                    if time.monotonic()-last_attempt.get(key,-1e9)>=90:
                        last_attempt[key]=time.monotonic()
                        result=node_call('probe',root,runtime,slot,job,python,runner=runner)
                        good=(result.get('status')=='ACTUAL_RUNTIME_OBSERVED_NO_FIT'
                            and result.get('step_returncode')==0 and isinstance(result.get('runtime_profile'),dict))
                        observed[str(slot)]=dict(result,verified=good)
                        pr.put(work/'node-probes'/('slot-%d-%s.json'%(slot,job)),result)
                # Deliberately no refresh call. A new child can legitimately
                # lack repair-active.json until imports complete; every claim
                # and allocation replacement must remain free to finish.
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(one,jobs))
            # Readers bind every observation back to the current allocation ID.
            pr.put(work/'hardware-observations.json',dict(schema_version=pr.HARDWARE_SCHEMA,
                reference_runtime_profile=reference,reference_source=source,slots=observed,
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                refresh_policy='DISABLED_NEW_ACTIONS_NEVER_SIGNALLED',
                scientific_children_signalled=0,new_allocations_created_by_probe=False))
        except Exception as error:
            pr.emit('R1_EXISTING_ALLOCATION_OBSERVATION_RETRY',error_type=type(error).__name__)
        stop.wait(15)


def manage(args):
    root=Path(args.root).resolve();work=Path(args.work);runtime=root/pr.OPERATIONAL
    old=original_package(root,args.home);stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    with pr.lock(work/'manager.lock',blocking=False):
        rows=find_controllers(root,old)
        for row in rows:
            if row['pid']==os.getpid():continue
            if row.get('current_patch'):raise RuntimeError('ANOTHER_REPAIRED_CONTROLLER_IS_RUNNING')
            pr.put(work/('old-controller-%d.json'%row['pid']),stop_old_controller(row,root,old))
        until=time.monotonic()+15
        while lock_live(runtime/'controller.lock') and time.monotonic()<until:
            if stop.wait(.1):return 75
        if lock_live(runtime/'controller.lock'):raise RuntimeError('OLD_CONTROLLER_LOCK_NOT_RELEASED')
        # Restore the original operational import path before the asynchronous
        # probe reader uses its exact hardware-class canonicalizer.
        pr.load_operations(root,old)
        watcher=threading.Thread(target=watch_allocations,args=(root,runtime,work,args.python,stop),daemon=True)
        watcher.start()
        try:
            return pr.controller_loop(root,old,work,submit=args.submit,stop=stop,launch_token=args.launch_token)
        finally:stop.set();watcher.join(timeout=.5)


def start(args,*,popen=subprocess.Popen,clock=time.monotonic,sleep=time.sleep):
    if not args.submit:raise ValueError('EXPLICIT_SUBMIT_REQUIRED')
    root=Path(args.root).resolve();package=Path(args.package).resolve()
    work=Path(args.work or Path(args.home)/'paper3-r1-full-repair/work').resolve();work.mkdir(parents=True,exist_ok=True)
    with pr.lock(work/'start.lock'):
        applied=install_hooks.apply(root,args.home,package=package)
        old=original_package(root,args.home)
        rows=find_controllers(root,old)
        existing=next((r for r in rows if r.get('current_patch')),None)
        previous=pr.read_json(work/'controller-process.json',{}) or {}
        if existing and previous.get('pid')==existing['pid'] and previous.get('launch_token')==option(existing['argv'],'--launch-token'):
            ready=pr.read_json(work/'controller-ready.json',{}) or {}
            good=(ready.get('pid')==existing['pid'] and ready.get('launch_token')==previous['launch_token']
                  and ready.get('first_slurm_observation_completed') is True
                  and ready.get('queue_observation_successful') is True)
            return dict(status='EXISTING_R1_REPAIR_CONTROLLER_REUSED' if good else 'EXISTING_R1_REPAIR_CONTROLLER_STARTUP_PENDING',
                        pid=existing['pid'],work=str(work),terminal_can_close=True)
        token=hashlib.sha256(os.urandom(32)).hexdigest()
        command=[args.python,'-I','-S','-B','-u',str(HERE/'supervisor.py'),'manage','--root',str(root),
            '--home',str(args.home),'--package',str(package),'--work',str(work),'--submit','--launch-token',token,'--python',args.python]
        with (work/'controller.log').open('ab',buffering=0) as log:
            child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        pr.put(work/'controller-process.json',dict(pid=child.pid,launch_token=token,root=str(root),started_utc=time.time()))
        status='STARTUP_PENDING';until=clock()+8
        while True:
            ready=pr.read_json(work/'controller-ready.json',{}) or {}
            if child.poll() is not None:status='STARTUP_FAILED';break
            if (ready.get('pid')==child.pid and ready.get('launch_token')==token
                and ready.get('first_slurm_observation_completed') is True
                and ready.get('queue_observation_successful') is True):status='R1_REPAIR_CONTROLLER_READY';break
            if clock()>=until:break
            sleep(.1)
        value=dict(status=status,pid=child.pid,work=str(work),terminal_can_close=True,
            allocation_cancellations=0,scientific_files_changed=0,completed_results_reset=0,
            scientific_refresh='owned child SIGINT checkpoint unwind only; no SIGKILL',
            parallelism='independent actions on measured compatible slots; serial scientific barriers retained',
            hooks_applied=applied)
        pr.put(work/'STARTED.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('mode',choices=('start','check','manage'))
    p.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    p.add_argument('--home',type=Path,default=Path.home());p.add_argument('--package',type=Path,default=HERE)
    p.add_argument('--work',type=Path);p.add_argument('--python',default=sys.executable)
    p.add_argument('--submit',action='store_true');p.add_argument('--launch-token');p.add_argument('--refresh-pretraining',action='store_true')
    args=p.parse_args(argv)
    if args.mode=='manage':
        result=manage(args)
        return result if isinstance(result,int) else 0
    value=start(args) if args.mode=='start' else install_hooks.inspect(args.root,args.home)
    print(json.dumps(dict(event='R1_COMPLETE_REPAIR',**value),ensure_ascii=False),flush=True)
    return 2 if value.get('status')=='STARTUP_FAILED' else 0


if __name__=='__main__':raise SystemExit(main())
