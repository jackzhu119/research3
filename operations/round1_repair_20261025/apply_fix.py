"""Install the R1 wrapper correction in place; never alter scientific outputs."""
import argparse
import ast
import fcntl
import hashlib
import json
import getpass
import os
from pathlib import Path
import stat
import re
import subprocess
import sys
import time

MARKER='PAPER3_ACTION_STARTUP_CONTEXT_V1'
MODULES=('audited_runtime.py','hardware_comparison.py','action_startup.py')


def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),ensure_ascii=False),flush=True)


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def option(argv,flag):
    try:return argv[argv.index(flag)+1]
    except (ValueError,IndexError):return None


def package_from(value):
    if not isinstance(value,str):return None
    p=Path(value).expanduser().resolve()
    if (p/'kernel_bridge.py').is_file() and (p/'worker.py').is_file():return p
    if (p/'round1/kernel_bridge.py').is_file() and (p/'round1/worker.py').is_file():return p/'round1'
    return None


def discover_packages(runtime, *, home=None):
    runtime=Path(runtime);launch=read(runtime/'launch.json',{}) or {};state=read(runtime/'scheduler-state.json',{}) or {}
    raw=[launch.get('package')];command=launch.get('worker_command',[])
    if isinstance(command,list):raw.append(option(command,'--package'))
    for slot in state.get('slots',{}).values():
        for attempt in slot.get('attempts',[]):
            command=attempt.get('command',[])
            if isinstance(command,list):raw.append(option(command,'--package'))
    found={p for v in raw if (p:=package_from(v)) is not None}
    if not found:
        base=Path(home or Path.home())/'paper3-parallel-migrations'
        found={p.parent for p in base.glob('*/round1/kernel_bridge.py') if (p.parent/'worker.py').is_file()}
    return sorted(found)


def patch_bridge(raw):
    source=raw.decode();tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='perform_action']
    if len(functions)!=1:raise ValueError('single_perform_action_required')
    node=functions[0]
    if MARKER in source:
        matching=[n for n in ast.walk(node) if isinstance(n,ast.With) and any(
            isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Attribute)
            and i.context_expr.func.attr=='active' for i in n.items)]
        if len(matching)!=1:raise ValueError('startup_marker_without_single_active_context')
        return raw
    positions=[]
    for i,n in enumerate(node.body):
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='activate_runtime':
            positions.append(i)
    if len(positions)!=1:raise ValueError('single_runtime_activation_required')
    at=positions[0]+1;tail=node.body[at:]
    if not tail:raise ValueError('action_tail_missing')
    statements=ast.parse(repr(MARKER)+'\nimport action_startup\nwith action_startup.active(root, action):\n    pass\n').body
    statements[-1].body=tail
    node.body=node.body[:at]+statements
    replacement=ast.unparse(ast.fix_missing_locations(node))+'\n'
    lines=source.splitlines(keepends=True);result=''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])
    ast.parse(result)
    return result.encode()


def atomic_file(path,data,mode):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-fix-'+str(os.getpid()))
    try:
        with temp.open('wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.chmod(temp,mode);os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def apply(runtime,payload, *, home=None):
    runtime=Path(runtime).resolve();payload=Path(payload).resolve();packages=discover_packages(runtime,home=home)
    if not packages:raise FileNotFoundError('Existing R1 package not found in launch/scheduler metadata')
    supplied={n:(payload/n).read_bytes() for n in MODULES}
    for name,data in supplied.items():ast.parse(data.decode(),filename=name)
    plan=[]
    for package in packages:
        bridge=package/'kernel_bridge.py'
        if bridge.is_symlink() or bridge.stat().st_uid!=os.getuid():raise ValueError('R1 wrapper file not owned regular path')
        values=dict(supplied);values['kernel_bridge.py']=patch_bridge(bridge.read_bytes())
        for name,data in values.items():
            target=package/name
            if target.is_symlink():raise ValueError('R1 wrapper target symlink')
            if target.exists() and target.stat().st_uid!=os.getuid():raise ValueError('R1 wrapper target not owned')
            old=target.read_bytes() if target.exists() else None
            mode=stat.S_IMODE(target.stat().st_mode) if old is not None else 0o600
            plan.append((package,target,data,old,mode))
    root=runtime/'repair-backups'/'gpu-runtime-fix-v1';root.mkdir(parents=True,exist_ok=True)
    changed=[]
    with (runtime/'hardware-fix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for package,target,data,old,mode in plan:
            if old==data:continue
            digest=hashlib.sha256(old).hexdigest() if old is not None else None
            backup=None
            if old is not None:
                backup=root/hashlib.sha256(str(package).encode()).hexdigest()[:16]/(target.name+'.'+digest)
                if not backup.exists():atomic_file(backup,old,mode)
            atomic_file(target,data,mode)
            changed.append(dict(path=str(target),old_sha256=digest,new_sha256=hashlib.sha256(data).hexdigest(),
                                backup=str(backup) if backup else None,mode=oct(mode)))
        receipt=dict(schema_version='paper3-r1-wrapper-hotfix-v1',packages=[str(p) for p in packages],files_changed=changed,
                     original_checkpoint_and_scientific_sources_changed=False,slurm_allocations_cancelled=0,
                     note='Already-loaded children retain old functions until they return or are refreshed; future children read corrected files.')
        atomic_file(runtime/'hardware-fix-applied.json',(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n').encode(),0o600)
    emit('R1_WRAPPER_FIX_APPLIED',packages=[str(p) for p in packages],files_changed=len(changed),checkpoint_files_changed=0)
    return receipt


def refresh_current(runtime,node_helper, *, python=sys.executable,runner=subprocess.run):
    runtime=Path(runtime).resolve();results=[]
    for slot in range(3):
        row=read(runtime/'worker-status'/('slot-%d.json'%slot),{}) or {}
        job=str(row.get('job_id',''))
        if not job.isdigit() or not row.get('child_pid'):continue
        command=['srun','--jobid='+job,'--overlap','-N1','-n1','-c1','--mem=256M','--gres=none','--time=00:02:00',
                 python,'-I','-S','-B',str(Path(node_helper).resolve()),'--runtime',str(runtime),'--job-id',job,'--slot',str(slot)]
        try:
            response=runner(command,capture_output=True,text=True,timeout=40)
            if response.returncode==0:
                output=response.stdout.strip();emit('R1_CHILD_REFRESH_RESULT',job_id=job,slot=slot,result=output[-3000:])
                results.append(dict(job_id=job,slot=slot,returncode=0,result=output))
            else:
                emit('WAIT_NEXT_CHILD',job_id=job,slot=slot,reason='allocation_step_not_available',returncode=response.returncode)
                results.append(dict(job_id=job,slot=slot,status='WAIT_NEXT_CHILD'))
        except (OSError,subprocess.TimeoutExpired) as error:
            emit('WAIT_NEXT_CHILD',job_id=job,slot=slot,reason=type(error).__name__)
            results.append(dict(job_id=job,slot=slot,status='WAIT_NEXT_CHILD'))
    return results


def checkpoint_restart(runtime, *, runner=subprocess.run, python=sys.executable):
    """Use the existing controller to replace only this round's running workers."""
    runtime=Path(runtime).resolve()
    launch=read(runtime/'launch.json',{}) or {}
    state=read(runtime/'scheduler-state.json',{}) or {}
    marker=state.get('marker','')
    if not re.fullmatch(r'paper3r1p-[0-9a-f]{16}',str(marker)):
        raise ValueError('Existing first-round scheduler marker missing')
    # The resident controller must own its lock before any worker is asked to
    # checkpoint and exit. Start that same controller if it is no longer alive.
    with (runtime/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            controller_ready=True
        else:
            fcntl.flock(lock,fcntl.LOCK_UN)
            package=package_from(launch.get('package'))
            if package is None or not (package/'START.py').is_file():
                raise ValueError('Existing controller package not found; allocations retained')
            with (runtime/'controller.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([python,'-I','-S','-B','-u',str(package/'START.py'),
                    '--controller','--root',str(runtime.parents[1])],stdin=subprocess.DEVNULL,
                    stdout=log,stderr=log,start_new_session=True,close_fds=True)
            emit('R1_EXISTING_CONTROLLER_STARTED',pid=child.pid)
            controller_ready=False
            for _ in range(25):
                time.sleep(.2)
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    controller_ready=True;break
                else:fcntl.flock(lock,fcntl.LOCK_UN)
    if not controller_ready:
        emit('R1_RESTART_DEFERRED',reason='controller_starting_allocations_retained')
        return []
    try:
        response=runner(['squeue','--noheader','--user',getpass.getuser(),
            '--format=%i|%k|%T|%j'],capture_output=True,text=True,timeout=20)
    except (OSError,subprocess.TimeoutExpired) as error:
        emit('R1_RESTART_DEFERRED',reason=type(error).__name__,allocations_retained=True)
        return []
    if response.returncode:
        emit('R1_RESTART_DEFERRED',reason='queue_read_failed_allocations_retained')
        return []
    requested=[]
    for line in response.stdout.splitlines():
        row=[s.strip() for s in line.split('|')]
        if (len(row)<4 or not row[0].isdigit() or row[1]!=marker or row[2]!='RUNNING'
                or not re.fullmatch(re.escape(marker)+r'-s[012]-a[1-9][0-9]*',row[3])):
            continue
        # USR1 is the batch script's checkpoint-and-exit request. It is not a
        # force cancellation; the existing controller observes exit and submits
        # the next attempt with the same scientific state and corrected files.
        try:
            result=runner(['scancel','--signal=USR1','--batch',row[0]],
                capture_output=True,text=True,timeout=20)
        except (OSError,subprocess.TimeoutExpired) as error:
            emit('R1_RESTART_DEFERRED',job_id=row[0],reason=type(error).__name__,allocations_retained=True)
            continue
        emit('R1_CHECKPOINT_REQUEUE_REQUESTED',job_id=row[0],returncode=result.returncode,
            signal='USR1',pending_jobs_preserved=True,force_kill=False)
        if result.returncode==0:requested.append(row[0])
    return requested


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--payload-dir',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--apply',action='store_true');parser.add_argument('--refresh-current',action='store_true')
    parser.add_argument('--restart-workers',action='store_true',help='Checkpoint first-round running workers and let their existing controller submit corrected attempts.')
    args=parser.parse_args();runtime=Path(args.root).expanduser().resolve()/'runs/training_campaign_parallel_v1'
    if not args.apply:
        emit('R1_FIX_PREVIEW',packages=[str(p) for p in discover_packages(runtime)],training_jobs_changed=0)
        return 0
    apply(runtime,Path(args.payload_dir))
    if args.restart_workers:
        checkpoint_restart(runtime)
    elif args.refresh_current:
        refresh_current(runtime,Path(__file__).with_name('node_refresh.py'))
    emit('R1_FIX_READY_FOR_EXISTING_SUPERVISORS',replacement_submission_by_existing_controller=bool(args.restart_workers),checkpoint_files_preserved=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
