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


def _controller_process(pid,proc_root=Path('/proc')):
    folder=Path(proc_root)/str(pid)
    try:
        if folder.stat().st_uid!=os.getuid():return None
        tail=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if tail[0] in ('Z','X','x'):return None
        argv=[p.decode(errors='replace') for p in (folder/'cmdline').read_bytes().split(b'\0') if p]
        return dict(pid=int(pid),uid=folder.stat().st_uid,start_ticks=int(tail[19]),argv=argv)
    except (OSError,ValueError,IndexError):return None


def _controller_matches(row,root,packages):
    argv=row.get('argv',[])
    if row.get('uid')!=os.getuid() or '--controller' not in argv or option(argv,'--root')!=str(Path(root).resolve()):return False
    allowed={str(Path(p).resolve()/'START.py') for p in packages}
    return any(token in allowed for token in argv)


def controller_processes(root,packages,proc_root=Path('/proc')):
    rows=[]
    try:folders=list(Path(proc_root).iterdir())
    except OSError:return rows
    for folder in folders:
        if folder.name.isdigit():
            row=_controller_process(int(folder.name),proc_root)
            if row and _controller_matches(row,root,packages):rows.append(row)
    return rows


def lock_held(path):
    with Path(path).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        else:fcntl.flock(lock,fcntl.LOCK_UN);return False


def standalone_launch(runtime, *, base=None):
    runtime=Path(runtime).resolve();launch=read(runtime/'launch.json',{}) or {}
    source=package_from(launch.get('package')) or package_from(option(launch.get('worker_command',[]),'--package'))
    if source is None or not (source/'START.py').is_file():raise ValueError('R1_controller_source_package_required')
    contents={p.name:p.read_bytes() for p in sorted(source.glob('*.py')) if p.is_file() and not p.is_symlink()}
    needed={'START.py','worker.py','kernel_bridge.py','audited_runtime.py','hardware_comparison.py','action_startup.py','scheduler.py'}
    if not needed<=set(contents):raise ValueError('standalone_R1_modules_incomplete')
    identity=hashlib.sha256(json.dumps({n:hashlib.sha256(c).hexdigest() for n,c in contents.items()},sort_keys=True).encode()).hexdigest()
    destination=Path(base or (Path(__file__).resolve().parent/'active-round1'))/identity[:20]
    destination.mkdir(parents=True,exist_ok=True)
    for name,data in contents.items():
        target=destination/name
        if not target.is_file() or target.read_bytes()!=data:
            atomic_file(target,data,stat.S_IMODE((source/name).stat().st_mode))
    packages=set(discover_packages(runtime));packages.add(source)
    prefixes=sorted((str(p) for p in packages),key=len,reverse=True)
    def remap(value):
        for before in prefixes:
            if value==before or value.startswith(before+os.sep):return str(destination)+value[len(before):]
        return value
    adjusted=dict(launch);adjusted['package']=str(destination)
    adjusted['worker_command']=[remap(value) for value in launch.get('worker_command',[])]
    if option(adjusted['worker_command'],'--package')!=str(destination):raise ValueError('standalone_worker_package_not_bound')
    old=(runtime/'launch.json').read_bytes();new=(json.dumps(adjusted,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()
    if old!=new:
        backup=runtime/'repair-backups/durable-switch-v1'/('launch.'+hashlib.sha256(old).hexdigest()+'.json')
        if not backup.exists():atomic_file(backup,old,0o600)
        atomic_file(runtime/'launch.json',new,stat.S_IMODE((runtime/'launch.json').stat().st_mode))
    emit('R1_STANDALONE_PACKAGE_READY',package=str(destination),old_packages=[str(p) for p in sorted(packages)],scientific_outputs_changed=0)
    return destination,packages,adjusted


def ensure_standalone_controller(runtime,package,old_packages, *, python=sys.executable,
        scanner=controller_processes,snapshot=_controller_process,kill=os.kill,held=lock_held,
        spawner=subprocess.Popen,sleep=time.sleep):
    runtime=Path(runtime).resolve();root=runtime.parents[1];all_packages=set(old_packages)|{Path(package)}
    rows=scanner(root,all_packages);old=[row for row in rows if str(Path(package)/'START.py') not in row['argv']]
    if held(runtime/'controller.lock') and not rows:
        emit('DURABLE_SWITCH_PENDING',reason='controller_lock_is_not_owned_by_a_verified_local_process',allocations_retained=True)
        return False
    for row in old:
        current=snapshot(row['pid'])
        if not current or current!=row or not _controller_matches(current,root,old_packages):
            emit('DURABLE_SWITCH_PENDING',reason='controller_identity_changed',allocations_retained=True);return False
        kill(row['pid'],__import__('signal').SIGTERM)
        emit('R1_OLD_CONTROLLER_RETIRED',pid=row['pid'],signal='SIGTERM',training_workers_signalled=False)
    for _ in range(50):
        if not held(runtime/'controller.lock'):break
        current=scanner(root,all_packages)
        if current and all(str(Path(package)/'START.py') in row['argv'] for row in current):
            emit('R1_STANDALONE_CONTROLLER_REUSED',package=str(package));return True
        sleep(.2)
    else:
        emit('DURABLE_SWITCH_PENDING',reason='previous_controller_lock_not_released',allocations_retained=True);return False
    with (runtime/'controller.log').open('ab',buffering=0) as log:
        child=spawner([python,'-I','-S','-B','-u',str(Path(package)/'START.py'),'--controller','--root',str(root)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
    for _ in range(50):
        if child.poll() is not None:break
        if held(runtime/'controller.lock'):
            observed=snapshot(child.pid)
            if observed and _controller_matches(observed,root,{Path(package)}):
                emit('R1_STANDALONE_CONTROLLER_STARTED',pid=child.pid,package=str(package));return True
        sleep(.2)
    emit('DURABLE_SWITCH_PENDING',reason='standalone_controller_start_not_confirmed',allocations_retained=True)
    return False


def replace_old_jobs(runtime,package,old_packages, *, runner=subprocess.run):
    runtime=Path(runtime).resolve();state=read(runtime/'scheduler-state.json',{}) or {};marker=state.get('marker','')
    if not re.fullmatch(r'paper3r1p-[0-9a-f]{16}',str(marker)):raise ValueError('Existing first-round scheduler marker missing')
    old_ids={}
    for slot in state.get('slots',{}).values():
        for attempt in slot.get('attempts',[]):
            job=str(attempt.get('job_id',''));command=attempt.get('command',[])
            selected=option(command,'--package') if isinstance(command,list) else None
            if job.isdigit() and selected and Path(selected).resolve()!=Path(package).resolve() and Path(selected).resolve() in {Path(p).resolve() for p in old_packages}:
                old_ids[job]=attempt
    try:response=runner(['squeue','--noheader','--user',getpass.getuser(),'--format=%i|%k|%T|%j'],capture_output=True,text=True,timeout=20)
    except (OSError,subprocess.TimeoutExpired) as error:
        emit('DURABLE_SWITCH_PENDING',reason=type(error).__name__,allocations_retained=True);return []
    if response.returncode:
        emit('DURABLE_SWITCH_PENDING',reason='queue_read_failed',allocations_retained=True);return []
    requested=[]
    for line in response.stdout.splitlines():
        row=[p.strip() for p in line.split('|')]
        if (len(row)<4 or row[0] not in old_ids or row[1]!=marker or
                not re.fullmatch(re.escape(marker)+r'-s[012]-a[1-9][0-9]*',row[3]) or old_ids[row[0]].get('name')!=row[3]):continue
        if row[2]=='PENDING':command=['scancel',row[0]]
        elif row[2]=='RUNNING':command=['scancel','--signal=USR1','--batch',row[0]]
        else:continue
        try:result=runner(command,capture_output=True,text=True,timeout=20)
        except (OSError,subprocess.TimeoutExpired) as error:
            emit('R1_OLD_WORKER_SWITCH_WAIT',job_id=row[0],reason=type(error).__name__);continue
        emit('R1_OLD_WORKER_SWITCH_REQUESTED',job_id=row[0],old_state=row[2],returncode=result.returncode,
            action='cancel_old_pending' if row[2]=='PENDING' else 'checkpoint_and_exit_old_running',new_package=str(package),force_kill=False)
        if result.returncode==0:requested.append(row[0])
    return requested


def checkpoint_restart(runtime, *, runner=subprocess.run, python=sys.executable, standalone_base=None):
    """Move R1 off the old combined install, then use its scheduler to replace old attempts."""
    runtime=Path(runtime).resolve();state=read(runtime/'scheduler-state.json',{}) or {}
    if not re.fullmatch(r'paper3r1p-[0-9a-f]{16}',str(state.get('marker',''))):raise ValueError('Existing first-round scheduler marker missing')
    package,old_packages,launch=standalone_launch(runtime,base=standalone_base)
    if not ensure_standalone_controller(runtime,package,old_packages,python=python):
        atomic_file(runtime/'durable-switch-status.json',json.dumps(dict(status='DURABLE_SWITCH_PENDING',package=str(package),old_jobs_preserved=True)).encode(),0o600)
        return []
    requested=replace_old_jobs(runtime,package,old_packages,runner=runner)
    atomic_file(runtime/'durable-switch-status.json',json.dumps(dict(status='DURABLE_CONTROLLER_READY',package=str(package),replacement_requests=requested,scientific_files_changed=False)).encode(),0o600)
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
    emit('R1_FIX_APPLIED',switch=read(runtime/'durable-switch-status.json',{}) if args.restart_workers else None,
         checkpoint_files_preserved=True,gpu_training_resumed_claimed=False)
    return 0


if __name__=='__main__':raise SystemExit(main())
