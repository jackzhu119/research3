#!/usr/bin/env python3
"""Stop R1 automatic child refresh; replace only its owned CPU controller."""
from __future__ import annotations
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

HOTFIX_ID='paper3-r1-refresh-disabled-v1'
OLD_PACKAGE='2bfd664251ee5d1bcaf5'
OLD_MANIFEST_SHA256='f30f3e60102b7bfaa54d758a69427f7e9fc27d7330ab22c59d9a458f08c9fbc4'
OLD_SUPERVISOR_SHA256='b09934e9f6246492388ba57159e1f2af41749c5a857dc9410f3252d3995bbfc9'
SUPERVISOR_B85='__SUPERVISOR_B85__'
SUPERVISOR_SHA256='__SUPERVISOR_SHA256__'
DEPENDENCIES=('parallel_runtime.py','install_hooks.py','atomic_io.py','newnode_tool.py')
RUNTIME='runs/training_campaign_parallel_v1'


def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()


def read(path):
    with Path(path).open('rb') as handle:
        size=os.fstat(handle.fileno()).st_size
        if not 0 <= size <= 1024*1024:raise ValueError('HOTFIX_RECORD_TOO_LARGE')
        data=handle.read(size+1)
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('HOTFIX_RECORD_NOT_OBJECT')
    return value


def optional(path):
    try:return read(path)
    except FileNotFoundError:return {}


def regular(path):
    value=Path(path).lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid!=os.getuid():
        raise ValueError('HOTFIX_FILE_NOT_OWNED_REGULAR')


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():regular(path)
    fd,temp=tempfile.mkstemp(prefix='.refresh-hotfix-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:Path(temp).unlink(missing_ok=True)


def option(args,name):
    if args.count(name)!=1:return None
    index=args.index(name)
    return args[index+1] if index+1<len(args) else None


def identity(pid,proc_root=Path('/proc')):
    try:
        if type(pid) is not int or pid<=1:return None
        folder=Path(proc_root)/str(pid)
        if folder.stat().st_uid!=os.getuid():return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return None
        argv=[v.decode('utf-8') for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid,uid=folder.stat().st_uid,start_ticks=fields[19],argv=argv,ppid=int(fields[1]))
    except (OSError,ValueError,IndexError,UnicodeError):return None


def checked_manager(record,root,home,work,old,overlay,*,process=identity):
    pid=record.get('pid');value=process(pid)
    if value is None:raise ValueError('R1_MANAGEMENT_PROCESS_NOT_FOUND')
    args=value['argv'];at=1
    if not args or Path(args[0]).resolve()!=Path(sys.executable).resolve():
        raise ValueError('R1_MANAGEMENT_PYTHON_DIFFERS')
    while at<len(args) and args[at] in ('-I','-S','-B','-u','-E','-s','-O','-OO'):at+=1
    allowed={old/'supervisor.py',overlay/'supervisor.py'}
    if (at+1>=len(args) or Path(args[at]).resolve() not in allowed or args[at+1]!='manage'
            or '--child' in args or '--controller' in args
            or option(args,'--root')!=str(root) or option(args,'--home')!=str(home)
            or option(args,'--work')!=str(work) or option(args,'--package')!=str(old)
            or option(args,'--launch-token')!=record.get('launch_token')
            or not record.get('launch_token') or record.get('root')!=str(root)):
        raise ValueError('R1_MANAGEMENT_IDENTITY_DIFFERS')
    if record.get('start_ticks') is not None and record['start_ticks']!=value['start_ticks']:
        raise ValueError('R1_MANAGEMENT_PID_REUSED')
    return value


def verified_files(old):
    regular(old/'MANIFEST.json');raw=(old/'MANIFEST.json').read_bytes()
    if sha(raw)!=OLD_MANIFEST_SHA256:raise ValueError('ORIGINAL_REPAIR_MANIFEST_DIFFERS')
    manifest=json.loads(raw);files={}
    for name in ('supervisor.py',*DEPENDENCIES):
        path=old/name;regular(path);data=path.read_bytes();record=manifest['files'][name]
        if len(data)!=record['bytes'] or sha(data)!=record['sha256']:
            raise ValueError('ORIGINAL_REPAIR_FILE_DIFFERS: '+name)
        files[name]=data
    if sha(files['supervisor.py'])!=OLD_SUPERVISOR_SHA256:
        raise ValueError('ORIGINAL_SUPERVISOR_SHA256_DIFFERS')
    replacement=base64.b85decode(SUPERVISOR_B85)
    if sha(replacement)!=SUPERVISOR_SHA256:raise ValueError('FIXED_SUPERVISOR_SHA256_DIFFERS')
    compile(replacement,'supervisor.py','exec');files['supervisor.py']=replacement
    return files


def overlay_install(overlay,files):
    if overlay.is_symlink():raise ValueError('HOTFIX_OVERLAY_IS_SYMLINK')
    overlay.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name,data in files.items():
        path=overlay/name
        if path.exists() or path.is_symlink():
            regular(path)
            if path.read_bytes()!=data:raise ValueError('EXISTING_HOTFIX_OVERLAY_DIFFERS')
        else:atomic(path,data)


def ready_for(work,pid,token):
    value=optional(work/'controller-ready.json')
    return (value.get('pid')==pid and value.get('launch_token')==token
        and value.get('first_slurm_observation_completed') is True
        and value.get('queue_observation_successful') is True)


def manager_or_stopped(record,root,home,work,old,overlay,*,process):
    """Recover only a management start/stop durably recorded by this hotfix."""
    if process(record.get('pid')) is not None:
        return checked_manager(record,root,home,work,old,overlay,process=process),False
    before=optional(work/'refresh-hotfix-before.json')
    own=optional(work/'refresh-hotfix-applied.json')
    if (before.get('hotfix_id')==HOTFIX_ID
            and before.get('replacement_supervisor_sha256')==SUPERVISOR_SHA256
            and before.get('old_supervisor_sha256')==OLD_SUPERVISOR_SHA256
            and before.get('record')==record and isinstance(before.get('identity'),dict)):
        prior=before['identity']
    elif (record.get('hotfix_id')==HOTFIX_ID and record.get('supervisor_sha256')==SUPERVISOR_SHA256
            and own.get('hotfix_id')==HOTFIX_ID and own.get('supervisor_sha256')==SUPERVISOR_SHA256
            and own.get('manager_pid')==record.get('pid')
            and own.get('launch_token')==record.get('launch_token')
            and own.get('start_ticks')==record.get('start_ticks')
            and isinstance(record.get('argv'),list)):
        prior=dict(pid=record['pid'],uid=os.getuid(),start_ticks=record.get('start_ticks'),argv=record['argv'])
    else:raise ValueError('R1_MANAGEMENT_PROCESS_NOT_FOUND_NO_RECORDED_HOTFIX_STOP')
    checked_manager(record,root,home,work,old,overlay,process=lambda pid:prior)
    return prior,True


def refresh_steps(manager,root,old,*,process=identity,proc_root=Path('/proc'),orphans=False):
    """Identify only this R1 CPU refresh srun, never a fit or an allocation."""
    runtime=root/RUNTIME
    state=optional(runtime/'scheduler-state.json');jobs={}
    for slot in range(3):
        attempts=state.get('slots',{}).get(str(slot),{}).get('attempts',[])
        if attempts and str(attempts[-1].get('job_id','')).isdigit():
            jobs[str(slot)]=str(attempts[-1]['job_id'])
    found=[]
    for path in Path(proc_root).iterdir():
        if not path.name.isdigit():continue
        value=process(int(path.name))
        if value is None or value.get('uid')!=os.getuid():continue
        if not orphans and value.get('ppid')!=manager['pid']:continue
        try:
            if int(value['start_ticks'])<int(manager['start_ticks']):continue
        except (ValueError,TypeError,KeyError):continue
        args=value['argv']
        if not args or Path(args[0]).name!='srun':continue
        for slot,job in jobs.items():
            expected=['--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1',
                '--mem=256M','--gres=none','--time=00:04:00',sys.executable,'-I','-S','-B',
                str(old/'newnode_tool.py'),'refresh','--root',str(root),'--runtime',str(runtime),
                '--job-id',job,'--slot',slot]
            if args[1:]==expected:found.append(value);break
    return found


def same_step(current,previous):
    return current is not None and all(current.get(k)==previous.get(k)
        for k in ('pid','uid','start_ticks','argv'))


def apply(home,root,*,activate=False,process=identity,send=os.kill,popen=subprocess.Popen,
          clock=time.monotonic,sleep=time.sleep,random_bytes=os.urandom,scan_steps=refresh_steps):
    home=Path(home).expanduser().resolve();root=Path(root).expanduser().resolve()
    base=home/'paper3-r1-full-repair';old=base/OLD_PACKAGE;work=base/'work'
    overlay=base/('refresh-fixed-'+SUPERVISOR_SHA256[:20])
    if base.is_symlink() or old.is_symlink() or work.is_symlink():
        raise ValueError('HOTFIX_BASE_PATH_IS_SYMLINK')
    files=verified_files(old)
    regular(work/'controller-process.json');record=read(work/'controller-process.json')
    current,stopped=manager_or_stopped(record,root,home,work,old,overlay,process=process)
    result=dict(hotfix_id=HOTFIX_ID,old_package=str(old),overlay=str(overlay),work=str(work),
        manager_pid=current['pid'],scientific_children_signalled=0,allocations_cancelled=0,
        completed_results_reset=0,scientific_files_changed=0,old_package_files_changed=0,
        automatic_child_refresh=False)
    if not activate:return dict(result,status='READ_ONLY_HOTFIX_READY')
    with (work/'refresh-hotfix.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Re-read after locking: no stale PID or overlapping installer may act.
        record=read(work/'controller-process.json')
        current,stopped=manager_or_stopped(record,root,home,work,old,overlay,process=process)
        own=optional(work/'refresh-hotfix-applied.json')
        pending_identity=(own.get('start_ticks') is None and record.get('start_ticks') is None)
        if (not stopped and own.get('hotfix_id')==HOTFIX_ID and own.get('supervisor_sha256')==SUPERVISOR_SHA256
                and own.get('manager_pid')==current['pid']
                and own.get('launch_token')==record.get('launch_token')
                and (own.get('start_ticks')==current['start_ticks'] or pending_identity)
                and str(overlay/'supervisor.py') in current['argv']):
            if pending_identity:
                record['start_ticks']=current['start_ticks'];own['start_ticks']=current['start_ticks']
                atomic(work/'controller-process.json',encoded(record))
                atomic(work/'refresh-hotfix-applied.json',encoded(own))
            return dict(result,manager_pid=current['pid'],status='R1_REFRESH_HOTFIX_ALREADY_RUNNING',
                controller_ready=ready_for(work,current['pid'],record['launch_token']))
        overlay_install(overlay,files)
        saved=optional(work/'refresh-hotfix-steps.json')
        captured=(saved.get('steps',[]) if saved.get('hotfix_id')==HOTFIX_ID
            and saved.get('manager_identity')==current else [])
        requested=set()
        def collect_steps(orphans=False):
            nonlocal captured
            rows=scan_steps(current,root,old,process=process,orphans=orphans)
            for row in rows:
                if not any(same_step(row,prior) for prior in captured):captured.append(row)
            atomic(work/'refresh-hotfix-steps.json',encoded(dict(hotfix_id=HOTFIX_ID,
                manager_identity=current,steps=captured,scientific_workers_signalled=0)))
        def stop_steps():
            for row in captured:
                signature=(row['pid'],row['start_ticks'])
                if signature in requested or not same_step(process(row['pid']),row):continue
                # The captured exact srun command launches only a metadata
                # management refresh helper on its own 256M, --gres=none step.
                try:send(row['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
                requested.add(signature)
        collect_steps();stop_steps()
        # Recheck identity immediately before the management controller signal.
        if not stopped:
            check=checked_manager(record,root,home,work,old,overlay,process=process)
            if check!=current:raise ValueError('R1_MANAGEMENT_CHANGED_BEFORE_SIGNAL')
            atomic(work/'refresh-hotfix-before.json',encoded(dict(record=record,identity=current,
                hotfix_id=HOTFIX_ID,old_supervisor_sha256=OLD_SUPERVISOR_SHA256,
                replacement_supervisor_sha256=SUPERVISOR_SHA256)))
            try:send(current['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            until=clock()+12
            while process(current['pid'])==current and clock()<until:
                collect_steps();stop_steps();sleep(.1)
            if process(current['pid'])==current:
                raise RuntimeError('R1_MANAGEMENT_TERM_PENDING_NO_WORKER_SIGNAL_SENT')
        # A launcher can be created immediately before its old parent exits.
        # Adopt it only through its exact old helper/root/current-job argv and
        # a start time at least as recent as the verified old manager.
        # Re-scan even when the first scan is empty: a launcher forked just
        # before shutdown can still be completing exec at the first read.
        # This short management-only settling interval leaves scientific
        # workers running and covers the ordinary fork/exec handover window.
        settle_until=clock()+1
        until=clock()+12
        while True:
            collect_steps(orphans=True);stop_steps()
            pending=any(same_step(process(row['pid']),row) for row in captured)
            if not pending and clock()>=settle_until:break
            if clock()>=until:
                raise RuntimeError('OLD_REFRESH_STEP_PENDING_NO_NEW_CONTROLLER_STARTED')
            sleep(.1)
        # PID reuse is not ours and must never receive another signal.
        token=hashlib.sha256(random_bytes(32)).hexdigest()
        command=[sys.executable,'-I','-S','-B','-u',str(overlay/'supervisor.py'),'manage',
            '--root',str(root),'--home',str(home),'--package',str(old),'--work',str(work),
            '--submit','--launch-token',token,'--python',sys.executable]
        with (work/'controller.log').open('ab',buffering=0) as log:
            child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                start_new_session=True,close_fds=True)
        fresh=dict(pid=child.pid,launch_token=token,root=str(root),started_utc=time.time(),
            argv=command,hotfix_id=HOTFIX_ID,supervisor_sha256=SUPERVISOR_SHA256)
        # Persist this exact child handle/nonce before /proc can lag its exec.
        # A failed or interrupted startup can therefore be recovered on rerun.
        atomic(work/'controller-process.json',encoded(fresh))
        receipt=dict(result,status='R1_REFRESH_HOTFIX_STARTUP_PENDING',manager_pid=child.pid,
            launch_token=token,start_ticks=None,supervisor_sha256=SUPERVISOR_SHA256,
            old_manager_pid=current['pid'],management_signals=['SIGTERM'],
            old_refresh_step_pids=[row['pid'] for row in captured],
            management_refresh_steps_signalled=len(requested),
            old_refresh_steps_remaining=0,
            updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        until=clock()+2;new=None
        while child.poll() is None:
            try:new=checked_manager(fresh,root,home,work,old,overlay,process=process)
            except ValueError:new=None
            if new is not None or clock()>=until:break
            sleep(.05)
        if new is None:
            receipt['status']=('R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED' if child.poll() is not None
                else 'R1_REFRESH_HOTFIX_STARTUP_PENDING_IDENTITY')
            atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
            return receipt
        fresh['start_ticks']=new['start_ticks'];receipt['start_ticks']=new['start_ticks']
        atomic(work/'controller-process.json',encoded(fresh))
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        until=clock()+8
        while child.poll() is None and not ready_for(work,child.pid,token) and clock()<until:sleep(.1)
        if child.poll() is not None:
            receipt['status']='R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED'
        elif ready_for(work,child.pid,token):receipt['status']='R1_REFRESH_HOTFIX_CONTROLLER_READY'
        atomic(work/'refresh-hotfix-applied.json',encoded(receipt))
        return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--home',type=Path,default=Path.home())
    p.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    p.add_argument('--apply',action='store_true');args=p.parse_args(argv)
    value=apply(args.home,args.root,activate=args.apply)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return 2 if value.get('status')=='R1_REFRESH_HOTFIX_CONTROLLER_START_FAILED' else 0


if __name__=='__main__':raise SystemExit(main())
