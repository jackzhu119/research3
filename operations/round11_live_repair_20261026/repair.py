#!/usr/bin/env python3
"""Detached operational sidecars; never change an R11 scientific package/plan."""
from __future__ import annotations
import argparse
import fcntl
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
from controller_fix import bounded_read,controller_loop,load_original,validate_operational_history
DEFAULT_PLAN=Path('/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json')


def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),ensure_ascii=False),flush=True)


def lock_is_live(path):
    path=Path(path)
    if not path.is_file():return False
    with path.open('r') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(handle,fcntl.LOCK_UN)
    return False


def process_identity(pid,*,proc_root='/proc'):
    try:
        pid=int(pid)
        if pid<=1:return None
        base=Path(proc_root)/str(pid)
        if base.stat().st_uid!=os.getuid():return None
        args=(base/'cmdline').read_bytes().split(b'\0');args=[x.decode('utf-8') for x in args if x]
        stat=(base/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        return {'pid':pid,'uid':base.stat().st_uid,'start_ticks':fields[19],'args':args}
    except (OSError,ValueError,IndexError,UnicodeError):return None


def _argument(args,name):
    try:return args[args.index(name)+1]
    except (ValueError,IndexError):return None


def _owned_operation_identity(pid,env,role,*,proc_root='/proc'):
    identity=process_identity(pid,proc_root=proc_root)
    if identity is None:return None
    args=identity['args'];plan_arg=_argument(args,'--plan')
    if args.count('--plan')!=1:return None
    if not plan_arg:return None
    source=Path(plan_arg).expanduser().resolve()
    launch=bounded_read(env['runtime']/'launch.json',{})
    allowed={env['path'].resolve()}
    if launch.get('plan_sha256')==env['header']['plan_sha256'] and launch.get('campaign_identity')==env['header']['recipe_sha256']:
        allowed.add(Path(launch.get('plan','')).expanduser().resolve())
    if source not in allowed or not source.is_file() or env['common'].sha(source)!=env['header']['plan_sha256']:return None
    interpreters={Path(sys.executable).resolve(),Path(env['header'].get('python',sys.executable)).expanduser().resolve()}
    if launch.get('python'):interpreters.add(Path(launch['python']).expanduser().resolve())
    if not args or not Path(args[0]).is_absolute() or Path(args[0]).resolve() not in interpreters:return None
    # Match the actual Python script position, never arbitrary argv membership.
    position=1
    while position<len(args) and args[position] in ('-I','-S','-B','-u','-E','-s','-O','-OO','-q'):
        position+=1
    if position+1>=len(args):return None
    script=Path(args[position]).expanduser().resolve();mode=args[position+1]
    original_entry=env['package']/'paper3r11/platform/entry.py'
    own_entry=HERE/'repair.py'
    original=(role=='paper' and script==original_entry.resolve() and mode=='upload'
              and args.count('--channel')==1 and _argument(args,'--channel')=='paper')
    sidecar=(script==own_entry.resolve() and mode==role)
    if not (original or sidecar):return None
    return identity


def owned_paper_identity(pid,env,*,proc_root='/proc'):
    return _owned_operation_identity(pid,env,'paper',proc_root=proc_root)


def owned_controller_identity(pid,env,*,proc_root='/proc'):
    return _owned_operation_identity(pid,env,'controller',proc_root=proc_root)


def stop_owned_paper(pid,env,*,signal_fn=os.kill,identity_fn=owned_paper_identity):
    first=identity_fn(pid,env)
    if first is None:return {'status':'NOT_AN_OWNED_PAPER_PROCESS','signal_sent':False}
    second=identity_fn(pid,env)
    if second is None or any(first[k]!=second[k] for k in ('pid','uid','start_ticks','args')):
        return {'status':'PROCESS_IDENTITY_CHANGED','signal_sent':False}
    try:signal_fn(int(pid),signal.SIGTERM)
    except ProcessLookupError:return {'status':'OWNED_PAPER_ALREADY_EXITED','signal_sent':False}
    return {'status':'OWNED_PAPER_SIGTERM_SENT','signal_sent':True,'pid':int(pid),'start_ticks':first['start_ticks']}


def sidecar_ready(kind,identity,env):
    path=_argument(identity['args'],'--work');token=_argument(identity['args'],'--launch-token')
    if not path or not token:return False
    value=bounded_read(Path(path)/(kind+'-ready.json'),{})
    flags=('memory_patch_loaded','first_slurm_observation_completed') if kind=='controller' else ('memory_patch_loaded','paper_patch_loaded','original_paper_lock_held')
    return (value.get('pid')==identity['pid'] and value.get('recipe_sha256')==env['header']['recipe_sha256']
            and value.get('launch_token')==token and all(value.get(flag) is True for flag in flags))


def check(args):
    env=load_original(args.package,args.plan);history=validate_operational_history(env)
    return dict(status='READ_ONLY_REPAIR_CHECK_PASSED',**history,plan=str(env['path']),package=str(env['package']),
        recipe_sha256=env['header']['recipe_sha256'],total_tasks=env['header']['total_tasks'],
        controller_lock_live=lock_is_live(env['runtime']/'controller.lock'),
        paper_lock_live=lock_is_live(env['runtime']/'paper-upload.lock'),
        existing_controller_error=bounded_read(env['runtime']/'controller-error.json',{}),
        existing_paper_status=bounded_read(env['runtime']/'upload-paper/status.json',{}),
        scientific_sources_changed=False,slurm_calls=0,gpu_used=False)


def start_repair(args,*,popen=subprocess.Popen,sleep=time.sleep):
    env=load_original(args.package,args.plan);history=validate_operational_history(env)
    common=env['common'];work=Path(args.work or Path.home()/'paper3-r11-operational-repair/work'/env['header']['recipe_sha256'][:16]).expanduser().resolve()
    work.mkdir(parents=True,exist_ok=True);results=[]
    python=str(Path(args.python or sys.executable).expanduser().absolute())
    with common.locked(work/'launch.lock'):
        proof=dict(schema_version='paper3-r11-operational-repair-input-v1',plan_sha256=env['header']['plan_sha256'],
            recipe_sha256=env['header']['recipe_sha256'],original_package=str(env['package']),
            original_runtime=str(env['runtime']),history=history,scientific_files_modified=False,
            submit_reaffirmed=bool(args.submit),updated_utc=common.utc())
        common.atomic(work/'input-proof.json',proof)
        def spawn(kind,wait_paper=False):
            launch_token=common.digest([os.getpid(),time.time_ns(),kind,env['header']['recipe_sha256']])
            command=[python,'-I','-S','-B','-u',str(HERE/'repair.py'),kind,'--plan',str(env['path']),
                '--package',str(env['package']),'--work',str(work),'--launch-token',launch_token]
            if kind=='controller' and args.submit:command+=['--submit']
            if kind=='paper' and wait_paper:command+=['--wait-paper-lock']
            if kind=='paper' and args.credentials:command+=['--credentials',str(Path(args.credentials).expanduser())]
            with (work/(kind+'.log')).open('ab',buffering=0) as log:
                child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
            receipt=dict(kind=kind,pid=child.pid,launch_token=launch_token,started_utc=common.utc(),command_scope='operational_sidecar_only',log=str(work/(kind+'.log')))
            deadline=time.monotonic()+5;ready_path=work/(kind+'-ready.json')
            while True:
                ready=bounded_read(ready_path,{})
                code=child.poll() if hasattr(child,'poll') else None
                if code is not None and code!=0:
                    receipt.update(status='STARTUP_FAILED',exit_code=code);break
                flags=('memory_patch_loaded','first_slurm_observation_completed') if kind=='controller' else ('memory_patch_loaded','paper_patch_loaded','original_paper_lock_held')
                if (ready.get('pid')==child.pid and ready.get('recipe_sha256')==env['header']['recipe_sha256']
                        and ready.get('launch_token')==launch_token and all(ready.get(flag) is True for flag in flags)):
                    receipt.update(status='READY',ready=ready);break
                if code is not None:
                    receipt.update(status='STARTUP_FAILED',exit_code=code);break
                if time.monotonic()>=deadline:
                    receipt.update(status='STARTUP_PENDING');break
                sleep(.05)
            common.atomic(work/(kind+'-process.json'),receipt);return receipt
        if lock_is_live(env['runtime']/'controller.lock'):
            candidates=[bounded_read(work/'controller-process.json',{}).get('pid'),
                bounded_read(env['runtime']/'controller-process.json',{}).get('pid')]
            owned=next((owned_controller_identity(pid,env) for pid in candidates if pid and owned_controller_identity(pid,env)),None)
            status=('EXISTING_REPAIRED_CONTROLLER_REUSED' if sidecar_ready('controller',owned,env) else 'EXISTING_REPAIRED_CONTROLLER_STARTUP_PENDING') if owned else 'EXISTING_CONTROLLER_LEFT_RUNNING'
            results.append(dict(kind='controller',status=status,pid=owned['pid'] if owned else None))
        elif not args.submit:
            results.append(dict(kind='controller',status='REQUIRES_EXPLICIT_SUBMIT_REAFFIRMATION'))
        else:
            results.append(spawn('controller'))
        # Never stop or restart full/report/monitor, scientific Python children,
        # Slurm workers, or another user's process.
        paper_lock=env['runtime']/'paper-upload.lock'
        if lock_is_live(paper_lock):
            candidates=[bounded_read(work/'paper-process.json',{}).get('pid'),
                bounded_read(env['runtime']/'upload-paper/process.json',{}).get('pid'),
                bounded_read(env['runtime']/'paper-upload-process.json',{}).get('pid')]
            stopped=None
            for pid in dict.fromkeys(candidates):
                if pid and owned_paper_identity(pid,env):
                    # An already repaired paper sidecar is simply reused.
                    identity=owned_paper_identity(pid,env)
                    if str(HERE/'repair.py') in identity['args']:
                        stopped={'status':'EXISTING_REPAIRED_PAPER_REUSED' if sidecar_ready('paper',identity,env) else 'EXISTING_REPAIRED_PAPER_STARTUP_PENDING','pid':pid};break
                    stopped=stop_owned_paper(pid,env);break
            if stopped is None:results.append(dict(kind='paper',status='LIVE_PROCESS_OWNERSHIP_UNCONFIRMED_NO_SIGNAL'))
            elif stopped['status'].startswith('EXISTING_REPAIRED_PAPER_'):results.append(dict(kind='paper',**stopped))
            elif not stopped.get('signal_sent'):
                if not lock_is_live(paper_lock):results.append(spawn('paper'))
                else:results.append(dict(kind='paper',**stopped))
            else:
                results.append(spawn('paper',wait_paper=lock_is_live(paper_lock)))
        else:results.append(spawn('paper'))
        statuses={r.get('status') for r in results}
        partial={'LIVE_PROCESS_OWNERSHIP_UNCONFIRMED_NO_SIGNAL','REQUIRES_EXPLICIT_SUBMIT_REAFFIRMATION','EXISTING_CONTROLLER_LEFT_RUNNING','NOT_AN_OWNED_PAPER_PROCESS','PROCESS_IDENTITY_CHANGED'}
        overall=('STARTUP_FAILED' if 'STARTUP_FAILED' in statuses else 'STARTUP_PENDING' if any(str(s).endswith('STARTUP_PENDING') for s in statuses)
                 else 'PARTIAL_REPAIR' if statuses&partial else 'OPERATIONAL_REPAIR_READY')
        result=dict(status=overall,work=str(work),processes=results,
            terminal_can_close=True,scientific_sources_changed=False,scientific_plan_changed=False,
            scientific_workers_signalled=0,slurm_jobs_cancelled=0,existing_four_slots_preserved=True,
            allocation_replacement_requires_explicit_submit=True,submit_reaffirmed=bool(args.submit))
        common.atomic(work/'STARTED.json',result)
    return result


def paper_loop(args,stop):
    """Reuse the original secure loop, with a real lock-acquired readiness hook."""
    env=load_original(args.package,args.plan);common=env['common']
    import importlib
    uploads=importlib.import_module('paper3r11.platform.uploads')
    from paper_fix import install_patch
    work=Path(args.work).expanduser().resolve()
    def ready(payload):
        if (payload.get('pid')!=os.getpid() or payload.get('recipe_sha256')!=env['header']['recipe_sha256']
                or payload.get('role')!='paper' or payload.get('lock_acquired') is not True):
            raise ValueError('PAPER_READY_OWNERSHIP_DIFFERS')
        common.atomic(work/'paper-ready.json',dict(status='READY',pid=os.getpid(),
            recipe_sha256=env['header']['recipe_sha256'],memory_patch_loaded=True,paper_patch_loaded=True,
            launch_token=args.launch_token,original_paper_lock_held=True,upload_progress_claimed=False,
            patch_id=payload.get('patch_id'),updated_utc=common.utc()))
    patch=install_patch(uploads,ready_callback=ready);common.atomic(work/'paper-patch.json',patch)
    if args.wait_paper_lock:
        deadline=time.monotonic()+300
        while lock_is_live(env['runtime']/'paper-upload.lock'):
            if stop.wait(.5):return {'status':'STOPPED_BEFORE_PAPER_LOCK'}
            if time.monotonic()>=deadline:raise RuntimeError('OWNED_PAPER_LOCK_RELEASE_TIMEOUT')
    return uploads.channel_loop(env['path'],'paper',credentials=args.credentials,stop=stop)


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('check','start','controller','paper'))
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN);p.add_argument('--package',type=Path)
    p.add_argument('--work',type=Path);p.add_argument('--python');p.add_argument('--credentials',type=Path)
    p.add_argument('--launch-token')
    p.add_argument('--wait-paper-lock',action='store_true')
    p.add_argument('--submit',action='store_true');return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=='check':result=check(args)
    elif args.mode=='start':result=start_repair(args)
    else:
        work=Path(args.work).expanduser().resolve();work.mkdir(parents=True,exist_ok=True);stop=threading.Event()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
        if args.mode=='controller':
            try:result=controller_loop(args.plan,work,package=args.package,submit=args.submit,stop=stop,launch_token=args.launch_token)
            except BlockingIOError:result={'status':'EXISTING_CONTROLLER_LEFT_RUNNING'}
        else:
            result=paper_loop(args,stop)
    emit('R11_OPERATIONAL_REPAIR_RESULT',**result);return 2 if result.get('status') in ('STARTUP_FAILED','PARTIAL_REPAIR') else 0


if __name__=='__main__':raise SystemExit(main())
