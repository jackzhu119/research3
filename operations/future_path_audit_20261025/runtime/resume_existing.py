#!/usr/bin/env python3
"""Resume existing Paper3 controllers without re-entering pinned installers.

The immutable numerical plan and saved launch are inputs, never rebuilt. This
helper neither downloads nor directly submits/cancels Slurm jobs. Its detached
controller process calls the original installed controller_loop with exactly
the saved launch configuration, so the existing scheduler owns continuation.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

OUTPUTS={5:'followup_20261024_r5',6:'followup_20261024_r6',
         8:'followup_20261025_r8',9:'followup_20261025_r9'}


def emit(event,**fields):
    print(json.dumps(dict(event=event,**fields)),flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def context(root,number,runtime=None):
    """Resolve only the existing plan, launch, and its declared operational code."""
    root=Path(root).expanduser().resolve()
    output=root/'runs'/OUTPUTS[number]
    runtime=Path(runtime).expanduser().resolve() if runtime else output/'runtime/three_jobs'
    launch_path=runtime/'launch.json'
    launch=read(launch_path)
    plan_path=Path(launch.get('plan',output/'plan.json')).resolve()
    raw=plan_path.read_bytes();plan=json.loads(raw)
    # This compares two small declared identities, not training arrays/source.
    if launch.get('original_plan_sha256') not in (None,hashlib.sha256(raw).hexdigest()):
        raise ValueError('Saved launch and original plan differ; neither was rewritten')
    if Path(plan['root']).resolve()!=root or Path(plan['output']).resolve()!=output:
        raise ValueError('Existing plan belongs to a different root/output')
    commands=launch.get('worker_command',[])
    operation=next((Path(v).resolve().parent for v in commands
                    if isinstance(v,str) and Path(v).name=='three_jobs.py'),None)
    if operation is None or not (operation/'coordinator.py').is_file():
        raise ValueError('Saved operational coordinator is unavailable')
    return {'round':number,'root':root,'output':output,'runtime':runtime,
            'plan_path':plan_path,'plan':plan,'launch':launch,
            'operation':operation,'plan_bytes':raw,'launch_bytes':launch_path.read_bytes()}


def lock_held(path):
    with Path(path).open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def receipt_alive(path,command):
    try:
        receipt=read(path);pid=int(receipt['pid']);folder=Path('/proc')/str(pid)
        if folder.stat().st_uid!=os.getuid():return False
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return False
        if int(fields[19])!=receipt['start_ticks']:return False
        if (Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=receipt['boot_id']):return False
        args=[v.decode(errors='replace') for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
        return all(value in args for value in command[1:] if value not in ('-I','-S','-B','-u'))
    except (OSError,ValueError,KeyError,TypeError,IndexError):return False


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def command_for(c,kind,credentials=None):
    launch,op=c['launch'],c['operation']
    command=[launch['python'],'-I','-S','-B','-u']
    if kind=='controller':
        command += [str(Path(__file__).resolve()),'--run-controller','--root',str(c['root']),
                    '--round',str(c['round']),'--runtime',str(c['runtime'])]
    elif kind=='monitor':
        command += [str(op/'monitor.py'),'--round',str(c['round']),
                    '--root',str(c['root']),'--operational',str(c['runtime'])]
    elif kind in ('paper','archive') and c['round']==9:
        command += [str(op/'result_channels.py'),'--channel',kind,'--root',str(c['root']),
                    '--output',str(c['output']),'--operational',str(c['runtime'])]
    else:raise ValueError('Unknown existing component')
    if credentials and kind!='controller':command += ['--credentials',str(credentials)]
    return command


def start(c,kind,*,credentials=None,popen=None):
    runtime=c['runtime'];command=command_for(c,kind,credentials)
    receipt_path=runtime/(kind+'-resume-process.json')
    if lock_held(runtime/(kind+'.lock')) or receipt_alive(receipt_path,command):
        emit('EXISTING_COMPONENT_REUSED',round=c['round'],kind=kind)
        return {'kind':kind,'reused':True}
    popen=popen or subprocess.Popen
    with (runtime/(kind+'.log')).open('ab',buffering=0) as handle:
        child=popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=handle,
                    start_new_session=True,close_fds=True)
    receipt={'pid':child.pid,'kind':kind,'round':c['round'],'command':command,'started_at':time.time()}
    try:
        receipt['start_ticks']=int((Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
        receipt['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (OSError,ValueError,IndexError):pass
    atomic_json(receipt_path,receipt)
    emit('EXISTING_COMPONENT_STARTED',round=c['round'],kind=kind,pid=child.pid,
         launch_preserved=True,plan_preserved=True)
    return receipt


def run_controller(c):
    """Call the old controller, avoiding prepare_context's launch reconstruction."""
    sys.path.insert(0,str(c['operation']))
    path=c['operation']/'coordinator.py'
    spec=importlib.util.spec_from_file_location('_paper3_existing_controller',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.set())
    return module.controller_loop({k:v for k,v in c.items()
        if k not in ('operation','plan_bytes','launch_bytes')},stop=stop)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--round',type=int,choices=OUTPUTS,required=True,dest='number')
    parser.add_argument('--runtime')
    parser.add_argument('--credentials')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-controller',action='store_true')
    args=parser.parse_args(argv)
    c=context(args.root,args.number,args.runtime)
    if args.run_controller:return run_controller(c)
    emit('EXISTING_PLAN_RESUME_READY',round=args.number,plan=str(c['plan_path']),
         partition=c['launch'].get('partition'),cpus=c['launch'].get('cpus'),
         memories=c['launch'].get('memories'),package=c['launch'].get('package'),
         installer_reentered=False,scientific_plan_rebuilt=False)
    if not args.execute:return 0
    with (c['runtime']/'resume-existing.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        stop=c['runtime']/'STOP'
        if stop.exists():
            history=c['runtime']/'startup-history';history.mkdir(exist_ok=True)
            os.replace(stop,history/('STOP-resumed-'+str(time.time_ns())))
        components=('controller','monitor','paper','archive') if args.number==9 else ('controller','monitor')
        for kind in components:start(c,kind,credentials=args.credentials)
    return 0


if __name__=='__main__':raise SystemExit(main())
