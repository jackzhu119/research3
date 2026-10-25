#!/usr/bin/env python3
"""Install the owned-child correction and checkpoint-safe recovery command."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import zlib

PAYLOAD = '__PAYLOAD__'
PAYLOAD_SHA256 = '__PAYLOAD_SHA256__'


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):return hashlib.sha256(data).hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temp=path.with_name('.'+path.name+'.future-fix-'+str(os.getpid()))
    try:
        with temp.open('wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.chmod(temp,mode);os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def put(path,value):atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def apply(root,home):
    raw=zlib.decompress(base64.b85decode(PAYLOAD))
    if sha(raw)!=PAYLOAD_SHA256:raise ValueError('Embedded future repair differs')
    files={name:base64.b64decode(value) for name,value in json.loads(raw).items()}
    folder=home/'paper3-runtime-fixes'/'future-path-v2'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name,data in files.items():
            if name not in ('r1_spawn_bootstrap.py','resume_existing.py'):
                raise ValueError('Unexpected embedded operational file')
            compile(data,name,'exec')
            target=folder/name
            if not target.exists() or target.read_bytes()!=data:atomic(target,data)
        helper=runpy.run_path(str(folder/'r1_spawn_bootstrap.py'),run_name='_paper3_future_bootstrap')
        launch_path=root/'runs/training_campaign_parallel_v1/launch.json'
        r1={'status':'NO_EXISTING_R1_PARALLEL_LAUNCH','scientific_files_changed':0}
        if launch_path.is_file():
            launch_raw=launch_path.read_bytes();launch=json.loads(launch_raw)
            package=Path(launch['package']).expanduser().resolve()
            direct=Path(launch['direct_package']).expanduser().resolve()
            target=direct/'campaign_recovery.py'
            original=target.read_bytes()
            fixed,changed=helper['corrected_recovery_source'](original.decode())
            fixed=fixed.encode();compile(fixed,str(target),'exec')
            # Both directories occur in the inherited owned-child sys.path.
            for destination in {package,direct}:
                path=destination/'r1_spawn_bootstrap.py'
                if not path.exists() or path.read_bytes()!=files['r1_spawn_bootstrap.py']:
                    atomic(path,files['r1_spawn_bootstrap.py'])
            backup=folder/'backups'/('campaign_recovery.'+sha(original)+'.py')
            if not backup.exists():atomic(backup,original)
            if original!=fixed:atomic(target,fixed)
            if launch_path.read_bytes()!=launch_raw:
                raise RuntimeError('R1 launch changed concurrently; scientific files were not written')
            r1={'status':'OWNED_CHILD_BOOTSTRAP_INSTALLED','package':str(package),'direct_package':str(direct),
                'path':str(target),'before_sha256':sha(original),'after_sha256':sha(fixed),
                'changed_entrypoints':changed,'scientific_files_changed':0,
                'launch_preserved':True,'bootstrap_sha256':sha(files['r1_spawn_bootstrap.py'])}
            emit('R1_FUTURE_CHILD_FIX_APPLIED',**r1)
        record={'schema_version':'paper3-future-path-repair-v2',
                'payload_sha256':PAYLOAD_SHA256,'r1':r1,
                'resume_helper':str(folder/'resume_existing.py'),
                'recovery_rounds':[5,6,8,9],
                'scientific_files_changed':0,'plans_changed':0,'checkpoints_changed':0,
                'slurm_jobs_cancelled':0,'slurm_jobs_submitted':0,
                'existing_r45689_runtime_corrections_replaced':False,
                'note':'Default installation does not restart any current job, controller or uploader. Owned children load corrected module on their next start.'}
        put(folder/'APPLIED.json',record)
        runtime=root/'runs/training_campaign_parallel_v1'
        if runtime.is_dir():put(runtime/'future-child-fix.json',record)
    emit('FUTURE_PATH_REPAIR_INSTALLED',record=str(folder/'APPLIED.json'),
         owned_child_bootstrap=r1['status'],resume_helper=str(folder/'resume_existing.py'),
         checkpoints_changed=0,jobs_cancelled=0,jobs_submitted=0)
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--resume-round',type=int,choices=(5,6,8,9),action='append',default=[])
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    emit('FUTURE_PATH_REPAIR_STARTED',imports_torch=False,data_download=False,full_test=False)
    folder=apply(root,home)
    if args.resume_round:
        # Explicit recovery calls the original installed controllers with their
        # saved plans. Default installation never reaches this branch.
        helper=runpy.run_path(str(folder/'resume_existing.py'),run_name='_paper3_saved_plan_resume')
        for number in dict.fromkeys(args.resume_round):
            helper['main'](['--root',str(root),'--round',str(number),'--execute'])
    return 0


if __name__=='__main__':raise SystemExit(main())
