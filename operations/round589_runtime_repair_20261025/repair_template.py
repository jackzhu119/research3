#!/usr/bin/env python3
"""Apply the verified R5/R8/R9 startup corrections without replacing plans."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

HOOK_B85 = '__HOOK_B85__'
HOOK_SHA256 = '__HOOK_SHA256__'
MARKER = '# PAPER3_R589_RUNTIME_FIX_V1'
OUTPUTS = {5:'followup_20261024_r5', 8:'followup_20261025_r8', 9:'followup_20261025_r9'}


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):
        return {}


def atomic(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary=path.with_name('.'+path.name+'.repair-'+str(os.getpid()))
    try:
        with temporary.open('wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary,mode)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def wrappers(home, root):
    """Read only three declared launches and shallow, known installation roots."""
    found={}
    def add(path, round_number=None):
        path=Path(path).expanduser()
        if path.is_file():
            found.setdefault(path.resolve(),set())
            if round_number:found[path.resolve()].add(round_number)
    for number,folder in OUTPUTS.items():
        output=root/'runs'/folder
        for relative in ('runtime/three_jobs/launch.json','runtime/launch.json'):
            launch=read(output/relative)
            for value in launch.get('worker_command',[]):
                if isinstance(value,str) and Path(value).name=='three_jobs.py':
                    add(Path(value).parent/'backend_entry.py',number)
            for key in ('package','scientific_package','source_package'):
                if isinstance(launch.get(key),str):
                    add(Path(launch[key])/'runtime/backend_entry.py',number)
    for pattern in ('paper3-parallel-migrations/*/round56/backend_entry.py',
                    'paper3-round8/*/runtime/backend_entry.py',
                    'paper3-round8/runtime/backend_entry.py',
                    'paper3-round9/*/runtime/backend_entry.py',
                    'paper3-round9/runtime/backend_entry.py'):
        for path in home.glob(pattern):add(path)
    return found


def patched_backend(original, hook_path):
    text=original.decode('utf-8')
    if MARKER in text:
        start=text.index('    '+MARKER)
        end=text.index('    # END_PAPER3_R589_RUNTIME_FIX_V1',start)
        end=text.index('\n',end)+1
        text=text[:start]+text[end:]
    anchor='    plan=json.loads(plan_path.read_text())\n'
    if text.count(anchor)!=1 or 'runpy.run_path(str(entry)' not in text:
        raise ValueError('Unrecognized operational backend; no scientific files were changed')
    block=('    '+MARKER+'\n'
        '    if entry.parent.name in ("paper3r5", "paper3r8", "paper3r9"):\n'
        '        runpy.run_path('+repr(str(hook_path))+', run_name="_paper3_runtime_fix")["install"](entry.parent.name)\n'
        '    # END_PAPER3_R589_RUNTIME_FIX_V1\n')
    result=text.replace(anchor,anchor+block).encode()
    compile(result,'backend_entry.py','exec')
    return result


def inspect_round(root, number):
    output=root/'runs'/OUTPUTS[number]
    plan_path=output/'plan.json'
    if not plan_path.is_file():return {'round':number,'status':'NOT_INSTALLED'}
    data=plan_path.read_bytes();plan=json.loads(data)
    runtime=output/'runtime/three_jobs'
    progress=read(runtime/'progress.json')
    rows=[]
    for slot in range(3):
        current=read(runtime/('slot-%d.json'%slot))
        nested=current.get('worker_progress') or {}
        rows.append({'slot':slot,'job_id':current.get('job_id'),
            'status':current.get('status'),'task':current.get('task_key'),
            'worker_progress':{k:nested[k] for k in ('status','stage','outer_update','session','sessions_total','updated_utc') if k in nested}})
    return {'round':number,'status':'EXISTING_PLAN_PRESERVED','plan':str(plan_path),
            'plan_sha256':sha(data),'recipe_sha256':plan.get('recipe_sha256'),
            'total_tasks':len(plan.get('tasks',[])),
            'last_observed_complete_tasks':progress.get('complete_tasks'),
            'progress_updated_utc':progress.get('updated_utc'),'slots':rows}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    if args.status:
        for number in OUTPUTS:emit('ROUND_REPAIR_PROGRESS',**inspect_round(root,number))
        return 0
    emit('R589_REPAIR_STARTED',root=str(root),scope=[5,8,9],
         imports_torch=False,full_data_rescan=False)
    hook=zlib.decompress(base64.b85decode(HOOK_B85))
    if sha(hook)!=HOOK_SHA256:raise ValueError('Embedded repair payload differs')
    compile(hook,'runtime_hook.py','exec')
    folder=home/'paper3-runtime-fixes'/'r589-dtype-cublas-v1'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hook_path=folder/'runtime_hook.py'
        if not hook_path.exists() or hook_path.read_bytes()!=hook:atomic(hook_path,hook)
        candidates=wrappers(home,root)
        if not candidates:
            raise SystemExit('No installed R5/R8/R9 operational backend found; send this output for the exact existing path.')
        before={n:inspect_round(root,n) for n in OUTPUTS}
        changed=[]
        for path,numbers in sorted(candidates.items()):
            original=path.read_bytes()
            fixed=patched_backend(original,hook_path)
            backup=folder/'backups'/(sha(original)+'.py')
            if not backup.exists():atomic(backup,original)
            if original!=fixed:atomic(path,fixed)
            item={'path':str(path),'before_sha256':sha(original),'after_sha256':sha(fixed),
                  'changed':original!=fixed,'rounds_from_launch':sorted(numbers)}
            changed.append(item);emit('R589_BACKEND_PATCHED',**item)
        after={n:inspect_round(root,n) for n in OUTPUTS}
        for number in OUTPUTS:
            if before[number].get('plan_sha256')!=after[number].get('plan_sha256'):
                raise RuntimeError('Plan concurrently changed; repair never writes plan files')
        record={'schema_version':'paper3-r589-runtime-corrections-v1',
                'applied_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'hook_sha256':HOOK_SHA256,'operational_files':changed,
                'rounds':after,'scientific_source_files_changed':0,
                'plan_files_changed':0,'checkpoint_files_changed':0,'result_files_changed':0,
                'jobs_cancelled':0,'jobs_submitted':0,
                'activation':'Existing child retry and future worker launches read the fixed backend. Preserve existing scheduler jobs.',
                'corrections':['R5 deterministic cuBLAS environment before Torch',
                               'R5 support_EA reference dispatch',
                               'R8 balanced evaluation/support loss precision',
                               'R8 future meta-training query precision',
                               'R9 inherited balanced loss precision']}
        write_json(folder/'APPLIED.json',record)
        for number in OUTPUTS:
            output=root/'runs'/OUTPUTS[number]
            if output.is_dir():write_json(output/'runtime'/'runtime-corrections.json',
                {k:v for k,v in record.items() if k!='rounds'})
        emit('R589_REPAIR_APPLIED',hook_sha256=HOOK_SHA256,backends=len(changed),
             science_source_files_changed=0,checkpoint_files_changed=0,
             plan_files_changed=0,jobs_cancelled=0,jobs_submitted=0,
             activation='NEXT_CHILD_ATTEMPT_AND_ALL_FUTURE_WORKERS',
             next_attempt_may_wait_seconds=300,record=str(folder/'APPLIED.json'))
        for number in OUTPUTS:emit('ROUND_REPAIR_PROGRESS',**after[number])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
