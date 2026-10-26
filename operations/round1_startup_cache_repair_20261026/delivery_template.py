#!/usr/bin/env python3
"""Attach persistent frozen-input digest caching to existing R1 startup only."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import zlib

HOOK_B85='__HOOK_B85__'
HOOK_SHA256='__HOOK_SHA256__'
BASE_STARTUP_SHA256='e2141c386bf7590e4d738e11661b7f0cd297c06b9f4ae5ed7fa5c2c72fe1df42'
BASE_BOOTSTRAP_SHA256='43513735525e678138d23fa848d901c3c780ece11a57cba3d675f158d13ca9d8'
MARKER='PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1'
SPAWN_MARKER='PAPER3_R1_OWNED_DIGEST_CACHE_V1'


def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields),ensure_ascii=False),flush=True)


def sha(data):return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    mode=path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary=path.with_name('.'+path.name+'.r1-cache-'+str(os.getpid()))
    try:
        with temporary.open('wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,mode);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def write_json(path,value):
    atomic(path,(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def option(command,flag):
    try:return command[command.index(flag)+1]
    except (ValueError,IndexError):return None


def packages(runtime,home):
    """Follow only R1 launch and durable scheduler package declarations."""
    launch=read(runtime/'launch.json');state=read(runtime/'scheduler-state.json')
    values=[launch.get('package')]
    if isinstance(launch.get('worker_command'),list):
        values.append(option(launch['worker_command'],'--package'))
    for slot in state.get('slots',{}).values():
        for attempt in slot.get('attempts',[]):
            if isinstance(attempt.get('command'),list):
                values.append(option(attempt['command'],'--package'))
    found=set()
    def add(value):
        if not isinstance(value,str):return
        path=Path(value).expanduser().resolve()
        if (path/'round1/action_startup.py').is_file():path=path/'round1'
        if ((path/'action_startup.py').is_file() and (path/'kernel_bridge.py').is_file()
                and (path/'worker.py').is_file()):found.add(path)
    for value in values:add(value)
    if not found:
        for pattern in ('paper3-r1-repairs/*/active-round1/*/action_startup.py',
                        'paper3-parallel-migrations/*/round1/action_startup.py'):
            for path in Path(home).glob(pattern):add(str(path.parent))
    return sorted(found)


def patched_startup(original,helper_path):
    text=original.decode('utf-8')
    begin='\n# '+MARKER+'\n';end='# END_'+MARKER+'\n'
    if begin in text or end in text:
        if text.count(begin)!=1 or text.count(end)!=1:
            raise ValueError('Incomplete or duplicate R1 startup cache marker')
        start=text.index(begin);finish=text.index(end,start)+len(end)
        text=text[:start]+text[finish:]
    if sha(text.encode())!=BASE_STARTUP_SHA256:
        raise ValueError('Unrecognized R1 operational action_startup.py; nothing was switched')
    block=(begin+
        '_paper3_uncached_startup_active = active\n'
        '@contextmanager\n'
        'def active(root, action, *, modules=None, torch_module=None, emit=None,\n'
        '           action_file=None, traceback_interval=300):\n'
        '    _cache = __import__("runpy").run_path('+repr(str(helper_path))+', run_name="_paper3_r1_digest_cache")\n'
        '    with _cache["active_digest_cache"](root, action, emit=emit):\n'
        '        with _paper3_uncached_startup_active(root, action, modules=modules,\n'
        '                torch_module=torch_module, emit=emit, action_file=action_file,\n'
        '                traceback_interval=traceback_interval):\n'
        '            yield\n'+end)
    fixed=(text+block).encode();compile(fixed,'action_startup.py','exec')
    return fixed


def bootstrap_files(runtime,found):
    launch=read(runtime/'launch.json');paths=set()
    for package in found:
        path=package/'r1_spawn_bootstrap.py'
        if path.is_file():paths.add(path.resolve())
    if isinstance(launch.get('direct_package'),str):
        path=Path(launch['direct_package']).expanduser()/'r1_spawn_bootstrap.py'
        if path.is_file():paths.add(path.resolve())
    return sorted(paths)


def patched_bootstrap(original,helper_path):
    text=original.decode('utf-8');begin='\n# '+SPAWN_MARKER+'\n';end='# END_'+SPAWN_MARKER+'\n'
    if begin in text or end in text:
        if text.count(begin)!=1 or text.count(end)!=1:
            raise ValueError('Incomplete or duplicate owned-stage digest cache marker')
        start=text.index(begin);finish=text.index(end,start)+len(end)
        text=text[:start]+text[finish:]
    if sha(text.encode())!=BASE_BOOTSTRAP_SHA256:
        raise ValueError('Unrecognized R1 operational spawn bootstrap; nothing was switched')
    block=(begin+
        '_paper3_original_bootstrap_owned_stage = bootstrap_owned_stage\n'
        'def bootstrap_owned_stage(root):\n'
        '    result = _paper3_original_bootstrap_owned_stage(root)\n'
        '    if isinstance(result, dict) and result.get("status") == "NOT_APPLICABLE_NO_PARALLEL_LAUNCH":\n'
        '        return result\n'
        '    try:\n'
        '        __import__("runpy").run_path('+repr(str(helper_path))+', run_name="_paper3_r1_owned_digest_cache")["install_process_cache"](root)\n'
        '    except Exception as error:\n'
        '        print(json.dumps({"event": "R1_OWNED_DIGEST_CACHE_UNAVAILABLE", "error_type": type(error).__name__}), flush=True)\n'
        '    return result\n'+end)
    fixed=(text+block).encode();compile(fixed,'r1_spawn_bootstrap.py','exec')
    return fixed


def preserved_sources(root,found):
    values={}
    for package in found:
        for name in ('worker.py','kernel_bridge.py','audited_runtime.py',
                     'hardware_comparison.py','scheduler.py','START.py'):
            path=package/name
            if path.is_file():values[path]=path.read_bytes()
    for name in ('p3work/contracts.py','tools/r2_backup.py'):
        path=root/name
        if path.is_file():values[path]=path.read_bytes()
    return values


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home',type=Path,default=Path.home())
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve();home=args.home.expanduser().resolve()
    runtime=root/'runs/training_campaign_parallel_v1'
    emit('R1_STARTUP_CACHE_REPAIR_STARTED',imports_torch=False,full_data_rescan=False)
    helper=zlib.decompress(base64.b85decode(HOOK_B85))
    if sha(helper)!=HOOK_SHA256:raise ValueError('Embedded R1 digest cache SHA differs')
    compile(helper,'r1_digest_cache.py','exec')
    folder=home/'paper3-runtime-fixes'/'r1-digest-cache-v1'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'apply.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        helper_path=folder/'r1_digest_cache.py'
        found=packages(runtime,home)
        if not found:raise ValueError('Existing R1 startup package not found; no files were switched')
        prepared=[]
        for package in found:
            path=package/'action_startup.py';original=path.read_bytes()
            prepared.append((path,original,patched_startup(original,helper_path)))
        owned_bootstraps=bootstrap_files(runtime,found)
        for path in owned_bootstraps:
            original=path.read_bytes();prepared.append((path,original,patched_bootstrap(original,helper_path)))
        science=preserved_sources(root,found)
        metadata={p:p.read_bytes() for p in (runtime/'launch.json',runtime/'scheduler-state.json') if p.is_file()}
        if not helper_path.is_file() or helper_path.read_bytes()!=helper:atomic(helper_path,helper)
        changed=[]
        for path,original,fixed in prepared:
            backup=folder/'backups'/(sha(original)+'.py')
            if not backup.is_file():atomic(backup,original)
            if original!=fixed:atomic(path,fixed)
            row={'path':str(path),'before_sha256':sha(original),'after_sha256':sha(fixed),'changed':original!=fixed}
            changed.append(row);emit('R1_STARTUP_CACHE_ENTRY_PATCHED',**row)
        for path,value in science.items():
            if path.read_bytes()!=value:raise RuntimeError('Native R1 source changed concurrently: '+str(path))
        # Scheduler state is live; record its original signature rather than
        # treating concurrent job progress as an installer failure.
        receipt={'schema_version':'paper3-r1-persistent-digest-startup-repair-v1',
            'applied_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'helper_sha256':HOOK_SHA256,'operational_files':changed,
            'owned_stage_bootstrap_files':len(owned_bootstraps),
            'preserved_native_sources':[{'path':str(p),'sha256':sha(v)} for p,v in sorted(science.items())],
            'original_metadata_sha256':{str(p):sha(v) for p,v in metadata.items()},
            'scientific_files_changed':0,'checkpoint_files_changed':0,'plan_files_changed':0,
            'jobs_submitted':0,'jobs_cancelled':0,'signals_sent':0,'other_rounds_changed':0,
            'activation':'NEXT_R1_ACTION_CHILD_ONLY_CURRENT_CHILD_PRESERVED',
            'cold_cache_behavior':'Successful native hashes are cached; the first uncached read still uses the original hash function'}
        write_json(folder/'APPLIED.json',receipt)
        write_json(runtime/'digest-cache-repair.json',receipt)
        emit('R1_STARTUP_CACHE_REPAIR_APPLIED',helper_sha256=HOOK_SHA256,
             packages=len(found),owned_stage_bootstrap_files=len(owned_bootstraps),
             scientific_files_changed=0,checkpoint_files_changed=0,
             jobs_cancelled=0,jobs_submitted=0,signals_sent=0,
             activation=receipt['activation'],record=str(folder/'APPLIED.json'))
    return 0


if __name__=='__main__':raise SystemExit(main())
