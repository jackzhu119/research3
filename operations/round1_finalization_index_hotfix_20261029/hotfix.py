"""Live R1 operational hook: ignore only the self-mutating digest index.

No Slurm allocation is cancelled/submitted. Only an exactly bound, already
ready, CPU-only finalization verifier may be asked to unwind with SIGINT so the
existing supervisor naturally starts its next action child with the new hook.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
RUNTIME = 'runs/training_campaign_parallel_v1'
OLD_SHA = '865c661075c5c7769c567b431cb70f7997b852f8e641ba3cccd803d2b57fffca'
VERSION = 'paper3-r1-finalization-verification-reuse-v2'
BEGIN = '\n# PAPER3_R1_FINALIZATION_INDEX_FIX_V2_BEGIN\n'
END = '# PAPER3_R1_FINALIZATION_INDEX_FIX_V2_END\n'
CPU_ONLY = {'strongest_control', 'freeze'}


def sha(raw): return hashlib.sha256(raw).hexdigest()
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
def emit(event, **fields): print(json.dumps(dict(event=event,utc=utc(),**fields),ensure_ascii=False),flush=True)


def read(path, default=None, limit=8*1024*1024):
    try:
        with Path(path).open('rb') as handle: raw = handle.read(limit+1)
        if len(raw)>limit: raise ValueError('OPERATIONAL_RECORD_TOO_LARGE')
        return json.loads(raw)
    except FileNotFoundError: return default


def atomic_bytes(path, raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    mode=stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temporary=tempfile.mkstemp(prefix='.r1-cache-fix-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle: handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,mode);os.replace(temporary,path)
    finally: Path(temporary).unlink(missing_ok=True)


def atomic(path,value): atomic_bytes(path,(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())


@contextmanager
def locked(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB));yield handle


def option(argv,name):
    if not isinstance(argv,list) or argv.count(name)!=1:return None
    at=argv.index(name);return argv[at+1] if at+1<len(argv) else None


def identity(pid,proc=Path('/proc')):
    try:
        if type(pid) is not int or pid<=1:return None
        folder=Path(proc)/str(pid)
        if folder.stat().st_uid!=os.getuid():return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return None
        return dict(pid=pid,ppid=int(fields[1]),uid=os.getuid(),start_ticks=fields[19],
            argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v])
    except (OSError,ValueError,IndexError,UnicodeError):return None


def regular_owned(path):
    info=Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid():
        raise ValueError('EXPECTED_OWNED_REGULAR_OPERATIONAL_FILE')


def script(argv):
    if not argv or Path(argv[0]).resolve()!=Path(sys.executable).resolve():return None
    at=1
    while at<len(argv) and argv[at] in ('-I','-S','-B','-u','-E','-s','-O','-OO'):at+=1
    return Path(argv[at]).resolve() if at<len(argv) else None


def make_hook(old_module_path,old_sha,new_helper_path,new_sha):
    spec=dict(old_path=str(Path(old_module_path).resolve()),old_sha=old_sha,
        new_path=str(Path(new_helper_path).resolve()),new_sha=new_sha)
    block='''
import hashlib as _r1fix_hashlib, importlib.util as _r1fix_importlib, json as _r1fix_json
import os as _r1fix_os, sys as _r1fix_sys, time as _r1fix_time
from pathlib import Path as _r1fix_Path
_r1fix_spec = __SPEC__
_r1fix_original = _r1fix_sys.modules.get('finalization_cache')
if _r1fix_original is not None and _r1fix_Path(getattr(_r1fix_original,'__file__','')).resolve() == _r1fix_Path(_r1fix_spec['old_path']):
    if _r1fix_hashlib.sha256(_r1fix_Path(_r1fix_spec['old_path']).read_bytes()).hexdigest() != _r1fix_spec['old_sha']:
        raise ValueError('R1_FINALIZATION_OLD_HELPER_SHA256_DIFFERS')
    if _r1fix_hashlib.sha256(_r1fix_Path(_r1fix_spec['new_path']).read_bytes()).hexdigest() != _r1fix_spec['new_sha']:
        raise ValueError('R1_FINALIZATION_FIXED_HELPER_SHA256_DIFFERS')
    _r1fix_key = '_paper3_r1_finalization_fixed_' + _r1fix_spec['new_sha']
    _r1fix_module = _r1fix_sys.modules.get(_r1fix_key)
    if _r1fix_module is None:
        _r1fix_loader = _r1fix_importlib.spec_from_file_location(_r1fix_key,_r1fix_spec['new_path'])
        _r1fix_module = _r1fix_importlib.module_from_spec(_r1fix_loader)
        _r1fix_sys.modules[_r1fix_key] = _r1fix_module
        _r1fix_loader.loader.exec_module(_r1fix_module)
    _r1fix_original.install = _r1fix_module.install
    _r1fix_original._active_hotfix_helper_sha = _r1fix_spec['new_sha']
    print(_r1fix_json.dumps(dict(event='R1_FINALIZATION_INDEX_FIX_IMPORTED',
        utc=_r1fix_time.strftime('%Y-%m-%dT%H:%M:%SZ',_r1fix_time.gmtime()),
        pid=_r1fix_os.getpid(),helper_sha256=_r1fix_spec['new_sha'],
        version=_r1fix_module.VERSION,scientific_source_files_changed=False)),flush=True)
'''.replace('__SPEC__',repr(spec))
    return BEGIN+block+END


def patch_startup(path,hook,work):
    path,work=Path(path),Path(work);regular_owned(path)
    with locked(work/'install.lock'):
        original=path.read_bytes();source=original.decode('utf-8')
        if BEGIN in source or END in source:
            if source.count(BEGIN)!=1 or source.count(END)!=1:
                raise ValueError('INCOMPLETE_FINALIZATION_HOTFIX_BLOCK')
            at=source.index(BEGIN);stop=source.index(END,at)+len(END)
            source=source[:at]+source[stop:]
        patched=(source+hook).encode();compile(patched,str(path),'exec')
        backup=work/'backups'/(sha(original)+'.py')
        if not backup.exists():atomic_bytes(backup,original)
        if patched!=original:
            if path.read_bytes()!=original:raise RuntimeError('STARTUP_CHANGED_CONCURRENTLY')
            atomic_bytes(path,patched)
    return dict(path=str(path),before_sha256=sha(original),after_sha256=sha(patched),changed=patched!=original)


def context(root,home,work):
    root,home,work=(Path(x).expanduser().resolve() for x in (root,home,work))
    config=read(home/'paper3-r1-single-gpu/work/config.json',{})
    if config.get('root')!=str(root) or config.get('runtime')!=str(root/RUNTIME):
        raise ValueError('EXACT_SINGLE_GPU_CONFIGURATION_REQUIRED')
    command=config.get('worker_command',[])
    if option(command,'--root')!=str(root):raise ValueError('WORKER_ROOT_DIFFERS')
    runner=script(command)
    if runner is None or runner.name!='worker_runner.py' or not runner.is_relative_to(home/'paper3-r1-single-gpu'):
        raise ValueError('EXACT_SINGLE_GPU_WORKER_RUNNER_REQUIRED')
    old=runner.parent/'finalization_cache.py';regular_owned(old)
    digest=sha(old.read_bytes())
    fixed=HERE/'finalization_cache.py';regular_owned(fixed)
    new_sha=sha(fixed.read_bytes())
    if digest not in (OLD_SHA,new_sha):raise ValueError('OBSERVED_FINALIZATION_HELPER_DIFFERS')
    native=Path(config['native_package']).resolve()
    if option(command,'--package')!=str(native) or not native.is_relative_to(home):
        raise ValueError('EXACT_NATIVE_OPERATIONAL_PACKAGE_REQUIRED')
    startup=native/'action_startup.py';regular_owned(startup)
    return dict(root=str(root),home=str(home),work=str(work),runtime=str(root/RUNTIME),
        native=str(native),runner=str(runner),old_helper=str(old),old_sha=digest,
        fixed_helper=str(fixed),fixed_sha=new_sha,version=VERSION,python=config.get('python',sys.executable))


def probe_log(path,token):
    try:
        with Path(path).open('rb') as handle:raw=handle.read(256*1024)
    except FileNotFoundError:return dict(ready=False,fixed_imported=False,scope_active=False,old_scope_active=False)
    imported=False;scope=False;old_scope=False
    for line in raw.decode('utf-8',errors='replace').splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('event')=='R1_FINALIZATION_INDEX_FIX_IMPORTED' and row.get('version')==VERSION:imported=True
        if row.get('event')=='R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE':
            if row.get('version')==VERSION:scope=True
            elif row.get('version')=='paper3-r1-finalization-verification-reuse-v1':old_scope=True
    return dict(ready=('KERNEL_READY '+token).encode() in raw,
        fixed_imported=imported,scope_active=scope,old_scope_active=old_scope)


def binding(config,*,clock=time.time):
    root=Path(config['root']);runtime=root/RUNTIME
    claims=read(runtime/'claims.json',{}) or {}
    worker=read(runtime/'worker-status/slot-0.json',{}) or {}
    key=worker.get('action_key');kind=worker.get('action_kind')
    if key not in claims:return dict(status='WAITING_FOR_CURRENT_CLAIM')
    claim=claims[key];token=claim.get('token');job=str(claim.get('job_id',''))
    if (kind not in CPU_ONLY or claim.get('action_kind')!=kind or key not in ('strongest-control','freeze')
        or worker.get('claim_token')!=token or str(worker.get('job_id'))!=job
        or claim.get('slot')!=0 or worker.get('slot')!=0 or worker.get('pid')!=claim.get('pid')
        or not job.isdigit() or not isinstance(token,str) or len(token)!=32
        or any(c not in '0123456789abcdef' for c in token)):
        return dict(status='CURRENT_ACTION_PRESERVED',action_key=key,action_kind=kind)
    if type(worker.get('updated_at')) not in (int,float) or clock()-worker['updated_at']>180:
        return dict(status='WAITING_FOR_FRESH_WORKER_RECORD',action_key=key)
    folder=runtime/'actions'/sha(key.encode())/'attempts'/token
    record=read(folder/'action.json',{})
    action=record.get('action',{})
    if (action.get('key')!=key or action.get('kind')!=kind
        or {'strongest-control':'strongest_control','freeze':'freeze'}.get(key)!=kind
        or action.get('serial') is not True or claim.get('serial') is not True or record.get('claim')!=claim):
        return dict(status='ACTION_CLAIM_BINDING_CHANGED')
    action_file=(folder/'action.json').resolve();log=(folder/'kernel.log').resolve()
    probe=probe_log(log,token)
    return dict(status='BOUND_CPU_FINALIZATION',action_key=key,action_kind=kind,
        token=token,job_id=job,slot=0,parent_pid=worker.get('pid'),child_pid=worker.get('child_pid'),
        action_file=str(action_file),log=str(log),returned_exists=(folder/'returned.json').exists(),**probe)


def node_activate(config,*,process=identity,send=os.kill,clock=time.time):
    current=binding(config,clock=clock)
    if current.get('status')!='BOUND_CPU_FINALIZATION':return current
    if current['fixed_imported'] or current['scope_active']:
        return dict(current,status='FIXED_HELPER_ALREADY_LOADED',signal_sent=False)
    if not current['ready'] or not current['old_scope_active'] or current['returned_exists']:
        return dict(current,status='WAITING_FOR_READY_OLD_VERIFIER_OR_NATURAL_RETRY',signal_sent=False)
    if os.environ.get('SLURM_JOB_ID')!=current['job_id']:
        return dict(status='ALLOCATION_BINDING_DIFFERS',signal_sent=False)
    value=process(current['child_pid'])
    if value is None:return dict(status='OLD_VERIFIER_ALREADY_ENDED',signal_sent=False)
    argv=value['argv']
    if (value.get('uid')!=os.getuid() or value.get('ppid')!=current['parent_pid'] or script(argv)!=Path(config['runner'])
        or '--child' not in argv or option(argv,'--root')!=config['root']
        or option(argv,'--runtime-dir')!=config['runtime'] or option(argv,'--slot')!='0'
        or option(argv,'--package')!=config['native'] or option(argv,'--action-file')!=current['action_file']):
        return dict(status='PROCESS_IDENTITY_DIFFERS',signal_sent=False)
    request=Path(config['work'])/'refreshed-claims'/(current['token']+'.json')
    try:
        with locked(request.with_suffix('.lock'),blocking=False):
            if request.exists():return dict(status='THIS_VERIFIER_REFRESH_ALREADY_REQUESTED',signal_sent=False)
            latest=binding(config,clock=clock)
            if latest!=current or process(value['pid'])!=value:
                return dict(status='CURRENT_ACTION_OR_PROCESS_CHANGED',signal_sent=False)
            intent=dict(status='CPU_FINALIZATION_REFRESH_INTENT',job_id=current['job_id'],
                action_key=current['action_key'],claim_token=current['token'],child_pid=value['pid'],
                child_start_ticks=value['start_ticks'],signal_name='SIGINT',signal_sent=False,
                optimizer_processes_signalled=0,slurm_jobs_cancelled=0,slurm_jobs_submitted=0,updated_utc=utc())
            # Persist before signaling: a remote-step timeout cannot authorize a
            # second request to this same scientific action process.
            atomic(request,intent)
            if binding(config,clock=clock)!=current or process(value['pid'])!=value:
                result=dict(intent,status='CURRENT_ACTION_OR_PROCESS_CHANGED')
                atomic(request,result);return result
            # CPU-only selection/freeze, never an optimizer. The original
            # child_main unwinds SIGINT and releases its own exact claim.
            try:send(value['pid'],signal.SIGINT)
            except ProcessLookupError:
                result=dict(intent,status='OLD_VERIFIER_ALREADY_ENDED')
            except OSError as error:
                result=dict(intent,status='CPU_REFRESH_SIGNAL_UNCERTAIN',error_type=type(error).__name__)
            else:
                result=dict(intent,status='CPU_FINALIZATION_REFRESH_REQUESTED',signal_sent=True)
            atomic(request,result);return result
    except BlockingIOError:
        return dict(status='THIS_VERIFIER_REFRESH_REQUEST_IN_PROGRESS',signal_sent=False)


def activation_step(config,*,runner=subprocess.run):
    row=binding(config)
    if row.get('status')!='BOUND_CPU_FINALIZATION':return row
    if row['scope_active']:return dict(row,status='FIXED_VERIFICATION_SCOPE_ACTIVE')
    if row['fixed_imported']:return dict(row,status='FIXED_HELPER_IMPORTED_WAITING_FOR_NATIVE_SCOPE')
    if not row['ready'] or not row['old_scope_active'] or row['returned_exists']:
        return dict(row,status='WAITING_FOR_OLD_VERIFIER_READY_OR_NATURAL_RETRY')
    request=Path(config['work'])/'refreshed-claims'/(row['token']+'.json')
    if request.exists():return dict(row,status='WAITING_FOR_SAME_ALLOCATION_RETRY')
    queue=runner(['squeue','-h','-j',row['job_id'],'-o','%i|%j|%T|%N'],capture_output=True,text=True,timeout=12)
    if queue.returncode:return dict(status='QUEUE_OBSERVATION_RETRY')
    exact=[line.split('|') for line in queue.stdout.splitlines() if line.split('|',1)[0]==row['job_id']]
    if len(exact)!=1 or len(exact[0])!=4 or not exact[0][1].startswith('paper3r1s-') or exact[0][2]!='RUNNING':
        return dict(status='ALLOCATION_NOT_OBSERVED_RUNNING')
    command=['srun','--jobid='+row['job_id'],'--overlap','--nodes=1','--ntasks=1',
        '--cpus-per-task=1','--mem=128M','--gres=none','--time=00:02:00',
        config['python'],'-I','-S','-B','-u',str(HERE/'hotfix.py'),'--node-activate',
        '--root',config['root'],'--home',config['home'],'--work',config['work']]
    response=runner(command,capture_output=True,text=True,timeout=40)
    for line in reversed(response.stdout.splitlines()):
        try:result=json.loads(line)
        except ValueError:continue
        if result.get('event')=='R1_FINALIZATION_NODE_ACTIVATION':
            return result.get('result',{})
    return dict(status='NODE_METADATA_STEP_RETRY',returncode=response.returncode)


def completed_status(config):
    root=Path(config['root'])
    strongest=read(root/'runs/training_campaign_v1/strongest_control.json',{}) or {}
    freeze=read(root/'runs/training_campaign_v1/pre_internal_freeze.json',{}) or {}
    if freeze.get('status')=='IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE':
        return 'FINALIZATION_COMPLETED_NATIVE_FREEZE_OBSERVED'
    if strongest.get('status')=='STRONGEST_DEVELOPMENT_CONTROL_FROZEN':
        return 'STRONGEST_CONTROL_COMPLETED_NATIVE_RECEIPT_OBSERVED'
    return None


def watch(args,*,step=activation_step,maximum_cycles=None):
    config=context(args.root,args.home,args.work)
    stop=threading.Event()
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda *_:stop.set())
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with locked(Path(args.work)/'watch.lock'):
        sequence=0
        while not stop.is_set():
            sequence+=1
            try:
                if (Path(config['runtime'])/'STOP').exists() or (Path(args.work)/'STOP').exists():
                    result=dict(status='FIX_INSTALLED_CAMPAIGN_STOP_RETAINED',version=VERSION,
                        helper_sha256=config['fixed_sha'],updated_utc=utc(),
                        slurm_jobs_cancelled=0,slurm_jobs_submitted=0,signal_sent=False)
                    atomic(Path(args.work)/'status.json',result);emit('R1_FINALIZATION_FIX_WATCH',**result)
                    return 0
                result=step(config)
                if result.get('status') not in ('FIXED_VERIFICATION_SCOPE_ACTIVE',):
                    done=completed_status(config)
                    if done:result=dict(result,status=done)
                result.update(version=VERSION,helper_sha256=config['fixed_sha'],cycle=sequence,
                    updated_utc=utc(),slurm_jobs_cancelled=0,slurm_jobs_submitted=0)
                atomic(Path(args.work)/'status.json',result)
                atomic(Path(config['runtime'])/'finalization-hotfix-status.json',result)
                emit('R1_FINALIZATION_FIX_WATCH',**result)
                if result.get('status') in ('FIXED_VERIFICATION_SCOPE_ACTIVE',
                    'FINALIZATION_COMPLETED_NATIVE_FREEZE_OBSERVED','STRONGEST_CONTROL_COMPLETED_NATIVE_RECEIPT_OBSERVED'):
                    return 0
            except Exception as error:
                result=dict(status='ACTIVATION_RETRYING',error_type=type(error).__name__,updated_utc=utc(),
                    slurm_jobs_cancelled=0,slurm_jobs_submitted=0)
                atomic(Path(args.work)/'status.json',result);emit('R1_FINALIZATION_FIX_RETRY',**result)
            if maximum_cycles is not None and sequence>=maximum_cycles:return 0
            stop.wait(15)
    return 0


def apply(root,home,work,*,popen=subprocess.Popen):
    config=context(root,home,work);work=Path(config['work']);work.mkdir(parents=True,exist_ok=True)
    hook=make_hook(config['old_helper'],config['old_sha'],config['fixed_helper'],config['fixed_sha'])
    change=patch_startup(Path(config['native'])/'action_startup.py',hook,work)
    value=dict(status='FIX_INSTALLED_ACTIVATION_PENDING',version=VERSION,helper_sha256=config['fixed_sha'],
        operational_change=change,immutable_release_files_changed=0,scientific_source_files_changed=0,
        optimizer_processes_signalled=0,slurm_jobs_cancelled=0,slurm_jobs_submitted=0,updated_utc=utc())
    atomic(work/'applied.json',value)
    with locked(work/'start.lock'):
        record=read(work/'watch-process.json',{}) or {};current=identity(record.get('pid'))
        if current is not None and current.get('argv')==record.get('argv') and current.get('start_ticks')==record.get('start_ticks'):
            return dict(value,watch_pid=current['pid'],watch_reused=True,terminal_can_close=True)
        command=[config['python'],'-I','-S','-B','-u',str(HERE/'hotfix.py'),'--watch',
            '--root',config['root'],'--home',config['home'],'--work',config['work']]
        with (work/'activation.log').open('ab',buffering=0) as log:
            child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        until=time.monotonic()+2;current=None
        while time.monotonic()<until:
            current=identity(child.pid)
            if current is not None and current['argv']==command:break
            if child.poll() is not None:break
            time.sleep(.1)
        if child.poll() is not None and not (work/'status.json').exists():
            raise RuntimeError('ACTIVATION_WATCHER_START_FAILED_READ_ACTIVATION_LOG')
        record=dict(pid=child.pid,argv=command,start_ticks=current.get('start_ticks') if current else None,updated_utc=utc())
        atomic(work/'watch-process.json',record)
    return dict(value,watch_pid=child.pid,terminal_can_close=True,activation_status=read(work/'status.json',{}))


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--apply',action='store_true');modes.add_argument('--check',action='store_true')
    modes.add_argument('--watch',action='store_true');modes.add_argument('--node-activate',action='store_true')
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--home',default=str(Path.home()));parser.add_argument('--work')
    args=parser.parse_args(argv)
    if args.work is None:args.work=str(Path(args.home).expanduser().resolve()/'paper3-r1-finalization-hotfix/work')
    if args.watch:return watch(args)
    if args.node_activate:
        result=node_activate(context(args.root,args.home,args.work))
        emit('R1_FINALIZATION_NODE_ACTIVATION',result=result);return 0
    if args.check:
        result=dict(applied=read(Path(args.work)/'applied.json',{}),activation=read(Path(args.work)/'status.json',{}))
        try:
            config=context(args.root,args.home,args.work)
            observed=binding(config)
            done=completed_status(config)
            if done:observed=dict(observed,status=done)
            elif observed.get('scope_active'):observed=dict(observed,status='FIXED_VERIFICATION_SCOPE_ACTIVE')
            elif observed.get('fixed_imported'):observed=dict(observed,status='FIXED_HELPER_IMPORTED_WAITING_FOR_NATIVE_SCOPE')
            result['current_observation']=dict(observed,observed_utc=utc())
        except Exception as error:
            result['current_observation']=dict(status='OBSERVATION_UNAVAILABLE',error_type=type(error).__name__)
    else:result=apply(args.root,args.home,args.work)
    emit('R1_FINALIZATION_FIX_RESULT',**result);return 0


if __name__=='__main__':raise SystemExit(main())
