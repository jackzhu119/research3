"""Detached R1/R11 result export; no scientific scheduler control."""
from __future__ import annotations
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
ROLES=('r1-paper','r11-paper','r1-full','r11-full')
MAX_METADATA=256*1024*1024


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def emit(event,**fields):print(json.dumps(dict(event=event,utc=utc(),**fields),ensure_ascii=False),flush=True)


def read(path,default=None,limit=64*1024*1024,max_bytes=None):
    if max_bytes is not None:limit=max_bytes
    limit=min(int(limit),MAX_METADATA)
    try:
        with Path(path).open('rb') as handle:raw=handle.read(limit+1)
        if len(raw)>limit:raise ValueError('EXPORT_METADATA_TOO_LARGE')
        return json.loads(raw)
    except FileNotFoundError:return default


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('EXPORT_STATE_MUST_NOT_BE_SYMLINK')
    fd,temporary=tempfile.mkstemp(prefix='.result-export-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(encoded(value));handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def client(work,credentials=None):
    from paper_credentials import upload_discover_token
    from transport.lossless_publish import VerifiedGitHub
    class FullReadVerified(VerifiedGitHub):
        def _asset_hash(self,asset):
            return super()._asset_hash({k:v for k,v in asset.items() if k!='digest'})
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    return FullReadVerified(upload_discover_token(credentials),emit,branch='paper3-results',cache_path=work/'verified-blobs.json')


def identity(pid):
    try:
        if type(pid) is not int or pid<=1:return None
        folder=Path('/proc')/str(pid)
        if folder.stat().st_uid!=os.getuid():return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return None
        return dict(pid=pid,uid=os.getuid(),start_ticks=fields[19],
            argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v])
    except (OSError,ValueError,IndexError,UnicodeError):return None


def same(record,current):
    return (isinstance(record,dict) and current is not None and all(record.get(k)==current.get(k)
        for k in ('pid','uid','start_ticks','argv')))


def stop_uploader(record,*,process=identity,send=os.kill,clock=time.monotonic,sleep=time.sleep):
    """Only an adapter-verified exact legacy CPU full uploader can be handed off."""
    if record.get('role')!='full' or record.get('kind')!='uploader' or record.get('uid')!=os.getuid():
        raise ValueError('EXACT_FULL_UPLOADER_ROLE_REQUIRED')
    expected={k:record.get(k) for k in ('pid','uid','start_ticks','argv')}
    signals=[]
    for number,grace in ((signal.SIGTERM,12),(signal.SIGKILL,5)):
        if not same(expected,process(record['pid'])):
            return dict(status='LEGACY_CPU_UPLOADER_ENDED_OR_IDENTITY_CHANGED',pid=record['pid'],signals=signals,scientific_processes_signalled=0)
        try:send(record['pid'],number)
        except ProcessLookupError:break
        signals.append(signal.Signals(number).name)
        emit('R11_LEGACY_CPU_UPLOAD_HANDOVER',pid=record['pid'],signal=signals[-1],scientific_processes_signalled=0)
        until=clock()+grace
        while same(expected,process(record['pid'])) and clock()<until:sleep(.1)
    if same(expected,process(record['pid'])):raise RuntimeError('LEGACY_CPU_UPLOADER_EXIT_PENDING')
    return dict(status='LEGACY_CPU_UPLOADER_STOPPED',pid=record['pid'],signals=signals,scientific_processes_signalled=0)


def handoff_full(args):
    module=importlib.import_module('r11_export')
    records=module.legacy_uploader_candidates(args,'full')
    results=[stop_uploader(row) for row in records]
    value=dict(results=results,scientific_processes_signalled=0,slurm_jobs_cancelled=0,slurm_jobs_submitted=0,updated_utc=utc())
    atomic(Path(args.work)/'r11-full'/'legacy-handover.json',value)
    return value


def command(args,mode,role):
    values=[sys.executable,'-I','-S','-B','-u',str(HERE/'delivery.py'),mode,'--role',role,
        '--root',args.root,'--work',args.work,'--plan',args.plan]
    if args.credentials:values+=['--credentials',args.credentials]
    return values


def once(args):
    module=importlib.import_module('r1_export' if args.role.startswith('r1-') else 'r11_export')
    function=module.paper_once if args.role.endswith('paper') else module.full_once
    result=function(args)
    if not isinstance(result,dict):raise ValueError('EXPORT_ADAPTER_RESULT_REQUIRED')
    result.update(role=args.role,updated_utc=utc(),training_controlled=False,slurm_jobs_submitted=0,slurm_jobs_cancelled=0)
    if args.role.endswith('paper') and result.get('paper_ready'):
        result['priority_ready']=True
    atomic(Path(args.work)/args.role/'status.json',result)
    emit('PAPER3_R1_R11_EXPORT_PROGRESS',**result)
    return result


def paper_ready(work):
    return all((read(Path(work)/role/'status.json',{}) or {}).get('priority_ready') is True for role in ('r1-paper','r11-paper'))


def worker(args):
    folder=Path(args.work)/args.role;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:emit('EXISTING_EXPORT_ROLE_REUSED',role=args.role);return 0
        current=identity(os.getpid()) or dict(pid=os.getpid(),uid=os.getuid(),argv=[],start_ticks=None)
        current.update(role=args.role,started_utc=utc());atomic(folder/'process.json',current)
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        try:os.nice(10)
        except OSError:pass
        handed=False
        while not (Path(args.work)/'STOP').exists():
            if args.role.endswith('full') and not paper_ready(args.work):
                status=dict(status='WAITING_FOR_PAPER_PRIORITY',role=args.role,updated_utc=utc(),training_controlled=False)
                atomic(folder/'status.json',status);emit('PAPER3_EXPORT_WAITING_FOR_PAPER',role=args.role);time.sleep(10);continue
            if args.role=='r11-full' and not handed:
                try:
                    handoff_full(args);handed=True
                except Exception as error:
                    atomic(folder/'status.json',dict(status='UPLOAD_HANDOVER_RETRY',error_type=type(error).__name__,updated_utc=utc(),training_controlled=False))
                    emit('R11_UPLOAD_HANDOVER_RETRY',error_type=type(error).__name__);time.sleep(20);continue
            # One group in a fresh process; metadata heap and stale caches do not
            # accumulate across thousands of completed experiment outputs.
            child=subprocess.Popen(command(args,'--once',args.role),stdin=subprocess.DEVNULL,close_fds=True)
            while child.poll() is None:
                atomic(folder/'heartbeat.json',dict(status='EXPORT_CHILD_ACTIVE',pid=os.getpid(),child_pid=child.pid,updated_utc=utc(),training_controlled=False))
                time.sleep(5)
            result=read(folder/'status.json',{}) or {}
            if child.returncode==0 and (result.get('complete') is True or result.get('priority_ready') is True):return 0
            if child.returncode:
                atomic(folder/'status.json',dict(result,status='RETRYING',role=args.role,child_exit_code=child.returncode,
                    updated_utc=utc(),training_controlled=False))
                emit('PAPER3_EXPORT_RETRY',role=args.role,child_exit_code=child.returncode);time.sleep(30)
            elif result.get('waiting') or result.get('status') not in ('UPLOADING','SOURCE_INVENTORY_ADVANCED') or result.get('pending_receipt_errors'):
                # Existing upload locks and missing native receipts are observed
                # again without repeatedly scanning large metadata in a loop.
                time.sleep(30)
        return 0


def start(args):
    work=Path(args.work);work.mkdir(parents=True,exist_ok=True);rows=[]
    with (work/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # This explicit upload start clears only this upload service's STOP.
        # Neither the R1 nor R11 scientific STOP is ever removed.
        (work/'STOP').unlink(missing_ok=True)
        for role in ROLES:
            folder=work/role;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'worker.lock').open('a') as rolelock:
                try:fcntl.flock(rolelock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    rows.append(dict(role=role,status='EXISTING_BACKGROUND_WORKER_REUSED',process=read(folder/'process.json',{})));continue
            # Explicit start rechecks even an earlier COMPLETE export, allowing
            # genuinely new completed fits to be captured without retraining.
            previous=read(folder/'status.json',{}) or {}
            if previous.get('priority_ready'):
                atomic(folder/'previous-paper-status.json',previous)
                previous=dict(previous,priority_ready=False,status='RECHECKING_AVAILABLE_PAPER_EVIDENCE')
                atomic(folder/'status.json',previous)
            with (folder/'upload.log').open('ab',buffering=0) as log:
                child=subprocess.Popen(command(args,'--worker',role),stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True,close_fds=True)
            acknowledged=False
            for _ in range(35):
                record=read(folder/'process.json',{}) or {}
                if record.get('pid')==child.pid:acknowledged=True;break
                if child.poll() is not None:break
                time.sleep(.1)
            rows.append(dict(role=role,pid=child.pid,acknowledged=acknowledged,log=str(folder/'upload.log')))
        result=dict(status='R1_R11_RESULT_EXPORT_BACKGROUND_STARTED',work=str(work),roles=rows,
            terminal_can_close=all(row.get('acknowledged') or row.get('status')=='EXISTING_BACKGROUND_WORKER_REUSED' for row in rows),
            scientific_stops_retained=True,slurm_jobs_submitted=0,slurm_jobs_cancelled=0,
            priority='Available exact paper evidence first; all completed scientific output follows.',updated_utc=utc())
        atomic(work/'STARTED.json',result);emit('PAPER3_R1_R11_EXPORT_STARTED',**result);return 0


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--start',action='store_true');modes.add_argument('--check',action='store_true')
    modes.add_argument('--worker',action='store_true');modes.add_argument('--once',action='store_true')
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work',default=str(Path.home()/'paper3-r1-r11-results-upload'))
    parser.add_argument('--plan','--r11-plan',dest='plan');parser.add_argument('--credentials');parser.add_argument('--role',choices=ROLES,default='r1-paper')
    args=parser.parse_args(argv);args.root=str(Path(args.root).expanduser().resolve());args.work=str(Path(args.work).expanduser().resolve())
    args.plan=str(Path(args.plan).expanduser().resolve()) if args.plan else str(Path(args.root)/'runs/r11_campaign/plan.json')
    args.r11_plan=args.plan
    if args.check:
        emit('PAPER3_R1_R11_EXPORT_STATUS',startup=read(Path(args.work)/'STARTED.json',{}),
            roles={role:dict(status=read(Path(args.work)/role/'status.json',{}),heartbeat=read(Path(args.work)/role/'heartbeat.json',{})) for role in ROLES});return 0
    if args.worker:return worker(args)
    if args.once:
        try:once(args);return 0
        except Exception as error:
            frames=traceback.extract_tb(error.__traceback__)
            site=frames[-1] if frames else None
            result=dict(status='RETRYING',role=args.role,error_type=type(error).__name__,
                error_site=dict(file=Path(site.filename).name,line=site.lineno,function=site.name) if site else None,
                http_status=getattr(error,'status',None),updated_utc=utc(),training_controlled=False)
            atomic(Path(args.work)/args.role/'status.json',result);emit('PAPER3_EXPORT_OPERATION_RETRY',**result);return 1
    return start(args)


if __name__=='__main__':raise SystemExit(main())
