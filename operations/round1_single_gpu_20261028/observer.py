"""Independent bounded R1 progress return; never controls scientific processes.

Every collection and delivery uses a fresh owned subprocess. The observer owns
the original monitor.lock so an old uploader must be handed off by the installer.
"""
from __future__ import annotations
import argparse
import fcntl
import getpass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
OPERATIONAL = 'runs/training_campaign_parallel_v1'
MAX_JSON = 2 * 1024 * 1024
MAX_SNAPSHOT = 512 * 1024
COMPLETE = {'COMPLETE', 'REUSED_COMPLETE', 'ALREADY_COMPLETE', 'FINISHED', 'REUSED'}
ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')
NUMERIC = {'outer_update', 'outer_updates_completed', 'inner_updates_actual',
    'episodes_completed', 'confirmed_outer_updates_observed',
    'last_observed_trajectory_update', 'observed_optimizer_calls_total',
    'completed_updates', 'completed_epochs', 'epoch', 'session'}
COMMITTED_MAPS = {'predictions':'committed_predictions', 'scores':'scored_sessions',
                  'completed':'completed_fit_receipts'}
EVENT_KEYS = NUMERIC | {'event', 'stage', 'phase', 'status', 'utc', 'action_key',
    'action_kind', 'kind', 'error_type', 'returncode', 'receipt_complete'}
TASK_KEYS = ('key', 'kind', 'method', 'arm', 'track', 'status', 'slot', 'job_id', 'error_type')
JOB_KEYS = ('slot', 'job_id', 'state', 'slurm_state', 'queue_reason', 'node', 'cpus',
    'memory', 'partition', 'updated_utc', 'attempt', 'name')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def emit(event, **fields):
    value=dict(event=event, utc=utc(), **fields);value['training_controlled']=False
    print(json.dumps(value,
                     ensure_ascii=False, allow_nan=False), flush=True)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    temporary = path.with_name('.' + path.name + '.tmp-' + str(os.getpid()))
    try:
        with temporary.open('wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path, cap=MAX_JSON):
    try:
        with Path(path).open('rb') as stream:
            stamp = os.fstat(stream.fileno()); raw = stream.read(cap + 1)
        if len(raw) > cap:
            return {}, dict(error='JSON_TOO_LARGE', mtime=stamp.st_mtime)
        value = json.loads(raw)
        if not isinstance(value, dict):
            return {}, dict(error='JSON_NOT_OBJECT', mtime=stamp.st_mtime)
        return value, dict(mtime=stamp.st_mtime, bytes=len(raw))
    except (OSError, ValueError, UnicodeError) as error:
        return {}, dict(error=type(error).__name__)


def age(stamp, now):
    return round(max(0., now-stamp), 1) if isinstance(stamp, (int, float)) else None


def clean(value):
    if isinstance(value, str):
        # Arbitrary diagnostic strings are never published, and known token
        # spellings are removed even from fields intended to contain statuses.
        value = re.sub(r'github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+', '[REDACTED]', value)
        value = re.sub(r'(?i)bearer\s+\S+', '[REDACTED]', value)
        return ''.join(c for c in value[:240] if c.isprintable())
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def scalars(value, keys):
    return {k: clean(value[k]) for k in keys if k in value and
            (value[k] is None or isinstance(value[k], (str, int, float, bool)))}


def safe_path(root, candidate):
    try:
        path = Path(candidate)
        if '..' in path.parts:
            return None
        path = path if path.is_absolute() else root/path
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return None
        if any(x in {'private', '.git', '.ssh', 'credentials', 'secrets'} for x in resolved.parts):
            return None
        return path
    except (OSError, TypeError, ValueError):
        return None


def progress_values(path):
    """Return counters only, including native committed internal predictions.

    Large native progress documents are parsed with the retained strict streaming
    tokenizer; prediction values and labels are never retained, and entry keys
    remain private to this metadata subprocess and are never published.
    """
    value,meta=read(path)
    if meta.get('error') != 'JSON_TOO_LARGE':
        selected=scalars(value,NUMERIC | {'status','stage','phase'})
        selected.update({target:len(value[name]) for name,target in COMMITTED_MAPS.items()
                         if isinstance(value.get(name),dict)})
        return selected,meta
    import receipt_projection
    class ProgressStream(receipt_projection._Stream):
        def members(self):
            self.whitespace()
            if self.peek()!='{':
                self.value(False,1);return None
            self.position+=1;self.whitespace();seen=set()
            if self.peek()=='}':self.position+=1;return 0
            while True:
                self.whitespace()
                if self.peek()!='"':self.error('COMMITTED_ENTRY_KEY_REQUIRED')
                key=self.string(True,key=True)
                if key is receipt_projection._TOO_LONG:self.error('COMMITTED_ENTRY_KEY_TOO_LONG')
                if key in seen:self.error('DUPLICATE_COMMITTED_ENTRY_KEY')
                seen.add(key);self.whitespace();self.expect(':');self.value(False,2)
                self.whitespace();char=self.take()
                if char=='}':return len(seen)
                if char!=',':self.error('COMMITTED_ENTRY_SEPARATOR_REQUIRED')
        def document(self):
            self.whitespace();self.expect('{');self.whitespace();result={}
            if self.peek()=='}':self.position+=1
            else:
                while True:
                    self.whitespace()
                    if self.peek()!='"':self.error('PROGRESS_FIELD_NAME_REQUIRED')
                    name=self.string(True,key=True);self.whitespace();self.expect(':')
                    if name in COMMITTED_MAPS:
                        count=self.members()
                        if count is not None:result[COMMITTED_MAPS[name]]=count
                    else:
                        keep=name in NUMERIC or name in ('status','stage','phase')
                        child=self.value(keep,1)
                        if keep:result[name]=child
                    self.whitespace();char=self.take()
                    if char=='}':break
                    if char!=',':self.error('PROGRESS_FIELD_SEPARATOR_REQUIRED')
            self.whitespace()
            if self.peek():self.error('TRAILING_PROGRESS_DATA')
            return result
    try:
        with Path(path).open('r',encoding='utf-8',errors='strict',newline='') as stream:
            stamp=os.fstat(stream.fileno());selected=ProgressStream(stream).document()
        return scalars(selected,NUMERIC | set(COMMITTED_MAPS.values()) | {'status','stage','phase'}),dict(
            mtime=stamp.st_mtime,bytes=stamp.st_size,streamed_counter_projection=True)
    except (OSError,ValueError,UnicodeError) as error:
        return {},dict(error=type(error).__name__)


def queue(ids=(), runner=subprocess.run):
    try:
        result = runner(['squeue', '-h', '-u', getpass.getuser(), '-o',
                         '%i|%j|%T|%P|%M|%R|%N'], stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=8, check=False)
        if result.returncode:
            return dict(status='UNAVAILABLE', error_type='SQUEUE_RETURNED_NONZERO', jobs=[])
        jobs = []; ids = set(map(str, ids))
        for line in result.stdout.splitlines():
            cells = [x.strip() for x in line.split('|', 6)]
            if len(cells) != 7:
                continue
            if cells[0] not in ids and not cells[1].startswith(('paper3r1p-', 'paper3r1s-', 'paper3-r1-')):
                continue
            jobs.append(dict(zip(('job_id', 'name', 'state', 'partition', 'elapsed',
                                  'reason_or_node', 'node'), map(clean, cells))))
        return dict(status='OBSERVED', observed_utc=utc(), jobs=jobs)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status='UNAVAILABLE', error_type=type(error).__name__, jobs=[])


def log_observation(path, now):
    try:
        with Path(path).open('rb') as stream:
            stamp = os.fstat(stream.fileno()); stream.seek(max(0, stamp.st_size-12288)); raw = stream.read(12288)
        events = []; errors = []
        for line in raw.decode('utf-8', 'replace').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                found = re.match(r'^([A-Za-z_][A-Za-z_0-9]*(?:Error|Exception))(?::|$)', line)
                if found:
                    errors.append(found.group(1))
                continue
            if isinstance(value, dict):
                event = scalars(value, EVENT_KEYS)
                if event:
                    events.append(event)
        return dict(age_seconds=age(stamp.st_mtime, now), bytes=stamp.st_size,
                    tail_sha256=hashlib.sha256(raw).hexdigest(), recent_events=events[-6:],
                    observed_error_types=list(dict.fromkeys(errors))[-4:],
                    raw_log_text_published=False)
    except OSError as error:
        return dict(read_error=type(error).__name__)


def collect(root, home, work, *, runner=subprocess.run, now=None):
    root, home, work = Path(root).absolute(), Path(home).absolute(), Path(work).absolute()
    now = time.time() if now is None else now; runtime = root/OPERATIONAL
    progress, pmeta = read(runtime/'progress.json')
    controller, cmeta = read(work/'controller-status.json')
    allocation, ameta = read(work/'allocation-state.json')
    claims, _ = read(runtime/'claims.json')
    frontier, qmeta = read(runtime/'queue.json')
    values = progress.get('tasks', [])
    values = list(values.values()) if isinstance(values, dict) else values
    values = values if isinstance(values, list) else []
    tasks = []; active = []
    for source in values:
        if not isinstance(source, dict):
            continue
        task = scalars(source, TASK_KEYS)
        task['observations'] = []
        if str(task.get('status', '')).upper() not in COMPLETE:
            active.append(task)
            paths = []
            if source.get('progress_path'):
                paths.append(source['progress_path'])
            if source.get('expected_receipt'):
                base = Path(source['expected_receipt']).parent
                paths.extend(str(base/n) for n in ('progress.json', 'execution_progress.json'))
            for candidate in dict.fromkeys(paths):
                path = safe_path(root, candidate)
                if path is None:
                    continue
                data, meta = progress_values(path)
                if data or meta.get('error') != 'FileNotFoundError':
                    task['observations'].append(dict(path=path.relative_to(root).as_posix(),
                        age_seconds=age(meta.get('mtime'), now), values=data,
                        read_error=meta.get('error')))
        tasks.append(task)
    scheduler = controller.get('scheduler', {})
    scheduler = scheduler if isinstance(scheduler, dict) else {}
    raw_jobs = scheduler.get('jobs', progress.get('jobs', []))
    raw_jobs = list(raw_jobs.values()) if isinstance(raw_jobs, dict) else raw_jobs
    jobs = [scalars(j, JOB_KEYS) for j in raw_jobs if isinstance(j, dict)] if isinstance(raw_jobs,list) else []
    live = queue([j['job_id'] for j in jobs if j.get('job_id')], runner=runner)
    current = {str(j['job_id']):j for j in live.get('jobs', [])}
    for job in jobs:
        match = current.get(str(job.get('job_id')))
        if match:
            job.update(live_state=match['state'], live_reason_or_node=match['reason_or_node'])
        elif live['status']=='OBSERVED':
            job['live_state']='NOT_IN_CURRENT_QUEUE'
    worker, wmeta = read(runtime/'worker-status/slot-0.json')
    w = scalars(worker, ('slot','job_id','pid','child_pid','phase','updated_utc','action_key','action_kind'))
    w['age_seconds'] = age(wmeta.get('mtime'), now)
    w['actual_numeric_counters'] = []
    for row in worker.get('action_progress', []) if isinstance(worker.get('action_progress'), list) else []:
        if isinstance(row, dict) and isinstance(row.get('values'), dict):
            w['actual_numeric_counters'].append(dict(file=clean(row.get('file')),
                                                   values=scalars(row['values'],NUMERIC)))
    key, token = worker.get('action_key'), worker.get('claim_token')
    claim = claims.get(key,{}) if isinstance(key,str) else {}
    bound = (isinstance(claim,dict) and claim.get('slot')==0
        and str(claim.get('job_id'))==str(worker.get('job_id')) and claim.get('token')==token
        and isinstance(token,str) and re.fullmatch('[0-9a-f]{32}',token) is not None)
    w['claim_bound'] = bool(bound)
    if bound:
        logfile=runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'kernel.log'
        w['current_action_log']=log_observation(logfile,now)
    job = str(worker.get('job_id',''))
    if re.fullmatch('[1-9][0-9]*',job):
        # New single allocation uses this exact declared path. No recursive logs scan.
        w['allocation_log']=log_observation(runtime/'logs'/('single-gpu-'+job+'.out'),now)
    main = [t for t in tasks if t.get('arm') not in ARMS and not str(t.get('kind','')).startswith('extension_')]
    extensions = {}
    for arm in ARMS:
        rows=[t for t in tasks if t.get('arm')==arm]
        base=root/'runs/source_extension_downstream'/arm
        catalog, catmeta=read(base/'catalog.json')
        receipt, _=read(base/'pre_internal_development_freeze.json')
        complete=receipt.get('status')=='SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN'
        extensions[arm]=dict(arm=arm,status='COMPLETE_RECEIPT_OBSERVED' if complete else (
            'RUNNING' if any(t.get('status')=='RUNNING' for t in rows) else 'INITIALIZED_OR_WAITING' if catalog else 'NOT_STARTED'),
            observed_actions=len(rows),completed_observed_actions=sum(t.get('status') in COMPLETE for t in rows),
            active_actions=[t for t in rows if t.get('status') not in COMPLETE],catalog_age_seconds=age(catmeta.get('mtime'),now),
            completion_receipt_status=receipt.get('status'),native_completed_artifact_counts={k:len(v) for k,v in catalog.items() if isinstance(v,dict)},
            count_scope='Native receipt catalog counts; wrapper actions are not individual fits.')
    snapshot=dict(schema_version='paper3-round1-parallel-monitor-v1',experiment='paper3-original-campaign-round1',round=1,
        updated_utc=utc(),sampled_unix=now,
        coordinator=dict(progress_path=OPERATIONAL+'/progress.json',progress_age_seconds=age(pmeta.get('mtime'),now),
                         **scalars(progress,('status','phase','updated_utc','partition','max_parallel_jobs'))),
        parallel_plan_total_tasks=len(tasks) if progress else None,
        parallel_plan_complete_tasks=sum(t.get('status') in COMPLETE for t in tasks) if progress else None,
        completed_by_kind={str(k):v for k,v in progress.get('completed_by_kind',{}).items() if type(v) is int}
            if isinstance(progress.get('completed_by_kind'),dict) else {},
        tasks=tasks,jobs=jobs,live_queue=live,
        mainline=dict(active_actions=[t for t in main if t.get('status') not in COMPLETE],observed_actions=len(main),
            completed_observed_actions=sum(t.get('status') in COMPLETE for t in main),
            coordinator_declares_complete=progress.get('mainline_complete') is True,
            count_scope='Observed primary actions; not a duration-weighted overall percentage.'),
        extension_arms=extensions,mainline_complete=progress.get('mainline_complete') is True,
        all_original_complete=progress.get('all_original_complete') is True,
        resources_target=dict(jobs=1,gpus_per_job=1,gpus_total=1,cpus_per_job=4,cpus_total=4,
            memory_per_job=['64G'],memory_total='64G',partition='hpc_gpu',
            configuration_source='explicit_single_gpu_4cpu_64G_request',observed_gpu_hardware_claimed=False),
        single_gpu_controller=dict(**scalars(controller,('status','phase','cycle','updated_utc','inflight_jobs','jobs_submitted')),
            age_seconds=age(cmeta.get('mtime'),now),read_error=cmeta.get('error')),
        single_gpu_worker=w,allocation_metadata_age_seconds=age(ameta.get('mtime'),now),
        dispatch_queue_age_seconds=age(qmeta.get('mtime'),now),
        interpretation='Sampling/queue/cache activity is not proof of numerical optimizer progress.',
        report_recomputed=False,trial_predictions_read=False,scientific_completion_claimed=False,
        training_controlled=False)
    if pmeta.get('error'):
        snapshot['coordinator']['read_error']=pmeta['error']
    return snapshot


def process_record(pid, proc=Path('/proc')):
    try:
        folder=Path(proc)/str(int(pid))
        if folder.stat().st_uid != os.getuid():
            return None
        raw=(folder/'cmdline').read_bytes()
        if len(raw)>65536:
            return None
        argv=[x.decode('utf-8','strict') for x in raw.rstrip(b'\0').split(b'\0')]
        fields=(folder/'stat').read_text().rsplit(')',1)[-1].split()
        if len(fields)<20 or fields[0] in ('Z','X','x'):
            return None
        return dict(pid=int(pid),uid=os.getuid(),argv=argv,start_ticks=fields[19])
    except (OSError,ValueError,UnicodeError):
        return None


def process_matches(record, proc=Path('/proc')):
    observed=process_record(record.get('pid',0),proc)
    return observed is not None and all(observed.get(k)==record.get(k) for k in ('pid','uid','argv','start_ticks'))


def legacy_monitor_candidates(root, home, work, *, proc=Path('/proc')):
    """Return exact identities only; the installer owns any later stop action.

    Original monitor argv declares exact --root and --operational. Historical
    single-node wrappers instead declare --work and bind root in config.json.
    No broad process-name matching or process-group signalling is used.
    """
    root,home,work=Path(root).absolute(),Path(home).absolute(),Path(work).absolute()
    runtime=root/OPERATIONAL; found=[]
    def option(argv,name):
        try:return argv[argv.index(name)+1]
        except (ValueError,IndexError):return None
    for folder in list(Path(proc).iterdir())[:131072]:
        if not folder.name.isdigit():continue
        row=process_record(int(folder.name),proc)
        if not row:continue
        argv=row['argv']
        scripts=[Path(a) for a in argv if a.endswith('.py')]
        if not any(p.name in ('monitor.py','monitor_native.py','observer.py') for p in scripts):continue
        if str(HERE/'observer.py') in argv and option(argv,'--work')==str(work):continue
        direct=option(argv,'--root')==str(root) and option(argv,'--operational')==str(runtime)
        wrapper=False
        prior=option(argv,'--work')
        if prior:
            base=Path(prior).absolute()
            if base.is_relative_to(home) and base!=work:
                config,_=read(base/'config.json',65536)
                wrapper=str(config.get('root',''))==str(root) and str(config.get('original_runtime',''))==str(runtime)
        if direct or wrapper:found.append(row)
    return found


def _command(mode,root,home,work,credentials=None):
    result=[sys.executable,'-I','-S','-B','-u',str(HERE/'observer.py'),mode,
            '--root',str(root),'--home',str(home),'--work',str(work)]
    if credentials:result.extend(['--credentials',str(credentials)])
    return result


def _bounded(command,timeout,cap=MAX_SNAPSHOT+65536):
    result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,timeout=timeout,check=False,close_fds=True)
    if result.returncode or len(result.stdout)>cap:
        raise RuntimeError('OBSERVER_HELPER_FAILED')
    value=json.loads(result.stdout)
    if not isinstance(value,dict):raise ValueError('OBSERVER_HELPER_INVALID')
    return value


def once(root,home,work,credentials=None,*,runner=_bounded):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    observed=runner(_command('--collect-cycle',root,home,work),30)
    raw=json.dumps(observed,ensure_ascii=False,allow_nan=False).encode()
    if len(raw)>MAX_SNAPSHOT:raise ValueError('SNAPSHOT_TOO_LARGE')
    atomic(work/'observer-snapshot.json',observed)
    receipt=runner(_command('--publish-cycle',root,home,work,credentials),52,MAX_SNAPSHOT)
    status=dict(status='SYNCED' if receipt.get('status')=='PROGRESS_UPLOADED_VERIFIED' else 'RETRYING',
        updated_utc=utc(),sample_updated_utc=observed.get('updated_utc'),
        complete_tasks=observed.get('parallel_plan_complete_tasks'),total_tasks=observed.get('parallel_plan_total_tasks'),
        phase=observed.get('coordinator',{}).get('phase'),delivery=receipt,training_controlled=False)
    atomic(work/'observer-status.json',status)
    emit('R1_SINGLE_GPU_OBSERVER_CYCLE',**status)
    return status


def run(root,home,work,credentials=None,*,interval=180,stop=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);runtime=Path(root)/OPERATIONAL
    runtime.mkdir(parents=True,exist_ok=True);stop=stop or threading.Event()
    dispatcher=None
    try:
        import results_observer
        dispatcher=results_observer.Dispatcher(root,home,work,credentials)
    except Exception as error:
        emit('R1_RESULTS_OBSERVER_START_RETRY',error_type=type(error).__name__)
    with (runtime/'monitor.lock').open('a') as lock:
        while not stop.is_set():
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                atomic(work/'observer-status.json',dict(status='WAITING_LEGACY_MONITOR_EXIT',updated_utc=utc(),training_controlled=False))
                emit('R1_SINGLE_GPU_OBSERVER_WAITING_MONITOR_LOCK');stop.wait(5)
        if stop.is_set():return 0
        emit('R1_SINGLE_GPU_OBSERVER_STARTED',pid=os.getpid(),interval=interval,
             branch='paper3-r1-progress',path='monitoring/paper3-round1/status.json')
        while not stop.is_set():
            started=time.monotonic()
            try:once(root,home,work,credentials)
            except Exception as error:
                atomic(work/'observer-status.json',dict(status='RETRYING',stage='COLLECT_OR_UPLOAD',
                    error_type=type(error).__name__,updated_utc=utc(),training_controlled=False))
                emit('R1_SINGLE_GPU_OBSERVER_RETRY',error_type=type(error).__name__)
            # Existing result summaries use their own low-priority child and
            # never delay live progress collection or scientific dispatch.
            try:
                if dispatcher is None:
                    import results_observer
                    dispatcher=results_observer.Dispatcher(root,home,work,credentials)
                snapshot,_=read(work/'observer-snapshot.json',MAX_SNAPSHOT)
                dispatcher.tick(snapshot)
            except Exception as error:
                emit('R1_RESULTS_OBSERVER_RETRY',error_type=type(error).__name__)
            stop.wait(max(1.,float(interval)-(time.monotonic()-started)))
    return 0


def start(root,home,work,credentials=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    with (work/'observer-start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        saved,_=read(work/'observer-process.json',65536)
        if process_matches(saved):return dict(status='ALREADY_RUNNING',**saved)
        token=secrets.token_hex(16)
        argv=_command('--run',root,home,work,credentials)+['--launch-token',token]
        with (work/'observer.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                close_fds=True,start_new_session=True)
        record=process_record(child.pid) or dict(pid=child.pid,uid=os.getuid(),argv=argv,start_ticks=None)
        record.update(root=str(Path(root).absolute()),work=str(work.absolute()),launch_token=token,started_utc=utc())
        atomic(work/'observer-process.json',record)
        return dict(status='STARTING',**record)


def _publish(root,home,work,credentials=None):
    spec=importlib.util.spec_from_file_location('_r1_single_gpu_upload',HERE/'upload_transport.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=Path(work)/'observer-snapshot.json'
    with data.open('rb') as stream:raw=stream.read(MAX_SNAPSHOT+1)
    # Transport step events remain private to this owned helper; only its
    # sanitized final receipt is returned to the guardian.
    return module.publish(raw,home,credentials=credentials,deadline_seconds=45,emit=lambda _event:None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--credentials',type=Path)
    parser.add_argument('--interval',type=float,default=180);parser.add_argument('--launch-token')
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('run','once','collect-cycle','publish-cycle'):mode.add_argument('--'+name,action='store_true')
    args=parser.parse_args(argv)
    if args.collect_cycle:
        print(json.dumps(collect(args.root,args.home,args.work),ensure_ascii=False,allow_nan=False));return 0
    if args.publish_cycle:
        print(json.dumps(_publish(args.root,args.home,args.work,args.credentials),ensure_ascii=False,allow_nan=False));return 0
    if args.once:
        runtime=args.root/OPERATIONAL;runtime.mkdir(parents=True,exist_ok=True)
        with (runtime/'monitor.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                emit('R1_SINGLE_GPU_OBSERVER_ALREADY_RUNNING');return 0
            once(args.root,args.home,args.work,args.credentials);return 0
    stop=threading.Event()
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:stop.set())
    return run(args.root,args.home,args.work,args.credentials,interval=args.interval,stop=stop)


if __name__=='__main__':raise SystemExit(main())
