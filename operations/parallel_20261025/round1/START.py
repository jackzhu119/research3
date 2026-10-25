"""Migrate the original campaign to three held single-GPU jobs, then resume."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from kernel_bridge import plan_ready, action_complete
from scheduler import ThreeJobScheduler
from handoff import checkpoint_handoff_step
from retire_legacy import retire_legacy_monitor
from audited_runtime import inspect_primary_measurements

DEFAULT_ROOT = '/public/home/zhuziyuan/projects/paper3'
OPERATIONAL = 'runs/training_campaign_parallel_v1'
SCHOOL_PYTHON = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python'


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, ensure_ascii=False, sort_keys=True), flush=True)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield handle


def command(argv):
    return subprocess.run(argv, text=True, capture_output=True, timeout=20)


def discover_context(root, old_job, direct=None, manifest=None):
    """Find already deployed operational adapters; never replace project code."""
    arguments = []
    try:
        observed = command(['scontrol', 'show', 'job', '-o', str(old_job)])
        match = re.search(r'(?:^|\s)Command=(.*?)(?:\s+WorkDir=|\s+StdErr=|$)', observed.stdout)
        if match:
            import shlex
            arguments = shlex.split(match.group(1))
    except (OSError, subprocess.TimeoutExpired):
        pass
    direct_candidates = [Path(direct).expanduser()] if direct else []
    manifest_candidates = [Path(manifest).expanduser()] if manifest else []
    for argument in arguments:
        path = Path(argument)
        if path.name == 'direct_runner.py': direct_candidates.append(path.parent)
        if path.name == 'payload_manifest.json': manifest_candidates.append(path)
    legacy = read(root/'.paper3-hpc/direct-train/state.json', {}) or {}
    for attempt in reversed(legacy.get('history', [])):
        name = (attempt.get('submission') or {}).get('package')
        if name: direct_candidates.append(Path(name))
    home = Path.home()
    direct_candidates += sorted((home/'paper3-direct-train').glob('*/direct_runner.py'),
                                key=lambda p:p.stat().st_mtime, reverse=True)
    normalized = [p.parent if p.name=='direct_runner.py' else p for p in direct_candidates]
    direct_path = next((p for p in normalized if (p/'direct_runner.py').is_file()
                        and (p/'campaign_recovery.py').is_file()), None)
    if direct_path is None:
        raise FileNotFoundError('Existing direct_runner.py/campaign_recovery.py installation not found')
    if not manifest_candidates:
        source = (direct_path/'launch.py').read_text()
        tree = __import__('ast').parse(source)
        for node in tree.body:
            if isinstance(node, __import__('ast').Assign) and any(
                    isinstance(t,__import__('ast').Name) and t.id=='MANIFEST' for t in node.targets):
                try: manifest_candidates.append(Path(__import__('ast').literal_eval(node.value)))
                except (ValueError,TypeError): pass
    manifest_candidates += sorted((home/'paper3-runtime-repairs').glob('*/payload_manifest.json'),
                                  key=lambda p:p.stat().st_mtime, reverse=True)
    manifest_path = next((p for p in manifest_candidates if p.is_file() and all(
        (p.parent/name).is_file() for name in ('episode_json_compat.py','runtime_policy.py','runtime_transfer.py'))),None)
    if manifest_path is None:
        raise FileNotFoundError('Existing runtime repair manifest/hooks not found')
    return {'direct_package': str(direct_path.resolve()), 'payload_manifest': str(manifest_path.resolve())}


def memory_mib(text):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGTP]?)', str(text).strip(), re.I)
    if not match: return None
    return int(float(match[1])*{'':1,'K':1/1024,'M':1,'G':1024,'T':1024**2,'P':1024**3}[match[2].upper()])


def confirm_three_held(snapshot, *, run=command):
    """A successful sbatch ACK alone is not the required queued replacement."""
    jobs = snapshot.get('jobs', [])
    if len(jobs)!=3 or any(not row.get('job_id') for row in jobs) or len({str(row.get('job_id')) for row in jobs})!=3:
        return {'ready':False, 'reason':'three_replacement_acknowledgements_pending'}
    observed = run(['squeue','--array','--noheader','--user',__import__('getpass').getuser(),
                    '--format=%i|%k|%T|%j|%P'])
    if observed.returncode:
        return {'ready':False,'reason':'new_queue_observation_retry'}
    queue = {v[0]:v for line in observed.stdout.splitlines()
             if len(v:=[x.strip() for x in line.split('|')])>=5}
    verified = []
    for job in jobs:
        row = queue.get(str(job['job_id']))
        if (not row or row[1]!=snapshot['marker'] or row[2]!='PENDING' or row[4]!='hpc_gpu'
                or not re.fullmatch(re.escape(snapshot['marker'])+'-s'+str(job['slot'])+'-a[1-9][0-9]*',row[3])):
            return {'ready':False,'reason':'three_owned_pending_jobs_not_yet_visible'}
        detail = run(['scontrol','show','job','-o',str(job['job_id'])])
        if detail.returncode: return {'ready':False,'reason':'new_job_resource_observation_retry'}
        fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)',detail.stdout))
        tres = fields.get('TRES',fields.get('ReqTRES',''))
        gpu = fields.get('TresPerNode',fields.get('Gres',''))
        cpus = fields.get('NumCPUs')
        mem = fields.get('MinMemoryNode')
        if mem is None:
            match = re.search(r'(?:^|,)mem=([^,]+)',tres)
            mem = match.group(1) if match else ''
        observed_resources = {'job_id':str(job['job_id']), 'Partition':fields.get('Partition'),
            'NumNodes':fields.get('NumNodes'), 'NumCPUs':cpus, 'memory':mem,
            'MinMemoryNode':fields.get('MinMemoryNode'), 'TRES':tres, 'TresPerNode':gpu}
        expected_resources = {'partition':'hpc_gpu', 'nodes':1, 'cpus':12,
                              'memory':job['memory'], 'gpus':1}
        resource_context = '; observed='+json.dumps(observed_resources,sort_keys=True)+\
                           '; expected='+json.dumps(expected_resources,sort_keys=True)
        # Slurm 20.11 displays a pending fixed one-node request as its exact
        # minimum-maximum range. A larger or open range is not a one-node job.
        if fields.get('Partition')!='hpc_gpu' or fields.get('NumNodes') not in ('1','1-1') or cpus!='12':
            raise ValueError('Replacement job partition/node/CPU differs from requested resources'+resource_context)
        if (fields.get('Reason') not in ('JobHeldUser','JobHeldAdmin')
                and not (fields.get('Reason') is None and fields.get('Priority')=='0')):
            return {'ready':False,'reason':'replacement_hold_not_confirmed'}
        if memory_mib(mem)!=memory_mib(job['memory']):
            raise ValueError('Replacement job memory differs from requested resources'+resource_context)
        if not (re.search(r'gpu(?::[^,=]+)*:1(?:,|$)',gpu) or re.search(r'gres/gpu=1(?:,|$)',tres)):
            raise ValueError('Replacement job must request exactly one GPU'+resource_context)
        verified.append({'slot':job['slot'],'job_id':str(job['job_id']),'memory':job['memory'],
                         'cpus':12,'gpus':1,'partition':'hpc_gpu'})
    return {'ready':True,'jobs':verified,'marker':snapshot['marker']}


def legacy_target(root, launch, retired):
    """Resolve a just-finished legacy sbatch under its now-retired launch lock."""
    if not retired.get('retirement_complete') and retired.get('status')!='READY':
        return None
    state=read(root/'.paper3-hpc/direct-train/state.json',{}) or {}
    target=str(state.get('current_job_id') or launch['old_job'])
    comments={str(row['job_id']):(row.get('submission') or {}).get('comment')
              for row in state.get('history',[]) if row.get('job_id')}
    comment=comments.get(target)
    if target==str(launch['old_job']) and not comment:
        comment='paper3-direct-debd49e881f04859ba643afa8a68a2c8'
    intent=state.get('intent')
    if intent:
        wanted=intent.get('comment')
        if not re.fullmatch(r'paper3-direct-[0-9a-f]{32}',str(wanted)):
            raise ValueError('Unrecognized legacy submission identity')
        result=command(['squeue','--array','--noheader','--user',__import__('getpass').getuser(),
                        '--format=%i|%k|%j'])
        if result.returncode:return None
        matches=[]
        for line in result.stdout.splitlines():
            row=[x.strip() for x in line.split('|')]
            if len(row)>=3 and row[1]==wanted and row[2]=='paper3-train-direct':matches.append(row[0])
        if not matches:
            start=str(intent.get('started_utc','')).split('+')[0]
            if not start:return None
            result=command(['sacct','-X','--noheader','--parsable2','--user',__import__('getpass').getuser(),
                '--starttime='+start,'--format=JobIDRaw,Comment%128,JobName%128'])
            if result.returncode:return None
            for line in result.stdout.splitlines():
                row=[x.strip() for x in line.split('|')]
                if len(row)>=3 and row[1]==wanted and row[2]=='paper3-train-direct' and row[0].isdigit():matches.append(row[0])
        if len(set(matches))!=1:return None
        target,comment=matches[0],wanted
    if not target.isdigit() or not re.fullmatch(r'paper3-direct-[0-9a-f]{32}',str(comment)):
        return None
    return {'job_id':target,'comment':comment,'requested_old_job':str(launch['old_job'])}


def reconcile_claims(runtime, snapshot):
    terminal = set(snapshot.get('terminal_job_ids', []))
    if not terminal: return
    with lock(runtime/'.queue.lock'):
        claims = read(runtime/'claims.json', {}) or {}
        changed = False
        for key, claim in list(claims.items()):
            if str(claim.get('job_id')) not in terminal: continue
            digest = hashlib.sha256(key.encode()).hexdigest()
            # The worker owns this additional lock until its child exits.
            try:
                with lock(runtime/'action-locks'/(digest+'.lock'),blocking=False):
                    claims.pop(key); changed=True
                    emit('TERMINAL_WORKER_CLAIM_RELEASED',key=key,job_id=claim.get('job_id'))
            except BlockingIOError:
                continue
        if changed: put(runtime/'claims.json',claims)


def publish_progress(root, runtime, frontier, snapshot, state, *, handoff_status=None):
    tasks = state.setdefault('tasks', {})
    original = read(root/'runs/training_campaign_v1/campaign_progress.json',{}) or {}
    for row in original.get('actions', []):
        if row.get('key') not in tasks:
            tasks[row['key']]={'key':row['key'],'kind':row.get('kind'), 'status':'COMPLETE',
                'expected_receipt':(row.get('receipt') or {}).get('path'), 'historical_reused':True}
    for action in frontier.get('phase_actions', []):
        record = tasks.setdefault(action['key'],{})
        record.update(action)
        record['status']='COMPLETE' if action_complete(root,action) else 'WAITING_DEPENDENCY_OR_SLOT'
        record['progress_path']=str(Path(action['expected_receipt']).parent/'progress.json')
    backoff=read(runtime/'action-backoff.json',{}) or {}
    for key,value in backoff.items():
        if key in tasks and tasks[key].get('status')!='COMPLETE':
            tasks[key]['error_type']=value.get('error')
            tasks[key]['retry_at']=value.get('retry_after')
            if value.get('retry_after',0)>time.time():tasks[key]['status']='RETRY_WAIT'
    claims = read(runtime/'claims.json',{}) or {}
    for key,claim in claims.items():
        if key in tasks and tasks[key].get('status')!='COMPLETE':
            tasks[key].update(status='RUNNING',slot=claim.get('slot'),job_id=claim.get('job_id'))
    status = ('ALL_ORIGINAL_WORK_COMPLETE' if frontier.get('all_complete') else
              'WAITING_SCIENTIFIC_INPUT' if frontier.get('blocked') else
              'RUNNING_OR_QUEUED' if handoff_status is None else handoff_status)
    progress = {'schema_version':'paper3-r1-parallel-progress-v1','experiment':'paper3-round1',
        'status':status, 'phase':frontier.get('phase'),
        'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'mainline_complete':frontier.get('mainline_complete',False),
        'all_original_complete':frontier.get('all_complete',False),
        'all_complete':frontier.get('all_complete',False),
        'phase_completed':frontier.get('phase_completed'), 'phase_total':frontier.get('phase_total'),
        'completed_by_kind':dict(Counter(v.get('kind') for v in tasks.values() if v.get('status')=='COMPLETE')),
        'planned_counts':{'meta_engineering':5,'meta_screening':45,'meta_selection':5,
            'calibration_profile':14,'development_screening':118,'development_selection':14,
            'meta_confirmation':10,'development_confirmation':28,'internal_check':1,'extension_arms':2},
        'blocked':frontier.get('blocked',[]),'tasks':list(tasks.values()),'jobs':snapshot.get('jobs',[]),
        'requested_resources':{'partition':'hpc_gpu','gpu_model':'A100-PCIe-40GB','jobs':3,'nodes_per_job':1,
            'gpus_per_job':1,'gpus_total':3,'cpus_per_job':12,'cpus_total':36,
            'memories':['43G','43G','42G'],'memory_total':'128G'},
        'scientific_outputs_reused_in_place':True,'original_output':'runs/training_campaign_v1',
        'cumulative_12_hour_limit':False}
    put(runtime/'progress.json',progress)
    state.update(status=status,phase=frontier.get('phase'),updated_utc=progress['updated_utc'])
    put(runtime/'state.json',state)
    return progress


def controller_loop(root, runtime, launch, *, stop_requested=lambda:False):
    with lock(runtime/'controller.lock',blocking=False):
        scheduler = ThreeJobScheduler(runtime,launch,emit=emit)
        state = read(runtime/'state.json',{}) or {}
        handoff_ready = bool(read(runtime/'HANDOFF_READY.json',{}))
        legacy_retired = False
        while not stop_requested():
            try:
                frontier = plan_ready(root)
                scheduler.launch['submit_hold']=not handoff_ready
                desired = (False if frontier.get('all_complete') else True if not handoff_ready else
                           [1,2] if frontier.get('mainline_complete') else True)
                snapshot = scheduler.step(keep_workers=desired)
                reconcile_claims(runtime,snapshot)
                if frontier.get('all_complete') and not handoff_ready:
                    with lock(runtime/'.queue.lock'):
                        put(runtime/'queue.json',{'status':'COMPLETE','ready_actions':[]})
                    publish_progress(root,runtime,frontier,snapshot,state)
                    emit('ALL_ORIGINAL_R1_WORK_ALREADY_COMPLETE',new_scientific_tasks_started=0)
                    return 0
                if not handoff_ready:
                    queued = confirm_three_held(snapshot)
                    publish_progress(root,runtime,frontier,snapshot,state,handoff_status='WAITING_THREE_REPLACEMENT_JOBS')
                    if not queued['ready']:
                        emit('R1_REPLACEMENTS_QUEUE_WAIT',reason=queued['reason'])
                        time.sleep(15);continue
                    put(runtime/'replacement-queue-verified.json',queued)
                    retired = retire_legacy_monitor(root,Path(launch['direct_package']),runtime,emit=emit)
                    target=legacy_target(root,launch,retired)
                    if target is None:
                        emit('LEGACY_MANAGER_RETIREMENT_WAIT',status=retired.get('status'),
                             current_job=retired.get('existing_current_job_id'))
                        time.sleep(15);continue
                    legacy_retired=True
                    put(runtime/'legacy-handoff-target.json',target)
                    saved = checkpoint_handoff_step(root,runtime/'legacy-handoffs'/target['job_id'],
                        old_job=target['job_id'],expected_comment=target['comment'],emit=emit,
                        replacement_queue_proof=queued)
                    publish_progress(root,runtime,frontier,snapshot,state,handoff_status=saved['status'])
                    if saved['status']!='READY': time.sleep(10);continue
                    put(runtime/'HANDOFF_READY.json',{'status':'READY','old_job':target['job_id'],
                        'checkpoint':saved.get('quiescence',{}).get('checkpoint'),
                        'saved_utc':time.time(),'old_outputs_changed_by_migration':False})
                    # Capture actual completion at the boundary, after native checkpoint close.
                    put(runtime/'migration-progress-snapshot.json',{
                        'campaign':read(root/'runs/training_campaign_v1/campaign_progress.json',{}),
                        'frontier':plan_ready(root),'handoff':saved})
                    handoff_ready=True
                elif not legacy_retired:
                    retired=retire_legacy_monitor(root,Path(launch['direct_package']),runtime,emit=emit)
                    if legacy_target(root,launch,retired) is None:
                        emit('LEGACY_MANAGER_RETIREMENT_WAIT',status=retired.get('status'))
                        time.sleep(15);continue
                    legacy_retired=True
                frontier=plan_ready(root)
                with lock(runtime/'.queue.lock'):
                    put(runtime/'queue.json',{'status':'COMPLETE' if frontier.get('all_complete') else 'RUNNING',
                        'phase':frontier['phase'],'ready_actions':frontier.get('ready_actions',[]),
                        'retired_slots':[0] if frontier.get('mainline_complete') else [],
                        'updated_utc':time.time()})
                # Release only this operational marker's acknowledged jobs. Repeating is harmless.
                released=read(runtime/'released-jobs.json',{}) or {}
                for job in snapshot.get('jobs',[]):
                    if (job.get('job_id') and job.get('slurm_state')=='PENDING' and job.get('held_requested')
                            and str(job['job_id']) not in released):
                        result=command(['scontrol','release',str(job['job_id'])])
                        emit('REPLACEMENT_RELEASE_REQUESTED',slot=job['slot'],job_id=job['job_id'],returncode=result.returncode)
                        if result.returncode==0:
                            released[str(job['job_id'])]={'released_utc':time.time(),'slot':job['slot']}
                            put(runtime/'released-jobs.json',released)
                progress=publish_progress(root,runtime,frontier,snapshot,state)
                emit('R1_PARALLEL_PROGRESS',phase=progress['phase'],status=progress['status'],
                     completed_by_kind=progress['completed_by_kind'],jobs=progress['jobs'])
                if frontier.get('all_complete') and snapshot.get('inflight_jobs',0)==0:
                    emit('ALL_ORIGINAL_R1_WORK_COMPLETE');return 0
            except Exception as error:
                emit('R1_COORDINATOR_RETRY',error_type=type(error).__name__,reason=str(error),
                     original_results_preserved=True)
            time.sleep(30)
    return 75


def is_locked(path):
    try:
        with lock(path,blocking=False): return False
    except BlockingIOError: return True


def prepare(args):
    root=Path(args.root).expanduser().resolve()
    runtime=root/OPERATIONAL
    frontier=plan_ready(root)
    measurements=inspect_primary_measurements(root)
    context=discover_context(root,args.old_job,args.direct_package,args.payload_manifest)
    worker=[args.python,'-I','-S','-B','-u',str(PACKAGE/'worker.py'), '--root',str(root),
        '--runtime-dir',str(runtime),'--slot','{slot}','--package',str(PACKAGE),
        '--payload-manifest',context['payload_manifest'],'--direct-package',context['direct_package'],
        '--prior-job','769863','--allocation-seconds','442800']
    launch={**context,'root':str(root),'old_job':str(args.old_job),'python':args.python,
        'package':str(PACKAGE),'worker_command':worker,'partition':'hpc_gpu','account':__import__('getpass').getuser(),
        'cpus':12,'gres':'gpu:1','memories':['43G','43G','42G'],'time_limit':'5-03:00:00','submit_hold':True}
    launch['primary_measurement_inspection']=measurements
    emit('R1_PARALLEL_PREPARED',phase=frontier['phase'],phase_completed=frontier.get('phase_completed'),
         phase_total=frontier.get('phase_total'),blocked=frontier.get('blocked'),old_job=args.old_job,
         partition='hpc_gpu',jobs=3,gpus=3,cpus_total=36,memory_total='128G',
         scientific_results_reused=True,no_full_test_or_data_rescan=True,
         prior_primary_measurements_on_other_hardware=len(measurements['incompatible_measurements']))
    return root,runtime,frontier,launch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=DEFAULT_ROOT)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--old-job',default='770043')
    parser.add_argument('--direct-package')
    parser.add_argument('--payload-manifest')
    parser.add_argument('--credentials')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--controller',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.controller:
        runtime=Path(args.root)/OPERATIONAL
        return controller_loop(Path(args.root),runtime,read(runtime/'launch.json'))
    root,runtime,frontier,launch=prepare(args)
    if not args.execute:return 0
    runtime.mkdir(parents=True,exist_ok=True)
    with lock(runtime/'startup.lock'):
        put(runtime/'launch.json',launch)
        if not (runtime/'queue.json').exists():put(runtime/'queue.json',{'status':'WAITING_HANDOFF','ready_actions':[]})
        put(runtime/'prepared-frontier.json',frontier)
        started={}
        processes=[('controller',runtime/'controller.lock',[args.python,'-I','-S','-B','-u',str(PACKAGE/'START.py'),
            '--controller','--root',str(root)]),
            ('monitor',runtime/'monitor.lock',[args.python,'-I','-S','-B','-u',str(PACKAGE/'monitor.py'),
             '--root',str(root),'--operational',str(runtime),'--interval','180'])]
        if args.credentials:processes[1][2].extend(['--credentials',args.credentials])
        for name,owned,argv in processes:
            if is_locked(owned):started[name]='ALREADY_RUNNING';continue
            with (runtime/(name+'.log')).open('ab',buffering=0) as log:
                process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                    start_new_session=True,close_fds=True)
            started[name]=process.pid
        emit('R1_PARALLEL_BACKGROUND_STARTED',processes=started,controller_log=str(runtime/'controller.log'),
            monitor_log=str(runtime/'monitor.log'),progress=str(runtime/'progress.json'),
            note='Background PIDs are startup acknowledgements; actual Slurm submissions are in scheduler receipts.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
