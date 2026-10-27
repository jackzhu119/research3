"""Operational-only, in-memory repair for the immutable R11 runtime."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

CHUNK_BYTES=1024*1024
DEFAULT_RECIPE='97e4cadb17278532f530d5c40ec7c8fbf928ebe308f60e5d9ada5eec7a0a864b'


def bounded_read(path,default=None,max_bytes=16*1024*1024):
    """Read actual bytes, never allocate the declared 16/64MiB ceiling upfront.

    Existing malformed, absent and over-limit records retain the original
    default semantics. Runtime callers still refuse an invalid state; nothing
    turns a corrupted tasks/scheduler record into a fresh campaign.
    """
    try:
        with Path(path).open('rb') as handle:
            size=os.fstat(handle.fileno()).st_size
            if size>max_bytes or size<0:return default
            parts=[];count=0
            while count<=size:
                block=handle.read(min(CHUNK_BYTES,size+1-count))
                if not block:break
                parts.append(block);count+=len(block)
            if count>size or count>max_bytes:return default
        return json.loads(b''.join(parts))
    except (OSError,ValueError):return default


def install_read_patch(common):
    old=common.read
    if old is bounded_read:return {'status':'ALREADY_PATCHED'}
    common.read=bounded_read;aliases=0
    for name,module in list(sys.modules.items()):
        if name.startswith('paper3r11.') and module is not None and getattr(module,'read',None) is old:
            setattr(module,'read',bounded_read);aliases+=1
    return {'status':'IN_MEMORY_READ_PATCH_INSTALLED','aliases':aliases,'source_files_changed':False}


def install_plan_cache(common,path,plan):
    """One already verified object for this operational process and exact plan."""
    path=Path(path).resolve();old=common.load_plan
    original=getattr(old,'_repair_original_loader',old)
    expected=common.sha(path)
    def signature():
        s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    observed=[signature()]
    def cached(candidate,*,validate_science_stage=True):
        if Path(candidate).expanduser().resolve()!=path:
            return original(candidate,validate_science_stage=validate_science_stage)
        current=signature()
        if current!=observed[0]:
            if common.sha(path)!=expected:raise ValueError('ORIGINAL_AUTHORIZED_PLAN_CHANGED')
            observed[0]=current
        return path,plan
    cached._repair_original_loader=original
    common.load_plan=cached
    for name,module in list(sys.modules.items()):
        if name.startswith('paper3r11.') and module is not None and getattr(module,'load_plan',None) is old:
            setattr(module,'load_plan',cached)
    return {'status':'VERIFIED_OPERATIONAL_PLAN_OBJECT_REUSED','plan_sha256':expected,'source_file_validation':'once_at_startup'}


def package_from_plan(path):
    """Extract the early sorted package token without retaining a second full plan."""
    decoder=json.JSONDecoder();prefix=b''
    with Path(path).open('rb') as handle:
        while len(prefix)<1024*1024:
            block=handle.read(65536)
            if not block:break
            prefix+=block
            text=prefix.decode('utf-8',errors='ignore')
            match=re.search(r'"package"\s*:\s*("(?:[^"\\]|\\.)*")',text)
            if match:
                value=decoder.decode(match.group(1))
                if isinstance(value,str) and Path(value).is_absolute():return Path(value).resolve()
    raise ValueError('ORIGINAL_PACKAGE_ARGUMENT_REQUIRED')


def load_original(package,plan_path,*,expected_recipe=DEFAULT_RECIPE):
    """Import only the original stdlib operational modules; patch before reads."""
    path=Path(plan_path).expanduser().resolve();package=Path(package).expanduser().resolve() if package else package_from_plan(path)
    if not (package/'paper3r11/platform/common.py').is_file():raise ValueError('ORIGINAL_PACKAGE_NOT_FOUND')
    existing=sys.modules.get('paper3r11')
    if existing is not None and Path(existing.__file__).resolve().parent!=package/'paper3r11':
        raise ValueError('ANOTHER_R11_PACKAGE_ALREADY_IMPORTED')
    if str(package) not in sys.path:sys.path.insert(0,str(package))
    common=importlib.import_module('paper3r11.platform.common');patched=install_read_patch(common)
    loader=getattr(common.load_plan,'_repair_original_loader',common.load_plan)
    _,plan=loader(path)
    if Path(plan['package']).resolve()!=package:raise ValueError('ORIGINAL_PACKAGE_PLAN_BINDING_DIFFERS')
    if expected_recipe and plan['recipe_sha256']!=expected_recipe:raise ValueError('ORIGINAL_R11_RECIPE_DIFFERS')
    runtime=Path(plan['output'])/'runtime'
    if not runtime.is_dir():raise ValueError('ORIGINAL_STARTED_RUNTIME_REQUIRED')
    header={k:plan[k] for k in ('root','output','package','recipe_sha256','stage','resources')}
    header['python']=plan.get('python',sys.executable)
    header.update(total_tasks=len(plan['tasks']),plan_sha256=common.sha(path),plan_path=str(path))
    cache=install_plan_cache(common,path,plan)
    # The cache and controller/paper runtime share this same verified object.
    del plan
    rt=importlib.import_module('paper3r11.platform.runtime')
    return {'common':common,'runtime_module':rt,'package':package,'path':path,'runtime':runtime,'header':header,'read_patch':patched,'plan_cache':cache}


def scheduler_identity(runtime,recipe):
    import hashlib
    return 'paper3r11p-'+hashlib.sha256(json.dumps([str(Path(runtime).resolve()),recipe],separators=(',',':')).encode()).hexdigest()[:16]


def validate_operational_history(env):
    h=env['header'];runtime=env['runtime'];read=bounded_read
    authorization=read(runtime/'SUBMIT_AUTHORIZATION.json',{})
    if (authorization.get('explicit_submit') is not True or authorization.get('plan_sha256')!=h['plan_sha256']
            or authorization.get('recipe_sha256')!=h['recipe_sha256']):raise ValueError('ORIGINAL_EXACT_SUBMIT_AUTHORIZATION_REQUIRED')
    state=read(runtime/'scheduler-state.json',None)
    marker=scheduler_identity(runtime,h['recipe_sha256'])
    if not isinstance(state,dict) or state.get('marker')!=marker:raise ValueError('ORIGINAL_SCHEDULER_HISTORY_REQUIRED')
    slots=state.get('slots',{})
    if set(slots)!={'0','1','2','3'}:raise ValueError('ORIGINAL_FOUR_SLOT_HISTORY_REQUIRED')
    jobs=[]
    for slot in range(4):
        attempts=slots[str(slot)].get('attempts')
        if not isinstance(attempts,list) or not attempts:raise ValueError('ORIGINAL_ALLOCATION_INTENTS_REQUIRED')
        for attempt in attempts:
            if attempt.get('marker')!=marker:raise ValueError('SCHEDULER_ATTEMPT_MARKER_DIFFERS')
            job=attempt.get('job_id')
            if job is not None:
                if not str(job).isdigit():raise ValueError('SCHEDULER_JOB_ID_INVALID')
                jobs.append(str(job))
    if not jobs:raise ValueError('EXISTING_SCIENTIFIC_ALLOCATIONS_REQUIRED')
    return {'marker':marker,'known_job_ids':sorted(set(jobs)),'prior_authorization_verified':True,
            'tasks_reset':False,'scientific_results_changed':False}


def controller_loop(plan_path,work,*,package=None,submit=False,runner=None,stop=None,once=False,expected_recipe=DEFAULT_RECIPE,launch_token=None):
    env=load_original(package,plan_path,expected_recipe=expected_recipe);history=validate_operational_history(env)
    rt,common=env['runtime_module'],env['common'];work=Path(work);work.mkdir(parents=True,exist_ok=True)
    ctx=rt.context(env['path']);rt._authorization(ctx);stop=stop or threading.Event()
    # Preserve the original exact worker command, Python and authorized marker.
    launch=bounded_read(env['runtime']/'launch.json',{})
    if (launch.get('plan_sha256')!=env['header']['plan_sha256'] or launch.get('campaign_identity')!=env['header']['recipe_sha256']
            or Path(launch.get('runtime','')).resolve()!=env['runtime'].resolve()):raise ValueError('ORIGINAL_LAUNCH_HISTORY_DIFFERS')
    ctx['launch']=launch
    with common.locked(env['runtime']/'controller.lock',nonblocking=True):
        previous=bounded_read(env['runtime']/'controller-error.json',{})
        if previous and not (work/'original-controller-error.json').exists():common.atomic(work/'original-controller-error.json',previous)
        previous_process=bounded_read(env['runtime']/'controller-process.json',{})
        if previous_process and not (work/'original-controller-process.json').exists():common.atomic(work/'original-controller-process.json',previous_process)
        common.atomic(env['runtime']/'controller-process.json',dict(kind='controller',pid=os.getpid(),started_utc=common.utc(),
            operational_repair=True,source_files_changed=False,submit_reaffirmed=bool(submit)))
        common.atomic(env['runtime']/'controller-ready.json',dict(status='BOOTSTRAP_READY',pid=os.getpid(),
            recipe_sha256=env['header']['recipe_sha256'],updated_utc=common.utc(),first_slurm_observation_performed=False,
            training_progress_claimed=False,operational_repair=True))
        scheduler=rt.FourJobScheduler(env['runtime'],launch,runner=runner,emit=rt.emit)
        # Existing intents must first be reconciled with queue/accounting. This
        # cycle cannot submit even when every slot happens to be unallocated.
        first=scheduler.step(keep_workers=False)
        common.atomic(work/'first-observation.json',dict(summary=first,history=history,first_cycle_submit_allowed=False,updated_utc=common.utc()))
        common.atomic(work/'controller-ready.json',dict(status='READY',pid=os.getpid(),
            recipe_sha256=env['header']['recipe_sha256'],memory_patch_loaded=True,
            launch_token=launch_token,first_slurm_observation_completed=True,training_progress_claimed=False,updated_utc=common.utc()))
        common.atomic(env['runtime']/'controller-error.json',{})
        stamp=rt.file_signature(env['path'])
        while not stop.is_set():
            current=rt.file_signature(env['path'])
            if current!=stamp:
                if common.sha(env['path'])!=launch['plan_sha256']:raise ValueError('AUTHORIZED_PLAN_CHANGED')
                stamp=current
            snapshot=rt.status(env['path'],plan=ctx['plan'])
            remaining=snapshot['runnable_remaining_tasks']
            summary=first if first is not None else scheduler.step(keep_workers=bool(remaining) and bool(submit))
            first=None
            result=dict(summary,task_status=snapshot,updated_utc=common.utc(),operational_repair=True,
                        automatic_replacement_authorized=bool(submit),scientific_sources_changed=False)
            common.atomic(env['runtime']/'controller-status.json',result);common.atomic(work/'controller-status.json',result)
            if once:return result
            if summary.get('controller_status','').startswith('BLOCKED_') and not summary.get('inflight_jobs'):return result
            if not remaining and summary.get('controller_status')=='FINISHED':return result
            stop.wait(float(launch.get('poll_seconds',30)))
    return {'status':'STOPPED'}
