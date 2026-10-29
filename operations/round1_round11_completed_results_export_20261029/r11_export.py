"""Export all currently sealed R11 results, reusing the original upload ledger.

This module imports only the installed stdlib operational package. It does not
run preparation, a report builder, a scheduler, or a scientific worker. Every
call is designed for one disposable metadata process owned by the outer driver.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
import urllib.parse

HERE=Path(__file__).resolve().parent
PREFIX='results/paper3/round-11'
LIMIT=256*1024*1024
_ABSENT=object()


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def emit(event,**fields):print(json.dumps(dict(event=event,utc=utc(),**fields),ensure_ascii=False),flush=True)


def read(path,default=None,max_bytes=LIMIT):
    try:
        with Path(path).open('rb') as handle:
            size=os.fstat(handle.fileno()).st_size
            if size<0 or size>max_bytes:raise ValueError('R11_EXPORT_METADATA_SIZE_EXCEEDED')
            parts=[];count=0
            while count<=size:
                block=handle.read(min(1024*1024,size+1-count))
                if not block:break
                parts.append(block);count+=len(block)
        if count!=size or count>max_bytes:raise ValueError('R11_EXPORT_METADATA_CHANGED_WHILE_READING')
        return json.loads(b''.join(parts))
    except FileNotFoundError:return default


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.r11-export-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def atomic(path,value):atomic_bytes(path,encoded(value))


@contextmanager
def lock(path,blocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));yield handle


def signature(path):
    try:
        s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    except FileNotFoundError:return None


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def argument(argv,name):
    if not isinstance(argv,list) or argv.count(name)!=1:return None
    at=argv.index(name);return argv[at+1] if at+1<len(argv) else None


def plan_path(args):
    value=getattr(args,'r11_plan',None) or getattr(args,'plan',None) or Path(args.root)/'runs/r11_campaign/plan.json'
    return Path(value).expanduser().resolve()


def _package(path):
    # The signed plan is sorted and package is near its header; never retain a
    # second complete 10k-task plan just to select the existing module path.
    with Path(path).open('rb') as handle:
        raw=handle.read(1024*1024)
    match=re.search(rb'"package"\s*:\s*("(?:[^"\\]|\\.)*")',raw)
    if not match:raise ValueError('R11_ORIGINAL_PACKAGE_NOT_DECLARED')
    return Path(json.loads(match.group(1))).expanduser().resolve()


def load_environment(args):
    path=plan_path(args);package=_package(path)
    if not (package/'paper3r11/platform/common.py').is_file():raise ValueError('R11_ORIGINAL_PACKAGE_MISSING')
    current=sys.modules.get('paper3r11')
    if current is not None and Path(current.__file__).resolve().parent!=package/'paper3r11':
        raise ValueError('ANOTHER_R11_PACKAGE_IS_ALREADY_IMPORTED')
    if str(package) not in sys.path:sys.path.insert(0,str(package))
    common=importlib.import_module('paper3r11.platform.common')
    original_read,original_atomic,original_bytes=common.read,common.atomic,common.atomic_bytes
    common.read=read;common.atomic=atomic;common.atomic_bytes=atomic_bytes
    # A fresh subprocess owns these aliases; no installed source is modified.
    for name,module in list(sys.modules.items()):
        if name.startswith('paper3r11.') and module is not None:
            if getattr(module,'read',None) is original_read:module.read=read
            if getattr(module,'atomic',None) is original_atomic:module.atomic=atomic
            if getattr(module,'atomic_bytes',None) is original_bytes:module.atomic_bytes=atomic_bytes
    _,plan=common.load_plan(path,validate_science_stage=False)
    root=Path(args.root).expanduser().resolve()
    if Path(plan['root']).resolve()!=root or Path(plan['package']).resolve()!=package:
        raise ValueError('R11_PLAN_ROOT_OR_PACKAGE_DIFFERS')
    output=Path(plan['output']).resolve();runtime=output/'runtime'
    if not output.is_relative_to(root/'runs') or not runtime.is_dir():
        raise ValueError('R11_EXISTING_OUTPUT_REQUIRED')
    u=importlib.import_module('paper3r11.platform.uploads')
    rt=importlib.import_module('paper3r11.platform.runtime')
    # Never re-read the same plan repeatedly inside a paper/one-group cycle.
    original_loader=common.load_plan;stamp=signature(path);plan_sha=sha(path)
    def same_plan(candidate,**kwargs):
        if Path(candidate).expanduser().resolve()!=path:return original_loader(candidate,**kwargs)
        if signature(path)!=stamp and sha(path)!=plan_sha:raise ValueError('R11_PLAN_CHANGED_DURING_EXPORT')
        return path,plan
    u.load_plan=same_plan
    work=Path(args.work)/'r11';work.mkdir(parents=True,exist_ok=True)
    return dict(path=path,package=package,plan=plan,root=root,output=output,runtime=runtime,
                u=u,rt=rt,common=common,work=work,plan_sha256=plan_sha)


def completion_rows(env):
    """Freeze sealed metadata, and refresh only when the task-state file changes."""
    path=env['work']/'completion-index.json';source=env['runtime']/'tasks.json'
    with lock(env['work']/'completion-index.lock',blocking=True):
        stamp=signature(source);saved=read(path,{}) or {}
        if (saved.get('plan_sha256')==env['plan_sha256']
            and saved.get('recipe_sha256')==env['plan']['recipe_sha256']
            and saved.get('task_state_signature')==stamp and isinstance(saved.get('rows'),dict)):
            rows=saved['rows']
        else:
            native=env['rt'].completed_task_rows(env['plan'])
            if signature(source)!=stamp:raise RuntimeError('R11_COMPLETED_STATE_CHANGED_RETRY')
            rows={key:{field:row[field] for field in ('task_identity','result_sha256','completion_sha256')}
                  for key,row in native.items()}
            atomic(path,dict(plan_sha256=env['plan_sha256'],recipe_sha256=env['plan']['recipe_sha256'],
                task_state_signature=stamp,rows=rows,captured_utc=utc()))
            emit('R11_COMPLETED_RECEIPTS_FROZEN',available_completed_tasks=len(rows),
                 planned_tasks=len(env['plan']['tasks']),scientific_outputs_rehashed=False)
        identities=env['rt'].plan_index(env['plan'])[1]
        for key,row in rows.items():
            if (key not in identities or row.get('task_identity')!=identities[key]
                or any(not re.fullmatch(r'[0-9a-f]{64}',str(row.get(name,'')))
                       for name in ('result_sha256','completion_sha256'))):
                raise ValueError('R11_FROZEN_COMPLETION_METADATA_DIFFERS')
    return rows


def _client(args,env,work,channel):
    # Native publish_snapshot deliberately requires its own VerifiedGitHub
    # class. FullVerifiedGitHub always downloads each entire archive part.
    u=env['u'];factory=u.FullVerifiedGitHub if channel=='full' else u.VerifiedGitHub
    return factory(u.discover_token(getattr(args,'credentials',None)),emit,
        branch='paper3-results',cache_path=Path(work)/'git-blobs.json')


def _remote_json(gh,path):
    item=gh.api(gh.prefix+'/contents/'+urllib.parse.quote(path,safe='/')+'?ref='+urllib.parse.quote(gh.branch,safe='/'),allow404=True)
    if not item:return {}
    if item.get('encoding')!='base64' or not item.get('content'):
        if not item.get('sha'):raise ValueError('R11_REMOTE_INDEX_REPRESENTATION')
        item=gh.api(gh.prefix+'/git/blobs/'+item['sha'])
    return json.loads(base64.b64decode(item['content']))


def _valid_paper_group(group,identity):
    receipt=group.get('receipt',{}) if isinstance(group,dict) else {}
    return (group.get('identity')==identity
        and receipt.get('remote_complete_blob_GET_SHA256_verified') is True
        and receipt.get('collection_status')=='COMPLETE')


def _publish_summary(gh,env,channel,rows,groups,reports,*,complete,delta_group=None):
    plan=env['plan'];keys=sorted(rows);key_sha=sha_bytes(encoded(keys))
    kind='paper' if channel=='paper' else 'full'
    previous=read(env['work']/(kind+'-last-published.json'),{}) or {}
    prior=previous.get('receipt',{})
    # The old groups ledger is authoritative and is persisted after each task.
    # Publish its full baseline once, then only the new task record per cycle.
    # The final pointer always embeds the complete available task set, so users
    # never need to traverse this progress chain to restore finished output.
    incremental=(channel=='full' and not complete and delta_group in groups
        and prior.get('status')=='UPLOADING_AVAILABLE_COMPLETED_OUTPUT'
        and prior.get('completed_task_keys_sha256')==key_sha
        and prior.get('uploaded_completed_tasks')==len(groups)-1 and bool(prior.get('manifest')))
    records={delta_group:groups[delta_group]} if incremental else groups
    payload=dict(schema_version='paper3-r11-current-completed-export-v2',round=11,channel=channel,
        stage=plan['stage'],recipe_sha256=plan['recipe_sha256'],
        observed_completed_tasks=len(rows),uploaded_completed_tasks=len(groups),planned_tasks=len(plan['tasks']),
        completed_task_keys_sha256=key_sha,tasks=records,reports=reports,
        task_index_mode='DELTA' if incremental else 'FULL',
        manifest_includes_all_uploaded_task_records=not incremental,
        previous_manifest=prior['manifest'] if incremental else None,
        available_completed_output_complete=bool(complete),all_planned_task_receipts_complete=len(rows)==len(plan['tasks']),
        scientific_completion_claimed=False,
        remote_verification_scope='Original upload whole-GET/SHA receipts retained; no repeated download of already verified groups.',
        remote_full_GET_SHA256_verified=channel=='full',
        remote_complete_blob_GET_SHA256_verified=channel=='paper',
        original_scientific_files_modified=False,training_controlled=False)
    if channel=='full':
        payload['ancillary_snapshot_scope']='Exact existing plan/reports and regular .out/.err/.log files under runtime/logs only. Job logs are an existing snapshot, not sealed scientific task evidence; manager/uploader logs are excluded.'
    snapshot=sha_bytes(encoded(payload))
    target=env['u'].stage_prefix(plan)+'/completed-export/'+kind+'/'+snapshot+'/manifest.json'
    pointer=dict(schema_version='paper3-r11-current-completed-pointer-v1',round=11,channel=kind,
        status='AVAILABLE_COMPLETED_OUTPUT_UPLOADED' if complete else 'UPLOADING_AVAILABLE_COMPLETED_OUTPUT',
        manifest=target,snapshot_id=snapshot,recipe_sha256=plan['recipe_sha256'],
        observed_completed_tasks=len(rows),uploaded_completed_tasks=len(groups),planned_tasks=len(plan['tasks']),
        completed_task_keys_sha256=key_sha,task_index_mode=payload['task_index_mode'],
        manifest_includes_all_uploaded_task_records=not incremental,
        available_completed_output_complete=bool(complete),all_planned_task_receipts_complete=len(rows)==len(plan['tasks']),
        scientific_completion_claimed=False,remote_full_GET_SHA256_verified=channel=='full',
        remote_complete_blob_GET_SHA256_verified=channel=='paper',updated_utc=utc(),training_controlled=False)
    paths={target:encoded(payload),PREFIX+'/COMPLETED_'+kind.upper()+'_LATEST.json':encoded(pointer)}
    if channel=='full':paths[PREFIX+'/BACKFILL_LATEST.json']=encoded(pointer)
    if previous.get('snapshot_id')==snapshot:return dict(previous['receipt'],reused_verified_index=True)
    commit=gh.git_publish(paths,'Export all available completed R11 '+kind+' output')
    pointer['commit']=commit
    atomic(env['work']/(kind+'-last-published.json'),dict(snapshot_id=snapshot,receipt=pointer))
    return pointer


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()


def _report_paths(env):
    u=env['u'];output=env['output'];sources={}
    canonical=output/'plan.json'
    if canonical.is_file() and not canonical.is_symlink():
        if sha(canonical)!=env['plan_sha256']:raise ValueError('R11_OUTPUT_PLAN_DIFFERS_FROM_EXPORT_PLAN')
        sources['plan.json']=canonical
    elif env['path'].parent==output:
        sources[env['path'].name]=env['path']
    else:
        raise ValueError('R11_ORIGINAL_OUTPUT_PLAN_REQUIRED')
    for path in output.iterdir():
        if (path.is_file() and not path.is_symlink() and not path.name.startswith('.')
            and path.name.lower() not in ('github.json','credentials.json','token.txt')
            and path.suffix.lower() in ('.json','.jsonl','.csv','.tsv','.md','.txt','.yaml','.yml')):
            u.safe_file(output,path);sources[path.name]=path
    reports=output/'reports'
    if reports.is_dir() and not reports.is_symlink():
        for path in u.scientific_files(reports):
            if path.suffix.lower() in ('.json','.jsonl','.csv','.tsv','.md','.txt','.yaml','.yml'):
                sources['reports/'+u.safe_file(reports,path)]=path
    return sources


def _full_metadata_paths(env):
    """Existing scientific job logs accompany, but never establish, completion."""
    sources=_report_paths(env);output=env['output'];logs=env['runtime']/'logs'
    management=('upload','full-upload','paper-upload','monitor-upload','controller',
        'manager','management','supervisor','guardian','reporter')
    if logs.is_dir() and not logs.is_symlink():
        for directory,dirs,files in os.walk(logs,followlinks=False):
            parent=Path(directory)
            dirs[:]=sorted(name for name in dirs if name not in env['u'].EXCLUDED
                and not name.lower().startswith(('.env',*management)) and not (parent/name).is_symlink())
            for name in sorted(files):
                path=parent/name
                if (path.is_file() and not path.is_symlink() and path.suffix.lower() in ('.out','.err','.log')
                    and not name.lower().startswith(management)):
                    relative=env['u'].safe_file(output,path);sources[relative]=path
    return list(sources.values())


def _paper_report_snapshot(env,gh):
    """Independently preserve current reports without contending with a daemon.

    Old report receipts prove their old bytes, not the current mutable sources.
    Only this export's SHA-verified receipt bound to the current source signature
    map is reusable. Neither the old paper lock nor its report ledger is touched.
    """
    u=env['u'];work=env['work'];state_path=work/'paper-report-snapshot.json'
    with lock(work/'paper-report-snapshot.lock',blocking=True):
        sources=_report_paths(env)
        stamps={name:u.signature(path) for name,path in sorted(sources.items())}
        source_identity=u.digest(stamps);saved=read(state_path,{}) or {}
        if (saved.get('recipe_sha256')==env['plan']['recipe_sha256']
            and saved.get('source_signatures')==stamps
            and saved.get('reports',{}).get('receipt',{}).get('remote_complete_blob_GET_SHA256_verified') is True):
            return saved['reports']
        stage=work/'paper-report-snapshots'/source_identity;stage.mkdir(parents=True,exist_ok=True)
        staged=[]
        for name,path in sorted(sources.items()):
            u._copy_stable(path,stage/name)
            staged.append(u.stage_file(stage,name,work/'paper-report-staged'))
        captured={name:u.signature(path) for name,path in sorted(_report_paths(env).items())}
        if captured!=stamps:raise RuntimeError('R11_PAPER_REPORT_SOURCE_CHANGED_RETRY')
        report_identity=sha_bytes(encoded([{k:r[k] for k in ('source_path','source_bytes','source_sha256','source_version')} for r in staged]))
        receipt=u.publish_snapshot(gh,11,staged,work/'paper-report-upload',metadata=dict(
            collection_status='COMPLETE',required_missing=[],required_warnings=[],
            source_signatures_sha256=source_identity,
            scope='Exact current existing reports and plan snapshot; no scientific recomputation.',training_controlled=False),
            prefix_root=u.stage_prefix(env['plan'])+'/completed-existing-reports/'+source_identity)
        reports=dict(identity=report_identity,source_signatures_sha256=source_identity,receipt=receipt)
        atomic(state_path,dict(recipe_sha256=env['plan']['recipe_sha256'],source_signatures=stamps,
            reports=reports,updated_utc=utc(),scientific_outputs_modified=False))
        current={name:u.signature(path) for name,path in sorted(_report_paths(env).items())}
        if current!=stamps:raise RuntimeError('R11_PAPER_REPORT_SOURCE_CHANGED_RETRY')
        return reports


def paper_once(args,*,env=None,client=None):
    env=env or load_environment(args);u=env['u'];plan=env['plan'];rows=completion_rows(env)
    if not rows:raise ValueError('R11_NO_ACTUAL_COMPLETED_RESULTS')
    work=env['runtime']/'upload-paper';work.mkdir(exist_ok=True)
    saved=read(work/'published-groups.json',{}) or {}
    if saved and saved.get('recipe_sha256')!=plan['recipe_sha256']:raise ValueError('R11_PAPER_LEDGER_RECIPE_DIFFERS')
    if not saved:saved=dict(recipe_sha256=plan['recipe_sha256'],tasks={})
    groups=saved.get('tasks',{});missing=[key for key,row in rows.items() if not _valid_paper_group(groups.get(key,{}),u._capture_identity(row))]
    if missing:
        try:
            with lock(env['runtime']/'paper-upload.lock'):
                # Only a missing-paper task is captured; never rerun the whole
                # scientific report or scan all already-published payloads.
                import r11_paper_fix
                r11_paper_fix.install_patch(u)
                key=sorted(missing)[0];row=rows[key]
                state=u._paper_capture_state(work,plan)
                stage=work/'captured'/plan['recipe_sha256']
                capture=u._capture_paper_task(plan,key,row,stage,work)
                state['tasks'][key]=capture;atomic(work/'capture-index.json',state)
                gh=client or _client(args,env,work,'paper')
                has_paper=capture['coverage']['has_declared_paper_artifacts']
                task_metadata=dict(stage=plan['stage'],recipe_sha256=plan['recipe_sha256'],task_key=key,
                    collection_status='COMPLETE' if has_paper else 'PARTIAL',
                    required_missing=[] if has_paper else [key+':paper_artifacts'],required_warnings=[],
                    paper_task_coverage=[capture['coverage']],training_controlled=False)
                receipt=r11_paper_fix._publish_scoped(u,gh,capture['files'],work,task_metadata,
                    u.stage_prefix(plan)+'/per-task/'+key+'/'+capture['identity'],capture.get('binary_representations',[]))
                saved['tasks'][key]=dict(identity=capture['identity'],receipt=receipt)
                atomic(work/'published-groups.json',saved)
                if not has_paper:
                    return dict(status='COMPLETED_TASK_HAS_NO_DECLARED_PAPER_ARTIFACTS',paper_ready=False,
                        complete=False,task_key=key,available_completed_tasks=len(rows),training_controlled=False)
                return dict(status='UPLOADING',paper_ready=False,complete=False,
                    available_completed_tasks=len(rows),uploaded_tasks=len(rows)-len(missing)+1,
                    task_key=key,training_controlled=False)
        except BlockingIOError:
            return dict(status='WAITING_EXISTING_PAPER_UPLOADER',paper_ready=False,complete=False,
                available_completed_tasks=len(rows),missing_tasks=len(missing),existing_uploader_preserved=True)
    # Existing 3190 task receipts can be reused while the old daemon owns its
    # lock. We publish an exact completed-set index, not a false 10270/10270.
    selected={key:groups[key] for key in sorted(rows)}
    report_work=env['work']/'paper-report-upload';report_work.mkdir(exist_ok=True)
    gh=client or _client(args,env,report_work,'paper')
    reports=_paper_report_snapshot(env,gh)
    receipt=_publish_summary(gh,env,'paper',rows,selected,reports,complete=True)
    return dict(status='COMPLETE_AVAILABLE',paper_ready=True,complete=True,
        available_completed_tasks=len(rows),uploaded_tasks=len(selected),declared_tasks=len(plan['tasks']),
        all_planned_task_receipts_complete=len(rows)==len(plan['tasks']),
        scientific_completion_claimed=False,pointer=receipt,training_controlled=False)


def full_once(args,*,env=None,client=None):
    env=env or load_environment(args);u=env['u'];plan=env['plan'];runtime=env['runtime']
    try:
        with lock(runtime/'full-upload.lock'):
            work=runtime/'upload-full';work.mkdir(exist_ok=True);rows=completion_rows(env)
            if not rows:raise ValueError('R11_NO_ACTUAL_COMPLETED_RESULTS')
            saved=read(work/'groups.json',{}) or {}
            if saved and saved.get('recipe_sha256')!=plan['recipe_sha256']:raise ValueError('R11_FULL_LEDGER_RECIPE_DIFFERS')
            if not saved:saved=dict(recipe_sha256=plan['recipe_sha256'],tasks={})
            source_state=read(work/'source-captures.json',{}) or {}
            captures=source_state.get('tasks',{}) if source_state.get('recipe_sha256')==plan['recipe_sha256'] else {}
            selected={key:group for key,group in saved.get('tasks',{}).items() if key in rows
                and group.get('identity')==rows[key]['completion_sha256'] and group.get('remote_full_GET_SHA256_verified') is True}
            gh=client or _client(args,env,work,'full')
            missing=[key for key in sorted(rows) if key not in selected]
            if missing:
                key=missing[0];row=rows[key]
                emit('R11_FULL_TASK_EXPORT_STARTED',task_key=key,already_uploaded_tasks=len(selected),available_completed_tasks=len(rows))
                capture=u._full_source_capture(plan,key,row,work,captures)
                folder=env['output']/key
                paths=[folder/relative for relative in capture['paths']]
                group=u.archive_group(gh,folder,paths,work,key,row['completion_sha256'],
                    plan['recipe_sha256'],u.stage_prefix(plan),expected_files=capture['expected_files'])
                saved['tasks'][key]=group;atomic(work/'groups.json',saved);selected[key]=group
                receipt=_publish_summary(gh,env,'full',rows,selected,saved.get('reports',{}),complete=False,delta_group=key)
                result=dict(status='UPLOADING',complete=False,available_completed_tasks=len(rows),
                    uploaded_tasks=len(selected),declared_tasks=len(plan['tasks']),task_key=key,
                    pointer=receipt,all_planned_task_receipts_complete=len(rows)==len(plan['tasks']),
                    scientific_completion_claimed=False,training_controlled=False)
                atomic(work/'completed-export-status.json',result)
                return result
            # Group output is sealed; only the small mutable report tree is
            # inventoried at the end, never each task and never raw EEG inputs.
            paths=_full_metadata_paths(env)
            report_identity=u.digest({str(p.relative_to(env['output'])):u.signature(p) for p in paths})
            if paths and saved.get('reports',{}).get('identity')!=report_identity:
                saved['reports']=u.archive_group(gh,env['output'],paths,work,'plan-and-reports',
                    report_identity,plan['recipe_sha256'],u.stage_prefix(plan))
                atomic(work/'groups.json',saved)
            # Read new metadata if tasks.json changed during the final report
            # capture; newly completed work will be appended next cycle.
            current=completion_rows(env)
            if current!=rows:
                return dict(status='UPLOADING_NEW_COMPLETIONS',complete=False,
                    available_completed_tasks=len(current),uploaded_tasks=len(selected),training_controlled=False)
            receipt=_publish_summary(gh,env,'full',rows,selected,saved.get('reports',{}),complete=True)
            result=dict(status='COMPLETE_AVAILABLE',complete=True,available_completed_tasks=len(rows),
                uploaded_tasks=len(selected),declared_tasks=len(plan['tasks']),pointer=receipt,
                all_planned_task_receipts_complete=len(rows)==len(plan['tasks']),
                scientific_completion_claimed=False,training_controlled=False)
            atomic(work/'completed-export-status.json',result)
            return result
    except BlockingIOError:
        return dict(status='WAITING_EXISTING_FULL_UPLOADER',complete=False,existing_uploader_preserved=True,
            training_controlled=False,reason='Exact existing full-upload.lock still held; no duplicate writer started.')


def process_identity(pid,proc_root=Path('/proc')):
    try:
        if type(pid) is not int or pid<=1:return None
        folder=Path(proc_root)/str(pid)
        if folder.stat().st_uid!=os.getuid():return None
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return None
        return dict(pid=pid,uid=os.getuid(),start_ticks=fields[19],
            argv=[v.decode() for v in (folder/'cmdline').read_bytes().split(b'\0') if v])
    except (OSError,ValueError,IndexError,UnicodeError):return None


def legacy_uploader_candidates(args,role='full',*,process=process_identity,env=None):
    """Return exact old CPU upload identities; caller alone decides handoff."""
    if role not in ('full','paper'):raise ValueError('UPLOAD_ROLE_REQUIRED')
    env=env or load_environment(args);runtime=env['runtime'];plan=env['plan'];path=env['path']
    paths=[runtime/('upload-'+role)/'process.json',runtime/(role+'-upload-process.json')]
    launch=read(runtime/'launch.json',{}) or {};allowed={path}
    if launch.get('plan_sha256')==env['plan_sha256'] and launch.get('campaign_identity')==plan['recipe_sha256']:
        candidate=Path(launch.get('plan','')).expanduser().resolve()
        if candidate.is_file() and sha(candidate)==env['plan_sha256']:allowed.add(candidate)
    entry=(env['package']/'paper3r11/platform/entry.py').resolve()
    interpreters={Path(sys.executable).resolve(),Path(plan.get('python',sys.executable)).expanduser().resolve()}
    result=[];seen=set()
    for record_path in paths:
        record=read(record_path,{}) or {};pid=record.get('pid')
        if pid in seen:continue
        seen.add(pid);value=process(pid)
        if value is None or value.get('uid')!=os.getuid():continue
        argv=value.get('argv',[])
        if not argv or not Path(argv[0]).is_absolute() or Path(argv[0]).resolve() not in interpreters:continue
        at=1
        while at<len(argv) and argv[at] in ('-I','-S','-B','-u','-E','-s','-O','-OO','-q'):at+=1
        if at+1>=len(argv) or Path(argv[at]).resolve()!=entry or argv[at+1]!='upload':continue
        if argument(argv,'--channel')!=role or argument(argv,'--plan') is None:continue
        source=Path(argument(argv,'--plan')).expanduser().resolve()
        if source not in allowed or not source.is_file() or sha(source)!=env['plan_sha256']:continue
        if '--child' in argv or 'science' in argv[at+1:at+2] or 'worker' in argv[at+1:at+2]:continue
        if record.get('start_ticks') is not None and record['start_ticks']!=value['start_ticks']:continue
        if record.get('argv') is not None and record['argv']!=argv:continue
        if record.get('channel',record.get('kind',role))!=role:continue
        result.append(dict(value,role=role,kind='uploader',plan=str(source),binding_plan=str(source),plan_sha256=env['plan_sha256'],
            record=str(record_path),scientific_worker=False,upload_lock=str(runtime/(role+'-upload.lock'))))
    return result
