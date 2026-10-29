"""Export exact already-produced Round1 science; never controls experiments.

Completion is read from native receipts, not Slurm or observation heartbeats.
No model/EEG loader, report builder, optimizer or scheduler is imported here.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time

PREFIX='results/paper3/round-01'
OPERATIONAL='runs/training_campaign_parallel_v1'
ARMS=('BASE4_PLUS_YANG','BASE4_PLUS_DREYER')
CAMPAIGN='runs/training_campaign_v1'
FORBIDDEN={'private','secrets','credentials','.git','.ssh','.aws','__pycache__','runtime',
    'cache','data','datasets','raw','raw_eeg','raw-eeg','github.json','credentials.json',
    'token','token.txt','.env','stop'}
COMPLETE={
 'ENGINEERING_PILOT_COMPLETE','SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING',
 'SOURCE_CLASSICAL_CORE_COMPLETE','SOURCE_REFERENCE_COMPLETE','META_ENGINEERING_COMPLETE',
 'META_JOB_COMPLETE','META_CONFIGURATION_FROZEN','DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE',
 'DEVELOPMENT_EVALUATION_COMPLETE','DEVELOPMENT_CONFIGURATION_FROZEN',
 'STRONGEST_DEVELOPMENT_CONTROL_FROZEN','IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE',
 'ONE_TIME_INTERNAL_CHECK_COMPLETE','SOURCE_F0_COMPLETE','SOURCE_FIT_COMPLETE',
 'SOURCE_CLASSICAL_FIT_COMPLETE','SOURCE_CLASSICAL_COMPLETE','PREDICTIONS_COMMITTED','FIT_PARAMETERS_COMMITTED',
 'SOURCE_EXTENSION_F0_COMPLETE','SOURCE_EXTENSION_REFERENCE_COMPLETE',
 'SOURCE_EXTENSION_FIT_PAYLOAD_COMMITTED','SOURCE_EXTENSION_CLASSICAL_BANK_COMPLETE',
 'SOURCE_EXTENSION_META_CONFIGURATION_FROZEN','SOURCE_EXTENSION_META_ENGINEERING_COMPLETE',
 'SOURCE_EXTENSION_META_COMPLETE','SOURCE_EXTENSION_CALIBRATION_PROFILE_COMPLETE',
 'SOURCE_EXTENSION_PREDICTIONS_COMMITTED','SOURCE_EXTENSION_VALIDATION_COMPLETE',
 'SOURCE_EXTENSION_F0_ARM_STAGE_COMPLETE','SOURCE_EXTENSION_REFERENCE_ARM_STAGE_COMPLETE',
 'SOURCE_EXTENSION_METHOD_CONFIGURATION_FROZEN','SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN'}
METADATA_PARENTS={CAMPAIGN,CAMPAIGN+'/selections',CAMPAIGN+'/meta_selections'}


def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def emit(event,**fields):print(json.dumps(dict(event=event,utc=utc(),**fields),ensure_ascii=False),flush=True)
def read(path,default=None,limit=32*1024*1024):
    try:
        with Path(path).open('rb') as stream:data=stream.read(limit+1)
        if len(data)>limit:raise ValueError('R1_METADATA_RECORD_TOO_LARGE')
        return json.loads(data)
    except FileNotFoundError:return default


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.r1-export-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:handle.write(encoded(value));handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:fcntl.flock(handle,fcntl.LOCK_EX);yield


def signature(path):
    item=Path(path).stat()
    if not stat.S_ISREG(item.st_mode):raise ValueError('R1_SCIENCE_FILE_NOT_REGULAR')
    return [item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns]


def digest_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def safe(root,candidate,*,exists=True):
    root=Path(root).resolve();value=Path(candidate)
    if not value.is_absolute():value=root/value
    try:relative=value.relative_to(root)
    except ValueError:raise ValueError('R1_SOURCE_OUTSIDE_PROJECT') from None
    if '..' in relative.parts or any(p.lower() in FORBIDDEN or p.startswith('.') for p in relative.parts):
        raise ValueError('R1_PRIVATE_OR_INPUT_PATH_REFUSED')
    current=root
    for part in relative.parts:
        current/=part
        if current.is_symlink():raise ValueError('R1_SOURCE_SYMLINK_REFUSED')
    if not value.resolve(strict=exists).is_relative_to(root):raise ValueError('R1_SOURCE_ESCAPED_ROOT')
    return value


def science_paths(root,base):
    """Walk one completed output, keeping model formats and prediction arrays."""
    root=Path(root).resolve();base=safe(root,base)
    if base.is_file():yield base;return
    for directory,dirs,names in os.walk(base,followlinks=False):
        folder=Path(directory)
        dirs[:]=sorted(v for v in dirs if v.lower() not in FORBIDDEN and not v.startswith('.') and not (folder/v).is_symlink())
        for name in sorted(names):
            if name.lower() in FORBIDDEN or name.startswith('.') or name.endswith('.lock'):continue
            path=folder/name
            if path.is_symlink():continue
            if path.is_file():yield safe(root,path)


def projected(path,cache):
    """Fully parse each changed receipt once without retaining its big payload."""
    key=str(path);before=signature(path);prior=cache.get(key,{})
    if prior.get('signature')==before:return prior
    import receipt_projection as projection
    fields=projection.FIELDS
    try:
        projection.FIELDS=fields|{'synthetic'}
        value=projection.read_projected(path,{})
    finally:projection.FIELDS=fields
    checksum=digest_file(path)
    if signature(path)!=before:raise RuntimeError('R1_RECEIPT_CHANGED_DURING_DISCOVERY')
    item=dict(signature=before,status=value.get('status'),arm=value.get('arm'),
        synthetic=value.get('synthetic'),sha256=checksum)
    cache[key]=item
    return item


def _base_for(root,path):
    relative=path.relative_to(root)
    if relative.parent.as_posix() in METADATA_PARENTS:return None
    if relative.parts[:2]==('runs','internal_validation_once') and relative.parent.name=='predictions':
        return path.parent.parent
    return path.parent


def _candidate_paths(root,progress):
    values={};warnings=[]
    def add(path,base=None,expected=None,sha256=None,required=False):
        declared_path=str(path)
        try:path=safe(root,path)
        except FileNotFoundError:
            if required:warnings.append(dict(path=declared_path,error_type='DECLARED_COMPLETED_RECEIPT_MISSING'))
            return
        except ValueError:
            if required:warnings.append(dict(path=declared_path,error_type='DECLARED_COMPLETED_RECEIPT_PATH_REFUSED'))
            return
        if path.is_file():
            prior=values.get(path,{})
            values[path]=dict(base=base or prior.get('base'),expected=expected or prior.get('expected'),
                sha256=sha256 or prior.get('sha256'),required=required or prior.get('required',False))
    tasks=progress.get('tasks',[])
    if isinstance(tasks,dict):tasks=tasks.values()
    elif not isinstance(tasks,list):tasks=[]
    for task in tasks:
        if isinstance(task,dict) and isinstance(task.get('expected_receipt'),str):
            add(task['expected_receipt'],required=task.get('status')=='COMPLETE')
    named={
      'runs/controlled_source_pilot_seed20261018/engineering_receipt.json':'ENGINEERING_PILOT_COMPLETE',
      'runs/source_training_v2/source_profile_development.json':'SOURCE_PROFILE_NEURAL_COMPLETE_CLASSICAL_PENDING',
      'runs/source_classical_v2/receipt.json':'SOURCE_CLASSICAL_CORE_COMPLETE',
      CAMPAIGN+'/strongest_control.json':'STRONGEST_DEVELOPMENT_CONTROL_FROZEN',
      CAMPAIGN+'/pre_internal_freeze.json':'IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE',
      'runs/internal_validation_once/receipt.json':'ONE_TIME_INTERNAL_CHECK_COMPLETE'}
    for path,expected in named.items():add(path,expected=expected)
    patterns=(CAMPAIGN+'/references/*/receipt.json',CAMPAIGN+'/profiles/*/receipt.json',
       CAMPAIGN+'/development/*/receipt.json',CAMPAIGN+'/selections/*.json',CAMPAIGN+'/meta_selections/*.json',
       'runs/meta/*/receipt.json','runs/source_training_v2/f0_receipt_seed-*.json',
       'runs/source_training_v2/*/status.json','runs/source_classical_v2/*/fit_receipt.json',
       CAMPAIGN+'/profiles/*/fits/*/attempt-*/candidate_state.json',
       'runs/internal_validation_once/candidates/*/predictions/receipt.json')
    for pattern in patterns:
        for path in root.glob(pattern):add(path)
    # Catalogs explicitly declare each arm's owned output, including extension
    # f0 models stored outside source_extension_downstream. No generic input
    # descriptor closure is followed into raw data or shared cache directories.
    sections=('source_f0','references','source_banks','meta','meta_engineering','meta_selections','configurations','validation','profiles','final')
    catalogs=[]
    for arm in ARMS:
        base=root/'runs/source_extension_downstream'/arm
        add(base/'pre_internal_development_freeze.json',expected='SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN')
        for path in (base/'launch.json',base/'catalog.json'):
            if not path.is_file():continue
            path=safe(root,path);catalogs.append(path)
            try:catalog=read(path,{})
            except (OSError,ValueError) as error:
                warnings.append(dict(path=path.relative_to(root).as_posix(),error_type='EXTENSION_CATALOG_'+type(error).__name__));continue
            if not isinstance(catalog,dict) or catalog.get('arm')!=arm:
                warnings.append(dict(path=path.relative_to(root).as_posix(),error_type='R1_EXTENSION_CATALOG_ARM_DIFFERS'));continue
            def refs(value):
                if isinstance(value,dict):
                    if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                        candidate=Path(value['path'])
                        permitted=(candidate.parts[:3]==('runs','source_extension_downstream',arm)
                            or candidate.parts[:3]==('runs','source_extensions',arm))
                        if permitted:add(candidate,sha256=value['sha256'],required=True)
                    else:
                        for child in value.values():refs(child)
                elif isinstance(value,list):
                    for child in value:refs(child)
            for section in sections:refs(catalog.get(section,{}))
        # Native stage receipts can be committed just before their catalog entry.
        for pattern in ('reference/*/receipt.json','source_f0/*/receipt.json','reference_stage/*/receipt.json'):
            for path in base.glob(pattern):add(path)
        for path in (root/'runs/source_extensions'/arm).glob('seed-*/f0_receipt.json'):add(path)
    return values,catalogs,warnings


def _metadata_paths(root):
    paths=[]
    roots=[root/CAMPAIGN,root/'runs/internal_validation_once',
        root/'runs/source_training_v2',root/'runs/source_classical_v2']
    roots += [root/'runs/source_extension_downstream'/arm for arm in ARMS]
    for base in roots:
        if not base.is_dir():continue
        for path in base.iterdir():
            if path.is_file() and path.suffix.lower() in ('.json','.jsonl','.csv','.md','.txt','.yaml','.yml'):
                try:paths.append(safe(root,path))
                except (ValueError,FileNotFoundError):pass
        reports=base/'reports'
        if reports.is_dir():
            for path in science_paths(root,reports):
                if path.suffix.lower() in ('.json','.jsonl','.csv','.md','.txt'):paths.append(path)
    for relative in ('contracts/input_contract.json','status/tasks.json','runs/internal_validation_once_claim.json',
        'runs/training_campaign_v1_claim.json'):
        try:paths.append(safe(root,relative))
        except (ValueError,FileNotFoundError):pass
    return sorted(set(paths))


def _directory_version(root,base,cache):
    """Notice files added after a seal without hashing old checkpoints again."""
    def current(relative):
        path=safe(root,relative);item=path.stat()
        if not stat.S_ISDIR(item.st_mode):raise ValueError('R1_OUTPUT_DIRECTORY_REQUIRED')
        return [item.st_dev,item.st_ino,item.st_mtime_ns,item.st_ctime_ns]
    prior=cache.get(base,{})
    try:
        if prior.get('directories') and all(current(row['path'])==row['signature'] for row in prior['directories']):
            return prior['identity']
    except FileNotFoundError:pass
    rows=[]
    for directory,dirs,names in os.walk(safe(root,base),followlinks=False):
        folder=Path(directory)
        dirs[:]=sorted(v for v in dirs if v.lower() not in FORBIDDEN and not v.startswith('.') and not (folder/v).is_symlink())
        relative=folder.relative_to(root).as_posix()
        rows.append(dict(path=relative,signature=current(relative)))
    identity=hashlib.sha256(encoded(rows)).hexdigest()
    cache[base]=dict(identity=identity,directories=rows);return identity


def _freeze_metadata(root,paths,work,seals):
    """Capture original bytes once per completed frontier, not per heartbeat.

    Active progress/budget ledgers can keep changing even when an export has
    finished. They are retained as exact observed files in the frontier's
    frozen snapshot; a new completed receipt captures their newer version.
    Static configuration and existing reports independently advance an export.
    """
    volatile={'progress.json','campaign_progress.json','wall_budget.json'}
    paths=sorted(set(paths));static=[]
    for path in paths:
        if path.name not in volatile:
            static.append(dict(path=path.relative_to(root).as_posix(),signature=signature(path)))
    frontier=[{k:s[k] for k in ('path','status','sha256')} for s in seals]
    epoch=hashlib.sha256(encoded(dict(seals=frontier,static=static))).hexdigest()
    prior=read(work/'metadata-freeze.json',{}) or {}
    if prior.get('epoch')==epoch and all((work/'metadata'/epoch/row['path']).is_file() for row in prior.get('files',[])):
        return epoch,prior['files']
    rows=[];base=work/'metadata'/epoch
    for path in paths:
        relative=path.relative_to(root).as_posix();target=base/relative
        target.parent.mkdir(parents=True,exist_ok=True);before=signature(path)
        fd,tmp=tempfile.mkstemp(prefix='.metadata-',dir=target.parent);checksum=hashlib.sha256();size=0
        try:
            with path.open('rb') as source,os.fdopen(fd,'wb') as output:
                for block in iter(lambda:source.read(1024*1024),b''):
                    output.write(block);checksum.update(block);size+=len(block)
                output.flush();os.fsync(output.fileno())
            if signature(path)!=before:raise RuntimeError('R1_METADATA_CHANGED_DURING_CAPTURE')
            os.replace(tmp,target)
        finally:Path(tmp).unlink(missing_ok=True)
        rows.append(dict(path=relative,sha256=checksum.hexdigest(),bytes=size))
    atomic(work/'metadata-freeze.json',dict(epoch=epoch,files=rows));return epoch,rows


def inventory(root,work):
    root=Path(root).expanduser().resolve();work=Path(work)/'r1-inventory';work.mkdir(parents=True,exist_ok=True)
    with lock(work/'inventory.lock'):
        observation_warnings=[]
        try:
            progress=read(root/OPERATIONAL/'progress.json',{}) or {}
            if not isinstance(progress,dict):raise ValueError('R1_PROGRESS_OBJECT_REQUIRED')
        except (OSError,ValueError) as error:
            progress={};observation_warnings.append(dict(path=OPERATIONAL+'/progress.json',error_type='R1_PROGRESS_'+type(error).__name__))
        cache=read(work/'receipt-cache.json',{}) or {}
        directory_cache=read(work/'directory-cache.json',{}) or {}
        candidates,catalogs,warnings=_candidate_paths(root,progress)
        warnings=observation_warnings+warnings
        completed=[];groups=[];metadata=set(_metadata_paths(root))|set(catalogs)
        for path,declared in sorted(candidates.items(),key=lambda x:x[0].as_posix()):
            try:item=projected(path,cache)
            except (OSError,ValueError,RuntimeError) as error:
                warnings.append(dict(path=path.relative_to(root).as_posix(),error_type=type(error).__name__));continue
            if item['synthetic'] is True or item['status'] not in COMPLETE:
                if declared.get('required'):
                    warnings.append(dict(path=path.relative_to(root).as_posix(),error_type='DECLARED_COMPLETE_NATIVE_STATUS_OR_REAL_DATA_DIFFERS'))
                continue
            if declared.get('expected') and item['status']!=declared['expected']:
                warnings.append(dict(path=path.relative_to(root).as_posix(),error_type='NATIVE_COMPLETION_STATUS_DIFFERS'));continue
            if declared.get('sha256') and item['sha256']!=declared['sha256']:
                warnings.append(dict(path=path.relative_to(root).as_posix(),error_type='NATIVE_CATALOG_RECEIPT_SHA256_DIFFERS'));continue
            relative=path.relative_to(root).as_posix();base=_base_for(root,path)
            seal=dict(path=relative,status=item['status'],sha256=item['sha256'],signature=item['signature'])
            completed.append(seal)
            metadata.add(path)
            if base is None:continue
            groups.append(dict(base=base.relative_to(root).as_posix(),seals=[seal]))
        # A completed source-profile/internal/freeze root subsumes its child
        # outputs. Shared references are therefore archived once, never once
        # for each development job that reads them.
        selected=[]
        for group in sorted(groups,key=lambda g:(len(Path(g['base']).parts),g['base'])):
            owner=next((old for old in selected if Path(group['base']).is_relative_to(Path(old['base']))),None)
            if owner:owner['seals'].extend(group['seals'])
            else:selected.append(group)
        for group in selected:
            group['key']='output:'+group['base']
            group['seals']=sorted(group['seals'],key=lambda s:s['path'])
            group['directory_version']=_directory_version(root,group['base'],directory_cache)
            group['identity']=hashlib.sha256(encoded(dict(seals=group['seals'],directories=group['directory_version']))).hexdigest()
        statuses={s['path']:s['status'] for s in completed}
        mainline=(statuses.get('runs/internal_validation_once/receipt.json')=='ONE_TIME_INTERNAL_CHECK_COMPLETE'
            and statuses.get(CAMPAIGN+'/pre_internal_freeze.json')=='IMMUTABLE_PRE_INTERNAL_DEVELOPMENT_FREEZE')
        extension_complete={arm:statuses.get('runs/source_extension_downstream/'+arm+'/pre_internal_development_freeze.json')=='SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN' for arm in ARMS}
        science=bool(mainline and all(extension_complete.values()))
        # No observation heartbeat enters this content identity. Exact mutable
        # metadata is frozen for the completed frontier so packing cannot race
        # a controller that is merely updating its observation timestamp.
        epoch,metadata_rows=_freeze_metadata(root,metadata,work,completed)
        body=dict(schema_version='paper3-round1-exact-completed-output-inventory-v1',
            groups=sorted(selected,key=lambda g:g['key']),completed_receipts=sorted(completed,key=lambda s:s['path']),
            metadata_files=metadata_rows,metadata_epoch=epoch,warnings=warnings,mainline_complete=mainline,
            extension_complete=extension_complete,whole_round_completed=science,science_complete=science,
            completed_native_receipts=len(completed),counts_are_not_duration_weighted=True,
            scope='Exact produced native Round1 output folders with completed receipts; includes original result evaluation fields and all model/checkpoint formats, not raw EEG, input caches, environment or credentials.')
        body['inventory_id']=hashlib.sha256(encoded(body)).hexdigest()
        atomic(work/'receipt-cache.json',cache);atomic(work/'directory-cache.json',directory_cache)
        atomic(work/'inventory.json',body)
        return body


def _client(args,work):
    from delivery import client
    return client(work,getattr(args,'credentials',None))


def _group_files(root,group):
    base=safe(root,group['base'])
    paths=list(science_paths(root,base))
    for seal in group['seals']:
        receipt=safe(root,seal['path'])
        if receipt not in paths or digest_file(receipt)!=seal['sha256']:
            raise ValueError('R1_COMPLETE_SEAL_CHANGED_REFRESH_INVENTORY')
    return paths


def paper_once(args):
    root=Path(args.root).resolve();work=Path(args.work)/'r1-paper';work.mkdir(parents=True,exist_ok=True)
    bundle=inventory(root,args.work);snapshot=Path(args.work)/'r1-inventory'/'metadata'/bundle['metadata_epoch']
    paths={row['path']:(snapshot,snapshot/row['path']) for row in bundle['metadata_files']}
    for row in bundle['completed_receipts']:paths.setdefault(row['path'],(root,root/row['path']))
    # Existing CSV/report/summary artifacts are copied exactly; no statistic is
    # generated here and fitted parameter JSON is reserved for the full archive.
    for group in bundle['groups']:
        base=root/group['base']
        for path in science_paths(root,base):
            name=path.name.lower()
            if path.suffix.lower()=='.csv' or ('report' in name or 'summary' in name or 'metrics' in name) and path.suffix.lower() in ('.json','.jsonl','.txt','.md'):
                paths.setdefault(path.relative_to(root).as_posix(),(root,path))
    if not paths:raise ValueError('NO_EXISTING_R1_PAPER_DATA')
    source_rows=[dict(path=name,signature=signature(path)) for name,(base,path) in sorted(paths.items())]
    reports=[name for name in paths if '/reports/' in name or name.endswith('.csv')]
    latest_complete=max((s['signature'][3] for s in bundle['completed_receipts']),default=0)
    latest_report=max((signature(paths[name][1])[3] for name in reports),default=0)
    metadata=dict(collection_status='AVAILABLE_EXISTING_COMPLETED_PAPER_EVIDENCE',
        completed_native_receipts=bundle['completed_native_receipts'],mainline_complete=bundle['mainline_complete'],
        extension_complete=bundle['extension_complete'],whole_round_completed=bundle['whole_round_completed'],
        required_missing=[],required_warnings=bundle['warnings'],report_recomputed=False,
        statistics_recomputed=False,exact_original_bytes=True,source_inventory_id=bundle['inventory_id'],
        report_observation=dict(existing_report_files=len(reports),
            status='EXISTING_REPORTS_MAY_PRECEDE_LATEST_RECEIPT' if reports and latest_report<latest_complete else
            'EXISTING_REPORTS_PRESENT_FRESHNESS_NOT_RECOMPUTED' if reports else 'NO_EXISTING_REPORT_FILES',
            absence_does_not_block_original_receipt_export=True),
        scope='Complete existing native receipts plus existing CSV/reports/selection/configuration metadata; no model fit or report recomputation.')
    identity=hashlib.sha256(encoded(dict(files=source_rows,metadata=metadata))).hexdigest()
    prior=read(work/'uploaded.json',{}) or {}
    if prior.get('identity')==identity:return dict(prior,paper_ready=True,status='AVAILABLE_PAPER_UPLOADED',complete=True)
    from transport.lossless_publish import stage_file,publish_snapshot
    gh=_client(args,work)
    staged=[]
    for i,(name,(base,path)) in enumerate(sorted(paths.items()),1):
        emit('R1_PAPER_SOURCE_STAGING',file=i,total_files=len(paths),source_path=name)
        staged.append(stage_file(base,name,work/'staged',notify=emit))
    receipt=publish_snapshot(gh,1,staged,work,metadata=metadata)
    if receipt.get('status')!='PAPER_DATA_FILES_VERIFIED':raise RuntimeError('R1_PAPER_UPLOAD_NOT_VERIFIED')
    result=dict(status='AVAILABLE_PAPER_UPLOADED',paper_ready=True,complete=True,identity=identity,
        files=len(paths),completed_native_receipts=bundle['completed_native_receipts'],
        pending_receipt_errors=len(bundle['warnings']),required_warnings=bundle['warnings'],
        science_complete=bundle['science_complete'],whole_round_completed=bundle['whole_round_completed'],
        pointer=PREFIX+'/PAPER_DATA_LATEST.json',receipt=receipt,updated_utc=utc(),training_controlled=False)
    atomic(work/'uploaded.json',result);return result


def full_once(args):
    root=Path(args.root).resolve();work=Path(args.work)/'r1-full';work.mkdir(parents=True,exist_ok=True)
    bundle=inventory(root,args.work)
    if not bundle['groups'] and not bundle['metadata_files']:raise ValueError('NO_COMPLETED_R1_OUTPUTS')
    import group_archive as archive
    archive.ROUND=1;archive.PREFIX=PREFIX
    state=read(work/'state.json',dict(groups={})) or dict(groups={})
    gh=_client(args,work)
    selected=bundle['groups']+([dict(key='metadata-and-reports',identity=hashlib.sha256(encoded(bundle['metadata_files'])).hexdigest(),
        paths=[row['path'] for row in bundle['metadata_files']],snapshot=bundle['metadata_epoch'],seals=[])] if bundle['metadata_files'] else [])
    for group in selected:
        if state['groups'].get(group['key'],{}).get('identity')==group['identity']:continue
        source_base=Path(args.work)/'r1-inventory'/'metadata'/group['snapshot'] if 'snapshot' in group else root
        paths=[safe(source_base,p) for p in group['paths']] if 'paths' in group else _group_files(root,group)
        emit('R1_FULL_GROUP_STARTED',group=group['key'],files=len(paths),archived_groups=len(state['groups']),available_groups=len(selected))
        result=archive.archive_group(gh,None,source_base,paths,work,group['key'],group['identity'])
        state['groups'][group['key']]={k:v for k,v in result.items() if k!='capture_files'}
        atomic(work/'state.json',state)
        spool=work/'groups'/hashlib.sha256(group['key'].encode()).hexdigest()/group['identity']
        packed=read(spool/'packed.json',{}) or {}
        for part in packed.get('parts',[]):
            name=part.get('name','')
            if Path(name).name==name and name.startswith('round-'):(spool/name).unlink(missing_ok=True)
        return dict(status='UPLOADING',complete=False,round=1,archived_groups=len(state['groups']),
            available_groups=len(selected),completed_native_receipts=bundle['completed_native_receipts'],
            science_complete=bundle['science_complete'],whole_round_completed=bundle['whole_round_completed'],
            updated_utc=utc(),training_controlled=False)
    # Re-discover only receipt/signature metadata. Parsed large receipts are
    # reused from the signature cache, never reloaded once per archived group.
    latest=inventory(root,args.work)
    if latest['inventory_id']!=bundle['inventory_id']:
        return dict(status='SOURCE_INVENTORY_ADVANCED',complete=False,round=1,updated_utc=utc(),training_controlled=False)
    groups=[state['groups'][group['key']] for group in selected]
    pending=len(bundle['warnings'])
    manifest=dict(schema_version='paper3-round1-available-completed-full-output-v1',round=1,
        source_inventory=bundle,groups=groups,archived_groups=len(groups),
        files=sum(g['files'] for g in groups),archive_parts=sum(g['archive_parts'] for g in groups),
        source_bytes=sum(g['source_bytes'] for g in groups),
        pending_receipt_errors=pending,unresolved_receipts=bundle['warnings'],
        remote_full_GET_SHA256_verified=True,science_complete=bundle['science_complete'],
        whole_round_completed=bundle['whole_round_completed'],training_controlled=False)
    checksum=hashlib.sha256(encoded(manifest)).hexdigest();target=PREFIX+'/completed-export/'+checksum+'/manifest.json'
    pointer=dict(schema_version='paper3-round1-completed-export-pointer-v1',round=1,
        status='AVAILABLE_OUTPUT_UPLOADED_WITH_RECEIPT_GAPS' if pending else 'AVAILABLE_COMPLETED_OUTPUT_UPLOADED',
        manifest=target,snapshot_id=checksum,
        completed_native_receipts=bundle['completed_native_receipts'],archived_groups=len(groups),
        files=manifest['files'],archive_parts=manifest['archive_parts'],source_bytes=manifest['source_bytes'],
        pending_receipt_errors=pending,unresolved_receipts=bundle['warnings'],
        remote_full_GET_SHA256_verified=True,science_complete=bundle['science_complete'],
        whole_round_completed=bundle['whole_round_completed'],scope=bundle['scope'],updated_utc=utc(),training_controlled=False)
    commit=gh.git_publish({target:encoded(manifest),PREFIX+'/COMPLETED_OUTPUT_LATEST.json':encoded(pointer)},
        'Archive all available completed native Round1 output')
    result=dict(pointer,complete=not pending,commit=commit,retry_seconds=30 if pending else 0)
    atomic(work/'uploaded.json',result);return result
