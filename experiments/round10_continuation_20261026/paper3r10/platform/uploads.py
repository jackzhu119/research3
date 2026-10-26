"""Independent R10 paper evidence, full archive and status delivery processes.

Only user-started background processes import this module. Network failure never
changes scientific task state, claims, checkpoints, or training exit codes.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse

from .common import atomic, atomic_bytes, digest, encoded, emit, load_plan, read, sha, utc, verify_result
from .runtime import runtime_dir, status, completed_task_rows, plan_index
from .vendor.lossless_publish import VerifiedGitHub, publish_snapshot, stage_file
from .vendor.paper3_result_transport import ApiError

PREFIX='results/paper3/round-10'
PART_BYTES=32*1024*1024
EXCLUDED={'.lock','.git','private','secrets','credentials','__pycache__'}
_REPORT_COMPLETION_CACHE={}
_STATE_BYTES=256*1024*1024


def discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN','GITHUB_TOKEN','GH_TOKEN'):
        value=os.environ.get(name,'').strip()
        if value:return value
    candidates=[]
    for value in (explicit,os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if value:candidates.append(Path(value).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup','paper3-round3','paper3-round4','paper3-round5','paper3-round6',
                 'paper3-round7','paper3-round4-hpcgpu','paper3-round5-hpcgpu','paper3-round6-hpcgpu'):
        base=Path.home()/name
        candidates.append(base/'private/github.json')
        candidates.extend(sorted(base.glob('*/private/github.json')))
    for p in candidates:
        value=read(p,{})
        token=value.get('token') if isinstance(value,dict) else None
        if isinstance(token,str) and token.strip():return token.strip()
    raise RuntimeError('Existing Paper3 GitHub credential is unavailable')


def safe_file(base,path):
    base=Path(base).resolve();path=Path(path)
    relative=path.relative_to(base)
    if (path.is_symlink() or not path.is_file() or any(x in EXCLUDED or x.lower().startswith('.env') for x in relative.parts)
            or path.name.lower() in ('github.json','credentials.json')
            or not path.resolve().is_relative_to(base)):
        raise ValueError('Archive source is outside the declared scientific output')
    return relative.as_posix()


def signature(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def _copy_stable(source,target,*,expected_sha256=None):
    """Capture one source once, streaming exact bytes into owned immutable storage."""
    before=signature(source)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+target.name+'.',dir=target.parent)
    checksum=hashlib.sha256();count=0
    try:
        with Path(source).open('rb') as incoming,os.fdopen(descriptor,'wb') as outgoing:
            for block in iter(lambda:incoming.read(1024*1024),b''):
                outgoing.write(block);checksum.update(block);count+=len(block)
            outgoing.flush();os.fsync(outgoing.fileno())
        if signature(source)!=before or count!=before[2]:
            raise RuntimeError('Scientific source changed while capturing paper evidence')
        value=checksum.hexdigest()
        if expected_sha256 is not None and value!=expected_sha256:
            raise ValueError('Paper artifact SHA differs from scientific result')
        os.replace(temporary,target)
        return dict(bytes=count,sha256=value,source_signature=before)
    finally:
        Path(temporary).unlink(missing_ok=True)


def report_completion(output,recipe,expected_keys):
    """Exact recipe and task identities, with SHA-bound compact report coverage."""
    output=Path(output);expected=sorted(expected_keys)
    report_path=output/'reports/report.json';receipt_path=output/'reports/report-status.json'
    def stamp(path):
        try:return signature(path)
        except OSError:return None
    stamps=(stamp(report_path),stamp(receipt_path))
    cache_key=(str(output.resolve()),recipe,digest(expected))
    previous=_REPORT_COMPLETION_CACHE.get(cache_key)
    if previous and previous['stamps']==stamps:return previous['complete']
    receipt=read(receipt_path,{})
    def exact(value):
        return (isinstance(value,dict) and value.get('status')=='COMPLETE'
            and value.get('recipe_sha256')==recipe and value.get('total_tasks')==len(expected)
            and value.get('sealed_complete_tasks')==len(expected)
            and value.get('complete_task_keys_sha256')==digest(expected))
    if exact(receipt):
        complete=bool(stamps[0] and receipt.get('report_sha256')==sha(report_path))
    else:
        report=read(report_path,{},max_bytes=_STATE_BYTES)
        report=report if isinstance(report,dict) else {}
        coverage=report.get('coverage',{})
        coverage=coverage if isinstance(coverage,dict) else {}
        complete=(report.get('status')=='COMPLETE' and report.get('recipe_sha256')==recipe
                  and exact(dict(coverage,status='COMPLETE',recipe_sha256=recipe))
                  and coverage.get('complete_task_keys')==expected)
    if stamps!=(stamp(report_path),stamp(receipt_path)):return False
    _REPORT_COMPLETION_CACHE[cache_key]=dict(stamps=stamps,complete=complete)
    return complete


def stage_prefix(plan):
    return PREFIX+'/stages/'+plan['stage']+'/'+plan['recipe_sha256'][:16]


STAGE_RANK={'mechanism':0,'formal_exploratory':1,'formal_confirmatory':2}

def merge_stage_index(previous,plan,channel,reference):
    """Merge only the finite registered stages; active_stage cannot move back."""
    old=previous if isinstance(previous,dict) and previous.get('schema_version')=='paper3-r10-stage-index-v1' else {}
    merged=json.loads(json.dumps(old)) if old else dict(schema_version='paper3-r10-stage-index-v1',round=10,stages={})
    stages=merged.setdefault('stages',{});runs=stages.setdefault(plan['stage'],{})
    run=runs.setdefault(plan['recipe_sha256'],dict(stage=plan['stage'],recipe_sha256=plan['recipe_sha256'],channels={}))
    run['channels'][channel]=reference
    active=merged.get('active_stage')
    if active not in STAGE_RANK or STAGE_RANK[plan['stage']]>=STAGE_RANK[active]:
        merged.update(active_stage=plan['stage'],active_recipe_sha256=plan['recipe_sha256'])
    merged.update(updated_utc=utc(),scientific_completion_claimed=False)
    return merged


def publish_stage_index(gh,plan,channel,reference,*,monitor_snapshot=None):
    """Head-bound merge and non-force CAS prevents cross-process lost updates."""
    gh.ensure_branch();quoted=urllib.parse.quote(gh.branch,safe='/')
    index_path=PREFIX+'/INDEX.json';attempt=0
    while True:
        head=gh.api(gh.prefix+'/git/ref/heads/'+quoted)['object']['sha']
        try:
            old=gh.api(gh.prefix+'/contents/'+urllib.parse.quote(index_path,safe='/')+'?ref='+head)
            if old.get('encoding','base64')!='base64':
                old=gh.api(gh.prefix+'/git/blobs/'+old['sha'])
            previous=json.loads(base64.b64decode(old['content']))
        except ApiError as error:
            if error.status!=404:raise
            previous={}
        merged=merge_stage_index(previous,plan,channel,reference)
        files={index_path:encoded(merged)}
        if monitor_snapshot is None:
            files[PREFIX+'/PAPER_DATA_LATEST.json']=encoded(merged)
        elif merged['active_stage']==plan['stage'] and merged['active_recipe_sha256']==plan['recipe_sha256']:
            files['monitoring/paper3-round10/status.json']=encoded(monitor_snapshot)
        entries=[dict(path=p,mode='100644',type='blob',sha=gh._blob(v)) for p,v in sorted(files.items())]
        base=gh.api(gh.prefix+'/git/commits/'+head)['tree']['sha']
        tree=gh.api(gh.prefix+'/git/trees',dict(base_tree=base,tree=entries),'POST')['sha']
        if tree==base:return head
        commit=gh.api(gh.prefix+'/git/commits',dict(message='Merge R10 stage evidence index',tree=tree,parents=[head]),'POST')['sha']
        try:
            gh.api(gh.prefix+'/git/refs/heads/'+quoted,dict(sha=commit,force=False),'PATCH')
            return commit
        except ApiError as error:
            if error.status not in (409,422):raise
            attempt+=1;gh._delay(attempt)


def _capture_identity(row):
    return digest({name:row[name] for name in ('task_identity','result_sha256','completion_sha256')})


def _capture_paper_task(plan,key,row,stage,work):
    """Only a newly sealed task touches its original scientific directory."""
    identity=_capture_identity(row);relative=Path('per-task')/key/identity
    capture=stage/relative;folder=Path(plan['output'])/key
    saved=read(capture/'capture.json',{},max_bytes=_STATE_BYTES)
    if (saved.get('identity')==identity and saved.get('recipe_sha256')==plan['recipe_sha256']
            and saved.get('task_key')==key):
        return saved
    result_path=folder/'result.json';completion_path=folder/'completion.json'
    if sha(result_path)!=row['result_sha256'] or sha(completion_path)!=row['completion_sha256']:
        raise ValueError('Cached completed-task receipt differs from scientific source')
    result=read(result_path,{})
    declared=result.get('paper_artifacts',[]);sources=[('result.json',row['result_sha256']),
                                                  ('completion.json',row['completion_sha256'])]
    for item in declared:
        name=item['path'] if isinstance(item,dict) else item
        safe_file(folder,folder/name)
        sources.append((name,item.get('sha256') if isinstance(item,dict) else None))
    files=[];captured=[];paths=[]
    for name,expected in sources:
        safe_file(folder,folder/name)
        proof=_copy_stable(folder/name,capture/name,expected_sha256=expected)
        staged=stage_file(capture,name,work/'staged')
        if staged['source_sha256']!=proof['sha256']:
            raise ValueError('Captured paper source differs from its staged fragments')
        files.append(dict(staged,source_path=key+'/'+name))
        paths.append((relative/name).as_posix())
        if name not in ('result.json','completion.json'):
            captured.append(dict(source_path=key+'/'+name,bytes=proof['bytes'],sha256=proof['sha256']))
    record=dict(identity=identity,recipe_sha256=plan['recipe_sha256'],task_key=key,
                completion_sha256=row['completion_sha256'],files=files,paths=paths,
                coverage=dict(task_key=key,recipe_sha256=plan['recipe_sha256'],
                    captured_paper_artifacts=captured,has_declared_paper_artifacts=bool(declared)))
    atomic(capture/'capture.json',record)
    return record


def _paper_capture_state(work,plan):
    value=read(Path(work)/'capture-index.json',{},max_bytes=_STATE_BYTES)
    if value.get('recipe_sha256')!=plan['recipe_sha256']:
        value=dict(recipe_sha256=plan['recipe_sha256'],tasks={},reports={})
    return value


def paper_collection(plan_path,work):
    """Incremental immutable task captures; reports alone track changing snapshots.

    Completed task identities come from the runtime's durable sealed cache. Once
    captured, neither their original files nor their staged chunks are reread on
    later completions. Each task is published at its own immutable remote root.
    """
    path,plan=load_plan(plan_path,validate_science_stage=False);output=Path(plan['output']);work=Path(work)
    stage=work/'captured'/plan['recipe_sha256'];state=_paper_capture_state(work,plan)
    rows=completed_task_rows(plan);changed=False;missing=[];coverage=[];paths=[]
    for key,row in sorted(rows.items()):
        identity=_capture_identity(row);capture=state['tasks'].get(key)
        if capture is None or capture.get('identity')!=identity:
            capture=_capture_paper_task(plan,key,row,stage,work)
            state['tasks'][key]=capture;changed=True
        coverage.append(capture['coverage']);paths.extend(capture['paths'])
        if not capture['coverage']['has_declared_paper_artifacts']:missing.append(key+':paper_artifacts')
    # Replacing a report never recopies the immutable per-task tree. Report files
    # are discovered only beneath reports/, never by walking task directories.
    sources={'plan.json':path};reports=output/'reports'
    if reports.is_dir():
        for source in scientific_files(reports):
            if source.suffix.lower() in ('.json','.jsonl','.csv','.md','.tsv','.txt','.yaml','.yml'):
                sources['reports/'+safe_file(reports,source)]=source
    report_records={}
    for name,source in sorted(sources.items()):
        stamp=signature(source);previous=state['reports'].get(name)
        if previous is None or previous.get('source_signature')!=stamp:
            _copy_stable(source,stage/name)
            previous=dict(source_signature=stamp,file=stage_file(stage,name,work/'staged'))
            changed=True
        report_records[name]=previous;paths.append(name)
    if set(report_records)!=set(state['reports']):changed=True
    state['reports']=report_records
    if changed:atomic(work/'capture-index.json',state)
    expected=[task['key'] for task in plan['tasks']]
    report_complete=report_completion(output,plan['recipe_sha256'],expected)
    if not report_complete:missing.append('reports/report.json:complete_exact_recipe')
    metadata={'stage':plan['stage'],'recipe_sha256':plan['recipe_sha256'],
              'complete_tasks':len(rows),'total_tasks':len(plan['tasks']),
              'collection_status':'COMPLETE' if len(rows)==len(plan['tasks']) and not missing else 'PARTIAL',
              'required_missing':missing,'required_warnings':[],
              'paper_task_coverage':coverage,'paper_tasks_with_declared_artifacts':sum(bool(r['captured_paper_artifacts']) for r in coverage),
              'scientific_metrics_recomputed':False,'declared_paper_artifacts_exact_unsanitized_bytes':True,
              'completed_tasks_from_verified_runtime_cache':True,'incremental_immutable_task_captures':True,
              'training_controlled':False}
    atomic(stage/'collection.json',dict(metadata=metadata,
        capture_identities={key:state['tasks'][key]['identity'] for key in sorted(rows)}))
    return stage,sorted(paths),metadata


def paper_cycle(gh,plan_path,work):
    stage,paths,metadata=paper_collection(plan_path,work);work=Path(work)
    _,plan=load_plan(plan_path,validate_science_stage=False);state=_paper_capture_state(work,plan)
    saved=read(work/'published-groups.json',{},max_bytes=_STATE_BYTES)
    if saved.get('recipe_sha256')!=plan['recipe_sha256']:
        saved=dict(recipe_sha256=plan['recipe_sha256'],tasks={})
    active={row['task_key'] for row in metadata['paper_task_coverage']}
    for key in sorted(active):
        capture=state['tasks'][key];previous=saved['tasks'].get(key,{})
        if previous.get('identity')==capture['identity']:continue
        task_metadata=dict(stage=plan['stage'],recipe_sha256=plan['recipe_sha256'],task_key=key,
            collection_status='COMPLETE' if capture['coverage']['has_declared_paper_artifacts'] else 'PARTIAL',
            required_missing=[] if capture['coverage']['has_declared_paper_artifacts'] else [key+':paper_artifacts'],
            required_warnings=[],paper_task_coverage=[capture['coverage']],training_controlled=False)
        prefix=stage_prefix(plan)+'/per-task/'+key+'/'+capture['identity']
        receipt=publish_snapshot(gh,10,capture['files'],work,metadata=task_metadata,prefix_root=prefix)
        saved['tasks'][key]=dict(identity=capture['identity'],receipt=receipt)
        atomic(work/'published-groups.json',saved)
    report_files=[record['file'] for _,record in sorted(state['reports'].items())]
    report_identity=digest([{name:record.get(name) for name in ('source_path','source_bytes','source_sha256','source_version')}
                            for record in report_files])
    if saved.get('reports',{}).get('identity')!=report_identity:
        receipt=publish_snapshot(gh,10,report_files,work,metadata=dict(stage=plan['stage'],
            recipe_sha256=plan['recipe_sha256'],collection_status=metadata['collection_status'],
            required_missing=metadata['required_missing'],required_warnings=[],training_controlled=False),
            prefix_root=stage_prefix(plan)+'/reports')
        saved['reports']=dict(identity=report_identity,receipt=receipt)
        atomic(work/'published-groups.json',saved)
    identity=digest(dict(tasks={key:saved['tasks'][key]['identity'] for key in sorted(active)},
                         reports=report_identity,metadata=metadata))
    previous=read(work/'last-published.json',{})
    if previous.get('identity')==identity and previous.get('receipt'):return previous['receipt']
    groups={key:saved['tasks'][key] for key in sorted(active)}
    manifest=dict(schema_version='paper3-r10-incremental-paper-manifest-v1',round=10,
        recipe_sha256=plan['recipe_sha256'],snapshot_id=identity,tasks=groups,reports=saved['reports'],
        metadata=metadata,exact_original_bytes=True,remote_complete_blob_GET_SHA256_verified=True,
        scientific_metrics_recomputed=False,training_controlled=False)
    root=stage_prefix(plan);manifest_path=root+'/paper-data/'+identity+'/manifest.json'
    receipts=[group['receipt'] for group in groups.values()]+[saved['reports']['receipt']]
    receipt=dict(schema_version='paper3-r10-incremental-paper-pointer-v1',round=10,stage=plan['stage'],
        recipe_sha256=plan['recipe_sha256'],snapshot_id=identity,manifest=manifest_path,
        status='PAPER_DATA_FILES_VERIFIED' if metadata['collection_status']=='COMPLETE' else 'PAPER_DATA_PARTIAL_FILES_VERIFIED',
        collection_status=metadata['collection_status'],complete_tasks=metadata['complete_tasks'],
        total_tasks=metadata['total_tasks'],files=sum(r.get('files',0) for r in receipts),
        bytes=sum(r.get('bytes',0) for r in receipts),incremental_immutable_task_captures=True,
        exact_original_bytes=True,remote_complete_blob_GET_SHA256_verified=True,
        full_raw_results_uploaded=False,training_controlled=False)
    receipt['commit']=gh.git_publish({manifest_path:encoded(manifest),root+'/PAPER_DATA_LATEST.json':encoded(receipt)},
                                    'Update R10 incremental paper evidence index')
    publish_stage_index(gh,plan,'paper',receipt)
    atomic(work/'last-published.json',dict(identity=identity,receipt=receipt))
    return receipt


class FullVerifiedGitHub(VerifiedGitHub):
    """Always GET archive bytes as well as checking GitHub asset metadata."""
    def _asset_hash(self,asset):
        h=hashlib.sha256();count=0
        with self._request(self.prefix_url+'/releases/assets/%s'%asset['id'],accept='application/octet-stream') as response:
            if getattr(response,'status',200)!=200:
                raise RuntimeError('Archive verification requires a full HTTP200 response')
            while True:
                block=response.read(1024*1024)
                if not block:break
                h.update(block);count+=len(block)
        if count!=asset['size']:raise RuntimeError('Archive GET is incomplete')
        return h.hexdigest()


class HashReader:
    def __init__(self,handle):self.handle=handle;self.hash=hashlib.sha256();self.count=0
    def read(self,size=-1):
        block=self.handle.read(size);self.hash.update(block);self.count+=len(block);return block


def scientific_files(base):
    base=Path(base)
    for directory,dirs,files in os.walk(base,followlinks=False):
        parent=Path(directory)
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (parent/d).is_symlink())
        for name in sorted(files):
            if name in EXCLUDED or name.endswith('.lock') or '.tmp-' in name:continue
            p=parent/name
            if p.is_symlink():raise ValueError('Scientific output unexpectedly contains a symlink')
            safe_file(base,p)
            yield p


def pack_group(base,paths,work,key,identity,part_bytes=PART_BYTES):
    """Stable completed output -> deterministic gzip/tar and bounded parts."""
    base=Path(base);paths=list(paths);folder=Path(work)/'groups'/hashlib.sha256(key.encode()).hexdigest()/identity
    folder.mkdir(parents=True,exist_ok=True)
    manifest=read(folder/'manifest.json',{})
    current={safe_file(base,p):signature(p) for p in paths}
    if (manifest.get('source_signatures')==current and manifest.get('identity')==identity
            and all((folder/p['name']).is_file() and sha(folder/p['name'])==p['sha256'] for p in manifest.get('parts',[]))):
        return folder,manifest
    archive=folder/'archive.tar.gz';temporary=folder/'archive.tar.gz.part';records=[]
    with temporary.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as gz:
        with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for p in paths:
                rel=safe_file(base,p);before=signature(p)
                info=tarfile.TarInfo(rel);info.size=p.stat().st_size;info.mode=0o600;info.mtime=0;info.uid=info.gid=0
                with p.open('rb') as handle:
                    reader=HashReader(handle);tar.addfile(info,reader)
                if signature(p)!=before or reader.count!=info.size:raise RuntimeError('Completed scientific file changed during archive')
                records.append(dict(path=rel,bytes=reader.count,sha256=reader.hash.hexdigest()))
        raw.flush()
    os.replace(temporary,archive)
    parts=[]
    with archive.open('rb') as source:
        number=0
        while True:
            block=source.read(part_bytes)
            if not block:break
            number+=1;name='part-%05d.bin'%number;atomic_bytes(folder/name,block)
            parts.append(dict(number=number,name=name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
    manifest=dict(schema_version='paper3-r10-complete-group-v1',key=key,identity=identity,
                  compression='gzip-level1 deterministic tar.PAX',files=records,parts=parts,
                  source_signatures=current,full_original_bytes=True)
    atomic(folder/'manifest.json',manifest)
    return folder,manifest


def archive_group(gh,base,paths,work,key,identity,recipe,prefix,*,expected_files=None):
    folder,manifest=pack_group(base,paths,work,key,identity)
    if expected_files is not None:
        captured={item['path']:item for item in manifest['files']}
        for name,expected in expected_files.items():
            if name not in captured or any(captured[name].get(field)!=expected[field] for field in ('bytes','sha256')):
                raise ValueError('Archived scientific bytes differ from the sealed completion')
    tag='paper3-r10-'+recipe[:12]+'-'+hashlib.sha256(key.encode()).hexdigest()[:12]+'-'+identity[:12]
    release=gh.ensure_release(tag,'Paper3 R10 '+key,'Exact scientific outputs, model states and prediction evidence. SHA256 manifests accompany parts.')
    uploaded=[]
    for part in manifest['parts']:
        item=gh.upload_part(release,folder/part['name'],part['name'],part['sha256'])
        uploaded.append(dict(part,remote=item))
    result=dict(manifest,parts=uploaded,remote_full_GET_SHA256_verified=True,training_controlled=False)
    target=prefix+'/full/'+recipe+'/'+hashlib.sha256(key.encode()).hexdigest()+'/'+identity+'/manifest.json'
    commit=gh.git_publish({target:encoded(result)},'Archive verified R10 scientific output '+key)
    return dict(key=key,identity=identity,manifest=target,commit=commit,
                source_files=len(manifest['files']),parts=len(uploaded),remote_full_GET_SHA256_verified=True)


def _full_source_capture(plan,key,row,work,captures):
    """Persist one full verification/inventory per newly completed task identity."""
    identity=_capture_identity(row);previous=captures.get(key,{})
    if previous.get('identity')==identity:
        capture=read(previous['manifest'],{},max_bytes=_STATE_BYTES)
        if capture.get('identity')==identity and capture.get('recipe_sha256')==plan['recipe_sha256']:
            return capture
    task=plan_index(plan)[0][key]
    if not verify_result(plan,task):raise ValueError('Completed R10 archive artifacts changed after sealing')
    folder=Path(plan['output'])/key
    if sha(folder/'result.json')!=row['result_sha256'] or sha(folder/'completion.json')!=row['completion_sha256']:
        raise ValueError('Cached completed-task receipt differs from scientific source')
    result=read(folder/'result.json',{})
    expected={item['path']:dict(bytes=item['bytes'],sha256=item['sha256']) for item in result['artifacts']}
    expected.update({name:dict(bytes=(folder/name).stat().st_size,sha256=row[field])
                     for name,field in (('result.json','result_sha256'),('completion.json','completion_sha256'))})
    paths=[safe_file(folder,path) for path in scientific_files(folder)]
    capture=dict(schema_version='paper3-r10-verified-source-capture-v1',identity=identity,
        recipe_sha256=plan['recipe_sha256'],task_key=key,completion_sha256=row['completion_sha256'],
        paths=paths,expected_files=expected,full_artifacts_verified_once=True,verified_utc=utc())
    target=Path(work)/'source-captures'/key/identity/'manifest.json';atomic(target,capture)
    captures[key]=dict(identity=identity,manifest=str(target))
    atomic(Path(work)/'source-captures.json',dict(recipe_sha256=plan['recipe_sha256'],tasks=captures))
    return capture


def full_cycle(gh,plan_path,work):
    path,plan=load_plan(plan_path,validate_science_stage=False);output=Path(plan['output']);work=Path(work)
    saved=read(work/'groups.json',{},max_bytes=_STATE_BYTES)
    if saved.get('recipe_sha256')!=plan['recipe_sha256']:
        saved=dict(recipe_sha256=plan['recipe_sha256'],tasks={})
    source_state=read(work/'source-captures.json',{},max_bytes=_STATE_BYTES)
    captures=source_state.get('tasks',{}) if source_state.get('recipe_sha256')==plan['recipe_sha256'] else {}
    rows=completed_task_rows(plan)
    for key,row in sorted(rows.items()):
        identity=row['completion_sha256'];previous=saved['tasks'].get(key,{})
        if previous.get('identity')==identity:continue
        capture=_full_source_capture(plan,key,row,work,captures);folder=output/key
        paths=[folder/relative for relative in capture['paths']]
        saved['tasks'][key]=archive_group(gh,folder,paths,work,key,identity,plan['recipe_sha256'],stage_prefix(plan),
                                         expected_files=capture['expected_files'])
        atomic(work/'groups.json',saved)
    # Only the mutable report tree is rescanned. Completed task directories and
    # their full payload hashes are never revisited after a verified publication.
    report_paths=[]
    if (output/'plan.json').is_file():report_paths.append(output/'plan.json')
    if (output/'reports').is_dir():report_paths+=list(scientific_files(output/'reports'))
    report_identity=digest({str(p.relative_to(output)):signature(p) for p in report_paths})
    if report_paths and saved.get('reports',{}).get('identity')!=report_identity:
        saved['reports']=archive_group(gh,output,report_paths,work,'plan-and-reports',report_identity,plan['recipe_sha256'],stage_prefix(plan))
        atomic(work/'groups.json',saved)
    done=(len(rows)==len(plan['tasks']) and report_completion(output,plan['recipe_sha256'],[t['key'] for t in plan['tasks']])
          and bool(saved.get('reports')))
    identity=digest(dict(tasks={key:saved['tasks'][key]['identity'] for key in sorted(rows)},
                         reports=report_identity,complete=done))
    last=read(work/'last-published.json',{})
    if last.get('identity')==identity and last.get('receipt'):return last['receipt']
    published=dict(tasks={key:saved['tasks'][key] for key in sorted(rows)})
    if saved.get('reports'):published['reports']=saved['reports']
    pointer=dict(schema_version='paper3-r10-full-archive-pointer-v1',round=10,stage=plan['stage'],
                 recipe_sha256=plan['recipe_sha256'],status='COMPLETE' if done else 'PARTIAL',
                 complete_tasks=len(rows),total_tasks=len(plan['tasks']),groups=published,
                 completed_tasks_from_verified_runtime_cache=True,full_task_artifacts_verified_once=True,
                 independent_from_paper_channel=True,remote_full_GET_SHA256_verified=True,training_controlled=False)
    pointer['commit']=gh.git_publish({stage_prefix(plan)+'/LATEST.json':encoded(pointer)},'Update R10 independent full archive index')
    # The stage index points to the full channel's immutable commit/path; it
    # must not duplicate thousands of archive-group inventories in INDEX.json.
    reference={key:pointer[key] for key in ('status','stage','recipe_sha256','complete_tasks',
        'total_tasks','commit','remote_full_GET_SHA256_verified','training_controlled')}
    reference['path']=stage_prefix(plan)+'/LATEST.json'
    publish_stage_index(gh,plan,'full',reference)
    atomic(work/'last-published.json',dict(identity=identity,receipt=pointer))
    return pointer


def monitor_cycle(gh,plan_path,work):
    # Create the progress branch before reading task state: a local snapshot
    # failure must not leave its existence coupled to a successful publish.
    head=gh.ensure_branch()
    atomic(Path(work)/'branch-ready.json',dict(branch=gh.branch,head=head,status='BRANCH_READY',updated_utc=utc()))
    _,plan=load_plan(plan_path,validate_science_stage=False)
    snapshot=status(plan_path,plan=plan)
    target='monitoring/paper3-round10/'+plan['stage']+'/'+plan['recipe_sha256']+'/status.json'
    commit=gh.git_publish({target:encoded(snapshot)},'Update R10 stage operational progress')
    publish_stage_index(gh,plan,'monitor',dict(path=target,commit=commit),monitor_snapshot=snapshot)
    return dict(status=snapshot['status'],commit=commit,updated_utc=utc(),training_controlled=False)


def channel_loop(plan_path,channel,*,credentials=None,gh_factory=None,stop=None,once=False,interval=120):
    path,plan=load_plan(plan_path,validate_science_stage=False);runtime=runtime_dir(plan)
    work=runtime/('upload-'+channel);work.mkdir(parents=True,exist_ok=True);stop=stop or threading.Event()
    with (runtime/(channel+'-upload.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'EXISTING_PROCESS_REUSED','channel':channel}
        atomic(work/'process.json',dict(pid=os.getpid(),channel=channel,started_utc=utc()))
        try:os.nice(10)
        except OSError:pass
        while not stop.is_set():
            try:
                branch='paper3-r10-progress' if channel=='monitor' else 'paper3-results'
                factory=gh_factory or (FullVerifiedGitHub if channel=='full' else VerifiedGitHub)
                gh=factory(discover_token(credentials),emit,branch=branch,cache_path=work/'git-blobs.json')
                function={'paper':paper_cycle,'full':full_cycle,'monitor':monitor_cycle}[channel]
                result=function(gh,path,work)
                atomic(work/'status.json',dict(result,channel=channel,updated_utc=utc(),training_controlled=False))
                if result.get('status') in ('COMPLETE','PAPER_DATA_FILES_VERIFIED'):
                    return result
            except Exception as error:
                # Never publish HTTP bodies, request headers or token values.
                result=dict(status='RETRYING',channel=channel,error_type=type(error).__name__,
                            http_status=getattr(error,'status',None),updated_utc=utc(),training_controlled=False)
                atomic(work/'status.json',result);emit('R10_UPLOAD_RETRY',**result)
            if once:return result
            stop.wait(interval)
    return {'status':'STOPPED','channel':channel}


def start_uploads(plan_path,*,python=None,credentials=None,popen=None):
    path,plan=load_plan(plan_path,validate_science_stage=False);runtime=runtime_dir(plan);runtime.mkdir(parents=True,exist_ok=True)
    python=python or plan.get('python') or sys.executable;popen=popen or subprocess.Popen;results=[]
    entry=Path(plan['package'])/'paper3r10/platform/entry.py'
    for channel in ('report','paper','full','monitor'):
        with (runtime/(channel+'-upload.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);live=False
            except BlockingIOError:live=True
        if live:
            results.append(dict(kind=channel,reused=True));continue
        command=[python,'-I','-S','-B','-u',str(entry),'report-loop' if channel=='report' else 'upload','--plan',str(path)]
        if channel!='report':command+=['--channel',channel]
        if credentials and channel!='report':command+=['--credentials',str(Path(credentials).expanduser())]
        with (runtime/(channel+'-upload.log')).open('ab',buffering=0) as log:
            child=popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
        receipt=dict(kind=channel,pid=child.pid,reused=False,started_utc=utc())
        atomic(runtime/(channel+'-upload-process.json'),receipt);results.append(receipt)
    return results


def report_loop(plan_path,*,runner=None,stop=None,once=False,interval=60,python=None):
    """Independent bounded CPU reporting; failure never changes science state."""
    path,plan=load_plan(plan_path,validate_science_stage=False);runtime=runtime_dir(plan)
    runtime.mkdir(parents=True,exist_ok=True);stop=stop or threading.Event();runner=runner or subprocess.run
    work=runtime/'reporter';work.mkdir(exist_ok=True)
    saved=read(work/'last-generated.json',{});last=saved.get('signature') if saved.get('recipe_sha256')==plan['recipe_sha256'] else None
    with (runtime/'report-upload.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='EXISTING_REPORTER_REUSED')
        while not stop.is_set():
            signatures={key:row['completion_sha256'] for key,row in completed_task_rows(plan).items()}
            expected=[t['key'] for t in plan['tasks']]
            if len(signatures)==len(expected) and report_completion(plan['output'],plan['recipe_sha256'],expected):
                result=dict(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],complete_tasks=len(signatures),
                            total_tasks=len(expected),training_controlled=False,updated_utc=utc())
                atomic(runtime/'reporter-status.json',result);return result
            signature=digest(signatures)
            result=dict(status='WAITING_FOR_NEW_COMPLETION',complete_tasks=len(signatures),total_tasks=len(expected))
            if signature!=last and signatures:
                command=[python or plan.get('python') or sys.executable,'-I','-S','-B','-u',
                         str(Path(plan['package'])/'START.py'),'report','--plan',str(path)]
                try:
                    with (work/'report.log').open('ab',buffering=0) as log:
                        completed=runner(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,timeout=300,check=False)
                    if completed.returncode:raise RuntimeError('ReportProcessFailed')
                    last=signature
                    atomic(work/'last-generated.json',dict(recipe_sha256=plan['recipe_sha256'],signature=signature))
                    done=len(signatures)==len(expected) and report_completion(plan['output'],plan['recipe_sha256'],expected)
                    result.update(status='COMPLETE' if done else 'PARTIAL_REPORT_GENERATED')
                except Exception as error:
                    result.update(status='RETRYING',error_type=type(error).__name__)
            result.update(recipe_sha256=plan['recipe_sha256'],training_controlled=False,updated_utc=utc())
            atomic(runtime/'reporter-status.json',result)
            if once or result['status']=='COMPLETE':return result
            stop.wait(interval)
    return dict(status='STOPPED',training_controlled=False)


def report_main(argv=None):
    parser=argparse.ArgumentParser(description='Independent R10 CPU report generator; no network or credentials')
    parser.add_argument('--plan',required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    report_loop(args.plan,once=args.once,stop=stop);return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--channel',choices=('paper','full','monitor'),required=True)
    parser.add_argument('--credentials');parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=float,default=120)
    args=parser.parse_args(argv);stop=threading.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
    channel_loop(args.plan,args.channel,credentials=args.credentials,once=args.once,interval=args.interval,stop=stop)
    return 0


if __name__=='__main__':raise SystemExit(main())
