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
import threading
import time
import urllib.parse

from .common import atomic, atomic_bytes, digest, encoded, emit, load_plan, read, sha, utc, verify_result
from .runtime import runtime_dir, status
from .vendor.lossless_publish import VerifiedGitHub, publish_snapshot, stage_file
from .vendor.paper3_result_transport import ApiError

PREFIX='results/paper3/round-10'
PART_BYTES=32*1024*1024
EXCLUDED={'.lock','.git','private','secrets','credentials','__pycache__'}


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


def _copy_stable(source,target):
    before=signature(source)
    data=Path(source).read_bytes()
    if signature(source)!=before:raise RuntimeError('Scientific source changed while capturing paper evidence')
    atomic_bytes(target,data)


def report_completion(output,recipe,expected_keys):
    """Exact recipe and task identities, with SHA-bound compact report coverage."""
    output=Path(output);expected=sorted(expected_keys);receipt=read(output/'reports/report-status.json',{})
    def exact(value):
        return (isinstance(value,dict) and value.get('status')=='COMPLETE'
            and value.get('recipe_sha256')==recipe and value.get('total_tasks')==len(expected)
            and value.get('sealed_complete_tasks')==len(expected)
            and value.get('complete_task_keys_sha256')==digest(expected))
    if exact(receipt):
        report=output/'reports/report.json'
        if report.is_file() and receipt.get('report_sha256')==sha(report):return True
    report=read(output/'reports/report.json',{},max_bytes=256*1024*1024)
    coverage=report.get('coverage',{}) if isinstance(report,dict) else {}
    return (report.get('status')=='COMPLETE' and report.get('recipe_sha256')==recipe
            and exact(dict(coverage,status='COMPLETE',recipe_sha256=recipe))
            and coverage.get('complete_task_keys')==expected)


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


def paper_collection(plan_path,work):
    path,plan=load_plan(plan_path,validate_science_stage=False);output=Path(plan['output']);work=Path(work)
    sources={'plan.json':path};complete=0;missing=[];coverage=[]
    for task in plan['tasks']:
        if not verify_result(plan,task,verify_artifacts=False):continue
        complete+=1;folder=output/task['key'];result=read(folder/'result.json',{})
        sources[task['key']+'/result.json']=folder/'result.json'
        sources[task['key']+'/completion.json']=folder/'completion.json'
        declared=result.get('paper_artifacts',[])
        if not declared:missing.append(task['key']+':paper_artifacts')
        captured=[]
        for item in declared:
            relative=item['path'] if isinstance(item,dict) else item
            p=folder/relative;safe_file(folder,p)
            if isinstance(item,dict) and sha(p)!=item.get('sha256'):
                raise ValueError('Paper artifact SHA differs from scientific result')
            sources[task['key']+'/'+relative]=p
            captured.append(dict(source_path=task['key']+'/'+relative,bytes=p.stat().st_size,sha256=sha(p)))
        coverage.append(dict(task_key=task['key'],recipe_sha256=plan['recipe_sha256'],
                             captured_paper_artifacts=captured,has_declared_paper_artifacts=bool(declared)))
    reports=output/'reports'
    if reports.is_dir():
        for p in scientific_files(reports):
            if p.suffix.lower() in ('.json','.jsonl','.csv','.md','.tsv','.txt','.yaml','.yml'):
                sources['reports/'+safe_file(reports,p)]=p
    report_complete=report_completion(output,plan['recipe_sha256'],[t['key'] for t in plan['tasks']])
    if not report_complete:missing.append('reports/report.json:complete_exact_recipe')
    source_stamps={name:signature(p) for name,p in sources.items()}
    identity=hashlib.sha256(encoded(source_stamps)).hexdigest()
    stage=work/'collections'/identity
    if not (stage/'collection.json').is_file():
        for name,p in sources.items():_copy_stable(p,stage/name)
        atomic(stage/'collection.json',dict(source_signatures=source_stamps))
    metadata={'stage':plan['stage'],'recipe_sha256':plan['recipe_sha256'],
              'complete_tasks':complete,'total_tasks':len(plan['tasks']),
              'collection_status':'COMPLETE' if complete==len(plan['tasks']) and not missing else 'PARTIAL',
              'required_missing':missing,'required_warnings':[],
              'paper_task_coverage':coverage,'paper_tasks_with_declared_artifacts':sum(bool(r['captured_paper_artifacts']) for r in coverage),
              'scientific_metrics_recomputed':False,'declared_paper_artifacts_exact_unsanitized_bytes':True,
              'training_controlled':False}
    return stage,sorted(sources),metadata


def paper_cycle(gh,plan_path,work):
    stage,paths,metadata=paper_collection(plan_path,work)
    identity=hashlib.sha256(encoded(dict(collection=stage.name,metadata=metadata))).hexdigest()
    previous=read(Path(work)/'last-published.json',{})
    if previous.get('identity')==identity and previous.get('receipt'):
        return previous['receipt']
    files=[stage_file(stage,name,Path(work)/'staged') for name in paths]
    _,plan=load_plan(plan_path,validate_science_stage=False)
    receipt=publish_snapshot(gh,10,files,work,metadata=metadata,prefix_root=stage_prefix(plan))
    receipt.update(stage=plan['stage'],recipe_sha256=plan['recipe_sha256'])
    publish_stage_index(gh,plan,'paper',receipt)
    atomic(Path(work)/'last-published.json',dict(identity=identity,receipt=receipt))
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


def archive_group(gh,base,paths,work,key,identity,recipe,prefix):
    folder,manifest=pack_group(base,paths,work,key,identity)
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


def full_cycle(gh,plan_path,work):
    path,plan=load_plan(plan_path,validate_science_stage=False);output=Path(plan['output']);work=Path(work)
    saved=read(work/'groups.json',{'tasks':{}});complete=[]
    for task in plan['tasks']:
        if not verify_result(plan,task,verify_artifacts=False):continue
        complete.append(task['key']);folder=output/task['key'];identity=sha(folder/'completion.json')
        previous=saved['tasks'].get(task['key'],{})
        if previous.get('identity')!=identity:
            if not verify_result(plan,task):raise ValueError('Completed R10 archive artifacts changed after sealing')
            paths=list(scientific_files(folder))
            saved['tasks'][task['key']]=archive_group(gh,folder,paths,work,task['key'],identity,plan['recipe_sha256'],stage_prefix(plan))
            atomic(work/'groups.json',saved)
    # The plan and report tree are a separate immutable snapshot. Raw EEG and
    # shared source models outside output are never traversed by this channel.
    report_paths=[]
    if (output/'plan.json').is_file():report_paths.append(output/'plan.json')
    if (output/'reports').is_dir():report_paths+=list(scientific_files(output/'reports'))
    report_identity=hashlib.sha256(encoded({str(p.relative_to(output)):signature(p) for p in report_paths})).hexdigest()
    if report_paths and saved.get('reports',{}).get('identity')!=report_identity:
        saved['reports']=archive_group(gh,output,report_paths,work,'plan-and-reports',report_identity,plan['recipe_sha256'],stage_prefix(plan))
        atomic(work/'groups.json',saved)
    done=(len(complete)==len(plan['tasks']) and report_completion(output,plan['recipe_sha256'],[t['key'] for t in plan['tasks']])
          and bool(saved.get('reports')))
    pointer=dict(schema_version='paper3-r10-full-archive-pointer-v1',round=10,stage=plan['stage'],
                 recipe_sha256=plan['recipe_sha256'],status='COMPLETE' if done else 'PARTIAL',
                 complete_tasks=len(complete),total_tasks=len(plan['tasks']),groups=saved,
                 independent_from_paper_channel=True,remote_full_GET_SHA256_verified=True,training_controlled=False)
    pointer['commit']=gh.git_publish({stage_prefix(plan)+'/LATEST.json':encoded(pointer)},'Update R10 independent full archive index')
    publish_stage_index(gh,plan,'full',pointer)
    return pointer


def monitor_cycle(gh,plan_path,work):
    snapshot=status(plan_path)
    _,plan=load_plan(plan_path,validate_science_stage=False)
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
    work=runtime/'reporter';work.mkdir(exist_ok=True);last=None
    with (runtime/'report-upload.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return dict(status='EXISTING_REPORTER_REUSED')
        while not stop.is_set():
            signatures={t['key']:sha(Path(plan['output'])/t['key']/'completion.json') for t in plan['tasks']
                        if verify_result(plan,t,verify_artifacts=False)}
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
