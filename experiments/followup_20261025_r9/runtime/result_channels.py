"""Round 9 independent incremental paper-evidence and scientific-archive channels.

Only declared plan/report paths are inspected by the paper channel. The archive
walks each COMPLETE scientific task once per receipt identity; it never packages
another round, raw datasets, runtime, credentials or the Python environment.
"""
import argparse
import base64
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import tarfile
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from three_jobs import atomic, read, emit
from paper3_result_transport import GitHub
from paper_credentials import upload_discover_token

PREFIX='results/paper3/round-09'
CHUNK_BYTES=1024*1024
ARCHIVE_BYTES=32*1024*1024
EXCLUDED={'private','runtime','logs','__pycache__','.git','credentials','github.json','credentials.json','token','token.txt','.env','STOP','.lock'}


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def signature(st):
    return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]


def source_signature(path):
    return signature(Path(path).stat())


def paper_sources(output):
    """Shallow declared reports only: no predictions or task-tree inspection."""
    output=Path(output)
    paths=[output/'plan.json',output/'report.json']
    paths += sorted(output.glob('*.csv'))
    reports=output/'reports'
    if reports.is_dir() and not reports.is_symlink():
        paths += sorted(p for p in reports.iterdir() if p.suffix.lower() in ('.csv','.json','.md') and p.is_file())
    unique=[]
    for path in paths:
        if path.is_file() and not path.is_symlink() and path not in unique:
            unique.append(path)
    priority={'plan.json':0,'report.json':1,'full-report.json':2}
    return sorted(unique,key=lambda p:(priority.get(p.name,3),p.relative_to(output).as_posix()))


def capture(path, work, *, chunk_bytes=CHUNK_BYTES):
    """Freeze one file as bounded exact-byte chunks, no whole-file size gate."""
    path=Path(path);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(str(path).encode()).hexdigest()
    receipt=work/(key+'.json')
    prior=read(receipt,{})
    before=source_signature(path)
    if prior.get('signature')==before and all((work/p['local']).is_file() for p in prior.get('parts',[])):
        return prior
    pieces=[];whole=hashlib.sha256();offset=0
    with path.open('rb') as handle:
        opened=signature(os.fstat(handle.fileno()))
        if before!=opened: raise RuntimeError('Source changed before capture')
        while True:
            block=handle.read(chunk_bytes)
            if not block:break
            digest=hashlib.sha256(block).hexdigest();local=digest+'.chunk'
            destination=work/local
            if not destination.is_file():
                temp=destination.with_name(destination.name+'.tmp-'+str(os.getpid()))
                temp.write_bytes(block);os.replace(temp,destination)
            pieces.append({'local':local,'sha256':digest,'bytes':len(block),'byte_offset':offset})
            offset+=len(block);whole.update(block)
        if signature(os.fstat(handle.fileno()))!=before or source_signature(path)!=before:
            raise RuntimeError('Source changed during capture; retry later')
    value={'source':str(path),'signature':before,'bytes':offset,'sha256':whole.hexdigest(),'parts':pieces}
    atomic(receipt,value)
    return value


def csv_views(path, *, chunk_bytes=CHUNK_BYTES):
    """Complete row-preserving CSV parts with repeated header, bounded by a row."""
    old=csv.field_size_limit();csv.field_size_limit(sys.maxsize)
    try:
        with Path(path).open('r',encoding='utf-8-sig',newline='') as handle:
            reader=csv.reader(handle,strict=True);header=next(reader,[])
            def row_bytes(row):
                buf=io.StringIO(newline='');csv.writer(buf,lineterminator='\n').writerow(row)
                return buf.getvalue().encode()
            prefix=row_bytes(header);pending=[];size=len(prefix);count=offset=0
            for row in reader:
                block=row_bytes(row)
                if pending and size+len(block)>chunk_bytes:
                    yield prefix+b''.join(pending),offset,count
                    offset+=count;pending=[];size=len(prefix);count=0
                pending.append(block);size+=len(block);count+=1
            if pending or offset==0:yield prefix+b''.join(pending),offset,count
    finally:csv.field_size_limit(old)


def publish_paper_file(gh, output, path, work, old=None):
    frozen=capture(path,work/'capture')
    if old and old.get('sha256')==frozen['sha256']:return {**old,'signature':frozen['signature']}
    rel=path.relative_to(output).as_posix();base=PREFIX+'/paper-data/files/'+frozen['sha256']+'/'+rel
    parts=[]
    for index,row in enumerate(frozen['parts']):
        block=(work/'capture'/row['local']).read_bytes()
        if hashlib.sha256(block).hexdigest()!=row['sha256']:raise RuntimeError('Captured evidence chunk changed; retain source and retry')
        # Base64 gives exact original byte reconstruction even if a UTF-8 codepoint
        # crosses a chunk boundary. Complete human-readable CSV parts are separate.
        target=base+'/raw-%05d.b64'%index
        content=base64.b64encode(block)+b'\n'
        gh.git_publish({target:content},'Paper3 R9 exact evidence chunk')
        parts.append({k:v for k,v in row.items() if k!='local'}|{'path':target,'encoding':'base64'})
    views=[]
    if path.suffix.lower()=='.csv':
        for index,(block,offset,rows) in enumerate(csv_views(path)):
            target=base+'/rows-%05d.csv'%index
            gh.git_publish({target:block},'Paper3 R9 complete CSV rows')
            views.append({'path':target,'sha256':hashlib.sha256(block).hexdigest(),'data_row_offset':offset,'data_rows':rows,'header_repeated':True})
        if source_signature(path)!=frozen['signature']:raise RuntimeError('CSV changed while rendering; retry')
    elif frozen['bytes']<=CHUNK_BYTES:
        block=b''.join((work/'capture'/p['local']).read_bytes() for p in frozen['parts'])
        # Exact original small JSON/report remains directly readable through GitHub.
        target=base+'/original'+path.suffix
        gh.git_publish({target:block},'Paper3 R9 readable evidence file')
        views.append({'path':target,'sha256':frozen['sha256'],'bytes':frozen['bytes']})
    record={'source_path':rel,'sha256':frozen['sha256'],'bytes':frozen['bytes'],'signature':frozen['signature'],'parts':parts,'readable_views':views,'statistics_recomputed':False}
    gh.git_publish({base+'/manifest.json':encoded(record)},'Paper3 R9 evidence file manifest')
    return record


def report_completion(output):
    plan=read(Path(output)/'plan.json',{})
    tasks=plan.get('tasks',[])
    complete=[]
    for task in tasks:
        key=task.get('key','')
        if Path(key).name!=key or not key:continue
        p=Path(output)/key/'result.json';row=read(p,{})
        if row.get('status')=='COMPLETE' and row.get('recipe_sha256')==plan.get('recipe_sha256'):
            complete.append((task,p))
    report=read(Path(output)/'reports/report.json',{})
    report_identity=report.get('recipe_sha256',report.get('provenance',{}).get('recipe_sha256'))
    scientific_complete=(bool(tasks) and len(complete)==len(tasks) and report.get('status')=='COMPLETE'
        and report_identity==plan.get('recipe_sha256')
        and report.get('complete_tasks')==len(tasks) and report.get('total_tasks')==len(tasks))
    return plan,complete,scientific_complete


def paper_cycle(gh,output,work):
    output=Path(output);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    state=read(work/'state.json',{'files':{}})
    for path in paper_sources(output):
        rel=path.relative_to(output).as_posix();prior=state['files'].get(rel)
        if prior and prior.get('signature')==source_signature(path):continue
        state['files'][rel]=publish_paper_file(gh,output,path,work,prior)
        atomic(work/'state.json',state)
    plan,complete,scientific_complete=report_completion(output)
    current_paths={p.relative_to(output).as_posix() for p in paper_sources(output)}
    records=[row for key,row in sorted(state['files'].items()) if key in current_paths]
    if any(source_signature(output/r['source_path'])!=r['signature'] for r in records):
        raise RuntimeError('Report changed during paper publication; retry current report without training changes')
    manifest={'schema_version':'paper3-r9-paper-evidence-v1','round':9,'files':records,
        'complete_tasks':len(complete),'total_tasks':len(plan.get('tasks',[])),
        'scientific_completion_claimed':scientific_complete,'trial_prediction_tree_scanned':False,
        'source_reports_may_be_partial':not scientific_complete,'statistics_recomputed':False,
        'reconstruction':'For each source file, sort raw parts by byte_offset, base64-decode and concatenate; verify source bytes/SHA256. CSV row views repeat the header and preserve all field values/order.'}
    snapshot=hashlib.sha256(encoded(manifest)).hexdigest()
    if state.get('published_snapshot')==snapshot:return state
    path=PREFIX+'/paper-data/'+snapshot+'/manifest.json'
    pointer={'round':9,'status':'PAPER_REPORT_FILES_UPLOADED' if any(r['source_path'].startswith('reports/') for r in records) else 'PAPER_FILES_PARTIAL',
        'snapshot_id':snapshot,'manifest':path,'files':len(records),'complete_tasks':len(complete),
        'total_tasks':len(plan.get('tasks',[])),'scientific_completion_claimed':scientific_complete,'full_raw_results_uploaded':False}
    text='# 第九轮 · 论文分析数据\n\n[最新清单](PAPER_DATA_LATEST.json)。每个原文件含完整字节分片与SHA；CSV另有完整行视图。逐参与者/会话小表随报告生成即上传。\n\n训练完成、报告上传、完整备份分别记账；报告未完成时仅为部分证据。\n'
    commit=gh.git_publish({path:encoded(manifest),PREFIX+'/PAPER_DATA_LATEST.json':encoded(pointer),PREFIX+'/PAPER_DATA_ZH.md':text.encode()},'Publish Round9 complete paper-evidence index')
    state.update(published_snapshot=snapshot,status='COMPLETE' if scientific_complete else 'UPLOADING',commit=commit)
    atomic(work/'state.json',state);emit('ROUND_9_PAPER_DATA_PUBLISHED',snapshot=snapshot,files=len(records),complete_tasks=len(complete),training_controlled=False)
    return state


def task_inventory(output,key):
    base=Path(output)/key
    if base.is_symlink():raise ValueError('Scientific task folder is a symlink')
    for directory,dirs,files in os.walk(base,followlinks=False):
        folder=Path(directory)
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (folder/d).is_symlink())
        for name in sorted(files):
            path=folder/name
            if name not in EXCLUDED and not path.is_symlink() and path.is_file():yield path


class _PartWriter:
    def __init__(self, directory, round_number, part_bytes):
        self.directory = Path(directory)
        self.round_number = round_number
        self.part_bytes = part_bytes
        self.parts = []
        self.handle = None
        self.size = self.position = 0
        self.hasher = None

    def _open(self):
        name = 'round-%02d.tar.gz.part-%05d' % (self.round_number, len(self.parts) + 1)
        self.name = name
        self.temp = self.directory / (name + '.tmp')
        self.handle = self.temp.open('wb')
        self.size = 0
        self.hasher = hashlib.sha256()

    def _finish(self):
        if self.handle is None:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        target = self.directory / self.name
        os.replace(self.temp, target)
        self.parts.append({'name': self.name, 'bytes': self.size,
                           'sha256': self.hasher.hexdigest()})
        self.handle = None

    def write(self, data):
        view = memoryview(data)
        consumed = 0
        while consumed < len(view):
            if self.handle is None:
                self._open()
            end = min(len(view), consumed + self.part_bytes - self.size)
            chunk = view[consumed:end]
            self.handle.write(chunk)
            self.hasher.update(chunk)
            amount = len(chunk)
            consumed += amount
            self.size += amount
            self.position += amount
            if self.size == self.part_bytes:
                self._finish()
        return consumed

    def flush(self):
        if self.handle is not None:
            self.handle.flush()

    def tell(self):
        return self.position

    def close(self):
        self._finish()


class _HashReader:
    def __init__(self,handle):self.handle=handle;self.digest=hashlib.sha256();self.bytes=0
    def read(self,size=-1):
        block=self.handle.read(size);self.digest.update(block);self.bytes+=len(block);return block


def pack_group(output,paths,work,key,identity,*,part_bytes=ARCHIVE_BYTES):
    """One deterministic tar.gz per completed task/report snapshot, bounded parts.

    A durable packed manifest is committed only after every input has a stable
    descriptor and SHA. Interrupted network transfers reuse these complete parts.
    """
    output=Path(output);work=Path(work)
    folder=work/'groups'/hashlib.sha256(key.encode()).hexdigest()/identity
    folder.mkdir(parents=True,exist_ok=True)
    prior=read(folder/'packed.json',{})
    if prior and all((folder/p['name']).is_file() for p in prior.get('parts',[])):
        return folder,prior
    paths=sorted(paths,key=lambda path:path.relative_to(output).as_posix())
    records=[];writer=_PartWriter(folder,9,part_bytes)
    try:
        with gzip.GzipFile(filename='',mode='wb',fileobj=writer,compresslevel=1,mtime=0) as compressed:
            with tarfile.open(mode='w|',fileobj=compressed,format=tarfile.PAX_FORMAT) as archive:
                for index,path in enumerate(paths):
                    before=source_signature(path);relative=path.relative_to(output).as_posix()
                    with path.open('rb') as handle:
                        if signature(os.fstat(handle.fileno()))!=before:raise RuntimeError('Archive input changed before read')
                        info=tarfile.TarInfo(relative);info.size=before[2];info.mode=0o600;info.mtime=0
                        info.uid=info.gid=0;info.uname=info.gname=''
                        reader=_HashReader(handle);archive.addfile(info,reader)
                        if reader.bytes!=before[2] or signature(os.fstat(handle.fileno()))!=before or source_signature(path)!=before:
                            raise RuntimeError('Archive input changed while packing')
                    records.append({'source_path':relative,'bytes':reader.bytes,'sha256':reader.digest.hexdigest(),'signature':before})
                    if index%100==0:emit('ROUND_9_ARCHIVE_PACK_PROGRESS',group=key,files=index+1,total_files=len(paths),training_controlled=False)
        writer.close()
    except BaseException:
        writer.close()
        raise
    manifest={'key':key,'identity':identity,'compression':'deterministic gzip level1, tar PAX','parts':writer.parts,'files':records,
        'source_bytes':sum(row['bytes'] for row in records),'archive_bytes':sum(row['bytes'] for row in writer.parts)}
    atomic(folder/'packed.json',manifest)
    return folder,manifest


def archive_group(gh,release,output,paths,work,key,identity):
    # GitHub caps assets per Release. Spread finite R9 task groups across 64
    # deterministic releases; only buckets actually used are created.
    if release is None:
        bucket=int(hashlib.sha256(key.encode()).hexdigest()[:2],16)%64
        release=gh.ensure_release('paper3-r9-archive-b%02d'%bucket,'Paper3 Round9 scientific archive bucket %02d'%bucket,
            'Aggregated COMPLETE task tar.gz parts. Per-file hashes and part URLs are indexed under results/paper3/round-09/.')
    folder,packed=pack_group(output,paths,work,key,identity)
    parts=[]
    for index,item in enumerate(packed['parts']):
        receipt=work/'parts'/(item['sha256']+'.json');previous=read(receipt,{})
        if previous.get('verified') is not True:
            uploaded=gh.upload_part(release,folder/item['name'],'r9-'+item['sha256']+'.tar.gz.part',item['sha256'])
            previous={**uploaded,'verified':True,'release_id':release['id']};atomic(receipt,previous)
        parts.append({'sha256':item['sha256'],'bytes':item['bytes'],'index':index,'url':previous['url'],'asset_id':previous['id']})
    manifest={'schema_version':'paper3-r9-scientific-archive-group-v1','key':key,
        'format':'Concatenate parts in index order into one .tar.gz; extract archive; verify every file bytes/SHA256.',
        'parts':parts,'files':[{k:v for k,v in row.items() if k!='signature'} for row in packed['files']],
        'source_bytes':packed['source_bytes'],'archive_bytes':packed['archive_bytes']}
    content_sha=hashlib.sha256(encoded(manifest)).hexdigest();target=PREFIX+'/archive-groups/'+content_sha+'/manifest.json'
    ack=read(folder/'published.json',{})
    if ack.get('manifest')!=target:
        commit=gh.git_publish({target:encoded(manifest)},'Publish Round9 aggregated scientific task archive')
        atomic(folder/'published.json',{'manifest':target,'commit':commit})
    return {'key':key,'identity':identity,'manifest':target,'files':len(manifest['files']),'source_bytes':packed['source_bytes'],
        'archive_bytes':packed['archive_bytes'],'archive_parts':len(parts),'capture_files':packed['files']}


def archive_cycle(gh,output,work):
    output=Path(output);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    state=read(work/'state.json',{'tasks':{}})
    plan,complete,scientific_complete=report_completion(output)
    if not plan:return state
    release=None # Per-group deterministic bucket avoids one Release asset cap.
    for task,receipt in complete:
        identity=hashlib.sha256(receipt.read_bytes()).hexdigest();key=task['key']
        if state['tasks'].get(key,{}).get('identity')==identity:continue
        native=output/key/'.lock'
        if native.exists():
            with native.open('a') as task_lock:
                try:fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
        # Each immutable COMPLETE task is inventoried/packed once. A lost network
        # acknowledgement retries existing archive parts, never all trial files.
        cache=work/'group-inputs'/hashlib.sha256(key.encode()).hexdigest()/(identity+'.json')
        inputs=read(cache,{})
        if inputs:
            paths=[output/name for name in inputs['paths']]
        else:
            paths=list(task_inventory(output,key));atomic(cache,{'paths':[p.relative_to(output).as_posix() for p in paths]})
        group=archive_group(gh,release,output,paths,work,key,identity)
        state['tasks'][key]={k:v for k,v in group.items() if k!='capture_files'}
        atomic(work/'state.json',state)
        emit('ROUND_9_SCIENTIFIC_TASK_ARCHIVED',task=key,archived_tasks=len(state['tasks']),archive_parts=state['tasks'][key]['archive_parts'],training_controlled=False)
    paths=paper_sources(output)
    stamps=[{'path':p.relative_to(output).as_posix(),'signature':source_signature(p)} for p in paths]
    report_identity=hashlib.sha256(encoded(stamps)).hexdigest()
    if state.get('reports',{}).get('identity')!=report_identity:
        state['reports']=archive_group(gh,release,output,paths,work,'plan-and-reports',report_identity)
        atomic(work/'state.json',state)
    # Report regeneration may atomically rewrite/delete shards. Never mark an old
    # partial capture as the final report; completed training is unaffected.
    current=[{'path':p.relative_to(output).as_posix(),'signature':source_signature(p)} for p in paper_sources(output)]
    if current!=stamps:raise RuntimeError('Report changed during archive publication; retry latest report')
    scientific_complete=scientific_complete and all(state['tasks'].get(t['key'],{}).get('identity')==hashlib.sha256(p.read_bytes()).hexdigest() for t,p in complete)
    groups=[{k:v for k,v in value.items() if k!='capture_files'} for _,value in sorted(state['tasks'].items())]
    groups.append({k:v for k,v in state['reports'].items() if k!='capture_files'})
    manifest={'schema_version':'paper3-r9-incremental-scientific-archive-v2','round':9,
        'complete_tasks':len(complete),'archived_tasks':len(state['tasks']),'total_tasks':len(plan.get('tasks',[])),
        'scientific_completion_claimed':scientific_complete,'groups':groups,
        'source_files':sum(group['files'] for group in groups),'archive_parts':sum(group['archive_parts'] for group in groups),
        'scope':'All regular scientific files under COMPLETE task folders plus current plan/reports: predictions, validation, model/optimizer/RNG checkpoints if produced. Excludes runtime/logs/credentials/raw datasets/shared source models/venv. In-flight tasks archived on completion.',
        'restore':'Read each group manifest; download and verify parts, concatenate in index order as tar.gz, extract, verify every source file bytes/SHA256.'}
    snapshot=hashlib.sha256(encoded(manifest)).hexdigest()
    if state.get('published_snapshot')==snapshot:return state
    path=PREFIX+'/final/'+snapshot+'/manifest.json'
    pointer={'round':9,'snapshot_id':snapshot,'manifest':path,'status':'FULL_DECLARED_SCIENTIFIC_OUTPUT_UPLOADED' if scientific_complete else 'PARTIAL_SCIENTIFIC_OUTPUT_UPLOADED',
        'scientific_completion_claimed':scientific_complete,'files':manifest['source_files'],'archive_parts':manifest['archive_parts'],'archived_tasks':len(state['tasks']),'total_tasks':len(plan.get('tasks',[])),
        'full_raw_results_uploaded':scientific_complete,'scope':manifest['scope']}
    uploads={path:encoded(manifest),PREFIX+'/ARCHIVE_PROGRESS.json':encoded(pointer)}
    if scientific_complete:uploads[PREFIX+'/LATEST.json']=encoded(pointer)
    commit=gh.git_publish(uploads,'Publish Round9 aggregated scientific-output archive index')
    state.update(published_snapshot=snapshot,status='COMPLETE' if scientific_complete else 'UPLOADING',commit=commit)
    atomic(work/'state.json',state);emit('ROUND_9_ARCHIVE_INDEX_PUBLISHED',snapshot=snapshot,complete=scientific_complete,training_controlled=False)
    return state


def run_channel(args, *, gh_factory=None, stop=None):
    operational=Path(args.operational);operational.mkdir(parents=True,exist_ok=True)
    work=operational/('upload-'+args.channel);work.mkdir(exist_ok=True)
    stop=stop or threading.Event()
    with (operational/(args.channel+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:emit('ROUND_9_UPLOAD_EXISTING_PROCESS_REUSED',channel=args.channel);return 0
        cycle=paper_cycle if args.channel=='paper' else archive_cycle
        while not stop.is_set():
            try:
                token=upload_discover_token(args.credentials)
                gh=(gh_factory or GitHub)(token,emit,cache_path=work/'blobs.json')
                state=cycle(gh,Path(args.output),work)
                if state.get('status')=='COMPLETE':return 0
            except Exception as error:
                atomic(work/'retry.json',{'status':'RETRYING','error_type':type(error).__name__,'training_controlled':False,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
                emit('ROUND_9_UPLOAD_RETRY',channel=args.channel,error_type=type(error).__name__,training_controlled=False)
            if args.once:return 0
            stop.wait(args.interval)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',choices=('paper','archive'),required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--operational',required=True)
    parser.add_argument('--credentials')
    parser.add_argument('--interval',type=float,default=180)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    return run_channel(args,stop=stop)

if __name__=='__main__':raise SystemExit(main())
