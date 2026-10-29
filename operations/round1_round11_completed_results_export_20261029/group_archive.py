"""Adapted tested per-task Release archive; immutable inputs and full GET proofs."""
import gzip,hashlib,io,json,os,tarfile,time
from pathlib import Path
from delivery import atomic,read,emit,encoded
ROUND=8
PREFIX='results/paper3/round-08'
ARCHIVE_BYTES=32*1024*1024
EXCLUDED={'private','runtime','logs','__pycache__','.git','.ssh','credentials','github.json','credentials.json','token','token.txt','.env','STOP','.lock','evaluation-labels.npz'}


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
    records=[];writer=_PartWriter(folder,ROUND,part_bytes)
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
                    if index%100==0:emit('ROUND_BACKFILL_ARCHIVE_PACK_PROGRESS',group=key,files=index+1,total_files=len(paths),training_controlled=False)
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
        release=gh.ensure_release('paper3-r%d-backfill-archive-b%02d'%(ROUND,bucket),'Paper3 Round%d scientific archive bucket %02d'%(ROUND,bucket),
            'Aggregated COMPLETE task tar.gz parts. Per-file hashes and part URLs are indexed under results/paper3/round-09/.')
    folder,packed=pack_group(output,paths,work,key,identity)
    parts=[]
    for index,item in enumerate(packed['parts']):
        receipt=work/'parts'/(item['sha256']+'.json');previous=read(receipt,{})
        if previous.get('verified') is not True:
            uploaded=gh.upload_part(release,folder/item['name'],'r%d-'%ROUND+item['sha256']+'.tar.gz.part',item['sha256'])
            previous={**uploaded,'verified':True,'release_id':release['id']};atomic(receipt,previous)
        parts.append({'sha256':item['sha256'],'bytes':item['bytes'],'index':index,'url':previous['url'],'asset_id':previous['id']})
    manifest={'schema_version':'paper3-backfill-scientific-archive-group-v1','key':key,
        'format':'Concatenate parts in index order into one .tar.gz; extract archive; verify every file bytes/SHA256.',
        'parts':parts,'files':[{k:v for k,v in row.items() if k!='signature'} for row in packed['files']],
        'source_bytes':packed['source_bytes'],'archive_bytes':packed['archive_bytes']}
    content_sha=hashlib.sha256(encoded(manifest)).hexdigest();target=PREFIX+'/archive-groups/'+content_sha+'/manifest.json'
    ack=read(folder/'published.json',{})
    if ack.get('manifest')!=target:
        commit=gh.git_publish({target:encoded(manifest)},'Publish Paper3 round%d scientific task archive'%ROUND)
        atomic(folder/'published.json',{'manifest':target,'commit':commit})
    return {'key':key,'identity':identity,'manifest':target,'files':len(manifest['files']),'source_bytes':packed['source_bytes'],
        'archive_bytes':packed['archive_bytes'],'archive_parts':len(parts),'capture_files':packed['files']}
