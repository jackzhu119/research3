from pathlib import Path
import base64,hashlib,io,json,zipfile

ROOT=Path(__file__).resolve().parent
TEMPLATE='''#!/usr/bin/env python3
"""Fix R1's dynamic digest index classification inside its existing allocation."""
from __future__ import annotations
import base64,fcntl,hashlib,io,json,os,sys,tempfile,zipfile
from pathlib import Path
PAYLOAD="__PAYLOAD__"
PAYLOAD_SHA256="__SHA__"

def main():
 print('修复程序已启动：保留现有作业，修正对照校验缓存索引误报。',flush=True)
 args=sys.argv[1:]
 home=Path(args[args.index('--home')+1]).expanduser().resolve() if '--home' in args else Path.home()
 base=home/'paper3-r1-finalization-hotfix';base.mkdir(parents=True,exist_ok=True,mode=0o700)
 raw=base64.b85decode(PAYLOAD)
 if hashlib.sha256(raw).hexdigest()!=PAYLOAD_SHA256:raise ValueError('HOTFIX_PAYLOAD_SHA256_DIFFERS')
 target=base/PAYLOAD_SHA256[:20]
 with (base/'install.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  archive=zipfile.ZipFile(io.BytesIO(raw));manifest=json.loads(archive.read('MANIFEST.json'))
  if set(manifest['files'])!={'hotfix.py','finalization_cache.py'} or set(archive.namelist())!={*manifest['files'],'MANIFEST.json'}:raise ValueError('HOTFIX_MEMBER_SET_DIFFERS')
  target.mkdir(exist_ok=True,mode=0o700)
  for name,record in manifest['files'].items():
   data=archive.read(name)
   if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('HOTFIX_MEMBER_SHA256_DIFFERS')
   path=target/name
   if path.exists():
    if path.is_symlink() or path.read_bytes()!=data:raise ValueError('INSTALLED_HOTFIX_DIFFERS')
   else:
    fd,tmp=tempfile.mkstemp(prefix='.install-',dir=target)
    try:
     with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
     os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
 print(json.dumps(dict(event='R1_FINALIZATION_FIX_PACKAGE_READY',package=str(target),
   dependency_installations=0,slurm_jobs_cancelled=0,slurm_jobs_submitted=0),ensure_ascii=False),flush=True)
 sys.path.insert(0,str(target))
 import hotfix
 return hotfix.main(args)

if __name__=='__main__':raise SystemExit(main())
'''


def build():
 files={name:(ROOT/name).read_bytes() for name in ('hotfix.py','finalization_cache.py')}
 for name,data in files.items():compile(data,name,'exec')
 manifest=dict(schema_version='paper3-r1-finalization-index-hotfix-v2',files={
  name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
  for name,data in {**files,'MANIFEST.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}.items():
   entry=zipfile.ZipInfo(name,date_time=(2026,10,29,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100600<<16
   archive.writestr(entry,data)
 payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
 raw=TEMPLATE.replace('__PAYLOAD__',base64.b85encode(payload).decode()).replace('__SHA__',digest).encode()
 compile(raw,'FIX-PAPER3-R1-FINALIZATION.py','exec')
 (ROOT/'FIX-PAPER3-R1-FINALIZATION.py').write_bytes(raw)
 release=dict(manifest,entry='FIX-PAPER3-R1-FINALIZATION.py',entry_bytes=len(raw),entry_sha256=hashlib.sha256(raw).hexdigest(),
   payload_sha256=digest,slurm_allocations_cancelled=0,slurm_allocations_submitted=0,
   numerical_training_processes_signalled=0,old_immutable_package_files_changed=0,
   activation='Precisely bound CPU-only strongest-control/freeze verifier can SIGINT unwind once; existing parent naturally retries in same allocation.')
 (ROOT/'RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
 return release


if __name__=='__main__':print(json.dumps(build(),indent=2))
