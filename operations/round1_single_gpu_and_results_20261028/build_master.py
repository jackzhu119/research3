from pathlib import Path
import base64,hashlib,io,json,zipfile

ROOT=Path(__file__).resolve().parent
SOURCES={
 'START-PAPER3-R1-1GPU.py':Path('/workspace/paper3-r1-single-gpu-20261028/delivery/START-PAPER3-R1-1GPU.py'),
 'START-PAPER3-MISSING-RESULTS.py':Path('/workspace/paper3-missing-results-upload-20261028/START-PAPER3-MISSING-RESULTS.py'),
}
TEMPLATE='''#!/usr/bin/env python3
"""Start R1 single-GPU continuation and independent missing-results delivery."""
from __future__ import annotations
import base64,fcntl,hashlib,io,json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
PAYLOAD="__PAYLOAD__"
PAYLOAD_SHA256="__HASH__"
FILES=__FILES__

def main():
 print('程序已启动：准备第一轮单卡续跑和独立结果补传。',flush=True)
 args=sys.argv[1:]
 if any(v not in ('--apply','--check','--uploads-only','--r1-only') for v in args):
  raise ValueError('SUPPORTED_OPTIONS: --apply --check --uploads-only --r1-only')
 base=Path.home()/'paper3-r1-and-upload';base.mkdir(parents=True,exist_ok=True,mode=0o700)
 raw=base64.b85decode(PAYLOAD)
 if hashlib.sha256(raw).hexdigest()!=PAYLOAD_SHA256:raise ValueError('COMBINED_PAYLOAD_SHA256_DIFFERS')
 target=base/PAYLOAD_SHA256[:20]
 with (base/'install.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);target.mkdir(exist_ok=True,mode=0o700)
  archive=zipfile.ZipFile(io.BytesIO(raw))
  if set(archive.namelist())!=set(FILES):raise ValueError('COMBINED_MEMBER_SET_DIFFERS')
  for name,expected in FILES.items():
   data=archive.read(name)
   if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('COMBINED_MEMBER_SHA256_DIFFERS')
   path=target/name
   if path.exists():
    if path.is_symlink() or path.read_bytes()!=data:raise ValueError('INSTALLED_COMBINED_ENTRY_DIFFERS')
   else:
    fd,tmp=tempfile.mkstemp(prefix='.install-',dir=target)
    try:
     with os.fdopen(fd,'wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
     os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
 tasks=[]
 if '--uploads-only' not in args:tasks.append(('R1','START-PAPER3-R1-1GPU.py','--check' if '--check' in args else '--apply'))
 if '--r1-only' not in args:tasks.append(('RESULTS','START-PAPER3-MISSING-RESULTS.py','--check' if '--check' in args else '--start'))
 results=[]
 for name,entry,mode in tasks:
  print('开始：'+name+' '+mode,flush=True)
  try:
   result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(target/entry),mode],stdin=subprocess.DEVNULL,timeout=120)
   results.append(dict(component=name,exit_code=result.returncode))
  except subprocess.TimeoutExpired:
   results.append(dict(component=name,exit_code=124,startup_status='UNCONFIRMED_READ_COMPONENT_LOG'))
 value=dict(event='PAPER3_COMBINED_LAUNCH_FINISHED',components=results,package=str(target),
   note='后台启动回执不代表GPU已运行或上传已完成；请看各组件状态。')
 print(json.dumps(value,ensure_ascii=False),flush=True)
 return 0 if all(row['exit_code']==0 for row in results) else 1

if __name__=='__main__':raise SystemExit(main())
'''


def build():
 files={name:path.read_bytes() for name,path in SOURCES.items()}
 buffer=io.BytesIO()
 with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
  for name,data in files.items():
   entry=zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o100600<<16
   archive.writestr(entry,data)
 payload=buffer.getvalue();digest=hashlib.sha256(payload).hexdigest()
 manifests={name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
 source=TEMPLATE.replace('__PAYLOAD__',base64.b85encode(payload).decode()).replace('__HASH__',digest).replace('__FILES__',repr(manifests)).encode()
 compile(source,'START-PAPER3-R1-AND-UPLOAD.py','exec')
 (ROOT/'START-PAPER3-R1-AND-UPLOAD.py').write_bytes(source)
 manifest=dict(entry='START-PAPER3-R1-AND-UPLOAD.py',bytes=len(source),sha256=hashlib.sha256(source).hexdigest(),payload_sha256=digest,components=manifests)
 (ROOT/'RELEASE.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 return manifest


if __name__=='__main__':print(json.dumps(build(),indent=2))
