#!/usr/bin/env python3
"""Offline R11 I/O recovery. Default installs only; --repair --submit resumes failed slots."""
from __future__ import annotations
import argparse,base64,fcntl,hashlib,io,json,os,stat,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath
PAYLOAD_B85='__PAYLOAD_B85__'
PAYLOAD_SHA256='__PAYLOAD_SHA256__'
MANIFEST_SHA256='__MANIFEST_SHA256__'


def sha(data):return hashlib.sha256(data).hexdigest()
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
def emit(event,**fields):print(json.dumps(dict(event=event,**fields),ensure_ascii=False),flush=True)
def safe(name):
 p=PurePosixPath(name)
 if (not name or p.is_absolute() or p.as_posix()!=name or '\\' in name or any(x in ('','.','..') for x in name.split('/'))
     or any(x.lower() in ('private','.git','.ssh','secrets','credentials') for x in p.parts)):
  raise ValueError('UNSAFE_REPAIR_PACKAGE_PATH')
 return p

def install(home):
 raw=base64.b85decode(PAYLOAD_B85)
 if sha(raw)!=PAYLOAD_SHA256:raise ValueError('REPAIR_PACKAGE_SHA256_DIFFERS')
 z=zipfile.ZipFile(io.BytesIO(raw));manifest_bytes=z.read('MANIFEST.json')
 if sha(manifest_bytes)!=MANIFEST_SHA256:raise ValueError('REPAIR_MANIFEST_SHA256_DIFFERS')
 m=json.loads(manifest_bytes);files=m['files'];names=z.namelist()
 if len(names)!=len(set(names)) or set(names)!={'MANIFEST.json',*files}:raise ValueError('REPAIR_MANIFEST_COVERAGE_DIFFERS')
 if sum(i['bytes'] for i in files.values())>32*1024*1024:raise ValueError('REPAIR_CODE_SIZE_LIMIT')
 for name,row in files.items():
  safe(name);info=z.getinfo(name);mode=(info.external_attr>>16)&0xffff
  if info.is_dir() or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or info.file_size!=row['bytes']:
   raise ValueError('REPAIR_MEMBER_TYPE_OR_SIZE_DIFFERS')
 base=Path(home).expanduser().absolute()/'paper3-r11-io-repair'
 if base.is_symlink():raise ValueError('REPAIR_INSTALL_BASE_IS_SYMLINK')
 base.mkdir(parents=True,exist_ok=True,mode=0o700);dest=base/PAYLOAD_SHA256[:20]
 def verify(folder):
  if folder.is_symlink() or not folder.is_dir():return False
  if not (folder/'MANIFEST.json').is_file() or sha((folder/'MANIFEST.json').read_bytes())!=MANIFEST_SHA256:return False
  for name,row in files.items():
   p=folder/name
   if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p.read_bytes())!=row['sha256']:return False
  return True
 lock_path=base/'install.lock'
 if lock_path.is_symlink():raise ValueError('REPAIR_INSTALL_LOCK_IS_SYMLINK')
 with lock_path.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if dest.exists():
   if not verify(dest):raise ValueError('EXISTING_IMMUTABLE_REPAIR_PACKAGE_DIFFERS')
  else:
   stage=Path(tempfile.mkdtemp(prefix='.install-',dir=base))
   try:
    for name,row in files.items():
     data=z.read(name)
     if sha(data)!=row['sha256']:raise ValueError('REPAIR_MEMBER_SHA256_DIFFERS')
     p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o600)
    (stage/'MANIFEST.json').write_bytes(manifest_bytes)
    if not verify(stage):raise ValueError('STAGED_REPAIR_VERIFY_FAILED')
    os.replace(stage,dest)
   finally:
    if stage.exists():
     import shutil
     shutil.rmtree(stage)
  receipt=dict(package=str(dest),payload_sha256=PAYLOAD_SHA256,manifest_sha256=MANIFEST_SHA256,
   files=len(files),scientific_package_files_changed=False,scientific_plan_changed=False,jobs_submitted_during_install=0)
  temp=base/('CURRENT.tmp-'+str(os.getpid()));temp.write_bytes(encoded(receipt));os.replace(temp,base/'CURRENT.json')
 emit('R11_IO_REPAIR_INSTALLED',**receipt);return dest


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
 p.add_argument('--home',type=Path,default=Path.home());p.add_argument('--install',action='store_true')
 p.add_argument('--check',action='store_true');p.add_argument('--repair',action='store_true');p.add_argument('--submit',action='store_true')
 p.add_argument('--python',default='/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python')
 p.add_argument('--plan',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3/runs/r11_campaign/plan.json'))
 p.add_argument('--package',type=Path);p.add_argument('--work',type=Path)
 args=p.parse_args(argv)
 if args.check and args.repair:raise ValueError('CHOOSE_CHECK_OR_REPAIR')
 if args.submit and not args.repair:raise ValueError('SUBMIT_REQUIRES_EXPLICIT_REPAIR')
 package=install(args.home)
 if not args.check and not args.repair:
  emit('R11_IO_REPAIR_CODE_READY_NO_ACTION',next_operation='--check or --repair --submit',scientific_workers_stopped=0);return 0
 command=[str(Path(args.python).expanduser().absolute()),'-I','-S','-B','-u',str(package/'supervisor.py'),
  'start' if args.repair else 'check','--plan',str(args.plan)]
 if args.package:command+=['--package',str(args.package)]
 if args.work:command+=['--work',str(args.work)]
 if args.submit:command+=['--submit']
 command+=['--python',str(Path(args.python).expanduser().absolute())]
 import subprocess
 return subprocess.run(command,check=False).returncode
if __name__=='__main__':raise SystemExit(main())
