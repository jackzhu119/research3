#!/usr/bin/env python3
"""Self-contained R11 installer; default action installs code and nothing else."""
from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import zipfile

PAYLOAD_B85='__PAYLOAD_B85__'
PAYLOAD_SHA256='__PAYLOAD_SHA256__'
MANIFEST_SHA256='__MANIFEST_SHA256__'
MAX_PAYLOAD_BYTES=128*1024*1024


def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                          **fields),ensure_ascii=False),flush=True)


def sha(data):return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(data);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:Path(temporary).unlink(missing_ok=True)


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()


def safe_member(name):
    p=PurePosixPath(name)
    if (not isinstance(name,str) or not name or '\\' in name or p.is_absolute()
            or p.as_posix()!=name or any(x in ('','.','..') for x in name.split('/'))
            or any(x.lower() in ('private','.git','.ssh','.aws','secrets','credentials') or x.lower().startswith('.env') for x in p.parts)
            or p.name.lower() in ('github.json','credentials.json')):
        raise ValueError('R11 package contains an unsafe member name')
    return p


def payload():
    data=base64.b85decode(PAYLOAD_B85)
    if sha(data)!=PAYLOAD_SHA256:raise ValueError('Embedded R11 package SHA256 differs')
    archive=zipfile.ZipFile(io.BytesIO(data))
    names=archive.namelist()
    if len(names)!=len(set(names)) or 'MANIFEST.json' not in names:
        raise ValueError('R11 package contains duplicate or missing manifest members')
    info=archive.getinfo('MANIFEST.json')
    if info.file_size>4*1024*1024:raise ValueError('R11 package manifest is unexpectedly large')
    manifest_bytes=archive.read('MANIFEST.json')
    if sha(manifest_bytes)!=MANIFEST_SHA256:raise ValueError('R11 manifest SHA256 differs')
    manifest=json.loads(manifest_bytes)
    if manifest.get('schema_version')!='paper3-r11-code-package-v1':raise ValueError('R11 code package schema differs')
    files=manifest.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('R11 code package file list is empty')
    if set(names)!={'MANIFEST.json',*files}:raise ValueError('R11 package member coverage differs')
    total=0
    for name,row in files.items():
        safe_member(name);entry=archive.getinfo(name)
        mode=(entry.external_attr>>16)&0xffff
        if entry.is_dir() or (stat.S_IFMT(mode) not in (0,stat.S_IFREG)):
            raise ValueError('R11 package cannot contain links or special files')
        if (not isinstance(row,dict) or type(row.get('bytes')) is not int or row['bytes']<0
                or entry.file_size!=row['bytes'] or len(str(row.get('sha256','')))!=64):
            raise ValueError('R11 declared member identity differs')
        total+=row['bytes']
    if total>MAX_PAYLOAD_BYTES:raise ValueError('R11 package exceeds its declared code-only size bound')
    return archive,manifest,manifest_bytes


def verify_install(folder,manifest):
    folder=Path(folder)
    if folder.is_symlink() or not folder.is_dir():return False
    try:
        if sha_file(folder/'MANIFEST.json')!=MANIFEST_SHA256:return False
        for name,row in manifest['files'].items():
            path=folder/str(safe_member(name))
            if (path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']
                    or any(parent.is_symlink() for parent in path.parents)
                    or sha_file(path)!=row['sha256']):return False
        return True
    except (OSError,ValueError):return False


def install(home):
    """Decode exact files into a new content-addressed directory; no imports/jobs."""
    archive,manifest,manifest_bytes=payload()
    base=Path(home).expanduser().resolve()/'paper3-round11'
    if base.is_symlink():raise ValueError('R11 install base cannot be a symlink to an existing experiment')
    base.mkdir(parents=True,exist_ok=True)
    destination=base/PAYLOAD_SHA256[:20]
    if (base/'install.lock').is_symlink():raise ValueError('R11 install lock cannot be a symlink')
    with (base/'install.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reused=False
        if (destination.is_dir() and not destination.is_symlink()
                and destination.stat().st_uid==os.getuid() and not any(destination.iterdir())):
            destination.rmdir()  # An empty code-version placeholder contains no scientific state.
        if destination.exists():
            if not verify_install(destination,manifest):
                raise ValueError('Existing immutable R11 package differs; it was not overwritten')
            reused=True
        else:
            stage=Path(tempfile.mkdtemp(prefix='.install-',dir=base))
            try:
                for name,row in manifest['files'].items():
                    path=stage/str(safe_member(name));path.parent.mkdir(parents=True,exist_ok=True)
                    h=hashlib.sha256();count=0
                    with archive.open(name) as source,path.open('xb') as target:
                        while True:
                            block=source.read(min(1024*1024,row['bytes']-count+1))
                            if not block:break
                            count+=len(block)
                            if count>row['bytes']:raise ValueError('R11 member exceeded declared byte count')
                            h.update(block);target.write(block)
                        target.flush();os.fsync(target.fileno())
                    if count!=row['bytes'] or h.hexdigest()!=row['sha256']:
                        raise ValueError('R11 member SHA256 verification failed')
                    os.chmod(path,0o700 if name.endswith('.sh') else 0o600)
                atomic(stage/'MANIFEST.json',manifest_bytes)
                if not verify_install(stage,manifest):raise ValueError('R11 staged package verification failed')
                os.replace(stage,destination)  # Ordinary rename works on CentOS7/NFS; no renameat2.
                descriptor=os.open(base,os.O_RDONLY|os.O_DIRECTORY)
                try:
                    try:os.fsync(descriptor)
                    except OSError as error:
                        if error.errno not in (errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP):raise
                finally:os.close(descriptor)
            finally:
                if stage.exists():shutil.rmtree(stage)
        receipt=dict(schema_version='paper3-r11-installation-v1',package=str(destination),
                     payload_sha256=PAYLOAD_SHA256,manifest_sha256=MANIFEST_SHA256,
                     files=len(manifest['files']),reused=reused,installed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                     training_jobs_submitted=0,training_jobs_cancelled=0,torch_imported=False,
                     dependencies_installed=False,data_downloaded=False,scientific_outputs_changed=False)
        atomic(base/'CURRENT.json',json_bytes(receipt))
    emit('R11_INSTALL_COMPLETE',**receipt)
    return destination


def requested_command(args,package):
    modes=[args.validate,args.gpu_precheck,args.prepare is not None,args.start,args.resume,args.status,args.export]
    if args.submit and not args.resume and not args.start:modes.append(True)
    if sum(bool(x) for x in modes)>1:raise ValueError('Choose one explicit R11 operation per invocation')
    if args.start:
        command=['launch','--root',str(args.root)]
        if args.registry:command+=['--registry',str(args.registry)]
        if args.submit:command+=['--submit']
        return command+args.extra
    if args.validate:return ['validate','--cpu',*args.extra]
    if args.gpu_precheck:
        if not args.registry:raise ValueError('GPU precheck needs an explicit study registry')
        return ['gpu-precheck','--root',str(args.root),'--registry',str(args.registry),*args.extra]
    if args.prepare:
        if not args.registry:raise ValueError('Plan preparation needs an explicit study registry')
        command=['plan','--stage',args.prepare,'--root',str(args.root),'--registry',str(args.registry)]
        if args.output:command+=['--output',str(args.output)]
        return command+args.extra
    if args.resume or args.status or args.export or args.submit:
        if not args.plan:raise ValueError('This R11 operation requires an existing explicit --plan')
        mode='resume' if args.resume else 'status' if args.status else 'export' if args.export else 'submit'
        command=[mode,'--plan',str(args.plan)]
        if args.submit:command+=['--submit']
        return command+args.extra
    return None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--python',default='/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python')
    parser.add_argument('--install',action='store_true',help='Install only; this is the default')
    parser.add_argument('--validate',action='store_true',help='Explicit CPU validation; no GPU or Slurm')
    parser.add_argument('--gpu-precheck',action='store_true',help='Explicit finite GPU check in a user-owned allocation')
    parser.add_argument('--prepare',choices=('campaign','development','exploratory','confirmatory'))
    parser.add_argument('--registry',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--start',action='store_true',help='Install and prepare finite development plan; --submit additionally starts four jobs')
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true')
    parser.add_argument('--export',action='store_true');parser.add_argument('--plan',type=Path)
    parser.add_argument('extra',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.extra[:1]==['--']:args.extra=args.extra[1:]
    # Validate mode combinations before touching the filesystem.
    command=requested_command(args,None)
    package=install(args.home)
    if command is None:
        emit('R11_CODE_READY_NO_EXECUTION',package=str(package),
             next_cpu_validation='Re-run this installer with --validate',
             training_submission_requires='Explicit --start --submit or --submit --plan PATH')
        return 0
    python=Path(args.python).expanduser()
    if not python.is_file():raise ValueError('Use --python to select the existing compatible Python environment')
    entry=package/'START.py'
    if not entry.is_file():raise ValueError('R11 START.py is missing from the verified package')
    emit('R11_EXPLICIT_OPERATION_STARTED',operation=command[0],training_submission_authorized=bool(args.submit))
    import subprocess
    return subprocess.run([str(python),'-I','-S','-B','-u',str(entry),*command],check=False).returncode


if __name__=='__main__':raise SystemExit(main())
