#!/usr/bin/env python3
"""Install the r1-r11-results evidence uploader, then start its detached driver."""
import argparse
import base64
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zlib

PAYLOAD_SHA256='__PAYLOAD_SHA256__'
PAYLOAD_B85='__PAYLOAD_B85__'
FILE_HASHES=__FILE_HASHES__
SCHEMA='paper3-r1-r11-results-uploader-install-v1'


def sha(data):return hashlib.sha256(data).hexdigest()


def emit(event,**fields):
    print(json.dumps(dict(event=event,**fields),ensure_ascii=False),flush=True)


def manifest():
    return {'schema_version':SCHEMA,'payload_sha256':PAYLOAD_SHA256,'files':FILE_HASHES}


def verify_install(destination):
    destination=Path(destination)
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeError('Existing uploader package is not a regular install directory')
    receipt=destination/'INSTALL_MANIFEST.json'
    try:actual=json.loads(receipt.read_text())
    except (OSError,ValueError) as error:
        raise RuntimeError('Existing uploader package has no valid install receipt') from error
    if receipt.is_symlink() or actual!=manifest():
        raise RuntimeError('Existing uploader install receipt differs; original package retained')
    for name,expected in FILE_HASHES.items():
        path=destination/name
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes())!=expected:
            raise RuntimeError('Installed uploader source differs; original package retained: '+name)
    return destination


def unpack_payload():
    packed=base64.b85decode(PAYLOAD_B85)
    if sha(packed)!=PAYLOAD_SHA256:
        raise ValueError('Embedded completed-results uploader payload SHA differs')
    raw=zlib.decompress(packed)
    if len(raw)>4*1024*1024:
        raise ValueError('Unexpected completed-results uploader source size')
    document=json.loads(raw)
    if not isinstance(document,dict) or set(document)!=set(FILE_HASHES):
        raise ValueError('Uploader payload file set differs')
    files={}
    for name,encoded in document.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or relative.suffix!='.py':
            raise ValueError('Unexpected uploader source path')
        if not isinstance(encoded,str):raise ValueError('Uploader source must be UTF8 text')
        data=encoded.encode('utf-8')
        if sha(data)!=FILE_HASHES[name]:raise ValueError('Embedded uploader file SHA differs: '+name)
        compile(data,name,'exec')
        files[name]=data
    if 'delivery.py' not in files:raise ValueError('Uploader driver is absent')
    return files


def install(base=None):
    # Decode/verify the tiny public-code payload before creating any files.
    files=unpack_payload()
    base=(Path(base).expanduser() if base is not None else
          Path.home()/'paper3-r1-r11-results-uploader').resolve()
    base.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination=base/PAYLOAD_SHA256[:20]
    with (base/('.install-'+PAYLOAD_SHA256[:20]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():return verify_install(destination)
        stage=Path(tempfile.mkdtemp(prefix='.'+PAYLOAD_SHA256[:20]+'.tmp-',dir=base))
        try:
            for name,data in files.items():
                path=stage/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                with path.open('xb') as handle:
                    handle.write(data);handle.flush();os.fsync(handle.fileno())
                path.chmod(0o600)
            receipt=stage/'INSTALL_MANIFEST.json'
            with receipt.open('x') as handle:
                json.dump(manifest(),handle,sort_keys=True,indent=2);handle.write('\n')
                handle.flush();os.fsync(handle.fileno())
            receipt.chmod(0o600)
            # Ordinary same-filesystem rename is compatible with the school's
            # older kernel and NFS. The payload-specific flock serializes peers.
            os.rename(stage,destination)
            return verify_install(destination)
        finally:
            if stage.exists():shutil.rmtree(stage)


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--install-base',type=Path)
    parser.add_argument('--install-only',action='store_true')
    args,forwarded=parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    emit('R1_R11_EXPORT_BOOTSTRAP_STARTED',message='补传程序已启动；正在安装离线上传代码，不申请GPU。')
    destination=install(args.install_base)
    emit('R1_R11_EXPORT_PACKAGE_READY',package=str(destination),payload_sha256=PAYLOAD_SHA256,
         scientific_jobs_changed=0,downloads_performed=False,torch_imported=False)
    if args.install_only:return 0
    if not any(value in ('--start','--worker','--once','--check','--help','-h') for value in forwarded):
        forwarded=['--start',*forwarded]
    sys.path.insert(0,str(destination))
    driver=importlib.import_module('delivery')
    return driver.main(forwarded)


if __name__=='__main__':raise SystemExit(main())
