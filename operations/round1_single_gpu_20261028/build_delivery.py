"""Create one offline entry; no dependencies or GPU checks on school login."""
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parent
MEMBERS = ('launch.py', 'single_gpu.py', 'worker_runner.py', 'receipt_projection.py',
           'finalization_cache.py', 'observer.py', 'upload_transport.py',
           'results_observer.py', 'auto_summary.py', 'summary_transport.py')
TEMPLATE = '''#!/usr/bin/env python3
"""Resume the original R1 experiment on one GPU / four CPUs / 64G."""
from __future__ import annotations
import base64, fcntl, hashlib, io, json, os, sys, tempfile, zipfile
from pathlib import Path
PAYLOAD = "__PAYLOAD__"
PAYLOAD_SHA256 = "__HASH__"
MEMBERS = __MEMBERS__

def main():
    print(json.dumps(dict(event='R1_SINGLE_GPU_BOOTSTRAP_STARTED',
        message='程序已启动；复用学校现有环境，准备单卡续跑管理程序。'), ensure_ascii=False), flush=True)
    args = sys.argv[1:]
    home = Path(args[args.index('--home')+1]).expanduser().resolve() if '--home' in args else Path.home()
    base = home/'paper3-r1-single-gpu'
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = base64.b85decode(PAYLOAD)
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
        raise ValueError('SINGLE_GPU_PAYLOAD_SHA256_DIFFERS')
    target = base/PAYLOAD_SHA256[:20]
    with (base/'install.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        archive = zipfile.ZipFile(io.BytesIO(raw))
        manifest = json.loads(archive.read('MANIFEST.json'))
        if set(archive.namelist()) != {*MEMBERS, 'MANIFEST.json'} or set(manifest['files']) != set(MEMBERS):
            raise ValueError('SINGLE_GPU_PACKAGE_MEMBER_SET_DIFFERS')
        target.mkdir(exist_ok=True, mode=0o700)
        for name in MEMBERS:
            data = archive.read(name); expected = manifest['files'][name]
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('SINGLE_GPU_MEMBER_SHA256_DIFFERS: '+name)
            path = target/name
            if path.exists():
                if path.is_symlink() or path.read_bytes() != data:
                    raise ValueError('EXISTING_SINGLE_GPU_RELEASE_DIFFERS')
            else:
                fd, temporary = tempfile.mkstemp(prefix='.install-', dir=target)
                try:
                    with os.fdopen(fd,'wb') as out:
                        out.write(data); out.flush(); os.fsync(out.fileno())
                    os.replace(temporary,path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
    print(json.dumps(dict(event='R1_SINGLE_GPU_PACKAGE_READY',package=str(target),
        dependency_installations=0,gpu_preflight_jobs=0),ensure_ascii=False),flush=True)
    sys.path.insert(0,str(target))
    import launch
    return launch.main(args)

if __name__=='__main__':raise SystemExit(main())
'''


def build():
    files = {name:(ROOT/name).read_bytes() for name in MEMBERS}
    for name, data in files.items():
        compile(data, name, 'exec')
    manifest = dict(schema_version='paper3-r1-single-gpu-delivery-v1', files={
        name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in files.items()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name, data in {**files,'MANIFEST.json':(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()}.items():
            entry = zipfile.ZipInfo(name,date_time=(2026,10,28,0,0,0))
            entry.compress_type = zipfile.ZIP_DEFLATED; entry.external_attr = 0o100600 << 16
            archive.writestr(entry,data)
    payload = buffer.getvalue(); digest = hashlib.sha256(payload).hexdigest()
    source = TEMPLATE.replace('__PAYLOAD__',base64.b85encode(payload).decode()).replace('__HASH__',digest).replace('__MEMBERS__',repr(MEMBERS)).encode()
    compile(source,'START-PAPER3-R1-1GPU.py','exec')
    out=ROOT/'delivery';out.mkdir(exist_ok=True)
    (out/'START-PAPER3-R1-1GPU.py').write_bytes(source)
    (out/'paper3-r1-single-gpu.zip').write_bytes(payload)
    release=dict(manifest,entry='START-PAPER3-R1-1GPU.py',entry_bytes=len(source),
        entry_sha256=hashlib.sha256(source).hexdigest(),payload_sha256=digest,
        resources=dict(jobs=1,gpus=1,cpus=4,memory='64G',partition='hpc_gpu'),
        changes_scientific_source_files=False,dependencies_installed=0,real_school_gpu_test=False)
    (out/'RELEASE.json').write_text(json.dumps(release,sort_keys=True,indent=2)+'\n')
    return release


if __name__=='__main__':print(json.dumps(build(),indent=2))
