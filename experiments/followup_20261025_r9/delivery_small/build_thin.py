#!/usr/bin/env python3
"""Build a small R9 source-only installer; keep scientific source byte identities."""
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import textwrap
import zipfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'


def build():
    spec = importlib.util.spec_from_file_location('old_r9_builder', '/workspace/paper3-round9/build_online.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    files = {p.relative_to(SOURCE).as_posix():p.read_bytes()
             for p in sorted(SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in files.items()}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name,data in files.items():
            item=zipfile.ZipInfo(name,date_time=(2000,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o100600<<16
            archive.writestr(item,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    payload=stream.getvalue()
    encoded=base64.b85encode(payload).decode()
    wrapper=old.INSTALLER.replace('Path.home() / "paper3-round9"','Path.home() / "paper3-round9-thin"')
    wrapper=wrapper.replace('parser.add_argument("--install-base", help="Override ~/paper3-round9 (useful for isolated local checks)")',
        'parser.add_argument("--install-base", help="Override ~/paper3-round9-thin")\n'
        '    parser.add_argument("--cache-home", help="Home directory holding prior round installs")\n'
        '    parser.add_argument("--cache-root", action="append", default=[], help="Existing package/data root; can repeat")\n'
        '    parser.add_argument("--data-base", action="append", default=[], help="HTTPS root for individually missing arrays; can repeat")\n'
        '    parser.add_argument("--data-archive", action="append", default=[], help="Fallback cache ZIP URL#sha256=...; only used for missing local files")')
    wrapper=wrapper.replace('    command = [sys.executable, "-I", "-S", "-B", "-u", str(destination / "START.py"), *forwarded]',
        '    sys.path.insert(0, str(destination))\n'
        '    from cache_resolution import ensure_data\n'
        '    ensure_data(destination, cache_home=args.cache_home, cache_roots=args.cache_root, data_bases=args.data_base, data_archives=args.data_archive)\n'
        '    command = [sys.executable, "-I", "-S", "-B", "-u", str(destination / "START.py"), *forwarded]')
    wrapper=wrapper.replace('__PAYLOAD_SHA256__',json.dumps(hashlib.sha256(payload).hexdigest()))
    wrapper=wrapper.replace('__FILE_HASHES__',json.dumps(hashes,sort_keys=True,indent=2))
    wrapper=wrapper.replace('__REQUIRED_FILES__',repr(list(old.REQUIRED_FILES)+['cache_resolution.py']))
    wrapper=wrapper.replace('__PAYLOAD_B85__','\n'.join('    '+repr(c) for c in textwrap.wrap(encoded,100)))
    compile(wrapper,'START-PAPER3-R9-SMALL.py','exec')
    output=HERE/'START-PAPER3-R9-SMALL.py'
    output.write_text(wrapper)
    result=dict(entrypoint=output.name,entrypoint_bytes=output.stat().st_size,
        entrypoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),payload_bytes=len(payload),
        source_bytes=sum(map(len,files.values())),files=hashes,contains_X_arrays=False,
        existing_caches_reused=True,scientific_source_changed=False)
    (HERE/'PACKAGE.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    return result


if __name__=='__main__':build()
