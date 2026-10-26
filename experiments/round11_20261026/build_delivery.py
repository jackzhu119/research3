"""Deterministic R11 source and bounded evidence package; no school arrays/secrets."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import stat
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def source_files(root=ROOT):
    root=Path(root);found=set()
    for name in ('START.py','build_delivery.py','delivery_template.py'):
        if (root/name).is_file():found.add(root/name)
    for pattern in ('R11_*.md','requirements*.txt','pyproject.toml','README.md'):
        found.update(p for p in root.glob(pattern) if p.is_file())
    for folder in ('paper3r11','tests','configs'):
        base=root/folder
        if base.is_dir():
            for p in base.rglob('*'):
                if (p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.py','.json','.yaml','.yml','.toml','.md','.txt')
                        and not any(x in ('__pycache__','.pytest_cache','private','.git') for x in p.parts)):
                    found.add(p)
    # Bundle the authored research and compact inspectable evidence, never
    # restored raw trees, archives, papers or recursive historical outputs.
    for base in (root/'research',):
        if base.is_dir():
            found.update(p for p in base.rglob('*') if p.is_file() and not p.is_symlink()
                         and (p.suffix.lower() in ('.md','.json','.py') or
                              (p.suffix.lower()=='.csv' and p.is_relative_to(root/'research/evidence')))
                         and p.stat().st_size<=4*1024*1024
                         and not any(x in ('private','.git','__pycache__') for x in p.parts)
                         and not any(x.is_symlink() for x in p.parents if x!=root.parent))
    for name in ('early','late'):
        base=root/'evidence'/name
        if base.is_dir() and not base.is_symlink():
            found.update(p for p in base.iterdir() if p.is_file() and not p.is_symlink()
                         and p.suffix.lower() in ('.csv','.json','.py') and p.stat().st_size<=4*1024*1024
                         and p.name.lower() not in ('github.json','credentials.json'))
    return sorted(found)


def build(root=ROOT,*,files=None):
    root=Path(root);paths=source_files(root) if files is None else [root/p for p in files]
    if not any(p.relative_to(root).as_posix()=='START.py' for p in paths):raise ValueError('R11 START.py must exist before building delivery')
    contents={p.relative_to(root).as_posix():p.read_bytes() for p in paths}
    manifest={'schema_version':'paper3-r11-code-package-v1','round':11,
              'scope':'Code, tests, configurations, authored research and bounded source-backed CSV/JSON evidence; no credentials, raw EEG, historical raw output trees, checkpoints or papers',
              'evidence_policy':{'research_extensions':['.md','.json','.py','.csv'],'research_csv_scope':'research/evidence only','historical_evidence_top_level_only':True,'maximum_evidence_file_bytes':4*1024*1024},
              'files':{name:{'bytes':len(data),'sha256':sha(data)} for name,data in contents.items()}}
    encoded=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    contents['MANIFEST.json']=encoded
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3
            info.external_attr=(stat.S_IFREG|0o600)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    payload=buffer.getvalue()
    source=(root/'delivery_template.py').read_text()
    entry=source.replace('__PAYLOAD_B85__',base64.b85encode(payload).decode()).replace('__PAYLOAD_SHA256__',sha(payload)).replace('__MANIFEST_SHA256__',sha(encoded)).encode()
    compile(entry,'START-PAPER3-R11.py','exec')
    output=root/'delivery';output.mkdir(exist_ok=True)
    (output/'START-PAPER3-R11.py').write_bytes(entry)
    (output/'paper3-r11-code.zip').write_bytes(payload)
    (output/'MANIFEST.json').write_bytes(encoded)
    record={'schema_version':'paper3-r11-delivery-v1','entry':'START-PAPER3-R11.py','entry_bytes':len(entry),
            'entry_sha256':sha(entry),'payload_bytes':len(payload),'payload_sha256':sha(payload),
            'manifest_sha256':sha(encoded),'files':len(manifest['files']),
            'default_action':'install_only','gpu_precheck_executed':False,'slurm_jobs_submitted':0,
            'network_operations_during_build':0}
    (output/'PACKAGE.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':print(json.dumps(build(),indent=2))
