"""Build a deterministic, code-only continuation/diagnostic installer."""
from pathlib import Path
import base64, hashlib, io, json, stat, zipfile

ROOT=Path(__file__).resolve().parent
REQUIRED={'entry.py','selection_cache.py','b5_retry_budget.py','install_selection.py',
          'diagnostic_upload.py','installer_template.py','build_delivery.py'}
SCHEMA='paper3-r1-r11-continuation-package-v1'

def sha(data):return hashlib.sha256(data).hexdigest()

def build(root=ROOT):
    root=Path(root)
    if any(not(root/n).is_file() or (root/n).is_symlink() for n in REQUIRED):
        raise ValueError('REQUIRED_CONTINUATION_CODE_MISSING')
    files={}
    for p in sorted(root.rglob('*')):
        name=p.relative_to(root)
        if any(x in ('delivery','__pycache__','.git','private','credentials') for x in name.parts):continue
        if not p.is_file() or p.is_symlink() or p.suffix not in ('.py','.md','.json'):continue
        if p.name.lower() in ('credentials.json','github.json','manifest.json') or p.stat().st_size>4*1024*1024:
            raise ValueError('UNEXPECTED_CONTINUATION_PACKAGE_MEMBER')
        data=p.read_bytes()
        if p.suffix=='.py':compile(data,str(name),'exec')
        files[name.as_posix()]=data
    if sum(map(len,files.values()))>16*1024*1024:raise ValueError('CONTINUATION_PACKAGE_TOO_LARGE')
    manifest=dict(schema_version=SCHEMA,scope='Operational code/tests/proof only; no EEG, credentials, plans, fits or checkpoints',
        entry='entry.py',files={n:dict(bytes=len(v),sha256=sha(v)) for n,v in files.items()})
    metadata=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted({**files,'MANIFEST.json':metadata}.items()):
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.create_system=3
            info.external_attr=(stat.S_IFREG|0o600)<<16;info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    payload=buffer.getvalue()
    entry=(root/'installer_template.py').read_text().replace('__PAYLOAD_B85__',base64.b85encode(payload).decode()).replace('__PAYLOAD_SHA256__',sha(payload)).replace('__MANIFEST_SHA256__',sha(metadata)).encode()
    compile(entry,'FIX-PAPER3-R1-R11-CONTINUATION.py','exec')
    delivery=root/'delivery';delivery.mkdir(exist_ok=True)
    (delivery/'FIX-PAPER3-R1-R11-CONTINUATION.py').write_bytes(entry)
    (delivery/'paper3-r1-r11-continuation.zip').write_bytes(payload)
    (delivery/'MANIFEST.json').write_bytes(metadata)
    record=dict(entry='FIX-PAPER3-R1-R11-CONTINUATION.py',entry_sha256=sha(entry),entry_bytes=len(entry),
        payload_sha256=sha(payload),payload_bytes=len(payload),manifest_sha256=sha(metadata),files=len(files),
        default='INSTALL_ONLY',activation='--apply',scientific_jobs_signalled=0,slurm_jobs_submitted=0,
        r11_new_code_applied=False,r11_report_only_memory_fix_applied=False,existing_B2_interrupted=False)
    (delivery/'PACKAGE.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':print(json.dumps(build(),indent=2))
