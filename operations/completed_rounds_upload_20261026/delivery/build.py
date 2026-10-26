"""Build a deterministic stdlib-only uploader bootstrap from an explicit allowlist."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib

REQUIRED_FILES=(
    'upload_driver.py','collector.py','paper_credentials.py',
    'transport/__init__.py','transport/lossless_publish.py','transport/paper3_result_transport.py',
    'archive/__init__.py','archive/full_channel.py','archive/paper3_result_pack.py',
)


def build(source=None,output=None):
    here=Path(__file__).resolve().parent
    source=Path(source).resolve() if source else here.parent
    output=Path(output).resolve() if output else here/'START-PAPER3-COMPLETED-UPLOAD.py'
    files={}
    for name in REQUIRED_FILES:
        path=source/name
        if not path.is_file():raise FileNotFoundError('Required uploader module is not ready: '+name)
        data=path.read_bytes();compile(data,name,'exec')
        # The source code may name environment keys, never literal credentials.
        if re.search(rb'(?:github_pat_[A-Za-z0-9_]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16})',data):
            raise ValueError('Credential-shaped literal must not enter public uploader payload: '+name)
        files[name]=data
    document={name:data.decode('utf-8') for name,data in sorted(files.items())}
    packed=zlib.compress(json.dumps(document,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode(),9)
    digest=hashlib.sha256(packed).hexdigest()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}
    template=(here/'bootstrap_template.py').read_text()
    text=template.replace('__PAYLOAD_SHA256__',digest).replace('__PAYLOAD_B85__',base64.b85encode(packed).decode())
    text=text.replace('__FILE_HASHES__',repr(hashes));data=text.encode();compile(data,output.name,'exec')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(data)
    metadata={'entrypoint':output.name,'entrypoint_bytes':len(data),
        'entrypoint_sha256':hashlib.sha256(data).hexdigest(),'payload_sha256':digest,
        'files':hashes,'file_count':len(files),'source_bytes':sum(map(len,files.values())),
        'source_only':True,'includes_credentials':False,'includes_user_outputs':False,
        'imports_torch_at_install':False,'scientific_jobs_changed':0}
    (output.parent/'PACKAGE.json').write_text(json.dumps(metadata,sort_keys=True,indent=2)+'\n')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.output),sort_keys=True,indent=2))


if __name__=='__main__':main()
