"""Embed only the fixed R1 controller; original installation stays immutable."""
from pathlib import Path
import base64,hashlib,json

ROOT=Path(__file__).resolve().parent


def build(root=ROOT):
    root=Path(root);source=(root/'supervisor.py').read_bytes()
    if b'R1_AUTOMATIC_CHILD_REFRESH_DISABLED' not in source:
        raise ValueError('AUTOMATIC_REFRESH_DISABLE_FIX_REQUIRED')
    digest=hashlib.sha256(source).hexdigest()
    raw=(root/'refresh_hotfix_template.py').read_text().replace('__SUPERVISOR_B85__',base64.b85encode(source).decode()).replace('__SUPERVISOR_SHA256__',digest).encode()
    compile(raw,'HOTFIX-PAPER3-R1-REFRESH.py','exec')
    out=root/'delivery';out.mkdir(exist_ok=True)
    target=out/'HOTFIX-PAPER3-R1-REFRESH.py';target.write_bytes(raw)
    value=dict(hotfix_id='paper3-r1-refresh-disabled-v1',entry=target.name,
        entry_sha256=hashlib.sha256(raw).hexdigest(),entry_bytes=len(raw),
        supervisor_sha256=digest,default='read_only_inspection',activation='--apply',
        scientific_children_signalled=0,old_package_files_changed=0)
    (out/'REFRESH_HOTFIX_PACKAGE.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':print(json.dumps(build(),indent=2))
