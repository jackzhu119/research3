from pathlib import Path
import base64
import hashlib
import json
import zlib

root=Path(__file__).resolve().parent
hook=(root/'r58_runtime_hook.py').read_bytes()
sha=lambda data:hashlib.sha256(data).hexdigest()
template=(root/'repair_template.py').read_text()
text=template.replace('__HOOK_B85__',base64.b85encode(zlib.compress(hook,9)).decode()).replace('__HOOK_SHA256__',sha(hook))
data=text.encode()
compile(data,'FIX-PAPER3-R589.py','exec')
(root/'FIX-PAPER3-R589.py').write_bytes(data)
(root/'PACKAGE.json').write_text(json.dumps({'entry':'FIX-PAPER3-R589.py',
    'bytes':len(data),'sha256':sha(data),'runtime_hook_sha256':sha(hook),
    'rounds':[5,8,9],'school_jobs_submitted_by_assistant':0,
    'native_scientific_source_files_changed':0},indent=2)+'\n')
print((root/'PACKAGE.json').read_text())
