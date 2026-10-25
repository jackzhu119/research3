import base64
import hashlib
import json
from pathlib import Path
import zlib

root=Path(__file__).resolve().parent
files={'r1_spawn_bootstrap.py':(root/'r1/r1_spawn_bootstrap.py').read_bytes(),
       'resume_existing.py':(root/'runtime/resume_existing.py').read_bytes()}
raw=json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()},sort_keys=True,separators=(',',':')).encode()
sha=lambda value:hashlib.sha256(value).hexdigest()
text=(root/'future_fix_template.py').read_text().replace('__PAYLOAD_SHA256__',sha(raw)).replace('__PAYLOAD__',base64.b85encode(zlib.compress(raw,9)).decode())
data=text.encode();compile(data,'FIX-PAPER3-FUTURE.py','exec')
(root/'FIX-PAPER3-FUTURE.py').write_bytes(data)
record={'entry':'FIX-PAPER3-FUTURE.py','bytes':len(data),'sha256':sha(data),'payload_sha256':sha(raw),'files':{n:sha(v) for n,v in files.items()},'school_jobs_changed_by_assistant':0}
(root/'PACKAGE.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
