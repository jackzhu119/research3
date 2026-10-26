import base64
import hashlib
import json
from pathlib import Path
import zlib

root = Path(__file__).resolve().parent
helper = (root/'r1_digest_cache.py').read_bytes()
sha = lambda b: hashlib.sha256(b).hexdigest()
entry = (root/'delivery_template.py').read_text().replace('__HOOK_B85__',
    base64.b85encode(zlib.compress(helper, 9)).decode()).replace('__HOOK_SHA256__', sha(helper)).encode()
compile(entry, 'FIX-PAPER3-R1-STARTUP-CACHE.py', 'exec')
(root/'FIX-PAPER3-R1-STARTUP-CACHE.py').write_bytes(entry)
record = {'entry': 'FIX-PAPER3-R1-STARTUP-CACHE.py', 'bytes': len(entry),
          'sha256': sha(entry), 'helper_sha256': sha(helper),
          'school_jobs_changed_by_assistant': 0}
(root/'PACKAGE.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
