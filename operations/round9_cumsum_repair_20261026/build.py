"""Build the small standard-library-only school installer."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

root = Path(__file__).resolve().parent
hook = (root/'r9_runtime_hook.py').read_bytes()
digest = lambda b: hashlib.sha256(b).hexdigest()
entry = (root/'delivery_template.py').read_text().replace(
    '__HOOK_B85__', base64.b85encode(zlib.compress(hook, 9)).decode()).replace(
    '__HOOK_SHA256__', digest(hook)).encode()
compile(entry, 'FIX-PAPER3-R9-CUMSUM.py', 'exec')
(root/'FIX-PAPER3-R9-CUMSUM.py').write_bytes(entry)
metadata = {'entry': 'FIX-PAPER3-R9-CUMSUM.py', 'bytes': len(entry),
    'sha256': digest(entry), 'runtime_hook_sha256': digest(hook),
    'native_gp_sha256': '5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7',
    'executed_gp_sha256': 'bb3dbd077632251d8fd34a43f83645c8f411f2954d8110c8d907da9e840e17ee',
    'school_jobs_changed_by_assistant': 0}
(root/'PACKAGE.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps(metadata))
