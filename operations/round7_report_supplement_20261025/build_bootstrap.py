"""Build the self-contained stdlib-only online entrypoint, without credentials."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('report_supplement.py', 'restore_report.py', 'paper3_result_transport.py', 'paper_credentials.py')
files = {name: (ROOT / name).read_text() for name in NAMES}
raw = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
payload_hash = hashlib.sha256(raw).hexdigest()
payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
header = '''#!/usr/bin/env python3
"""Install/start only the Paper3 R7 aggregate report upload; never submit training."""
import base64,gzip,json,hashlib,pathlib,os,sys,fcntl,tempfile,runpy,shutil
PAYLOAD_SHA = %r
PAYLOAD = %r
''' % (payload_hash, payload)
body = '''
root=pathlib.Path.home()/"paper3-r7-report-uploader"/PAYLOAD_SHA[:20]
root.parent.mkdir(parents=True,exist_ok=True)
with (root.parent/"install.lock").open("a") as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 raw=gzip.decompress(base64.b64decode(PAYLOAD))
 if hashlib.sha256(raw).hexdigest()!=PAYLOAD_SHA:raise SystemExit("PACKAGE_HASH_MISMATCH")
 files=json.loads(raw)
 expected={n:hashlib.sha256(c.encode()).hexdigest() for n,c in files.items()}
 good=root.is_dir() and all((root/n).is_file() and hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in expected.items())
 if not good:
  staging=pathlib.Path(tempfile.mkdtemp(prefix=".install-",dir=root.parent))
  try:
   for n,c in files.items():(staging/n).write_text(c)
   if root.exists():
    for n in files:os.replace(staging/n,root/n)
   else:os.rename(staging,root)
  finally:
   if staging.exists():shutil.rmtree(staging)
print(json.dumps({"event":"R7_REPORT_PACKAGE_READY","package":str(root),"payload_sha256":PAYLOAD_SHA,"training_jobs_changed":0}),flush=True)
sys.path.insert(0,str(root))
runpy.run_path(str(root/"report_supplement.py"),run_name="__main__")
'''
target = ROOT / 'START-PAPER3-R7-REPORT.py'
target.write_text(header + body)
metadata = dict(entrypoint=target.name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                payload_sha256=payload_hash, bytes=target.stat().st_size,
                files=[dict(path=n, sha256=hashlib.sha256(files[n].encode()).hexdigest()) for n in NAMES])
(ROOT / 'PACKAGE.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata, indent=2))
