"""Build a self-contained deployment file. Private config stays out of GitHub."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

root=Path(__file__).resolve().parent
files=[root/n for n in ('START.py','entry.py','train.sbatch','README.md','DESIGN_ZH.md','KERNEL_ORIGIN.json','VALIDATION.json')]
files += sorted((root/'paper3v2').glob('*.py')) + sorted((root/'p3work').glob('*.py'))
files += [root/'private/github.json']
payload=json.dumps({str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in files},sort_keys=True,separators=(',',':')).encode()
compressed=zlib.compress(payload,9); bundle=hashlib.sha256(compressed).hexdigest()
encoded=base64.b85encode(compressed).decode()
header='''#!/usr/bin/env python3
"""Paper3: install the prepared three-GPU experiment and optionally submit it.
Contains the user's preconfigured synchronization account. Stdlib only.
"""
import base64, fcntl, hashlib, json, os
from pathlib import Path
import subprocess, sys, tempfile, zlib
'''
body='''
def main():
    payload=base64.b85decode(BUNDLE_B85)
    if hashlib.sha256(payload).hexdigest()!=BUNDLE_ID:
        raise SystemExit("Installer is incomplete; copy the full file again")
    files=json.loads(zlib.decompress(payload))
    install_base=Path(os.environ.get("PAPER3_FOLLOWUP_INSTALL_BASE",str(Path.home()/"paper3-followup")))
    install_base.mkdir(parents=True,exist_ok=True)
    lock=(install_base/"install.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    target=install_base/BUNDLE_ID[:20]
    target.mkdir(exist_ok=True)
    for name,data in files.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid package member")
        path=target/relative;path.parent.mkdir(parents=True,exist_ok=True)
        raw=base64.b64decode(data)
        if not path.exists() or path.read_bytes()!=raw:
            fd,tmp=tempfile.mkstemp(prefix=".install-",dir=path.parent)
            try:
                with os.fdopen(fd,"wb") as stream:
                    stream.write(raw);stream.flush();os.fsync(stream.fileno())
                os.replace(tmp,path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        path.chmod(0o600 if relative.parts[0]=="private" else 0o644)
    (target/"private").chmod(0o700)
    lock.close()
    print("INSTALLED "+str(target),flush=True)
    return subprocess.call([sys.executable,"-I","-S","-B","-u",str(target/"START.py"),*sys.argv[1:]])
if __name__=="__main__":
    raise SystemExit(main())
'''
output=root/'START-PAPER3-FOLLOWUP.py'
output.write_text(header+'\nBUNDLE_ID='+repr(bundle)+'\nBUNDLE_B85='+repr(encoded)+'\n'+body)
output.chmod(0o600)
print(json.dumps({'installer':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
