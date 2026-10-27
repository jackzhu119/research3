import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, writer):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def put(path, value):
    atomic(path, lambda p: Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)))

def emit(event, **values):
    print(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **values}, ensure_ascii=False, allow_nan=False), flush=True)

def artifact(root, value, *, verify=True):
    path = (Path(root) / value["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact outside project")
    if verify and sha(path) != value["sha256"]:
        raise ValueError("Artifact content differs: " + value["path"])
    return path
