"""-I entry: add the isolated extension and read-only original project explicitly."""
import os
import hashlib
import json
from pathlib import Path
import runpy
import signal
import sys
import threading
import time

package, project, mode = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve(), sys.argv[2]
sys.path[:0] = [str(package), str(project)]
# Reuse a node-local runtime prepared by v1 when this exact Python environment matches.
# No payload hash walk/copy; the scientific extension has no sitecustomize overlay.
cache_base = Path(f"/tmp/paper3-runtime-{os.getuid()}")
if cache_base.is_dir():
    for cache in sorted(cache_base.iterdir()):
        try:
            manifest = cache / "manifest.json"
            if not manifest.is_file() or manifest.stat().st_size > 16 * 1024**2:
                continue
            value = json.loads(manifest.read_text())
            key = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            paths = [cache / "stdlib", cache / "site"]
            if (cache.name != key or value.get("python") != sys.version
                    or value.get("executable") != sys.executable or not all(p.is_dir() for p in paths)):
                continue
            sys.path[2:2] = [str(p) for p in paths]
            print(json.dumps({"event": "NODE_RUNTIME_REUSED", "cache": str(cache), "files_rehashed": 0}), flush=True)
            break
        except (OSError, ValueError, TypeError):
            pass
if mode == "worker":
    signal.signal(signal.SIGUSR1, lambda *_: os.environ.update(PAPER3V2_STOP_REQUESTED="1"))
    signal.signal(signal.SIGTERM, lambda *_: os.environ.update(PAPER3V2_STOP_REQUESTED="1"))
    def import_progress():
        while "paper3v2.worker" not in sys.modules:
            time.sleep(30)
            # The worker module is executed as __main__; torch availability ends import reporting.
            if hasattr(sys.modules.get("torch"), "Tensor"):
                return
            print(json.dumps({"event": "PYTHON_IMPORT_IN_PROGRESS", "training_started": False}), flush=True)
    threading.Thread(target=import_progress, daemon=True).start()
sys.argv = [str(package / "entry.py"), *sys.argv[3:]]
runpy.run_module("paper3v2." + mode, run_name="__main__")
