"""-I -S entry: bypass startup hooks, then select known package paths explicitly."""
import os
import hashlib
import importlib
import json
from pathlib import Path
import signal
import sys
import sysconfig
import threading
import time


def package_paths(executable=None):
    """Locate the existing venv without executing site.py, .pth or sitecustomize.

    Python 3.12 -S can report the base sys.prefix even for a venv; use the
    interpreter path as the primary selector, with sysconfig as a base fallback.
    """
    executable = Path(executable or sys.executable)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [executable.parent.parent / "lib" / version / "site-packages",
                  executable.parent.parent / "lib64" / version / "site-packages"]
    candidates += [Path(sysconfig.get_paths()[name]) for name in ("purelib", "platlib")]
    result = []
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in result:
            result.append(str(candidate))
    return result


def reuse_node_runtime(cache_base):
    # Cached packages are optional; no whole-runtime copy or rehash before fitting.
    try:
        caches = sorted(Path(cache_base).iterdir()) if Path(cache_base).is_dir() else []
    except OSError:
        caches = []
    for cache in caches:
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
            print(json.dumps({"event": "NODE_RUNTIME_REUSED", "cache": str(cache), "files_rehashed": 0}), flush=True)
            return [str(p) for p in paths]
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return []


def main():
    package, project, mode = Path(__file__).resolve().parent, Path(sys.argv[1]).resolve(), sys.argv[2]
    import_done = threading.Event()
    if mode == "worker":
        signal.signal(signal.SIGUSR1, lambda *_: os.environ.update(PAPER3V2_STOP_REQUESTED="1"))
        signal.signal(signal.SIGTERM, lambda *_: os.environ.update(PAPER3V2_STOP_REQUESTED="1"))
        def import_progress():
            while not import_done.wait(30):
                print(json.dumps({"event": "PYTHON_IMPORT_IN_PROGRESS", "training_started": False}), flush=True)
        threading.Thread(target=import_progress, daemon=True).start()
    cache_paths = reuse_node_runtime(Path(f"/tmp/paper3-runtime-{os.getuid()}"))
    sys.path[:0] = [str(package), str(project), *cache_paths]
    for path in package_paths():
        if path not in sys.path:
            sys.path.append(path)
    sys.argv = [str(package / "entry.py"), *sys.argv[3:]]
    try:
        module = importlib.import_module("paper3v2." + mode)
    finally:
        import_done.set()
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
