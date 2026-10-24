"""Isolated -I -S entry: explicitly select packages without executing .pth files."""
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import sysconfig
import threading


def package_paths(executable=None):
    # Python 3.12 -S reports the base prefix; derive venv paths from executable.
    executable = Path(executable or sys.executable).absolute()
    version = "python%d.%d" % sys.version_info[:2]
    candidates = [executable.parent.parent / lib / version / "site-packages"
                  for lib in ("lib", "lib64")]
    candidates += [Path(sysconfig.get_paths()[key]) for key in ("purelib", "platlib")]
    result = []
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in result:
            result.append(str(candidate))
    return result


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("worker", "allocation", "monitor", "controller", "report"):
        raise SystemExit("Usage: entry.py {worker,allocation,monitor,controller,report} --plan PATH ...")
    mode, rest = args[0], args[1:]
    package = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package))
    if "--plan" in rest:
        plan_path = Path(rest[rest.index("--plan") + 1])
        plan = json.loads(plan_path.read_text())
        if plan.get("root"):
            sys.path.append(str(Path(plan["root"]).resolve()))
    for path in package_paths():
        if path not in sys.path:
            sys.path.append(path)
    import_done = threading.Event()
    if mode in ("worker", "allocation"):
        def stop_requested(*_):
            os.environ["PAPER3R6_STOP_REQUESTED"] = "1"
        signal.signal(signal.SIGUSR1, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)
        def import_progress():
            while not import_done.wait(30):
                print(json.dumps({"event": "PYTHON_IMPORT_IN_PROGRESS", "training_started": False}), flush=True)
        threading.Thread(target=import_progress, daemon=True).start()
    sys.argv = [str(Path(__file__)), *rest]
    try:
        module = importlib.import_module("paper3r6." + ("runtime" if mode == "controller" else mode))
    finally:
        import_done.set()
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
