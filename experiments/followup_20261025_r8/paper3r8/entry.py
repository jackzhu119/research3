"""Stdlib-only isolated entry with immediate import heartbeat."""
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import sysconfig
import threading


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ('worker','report'):
        raise SystemExit('Usage: entry.py {worker,report} --plan PATH ...')
    mode,rest=args[0],args[1:]
    package=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(package))
    executable=Path(sys.executable).absolute()
    version='python%d.%d'%sys.version_info[:2]
    candidates=[executable.parent.parent/lib/version/'site-packages' for lib in ('lib','lib64')]
    candidates += [Path(sysconfig.get_paths()[key]) for key in ('purelib','platlib')]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:sys.path.append(str(p))
    done=threading.Event()
    if mode=='worker':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        def stop(*_):os.environ['PAPER3R8_STOP_REQUESTED']='1'
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        print(json.dumps({'event':'R8_SCIENCE_PROCESS_STARTED','pid':os.getpid(),
                          'training_started':False}),flush=True)
        def heartbeat():
            while not done.wait(30):
                print(json.dumps({'event':'PYTHON_IMPORT_IN_PROGRESS','training_started':False}),flush=True)
        threading.Thread(target=heartbeat,daemon=True).start()
    sys.argv=[str(Path(__file__)),*rest]
    try:module=importlib.import_module('paper3r8.'+mode)
    finally:done.set()
    return module.main()


if __name__=='__main__':
    raise SystemExit(main())
