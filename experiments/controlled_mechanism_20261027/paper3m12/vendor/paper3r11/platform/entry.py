"""Restore isolated Python paths, set deterministic environment, then dispatch."""
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import signal
import threading


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    package = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package))
    version = 'python%d.%d'%sys.version_info[:2]
    executable = Path(sys.executable).absolute()
    candidates = [executable.parent.parent/lib/version/'site-packages' for lib in ('lib','lib64')]
    candidates += [Path(sysconfig.get_paths()[key]) for key in ('purelib','platlib')]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    if not args:
        raise SystemExit('Expected runtime command or scientific worker dispatch')
    if args[0] in ('science','science-daemon'):
        stop=threading.Event();done=threading.Event()
        def request_stop(*_):
            os.environ['PAPER3_R11_STOP_REQUESTED']='1'
            stop.set()
        for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
        print(json.dumps({'event':'R11_SCIENTIFIC_IMPORT_STARTED','training_started':False,'pid':os.getpid()}),flush=True)
        def heartbeat():
            while not done.wait(30):
                print(json.dumps({'event':'R11_IMPORT_HEARTBEAT','training_started':False}),flush=True)
        threading.Thread(target=heartbeat,daemon=True).start()
        from paper3r11.platform.common import load_plan
        try:
            plan_path=args[args.index('--plan')+1]
        except (ValueError,IndexError):
            raise SystemExit('Scientific worker requires --plan')
        try:
            _,plan=load_plan(plan_path)  # Verify code/recipe before numerical imports.
            print(json.dumps({'event':'R11_IMPORT_RUNTIME_SELECTED','runtime':'existing_python_site_packages',
                              'runtime_files_rehashed':0,'files_copied':0}),flush=True)
            module_name=plan.get('science_module','paper3r11.worker')
            if module_name!='paper3r11.worker' and not plan.get('synthetic_fixture'):
                raise ValueError('Alternate scientific modules are only software fixtures')
            module=importlib.import_module(module_name)
        finally:done.set()
        if stop.is_set():return 75
        if args[0]=='science-daemon':
            return importlib.import_module('paper3r11.platform.science_daemon').main(args[1:],plan=plan,worker=module,stop=stop)
        return module.main(args[1:])
    if args[0]=='report-loop':
        return importlib.import_module('paper3r11.platform.uploads').report_main(args[1:])
    if args[0]=='upload':
        return importlib.import_module('paper3r11.platform.uploads').main(args[1:])
    return importlib.import_module('paper3r11.platform.runtime').main(args)


if __name__=='__main__':
    raise SystemExit(main())
