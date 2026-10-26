#!/usr/bin/env python3
"""Explicit R11 commands; default/help never submits work or initializes CUDA."""
import os
from pathlib import Path
import sys
import sysconfig

package=Path(__file__).resolve().parent
sys.path.insert(0,str(package))
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
if len(sys.argv)>1 and sys.argv[1] in ('validate','gpu-precheck','report','select'):
    version='python%d.%d'%sys.version_info[:2]
    prefix=Path(sys.executable).absolute().parent.parent
    paths=[prefix/lib/version/'site-packages' for lib in ('lib','lib64')]
    paths += [Path(sysconfig.get_paths()[k]) for k in ('purelib','platlib')]
    for path in paths:
        if path.is_dir() and str(path) not in sys.path:sys.path.append(str(path))
from paper3r11.cli import main
if __name__=='__main__':raise SystemExit(main())
