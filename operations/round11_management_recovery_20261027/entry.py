"""Isolated Python bootstrap for operational R11 recovery and original science."""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import sys
import sysconfig


def main(argv=None):
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # -I -S keeps login-site modifications out. Restore only this selected
    # interpreter's existing venv dependencies for report/scientific children.
    binary = Path(sys.executable).absolute()
    version = 'python%d.%d' % sys.version_info[:2]
    candidates = [binary.parent.parent/lib/version/'site-packages' for lib in ('lib', 'lib64')]
    candidates.extend(Path(sysconfig.get_paths()[key]) for key in ('purelib', 'platlib'))
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return importlib.import_module('supervisor').main(sys.argv[1:] if argv is None else argv)


if __name__ == '__main__':
    raise SystemExit(main())
