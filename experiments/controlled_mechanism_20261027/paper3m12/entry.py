"""Isolated-school Python bootstrap; scientific imports belong to workers only."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    package = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package))
    version = 'python%d.%d' % sys.version_info[:2]
    binary = Path(sys.executable).absolute()
    candidates = [binary.parent.parent / lib / version / 'site-packages' for lib in ('lib', 'lib64')]
    candidates.extend(Path(sysconfig.get_paths()[name]) for name in ('purelib', 'platlib'))
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if not args:
        raise SystemExit('Expected M12 prepare, start, status, controller or worker command')
    if args[0] == 'prepare':
        import argparse
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--root', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--package', type=Path, default=package)
        values = parser.parse_args(args[1:])
        result = importlib.import_module('paper3m12.plan').prepare(values.root, values.output, values.package)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        return 0
    if args[0] in {'report-loop', 'upload'}:
        return importlib.import_module('paper3m12.uploads').main(args)
    return importlib.import_module('paper3m12.runtime').main(args)


if __name__ == '__main__':
    raise SystemExit(main())
