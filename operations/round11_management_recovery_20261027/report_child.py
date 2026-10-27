#!/usr/bin/env python3
"""Owned CPU-only R11 compact report child with the operational I/O repair."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig


def main(argv=None):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    version = 'python%d.%d' % sys.version_info[:2]
    binary = Path(sys.executable).absolute()
    candidates = [binary.parent.parent/lib/version/'site-packages' for lib in ('lib', 'lib64')]
    candidates += [Path(sysconfig.get_paths()[name]) for name in ('purelib', 'platlib')]
    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--expected-recipe', required=True)
    args = parser.parse_args(argv)
    from environment import load_env
    env = load_env(args.package, args.plan, expected_recipe=args.expected_recipe)
    report = importlib.import_module('paper3r11.report').generate_report(env['path'])
    coverage = report.get('coverage', {})
    print(json.dumps(dict(status=report.get('status'), recipe_sha256=report.get('recipe_sha256'),
                          sealed_complete_tasks=coverage.get('sealed_complete_tasks'),
                          total_tasks=coverage.get('total_tasks'), CPU_only=True,
                          scientific_computation_started=False), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
