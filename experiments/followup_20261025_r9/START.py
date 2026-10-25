#!/usr/bin/env python3
"""Prepare Round9 using only stdlib, then start the three-job controller."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
from paper3r9.plan import build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output')
    parser.add_argument('--credentials')
    parser.add_argument('--account')
    parser.add_argument('--submit', '--execute', dest='submit', action='store_true')
    args = parser.parse_args(argv)
    plan = build(args.root, args.output)
    if not Path(args.root).is_dir():
        raise SystemExit('Project root does not exist: '+args.root)
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    path = output/'plan.json'
    with (output/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            previous = json.loads(path.read_text())
            if previous['recipe_sha256'] != plan['recipe_sha256']:
                raise SystemExit('Existing Round9 plan differs; use --output for a separate experiment.')
        else:
            temporary = output/'.plan.json.tmp'
            temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(path)
    spec = importlib.util.spec_from_file_location('round9_runtime_start', PACKAGE/'runtime/START.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    command = ['--plan', str(path), '--package', str(PACKAGE), '--python', args.python]
    if args.credentials: command += ['--credentials', args.credentials]
    if args.account: command += ['--account', args.account]
    if args.submit: command += ['--submit']
    return module.main(command)


if __name__ == '__main__':
    raise SystemExit(main())
