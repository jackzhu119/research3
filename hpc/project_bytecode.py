#!/usr/bin/env python3
"""Cache verified project bytecode locally without moving source or changing it."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shlex
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.environment import verify_code
from hpc.resources import allocation, write_json


class BytecodeError(ValueError):
    pass


def need(value, code):
    if not value:
        raise BytecodeError(code)


def real(path):
    path = Path(os.path.abspath(path))
    need(not any(p.is_symlink() for p in (path, *path.parents)), 'project_bytecode_symlink_forbidden')
    return path


def compile_project(root, cache_root, *, environ=None, hostname=None, progress=print):
    """Explicit compilation is allowed while -B still prevents incidental writes.

    Timestamp pycs preserve the original filename and normal CPython source
    invalidation. Full frozen source verification remains mandatory on each
    scientific stage; compilation never substitutes for that verification.
    """
    allocated = allocation(environ=environ, hostname=hostname)
    root, cache_root = real(root), real(cache_root)
    need(root.is_dir() and cache_root.is_dir() and cache_root.stat().st_uid == os.getuid()
         and cache_root.stat().st_mode & 0o077 == 0, 'verified_private_node_cache_required')
    identity = verify_code(root)
    value = json.loads((root / 'hpc/scientific_code.json').read_bytes())
    rows = {r['path']: r for r in value['baseline_files'] + value.get('deployment_support_files', [])}
    selected = sorted(name for name in rows if name.endswith('.py'))
    prefix = real(cache_root / 'project-pycache')
    prefix.mkdir(mode=0o700, exist_ok=True)
    previous = sys.pycache_prefix
    compiled = []
    try:
        sys.pycache_prefix = str(prefix)
        for index, name in enumerate(selected, 1):
            source = real(root / name)
            before = source.stat()
            destination = real(importlib.util.cache_from_source(str(source)))
            need(destination.is_relative_to(prefix), 'project_bytecode_cache_escape')
            py_compile.compile(str(source), cfile=str(destination), dfile=str(source),
                doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
            after = source.stat()
            need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'project_source_changed_during_bytecode_preparation')
            raw = destination.read_bytes()
            compiled.append({'source': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            if progress and (index == 1 or index % 25 == 0 or index == len(selected)):
                progress('PROJECT_BYTECODE: ' + str(index) + '/' + str(len(selected)))
    finally:
        sys.pycache_prefix = previous
    need(verify_code(root) == identity, 'project_source_changed_during_bytecode_preparation')
    return {'schema_version': 'paper3-hpc-local-project-bytecode-v1',
        'status': 'FROZEN_PROJECT_BYTECODE_PREPARED', 'allocation': allocated,
        'cache_prefix': str(prefix), 'software_identity': identity, 'files': compiled,
        'source_paths_unchanged': True, 'source_modified': False, 'scientific_fit_performed': False,
        'original_test_deadlines_changed': False, 'scientific_budget_changed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = real(args.root)
        output, env_output = real(args.output), real(args.env_output)
        need(output.is_relative_to(root) and env_output.is_relative_to(root), 'project_bytecode_report_must_be_inside_project')
        result = compile_project(root, args.cache_root, progress=lambda text: print(text, flush=True))
        write_json(output, result)
        env_output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.bytecode-', dir=env_output.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                stream.write('export PYTHONPYCACHEPREFIX=' + shlex.quote(result['cache_prefix']) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, env_output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(result['status'], flush=True)
        return 0
    except (OSError, ValueError, py_compile.PyCompileError) as exc:
        reason = str(exc) if isinstance(exc, BytecodeError) else type(exc).__name__
        print('BLOCKED: ' + reason, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
