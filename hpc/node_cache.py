#!/usr/bin/env python3
"""Verify a node-local read cache of the existing Python runtime; no install/fit.

Only installed distribution RECORD files and stock stdlib Python sources are
eligible. The original interpreter, project, CUDA driver, scientific budgets,
thread limits and checkpoint paths are unchanged.
"""
from __future__ import annotations

import argparse
import base64
from contextvars import ContextVar, copy_context
from concurrent.futures import ThreadPoolExecutor
import fcntl
from functools import wraps
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import sys
import sysconfig
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.resources import allocation, write_json

MAX_BYTES = 16 * 1024**3
RESERVE_BYTES = 1024**3
LOCAL_FS = frozenset({'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'tmpfs', 'ramfs', 'overlay', 'zfs'})
ROOT_DISTRIBUTIONS = ('numpy', 'scipy', 'scikit-learn', 'torch', 'pytest')
# Keep trust stores and credential-aware clients in their original environment.
# A whole package absent from the overlay still imports from the original site.
ORIGINAL_CLIENTS = frozenset({'boto3', 'botocore', 'certifi', 'requests', 'keyring',
                               'oauthlib', 'requests-oauthlib'})
FORBIDDEN = frozenset({'.aws', '.ssh', '.git', '.env', 'credentials', 'pip.conf',
                       'direct_url.json', 'sitecustomize.py', 'usercustomize.py'})
_SOURCE_PARENTS = ContextVar('paper3_verified_source_parents', default=None)


class CacheError(ValueError):
    pass


def need(value, code):
    if not value:
        raise CacheError(code)


def regular(path, *, source=False):
    path = Path(os.path.abspath(path))
    need(not path.is_symlink(), 'node_cache_symlink_forbidden')
    checked = _SOURCE_PARENTS.get() if source else None
    pending = []
    for parent in path.parents:
        if checked is not None and parent in checked:
            break
        need(not parent.is_symlink(), 'node_cache_symlink_forbidden')
        pending.append(parent)
    if checked is not None:
        checked.update(pending)
    need(stat.S_ISREG(path.stat().st_mode), 'node_cache_regular_file_required')
    return path


def digest(path, *, source=False):
    h = hashlib.sha256()
    with regular(path, source=source).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def _local(path):
    """Use kernel mount evidence, never a hostname or path-name assumption."""
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        separator = fields.index('-')
        mount = Path(fields[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\'))
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), fields[separator + 1]))
    return bool(matches and max(matches)[1] in LOCAL_FS)


def _select_distributions(site=None):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    available = {}
    active = metadata.distributions(path=[str(site)]) if site is not None else metadata.distributions()
    for dist in active:
        if dist.metadata.get('Name'):
            available.setdefault(canonicalize_name(dist.metadata['Name']), dist)
    selected, pending = {}, list(ROOT_DISTRIBUTIONS)
    environment = {**default_environment(), 'extra': ''}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in selected or name in ORIGINAL_CLIENTS:
            continue
        need(name in available, 'node_cache_existing_distribution_missing_' + name.replace('-', '_'))
        dist = selected[name] = available[name]
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate(environment):
                key = canonicalize_name(requirement.name)
                need(key in available and requirement.specifier.contains(available[key].version, prereleases=True),
                     'node_cache_existing_dependency_unsatisfied')
                pending.append(key)
    return list(selected.values())


def _eligible(relative):
    path = PurePosixPath(str(relative))
    return (not path.is_absolute() and '..' not in path.parts
            and not any(p in FORBIDDEN or p == '__pycache__' for p in path.parts)
            and path.suffix not in {'.pth', '.pyc', '.pyo', '.key'}
            and not any(p.startswith('.') for p in path.parts))


def _inventory(site, distributions, stdlib, progress=None):
    rows = {}
    versions = {}
    last_progress = time.monotonic()
    for dist in distributions:
        name = dist.metadata.get('Name', '')
        need(name and dist.files, 'node_cache_existing_RECORD_required')
        versions[name] = dist.version
        record_found = False
        for entry in dist.files:
            relative = PurePosixPath(str(entry))
            if relative.name == 'RECORD' and any(p.endswith('.dist-info') for p in relative.parts):
                record_found = True
            if not _eligible(relative):
                continue
            source = Path(dist.locate_file(entry))
            source_site = site if site is not None else Path(dist.locate_file('')).absolute()
            need(source.absolute().is_relative_to(source_site), 'node_cache_distribution_outside_site')
            source = regular(source, source=True)
            sha = entry.hash
            if sha is not None:
                need(sha.mode == 'sha256', 'node_cache_RECORD_sha256_required')
                try:
                    expected = base64.urlsafe_b64decode(sha.value + '=' * (-len(sha.value) % 4)).hex()
                except (ValueError, TypeError):
                    raise CacheError('node_cache_invalid_RECORD_hash') from None
                need(len(expected) == 64, 'node_cache_invalid_RECORD_hash')
            else:
                # Only the RECORD listing itself has no self-referential digest.
                need(relative.name == 'RECORD' and any(p.endswith('.dist-info') for p in relative.parts),
                     'node_cache_unhashed_runtime_file_forbidden')
                expected = digest(source, source=True)
            size = source.stat().st_size
            need(entry.size is None or size == entry.size, 'node_cache_source_size_mismatch')
            key = 'site/' + relative.as_posix()
            row = {'source': str(source), 'path': key, 'bytes': size, 'sha256': expected}
            need(key not in rows or rows[key] == row, 'node_cache_distribution_collision')
            rows[key] = row
            if progress and time.monotonic() - last_progress >= 5:
                progress(f'NODE_CACHE: inspected {len(rows)} runtime RECORD files')
                last_progress = time.monotonic()
        need(record_found, 'node_cache_existing_RECORD_required')
    for current, directories, files in os.walk(stdlib, followlinks=False):
        directories[:] = [p for p in directories if p not in {'site-packages', '__pycache__', 'test', 'tests', 'ensurepip', 'lib-dynload'}
                          and not p.startswith(('config-', '.'))]
        for name in files:
            relative = Path(current, name).relative_to(stdlib)
            if not _eligible(relative) or not name.endswith('.py'):
                continue
            source = regular(Path(current, name), source=True)
            key = 'stdlib/' + relative.as_posix()
            rows[key] = {'source': str(source), 'path': key, 'bytes': source.stat().st_size,
                         'sha256': digest(source, source=True)}
    need(rows, 'node_cache_no_runtime_files')
    return [rows[k] for k in sorted(rows)], versions


def _copy_verified(row, target):
    source = regular(row['source'], source=True)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    if target.exists():
        need(target.stat().st_size == row['bytes'] and digest(target) == row['sha256'],
             'node_cache_cached_SHA256_mismatch')
        need(digest(source, source=True) == row['sha256'], 'node_cache_source_SHA256_mismatch')
        return
    fd, name = tempfile.mkstemp(prefix='.copy-', dir=target.parent)
    temporary = Path(name)
    try:
        with source.open('rb') as original, os.fdopen(fd, 'wb') as output:
            for block in iter(lambda: original.read(4 * 1024**2), b''):
                h.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        need((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
             and before.st_size == row['bytes'], 'node_cache_source_changed_during_copy')
        need(h.hexdigest() == row['sha256'] and digest(temporary) == row['sha256'],
             'node_cache_source_SHA256_mismatch')
        os.chmod(temporary, stat.S_IMODE(before.st_mode) & 0o755)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _one_transaction(function):
    @wraps(function)
    def call(*args, **kwargs):
        token = _SOURCE_PARENTS.set(set())
        try:
            return function(*args, **kwargs)
        finally:
            _SOURCE_PARENTS.reset(token)
    return call


@_one_transaction
def prepare_cache(output, *, environ=None, hostname=None, distributions=None, stdlib=None,
                  site_root=None, scratch=None, local_checker=None, max_bytes=None, progress=print):
    env = os.environ if environ is None else environ
    allocated = allocation(environ=env, hostname=hostname)  # before any directory/file writes
    site = Path(site_root).absolute() if site_root is not None else None
    stdlib = Path(stdlib or sysconfig.get_path('stdlib')).absolute()
    checker = local_checker or _local
    candidates = [scratch] if scratch is not None else [env.get('SLURM_TMPDIR'), env.get('TMPDIR'), '/tmp']
    selected = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).absolute()
        if path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)) and checker(path):
            selected = path
            break
    need(selected is not None, 'node_cache_node_local_scratch_required')
    if max_bytes is None:
        raw = env.get('PAPER3_HPC_NODE_CACHE_MAX_GIB', '16')
        need(raw.isdigit() and 1 <= int(raw) <= 16, 'node_cache_max_GiB_must_be_1_through_16')
        max_bytes = int(raw) * 1024**3
    need(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES, 'node_cache_size_limit_invalid')
    if progress:
        progress('NODE_CACHE: inspecting existing runtime RECORDs; no package installation')
    rows, versions = _inventory(site, distributions if distributions is not None else _select_distributions(site),
                                stdlib, progress=progress)
    total = sum(row['bytes'] for row in rows)
    need(total <= max_bytes, 'node_cache_runtime_exceeds_configured_size_cap')
    memory_gib = allocated.get('memory_gib')
    need(memory_gib is not None and memory_gib > 0 and total <= memory_gib * 1024**3 / 4,
         'node_cache_runtime_exceeds_quarter_allocation_RAM')
    fingerprint = {'python': sys.version, 'executable': sys.executable,
                   'site': str(site) if site is not None else 'active_interpreter_distribution_roots',
                   'stdlib': str(stdlib), 'versions': versions, 'files': rows}
    raw = json.dumps(fingerprint, sort_keys=True, separators=(',', ':')).encode()
    key = hashlib.sha256(raw).hexdigest()
    parent = selected / ('paper3-runtime-' + str(os.getuid()))
    need(not parent.is_symlink(), 'node_cache_symlink_forbidden')
    parent.mkdir(mode=0o700, exist_ok=True)
    need(parent.stat().st_uid == os.getuid() and stat.S_IMODE(parent.stat().st_mode) & 0o077 == 0,
         'node_cache_private_owned_directory_required')
    cache = parent / key
    cache.mkdir(mode=0o700, exist_ok=True)
    need(not cache.is_symlink(), 'node_cache_symlink_forbidden')
    missing = sum(row['bytes'] for row in rows if not (cache / row['path']).exists())
    need(shutil.disk_usage(cache).free >= missing + RESERVE_BYTES, 'node_cache_insufficient_local_disk')
    with (cache / 'owner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CacheError('node_cache_existing_builder_still_running') from None
        done = copied = 0
        last = time.monotonic()
        workers = min(8, max(1, int(allocated.get('cpus') or 1)))
        # Bounded deployment IO only; scientific kernels retain their original
        # single-thread profile. Copy the validated source-directory context.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(rows), 64):
                batch = rows[offset:offset + 64]
                pending = [(row, pool.submit(copy_context().run, _copy_verified,
                            row, cache / row['path'])) for row in batch]
                for row, future in pending:
                    future.result()
                    done += 1; copied += row['bytes']
                    if progress and (time.monotonic() - last >= 5 or done == len(rows)):
                        progress(f'NODE_CACHE: verified {done}/{len(rows)} files; {copied}/{total} bytes')
                        last = time.monotonic()
        write_json(cache / 'manifest.json', fingerprint)
    result = {'schema_version': 'paper3-hpc-node-local-runtime-cache-v1', 'status': 'NODE_LOCAL_CACHE_VERIFIED',
              'allocation': allocated, 'cache_root': str(cache), 'manifest_sha256': key,
              'files_verified': len(rows), 'bytes_verified': total, 'versions': versions,
              'deployment_copy_workers': workers,
              'pythonpath': os.pathsep.join(map(str, (cache / 'stdlib', cache / 'site'))),
              'stdlib_extension_libraries_cached': False,
              'python_executable': sys.executable, 'same_python_executable': True,
              'scientific_code_copied': False, 'scientific_budgets_changed': False,
              'dependencies_installed': 0, 'credentials_copied': False, 'numerical_fit_performed': False}
    write_json(Path(output), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_cache(args.output, progress=lambda s: print(s, flush=True))
        line = ('export PYTHONPATH=' + shlex.quote(result['pythonpath']) + '\n'
                + 'export PAPER3_HPC_NODE_CACHE_ROOT=' + shlex.quote(result['cache_root']) + '\n')
        args.env_output.parent.mkdir(parents=True, exist_ok=True)
        need(not any(p.is_symlink() for p in (args.env_output, *args.env_output.parents)),
             'node_cache_env_symlink_forbidden')
        fd, temporary = tempfile.mkstemp(prefix='.node-env-', dir=args.env_output.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(line); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, args.env_output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(json.dumps({'status': result['status'], 'files_verified': result['files_verified'],
                          'bytes_verified': result['bytes_verified'], 'numerical_fit_performed': False}), flush=True)
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'NODE_CACHE_BLOCKED', 'reason': str(exc) if isinstance(exc, CacheError)
                          else type(exc).__name__, 'numerical_fit_performed': False}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
