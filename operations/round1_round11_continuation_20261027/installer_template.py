#!/usr/bin/env python3
"""Visible R1 continuation recovery; applying the future-action hooks is explicit."""
from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import time
import zipfile

PAYLOAD_B85 = '__PAYLOAD_B85__'
PAYLOAD_SHA256 = '__PAYLOAD_SHA256__'
MANIFEST_SHA256 = '__MANIFEST_SHA256__'
SCHEMA = 'paper3-r1-r11-continuation-package-v1'
BASE_NAME = 'paper3-r1-r11-continuation'
DEFAULT_ROOT = '/public/home/zhuziyuan/projects/paper3'
DEFAULT_PYTHON = '/public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python'
MAX_CODE_BYTES = 64 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError('PACKAGE_MEMBER_NAME_NOT_TEXT')
    path = PurePosixPath(name)
    forbidden = {'private', '.git', '.ssh', 'secrets', 'credentials', '__pycache__'}
    if (not name or path.is_absolute() or path.as_posix() != name or '\\' in name
            or any(part in ('', '.', '..') or part.lower() in forbidden for part in name.split('/'))
            or path.name.lower() in {'github.json', 'credentials.json', '.env'}):
        raise ValueError('UNSAFE_PACKAGE_MEMBER_PATH')
    return path


def strict_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('DUPLICATE_PACKAGE_MANIFEST_KEY')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def checked_payload():
    raw = base64.b85decode(PAYLOAD_B85)
    if sha(raw) != PAYLOAD_SHA256:
        raise ValueError('PACKAGE_SHA256_DIFFERS')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('DUPLICATE_PACKAGE_MEMBER')
    if 'MANIFEST.json' not in names or archive.getinfo('MANIFEST.json').file_size > 2 * 1024 * 1024:
        raise ValueError('PACKAGE_MANIFEST_REQUIRED_OR_TOO_LARGE')
    manifest_data = archive.read('MANIFEST.json')
    if sha(manifest_data) != MANIFEST_SHA256:
        raise ValueError('PACKAGE_MANIFEST_SHA256_DIFFERS')
    manifest = strict_json(manifest_data)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != SCHEMA:
        raise ValueError('PACKAGE_MANIFEST_SCHEMA_DIFFERS')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or len(files) > 4096:
        raise ValueError('PACKAGE_FILE_INVENTORY_INVALID')
    if set(names) != {'MANIFEST.json', *files}:
        raise ValueError('PACKAGE_MANIFEST_COVERAGE_DIFFERS')
    total = 0
    for name, row in files.items():
        safe_name(name)
        if (not isinstance(row, dict) or type(row.get('bytes')) is not int
                or row['bytes'] < 0 or not isinstance(row.get('sha256'), str)
                or len(row['sha256']) != 64):
            raise ValueError('PACKAGE_FILE_DESCRIPTOR_INVALID')
        info = archive.getinfo(name)
        mode = (info.external_attr >> 16) & 0xffff
        if (info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or info.file_size != row['bytes']):
            raise ValueError('PACKAGE_MEMBER_TYPE_OR_SIZE_DIFFERS')
        total += row['bytes']
    if total > MAX_CODE_BYTES:
        raise ValueError('PACKAGE_CODE_SIZE_LIMIT')
    if not {'entry.py', 'selection_cache.py', 'b5_retry_budget.py', 'install_selection.py', 'diagnostic_upload.py'} <= set(files):
        raise ValueError('PACKAGE_DISPATCH_ENTRY_REQUIRED')
    return archive, manifest_data, manifest


def package_path(home):
    return Path(home).expanduser().absolute() / BASE_NAME / PAYLOAD_SHA256[:20]


def _no_symlink_components(path, base):
    relative = path.relative_to(base)
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return path.resolve().is_relative_to(base.resolve())


def verify_install(folder, manifest_data, manifest):
    """Check only this new immutable installation, never live R1/R11 outputs.

    Source/runtime consistency for the scientific plan belongs to one-time
    metadata preparation. Unrelated jobs may legitimately update their own
    runtime/status/results while this installer runs; no such file is read or
    compared here.
    """
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        return False
    recorded = folder / 'MANIFEST.json'
    if (not _no_symlink_components(recorded, folder) or not recorded.is_file()
            or recorded.read_bytes() != manifest_data):
        return False
    expected = {'MANIFEST.json', *manifest['files']}
    for item in folder.rglob('*'):
        if item.is_symlink():
            return False
        if item.is_file() and item.relative_to(folder).as_posix() not in expected:
            return False
    for name, row in manifest['files'].items():
        path = folder / name
        if (not _no_symlink_components(path, folder) or not path.is_file()
                or path.stat().st_size != row['bytes'] or sha(path.read_bytes()) != row['sha256']):
            return False
    return True


def atomic_bytes(path, data):
    path = Path(path)
    transient = {errno.ENOENT, errno.EINTR, errno.EAGAIN, errno.EIO,
                 getattr(errno, 'ESTALE', 116), errno.ETIMEDOUT}
    for attempt in range(4):
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix='.install-receipt-', dir=path.parent)
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return
        except OSError as error:
            # A shared-filesystem rename may succeed before its RPC fails.
            try:
                if path.is_file() and path.read_bytes() == data:
                    return
            except OSError:
                pass
            if error.errno not in transient or attempt == 3:
                raise
            time.sleep(.02 * (2**attempt))
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass


def install(home):
    archive, manifest_data, manifest = checked_payload()
    base = Path(home).expanduser().absolute() / BASE_NAME
    if base.is_symlink():
        raise ValueError('INSTALL_BASE_IS_SYMLINK')
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = package_path(home)
    lock_path = base / 'install.lock'
    if lock_path.is_symlink():
        raise ValueError('INSTALL_LOCK_IS_SYMLINK')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            if not verify_install(destination, manifest_data, manifest):
                raise ValueError('EXISTING_IMMUTABLE_PACKAGE_DIFFERS')
        else:
            staging = Path(tempfile.mkdtemp(prefix='.install-', dir=base))
            try:
                for name, row in manifest['files'].items():
                    data = archive.read(name)
                    if len(data) != row['bytes'] or sha(data) != row['sha256']:
                        raise ValueError('PACKAGE_MEMBER_SHA256_DIFFERS')
                    path = staging / name
                    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with path.open('xb') as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    path.chmod(0o600)
                (staging / 'MANIFEST.json').write_bytes(manifest_data)
                if not verify_install(staging, manifest_data, manifest):
                    raise ValueError('STAGED_PACKAGE_VERIFY_FAILED')
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    import shutil
                    shutil.rmtree(staging)
        receipt = dict(package=str(destination), payload_sha256=PAYLOAD_SHA256,
                       manifest_sha256=MANIFEST_SHA256, files=len(manifest['files']),
                       scientific_files_changed=0, slurm_calls=0, gpu_used=False)
        atomic_bytes(base / 'CURRENT.json', encoded(receipt))
    emit('R1_CONTINUATION_PACKAGE_INSTALLED', **receipt)
    return destination


def existing_install(home):
    _, manifest_data, manifest = checked_payload()
    destination = package_path(home)
    if not verify_install(destination, manifest_data, manifest):
        raise ValueError('EXACT_INSTALLED_PACKAGE_REQUIRED_FOR_CHECK')
    return destination


def main(argv=None):
    emit('R1_CONTINUATION_BOOTSTRAP_START', scientific_jobs_signalled=0,
         message='程序已启动；安装后逐项输出检查结果。')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args(argv)
    if args.check and any((args.install, args.apply)):
        raise ValueError('CHECK_AND_APPLY_ARE_SEPARATE_MODES')
    package = existing_install(args.home) if args.check else install(args.home)
    if not (args.apply or args.check):
        emit('R1_CONTINUATION_CODE_READY_NO_ACTION', package=str(package),
             next_operation='--apply', jobs_submitted=0, scientific_jobs_signalled=0)
        return 0
    import subprocess
    command = [str(Path(args.python).expanduser().absolute()), '-I', '-S', '-B', '-u',
               str(package/'entry.py'), '--root', str(args.root.expanduser().absolute()),
               '--home', str(args.home.expanduser().absolute())]
    if args.apply:
        command.append('--apply')
    if args.no_upload:
        command.append('--no-upload')
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
