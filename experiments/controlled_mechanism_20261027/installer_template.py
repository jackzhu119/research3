#!/usr/bin/env python3
"""Offline Paper3 mechanism-study installer; GPU submission is explicit."""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

PAYLOAD_B85 = '__PAYLOAD_B85__'
PAYLOAD_SHA256 = '__PAYLOAD_SHA256__'
MANIFEST_SHA256 = '__MANIFEST_SHA256__'
SCHEMA = 'paper3-mechanism12-package-v1'
BASE_NAME = 'paper3-mechanism12'
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
    if 'paper3m12/entry.py' not in files:
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
    for name, row in manifest['files'].items():
        path = folder / name
        if (not _no_symlink_components(path, folder) or not path.is_file()
                or path.stat().st_size != row['bytes'] or sha(path.read_bytes()) != row['sha256']):
            return False
    return True


def atomic_bytes(path, data):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.install-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
    emit('M12_PACKAGE_INSTALLED', **receipt)
    return destination


def existing_install(home):
    _, manifest_data, manifest = checked_payload()
    destination = package_path(home)
    if not verify_install(destination, manifest_data, manifest):
        raise ValueError('EXACT_INSTALLED_PACKAGE_REQUIRED_FOR_STATUS')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--output', type=Path, default=Path('runs/r12_controlled_mechanism'))
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare and args.resume:
        raise ValueError('CHOOSE_PREPARE_OR_RESUME')
    if args.status and any((args.prepare, args.submit, args.resume, args.install)):
        raise ValueError('STATUS_IS_READ_ONLY_USE_SEPARATELY')
    if args.submit and not (args.prepare or args.resume):
        raise ValueError('SUBMIT_REQUIRES_PREPARE_OR_RESUME')
    if args.resume and not args.submit:
        raise ValueError('RESUME_REQUIRES_EXPLICIT_SUBMIT')
    root = args.root.expanduser().resolve()
    output = args.output.expanduser()
    output = (output if output.is_absolute() else root / output).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError('OUTPUT_MUST_BE_INSIDE_PROJECT_ROOT')
    python = str(Path(args.python).expanduser().absolute())
    package = existing_install(args.home) if args.status else install(args.home)
    if not any((args.prepare, args.submit, args.status)):
        emit('M12_CODE_READY_NO_ACTION', package=str(package),
             next_operation='--prepare --submit', scientific_workers_stopped=0,
             slurm_calls=0, gpu_used=False)
        return 0
    import subprocess
    entry = str(package / 'paper3m12/entry.py')
    prefix = [python, '-I', '-S', '-B', '-u', entry]
    plan = output / 'plan.json'
    if args.status:
        return subprocess.run(prefix + ['status', '--plan', str(plan)], check=False).returncode
    if args.prepare:
        command = prefix + ['prepare', '--root', str(root), '--output', str(output),
                            '--package', str(package)]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    if args.submit:
        return subprocess.run(prefix + ['start', '--plan', str(plan), '--submit'], check=False).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
