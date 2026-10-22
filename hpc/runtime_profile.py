"""Capture the installed, explicitly selected school runtime; never install it.

The retained RunPod lock is not a school dependency specification.  This
adapter records the complete installed inventory and binds it to one registered
school profile.  It changes no model, optimizer, data, randomness or fit budget.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import tempfile

PROFILE = 'school-cu121-v1'
REGISTRY = {
    'schema_version': 'paper3-hpc-registered-runtime-v1',
    'profile': PROFILE,
    'python': '3.12.3',
    'pins': {'torch': '2.5.1+cu121', 'numpy': '2.2.6', 'scipy': '1.15.3',
             'scikit-learn': '1.6.1', 'pytest': '8.4.2'},
}
INVENTORY = 'audit/gpu_resolved_dependencies.json'
EVIDENCE = 'audit/hpc_runtime_profile.json'
MODULES = {'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy',
           'scikit-learn': 'sklearn', 'pytest': 'pytest'}


class RuntimeProfileError(ValueError):
    pass


def _need(value, code):
    if not value:
        raise RuntimeProfileError(code)


def _raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + '\n').encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _path(value):
    value = Path(os.path.abspath(value))
    _need(not any(part.is_symlink() for part in (value, *value.parents)),
          'runtime_profile_symlink_forbidden')
    return value


def _root(root):
    root = _path(root)
    _need(root.is_dir(), 'runtime_profile_existing_root_required')
    return root


def selected_profile(root):
    """An explicit school selector cannot silently fall back to RunPod pins."""
    profile = os.environ.get('PAPER3_HPC_RUNTIME_PROFILE')
    if profile is None:
        return None
    _need(profile == PROFILE, 'runtime_profile_not_registered')
    root = _root(root)
    _need(os.environ.get('PAPER3_HPC_ROOT') == str(root)
          and os.environ.get('PAPER3_HPC_STAGE') in {'pilot', 'train'},
          'runtime_profile_explicit_project_and_stage_required')
    return profile


def registered_requirements(profile):
    _need(profile == PROFILE, 'runtime_profile_not_registered')
    return tuple(name + '==' + version for name, version in sorted(REGISTRY['pins'].items()))


def _installed_inventory():
    names = {}
    for distribution in metadata.distributions():
        name, version = distribution.metadata.get('Name'), distribution.version
        _need(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', name)
              and isinstance(version, str) and re.fullmatch(r'[A-Za-z0-9_.+!-]+', version),
              'runtime_profile_distribution_metadata_invalid')
        name = re.sub(r'[-_.]+', '-', name).lower()
        _need(name not in names or names[name] == version,
              'runtime_profile_conflicting_installed_distributions')
        # The verified node overlay and original site can expose the same
        # distribution twice. Equal versions are one installed identity.
        names[name] = version
    _need(names, 'runtime_profile_complete_inventory_required')
    return [{'name': name, 'version': version} for name, version in sorted(names.items())]


def _runtime_versions():
    return {name: str(importlib.import_module(module).__version__)
            for name, module in MODULES.items()}


def _read(path):
    path = _path(path)
    _need(path.is_file(), 'runtime_profile_existing_evidence_required')
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_size <= 2 * 1024**2,
          'runtime_profile_regular_bounded_evidence_required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(2 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _need(identity(before) == identity(opened) == identity(after) == identity(path.stat())
          and len(raw) == before.st_size,
          'runtime_profile_evidence_changed_during_read')
    return raw


def _descriptor(root, relative, raw):
    return {'path': relative, 'sha256': _sha(raw)}


def _observation(root, profile):
    root = _root(root)
    _need(profile == PROFILE and selected_profile(root) == profile,
          'runtime_profile_explicit_selection_required')
    _need(platform.python_version() == REGISTRY['python'],
          'runtime_profile_python_version_mismatch')
    inventory = _installed_inventory()
    names = {row['name']: row['version'] for row in inventory}
    _need(all(names.get(name) == version for name, version in REGISTRY['pins'].items()),
          'runtime_profile_registered_pin_mismatch')
    versions = _runtime_versions()
    _need(versions == REGISTRY['pins'],
          'runtime_profile_imported_modules_mismatch')
    inventory_raw = _raw(inventory)
    evidence = {
        'schema_version': 'paper3-hpc-installed-runtime-profile-v1',
        'profile': profile, 'project': str(root), 'python': platform.python_version(),
        'python_executable': sys.executable,
        'registered_profile_sha256': _sha(_raw(REGISTRY)),
        'original_RunPod_lock_sha256': _sha(_read(root / 'requirements-gpu.lock')),
        'resolved_dependencies': _descriptor(root, INVENTORY, inventory_raw),
        'inventory_packages': len(inventory), 'actual_module_versions': versions,
        'packages_installed_by_capture': 0, 'network_operations_by_capture': 0,
        'scientific_parameters_changed': False,
    }
    return root, inventory_raw, _raw(evidence)


def _publish(path, raw):
    path = _path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _path(path.parent)
    if path.exists():
        _need(_read(path) == raw, 'runtime_profile_existing_evidence_differs_preserve_original')
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.runtime-profile-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _need(_read(path) == raw, 'runtime_profile_existing_evidence_differs_preserve_original')
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_runtime_profile(root, profile):
    """Recheck both immutable documents against the entire current runtime."""
    root, inventory, evidence = _observation(root, profile)
    _need(_read(root / INVENTORY) == inventory,
          'runtime_profile_full_inventory_changed_preserve_original')
    _need(_read(root / EVIDENCE) == evidence,
          'runtime_profile_provenance_changed_preserve_original')
    return _descriptor(root, EVIDENCE, evidence)


def prepare_runtime_profile(root, profile):
    """Write genuine runtime evidence only inside the allocated GPU stage."""
    root = _root(root)
    _need(selected_profile(root) == profile, 'runtime_profile_explicit_selection_required')
    from hpc.run_stage import guard
    try:
        guard(root, os.environ['PAPER3_HPC_STAGE'])
    except ValueError:
        raise RuntimeProfileError('runtime_profile_allocated_GPU_job_required') from None
    root, inventory, evidence = _observation(root, profile)
    # Do not reconstruct an orphaned proof or replace an older environment's
    # evidence. A crash after publishing only the genuine inventory is safe.
    _need(not (root / EVIDENCE).exists() or (root / INVENTORY).exists(),
          'runtime_profile_provenance_without_inventory_preserve_original')
    if (root / INVENTORY).exists():
        _need(_read(root / INVENTORY) == inventory,
              'runtime_profile_existing_evidence_differs_preserve_original')
    if (root / EVIDENCE).exists():
        _need(_read(root / EVIDENCE) == evidence,
              'runtime_profile_existing_evidence_differs_preserve_original')
    _publish(root / INVENTORY, inventory)
    _publish(root / EVIDENCE, evidence)
    return validate_runtime_profile(root, profile)
