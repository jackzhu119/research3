"""Install a selection cache into future owned R1 startup imports only.

The current child and its in-memory modules are deliberately never inspected or
signalled. Only the two retained operational action_startup.py files may change.
"""
from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

MARKER = 'PAPER3_R1_SELECTION_CONTINUATION_V1'
BASE_SHA256 = '1aeaabfc2329ce460cf4542cd49944753208efc502274c5cab4e6f178383cd0a'
RUNTIME = 'runs/training_campaign_parallel_v1'
TARGETS = (
    'paper3-parallel-migrations/fa00be482b1440d3fae7fff1/round1',
    'paper3-r1-repairs/d8ee006073f23292089b/active-round1/c36635429230c732b802',
)
HELPERS = ('selection_cache.py', 'b5_retry_budget.py')
LIMIT = 4 * 1024 * 1024
_MISSING = object()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path, *, owned=True):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (owned and info.st_uid != os.getuid()):
        raise ValueError('R1_SELECTION_FILE_NOT_OWNED_REGULAR')
    if info.st_size > LIMIT:
        raise ValueError('R1_SELECTION_METADATA_TOO_LARGE')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('R1_SELECTION_SYMLINK_COMPONENT')
    with path.open('rb') as handle:
        data = handle.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError('R1_SELECTION_METADATA_TOO_LARGE')
    return data


def _json(path, *, optional=False):
    try:
        value = json.loads(_bytes(path))
    except FileNotFoundError:
        if optional:
            return {}
        raise
    if not isinstance(value, dict):
        raise ValueError('R1_SELECTION_METADATA_NOT_OBJECT')
    return value


def _option(command, option):
    if not isinstance(command, list) or command.count(option) != 1:
        return None
    at = command.index(option)
    return command[at + 1] if at + 1 < len(command) else None


def discover_packages(root, home):
    """Use retained launch/history, with only the two exact known fallbacks."""
    root, home = Path(root).expanduser().absolute(), Path(home).expanduser().absolute()
    runtime = root / RUNTIME
    launch = _json(runtime / 'launch.json', optional=True)
    state = _json(runtime / 'scheduler-state.json', optional=True)
    allowed = {home / name for name in TARGETS}
    values = [launch.get('package'), _option(launch.get('worker_command'), '--package')]
    slots = state.get('slots', {})
    if not isinstance(slots, dict):
        raise ValueError('R1_SELECTION_SCHEDULER_HISTORY_INVALID')
    for slot in slots.values():
        if not isinstance(slot, dict) or not isinstance(slot.get('attempts', []), list):
            raise ValueError('R1_SELECTION_SCHEDULER_HISTORY_INVALID')
        for attempt in slot.get('attempts', []):
            if isinstance(attempt, dict):
                values.append(_option(attempt.get('command'), '--package'))
    found = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        path = Path(value).expanduser().absolute()
        if path / 'round1' in allowed:
            path = path / 'round1'
        if path not in allowed and ((path / 'action_startup.py').exists()
                                    or (path / 'round1/action_startup.py').exists()):
            raise ValueError('R1_SELECTION_UNRECOGNIZED_ACTIVE_OPERATIONAL_PACKAGE')
        if path in allowed and (path / 'action_startup.py').exists():
            found.add(path)
    # Both retained launch families can be needed by a later replacement child.
    # Do not glob archived packages or touch another version of the experiment.
    for path in allowed:
        if (path / 'action_startup.py').exists() or (path / 'action_startup.py').is_symlink():
            found.add(path)
    return sorted(found)


def strip_own_block(source):
    begin, end = '\n# ' + MARKER + '\n', '# END_' + MARKER + '\n'
    if MARKER not in source:
        return source
    if source.count(begin) != 1 or source.count(end) != 1 or source.count(MARKER) != 2:
        raise ValueError('R1_SELECTION_INCOMPLETE_OR_DUPLICATE_OWN_BLOCK')
    at = source.index(begin)
    stop = source.find(end, at)
    if stop < at or stop + len(end) != len(source):
        raise ValueError('R1_SELECTION_OWN_BLOCK_NOT_EXACT_TRAILER')
    return source[:at]


def _old_specs(source):
    tree = ast.parse(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == '_paper3_r1_complete_helper'
               for node in tree.body):
        raise ValueError('R1_SELECTION_EXISTING_COMPLETE_HELPER_REQUIRED')
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '_paper3_r1_complete_specs'
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('R1_SELECTION_EXISTING_COMPLETE_SPECS_REQUIRED')
    try:
        specs = ast.literal_eval(assignments[0].value)
        path, digest = specs['verified_context_cache.py']
    except (ValueError, TypeError, KeyError):
        raise ValueError('R1_SELECTION_EXISTING_CACHE_SPEC_INVALID') from None
    if (not isinstance(path, str) or not Path(path).is_absolute()
            or not isinstance(digest, str) or len(digest) != 64):
        raise ValueError('R1_SELECTION_EXISTING_CACHE_SPEC_INVALID')
    data = _bytes(path)
    if sha(data) != digest:
        raise ValueError('R1_SELECTION_EXISTING_CACHE_SHA256_DIFFERS')
    return specs


def checked_helpers(package):
    package = Path(package).expanduser().absolute()
    manifest = _json(package / 'MANIFEST.json')
    files = manifest.get('files')
    if not isinstance(files, dict):
        raise ValueError('R1_SELECTION_PACKAGE_MANIFEST_REQUIRED')
    result = {}
    for name in HELPERS:
        row = files.get(name)
        if not isinstance(row, dict):
            raise ValueError('R1_SELECTION_HELPER_MANIFEST_REQUIRED')
        path = package / name
        data = _bytes(path)
        if len(data) != row.get('bytes') or sha(data) != row.get('sha256'):
            raise ValueError('R1_SELECTION_HELPER_SHA256_DIFFERS')
        compile(data, str(path), 'exec')
        result[name] = (path, sha(data))
    return result


def patch_source(original, helpers, root, *, expected_base_sha=BASE_SHA256):
    source = strip_own_block(original.decode('utf-8'))
    if sha(source.encode('utf-8')) != expected_base_sha:
        raise ValueError('R1_SELECTION_UNKNOWN_STARTUP_BASE_VERSION')
    _old_specs(source)
    if not isinstance(helpers, dict) or set(helpers) != set(HELPERS):
        raise ValueError('R1_SELECTION_EXACT_HELPER_SET_REQUIRED')
    path, checksum = helpers['selection_cache.py']
    b5_path, b5_checksum = helpers['b5_retry_budget.py']
    project = str(Path(root).expanduser().absolute())
    block = (
        "_paper3_r1_selection_continuation = _paper3_r1_complete_helper(%r, %r)\n"
        "_paper3_r1_selection_old_cache = _paper3_r1_complete_helper(\n"
        "    *_paper3_r1_complete_specs['verified_context_cache.py'])\n"
        "_paper3_r1_selection_continuation.install(\n"
        "    __import__('sys').modules[__name__], _paper3_r1_selection_old_cache, %r)\n"
        "_paper3_r1_b5_retry_continuation = _paper3_r1_complete_helper(%r, %r)\n"
        "_paper3_r1_b5_retry_continuation.install(__import__('sys').modules[__name__], %r)\n"
    ) % (str(path), checksum, project, str(b5_path), b5_checksum, project)
    fixed = (source + '\n# ' + MARKER + '\n' + block + '# END_' + MARKER + '\n').encode()
    compile(fixed, 'action_startup.py', 'exec')
    return fixed


def atomic(path, data, *, expected=_MISSING):
    """Private same-filesystem staging, read-before-replace and bounded retries."""
    path = Path(path)
    transient = {errno.ENOENT, errno.EINTR, errno.EAGAIN, errno.EIO,
                 getattr(errno, 'ESTALE', 116), errno.ETIMEDOUT}
    for attempt in range(4):
        stage = None
        try:
            for parent in (path.parent, *path.parents):
                if parent.is_symlink():
                    raise ValueError('R1_SELECTION_ATOMIC_PARENT_IS_SYMLINK')
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = stat.S_IMODE(path.lstat().st_mode) if path.exists() else 0o600
            stage = Path(tempfile.mkdtemp(prefix='.r1-selection-stage-', dir=path.parent))
            temporary = stage / 'payload'
            with temporary.open('xb') as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            temporary.chmod(mode)
            try:
                current = _bytes(path)
            except FileNotFoundError:
                current = _MISSING
            if current == data:
                return
            if expected is _MISSING:
                if current is not _MISSING:
                    raise RuntimeError('R1_SELECTION_NEW_FILE_ALREADY_DIFFERS')
                os.link(temporary, path)
            else:
                if current != expected:
                    raise RuntimeError('R1_SELECTION_SOURCE_CHANGED_BEFORE_REPLACE')
                os.replace(temporary, path)
            if _bytes(path) != data:
                raise RuntimeError('R1_SELECTION_ATOMIC_WRITE_NOT_OBSERVED')
            return
        except OSError as error:
            try:
                if _bytes(path) == data:
                    return
            except OSError:
                pass
            if error.errno not in transient or attempt == 3:
                raise
            time.sleep(.02 * (2 ** attempt))
        finally:
            if stage is not None:
                shutil.rmtree(stage, ignore_errors=True)


def apply(root, home, package, *, expected_base_sha=BASE_SHA256):
    """Only future startup imports change; there is no process-control channel."""
    root, home = Path(root).expanduser().absolute(), Path(home).expanduser().absolute()
    package = Path(package).expanduser().absolute()
    helpers = checked_helpers(package)
    found = discover_packages(root, home)
    if not found:
        raise ValueError('R1_SELECTION_EXISTING_OPERATIONAL_TARGET_NOT_FOUND')
    base = home / 'paper3-r1-selection-continuation'
    if base.is_symlink():
        raise ValueError('R1_SELECTION_OWN_BASE_IS_SYMLINK')
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not stat.S_ISDIR(base.lstat().st_mode) or base.stat().st_uid != os.getuid():
        raise ValueError('R1_SELECTION_OWN_BASE_NOT_OWNED_DIRECTORY')
    lock_path = base / 'apply.lock'
    if lock_path.is_symlink():
        raise ValueError('R1_SELECTION_OWN_LOCK_IS_SYMLINK')
    with lock_path.open('a') as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('R1_SELECTION_OWN_LOCK_NOT_OWNED_REGULAR')
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepared, preserved = [], {}
        for directory in found:
            target = directory / 'action_startup.py'
            original = _bytes(target)
            fixed = patch_source(original, helpers, root, expected_base_sha=expected_base_sha)
            prepared.append((target, original, fixed))
            old_specs = _old_specs(strip_own_block(original.decode('utf-8')))
            old_cache = Path(old_specs['verified_context_cache.py'][0])
            preserved[old_cache] = _bytes(old_cache)
            for name in ('worker.py', 'kernel_bridge.py', 'audited_runtime.py', 'scheduler.py', 'START.py'):
                path = directory / name
                if path.exists():
                    preserved[path] = _bytes(path)
        for relative in ('p3work/development_evaluation.py', 'p3work/classical_training.py',
                         'p3work/method_registry.py', 'p3work/meta_training.py'):
            path = root / relative
            if path.exists():
                preserved[path] = _bytes(path)
        changed = []
        try:
            for target, original, fixed in prepared:
                if original == fixed:
                    continue
                backup = base / 'backups' / (sha(original) + '.py')
                atomic(backup, original)
                if checked_helpers(package) != helpers:
                    raise RuntimeError('R1_SELECTION_HELPER_CHANGED_DURING_PREPARATION')
                if _bytes(target) != original:
                    raise RuntimeError('R1_SELECTION_SOURCE_CHANGED_DURING_PREPARATION')
                atomic(target, fixed, expected=original)
                changed.append((target, original, fixed))
            if any(_bytes(path) != data for path, data in preserved.items()):
                raise RuntimeError('R1_SELECTION_PRESERVED_SOURCE_CHANGED_CONCURRENTLY')
            if checked_helpers(package) != helpers:
                raise RuntimeError('R1_SELECTION_HELPER_CHANGED_DURING_APPLY')
        except BaseException as original_error:
            # Never overwrite a third party's later edit during rollback.
            rollback_errors = []
            for target, original, fixed in reversed(changed):
                try:
                    if _bytes(target) == fixed:
                        atomic(target, original, expected=fixed)
                except (OSError, ValueError, RuntimeError) as error:
                    rollback_errors.append(dict(path=str(target), error_type=type(error).__name__))
            if rollback_errors and hasattr(original_error, 'add_note'):
                original_error.add_note('R1 selection rollback could not observe/restore: '
                                        + json.dumps(rollback_errors, sort_keys=True))
            raise
        receipt = dict(
            schema_version='paper3-r1-selection-startup-continuation-v1',
            status='R1_FUTURE_STARTUP_CONTINUATION_INSTALLED', activation='FUTURE_ACTIONS_ONLY',
            package=str(package), root=str(root),
            helpers={name:dict(path=str(path), sha256=checksum) for name,(path,checksum) in helpers.items()},
            operational_files=[dict(path=str(path), before_sha256=sha(before),
                                    after_sha256=sha(after), changed=before != after)
                               for path, before, after in prepared],
            preserved_sources=[dict(path=str(path), sha256=sha(data))
                               for path, data in sorted(preserved.items())],
            existing_children_modified=False, current_selection_interrupted=False,
            jobs_submitted=0, jobs_cancelled=0, signals_sent=0, scientific_files_changed=0,
            plan_files_changed=0, checkpoint_files_changed=0, controller_or_queue_modified=False)
        output = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
        receipt_path = base / ('applied-' + sha(output) + '.json')
        try:
            atomic(receipt_path, output)
        except Exception as error:
            # The two hooks were already read-back verified. A telemetry-only
            # failure must not be labelled as an unapplied operational change.
            return dict(receipt, receipt_path=str(receipt_path), receipt_persisted=False,
                        receipt_error_type=type(error).__name__)
        return dict(receipt, receipt_path=str(receipt_path), receipt_persisted=True)
