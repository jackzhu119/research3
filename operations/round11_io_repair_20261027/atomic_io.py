"""Bounded, whole-file R11 writes for a shared filesystem; standard library only.

The signed R11 sources and encoded result bytes are unchanged.  Staging lives
in an operation-owned, private directory on the destination filesystem.  Only
that operation's payload/directory is removed; no glob or age-based cleanup is
performed.  Transient I/O is retried; an unsuccessful scientific write raises.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

REPAIR_ID = 'paper3-r11-private-staging-atomic-v1'
MAX_ATTEMPTS = 6
RETRY_DELAYS = (0.05, 0.15, 0.4, 0.8, 1.6)
TRANSIENT_ERRNOS = frozenset(
    code for code in (
        errno.ENOENT, errno.EINTR, errno.EAGAIN, errno.EBUSY, errno.EIO,
        getattr(errno, 'ESTALE', None), getattr(errno, 'ETIMEDOUT', None),
    ) if code is not None
)


def _matches(path, data):
    """A rename RPC can report a failure after the server has committed it."""
    try:
        with Path(path).open('rb') as handle:
            if os.fstat(handle.fileno()).st_size != len(data):
                return False
            expected = hashlib.sha256(data).digest()
            actual = hashlib.sha256()
            count = 0
            while True:
                block = handle.read(min(1024 * 1024, len(data) + 1 - count))
                if not block:
                    break
                count += len(block)
                if count > len(data):
                    return False
                actual.update(block)
            return count == len(data) and actual.digest() == expected
    except OSError:
        return False


def _cleanup_owned(stage, payload):
    # Each path was generated exclusively for this invocation.  In particular,
    # never sweep the destination directory or another writer's staging area.
    try:
        payload.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        stage.rmdir()
    except OSError:
        pass


def atomic_bytes(path, data, *, attempts=MAX_ATTEMPTS, sleep=time.sleep):
    """Replace a whole file, retrying a bounded set of transient I/O failures.

    Permanent failures (permission, full disk, invalid path, cross-device
    rename, etc.) propagate immediately.  After exhaustion the last error
    propagates.  Existing target content is never removed before replacement.
    Nothing reports success unless rename succeeded or exact destination
    bytes prove an ambiguously acknowledged rename actually committed.
    """
    path = Path(path)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError('atomic_bytes needs a bytes-like payload')
    data = bytes(data)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError('attempts must be a positive integer')
    for attempt in range(attempts):
        stage = payload = None
        commit_attempted = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # mkdtemp creates mode 0700; staging remains on the same filesystem.
            # The payload name does not imitate a published .json / .npz file.
            stage = Path(tempfile.mkdtemp(prefix='.r11-atomic-', dir=path.parent))
            payload = stage / 'payload.bin'
            descriptor = os.open(payload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            commit_attempted = True
            os.replace(payload, path)
            return
        except OSError as error:
            retryable = error.errno in TRANSIENT_ERRNOS
            if retryable and commit_attempted and _matches(path, data):
                return
            if not retryable or attempt + 1 >= attempts:
                raise
        finally:
            if stage is not None and payload is not None:
                _cleanup_owned(stage, payload)
        sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])


def install_atomic_patch(common):
    """Patch common and exact already-imported aliases in this process only."""
    previous_bytes = common.atomic_bytes
    previous_atomic = common.atomic
    already_patched = previous_bytes is atomic_bytes

    def atomic(path, value):
        return atomic_bytes(path, common.encoded(value))

    if not already_patched:
        common.atomic_bytes = atomic_bytes
        common.atomic = atomic
    else:
        atomic = previous_atomic
    aliases = 0
    for name, module in list(sys.modules.items()):
        if name.startswith('paper3r11.') and module is not None:
            if getattr(module, 'atomic_bytes', None) is previous_bytes:
                module.atomic_bytes = atomic_bytes
                aliases += 1
            if getattr(module, 'atomic', None) is previous_atomic:
                module.atomic = atomic
                aliases += 1
    scheduler = sys.modules.get('paper3r11.platform.scheduler')
    scheduler_patched = False
    if scheduler is not None and hasattr(scheduler, '_atomic'):
        if getattr(scheduler._atomic, '_r11_atomic_repair_id', None) != REPAIR_ID:
            def scheduler_atomic(path, value):
                # Preserve scheduler._atomic's original JSON bytes, including
                # its original serializer options; only the write path changes.
                data = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   indent=2) + '\n').encode('utf-8')
                return atomic_bytes(path, data)
            scheduler_atomic._r11_atomic_repair_id = REPAIR_ID
            scheduler._atomic = scheduler_atomic
            scheduler_patched = True
    return dict(status='ALREADY_PATCHED' if already_patched else 'IN_MEMORY_ATOMIC_PATCH_INSTALLED',
                repair_id=REPAIR_ID, aliases=aliases, scheduler_patched=scheduler_patched,
                attempts=MAX_ATTEMPTS, source_files_changed=False, scientific_bytes_changed=False)


# Convenient name for the worker/controller sidecar's bootstrap.
install_patch = install_atomic_patch
