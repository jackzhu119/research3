"""Small durable records and non-fatal telemetry for the M12 experiment.

No scientific imports, memory-limit changes, network calls or signals occur here.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

TRANSIENT = {errno.ENOENT, errno.EAGAIN, errno.EINTR, errno.EIO,
             getattr(errno, 'ESTALE', 116), errno.ETIMEDOUT}


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


sha = sha_file


def read_json(path, default=None, max_bytes=8 * 1024 * 1024):
    """Allocate only the actual file size, never max_bytes+1 on a tiny record."""
    try:
        with Path(path).open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if not 0 <= size <= max_bytes:
                raise ValueError('M12 JSON record exceeds declared size limit')
            data = handle.read(size + 1)
        if len(data) != size:
            raise ValueError('M12 JSON record changed during reading')
        return json.loads(data)
    except FileNotFoundError:
        return default


read = read_json


def atomic_bytes(path, data, *, attempts=5, sleep=time.sleep):
    """Durable unique staging; retry narrow shared-filesystem interruptions only.

    An ambiguous replace acknowledgement is accepted only when the target has
    the exact requested bytes. ENOSPC/permissions and scientific errors propagate.
    """
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError('atomic_bytes requires exact bytes')
    last = None
    for attempt in range(attempts):
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stage = path.parent / '.m12-atomic'
            stage.mkdir(mode=0o700, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix='write-', dir=stage)
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return path
        except OSError as error:
            last = error
            try:
                if path.is_file() and path.stat().st_size == len(data) and path.read_bytes() == data:
                    return path
            except OSError:
                pass
            if error.errno not in TRANSIENT or attempt + 1 == attempts:
                raise
            sleep(min(.02 * (2 ** attempt), .2))
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass
    raise last


def atomic_json(path, value, **kwargs):
    return atomic_bytes(path, encoded(value), **kwargs)


atomic = atomic_json


def best_effort_json(path, value):
    """Operational status must not interrupt scientific work."""
    try:
        atomic_json(path, value)
        return True
    except (OSError, TypeError, ValueError):
        return False


def safe_read(path, default=None, max_bytes=8 * 1024 * 1024):
    try:
        return read_json(path, default, max_bytes=max_bytes)
    except (OSError, ValueError, TypeError):
        return default


def emit(event, **values):
    try:
        print(json.dumps(dict(event=event, utc=utc(), **values), ensure_ascii=False,
                         allow_nan=False), flush=True)
    except (OSError, TypeError, ValueError):
        pass


@contextmanager
def lock(path, blocking=True, *, nonblocking=None):
    if nonblocking is not None:
        blocking = not nonblocking
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


locked = lock


def task_key(task):
    value = task.get('key')
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,240}', value) or value in {'.', '..'}:
        raise ValueError('M12 task requires a safe fixed key')
    return value


def task_identity(plan, task=None):
    return digest(plan if task is None else task)


def task_folder(plan, task):
    return Path(plan['output']) / 'tasks' / task_key(task)


def load_plan(path):
    path = Path(path).expanduser().resolve()
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get('tasks'), list):
        raise ValueError('M12 plan with an explicit task list is required')
    root = Path(value.get('root', path.parent)).expanduser().resolve()
    output = Path(value['output']).expanduser()
    if not output.is_absolute():
        output = root / output
    value['root'] = str(root)
    value['output'] = str(output.resolve())
    tasks = value['tasks']
    keys = [task_key(task) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate M12 scientific task keys')
    known = set(keys)
    for task in tasks:
        dependencies = task.get('dependencies', task.get('depends_on', []))
        if not isinstance(dependencies, list) or not set(dependencies) <= known or task['key'] in dependencies:
            raise ValueError('M12 task dependencies must bind registered task keys')
        gate = task.get('gate')
        if gate and (not isinstance(gate, dict) or gate.get('task') not in known or gate.get('task') == task['key']):
            raise ValueError('M12 gate must bind another registered task')
    # Detect a malformed DAG before allocating GPUs, not at each fit.
    done = set()
    while len(done) < len(keys):
        ready = [t['key'] for t in tasks if t['key'] not in done
                 and (set(t.get('dependencies', t.get('depends_on', [])))
                      | ({t['gate']['task']} if t.get('gate') else set())) <= done]
        if not ready:
            raise ValueError('M12 task dependency graph contains a cycle')
        done.update(ready)
    value.setdefault('recipe_sha256', digest({k: v for k, v in value.items()
                                             if k not in {'recipe_sha256', 'runtime', 'python'}}))
    value.setdefault('runtime', str(Path(value['output']) / 'runtime'))
    runtime = Path(value['runtime'])
    if not runtime.is_absolute():
        value['runtime'] = str(root / runtime)
    value['_plan_path'] = str(path)
    return path, value


def verify_result(plan, task, *, deep=False):
    """Check an actual scientific completion, never manufacture one for status.

    The worker seals artifacts. Routine status checks the small receipt only;
    explicit deep recovery/archive verification can stream all artifact hashes.
    """
    folder = task_folder(plan, task)
    result = safe_read(folder / 'result.json')
    if not isinstance(result, dict) or result.get('status') != 'COMPLETE':
        return None
    key = result.get('task_key')
    if key is None:
        inner = result.get('task')
        key = inner.get('key') if isinstance(inner, dict) else None
    if key != task['key']:
        return None
    if result.get('task_identity') != digest(task):
        return None
    if plan.get('recipe_sha256') is not None and result.get('recipe_sha256') != plan['recipe_sha256']:
        return None
    files = result.get('files')
    if not isinstance(files, (dict, list)):
        return None
    if deep:
        rows = files.items() if isinstance(files, dict) else [(row.get('path'), row) for row in files]
        output = Path(plan['output']).resolve()
        for name, item in rows:
            if not isinstance(item, dict) or not isinstance(name, str):
                return None
            relative = Path(item.get('path', name))
            path = relative if relative.is_absolute() else output / relative
            try:
                if not path.resolve().is_relative_to(output) or path.is_symlink() or not path.is_file():
                    return None
                if path.stat().st_size != item['bytes'] or sha_file(path) != item['sha256']:
                    return None
            except (OSError, KeyError, ValueError):
                return None
    return result
