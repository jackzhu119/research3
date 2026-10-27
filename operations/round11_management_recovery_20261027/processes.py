"""Exact same-user R11 management process identities; no scientific signals."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROLES = ('controller', 'paper', 'full', 'report', 'monitor')
PYTHON_FLAGS = frozenset(('-I', '-S', '-B', '-u', '-E', '-s', '-O', '-OO', '-q'))
OLD_HASHES = {
    ('paper3-r11-operational-repair', 'repair.py'): '3d63152c39ccaa50f99823e6159848fa7ba637128635c08b033726871fcd85e9',
    ('paper3-r11-io-repair', 'resume_io.py'): 'da7078c0251e115b002b4fd3e8bea6d431b52bbafca8474042e9402d6d39cdd2',
}


def argument(args, name):
    if args.count(name) != 1:
        return None
    at = args.index(name)
    return args[at + 1] if at + 1 < len(args) else None


def raw_identity(pid, *, proc_root='/proc'):
    try:
        pid = int(pid)
        if pid <= 1:
            return None
        folder = Path(proc_root) / str(pid)
        owner = folder.stat().st_uid
        if owner != os.getuid():
            return None
        fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X', 'x'):
            return None
        args = [part.decode('utf-8') for part in (folder/'cmdline').read_bytes().split(b'\0') if part]
        return dict(pid=pid, uid=owner, start_ticks=fields[19], args=args)
    except (OSError, ValueError, IndexError, UnicodeError, TypeError):
        return None


def _script_position(args):
    at = 1
    while at < len(args) and args[at] in PYTHON_FLAGS:
        at += 1
    return at


def _trusted_script(script, env):
    script = Path(script).resolve()
    current = script in {HERE/'entry.py', HERE/'supervisor.py'}
    if current:
        return 'current'
    if script == Path(env['package']).resolve()/'paper3r11/platform/entry.py':
        return 'original'
    for (base, filename), expected in OLD_HASHES.items():
        if (script.name == filename and script.parent.parent == Path.home()/base
                and script.is_file()
                and hashlib.sha256(script.read_bytes()).hexdigest() == expected):
            return 'previous'
    return None


def _authorized_launch(env):
    """Only the persisted exact campaign launch can authorize a plan alias."""
    try:
        path = Path(env['runtime'])/'launch.json'
        with path.open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if not 0 < size <= 1024*1024:
                return {}
            data = handle.read(size+1)
        if len(data) != size:
            return {}
        value = json.loads(data)
        if (not isinstance(value,dict)
                or value.get('plan_sha256') != env['header']['plan_sha256']
                or value.get('campaign_identity') != env['header']['recipe_sha256']):
            return {}
        return value
    except (OSError, ValueError, TypeError):
        return {}


def identify(pid, env, role, *, proc_root='/proc'):
    """Match actual Python script position, complete role and frozen plan.

    Merely mentioning an entry path or role in another command is insufficient.
    The worker/science/report-child modes are deliberately absent here.
    """
    if role not in ROLES:
        return None
    value = raw_identity(pid, proc_root=proc_root)
    if value is None:
        return None
    try:
        args = value['args']
        launch = _authorized_launch(env)
        interpreters = {Path(sys.executable).resolve(),
                        Path(env['header'].get('python', sys.executable)).expanduser().resolve()}
        if isinstance(launch.get('python'),str) and Path(launch['python']).is_absolute():
            interpreters.add(Path(launch['python']).expanduser().resolve())
        if not args or not Path(args[0]).is_absolute() or Path(args[0]).resolve() not in interpreters:
            return None
        at = _script_position(args)
        if at + 1 >= len(args) or not Path(args[at]).is_absolute():
            return None
        script = Path(args[at]).resolve(); trust = _trusted_script(script, env)
        if trust is None:
            return None
        mode = args[at + 1]
        if trust == 'original':
            if role == 'controller':
                matches = mode == 'controller'
            elif role == 'report':
                matches = mode == 'report-loop'
            else:
                matches = mode == 'upload' and argument(args, '--channel') == role
        else:
            matches = mode == role
        if not matches:
            return None
        tail = args[at + 2:]
        values = {'--plan', '--package', '--work', '--python', '--credentials', '--launch-token', '--channel', '--interval'}
        flags = {'--submit', '--wait-paper-lock'}
        seen = set(); pos = 0
        while pos < len(tail):
            key = tail[pos]
            if key in seen or key not in values | flags:
                return None
            seen.add(key)
            if key in values:
                if pos + 1 >= len(tail) or tail[pos + 1].startswith('--'):
                    return None
                if key == '--interval':
                    interval = float(tail[pos+1])
                    if not math.isfinite(interval) or interval <= 0:
                        return None
                pos += 2
            else:
                pos += 1
        plan = argument(args, '--plan')
        if not plan:
            return None
        candidate = Path(plan).expanduser().resolve()
        allowed_plans = {Path(env['path']).resolve()}
        alias = launch.get('plan')
        if isinstance(alias,str) and Path(alias).is_absolute():
            allowed_plans.add(Path(alias).expanduser().resolve())
        if (candidate not in allowed_plans or not candidate.is_file()
                or env['common'].sha(candidate) != env['header']['plan_sha256']):
            return None
        return dict(value, role=role, script=str(script), current_patch=trust == 'current',
                    launch_token=argument(args, '--launch-token'), work=argument(args, '--work'))
    except (OSError, ValueError, TypeError):
        return None


def find_owned(env, role, candidates=(), *, proc_root='/proc', identify_fn=identify):
    """Recover lost PID telemetry without signalling or reading environments."""
    found = []; seen = set()
    for pid in candidates:
        if pid is None or str(pid) in seen:
            continue
        seen.add(str(pid))
        item = identify_fn(pid, env, role, proc_root=proc_root)
        if item is not None:
            found.append(item)
    try:
        for folder in Path(proc_root).iterdir():
            if not folder.name.isdigit() or folder.name in seen:
                continue
            seen.add(folder.name)
            item = identify_fn(int(folder.name), env, role, proc_root=proc_root)
            if item is not None:
                found.append(item)
    except OSError:
        pass
    return sorted(found, key=lambda item: (not item['current_patch'], item['pid']))


def same_identity(left, right):
    return right is not None and all(left.get(key) == right.get(key)
                                     for key in ('pid', 'uid', 'start_ticks', 'args', 'role'))


def stop_owned(identity, env, *, identify_fn=identify, send=os.kill,
               sleep=time.sleep, clock=time.monotonic, grace=2.):
    """Request graceful management stop, with no escalation or group signal."""
    role = identity.get('role')
    first = identify_fn(identity.get('pid'), env, role)
    if not same_identity(identity, first):
        return dict(status='PROCESS_IDENTITY_CHANGED_NO_SIGNAL', signals=[])
    second = identify_fn(first['pid'], env, role)
    if not same_identity(first, second):
        return dict(status='PROCESS_IDENTITY_CHANGED_NO_SIGNAL', signals=[])
    try:
        send(first['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return dict(status='OWNED_HELPER_ALREADY_EXITED', signals=[], role=role)
    until = clock() + max(0., grace)
    while clock() < until and same_identity(first, identify_fn(first['pid'], env, role)):
        sleep(min(.05, max(0., until-clock())))
    remaining = same_identity(first, identify_fn(first['pid'], env, role))
    return dict(status='OWNED_HELPER_GRACEFUL_STOP_PENDING' if remaining else 'OWNED_HELPER_ENDED',
                role=role, pid=first['pid'], start_ticks=first['start_ticks'], signals=['SIGTERM'],
                scientific_workers_signalled=0, escalation_permitted=False)


def lock_live(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False
