"""Attach the R1 action-local repair to owned operational startup files only."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

MARKER = 'PAPER3_R1_COMPLETE_RUNTIME_REPAIR_V1'
HELPERS = ('verified_context_cache.py', 'newdiagnostics.py')
RUNTIME = 'runs/training_campaign_parallel_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        with Path(path).open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if size > 16 * 1024 * 1024:
                return {}
            value = json.loads(handle.read(size + 1))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic(path, data):
    """Independent stdlib installer write; no scientific modules are imported."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    fd, temp = tempfile.mkstemp(prefix='.r1-repair-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def option(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def discover_packages(root, home):
    runtime = Path(root) / RUNTIME
    launch, state = read(runtime/'launch.json'), read(runtime/'scheduler-state.json')
    values = [launch.get('package')]
    command = launch.get('worker_command')
    if isinstance(command, list):
        values.append(option(command, '--package'))
    for slot in state.get('slots', {}).values():
        for attempt in slot.get('attempts', []):
            command = attempt.get('command')
            if isinstance(command, list):
                values.append(option(command, '--package'))
    found = set()
    def add(value):
        if not isinstance(value, str) or not value:
            return
        path = Path(value).expanduser()
        if (path/'round1/action_startup.py').is_file():
            path = path/'round1'
        if all((path/name).is_file() for name in ('action_startup.py','kernel_bridge.py','worker.py')):
            found.add(path.absolute())
    for value in values:
        add(value)
    if not found:
        for pattern in ('paper3-r1-repairs/*/active-round1/*/action_startup.py',
                        'paper3-parallel-migrations/*/round1/action_startup.py'):
            for source in Path(home).glob(pattern):
                add(str(source.parent))
    return sorted(found)


def strip_own_block(source):
    begin, end = '\n# '+MARKER+'\n', '# END_'+MARKER+'\n'
    if begin not in source and end not in source:
        return source
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('INCOMPLETE_OR_DUPLICATE_R1_COMPLETE_REPAIR_BLOCK')
    at = source.index(begin); stop = source.index(end, at) + len(end)
    return source[:at] + source[stop:]


def checked_helpers(package):
    package = Path(package).expanduser().resolve()
    manifest = read(package/'MANIFEST.json')
    rows = manifest.get('files')
    if not isinstance(rows, dict):
        raise ValueError('INSTALLED_REPAIR_MANIFEST_REQUIRED')
    helpers = {}
    for name in HELPERS:
        row = rows.get(name, {})
        path = package/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('REPAIR_HELPER_NOT_REGULAR')
        data = path.read_bytes()
        if len(data) != row.get('bytes') or sha(data) != row.get('sha256'):
            raise ValueError('REPAIR_HELPER_SHA256_DIFFERS')
        compile(data, str(path), 'exec')
        helpers[name] = (path, sha(data))
    return helpers


def patch_source(original, helpers):
    source = strip_own_block(original.decode('utf-8'))
    tree = ast.parse(source)
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'active'
               for node in tree.body):
        raise ValueError('ORIGINAL_R1_STARTUP_ACTIVE_CONTEXT_REQUIRED')
    specs = {name: (str(path), digest) for name, (path, digest) in helpers.items()}
    block = '''
def _paper3_r1_complete_helper(_path, _expected):
    import hashlib as _h, importlib.util as _iu, sys as _s
    from pathlib import Path as _P
    _key = "_paper3_r1_complete_" + _h.sha256((_path + _expected).encode()).hexdigest()
    _module = _s.modules.get(_key)
    if _module is None:
        if _h.sha256(_P(_path).read_bytes()).hexdigest() != _expected:
            raise ValueError("Installed R1 complete-repair helper SHA256 differs")
        _spec = _iu.spec_from_file_location(_key, _path)
        if _spec is None or _spec.loader is None:
            raise RuntimeError("R1 complete-repair helper loader unavailable")
        _module = _iu.module_from_spec(_spec)
        _s.modules[_key] = _module
        try:
            _spec.loader.exec_module(_module)
        except BaseException:
            _s.modules.pop(_key, None)
            raise
    return _module
_paper3_r1_complete_specs = __SPECS__
def _paper3_r1_complete_note(root, action, action_file):
    import hashlib as _h, json as _j, os as _o, sys as _s, tempfile as _t, time as _time
    from pathlib import Path as _P
    _temporary = None
    try:
        if action_file is None:
            _at = _s.argv.index('--action-file')
            action_file = _s.argv[_at + 1]
        _candidate = _P(action_file).resolve()
        with _candidate.open('rb') as _handle:
            _record = _j.loads(_handle.read(65537))
        _claim = _record['claim']
        _token = _claim['token']
        _key = action['key']
        if (not isinstance(_token, str) or not _token.isalnum()
                or _record['action']['key'] != _key):
            return
        _expected = (_P(root).resolve() / 'runs/training_campaign_parallel_v1/actions'
            / _h.sha256(str(_key).encode()).hexdigest() / 'attempts' / _token / 'action.json')
        if _candidate != _expected.resolve():
            return
        _value = dict(repair_id='paper3-r1-complete-runtime-v1', pid=_o.getpid(),
            action_key=_key, claim_token=_token,
            helpers_sha256={_n:_v[1] for _n,_v in _paper3_r1_complete_specs.items()},
            started_utc=_time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime()))
        _descriptor, _temporary = _t.mkstemp(prefix='.repair-active-', dir=_candidate.parent)
        with _o.fdopen(_descriptor, 'w') as _handle:
            _j.dump(_value, _handle, sort_keys=True)
            _handle.write('\\n'); _handle.flush(); _o.fsync(_handle.fileno())
        _o.replace(_temporary, _candidate.with_name('repair-active.json'))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        # Failure to publish an operational activation marker cannot interrupt
        # the existing scientific action. Its next parent heartbeat is retained.
        pass
    finally:
        if _temporary is not None:
            try:_P(_temporary).unlink(missing_ok=True)
            except OSError:pass
_paper3_r1_before_complete_repair_active = active
@contextmanager
def active(root, action, *, modules=None, torch_module=None, emit=None,
           action_file=None, traceback_interval=300):
    _cache = _paper3_r1_complete_helper(*_paper3_r1_complete_specs['verified_context_cache.py'])
    _paper3_r1_complete_note(root, action, action_file)
    with _cache.active(root, action):
        with _paper3_r1_before_complete_repair_active(root, action, modules=modules,
                torch_module=torch_module, emit=emit, action_file=action_file,
                traceback_interval=traceback_interval):
            yield
_paper3_r1_complete_helper(*_paper3_r1_complete_specs['newdiagnostics.py']).install(__import__('sys').modules[__name__])
'''.replace('__SPECS__', repr(specs))
    fixed = (source + '\n# '+MARKER+'\n' + block + '# END_'+MARKER+'\n').encode('utf-8')
    compile(fixed, 'action_startup.py', 'exec')
    return fixed


def inspect(root, home):
    root, home = Path(root).expanduser().resolve(), Path(home).expanduser().resolve()
    found = discover_packages(root, home)
    return dict(status='READ_ONLY_STARTUP_HOOK_INSPECTION', packages=[str(p) for p in found],
        startup_files=[dict(path=str(p/'action_startup.py'),
            sha256=sha((p/'action_startup.py').read_bytes()),
            installed=('# '+MARKER) in (p/'action_startup.py').read_text()) for p in found],
        scientific_files_changed=0, jobs_submitted=0, signals_sent=0)


def apply(root, home, *, package):
    root, home = Path(root).expanduser().resolve(), Path(home).expanduser().resolve()
    package = Path(package).expanduser().resolve()
    helpers = checked_helpers(package)
    runtime = root/RUNTIME
    found = discover_packages(root, home)
    if not found:
        raise ValueError('EXISTING_R1_OPERATIONAL_PACKAGE_NOT_FOUND')
    base = home/'paper3-r1-full-repair'; base.mkdir(parents=True, exist_ok=True)
    with (base/'apply.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepared, preserved = [], {}
        for directory in found:
            target = directory/'action_startup.py'
            info = target.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError('R1_STARTUP_NOT_OWNED_REGULAR_FILE')
            original = target.read_bytes()
            prepared.append((target, original, patch_source(original, helpers)))
            for name in ('worker.py','kernel_bridge.py','audited_runtime.py','scheduler.py','START.py'):
                p = directory/name
                if p.is_file():
                    preserved[p] = p.read_bytes()
        # A few immutable scientific adapters, not a traversal of source data.
        for relative in ('p3work/development_evaluation.py','p3work/classical_training.py',
                         'p3work/method_registry.py','p3work/meta_training.py'):
            p = root/relative
            if p.is_file():
                preserved[p] = p.read_bytes()
        changed = []
        try:
            for target, original, fixed in prepared:
                backup = base/'backups'/(sha(original)+'.py')
                if not backup.is_file():
                    atomic(backup, original)
                if original != fixed:
                    if target.read_bytes() != original:
                        raise RuntimeError('R1_STARTUP_CHANGED_DURING_REPAIR')
                    atomic(target, fixed); changed.append((target, original))
            if any(p.read_bytes() != data for p, data in preserved.items()):
                raise RuntimeError('PRESERVED_R1_SOURCE_CHANGED_CONCURRENTLY')
        except BaseException:
            for target, original in reversed(changed):
                atomic(target, original)
            raise
        receipt = dict(schema_version='paper3-r1-complete-startup-hook-v1',
            package=str(package), helpers={n:dict(path=str(p),sha256=h) for n,(p,h) in helpers.items()},
            operational_files=[dict(path=str(p),before_sha256=sha(before),after_sha256=sha(after),
                                    changed=before!=after) for p,before,after in prepared],
            preserved_sources=[dict(path=str(p),sha256=sha(data)) for p,data in sorted(preserved.items())],
            activation='FUTURE_R1_ACTION_CHILDREN_EXISTING_ALLOCATION_PRESERVED',
            jobs_submitted=0, jobs_cancelled=0, signals_sent=0, scientific_files_changed=0,
            checkpoint_files_changed=0, plan_files_changed=0, controller_or_queue_modified=False)
        raw = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
        atomic(base/'HOOKS_APPLIED.json', raw)
        atomic(runtime/'full-repair-hooks.json', raw)
        return receipt
