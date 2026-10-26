#!/usr/bin/env python3
"""Install narrow R1 clock/driver resume hooks; preserve native training files."""
import argparse
import ast
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

PAYLOAD_B85 = '__PAYLOAD_B85__'
PAYLOAD_SHA256 = '__PAYLOAD_SHA256__'
SCHEMA = 'paper3-r1-resume-delivery-payload-v1'
MARKERS = {'action_startup.py': 'PAPER3_R1_RESUME_ACTION_V3',
           'r1_spawn_bootstrap.py': 'PAPER3_R1_RESUME_OWNED_STAGE_V3',
           'audited_runtime.py': 'PAPER3_R1_RESUME_RUNTIME_INSTALL_V3'}
DIGEST_MARKERS = {'action_startup.py': 'PAPER3_R1_PERSISTENT_DIGEST_CACHE_V1',
                  'r1_spawn_bootstrap.py': 'PAPER3_R1_OWNED_DIGEST_CACHE_V1'}


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          **fields), ensure_ascii=False), flush=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    temporary = path.with_name('.' + path.name + '.r1-resume-' + str(os.getpid()))
    try:
        with temporary.open('wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def option(command, flag):
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def packages(runtime, home):
    """Follow R1 launch and scheduler declarations, not scientific data trees."""
    launch, state = read(runtime / 'launch.json'), read(runtime / 'scheduler-state.json')
    values = [launch.get('package')]
    if isinstance(launch.get('worker_command'), list):
        values.append(option(launch['worker_command'], '--package'))
    for slot in state.get('slots', {}).values():
        for attempt in slot.get('attempts', []):
            if isinstance(attempt.get('command'), list):
                values.append(option(attempt['command'], '--package'))
    found = set()

    def add(value):
        if not isinstance(value, str):
            return
        path = Path(value).expanduser().resolve()
        if (path / 'round1/action_startup.py').is_file():
            path = path / 'round1'
        if all((path / name).is_file() for name in ('action_startup.py', 'kernel_bridge.py', 'worker.py')):
            found.add(path)

    for value in values:
        add(value)
    if not found:
        for pattern in ('paper3-r1-repairs/*/active-round1/*/action_startup.py',
                        'paper3-parallel-migrations/*/round1/action_startup.py'):
            for path in Path(home).glob(pattern):
                add(str(path.parent))
    return sorted(found)


def bootstrap_files(runtime, found):
    paths = set()
    for package in found:
        path = package / 'r1_spawn_bootstrap.py'
        if path.is_file():
            paths.add(path.resolve())
    launch = read(runtime / 'launch.json')
    if isinstance(launch.get('direct_package'), str):
        path = Path(launch['direct_package']).expanduser() / 'r1_spawn_bootstrap.py'
        if path.is_file():
            paths.add(path.resolve())
    return sorted(paths)


def strip_block(text, marker):
    begin, end = '\n# ' + marker + '\n', '# END_' + marker + '\n'
    if begin not in text and end not in text:
        return text, None
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError('Incomplete or duplicate R1 operational hook: ' + marker)
    start = text.index(begin)
    finish = text.index(end, start) + len(end)
    return text[:start] + text[finish:], text[start:finish]


def load_payload():
    data = zlib.decompress(base64.b85decode(PAYLOAD_B85))
    if sha(data) != PAYLOAD_SHA256:
        raise ValueError('Embedded R1 resume payload SHA256 differs')
    value = json.loads(data)
    if value.get('schema_version') != SCHEMA:
        raise ValueError('Unknown R1 resume payload schema')
    helpers = value.get('helpers', {})
    if not {'r1_clock_recovery.py', 'r1_driver_compat.py'} <= set(helpers):
        raise ValueError('Both narrow R1 resume helpers are required')
    for name, source in helpers.items():
        if Path(name).name != name or not name.endswith('.py') or not isinstance(source, str):
            raise ValueError('Invalid embedded helper name or source')
        compile(source, name, 'exec')
    for key in ('startup_install', 'bootstrap_install', 'runtime_install'):
        names = value.get(key)
        if not isinstance(names, list) or any(name not in helpers for name in names):
            raise ValueError('Invalid helper installation scope: ' + key)
    return value


def validate_source(original, name, payload):
    text = original.decode('utf-8')
    preserved, _ = strip_block(text, MARKERS[name])
    canonical = preserved
    if name in DIGEST_MARKERS:
        canonical, digest_block = strip_block(canonical, DIGEST_MARKERS[name])
        if digest_block:
            # The existing cache remains byte-for-byte intact; verify that its
            # run_path target is the previously published helper, not an unknown overlay.
            tree = ast.parse(digest_block)
            helper_paths = [node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'run_path' and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)]
            expected = payload.get('existing_digest_helper_sha256')
            if (len(helper_paths) != 1 or Path(helper_paths[0]).name != 'r1_digest_cache.py'
                    or not expected or sha(Path(helper_paths[0]).read_bytes()) != expected):
                raise ValueError('Existing R1 digest helper lineage differs')
    expected = payload['lineage_sha256'].get(name)
    if sha(canonical.encode()) != expected:
        raise ValueError('Unrecognized R1 operational source: ' + name)
    return preserved


def loader_source(helper_paths, names):
    # The actual helper source digest is bound into each generated import key.
    specifications = [(str(helper_paths[name][0]), helper_paths[name][1]) for name in names]
    return (
        'def _paper3_r1_resume_load_helper(_path, _expected):\n'
        '    import importlib.util as _util, sys as _sys, hashlib as _hashlib\n'
        '    from pathlib import Path as _Path\n'
        '    _key = "_paper3_r1_resume_" + _hashlib.sha256((_path + _expected).encode()).hexdigest()\n'
        '    _module = _sys.modules.get(_key)\n'
        '    if _module is None:\n'
        '        if _hashlib.sha256(_Path(_path).read_bytes()).hexdigest() != _expected:\n'
        '            raise ValueError("Installed R1 resume helper SHA256 differs")\n'
        '        _spec = _util.spec_from_file_location(_key, _path)\n'
        '        if _spec is None or _spec.loader is None:\n'
        '            raise RuntimeError("R1 resume helper loader is unavailable")\n'
        '        _module = _util.module_from_spec(_spec)\n'
        '        _sys.modules[_key] = _module\n'
        '        try:\n'
        '            _spec.loader.exec_module(_module)\n'
        '        except BaseException:\n'
        '            _sys.modules.pop(_key, None)\n'
        '            raise\n'
        '    return _module\n'
        'def _paper3_r1_resume_install_helpers(root):\n'
        '    for _path, _expected in ' + repr(specifications) + ':\n'
        '        _paper3_r1_resume_load_helper(_path, _expected).install(root)\n')


def patch_source(original, name, helper_paths, payload):
    text = validate_source(original, name, payload)
    marker = MARKERS[name]
    begin, end = '\n# ' + marker + '\n', '# END_' + marker + '\n'
    if name == 'action_startup.py':
        block = (loader_source(helper_paths, payload['startup_install']) +
            '_paper3_r1_before_resume_active = active\n'
            '@contextmanager\n'
            'def active(root, action, *, modules=None, torch_module=None, emit=None,\n'
            '           action_file=None, traceback_interval=300):\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    with _paper3_r1_before_resume_active(root, action, modules=modules,\n'
            '            torch_module=torch_module, emit=emit, action_file=action_file,\n'
            '            traceback_interval=traceback_interval):\n'
            '        yield\n')
    elif name == 'r1_spawn_bootstrap.py':
        block = (loader_source(helper_paths, payload['bootstrap_install']) +
            '_paper3_r1_before_resume_owned_stage = bootstrap_owned_stage\n'
            'def bootstrap_owned_stage(root):\n'
            '    result = _paper3_r1_before_resume_owned_stage(root)\n'
            '    if isinstance(result, dict) and result.get("status") == "NOT_APPLICABLE_NO_PARALLEL_LAUNCH":\n'
            '        return result\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    return result\n')
    elif name == 'audited_runtime.py':
        block = (loader_source(helper_paths, payload['runtime_install']) +
            '_paper3_r1_before_resume_runtime_install = install\n'
            'def install(root, *args, **kwargs):\n'
            '    result = _paper3_r1_before_resume_runtime_install(root, *args, **kwargs)\n'
            '    _paper3_r1_resume_install_helpers(root)\n'
            '    return result\n')
    else:
        raise ValueError('Unsupported R1 operational hook')
    fixed = (text + begin + block + end).encode()
    compile(fixed, name, 'exec')
    return fixed


def preserved_sources(root, found, bootstraps):
    paths = set()
    for package in found:
        paths.update(p for p in package.glob('*.py') if p.name not in MARKERS)
    for bootstrap in bootstraps:
        paths.update(p for p in bootstrap.parent.glob('*.py') if p.name not in MARKERS)
    for relative in ('p3work/meta_training.py', 'p3work/source_training.py',
                     'p3work/contracts.py', 'tools/r2_backup.py'):
        path = root / relative
        if path.is_file():
            paths.add(path)
    return {path: path.read_bytes() for path in paths if path.is_file()}


def refresh_current(runtime, node_helper, *, python=None, runner=None):
    """Reuse the audited allocation-local pretraining-only refresh protocol."""
    import concurrent.futures
    import subprocess
    if runner is None:
        runner = subprocess.run
    python = python or sys.executable
    tasks = []
    for slot in range(3):
        row = read(runtime / 'worker-status' / ('slot-%d.json' % slot))
        job = str(row.get('job_id', ''))
        if job.isdigit() and row.get('child_pid'):
            tasks.append((slot, job))
    emit('R1_PRETRAIN_REFRESH_CHECK_STARTED', allocations=len(tasks),
         healthy_training_children_preserved=True, new_slurm_allocations=0)

    def one(item):
        slot, job = item
        command = ['srun', '--jobid=' + job, '--overlap', '-N1', '-n1', '-c1', '--mem=256M',
                   '--gres=none', '--time=00:02:00', python, '-I', '-S', '-B', str(node_helper),
                   '--runtime', str(runtime), '--job-id', job, '--slot', str(slot)]
        try:
            response = runner(command, capture_output=True, text=True, timeout=40)
            value = None
            for line in reversed(response.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict):
                        value = candidate
                        break
                except ValueError:
                    pass
            if response.returncode != 0 or value is None:
                return {'job_id': job, 'slot': slot, 'status': 'WAIT_NEXT_CHILD',
                        'reason': 'allocation_step_not_available', 'returncode': response.returncode}
            return dict(value, job_id=job, slot=slot)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {'job_id': job, 'slot': slot, 'status': 'WAIT_NEXT_CHILD', 'reason': type(error).__name__}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(one, tasks):
            results.append(result)
            emit('R1_PRETRAIN_REFRESH_RESULT', **result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/public/home/zhuziyuan/projects/paper3'))
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--apply', action='store_true', default=True)
    parser.add_argument('--refresh-pretraining', action='store_true')
    args = parser.parse_args(argv)
    root, home = args.root.expanduser().resolve(), args.home.expanduser().resolve()
    runtime = root / 'runs/training_campaign_parallel_v1'
    payload = load_payload()
    if args.refresh_pretraining and 'node_refresh.py' not in payload['helpers']:
        raise ValueError('Embedded pretraining refresh helper is unavailable')
    emit('R1_RESUME_REPAIR_STARTED', imports_torch=False, full_data_rescan=False,
         refresh_pretraining_requested=args.refresh_pretraining)
    folder = home / 'paper3-runtime-fixes/r1-resume-v3'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'apply.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content_folder = folder / PAYLOAD_SHA256[:20]
        helper_paths = {name: (content_folder / name, sha(source.encode()))
                        for name, source in payload['helpers'].items()}
        found = packages(runtime, home)
        if not found:
            raise ValueError('Existing R1 operational packages were not found; no files were switched')
        bootstraps = bootstrap_files(runtime, found)
        prepared = []
        for package in found:
            for name in ('action_startup.py', 'audited_runtime.py'):
                path = package / name
                if path.is_symlink() or path.stat().st_uid != os.getuid():
                    raise ValueError('R1 operational target is not an owned regular file')
                original = path.read_bytes()
                prepared.append((path, original, patch_source(original, name, helper_paths, payload)))
        for path in bootstraps:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                raise ValueError('R1 owned-stage bootstrap target is not an owned regular file')
            original = path.read_bytes()
            prepared.append((path, original, patch_source(original, 'r1_spawn_bootstrap.py', helper_paths, payload)))
        # Do not import helpers or native training modules in this installer.
        science = preserved_sources(root, found, bootstraps)
        metadata = {p: p.read_bytes() for p in (runtime / 'launch.json', runtime / 'scheduler-state.json') if p.is_file()}
        for name, source in payload['helpers'].items():
            path = helper_paths[name][0]
            data = source.encode()
            if not path.is_file() or path.read_bytes() != data:
                atomic(path, data)
        changes = []
        for path, original, fixed in prepared:
            backup = folder / 'backups' / (sha(original) + '.py')
            if not backup.is_file():
                atomic(backup, original)
            if original != fixed:
                atomic(path, fixed)
            row = {'path': str(path), 'before_sha256': sha(original), 'after_sha256': sha(fixed),
                   'changed': original != fixed}
            changes.append(row)
            emit('R1_RESUME_OPERATIONAL_HOOK_INSTALLED', **row)
        for path, data in science.items():
            if path.read_bytes() != data:
                raise RuntimeError('Preserved R1 source changed concurrently: ' + str(path))
        receipt = {'schema_version': 'paper3-r1-clock-driver-resume-repair-v3',
                   'applied_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'payload_sha256': PAYLOAD_SHA256,
                   'helpers': {name: {'path': str(path), 'sha256': digest}
                               for name, (path, digest) in helper_paths.items()},
                   'operational_files': changes, 'owned_stage_bootstrap_files': len(bootstraps),
                   'preserved_native_sources': [{'path': str(path), 'sha256': sha(data)} for path, data in sorted(science.items())],
                   'original_metadata_sha256': {str(path): sha(data) for path, data in metadata.items()},
                   'scientific_files_changed': 0, 'checkpoint_files_changed': 0, 'plan_files_changed': 0,
                   'jobs_submitted': 0, 'jobs_cancelled': 0, 'signals_sent': 0, 'other_rounds_changed': 0,
                   'activation': 'FUTURE_R1_ACTION_AND_OWNED_CHILDREN_CURRENT_TRAINING_PRESERVED',
                   'refresh_pretraining_requested': args.refresh_pretraining}
        write_json(folder / 'APPLIED.json', receipt)
        write_json(runtime / 'resume-repair.json', receipt)
        emit('R1_RESUME_REPAIR_APPLIED', payload_sha256=PAYLOAD_SHA256, packages=len(found),
             owned_stage_bootstrap_files=len(bootstraps), checkpoint_files_changed=0,
             jobs_cancelled=0, jobs_submitted=0, activation=receipt['activation'],
             record=str(folder / 'APPLIED.json'))
    if args.refresh_pretraining:
        receipt['refresh_results'] = refresh_current(runtime, helper_paths['node_refresh.py'][0])
        receipt['signals_sent'] = sum(row.get('status') == 'PRETRAIN_CHILD_REFRESH_REQUESTED'
                                      for row in receipt['refresh_results'])
        write_json(folder / 'APPLIED.json', receipt)
        write_json(runtime / 'resume-repair.json', receipt)
    emit('R1_RESUME_REPAIR_COMPLETE', jobs_cancelled=0, jobs_submitted=0,
         signals_sent=receipt['signals_sent'], record=str(folder / 'APPLIED.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
