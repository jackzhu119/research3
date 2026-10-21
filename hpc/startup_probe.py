#!/usr/bin/env python3
"""Measure isolated CPU interpreter/import startup inside a Slurm allocation.

This deployment probe does not fit, change a scientific budget, or adjust any
original test deadline. The fixed 180s limit is exclusively its own subprocess
diagnostic limit. Filesystem/page-cache state is observed, never manipulated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc.resources import ResourceError, allocation, write_json

TIMEOUT_SECONDS = 180
# These are the exact import paths used by the seven reported software fixtures.
MODULES = (
    ('p3work.classical_training', 20),
    ('tools.continue_local_preparation_parallel', 5),
    ('tools.prepare_source_cache', 20),
    ('tools.source_preparation_batch', 20),
    ('tools.source_preparation_parallel', 20),
    ('tools.source_preparation_small_files', 20),
    ('p3work.training_state', 30),
)
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
NUMERIC_LIBRARIES = ('numpy', 'scipy', 'sklearn', 'torch')
WORKER_PREFIX = 'PAPER3_IMPORT_PROBE_RESULT_V1 '
WORKER = r'''
import importlib, json, resource, sys, time
before = resource.getrusage(resource.RUSAGE_SELF)
started = time.perf_counter()
error = None
try:
    importlib.import_module(sys.argv[1])
except BaseException as exc:
    error = {'error_type':type(exc).__name__}
    if isinstance(exc, ModuleNotFoundError): error['missing_module'] = exc.name
elapsed = time.perf_counter() - started
after = resource.getrusage(resource.RUSAGE_SELF)
value = {'status':'IMPORT_COMPLETE' if error is None else 'IMPORT_FAILED',
    'module':sys.argv[1], 'import_wall_seconds':elapsed,
    'user_seconds':after.ru_utime-before.ru_utime,
    'system_seconds':after.ru_stime-before.ru_stime,
    'peak_rss_bytes':int(after.ru_maxrss)*1024,
    'numeric_imports':{name:name in sys.modules for name in ('numpy','scipy','sklearn','torch')},
    'numerical_fit_performed':False, 'CUDA_operation_performed':False}
if error is not None: value.update(error)
print('\nPAPER3_IMPORT_PROBE_RESULT_V1 ' + json.dumps(value, allow_nan=False), flush=True)
raise SystemExit(0 if error is None else 2)
'''


class ProbeError(ValueError):
    pass


def _capture(value):
    """Keep only byte counts/hashes of unexpected library output, never text."""
    raw = value if isinstance(value, bytes) else (value or '').encode()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _probe_module(root, module, original_fixture_timeout, *, runner=subprocess.run):
    started = time.perf_counter()
    base = {'module': module, 'deployment_probe_timeout_seconds': TIMEOUT_SECONDS,
            'related_original_software_fixture_timeout_seconds': original_fixture_timeout,
            'scientific_fit_budget_changed': False, 'numerical_fit_performed': False}
    try:
        child = runner([sys.executable, '-B', '-c', WORKER, module], cwd=root,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        return {**base, 'status': 'IMPORT_PROBE_TIMEOUT', 'process_wall_seconds': time.perf_counter()-started,
                'import_metrics_available': False, 'stdout': _capture(exc.stdout), 'stderr': _capture(exc.stderr)}
    elapsed = time.perf_counter()-started
    try:
        lines = child.stdout.splitlines(keepends=True)
        framed = [line for line in lines if line.startswith(WORKER_PREFIX)]
        if len(framed) != 1:
            raise ValueError('one_framed_import_observation_required')
        value = json.loads(framed[0][len(WORKER_PREFIX):])
        if (not isinstance(value, dict) or value.get('module') != module
                or value.get('status') not in {'IMPORT_COMPLETE', 'IMPORT_FAILED'}
                or value.get('numerical_fit_performed') is not False
                or value.get('CUDA_operation_performed') is not False
                or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0
                       for key in ('import_wall_seconds', 'user_seconds', 'system_seconds', 'peak_rss_bytes'))
                or not isinstance(value.get('numeric_imports'), dict)
                or set(value['numeric_imports']) != set(NUMERIC_LIBRARIES)
                or any(type(flag) is not bool for flag in value['numeric_imports'].values())
                or (value['status'] == 'IMPORT_COMPLETE') != (child.returncode == 0)):
            raise ValueError('invalid_import_observation')
        if value['status'] == 'IMPORT_FAILED':
            for field in ('error_type', 'missing_module'):
                if field in value and (not isinstance(value[field], str)
                        or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]{0,159}', value[field]) is None):
                    raise ValueError('invalid_import_failure_identifier')
    except (ValueError, TypeError, KeyError):
        return {**base, 'status': 'IMPORT_PROBE_INVALID_RESULT', 'returncode': child.returncode,
                'process_wall_seconds': elapsed, 'import_metrics_available': False,
                'stdout': _capture(child.stdout), 'stderr': _capture(child.stderr)}
    retained = {key: value[key] for key in ('status', 'module', 'import_wall_seconds',
        'user_seconds', 'system_seconds', 'peak_rss_bytes', 'numeric_imports',
        'numerical_fit_performed', 'CUDA_operation_performed', 'error_type', 'missing_module')
        if key in value}
    incidental = ''.join(line for line in lines if not line.startswith(WORKER_PREFIX))
    return {**base, **retained, 'returncode': child.returncode, 'process_wall_seconds': elapsed,
            'import_metrics_available': True, 'stderr': _capture(child.stderr),
            'incidental_stdout': _capture(incidental),
            'original_software_timeout_headroom_seconds': original_fixture_timeout-elapsed}


def run_probe(root, output, *, environ=None, hostname=None, runner=subprocess.run):
    # Guard before spawning or publishing any diagnostic, including on login1.
    allocated = allocation(environ=environ, hostname=hostname)
    root = Path(os.path.abspath(root))
    if not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents)):
        raise ProbeError('existing_real_project_directory_required')
    output = Path(output)
    output = output if output.is_absolute() else root / output
    if (not output.resolve().is_relative_to(root)
            or any(path.is_symlink() for path in (output, *output.parents))):
        raise ProbeError('startup_report_must_be_inside_real_project')
    observations = []
    result = {'schema_version': 'paper3-hpc-startup-import-probe-v1',
        'status': 'IMPORT_PROBES_RUNNING_NOT_COMPLETE',
        'utc': datetime.now(timezone.utc).isoformat(), 'allocation': allocated,
        'python': platform.python_version(), 'python_executable': sys.executable,
        'filesystem_cache_state': 'uncontrolled_may_have_been_warmed',
        'cpu_accounting_scope': 'child module import; RSS is child lifetime peak on Linux',
        'process_wall_scope': 'parent launch through subprocess exit',
        'scientific_thread_environment_changed': False,
        'thread_environment': {key: os.environ.get(key) for key in THREAD_VARIABLES},
        'modules': observations, 'scientific_fit_budget_changed': False,
        'original_test_deadlines_changed': False, 'numerical_fit_performed': False,
        'CUDA_operation_performed': False, 'network_actions': False,
        'diagnostic_guidance': [
            'Compare independent interpreter import/exit wall time with the original software fixture timeout.',
            'Import startup is only part of each fixture; no successful fit or recovery is inferred from this probe.',
            'If startup is slow, inspect shared-filesystem latency and rerun the named tests with --durations=10.',
            'Do not increase scientific fit caps or discard charged startup time to make a fixture pass.']}
    result.update(modules_completed=0, modules_total=len(MODULES), active_module=None)

    def publish(event, **fields):
        result['utc'] = datetime.now(timezone.utc).isoformat()
        # Persist before emitting progress. Failed evidence publication must not
        # be treated as successful import verification.
        write_json(output, result)
        print(json.dumps({'event': event, 'status': result['status'],
            'modules_completed': len(observations), 'modules_total': len(MODULES),
            'numerical_fit_performed': False, **fields}, allow_nan=False), flush=True)

    publish('STARTUP_PROBE_STARTED')
    for index, (module, original_timeout) in enumerate(MODULES, 1):
        result.update(active_module=module, active_module_index=index)
        publish('IMPORT_PROBE_STARTED', module=module,
                deployment_probe_timeout_seconds=TIMEOUT_SECONDS)
        try:
            row = _probe_module(root, module, original_timeout, runner=runner)
        except BaseException:
            result['status'] = 'IMPORT_PROBES_INTERRUPTED_NOT_COMPLETE'
            publish('IMPORT_PROBE_INTERRUPTED', module=module)
            raise
        observations.append(row)
        result.update(modules_completed=len(observations), active_module=None)
        detail = {key: row[key] for key in ('module', 'status', 'process_wall_seconds',
            'error_type', 'missing_module') if key in row}
        detail['import_status'] = detail.pop('status')
        publish('IMPORT_PROBE_FINISHED', **detail)
    result['status'] = ('IMPORT_PROBES_COMPLETE'
        if all(row['status'] == 'IMPORT_COMPLETE' for row in observations)
        else 'IMPORT_PROBES_INCOMPLETE')
    result['active_module_index'] = None
    publish('STARTUP_PROBE_FINISHED')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.root, args.output)
        print(json.dumps({'status': result['status'], 'modules': len(result['modules']),
                          'numerical_fit_performed': False, 'output': str(args.output)}))
        return 0 if result['status'] == 'IMPORT_PROBES_COMPLETE' else 2
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'STARTUP_PROBE_BLOCKED', 'error_type': type(exc).__name__,
                          'reason': str(exc) if isinstance(exc, (ProbeError, ResourceError)) else 'probe_runtime_failed',
                          'numerical_fit_performed': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
