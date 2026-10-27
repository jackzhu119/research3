"""Read-only allocation probe and exact-child graceful R1 checkpoint refresh.

This helper is run by srun inside an already owned allocation. It never submits
or cancels an allocation and never sends a signal to a worker parent or process
group. Probe restores the existing school environment and calls only the native
hardware/runtime observer; no model fit or source-data validation is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import sysconfig

REPAIR_ID = 'paper3-r1-complete-runtime-v1'
RUNTIME_RELATIVE = 'runs/training_campaign_parallel_v1'
_CLAIM_FIELDS = ('job_id', 'slot', 'pid', 'child_pid', 'action_key', 'claim_token')
_PROC_FIELDS = ('uid', 'ppid', 'start_ticks', 'argv', 'slurm_job_id')


def read(path, default=None, *, max_bytes=16*1024*1024):
    try:
        with Path(path).open('rb') as handle:
            size = os.fstat(handle.fileno()).st_size
            if not 0 <= size <= max_bytes:
                return default
            data = handle.read(size+1)
        if len(data) != size:
            return default
        return json.loads(data)
    except (OSError, ValueError):
        return default


def option(argv, name):
    if argv.count(name) != 1:
        return None
    at = argv.index(name)
    return argv[at+1] if at+1 < len(argv) else None


def process(pid, proc_root=Path('/proc')):
    pid = int(pid)
    if pid <= 1:
        raise ValueError('invalid_process_pid')
    folder = Path(proc_root)/str(pid)
    fields = (folder/'stat').read_text().rsplit(')', 1)[1].split()
    argv = [x.decode('utf-8') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
    # Read selected fields only. No complete environment is retained or printed.
    selected = {}
    for item in (folder/'environ').read_bytes().split(b'\0'):
        name, sep, value = item.partition(b'=')
        if sep and name in (b'SLURM_JOB_ID', b'CUDA_VISIBLE_DEVICES', b'CUBLAS_WORKSPACE_CONFIG'):
            selected[name.decode()] = value.decode('utf-8')
    return dict(uid=folder.stat().st_uid, ppid=int(fields[1]), start_ticks=int(fields[19]),
        state=fields[0], argv=argv, slurm_job_id=selected.get('SLURM_JOB_ID'),
        visible_devices=selected.get('CUDA_VISIBLE_DEVICES'),
        cublas_workspace_config=selected.get('CUBLAS_WORKSPACE_CONFIG'))


def _script(argv):
    if not argv or not Path(argv[0]).is_absolute() or Path(argv[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError('worker_python_identity_differs')
    at = 1
    while at < len(argv) and argv[at] in ('-I', '-S', '-B', '-u', '-E', '-s', '-O', '-OO', '-q'):
        at += 1
    if at >= len(argv) or not Path(argv[at]).is_absolute():
        raise ValueError('worker_script_position_invalid')
    return Path(argv[at]).resolve()


def _check_worker(observed, root, runtime, slot, job_id, *, child=False, script=None):
    args = observed['argv']; actual = _script(args)
    package = option(args, '--package')
    if (observed.get('uid') != os.getuid() or observed.get('state') in ('Z', 'X', 'x')
            or observed.get('slurm_job_id') != job_id or not package
            or actual != Path(package).expanduser().resolve()/'worker.py'
            or (script is not None and actual != script)
            or option(args, '--root') != str(root)
            or option(args, '--runtime-dir') != str(runtime)
            or option(args, '--slot') != str(slot)
            or args.count('--child') != int(child)):
        raise ValueError('worker_process_identity_mismatch')
    return actual


def _context(root, runtime, job_id, slot, *, proc_root=Path('/proc'),
             allocation_job=None, process_fn=process, include_child=True):
    runtime = Path(runtime).expanduser().resolve()
    root = Path(root).expanduser().resolve() if root is not None else runtime.parents[1]
    job_id = str(job_id)
    if slot not in (0, 1, 2) or not re.fullmatch(r'[0-9]{1,20}', job_id):
        raise ValueError('invalid_job_or_slot')
    allocation = allocation_job if allocation_job is not None else os.environ.get('SLURM_JOB_ID', '')
    if str(allocation) != job_id:
        raise ValueError('allocation_job_mismatch')
    if runtime != root/RUNTIME_RELATIVE:
        raise ValueError('root_runtime_binding_differs')
    status_path = runtime/'worker-status'/('slot-%d.json'%slot)
    row = read(status_path, {})
    if not isinstance(row, dict) or str(row.get('job_id')) != job_id or row.get('slot') != slot:
        raise ValueError('worker_slot_or_job_changed')
    parent = int(row['pid']); parent_proc = process_fn(parent, proc_root)
    script = _check_worker(parent_proc, root, runtime, slot, job_id)
    ctx = dict(root=root, runtime=runtime, job_id=job_id, slot=slot, row=row,
               status_path=status_path, parent=parent, parent_proc=parent_proc, script=script)
    if not include_child or row.get('child_pid') is None:
        return ctx
    child = int(row['child_pid'])
    if child <= 1 or child == parent:
        raise ValueError('invalid_child_pid')
    key, token = row['action_key'], row['claim_token']
    if not isinstance(key, str) or not key or not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{32}', token):
        raise ValueError('invalid_current_claim')
    action_file = runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'action.json'
    if action_file.is_symlink() or action_file.stat().st_uid != os.getuid():
        raise ValueError('action_file_not_owned_regular')
    document = read(action_file, {}); claim = document['claim']
    if (document['action']['key'] != key or claim.get('token') != token
            or str(claim.get('job_id')) != job_id or claim.get('slot') != slot
            or claim.get('pid') != parent):
        raise ValueError('action_claim_changed')
    claimed = claim.get('claimed_utc')
    if type(claimed) not in (int, float) or not math.isfinite(claimed) or claimed < 0:
        raise ValueError('invalid_claim_time')
    child_proc = process_fn(child, proc_root)
    _check_worker(child_proc, root, runtime, slot, job_id, child=True, script=script)
    if child_proc.get('ppid') != parent or option(child_proc['argv'], '--action-file') != str(action_file):
        raise ValueError('child_parent_or_action_changed')
    ctx.update(child=child, child_proc=child_proc, action_key=key, token=token,
               action_file=action_file, document=document)
    return ctx


def _recheck(ctx, *, proc_root, process_fn, parent_only=False):
    now = read(ctx['status_path'], {})
    row_fields = ('job_id', 'slot', 'pid') if parent_only else _CLAIM_FIELDS
    if any(now.get(k) != ctx['row'].get(k) for k in row_fields):
        raise ValueError('worker_or_claim_changed_before_action')
    for pid_key, proc_key in (('parent', 'parent_proc'), ('child', 'child_proc')):
        if pid_key not in ctx:
            continue
        current = process_fn(ctx[pid_key], proc_root)
        original = ctx[proc_key]
        identity = _PROC_FIELDS + ('visible_devices', 'cublas_workspace_config') if parent_only else _PROC_FIELDS
        if current.get('state') in ('Z', 'X', 'x') or any(current.get(k) != original.get(k) for k in identity):
            raise ValueError('pid_identity_changed_before_action')
    if not parent_only and 'action_file' in ctx:
        again = read(ctx['action_file'], {})
        if again != ctx['document']:
            raise ValueError('action_record_changed_before_action')


def _repair_active(ctx):
    if 'action_file' not in ctx:
        return False
    path = ctx['action_file'].with_name('repair-active.json')
    value = read(path, {})
    helpers = value.get('helpers_sha256', {}) if isinstance(value, dict) else {}
    return (isinstance(value, dict) and value.get('repair_id') == REPAIR_ID
        and value.get('action_key') == ctx['action_key'] and value.get('claim_token') == ctx['token']
        and value.get('pid') == ctx['child'] and set(helpers) == {'verified_context_cache.py', 'newdiagnostics.py'}
        and all(isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) for v in helpers.values()))


def refresh(runtime, job_id, slot, *, root=None, proc_root=Path('/proc'), allocation_job=None,
            kill=os.kill, process_fn=process):
    """Request original SIGINT checkpoint unwind for the exact old action child."""
    try:
        ctx = _context(root, runtime, job_id, slot, proc_root=proc_root,
                       allocation_job=allocation_job, process_fn=process_fn)
        if 'child' not in ctx:
            return dict(status='IDLE_PARENT_REUSED_NO_SIGNAL', job_id=str(job_id), slot=slot)
        if _repair_active(ctx):
            return dict(status='CURRENT_REPAIRED_CHILD_KEPT', job_id=str(job_id), slot=slot,
                        child_pid=ctx['child'], signal_sent=False)
        _recheck(ctx, proc_root=proc_root, process_fn=process_fn)
        # A child may enter its newly repaired context during the identity read.
        if _repair_active(ctx):
            return dict(status='CURRENT_REPAIRED_CHILD_KEPT', job_id=str(job_id), slot=slot,
                        child_pid=ctx['child'], signal_sent=False)
        kill(ctx['child'], signal.SIGINT)
        return dict(status='CHECKPOINT_REFRESH_REQUESTED', job_id=str(job_id), slot=slot,
            child_pid=ctx['child'], parent_pid=ctx['parent'], action_key=ctx['action_key'],
            signal='SIGINT', signal_sent=True, parent_signalled=False, allocation_cancelled=False,
            scientific_files_modified=False, committed_results_retained=True,
            unfinished_fit_may_need_resume=True)
    except (OSError, ValueError, KeyError, TypeError, IndexError, UnicodeError) as error:
        return dict(status='WAIT_NEXT_CHILD', job_id=str(job_id), slot=slot, signal_sent=False,
                    reason=str(error) if isinstance(error, ValueError) else type(error).__name__,
                    parent_signalled=False, allocation_cancelled=False)


def _native_runtime(root, parent):
    version = 'python%d.%d'%sys.version_info[:2]
    executable = Path(sys.executable).absolute()
    paths = [executable.parent.parent/lib/version/'site-packages' for lib in ('lib', 'lib64')]
    paths += [Path(sysconfig.get_paths()[k]) for k in ('purelib', 'platlib')]
    for path in paths:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.append(str(path))
    sys.path.insert(0, str(root))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'  # Exact final kernel_bridge.activate_runtime policy.
    if parent.get('visible_devices') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = parent['visible_devices']
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = parent.get('cublas_workspace_config') or ':4096:8'
    module = importlib.import_module('p3work.development_evaluation')
    if Path(module.__file__).resolve() != root/'p3work/development_evaluation.py':
        raise ValueError('native_runtime_module_outside_original_root')
    module.torch.set_num_threads(1)
    return module._production_runtime()


def probe(root, runtime, job_id, slot, *, proc_root=Path('/proc'), allocation_job=None,
          process_fn=process, runtime_observer=_native_runtime):
    """Observe the native full runtime on the already allocated GPU, without fit."""
    try:
        ctx = _context(root, runtime, job_id, slot, proc_root=proc_root,
                       allocation_job=allocation_job, process_fn=process_fn, include_child=False)
        _recheck(ctx, proc_root=proc_root, process_fn=process_fn, parent_only=True)
        profile = runtime_observer(ctx['root'], ctx['parent_proc'])
        if not isinstance(profile, dict):
            raise ValueError('native_runtime_profile_not_a_mapping')
        _recheck(ctx, proc_root=proc_root, process_fn=process_fn, parent_only=True)
        return dict(status='ACTUAL_RUNTIME_OBSERVED_NO_FIT', job_id=str(job_id), slot=slot,
            runtime_profile=profile, source='p3work.development_evaluation._production_runtime',
            scientific_fit_executed=False, allocation_submitted=False, files_modified=False)
    except (OSError, ValueError, KeyError, TypeError, IndexError, UnicodeError, ImportError, RuntimeError) as error:
        return dict(status='PROBE_UNAVAILABLE', job_id=str(job_id), slot=slot,
            error_type=type(error).__name__, reason=str(error) if isinstance(error, ValueError) else None,
            scientific_fit_executed=False, files_modified=False)


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode', choices=('probe', 'refresh')); p.add_argument('--root', type=Path)
    p.add_argument('--runtime', type=Path, required=True); p.add_argument('--job-id', required=True)
    p.add_argument('--slot', type=int, choices=(0, 1, 2), required=True); args = p.parse_args(argv)
    root = args.root or args.runtime.resolve().parents[1]
    result = (probe(root, args.runtime, args.job_id, args.slot) if args.mode == 'probe'
              else refresh(args.runtime, args.job_id, args.slot, root=root))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result['status'] in ('ACTUAL_RUNTIME_OBSERVED_NO_FIT', 'IDLE_PARENT_REUSED_NO_SIGNAL',
        'CURRENT_REPAIRED_CHILD_KEPT', 'CHECKPOINT_REFRESH_REQUESTED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
