"""Retire only the old login-side automatic single-GPU submitter.

Called only by the new coordinator's explicitly executed migration.  The
original Slurm allocation and scientific processes are not signalled here.
The old launcher journals remain untouched.  Both old launcher locks stay held
for this coordinator's lifetime, preventing a late old monitor from submitting.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import time


_LEGACY_HELD_LOCKS = {}


def _legacy_read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def _legacy_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-%d' % os.getpid())
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _legacy_option(arguments, option):
    values = []
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 == len(arguments):
                return None
            values.append(arguments[index + 1])
        elif argument.startswith(option + '='):
            values.append(argument.split('=', 1)[1])
    return values[0] if len(values) == 1 else None


def _legacy_read_process(pid, proc=Path('/proc')):
    """Read Linux start ticks and argv without trusting pid alone."""
    base = Path(proc)/str(pid)
    try:
        uid = base.stat().st_uid
        command = (base/'cmdline').read_bytes()
        raw = (base/'stat').read_text()
        end = raw.rfind(')')
        fields = raw[end + 2:].split()
        if end < 1 or len(fields) < 20:
            return None
        return {'pid': int(pid), 'uid': uid, 'start_time_ticks': int(fields[19]),
                'state': fields[0],
                'argv': [os.fsdecode(value) for value in command.split(b'\0') if value]}
    except (OSError, ValueError):
        return None


def _legacy_matches(row, root, script):
    if not row or row.get('uid') != os.getuid() or row.get('state') == 'Z':
        return False
    argv = row.get('argv', [])
    if not argv or '--monitor' not in argv or '--execute' not in argv:
        return False
    # A literal path inside a -c string or another argument is insufficient.
    if str(script) not in argv[1:]:
        return False
    index = argv.index(str(script))
    if index < 1 or any(not value.startswith('-') for value in argv[1:index]):
        return False
    if not Path(argv[0]).name.startswith('python'):
        return False
    selected_root = _legacy_option(argv[index + 1:], '--root')
    return selected_root is not None and Path(selected_root).absolute() == root


def _legacy_processes(root, script):
    result = []
    try:
        directories = list(Path('/proc').iterdir())
    except OSError:
        return None
    for directory in directories:
        if directory.name.isdigit():
            row = _legacy_read_process(int(directory.name))
            if _legacy_matches(row, root, script):
                result.append(row)
    return result


def _legacy_try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _legacy_evidence(state):
    ids, comments, intents = set(), set(), []
    current = state.get('current_job_id')
    if current is not None:
        ids.add(str(current))
    for history in state.get('history', []):
        if history.get('job_id') is not None:
            ids.add(str(history['job_id']))
        submission = history.get('submission') or {}
        if submission.get('comment'):
            comments.add(submission['comment'])
    intent = state.get('intent')
    if isinstance(intent, dict):
        intents.append(dict(intent))
        if intent.get('comment'):
            comments.add(intent['comment'])
    return {'existing_current_job_id': str(current) if current is not None else None,
            'legacy_job_ids': sorted(ids), 'legacy_comments': sorted(comments),
            'unresolved_submission_intents': intents}


def retire_legacy_monitor(root, direct_package, runtime, emit=None):
    """Return READY/WAITING while holding launcher locks in module globals.

    No scheduler command or scientific signal is issued. A caller receiving
    unresolved submission intents or a changed current job must reconcile those
    exact legacy jobs before starting new workers. This helper does not hide
    an old monitor's just-accepted sbatch submission.
    """
    root, direct_package, runtime = (Path(value).absolute()
                                     for value in (root, direct_package, runtime))
    runtime.mkdir(parents=True, exist_ok=True)
    proof_path = runtime/'retired-legacy-monitor.json'
    script = direct_package/'launch.py'
    old_base = root/'.paper3-hpc/direct-train'
    key = (str(root), str(direct_package))
    record = _legacy_read_json(proof_path, {})
    if not isinstance(record, dict):
        record = {}
    if record and (record.get('root') != str(root) or record.get('direct_package') != str(direct_package)):
        raise ValueError('Legacy retirement record belongs to another root or package')
    record.setdefault('schema_version', 'paper3-retired-legacy-submitter-v1')
    record.setdefault('root', str(root))
    record.setdefault('direct_package', str(direct_package))
    record.setdefault('retired_monitors', [])

    def publish(status, reason, **fields):
        record.update(status=status, reason=reason,
                      observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      scientific_processes_signalled=False, scheduler_jobs_cancelled=False,
                      old_launch_journal_modified=False, **fields)
        _legacy_atomic(proof_path, record)
        if emit:
            emit('LEGACY_MONITOR_RETIREMENT', status=status, reason=reason,
                 existing_current_job_id=record.get('existing_current_job_id'),
                 unresolved_submission_intents=record.get('unresolved_submission_intents', []))
        return dict(record)

    with (runtime/'.retire-legacy.lock').open('a') as serial:
        try:
            fcntl.flock(serial, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'WAITING', 'reason': 'another_retirement_helper_active'}
        held = _LEGACY_HELD_LOCKS.setdefault(key, {})
        if 'launch' not in held:
            handle = _legacy_try_lock(old_base/'launch.lock')
            if handle is None:
                # Old step() can be in sbatch or reconciliation; do not kill it
                # mid-submission. Retry after its bounded command completes.
                return publish('WAITING', 'legacy_launch_step_finishing')
            held['launch'] = handle
        state = _legacy_read_json(old_base/'state.json', {})
        if not isinstance(state, dict) or (state and state.get('root') != str(root)):
            raise ValueError('Old launch journal does not match the exact project')
        record.setdefault('original_launch_state', state)
        record.update(_legacy_evidence(state))
        processes = _legacy_processes(root, script)
        if processes is None:
            return publish('WAITING', 'process_identity_unavailable')
        for observed in processes:
            current = _legacy_read_process(observed['pid'])
            if (not _legacy_matches(current, root, script)
                    or current['start_time_ticks'] != observed['start_time_ticks']):
                continue
            identity = {'pid': observed['pid'], 'start_time_ticks': observed['start_time_ticks'],
                        'script': str(script), 'root': str(root), 'uid': observed['uid']}
            try:
                os.kill(observed['pid'], signal.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                return publish('WAITING', 'legacy_monitor_signal_not_permitted')
            if identity not in record['retired_monitors']:
                record['retired_monitors'].append(identity)
        remaining = _legacy_processes(root, script)
        if remaining is None or remaining:
            return publish('WAITING', 'waiting_for_exact_legacy_monitor_exit')
        if 'monitor' not in held:
            handle = _legacy_try_lock(old_base/'monitor.lock')
            if handle is None:
                return publish('WAITING', 'legacy_monitor_lock_still_owned')
            held['monitor'] = handle
        # Both locks now exclude late old-launcher steps/monitor starts. Its
        # journal is a stable final snapshot, including midflight submissions.
        final = _legacy_read_json(old_base/'state.json', {})
        if final != state:
            return publish('WAITING', 'legacy_journal_changed_despite_launcher_lock')
        if state.get('intent'):
            return publish('WAITING', 'legacy_submission_ack_requires_reconciliation',
                           launch_lock_held=True, monitor_lock_held=True,
                           auto_resubmission_disabled=True, retirement_complete=True,
                           requires_legacy_job_reconciliation=True)
        current = record.get('existing_current_job_id')
        if current is not None and current != '770043':
            return publish('WAITING', 'legacy_current_job_changed_requires_handoff',
                           launch_lock_held=True, monitor_lock_held=True,
                           auto_resubmission_disabled=True, retirement_complete=True,
                           requires_legacy_job_reconciliation=True)
        return publish('READY', 'legacy_auto_resubmitter_retired',
                       launch_lock_held=True, monitor_lock_held=True,
                       auto_resubmission_disabled=True, retirement_complete=True,
                       requires_legacy_job_reconciliation=False)
