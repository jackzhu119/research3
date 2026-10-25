"""Cooperative handoff of the original R1 allocation; never alter model files.

The old batch runner converts USR1 to a native kernel interruption. The kernel
saves its own checkpoint and closes journals. This module only asks that runner
to pause and observes scheduler, lock, receipt and checkpoint metadata.
"""
from contextlib import ExitStack
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


OLD_JOB = '770043'
OLD_COMMENT = 'paper3-direct-debd49e881f04859ba643afa8a68a2c8'
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'BOOT_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'DEADLINE'}
META_KINDS = {'meta_engineering', 'meta_screening', 'meta_confirmation'}
COMPLETE_META = {'META_JOB_COMPLETE', 'META_ENGINEERING_COMPLETE'}


class HandoffTransient(RuntimeError):
    """Observation unavailable; preserve old owner and retry, not re-fit."""


class HandoffOwnershipError(RuntimeError):
    """The requested job is not the expected original R1 allocation."""


def _handoff_read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as error:
        raise HandoffTransient('read_' + type(error).__name__) from None


def _handoff_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _handoff_emit(callback, event, **fields):
    if callback is not None:
        callback(event, **fields)


def _handoff_command(run, command):
    runner = run or subprocess.run
    try:
        result = runner(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HandoffTransient('scheduler_' + type(error).__name__) from None
    return result


def _handoff_state(value):
    return value.strip().split()[0].rstrip('+').upper() if value.strip() else 'UNKNOWN'


def inspect_old_allocation(old_job=OLD_JOB, *, run=None, expected_comment=OLD_COMMENT,
                           expected_partition='hpc_gpu1', account=None):
    """Use scheduler job ownership, never a login-node PID, as authority."""
    old_job = str(old_job)
    if not re.fullmatch(r'[0-9]{1,20}', old_job):
        raise ValueError('numeric_old_job_required')
    account = account or getpass.getuser()
    q = _handoff_command(run, ['squeue', '--array', '--noheader', '--jobs=' + old_job,
                              '--format=%i|%u|%k|%T|%j|%P|%N'])
    if q.returncode != 0:
        # A completed job can be absent from slurmctld. Accounting still must
        # positively identify the terminal parent before any replacement runs.
        output = (q.stderr or '').lower()
        if not any(text in output for text in ('invalid job id', 'invalid job_id', 'invalid jobid')):
            raise HandoffTransient('squeue_nonzero')
    found = []
    for raw in (q.stdout or '').splitlines():
        fields = [part.strip() for part in raw.split('|')]
        if len(fields) >= 7 and fields[0] == old_job:
            found.append(fields)
    if len(found) > 1:
        raise HandoffOwnershipError('ambiguous_original_parent_job')
    if found:
        fields = found[0]
        if fields[1] != account or fields[2] != expected_comment or fields[5] != expected_partition:
            raise HandoffOwnershipError('original_job_user_comment_or_partition_mismatch')
        return {'job_id': old_job, 'state': _handoff_state(fields[3]),
                'name': fields[4], 'partition': fields[5], 'nodes': fields[6],
                'comment': fields[2], 'user': fields[1], 'source': 'squeue', 'terminal': False}
    a = _handoff_command(run, ['sacct', '-X', '--noheader', '--parsable2', '--jobs=' + old_job,
                              '--format=JobIDRaw,User%64,State%32,ExitCode,JobName%128,Partition%64,Comment%128,NodeList%256'])
    if a.returncode != 0:
        raise HandoffTransient('sacct_nonzero')
    parents = []
    for raw in (a.stdout or '').splitlines():
        fields = [part.strip() for part in raw.split('|')]
        if len(fields) >= 8 and fields[0] == old_job:
            parents.append(fields)
    if len(parents) != 1:
        raise HandoffTransient('terminal_parent_accounting_not_yet_visible')
    fields = parents[0]
    # Some installations do not store the Comment field in accounting; the
    # exact persisted job ID and native project receipts still constrain reuse.
    if (fields[1] not in ('', account) or fields[5] != expected_partition
            or fields[6] not in ('', '(null)', expected_comment)):
        raise HandoffOwnershipError('original_accounting_user_comment_or_partition_mismatch')
    if fields[6] in ('', '(null)') and fields[4] != 'paper3-train-direct':
        raise HandoffOwnershipError('commentless_original_accounting_job_name_mismatch')
    state = _handoff_state(fields[2])
    return {'job_id': old_job, 'state': state, 'exit_code': fields[3], 'name': fields[4],
            'partition': fields[5], 'nodes': fields[7], 'comment': fields[6],
            'user': fields[1], 'source': 'sacct', 'terminal': state in TERMINAL}


def _handoff_relative(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise HandoffOwnershipError('active_receipt_path_not_root_relative')
    result = root / path
    if any(part.is_symlink() for part in (result, *result.parents) if part != root.parent):
        raise HandoffOwnershipError('active_receipt_path_is_symbolic_link')
    return result


def _handoff_file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint_quiescence(root, old_job=OLD_JOB, *, skip_owner_locks=False):
    """Do not load torch or read datasets; native resumed kernels verify state."""
    root = Path(root).resolve()
    campaign = _handoff_read(root/'runs/training_campaign_v1/campaign_progress.json', {})
    if not isinstance(campaign, dict) or not campaign:
        raise HandoffTransient('original_campaign_progress_not_readable')
    active = campaign.get('active') or {}
    record = campaign.get('active_process')
    owner_lock_paths = [root/'.paper3-hpc/direct-train.lock', root/'runs/.training_campaign_v1.lock']
    lock_paths = [] if skip_owner_locks else list(owner_lock_paths)
    checkpoint = None
    receipt_status = None
    active_progress = {}
    active_kernel_wall = {}
    native_job_lock = None
    if active.get('expected_receipt'):
        receipt = _handoff_relative(root, active['expected_receipt'])
        folder = receipt.parent
        native_job_lock = folder/'.job.lock'
        lock_paths.append(native_job_lock)
        receipt_value = _handoff_read(receipt, {}) or {}
        receipt_status = receipt_value.get('status')
        if active.get('kind') in META_KINDS:
            latest = _handoff_read(folder/'latest_checkpoint.json')
            progress = _handoff_read(folder/'progress.json', {}) or {}
            active_progress = progress
            if latest is not None:
                filename = latest.get('path') if isinstance(latest, dict) else None
                if (not isinstance(filename, str) or Path(filename).name != filename
                        or not filename or latest.get('relative_to_job_output') is not True
                        or not re.fullmatch(r'[0-9a-f]{64}', str(latest.get('sha256', '')))):
                    raise HandoffTransient('original_latest_checkpoint_descriptor_not_valid')
                path = folder/filename
                try:
                    if path.is_symlink() or _handoff_file_sha(path) != latest['sha256']:
                        raise HandoffTransient('original_latest_checkpoint_digest_mismatch')
                except OSError as error:
                    raise HandoffTransient('checkpoint_' + type(error).__name__) from None
                checkpoint = {**latest, 'path': str(path.relative_to(root))}
                wall = _handoff_read(folder/'wall_budget.json', {}) or {}
                active_kernel_wall = wall
                if wall and wall.get('interval') is not None:
                    return {'ready': False, 'reason': 'active_kernel_wall_interval_still_open'}
            elif receipt_status not in COMPLETE_META:
                observed = [progress.get(key, 0) for key in ('outer_updates_completed',
                    'outer_attempts_dispatched', 'confirmed_outer_updates_observed', 'inner_updates_actual')]
                # An initialization that never dispatched is not a lost model.
                if any(isinstance(v, (int,float)) and v != 0 for v in observed):
                    return {'ready': False, 'reason': 'observed_meta_work_without_published_latest_checkpoint'}
    if active and isinstance(record, dict) and record.get('cleanup_verified') is not True:
        return {'ready': False, 'reason': 'native_stage_group_cleanup_not_yet_confirmed'}
    wall = _handoff_read(root/'runs/training_campaign_v1/wall_budget.json', {}) or {}
    if wall.get('interval') is not None:
        return {'ready': False, 'reason': 'campaign_wall_interval_still_open'}
    with ExitStack() as stack:
        for path in lock_paths:
            try:
                handle = stack.enter_context(path.open('rb'))
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {'ready': False, 'reason': 'old_owner_lock_still_held',
                        'lock': str(path.relative_to(root))}
        direct = _handoff_read(root/'.paper3-hpc/jobs'/str(old_job)/'direct_result.json', {}) or {}
        if direct and str(direct.get('job_id', old_job)) != str(old_job):
            raise HandoffOwnershipError('old_direct_result_job_identity_mismatch')
        return {'ready': True, 'campaign_status': campaign.get('status'),
                'active': active, 'active_receipt_status': receipt_status,
                'checkpoint': checkpoint, 'direct_status': direct.get('status'),
                'old_locks_observed_free': [str(p.relative_to(root)) for p in lock_paths],
                'campaign_wall_interval_closed': True,
                'campaign_wall_journal_present': bool(wall),
                'active_kernel_wall_journal_present': bool(active_kernel_wall),
                'active_kernel_wall_interval_closed': bool(active_kernel_wall) and active_kernel_wall.get('interval') is None,
                'active_process_cleanup_verified': isinstance(record, dict) and record.get('cleanup_verified') is True,
                'observed_outer_updates_completed': active_progress.get('outer_updates_completed'),
                'native_job_lock_observed_free': bool(native_job_lock and native_job_lock.is_file()),
                'owner_locks_skipped': [str(p.relative_to(root)) for p in owner_lock_paths] if skip_owner_locks else [],
                'prior_completed_actions': len(campaign.get('actions', [])),
                'source_outputs_rewritten': False}


def _handoff_replacement_proof(value):
    """Caller has already observed the replacement queue before this handoff."""
    if not isinstance(value, dict) or value.get('ready') is not True:
        return None
    jobs = value.get('jobs')
    if not isinstance(jobs, list) or len(jobs) != 3:
        return None
    identifiers = [str(job.get('job_id','')) for job in jobs if isinstance(job,dict)]
    if len(identifiers)!=3 or len(set(identifiers))!=3 or any(not re.fullmatch(r'\d+',job) for job in identifiers):
        return None
    return {'ready':True,'marker':value.get('marker'),'jobs':jobs}


def _handoff_legacy_backup_proof(root, old_job):
    """A completed native cleanup and closed journals precede legacy backup.

    The original execute_campaign finally closes its journal before traversing
    backup_manifest(root). Only the native task lock is required here: the old
    direct/campaign owner locks intentionally remain held during bookkeeping.
    """
    proof = inspect_checkpoint_quiescence(root, old_job, skip_owner_locks=True)
    if not proof.get('ready'):
        return None
    active = proof.get('active') or {}
    checkpoint = proof.get('checkpoint') or {}
    cursor, observed = checkpoint.get('outer_cursor'), proof.get('observed_outer_updates_completed')
    if (active.get('kind') not in META_KINDS or not proof.get('active_process_cleanup_verified')
            or not proof.get('native_job_lock_observed_free')
            or not proof.get('campaign_wall_journal_present')
            or not proof.get('campaign_wall_interval_closed')
            or not proof.get('active_kernel_wall_journal_present')
            or not proof.get('active_kernel_wall_interval_closed')
            or type(cursor) is not int or type(observed) is not int or cursor < 0 or cursor != observed):
        return None
    return proof


def checkpoint_handoff_step(root, state_dir, *, old_job=OLD_JOB, run=None, emit=None,
                            expected_comment=OLD_COMMENT, expected_partition='hpc_gpu1', account=None,
                            replacement_queue_proof=None):
    """One nonblocking controller iteration; signal only this exact old batch."""
    root, state_dir = Path(root).resolve(), Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir/'handoff.json'
    previous = _handoff_read(path, {}) or {}
    if previous and (previous.get('old_job') != str(old_job) or previous.get('root') != str(root)):
        raise HandoffOwnershipError('handoff_state_belongs_to_another_original_run')
    allocation = inspect_old_allocation(old_job, run=run, expected_comment=expected_comment,
        expected_partition=expected_partition, account=account)
    value = {**previous, 'schema_version': 'paper3-r1-cooperative-checkpoint-handoff-v1',
             'old_job': str(old_job), 'root': str(root), 'updated_utc': time.time(),
             'old_allocation': allocation, 'training_kernel_changed': False}
    if not allocation['terminal']:
        replacement_proof = _handoff_replacement_proof(replacement_queue_proof)
        success_at = previous.get('last_successful_signal_utc')
        if (success_at is not None and time.time()-success_at >= 15 and replacement_proof is not None
                and allocation['state'] in {'RUNNING','SUSPENDED','COMPLETING'}
                and time.time()-previous.get('legacy_backup_cancel_requested_utc',0) >= 30):
            proof = _handoff_legacy_backup_proof(root, old_job)
            if proof is not None:
                # Save authorization and native checkpoint proof BEFORE asking
                # Slurm to end legacy bookkeeping. No process-group kill or
                # node-local PID authority is introduced by this shortcut.
                value.update(status='CHECKPOINT_SAVED_LEGACY_BACKUP_ONLY',
                    legacy_bookkeeping_drain={'reason':'CHECKPOINT_SAVED_LEGACY_BACKUP_ONLY',
                        'native_checkpoint_proof':proof,'replacement_queue_proof':replacement_proof,
                        'successful_usr1_utc':success_at,'proof_saved_utc':time.time()})
                _handoff_atomic(path,value)
                result = _handoff_command(run,['scancel',str(old_job)])
                value['legacy_backup_cancel_requested_utc'] = time.time()
                value['legacy_backup_cancel_returncode'] = result.returncode
                value['status'] = 'WAITING_OLD_BOOKKEEPING_TERMINAL'
                _handoff_atomic(path,value)
                _handoff_emit(emit,'OLD_CHECKPOINT_SAVED_LEGACY_BACKUP_CANCEL_REQUESTED',
                    old_job=str(old_job),reason='CHECKPOINT_SAVED_LEGACY_BACKUP_ONLY',
                    checkpoint=proof['checkpoint'],returncode=result.returncode)
                return value
        # A pending allocation has no running kernel. The scheduler-side state
        # filter prevents this request killing it if it races into RUNNING.
        if allocation['state'] == 'PENDING' and time.time()-previous.get('last_pending_cancel_utc', 0) >= 30:
            result = _handoff_command(run, ['scancel', '--state=PENDING', str(old_job)])
            value['last_pending_cancel_utc'] = time.time()
            value['last_pending_cancel_returncode'] = result.returncode
            _handoff_emit(emit, 'OLD_QUEUED_ALLOCATION_REMOVED', old_job=str(old_job),
                          state_filter='PENDING', returncode=result.returncode)
        # USR1 asks the existing runner to save and return normally. Never use
        # unconditional cancellation, TERM, KILL, a login PID or remote shell.
        last_signal = previous.get('last_signal_utc', 0)
        if allocation['state'] in {'RUNNING','SUSPENDED','COMPLETING'} and time.time()-last_signal >= 300:
            result = _handoff_command(run, ['scancel', '--batch', '--signal=USR1', str(old_job)])
            value['last_signal_utc'] = time.time()
            value['signal_attempts'] = previous.get('signal_attempts', 0)+1
            value['last_signal_returncode'] = result.returncode
            if result.returncode == 0:
                value['last_successful_signal_utc'] = value['last_signal_utc']
            _handoff_emit(emit, 'OLD_RUNNER_CHECKPOINT_REQUESTED', old_job=str(old_job),
                          signal='USR1', returncode=result.returncode)
        value['status'] = 'WAITING_OLD_JOB'
        _handoff_atomic(path, value)
        return value
    quiescence = inspect_checkpoint_quiescence(root, old_job)
    value['quiescence'] = quiescence
    value['status'] = 'READY' if quiescence['ready'] else 'WAITING_QUIESCENT'
    _handoff_atomic(path, value)
    _handoff_emit(emit, 'OLD_RUNNER_HANDOFF_READY' if quiescence['ready'] else 'OLD_RUNNER_CLEANUP_WAIT',
                  old_job=str(old_job), old_scheduler_state=allocation['state'],
                  checkpoint=quiescence.get('checkpoint'), reason=quiescence.get('reason'))
    return value


def ensure_checkpoint_handoff(root, state_dir, *, poll_seconds=10, stop_requested=None, emit=None, **kwargs):
    while True:
        if stop_requested is not None and stop_requested():
            return {'status': 'CONTROLLER_STOPPED_BEFORE_REPLACEMENT_DISPATCH'}
        try:
            value = checkpoint_handoff_step(root, state_dir, emit=emit, **kwargs)
            if value['status'] == 'READY':
                return value
        except HandoffTransient as error:
            _handoff_emit(emit, 'OLD_RUNNER_OBSERVATION_RETRY', reason=str(error))
        time.sleep(max(1, min(float(poll_seconds), 30)))
