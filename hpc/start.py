#!/usr/bin/env python3
"""Plan, or explicitly fetch frozen data and submit one guarded Slurm chain.

This submission-host coordinator never executes a scientific stage locally.
Existing attempts, uncertain sbatch outcomes and original scientific claims
are retained. Repeating the command never authorizes a second training chain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc import submit
from hpc.environment import verify_code

SCHEMA = 'paper3-hpc-one-command-start-v1'
STAGES = ('test', 'prepare', 'train')
QUEUE_FORMAT = '%i|%j|%T|%Z'
JOB_ID = re.compile(r'[0-9]+(?:_[0-9]+)?(?:\+[0-9]+)?\Z')
NO_SUBMISSION_ERRORS = frozenset({'deploy_hpc_code_to_project_first',
    'existing_school_python_not_executable', 'sbatch_not_available_on_submission_host',
    'Slurm_test_only_rejected_request_no_job_submitted'})


class StartError(ValueError):
    pass


def need(condition, code):
    if not condition:
        raise StartError(code)


def safe(path):
    path = Path(os.path.abspath(path))
    need(not any(item.is_symlink() for item in (path, *path.parents)), 'start_symlink_path_forbidden')
    need(not any(ord(character) < 32 or ord(character) == 127 for character in str(path)), 'start_control_character_path_forbidden')
    return path


def read(path):
    path = safe(path)
    before = path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= 8 * 1024**2, 'bounded_regular_start_control_file_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        raw = stream.read(8 * 1024**2 + 1)
    need(len(raw) <= 8 * 1024**2, 'bounded_regular_start_control_file_required')
    value = json.loads(raw)
    need(isinstance(value, dict), 'start_control_object_required')
    return value


def write(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.start-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def owner(root):
    path = safe(root / '.paper3-hpc/start/owner.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise StartError('existing_start_coordinator_is_still_running_no_duplicate_fetch_or_submission') from None
        yield


def records(root, relative, pattern):
    parent = safe(root / relative)
    if not parent.exists():
        return []
    paths = list(parent.iterdir())
    need(len(paths) <= 4096, 'too_many_start_control_files_review_required')
    return [(path, read(path)) for path in sorted(paths) if re.fullmatch(pattern, path.name)]


def submissions(root):
    result = []
    for path, row in records(root, '.paper3-hpc/submissions', r'[0-9]+-[a-f0-9]+\.json'):
        if row.get('project') != str(root):
            continue
        need(row.get('schema_version') == 'paper3-slurm-submission-v1'
             and row.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
             and JOB_ID.fullmatch(str(row.get('job_id', ''))) is not None,
             'existing_project_submission_receipt_requires_review')
        result.append(row)
    return result


def active_jobs(root, receipts, *, runner=subprocess.run):
    """Scope by real root-bound receipts or exact Paper3 WorkDir, never name alone."""
    need('|' not in str(root), 'project_path_unsupported_by_Slurm20_queue_format')
    response = runner(['squeue', '--noheader', '--array', '--user', str(os.getuid()), '--format=' + QUEUE_FORMAT],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=30)
    need(response.returncode == 0 and isinstance(response.stdout, str) and len(response.stdout) <= 1024**2,
         'own_account_squeue_unavailable_no_download_or_submission')
    known = {row['job_id'] for row in receipts}
    result = []
    for line in response.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split('|')]
        need(len(fields) == 4 and JOB_ID.fullmatch(fields[0]) is not None and fields[2],
             'own_account_squeue_result_unrecognized_no_submission')
        job_id, name, state, workdir = fields
        same_project = bool(workdir.startswith('/') and os.path.abspath(workdir) == str(root))
        if job_id in known or (name.startswith('paper3-') and same_project):
            result.append({'job_id': job_id, 'job_name': name, 'state': state,
                           'binding': 'root_submission_receipt' if job_id in known else 'exact_project_workdir'})
    return result


def source_identity(root, *, runner=subprocess.run, verifier=verify_code):
    checked = verifier(root)
    value = runner(['git', 'rev-parse', '--verify', 'HEAD'], cwd=root, stdin=subprocess.DEVNULL,
                   capture_output=True, text=True, check=False, timeout=30)
    commit = value.stdout.strip() if value.returncode == 0 else ''
    need(re.fullmatch(r'[a-f0-9]{40}', commit) is not None, 'verified_local_source_commit_required_no_remote_Git_operation')
    return {'source_commit': commit, 'code_manifest_sha256': checked['manifest_sha256']}


def requests(args, root, config):
    result = []
    for stage in STAGES:
        train = stage == 'train'
        options = SimpleNamespace(stage=stage, root=root, python=args.python or config.get('python') or sys.executable,
            account=None, partition=None, dependency=None,
            cpus=args.cpus if train else None, mem_gib=args.mem_gib if train else None,
            gpus=args.gpus if train else None, time=args.time if train else None)
        result.append(submit.build_request(options, config))
    return result


def monitor_command(root):
    return shlex.join([sys.executable, str(root / 'hpc/monitor.py'), '--root', str(root)])


def prior_result(root, attempt, receipts):
    """Recover durable API receipts, but never replay an already intended submit."""
    recovered = []
    for intent in attempt.get('submission_intents', []):
        matches = [row for row in receipts if all(row.get(key) == value for key, value in intent['request'].items())]
        need(len({row['job_id'] for row in matches}) <= 1, 'multiple_real_submissions_for_one_intent_require_review')
        recovered.extend(matches)
    jobs = {row['stage']: row['job_id'] for row in attempt.get('submitted_jobs', []) + recovered}
    unknown = [row['stage'] for row in attempt.get('submission_intents', [])
               if row.get('state') != 'NOT_SUBMITTED_BLOCKED' and row['stage'] not in jobs]
    if not jobs and not unknown:
        return None
    failures = []
    for stage, job_id in jobs.items():
        path = root / '.paper3-hpc/jobs' / job_id / ('stage-' + stage + '.json')
        if path.exists():
            terminal = read(path)
            if (terminal.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
                    and terminal.get('job_id') == job_id and terminal.get('stage') == stage
                    and type(terminal.get('returncode')) is int and terminal['returncode'] != 0):
                failures.append({'stage': stage, 'job_id': job_id, 'returncode': terminal['returncode'],
                                 'receipt': str(path.relative_to(root))})
    return {'status': ('SUBMISSION_OUTCOME_UNCERTAIN_NO_RESUBMISSION' if unknown else
                       'PRIOR_STAGE_FAILED_MANUAL_REVIEW_REQUIRED' if failures else
                       'EXISTING_PARTIAL_CHAIN_REQUIRES_REVIEW' if set(jobs) != set(STAGES) else
                       'EXISTING_CHAIN_NO_RESUBMISSION'),
            'job_ids': jobs, 'unknown_submission_stages': unknown, 'failed_stages': failures,
            'missing_stages': [stage for stage in STAGES if stage not in jobs],
            'minimal_gap': ('Inspect the failed stage receipt, fix that specific failure and review pending afterok dependencies before explicitly authorizing recovery. No jobs are cancelled or requeued; original claims and checkpoints remain in place.'
                if failures else 'Inspect the recorded jobs and attempt; authorize recovery only after resolving the original outcome. Existing scientific claims and checkpoints remain in place.'),
            'source_identity': attempt.get('source_identity'), 'monitor_command': monitor_command(root),
            'scientific_completion_claimed': False, 'new_jobs_submitted': False}


def start(args, *, runner=subprocess.run, verifier=verify_code, preparer=None, submitter=submit.submit):
    root = safe(args.root)
    need(root.is_dir(), 'existing_project_directory_required')
    config = submit.load_config(args.config or root / 'hpc/resources.json')
    planned = requests(args, root, config)
    if not args.execute or args.dry_run:
        return {'schema_version': SCHEMA, 'status': 'PLAN_ONLY_NO_FETCH_OR_SUBMISSION',
                'stages': [{'stage': row['stage'], 'request': row,
                    'dependency_on_stage': None if index == 0 else STAGES[index-1],
                    'dependency_bound_only_after_real_job_id': index > 0} for index, row in enumerate(planned)],
                'execute_requires': ['--execute', '--network-host'], 'scientific_completion_claimed': False}
    need(args.network_host, 'execute_requires_explicit_network_host_permission')
    need(not os.environ.get('SLURM_JOB_ID'), 'network_fetch_must_run_on_permitted_submission_host_not_compute_job')
    with owner(root):
        existing = submissions(root)
        queued = active_jobs(root, existing, runner=runner)
        prior_attempts = []
        for path, attempt in records(root, '.paper3-hpc/start', r'attempt-[a-f0-9]{32}\.json'):
            need(attempt.get('schema_version') == SCHEMA and attempt.get('project') == str(root),
                 'prior_start_attempt_identity_requires_review')
            previous_result = prior_result(root, attempt, existing)
            if previous_result:
                prior_attempts.append((path, previous_result))
        if queued:
            queued_ids = {row['job_id'] for row in queued}
            blocked = [prior for path, prior in prior_attempts
                       if set(prior['job_ids'].values()) & queued_ids and prior['failed_stages']]
            if blocked:
                return {'status': 'EXISTING_CHAIN_HAS_FAILED_STAGE_REVIEW_REQUIRED', 'active_jobs': queued,
                        'job_ids': blocked[0]['job_ids'], 'failed_stages': blocked[0]['failed_stages'],
                        'minimal_gap': blocked[0]['minimal_gap'], 'monitor_command': monitor_command(root),
                        'scientific_completion_claimed': False, 'new_jobs_submitted': False}
            return {'status': 'EXISTING_ACTIVE_JOBS_NO_DUPLICATE_FETCH_OR_SUBMISSION', 'active_jobs': queued,
                    'job_ids': [row['job_id'] for row in queued], 'monitor_command': monitor_command(root),
                    'scientific_completion_claimed': False, 'new_jobs_submitted': False}
        if prior_attempts:
            path, prior = prior_attempts[0]
            return {**prior, 'attempt': str(path.relative_to(root))}
        historical_train = [row for row in existing if row.get('stage') == 'train']
        if historical_train:
            return {'status': 'EXISTING_TRAIN_SUBMISSION_REQUIRES_REVIEW',
                    'job_ids': [row['job_id'] for row in historical_train], 'monitor_command': monitor_command(root),
                    'minimal_gap': 'Inspect the existing training outcome and checkpoint before authorizing another submission.',
                    'scientific_completion_claimed': False, 'new_jobs_submitted': False}
        identity = source_identity(root, runner=runner, verifier=verifier)
        attempt_id = uuid.uuid4().hex
        path = root / '.paper3-hpc/start' / ('attempt-' + attempt_id + '.json')
        value = {'schema_version': SCHEMA, 'attempt_id': attempt_id, 'project': str(root),
            'source_identity': identity, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'FETCHING_FROZEN_DATA', 'submission_intents': [], 'submitted_jobs': [],
            'credentials_recorded': False, 'scientific_completion_claimed': False}
        write(path, value)
        if preparer is None:
            from hpc.stage_data import prepare_data
            preparer = prepare_data
        try:
            data = preparer(root, mode='fetch', execute=True, network_host=True, prompt=sys.stdin.isatty(),
                progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
            value['data_receipt'] = {key: data.get(key) for key in ('status', 'attempt', 'error_code', 'failure_detail', 'minimal_gap') if key in data}
            if data.get('status') != 'ready':
                value['status'] = 'DATA_NOT_READY_NO_JOBS_SUBMITTED'
                write(path, value)
                return {**value['data_receipt'], 'status': value['status'], 'attempt': str(path.relative_to(root)),
                        'job_ids': {}, 'new_jobs_submitted': False, 'scientific_completion_claimed': False}
            # Detect manually queued project work that appeared while fetching.
            queued = active_jobs(root, submissions(root), runner=runner)
            if queued:
                value.update(status='ACTIVE_JOBS_APPEARED_NO_NEW_SUBMISSION', active_jobs=queued)
                write(path, value)
                return {**value, 'monitor_command': monitor_command(root), 'new_jobs_submitted': False}
            need(source_identity(root, runner=runner, verifier=verifier) == identity,
                 'source_identity_changed_during_fetch_no_submission')
            previous = None
            for request in planned:
                stage = request['stage']
                need(source_identity(root, runner=runner, verifier=verifier) == identity,
                     'source_identity_changed_before_submission_preserve_existing_jobs')
                options = SimpleNamespace(stage=stage, root=root, python=request['python'],
                    account=None, partition=None, dependency='afterok:' + previous if previous else None,
                    cpus=args.cpus if stage == 'train' else None, mem_gib=args.mem_gib if stage == 'train' else None,
                    gpus=args.gpus if stage == 'train' else None, time=args.time if stage == 'train' else None)
                bound = {**submit.build_request(options, config), 'start_source_identity': identity,
                         'start_attempt_id': attempt_id}
                intent = {'stage': stage, 'state': 'SUBMISSION_INTENT_OUTCOME_UNKNOWN', 'request': bound}
                value['submission_intents'].append(intent); value['status'] = 'SUBMITTING_' + stage.upper()
                write(path, value)  # fsync intent before entering any sbatch-capable API.
                try:
                    receipt = submitter(bound)
                except BaseException as exc:
                    if isinstance(exc, submit.SubmissionError) and str(exc) in NO_SUBMISSION_ERRORS:
                        intent['state'] = 'NOT_SUBMITTED_BLOCKED'
                    value['status'] = 'SUBMISSION_BLOCKED_PRESERVE_INTENT'
                    value['error_type'] = type(exc).__name__
                    write(path, value)
                    raise
                need(receipt.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
                     and re.fullmatch(r'[0-9]+', str(receipt.get('job_id', ''))) is not None
                     and all(receipt.get(key) == item for key, item in bound.items()),
                     'submission_receipt_unverified_preserve_intent_no_replay')
                previous = receipt['job_id']
                intent['state'] = 'SUBMISSION_RECEIPT_VERIFIED'
                value['submitted_jobs'].append(receipt)
                write(path, value)
            value['status'] = 'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE'
            write(path, value)
            return {'status': value['status'], 'attempt': str(path.relative_to(root)), 'source_identity': identity,
                    'job_ids': {row['stage']: row['job_id'] for row in value['submitted_jobs']},
                    'monitor_command': monitor_command(root), 'scientific_completion_claimed': False,
                    'new_jobs_submitted': True}
        except BaseException as exc:
            value['error_type'] = type(exc).__name__
            if value['status'] == 'FETCHING_FROZEN_DATA':
                value['status'] = 'FETCH_INTERRUPTED_NO_JOBS_SUBMITTED'
            write(path, value)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--python', type=Path, help='Reuse the existing school interpreter; never install dependencies')
    parser.add_argument('--cpus', type=int, help='Override train resources only')
    parser.add_argument('--mem-gib', type=int, help='Override train resources only')
    parser.add_argument('--gpus', type=int, help='Override allocation only; no multi-GPU algorithm change')
    parser.add_argument('--time', help='Train Slurm wall time only; original scientific caps remain fixed')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--network-host', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = start(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if result['status'] in {'PLAN_ONLY_NO_FETCH_OR_SUBMISSION',
            'EXISTING_ACTIVE_JOBS_NO_DUPLICATE_FETCH_OR_SUBMISSION', 'ACTIVE_JOBS_APPEARED_NO_NEW_SUBMISSION',
            'EXISTING_CHAIN_NO_RESUBMISSION', 'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE'} else 2
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        detail = ({'errno': exc.errno, 'errno_name': errno.errorcode.get(exc.errno, 'UNKNOWN_ERRNO')}
                  if isinstance(exc, OSError) else None)
        print(json.dumps({'status': 'START_BLOCKED_PRESERVE_EXISTING_JOBS_AND_CLAIMS',
            'reason': str(exc) if isinstance(exc, (StartError, submit.SubmissionError)) else type(exc).__name__,
            'failure_detail': detail,
            'scientific_completion_claimed': False}), flush=True)
        return 2
    except KeyboardInterrupt:
        print('Start interrupted; recorded intents, jobs and scientific claims are preserved.', flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
