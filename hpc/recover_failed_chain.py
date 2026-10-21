#!/usr/bin/env python3
"""Explicitly retire a failed CPU-test chain whose dependents never started.

This submission-host utility never submits a job, runs a fit, edits an original
receipt, or resets a claim/budget. Scheduler mutations require --execute and are
restricted to the original, still-PENDING descendant job IDs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc import start
from hpc.run_stage import StageError, owner_lock

SCHEMA = 'paper3-hpc-failed-prefit-recovery-v1'
TERMINAL_TEST = {'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
FIELDS = ('job_id', 'state', 'exit_code', 'start', 'elapsed', 'account', 'user', 'name', 'workdir', 'end')
UNKNOWN_START = {'Unknown', 'None', 'N/A', ''}
SAFE_STATES = TERMINAL_TEST | {'PENDING', 'RUNNING', 'COMPLETED', 'CONFIGURING', 'COMPLETING',
    'BOOT_FAIL', 'DEADLINE', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'RESIZING',
    'REVOKED', 'SIGNALING', 'SPECIAL_EXIT', 'STAGE_OUT', 'STOPPED', 'SUSPENDED'}
DEPENDENCY_SUFFIX = r'(?:\((?:unfulfilled|fulfilled|unsatisfied|satisfied)\))?'


class RecoveryError(ValueError):
    def __init__(self, code, diagnostic=None):
        super().__init__(code)
        self.diagnostic = diagnostic


def need(value, code):
    if not value:
        raise RecoveryError(code)


def _hash(path):
    path = start.safe(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(value):
    return value.split()[0].rstrip('+') if value.strip() else ''


def _timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?', value):
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return False
    return True


def _initial_pending_control(control, binding, stage, user, uid):
    job_id = binding['jobs'][stage]['job_id']
    parent = binding['jobs']['test' if stage == 'prepare' else 'prepare']['job_id']
    return (isinstance(control, dict) and control.get('job_id') == job_id
        and control.get('state') == 'PENDING' and control.get('uid') == uid
        and control.get('user') == user and control.get('account') == binding['account']
        and control.get('workdir') == binding['project'] and control.get('name') == 'paper3-' + stage
        and control.get('runtime') == '00:00:00' and control.get('restarts') == 0
        and isinstance(control.get('dependency'), str)
        and re.fullmatch('afterok:' + parent + DEPENDENCY_SUFFIX, control['dependency']) is not None)


def _validate_intent(binding, intent, user):
    """Validate the original pre-cancel witness, including legacy 7ae receipts."""
    need(isinstance(intent, dict) and intent.get('schema_version') == SCHEMA
         and intent.get('status') == 'PENDING_CANCEL_INTENT_PRESERVE_ORIGINALS'
         and intent.get('project') == binding['project'] and intent.get('binding') == binding
         and intent.get('failed_test_job') == binding['jobs']['test']['job_id']
         and type(intent.get('uid')) is int and intent['uid'] == os.getuid()
         and intent.get('user') == user and intent.get('scientific_fit_performed') is False
         and intent.get('scientific_budget_changed') is False,
         'recovery_original_cancel_intent_invalid')
    rows = intent.get('accounting_before', {})
    controls = intent.get('live_pending_before', {})
    ids = [binding['jobs'][s]['job_id'] for s in start.STAGES]
    need(isinstance(rows, dict) and set(rows) == set(ids), 'recovery_initial_accounting_incomplete')
    need(isinstance(controls, dict), 'recovery_initial_live_pending_proof_invalid')
    for stage, job_id in zip(start.STAGES, ids):
        row = rows[job_id]
        need(isinstance(row, dict) and all(isinstance(row.get(k), str) for k in FIELDS[:-1])
             and ('end' not in row or isinstance(row['end'], str)) and row.get('job_id') == job_id
             and row.get('account') == binding['account'] and row.get('user') == user
             and row.get('workdir') == binding['project'] and row.get('name') == 'paper3-' + stage,
             'recovery_initial_identity_invalid')
        if stage == 'test':
            exit_code = re.fullmatch(r'(\d+):(\d+)', row.get('exit_code', ''))
            need(row.get('state') in TERMINAL_TEST and exit_code and any(map(int, exit_code.groups())),
                 'recovery_initial_test_not_failed')
        else:
            # This witness is always the strict original pre-cancel observation.
            # A normalized timestamp can never establish its own provenance.
            need(row.get('state') in {'PENDING', 'CANCELLED'} and row.get('start') in UNKNOWN_START
                 and row.get('elapsed') == '0', 'recovery_initial_descendant_not_never_started')
            if row['state'] == 'PENDING':
                need(_initial_pending_control(controls.get(stage),
                                              binding, stage, user, intent['uid']),
                     'recovery_initial_live_pending_proof_invalid')
    return intent


def _load_intent(binding, user):
    path = start.safe(Path(binding['project']) / '.paper3-hpc/recoveries'
                      / ('test-' + binding['jobs']['test']['job_id'] + '.intent.json'))
    if not path.exists():
        return None
    return _validate_intent(binding, start.read(path), user)


def _never_started(row, binding, intent=None):
    if row.get('state') not in {'PENDING', 'CANCELLED'} or row.get('elapsed') != '0':
        return False
    if row.get('start') in UNKNOWN_START:
        return True
    # Slurm 20.11 normalizes a cancelled job's missing start to its end. Accept
    # that only after a fully bound, original PENDING/unknown-start witness.
    if (row.get('state') != 'CANCELLED' or not _timestamp(row.get('start'))
            or row.get('start') != row.get('end') or intent is None):
        return False
    original = intent['accounting_before'].get(row.get('job_id'), {})
    return (original.get('state') == 'PENDING' and original.get('start') in UNKNOWN_START
            and original.get('elapsed') == '0')


def _safe_accounting(row):
    """Retain only typed scheduler fields, never arbitrary returned text."""
    def clock(value):
        return value if value in UNKNOWN_START or _timestamp(value) else 'UNRECOGNIZED_VALUE'
    return {'job_id': row['job_id'], 'state': row['state'] if row['state'] in SAFE_STATES else 'UNRECOGNIZED_VALUE',
            'start': clock(row.get('start', '')), 'end': clock(row.get('end', '')),
            'elapsed': row['elapsed'] if re.fullmatch(r'\d{1,20}', row['elapsed']) else 'UNRECOGNIZED_VALUE',
            'exit_code': row['exit_code'] if re.fullmatch(r'\d{1,10}:\d{1,10}', row['exit_code']) else 'UNRECOGNIZED_VALUE'}


def _diagnostic(binding, rows, intent):
    children = []
    for stage in ('prepare', 'train'):
        row = rows[binding['jobs'][stage]['job_id']]
        children.append({'stage': stage, 'actual': _safe_accounting(row),
            'checks': {'state_is_pending_or_cancelled': row['state'] in {'PENDING', 'CANCELLED'},
                       'start_is_unknown': row['start'] in UNKNOWN_START, 'elapsed_is_zero': row['elapsed'] == '0',
                       'cancelled_equal_start_end': row['state'] == 'CANCELLED' and _timestamp(row['start'])
                                                    and row['start'] == row.get('end'),
                       'never_started_proven': _never_started(row, binding, intent)},
            'expected': {'elapsed': '0', 'state': ['PENDING', 'CANCELLED'],
                         'start': 'unknown, or cancelled Start=End backed by original pending cancel intent'}})
    return {'phase': 'validated_accounting_never_started_check', 'children': children,
            'accounting': [_safe_accounting(rows[binding['jobs'][s]['job_id']]) for s in start.STAGES],
            'cancel_intent_available_and_validated': intent is not None,
            'read_only_diagnostic': True,
            'next_read_only_command': ['sacct', '-X', '-j', ','.join(binding['jobs'][s]['job_id'] for s in start.STAGES),
                '-n', '-P', '-o', 'JobIDRaw,State%40,Start,End,ElapsedRaw,NNodes,AllocCPUS,NodeList%64']}


def _bound(root, failed_test_job):
    """Bind the complete original durable chain; never guess from a job name."""
    need(isinstance(failed_test_job, str) and re.fullmatch(r'[0-9]+', failed_test_job), 'recovery_test_job_id_required')
    choices = []
    for path, attempt in start.records(root, '.paper3-hpc/start', r'attempt-[a-f0-9]{32}\.json'):
        rows = attempt.get('submitted_jobs', [])
        if any(row.get('stage') == 'test' and row.get('job_id') == failed_test_job for row in rows):
            choices.append((path, attempt))
    need(len(choices) == 1, 'recovery_exact_original_attempt_required')
    path, attempt = choices[0]
    need(attempt.get('schema_version') == start.SCHEMA and attempt.get('project') == str(root)
         and attempt.get('status') == 'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE',
         'recovery_complete_original_chain_required')
    rows = attempt.get('submitted_jobs', [])
    intents = attempt.get('submission_intents', [])
    need(len(rows) == len(intents) == 3 and all(isinstance(r, dict) for r in rows + intents)
         and all(isinstance(r.get('request'), dict) for r in intents)
         and [r.get('stage') for r in rows] == list(start.STAGES)
         and [r.get('stage') for r in intents] == list(start.STAGES)
         and all(r.get('state') == 'SUBMISSION_RECEIPT_VERIFIED' for r in intents),
         'recovery_unknown_or_partial_submission_forbidden')
    saved = start.records(root, '.paper3-hpc/submissions', r'[0-9]+-[a-f0-9]+\.json')
    original = {}
    accounts = set()
    for index, row in enumerate(rows):
        stage = start.STAGES[index]
        job_id = row.get('job_id', '')
        need(isinstance(job_id, str) and re.fullmatch(r'[0-9]+', job_id)
             and row.get('stage') == stage and row.get('project') == str(root)
             and row.get('schema_version') == 'paper3-slurm-submission-v1'
             and row.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
             and row.get('start_attempt_id') == attempt.get('attempt_id')
             and row.get('start_source_identity') == attempt.get('source_identity')
             and all(row.get(key) == value for key, value in intents[index]['request'].items()),
             'recovery_original_submission_binding_required')
        matches = [(p, r) for p, r in saved if r.get('job_id') == job_id]
        need(len(matches) == 1 and matches[0][1] == row, 'recovery_durable_submission_required')
        accounts.add(row.get('resources', {}).get('account'))
        commands = row.get('command', [])
        need(isinstance(commands, list) and all(isinstance(c, str) for c in commands),
             'recovery_original_command_required')
        dependencies = [c for c in commands if c.startswith('--dependency=')]
        expected = [] if index == 0 else ['--dependency=afterok:' + rows[index-1]['job_id']]
        need(dependencies == expected and '--job-name=paper3-' + stage in commands
             and '--chdir=' + str(root) in commands, 'recovery_original_dependency_binding_required')
        original[stage] = {'job_id': job_id, 'submission': str(matches[0][0].relative_to(root)),
                           'submission_sha256': _hash(matches[0][0])}
    need(len(accounts) == 1 and None not in accounts, 'recovery_original_account_required')
    account = next(iter(accounts))
    need(re.fullmatch(r'[A-Za-z0-9_-]+', account), 'recovery_original_account_required')
    need(len({r['job_id'] for r in original.values()}) == 3
         and original['test']['job_id'] == failed_test_job, 'recovery_original_job_ids_required')
    stage_path = start.safe(root / '.paper3-hpc/jobs' / failed_test_job / 'stage-test.json')
    failed = start.read(stage_path)
    need(failed.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
         and failed.get('stage') == 'test' and failed.get('job_id') == failed_test_job
         and type(failed.get('returncode')) is int and failed['returncode'] != 0
         and failed.get('status') == 'STOPPED_OR_BLOCKED_NOT_COMPLETE'
         and failed.get('scientific_completion_claimed') is False
         and failed.get('external_query_accessed') is False,
         'recovery_original_failed_CPU_test_receipt_required')
    for stage in ('prepare', 'train'):
        need(not start.safe(root / '.paper3-hpc/jobs' / original[stage]['job_id']).exists(),
             'recovery_descendant_has_execution_artifacts_preserve_existing_jobs')
    binding = {'attempt': str(path.relative_to(root)), 'attempt_sha256': _hash(path),
               'failed_test_receipt': str(stage_path.relative_to(root)),
               'failed_test_receipt_sha256': _hash(stage_path), 'jobs': original,
               'account': account, 'project': str(root)}
    return binding


def _run(command, runner):
    try:
        value = runner(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        raise RecoveryError('recovery_scheduler_observation_unavailable') from None
    need(value.returncode == 0 and isinstance(value.stdout, str)
         and len(value.stdout) <= 1024**2, 'recovery_scheduler_observation_failed')
    return value.stdout


def _accounting_rows(binding, *, runner=subprocess.run, username=None):
    ids = [binding['jobs'][s]['job_id'] for s in start.STAGES]
    format_ = 'JobIDRaw,State%64,ExitCode,Start,ElapsedRaw,Account%64,User%64,JobName%64,WorkDir%4096,End'
    raw = _run(['sacct', '--allocations', '--jobs=' + ','.join(ids), '--noheader', '--parsable2',
                '--format=' + format_], runner)
    found = {}
    user = username or pwd.getpwuid(os.getuid()).pw_name
    for line in raw.splitlines():
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split('|')]
        need(len(cells) == len(FIELDS), 'recovery_sacct_record_unrecognized')
        row = dict(zip(FIELDS, cells))
        need(row['job_id'] in ids and row['job_id'] not in found, 'recovery_sacct_exact_jobs_required')
        row['state'] = _state(row['state'])
        stage = next(s for s in start.STAGES if binding['jobs'][s]['job_id'] == row['job_id'])
        need(row['account'] == binding['account'] and row['user'] == user
             and row['name'] == 'paper3-' + stage and row['workdir'] == binding['project'],
             'recovery_scheduler_job_identity_mismatch')
        found[row['job_id']] = row
    need(set(found) == set(ids), 'recovery_all_three_accounting_records_required')
    return found


def accounting(binding, *, runner=subprocess.run, username=None, intent=None):
    user = username or pwd.getpwuid(os.getuid()).pw_name
    if intent is not None:
        _validate_intent(binding, intent, user)
    found = _accounting_rows(binding, runner=runner, username=user)
    ids = [binding['jobs'][s]['job_id'] for s in start.STAGES]
    test = found[ids[0]]
    exit_code = re.fullmatch(r'(\d+):(\d+)', test['exit_code'])
    need(test['state'] in TERMINAL_TEST and exit_code and any(map(int, exit_code.groups())),
         'recovery_scheduler_failed_test_confirmation_required')
    for job_id in ids[1:]:
        row = found[job_id]
        if not _never_started(row, binding, intent):
            raise RecoveryError('recovery_descendant_may_have_started_preserve_existing_jobs',
                                _diagnostic(binding, found, intent))
    return found


def _control_rows(binding, stage, *, runner=subprocess.run, username=None):
    job_id = binding['jobs'][stage]['job_id']
    text = _run(['scontrol', '--oneliner', 'show', 'job', job_id], runner)
    matches = list(re.finditer(r'(?<!\S)([A-Za-z][A-Za-z0-9_]*)=', text))
    row = {m.group(1): text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
           for i, m in enumerate(matches)}
    user = username or pwd.getpwuid(os.getuid()).pw_name
    owner = re.fullmatch(r'([^()\s]+)\((\d+)\)', row.get('UserId', ''))
    need(row.get('JobId') == job_id and row.get('JobName') == 'paper3-' + stage
         and row.get('WorkDir') == binding['project'] and row.get('Account') == binding['account']
         and owner and owner.group(1) == user and int(owner.group(2)) == os.getuid(),
         'recovery_live_job_identity_or_start_state_mismatch')
    return row


def _control(binding, stage, *, runner=subprocess.run, username=None):
    job_id = binding['jobs'][stage]['job_id']
    row = _control_rows(binding, stage, runner=runner, username=username)
    user = username or pwd.getpwuid(os.getuid()).pw_name
    owner = re.fullmatch(r'([^()\s]+)\((\d+)\)', row.get('UserId', ''))
    expected_dependency = binding['jobs']['test' if stage == 'prepare' else 'prepare']['job_id']
    dependency = row.get('Dependency', '')
    need(re.fullmatch(r'afterok:' + expected_dependency + r'(?:\((?:unfulfilled|fulfilled|unsatisfied|satisfied)\))?', dependency),
         'recovery_live_dependency_mismatch')
    need(row.get('JobId') == job_id and row.get('JobName') == 'paper3-' + stage
         and row.get('JobState') == 'PENDING' and row.get('RunTime') == '00:00:00'
         and row.get('Restarts') == '0' and row.get('WorkDir') == binding['project']
         and row.get('Account') == binding['account'] and owner
         and owner.group(1) == user and int(owner.group(2)) == os.getuid(),
         'recovery_live_job_identity_or_start_state_mismatch')
    return {'job_id': job_id, 'state': 'PENDING', 'dependency': dependency,
            'user': user, 'uid': os.getuid(), 'account': binding['account'],
            'workdir': binding['project'], 'name': 'paper3-' + stage,
            'runtime': '00:00:00', 'restarts': 0}


def diagnose(root, failed_test_job, *, runner=subprocess.run, username=None):
    """Read original bindings and scheduler facts; never take locks or mutate."""
    root = start.safe(root)
    need(root.is_dir(), 'recovery_existing_project_required')
    user = username or pwd.getpwuid(os.getuid()).pw_name
    binding = _bound(root, failed_test_job)
    intent_path = start.safe(root / '.paper3-hpc/recoveries' / ('test-' + failed_test_job + '.intent.json'))
    intent, intent_error = None, None
    try:
        intent = _load_intent(binding, user)
    except (RecoveryError, start.StartError, OSError, ValueError, TypeError, KeyError) as exc:
        intent_error = str(exc) if isinstance(exc, (RecoveryError, start.StartError)) else type(exc).__name__
    rows = _accounting_rows(binding, runner=runner, username=user)
    result = {'schema_version': SCHEMA, 'status': 'READ_ONLY_RECOVERY_DIAGNOSTIC',
              'project': str(root), 'failed_test_job': failed_test_job,
              'diagnostic': _diagnostic(binding, rows, intent),
              'cancel_intent': {'exists': intent_path.exists(), 'validated': intent is not None,
                                'error_code': intent_error},
              'live_controls': {}, 'read_only_diagnostic': True,
              'new_jobs_submitted': False, 'scientific_fit_performed': False}
    for stage in ('prepare', 'train'):
        try:
            row = _control_rows(binding, stage, runner=runner, username=user)
            dependency = row.get('Dependency', '')
            runtime = row.get('RunTime', '')
            restarts = row.get('Restarts', '')
            start_time = row.get('StartTime', '')
            reason = row.get('Reason', '')
            result['live_controls'][stage] = {
                'job_id': binding['jobs'][stage]['job_id'], 'identity_validated': True,
                'state': row.get('JobState') if row.get('JobState') in SAFE_STATES else 'UNRECOGNIZED_VALUE',
                'runtime': runtime if re.fullmatch(r'(?:\d+-)?\d{2,3}:\d{2}:\d{2}', runtime) else 'UNRECOGNIZED_VALUE',
                'restarts': restarts if re.fullmatch(r'\d{1,10}', restarts) else 'UNRECOGNIZED_VALUE',
                'start': start_time if start_time in UNKNOWN_START or _timestamp(start_time) else 'UNRECOGNIZED_VALUE',
                'dependency': dependency if re.fullmatch(r'(?:afterok|afterany|afternotok):\d+' + DEPENDENCY_SUFFIX, dependency)
                                            else 'UNRECOGNIZED_VALUE',
                'reason': reason if reason in {'Dependency', 'DependencyNeverSatisfied', 'Resources', 'Priority',
                    'None', 'QOSMaxJobsPerUserLimit', 'JobHeldUser', 'JobHeldAdmin', 'BeginTime'} else 'UNRECOGNIZED_VALUE'}
        except (RecoveryError, start.StartError, OSError, ValueError, TypeError, KeyError) as exc:
            result['live_controls'][stage] = {'job_id': binding['jobs'][stage]['job_id'],
                'identity_validated': False,
                'error_code': str(exc) if isinstance(exc, (RecoveryError, start.StartError)) else type(exc).__name__}
    return result


def superseded_attempts(root):
    """Pure read-only validation used by start.py; no scheduler or cancellation."""
    root = start.safe(root)
    result = {}
    for path, value in start.records(root, '.paper3-hpc/recoveries', r'test-[0-9]+\.json'):
        need(value.get('schema_version') == SCHEMA and value.get('status') == 'RECOVERY_VERIFIED_NO_FIT_REPLAY'
             and value.get('project') == str(root) and value.get('scientific_fit_performed') is False
             and value.get('scientific_budget_changed') is False,
             'recovery_final_proof_invalid')
        test_id = value.get('failed_test_job', '')
        binding = _bound(root, test_id)
        need(value.get('binding') == binding, 'recovery_original_records_changed')
        intent_relative = '.paper3-hpc/recoveries/test-' + test_id + '.intent.json'
        need(value.get('intent') == intent_relative, 'recovery_original_cancel_intent_required')
        intent_path = start.safe(root / intent_relative)
        need(intent_path.is_file() and _hash(intent_path) == value.get('intent_sha256'),
             'recovery_original_cancel_intent_changed')
        intent = start.read(intent_path)
        _validate_intent(binding, intent, value.get('user'))
        need(intent.get('schema_version') == SCHEMA
             and intent.get('status') == 'PENDING_CANCEL_INTENT_PRESERVE_ORIGINALS'
             and intent.get('binding') == binding and intent.get('uid') == value.get('uid')
             and intent.get('user') == value.get('user')
             and intent.get('scientific_fit_performed') is False
             and intent.get('scientific_budget_changed') is False,
             'recovery_original_cancel_intent_invalid')
        before = intent.get('accounting_before', {})
        ids = [binding['jobs'][s]['job_id'] for s in start.STAGES]
        need(set(before) == set(ids), 'recovery_initial_accounting_incomplete')
        for index, job_id in enumerate(ids):
            row = before[job_id]
            stage = start.STAGES[index]
            need(row.get('job_id') == job_id and row.get('account') == binding['account']
                 and row.get('workdir') == str(root) and row.get('name') == 'paper3-' + stage
                 and row.get('user') == value.get('user'), 'recovery_initial_identity_invalid')
            if index == 0:
                need(row.get('state') in TERMINAL_TEST, 'recovery_initial_test_not_failed')
            else:
                need(row.get('state') in {'PENDING', 'CANCELLED'}
                     and row.get('start') in UNKNOWN_START and row.get('elapsed') == '0',
                     'recovery_initial_descendant_not_never_started')
                if row['state'] == 'PENDING':
                    control = intent.get('live_pending_before', {}).get(stage, {})
                    expected = ids[index-1]
                    need(control.get('job_id') == job_id and control.get('state') == 'PENDING'
                         and control.get('uid') == value.get('uid') and control.get('user') == value.get('user')
                         and control.get('account') == binding['account']
                         and control.get('workdir') == str(root) and control.get('name') == 'paper3-' + stage
                         and control.get('runtime') == '00:00:00' and control.get('restarts') == 0
                         and re.fullmatch(r'afterok:' + expected + r'(?:\((?:unfulfilled|fulfilled|unsatisfied|satisfied)\))?',
                                          control.get('dependency', '')),
                         'recovery_initial_live_pending_proof_invalid')
        final = value.get('accounting_after', {})
        ids = [binding['jobs'][s]['job_id'] for s in start.STAGES]
        need(set(final) == set(ids), 'recovery_final_accounting_incomplete')
        for index, job_id in enumerate(ids):
            row = final[job_id]
            need(isinstance(row, dict) and all(isinstance(row.get(k), str) for k in FIELDS[:-1])
                 and ('end' not in row or isinstance(row['end'], str))
                 and row.get('job_id') == job_id and row.get('account') == binding['account']
                 and row.get('workdir') == str(root)
                 and row.get('name') == 'paper3-' + start.STAGES[index]
                 and row.get('user') == value.get('user'), 'recovery_final_identity_invalid')
            if index == 0:
                exit_code = re.fullmatch(r'(\d+):(\d+)', row.get('exit_code', ''))
                need(row.get('state') in TERMINAL_TEST and exit_code
                     and any(map(int, exit_code.groups())), 'recovery_final_test_not_failed')
            else:
                need(row.get('state') == 'CANCELLED' and _never_started(row, binding, intent),
                     'recovery_final_descendant_not_never_started')
        need(value.get('queue_after') == [] and type(value.get('uid')) is int
             and value['uid'] == os.getuid() and value.get('user') == pwd.getpwuid(os.getuid()).pw_name,
             'recovery_final_owner_or_queue_invalid')
        need(binding['attempt'] not in result, 'recovery_duplicate_final_proof')
        result[binding['attempt']] = {'job_ids': {s: binding['jobs'][s]['job_id'] for s in start.STAGES},
                                    'receipt': str(path.relative_to(root))}
    return result


def recover(root, failed_test_job, *, execute=False, dry_run=False, runner=subprocess.run,
            username=None, sleeper=time.sleep):
    root = start.safe(root)
    need(root.is_dir() and '|' not in str(root), 'recovery_existing_project_required')
    need(not os.environ.get('SLURM_JOB_ID'), 'recovery_submission_host_required')
    binding = _bound(root, failed_test_job)
    user = username or pwd.getpwuid(os.getuid()).pw_name
    acting = execute and not dry_run
    with (start.owner(root) if acting else nullcontext()), \
         (start.staged_owner(root) if acting else nullcontext()), \
         (owner_lock(root) if acting else nullcontext()):
        need(_bound(root, failed_test_job) == binding, 'recovery_original_records_changed')
        intent_path = root / '.paper3-hpc/recoveries' / ('test-' + failed_test_job + '.intent.json')
        pinned_intent_sha = _hash(intent_path) if intent_path.exists() else None
        intent_witness = _load_intent(binding, user)
        need(pinned_intent_sha is None or _hash(intent_path) == pinned_intent_sha,
             'recovery_original_cancel_intent_changed')
        before = accounting(binding, runner=runner, username=user, intent=intent_witness)
        controls = {}
        for stage in ('prepare', 'train'):
            if before[binding['jobs'][stage]['job_id']]['state'] == 'PENDING':
                controls[stage] = _control(binding, stage, runner=runner, username=user)
        result = {'schema_version': SCHEMA, 'status': 'PLAN_CANCEL_ONLY_NEVER_STARTED_DEPENDENTS',
                  'project': str(root), 'failed_test_job': failed_test_job, 'binding': binding,
                  'uid': os.getuid(), 'user': user, 'accounting_before': before,
                  'live_pending_before': controls, 'scientific_fit_performed': False,
                  'scientific_budget_changed': False, 'new_jobs_submitted': False}
        if not acting:
            return result
        destination = root / '.paper3-hpc/recoveries' / ('test-' + failed_test_job + '.json')
        if destination.exists():
            checked = superseded_attempts(root)
            need(binding['attempt'] in checked, 'recovery_existing_proof_requires_review')
            return {**start.read(destination), 'reused_existing_proof': True}
        if intent_path.exists():
            old = start.read(intent_path)
            need(old.get('binding') == binding and old.get('uid') == os.getuid()
                 and old.get('user') == user, 'recovery_existing_cancel_intent_mismatch')
        else:
            start.write(intent_path, {**result, 'status': 'PENDING_CANCEL_INTENT_PRESERVE_ORIGINALS',
                                      'created_at_utc': datetime.now(timezone.utc).isoformat()})
        if pinned_intent_sha is None:
            pinned_intent_sha = _hash(intent_path)
        intent_witness = _load_intent(binding, user)
        need(_hash(intent_path) == pinned_intent_sha, 'recovery_original_cancel_intent_changed')
        for stage in ('train', 'prepare'):
            if stage in controls:
                # Recheck immediately, then use Slurm's state filter to refuse a
                # newly RUNNING job even if it races this last observation.
                need(_hash(intent_path) == pinned_intent_sha, 'recovery_original_cancel_intent_changed')
                _control(binding, stage, runner=runner, username=user)
                _run(['scancel', '--state=PENDING', '--user=' + str(os.getuid()),
                      binding['jobs'][stage]['job_id']], runner)
        after = None
        for retry in range(4):
            after = accounting(binding, runner=runner, username=user, intent=intent_witness)
            if all(after[binding['jobs'][s]['job_id']]['state'] == 'CANCELLED' for s in ('prepare', 'train')):
                break
            if retry < 3:
                sleeper(1)
        need(all(after[binding['jobs'][s]['job_id']]['state'] == 'CANCELLED' for s in ('prepare', 'train')),
             'recovery_cancellation_unconfirmed_preserve_intent')
        ids = {binding['jobs'][s]['job_id'] for s in start.STAGES}
        queued = _run(['squeue', '--noheader', '--array', '--user=' + str(os.getuid()),
                       '--format=%i|%T'], runner)
        for line in queued.splitlines():
            if not line.strip():
                continue
            cells = [c.strip() for c in line.split('|')]
            need(len(cells) == 2 and start.JOB_ID.fullmatch(cells[0]) and cells[1],
                 'recovery_queue_record_unrecognized')
            need(cells[0] not in ids, 'recovery_original_jobs_still_queued')
        need(_bound(root, failed_test_job) == binding, 'recovery_original_records_changed')
        need(_hash(intent_path) == pinned_intent_sha, 'recovery_original_cancel_intent_changed')
        result.update(status='RECOVERY_VERIFIED_NO_FIT_REPLAY', accounting_after=after, queue_after=[],
                      intent=str(intent_path.relative_to(root)), intent_sha256=pinned_intent_sha,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        start.write(destination, result)
        superseded_attempts(root)
        return {**result, 'receipt': str(destination.relative_to(root))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--failed-test-job', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--diagnose', action='store_true',
                        help='Read-only original binding/accounting/live-control diagnostic; never cancel or submit')
    args = parser.parse_args(argv)
    try:
        result = (diagnose(args.root, args.failed_test_job) if args.diagnose else
                  recover(args.root, args.failed_test_job, execute=args.execute, dry_run=args.dry_run))
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    except (RecoveryError, start.StartError, StageError, OSError, ValueError, TypeError, KeyError) as exc:
        code = str(exc) if isinstance(exc, (RecoveryError, start.StartError, StageError)) else type(exc).__name__
        print(json.dumps({'status': 'RECOVERY_BLOCKED_PRESERVE_ORIGINAL_JOBS_AND_CLAIMS',
                          'reason': code, 'diagnostic': getattr(exc, 'diagnostic', None),
                          'scientific_fit_performed': False}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
