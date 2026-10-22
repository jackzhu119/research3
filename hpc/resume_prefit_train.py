#!/usr/bin/env python3
"""Explicitly retest deployment and resume a verified zero-update failed pilot.

Submission-host metadata adapter only: never captures dependencies, cancels a
job, deletes a claim, or runs a fit. Actual data/pilot checks remain on GPU.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc import start, submit
from hpc.environment import verify_code
from hpc.pilot_prefit import PrefitError
from hpc.run_stage import owner_lock

SCHEMA = 'paper3-hpc-prefit-train-resume-v1'
STAGES = ('test', 'train')
PILOT_REASON = ('stage_blocked:SourceTrainingError:'
    'Actual completed 200-update engineering pilot with exact resume is required')
FIELDS = ('job_id', 'state', 'exit_code', 'start', 'end', 'elapsed', 'account', 'user',
          'name', 'workdir', 'db_index', 'submit')
FORMAT = ('JobIDRaw,State%64,ExitCode,Start,End,ElapsedRaw,Account%64,User%64,'
          'JobName%64,WorkDir%4096,DBIndex,Submit')


class ResumeError(ValueError):
    pass


def need(value, code):
    if not value:
        raise ResumeError(code)


def descriptor(root, path):
    path = start.safe(path)
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def original_binding(root, job):
    """Bind the original full chain and its real local terminal receipts."""
    need(isinstance(job, str) and re.fullmatch(r'\d{1,20}', job), 'prefit_failed_train_job_id_required')
    matches = [(p, v) for p, v in start.records(root, '.paper3-hpc/start', r'attempt-[a-f0-9]{32}\.json')
               if any(r.get('stage') == 'train' and r.get('job_id') == job
                      for r in v.get('submitted_jobs', []))]
    need(len(matches) == 1, 'prefit_exact_original_chain_required')
    path, value = matches[0]
    need(value.get('schema_version') == start.SCHEMA and value.get('project') == str(root)
         and value.get('status') == 'SLURM_CHAIN_SUBMITTED_NOT_TRAINING_COMPLETE',
         'prefit_complete_original_chain_required')
    rows, intents = value.get('submitted_jobs', []), value.get('submission_intents', [])
    need(len(rows) == len(intents) == 3 and [r.get('stage') for r in rows] == list(start.STAGES)
         and [r.get('stage') for r in intents] == list(start.STAGES)
         and all(r.get('state') == 'SUBMISSION_RECEIPT_VERIFIED' for r in intents),
         'prefit_original_submission_outcome_unknown')
    source = value.get('source_identity', {})
    need(isinstance(source, dict) and re.fullmatch('[a-f0-9]{40}', source.get('source_commit', ''))
         and re.fullmatch('[a-f0-9]{64}', source.get('code_manifest_sha256', '')),
         'prefit_original_source_identity_required')
    saved = start.submissions(root)
    bound, files, accounts, data_binding = {}, [descriptor(root, path)], set(), None
    for i, (stage, row) in enumerate(zip(start.STAGES, rows)):
        request = intents[i].get('request', {})
        need(isinstance(request, dict) and row.get('project') == str(root)
             and row.get('start_attempt_id') == value.get('attempt_id')
             and row.get('start_source_identity') == source
             and row.get('schema_version') == 'paper3-slurm-submission-v1'
             and row.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
             and re.fullmatch(r'\d{1,20}', str(row.get('job_id', '')))
             and all(row.get(k) == v for k, v in request.items()),
             'prefit_original_submission_binding_invalid')
        selected = [r for r in saved if r.get('job_id') == row['job_id']]
        need(len(selected) == 1 and selected[0] == row, 'prefit_original_durable_submission_required')
        paths = [p for p, r in start.records(root, '.paper3-hpc/submissions', r'\d+-[a-f0-9]+\.json') if r == row]
        need(len(paths) == 1, 'prefit_original_submission_file_ambiguous')
        command = row.get('command', [])
        need(isinstance(command, list) and all(isinstance(c, str) for c in command),
             'prefit_original_command_required')
        dependencies = [c for c in command if c.startswith('--dependency=')]
        need(dependencies == ([] if i == 0 else ['--dependency=afterok:' + rows[i-1]['job_id']])
             and '--job-name=paper3-' + stage in command and '--chdir=' + str(root) in command,
             'prefit_original_dependency_binding_invalid')
        terminal_path = root / '.paper3-hpc/jobs' / row['job_id'] / ('stage-' + stage + '.json')
        terminal = start.read(terminal_path)
        expected = 2 if stage == 'train' else 0
        need(terminal.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
             and terminal.get('job_id') == row['job_id'] and terminal.get('stage') == stage
             and type(terminal.get('returncode')) is int and terminal['returncode'] == expected
             and terminal.get('scientific_completion_claimed') is False
             and terminal.get('external_query_accessed') is False,
             'prefit_original_stage_outcome_required')
        if stage == 'test':
            tested = terminal.get('result', {})
            need(tested.get('status') == 'ALL_SOFTWARE_TESTS_PASSED'
                 and tested.get('returncode') == 0
                 and tested.get('software_identity', {}).get('manifest_sha256') == source['code_manifest_sha256'],
                 'prefit_original_successful_test_identity_required')
        if stage in {'prepare', 'train'}:
            data = terminal.get('frozen_data_verification') or terminal.get('result', {})
            need(data.get('status') == 'ready' and data.get('frozen_data_files') == 21262
                 and data.get('locally_SHA_verified') == 21262 and data.get('missing_files') == []
                 and data.get('missing_cache_files') == 0,
                 'prefit_original_full_data_verification_required')
            refs = {key: data.get(key) for key in ('closure_sha256', 'index_sha256')}
            need(all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v) for v in refs.values()),
                 'prefit_original_frozen_data_identity_required')
            need(data_binding is None or refs == data_binding, 'prefit_original_data_identity_changed')
            data_binding = refs
        bound[stage] = row['job_id']; accounts.add(row.get('resources', {}).get('account'))
        files.extend((descriptor(root, paths[0]), descriptor(root, terminal_path)))
    need(bound['train'] == job and len(set(bound.values())) == 3 and len(accounts) == 1
         and re.fullmatch(r'[A-Za-z0-9_-]+', next(iter(accounts)) or ''),
         'prefit_original_account_or_job_identity_invalid')
    log = root / '.paper3-hpc/jobs' / job / 'campaign.log'
    failure = start.read(log)
    need(failure.get('status') == 'BLOCKED_PRESERVE_EXISTING_JOBS'
         and failure.get('error_type') == 'CampaignError' and failure.get('reason') == PILOT_REASON
         and failure.get('completion_claimed') is False and failure.get('external_query_accessed') is False,
         'prefit_exact_failed_pilot_gate_required')
    files.append(descriptor(root, log))
    return {'project': str(root), 'jobs': bound, 'account': next(iter(accounts)),
            'source_identity': source, 'files': files}


def accounting(root, binding, runner, user):
    command = ['sacct', '--duplicates', '--allocations', '--jobs=' + ','.join(binding['jobs'].values()),
               '--noheader', '--parsable2', '--format=' + FORMAT]
    try:
        result = runner(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        raise ResumeError('prefit_original_accounting_unavailable') from None
    need(result.returncode == 0 and isinstance(result.stdout, str) and len(result.stdout) <= 1024**2,
         'prefit_original_accounting_unavailable')
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        cells = [v.strip() for v in line.split('|')]
        need(len(cells) == len(FIELDS), 'prefit_accounting_fields_required')
        row = dict(zip(FIELDS, cells)); job = row['job_id']
        need(job in binding['jobs'].values() and job not in rows, 'prefit_ambiguous_accounting_history')
        stage = next(s for s, j in binding['jobs'].items() if j == job)
        need(row['account'] == binding['account'] and row['user'] == user
             and row['workdir'] == str(root) and row['name'] == 'paper3-' + stage,
             'prefit_accounting_original_identity_mismatch')
        state = row['state'].split()[0].rstrip('+') if row['state'].strip() else ''
        need((state, row['exit_code']) == (('FAILED', '2:0') if stage == 'train' else ('COMPLETED', '0:0')),
             'prefit_original_scheduler_terminal_outcome_required')
        need(re.fullmatch(r'[1-9]\d{0,19}', row['db_index']) and re.fullmatch(r'\d{1,20}', row['elapsed']),
             'prefit_original_database_identity_required')
        try:
            clocks = [datetime.fromisoformat(row[k]) for k in ('submit', 'start', 'end')]
            ordered = clocks[0] <= clocks[1] <= clocks[2]
        except (ValueError, TypeError):
            ordered = False
        need(ordered, 'prefit_original_accounting_time_invalid')
        rows[job] = row
    need(set(rows) == set(binding['jobs'].values()) and len({r['db_index'] for r in rows.values()}) == 3,
         'prefit_complete_original_accounting_required')
    return {'rows': [rows[binding['jobs'][s]] for s in start.STAGES]}


def requests(args, root, config, account=None, dependency=None):
    rows = []
    for stage in STAGES:
        train = stage == 'train'
        options = SimpleNamespace(stage=stage, root=root, python=args.python or config['python'],
            account=account, partition=None, dependency=dependency if train else None,
            cpus=args.cpus if train else None, mem_gib=args.mem_gib if train else None,
            gpus=args.gpus if train else None, time=args.time if train else None)
        rows.append(submit.build_request(options, config))
    return rows


def reconciled(root, value):
    """Only an actual matching durable submission receipt can resolve intent."""
    need(value.get('schema_version') == SCHEMA and value.get('project') == str(root)
         and isinstance(value.get('intents'), list) and len(value['intents']) <= 2
         and [i.get('stage') for i in value['intents']] == list(STAGES[:len(value['intents'])]),
         'prefit_resume_attempt_invalid')
    saved = start.submissions(root)
    confirmed = []
    for intent in value['intents']:
        request = intent.get('request', {})
        need(isinstance(request, dict) and request.get('prefit_resume_attempt_id') == value.get('attempt_id')
             and request.get('prefit_resume_failed_train_job') == value.get('failed_train_job')
             and request.get('start_source_identity') == value.get('source_identity')
             and request.get('stage') == intent.get('stage') and request.get('project') == str(root),
             'prefit_resume_intent_binding_invalid')
        commands = request.get('command', [])
        need(isinstance(commands, list) and all(isinstance(c, str) for c in commands)
             and '--job-name=paper3-' + intent['stage'] in commands
             and '--chdir=' + str(root) in commands
             and [c for c in commands if c.startswith('--dependency=')] ==
                 ([] if intent['stage'] == 'test' else ['--dependency=afterok:' + confirmed[0]['job_id']])
             and request.get('resources', {}).get('account') == value['original_binding']['account']
             and request.get('scientific_budget_changed') is False
             and request.get('scientific_threads_changed') is False,
             'prefit_resume_request_or_dependency_changed')
        matches = [r for r in saved if all(r.get(k) == v for k, v in request.items())]
        need(len(matches) == 1, 'prefit_unknown_submission_outcome_no_duplicate_authorized')
        receipt = matches[0]
        if intent.get('state') == 'SUBMISSION_RECEIPT_VERIFIED':
            need(intent.get('receipt') == receipt, 'prefit_durable_submission_receipt_changed')
        else:
            need(intent.get('state') == 'SUBMISSION_INTENT_OUTCOME_UNKNOWN', 'prefit_resume_intent_state_invalid')
            intent.update(state='SUBMISSION_RECEIPT_VERIFIED', receipt=receipt)
        confirmed.append(receipt)
    need(len({r['job_id'] for r in confirmed}) == len(confirmed), 'prefit_resume_job_ids_ambiguous')
    return confirmed


def resume(args, *, runner=subprocess.run, verifier=verify_code, inspector=None, submitter=submit.submit):
    root = start.safe(args.root)
    need(root.is_dir() and not os.environ.get('SLURM_JOB_ID'), 'prefit_submission_host_required')
    need(re.fullmatch(r'\d{1,20}', args.failed_train_job), 'prefit_failed_train_job_id_required')
    config = submit.load_config(root / 'hpc/resources.json')
    planned = requests(args, root, config)
    if not args.execute or args.dry_run:
        return {'schema_version': SCHEMA, 'status': 'PLAN_PREFIT_RETEST_THEN_TRAIN_NO_ACTION',
                'stages': [{'stage': r['stage'], 'request': r, 'dependency_on_stage': 'test' if r['stage']=='train' else None}
                           for r in planned], 'original_prefit_evidence_not_yet_validated': True,
                'new_jobs_submitted': False, 'scientific_completion_claimed': False}
    binding = original_binding(root, args.failed_train_job)
    destination = root / '.paper3-hpc/prefit-train-resumes' / ('train-' + args.failed_train_job + '.json')
    user = pwd.getpwuid(os.getuid()).pw_name

    def existing_result(confirmed):
        return {'status': 'EXISTING_PREFIT_RESUME_NO_RESUBMISSION',
                'job_ids': {r['stage']:r['job_id'] for r in confirmed},
                'attempt': str(destination.relative_to(root)), 'new_jobs_submitted': False,
                'scientific_completion_claimed': False}

    def prior_attempt():
        value = start.read(destination) if destination.exists() else None
        if value is None:
            return None, []
        need(value.get('original_binding') == binding and value.get('uid') == os.getuid()
             and value.get('user') == user, 'prefit_existing_resume_binding_changed')
        return value, reconciled(root, value)

    # A submitted test/train may already hold the controller lock. Returning
    # its two actual durable IDs is read-only and does not authorize more work.
    _, existing = prior_attempt()
    if len(existing) == 2:
        return existing_result(existing)
    with start.owner(root), start.staged_owner(root), ExitStack() as validation:
        validation.enter_context(owner_lock(root))
        need(original_binding(root, args.failed_train_job) == binding, 'prefit_original_records_changed')
        value, confirmed = prior_attempt()
        if value is not None:
            if len(confirmed) == 2:
                return existing_result(confirmed)
        identity = start.source_identity(root, runner=runner, verifier=verifier)
        if value is not None:
            need(value.get('source_identity') == identity, 'prefit_resume_source_changed_preserve_existing_jobs')
        history = accounting(root, binding, runner, user)
        if inspector is None:
            from hpc.pilot_prefit import inspect_failed_prefit
            inspector = inspect_failed_prefit
        evidence = inspector(root)
        need(isinstance(evidence, dict), 'prefit_readonly_pilot_evidence_required')
        if value is not None:
            need(value.get('prefit_inspection') == evidence, 'prefit_original_pilot_evidence_changed')
        allowed = {r['job_id'] for r in confirmed}
        queued = start.active_jobs(root, start.submissions(root), runner=runner)
        need(all(row['job_id'] in allowed for row in queued), 'prefit_existing_active_project_job_preserved')
        if confirmed:
            receipt_path = root / '.paper3-hpc/jobs' / confirmed[0]['job_id'] / 'stage-test.json'
            if receipt_path.exists():
                checked = start.read(receipt_path)
                need(checked.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
                     and checked.get('stage') == 'test' and checked.get('job_id') == confirmed[0]['job_id']
                     and type(checked.get('returncode')) is int and checked['returncode'] == 0
                     and checked.get('result', {}).get('software_identity', {}).get('manifest_sha256') == identity['code_manifest_sha256'],
                     'prefit_new_CPU_test_failed_no_train_submission')
        if value is None:
            value = {'schema_version': SCHEMA, 'project': str(root), 'failed_train_job': args.failed_train_job,
                     'attempt_id': uuid.uuid4().hex, 'uid': os.getuid(), 'user': user,
                     'original_binding': binding, 'source_identity': identity, 'original_accounting': history,
                     'prefit_inspection': evidence, 'intents': [], 'status': 'VALIDATED_BEFORE_SUBMISSION',
                     'scientific_completion_claimed': False, 'scientific_budget_changed': False}
            start.write(destination, value)
        # An immediately scheduled CPU test must be able to acquire its normal
        # controller lock. The submission owner still serializes durable intent.
        validation.close()
        previous = confirmed[0]['job_id'] if confirmed else None
        for stage in STAGES[len(confirmed):]:
            current = requests(args, root, config, account=binding['account'],
                               dependency='afterok:' + previous if previous else None)
            request = next(r for r in current if r['stage'] == stage)
            request.update(start_source_identity=identity, prefit_resume_attempt_id=value['attempt_id'],
                           prefit_resume_failed_train_job=args.failed_train_job)
            need(start.source_identity(root, runner=runner, verifier=verifier) == identity,
                 'prefit_source_changed_before_submission')
            intent = {'stage': stage, 'state': 'SUBMISSION_INTENT_OUTCOME_UNKNOWN', 'request': request}
            value['intents'].append(intent); value['status'] = 'SUBMITTING_' + stage.upper()
            start.write(destination, value)
            receipt = submitter(request)
            need(receipt.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
                 and re.fullmatch(r'\d{1,20}', str(receipt.get('job_id', '')))
                 and all(receipt.get(k) == v for k, v in request.items()),
                 'prefit_returned_submission_receipt_unverified')
            saved = reconciled(root, value)
            need(saved[-1] == receipt, 'prefit_returned_submission_not_durable')
            intent.update(state='SUBMISSION_RECEIPT_VERIFIED', receipt=receipt)
            previous = receipt['job_id']; confirmed = saved
            start.write(destination, value)
        value['status'] = 'SLURM_PREFIT_RETEST_TRAIN_CHAIN_SUBMITTED_NOT_COMPLETE'
        start.write(destination, value)
        return {'status': value['status'], 'attempt': str(destination.relative_to(root)),
                'job_ids': {r['stage']:r['job_id'] for r in confirmed},
                'prepare_reused_full_SHA_rechecked_by_train': True, 'new_jobs_submitted': True,
                'scientific_completion_claimed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--failed-train-job', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    for name in ('cpus', 'mem-gib', 'gpus'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--time')
    args = parser.parse_args(argv)
    try:
        result = resume(args)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        reason = str(exc) if isinstance(exc, (ResumeError, start.StartError, submit.SubmissionError, PrefitError)) else type(exc).__name__
        print(json.dumps({'status': 'PREFIT_RESUME_BLOCKED_PRESERVE_ORIGINALS', 'reason': reason,
                          'submission_outcome': 'INSPECT_DURABLE_INTENT_NO_AUTOMATIC_REPLAY',
                          'scientific_completion_claimed': False}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
