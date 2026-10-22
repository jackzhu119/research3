#!/usr/bin/env python3
"""Explicitly retest and finalize a completed pilot without dispatching it again.

This submission-host adapter binds the failed two-job prefit recovery and its
original three-job chain.  It never runs a fit, captures dependencies, removes
a claim, or declares the pilot complete.  Allocated GPU code owns independent
evidence verification and metadata finalization before the original campaign.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpc import resume_prefit_train as prefit, start, submit
from hpc.environment import verify_code
from hpc.run_stage import owner_lock

SCHEMA = 'paper3-hpc-completed-pilot-resume-v1'
DIRECTORY = '.paper3-hpc/completed-pilot-resumes'
STAGES = ('test', 'train')
FIELDS = (*prefit.FIELDS, 'node_list')
FORMAT = prefit.FORMAT + ',NodeList%4096'
# These two files are part of the genuine school checkpoint identity at 71ce.
# A deployment fix must not invalidate that identity to bypass its verifier.
FROZEN_CODE = {
    'tools/run_controlled_pilot.py': '95058f93b9ccaaa3143116b227eabef4fe046f924225c2ed274c1a80d09535a3',
    'p3work/source_training.py': '149bae9b9caf371b170e061f0975a95f2c1e2de69efe17c5d5ac972cc1d7567e',
}


class ResumeError(ValueError):
    pass


def need(value, code):
    if not value:
        raise ResumeError(code)


def proof_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source_identity(value):
    return (isinstance(value, dict) and
        re.fullmatch('[a-f0-9]{40}', value.get('source_commit', '')) is not None and
        re.fullmatch('[a-f0-9]{64}', value.get('code_manifest_sha256', '')) is not None)


def frozen_code(root):
    rows = {name: prefit.descriptor(root, root / name) for name in FROZEN_CODE}
    need(all(rows[name]['sha256'] == sha for name, sha in FROZEN_CODE.items()),
         'completed_pilot_frozen_checkpoint_code_changed')
    return rows


def _data_identity(value):
    need(isinstance(value, dict) and value.get('status') == 'ready'
         and type(value.get('frozen_data_files')) is int and value['frozen_data_files'] == 21262
         and type(value.get('locally_SHA_verified')) is int and value['locally_SHA_verified'] == 21262
         and value.get('missing_files') == [] and type(value.get('missing_cache_files')) is int
         and value['missing_cache_files'] == 0, 'completed_pilot_full_frozen_data_required')
    refs = {key: value.get(key) for key in ('closure_sha256', 'index_sha256')}
    need(all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v) for v in refs.values()),
         'completed_pilot_frozen_data_identity_required')
    return refs


def predecessor_binding(root, failed_job, user):
    """Find the actual acknowledged prefit two-job chain, never invent its CPU ID."""
    need(isinstance(failed_job, str) and re.fullmatch(r'\d{1,20}', failed_job),
         'completed_pilot_failed_train_job_id_required')
    candidates = []
    for path, value in start.records(root, '.paper3-hpc/prefit-train-resumes', r'train-\d{1,20}\.json'):
        intents = value.get('intents', [])
        if isinstance(intents, list) and any(isinstance(i, dict) and
                i.get('stage') == 'train' and i.get('receipt', {}).get('job_id') == failed_job
                for i in intents):
            candidates.append((path, value))
    # An acknowledgement crash can leave the actual train only in its durable
    # API receipt. Resolve that case through the exact old intent, not a name.
    if not candidates:
        for path, value in start.records(root, '.paper3-hpc/prefit-train-resumes', r'train-\d{1,20}\.json'):
            if not isinstance(value.get('intents'), list) or len(value['intents']) != 2:
                continue
            bound = prefit.reconciled(root, deepcopy(value))
            if len(bound) == 2 and bound[1]['job_id'] == failed_job:
                candidates.append((path, value))
    need(len(candidates) == 1, 'completed_pilot_exact_predecessor_resume_required')
    path, value = candidates[0]
    need(value.get('project') == str(root) and value.get('uid') == os.getuid()
         and value.get('user') == user and _source_identity(value.get('source_identity'))
         and value.get('scientific_completion_claimed') is False
         and value.get('scientific_budget_changed') is False
         and value.get('status') in {'SUBMITTING_TRAIN', 'SLURM_PREFIT_RETEST_TRAIN_CHAIN_SUBMITTED_NOT_COMPLETE'}
         and path.name == 'train-' + str(value.get('failed_train_job')) + '.json',
         'completed_pilot_predecessor_owner_or_state_invalid')
    rows = prefit.reconciled(root, deepcopy(value))
    need(len(rows) == 2 and [r['stage'] for r in rows] == list(STAGES)
         and rows[1]['job_id'] == failed_job, 'completed_pilot_predecessor_two_actual_jobs_required')
    ancestor = prefit.original_binding(root, value.get('failed_train_job'))
    need(value.get('original_binding') == ancestor, 'completed_pilot_ancestor_chain_changed')
    historical = value.get('prefit_inspection', {})
    need(historical.get('status') == 'KNOWN_ZERO_UPDATE_PREFIT_FAILURE_SAME_CLAIM_ONLY'
         and type(historical.get('scientific_updates_performed')) is int
         and historical['scientific_updates_performed'] == 0
         and historical.get('new_pilot_claim_authorized') is False,
         'completed_pilot_original_zero_update_proof_required')
    # The genuine runner embeds this allocated-job proof in its receipt. It
    # links the completed work to the selected failed train, rather than just
    # assuming that a global pilot belongs to the most recent failed chain.
    from hpc.pilot_prefit import PILOT, SCHEMA as PREFIT_SCHEMA
    proof_path = root / '.paper3-hpc/jobs' / failed_job / 'pilot_prefit_resume.json'
    job_proof = start.read(proof_path)
    proof_ref = prefit.descriptor(root, proof_path)
    receipt = start.read(root / PILOT / 'engineering_receipt.json')
    need(job_proof.get('schema_version') == PREFIT_SCHEMA
         and job_proof.get('status') == 'PREFIT_SAME_CLAIM_RESUME_PREPARED'
         and job_proof.get('project') == str(root) and job_proof.get('job_id') == failed_job
         and job_proof.get('facts') == historical
         and receipt.get('prefit_resume', {}).get('proof') == proof_ref,
         'completed_pilot_selected_failed_job_proof_required')
    files = [prefit.descriptor(root, path), proof_ref]
    saved_files = start.records(root, '.paper3-hpc/submissions', r'\d+-[a-f0-9]+\.json')
    for stage, row in zip(STAGES, rows):
        matches = [p for p, candidate in saved_files if candidate == row]
        need(len(matches) == 1, 'completed_pilot_predecessor_durable_file_ambiguous')
        terminal_path = root / '.paper3-hpc/jobs' / row['job_id'] / ('stage-' + stage + '.json')
        terminal = start.read(terminal_path)
        need(terminal.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
             and terminal.get('job_id') == row['job_id'] and terminal.get('stage') == stage
             and type(terminal.get('returncode')) is int
             and terminal['returncode'] == (2 if stage == 'train' else 0)
             and terminal.get('scientific_completion_claimed') is False
             and terminal.get('external_query_accessed') is False,
             'completed_pilot_predecessor_terminal_receipt_required')
        if stage == 'test':
            tested = terminal.get('result', {})
            need(tested.get('status') == 'ALL_SOFTWARE_TESTS_PASSED'
                 and type(tested.get('returncode')) is int and tested['returncode'] == 0
                 and tested.get('software_identity', {}).get('manifest_sha256') ==
                     value['source_identity']['code_manifest_sha256'],
                 'completed_pilot_predecessor_CPU_identity_required')
        else:
            refs = _data_identity(terminal.get('frozen_data_verification'))
            original_data = start.read(root / '.paper3-hpc/jobs' / ancestor['jobs']['prepare'] / 'stage-prepare.json')
            need(refs == _data_identity(original_data.get('frozen_data_verification') or original_data.get('result')),
                 'completed_pilot_original_frozen_data_changed')
        files.extend((prefit.descriptor(root, matches[0]), prefit.descriptor(root, terminal_path)))
    log = root / '.paper3-hpc/jobs' / failed_job / 'campaign.log'
    failure = start.read(log)
    need(failure.get('status') == 'BLOCKED_PRESERVE_EXISTING_JOBS'
         and failure.get('error_type') == 'CampaignError' and failure.get('reason') == prefit.PILOT_REASON
         and failure.get('completion_claimed') is False and failure.get('external_query_accessed') is False,
         'completed_pilot_exact_failed_campaign_gate_required')
    files.append(prefit.descriptor(root, log))
    return {'project': str(root), 'jobs': {r['stage']: r['job_id'] for r in rows},
            'account': ancestor['account'], 'source_identity': value['source_identity'],
            'ancestor': ancestor, 'frozen_data_identity': refs, 'files': files}


def accounting(root, binding, runner, user):
    command = ['sacct', '--duplicates', '--allocations', '--jobs=' + ','.join(binding['jobs'].values()),
               '--noheader', '--parsable2', '--format=' + FORMAT]
    try:
        result = runner(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        raise ResumeError('completed_pilot_predecessor_accounting_unavailable') from None
    need(result.returncode == 0 and isinstance(result.stdout, str) and len(result.stdout) <= 1024**2,
         'completed_pilot_predecessor_accounting_unavailable')
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        cells = [v.strip() for v in line.split('|')]
        need(len(cells) == len(FIELDS), 'completed_pilot_accounting_fields_required')
        row = dict(zip(FIELDS, cells)); job = row['job_id']
        need(job in binding['jobs'].values() and job not in rows, 'completed_pilot_accounting_history_ambiguous')
        stage = next(s for s, j in binding['jobs'].items() if j == job)
        need(row['account'] == binding['account'] and row['user'] == user
             and row['workdir'] == str(root) and row['name'] == 'paper3-' + stage,
             'completed_pilot_accounting_identity_mismatch')
        state = row['state'].split()[0].rstrip('+') if row['state'].strip() else ''
        need((state, row['exit_code']) == (('FAILED', '2:0') if stage == 'train' else ('COMPLETED', '0:0')),
             'completed_pilot_scheduler_terminal_outcome_required')
        need(re.fullmatch(r'[1-9]\d{0,19}', row['db_index']) and re.fullmatch(r'\d{1,20}', row['elapsed']),
             'completed_pilot_database_identity_required')
        try:
            clocks = [datetime.fromisoformat(row[k]) for k in ('submit', 'start', 'end')]
            ordered = clocks[0] <= clocks[1] <= clocks[2]
        except (ValueError, TypeError):
            ordered = False
        need(ordered, 'completed_pilot_accounting_time_invalid')
        if stage == 'train':
            need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', row['node_list']),
                 'completed_pilot_original_single_GPU_node_required')
        rows[job] = row
    need(set(rows) == set(binding['jobs'].values()) and len({r['db_index'] for r in rows.values()}) == 2,
         'completed_pilot_complete_predecessor_accounting_required')
    return {'rows': [rows[binding['jobs'][s]] for s in STAGES],
            'train_node': rows[binding['jobs']['train']]['node_list']}


def _inspection(root, inspector):
    if inspector is None:
        from hpc.pilot_finalize import inspect_completed_pilot
        inspector = inspect_completed_pilot
    value = inspector(root)
    need(isinstance(value, dict) and value.get('status') == 'COMPLETED_200_PILOT_METADATA_FINALIZATION_REQUIRED'
         and value.get('project') == str(root) and type(value.get('actual_completed_updates')) is int
         and value['actual_completed_updates'] == 200 and value.get('exact_resume_claimed') is True
         and value.get('numerical_evidence_not_yet_independently_verified') is True
         and value.get('code_files') == frozen_code(root), 'completed_pilot_strict_finalization_proof_required')
    return value


def requests(args, root, config, account=None, dependency=None, node=None):
    rows = prefit.requests(args, root, config, account=account, dependency=dependency)
    if node:
        need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,252}', node),
             'completed_pilot_original_single_GPU_node_required')
        train = rows[1]
        for key in ('command', 'scheduler_preflight'):
            train[key].insert(2, '--nodelist=' + node)
        train['resources']['node_list'] = node
    return rows


def reconciled(root, value):
    """Only actual durable API receipts resolve a new mode-specific intent."""
    need(value.get('schema_version') == SCHEMA and value.get('project') == str(root)
         and isinstance(value.get('intents'), list) and len(value['intents']) <= 2
         and [i.get('stage') for i in value['intents']] == list(STAGES[:len(value['intents'])])
         and _source_identity(value.get('source_identity')),
         'completed_pilot_resume_attempt_invalid')
    evidence = value.get('completed_pilot_inspection')
    need(isinstance(evidence, dict) and value.get('proof_sha256') == proof_sha256(evidence),
         'completed_pilot_stored_proof_changed')
    saved, confirmed = start.submissions(root), []
    for intent in value['intents']:
        request = intent.get('request', {})
        need(isinstance(request, dict) and request.get('completed_pilot_resume_attempt_id') == value.get('attempt_id')
             and request.get('completed_pilot_resume_failed_train_job') == value.get('failed_train_job')
             and request.get('completed_pilot_resume_proof_sha256') == value['proof_sha256']
             and request.get('start_source_identity') == value['source_identity']
             and request.get('stage') == intent['stage'] and request.get('project') == str(root),
             'completed_pilot_intent_binding_invalid')
        commands = request.get('command', [])
        need(isinstance(commands, list) and all(isinstance(c, str) for c in commands)
             and '--job-name=paper3-' + intent['stage'] in commands and '--chdir=' + str(root) in commands
             and [c for c in commands if c.startswith('--dependency=')] ==
                 ([] if intent['stage'] == 'test' else ['--dependency=afterok:' + confirmed[0]['job_id']])
             and [c for c in commands if c.startswith('--nodelist=')] ==
                 ([] if intent['stage'] == 'test' else ['--nodelist=' + value['predecessor_accounting']['train_node']])
             and request.get('resources', {}).get('account') == value['predecessor_binding']['account']
             and request.get('scientific_budget_changed') is False
             and request.get('scientific_threads_changed') is False,
             'completed_pilot_request_or_dependency_changed')
        matches = [row for row in saved if all(row.get(k) == v for k, v in request.items())]
        need(len(matches) == 1, 'completed_pilot_unknown_submission_outcome_no_duplicate_authorized')
        receipt = matches[0]
        if intent.get('state') == 'SUBMISSION_RECEIPT_VERIFIED':
            need(intent.get('receipt') == receipt, 'completed_pilot_durable_submission_changed')
        else:
            need(intent.get('state') == 'SUBMISSION_INTENT_OUTCOME_UNKNOWN', 'completed_pilot_intent_state_invalid')
            intent.update(state='SUBMISSION_RECEIPT_VERIFIED', receipt=receipt)
        confirmed.append(receipt)
    need(len({r['job_id'] for r in confirmed}) == len(confirmed), 'completed_pilot_resume_job_ids_ambiguous')
    return confirmed


def allocated_resume_proof(root, job_id=None, *, inspector=None, verified_identity=None):
    """Bind an opted-in allocated train; ordinary untagged jobs return None."""
    root = start.safe(root)
    current = os.environ.get('SLURM_JOB_ID')
    job_id = current if job_id is None else job_id
    need(isinstance(current, str) and current == job_id and re.fullmatch(r'\d{1,20}', job_id),
         'completed_pilot_allocated_job_binding_required')
    current_receipts = [row for row in start.submissions(root) if row['job_id'] == job_id]
    tagged = [row for row in current_receipts if any(key.startswith('completed_pilot_resume_') for key in row)]
    if not tagged:
        return None
    need(len(current_receipts) == len(tagged) == 1, 'completed_pilot_allocated_submission_ambiguous')
    matches = []
    for path, value in start.records(root, DIRECTORY, r'train-\d{1,20}\.json'):
        rows = reconciled(root, deepcopy(value))
        if any(r['stage'] == 'train' and r['job_id'] == job_id for r in rows):
            matches.append((path, value, rows))
    need(len(matches) == 1, 'completed_pilot_allocated_durable_resume_required')
    path, value, rows = matches[0]
    need(value.get('uid') == os.getuid() and value.get('user') == pwd.getpwuid(os.getuid()).pw_name
         and len(rows) == 2 and rows[1]['job_id'] == job_id and rows[1] == tagged[0],
         'completed_pilot_allocated_actor_or_chain_changed')
    need(os.environ.get('SLURM_JOB_NODELIST') == value['predecessor_accounting']['train_node'],
         'completed_pilot_allocated_original_GPU_node_required')
    checked = verify_code(root) if verified_identity is None else verified_identity
    need(isinstance(checked, dict) and checked.get('manifest_sha256') ==
         value['source_identity']['code_manifest_sha256'], 'completed_pilot_allocated_current_code_identity_changed')
    evidence = _inspection(root, inspector)
    need(evidence == value['completed_pilot_inspection'], 'completed_pilot_allocated_evidence_changed')
    tested = start.read(root / '.paper3-hpc/jobs' / rows[0]['job_id'] / 'stage-test.json')
    need(tested.get('schema_version') == 'paper3-hpc-stage-receipt-v1'
         and tested.get('stage') == 'test' and tested.get('job_id') == rows[0]['job_id']
         and type(tested.get('returncode')) is int and tested['returncode'] == 0
         and tested.get('result', {}).get('status') == 'ALL_SOFTWARE_TESTS_PASSED'
         and tested.get('result', {}).get('software_identity', {}).get('manifest_sha256') ==
             value['source_identity']['code_manifest_sha256'], 'completed_pilot_allocated_CPU_identity_required')
    return {'schema_version': SCHEMA, 'attempt': prefit.descriptor(root, path),
            'failed_train_job': value['failed_train_job'], 'job_id': job_id,
            'source_identity': value['source_identity'], 'proof_sha256': value['proof_sha256'],
            'inspection': evidence, 'prior_GPU_node': value['predecessor_accounting']['train_node'],
            'numerical_updates_authorized': False, 'scientific_completion_claimed': False}


def resume(args, *, runner=subprocess.run, verifier=verify_code, inspector=None, submitter=submit.submit):
    root = start.safe(args.root)
    need(root.is_dir() and not os.environ.get('SLURM_JOB_ID'), 'completed_pilot_submission_host_required')
    need(re.fullmatch(r'\d{1,20}', args.failed_train_job), 'completed_pilot_failed_train_job_id_required')
    config = submit.load_config(root / 'hpc/resources.json')
    planned = requests(args, root, config)
    if not args.execute or args.dry_run:
        return {'schema_version': SCHEMA, 'status': 'PLAN_COMPLETED_PILOT_RETEST_TRAIN_NO_ACTION',
                'stages': [{'stage': r['stage'], 'request': r,
                            'dependency_on_stage': 'test' if r['stage'] == 'train' else None} for r in planned],
                'completed_pilot_evidence_not_yet_validated': True, 'new_jobs_submitted': False,
                'scientific_completion_claimed': False}
    user = pwd.getpwuid(os.getuid()).pw_name
    binding = predecessor_binding(root, args.failed_train_job, user)
    destination = root / DIRECTORY / ('train-' + args.failed_train_job + '.json')

    def prior_attempt():
        value = start.read(destination) if destination.exists() else None
        if value is None:
            return None, []
        need(value.get('predecessor_binding') == binding and value.get('uid') == os.getuid()
             and value.get('user') == user, 'completed_pilot_existing_resume_binding_changed')
        return value, reconciled(root, value)

    def existing_result(rows):
        return {'status': 'EXISTING_COMPLETED_PILOT_RESUME_NO_RESUBMISSION',
                'job_ids': {r['stage']: r['job_id'] for r in rows}, 'attempt': str(destination.relative_to(root)),
                'new_jobs_submitted': False, 'scientific_completion_claimed': False}

    _, existing = prior_attempt()
    if len(existing) == 2:
        return existing_result(existing)
    with start.owner(root), start.staged_owner(root), ExitStack() as validation:
        validation.enter_context(owner_lock(root))
        need(predecessor_binding(root, args.failed_train_job, user) == binding,
             'completed_pilot_original_records_changed')
        value, confirmed = prior_attempt()
        if len(confirmed) == 2:
            return existing_result(confirmed)
        identity = start.source_identity(root, runner=runner, verifier=verifier)
        if value is not None:
            need(value.get('source_identity') == identity, 'completed_pilot_resume_source_changed_preserve_jobs')
        history = accounting(root, binding, runner, user)
        evidence = _inspection(root, inspector)
        if value is not None:
            need(value.get('completed_pilot_inspection') == evidence and value.get('predecessor_accounting') == history,
                 'completed_pilot_resume_evidence_changed')
        allowed = {r['job_id'] for r in confirmed}
        queued = start.active_jobs(root, start.submissions(root), runner=runner)
        need(all(row['job_id'] in allowed for row in queued), 'completed_pilot_active_project_job_preserved')
        if confirmed:
            cpu_path = root / '.paper3-hpc/jobs' / confirmed[0]['job_id'] / 'stage-test.json'
            if cpu_path.exists():
                cpu = start.read(cpu_path)
                need(cpu.get('schema_version') == 'paper3-hpc-stage-receipt-v1' and cpu.get('stage') == 'test'
                     and cpu.get('job_id') == confirmed[0]['job_id'] and type(cpu.get('returncode')) is int
                     and cpu['returncode'] == 0 and cpu.get('result', {}).get('status') == 'ALL_SOFTWARE_TESTS_PASSED'
                     and cpu.get('result', {}).get('software_identity', {}).get('manifest_sha256') ==
                         identity['code_manifest_sha256'], 'completed_pilot_new_CPU_test_failed_no_train_submission')
        if value is None:
            value = {'schema_version': SCHEMA, 'project': str(root), 'failed_train_job': args.failed_train_job,
                     'attempt_id': uuid.uuid4().hex, 'uid': os.getuid(), 'user': user,
                     'predecessor_binding': binding, 'predecessor_accounting': history,
                     'source_identity': identity, 'completed_pilot_inspection': evidence,
                     'proof_sha256': proof_sha256(evidence), 'intents': [], 'status': 'VALIDATED_BEFORE_SUBMISSION',
                     'scientific_completion_claimed': False, 'scientific_budget_changed': False}
            start.write(destination, value)
        # Retain durable submission owners, but let an immediately scheduled
        # CPU test acquire the normal deployment controller lock.
        validation.close()
        previous = confirmed[0]['job_id'] if confirmed else None
        for stage in STAGES[len(confirmed):]:
            request = next(r for r in requests(args, root, config, account=binding['account'],
                dependency='afterok:' + previous if previous else None, node=history['train_node']) if r['stage'] == stage)
            request.update(start_source_identity=identity, completed_pilot_resume_attempt_id=value['attempt_id'],
                completed_pilot_resume_failed_train_job=args.failed_train_job,
                completed_pilot_resume_proof_sha256=value['proof_sha256'])
            need(start.source_identity(root, runner=runner, verifier=verifier) == identity,
                 'completed_pilot_source_changed_before_submission')
            frozen_code(root)
            intent = {'stage': stage, 'state': 'SUBMISSION_INTENT_OUTCOME_UNKNOWN', 'request': request}
            value['intents'].append(intent); value['status'] = 'SUBMITTING_' + stage.upper()
            start.write(destination, value)
            receipt = submitter(request)
            need(receipt.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
                 and re.fullmatch(r'\d{1,20}', str(receipt.get('job_id', '')))
                 and all(receipt.get(k) == v for k, v in request.items()),
                 'completed_pilot_returned_submission_unverified')
            saved = reconciled(root, value)
            need(saved[-1] == receipt, 'completed_pilot_returned_submission_not_durable')
            intent.update(state='SUBMISSION_RECEIPT_VERIFIED', receipt=receipt)
            previous = receipt['job_id']; confirmed = saved
            start.write(destination, value)
        value['status'] = 'SLURM_COMPLETED_PILOT_RETEST_TRAIN_CHAIN_SUBMITTED_NOT_COMPLETE'
        start.write(destination, value)
        return {'status': value['status'], 'attempt': str(destination.relative_to(root)),
                'job_ids': {r['stage']: r['job_id'] for r in confirmed},
                'prepare_reused_full_SHA_rechecked_by_train': True, 'prior_GPU_node': history['train_node'],
                'pilot_optimizer_updates_requested': 0, 'new_jobs_submitted': True,
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
        print(json.dumps(resume(args), indent=2, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        # All helper codes are finite lowercase identifiers; never expose raw
        # SDK failure messages, absolute dependency paths, or traceback payloads.
        helper_error = type(exc).__module__ == 'hpc.pilot_finalize' and type(exc).__name__ == 'FinalizeError'
        reason = str(exc) if helper_error or isinstance(exc, (ResumeError, prefit.ResumeError, start.StartError, submit.SubmissionError)) else type(exc).__name__
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,159}', reason):
            reason = type(exc).__name__
        print(json.dumps({'status': 'COMPLETED_PILOT_RESUME_BLOCKED_PRESERVE_ORIGINALS', 'reason': reason,
            'submission_outcome': 'INSPECT_DURABLE_INTENT_NO_AUTOMATIC_REPLAY',
            'scientific_completion_claimed': False}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
