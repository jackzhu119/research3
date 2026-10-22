"""Resume the same claimed pilot after one proved, zero-update setup failure.

This deployment adapter never removes a claim, changes a ledger, performs a
fit, or declares a pilot complete.  Only an explicitly selected school GPU
allocation may prepare/consume its job-bound proof.  The original pilot still
owns all numerical work and the unchanged 1800-second/200-update limits.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

PILOT = 'runs/controlled_source_pilot_seed20261018'
CAMPAIGN = 'runs/training_campaign_v1'
SCHEMA = 'paper3-hpc-same-pilot-prefit-resume-v1'
FILES = {
    'receipt': PILOT + '/engineering_receipt.json',
    'pilot_claim': 'runs/controlled_pilot_job_claim.json',
    'campaign_claim': 'runs/training_campaign_v1_claim.json',
    'campaign_progress': CAMPAIGN + '/campaign_progress.json',
    'campaign_wall': CAMPAIGN + '/wall_budget.json',
}
OPTIONAL = {
    'artifact_manifest': PILOT + '/engineering_artifact_manifest.json',
    'backup_manifest': PILOT + '/engineering_backup_manifest.json',
}
ACTION = {'kind': 'pilot', 'key': 'pilot', 'expected_receipt': PILOT + '/engineering_receipt.json'}


class PrefitError(ValueError):
    pass


def _need(value, code):
    if not value:
        raise PrefitError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(path):
    path = Path(os.path.abspath(path))
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'prefit_symlink_forbidden')
    return path


def _root(root):
    root = _path(root)
    _need(root.is_dir(), 'prefit_existing_root_required')
    return root


def _read_checked(path):
    path = _path(path)
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_size <= 2 * 1024**2,
          'prefit_regular_bounded_JSON_required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno()); raw = stream.read(2 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    _need(signature(before) == signature(opened) == signature(after) == signature(path.stat())
          and len(raw) == before.st_size, 'prefit_original_changed_during_read')
    return raw


def _read(path):
    try:
        return _read_checked(path)
    except OSError:
        raise PrefitError('prefit_original_or_bound_proof_file_unavailable') from None


def _json(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError, UnicodeError):
        raise PrefitError('prefit_JSON_invalid') from None
    _need(isinstance(value, dict), 'prefit_JSON_object_required')
    return value


def _descriptor(root, relative, raw):
    return {'path': relative, 'bytes': len(raw), 'sha256': _sha(raw)}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        _need(parsed.tzinfo is not None and parsed.utcoffset() is not None,
              'prefit_aware_original_time_required')
        return parsed
    except (ValueError, TypeError, AttributeError):
        raise PrefitError('prefit_original_time_invalid') from None


def _output(root):
    output = _path(root / PILOT)
    _need(output.is_dir(), 'prefit_existing_pilot_output_required')
    allowed = {'engineering_receipt.json', 'engineering_artifact_manifest.json',
               'engineering_backup_manifest.json', 'prefit_history'}
    for path in output.iterdir():
        _path(path)
        _need(path.name in allowed, 'prefit_numerical_or_unknown_output_preserve_attempt')
        _need(path.is_dir() if path.name == 'prefit_history' else path.is_file(),
              'prefit_output_type_invalid')
        if path.name == 'prefit_history':
            names = {'original_' + name + '.json' for name in (*FILES, *OPTIONAL)}
            for folder in path.iterdir():
                _path(folder)
                _need(folder.is_dir() and re.fullmatch('[a-f0-9]{64}', folder.name),
                      'prefit_history_directory_invalid')
                for item in folder.iterdir():
                    _path(item)
                    _need(item.is_file() and item.name in names,
                          'prefit_history_numerical_or_unknown_output_preserve_attempt')
    return output


def _validate(root, raws):
    from p3work.contracts import digest
    from p3work.recovery_budget import PersistentWallBudget, WallBudgetError
    value = {name: _json(raw) for name, raw in raws.items()}
    receipt, claim = value['receipt'], value['pilot_claim']
    _need(receipt.get('status') == 'ENGINEERING_PILOT_FAILED'
          and receipt.get('scope') == 'ENGINEERING_SOURCE_PILOT_ONLY'
          and receipt.get('training_started') is False
          and receipt.get('exact_resume_verified') is False
          and receipt.get('user_started_gpu_attested') is True
          and type(receipt.get('seed')) is int and receipt['seed'] == 20261018
          and type(receipt.get('max_jobs')) is int and receipt['max_jobs'] == 1,
          'prefit_original_failed_pilot_identity_required')
    for key in ('actual_completed_updates', 'optimizer_updates_dispatched', 'prediction_replay_optimizer_updates'):
        _need(type(receipt.get(key)) is int and receipt[key] == 0,
              'prefit_any_or_unknown_update_count_preserve_attempt')
    _need(receipt.get('failure_type') == 'FileNotFoundError'
          and receipt.get('failure') == str(FileNotFoundError(2, 'No such file or directory',
                                                            str(root / 'audit/gpu_resolved_dependencies.json'))),
          'prefit_only_known_missing_dependency_inventory_is_recoverable')
    _need(receipt.get('checkpoint_files') == {} and receipt.get('checkpoint_io_timings') == []
          and not receipt.get('timing_segments')
          and not any(receipt.get(key) for key in ('identity', 'identity_sha256', 'model', 'optimizer',
                'resume_comparison', 'engineering_prediction_replay', 'resolved_dependencies')),
          'prefit_numerical_identity_checkpoint_or_timing_evidence_preserve_attempt')
    prior = receipt.get('total_monotonic_wall_seconds')
    _need(_number(prior) and prior < 1800, 'prefit_original_pilot_wall_exhausted_or_unknown')
    _need(_time(receipt.get('started_utc')) <= _time(receipt.get('ended_utc')),
          'prefit_original_pilot_time_order_invalid')
    _need(claim.get('scope') == 'ENGINEERING_SOURCE_PILOT_ONLY'
          and type(claim.get('max_jobs')) is int and claim['max_jobs'] == 1
          and claim.get('output') == PILOT and _time(claim.get('started_utc')) <= _time(receipt['started_utc']),
          'prefit_original_same_one_job_claim_required')
    parent, state, wall = value['campaign_claim'], value['campaign_progress'], value['campaign_wall']
    identity_sha = parent.get('identity_sha256')
    _need(parent.get('schema_version') == 'paper3-mainline-training-campaign-v1'
          and isinstance(parent.get('identity'), dict) and digest(parent['identity']) == identity_sha
          and parent.get('fixed_output') == CAMPAIGN
          and parent.get('claim_must_not_be_deleted_or_renamed_to_retry') is True,
          'prefit_original_campaign_claim_identity_invalid')
    process = state.get('active_process', {})
    _need(state.get('schema_version') == 'paper3-mainline-training-campaign-v1'
          and state.get('identity_sha256') == identity_sha and state.get('actions') == []
          and state.get('active') == ACTION and state.get('status') == 'INTERRUPTED_PRESERVE_SAME_JOB'
          and state.get('error_type') == 'CampaignError'
          and isinstance(process, dict) and process.get('cleanup_verified') is True
          and process.get('schema_version') == 'paper3-campaign-owned-stage-group-v1'
          and process.get('campaign_identity_sha256') == identity_sha and process.get('action_key') == 'pilot'
          and type(process.get('pid')) is int and process['pid'] > 1
          and process.get('pgid') == process['pid']
          and type(process.get('owner_pid')) is int and process['owner_pid'] > 0
          and process['owner_pid'] != process['pid']
          and type(process.get('pid_start_time_ticks')) is int and process['pid_start_time_ticks'] > 0
          and isinstance(process.get('boot_id'), str) and bool(process['boot_id'])
          and _number(process.get('absolute_deadline_monotonic')),
          'prefit_original_pilot_dispatch_cleanup_proof_required')
    checker = object.__new__(PersistentWallBudget); checker._state = wall
    try:
        checker._validate_state()
    except (WallBudgetError, KeyError, TypeError, ValueError):
        raise PrefitError('prefit_campaign_wall_journal_invalid_preserve_original') from None
    report = state.get('wall_budget', {})
    _need(wall.get('identity_sha256') == identity_sha and wall.get('interval') is None
          and wall.get('clock_block') is None and wall.get('initial_max_wall_seconds') == 43200
          and wall.get('max_wall_seconds') == 43200 and wall.get('amendments') == []
          and 0 < wall['durable_wall_seconds'] < wall['max_wall_seconds']
          and wall['closed_intervals'] > 0 and wall['durable_wall_seconds'] >= prior
          and isinstance(report, dict) and report.get('identity_sha256') == identity_sha
          and report.get('wall_seconds') == wall['durable_wall_seconds']
          and report.get('minimum_charged_seconds') == wall['durable_wall_seconds']
          and report.get('blocked') is False and report.get('clock_block') is None
          and report.get('interval_open') is False and report.get('exhausted') is False,
          'prefit_closed_unchanged_campaign_wall_proof_required')
    for name in OPTIONAL:
        if name in value:
            _need(not any(str(row.get('path', '')).endswith(('.pt', '.npy', '.npz', '.pkl'))
                          for row in value[name].get('files', []) if isinstance(row, dict)),
                  'prefit_published_numerical_artifact_preserve_attempt')
    return {'status': 'KNOWN_ZERO_UPDATE_PREFIT_FAILURE_SAME_CLAIM_ONLY',
            'project': str(root), 'pilot_output': PILOT, 'prior_wall_seconds': prior,
            'original_started_utc': receipt['started_utc'], 'campaign_identity_sha256': identity_sha,
            'campaign_charged_wall_seconds': wall['durable_wall_seconds'],
            'scientific_updates_performed': 0, 'new_pilot_claim_authorized': False,
            'original_claim_ledger_and_progress_changed': False}


def _collect(root):
    _output(root)
    raws = {name: _read(root / relative) for name, relative in FILES.items()}
    for name, relative in OPTIONAL.items():
        if _path(root / relative).exists():
            raws[name] = _read(root / relative)
    return raws


def inspect_failed_prefit(root):
    """Pure read-only classification; no GPU, lock, journal or proof creation."""
    root = _root(root); raws = _collect(root); result = _validate(root, raws)
    paths = {**FILES, **OPTIONAL}
    result['sources'] = {name: _descriptor(root, paths[name], raw) for name, raw in raws.items()}
    return result


def _allocation(root):
    from hpc.runtime_profile import PROFILE, selected_profile
    from hpc.run_stage import guard, StageError
    _need(selected_profile(root) == PROFILE, 'prefit_explicit_school_runtime_required')
    stage = os.environ.get('PAPER3_HPC_STAGE')
    try:
        guard(root, stage)
    except StageError:
        raise PrefitError('prefit_current_GPU_allocation_required') from None
    _need(stage in {'pilot', 'train'}, 'prefit_current_GPU_stage_required')
    return os.environ['SLURM_JOB_ID']


def _publish(path, raw):
    path = _path(path)
    missing, current = [], path.parent
    while not current.exists():
        missing.append(current); current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True); _path(path.parent)
    # Persist each newly introduced directory entry as well as the file's
    # immediate directory; a durable leaf must not depend on a missing parent.
    for folder in reversed(missing):
        for directory_path in (folder, folder.parent):
            directory = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
    if path.exists():
        _need(_read(path) == raw, 'prefit_existing_history_or_proof_changed')
        return
    fd, temporary = tempfile.mkstemp(prefix='.prefit-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _need(_read(path) == raw, 'prefit_history_or_proof_appeared_differently')
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _proof_path(root, job):
    return root / '.paper3-hpc/jobs' / job / 'pilot_prefit_resume.json'


def prepare_prefit_resume(root):
    root = _root(root); job = _allocation(root)
    output = _path(root / PILOT)
    if not output.exists():
        return {'status': 'PILOT_NOT_STARTED_NO_PREFIT_RESUME'}
    current = _json(_read(output / 'engineering_receipt.json'))
    if current.get('status') == 'ENGINEERING_PILOT_COMPLETE':
        return {'status': 'EXISTING_PILOT_REQUIRES_ORIGINAL_FULL_VERIFICATION'}
    facts = inspect_failed_prefit(root); raws = _collect(root)
    _need(facts == {**_validate(root, raws), 'sources': {
        name: _descriptor(root, {**FILES, **OPTIONAL}[name], raw) for name, raw in raws.items()}},
        'prefit_original_changed_before_history')
    prefix = PILOT + '/prefit_history/' + facts['sources']['receipt']['sha256']
    history = {}
    for name, raw in raws.items():
        relative = prefix + '/original_' + name + '.json'
        _publish(root / relative, raw); history[name] = _descriptor(root, relative, raw)
    proof = {'schema_version': SCHEMA, 'status': 'PREFIT_SAME_CLAIM_RESUME_PREPARED',
             'project': str(root), 'job_id': job, 'facts': facts, 'history': history,
             'runtime_profile': _descriptor(root, 'audit/hpc_runtime_profile.json',
                                             _read(root / 'audit/hpc_runtime_profile.json'))}
    raw = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _publish(_proof_path(root, job), raw)
    return {**facts, 'status': proof['status'], 'proof': _descriptor(root,
        _proof_path(root, job).relative_to(root).as_posix(), raw), 'history': history}


def consume_prefit_resume(root, output):
    """Read-only revalidation just before the original runner's first write."""
    root = _root(root); job = _allocation(root)
    _need(_path(output) == root / PILOT, 'prefit_same_output_required')
    _output(root)
    path = _proof_path(root, job); raw = _read(path); proof = _json(raw)
    _need(proof.get('schema_version') == SCHEMA
          and proof.get('status') == 'PREFIT_SAME_CLAIM_RESUME_PREPARED'
          and proof.get('project') == str(root) and proof.get('job_id') == job,
          'prefit_current_job_bound_proof_required')
    facts = proof.get('facts', {}); history = proof.get('history', {})
    _need(isinstance(history, dict) and set(FILES) <= set(history) <= set(FILES) | set(OPTIONAL),
          'prefit_complete_original_history_required')
    prefix = PILOT + '/prefit_history/' + facts.get('sources', {}).get('receipt', {}).get('sha256', '')
    raws = {}
    for name, item in history.items():
        relative = prefix + '/original_' + name + '.json'
        _need(isinstance(item, dict) and item.get('path') == relative,
              'prefit_history_path_binding_required')
        source = _read(root / relative)
        _need(item == _descriptor(root, relative, source), 'prefit_history_SHA_mismatch')
        raws[name] = source
    checked = _validate(root, raws)
    sources = {name: _descriptor(root, {**FILES, **OPTIONAL}[name], source)
               for name, source in raws.items()}
    _need(facts == {**checked, 'sources': sources}, 'prefit_history_classification_changed')
    for name in set(raws) - {'campaign_progress', 'campaign_wall'}:
        _need(_read(root / sources[name]['path']) == raws[name],
              'prefit_original_pilot_or_claim_changed_after_proof')
    profile = _read(root / 'audit/hpc_runtime_profile.json')
    _need(proof.get('runtime_profile') == _descriptor(root, 'audit/hpc_runtime_profile.json', profile),
          'prefit_registered_runtime_changed_after_proof')
    return {'prior_wall_seconds': checked['prior_wall_seconds'],
            'original_started_utc': checked['original_started_utc'],
            'same_pilot_claim_preserved': True, 'max_jobs': 1,
            'proof': _descriptor(root, path.relative_to(root).as_posix(), raw), 'history': history}
