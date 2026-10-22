"""Recover only metadata after a completed 200-update pilot's finalizer failed.

This adapter never executes a model, restores RNG, changes a claim/budget, or
changes the original verifier/runner. Candidate documents are not authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from hpc import pilot_prefit as prefit
from hpc.runtime_profile import EVIDENCE, INVENTORY, PROFILE, selected_profile, validate_runtime_profile

PILOT = prefit.PILOT
RECEIPT = PILOT + '/engineering_receipt.json'
CANONICAL = (RECEIPT, PILOT + '/engineering_artifact_manifest.json',
             PILOT + '/engineering_backup_manifest.json')
CHECKPOINTS = ('engineering_initial.pt', 'engineering_continuous100.pt',
               'engineering_resume50.pt', 'engineering_resumed100.pt')
PREDICTIONS = ('engineering_continuous100_predictions.json',
               'engineering_resumed100_predictions.json', 'engineering_prediction_replay.json')
COMPARISONS = ('model', 'optimizer', 'sampler_rng', 'torch_rng', 'cuda_rng',
               'python_rng', 'numpy_rng', 'update', 'identity_sha256')
RUNTIME = {'device': 'device', 'device_name': 'device_name', 'device_uuid': 'device_uuid',
    'device_total_memory': 'device_total_memory', 'compute_capability': 'compute_capability',
    'torch': 'torch', 'numpy': 'numpy', 'python': 'python', 'cuda_runtime': 'cuda_runtime',
    'cudnn': 'cudnn', 'driver_versions_local_nvidia_smi': 'driver_versions'}
SCHEMA = 'paper3-hpc-pilot-finalization-transaction-v1'


class FinalizeError(ValueError):
    def __init__(self, code, *, stage=None, cause=None):
        self.code = code
        self.details = {'stage': stage, 'error_type': type(cause).__name__ if cause else None}
        if cause and re.fullmatch(r'[a-z][a-z0-9_]{1,159}', str(cause)):
            self.details['reason_code'] = str(cause)
        super().__init__(code)


def need(value, code):
    if not value:
        raise FinalizeError(code)


def raw(value):
    # Matches the unchanged original runner's JSON writer, including key order.
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(root, relative):
    return prefit._read(prefit._path(root / relative))


def value(root, relative):
    result = json.loads(read(root, relative))
    need(isinstance(result, dict), 'pilot_finalize_JSON_object_required')
    return result


def descriptor(root, relative):
    path = prefit._path(root / relative)
    need(path.is_file(), 'pilot_finalize_original_file_missing')
    from hpc.handoff import hash_file
    digest, size = hash_file(path)
    return {'path': relative, 'bytes': size, 'sha256': digest}


def bound(root, ref, relative):
    actual = descriptor(root, relative)
    need(isinstance(ref, dict) and ref.get('path') == relative
         and ref.get('sha256') == actual['sha256']
         and ('bytes' not in ref or type(ref['bytes']) is int and ref['bytes'] == actual['bytes']),
         'pilot_finalize_original_reference_mismatch')
    return actual


def require_runtime_identity(root, runtime):
    """Cheap original device/runtime check before expensive frozen-data reads."""
    root = prefit._root(root)
    need(isinstance(runtime, dict), 'pilot_finalize_actual_runtime_required')
    identity = value(root, PILOT + '/identity.json')
    need(all(key in identity and other in runtime and identity[key] == runtime[other]
             for key, other in RUNTIME.items()), 'pilot_finalize_original_runtime_device_mismatch')
    return {'status': 'ORIGINAL_PILOT_RUNTIME_IDENTITY_MATCHED_NOT_COMPLETION',
            'identity': descriptor(root, PILOT + '/identity.json'),
            'device_uuid': identity['device_uuid'], 'optimizer_updates_performed': 0}


def _inspect_failed(root, originals=None):
    root = prefit._root(root)
    receipt = json.loads(originals[RECEIPT]) if originals is not None else value(root, RECEIPT)
    need(receipt.get('status') == 'ENGINEERING_PILOT_FAILED'
         and receipt.get('scope') == 'ENGINEERING_SOURCE_PILOT_ONLY'
         and receipt.get('training_started') is True
         and receipt.get('user_started_gpu_attested') is True
         and receipt.get('exact_resume_verified') is True
         and receipt.get('result_manifest_status') == 'BLOCKED_NO_COMPLETE_BACKUP_MANIFEST'
         and receipt.get('result_manifest_error_type') == 'PilotGateError'
         and not receipt.get('failure_type') and not receipt.get('failure')
         and not receipt.get('artifact_manifest') and not receipt.get('backup_manifest_path'),
         'pilot_finalize_only_completed_body_finalization_failure_supported')
    for key, expected in (('actual_completed_updates', 200), ('optimizer_updates_dispatched', 200),
                          ('prediction_replay_optimizer_updates', 0), ('seed', 20261018), ('max_jobs', 1)):
        need(type(receipt.get(key)) is int and receipt[key] == expected,
             'pilot_finalize_exact_original_update_counts_required')
    need(set(receipt.get('resume_comparison', {})) == set(COMPARISONS)
         and all(receipt['resume_comparison'][key] is True for key in COMPARISONS),
         'pilot_finalize_all_original_comparisons_required')
    wall_seconds = receipt.get('total_monotonic_wall_seconds')
    need(type(wall_seconds) in (float, int) and math.isfinite(wall_seconds)
         and 0 <= wall_seconds < 1800, 'pilot_finalize_original_pilot_budget_required')
    receipt_ref = ({'path': RECEIPT, 'bytes': len(originals[RECEIPT]), 'sha256': sha(originals[RECEIPT])}
                   if originals is not None else descriptor(root, RECEIPT))
    files = {'receipt': receipt_ref, 'identity': descriptor(root, PILOT + '/identity.json')}
    identity = value(root, files['identity']['path'])
    need(set(receipt.get('checkpoint_files', {})) == set(CHECKPOINTS),
         'pilot_finalize_all_four_checkpoints_required')
    for name in CHECKPOINTS:
        row = descriptor(root, PILOT + '/' + name)
        need(receipt['checkpoint_files'][name] == row['sha256'], 'pilot_finalize_checkpoint_SHA_mismatch')
        files[name] = row
    for name in PREDICTIONS:
        files[name] = descriptor(root, PILOT + '/' + name)
    bound(root, receipt.get('engineering_prediction_replay'), PILOT + '/engineering_prediction_replay.json')
    for key, relative in (('runtime_inventory', INVENTORY), ('runtime_profile', EVIDENCE)):
        ref = 'resolved_dependencies' if key == 'runtime_inventory' else 'deployment_runtime_profile'
        need(identity.get(ref) == receipt.get(ref), 'pilot_finalize_runtime_receipt_identity_mismatch')
        files[key] = bound(root, receipt.get(ref), relative)
    code = {}
    for relative in ('tools/run_controlled_pilot.py', 'p3work/source_training.py'):
        row = descriptor(root, relative)
        need(identity.get('cache_artifacts', {}).get(relative) == row['sha256'],
             'pilot_finalize_original_code_identity_changed')
        code[relative] = row
    need(identity.get('runner_sha256') == code['tools/run_controlled_pilot.py']['sha256'],
         'pilot_finalize_original_runner_identity_changed')
    for key in ('pilot_claim', 'campaign_claim', 'campaign_progress', 'campaign_wall'):
        files[key] = descriptor(root, prefit.FILES[key])
    claim = value(root, files['pilot_claim']['path'])
    need(claim.get('scope') == 'ENGINEERING_SOURCE_PILOT_ONLY' and claim.get('output') == PILOT
         and type(claim.get('max_jobs')) is int and claim['max_jobs'] == 1,
         'pilot_finalize_same_one_job_claim_required')
    campaign = value(root, files['campaign_claim']['path'])
    state = value(root, files['campaign_progress']['path'])
    wall = value(root, files['campaign_wall']['path'])
    from p3work.contracts import digest
    from p3work.recovery_budget import PersistentWallBudget
    campaign_sha = campaign.get('identity_sha256')
    need(isinstance(campaign.get('identity'), dict) and digest(campaign['identity']) == campaign_sha
         and campaign.get('fixed_output') == prefit.CAMPAIGN
         and campaign.get('claim_must_not_be_deleted_or_renamed_to_retry') is True
         and state.get('identity_sha256') == campaign_sha and state.get('active') == prefit.ACTION
         and state.get('actions') == [] and state.get('status') == 'INTERRUPTED_PRESERVE_SAME_JOB'
         and state.get('active_process', {}).get('cleanup_verified') is True,
         'pilot_finalize_original_closed_campaign_required')
    checker = object.__new__(PersistentWallBudget); checker._state = wall
    try:
        checker._validate_state()
    except (ValueError, KeyError, TypeError) as error:
        raise FinalizeError('pilot_finalize_original_wall_invalid', cause=error) from None
    need(wall.get('identity_sha256') == campaign_sha and wall.get('interval') is None
         and wall.get('clock_block') is None and wall.get('max_wall_seconds') == 43200
         and wall.get('initial_max_wall_seconds') == 43200 and wall.get('amendments') == []
         and wall_seconds <= wall['durable_wall_seconds'] < 43200
         and state.get('wall_budget', {}).get('wall_seconds') == wall['durable_wall_seconds'],
         'pilot_finalize_unchanged_closed_wall_required')
    for relative in CANONICAL[1:]:
        if originals is not None and relative in originals:
            files[Path(relative).name] = {'path': relative, 'bytes': len(originals[relative]),
                                        'sha256': sha(originals[relative])}
        elif originals is None and prefit._path(root / relative).exists():
            files[Path(relative).name] = descriptor(root, relative)
    return {'status': 'COMPLETED_200_PILOT_METADATA_FINALIZATION_REQUIRED', 'project': str(root),
        'pilot_output': PILOT, 'actual_completed_updates': 200, 'exact_resume_claimed': True,
        'numerical_evidence_not_yet_independently_verified': True, 'prior_pilot_wall_seconds': wall_seconds,
        'campaign_identity_sha256': campaign_sha, 'campaign_charged_wall_seconds': wall['durable_wall_seconds'],
        'code_files': code, 'files': files,
        'pilot_runtime': {key: identity.get(key) for key in RUNTIME}}


def _transaction_originals(root, intent, key):
    inspection = intent.get('inspection', {})
    need(intent.get('schema_version') == SCHEMA and inspection.get('project') == str(root)
         and inspection.get('files', {}).get('receipt', {}).get('sha256') == key,
         'pilot_finalize_transaction_originals_changed')
    originals = {row['path']: row for row in inspection.get('files', {}).values() if row['path'] in CANONICAL}
    need(intent.get('before') == originals and set(intent.get('archive', {})) == set(originals),
         'pilot_finalize_transaction_archive_binding_changed')
    retained = {}
    for relative, row in originals.items():
        saved = PILOT + '/finalization_history/' + key + '/original_' + Path(relative).name
        need(intent['archive'].get(relative) == saved, 'pilot_finalize_transaction_archive_binding_changed')
        data = read(root, saved)
        need(sha(data) == row['sha256'] and len(data) == row['bytes'], 'pilot_finalize_archived_original_changed')
        retained[relative] = data
    need(RECEIPT in retained and _inspect_failed(root, retained) == inspection,
         'pilot_finalize_transaction_original_evidence_changed')
    documents, _ = _documents(root, json.loads(retained[RECEIPT]))
    proposed = {p: {'path': p, 'bytes': len(b), 'sha256': sha(b)} for p, b in documents.items()}
    need(intent.get('proposed') == proposed, 'pilot_finalize_proposed_bytes_changed')
    for relative in CANONICAL:
        current = descriptor(root, relative) if (root / relative).exists() else None
        need(current == originals.get(relative) or current == proposed[relative],
             'pilot_finalize_unknown_canonical_change_preserved')
    return retained


def inspect_completed_pilot(root):
    """Read-only strict failure or known interrupted metadata transaction.

    A hand-edited COMPLETE status is never accepted. Transaction recovery
    revalidates the archived strict failure, every unchanged original, and
    independently recomputes all permitted canonical publication bytes.
    """
    root = prefit._root(root)
    base = prefit._path(root / '.paper3-hpc/pilot-finalization')
    if base.exists():
        entries = list(base.iterdir())
        need(len(entries) <= 8, 'pilot_finalize_ambiguous_transaction_history')
        intents = []
        for path in entries:
            prefit._path(path)
            need(path.is_dir() and re.fullmatch('[a-f0-9]{64}', path.name),
                 'pilot_finalize_unknown_transaction_path')
            if (path / 'intent.json').exists(): intents.append(path)
        need(len(intents) <= 1, 'pilot_finalize_ambiguous_transaction_history')
        if intents:
            path = intents[0]
            intent = value(root, path.relative_to(root).as_posix() + '/intent.json')
            _transaction_originals(root, intent, path.name)
            return intent['inspection']
    return _inspect_failed(root)


@contextmanager
def _locks(root):
    handles = []
    try:
        for relative in ('.paper3-hpc/handoff.lock', 'runs/.training_campaign_v1.lock'):
            path = prefit._path(root / relative); path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), 'a')
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise FinalizeError('pilot_finalize_active_campaign_or_handoff_preserved') from None
        yield
    finally:
        for handle in reversed(handles): handle.close()


def _documents(root, receipt, *, prefix='engineering'):
    """Independent finite candidate; the unchanged verifier checks every row."""
    names = ([prefit.FILES['pilot_claim'], PILOT + '/identity.json']
             + [PILOT + '/' + n for n in CHECKPOINTS + PREDICTIONS[:2]]
             + [PILOT + '/' + PREDICTIONS[2], INVENTORY, EVIDENCE])
    rows = []
    for path in names:
        item = descriptor(root, path)
        # Original _descriptor insertion order is part of the predicted bytes.
        rows.append({'path': path, 'sha256': item['sha256'], 'bytes': item['bytes'],
                     'role': 'result', 'dataset': 'Paper3SourceEngineering'})
    candidate = prefix != 'engineering'
    artifact = PILOT + ('/' + prefix + '.artifact.candidate' if candidate else '/engineering_artifact_manifest.json')
    receipt_path = PILOT + ('/' + prefix + '.receipt.candidate' if candidate else '/engineering_receipt.json')
    backup = PILOT + ('/' + prefix + '.backup.candidate' if candidate else '/engineering_backup_manifest.json')
    manifest = {'schema_version': 'paper3-source-engineering-artifacts-v1', 'scope': 'ENGINEERING_SOURCE_PILOT_ONLY',
                'scientific_result': False, 'pilot_status': 'ENGINEERING_PILOT_COMPLETE', 'files': rows}
    artifact_raw = raw(manifest)
    updated = dict(receipt); updated['status'] = 'ENGINEERING_PILOT_COMPLETE'
    updated['artifact_manifest'] = {'path': artifact, 'sha256': sha(artifact_raw), 'bytes': len(artifact_raw)}
    updated['backup_manifest_path'] = backup
    updated['result_manifest_status'] = 'LOCAL_MANIFESTS_COMPLETE_R2_NOT_ATTEMPTED'
    updated['backup_performed'] = False
    receipt_raw = raw(updated)
    backup_rows = rows + [{**row, 'role': 'result', 'dataset': 'Paper3SourceEngineering'} for row in (
        updated['artifact_manifest'], {'path': receipt_path, 'sha256': sha(receipt_raw), 'bytes': len(receipt_raw)})]
    return {artifact: artifact_raw, receipt_path: receipt_raw, backup: raw({'schema_version': 1, 'files': backup_rows})}, receipt_path


def _write(path, data):
    path = prefit._path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.pilot-finalize-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: Path(temp).unlink(missing_ok=True)


def finalize_completed_pilot(root, *, cache, runtime, inspection=None, verifier=None, finalizer=None):
    """Caller owns deployment/controller locks; this owns campaign/handoff locks.

    ``runtime`` must come from the original cuda_runtime in this allocation.
    Test injection is only an explicit software seam; defaults use originals.
    """
    root = prefit._root(root)
    from hpc.run_stage import guard
    need(selected_profile(root) == PROFILE, 'pilot_finalize_explicit_school_profile_required')
    need(os.environ.get('PAPER3_HPC_STAGE') == 'train', 'pilot_finalize_train_stage_required')
    guard(root, os.environ.get('PAPER3_HPC_STAGE'))
    need(cache.root == root and isinstance(runtime, dict), 'pilot_finalize_actual_cache_and_runtime_required')
    require_runtime_identity(root, runtime)
    from p3work.source_training import verify_engineering_pilot
    from tools.run_controlled_pilot import _finalize_result_manifests, _resolved_dependencies
    verifier = verifier or verify_engineering_pilot
    finalizer = finalizer or _finalize_result_manifests
    with _locks(root):
        inspection = inspection or inspect_completed_pilot(root)
        need(inspection.get('project') == str(root) and inspection.get('pilot_output') == PILOT,
             'pilot_finalize_original_inspection_required')
        key = inspection['files']['receipt']['sha256']
        need(re.fullmatch('[a-f0-9]{64}', key), 'pilot_finalize_original_receipt_SHA_required')
        base = '.paper3-hpc/pilot-finalization/' + key
        intent_path = base + '/intent.json'; journal_path = base + '/transaction.json'
        existing = value(root, intent_path) if (root / intent_path).exists() else None
        if existing:
            need(existing.get('schema_version') == SCHEMA and existing.get('inspection') == inspection,
                 'pilot_finalize_transaction_originals_changed')
            originals = {row['path']: row for row in inspection['files'].values() if row['path'] in CANONICAL}
            need(existing.get('before') == originals and set(existing.get('archive', {})) == set(originals),
                 'pilot_finalize_transaction_archive_binding_changed')
            for relative, row in originals.items():
                saved = PILOT + '/finalization_history/' + key + '/original_' + Path(relative).name
                need(existing['archive'].get(relative) == saved,
                     'pilot_finalize_transaction_archive_binding_changed')
                retained = descriptor(root, saved)
                need(retained['sha256'] == row['sha256'] and retained['bytes'] == row['bytes'],
                     'pilot_finalize_archived_original_changed')
        else:
            need(inspect_completed_pilot(root) == inspection, 'pilot_finalize_inspection_changed_before_recovery')
        need(all(inspection['pilot_runtime'].get(key) == runtime.get(other) for key, other in RUNTIME.items()),
             'pilot_finalize_original_runtime_device_mismatch')
        for row in list(inspection['files'].values()) + list(inspection['code_files'].values()):
            if existing and row['path'] in CANONICAL:
                continue
            need(descriptor(root, row['path']) == row, 'pilot_finalize_original_evidence_changed')
        try:
            profile = validate_runtime_profile(root, PROFILE)
            dependencies = _resolved_dependencies(root)
        except (ValueError, OSError) as error:
            raise FinalizeError('pilot_finalize_original_runtime_validation_failed', stage='runtime', cause=error) from None
        identity = value(root, PILOT + '/identity.json')
        need(profile == identity['deployment_runtime_profile'] and dependencies == identity['resolved_dependencies'],
             'pilot_finalize_original_runtime_proof_changed')
        if existing:
            proposed = existing['proposed']
            for relative in CANONICAL:
                current = descriptor(root, relative) if (root / relative).exists() else None
                before = existing['before'].get(relative)
                need(current == before or current == proposed[relative],
                     'pilot_finalize_unknown_canonical_change_preserved')
            original = json.loads(read(root, existing['archive'][RECEIPT]))
        else:
            original = value(root, RECEIPT)
        candidate_docs, candidate_receipt = _documents(root, original, prefix='.pilot-finalize-' + key)
        for relative, data in candidate_docs.items(): prefit._publish(root / relative, data)
        try:
            verifier(cache, candidate_receipt, runtime=runtime)
        except (ValueError, OSError, RuntimeError) as error:
            raise FinalizeError('pilot_finalize_original_deep_verifier_rejected_candidate', stage='candidate', cause=error) from None
        canonical_docs, _ = _documents(root, original)
        proposed = {p: {'path': p, 'bytes': len(b), 'sha256': sha(b)} for p, b in canonical_docs.items()}
        if existing:
            need(existing['proposed'] == proposed, 'pilot_finalize_proposed_bytes_changed')
        else:
            archive = {}; before = {}
            for relative in CANONICAL:
                if (root / relative).exists():
                    before[relative] = descriptor(root, relative)
                    saved = PILOT + '/finalization_history/' + key + '/original_' + Path(relative).name
                    prefit._publish(root / saved, read(root, relative)); archive[relative] = saved
            existing = {'schema_version': SCHEMA, 'inspection': inspection, 'before': before,
                        'archive': archive, 'proposed': proposed, 'numerical_work_performed': False}
            prefit._publish(root / intent_path, raw(existing))
        _write(root / journal_path, raw({'schema_version': SCHEMA, 'status': 'VERIFIED_BEFORE_CANONICAL_PUBLICATION',
                                       'intent_sha256': sha(raw(existing))}))
        # Original runner/source/verifier bytes are retained; only their original
        # metadata producer is invoked after the full candidate gate passed.
        candidate = dict(original); candidate['status'] = 'ENGINEERING_PILOT_COMPLETE'
        try:
            finalizer(root, root / PILOT, candidate)
            need(all(read(root, path) == data for path, data in canonical_docs.items()),
                 'pilot_finalize_original_producer_bytes_differed')
            verifier(cache, RECEIPT, runtime=runtime)
        except (ValueError, OSError, RuntimeError) as error:
            _write(root / journal_path, raw({'schema_version': SCHEMA, 'status': 'BLOCKED_PRESERVE_METADATA_TRANSACTION',
                                           'intent_sha256': sha(raw(existing)), 'error_type': type(error).__name__}))
            raise FinalizeError('pilot_finalize_canonical_publication_incomplete_preserve_transaction',
                                stage='canonical', cause=error) from None
        result = {'schema_version': SCHEMA, 'status': 'FINALIZED_ORIGINAL_FULL_VERIFICATION_PASSED',
                  'intent_sha256': sha(raw(existing)), 'receipt': descriptor(root, RECEIPT),
                  'original_failed_receipt': descriptor(root, existing['archive'][RECEIPT]),
                  'optimizer_updates_performed': 0, 'claims_or_budgets_changed': False}
        _write(root / journal_path, raw(result))
        return result
