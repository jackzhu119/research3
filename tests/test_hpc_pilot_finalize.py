"""Metadata transaction software doubles; never GPU/optimizer/actual fit evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hpc import pilot_finalize as repair, pilot_prefit as prefit, runtime_profile
from tools import run_controlled_pilot as pilot
from p3work import source_training
from test_hpc_pilot_prefit import project as prefit_project

ROOT = Path(__file__).resolve().parents[1]
REAL_PROFILE_VALIDATOR = runtime_profile.validate_runtime_profile
REAL_DEPENDENCIES = pilot._resolved_dependencies


@pytest.fixture
def completed(tmp_path, monkeypatch):
    root = prefit_project.__wrapped__(tmp_path, monkeypatch)
    folder = root / repair.PILOT
    runtime = {other: 'SOFTWARE_RUNTIME_TEST_DOUBLE_' + other for other in repair.RUNTIME.values()}
    runtime['device'] = 'cuda:0'
    identity = {key: runtime[other] for key, other in repair.RUNTIME.items()}
    identity['cache_artifacts'] = {}
    for relative in ('tools/run_controlled_pilot.py', 'p3work/source_training.py'):
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        identity['cache_artifacts'][relative] = repair.descriptor(root, relative)['sha256']
    identity['runner_sha256'] = identity['cache_artifacts']['tools/run_controlled_pilot.py']
    inventory = root / 'audit/gpu_resolved_dependencies.json'
    inventory.write_text('[{"name":"software-test-double","version":"1"}]\n')
    identity['resolved_dependencies'] = repair.descriptor(root, repair.INVENTORY)
    identity['deployment_runtime_profile'] = {key: value for key, value in
        repair.descriptor(root, repair.EVIDENCE).items() if key != 'bytes'}
    pilot._atomic_json(folder / 'identity.json', identity)
    receipt = json.loads((root / repair.RECEIPT).read_bytes())
    receipt.pop('failure_type'); receipt.pop('failure')
    receipt.update(training_started=True, actual_completed_updates=200, optimizer_updates_dispatched=200,
        exact_resume_verified=True, resume_comparison={name: True for name in repair.COMPARISONS},
        result_manifest_status='BLOCKED_NO_COMPLETE_BACKUP_MANIFEST', result_manifest_error_type='PilotGateError',
        total_monotonic_wall_seconds=9.607, resolved_dependencies=identity['resolved_dependencies'],
        deployment_runtime_profile=identity['deployment_runtime_profile'],
        fixture_scope='SOFTWARE_METADATA_DOUBLE_NOT_A_GPU_COMPLETION')
    receipt['checkpoint_files'] = {}
    for name in repair.CHECKPOINTS:
        (folder / name).write_bytes(('SOFTWARE_PLACEHOLDER_NOT_A_MODEL_' + name).encode())
        receipt['checkpoint_files'][name] = repair.descriptor(root, repair.PILOT + '/' + name)['sha256']
    for name in repair.PREDICTIONS:
        pilot._atomic_json(folder / name, {'scope': 'SOFTWARE_PREDICTION_DOUBLE_NO_MODEL'})
    receipt['engineering_prediction_replay'] = repair.descriptor(root, repair.PILOT + '/' + repair.PREDICTIONS[-1])
    pilot._atomic_json(folder / 'engineering_receipt.json', receipt)
    monkeypatch.setattr(repair, 'validate_runtime_profile', lambda *_: identity['deployment_runtime_profile'])
    monkeypatch.setattr(runtime_profile, 'validate_runtime_profile', lambda *_: identity['deployment_runtime_profile'])
    monkeypatch.setattr(pilot, '_resolved_dependencies', lambda *_: identity['resolved_dependencies'])
    cache = SimpleNamespace(root=root, input_contract_sha256='a' * 64, manifest_sha256='b' * 64)
    return root, cache, runtime


def saved(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*')
            if path.is_file() and not path.name.endswith('.lock')}


def software_verifier(root, calls, original):
    """Explicit software seam; never claims these placeholders are genuine."""
    def verify(cache, relative, *, runtime):
        assert cache.root == root
        document = json.loads((root / relative).read_bytes())
        assert document['fixture_scope'] == 'SOFTWARE_METADATA_DOUBLE_NOT_A_GPU_COMPLETION'
        assert document['status'] == 'ENGINEERING_PILOT_COMPLETE'
        calls.append(relative)
        if relative.endswith('.candidate'):
            assert (root / repair.RECEIPT).read_bytes() == original or len(calls) > 1
        for name, digest in document['checkpoint_files'].items():
            assert hashlib.sha256((root / repair.PILOT / name).read_bytes()).hexdigest() == digest
        manifest = root / document['artifact_manifest']['path']
        assert repair.sha(manifest.read_bytes()) == document['artifact_manifest']['sha256']
        return {'fixture_scope': 'SOFTWARE_VERIFIER_DOUBLE_NOT_GPU_PROOF'}
    return verify


def test_inspection_is_readonly_and_preserves_exact_existing_completed_work(completed):
    root, _, _ = completed; before = saved(root)
    report = repair.inspect_completed_pilot(root)
    assert report['actual_completed_updates'] == 200
    assert report['numerical_evidence_not_yet_independently_verified'] is True
    assert report['campaign_charged_wall_seconds'] == 388.038
    assert report['code_files']['tools/run_controlled_pilot.py']['sha256'] == repair.sha((ROOT / 'tools/run_controlled_pilot.py').read_bytes())
    assert saved(root) == before


@pytest.mark.parametrize('changes', [
    {'actual_completed_updates': 199}, {'actual_completed_updates': True},
    {'optimizer_updates_dispatched': 201}, {'exact_resume_verified': False},
    {'training_started': False}, {'prediction_replay_optimizer_updates': 1},
    {'failure_type': 'RuntimeError'}, {'failure': 'body failed'},
    {'result_manifest_error_type': 'UnknownError'}, {'total_monotonic_wall_seconds': 1800},
    {'resume_comparison': {name: True for name in repair.COMPARISONS[:-1]}},
])
def test_incomplete_or_body_failed_pilot_never_authorizes_finalization(completed, changes):
    root, _, _ = completed; document = json.loads((root / repair.RECEIPT).read_bytes())
    document.update(changes); pilot._atomic_json(root / repair.RECEIPT, document)
    before = saved(root)
    with pytest.raises(repair.FinalizeError): repair.inspect_completed_pilot(root)
    assert saved(root) == before


def test_candidate_gate_precedes_canonical_writes_and_original_finalizer_is_used(completed):
    root, cache, runtime = completed; before = saved(root); calls = []
    inspection = repair.inspect_completed_pilot(root)
    result = repair.finalize_completed_pilot(root, cache=cache, runtime=runtime,
        verifier=software_verifier(root, calls, before[repair.RECEIPT]))
    assert result['status'] == 'FINALIZED_ORIGINAL_FULL_VERIFICATION_PASSED'
    assert len(calls) == 2 and calls[0].endswith('.candidate') and calls[1] == repair.RECEIPT
    assert result['optimizer_updates_performed'] == 0 and result['claims_or_budgets_changed'] is False
    original = root / result['original_failed_receipt']['path']
    assert original.read_bytes() == before[repair.RECEIPT]
    for relative, data in before.items():
        if relative not in repair.CANONICAL: assert (root / relative).read_bytes() == data
    canonical = json.loads((root / repair.RECEIPT).read_bytes())
    assert canonical['result_manifest_error_type'] == 'PilotGateError'  # retained historical fact
    assert canonical['total_monotonic_wall_seconds'] == 9.607
    assert canonical['checkpoint_files'] == json.loads(before[repair.RECEIPT])['checkpoint_files']
    # Repeating resumes the same transaction; it creates no pilot or fit claim.
    again = repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
        verifier=software_verifier(root, calls, before[repair.RECEIPT]))
    assert again == result


def test_real_unchanged_deep_verifier_rejects_placeholder_models_before_promotion(completed):
    root, cache, runtime = completed; before = saved(root)
    with pytest.raises(repair.FinalizeError, match='deep_verifier_rejected_candidate'):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime)
    assert (root / repair.RECEIPT).read_bytes() == before[repair.RECEIPT]
    assert not (root / '.paper3-hpc/pilot-finalization').exists()
    assert not (root / repair.PILOT / 'finalization_history').exists()


def test_wrong_GPU_or_changed_checkpoint_never_creates_candidates(completed):
    root, cache, runtime = completed; before = saved(root)
    with pytest.raises(repair.FinalizeError, match='runtime_device_mismatch'):
        repair.finalize_completed_pilot(root, cache=cache, runtime={**runtime, 'device_uuid': 'OTHER_GPU'})
    assert saved(root) == before
    inspection = repair.inspect_completed_pilot(root)
    (root / repair.PILOT / repair.CHECKPOINTS[0]).write_bytes(b'SOFTWARE_CHANGED_CHECKPOINT')
    with pytest.raises(repair.FinalizeError):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection)
    assert not list((root / repair.PILOT).glob('*.candidate'))


def test_torn_original_metadata_publication_resumes_known_hash_transaction_without_work(completed, monkeypatch):
    root, cache, runtime = completed; before = saved(root); inspection = repair.inspect_completed_pilot(root)
    writer = pilot._atomic_json; calls = []; failed = [False]
    def interrupted(path, document):
        writer(path, document)
        if Path(path).name == 'engineering_artifact_manifest.json' and not failed[0]:
            failed[0] = True
            raise OSError('SOFTWARE_INTERRUPTION_AFTER_FIRST_ATOMIC_METADATA_RENAME')
    monkeypatch.setattr(pilot, '_atomic_json', interrupted)
    with pytest.raises(repair.FinalizeError, match='canonical_publication_incomplete'):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
            verifier=software_verifier(root, calls, before[repair.RECEIPT]))
    assert (root / repair.RECEIPT).read_bytes() == before[repair.RECEIPT]
    assert (root / repair.CANONICAL[1]).is_file()
    assert repair.inspect_completed_pilot(root) == inspection
    monkeypatch.setattr(pilot, '_atomic_json', writer)
    result = repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
        verifier=software_verifier(root, calls, before[repair.RECEIPT]))
    assert result['status'] == 'FINALIZED_ORIGINAL_FULL_VERIFICATION_PASSED'
    assert repair.inspect_completed_pilot(root) == inspection
    for relative, data in before.items():
        if relative not in repair.CANONICAL: assert (root / relative).read_bytes() == data


def test_foreign_write_after_interruption_blocks_without_overwriting_it(completed):
    root, cache, runtime = completed; before = saved(root); inspection = repair.inspect_completed_pilot(root)
    def fail(*args): raise OSError('SOFTWARE_STOP_BEFORE_METADATA_WRITE')
    with pytest.raises(repair.FinalizeError):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
            verifier=software_verifier(root, [], before[repair.RECEIPT]), finalizer=fail)
    path = root / repair.CANONICAL[1]; path.write_bytes(b'SOFTWARE_FOREIGN_METADATA_CHANGE')
    with pytest.raises(repair.FinalizeError, match='unknown_canonical_change'):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
            verifier=software_verifier(root, [], before[repair.RECEIPT]))
    assert path.read_bytes() == b'SOFTWARE_FOREIGN_METADATA_CHANGE'


def test_archived_failure_bytes_and_current_code_remain_bound_on_retry(completed):
    root, cache, runtime = completed; before = saved(root); inspection = repair.inspect_completed_pilot(root)
    def fail(*args): raise OSError('SOFTWARE_STOP')
    with pytest.raises(repair.FinalizeError):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection,
            verifier=software_verifier(root, [], before[repair.RECEIPT]), finalizer=fail)
    archive = root / repair.PILOT / 'finalization_history' / inspection['files']['receipt']['sha256'] / 'original_engineering_receipt.json'
    archive.write_bytes(archive.read_bytes() + b'\n')
    with pytest.raises(repair.FinalizeError, match='archived_original_changed'):
        repair.finalize_completed_pilot(root, cache=cache, runtime=runtime, inspection=inspection)


def test_no_GPU_allocation_and_symlink_are_rejected(completed, monkeypatch):
    root, cache, runtime = completed; before = saved(root)
    monkeypatch.delenv('SLURM_JOB_ID')
    with pytest.raises(ValueError): repair.finalize_completed_pilot(root, cache=cache, runtime=runtime)
    assert saved(root) == before
    checkpoint = root / repair.PILOT / repair.CHECKPOINTS[0]
    moved = checkpoint.with_suffix('.saved'); checkpoint.rename(moved); checkpoint.symlink_to(moved)
    with pytest.raises(ValueError): repair.inspect_completed_pilot(root)


def test_cheap_runtime_guard_reads_only_identity_before_cache_work(completed, monkeypatch):
    root, _, runtime = completed
    checked = []
    original = repair.read
    def only_identity(root, relative):
        checked.append(relative)
        assert relative == repair.PILOT + '/identity.json'
        return original(root, relative)
    monkeypatch.setattr(repair, 'read', only_identity)
    report = repair.require_runtime_identity(root, runtime)
    assert report['status'] == 'ORIGINAL_PILOT_RUNTIME_IDENTITY_MATCHED_NOT_COMPLETION'
    with pytest.raises(repair.FinalizeError, match='runtime_device_mismatch'):
        repair.require_runtime_identity(root, {**runtime, 'device_uuid': 'OTHER_PHYSICAL_GPU'})
    assert len(checked) == 2


def test_manual_COMPLETE_without_verified_transaction_is_not_recoverable(completed):
    root, _, _ = completed
    receipt = json.loads((root / repair.RECEIPT).read_bytes());receipt['status'] = 'ENGINEERING_PILOT_COMPLETE'
    pilot._atomic_json(root / repair.RECEIPT, receipt)
    with pytest.raises(repair.FinalizeError, match='only_completed_body'):
        repair.inspect_completed_pilot(root)


def test_original_deep_verifier_and_original_producer_accept_candidate_transaction_with_toy_states(
        completed, monkeypatch):
    """Actual verifier/serializer integration; toy CPU tensors are not fitted models."""
    import numpy as np
    import torch
    from p3work.contracts import SOURCE_IDS, digest
    root, cache, runtime = completed; folder = root / repair.PILOT
    for relative in (repair.INVENTORY, repair.EVIDENCE): (root / relative).unlink()
    (root / 'requirements-gpu.lock').write_bytes((ROOT / 'requirements-gpu.lock').read_bytes())
    inventory = [{'name': name, 'version': version} for name, version in sorted(runtime_profile.REGISTRY['pins'].items())]
    monkeypatch.setattr(runtime_profile, '_observe_installed_inventory', lambda root: inventory)
    monkeypatch.setattr(runtime_profile, '_runtime_versions', lambda: dict(runtime_profile.REGISTRY['pins']))
    monkeypatch.setattr(runtime_profile.platform, 'python_version', lambda: '3.12.3')
    monkeypatch.setattr(runtime_profile, 'validate_runtime_profile', REAL_PROFILE_VALIDATOR)
    monkeypatch.setattr(repair, 'validate_runtime_profile', REAL_PROFILE_VALIDATOR)
    monkeypatch.setattr(pilot, '_resolved_dependencies', REAL_DEPENDENCIES)
    profile = runtime_profile.prepare_runtime_profile(root, runtime_profile.PROFILE)
    runtime.update(torch='2.5.1+cu121', numpy='2.2.6')
    identity = json.loads((folder / 'identity.json').read_bytes())
    identity.update({key: runtime[other] for key, other in repair.RUNTIME.items()})
    identity.update(input_contract_sha256=cache.input_contract_sha256,
        cache_manifest_sha256=cache.manifest_sha256, tf32=False, mixed_precision=False,
        deterministic_algorithms=True, seed=20261018, batch_size=64, optimizer='Adam',
        learning_rate=.001, weight_decay=0, foreach=False, dropout=.25,
        resolved_dependencies=repair.descriptor(root, repair.INVENTORY),
        deployment_runtime_profile=profile, upstream_acceptance_sha256={})
    for key, relative in {'bridge_sha256':'p3work/f0_bridge.py',
            'geometry_sha256':'p3work/spatial_geometry.py','contracts_sha256':'p3work/contracts.py'}.items():
        (root / relative).write_bytes((ROOT / relative).read_bytes())
        identity[key] = repair.descriptor(root, relative)['sha256']
    cache.artifact_sha256 = dict(identity['cache_artifacts']); cache.upstream_acceptance_sha256 = {}
    cache.rows = [{'dataset': dataset, 'subject_id':'SOFTWARE_SUBJECT', 'event_uid':dataset+'/'+str(i),
                   'role':'source_train','label':i % 2} for dataset in sorted(SOURCE_IDS) for i in range(4)]
    cache.labels = np.asarray([row['label'] for row in cache.rows], dtype=np.int64)
    pilot._atomic_json(folder / 'identity.json', identity)
    receipt = json.loads((root / repair.RECEIPT).read_bytes())
    receipt.update(resolved_dependencies=identity['resolved_dependencies'], deployment_runtime_profile=profile)
    for name, update in zip(repair.CHECKPOINTS, (0, 100, 50, 100)):
        state = {'scope':'ENGINEERING_SOURCE_PILOT_ONLY','identity':identity,'identity_sha256':digest(identity),
                 'update':update,'model':{'SOFTWARE_TOY_STATE':torch.tensor([float(update)])},
                 'optimizer':{'step':update},'sampler_rng':torch.tensor([1],dtype=torch.uint8),
                 'torch_rng':torch.tensor([2],dtype=torch.uint8),'cuda_rng':[], 'python_rng':[1],'numpy_rng':[2]}
        torch.save(state, folder / name)
        receipt['checkpoint_files'][name] = repair.descriptor(root, repair.PILOT+'/'+name)['sha256']
    _, uids, labels = pilot.engineering_prediction_selection(cache)
    logits = np.asarray([[.5, -.5] if label == 0 else [-.5, .5] for label in labels], dtype=np.float32)
    probabilities = np.exp(logits.astype(np.float64)); probabilities /= probabilities.sum(1,keepdims=True)
    probabilities = probabilities.astype(np.float32); predicted = logits.argmax(1).tolist()
    common = {'scope':'ENGINEERING_SOURCE_PILOT_ONLY','role':'source_train','synthetic':False,
        'scientific_result':False,'identity_sha256':digest(identity),'input_contract_sha256':cache.input_contract_sha256,
        'cache_manifest_sha256':cache.manifest_sha256,'source_batch_sha256':digest({'event_uids':uids,'labels':labels}),
        'event_uids':uids,'labels':labels}
    branches = {}
    for branch, checkpoint, prediction in zip(('continuous','resumed'), repair.CHECKPOINTS[1::2],repair.PREDICTIONS[:2]):
        checkpoint_item = repair.descriptor(root, repair.PILOT+'/'+checkpoint)
        pilot._atomic_json(folder/prediction, {'schema_version':'paper3-source-engineering-predictions-v1',
            **common,'checkpoint':checkpoint_item,'model_update':100,'optimizer_updates':0,'eval_mode':True,
            'prediction_rule':'two_logits.argmax(-1); exact tie selects class 0','two_logits':logits.tolist(),
            'probabilities':probabilities.tolist(),'predictions':predicted,
            'metrics':pilot.engineering_prediction_metrics(labels,probabilities,predictions=predicted),
            'fixture_scope':'MATHEMATICAL_SOFTWARE_DOUBLE_NOT_GPU_OUTPUT'})
        branches[branch] = {'checkpoint':checkpoint_item,'predictions':repair.descriptor(root,repair.PILOT+'/'+prediction)}
    pilot._atomic_json(folder/repair.PREDICTIONS[2], {'schema_version':'paper3-source-engineering-prediction-replay-v1',
        **common,'selection_rule':pilot.PREDICTION_SELECTION_RULE,'optimizer_updates_for_replay':0,**branches,
        'comparisons':{key:True for key in ('logits','probabilities','predictions','labels','metrics')},'all_exact':True,
        'fixture_scope':'MATHEMATICAL_SOFTWARE_DOUBLE_NOT_GPU_OUTPUT'})
    receipt['engineering_prediction_replay'] = repair.descriptor(root, repair.PILOT+'/'+repair.PREDICTIONS[2])
    pilot._atomic_json(root/repair.RECEIPT, receipt)
    before = saved(root)
    result = repair.finalize_completed_pilot(root, cache=cache, runtime=runtime)
    assert result['status'] == 'FINALIZED_ORIGINAL_FULL_VERIFICATION_PASSED'
    proof = source_training.verify_engineering_pilot(cache, repair.RECEIPT, runtime=runtime)
    assert proof['actual_completed_updates'] == 200 and proof['prediction_replay_exact'] is True
    for relative, data in before.items():
        if relative not in repair.CANONICAL: assert (root/relative).read_bytes() == data
    # The original result collector and finite handoff retain the exact old
    # failure/archive as well as completed current files and both audit proofs.
    from hpc import handoff
    from p3work import training_campaign
    from tools import r2_backup, r2_restore
    from test_hpc_handoff import Bootstrap
    snap = handoff.snapshot(root, label='metadata_recovered',
        apis=(Bootstrap(),training_campaign,r2_backup,r2_restore))
    payload = root / snap['queue_path'] / 'payload'
    assert (payload / result['original_failed_receipt']['path']).read_bytes() == before[repair.RECEIPT]
    assert all((payload / name).read_bytes() == (root / name).read_bytes()
               for name in (repair.INVENTORY, repair.EVIDENCE))
    assert not list(payload.rglob('*.candidate'))
