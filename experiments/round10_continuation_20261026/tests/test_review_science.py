"""Independent CPU-only regression checks from the R10 science audit.

All data and successful-development receipts in this file are synthetic or
mocked. They verify numerical scoring and freeze identity, never EEG efficacy.
"""
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from paper3r10 import algorithm, selection, study
from paper3r10.platform import common
from paper3r10.statistics import SECONDARY_ABLATION_FAMILY


def test_zero_update_fallback_and_ea_have_identical_saturated_scoring():
    # Reuse the established guarded-label EEG-shaped fixture, but create a
    # source whose wrong-class logits saturate ordinary softmax. This exposes
    # artificial NLL gain if only the R10 fallback clips its anchor.
    import test_data_worker as fixtures
    case = fixtures.DataWorkerTests('runTest')
    case.setUpClass()
    case.setUp()
    try:
        head = nn.Linear(6, 2).double()
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([0., -1000.], dtype=torch.float64))
        adapt = fixtures.native('p3work.adapt')
        case.source = adapt.FrozenSourceModel(case.source.encoder, head, feature_dim=6)
        case.plan['synthetic_fixture'] = True
        case.task.update(method='profile', method_config={'amplitude': 0.})
        case.build_reference()
        case.evaluate()
        with np.load(next(case.folder.glob('sessions/*.predictions.npz')), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['probabilities'], saved['ea_probabilities'])
            np.testing.assert_allclose(np.exp(saved['log_probabilities']), saved['probabilities'],
                                       rtol=1e-14, atol=0.)
            assert np.all(saved['probabilities'] >= 1e-12)
        row = common.read(next(case.folder.glob('sessions/*.metrics.json')))
        assert row['balanced_nll'] == row['anchor_ea_balanced_nll']
        assert row['updated_query_fraction'] == 0.
    finally:
        case.tearDown()
        case.tearDownClass()


def _mocked_pass_freeze(tmp_path, monkeypatch):
    # This mocks only a successful development receipt. No fit or EEG file is
    # opened. The ordinary builder and freeze writer are exercised unchanged.
    root = tmp_path / 'source'
    root.mkdir()
    package = tmp_path / 'package'
    (package / 'paper3r10').mkdir(parents=True)
    (package / 'paper3r10/algorithm.py').write_text('# synthetic review fixture\n')
    payload = root / 'payload.json'
    common.atomic(payload, {'synthetic': True})
    item = study.describe(payload, root)
    registry = {'schema_version': 'paper3-r10-study-registry-v1', 'root': str(root),
        'source_by_seed': {str(seed): {'source_receipt': item, 'reference_receipt': item}
                           for seed in study.SEEDS},
        'external_manifests': {'pan2023': item, 'weibo2014': item},
        'baseline_artifacts': {str(seed): {method: {'artifact': item}
            for method in study.NEEDS_ARTIFACTS} for seed in study.SEEDS},
        'previously_used_participants': ['synthetic-old-person']}
    registry_path = root / 'registry.json'
    common.atomic(registry_path, registry)
    descriptor = study.describe(registry_path, root)
    plan = study.build_plan(root, descriptor, package=package)
    plan_path = root / 'mechanism.json'
    common.atomic(plan_path, plan)
    decision = {'status': 'PASS', 'scientific_scope': 'REAL_DATA',
        'study_registry_sha256': descriptor['sha256'],
        'decision_identity': 'a' * 64, 'selected_configuration': asdict(algorithm.Config()),
        'source_result_receipts': [], 'selected_candidate_id': 'a0.3-c0.5',
        'comparison_family': study.PRIMARY_FAMILY,
        'secondary_ablation_family': SECONDARY_ABLATION_FAMILY,
        'selection_participant_ids': ['mock-select'], 'check_participant_ids': ['mock-check'],
        'gates': {key: True for key in ('pairing_and_identity',)}}
    monkeypatch.setattr(selection, 'development_decision', lambda path: decision)
    monkeypatch.setattr(selection, 'load_completed', lambda path: (plan, [], {}, []))
    freeze = selection.freeze_development(plan_path, root / 'freeze.json')
    return root, package, registry_path, descriptor, plan, freeze


def test_actual_freeze_writer_emits_enforced_code_and_registry_identities(tmp_path, monkeypatch):
    root, _, _, descriptor, plan, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    saved = common.read(root / freeze['path'])
    assert saved['code_manifest_sha256'] == plan['code_manifest_sha256']
    assert saved['study_registry_sha256'] == descriptor['sha256']


def test_actual_freeze_rejects_code_change_on_formal_transfer(tmp_path, monkeypatch):
    root, package, _, descriptor, _, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    (package / 'paper3r10/algorithm.py').write_text('# changed synthetic implementation\n')
    with pytest.raises(ValueError, match='[Cc]ode'):
        study.build_plan(root, descriptor, stage='formal_exploratory', package=package, freeze=freeze)


def test_actual_freeze_rejects_registry_change_on_formal_transfer(tmp_path, monkeypatch):
    root, package, registry_path, _, _, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    registry = common.read(registry_path)
    registry['review_fixture_changed'] = True
    common.atomic(registry_path, registry)
    with pytest.raises(ValueError, match='registry'):
        study.build_plan(root, study.describe(registry_path, root), stage='formal_exploratory',
                         package=package, freeze=freeze)


def test_confirmation_manifest_without_review_or_sessions_is_blocked_before_jobs(tmp_path, monkeypatch):
    root, package, _, descriptor, _, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    manifest = root / 'unreviewed-confirmation.json'
    common.atomic(manifest, {'status': 'COMPLETE', 'cohort_kind': 'independent_confirmation',
        'dataset_identity': 'synthetic-new-name', 'previously_used_for_algorithm_design': False,
        'acquisition_order_verified': True, 'physical_run_id_available': True,
        'participants': ['synthetic-new-person']})
    with pytest.raises(ValueError):
        study.build_plan(root, descriptor, stage='formal_confirmatory', package=package,
                         freeze=freeze, confirmation=study.describe(manifest, root))


@pytest.mark.parametrize('field', ['code_manifest_sha256', 'study_registry_sha256', 'scientific_scope'])
def test_formal_transfer_rejects_freeze_with_missing_binding(tmp_path, monkeypatch, field):
    root, package, _, descriptor, _, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    path = root / freeze['path']
    saved = common.read(path)
    saved.pop(field)
    common.atomic(path, saved)
    with pytest.raises(ValueError):
        study.build_plan(root, descriptor, stage='formal_exploratory', package=package,
                         freeze=study.describe(path, root))


def test_formal_transfer_rejects_software_only_freeze(tmp_path, monkeypatch):
    root, package, _, descriptor, _, freeze = _mocked_pass_freeze(tmp_path, monkeypatch)
    path = root / freeze['path']
    saved = common.read(path)
    saved['scientific_scope'] = 'SOFTWARE_TEST_ONLY'
    common.atomic(path, saved)
    with pytest.raises(ValueError):
        study.build_plan(root, descriptor, stage='formal_exploratory', package=package,
                         freeze=study.describe(path, root))


def test_independent_cache_cannot_alias_support_and_future_storage(tmp_path):
    from paper3r10.confirmation import validate_confirmation_metadata
    root = tmp_path
    (root / 'contracts').mkdir()
    contract = {'synthetic_review_fixture': True}
    common.atomic(root / 'contracts/input_contract.json', contract)
    array = root / 'signals.npy'
    np.save(array, np.zeros((40, 11, 128)))
    rows = [{'participant_id': 'new', 'session_id': '1', 'tasknumber': 1,
             'event_uid': f'e{i}', 'trial_uid': f'e{i}', 'original_trial_index': i,
             'label': i % 2, 'valid': True, 'cache_index': i % 20,
             'physical_run_id': 'support' if i < 20 else 'future',
             'physical_run_complete_verified': True} for i in range(40)]
    events = root / 'events.json'
    common.atomic(events, rows)
    review = root / 'review.json'
    common.atomic(review, {'status': 'REVIEWED', 'prior_design_history_reviewed': True,
        'previously_used_for_algorithm_design': False, 'reviewer': 'synthetic reviewer',
        'reviewed_utc': '2026-10-26T00:00:00Z', 'dataset_identity': {'id': 'synthetic'},
        'participants': ['new']})
    manifest = root / 'confirmation.json'
    common.atomic(manifest, {'schema_version': 'paper3-r10-independent-cache-v1', 'status': 'COMPLETE',
        'cohort_kind': 'independent_confirmation', 'acquisition_order_verified': True,
        'physical_run_id_available': True, 'previously_used_for_algorithm_design': False,
        'signal_contract_sha256': common.digest(contract), 'dataset_identity': {'id': 'synthetic'},
        'participants': ['new'], 'independence_review': study.describe(review, root),
        'sessions': [{'participant_id': 'new', 'session_id': '1', 'acquisition_order_verified': True,
                      'events': study.describe(events, root), 'X': study.describe(array, root)}]})
    plan = {'root': str(root), 'confirmation_manifest': study.describe(manifest, root)}
    with pytest.raises(ValueError):
        validate_confirmation_metadata(plan, {'previously_used_participants': ['old']})
