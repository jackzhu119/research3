"""Retained scientific regressions that apply to the R11 protocol.
R10 foreign-freeze/PASS gates remain tested in the unchanged R10 package;
R11 replaces them with its signed campaign and task-role tests.
"""
import numpy as np
import pytest
import torch
from torch import nn
from paper3r11 import study
from paper3r11.platform import common

def test_zero_update_fallback_and_ea_have_identical_saturated_scoring():
    # Reuse the established guarded-label EEG-shaped fixture, but create a
    # source whose wrong-class logits saturate ordinary softmax. This exposes
    # artificial NLL gain if only the R11 fallback clips its anchor.
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


def test_independent_cache_cannot_alias_support_and_future_storage(tmp_path):
    from paper3r11.confirmation import validate_confirmation_metadata
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
    common.atomic(manifest, {'schema_version': 'paper3-r11-independent-cache-v1', 'status': 'COMPLETE',
        'cohort_kind': 'independent_confirmation', 'acquisition_order_verified': True,
        'physical_run_id_available': True, 'previously_used_for_algorithm_design': False,
        'signal_contract_sha256': common.digest(contract), 'dataset_identity': {'id': 'synthetic'},
        'participants': ['new'], 'independence_review': study.describe(review, root),
        'sessions': [{'participant_id': 'new', 'session_id': '1', 'acquisition_order_verified': True,
                      'events': study.describe(events, root), 'X': study.describe(array, root)}]})
    plan = {'root': str(root), 'confirmation_manifest': study.describe(manifest, root)}
    with pytest.raises(ValueError):
        validate_confirmation_metadata(plan, {'previously_used_participants': ['old']})


