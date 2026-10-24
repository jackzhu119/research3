"""Controlled native-CAR calibration shifts; software/algebra checks only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from paper3r6.diagnostics import (COUPLED_SCENARIOS, LEGACY_SCENARIOS, SCENARIOS,
                                 coupled_generators, coupled_matrix, coupled_operator,
                                 perturb_episode)
from p3work.adapt import CalibrationPrefix
from p3work.spatial_geometry import helmert_rows


@pytest.fixture(autouse=True)
def single_torch_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def session(person='S17', name='2'):
    return SimpleNamespace(participant_id=person, session_id=name)


def example():
    generator = torch.Generator().manual_seed(34)
    x = torch.randn(3, 11, 128, generator=generator, dtype=torch.float64)
    qx = torch.randn(9, 11, 128, generator=generator, dtype=torch.float64)
    x -= x.mean(1, keepdim=True); qx -= qx.mean(1, keepdim=True)
    support = CalibrationPrefix(x, torch.tensor([0, 1, 0]), ('e0', 'e2', 'e3'),
        ('e0', 'e1-invalid', 'e2', 'e3'), 'a' * 64)
    episode = SimpleNamespace(support=support, query_x=qx, query_y=torch.arange(9) % 2,
        query_ids=tuple(f'q{i}' for i in range(9)), run_ids=('r1',) * 9,
        audit={'fixed_query_after_B20': True})
    geometry = SimpleNamespace(q=helmert_rows(), rref_sqrt=torch.eye(10),
                               rref_invsqrt=torch.eye(10))
    return episode, geometry


def test_scenario_inventory_preserves_previous_contracts():
    assert len(LEGACY_SCENARIOS) == 6 and len(COUPLED_SCENARIOS) == 6
    assert len(set(SCENARIOS)) == 12
    e, g = example()
    assert perturb_episode(e, g, session(), 'clean')[0] is e
    with pytest.raises(ValueError, match='Unknown diagnostic'):
        perturb_episode(e, g, session(), 'coupled_ks99')


def test_generator_geometry_is_fixed_traceless_unit_and_has_genuine_commutator():
    k, s, k_null, seed = coupled_generators(session())
    torch.testing.assert_close(k.T, -k, atol=1e-14, rtol=0)
    torch.testing.assert_close(s.T, s, atol=1e-14, rtol=0)
    torch.testing.assert_close(k_null.T, -k_null, atol=1e-14, rtol=0)
    assert abs(float(s.trace())) < 1e-14
    for value in (k, s, k_null):
        assert float(value.norm()) == pytest.approx(1., abs=1e-14)
    assert float((k @ s - s @ k).norm()) > .01
    assert float((k_null @ s - s @ k_null).norm()) < 1e-14
    # The null changes the relative axes, preserving both component spectra.
    torch.testing.assert_close(torch.linalg.svdvals(k), torch.linalg.svdvals(k_null), atol=1e-14, rtol=0)
    torch.testing.assert_close(torch.sort(torch.linalg.eigvals(k).imag).values,
                               torch.sort(torch.linalg.eigvals(k_null).imag).values, atol=1e-14, rtol=0)
    assert isinstance(seed, int)


@pytest.mark.parametrize('strength', [.15, .35])
def test_two_noncommuting_orders_differ_but_have_equal_spectrum_and_amplitude(strength):
    k, s, k_null, _ = coupled_generators(session())
    ks, sk = (coupled_matrix(k, s, strength, order) for order in ('ks', 'sk'))
    null = coupled_matrix(k_null, s, strength, 'commuting')
    assert float((ks-sk).norm()) > .0001
    torch.testing.assert_close(torch.linalg.svdvals(ks), torch.linalg.svdvals(sk), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(torch.linalg.svdvals(ks), torch.linalg.svdvals(null), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close((ks-torch.eye(10)).norm(), (sk-torch.eye(10)).norm(), atol=2e-14, rtol=2e-14)
    assert float(torch.linalg.det(ks)) == pytest.approx(1., abs=2e-14)
    assert float(torch.linalg.det(sk)) == pytest.approx(1., abs=2e-14)


@pytest.mark.parametrize('strength', [.15, .35])
def test_commuting_null_agrees_for_both_orders_and_canonical_exponential(strength):
    _, s, k_null, _ = coupled_generators(session())
    left = coupled_matrix(k_null, s, strength, 'ks')
    right = coupled_matrix(k_null, s, strength, 'sk')
    canonical = coupled_matrix(k_null, s, strength, 'commuting')
    torch.testing.assert_close(left, right, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(left, canonical, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('scenario', COUPLED_SCENARIOS)
def test_same_actual_shift_changes_support_and_queries_without_mutating_cache_or_labels(scenario):
    episode, geometry = example()
    before_x, before_q = episode.support.x.clone(), episode.query_x.clone()
    shifted, metadata = perturb_episode(episode, geometry, session(), scenario)
    t, _ = coupled_operator(session(), scenario)
    operator = geometry.q.T @ t @ geometry.q
    torch.testing.assert_close(shifted.support.x, operator @ before_x)
    torch.testing.assert_close(shifted.query_x, operator @ before_q)
    assert not torch.equal(shifted.support.x, before_x)
    assert not torch.equal(shifted.query_x, before_q)
    torch.testing.assert_close(shifted.support.x.mean(1), torch.zeros(3, 128, dtype=torch.float64), atol=2e-14, rtol=0)
    torch.testing.assert_close(shifted.query_x.mean(1), torch.zeros(9, 128, dtype=torch.float64), atol=2e-14, rtol=0)
    assert shifted.support.event_uids == episode.support.event_uids
    assert shifted.support.acquired_event_uids == episode.support.acquired_event_uids
    assert len(shifted.support.acquired_event_uids) == 4 and len(shifted.support.x) == 3
    assert shifted.support.y is episode.support.y and shifted.query_y is episode.query_y
    assert shifted.query_ids == episode.query_ids and shifted.run_ids == episode.run_ids
    assert shifted.audit is episode.audit
    torch.testing.assert_close(episode.support.x, before_x, atol=0, rtol=0)
    torch.testing.assert_close(episode.query_x, before_q, atol=0, rtol=0)
    assert metadata['support_changed'] and metadata['query_changed'] and not metadata['labels_changed']
    assert metadata['scope'] == 'same_shift_support_and_future_query'
    assert metadata['source_reference_used'] is False
    assert metadata['car_generator_norm'] == pytest.approx(metadata['severity'], abs=1e-14)


def test_shift_is_independent_of_source_reference_and_does_not_change_global_rng():
    episode, geometry = example()
    reference = torch.diag(torch.linspace(.2, 3., 10))
    other = SimpleNamespace(q=geometry.q, rref_sqrt=reference, rref_invsqrt=reference.inverse())
    torch.manual_seed(456)
    before_rng = torch.get_rng_state().clone()
    first, recipe = perturb_episode(episode, geometry, session(), 'coupled_ks35')
    second, another = perturb_episode(episode, other, session(), 'coupled_ks35')
    assert torch.equal(torch.get_rng_state(), before_rng)
    torch.testing.assert_close(first.support.x, second.support.x, atol=0, rtol=0)
    torch.testing.assert_close(first.query_x, second.query_x, atol=0, rtol=0)
    assert recipe == another


def test_severities_share_direction_and_null_matches_component_spectra_not_displacement():
    small, low = coupled_operator(session(), 'coupled_ks15')
    large, high = coupled_operator(session(), 'coupled_sk35')
    _, null = coupled_operator(session(), 'coupled_null15')
    assert low['direction_seed'] == high['direction_seed'] == null['direction_seed']
    assert low['severity'] == .15 and high['severity'] == .35
    assert low['commutator_norm'] / high['commutator_norm'] == pytest.approx((.15/.35)**2)
    assert null['commutator_norm'] < 1e-15 and null['commuting_null']
    assert null['gain_spectrum_matched_to_noncommuting']
    assert null['rotation_spectrum_matched_to_noncommuting']
    assert null['null_gain_spectrum_matched_to_noncommuting']
    assert null['ks_sk_gain_spectrum_matched']
    assert not null['null_transform_displacement_matched_to_noncommuting']
    alternate, other = coupled_operator(session(name='3'), 'coupled_ks15')
    assert low['direction_seed'] != other['direction_seed']
    assert not torch.allclose(small, alternate)


def test_old_query_only_shift_and_label_noise_keep_their_distinct_scope():
    episode, geometry = example()
    changed, recipe = perturb_episode(episode, geometry, session(), 'gain10')
    assert changed.support is episode.support and recipe['support_changed'] is False
    assert not torch.equal(changed.query_x, episode.query_x)
    flipped, recipe = perturb_episode(episode, geometry, session(), 'label_noise20')
    assert flipped.query_x is episode.query_x and flipped.query_y is episode.query_y
    assert flipped.support.acquired_event_uids == episode.support.acquired_event_uids
    assert recipe['query_changed'] is False


def test_fixed_b20_future_query_is_identical_across_short_support_budgets(tmp_path):
    from test_external import catalog_fixture
    from paper3r6.external import ExternalCatalog
    root, cache, _, _ = catalog_fixture(tmp_path)
    catalog = ExternalCatalog(root, cache/'manifest.json')
    plan = catalog.plans[0]
    four, twenty = [catalog.materialize(plan, 'cpu', budget=budget) for budget in (4, 20)]
    geometry = SimpleNamespace(q=helmert_rows())
    a, ma = perturb_episode(four, geometry, plan, 'coupled_sk35')
    b, mb = perturb_episode(twenty, geometry, plan, 'coupled_sk35')
    assert a.query_ids == b.query_ids and a.support.acquired_event_uids == four.support.acquired_event_uids
    torch.testing.assert_close(a.query_x, b.query_x, atol=0, rtol=0)
    torch.testing.assert_close(a.support.x, b.support.x[:len(a.support.x)], atol=0, rtol=0)
    assert ma == mb
