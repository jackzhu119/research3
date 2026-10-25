"""Numerical mechanism/software tests; these do not establish EEG efficacy."""
from dataclasses import replace
from types import SimpleNamespace
import inspect

import pytest
import torch

from paper3r8 import transport as tr


torch.set_num_threads(2)


def spd(d=10, seed=1):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    a = a @ a.T + torch.eye(d, dtype=torch.float64)
    return a / a.trace()


def fixture(n=20):
    q = torch.zeros(10, 11, dtype=torch.float64)
    for k in range(1, 11):
        q[k-1, :k] = 1 / (k*(k+1))**.5
        q[k-1, k] = -k / (k*(k+1))**.5
    reference = spd()
    geometry = SimpleNamespace(q=q, rref_sqrt=tr.spd_sqrt(reference),
                               rref_invsqrt=tr.spd_invsqrt(reference))
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(n, 11, 32, dtype=torch.float64, generator=generator)
    x = x-x.mean(1, keepdim=True)
    y = torch.arange(n) % 2

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weights', torch.linspace(-.2, .2, 11, dtype=torch.float64))
        def features_and_logits(self, value):
            features = value[:, :, :8].mean(-1)
            margin = features @ self.weights
            return features, torch.stack((-margin/2, margin/2), 1)
        def source_logits(self, value):
            rms = value.square().mean((-2, -1), keepdim=True).sqrt()
            return self.features_and_logits(value/rms)[1]
    return Source(), geometry, SimpleNamespace(x=x, y=y)


@pytest.mark.parametrize('root', [tr.spd_sqrt, tr.spd_invsqrt])
@pytest.mark.parametrize('repeated', [False, True])
def test_root_gradient_matches_finite_differences_including_repeated_eigenvalues(root, repeated):
    a = (torch.eye(4, dtype=torch.float64)*.5 if repeated else spd(4)).requires_grad_(True)
    assert torch.autograd.gradcheck(root, (a,), eps=1e-6, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('kind', ['recolor', 'bures', 'metric_bures'])
def test_full_map_exactly_matches_regularized_covariance(kind):
    c, r = spd(seed=4), spd(seed=5)
    t = tr.covariance_map(c, r, kind)
    torch.testing.assert_close(t @ c @ t.T, r, atol=2e-13, rtol=2e-11)


def test_bures_map_has_minimum_euclidean_covariance_weighted_displacement():
    c, r = spd(seed=4), spd(seed=5)
    eye = torch.eye(10, dtype=c.dtype)
    cost = lambda t: torch.trace((t-eye) @ c @ (t-eye).T)
    b = tr.covariance_map(c, r, 'bures')
    e = tr.covariance_map(c, r, 'recolor')
    assert cost(b) < cost(e)-1e-6
    torch.testing.assert_close(b, b.T, atol=1e-13, rtol=1e-13)
    assert torch.linalg.eigvalsh(b).min() > 0


def test_metric_bures_minimizes_its_stated_source_whitened_displacement():
    c, r = spd(seed=4), spd(seed=5)
    eye = torch.eye(10, dtype=c.dtype)
    g = torch.linalg.inv(r)
    cost = lambda t: torch.trace(g @ (t-eye) @ c @ (t-eye).T)
    mapped = tr.covariance_map(c, r, 'metric_bures')
    assert cost(mapped) < cost(tr.covariance_map(c, r, 'bures'))-1e-6
    assert cost(mapped) < cost(tr.covariance_map(c, r, 'recolor'))-1e-6


def test_commuting_covariance_null_and_identity():
    c = torch.diag(torch.linspace(.01, .19, 10, dtype=torch.float64))
    r = torch.diag(torch.linspace(.19, .01, 10, dtype=torch.float64))
    matrices = [tr.covariance_map(c, r, kind) for kind in ('recolor', 'bures', 'metric_bures')]
    for mapped in matrices[1:]:
        torch.testing.assert_close(mapped, matrices[0], atol=1e-13, rtol=1e-13)
    r = spd()
    for kind in ('recolor', 'bures', 'metric_bures'):
        torch.testing.assert_close(tr.covariance_map(r, r, kind), torch.eye(10, dtype=r.dtype),
                                   atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('kind', ['recolor', 'bures', 'metric_bures'])
@pytest.mark.parametrize('alpha', [0., .3, 1.])
def test_partial_map_preserves_car_complement_and_is_invertible(kind, alpha):
    source, g, support = fixture()
    fitted = tr.fit_transport(source, g, support, tr.TransportConfig(kind=kind, alpha=alpha))
    assert fitted.fallback_reason is None
    perturbed = support.x + .005
    actual = tr.transform_signal(g, fitted, perturbed)
    torch.testing.assert_close(actual.mean(1), perturbed.mean(1), atol=1e-14, rtol=1e-13)
    assert torch.linalg.det(fitted.transform) > 0
    if alpha == 0:
        assert torch.equal(actual, perturbed)
    if alpha == 1:
        assert fitted.diagnostics['full_map_covariance_residual'] < 1e-11


def test_ea_reproduces_preceding_full_map_on_exact_car_data():
    source, g, support = fixture()
    fitted = tr.fit_transport(source, g, support, tr.TransportConfig(kind='ea'))
    cov = tr.trial_covariances(g.q, support.x).mean(0)
    c = .95*cov+.05*torch.eye(10, dtype=cov.dtype)/10
    eig, vectors = torch.linalg.eigh(c)
    inv = (vectors*eig.rsqrt())@vectors.T
    old_signal = g.q.T @ g.rref_sqrt @ inv @ g.q @ support.x
    expected = source.source_logits(old_signal)
    actual = tr.predict_transport(source, g, fitted, support.x)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-11)


@pytest.mark.parametrize('kind', ['recolor', 'bures', 'metric_bures'])
@pytest.mark.parametrize('mode', ['global', 'evidence'])
def test_outer_gradient_of_actual_prediction_matches_finite_difference(kind, mode):
    source, g, support = fixture()
    cfg = tr.TransportConfig(kind=kind, parameterization=mode)
    params = tr.init_meta_parameters(cfg)
    def value(p):
        fitted = tr.fit_transport(source, g, support, cfg, p)
        return tr.predict_transport(source, g, fitted, support.x[:5]).square().mean()
    grad = torch.autograd.grad(value(params), params)[0]
    step = 1e-5
    for index in (0, params.numel()-1):
        delta = torch.zeros_like(params).reshape(-1)
        delta[index] = step
        delta = delta.reshape_as(params)
        numerical = (value(params+delta)-value(params-delta))/(2*step)
        torch.testing.assert_close(grad.flatten()[index], numerical, atol=2e-10, rtol=1e-5)


def test_query_independent_fit_batch_independent_predict_and_no_source_mutation():
    source, g, support = fixture()
    before = {key: value.clone() for key, value in source.state_dict().items()}
    fitted = tr.fit_transport(source, g, support, tr.TransportConfig(kind='bures'))
    assert 'query' not in inspect.signature(tr.fit_transport).parameters
    whole = tr.predict_transport(source, g, fitted, support.x)
    single = torch.cat([tr.predict_transport(source, g, fitted, x[None]) for x in support.x])
    torch.testing.assert_close(whole, single, atol=1e-13, rtol=1e-12)
    assert all(torch.equal(value, source.state_dict()[key]) for key, value in before.items())


def test_pooled_covariance_fit_is_label_permutation_invariant_and_uses_one_class():
    source, g, support = fixture()
    cfg = tr.TransportConfig(kind='bures')
    original = tr.fit_transport(source, g, support, cfg)
    changed = tr.fit_transport(source, g, SimpleNamespace(x=support.x, y=support.y*0), cfg)
    assert changed.fallback_reason is None
    assert torch.equal(changed.transform, original.transform)
    assert changed.diagnostics['uses_support_labels'] is False


def test_empty_and_invalid_support_are_explicit_source_fallbacks():
    source, g, support = fixture()
    for x, y in ((support.x[:0], support.y[:0]), (support.x*0, support.y)):
        fitted = tr.fit_transport(source, g, SimpleNamespace(x=x, y=y), tr.TransportConfig())
        assert fitted.fallback_reason is not None
        assert torch.equal(tr.predict_transport(source, g, fitted, support.x), source.source_logits(support.x))


def test_identity_is_exact_even_without_support():
    source, g, support = fixture()
    fitted = tr.fit_transport(source, g, SimpleNamespace(x=support.x[:0], y=support.y[:0]),
                              tr.TransportConfig(kind='identity'))
    assert torch.equal(tr.predict_transport(source, g, fitted, support.x), source.source_logits(support.x))


def test_class_balance_control_retains_one_class_instead_of_deleting_session():
    source, g, support = fixture()
    support.y[:] = 0
    fitted = tr.fit_transport(source, g, support, tr.TransportConfig(covariance_weighting='class_balanced'))
    assert fitted.fallback_reason is None
    assert fitted.diagnostics['actual_covariance_weighting'] == 'pooled_one_class_support'


def test_fiber_preserves_covariance_and_moves_alignment_in_predicted_local_direction():
    c, r = spd(seed=4), spd(seed=5)
    contrast, ds = spd(seed=6)-spd(seed=7), spd(seed=8)-spd(seed=9)
    config = tr.TransportConfig(kind='fiber', beta=1e-4)
    aligned, da = tr.fiber_map(c, r, contrast, ds, config)
    reversed_map, dr = tr.fiber_map(c, r, contrast, ds, replace(config, fiber_control='reversed'))
    ea = tr.covariance_map(c, r, 'recolor')
    for mapped in (aligned, reversed_map):
        torch.testing.assert_close(mapped @ c @ mapped.T, r, atol=5e-13, rtol=1e-11)
    assert da['fiber_alignment_after'] > da['fiber_alignment_before']
    assert dr['fiber_alignment_after'] < dr['fiber_alignment_before']
    assert da['fiber_contrast_distance_after'] < da['fiber_contrast_distance_before']
    central = (da['fiber_alignment_after']-dr['fiber_alignment_after'])/(2*config.beta)
    assert central == pytest.approx(da['fiber_commutator_norm'], rel=1e-7, abs=1e-10)
    assert not torch.allclose(aligned, ea, atol=1e-8, rtol=1e-8)


def test_fiber_zero_beta_is_ea_and_random_control_matches_rotation_generator_norm():
    c, r = spd(seed=4), spd(seed=5)
    contrast, ds = spd(seed=6)-spd(seed=7), spd(seed=8)-spd(seed=9)
    config = tr.TransportConfig(kind='fiber', beta=0)
    mapped, _ = tr.fiber_map(c, r, contrast, ds, config)
    torch.testing.assert_close(mapped, tr.covariance_map(c, r, 'recolor'), atol=1e-13, rtol=1e-12)
    _, a = tr.fiber_map(c, r, contrast, ds, replace(config, beta=.2))
    _, b = tr.fiber_map(c, r, contrast, ds, replace(config, beta=.2, fiber_control='random'))
    assert a['fiber_direction_norm'] == pytest.approx(b['fiber_direction_norm'])


def test_fiber_commuting_null_does_not_fail_or_claim_identification():
    c, r = spd(seed=4), spd(seed=5)
    mapped, diagnostics = tr.fiber_map(c, r, torch.zeros_like(c), torch.eye(10, dtype=c.dtype),
                                      tr.TransportConfig(kind='fiber'))
    torch.testing.assert_close(mapped, tr.covariance_map(c, r, 'recolor'), atol=1e-13, rtol=1e-12)
    assert diagnostics['fiber_local_direction_unidentified']


def test_fiber_one_class_uses_ea_and_keeps_trial_data():
    source, g, support = fixture()
    support.y[:] = 0
    cfg = tr.TransportConfig(kind='fiber', alpha=1.)
    fitted = tr.fit_transport(source, g, support, cfg, {'class_reference': spd(seed=14)})
    ea = tr.fit_transport(source, g, support, tr.TransportConfig(kind='ea'))
    assert fitted.fallback_reason is None
    torch.testing.assert_close(fitted.transform, ea.transform, atol=1e-13, rtol=1e-12)
    assert fitted.diagnostics['fiber_two_support_classes'] is False


@pytest.mark.parametrize('seed', [1, 10, 20])
def test_bounded_fiber_step_obeys_stated_alignment_gain_bound(seed):
    c, r = spd(seed=seed), spd(seed=seed+1)
    contrast, ds = spd(seed=seed+2)-spd(seed=seed+3), spd(seed=seed+4)-spd(seed=seed+5)
    cfg = tr.TransportConfig(kind='fiber', beta=.35, beta_rule='bounded')
    _, d = tr.fiber_map(c, r, contrast, ds, cfg)
    actual = d['fiber_alignment_after']-d['fiber_alignment_before']
    bound = d['fiber_alignment_gain_lower_bound']
    assert 0 < d['fiber_beta'] <= .35
    assert actual >= bound-1e-12
    assert bound >= .5*d['fiber_beta']*d['fiber_aligned_initial_slope']-1e-12
    for control in ('reversed', 'random'):
        _, other = tr.fiber_map(c, r, contrast, ds, replace(cfg, fiber_control=control))
        assert other['fiber_beta'] == pytest.approx(d['fiber_beta'])
        assert other['fiber_alignment_gain_lower_bound'] is None


def test_bounded_fiber_zero_commutator_is_identity_rotation_and_zero_bound():
    c, r = spd(seed=4), spd(seed=5)
    mapped, d = tr.fiber_map(c, r, torch.zeros_like(c), torch.eye(10, dtype=c.dtype),
        tr.TransportConfig(kind='fiber', beta=.35, beta_rule='bounded'))
    assert d['fiber_beta'] == 0.
    assert d['fiber_alignment_gain_lower_bound'] == 0.
    torch.testing.assert_close(mapped, tr.covariance_map(c, r, 'recolor'), atol=1e-13, rtol=1e-12)


def test_bad_stationary_contrast_is_retained_without_claiming_global_optimum():
    c, r = spd(seed=4), spd(seed=5)
    contrast = spd(seed=6)-spd(seed=7)
    ci = tr.spd_invsqrt(c)
    dt = tr._sym(ci @ contrast @ ci)
    mapped, d = tr.fiber_map(c, r, contrast, -dt,
        tr.TransportConfig(kind='fiber', beta=.35, beta_rule='bounded'))
    assert d['fiber_local_direction_unidentified']
    assert d['fiber_alignment_before'] < 0
    assert d['fiber_beta'] == 0.
    torch.testing.assert_close(mapped, tr.covariance_map(c, r, 'recolor'), atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize('bad', [dict(alpha=2), dict(shrinkage=-.1), dict(kind='unknown'),
    dict(covariance_weighting='query'), dict(parameterization='unknown'), dict(beta=-1)])
def test_unknown_configuration_does_not_silently_change_scientific_method(bad):
    with pytest.raises(ValueError):
        tr.TransportConfig(**bad)
