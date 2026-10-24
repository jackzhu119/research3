"""Head-span geometry, prior normalization and support-only projection checks."""
import inspect

import pytest
import torch

from paper3r7.head_capacity import (effective_head_l2, identifiability,
                                  project_weights, projector)


def sample(n=20, p=32, r=4):
    generator = torch.Generator().manual_seed(704)
    f = torch.randn(n, p, generator=generator, dtype=torch.float64)
    j = torch.randn(n, r, generator=generator, dtype=torch.float64)
    y = torch.arange(n) % 2
    return j, f, y, torch.linspace(-1., 1., n, dtype=torch.float64)


@pytest.mark.parametrize('policy,expected', [('full', 32), ('fixed', 4), ('budget', 15), ('none', 0)])
def test_projector_dimensions_detach_and_orthogonality(policy, expected):
    _, f, _, _ = sample(); f.requires_grad_(True)
    u, d = projector(f, policy, rank=4)
    assert u.shape == (32, expected) and u.dtype == torch.float64 and not u.requires_grad
    torch.testing.assert_close(u.T @ u, torch.eye(expected, dtype=torch.float64), atol=2e-14, rtol=2e-14)
    assert d['effective_head_rank'] == expected
    assert d['head_prior_normalization_dim_including_bias'] == 33
    assert not d['projector_uses_labels'] and not d['projector_uses_queries']
    if policy == 'full':
        torch.testing.assert_close(u, torch.eye(32, dtype=torch.float64), atol=0, rtol=0)


@pytest.mark.parametrize('n,expected', [(4, 0), (8, 3), (12, 7), (16, 11), (20, 15)])
def test_budget_counts_actual_valid_rows_without_a_startup_gate(n, expected):
    _, f, _, _ = sample(n=n)
    u, d = projector(f, 'budget', rank=4)
    assert u.shape[1] == expected and d['valid_support_count'] == n
    if n == 4:
        assert not d['dimension_necessary_condition_only']
        assert d['identifiability_claimed'] is False


@pytest.mark.parametrize('basis', ['pca', 'random'])
def test_constant_support_does_not_invent_independent_head_directions(basis):
    u, d = projector(torch.ones(20, 32), 'budget', rank=4, basis=basis)
    assert u.shape == (32, 0) and d['centered_support_feature_rank'] == 0
    assert d['requested_head_rank'] == 15 and d['effective_head_rank'] == 0


def test_low_rank_support_caps_both_matched_dimension_bases():
    generator = torch.Generator().manual_seed(220)
    f = torch.randn(20, 2, generator=generator, dtype=torch.float64) @ torch.randn(2, 32, generator=generator, dtype=torch.float64)
    for basis in ('pca', 'random'):
        u, d = projector(f, 'fixed', rank=4, head_rank=8, basis=basis)
        assert u.shape[1] == d['centered_support_feature_rank'] == 2


def test_pca_is_input_only_and_random_space_is_seed_fixed_and_nested():
    _, f, _, _ = sample()
    pca, _ = projector(f, 'fixed', 4)
    shifted, _ = projector(f + torch.arange(32, dtype=f.dtype)[None], 'fixed', 4)
    torch.testing.assert_close(pca @ pca.T, shifted @ shifted.T, atol=3e-14, rtol=3e-14)
    a, _ = projector(f, 'fixed', 4, head_rank=4, basis='random')
    b, _ = projector(f.flip(0) * 2, 'fixed', 4, head_rank=8, basis='random')
    torch.testing.assert_close(a, b[:, :4], atol=0, rtol=0)
    assert 'y' not in inspect.signature(projector).parameters
    assert 'query' not in str(inspect.signature(projector))


def test_projection_preserves_full_head_norm_and_prior_under_dimension_change():
    _, f, _, _ = sample(); u, _ = projector(f, 'fixed', 4)
    a = torch.arange(1., 5., dtype=torch.float64)
    full = u @ a
    torch.testing.assert_close(full.square().sum(), a.square().sum(), atol=2e-13, rtol=2e-14)
    for h in (0, 1, 4, 16, 32):
        effective = effective_head_l2(.1, h, 32)
        assert effective / (h + 1) == pytest.approx(.1 / 33)
    torch.testing.assert_close(project_weights(full, u), full, atol=1e-14, rtol=1e-14)
    coefficients = torch.randn(32, dtype=torch.float64, requires_grad=True)
    projected = project_weights(coefficients, u)
    projected.sum().backward()
    torch.testing.assert_close(coefficients.grad, u @ (u.T @ torch.ones(32, dtype=u.dtype)))


def test_full_row_rank_head_absorbs_every_spatial_support_effect():
    j, f, y, m = sample()
    d = identifiability(j, f, y, m, .1, .01)
    assert d['weighted_head_span_rank'] == 20
    assert d['data_only_spatial_info_rank'] == d['data_only_spatial_info_rank_bound'] == 0
    assert d['head_absorbed_spatial_fraction'] == 1.
    assert torch.tensor(d['data_only_spatial_info']).abs().max() == 0
    assert min(d['penalized_head_spatial_eigenvalues']) > 0
    assert min(d['total_penalized_spatial_eigenvalues']) > .02
    assert not d['penalized_curvature_is_data_identifiability']


def test_data_information_rank_respects_remaining_observation_dimensions():
    j, f, y, m = sample(n=8, p=32, r=4)
    for h in (0, 2, 4, 7):
        u, _ = projector(f, 'fixed', 4, head_rank=h)
        d = identifiability(j, f @ u, y, m, .1, .01, full_head_dim=32)
        assert d['data_only_spatial_info_rank'] <= min(4, 8-d['weighted_head_span_rank'])
        assert d['head_prior_precision_eta'] == pytest.approx(.2/33)
        if h == 7:
            assert d['data_only_spatial_info_rank'] == 0


def test_penalized_schur_matches_direct_formula_and_ridge_zero_limit():
    j, f, y, m = sample(n=8, p=10, r=4)
    counts = torch.bincount(y).double()
    w = (.5/counts)[y] * m.sigmoid() * (-m).sigmoid()
    z = torch.cat((f, torch.ones(8, 1, dtype=f.dtype)), 1)
    a = j.T @ (w[:, None]*j)
    cross = j.T @ (w[:, None]*z)
    head = z.T @ (w[:, None]*z)
    eta = .2/11
    expected = a-cross @ torch.linalg.solve(head+eta*torch.eye(11, dtype=f.dtype), cross.T)
    actual = identifiability(j, f, y, m, .1, .01)
    torch.testing.assert_close(torch.tensor(actual['penalized_head_spatial_schur'], dtype=j.dtype), expected,
                               atol=3e-14, rtol=3e-12)
    inverse_coefficient = j.T @ torch.linalg.solve(z @ z.T, j)
    for penalty in (1e-6, 1e-8, 1e-10):
        d = identifiability(j, f, y, m, penalty, .01)
        observed = torch.tensor(d['penalized_head_spatial_schur'], dtype=j.dtype)/d['head_prior_precision_eta']
        torch.testing.assert_close(observed, inverse_coefficient, atol=2e-4, rtol=2e-4)
    zero = identifiability(j, f, y, m, 0., .01)
    assert torch.tensor(zero['penalized_head_spatial_schur']).abs().max() == 0


def test_weighted_information_invariant_to_invertible_head_reparameterization():
    j, f, y, m = sample(n=20, p=4)
    change = torch.tensor([[2., .2, 0, 0], [0, 1., .3, 0], [0, 0, .5, 0], [0, 0, 0, 1.5]], dtype=f.dtype)
    a = identifiability(j, f, y, m, 0., .01)
    b = identifiability(j, f @ change, y, m, 0., .01)
    torch.testing.assert_close(torch.tensor(a['data_only_spatial_info']), torch.tensor(b['data_only_spatial_info']), atol=1e-7, rtol=1e-7)


def test_empty_and_bias_only_cases_return_diagnostics_without_identifiability_claims():
    u, meta = projector(torch.empty(0, 8), 'budget', rank=4)
    assert u.shape == (8, 0) and meta['effective_head_rank'] == 0
    v = identifiability(torch.empty(0, 4), torch.empty(0, 0), torch.empty(0, dtype=torch.long),
                        torch.empty(0), .1, .01, full_head_dim=8)
    assert v['data_only_spatial_info_rank'] == 0 and not v['diagnostic_is_training_gate']
    j, f, y, m = sample(n=4, p=8)
    v = identifiability(j, f[:, :0], y, m, .1, .01, full_head_dim=8)
    assert v['weighted_head_span_rank'] == 1
    assert not v['necessary_observation_rank_condition']


def test_saturated_margin_does_not_create_false_observation_information():
    j, f, y, _ = sample(n=4, p=8)
    d = identifiability(j, f, y, torch.full((4,), 1000.), .1, .01)
    assert d['positive_curvature_rows'] == 0 and d['data_only_spatial_info_rank'] == 0
    assert not d['necessary_observation_rank_condition']


@pytest.mark.parametrize('policy,basis', [('bad', 'pca'), ('fixed', 'unknown')])
def test_unknown_projector_configuration_is_visible(policy, basis):
    with pytest.raises(ValueError):
        projector(torch.ones(4, 8), policy, 4, basis=basis)
