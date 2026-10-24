"""Mathematical properties of finite-support risk objectives, synthetic only."""
import math

import pytest
import torch

from paper3r7.risk import (class_balanced_weights, entropic_risk,
                          pointwise_losses, risk_diagnostics, support_risk)


def fixture():
    source = torch.tensor([[2., -1.], [.1, .5], [-2., 1.], [1., -.4],
                           [-1., .1], [.2, -.1], [.7, .2]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    candidate = source + torch.tensor([[.1, -.3], [.2, -.1], [.4, -.4], [-.3, .1],
                                      [-.2, .2], [.1, -.1], [-.1, .3]], dtype=torch.float64)
    return candidate.requires_grad_(), source.requires_grad_(), labels


@pytest.mark.parametrize("temperature", [.01, .5, 100., 1e14])
def test_identity_relative_risk_is_exact_zero(temperature):
    _, source, y = fixture()
    result = support_risk(source, source, y, temperature=temperature)
    assert result.item() == 0.
    diagnostic = risk_diagnostics(source, source, y, temperature=temperature)
    assert diagnostic["support_certificate_holds"]
    assert diagnostic["earned_kl_radius"] == 0.


def test_class_balancing_retains_half_mass_under_unequal_counts():
    candidate, source, y = fixture()
    q = class_balanced_weights(y)
    assert q[y == 0].sum() == pytest.approx(.5)
    assert q[y == 1].sum() == pytest.approx(.5)
    losses = pointwise_losses(candidate, y)
    expected = .5 * (losses[y == 0].mean() + losses[y == 1].mean())
    torch.testing.assert_close(support_risk(candidate, source, y, kind="ce"), expected)
    assert not torch.isclose(expected, losses.mean())


def test_large_temperature_recovers_mean_regret_value_and_ce_gradient():
    candidate, source, y = fixture()
    q = class_balanced_weights(y)
    mean = (q * (pointwise_losses(candidate, y) - pointwise_losses(source.detach(), y))).sum()
    actual = support_risk(candidate, source, y, temperature=1e12)
    torch.testing.assert_close(actual, mean, atol=3e-14, rtol=1e-12)
    first = torch.autograd.grad(actual, candidate, retain_graph=True)[0]
    second = torch.autograd.grad(support_risk(candidate, source, y, kind="ce"), candidate)[0]
    torch.testing.assert_close(first, second, atol=3e-14, rtol=1e-12)


@pytest.mark.parametrize("temperature", [.03, .5, 20., 1e14])
def test_constant_source_loss_reduces_relative_to_absolute_plus_constant(temperature):
    candidate, _, y = fixture()
    source = torch.zeros_like(candidate).requires_grad_()
    relative = support_risk(candidate, source, y, temperature=temperature)
    absolute = support_risk(candidate, source, y, kind="absolute_dro", temperature=temperature)
    torch.testing.assert_close(relative, absolute - math.log(2), atol=5e-15, rtol=2e-13)
    first = torch.autograd.grad(relative, candidate, retain_graph=True)[0]
    second = torch.autograd.grad(absolute, candidate)[0]
    torch.testing.assert_close(first, second, atol=3e-15, rtol=2e-13)


def test_donsker_varadhan_adversary_attains_the_variational_objective():
    candidate, source, y = fixture()
    q = class_balanced_weights(y)
    delta = pointwise_losses(candidate, y) - pointwise_losses(source.detach(), y)
    tau = .37
    p = (q.log() + delta / tau).softmax(0)
    kl = (p * (p.log() - q.log())).sum()
    primal = (p * delta).sum() - tau * kl
    dual = support_risk(candidate, source, y, temperature=tau)
    torch.testing.assert_close(primal, dual, atol=2e-15, rtol=1e-13)
    # Arbitrary reweighting obeys the bound, not just the Gibbs maximizer.
    other = torch.arange(1, len(q) + 1, dtype=torch.float64)
    other = other / other.sum()
    other_kl = (other * (other.log() - q.log())).sum()
    assert (other * delta).sum() <= dual + tau * other_kl + 1e-14


def test_variance_regret_is_second_order_expansion_with_cubic_remainder():
    values = torch.tensor([-.4, .2, .7, 1.4], dtype=torch.float64)
    q = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    mean = (q * values).sum()
    variance = (q * (values - mean).square()).sum()
    errors = []
    for tau in (10., 20., 40.):
        exact = entropic_risk(values, q, tau)
        second = mean + variance / (2 * tau)
        errors.append(abs(float(exact - second)))
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_variance_regret_api_matches_weighted_formula():
    candidate, source, y = fixture()
    q = class_balanced_weights(y)
    delta = pointwise_losses(candidate, y) - pointwise_losses(source.detach(), y)
    mean = (q * delta).sum()
    expected = mean + (q * (delta - mean).square()).sum() / 1.4
    torch.testing.assert_close(support_risk(candidate, source, y, kind="variance_regret", temperature=.7), expected)


@pytest.mark.parametrize("kind", ["ce", "absolute_dro", "relative_dro", "variance_regret"])
def test_reference_is_detached_but_candidate_retains_first_and_second_derivatives(kind):
    candidate, source, y = fixture()
    result = support_risk(candidate, source, y, kind=kind)
    grad, reference_grad = torch.autograd.grad(result, (candidate, source), allow_unused=True, create_graph=True)
    assert reference_grad is None
    assert torch.isfinite(grad).all() and grad.norm() > 0
    second = torch.autograd.grad(grad.square().sum(), candidate)[0]
    assert torch.isfinite(second).all() and second.norm() > 0


@pytest.mark.parametrize("kind", ["ce", "absolute_dro", "relative_dro", "variance_regret"])
def test_extreme_finite_logits_are_not_probability_clipped(kind):
    y = torch.tensor([0, 1])
    candidate = torch.tensor([[-500., 500.], [500., -500.]], dtype=torch.float64, requires_grad=True)
    source = torch.zeros_like(candidate)
    result = support_risk(candidate, source, y, kind=kind, temperature=.001)
    assert torch.isfinite(result)
    expected = 1000. if kind in ("ce", "absolute_dro") else 1000. - math.log(2)
    assert float(result.detach()) == pytest.approx(expected, abs=1e-9)
    assert torch.isfinite(torch.autograd.grad(result, candidate)[0]).all()


def test_diagnostic_preserves_negative_earned_radius_when_certificate_fails():
    y = torch.tensor([0, 1])
    source = torch.zeros((2, 2), dtype=torch.float64)
    wrong = torch.tensor([[-2., 2.], [2., -2.]], dtype=torch.float64)
    diagnostic = risk_diagnostics(wrong, source, y, temperature=.5, ridge=.1, kl_radius=.2)
    assert diagnostic["earned_kl_radius"] < 0
    assert not diagnostic["earned_kl_radius_nonnegative"]
    assert not diagnostic["support_certificate_holds"]
    assert diagnostic["certificate_bound_if_nonpositive"] is None
    assert diagnostic["support_only"] and not diagnostic["future_risk_guarantee"]


def test_diagnostic_reports_attained_support_bound_without_future_guarantee():
    y = torch.tensor([0, 1])
    source = torch.zeros((2, 2), dtype=torch.float64)
    better = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    diagnostic = risk_diagnostics(better, source, y, temperature=.5, ridge=.1, kl_radius=.2)
    assert diagnostic["support_certificate_holds"]
    assert diagnostic["earned_kl_radius"] > 0
    assert diagnostic["certificate_bound_if_nonpositive"] == pytest.approx(0., abs=1e-8)
    assert diagnostic["variational_regret_bound_at_requested_kl"] < 0
    assert diagnostic["log_exponential_moment"] < 0


def test_meta_parameter_gradient_matches_directional_difference():
    _, source, y = fixture()
    x = torch.tensor([-.8, -.5, -.2, .1, .3, .7, 1.1], dtype=torch.float64)
    parameter = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([.3, -.7], dtype=torch.float64)
    def objective(value):
        margin = value[0] * x + value[1] * x.square()
        logits = source.detach() + torch.stack((-margin / 2, margin / 2), 1)
        return support_risk(logits, source, y)
    gradient = torch.autograd.grad(objective(parameter), parameter)[0]
    epsilon = 1e-5
    difference = (objective(parameter.detach() + epsilon * direction) -
                  objective(parameter.detach() - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(gradient @ direction, difference, atol=1e-10, rtol=1e-7)
    assert gradient.norm() > 0


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature_is_a_task_input_error(temperature):
    candidate, source, y = fixture()
    with pytest.raises(ValueError, match="positive risk temperature"):
        support_risk(candidate, source, y, temperature=temperature)


def test_missing_class_is_explicit_for_callers_to_handle_as_fallback():
    logits = torch.zeros((4, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="Both support classes"):
        support_risk(logits, logits, torch.zeros(4, dtype=torch.long))
