"""Numerical/method-contract checks on synthetic EEG, not scientific results."""
from dataclasses import replace
from types import SimpleNamespace
import inspect
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)
from p3work.adapt import (CalibrationPrefix, FrozenSourceModel, InnerConfig,
                          FastWeights, inner_unroll, adapted_logits, balanced_bce)
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary
from paper3r6.algorithm import (Config, VARIANTS, fit, predict, support_tangent,
    ball_quadratic, information, _propose, _blocks, _actual, _bernoulli_kl, _class_weights)


def fixture(*, double=True, n=20):
    torch.manual_seed(711)
    model = CompactEEGNet(11, 128)
    if double:
        model = model.double()
    source = FrozenSourceModel(torch.nn.Sequential(model.features, torch.nn.Flatten(1)),
        model.classifier, feature_dim=64, input_layout="N1CT")
    factor = torch.randn(10, 10, dtype=torch.float64)
    reference = factor @ factor.T + .5 * torch.eye(10, dtype=torch.float64)
    reference = reference / reference.trace()
    geometry = SpatialGeometry.from_reference(reference)
    x = torch.randn(24, 11, 128, dtype=torch.float64)
    x -= x.mean(1, keepdim=True); x -= x.mean(2, keepdim=True)
    ids = tuple(f"prefix-{i}" for i in range(n))
    support = CalibrationPrefix(x[:n], torch.arange(n) % 2, ids, ids, "a" * 64,
                                "meta_train_support")
    b = initialize_dictionary(4, 711).requires_grad_(True)
    return source, geometry, support, b, x[20:]


def test_cached_identity_tangent_matches_direct_autograd_with_rms_and_recoloring():
    source, geometry, support, b, _ = fixture()
    m0, c, features = support_tangent(source, geometry, support)
    theta = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    def margin(t):
        weights = FastWeights(t, t.new_zeros(64), t.new_zeros(()))
        logits = _actual(source, geometry, b, weights, support.x)[0]
        return logits[:, 1] - logits[:, 0]
    actual = torch.autograd.functional.jacobian(margin, theta)
    torch.testing.assert_close(actual, c @ b, atol=2e-12, rtol=2e-10)
    torch.testing.assert_close(m0, margin(theta), atol=2e-12, rtol=2e-10)
    assert c.shape == (20, 99) and features.shape == (20, 64)
    assert not c.requires_grad and not features.requires_grad and not m0.requires_grad


def test_schur_and_full_joint_steps_are_equivalent_interior_not_new_models():
    torch.manual_seed(23)
    j = torch.randn(12, 4, dtype=torch.float64)
    z = torch.randn(12, 7, dtype=torch.float64)
    theta = torch.randn(4, dtype=torch.float64) * .02
    head = torch.randn(7, dtype=torch.float64) * .02
    margin = torch.randn(12, dtype=torch.float64)
    y = torch.arange(12) % 2
    config = Config()
    radius = j.new_tensor(1000.)
    joint = _propose(j, z, margin, y, theta, head, radius, config, mode="joint")
    schur = _propose(j, z, margin, y, theta, head, radius, config, mode="profiled")
    for a, b in zip(joint, schur):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("boundary", [False, True])
def test_quadratic_ball_first_derivative_matches_finite_difference(boundary):
    torch.manual_seed(3)
    factor = torch.randn(4, 4, dtype=torch.float64)
    h = (factor @ factor.T + torch.eye(4, dtype=torch.float64)).requires_grad_(True)
    rhs = (torch.randn(4, dtype=torch.float64) * 3).requires_grad_(True)
    radius = torch.tensor(.12 if boundary else 100., dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ball_quadratic, (h, rhs, radius), eps=1e-6,
                                   atol=1e-6, rtol=1e-4)
    result = ball_quadratic(h, rhs, radius)
    assert float(result.detach().norm()) <= float(radius.detach()) + 1e-12
    if boundary:
        torch.testing.assert_close(result.norm(), radius, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_cpu_spectral_root_matches_independent_full_matrix_solve(rank):
    torch.manual_seed(29 + rank)
    rotation = torch.linalg.qr(torch.randn(rank, rank, dtype=torch.float64)).Q
    values = torch.logspace(-5, 1, rank, dtype=torch.float64)
    h = rotation @ torch.diag(values) @ rotation.T
    rhs = torch.randn(rank, dtype=torch.float64)
    radius = .07
    identity = torch.eye(rank, dtype=torch.float64)
    lo, hi = 0., float(rhs.norm()) / radius
    for _ in range(80):
        mid = (lo + hi) / 2
        full = torch.linalg.solve(h + mid * identity, rhs)
        if full.norm() > radius:
            lo = mid
        else:
            hi = mid
    reference = torch.linalg.solve(h + hi * identity, rhs)
    actual = ball_quadratic(h, rhs, radius)
    torch.testing.assert_close(actual, reference, atol=2e-13, rtol=2e-11)
    assert float(actual.norm()) <= radius + 1e-13


def test_penalty_curvature_and_balanced_conditional_fisher_are_exact():
    torch.manual_seed(4)
    j, z = torch.randn(6, 4, dtype=torch.float64), torch.randn(6, 5, dtype=torch.float64)
    theta, head = torch.randn(4, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
    margin, y = torch.randn(6, dtype=torch.float64), torch.tensor([0, 0, 0, 0, 1, 1])
    cfg = Config(damping=0.)
    gt, gh, htt, cross, hhh = _blocks(j, z, margin, y, theta, head, cfg)
    a = torch.tensor([1/8] * 4 + [1/4] * 2, dtype=torch.float64)
    p = margin.sigmoid(); d = a * p * (1-p)
    torch.testing.assert_close(gt, j.T @ (a * (p-y)) + .02 * theta)
    torch.testing.assert_close(gh, z.T @ (a * (p-y)) + .0004 * head)
    torch.testing.assert_close(htt, j.T @ (d[:, None] * j) + .02 * torch.eye(4, dtype=j.dtype))
    torch.testing.assert_close(hhh, z.T @ (d[:, None] * z) + .0004 * torch.eye(5, dtype=j.dtype))


def test_head_interpolation_can_destroy_penalized_geometry_information():
    torch.manual_seed(19)
    j, z = torch.randn(12, 4, dtype=torch.float64), torch.randn(12, 18, dtype=torch.float64)
    margin, y = j.new_zeros(12), torch.arange(12) % 2
    weak, de_weak = information(j, z, margin, y, Config(head_l2=1e-10))
    strong, de_strong = information(j, z, margin, y, Config(head_l2=1.))
    assert de_weak < de_strong and float(de_weak) < 1e-6
    assert float(torch.linalg.eigvalsh(weak).min()) > -1e-12
    assert torch.trace(strong) > torch.trace(weak)


def test_sgd10_preserves_original_fitting_and_actual_prediction_path():
    source, geometry, support, b, query = fixture(double=False)
    expected, _, _ = inner_unroll(source, geometry, b, support, config=InnerConfig())
    actual = fit(source, geometry, b, support, Config(variant="sgd10"))
    for name in ("theta", "delta_w", "delta_b"):
        torch.testing.assert_close(getattr(expected, name), getattr(actual.weights, name), atol=1e-9, rtol=1e-6)
    torch.testing.assert_close(predict(source, geometry, b, actual, query),
        adapted_logits(source, geometry, b, query, expected), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_variants_are_finite_freeze_source_and_respect_radius(variant):
    source, geometry, support, b, query = fixture(double=False)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    cfg = Config(variant=variant, tangent_steps=2, head_l2=.1)
    fitted = fit(source, geometry, b, support, cfg)
    assert not fitted.fallback_reason
    assert torch.isfinite(predict(source, geometry, b, fitted, query)).all()
    assert float(fitted.weights.theta.norm()) <= cfg.radius + 1e-10
    if variant in ("tangent_information", "information_protected"):
        assert float(fitted.weights.theta.norm()) <= fitted.diagnostics["effective_radius"] + 1e-10
    for key, value in state.items():
        assert torch.equal(value, source.state_dict()[key])


@pytest.mark.parametrize("variant", ["tangent_profiled", "tangent_information"])
def test_complete_learner_is_invariant_to_orthogonal_dictionary_coordinate_change(variant):
    source, geometry, support, b, query = fixture()
    o = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    cfg = Config(variant=variant, tangent_steps=3, radius=.002)
    first = fit(source, geometry, b, support, cfg)
    second = fit(source, geometry, b @ o, support, cfg)
    torch.testing.assert_close(second.weights.theta, o.T @ first.weights.theta, atol=1e-11, rtol=1e-7)
    torch.testing.assert_close(second.weights.delta_w, first.weights.delta_w, atol=1e-10, rtol=1e-7)
    torch.testing.assert_close(predict(source, geometry, b @ o, second, query),
                               predict(source, geometry, b, first, query), atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize("variant", ["tangent_profiled", "tangent_information"])
def test_boundary_meta_gradient_through_support_solver_matches_directional_difference(variant):
    source, geometry, support, b, query = fixture()
    cfg = Config(variant=variant, radius=1e-5, tangent_steps=3, head_l2=.1)
    def value(dictionary):
        state = fit(source, geometry, dictionary, support, cfg, create_graph=True)
        return predict(source, geometry, dictionary, state, query).square().mean()
    loss = value(b)
    gradient = torch.autograd.grad(loss, b)[0]
    direction = torch.randn_like(b); direction = direction / direction.norm()
    eps = 2e-5
    numerical = (value(b + eps*direction).detach() - value(b - eps*direction).detach()) / (2*eps)
    analytic = (gradient * direction).sum()
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    torch.testing.assert_close(analytic, numerical, atol=2e-8, rtol=2e-3)


def test_information_scale_changes_only_radius_evidence_not_data_schur():
    source, geometry, support, b, _ = fixture()
    m0, c, features = support_tangent(source, geometry, support)
    from paper3r6.algorithm import _augmented
    schur_a, effective_a = information(c @ b, _augmented(features), m0, support.y,
        Config(head_l2=.1, information_scale=.5))
    schur_b, effective_b = information(c @ b, _augmented(features), m0, support.y,
        Config(head_l2=.1, information_scale=2.))
    torch.testing.assert_close(schur_a, schur_b)
    assert effective_a > effective_b


def test_four_episode_outer_gradient_and_stiefel_update_are_finite():
    from p3work.spatial_geometry import stiefel_step
    source, geometry, support, b, query = fixture(double=False)
    cfg = Config(variant="tangent_information", radius=1e-5,
                 tangent_steps=3, head_l2=.1)
    gradient = torch.zeros_like(b)
    for index in range(4):
        # Different support order/labels, no query statistics enter fit.
        episode_support = replace(support, y=(support.y + index) % 2)
        fitted = fit(source, geometry, b, episode_support, cfg, create_graph=True)
        pred = predict(source, geometry, b, fitted, query)
        loss = balanced_bce(pred, torch.arange(len(query)) % 2)
        gradient += torch.autograd.grad(loss / 4, b)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 0
    updated = stiefel_step(b, gradient, learning_rate=.01)
    torch.testing.assert_close(updated.T @ updated, torch.eye(4, dtype=b.dtype), atol=1e-10, rtol=1e-10)


def test_source_anchored_protection_checks_actual_nonlinear_support_predictions():
    source, geometry, support, b, _ = fixture()
    cfg = Config(variant="tangent_protected", tangent_steps=4, kl_epsilon=1e-9)
    fitted = fit(source, geometry, b, support, cfg, create_graph=True)
    pred = predict(source, geometry, b, fitted, support.x)
    base = source.source_logits(support.x).detach()
    kl = (_class_weights(support.y) * _bernoulli_kl(
        base[:, 1] - base[:, 0], pred[:, 1] - pred[:, 0])).sum()
    assert float(kl.detach()) <= cfg.kl_epsilon + 1e-12
    assert fitted.protection_fraction < 1
    assert torch.isfinite(torch.autograd.grad(pred.square().mean(), b)[0]).all()


def test_single_class_fallback_preserves_source_predictions():
    source, geometry, support, b, query = fixture()
    single = replace(support, y=torch.zeros_like(support.y))
    fitted = fit(source, geometry, b, single, Config())
    assert fitted.fallback_reason == "insufficient_class_support"
    torch.testing.assert_close(predict(source, geometry, b, fitted, query), source.source_logits(query))
    assert "query" not in inspect.signature(fit).parameters


def test_posthoc_relinearized_variant_refuses_unsupported_outer_derivatives():
    source, geometry, support, b, _ = fixture()
    with pytest.raises(ValueError, match="posthoc"):
        fit(source, geometry, b, support, Config(variant="relinearized"), create_graph=True)
