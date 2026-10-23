"""Mechanism controls and functional gate optimizer; synthetic software only."""
import io
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from paper3v2.algorithm import (Config, gate_adam, init_gate, fit, gated_matrix,
                               pct_weights, temporal_loss, outer_update,
                               fit_reference, reference_logits, REFERENCE_VARIANTS)
from test_algorithm import fixture


def test_mean_teacher_control_preserves_weight_mass_not_assignment():
    teacher = torch.tensor([[4., -4.], [-.3, .3], [1., -1.], [-3., 3.]], dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 1])
    soft = pct_weights(teacher, y, 'softPCT')
    control = pct_weights(teacher, y, 'softPCT_mean')
    assert soft.std() > .1
    assert torch.equal(control, control[0].expand_as(control))
    torch.testing.assert_close(control.sum(), soft.sum(), atol=1e-14, rtol=1e-14)
    # The total penalty generally differs when sample residuals differ, which is
    # precisely the assignment effect the control is intended to isolate.
    residual = torch.tensor([.01, 5., .1, 2.], dtype=torch.float64)
    assert abs(float((soft * residual).mean() - (control * residual).mean())) > .01


def test_runmean_control_distinguishes_equal_runs_from_trial_pooling():
    pred = torch.tensor([[2., -2.], [-2., 2.],
                         [-2., 2.], [2., -2.], [-2., 2.], [2., -2.], [-2., 2.], [2., -2.]])
    y = torch.arange(8) % 2
    runs = ('a', 'a', 'b', 'b', 'b', 'b', 'b', 'b')
    pooled = F.cross_entropy(pred, y)
    equal_run = (F.cross_entropy(pred[:2], y[:2]) + F.cross_entropy(pred[2:], y[2:])) / 2
    controlled = temporal_loss(pred, y, runs, Config(variant='runmean'))
    torch.testing.assert_close(controlled, .75 * pooled + .25 * equal_run)
    assert abs(float(controlled - pooled)) > .1
    assert temporal_loss(pred, y, runs, Config(variant='temporal')) > controlled


@pytest.mark.parametrize('variant,expected', [('symonly', [1., 0.]), ('skewonly', [0., 1.])])
def test_fixed_geometry_ablations_have_only_the_named_component(variant, expected):
    _, source, geometry, e, b = fixture()
    c = Config(variant=variant, inner_steps=2)
    weights, alpha, fallback = fit(source, geometry, b, init_gate(c, 'cpu'), e.support, c)
    torch.testing.assert_close(alpha, torch.tensor(expected, dtype=alpha.dtype))
    assert fallback is None
    theta = torch.tensor([.03, -.04, .01, .02], dtype=torch.float64)
    m = gated_matrix(geometry, b, theta, alpha)
    torch.testing.assert_close(m, m.T if variant == 'symonly' else -m.T, atol=1e-14, rtol=1e-14)
    assert float(m.norm().detach()) <= float(theta.norm()) + 1e-12


def test_functional_gate_adam_matches_torch_adam_and_does_not_mutate_state():
    torch.manual_seed(5)
    c = Config(variant='dual')
    g = init_gate(c, 'cpu')
    reference = torch.nn.Parameter(g.detach().clone())
    optimizer = torch.optim.Adam([reference], lr=c.gate_lr,
                                betas=(c.gate_beta1, c.gate_beta2), eps=c.gate_eps)
    state = None
    for magnitude in (1e-8, 1e-5, .05, 0., .2):
        gradient = torch.randn_like(g) * magnitude
        old_g = g.detach().clone()
        old_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()} if state else None
        next_g, next_state = gate_adam(g, gradient, state, c)
        reference.grad = gradient.clone(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(next_g, reference, atol=1e-14, rtol=1e-14)
        assert torch.equal(g, old_g)
        if state:
            for k in state:
                assert torch.equal(state[k], old_state[k]) if torch.is_tensor(state[k]) else state[k] == old_state[k]
        g, state = next_g, next_state
    assert state['step'] == 5


def test_adam_full_meta_update_resume_preserves_actual_moments():
    _, source, geometry, e, b = fixture()
    c = Config(variant='dual', inner_steps=2)
    g = init_gate(c, 'cpu')
    b, g, metrics, state = outer_update(source, geometry, b, g, [e], c)
    assert metrics['gate_parameter_delta_norm'] > 1e-6
    assert state['step'] == 1 and state['exp_avg'].norm() > 0
    stream = io.BytesIO()
    torch.save({'b': b.detach(), 'g': g.detach(), 'optimizer': state}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    continuous = outer_update(source, geometry, b, g, [e], c, gate_state=state)
    resumed = outer_update(source, geometry, restored['b'].requires_grad_(True),
                           restored['g'].requires_grad_(True), [e], c, gate_state=restored['optimizer'])
    assert torch.equal(continuous[0], resumed[0])
    assert torch.equal(continuous[1], resumed[1])
    for key in ('exp_avg', 'exp_avg_sq'):
        assert torch.equal(continuous[3][key], resumed[3][key])
    assert continuous[3]['step'] == resumed[3]['step'] == 2


def test_empty_gate_needs_no_optimizer_state():
    c = Config(variant='P'); g = init_gate(c, 'cpu')
    ng, state = gate_adam(g, torch.zeros_like(g), None, c)
    assert state is None and ng.numel() == 0


def test_head_only_reference_matches_original_zero_spatial_fit(monkeypatch):
    from p3work.adapt import inner_unroll, adapted_logits
    native, source, geometry, e, b = fixture()
    expected, steps, fallback = inner_unroll(native, geometry, b, e.support, head_only=True)
    calls = []
    original = source.features_and_logits
    def observed(x):
        calls.append(len(x))
        return original(x)
    monkeypatch.setattr(source, "features_and_logits", observed)
    state, metadata = fit_reference(source, geometry, e.support, "head_only")
    assert calls == [20]  # fixed support features computed once, no query seen
    assert steps == metadata["optimizer_updates"] == 10 and fallback is None
    assert expected.theta.norm() == 0 and metadata["fallback_reason"] is None
    torch.testing.assert_close(state["delta_w"], expected.delta_w, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(state["delta_b"], expected.delta_b, atol=1e-8, rtol=1e-6)
    actual = reference_logits(source, geometry, state, e.query_x)
    torch.testing.assert_close(actual, adapted_logits(native, geometry, b, e.query_x, expected),
                               atol=1e-7, rtol=1e-6)


def test_support_ea_matches_manual_covariance_mapping():
    from p3work.spatial_geometry import SpatialGeometry, joint_trial_rms
    _, source, _, e, _ = fixture()
    # Non-isotropic, non-diagonal reference detects accidentally whitening to I
    # or reversing the source square-root / support inverse-square-root order.
    factor = torch.randn(10, 10, dtype=torch.float64)
    reference = factor @ factor.T + torch.eye(10, dtype=torch.float64)
    reference /= reference.trace()
    geometry = SpatialGeometry.from_reference(reference)
    support = replace(e.support, x=e.support.x * torch.linspace(.2, 3., 20)[:, None, None])
    state, metadata = fit_reference(source, geometry, support, "support_EA")
    individual = []
    for x in support.x:
        z = geometry.q @ x
        c = z @ z.T
        individual.append(c / c.trace())
    regularized = .95 * torch.stack(individual).mean(0) + .005 * torch.eye(10, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(regularized)
    invsqrt = vectors @ torch.diag(values ** -.5) @ vectors.T
    expected_transform = geometry.q.T @ geometry.rref_sqrt @ invsqrt @ geometry.q
    torch.testing.assert_close(state["support_covariance"], regularized, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(state["transform"], expected_transform, atol=1e-12, rtol=1e-12)
    expected = source.features_and_logits(joint_trial_rms(expected_transform @ e.query_x))[1]
    torch.testing.assert_close(reference_logits(source, geometry, state, e.query_x), expected)
    assert metadata["uses_support_labels"] is False and metadata["optimizer_updates"] == 0


def test_support_ea_fit_is_label_and_amplitude_invariant_and_query_independent(monkeypatch):
    _, source, geometry, e, _ = fixture()
    def unexpected_source_access(_):
        pytest.fail("EA covariance fitting must not access source logits or query features")
    original = source.features_and_logits
    monkeypatch.setattr(source, "features_and_logits", unexpected_source_access)
    state, _ = fit_reference(source, geometry, e.support, "support_EA")
    other = replace(e.support, y=1 - e.support.y,
                    x=e.support.x * torch.linspace(.1, 4., 20)[:, None, None])
    changed, _ = fit_reference(source, geometry, other, "support_EA")
    torch.testing.assert_close(state["transform"], changed["transform"], atol=1e-12, rtol=1e-12)
    monkeypatch.setattr(source, "features_and_logits", original)
    before = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
    together = reference_logits(source, geometry, state, e.query_x)
    separate = torch.cat([reference_logits(source, geometry, state, q[None]) for q in e.query_x])
    torch.testing.assert_close(together, separate, atol=1e-7, rtol=1e-6)
    # Adversarial future inputs cannot change the previously fitted transform.
    reference_logits(source, geometry, state, 10 * e.query_x.flip(2))
    for key, value in before.items():
        assert torch.equal(state[key], value) if torch.is_tensor(value) else state[key] == value


@pytest.mark.parametrize("variant", REFERENCE_VARIANTS)
def test_references_preserve_source_and_retain_single_class_sessions(variant):
    _, source, geometry, e, _ = fixture()
    before = {key: value.clone() for key, value in source.state_dict().items()}
    support = replace(e.support, y=torch.zeros_like(e.support.y))
    state, metadata = fit_reference(source, geometry, support, variant)
    pred = reference_logits(source, geometry, state, e.query_x)
    assert torch.isfinite(pred).all() and pred.shape == (20, 2)
    if variant == "head_only":
        assert metadata["fallback_reason"] == "insufficient_class_support"
        assert torch.equal(pred, source.source_logits(e.query_x))
    else:
        assert metadata["fallback_reason"] is None  # EA uses no labels
    for key, value in before.items():
        assert torch.equal(source.state_dict()[key], value)


def test_support_ea_empty_prefix_has_explicit_frozen_source_fallback():
    _, source, geometry, e, _ = fixture()
    support = replace(e.support, x=e.support.x[:0], y=e.support.y[:0], event_uids=())
    state, metadata = fit_reference(source, geometry, support, "support_EA")
    assert metadata["fallback_reason"] == "empty_support"
    assert torch.equal(reference_logits(source, geometry, state, e.query_x), source.source_logits(e.query_x))
