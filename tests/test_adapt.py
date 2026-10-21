"""Synthetic software tests only: no real EEG pilot or training result."""
from dataclasses import replace
import inspect

import pytest
import torch
from torch import nn

from p3work.adapt import (CalibrationPrefix, FrozenSourceModel, InnerConfig, PredictionInputs,
    adapted_logits, balanced_bce, fit_target, module_state_sha256, predict, zero_fast_weights)
from p3work.methods.meta_train import MetaTrainEpisode, episode_objective, outer_step
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary, positive_qr


torch.set_num_threads(1)
SHA = "a" * 64  # Explicitly synthetic fixture, never a real protocol freeze.


def setup(dtype=torch.float64):
    torch.manual_seed(20261018)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * 6, 5), nn.Tanh(), nn.BatchNorm1d(5), nn.Dropout(0.4)).to(dtype)
    head = nn.Linear(5, 2).to(dtype)
    source = FrozenSourceModel(encoder, head, feature_dim=5)
    geom = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
    b = initialize_dictionary(2, 20261018)
    x = torch.randn(24, 11, 6, dtype=dtype)
    x = x - x.mean(1, keepdim=True)
    y = torch.tensor([0, 1] * 12)
    # Only 4 valid support trials out of the 20 acquired: invalid trials still cost.
    support = CalibrationPrefix(x[:4], y[:4], tuple(f"s{i}" for i in range(4)), tuple(f"s{i}" for i in range(20)), SHA)
    inputs = PredictionInputs(x[4:], tuple(f"q{i}" for i in range(20)), SHA)
    return source, geom, b, support, inputs, y[4:]


def test_zero_replay_original_logits_and_common_reference_offsets_float32():
    source, geometry, b, support, _, _ = setup(torch.float32)
    weights = zero_fast_weights(source, 2, device=b.device)
    old = source.source_logits(support.x)
    new = adapted_logits(source, geometry, b, support.x, weights)
    torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-6)
    assert torch.equal(new.argmax(1), old.argmax(1))
    torch.manual_seed(3)
    for scale in (0.1, 1.0, 10.0):
        offset = torch.randn(4, 1, 6, dtype=torch.float64) * scale
        replay = adapted_logits(source, geometry, b, support.x.double() + offset, weights)
        torch.testing.assert_close(replay, old, atol=2e-6, rtol=2e-6)


def test_target_fit_exact_ten_steps_and_frozen_parameters_and_buffers():
    source, geometry, b, support, inputs, _ = setup()
    before = module_state_sha256(source)
    source.train()  # A parent training-mode transition must not enable BN/dropout.
    assert not any(m.training for m in source.modules())
    fitted = fit_target(source, geometry, b, support)
    assert fitted.updates == 10 and fitted.fallback_reason is None
    assert fitted.trainable_parameters == 2 + 5 + 1
    assert fitted.weights.theta.norm() <= 0.15 + 1e-15
    result = predict(source, geometry, b, fitted, inputs)
    assert torch.equal(result.labels, result.logits.argmax(1))
    assert torch.equal(result.probabilities, result.logits.softmax(1))
    assert module_state_sha256(source) == before
    assert all(p.grad is None and not p.requires_grad for p in source.parameters())


def test_missing_class_falls_back_to_exact_source_and_does_not_replenish():
    source, geometry, b, support, inputs, _ = setup()
    support = replace(support, y=torch.zeros(4, dtype=torch.int64))
    fitted = fit_target(source, geometry, b, support)
    assert fitted.fallback_reason == "insufficient_class_support" and fitted.updates == 0
    assert len(fitted.acquired_event_uids) == 20 and len(fitted.support_event_uids) == 4
    assert torch.equal(predict(source, geometry, b, fitted, inputs).logits, source.source_logits(inputs.x))


def test_label_visibility_prefix_order_and_version_fail_closed():
    source, geometry, b, support, inputs, labels = setup()
    assert "query" not in inspect.signature(fit_target).parameters
    assert "y" not in PredictionInputs.__dataclass_fields__
    with pytest.raises(TypeError):
        fit_target(source, geometry, b, inputs)
    with pytest.raises(ValueError):
        replace(support, event_uids=tuple(reversed(support.event_uids)))
    with pytest.raises(PermissionError):
        replace(support, role="external_query")
    with pytest.raises(PermissionError):
        fit_target(source, geometry, b, replace(support, role="meta_train_support"))
    fitted = fit_target(source, geometry, b, support)
    with pytest.raises(ValueError):
        predict(source, geometry, b, fitted, replace(inputs, protocol_freeze_sha="b" * 64))
    with pytest.raises(ValueError):
        predict(source, geometry, b, fitted, replace(inputs, event_uids=("s19",) + inputs.event_uids[1:]))
    with pytest.raises(ValueError):
        predict(source, geometry, initialize_dictionary(2, 19), fitted, inputs)
    with pytest.raises(RuntimeError):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            fit_target(source, geometry, b, support)
    episode = MetaTrainEpisode(replace(support, role="meta_train_support"), inputs.x, labels,
                              inputs.event_uids, SHA, SHA)
    with pytest.raises(PermissionError):
        replace(episode, role="external_query")
    with pytest.raises(PermissionError):
        replace(episode, dataset="HS-Stroke")


def test_balanced_loss_and_tie_prediction_use_the_same_two_logits():
    logits = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]], dtype=torch.float64)
    y = torch.tensor([0, 0, 1])
    each = nn.functional.binary_cross_entropy_with_logits(logits[:, 1] - logits[:, 0], y.double(), reduction="none")
    torch.testing.assert_close(balanced_bce(logits, y), 0.5 * (each[:2].mean() + each[2]))
    source, geometry, b, support, inputs, _ = setup()
    with torch.no_grad():
        for p in source.source_head.parameters():
            p.zero_()
    source.frozen_state_sha256 = module_state_sha256(source)
    fitted = fit_target(source, geometry, b, support, method="B0")
    result = predict(source, geometry, b, fitted, inputs)
    assert torch.equal(result.labels, torch.zeros(20, dtype=torch.int64))
    assert torch.equal(result.probabilities, torch.full((20, 2), 0.5, dtype=torch.float64))


def test_head_additive_random_and_pct_kernels_and_counts():
    source, geometry, b, support, inputs, _ = setup()
    for method in ("B1", "B8", "B9", "P+PCT"):
        fitted = fit_target(source, geometry, b, support, method=method)
        assert fitted.updates == 10
        assert fitted.trainable_parameters == (6 if method == "B1" else 8)
        assert torch.isfinite(predict(source, geometry, b, fitted, inputs).logits).all()
        if method == "B1":
            assert torch.equal(fitted.weights.theta, torch.zeros_like(fitted.weights.theta))
    with pytest.raises(ValueError):
        fit_target(source, geometry, b, support, method="P", config=InnerConfig(pct_weight=0.1))
    with pytest.raises(NotImplementedError):
        fit_target(source, geometry, b, support, method="B10")


def test_full_ten_step_inner_to_outer_b_direction_gradient_finite_difference():
    source, geometry, b, support, inputs, labels = setup()
    b = b.requires_grad_(True)
    episode = MetaTrainEpisode(replace(support, role="meta_train_support"), inputs.x, labels,
                              inputs.event_uids, SHA, SHA)
    objective = episode_objective(source, geometry, b, episode)
    gradient, = torch.autograd.grad(objective, b)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-10
    torch.manual_seed(8)
    v = torch.randn_like(b)
    tangent = v - b @ (0.5 * (b.T @ v + v.T @ b))
    tangent = tangent / tangent.norm()
    analytic = (gradient * tangent).sum()
    h = 1e-4
    plus = episode_objective(source, geometry, positive_qr(b.detach() + h * tangent.detach()), episode)
    minus = episode_objective(source, geometry, positive_qr(b.detach() - h * tangent.detach()), episode)
    finite = (plus - minus) / (2 * h)
    torch.testing.assert_close(analytic, finite, atol=2e-8, rtol=2e-3)
    source.assert_unchanged()


def test_one_outer_step_is_full_unroll_and_retracts_with_four_episodes():
    source, geometry, b, support, inputs, labels = setup()
    episode = MetaTrainEpisode(replace(support, role="meta_train_support"), inputs.x, labels,
                              inputs.event_uids, SHA, SHA)
    # Repeating one synthetic fixture tests the batch operator, not subject sampling.
    bnext, receipt = outer_step(source, geometry, b.requires_grad_(True), (episode,) * 4)
    torch.testing.assert_close(bnext.T @ bnext, torch.eye(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert receipt["full_unroll"] is True and receipt["inner_steps"] == 10
    with pytest.raises(ValueError):
        outer_step(source, geometry, b, (episode,))
