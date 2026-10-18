"""Synthetic CPU software/math checks only, with no EEG or scientific fits."""
from dataclasses import replace
import inspect
from unittest.mock import patch

import pytest
import torch
from torch import nn

from p3work.adapt import (CalibrationPrefix, FrozenSourceModel, PredictionInputs,
                         balanced_bce, module_state_sha256)
from p3work.methods import lowrank
from p3work.methods.meta_train import MetaTrainEpisode
from p3work.spatial_geometry import SpatialGeometry, positive_qr


torch.set_num_threads(1)
SHA = "a" * 64  # Explicitly synthetic protocol/split/temporal fixture references.


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(20261018)
    # Local initialization uses a forked RNG scope, never a real source model.
    with torch.random.fork_rng():
        torch.manual_seed(20261018)
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * 6, 5), nn.Tanh(),
                                nn.BatchNorm1d(5), nn.Dropout(0.4)).to(dtype)
        head = nn.Linear(5, 2).to(dtype)
    source = FrozenSourceModel(encoder, head, feature_dim=5)
    diagonal = torch.linspace(0.05, 0.15, 10, dtype=torch.float64)
    geometry = SpatialGeometry.from_reference(torch.diag(diagonal))
    x = torch.randn(24, 11, 6, dtype=dtype, generator=generator)
    x = x - x.mean(1, keepdim=True)
    y = torch.tensor([0, 1] * 12)
    support = CalibrationPrefix(x[:4], y[:4], tuple(f"s{i}" for i in range(4)),
        tuple(f"s{i}" for i in range(20)), SHA)
    inputs = PredictionInputs(x[4:], tuple(f"q{i}" for i in range(20)), SHA)
    return source, geometry, support, inputs, y[4:]


def meta_episode(support, inputs, labels, suffix=""):
    return MetaTrainEpisode(replace(support, role="meta_train_support",
        event_uids=tuple(uid+suffix for uid in support.event_uids),
        acquired_event_uids=tuple(uid+suffix for uid in support.acquired_event_uids)),
        inputs.x, labels, tuple(uid+suffix for uid in inputs.event_uids), SHA, SHA)


def test_nine_grid_and_deterministic_thin_qr_without_global_rng_consumption():
    assert len(lowrank.LOWRANK_GRID) == len(set(lowrank.LOWRANK_GRID)) == 9
    torch.manual_seed(43)
    rng = torch.random.get_rng_state().clone()
    for rank in (1, 2, 4):
        u = lowrank.initialize_lowrank(rank, 20261018)
        assert u.shape == (10, rank) and u.dtype == torch.float64
        torch.testing.assert_close(u.T @ u, torch.eye(rank, dtype=torch.float64), atol=1e-14, rtol=0)
        assert torch.equal(u, lowrank.initialize_lowrank(rank, 20261018))
        assert not torch.equal(u, lowrank.initialize_lowrank(rank, 20261019))
    assert torch.equal(torch.random.get_rng_state(), rng)
    for kwargs in ({"rank": 3}, {"radius": 0.2}, {"steps": 9}, {"factor_lr": 0.01},
                   {"head_lr": 0.02}, {"pct_weight": 0.2}, {"pct_beta": 0.0}):
        with pytest.raises(ValueError):
            lowrank.LowRankConfig(**kwargs)


def test_zero_v_identity_replays_original_two_logits_and_reference_offset_float32():
    source, geometry, support, _, _ = fixture(torch.float32)
    u0 = lowrank.initialize_lowrank(2, 20261018)
    weights = lowrank.zero_lowrank_weights(source, u0)
    a, t, update = lowrank.lowrank_matrix(geometry, weights.u, weights.v)
    assert torch.equal(t, torch.eye(10, dtype=torch.float64))
    assert torch.equal(update, torch.zeros_like(update))
    torch.testing.assert_close(a, geometry.q.T @ geometry.q, atol=2e-15, rtol=2e-15)
    old = source.source_logits(support.x)
    new = lowrank.lowrank_logits(source, geometry, support.x, weights)
    torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-6)
    assert torch.equal(new.argmax(1), old.argmax(1))
    generator = torch.Generator().manual_seed(3)
    for scale in (0.1, 1.0, 10.0):
        offset = torch.randn(4, 1, 6, generator=generator, dtype=torch.float64) * scale
        new = lowrank.lowrank_logits(source, geometry, support.x.double() + offset, weights)
        torch.testing.assert_close(new, old, atol=2e-6, rtol=2e-6)


def test_other_valid_car_bases_are_rejected_in_favor_of_fixed_helmert_coordinates():
    source, geometry, support, _, _ = fixture()
    u0 = lowrank.initialize_lowrank(1, 20261018)
    weights = lowrank.zero_lowrank_weights(source, u0)
    rotation = torch.eye(10, dtype=torch.float64)
    rotation[0, 0] = -1
    alternate = replace(geometry, q=rotation @ geometry.q)
    alternate.validate()  # This is valid CAR geometry, but the protocol fixes Q.
    with pytest.raises(ValueError, match="fixed clinic11 Helmert"):
        lowrank.inner_unroll_lowrank(source, alternate, u0, support)
    with pytest.raises(ValueError, match="fixed clinic11 Helmert"):
        lowrank.lowrank_matrix(alternate, weights.u, weights.v)
    with pytest.raises(ValueError, match="fixed clinic11 Helmert"):
        lowrank.lowrank_logits(source, alternate, support.x, weights)


def test_projection_scales_v_only_using_product_norm_and_gives_singular_value_bound():
    generator = torch.Generator().manual_seed(4)
    u = torch.randn(10, 4, generator=generator, dtype=torch.float64, requires_grad=True) * 3
    v = torch.randn(10, 4, generator=generator, dtype=torch.float64, requires_grad=True) * 2
    before = (u @ v.T).norm()
    unew, vnew = lowrank.project_lowrank_update(u, v, 0.15)
    assert unew is u and torch.equal(unew, u)
    torch.testing.assert_close(vnew, v * (0.15 / before), atol=1e-16, rtol=1e-14)
    torch.testing.assert_close((unew @ vnew.T).norm(), torch.tensor(0.15, dtype=torch.float64), atol=1e-15, rtol=0)
    singular = torch.linalg.svdvals(torch.eye(10, dtype=torch.float64) + unew @ vnew.T)
    assert singular.min() >= 0.85 - 1e-14 and singular.max() <= 1.15 + 1e-14
    inactive_u, inactive_v = lowrank.project_lowrank_update(u, v * 1e-6, 0.15)
    assert inactive_u is u and torch.equal(inactive_v, v * 1e-6)
    gu, gv = torch.autograd.grad(vnew.square().sum(), (u, v), create_graph=True)
    assert torch.isfinite(gu).all() and torch.isfinite(gv).all()


def test_ten_simultaneous_steps_zero_first_u_gradient_and_frozen_source_state():
    source, geometry, support, inputs, _ = fixture()
    u0 = lowrank.initialize_lowrank(2, 20261018)
    before = module_state_sha256(source)
    snapshots = []
    original = lowrank.lowrank_logits
    def observe(source, geometry, x, weights):
        snapshots.append(tuple(t.detach().clone() for t in (weights.u, weights.v, weights.delta_w, weights.delta_b)))
        return original(source, geometry, x, weights)
    with patch.object(lowrank, "lowrank_logits", side_effect=observe):
        weights, updates, fallback = lowrank.inner_unroll_lowrank(source, geometry, u0, support,
            config=lowrank.LowRankConfig(rank=2))
    assert updates == len(snapshots) == 10 and fallback is None
    # At V=0, d(UV^T)/dU is exactly zero; V and head learn in that SAME step.
    assert torch.equal(snapshots[0][0], snapshots[1][0])
    assert snapshots[1][1].norm() > 0 and snapshots[1][2].norm() > 0
    assert not torch.equal(weights.u.detach(), u0)
    assert (weights.u @ weights.v.T).norm() <= 0.15 + 1e-14
    assert module_state_sha256(source) == before
    assert all(not p.requires_grad and p.grad is None for p in source.parameters())
    assert not any(m.training for m in source.modules())


@pytest.mark.parametrize("method", ["B6", "B7", "B10", "B11"])
def test_target_methods_fit_only_prefix_predict_from_same_logits_and_report_real_count(method):
    source, geometry, support, inputs, _ = fixture()
    config = lowrank.LowRankConfig(rank=4, pct_weight=0.1 if method in {"B7", "B11"} else 0.0)
    u0 = lowrank.initialize_lowrank(4, 20261018) if method in {"B10", "B11"} else None
    fitted = lowrank.fit_lowrank_target(source, geometry, support, method=method, config=config, u0=u0)
    assert fitted.method == method and fitted.updates == 10 and fitted.fallback_reason is None
    assert fitted.trainable_parameters == 20 * 4 + 5 + 1
    assert fitted.support_event_uids == support.event_uids
    assert fitted.acquired_event_uids == support.acquired_event_uids
    prediction = lowrank.predict_lowrank(source, geometry, fitted, inputs)
    assert torch.isfinite(prediction.logits).all()
    assert torch.equal(prediction.labels, prediction.logits.argmax(1))
    assert torch.equal(prediction.probabilities, prediction.logits.softmax(1))
    assert (fitted.weights.u @ fitted.weights.v.T).norm() <= config.radius + 1e-14
    source.assert_unchanged()


def test_pct_teacher_and_old_correct_mask_are_fixed_and_loss_is_two_logit_focal_matching():
    source, geometry, support, _, _ = fixture()
    u0 = lowrank.initialize_lowrank(1, 20261018)
    teacher = source.source_logits(support.x).detach()
    fixed = 1 + 5 * (teacher.argmax(1) == support.y)
    observed_weights, observed_losses = [], []
    original_logits, original_grad = lowrank.lowrank_logits, torch.autograd.grad
    def logits(source, geometry, x, weights):
        observed_weights.append(weights)
        return original_logits(source, geometry, x, weights)
    def gradients(loss, *args, **kwargs):
        observed_losses.append(loss.detach().clone())
        return original_grad(loss, *args, **kwargs)
    with patch.object(source, "source_logits", wraps=source.source_logits) as teacher_call, \
            patch.object(lowrank, "lowrank_logits", side_effect=logits), \
            patch.object(torch.autograd, "grad", side_effect=gradients):
        lowrank.inner_unroll_lowrank(source, geometry, u0, support,
            config=lowrank.LowRankConfig(pct_weight=0.1))
    assert teacher_call.call_count == 1 and len(observed_losses) == 10
    for w, loss in zip(observed_weights, observed_losses):
        z = original_logits(source, geometry, support.x, w)
        expected = balanced_bce(z, support.y) + 0.01 * (w.u @ w.v.T).square().sum()
        expected = expected + 0.001 * (w.delta_w.square().sum() + w.delta_b.square()) / 6
        expected = expected + 0.1 * (fixed * 0.5 * (z - teacher).square().sum(1)).mean()
        torch.testing.assert_close(loss, expected.detach(), atol=1e-14, rtol=1e-14)
    assert not teacher.requires_grad


@pytest.mark.parametrize("method", ["B6", "B7", "B10", "B11"])
def test_missing_class_is_exact_source_fallback_without_free_replacement(method):
    source, geometry, support, inputs, _ = fixture()
    support = replace(support, y=torch.zeros(4, dtype=torch.int64))
    u0 = lowrank.initialize_lowrank(1, 20261018) if method in {"B10", "B11"} else None
    fitted = lowrank.fit_lowrank_target(source, geometry, support, method=method, u0=u0)
    assert fitted.updates == 0 and fitted.fallback_reason == "insufficient_class_support"
    assert len(fitted.acquired_event_uids) == 20 and len(fitted.support_event_uids) == 4
    assert torch.equal(lowrank.predict_lowrank(source, geometry, fitted, inputs).logits, source.source_logits(inputs.x))


def test_target_permissions_meta_initialization_and_prediction_integrity_fail_closed():
    source, geometry, support, inputs, labels = fixture()
    u0 = lowrank.initialize_lowrank(1, 20261018)
    assert "query" not in inspect.signature(lowrank.fit_lowrank_target).parameters
    assert "y" not in PredictionInputs.__dataclass_fields__
    with pytest.raises(TypeError):
        lowrank.fit_lowrank_target(source, geometry, inputs)
    with pytest.raises(PermissionError):
        lowrank.fit_lowrank_target(source, geometry, replace(support, role="meta_validation_support"))
    with pytest.raises(ValueError):
        lowrank.fit_lowrank_target(source, geometry, support, method="B10")
    with pytest.raises(ValueError):
        lowrank.fit_lowrank_target(source, geometry, support, method="B6", u0=u0)
    with pytest.raises(ValueError):
        lowrank.fit_lowrank_target(source, geometry, support, method="B7", config=lowrank.LowRankConfig())
    fitted = lowrank.fit_lowrank_target(source, geometry, support)
    with pytest.raises(ValueError):
        lowrank.predict_lowrank(source, geometry, fitted, replace(inputs, protocol_freeze_sha="b"*64))
    with pytest.raises(ValueError):
        lowrank.predict_lowrank(source, geometry, fitted, replace(inputs, event_uids=("s19",) + inputs.event_uids[1:]))
    altered_geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64))
    with pytest.raises(ValueError):
        lowrank.predict_lowrank(source, altered_geometry, fitted, inputs)
    with torch.no_grad():
        fitted.weights.v[0, 0] += 0.01
    with pytest.raises(ValueError):
        lowrank.predict_lowrank(source, geometry, fitted, inputs)
    with pytest.raises(RuntimeError):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            lowrank.fit_lowrank_target(source, geometry, support)


def test_meta_validation_support_can_use_pure_inner_and_logit_kernels_without_query_fit_argument():
    source, geometry, support, inputs, _ = fixture()
    support = replace(support, role="meta_validation_support")
    u0 = lowrank.initialize_lowrank(1, 20261018)
    weights, steps, fallback = lowrank.inner_unroll_lowrank(source, geometry, u0, support)
    assert steps == 10 and fallback is None
    with torch.no_grad():
        z = lowrank.lowrank_logits(source, geometry, inputs.x, weights)
    assert z.shape == (20, 2) and torch.isfinite(z).all()


@pytest.mark.parametrize("method", ["B10", "B11"])
def test_full_ten_step_u0_meta_gradient_matches_stiefel_tangent_finite_difference(method):
    source, geometry, support, inputs, labels = fixture()
    u0 = lowrank.initialize_lowrank(1, 20261018).requires_grad_(True)
    episode = meta_episode(support, inputs, labels)
    config = lowrank.LowRankConfig(pct_weight=0.1 if method == "B11" else 0.0)
    objective = lowrank.lowrank_episode_objective(source, geometry, u0, episode, method=method, config=config)
    gradient, = torch.autograd.grad(objective, u0)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-10
    generator = torch.Generator().manual_seed(81)
    direction = torch.randn(u0.shape, generator=generator, dtype=torch.float64)
    tangent = direction - u0 @ (0.5 * (u0.T @ direction + direction.T @ u0))
    tangent = tangent.detach() / tangent.detach().norm()
    analytic = (gradient * tangent).sum()
    h = 1e-4
    plus = lowrank.lowrank_episode_objective(source, geometry, positive_qr(u0.detach()+h*tangent), episode, method=method, config=config)
    minus = lowrank.lowrank_episode_objective(source, geometry, positive_qr(u0.detach()-h*tangent), episode, method=method, config=config)
    finite = (plus-minus) / (2*h)
    torch.testing.assert_close(analytic, finite, atol=2e-8, rtol=2e-3)
    source.assert_unchanged()


@pytest.mark.parametrize("method", ["B10", "B11"])
def test_rank_two_full_unroll_gradient_through_active_v_projection_matches_finite_difference(method):
    source, geometry, support, inputs, labels = fixture()
    # A known synthetic head scale reliably activates the registered .05 bound.
    with torch.no_grad():
        for parameter in source.source_head.parameters():
            parameter.mul_(3)
    source.frozen_state_sha256 = module_state_sha256(source)
    u0 = lowrank.initialize_lowrank(2, 20261018).requires_grad_(True)
    episode = meta_episode(support, inputs, labels)
    config = lowrank.LowRankConfig(rank=2, radius=0.05, pct_weight=0.1 if method == "B11" else 0.0)
    attempted_norms, projected_norms = [], []
    projection = lowrank.project_lowrank_update
    def observe(u, v, radius):
        attempted_norms.append(float((u @ v.T).norm().detach()))
        unew, vnew = projection(u, v, radius)
        projected_norms.append(float((unew @ vnew.T).norm().detach()))
        return unew, vnew
    with patch.object(lowrank, "project_lowrank_update", side_effect=observe):
        loss = lowrank.lowrank_episode_objective(source, geometry, u0, episode, config=config, method=method)
    assert len(attempted_norms) == 10
    active = [i for i, norm in enumerate(attempted_norms) if norm > config.radius]
    assert len(active) >= 5
    assert all(projected_norms[i] <= config.radius + 1e-14 for i in range(10))
    assert all(abs(projected_norms[i] - config.radius) < 1e-14 for i in active)
    gradient, = torch.autograd.grad(loss, u0)
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-8
    generator = torch.Generator().manual_seed(91)
    direction = torch.randn(u0.shape, generator=generator, dtype=torch.float64)
    tangent = direction - u0 @ (0.5 * (u0.T @ direction + direction.T @ u0))
    tangent = tangent.detach() / tangent.detach().norm()
    analytic = (gradient * tangent).sum()
    h = 1e-5
    plus = lowrank.lowrank_episode_objective(source, geometry, positive_qr(u0.detach()+h*tangent),
                                            episode, config=config, method=method)
    minus = lowrank.lowrank_episode_objective(source, geometry, positive_qr(u0.detach()-h*tangent),
                                             episode, config=config, method=method)
    torch.testing.assert_close(analytic, (plus-minus)/(2*h), atol=2e-9, rtol=2e-4)
    source.assert_unchanged()


@pytest.mark.parametrize("method", ["B10", "B11"])
def test_four_episode_outer_step_has_independent_fast_initializations_and_updates_only_stiefel_u0(method):
    source, geometry, support, inputs, labels = fixture()
    episodes = tuple(meta_episode(replace(support, x=support.x*(1+i*0.01)), inputs, labels, f"_{i}") for i in range(4))
    u0 = lowrank.initialize_lowrank(2, 20261018).requires_grad_(True)
    original_u = u0.detach().clone()
    config = lowrank.LowRankConfig(rank=2, pct_weight=0.1 if method == "B11" else 0.0)
    initializations = []
    original = lowrank.zero_lowrank_weights
    def record(source, u0):
        w = original(source, u0)
        initializations.append(w)
        return w
    before = module_state_sha256(source)
    with patch.object(lowrank, "zero_lowrank_weights", side_effect=record):
        unext, receipt = lowrank.lowrank_outer_step(source, geometry, u0, episodes, method=method, config=config)
    assert len(initializations) == 4
    for w in initializations:
        assert torch.equal(w.u, original_u) and torch.count_nonzero(w.v) == 0
        assert torch.count_nonzero(w.delta_w) == 0 and w.delta_b == 0
    assert len({w.v.data_ptr() for w in initializations}) == 4
    torch.testing.assert_close(unext.T @ unext, torch.eye(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert unext.requires_grad and torch.equal(u0, original_u)
    assert receipt["full_unroll"] is True and receipt["inner_steps"] == 10 and receipt["episodes"] == 4
    assert receipt["method"] == method and receipt["outer_parameter"] == "U0_only"
    assert receipt["outer_learning_rate"] == 0.01 and receipt["gradient_norm_cap"] == 1.0
    assert module_state_sha256(source) == before


def test_active_outer_clip_is_exactly_tangent_projection_then_global_norm_cap_then_positive_qr():
    source, geometry, support, inputs, labels = fixture()
    episodes = tuple(meta_episode(support, inputs, labels, f"_{i}") for i in range(4))
    u0 = lowrank.initialize_lowrank(2, 20261018).requires_grad_(True)
    base = u0.detach()
    direction = torch.randn(base.shape, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    tangent_direction = direction - base @ (0.5 * (base.T @ direction + direction.T @ base))
    tangent_direction = tangent_direction / tangent_direction.norm()
    # The large normal component distinguishes clipping the Euclidean gradient
    # too early from clipping the projected tangent as required by the kernel.
    known_gradient = 100 * base + 7 * tangent_direction
    projected = known_gradient - base @ (0.5 * (base.T @ known_gradient + known_gradient.T @ base))
    assert projected.norm() > 1
    clipped = projected / projected.norm()
    expected = positive_qr(base - 0.01 * clipped)
    def affine_objective(source, geometry, u0, episode, **kwargs):
        return (u0 * known_gradient).sum()
    with patch.object(lowrank, "lowrank_episode_objective", side_effect=affine_objective) as objective:
        unext, receipt = lowrank.lowrank_outer_step(source, geometry, u0, episodes,
            config=lowrank.LowRankConfig(rank=2), method="B10")
    assert objective.call_count == 4
    torch.testing.assert_close(unext, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(unext.T @ unext, torch.eye(2, dtype=torch.float64), atol=1e-14, rtol=0)
    assert abs(receipt["euclidean_gradient_norm"] - float(known_gradient.norm())) < 1e-12
    assert receipt["gradient_norm_cap"] == 1.0 and receipt["outer_learning_rate"] == 0.01


def test_meta_permissions_mixed_versions_and_missing_class_zero_gradient_fallback():
    source, geometry, support, inputs, labels = fixture()
    u0 = lowrank.initialize_lowrank(1, 20261018).requires_grad_(True)
    episode = meta_episode(support, inputs, labels)
    with pytest.raises(TypeError):
        lowrank.lowrank_episode_objective(source, geometry, u0, inputs)
    with pytest.raises(ValueError):
        lowrank.lowrank_episode_objective(source, geometry, u0, episode, method="B6")
    with pytest.raises(ValueError):
        lowrank.lowrank_outer_step(source, geometry, u0, (episode,))
    with pytest.raises(ValueError):
        lowrank.lowrank_outer_step(source, geometry, u0, (episode,)*3 + (replace(episode, split_manifest_sha256="b"*64),))
    with pytest.raises(ValueError):
        lowrank.lowrank_outer_step(source, geometry, u0.detach(), (episode,)*4)
    invalid = replace(episode)
    object.__setattr__(invalid, "role", "external_query")
    with pytest.raises(PermissionError):
        lowrank.lowrank_episode_objective(source, geometry, u0, invalid)
    missing = replace(episode, support=replace(episode.support, y=torch.zeros_like(support.y)))
    loss = lowrank.lowrank_episode_objective(source, geometry, u0, missing)
    gradient, = torch.autograd.grad(loss, u0)
    assert torch.equal(gradient, torch.zeros_like(u0))
    expected = balanced_bce(source.source_logits(inputs.x), labels)
    torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    unext, receipt = lowrank.lowrank_outer_step(source, geometry, u0, (missing,)*4)
    assert receipt["euclidean_gradient_norm"] == 0
    torch.testing.assert_close(unext, u0, atol=1e-15, rtol=1e-15)
