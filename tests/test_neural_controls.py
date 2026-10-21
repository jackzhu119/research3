"""B2/B3 synthetic CPU software acceptance only, never real EEG fitting/pilot."""
from dataclasses import replace
import inspect
import itertools

import pytest
import torch
from torch import nn

from p3work.adapt import CalibrationPrefix, FrozenSourceModel, PredictionInputs, module_state_sha256
from p3work.contracts import SOURCE_IDS
from p3work.methods import neural_controls as controls
from p3work.methods.neural_controls import (AlignmentConfig, FineTuneConfig, SourceReferenceAuthorization,
    fit_full_finetune, fit_source_covariance_alignment, normalized_source_l2, predict_neural_control, support_alignment)
from p3work.spatial_geometry import (SourceCovarianceTrial, SpatialGeometry, helmert_rows,
                                    source_reference, spd_roots)

torch.set_num_threads(1)
SHA = "a" * 64  # Synthetic fixture identity, not a real frozen input protocol.


def fixture():
    torch.manual_seed(61)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * 12, 5), nn.BatchNorm1d(5), nn.Tanh(), nn.Dropout(.6)).double()
    source = FrozenSourceModel(encoder, nn.Linear(5, 2).double(), feature_dim=5)
    x = torch.randn(24, 11, 12, dtype=torch.float64)
    x -= x.mean(1, keepdim=True)
    x -= x.mean(2, keepdim=True)
    support = CalibrationPrefix(x[:4], torch.tensor([0, 1, 0, 1]), tuple(f"s{i}" for i in range(4)),
                                tuple(f"s{i}" for i in range(20)), SHA)
    query = PredictionInputs(x[4:], tuple(f"q{i}" for i in range(20)), SHA)
    source_trials = []
    for dataset in sorted(SOURCE_IDS):
        for i in range(3):
            sx = torch.randn(11, 12, dtype=torch.float64)
            sx -= sx.mean(0, keepdim=True)
            sx -= sx.mean(1, keepdim=True)
            source_trials.append(SourceCovarianceTrial(dataset, "source-person", f"{dataset}/{i}", sx))
    ref = source_reference(source_trials)
    geometry = SpatialGeometry.from_reference(ref.matrix)
    authorization = SourceReferenceAuthorization(ref, SHA, tuple(sorted(SOURCE_IDS)))
    return source, support, query, geometry, authorization


def fit(method, source, support, geometry, authorization):
    return (fit_full_finetune(source, support) if method == "B2" else
            fit_source_covariance_alignment(source, geometry, support, source_reference=authorization))


def test_registered_grids_have_nine_settings_and_fixed_ten_steps():
    assert len([FineTuneConfig(lr, l2) for lr, l2 in itertools.product((.0001, .0003, .001), (0, .0001, .001))]) == 9
    assert len([AlignmentConfig(shrink, lr) for shrink, lr in itertools.product((.01, .05, .1), (.003, .01, .03))]) == 9
    for factory, kwargs in ((FineTuneConfig, {"steps": 9}), (FineTuneConfig, {"learning_rate": .1}),
                            (AlignmentConfig, {"steps": 20}), (AlignmentConfig, {"head_l2": .01}),
                            (AlignmentConfig, {"shrinkage": .2})):
        with pytest.raises(ValueError):
            factory(**kwargs)


def test_b2_source_l2_includes_bias_and_divides_by_parameter_element_count():
    weight = nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
    bias = nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    loss = normalized_source_l2((weight, bias), (torch.zeros_like(weight), torch.zeros_like(bias)))
    torch.testing.assert_close(loss, torch.tensor(14 / 3, dtype=torch.float64))
    gw, gb = torch.autograd.grad(loss, (weight, bias))
    torch.testing.assert_close(gw, torch.tensor([2 / 3, 4 / 3], dtype=torch.float64))
    torch.testing.assert_close(gb, torch.tensor([2.], dtype=torch.float64))


def test_b2_updates_every_parameter_tensor_including_bn_affine_and_bias_without_source_or_buffer_change():
    source, support, query, _, _ = fixture()
    source_before = module_state_sha256(source)
    buffers_before = {n: t.clone() for n, t in source.named_buffers()}
    parameters_before = {n: t.clone() for n, t in source.named_parameters()}
    rng_before = torch.get_rng_state().clone()
    fitted = fit_full_finetune(source, support)
    assert fitted.method == "B2" and fitted.updates == 10 and fitted.fallback_reason is None
    assert fitted.trainable_parameters == sum(t.numel() for t in source.parameters())
    target_parameters = dict(fitted.target_model.named_parameters())
    assert set(target_parameters) == set(parameters_before)
    assert all(not torch.equal(t, parameters_before[n]) for n, t in target_parameters.items())
    assert any(n.endswith("bias") for n in target_parameters)
    for name, value in fitted.target_model.named_buffers():
        assert torch.equal(value, buffers_before[name])
    assert not any(m.training for m in fitted.target_model.modules())
    assert not any(p.requires_grad for p in fitted.target_model.parameters())
    assert torch.equal(torch.get_rng_state(), rng_before)  # Dropout stayed disabled.
    result = predict_neural_control(fitted, query)
    assert torch.equal(result.probabilities, result.logits.softmax(1))
    assert torch.equal(result.labels, result.logits.argmax(1))
    assert module_state_sha256(source) == source_before
    assert all(p.grad is None and not p.requires_grad for p in source.parameters())


def test_b2_sessions_restart_from_same_source_and_use_l2_to_source():
    source, support, query, _, _ = fixture()
    a = fit_full_finetune(source, support, config=FineTuneConfig(learning_rate=.001, l2_to_source=0))
    b = fit_full_finetune(source, support, config=FineTuneConfig(learning_rate=.001, l2_to_source=0))
    c = fit_full_finetune(source, support, config=FineTuneConfig(learning_rate=.001, l2_to_source=.001))
    for av, bv in zip(a.target_model.parameters(), b.target_model.parameters()):
        assert torch.equal(av, bv)
    assert any(not torch.equal(av, cv) for av, cv in zip(a.target_model.parameters(), c.target_model.parameters()))
    assert torch.equal(predict_neural_control(a, query).logits, predict_neural_control(b, query).logits)


def test_b3_unit_trace_equal_trial_covariance_shrinkage_and_correct_alignment_direction():
    source, support, _, geometry, authorization = fixture()
    alignment, ctarget, rt = support_alignment(geometry, support, shrinkage=.05)
    q = geometry.q
    expected = []
    for trial in support.x:
        covariance = (q @ trial) @ (q @ trial).T
        expected.append(covariance / torch.trace(covariance))
    torch.testing.assert_close(ctarget, torch.stack(expected).mean(0), atol=1e-14, rtol=0)
    torch.testing.assert_close(torch.trace(ctarget), torch.tensor(1., dtype=torch.float64), atol=1e-14, rtol=0)
    torch.testing.assert_close(rt, .95 * ctarget + .05 * torch.eye(10, dtype=torch.float64) / 10, atol=1e-14, rtol=0)
    _, rt_inverse = spd_roots(rt)
    torch.testing.assert_close(alignment, q.T @ geometry.rref_sqrt @ rt_inverse @ q, atol=1e-14, rtol=0)
    effective = q @ alignment @ q.T
    # Correct target→source direction maps the SHRUNK target reference to Rref.
    torch.testing.assert_close(effective @ rt @ effective.T, authorization.reference.matrix, atol=1e-12, rtol=1e-12)
    fitted = fit_source_covariance_alignment(source, geometry, support, source_reference=authorization)
    assert fitted.method == "B3" and fitted.updates == 10 and fitted.trainable_parameters == 6
    assert fitted.source_reference_authorization_sha256 is not None
    torch.testing.assert_close(fitted.target_covariance, ctarget, atol=0, rtol=0)


def test_b3_estimator_ignores_per_trial_common_positive_gain_and_common_reference():
    _, support, _, geometry, _ = fixture()
    a, ct, rt = support_alignment(geometry, support)
    gains = torch.tensor([1e-6, 2., 1e4, .2], dtype=torch.float64).reshape(4, 1, 1)
    b, ct2, rt2 = support_alignment(geometry, replace(support, x=support.x * gains))
    torch.testing.assert_close(ct, ct2, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)
    offsets = torch.randn(4, 1, 12, dtype=torch.float64)
    c, _, _ = support_alignment(geometry, replace(support, x=support.x + offsets))
    torch.testing.assert_close(a, c, atol=1e-13, rtol=1e-13)


def test_b3_matching_isotropic_covariance_recovers_car_identity():
    q = helmert_rows()
    x = (q.T @ q).repeat(4, 1, 1)
    support = CalibrationPrefix(x, torch.tensor([0, 1, 0, 1]), tuple(f"s{i}" for i in range(4)),
                                tuple(f"s{i}" for i in range(20)), SHA)
    geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
    alignment, _, _ = support_alignment(geometry, support)
    torch.testing.assert_close(alignment, q.T @ q, atol=1e-14, rtol=0)


@pytest.mark.parametrize("method", ["B2", "B3"])
def test_missing_class_falls_back_exactly_and_preserves_acquired_budget(method, monkeypatch):
    source, support, query, geometry, authorization = fixture()
    support = replace(support, y=torch.zeros(4, dtype=torch.int64))
    def must_not_estimate(*args, **kwargs):
        pytest.fail("Missing-class support attempted B3 covariance fitting")
    monkeypatch.setattr(controls, "_support_alignment", must_not_estimate)
    fitted = fit(method, source, support, geometry, authorization)
    assert fitted.updates == 0 and fitted.fallback_reason == "insufficient_class_support"
    assert len(fitted.acquired_event_uids) == 20 and len(fitted.support_event_uids) == 4
    assert fitted.alignment is None and fitted.target_covariance is None
    assert torch.equal(predict_neural_control(fitted, query).logits, source.source_logits(query.x))


@pytest.mark.parametrize("method", ["B2", "B3"])
def test_query_batch_statistics_cannot_change_fixed_predictions(method):
    source, support, query, geometry, authorization = fixture()
    fitted = fit(method, source, support, geometry, authorization)
    single = replace(query, x=query.x[:1], event_uids=query.event_uids[:1])
    changed = query.x.clone()
    changed[1:] *= 1000000  # Other future inputs must not determine normalization/BN/alignment.
    with_many = predict_neural_control(fitted, replace(query, x=changed)).logits[:1]
    alone = predict_neural_control(fitted, single).logits
    torch.testing.assert_close(with_many, alone, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("violation", ["role", "source_ids", "protocol", "event_digest", "geometry"])
def test_b3_source_reference_permissions_and_identity_fail_closed(violation):
    source, support, _, geometry, authorization = fixture()
    if violation == "role":
        authorization = replace(authorization, role="external_query")
    elif violation == "source_ids":
        authorization = replace(authorization, dataset_ids=("BNCI2014_001", "Cho2017", "Lee2019_MI", "Stieger2021"))
    elif violation == "protocol":
        authorization = replace(authorization, protocol_freeze_sha="b" * 64)
    elif violation == "event_digest":
        authorization = replace(authorization, reference=replace(authorization.reference, events_sha256="c" * 64))
    else:
        geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
    with pytest.raises((PermissionError, ValueError)):
        fit_source_covariance_alignment(source, geometry, support, source_reference=authorization)


@pytest.mark.parametrize("method", ["B2", "B3"])
def test_label_visibility_prefix_overlap_and_fitted_state_are_guarded(method):
    source, support, query, geometry, authorization = fixture()
    assert "query" not in inspect.signature(fit_full_finetune).parameters
    assert "query" not in inspect.signature(fit_source_covariance_alignment).parameters
    assert "y" not in PredictionInputs.__dataclass_fields__
    with pytest.raises(PermissionError):
        fit(method, source, replace(support, role="meta_train_support"), geometry, authorization)
    fitted = fit(method, source, support, geometry, authorization)
    with pytest.raises(ValueError):
        predict_neural_control(fitted, replace(query, protocol_freeze_sha="b" * 64))
    with pytest.raises(ValueError, match="overlaps"):
        predict_neural_control(fitted, replace(query, event_uids=("s19",) + query.event_uids[1:]))
    with torch.no_grad():
        if method == "B2":
            next(fitted.target_model.parameters()).add_(.1)
        else:
            fitted.alignment.add_(.1)
    with pytest.raises(ValueError, match="changed"):
        predict_neural_control(fitted, query)


@pytest.mark.parametrize("method", ["B2", "B3"])
def test_validation_prefix_has_separate_entry_and_keeps_its_permission_role(method):
    source, support, query, geometry, authorization = fixture()
    development = replace(support, role="meta_validation_support")
    with pytest.raises(PermissionError):
        fit(method, source, development, geometry, authorization)
    fitted = controls.fit_neural_control_development(source, development, method=method, geometry=geometry,
                                                     source_reference=authorization)
    target = fit(method, source, support, geometry, authorization)
    assert fitted.support_role == "meta_validation_support" and target.support_role == "calibration_prefix"
    torch.testing.assert_close(predict_neural_control(fitted, query).logits,
                               predict_neural_control(target, query).logits, atol=0, rtol=0)
    with pytest.raises(PermissionError):
        controls.fit_neural_control_development(source, replace(support, role="meta_train_support"), method=method,
                                                 geometry=geometry, source_reference=authorization)
