"""Synthetic CPU math/capability tests, never real EEG/source fit evidence."""
from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch
from scipy.linalg import helmert
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from p3work.adapt import CalibrationPrefix, PredictionInputs
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest
from p3work.methods.classical import (
    B4_GRID, B5_GRID, SourceTrainingInputs, balanced_logistic_objective,
    csp_features, csp_filters, fit_b4_source, fit_b4_target, fit_b4_development, fit_b5_source,
    fit_b5_target, fit_b5_development, hierarchical_class_means, hierarchical_weights,
    load_classical_model, matrix_function, predict_classical,
    save_classical_model, tangent_features, trace_covariances,
)


def contract():
    return {
        "status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": "short", "context_s": [0, 2], "crop_s": [0.5, 1.5],
        "n_times": 128,
        "window_audit_receipt": {
            "decision_pool_role": "meta_train", "split_sha256": "a" * 64,
            "raw_metadata_manifest_sha256": "b" * 64,
            "long": {"participant_mean_coverage": 0.7, "participant_mean_abs_class_exclusion_gap": 0.04},
            "short": {"participant_mean_coverage": 0.9, "participant_mean_abs_class_exclusion_gap": 0.02},
        },
    }


def synthetic_sources(held_out=None):
    rng = np.random.default_rng(20261018)
    q = helmert(11, full=False)
    rows, labels, trials = [], [], []
    for dataset in sorted(SOURCE_IDS - ({held_out} if held_out else set())):
        for subject in ("1", "2"):
            for index in range(8):
                label = index % 2
                z = rng.normal(size=(10, 128))
                z[label] *= 2 + int(subject) / 3
                z -= z.mean(axis=1, keepdims=True)
                trials.append(q.T @ z)
                labels.append(label)
                rows.append({"dataset": dataset, "subject": subject, "role": "source_train",
                             "event_uid": f"synthetic/{dataset}/{subject}/{index}"})
    return SourceTrainingInputs(np.stack(trials), np.array(labels, dtype=np.int64), tuple(rows),
                                contract(), "c" * 64, held_out_dataset=held_out, synthetic=True)


@pytest.fixture(scope="module")
def source_inputs():
    return synthetic_sources()


@pytest.fixture(scope="module")
def b4(source_inputs):
    return fit_b4_source(source_inputs, m=2)


@pytest.fixture(scope="module")
def b5(source_inputs):
    return fit_b5_source(source_inputs, shrink=0.05)


def prefix(inputs, *, missing_class=False, empty=False):
    indices = np.arange(20)
    if missing_class:
        indices = indices[inputs.y[indices] == 0]
    if empty:
        indices = indices[:0]
    acquired = tuple(f"target/{index}" for index in range(20))
    return CalibrationPrefix(torch.tensor(inputs.x[indices]), torch.tensor(inputs.y[indices]),
                             tuple(acquired[index] for index in indices), acquired, digest(inputs.signal_contract))


def predictions(inputs):
    return PredictionInputs(torch.tensor(inputs.x[-12:]), tuple(f"query/{i}" for i in range(12)),
                            digest(inputs.signal_contract))


def test_only_registered_source_training_roles_before_payload_inspection(source_inputs):
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("unauthorized signal was inspected")
    rows = [dict(row) for row in source_inputs.rows]
    rows[0].update(dataset="HSStroke2026_v6", role="sealed_query")
    for fit in (fit_b4_source, fit_b5_source):
        with pytest.raises(PermissionError):
            fit(replace(source_inputs, x=Unreadable(), rows=tuple(rows)))
        with pytest.raises(PermissionError):
            fit(np.zeros((20, 11, 128)))


def test_frozen_contract_and_unique_provenance_required(source_inputs):
    pending = dict(source_inputs.signal_contract, status="pending")
    with pytest.raises(ValueError, match="not frozen"):
        fit_b4_source(replace(source_inputs, signal_contract=pending))
    rows = list(source_inputs.rows)
    rows[1] = dict(rows[1], event_uid=rows[0]["event_uid"])
    with pytest.raises(ValueError, match="duplicate"):
        fit_b5_source(replace(source_inputs, rows=tuple(rows)))
    with pytest.raises(ValueError, match="membership"):
        fit_b4_source(replace(source_inputs, held_out_dataset="Cho2017",
                              rows=tuple(row for row in source_inputs.rows
                                         if row["dataset"] not in {"Cho2017", "PhysionetMI"})))


def test_source_lodo_explicit_and_not_a_target_prior(source_inputs):
    inputs = synthetic_sources("Cho2017")
    model = fit_b4_source(inputs)
    assert model.receipt["held_out_dataset"] == "Cho2017"
    assert set(model.receipt["source_datasets"]) == SOURCE_IDS - {"Cho2017"}
    with pytest.raises(ValueError, match="all-four"):
        fit_b4_target(model, prefix(source_inputs))


def test_actual_cache_subject_id_schema_and_label_binding(source_inputs):
    rows = tuple({**{key: value for key, value in row.items() if key != "subject"},
                  "subject_id": row["subject"], "label": int(label)}
                 for row, label in zip(source_inputs.rows, source_inputs.y))
    fitted = fit_b4_source(replace(source_inputs, rows=rows))
    assert set(fitted.receipt["participants_by_source"].values()) == {2}
    broken = (dict(rows[0], label=1 - rows[0]["label"]), *rows[1:])
    with pytest.raises(ValueError, match="label disagrees"):
        fit_b5_source(replace(source_inputs, rows=broken))
    conflict = (dict(rows[0], subject="conflict"), *rows[1:])
    with pytest.raises(ValueError, match="conflicting"):
        fit_b4_source(replace(source_inputs, rows=conflict))


def test_hierarchical_weights_and_class_means_do_not_overweight_large_people(source_inputs):
    original = source_inputs
    chosen = [index for index, row in enumerate(original.rows)
              if row["dataset"] == "Cho2017" and row["subject"] == "1"]
    indices = np.concatenate((np.arange(len(original.y)), np.tile(chosen, 7)))
    rows = tuple(original.rows[index] for index in indices)
    y = original.y[indices]
    covariance = trace_covariances(original.x)
    means = hierarchical_class_means(covariance[indices], rows, y)
    np.testing.assert_allclose(means, hierarchical_class_means(covariance, original.rows, original.y), atol=1e-15)
    weights = hierarchical_weights(rows, y)
    assert weights.sum() == pytest.approx(1)
    for dataset in SOURCE_IDS:
        mask = np.array([row["dataset"] == dataset for row in rows])
        assert weights[mask].sum() == pytest.approx(0.25)
        for subject in ("1", "2"):
            person = mask & np.array([row["subject"] == subject for row in rows])
            assert weights[person].sum() == pytest.approx(0.125)
            for label in (0, 1):
                assert weights[person & (y == label)].sum() == pytest.approx(0.0625)


def test_trace_covariance_common_gain_and_spd_without_amplitude_epsilon(source_inputs):
    x = source_inputs.x[:4]
    expected = trace_covariances(x, shrink=0.05)
    for gain in (1e-100, 1e100):
        np.testing.assert_allclose(trace_covariances(x * gain, shrink=0.05), expected, atol=3e-16)
    np.testing.assert_allclose(np.trace(expected, axis1=1, axis2=2), 1, atol=1e-15)
    assert np.linalg.eigvalsh(expected).min() >= 0.005 - 1e-15
    with pytest.raises(ValueError, match="zero-energy"):
        trace_covariances(np.zeros_like(x))
    with pytest.raises(ValueError, match="CAR11"):
        trace_covariances(x + 1)


def test_csp_order_sign_generalized_eigenproblem_and_tie_convention():
    mean0 = np.diag(np.linspace(1, 10, 10)) / 55
    mean1 = mean0[::-1, ::-1]
    filters, values = csp_filters(np.stack((mean0, mean1)), 3)
    c0, c1 = [0.9 * mean + 0.1 * np.eye(10) / 10 for mean in (mean0, mean1)]
    assert np.all(np.diff(values[:3]) >= 0)
    assert np.all(np.diff(values[3:]) <= 0)
    for vector, value in zip(filters, values):
        assert vector[np.argmax(np.abs(vector))] >= 0
        np.testing.assert_allclose(c1 @ vector, value * (c0 + c1) @ vector, atol=1e-15)
    np.testing.assert_allclose(filters @ (c0 + c1) @ filters.T, np.eye(6), atol=1e-14)
    tied = np.stack((np.eye(10) / 10,) * 2)
    first = csp_filters(tied, 3)[0]
    np.testing.assert_array_equal(first, csp_filters(tied, 3)[0])
    np.testing.assert_allclose(first @ (np.eye(10) / 5) @ first.T, np.eye(6), atol=1e-15)


def test_csp_log_normalized_variance(source_inputs, b4):
    feature = csp_features(source_inputs.x, b4.arrays["filters"])
    np.testing.assert_allclose(np.exp(feature).sum(axis=1), 1, atol=5e-16)
    np.testing.assert_allclose(csp_features(source_inputs.x * 19, b4.arrays["filters"]), feature, atol=2e-15)
    assert b4.receipt["source_classifier_weighting"].startswith("unweighted_trials_legacy_recipe")


def test_target_b4_alpha_one_reuses_source_csp_and_exact_target_lda(source_inputs, b4):
    support = prefix(source_inputs)
    fitted = fit_b4_target(b4, support, alpha=1, m=2)
    np.testing.assert_array_equal(fitted.arrays["filters"], b4.arrays["filters"])
    feature = csp_features(support.x.numpy(), b4.arrays["filters"])
    exact = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(feature, support.y.numpy())
    np.testing.assert_array_equal(fitted.arrays["coef"], exact.coef_.reshape(-1))
    assert fitted.receipt["target_optimizer_updates"] == 0
    assert fitted.receipt["target_exact_solves"] == 1
    assert fitted.receipt["acquired_trials"] == 20
    assert fitted.receipt["query_accessed"] is False


def test_b4_registered_nine_grid(source_inputs):
    assert len(B4_GRID) == 9
    for m in (1, 2, 3):
        model = fit_b4_source(source_inputs, m=m)
        for alpha in (0, 0.5, 1):
            fitted = fit_b4_target(model, prefix(source_inputs), alpha=alpha)
            assert fitted.receipt["status"] == "TARGET_FIT_COMPLETE"
            assert fitted.arrays["filters"].shape == (2 * m, 10)


@pytest.mark.parametrize("empty", [False, True])
def test_missing_support_classes_fall_back_to_each_own_source(source_inputs, b4, b5, empty):
    support = prefix(source_inputs, missing_class=True, empty=empty)
    for source, fit in ((b4, fit_b4_target), (b5, fit_b5_target)):
        fitted = fit(source, support)
        assert fitted.receipt["fallback_reason"] == "insufficient_class_support"
        assert fitted.receipt["acquired_trials"] == 20
        assert fitted.receipt["target_optimizer_updates"] == 0
        np.testing.assert_array_equal(predict_classical(fitted, predictions(source_inputs))["logits"],
                                      predict_classical(source, predictions(source_inputs))["logits"])


def test_log_euclidean_reference_and_vectorization_preserve_log_frobenius_norm(source_inputs, b5):
    covariance = trace_covariances(source_inputs.x, shrink=0.05)
    weights = hierarchical_weights(source_inputs.rows, source_inputs.y)
    expected = matrix_function(np.average(matrix_function(covariance, "log"), axis=0, weights=weights), "exp")
    np.testing.assert_array_equal(expected, b5.arrays["reference"])
    assert b5.receipt["reference"] == "weighted_log_Euclidean_not_affine_invariant"
    logs = matrix_function(b5.arrays["inverse_root"] @ covariance @ b5.arrays["inverse_root"], "log")
    feature = tangent_features(covariance, b5.arrays["inverse_root"])
    assert feature.shape[1] == 55
    np.testing.assert_allclose(np.sum(feature ** 2, axis=1), np.sum(logs ** 2, axis=(1, 2)), atol=1e-13)


def test_balanced_bce_source_penalty_including_bias_analytic_gradient():
    rng = np.random.default_rng(810)
    features = rng.normal(size=(9, 4))
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1])
    source = rng.normal(size=5)
    parameters = rng.normal(size=5)
    value, gradient = balanced_logistic_objective(parameters, features, labels, source, 0.1)
    numerical = np.empty_like(parameters)
    step = 1e-6
    for i in range(len(parameters)):
        delta = np.eye(5)[i] * step
        plus = balanced_logistic_objective(parameters + delta, features, labels, source, 0.1)[0]
        minus = balanced_logistic_objective(parameters - delta, features, labels, source, 0.1)[0]
        numerical[i] = (plus - minus) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=4e-10, rtol=1e-8)
    unpenalized_value, unpenalized_gradient = balanced_logistic_objective(parameters, features, labels, source, 0)
    assert value - unpenalized_value == pytest.approx(0.1 * np.sum((parameters - source) ** 2) / 5)
    assert gradient[-1] - unpenalized_gradient[-1] == pytest.approx(2 * 0.1 * (parameters[-1] - source[-1]) / 5)
    indices = np.concatenate((np.arange(9), np.tile([0, 1], 5)))
    repeated = balanced_logistic_objective(parameters, features[indices], labels[indices], source, 0.1)
    assert repeated[0] == pytest.approx(value)
    np.testing.assert_allclose(repeated[1], gradient, atol=3e-16)


def test_b5_target_frozen_scaler_deterministic_full_support_lbfgs(source_inputs, b5):
    before = b5.state_sha256
    first = fit_b5_target(b5, prefix(source_inputs), l2=0.1)
    second = fit_b5_target(b5, prefix(source_inputs), l2=0.1)
    np.testing.assert_array_equal(first.arrays["coef"], second.arrays["coef"])
    for key in ("reference", "inverse_root", "scaler_mean", "scaler_scale"):
        np.testing.assert_array_equal(first.arrays[key], b5.arrays[key])
    assert first.receipt["target_optimizer_updates"] <= 200
    assert first.receipt["gradient_tolerance"] == 1e-8
    assert first.receipt["target_trainable_parameters"] == 56
    assert type(first.receipt["converged"]) is bool
    assert first.receipt["converged"] == (first.receipt["optimizer_success"] and first.receipt["gradient_inf"] <= 1e-8)
    assert first.receipt["max_iterations"] == 200
    assert before == b5.state_sha256
    b5.assert_unchanged()


def test_b5_registered_nine_grid_and_matching_source_references(source_inputs):
    assert len(B5_GRID) == 9
    refs = []
    for shrink in (0.01, 0.05, 0.1):
        source = fit_b5_source(source_inputs, shrink=shrink)
        refs.append(source.arrays["reference"])
        for l2 in (0.01, 0.1, 1):
            target = fit_b5_target(source, prefix(source_inputs), l2=l2)
            assert target.receipt["shrink"] == shrink
            assert target.receipt["l2"] == l2
            assert target.receipt["target_optimizer_updates"] <= 200
    assert not np.allclose(refs[0], refs[-1])


def test_no_query_reference_reestimate_prediction_is_batch_invariant(source_inputs, b4, b5):
    inputs = predictions(source_inputs)
    for model in (b4, b5):
        batch = predict_classical(model, inputs)
        individual = [predict_classical(model, PredictionInputs(inputs.x[i:i + 1], (inputs.event_uids[i],),
                                                                inputs.protocol_freeze_sha))["logits"]
                      for i in range(len(inputs.x))]
        np.testing.assert_allclose(batch["logits"], np.concatenate(individual), atol=2e-14, rtol=1e-13)
        np.testing.assert_array_equal(batch["prediction"], batch["logits"].argmax(axis=1))
        np.testing.assert_allclose(batch["probability"].sum(axis=1), 1, atol=2e-16)
    with pytest.raises(PermissionError):
        fit_b5_target(b5, inputs)
    with pytest.raises(PermissionError):
        predict_classical(b5, prefix(source_inputs))


def test_signal_hash_binding_and_source_state_mutation_detection(source_inputs, b4, b5):
    wrong = replace(prefix(source_inputs), protocol_freeze_sha="d" * 64)
    with pytest.raises(ValueError, match="signal versions"):
        fit_b4_target(b4, wrong)
    with pytest.raises(ValueError, match="signal versions"):
        fit_b5_target(b5, wrong)
    wrong_length = replace(prefix(source_inputs), x=prefix(source_inputs).x[:, :, :64])
    with pytest.raises(ValueError, match="temporal demeaning|window lengths"):
        fit_b5_target(b5, wrong_length)
    model = fit_b4_source(source_inputs)
    with pytest.raises(ValueError):
        model.arrays["coef"][0] = 0
    model.receipt["source_trials"] += 1
    with pytest.raises(RuntimeError, match="mutated"):
        model.assert_unchanged()


def test_numerical_artifact_restore_full_hash_validation(source_inputs, b4, b5, tmp_path):
    for source, name in ((b4, "b4"), (b5, "b5")):
        directory = tmp_path / name
        saved = save_classical_model(source, directory)
        receipt_sha = file_digest(directory / "receipt.json")
        restored = load_classical_model(directory, expected_receipt_sha256=receipt_sha)
        assert restored.state_sha256 == source.state_sha256
        assert saved["fit_receipt"]["synthetic"] is True
        assert saved["fit_receipt"]["query_accessed"] is False
        with pytest.raises(FileExistsError):
            save_classical_model(source, directory)
        entry = saved["files"][0]
        path = directory / entry["path"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with pytest.raises(ValueError, match="hash/size"):
            load_classical_model(directory, expected_receipt_sha256=receipt_sha)


def test_invalid_grid_rejected(source_inputs, b4, b5):
    with pytest.raises(ValueError):
        fit_b4_target(b4, prefix(source_inputs), alpha=0.2)
    with pytest.raises(ValueError):
        fit_b4_target(b4, prefix(source_inputs), m=3)
    with pytest.raises(ValueError):
        fit_b5_source(source_inputs, shrink=0.2)
    with pytest.raises(ValueError):
        fit_b5_target(b5, prefix(source_inputs), l2=0)


@pytest.mark.parametrize("method", ["B4", "B5"])
@pytest.mark.parametrize("fallback", [False, True])
def test_query_rejects_entire_acquired_prefix_including_failed_qc(method, fallback, source_inputs, b4, b5):
    source, fit = (b4, fit_b4_target) if method == "B4" else (b5, fit_b5_target)
    # Trial 19 was acquired but excluded by frozen upstream QC. It must still
    # remain forbidden to future evaluation, including the missing-class path.
    support = prefix(source_inputs, missing_class=fallback)
    if not fallback:
        support = replace(support, x=support.x[:-1], y=support.y[:-1], event_uids=support.event_uids[:-1])
    fitted = fit(source, support)
    query = predictions(source_inputs)
    for forbidden in (support.event_uids[0], support.acquired_event_uids[-1]):
        overlapping = replace(query, event_uids=(forbidden, *query.event_uids[1:]))
        with pytest.raises(ValueError, match="overlaps.*acquired"):
            predict_classical(fitted, overlapping)
    assert predict_classical(fitted, query)["logits"].shape == (12, 2)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_development_fit_uses_only_validation_support_with_identical_math_and_clear_receipt(method, source_inputs, b4, b5):
    source, target_fit, development_fit = ((b4, fit_b4_target, fit_b4_development) if method == "B4"
                                          else (b5, fit_b5_target, fit_b5_development))
    support = prefix(source_inputs)
    validation = replace(support, role="meta_validation_support")
    target = target_fit(source, support)
    development = development_fit(source, validation)
    assert development.receipt["kind"] == "development_model"
    assert development.receipt["support_role"] == "meta_validation_support"
    assert development.receipt["fit_purpose"] == "development_model_selection"
    assert development.receipt["query_accessed"] is False
    assert development.receipt["source_state_sha256"] == source.state_sha256
    for key in target.arrays:
        np.testing.assert_array_equal(development.arrays[key], target.arrays[key])
    np.testing.assert_array_equal(predict_classical(target, predictions(source_inputs))["logits"],
                                  predict_classical(development, predictions(source_inputs))["logits"])
    assert "query" not in inspect.signature(development_fit).parameters
    assert "query_y" not in inspect.signature(development_fit).parameters
    assert "y" not in PredictionInputs.__dataclass_fields__
    for wrong in (support, replace(support, role="meta_train_support"), predictions(source_inputs)):
        with pytest.raises(PermissionError):
            development_fit(source, wrong)
    for wrong in (validation, replace(support, role="meta_train_support")):
        with pytest.raises(PermissionError):
            target_fit(source, wrong)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_development_missing_class_fallback_keeps_prefix_guard(method, source_inputs, b4, b5):
    source, fit = (b4, fit_b4_development) if method == "B4" else (b5, fit_b5_development)
    support = replace(prefix(source_inputs, missing_class=True), role="meta_validation_support")
    fitted = fit(source, support)
    assert fitted.receipt["kind"] == "development_model"
    assert fitted.receipt["fallback_reason"] == "insufficient_class_support"
    assert fitted.receipt["acquired_trials"] == 20
    query = predictions(source_inputs)
    with pytest.raises(ValueError, match="overlaps.*acquired"):
        predict_classical(fitted, replace(query, event_uids=("target/19", *query.event_uids[1:])))
    np.testing.assert_array_equal(predict_classical(fitted, query)["logits"], predict_classical(source, query)["logits"])
