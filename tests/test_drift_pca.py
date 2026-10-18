"""SOFTWARE_TEST_DOUBLE math only: no real Stieger PCA estimate or checkpoint."""
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
from scipy.linalg import expm
import torch

from p3work import drift_pca as pca
from p3work.contracts import digest, file_digest
from p3work.spatial_geometry import helmert_rows, traceless_basis
from test_classical_controls import contract as signal_contract


ROOT = Path(__file__).resolve().parents[1]


def explicit_software_parameters(**updates):
    value = pca.pending_parameter_contract()
    value.update(status="SOFTWARE_TEST_DOUBLE_PARAMETER_CONTRACT", synthetic=True)
    value["parameters"] = {
        "covariance_coordinates": "car10",
        "trial_covariance_normalization": "per_trial_trace_after_coordinate_map",
        "covariance_shrinkage": .07, "shrinkage_stage": "per_trial",
        "class_balance": "equal_class_then_trial", "session_trial_scope": "all_valid_lr_trials_of_session",
        "single_class_session_policy": "error_stop", "within_subject_centering": "equal_sessions",
        "pca_observation_weighting": "subject_equal_then_session_equal",
        "global_center_after_subject_centering": False, "insufficient_subject_sessions_policy": "error_stop",
        **updates,
    }
    return value


def modes():
    diagonal = helmert_rows(10).numpy()
    return np.stack([np.diag(diagonal[index]) for index in range(3)])


def session_fixture(*, scalar_nuisance=False, equal_variance=False):
    g1, g2, g3 = modes()
    sessions = []
    for person, offset in (("SOFTWARE_A", -2), ("SOFTWARE_B", 3)):
        for index, (a, b) in enumerate(((3, 0), (-3, 0), (0, 3 if equal_variance else 2), (0, -3 if equal_variance else -2))):
            log = offset * g3 + a * g1 + b * g2
            if scalar_nuisance:
                log += (.3 * index - .4) * np.eye(10)
            sessions.append(pca.SoftwareSessionCovariance(person, str(index), expm(log), 100 + index))
    return sessions


def estimate(sessions=None, parameter_contract=None, rank=2):
    return pca.fit_drift_pca_math(session_fixture() if sessions is None else sessions,
        explicit_software_parameters() if parameter_contract is None else parameter_contract,
        rank=rank, software_test_double=True)


def projection(generators):
    vectors = generators.reshape(len(generators), 100).T
    return vectors @ vectors.T


def test_protocol_does_not_supply_hidden_scientific_defaults_and_plan_writes_nothing(tmp_path, monkeypatch):
    parameters = pca.pending_parameter_contract()
    assert set(pca.unresolved_parameters(parameters)) == set(pca.UNRESOLVED_FIELDS)
    assert all(value is None for value in parameters["parameters"].values())
    with pytest.raises(pca.DriftPCAError, match="remain unresolved"):
        pca.validate_parameters(parameters, formal=True)
    def forbidden(*args, **kwargs):
        pytest.fail("Read-only preparation plan attempted an artifact write")
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(np, "save", forbidden)
    plan = pca.preparation_plan(ROOT)
    assert plan["status"] == "BLOCKED_PENDING_DRIFT_PCA_PARAMETER_FREEZE"
    assert plan["scientific_PCA_estimated"] is False and plan["writes"] is False
    assert plan["formal_estimator_implemented"] is False
    assert plan["authority"]["method_file_sha256"] == file_digest(ROOT / "protocol/METHOD_SPEC.md")
    assert not list(tmp_path.iterdir())


def test_scientific_execution_is_not_enabled_by_a_software_parameter_contract():
    parameters = explicit_software_parameters()
    assert pca.validate_parameters(parameters)
    with pytest.raises(pca.DriftPCAError, match="non-synthetic"):
        pca.validate_parameters(parameters, formal=True)
    with pytest.raises(PermissionError, match="no scientific estimator"):
        pca.fit_drift_pca_math(session_fixture(), parameters, rank=2)


def test_known_diagonal_drift_recovers_expected_symmetric_traceless_frobenius_directions():
    result = estimate()
    expected = modes()[:2]
    np.testing.assert_allclose(result.eigenvalues, [4.5, 2.0], atol=2e-14)
    np.testing.assert_allclose(projection(result.generators), projection(expected), atol=2e-14)
    np.testing.assert_allclose(result.generators, result.generators.transpose(0, 2, 1), atol=1e-15)
    np.testing.assert_allclose(np.trace(result.generators, axis1=1, axis2=2), 0, atol=2e-15)
    flattened = result.generators.reshape(2, 100)
    np.testing.assert_allclose(flattened @ flattened.T, np.eye(2), atol=2e-15)
    np.testing.assert_allclose(result.observation_weights, np.ones(8) / 8, atol=0)
    for person in result.participant_ids:
        indices = [i for i, key in enumerate(result.session_keys) if key[0] == person]
        np.testing.assert_allclose(result.centered_traceless_logs[indices].mean(0), 0, atol=1e-15)
    assert result.status == "SOFTWARE_TEST_DOUBLE_MATH_ONLY" and result.synthetic is True
    assert result.scientific_PCA_estimated is False


def test_person_offsets_and_session_scalar_gain_are_removed_without_global_amplitude_floor():
    baseline = estimate()
    nuisance = estimate(session_fixture(scalar_nuisance=True))
    np.testing.assert_allclose(projection(baseline.generators), projection(nuisance.generators), atol=2e-14)
    np.testing.assert_allclose(baseline.eigenvalues, nuisance.eigenvalues, atol=3e-14)
    tiny = [replace(session, covariance=expm(pca.symmetric_log(session.covariance) * 1e-8))
            for session in session_fixture()]
    small = estimate(tiny)
    assert small.eigenvalues[1] > 0
    np.testing.assert_allclose(small.eigenvalues / baseline.eigenvalues, [1e-16, 1e-16], rtol=1e-7, atol=0)


def test_frobenius_metric_counts_both_off_diagonal_entries():
    basis = traceless_basis(10, kind="symmetric").numpy()
    g1 = basis[:, 0].reshape(10, 10)
    g2 = basis[:, 1].reshape(10, 10)
    sessions = [pca.SoftwareSessionCovariance("SOFTWARE_PERSON", str(i), expm(a * g1 + b * g2), 20)
                for i, (a, b) in enumerate(((2, 0), (-2, 0), (0, 1), (0, -1)))]
    result = estimate(sessions)
    np.testing.assert_allclose(result.eigenvalues, [2., .5], atol=2e-14)
    np.testing.assert_allclose(projection(result.generators), projection(np.stack((g1, g2))), atol=3e-14)


def test_deterministic_tied_eigenspace_and_sign_convention():
    first, second = estimate(session_fixture(equal_variance=True)), estimate(session_fixture(equal_variance=True))
    np.testing.assert_array_equal(first.generators, second.generators)
    for generator in first.generators:
        flat = generator.reshape(-1)
        assert flat[np.argmax(np.abs(flat))] >= 0
    np.testing.assert_allclose(first.eigenvalues, [4.5, 4.5], atol=2e-14)


def test_explicit_subject_session_weights_and_trial_weighted_centering():
    sessions = session_fixture()
    sessions = sessions[:2] + sessions[4:]
    equal = estimate(sessions)
    np.testing.assert_allclose(equal.observation_weights, [.25, .25, .125, .125, .125, .125])
    weighted = explicit_software_parameters(within_subject_centering="valid_trial_count",
                                            pca_observation_weighting="session_equal")
    result = estimate(sessions, weighted)
    np.testing.assert_allclose(result.observation_weights, np.ones(6) / 6)
    for person in result.participant_ids:
        indices = [i for i, session in enumerate(sessions) if session.participant_id == person]
        trial_weights = [sessions[i].valid_trial_count for i in indices]
        np.testing.assert_allclose(np.average(result.centered_traceless_logs[indices], axis=0, weights=trial_weights), 0, atol=2e-15)
    global_centered = estimate(sessions, explicit_software_parameters(within_subject_centering="valid_trial_count",
        pca_observation_weighting="session_equal", global_center_after_subject_centering=True))
    np.testing.assert_allclose(np.average(global_centered.centered_traceless_logs, axis=0,
                                        weights=global_centered.observation_weights), 0, atol=1e-15)


def test_class_balanced_session_covariance_does_not_overweight_a_repeated_class():
    generator = np.random.default_rng(3)
    z = generator.normal(size=(8, 10, 128))
    z -= z.mean(2, keepdims=True)
    x = np.einsum("cd,ndt->nct", helmert_rows().numpy().T, z)
    y = np.array([0] * 3 + [1] * 5)
    parameters = explicit_software_parameters()
    expected = pca.session_covariance_math(x, y, parameters, software_test_double=True)
    indices = np.concatenate((np.arange(8), np.tile([0, 1, 2], 10)))
    actual = pca.session_covariance_math(x[indices], y[indices], parameters, software_test_double=True)
    np.testing.assert_allclose(expected, actual, atol=2e-16)
    np.testing.assert_allclose(np.trace(expected), 1, atol=1e-15)
    assert np.linalg.eigvalsh(expected).min() >= .007 - 1e-15
    gains = np.geomspace(1e-80, 1e80, 8)[:, None, None]
    np.testing.assert_allclose(pca.session_covariance_math(x * gains, y, parameters, software_test_double=True), expected, atol=2e-16)
    after = explicit_software_parameters(shrinkage_stage="after_balanced_session_mean")
    np.testing.assert_allclose(pca.session_covariance_math(x, y, after, software_test_double=True), expected, atol=2e-16)
    with pytest.raises(pca.DriftPCAError, match="both classes"):
        pca.session_covariance_math(x[y == 0], y[y == 0], parameters, software_test_double=True)


def test_whitening_coordinate_map_and_trace_order_are_explicit():
    generator = np.random.default_rng(9)
    z = generator.normal(size=(4, 10, 128))
    z -= z.mean(2, keepdims=True)
    x = np.einsum("cd,ndt->nct", helmert_rows().numpy().T, z)
    y = np.array([0, 1, 0, 1])
    inverse = np.diag(np.linspace(.5, 2., 10))
    parameters = explicit_software_parameters(covariance_coordinates="source_whitened_car10")
    actual = pca.session_covariance_math(x, y, parameters, inverse_source_root=inverse, software_test_double=True)
    mapped = inverse @ (helmert_rows().numpy() @ x)
    covariance = mapped @ mapped.transpose(0, 2, 1)
    covariance /= np.trace(covariance, axis1=1, axis2=2)[:, None, None]
    expected = .93 * (covariance[y == 0].mean(0) + covariance[y == 1].mean(0)) / 2 + .07 * np.eye(10) / 10
    np.testing.assert_allclose(actual, expected, atol=2e-16)
    with pytest.raises(pca.DriftPCAError, match="inverse root"):
        pca.session_covariance_math(x, y, parameters, software_test_double=True)
    with pytest.raises(pca.DriftPCAError, match="silently consume"):
        pca.session_covariance_math(x, y, explicit_software_parameters(), inverse_source_root=inverse, software_test_double=True)


def test_missing_observed_dimensions_or_sessions_never_invent_random_basis_vectors():
    with pytest.raises(pca.DriftPCAError, match="Insufficient observed"):
        estimate(rank=4)
    with pytest.raises(pca.DriftPCAError, match="two session"):
        estimate([session_fixture()[0], *session_fixture()[4:]])
    with pytest.raises(pca.DriftPCAError, match="Unique participant"):
        estimate(session_fixture() + session_fixture()[:1])
    with pytest.raises(pca.DriftPCAError, match="not SPD"):
        pca.symmetric_log(np.diag([0.] + [1.] * 9))


def test_role_permissions_precede_covariance_inspection_and_do_not_accept_external_aliases():
    class ForbiddenCovariance:
        def __array__(self, *args, **kwargs):
            pytest.fail("Unauthorized covariance payload was inspected")
    for role, dataset in (("meta_validation", "Stieger2021"), ("internal_holdout", "Stieger2021"),
                          ("meta_train", "HSStroke2026_v6")):
        bad = replace(session_fixture()[0], role=role, dataset=dataset, covariance=ForbiddenCovariance())
        with pytest.raises(PermissionError):
            estimate([bad, *session_fixture()[1:]])


def frozen_parameter_fixture(tmp_path):
    value = explicit_software_parameters()
    signal = signal_contract()
    value.update(status="FROZEN_IMPLEMENTATION_CONTRACT", synthetic=False,
                 input_contract_sha256=digest(signal), split_sha256=signal["window_audit_receipt"]["split_sha256"])
    for relative in ("protocol/METHOD_SPEC.md", "protocol/WORK_TASKS.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    amendment = {"status": "DRIFT_PCA_IMPLEMENTATION_AMENDMENT_FROZEN", "scope": pca.SOFTWARE_SCOPE,
                 "parameter_values_sha256": digest(value["parameters"]), "input_contract_sha256": digest(signal),
                 "split_sha256": value["split_sha256"], "allowed_dataset": "Stieger2021", "allowed_role": "meta_train",
                 "protocol_method_sha256": file_digest(tmp_path / "protocol/METHOD_SPEC.md"),
                 "protocol_work_tasks_sha256": file_digest(tmp_path / "protocol/WORK_TASKS.md"),
                 "registered_before_internal_or_external_query": True}
    path = tmp_path / "contracts/parameter_fixture_amendment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(amendment))
    value["implementation_amendment"] = {"path": path.relative_to(tmp_path).as_posix(), "sha256": file_digest(path)}
    (tmp_path / "contracts/input_contract.json").write_text(json.dumps(signal))
    return value


def test_unresolved_formal_inputs_stop_before_any_file_or_array_read(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Premature numerical data read"))
    with pytest.raises(pca.DriftPCAError, match="remain unresolved"):
        pca.verify_drift_inputs(tmp_path, pca.pending_parameter_contract(), source_receipt=None, reference_receipt=None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["W1", "W3"])
def test_formal_stage_gates_stop_before_source_or_numerical_cache_access(tmp_path, monkeypatch, stage):
    parameters = frozen_parameter_fixture(tmp_path)
    tasks = {"W1": {"status": "COMPLETE"}, "W3": {"status": "COMPLETE"}}
    tasks[stage]["status"] = "SOFTWARE_TEST_DOUBLE_PENDING"
    path = tmp_path / "status/tasks.json"
    path.parent.mkdir()
    path.write_text(json.dumps(tasks))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Premature numerical data read"))
    with pytest.raises(pca.DriftPCAError, match="Real complete W1 and W3"):
        pca.verify_drift_inputs(tmp_path, parameters, source_receipt="missing.json", reference_receipt="missing.json")


def test_mock_complete_w3_cannot_turn_software_fixtures_into_formal_pca_inputs(tmp_path):
    parameters = frozen_parameter_fixture(tmp_path)
    cache = tmp_path / "cache/software_fixture_manifest.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"synthetic": False, "test_double_not_research_evidence": True}))
    tasks = {"W1": {"status": "COMPLETE"}, "W3": {"status": "COMPLETE", "cache_manifest": {
        "path": cache.relative_to(tmp_path).as_posix(), "sha256": file_digest(cache)}}}
    path = tmp_path / "status/tasks.json"
    path.parent.mkdir()
    path.write_text(json.dumps(tasks))
    with pytest.raises(pca.DriftPCAError, match="software-double statuses"):
        pca.verify_drift_inputs(tmp_path, parameters, source_receipt="missing.json", reference_receipt="missing.json")


def test_math_does_not_consume_rng_or_emit_any_scientific_state(tmp_path):
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = estimate()
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert result.scientific_PCA_estimated is False
    assert not list(tmp_path.iterdir())
