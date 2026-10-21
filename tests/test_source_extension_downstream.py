"""Five-source downstream SOFTWARE acceptance on explicit synthetic CPU data.

These fixtures are mathematical input tensors, never real EEG, trained f0s,
completed meta fits, eligibility receipts, or production capabilities.  All
filesystem tests use an empty temporary root; sealed data are never opened.
"""
from dataclasses import replace
import copy
import hashlib
import inspect
import json
import signal
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
from torch import nn

from p3work import source_extension_downstream as downstream
from p3work.adapt import (CalibrationPrefix, FrozenSourceModel, PredictionInputs,
                         module_state_sha256)
from p3work.contracts import SOURCE_IDS
from p3work.contracts import file_digest
from p3work.method_registry import METHOD_IDS, registered_grid
from p3work.methods.lowrank import initialize_lowrank
from p3work.spatial_geometry import (SourceCovarianceTrial, helmert_rows,
                                    initialize_dictionary, source_reference)


torch.set_num_threads(1)
SIGNAL_SHA = "a" * 64  # Deliberately synthetic, not a real frozen protocol.
SPLIT_SHA = "b" * 64
SEED = 20261018
ARM = "BASE4_PLUS_YANG"
OTHER_ARM = "BASE4_PLUS_DREYER"
LEARNED = frozenset({"P", "P+PCT", "B8", "B10", "B11"})


def five_ids(arm=ARM):
    return frozenset(SOURCE_IDS | {downstream.ARMS[arm]})


def covariance_trials(arm=ARM):
    """Unequal trial counts require dataset/subject/trial, rather than trial, means."""
    q = helmert_rows()
    result = []
    for dataset_index, dataset in enumerate(sorted(five_ids(arm))):
        for person, n in (("small", 1), ("large", 4)):
            coordinate = (dataset_index + (person == "large")) % 10
            for index in range(n):
                z = torch.zeros(10, 128, dtype=torch.float64)
                z[coordinate, :2] = torch.tensor([1., -1.], dtype=torch.float64)
                result.append(SourceCovarianceTrial(dataset, person,
                    f"SYNTHETIC_ONLY/{dataset}/{person}/{index}", q.T @ z))
    return result


@pytest.fixture(scope="module")
def software():
    generator = torch.Generator().manual_seed(173)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(174)
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * 128, 6), nn.Tanh(),
                                nn.BatchNorm1d(6), nn.Dropout(.7)).double()
        source = FrozenSourceModel(encoder, nn.Linear(6, 2).double(), feature_dim=6)
    q = helmert_rows()
    xs, ys, rows = [], [], []
    for dataset in sorted(five_ids()):
        for person in ("one", "two"):
            for i in range(8):
                y = i % 2
                z = torch.randn(10, 128, dtype=torch.float64, generator=generator)
                z[y] *= 2.5
                z -= z.mean(1, keepdim=True)
                xs.append(q.T @ z)
                ys.append(y)
                rows.append({"dataset": dataset, "subject_id": person,
                    "role": "source_pretrain_train", "label": y,
                    "event_uid": f"SYNTHETIC_BANK_ONLY/{dataset}/{person}/{i}"})
    x, y = torch.stack(xs), torch.tensor(ys, dtype=torch.int64)
    reference = downstream.source_reference_five(covariance_trials(), five_ids())
    context = downstream.MathContext(source, reference, ARM, SEED, SIGNAL_SHA, SPLIT_SHA)
    support = CalibrationPrefix(x[:8], y[:8], tuple(f"synthetic/support/{i}" for i in range(8)),
        tuple(f"synthetic/support/{i}" for i in range(20)), SIGNAL_SHA,
        role="meta_validation_support")
    query = PredictionInputs(x[-6:], tuple(f"synthetic/future/{i}" for i in range(6)), SIGNAL_SHA)
    return SimpleNamespace(source=source, context=context, support=support, query=query,
                           x=x.numpy(), y=y.numpy(), rows=tuple(rows), reference=reference)


def configuration(method):
    return registered_grid(method)[0].copy()


def bank_configuration(method):
    config = configuration(method)
    return {"m": config["m"]} if method == "B4" else {"shrinkage": config["shrinkage"]}


def learned_tensor(method, config):
    """Explicit untrained synthetic tensors test math, not artifact provenance."""
    if method in {"B10", "B11"}:
        return initialize_lowrank(config["rank"], SEED + 11 + (method == "B11"))
    if method in {"P", "P+PCT", "B8"}:
        return initialize_dictionary(config["dimension"], SEED + 23 + sorted(LEARNED).index(method))
    return None


@pytest.fixture(scope="module")
def source_banks(software):
    return {method: downstream.fit_source_bank_math(software.x, software.y, software.rows,
        method, bank_configuration(method), five_ids(), SIGNAL_SHA) for method in ("B4", "B5")}


def fit(software, method, source_banks, *, support=None, context=None, config=None):
    config = configuration(method) if config is None else config
    return downstream.fit_support_math(context or software.context, support or software.support,
        method, config, learned=learned_tensor(method, config), bank=source_banks.get(method))


@pytest.mark.parametrize("arm", [ARM, OTHER_ARM])
def test_plan_is_exact_finite_and_cannot_issue_internal_or_external_access(arm, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A default software plan attempted CUDA or production context loading")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(downstream, "load_context", forbidden, raising=False)
    before = torch.get_rng_state().clone()
    result = downstream.build_plan(arm)
    assert result["planned_meta_jobs"] == 55
    assert result["screening_validation_jobs"] == 118
    assert result["additional_confirmation_jobs"] == 28
    assert result["source_bank_jobs"] == 6
    assert result["internal_allowed"] is False
    assert result["external_allowed"] is False
    assert result["actual_fits"] == 0
    assert torch.equal(before, torch.get_rng_state())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arm", ["BASE4", "Ma2022", "BASE4_PLUS_UNKNOWN", "../sealed"])
def test_unregistered_arm_fails_closed(arm, tmp_path):
    with pytest.raises((ValueError, PermissionError)):
        downstream.build_plan(arm, root=tmp_path)


def test_five_source_reference_uses_exact_uid_set_equal_hierarchy_and_fixed_shrinkage():
    trials = covariance_trials()
    result = downstream.source_reference_five(trials, five_ids())
    q = helmert_rows()
    dataset_means = []
    for dataset in sorted(five_ids()):
        people = sorted({r.subject for r in trials if r.dataset == dataset})
        means = []
        for person in people:
            covariance = []
            for row in trials:
                if (row.dataset, row.subject) == (dataset, person):
                    z = q @ row.x
                    c = z @ z.T
                    covariance.append(c / c.trace())
            means.append(torch.stack(covariance).mean(0))
        dataset_means.append(torch.stack(means).mean(0))
    expected = .95 * torch.stack(dataset_means).mean(0) + .005 * torch.eye(10, dtype=torch.float64)
    torch.testing.assert_close(result.matrix, expected, atol=2e-15, rtol=0)
    assert result.weighting == "dataset_equal/subject_equal/trial_equal" and result.shrinkage == .05
    assert result.event_uids == tuple(sorted(r.event_uid for r in trials))
    assert result.events_sha256 == hashlib.sha256("\n".join(result.event_uids).encode()).hexdigest()
    torch.testing.assert_close(result.sqrt @ result.invsqrt, torch.eye(10, dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.linalg.eigvalsh(result.matrix).min() >= .005 - 1e-15


def test_reference_ignores_positive_common_gain_and_reference_offsets():
    trials = covariance_trials()
    baseline = downstream.source_reference_five(trials, five_ids())
    changed = [replace(row, x=row.x * (10. ** (i % 7 - 3)) +
                       torch.linspace(-1., 1., row.x.shape[1], dtype=torch.float64)[None, :])
               for i, row in enumerate(trials)]
    actual = downstream.source_reference_five(changed, five_ids())
    torch.testing.assert_close(actual.matrix, baseline.matrix, atol=3e-14, rtol=1e-12)
    assert actual.events_sha256 == baseline.events_sha256


@pytest.mark.parametrize("violation", ["missing_source", "unexpected_source", "duplicate_uid", "missing_person",
    "source_validation", "meta_train", "sealed_query", "four_source_manifest", "unknown_fifth_manifest"])
def test_reference_provenance_and_source_only_roles_fail_closed(violation):
    trials = covariance_trials()
    expected = five_ids()
    if violation == "missing_source":
        trials = [r for r in trials if r.dataset != "Yang2025"]
    elif violation == "unexpected_source":
        trials[0] = replace(trials[0], dataset="HSStroke2026_v6")
    elif violation == "duplicate_uid":
        trials[1] = replace(trials[1], event_uid=trials[0].event_uid)
    elif violation == "missing_person":
        trials[0] = replace(trials[0], subject="")
    elif violation in {"source_validation", "meta_train", "sealed_query"}:
        trials[0] = replace(trials[0], role=violation)
    elif violation == "four_source_manifest":
        expected = SOURCE_IDS
        trials = [r for r in trials if r.dataset in SOURCE_IDS]
    else:
        expected = SOURCE_IDS | {"unregistered_fifth"}
        trials = [replace(r, dataset="unregistered_fifth") if r.dataset == "Yang2025" else r for r in trials]
    with pytest.raises((ValueError, PermissionError)):
        downstream.source_reference_five(trials, expected)


def test_primary_four_source_reference_remains_restricted():
    with pytest.raises(PermissionError, match="four"):
        source_reference(covariance_trials())
    assert SOURCE_IDS == frozenset({"BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI"})


def test_geometry_is_fixed_car11_and_context_is_math_only(software):
    geometry = software.context.geometry
    geometry.validate()
    assert torch.equal(geometry.q, helmert_rows())
    torch.testing.assert_close(geometry.q.T @ geometry.q,
        torch.eye(11, dtype=torch.float64) - torch.ones(11, 11, dtype=torch.float64) / 11,
        atol=3e-16, rtol=0)
    torch.testing.assert_close(geometry.rref_sqrt @ geometry.rref_invsqrt,
        torch.eye(10, dtype=torch.float64), atol=2e-14, rtol=0)
    assert not hasattr(software.context, "_seal")


@pytest.mark.parametrize("method", METHOD_IDS)
def test_all_fourteen_math_routes_predict_finite_two_logits_without_mutating_source(method, software, source_banks):
    before = module_state_sha256(software.source)
    buffers = {n: v.clone() for n, v in software.source.named_buffers()}
    rng = torch.get_rng_state().clone()
    fitted = fit(software, method, source_banks)
    result = downstream.predict_math(software.context, fitted, software.query)
    assert fitted.method == method and fitted.configuration == configuration(method)
    assert fitted.support_event_uids == software.support.event_uids
    assert fitted.acquired_event_uids == software.support.acquired_event_uids
    assert result.logits.shape == (len(software.query.x), 2)
    assert torch.isfinite(result.logits).all() and torch.isfinite(result.probabilities).all()
    torch.testing.assert_close(result.probabilities, result.logits.softmax(1), atol=1e-15, rtol=1e-14)
    assert torch.equal(result.labels, result.logits.argmax(1))
    assert result.event_uids == software.query.event_uids
    assert result.protocol_freeze_sha == SIGNAL_SHA
    assert module_state_sha256(software.source) == before
    assert all(torch.equal(v, buffers[n]) for n, v in software.source.named_buffers())
    assert all(p.grad is None and not p.requires_grad for p in software.source.parameters())
    assert not any(m.training for m in software.source.modules())
    assert torch.equal(rng, torch.get_rng_state())  # BN/dropout and random B9 stay frozen/local.


def test_b0_preserves_original_two_logits(software, source_banks):
    fitted = fit(software, "B0", source_banks)
    actual = downstream.predict_math(software.context, fitted, software.query)
    torch.testing.assert_close(actual.logits, software.source.source_logits(software.query.x), atol=0, rtol=0)


@pytest.mark.parametrize("method", ["B3", "B6", "B7", "B9", "P", "P+PCT", "B10", "B11"])
def test_single_class_support_retains_prefix_budget_and_replays_frozen_source(method, software, source_banks):
    support = replace(software.support, y=torch.zeros_like(software.support.y))
    fitted = fit(software, method, source_banks, support=support)
    assert fitted.updates == 0 and fitted.fallback_reason == "insufficient_class_support"
    assert fitted.support_event_uids == support.event_uids
    assert fitted.acquired_event_uids == support.acquired_event_uids
    result = downstream.predict_math(software.context, fitted, software.query)
    torch.testing.assert_close(result.logits, software.source.source_logits(software.query.x), atol=0, rtol=0)


@pytest.mark.parametrize("method", ["B0", "B2", "B3", "B4", "B5", "B6", "P", "B11"])
def test_predictions_are_per_trial_and_never_use_future_batch_statistics(method, software, source_banks):
    fitted = fit(software, method, source_banks)
    single = replace(software.query, x=software.query.x[:1], event_uids=software.query.event_uids[:1])
    changed = software.query.x.clone()
    changed[1:] *= 1e5
    actual = downstream.predict_math(software.context, fitted, replace(software.query, x=changed)).logits[:1]
    expected = downstream.predict_math(software.context, fitted, single).logits
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=1e-12)


@pytest.mark.parametrize("method", METHOD_IDS)
def test_support_role_freeze_and_all_acquired_uid_overlap_are_guarded(method, software, source_banks):
    with pytest.raises((PermissionError, ValueError)):
        fit(software, method, source_banks, support=replace(software.support, role="meta_train_support"))
    with pytest.raises((PermissionError, ValueError)):
        fit(software, method, source_banks, support=replace(software.support, role="calibration_prefix"))
    with pytest.raises((PermissionError, ValueError)):
        fit(software, method, source_banks, support=replace(software.support, protocol_freeze_sha="c" * 64))
    fitted = fit(software, method, source_banks)
    # s19 failed upstream QC and is not in x/y; it is still an acquired-prefix event.
    overlapping = replace(software.query, event_uids=(software.support.acquired_event_uids[-1],)
                          + software.query.event_uids[1:])
    with pytest.raises((PermissionError, ValueError)):
        downstream.predict_math(software.context, fitted, overlapping)
    with pytest.raises((PermissionError, ValueError)):
        downstream.predict_math(software.context, fitted,
                                replace(software.query, protocol_freeze_sha="d" * 64))


@pytest.mark.parametrize("method", sorted(LEARNED))
def test_learned_families_require_explicit_independent_matching_math_tensor(method, software):
    config = configuration(method)
    with pytest.raises((ValueError, PermissionError, TypeError)):
        downstream.fit_support_math(software.context, software.support, method, config)
    proper = learned_tensor(method, config)
    for invalid in (torch.zeros_like(proper), proper[:, :0], proper.float(), torch.full_like(proper, float("nan"))):
        with pytest.raises((ValueError, PermissionError, TypeError)):
            downstream.fit_support_math(software.context, software.support, method, config, learned=invalid)
    wrong = configuration(method)
    wrong["rank" if "rank" in wrong else "dimension"] *= 2
    with pytest.raises((ValueError, PermissionError, TypeError)):
        downstream.fit_support_math(software.context, software.support, method, wrong, learned=proper)


@pytest.mark.parametrize("method", ["B0", "B1", "B6", "B7", "B9"])
def test_nonlearned_routes_do_not_accept_an_ignored_learned_replacement(method, software):
    with pytest.raises((ValueError, PermissionError, TypeError)):
        downstream.fit_support_math(software.context, software.support, method, configuration(method),
                                   learned=initialize_dictionary(2, SEED))


@pytest.mark.parametrize("method", METHOD_IDS)
def test_nonregistered_configuration_cannot_change_grid_or_steps(method, software, source_banks):
    config = {**configuration(method), "steps": 1}
    with pytest.raises((ValueError, PermissionError, TypeError)):
        fit(software, method, source_banks, config=config)


@pytest.mark.parametrize("method", ["B3", "B4", "B5", "B9", "P", "B11"])
def test_fitted_state_cannot_be_transplanted_to_different_arm_or_seed(method, software, source_banks):
    fitted = fit(software, method, source_banks)
    for arm, seed, split in ((OTHER_ARM, SEED, SPLIT_SHA), (ARM, SEED + 1, SPLIT_SHA),
                             (ARM, SEED, "e" * 64)):
        with pytest.raises((ValueError, PermissionError)):
            context = downstream.MathContext(software.source, software.reference, arm, seed, SIGNAL_SHA, split)
            downstream.predict_math(context, fitted, software.query)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_classical_source_bank_requires_all_five_training_sources_and_bound_labels(method, software):
    rows = list(software.rows)
    mutations = [
        (replace_row(rows, 0, role="meta_train"), software.y),
        (replace_row(rows, 0, role="sealed_query"), software.y),
        (replace_row(rows, 0, dataset="HSStroke2026_v6"), software.y),
        (replace_row(rows, 1, event_uid=rows[0]["event_uid"]), software.y),
        (replace_row(rows, 0, label=1 - rows[0]["label"]), software.y),
    ]
    for invalid_rows, labels in mutations:
        with pytest.raises((ValueError, PermissionError)):
            downstream.fit_source_bank_math(software.x, labels, invalid_rows,
                method, bank_configuration(method), five_ids(), SIGNAL_SHA)
    indices = [i for i, row in enumerate(rows) if row["dataset"] != "Yang2025"]
    with pytest.raises((ValueError, PermissionError)):
        downstream.fit_source_bank_math(software.x[indices], software.y[indices], tuple(rows[i] for i in indices),
            method, bank_configuration(method), five_ids(), SIGNAL_SHA)


def replace_row(rows, index, **changes):
    result = [dict(row) for row in rows]
    result[index].update(changes)
    return tuple(result)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_source_bank_is_required_and_method_specific(method, software, source_banks):
    config = configuration(method)
    with pytest.raises((ValueError, PermissionError, TypeError)):
        downstream.fit_support_math(software.context, software.support, method, config)
    other = "B5" if method == "B4" else "B4"
    with pytest.raises((ValueError, PermissionError, TypeError)):
        downstream.fit_support_math(software.context, software.support, method, config, bank=source_banks[other])


def test_label_free_prediction_and_support_fit_interfaces():
    parameters = inspect.signature(downstream.fit_support_math).parameters
    assert "query" not in parameters and "query_y" not in parameters
    assert "y" not in PredictionInputs.__dataclass_fields__
    assert "labels" not in PredictionInputs.__dataclass_fields__


def test_b3_uses_its_own_five_source_authorization_without_weakening_primary_validator(software, source_banks, monkeypatch):
    from p3work.methods import neural_controls
    def forbidden(*args, **kwargs):
        pytest.fail("Extension B3 borrowed the primary four-source authorization")
    monkeypatch.setattr(neural_controls.SourceReferenceAuthorization, "validate", forbidden)
    fitted = fit(software, "B3", source_banks)
    result = downstream.predict_math(software.context, fitted, software.query)
    assert torch.isfinite(result.logits).all()


def alter_first_numerical_value(payload):
    for key, value in payload.items():
        if isinstance(value, torch.Tensor) and value.numel():
            value.reshape(-1)[0] += .125
            return True
        if isinstance(value, np.ndarray) and value.size:
            value.reshape(-1)[0] += .125
            return True
        if isinstance(value, dict) and alter_first_numerical_value(value):
            return True
    return False


@pytest.mark.parametrize("method", METHOD_IDS)
def test_any_changed_adaptation_payload_is_rejected_before_prediction(method, software, source_banks):
    fitted = copy.deepcopy(fit(software, method, source_banks))
    assert alter_first_numerical_value(fitted.payload)
    with pytest.raises((ValueError, PermissionError), match="changed"):
        downstream.predict_math(software.context, fitted, software.query)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_unauthorized_source_role_is_rejected_before_signal_payload_is_inspected(method, software):
    class Unreadable:
        def __array__(self, *args, **kwargs):
            pytest.fail("Unauthorized source signal payload was inspected")
    rows = replace_row(software.rows, 0, role="sealed_query")
    with pytest.raises(PermissionError):
        downstream.fit_source_bank_math(Unreadable(), software.y, rows, method,
            bank_configuration(method), five_ids(), SIGNAL_SHA)


def test_unauthorized_reference_role_is_rejected_before_signal_payload_is_inspected():
    class Unreadable:
        def detach(self):
            pytest.fail("Unauthorized reference signal payload was inspected")
    trials = covariance_trials()
    trials[0] = replace(trials[0], x=Unreadable(), role="meta_validation")
    with pytest.raises(PermissionError):
        downstream.source_reference_five(trials, five_ids())


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_source_bank_math_metadata_does_not_claim_a_completed_scientific_fit(method, source_banks):
    bank = source_banks[method]
    assert bank.metadata["scope"] == "MATHEMATICAL_PAYLOAD_NOT_SCIENTIFIC_RECEIPT"
    assert set(bank.metadata["source_dataset_ids"]) == five_ids()
    assert bank.metadata["signal_sha"] == SIGNAL_SHA
    assert not hasattr(bank, "_seal")
    bank.assert_unchanged()


def test_b3_alignment_direction_matches_its_own_five_source_reference(software, source_banks):
    fitted = fit(software, "B3", source_banks)
    alignment = fitted.payload["alignment"]
    target = fitted.payload["target_shrunk_covariance"]
    q = software.context.geometry.q
    effective = q @ alignment @ q.T
    torch.testing.assert_close(effective @ target @ effective.T, software.reference.matrix,
                               atol=2e-14, rtol=1e-12)


def test_b4_full_source_mix_keeps_source_filters_and_solves_only_on_support(software, source_banks):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from p3work.methods.classical import csp_features
    config = {"m": 1, "alpha": 1.}
    fitted = downstream.fit_support_math(software.context, software.support, "B4", config,
                                        bank=source_banks["B4"])
    arrays = fitted.payload["arrays"]
    np.testing.assert_array_equal(arrays["filters"], source_banks["B4"].arrays["filters"])
    features = csp_features(software.support.x.numpy(), arrays["filters"])
    exact = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[.5, .5]).fit(
        features, software.support.y.numpy())
    np.testing.assert_array_equal(arrays["coef"], exact.coef_.reshape(-1))
    np.testing.assert_array_equal(arrays["bias"], exact.intercept_[0])
    assert fitted.updates == 0 and fitted.fallback_reason is None


@pytest.mark.parametrize("path", ["../outside.json", "/absolute.json", "runs/../sealed/data.npy",
    "runs\\outside.json", ".env", "runs/.env.production", ".aws/config", "secrets/key.json",
    "legacy_baseline_v1/old.json", "external/data.npy", "sealed/query.npy",
    "internal_holdout/events.json", "cache/old.npz", "runs/result.pickle", "runs/private.pem"])
def test_extension_paths_cannot_open_legacy_sealed_or_private_inputs(path, tmp_path):
    with pytest.raises((ValueError, PermissionError)):
        downstream.local(tmp_path, path, exists=False)


def test_symlinks_cannot_disguise_an_input_path(tmp_path):
    value = tmp_path / "fixture.json"
    value.write_text('{"scope":"SYNTHETIC_SOFTWARE_ONLY"}')
    link = tmp_path / "alias.json"
    link.symlink_to(value)
    with pytest.raises((ValueError, PermissionError), match="symlink"):
        downstream.local(tmp_path, "alias.json")


@pytest.mark.parametrize("violation", ["sha", "bytes"])
def test_extension_artifact_descriptor_checks_actual_bytes(violation, tmp_path):
    value = tmp_path / "fixture.json"
    value.write_text('{"scope":"SYNTHETIC_SOFTWARE_ONLY"}')
    item = {"path": value.name, "sha256": file_digest(value), "bytes": value.stat().st_size}
    assert downstream.checked(tmp_path, item) == value
    item[violation if violation == "bytes" else "sha256"] = item["bytes"] + 1 if violation == "bytes" else "f" * 64
    with pytest.raises((ValueError, PermissionError), match="changed"):
        downstream.checked(tmp_path, item)


def production_catalog_stub():
    """No freeze, source checkpoint, reference, or completion proof is supplied."""
    return {"schema_version": downstream.SCHEMA, "arm": ARM,
        "input_contract_sha256": downstream.INPUT_SHA,
        "internal_allowed": False, "external_allowed": False,
        "fixture_scope": "SYNTHETIC_SOFTWARE_GATE_INPUT_NOT_A_COMPLETION_RECEIPT"}


@pytest.mark.parametrize("violation", ["missing_freeze", "schema", "arm", "signal", "internal", "external"])
def test_unqualified_production_catalog_cannot_reach_f0_episode_or_gpu(violation, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An unqualified production catalog reached a source, episode, or CUDA edge")
    monkeypatch.setattr(downstream.bridge, "load_verified_extension_source", forbidden)
    monkeypatch.setattr(downstream, "load_episode_catalog", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    catalog = production_catalog_stub()
    if violation == "schema":
        catalog["schema_version"] = "primary_four_source_schema_cannot_authorize_extension"
    elif violation == "arm":
        catalog["arm"] = "Ma2022"
    elif violation == "signal":
        catalog["input_contract_sha256"] = "f" * 64
    elif violation in {"internal", "external"}:
        catalog[violation + "_allowed"] = True
    with pytest.raises((ValueError, PermissionError)):
        downstream.load_context(tmp_path, catalog, SEED)
    with pytest.raises((ValueError, PermissionError)):
        downstream.prepare_reference(tmp_path, catalog, SEED)
    assert not list(tmp_path.iterdir())


def test_math_context_cannot_publish_or_load_production_fit_artifacts(software, source_banks, tmp_path):
    state = fit(software, "B0", source_banks)
    output = tmp_path / "must_not_be_created"
    with pytest.raises(PermissionError):
        downstream.save_adapter(software.context, state, output, {})
    with pytest.raises(PermissionError):
        downstream.load_adapter(software.context, {}, {})
    assert not output.exists() and not list(tmp_path.iterdir())


def test_direct_production_context_construction_cannot_issue_a_capability(software, tmp_path):
    with pytest.raises(PermissionError, match="factory"):
        downstream.ProductionContext(software.source, software.reference, ARM, SEED, SIGNAL_SHA, SPLIT_SHA,
            root=tmp_path, catalog=production_catalog_stub(), source_receipt={}, reference_receipt={},
            source_cache_sha256="c" * 64, training_UID_set_sha256="d" * 64, code={}, issuer=object())
    assert not list(tmp_path.iterdir())


def test_same_parameter_bytes_with_different_source_operator_cannot_reuse_an_adaptation(software, source_banks):
    source = copy.deepcopy(software.source)
    source.encoder[2] = nn.ReLU().eval()  # A zero-parameter operator change evades a weights-only hash.
    source.assert_unchanged()
    assert source.frozen_state_sha256 == software.source.frozen_state_sha256
    assert not torch.equal(source.source_logits(software.query.x), software.source.source_logits(software.query.x))
    context = downstream.MathContext(source, software.reference, ARM, SEED, SIGNAL_SHA, SPLIT_SHA)
    fitted = fit(software, "B0", source_banks)
    with pytest.raises((ValueError, PermissionError)):
        downstream.predict_math(context, fitted, software.query)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_float32_native_cache_roundoff_is_accepted_by_classical_source_bank(method, software):
    """A float64 cast must not turn legitimate cache roundoff into invalid CAR.

    This is synthetic, already CAR/time-demeaned input rounded once to the
    production cache dtype. Cache precision remains relevant to validation
    even though the covariance product itself uses float64 arithmetic.
    """
    from p3work.methods.classical import trace_covariances
    x = software.x.astype(np.float32)
    before = x.copy()
    trace_covariances(x)  # Establish the numerical kernel accepts this dtype.
    bank = downstream.fit_source_bank_math(x, software.y, software.rows, method,
        bank_configuration(method), five_ids(), SIGNAL_SHA)
    bank.assert_unchanged()
    assert bank.metadata["scope"] == "MATHEMATICAL_PAYLOAD_NOT_SCIENTIFIC_RECEIPT"
    assert all(np.isfinite(array).all() for array in bank.arrays.values())
    assert x.dtype == np.float32 and np.array_equal(x, before)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_float32_native_support_is_not_recast_before_classical_validation(method, software, source_banks):
    support = replace(software.support, x=software.support.x.float())
    before = support.x.clone()
    fitted = fit(software, method, source_banks, support=support)
    prediction = downstream.predict_math(software.context, fitted, software.query)
    assert fitted.support_event_uids == support.event_uids
    assert torch.isfinite(prediction.logits).all()
    assert support.x.dtype == torch.float32 and torch.equal(support.x, before)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_float32_native_query_is_not_recast_before_classical_validation(method, software, source_banks):
    fitted = fit(software, method, source_banks)
    query = replace(software.query, x=software.query.x.float())
    before = query.x.clone()
    prediction = downstream.predict_math(software.context, fitted, query)
    assert prediction.event_uids == query.event_uids
    assert torch.isfinite(prediction.logits).all()
    assert query.x.dtype == torch.float32 and torch.equal(query.x, before)


def _software_sealed_context(software, tmp_path, monkeypatch):
    """Isolate context immutability; this never supplies real receipt authority.

    The filesystem validators are deliberately replaced only in this empty
    temporary-root software test. No receipt, fit or eligibility is published.
    """
    context = downstream.ProductionContext(copy.deepcopy(software.source),
        copy.deepcopy(software.reference), ARM, SEED, SIGNAL_SHA, SPLIT_SHA,
        root=tmp_path, catalog=production_catalog_stub(),
        source_receipt={"path": "synthetic-f0.json", "sha256": "c" * 64},
        reference_receipt={"path": "synthetic-reference.json", "sha256": "d" * 64},
        source_cache_sha256="e" * 64, training_UID_set_sha256="f" * 64,
        code={}, issuer=downstream._ISSUER)
    # Issuer identity alone must not constitute a production capability.
    with pytest.raises(PermissionError, match="unissued"):
        context.assert_current()
    monkeypatch.setattr(downstream, "checked", lambda root, item: tmp_path / item["path"])
    monkeypatch.setattr(downstream, "_code", lambda root: {})
    downstream._seal_context(context)
    context.assert_current()
    return context


@pytest.mark.parametrize("mutation", ["matrix", "sqrt", "invsqrt", "matrix_stride", "sqrt_stride",
                                      "invsqrt_stride", "BN_eps", "operator", "feature_dim"])
def test_issued_context_rejects_pre_first_fit_reference_or_source_operator_change(
        mutation, software, tmp_path, monkeypatch):
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    before_weights = module_state_sha256(context.source)
    if mutation in {"matrix", "sqrt", "invsqrt"}:
        getattr(context.reference, mutation).mul_(1.0001)
    elif mutation.endswith("_stride"):
        value = getattr(context.reference, mutation.removesuffix("_stride"))
        before_stride = value.stride()
        value.transpose_(0, 1)
        assert value.stride() != before_stride
    elif mutation == "BN_eps":
        context.source.encoder[3].eps *= 2
    elif mutation == "operator":
        context.source.encoder[2] = nn.ReLU().eval()
    else:
        context.source.feature_dim += 1
    assert module_state_sha256(context.source) == before_weights
    with pytest.raises((ValueError, PermissionError, RuntimeError)):
        downstream._production(context)
    assert not list(tmp_path.iterdir())


def test_pure_math_payload_cannot_be_published_under_an_issued_context(
        software, tmp_path, monkeypatch):
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    state = downstream.fit_support_math(context, software.support, "B0", {})
    candidate = {"arm": ARM, "context": context.production_identity}
    folder = tmp_path / "runs" / "source_extension_downstream" / ARM / "synthetic-not-a-production-fit"
    with pytest.raises(PermissionError, match="production_dispatched_fit"):
        downstream.save_adapter(context, state, folder, candidate)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["B0", "B3", "B4", "B5"])
def test_support_math_rejects_ambient_autocast_for_all_operator_families(method, software, source_banks):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises((ValueError, RuntimeError), match="precision|autocast|AMP|mixed"):
            fit(software, method, source_banks)


@pytest.mark.parametrize("method", ["B0", "B3", "B4", "B5"])
def test_prediction_math_rejects_ambient_autocast_even_for_zero_update_or_numpy_routes(
        method, software, source_banks):
    state = fit(software, method, source_banks)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises((ValueError, RuntimeError), match="precision|autocast|AMP|mixed"):
            downstream.predict_math(software.context, state, software.query)


def _software_prediction_plan(software):
    """A plain CPU fixture; never a verified catalog plan or EEG observation."""
    values = {"participant_id": "SYNTHETIC_ONLY", "session_id": "software-1",
        "role": "meta_validation", "query_event_uids": list(software.query.event_uids)}
    return SimpleNamespace(**{**values, "query_event_uids": software.query.event_uids},
                           to_dict=lambda: copy.deepcopy(values))


def test_unissued_self_consistent_prediction_cannot_publish_under_issued_context(
        software, tmp_path, monkeypatch):
    """Correct shapes, softmax and UID strings cannot substitute for dispatch."""
    from p3work.adapt import PredictionResult
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    plan = _software_prediction_plan(software)
    logits = torch.zeros(len(plan.query_event_uids), 2, dtype=torch.float64)
    prediction = PredictionResult(logits, logits.softmax(1), logits.argmax(1),
        plan.query_event_uids, downstream.INPUT_SHA)
    output = tmp_path / "runs" / "source_extension_downstream" / ARM / "unissued-prediction-must-not-exist"
    with pytest.raises((ValueError, PermissionError), match="production_prediction|dispatch|unissued"):
        downstream._commit_prediction(context, {"fixture_scope": "SYNTHETIC_ONLY"}, plan,
            prediction, {"fixture_scope": "NO_ACTUAL_FIT"}, {}, output)
    assert not output.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", ["runs/source_extension_downstream/BASE4_PLUS_DREYER/cross-arm-prediction",
                                      "runs/primary-development/software-prediction-must-not-exist"])
def test_prediction_commit_rejects_other_arm_or_primary_output_path(relative, software, tmp_path, monkeypatch):
    """A fabricated software seal isolates output ownership, never a real fit."""
    from p3work.adapt import PredictionResult
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    plan = _software_prediction_plan(software)
    logits = torch.zeros(len(plan.query_event_uids), 2, dtype=torch.float64)
    prediction = PredictionResult(logits, logits.softmax(1), logits.argmax(1),
        plan.query_event_uids, downstream.INPUT_SHA)
    seal = {"context": context.production_identity, "job": {}, "plan": plan.to_dict(),
        "fitted": {}, "candidate_sha256": downstream.digest({}), "query_x_sha256": "0" * 64,
        "numeric_sha256": downstream._tensor_tree_sha({"logits": prediction.logits,
            "probabilities": prediction.probabilities, "labels": prediction.labels})}
    monkeypatch.setitem(downstream._PREDICTION_SEALS, id(prediction), (weakref.ref(prediction), seal))
    with pytest.raises((ValueError, PermissionError), match="owned_arm_output"):
        downstream._commit_prediction(context, {}, plan, prediction, {}, {}, tmp_path / relative)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["numeric", "context", "job", "plan", "fitted", "candidate", "object"])
def test_prediction_commit_rejects_numeric_or_dispatch_binding_mutation(
        mutation, software, tmp_path, monkeypatch):
    """Install an explicit software seal to isolate its rejection checks.

    This never calls a fitter, a real loader, the GPU or the publication success
    path. The fabricated seal supplies no actual scientific provenance.
    """
    from p3work.adapt import PredictionResult
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    plan = _software_prediction_plan(software)
    logits = torch.zeros(len(plan.query_event_uids), 2, dtype=torch.float64)
    prediction = PredictionResult(logits, logits.softmax(1), logits.argmax(1),
        plan.query_event_uids, downstream.INPUT_SHA)
    job = {"fixture_scope": "SYNTHETIC_ONLY"}
    fitted = {"fixture_scope": "NO_ACTUAL_FIT"}
    candidate = {"fixture_scope": "NO_REAL_CANDIDATE"}
    seal = {"context": context.production_identity, "job": copy.deepcopy(job),
        "plan": plan.to_dict(), "fitted": copy.deepcopy(fitted),
        "candidate_sha256": downstream.digest(candidate), "query_x_sha256": "0" * 64,
        "numeric_sha256": downstream._tensor_tree_sha({"logits": prediction.logits,
            "probabilities": prediction.probabilities, "labels": prediction.labels})}
    issued_object = prediction if mutation != "object" else replace(prediction)
    monkeypatch.setitem(downstream._PREDICTION_SEALS, id(prediction), (weakref.ref(issued_object), seal))
    if mutation == "numeric":
        # A common logit offset preserves probabilities/argmax but changes the
        # numerical prediction; structural consistency alone would accept it.
        prediction.logits.add_(1.)
    elif mutation == "context":
        seal["context"] = "0" * 64
    elif mutation == "job":
        job["different_job"] = True
    elif mutation == "plan":
        seal["plan"]["session_id"] = "other-software-session"
    elif mutation == "fitted":
        fitted["different_fitted_payload"] = True
    elif mutation == "candidate":
        candidate["different_support_binding"] = True
    output = tmp_path / "runs" / "source_extension_downstream" / ARM / "mutated-prediction-must-not-exist"
    with pytest.raises((ValueError, PermissionError), match="production_prediction|dispatch|unissued"):
        downstream._commit_prediction(context, job, plan, prediction, fitted, candidate, output)
    assert not output.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("binding", ["job", "fitted"])
def test_prediction_factory_seal_captures_original_binding_without_caller_dict_alias(
        binding, software, tmp_path, monkeypatch):
    """CPU provenance-wrapper isolation with explicitly substituted loaders.

    The fake loaders below supply no real fit/checkpoint/catalog authority. No
    fitter runs: an empty mathematical B0 state only requests a frozen-source
    forward. The only behavior under test is an immutable dispatch snapshot.
    """
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    context = replace(context, signal_sha=downstream.INPUT_SHA)
    downstream._seal_context(context)
    plan = _software_prediction_plan(software)
    query = replace(software.query, protocol_freeze_sha=downstream.INPUT_SHA)
    state = downstream._adapter(context, software.support, "B0", {}, {})
    catalog = SimpleNamespace(plan=lambda *args: plan,
        materialize=lambda *args, **kwargs: SimpleNamespace(prediction_inputs=query))
    monkeypatch.setattr(downstream, "load_adapter", lambda *args: state)
    monkeypatch.setattr(downstream, "load_episode_catalog", lambda *args, **kwargs: catalog)
    job = {"method": "B0", "fixture_scope": "NO_REAL_JOB"}
    fitted = {"path": "no-actual-fit.json", "fixture_scope": "SOFTWARE_LOADER_SUBSTITUTE"}
    candidate = {"fixture_scope": "NO_REAL_CANDIDATE"}
    prediction = downstream._production_predict(context, job, plan, fitted, candidate)
    if binding == "job":
        job["fixture_scope"] = "CALLER_CHANGED_AFTER_PREDICTION"
    else:
        fitted["path"] = "different-nonexistent-fit.json"
    output = tmp_path / "runs" / "source_extension_downstream" / ARM / "aliased-prediction-must-not-exist"
    with pytest.raises((ValueError, PermissionError), match="production_prediction|dispatch|unissued"):
        downstream._commit_prediction(context, job, plan, prediction, fitted, candidate, output)
    assert not output.exists() and not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", sorted(LEARNED))
def test_engineering_job_is_unique_predeclared_largest_grid_setting(method):
    job = downstream.engineering_job(method)
    assert job == downstream.MetaJob(method, 4 if method in {"B10", "B11"} else 8,
                                    .35, SEED, "engineering")
    assert job.updates == 100


@pytest.mark.parametrize("method,config", [(method, config) for method in sorted(LEARNED)
    for config in registered_grid(method)
    if (config.get("dimension", config.get("rank")), config["radius"])
       != (4 if method in {"B10", "B11"} else 8, .35)])
def test_noncanonical_engineering_grid_cannot_reach_context_even_in_default_plan(
        method, config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A noncanonical engineering job reached an actual context/GPU edge")
    monkeypatch.setattr(downstream, "load_context", forbidden)
    monkeypatch.setattr(downstream, "_runtime", forbidden)
    job = downstream.MetaJob(method, config.get("dimension", config.get("rank")),
                             config["radius"], SEED, "engineering")
    with pytest.raises((ValueError, PermissionError), match="engineering"):
        downstream.execute_meta(tmp_path, production_catalog_stub(), job)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", sorted(LEARNED))
def test_noncanonical_engineering_receipt_fails_before_loading_real_context(method, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An unqualified engineering receipt reached a real source/context")
    monkeypatch.setattr(downstream, "load_context", forbidden)
    job = downstream.MetaJob(method, 1 if method in {"B10", "B11"} else 2,
                             .05, SEED, "engineering")
    path = tmp_path / "synthetic-job-only-not-a-completion.json"
    path.write_text(json.dumps({"job": downstream.asdict(job), "fixture_scope": "SYNTHETIC_ONLY"}))
    with pytest.raises((ValueError, PermissionError), match="engineering"):
        downstream.verify_meta(tmp_path, downstream.descriptor(tmp_path, path), production_catalog_stub())


def _software_resource_ledger():
    claim = {"identity_sha256": "c" * 64, "dispatched": True}
    return {"identity_sha256": "a" * 64, "target_fits": {"software-target": copy.deepcopy(claim)},
        "bank_fits": {"software-bank": copy.deepcopy(claim)}, "meta_calls": {"software-meta": 5},
        "source_calls": {"software-source": 7}, "stages": {"software-stage": {
            "identity_sha256": "d" * 64, "kind": "bounded_stage", "dispatched": True,
            "wall_high_water_seconds": 4.}}, "wall_high_water_seconds": 8.}


@pytest.mark.parametrize("field", ["meta_calls", "source_calls"])
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "2", float("inf"), float("nan")])
def test_resource_ledger_rejects_negative_or_noninteger_physical_counts(field, invalid):
    ledger = _software_resource_ledger()
    ledger[field]["malformed-counter"] = invalid
    with pytest.raises((ValueError, PermissionError), match="physical|integer|ledger"):
        downstream._validate_ledger(ledger)


@pytest.mark.parametrize("field", ["target_fits", "bank_fits"])
@pytest.mark.parametrize("mutation", ["missing_dispatch", "false_dispatch", "malformed_sha"])
def test_resource_ledger_requires_complete_dispatch_claims(field, mutation):
    ledger = _software_resource_ledger()
    key = next(iter(ledger[field]))
    if mutation == "missing_dispatch":
        del ledger[field][key]["dispatched"]
    elif mutation == "false_dispatch":
        ledger[field][key]["dispatched"] = False
    else:
        ledger[field][key]["identity_sha256"] = "not-a-SHA"
    with pytest.raises((ValueError, PermissionError), match="dispatch|SHA"):
        downstream._validate_ledger(ledger)


@pytest.mark.parametrize("mutation", ["identity", "target_missing", "bank_missing", "target_changed",
    "meta_rollback", "source_rollback", "meta_missing", "stage_missing", "stage_changed", "stage_wall_rollback"])
def test_immutable_resource_snapshot_cannot_become_a_nonprefix_of_live_ledger(mutation):
    old = _software_resource_ledger()
    current = copy.deepcopy(old)
    if mutation == "identity":
        current["identity_sha256"] = "f" * 64
    elif mutation in {"target_missing", "bank_missing"}:
        current["target_fits" if mutation.startswith("target") else "bank_fits"].clear()
    elif mutation == "target_changed":
        current["target_fits"]["software-target"]["identity_sha256"] = "e" * 64
    elif mutation == "meta_rollback":
        current["meta_calls"]["software-meta"] -= 1
    elif mutation == "source_rollback":
        current["source_calls"]["software-source"] -= 1
    elif mutation == "meta_missing":
        current["meta_calls"].clear()
    elif mutation == "stage_missing":
        current["stages"].clear()
    elif mutation == "stage_changed":
        current["stages"]["software-stage"]["identity_sha256"] = "f" * 64
    else:
        current["stages"]["software-stage"]["wall_high_water_seconds"] -= 1
    with pytest.raises((ValueError, PermissionError), match="rolled_back|identity"):
        downstream._ledger_prefix(old, current)


def test_resource_prefix_accepts_only_preserved_claims_and_monotone_new_work():
    old = _software_resource_ledger()
    current = copy.deepcopy(old)
    current["meta_calls"]["software-meta"] += 1
    current["source_calls"]["software-source"] += 2
    current["stages"]["software-stage"]["wall_high_water_seconds"] += 1
    current["target_fits"]["new-software-dispatch"] = {
        "identity_sha256": "e" * 64, "dispatched": True}
    downstream._ledger_prefix(old, current)


def _software_arm_run(tmp_path, monkeypatch):
    """Real local operational journals; no real budget, data, fit or GPU."""
    budget = {"fixture_scope": "SOFTWARE_OPERATIONAL_TEST_ONLY", "max_total_wall_seconds": 100.,
        "max_target_fits": 10, "max_source_bank_fits": 10,
        "max_observed_outer_calls": 30, "max_source_f0_optimizer_calls": 30,
        "runtime": {"fixture_scope": "CPU_SOFTWARE_ONLY"}}
    budget_path = tmp_path / "software-budget-not-authoritative.json"
    budget_path.write_text(json.dumps(budget))
    monkeypatch.setattr(downstream, "_budget", lambda *args: (
        budget, downstream.descriptor(tmp_path, budget_path)))
    return downstream.ArmRun(tmp_path, production_catalog_stub(), {"fixture_scope": "CPU_SOFTWARE_ONLY"})


def test_fixed_budget_cannot_use_different_primary_measured_device(tmp_path, monkeypatch):
    run = _software_arm_run(tmp_path, monkeypatch)
    run.close()
    with pytest.raises(ValueError, match="actual_primary_profile_same_device_runtime_required"):
        downstream.ArmRun(tmp_path, production_catalog_stub(), {"fixture_scope": "OTHER_SOFTWARE_DEVICE"})


def _software_profile_plan(support, *, query_eligible=True, fallback_reason=None):
    return SimpleNamespace(query_eligible=query_eligible, fallback_reason=fallback_reason,
        acquired_event_uids=support.acquired_event_uids, support_event_uids=support.event_uids)


def test_profile_skips_first_query_eligible_fallback_using_only_support(software):
    """Synthetic metadata/support selection, never a fit or query read."""
    from p3work.development_evaluation import _profile_session as primary_profile_session
    support = software.support
    first = _software_profile_plan(support, fallback_reason="insufficient_class_support")
    ineligible = _software_profile_plan(support, query_eligible=False)
    chosen = _software_profile_plan(support)
    calls = []
    def materialize_support(plan, *, device):
        assert plan is chosen and device == "cpu"
        calls.append(plan)
        return support
    def forbidden_query(*args, **kwargs):
        raise AssertionError("profile selection must not materialize a query")
    catalog = SimpleNamespace(plans=(first, ineligible, chosen),
        materialize_support=materialize_support, materialize=forbidden_query)
    assert downstream._profile_session(catalog) == (chosen, support)
    assert primary_profile_session(catalog) == (chosen, support)
    assert calls == [chosen, chosen]


def test_profile_metadata_cannot_disguise_single_class_support(software):
    support = replace(software.support, y=torch.zeros_like(software.support.y))
    plan = _software_profile_plan(support)
    catalog = SimpleNamespace(plans=(plan,), materialize_support=lambda *args, **kwargs: support)
    with pytest.raises(ValueError, match="actual_two_class_B20_support_disagree"):
        downstream._profile_session(catalog)


def test_profile_refuses_catalog_without_nonfallback_two_class_B20(software):
    plan = _software_profile_plan(software.support, fallback_reason="insufficient_class_support")
    def forbidden(*args, **kwargs):
        raise AssertionError("rejected profile metadata must not read any tensor")
    with pytest.raises(ValueError, match="no_two_class_B20_profile_session"):
        downstream._profile_session(SimpleNamespace(plans=(plan,), materialize_support=forbidden))


@pytest.mark.parametrize("state_reason,timing", [
    ("insufficient_class_support", {"fallback_reason": "insufficient_class_support"}),
    (None, {"fallback_reason": "insufficient_class_support"}),
    ("insufficient_class_support", {"fallback_reason": None}),
    (None, {}),
])
def test_profile_cannot_use_fallback_or_missing_measurement(state_reason, timing):
    with pytest.raises(ValueError, match="fallback_cannot_supply_real_calibration_profile"):
        downstream._profile_fit_required(SimpleNamespace(fallback_reason=state_reason), timing)


@pytest.mark.parametrize("kind", ["target_fits", "bank_fits"])
def test_unknown_dispatched_numerical_fit_cannot_be_retried(kind, tmp_path, monkeypatch):
    run = _software_arm_run(tmp_path, monkeypatch)
    identity = {"fixture_scope": "DISPATCH_ONLY_NO_FIT"}
    try:
        assert run.dispatch_once(kind, "software", identity) is True
        with pytest.raises((ValueError, PermissionError), match="unknown_dispatched_fit"):
            run.dispatch_once(kind, "software", identity)
        assert run.dispatch_once(kind, "software", identity, committed=True) is False
    finally:
        run.close()


@pytest.mark.parametrize("kind", ["target_fits", "bank_fits", "meta_calls", "source_calls"])
def test_immutable_dispatch_or_call_record_blocks_rollback_even_without_snapshot(kind, tmp_path, monkeypatch):
    run = _software_arm_run(tmp_path, monkeypatch)
    if kind.endswith("fits"):
        run.dispatch_once(kind, "software", {"fixture_scope": "DISPATCH_ONLY_NO_FIT"})
    else:
        run.observe(kind, "software", 3)
    run.close()
    # Simulate an incomplete restore in this isolated temporary software root.
    # Preserve the snapshot bytes outside the directory read by the loader.
    (run.folder / "snapshots").rename(run.folder / "software-quarantined-snapshots")
    ledger = json.loads(run.path.read_text())
    ledger[kind].clear()
    run.path.write_text(json.dumps(ledger))
    with pytest.raises((ValueError, PermissionError), match="immutable_.*missing_from_live_ledger"):
        _software_arm_run(tmp_path, monkeypatch)


@pytest.mark.parametrize("stage", ["profile", "validation"])
def test_existing_verified_receipt_repairs_missing_catalog_reference_without_new_fit(
        stage, software, tmp_path, monkeypatch):
    """Control-flow recovery only: verifiers explicitly accept a software stub.

    No production verifier is satisfied, no context is loaded from disk, no
    numerical fitter or CUDA function runs, and no scientific receipt is made.
    """
    context = _software_sealed_context(software, tmp_path, monkeypatch)
    catalog = production_catalog_stub()
    runtime = {"fixture_scope": "SYNTHETIC_RECOVERY_ONLY"}
    profile_ref = {"fixture_scope": "NO_ACTUAL_PROFILE"}
    catalog["profiles"] = {"all118": profile_ref}
    monkeypatch.setattr(downstream, "_latest_catalog", lambda *args: copy.deepcopy(catalog))
    monkeypatch.setattr(downstream, "load_context", lambda *args: (context, None, SimpleNamespace(plans=[])))
    monkeypatch.setattr(downstream, "_runtime", lambda: runtime)
    monkeypatch.setattr(downstream, "_gpu", lambda *args: None)
    monkeypatch.setattr(downstream, "_require_owned_stage", lambda *args, **kwargs: None, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Existing receipt recovery attempted a fitter/resource runner")
    monkeypatch.setattr(downstream, "ArmRun", forbidden)
    monkeypatch.setattr(downstream, "_target_once", forbidden)
    calls = []
    def append_reference(root, launch, section, key, item):
        calls.append((section, key, item))
        return launch
    monkeypatch.setattr(downstream, "_update_catalog", append_reference)
    def software_verifier(root, item, launch):
        if "path" in item:
            assert json.loads((root / item["path"]).read_text())["fixture_scope"] == "NOT_A_REAL_COMPLETION"
        else:
            assert item == profile_ref
        return {"runtime": runtime, "geometry_sha256": context.geometry.sha256(),
            "source_feature_dim": context.source.feature_dim,
            "training_UID_set_sha256": context.training_UID_set_sha256}
    monkeypatch.setattr(downstream, "verify_profile", software_verifier)
    monkeypatch.setattr(downstream, "verify_validation", software_verifier)
    job = downstream._validation_job("B0", 0, SEED, "screening")
    relative = f"runs/source_extension_downstream/{ARM}/" + (
        "calibration_profile" if stage == "profile" else "validation/" + job["identity_sha256"])
    folder = tmp_path / relative
    folder.mkdir(parents=True)
    path = folder / "receipt.json"
    path.write_text(json.dumps({"fixture_scope": "NOT_A_REAL_COMPLETION"}))
    result = (downstream.execute_profile(tmp_path, catalog, execute=True, user_started_gpu=True)
        if stage == "profile" else downstream.execute_validation(tmp_path, catalog, "B0", 0, SEED,
            execute=True, user_started_gpu=True))
    assert result["status"] == "ALREADY_COMPLETE_VERIFIED" and result["new_fits"] == 0
    assert calls == [("profiles" if stage == "profile" else "validation",
                      "all118" if stage == "profile" else job["identity_sha256"], result["receipt"])]


def _software_stage_clock(monkeypatch):
    """Inject equal UTC/monotonic advances into the real local journal class."""
    from p3work.recovery_budget import PersistentWallBudget
    clock = SimpleNamespace(seconds=0.)
    def budget(path, identity, maximum, checkpoint_wall_seconds=0):
        return PersistentWallBudget(path, identity, maximum, checkpoint_wall_seconds,
            monotonic=lambda: clock.seconds, utc_now=lambda: 1800000000. + clock.seconds,
            boot_id=lambda: "SYNTHETIC_STAGE_CLOCK")
    monkeypatch.setattr(downstream, "PersistentWallBudget", budget)
    return clock


def test_stage_wall_resumes_cumulative_time_without_reset_and_requires_matching_dispatch(tmp_path, monkeypatch):
    clock = _software_stage_clock(monkeypatch)
    run = SimpleNamespace(state={"stages": {}}, save=lambda: None)
    folder = tmp_path / "software-stage"
    folder.mkdir()
    identity = {"fixture_scope": "NO_FIT_CPU_CLOCK_ONLY"}
    with downstream._stage_wall(run, folder, "software", identity, 10.) as wall:
        clock.seconds += 4.
        assert wall.wall_seconds == 4.
    assert run.state["stages"]["software"]["wall_high_water_seconds"] == 4.
    with downstream._stage_wall(run, folder, "software", identity, 10.) as wall:
        assert wall.wall_seconds == 4. and wall.remaining_seconds == 6.
        clock.seconds += 3.
    item = downstream.descriptor(tmp_path, folder / "stage_wall.json")
    resources = {"ledger": run.state}
    assert downstream._verify_stage_wall(tmp_path, item, identity, 10., resources, "software") == 7.
    rolled_back = copy.deepcopy(resources)
    rolled_back["ledger"]["stages"]["software"]["wall_high_water_seconds"] = 6.
    with pytest.raises((ValueError, PermissionError), match="high_water|dispatch"):
        downstream._verify_stage_wall(tmp_path, item, identity, 10., rolled_back, "software")
    (folder / "stage_wall.json").unlink()
    with pytest.raises((ValueError, PermissionError), match="missing_no_reset"):
        with downstream._stage_wall(run, folder, "software", identity, 10.):
            pytest.fail("A dispatched stage restarted after losing its durable clock")


def test_stage_wall_overrun_cannot_verify_or_restart_as_a_new_budget(tmp_path, monkeypatch):
    clock = _software_stage_clock(monkeypatch)
    run = SimpleNamespace(state={"stages": {}}, save=lambda: None)
    folder = tmp_path / "software-overrun"
    folder.mkdir()
    identity = {"fixture_scope": "NO_FIT_CPU_CLOCK_ONLY"}
    with downstream._stage_wall(run, folder, "software", identity, 5.) as wall:
        clock.seconds += 6.
        assert wall.exhausted
    item = downstream.descriptor(tmp_path, folder / "stage_wall.json")
    with pytest.raises((ValueError, PermissionError), match="overrun|closed_resource"):
        downstream._verify_stage_wall(tmp_path, item, identity, 5., {"ledger": run.state}, "software")
    with pytest.raises((ValueError, PermissionError), match="exhausted"):
        with downstream._stage_wall(run, folder, "software", identity, 5.):
            pytest.fail("An exhausted stage silently received a new budget")


@pytest.mark.parametrize("invalid", [0, -1, True, float("inf"), float("nan")])
def test_deadline_rejects_nonpositive_or_nonfinite_limits_without_entering(invalid):
    with pytest.raises((ValueError, PermissionError), match="positive.*deadline"):
        with downstream._deadline(invalid):
            pytest.fail("An invalid software deadline entered its body")


def test_deadline_interrupts_cpu_work_and_restores_signal_owner():
    original = signal.getsignal(signal.SIGALRM)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    began = time.monotonic()
    try:
        with pytest.raises((ValueError, PermissionError), match="deadline_expired"):
            with downstream._deadline(.025):
                while time.monotonic() - began < 2.:
                    pass
        assert time.monotonic() - began < 1.
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
        assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


def test_deadline_cannot_replace_an_existing_signal_owner():
    original = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    try:
        with pytest.raises((ValueError, PermissionError), match="deadline_owner"):
            with downstream._deadline(1.):
                pytest.fail("A software deadline replaced a preexisting signal owner")
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGALRM, original)


def test_source_bank_campaign_uses_six_source_only_configs_without_target_grid_fields():
    settings = downstream.source_bank_configurations()
    assert settings == [("B4", {"m": m}) for m in (1, 2, 3)] + [
        ("B5", {"shrinkage": shrink}) for shrink in (.01, .05, .1)]
    assert len({downstream._bank_key(method, config) for method, config in settings}) == 6
    for method in ("B4", "B5"):
        for config in registered_grid(method):
            source_config = {"m": config["m"]} if method == "B4" else {"shrinkage": config["shrinkage"]}
            assert (method, source_config) in settings
            assert set(source_config) < set(config)


@pytest.mark.parametrize("entry,args", [("prepare_f0", (SEED,)), ("prepare_reference_stage", (SEED,)),
    ("prepare_source_banks", ()), ("execute_meta", (downstream.MetaJob("P", 8, .35, SEED, "engineering"),)),
    ("execute_profile", ()), ("execute_validation", ("B0", 0, SEED))])
def test_direct_production_execution_cannot_bypass_owned_process_dispatch(entry, args, tmp_path, monkeypatch):
    """Even both user flags cannot grant the current test process fit authority."""
    monkeypatch.setattr(downstream, "_OWNED_STAGE", False)
    def forbidden(*args, **kwargs):
        pytest.fail("An unowned execution reached real context/runtime/cache loading")
    monkeypatch.setattr(downstream, "load_context", forbidden)
    monkeypatch.setattr(downstream, "_source_inputs", forbidden)
    monkeypatch.setattr(downstream, "_runtime", forbidden)
    with pytest.raises((ValueError, PermissionError), match="bounded_stage_runner_required"):
        getattr(downstream, entry)(tmp_path, production_catalog_stub(), *args,
                                  execute=True, user_started_gpu=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage,args", [("f0", (SEED,)), ("reference", (SEED,)), ("source-bank", ()),
    ("meta", (downstream.MetaJob("P", 8, .35, SEED, "engineering"),)),
    ("profile", ()), ("evaluate", ("B0", 0, SEED))])
def test_owned_stage_default_is_a_pure_plan_without_process_or_GPU_dispatch(stage, args, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A default owned-stage plan attempted process/GPU dispatch")
    monkeypatch.setattr(downstream.multiprocessing, "get_context", forbidden)
    monkeypatch.setattr(downstream, "_runtime", forbidden)
    monkeypatch.setattr(downstream, "load_context", forbidden)
    result = downstream.execute_stage(tmp_path, production_catalog_stub(), stage, args=args)
    assert result["status"] == "PLANNED_NOT_EXECUTED"
    assert result["internal_allowed"] is False and result["external_allowed"] is False
    assert not list(tmp_path.iterdir())


def test_owned_stage_requires_later_user_started_GPU_flag_before_any_budget_or_process(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A missing GPU authorization flag reached budget/process loading")
    monkeypatch.setattr(downstream, "_budget", forbidden)
    monkeypatch.setattr(downstream.multiprocessing, "get_context", forbidden)
    with pytest.raises(PermissionError, match="execute_and_later_user_started_GPU"):
        downstream.execute_stage(tmp_path, production_catalog_stub(), "profile", execute=True)
    assert not list(tmp_path.iterdir())
