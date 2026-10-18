"""Formal-dispatch SOFTWARE_TEST_DOUBLE fixtures, never real EEG/fit evidence.

The actual source/reference/learned artifact loaders have their own integrity
tests. They are explicitly replaced here to test registry calls and binding;
all tensor computations below use small generated CPU signals only.
"""
from dataclasses import replace
import inspect
import json

import numpy as np
import pytest
import torch
from torch import nn

from p3work import method_registry as registry
from p3work.adapt import CalibrationPrefix, FrozenSourceModel, PredictionInputs
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest, validate_signal_contract
from p3work.meta_artifacts import LearnedMetaParameter
from p3work.meta_training import MetaJob
from p3work.methods import lowrank
from p3work.spatial_geometry import SourceCovarianceTrial, initialize_dictionary, source_reference, tensor_sha256

torch.set_num_threads(1)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    contract = {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
        "selected_profile": "short", "context_s": [0, 2], "crop_s": [.5, 1.5], "n_times": 128,
        "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
            "raw_metadata_manifest_sha256": "b" * 64,
            "long": {"participant_mean_coverage": .5, "participant_mean_abs_class_exclusion_gap": 0.},
            "short": {"participant_mean_coverage": 1., "participant_mean_abs_class_exclusion_gap": 0.}}}
    signal_sha = validate_signal_contract(contract)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(62)
        source = FrozenSourceModel(nn.Sequential(nn.Flatten(), nn.Linear(11 * 128, 3), nn.Tanh()).double(),
                                   nn.Linear(3, 2).double(), feature_dim=3)
        x = torch.randn(8, 11, 128, dtype=torch.float64)
        x -= x.mean(1, keepdim=True)
        x -= x.mean(2, keepdim=True)
        trials = [SourceCovarianceTrial(dataset, "SOFTWARE_TEST_DOUBLE", dataset + "/fixture", x[0])
                  for dataset in sorted(SOURCE_IDS)]
        reference = source_reference(trials)
    (tmp_path / "source.json").write_text(json.dumps({"fixture": "SOFTWARE_TEST_DOUBLE", "not_scientific_evidence": True}))
    (tmp_path / "reference.json").write_text(json.dumps({"fixture": "SOFTWARE_TEST_DOUBLE", "not_scientific_evidence": True}))
    source_info = {"pipeline_seed": 20261018, "seed": 20261018, "signal_contract_sha256": signal_sha,
                   "source_cache_manifest_sha256": "d" * 64}
    reference_info = {"signal_contract_sha256": signal_sha, "source_f0_receipt_sha256": file_digest(tmp_path / "source.json"),
        "training_dataset_ids": sorted(SOURCE_IDS), "training_uid_set_sha256": reference.events_sha256,
        "source_cache_manifest": {"sha256": "d" * 64}}
    calls = []

    def fake_source(root, receipt, supplied):
        calls.append(("SOFTWARE_TEST_DOUBLE_SOURCE_LOADER", root, receipt, supplied))
        torch.rand(3)  # Verify the registry preserves caller RNG across bridge allocations.
        return source, source_info

    def fake_reference(root, receipt, f0_receipt, supplied):
        calls.append(("SOFTWARE_TEST_DOUBLE_REFERENCE_LOADER", root, receipt, f0_receipt, supplied))
        torch.rand(3)
        return reference, reference_info

    monkeypatch.setattr(registry, "load_verified_source", fake_source)
    monkeypatch.setattr(registry, "load_verified_reference", fake_reference)

    def build():
        return registry.Context.from_verified(tmp_path, "source.json", "reference.json", contract)

    def support(role="calibration_prefix", *, both_classes=True):
        return CalibrationPrefix(x[:4].clone(), torch.tensor([0, 1, 0, 1] if both_classes else [0] * 4),
            tuple(f"s{i}" for i in range(4)), tuple(f"s{i}" for i in range(4)), signal_sha, role=role)

    return {"root": tmp_path, "contract": contract, "signal_sha": signal_sha, "source": source, "reference": reference,
            "source_info": source_info, "reference_info": reference_info, "calls": calls, "build": build,
            "support": support, "inputs": PredictionInputs(x[4:].clone(), tuple(f"q{i}" for i in range(4)), signal_sha)}


def test_all_grids_are_preregistered_at_most_nine_and_return_copies():
    assert registry.METHOD_IDS == tuple(f"B{i}" for i in range(12)) + ("P", "P+PCT")
    assert len(registry.registered_grid("B0")) == 1
    for method in registry.METHOD_IDS:
        rows = registry.registered_grid(method)
        assert len(rows) <= 9
        assert len({digest(row) for row in rows}) == len(rows)
        if method != "B0":
            assert len(rows) == 9
    row = registry.registered_grid("B9")[0]
    row["dimension"] = 99
    assert registry.registered_grid("B9")[0]["dimension"] == 2
    status = registry.registry_status()
    assert status["B4"]["requires_verified_new_classical_source_bank"]
    assert status["B5"]["requires_verified_new_classical_source_bank"]
    assert all(status[m]["requires_independent_learned_artifact"] for m in registry.LEARNED_METHODS)
    with pytest.raises(ValueError):
        registry.registered_grid("legacy_baseline_v1")


def test_context_invokes_both_real_loader_interfaces_and_preserves_rng(fixture):
    before = torch.get_rng_state().clone()
    ctx = fixture["build"]()
    assert torch.equal(before, torch.get_rng_state())
    assert [v[0] for v in fixture["calls"]] == ["SOFTWARE_TEST_DOUBLE_SOURCE_LOADER", "SOFTWARE_TEST_DOUBLE_REFERENCE_LOADER"]
    assert fixture["calls"][1][3] == "source.json"
    assert ctx.pipeline_seed == 20261018 and ctx.signal_contract_sha256 == fixture["signal_sha"]
    assert ctx.source_state_sha256 == fixture["source"].frozen_state_sha256
    assert ctx.reference_array_sha256 == tensor_sha256(fixture["reference"].matrix)
    assert ctx.split_sha256 == fixture["contract"]["window_audit_receipt"]["split_sha256"]


@pytest.mark.parametrize("kind", ["seed", "source_signal", "ref_signal", "f0_receipt", "missing_source", "uid_set", "cache"])
def test_context_rejects_cross_artifact_identity(fixture, kind):
    if kind == "seed":
        fixture["source_info"]["seed"] = 20261019
    elif kind == "source_signal":
        fixture["source_info"]["signal_contract_sha256"] = "c" * 64
    elif kind == "ref_signal":
        fixture["reference_info"]["signal_contract_sha256"] = "c" * 64
    elif kind == "f0_receipt":
        fixture["reference_info"]["source_f0_receipt_sha256"] = "c" * 64
    elif kind == "missing_source":
        fixture["reference_info"]["training_dataset_ids"].pop()
    elif kind == "uid_set":
        fixture["reference_info"]["training_uid_set_sha256"] = "c" * 64
    else:
        fixture["reference_info"]["source_cache_manifest"]["sha256"] = "c" * 64
    with pytest.raises(ValueError):
        fixture["build"]()


def test_loader_failure_cannot_be_replaced_by_a_direct_context(fixture, monkeypatch):
    def blocked(*args):
        raise PermissionError("SOFTWARE_TEST_DOUBLE: real source gate blocked")
    monkeypatch.setattr(registry, "load_verified_source", blocked)
    with pytest.raises(PermissionError, match="source gate"):
        fixture["build"]()
    with pytest.raises(PermissionError, match="from_verified"):
        registry.Context().fit("B0", fixture["support"](), {})


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3", "B6", "B7", "B9"])
def test_synthetic_nonlearned_dispatch_is_support_only_and_predicts_same_logit_path(fixture, method):
    ctx = fixture["build"]()
    fitted = ctx.fit(method, fixture["support"](), registry.registered_grid(method)[0])
    assert fitted.method == fitted.payload.method == method
    assert fitted.support_role == "calibration_prefix"
    assert not fitted.scientific_meta_training_claim and fitted.learned_receipt is None
    assert fitted.payload.updates == (0 if method == "B0" else 10)
    prediction = ctx.predict(fitted, fixture["inputs"])
    assert torch.equal(prediction.probabilities, prediction.logits.softmax(1))
    assert torch.equal(prediction.labels, prediction.logits.argmax(1))
    ctx.source.assert_unchanged()


@pytest.mark.parametrize("method", ["B9", "B6", "B7"])
def test_random_controls_are_determined_only_by_pipeline_seed_and_do_not_consume_rng(fixture, method):
    ctx = fixture["build"]()
    row = registry.registered_grid(method)[0]
    before = torch.get_rng_state().clone()
    a = ctx.fit(method, fixture["support"](both_classes=False), row)
    b = ctx.fit(method, fixture["support"](both_classes=False), row)
    assert torch.equal(before, torch.get_rng_state())
    if method == "B9":
        assert torch.equal(a.dictionary, initialize_dictionary(row["dimension"], ctx.pipeline_seed))
        assert torch.equal(a.dictionary, b.dictionary)
    else:
        expected = lowrank.initialize_lowrank(row["rank"], ctx.pipeline_seed)
        assert a.payload.initial_u_sha256 == tensor_sha256(expected)
        assert torch.equal(a.payload.weights.u, expected)
        assert torch.equal(a.payload.weights.u, b.payload.weights.u)


@pytest.mark.parametrize("configuration", [{"dimension": 2, "radius": .05, "seed": 42},
    {"dimension": 2, "radius": .05, "B": 0}, {"dimension": 2., "radius": .05},
    {"dimension": True, "radius": .05}, {"dimension": 2, "radius": .2}])
def test_caller_dictionary_seed_and_off_grid_values_rejected(fixture, configuration):
    with pytest.raises(ValueError):
        fixture["build"]().fit("B9", fixture["support"](), configuration)


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3", "B6", "B7", "B9"])
def test_development_dispatch_keeps_capability_and_never_casts_it_to_target(fixture, method):
    ctx = fixture["build"]()
    support = fixture["support"]("meta_validation_support")
    row = registry.registered_grid(method)[0]
    with pytest.raises(PermissionError):
        ctx.fit(method, support, row)
    fitted = ctx.fit_development(method, support, row)
    assert fitted.support_role == support.role == "meta_validation_support"
    if method in {"B2", "B3"}:
        assert fitted.payload.support_role == "meta_validation_support"
    ctx.predict(fitted, fixture["inputs"])
    with pytest.raises(PermissionError):
        ctx.fit_development(method, fixture["support"](), row)
    with pytest.raises(PermissionError):
        ctx.fit_development(method, fixture["support"]("meta_train_support"), row)


@pytest.mark.parametrize("method", sorted(registry.LEARNED_METHODS))
def test_learned_methods_never_fall_back_to_random_initialization(fixture, method):
    ctx = fixture["build"]()
    for entry, support in ((ctx.fit, fixture["support"]()),
                           (ctx.fit_development, fixture["support"]("meta_validation_support"))):
        with pytest.raises(ValueError, match="own verified independent"):
            entry(method, support, registry.registered_grid(method)[0])
        with pytest.raises(ValueError, match="own verified independent"):
            entry(method, support, registry.registered_grid(method)[0], learned_receipt=torch.eye(10))


def learned_fixture(fixture, monkeypatch, method, *, mismatch=False):
    """Explicit SOFTWARE_TEST_DOUBLE loader; this does not issue a scientific artifact."""
    row = registry.registered_grid(method)[0]
    rank = row.get("dimension", row.get("rank"))
    ctx = fixture["build"]()
    path = fixture["root"] / (method + "-SOFTWARE_TEST_DOUBLE.json")
    path.write_text('{"fixture":"SOFTWARE_TEST_DOUBLE","scientific_fit_completed":false}')
    matrix = lowrank.initialize_lowrank(rank, ctx.pipeline_seed) if method in {"B10", "B11"} else initialize_dictionary(rank, ctx.pipeline_seed)
    value = LearnedMetaParameter(matrix, MetaJob(method, rank * 2 if mismatch else rank, row["radius"]),
        ctx.signal_contract_sha256, ctx.split_sha256, ctx.source_state_sha256, ctx.geometry_sha256,
        path.name, file_digest(path), 50)
    calls = []
    def verified_double(root, receipt, **expected):
        calls.append((receipt, expected))
        assert expected == {"expected_method": method, "expected_seed": ctx.pipeline_seed,
            "input_contract_sha256": ctx.signal_contract_sha256, "split_sha256": ctx.split_sha256,
            "source": ctx.source, "geometry": ctx.geometry}
        return value
    monkeypatch.setattr(registry.meta_artifacts, "load_learned_parameter", verified_double)
    return ctx, row, value, calls


@pytest.mark.parametrize("method", sorted(registry.LEARNED_METHODS))
def test_official_artifact_gate_and_target_wrapper_are_both_used(fixture, monkeypatch, method):
    ctx, row, value, calls = learned_fixture(fixture, monkeypatch, method)
    seen = []
    def wrapper(root, learned, source, geometry, support):
        seen.append((root, learned, source, geometry, support))
        # Synthetic missing-class kernel gives no training updates.
        if method in {"B10", "B11"}:
            return lowrank.fit_lowrank_target(source, geometry, support, method=method, u0=learned.matrix,
                config=lowrank.LowRankConfig(rank=learned.job.dimension, radius=learned.job.radius,
                                             pct_weight=.1 if method == "B11" else 0))
        return registry.adapt.fit_target(source, geometry, learned.matrix, support, method=method,
            config=registry.adapt.InnerConfig(radius=learned.job.radius, pct_weight=.1 if method == "P+PCT" else 0))
    monkeypatch.setattr(registry.meta_artifacts, "fit_verified_meta_method", wrapper)
    support = fixture["support"](both_classes=False)
    fitted = ctx.fit(method, support, row, learned_receipt=value.receipt_path)
    assert len(calls) == len(seen) == 1 and seen[0][-1] is support
    assert fitted.learned_receipt == {"path": value.receipt_path, "sha256": value.receipt_sha256}
    assert fitted.payload.updates == 0
    ctx.predict(fitted, fixture["inputs"])


@pytest.mark.parametrize("method", sorted(registry.LEARNED_METHODS))
def test_learned_development_uses_verified_parameter_and_preserves_role(fixture, monkeypatch, method):
    ctx, row, value, calls = learned_fixture(fixture, monkeypatch, method)
    def forbidden_target(*args):
        raise AssertionError("Development must never cast its role for the target wrapper")
    monkeypatch.setattr(registry.meta_artifacts, "fit_verified_meta_method", forbidden_target)
    support = fixture["support"]("meta_validation_support", both_classes=False)
    fitted = ctx.fit_development(method, support, row, learned_receipt=value.receipt_path)
    assert len(calls) == 1 and fitted.support_role == support.role
    if method in {"B10", "B11"}:
        assert fitted.payload.initial_u_sha256 == tensor_sha256(value.matrix)
    else:
        assert torch.equal(fitted.dictionary, value.matrix)
    ctx.predict(fitted, fixture["inputs"])


@pytest.mark.parametrize("method", sorted(registry.LEARNED_METHODS))
def test_verified_artifact_still_must_match_requested_configuration(fixture, monkeypatch, method):
    ctx, row, value, _ = learned_fixture(fixture, monkeypatch, method, mismatch=True)
    with pytest.raises(ValueError, match="dimension/radius"):
        ctx.fit(method, fixture["support"](), row, learned_receipt=value.receipt_path)


@pytest.mark.parametrize("method", ["B4", "B5"])
def test_classical_dispatch_requires_sha_bound_new_bank_and_never_accepts_legacy(fixture, method):
    ctx = fixture["build"]()
    for invalid in (None, "unbound.json", {"method": "legacy_source_csp_lda_v1"}, {"path": "unbound.json"}):
        with pytest.raises(ValueError, match="SHA-bound.*legacy"):
            ctx.fit(method, fixture["support"](), registry.registered_grid(method)[0], classic_model=invalid)


def bank_fixture(fixture, monkeypatch, method, *, violation=None):
    """Fake source-bank loader and numerical prior; no real source fit is claimed."""
    ctx = fixture["build"]()
    row = registry.registered_grid(method)[0]
    path = fixture["root"] / (method + "-bank-SOFTWARE_TEST_DOUBLE.json")
    path.write_text('{"fixture":"SOFTWARE_TEST_DOUBLE","scientific_fit_completed":false}')
    descriptor = {"path": path.name, "sha256": file_digest(path)}
    q = ctx.geometry.q.numpy()
    receipt = {"kind": "source_model", "status": "SOURCE_CLASSICAL_COMPLETE", "synthetic": False,
        "source_datasets": sorted(SOURCE_IDS), "held_out_dataset": None,
        "signal_contract_sha256": ctx.signal_contract_sha256, "n_times": ctx.contract["n_times"],
        "bank_training_uid_set_sha256": ctx.reference.events_sha256,
        "feature_mode": "log_normalized_variance" if method == "B4" else "log_Euclidean_TS_float64"}
    if method == "B4":
        receipt["m"] = row["m"]
        arrays = {"q": q, "class_means": np.stack([np.eye(10) / 10] * 2),
            "filters": np.eye(10)[[0, -1]], "eigenvalues": np.ones(10), "coef": np.zeros(2), "bias": np.array(0.)}
    else:
        receipt["shrink"] = row["shrinkage"]
        arrays = {"q": q, "reference": np.eye(10) / 10, "inverse_root": np.eye(10) * np.sqrt(10),
            "scaler_mean": np.zeros(55), "scaler_scale": np.ones(55), "coef": np.zeros(55), "bias": np.array(0.)}
    if violation == "uid": receipt["bank_training_uid_set_sha256"] = "c" * 64
    elif violation == "mode": receipt["feature_mode"] = "legacy_log_power"
    elif violation == "signal": receipt["signal_contract_sha256"] = "c" * 64
    elif violation == "window": receipt["n_times"] = 256
    elif violation == "configuration": receipt["m" if method == "B4" else "shrink"] = 2 if method == "B4" else .1
    elif violation == "synthetic": receipt["synthetic"] = True
    bank = registry.classical._model(method, arrays, receipt)
    calls = []
    def verified_double(root, supplied, **expected):
        calls.append((root, supplied, expected))
        assert root == ctx.root and supplied == descriptor
        assert expected == {"method": method, "expected_m": row.get("m"), "expected_shrink": row.get("shrinkage"),
            "input_sha": ctx.signal_contract_sha256, "cache_sha": ctx.source_cache_sha256}
        return bank
    monkeypatch.setattr(registry, "load_verified_source_bank", verified_double)
    return ctx, row, descriptor, calls


@pytest.mark.parametrize("method", ["B4", "B5"])
@pytest.mark.parametrize("development", [False, True])
def test_classical_new_bank_dispatch_binds_uid_input_grid_and_keeps_development_role(fixture, monkeypatch, method, development):
    ctx, row, descriptor, calls = bank_fixture(fixture, monkeypatch, method)
    support = fixture["support"]("meta_validation_support" if development else "calibration_prefix", both_classes=False)
    fitted = (ctx.fit_development if development else ctx.fit)(method, support, row, classic_model=descriptor)
    assert len(calls) == 1 and fitted.payload.receipt["support_role"] == support.role
    assert fitted.payload.receipt["fallback_reason"] == "insufficient_class_support"
    assert fitted.classical_receipt == descriptor and fitted.learned_receipt is None
    assert not fitted.scientific_meta_training_claim
    prediction = ctx.predict(fitted, fixture["inputs"])
    assert torch.equal(prediction.labels, prediction.logits.argmax(1))
    torch.testing.assert_close(prediction.probabilities, prediction.logits.softmax(1))


@pytest.mark.parametrize("method", ["B4", "B5"])
@pytest.mark.parametrize("violation", ["uid", "mode", "signal", "window", "configuration", "synthetic"])
def test_new_classical_bank_must_match_context_not_just_method_name(fixture, monkeypatch, method, violation):
    ctx, row, descriptor, _ = bank_fixture(fixture, monkeypatch, method, violation=violation)
    with pytest.raises(ValueError, match="differs from the source"):
        ctx.fit(method, fixture["support"](), row, classic_model=descriptor)


def test_classical_bank_evidence_cannot_change_after_fitting(fixture, monkeypatch):
    ctx, row, descriptor, _ = bank_fixture(fixture, monkeypatch, "B4")
    fitted = ctx.fit("B4", fixture["support"](both_classes=False), row, classic_model=descriptor)
    (fixture["root"] / descriptor["path"]).write_text("changed SOFTWARE_TEST_DOUBLE")
    with pytest.raises(ValueError, match="source-bank evidence"):
        ctx.predict(fitted, fixture["inputs"])


def test_query_labels_have_no_formal_input_and_wrong_capabilities_and_windows_reject(fixture):
    ctx = fixture["build"]()
    assert "query_y" not in inspect.signature(ctx.fit).parameters
    assert "y" not in inspect.signature(PredictionInputs).parameters
    with pytest.raises(TypeError):
        ctx.fit("B0", fixture["inputs"], {})
    with pytest.raises(ValueError, match="frozen window"):
        ctx.fit("B0", replace(fixture["support"](), x=fixture["support"]().x[:, :, :64]), {})
    with pytest.raises(ValueError, match="protocols differ"):
        ctx.fit("B0", replace(fixture["support"](), protocol_freeze_sha="c" * 64), {})
    fitted = ctx.fit("B0", fixture["support"](), {})
    with pytest.raises(ValueError, match="verified signal"):
        ctx.predict(fitted, replace(fixture["inputs"], protocol_freeze_sha="c" * 64))
    with pytest.raises(TypeError):
        ctx.predict(fitted, fixture["support"]())


@pytest.mark.parametrize("violation", ["metadata", "weights", "dictionary", "receipt", "source", "source_interface"])
def test_context_and_fitted_snapshots_reject_post_issue_changes(fixture, violation):
    ctx = fixture["build"]()
    fitted = ctx.fit("B9", fixture["support"](both_classes=False), registry.registered_grid("B9")[0])
    if violation == "metadata":
        fitted.configuration["radius"] = .15
    elif violation == "weights":
        fitted.payload.weights.delta_w.add_(1)
    elif violation == "dictionary":
        fitted.dictionary[0, 0] += .1
    elif violation == "receipt":
        (fixture["root"] / "source.json").write_text("changed")
    elif violation == "source_interface":
        ctx.source.input_layout = "N1CT"
    else:
        with torch.no_grad():
            next(ctx.source.parameters()).add_(1)
    with pytest.raises((ValueError, RuntimeError)):
        ctx.predict(fitted, fixture["inputs"])


def test_direct_fit_record_construction_cannot_claim_verified_learning(fixture):
    ctx = fixture["build"]()
    with pytest.raises(PermissionError, match="issue a registered"):
        ctx.predict(registry.RegisteredFit(), fixture["inputs"])
