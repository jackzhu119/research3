"""Explicit synthetic CPU fit persistence tests, never real EEG/learned proof.

Real source/reference/learned/source-bank loaders are explicit test doubles from
the registry fixtures. Every emitted fit receipt is marked synthetic and needs
the software_test_double opt-in; tests perform no GPU/network/data access.
"""
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import nn

from p3work import fit_artifacts as artifacts
from p3work import method_registry as registry
from p3work.adapt import PredictionInputs, module_state_sha256
from p3work.contracts import file_digest
from p3work.spatial_geometry import tensor_sha256
from p3work.training_state import capture_rng
from test_method_registry import fixture, bank_fixture, learned_fixture, _different_dense_layout


@pytest.fixture
def setup(fixture):
    root = fixture["root"]
    repository = Path(__file__).resolve().parents[1]
    for name in artifacts.CODE_FILES:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, path)
    source = fixture["source"]
    source.encoder = nn.Sequential(nn.Flatten(), nn.Linear(11 * 128, 3).double(), nn.BatchNorm1d(3).double(),
                                   nn.Tanh(), nn.Dropout(.4))
    source.encoder.register_buffer("SOFTWARE_NONPERSISTENT_BUFFER", torch.tensor([1., 2.]), persistent=False)
    for parameter in source.parameters(): parameter.requires_grad_(False)
    source.eval(); source.frozen_state_sha256 = module_state_sha256(source)
    return fixture


def candidate(ctx, method, row, role):
    return {"schema_version": "paper3-development-candidate-v1" if role == "meta_validation_support" else "paper3-target-candidate-v1",
        "purpose": "development_evaluation" if role == "meta_validation_support" else "target_inference",
        "method": method, "pipeline_seed": ctx.pipeline_seed, "configuration": row,
        "signal_contract_sha256": ctx.signal_contract_sha256, "final_protocol_freeze_sha256": None,
        "source_context_identity_sha256": ctx.identity_sha256, "job_sha256": "a" * 64,
        "catalog_identity_sha256": "b" * 64, "episode_plan_sha256": "c" * 64,
        "fixture_scope": "SOFTWARE_TEST_DOUBLE_NOT_SCIENTIFIC_EVIDENCE"}


def fitted_fixture(setup, monkeypatch, method="B9", *, role="meta_validation_support", both_classes=True):
    if method in registry.LEARNED_METHODS:
        ctx, row, learned, _ = learned_fixture(setup, monkeypatch, method)
        kwargs = {"learned_receipt": learned.receipt_path}
    elif method in registry.CLASSICAL_METHODS:
        ctx, row, descriptor, _ = bank_fixture(setup, monkeypatch, method)
        kwargs = {"classic_model": descriptor}
    else:
        ctx, row = setup["build"](), registry.registered_grid(method)[0]
        kwargs = {}
    support = replace(setup["support"](role, both_classes=both_classes), acquired_event_uids=tuple(f"s{i}" for i in range(20)))
    fitted = (ctx.fit_development if role == "meta_validation_support" else ctx.fit)(method, support, row, **kwargs)
    cid = candidate(ctx, method, row, role)
    descriptor = artifacts.save_registered_fit(ctx, fitted, "runs/fit-SOFTWARE_TEST_DOUBLE", candidate_identity=cid,
                                             software_test_double=True)
    return {"ctx": ctx, "row": row, "support": support, "fitted": fitted, "candidate": cid,
            "descriptor": descriptor, "inputs": setup["inputs"], "root": setup["root"]}


def load(f, **updates):
    options = {"expected_method": f["fitted"].method, "expected_configuration": f["row"],
        "expected_support_role": f["support"].role, "expected_acquired_event_uids": f["support"].acquired_event_uids,
        "expected_support_event_uids": f["support"].event_uids, "candidate_identity": f["candidate"], "software_test_double": True}
    options.update(updates)
    return artifacts.load_registered_fit(f["ctx"], f["descriptor"], **options)


def assert_rng_same(before):
    after = capture_rng(cuda=False)
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert torch.equal(before["torch_cpu"], after["torch_cpu"]) and after["cuda"] is None


@pytest.mark.parametrize("method", registry.METHOD_IDS)
@pytest.mark.parametrize("role", ["meta_validation_support", "calibration_prefix"])
def test_all_fourteen_methods_roundtrip_actual_parameters_with_identical_logits_no_refit_or_rng_restore(setup, monkeypatch, method, role):
    f = fitted_fixture(setup, monkeypatch, method, role=role)
    expected = f["ctx"].predict(f["fitted"], f["inputs"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference load attempted a fit or GPU action")
    monkeypatch.setattr(registry.Context, "fit", forbidden)
    monkeypatch.setattr(registry.Context, "fit_development", forbidden)
    monkeypatch.setattr(registry.adapt, "fit_target", forbidden)
    monkeypatch.setattr(registry.lowrank, "fit_lowrank_target", forbidden)
    monkeypatch.setattr(registry.neural_controls, "fit_full_finetune", forbidden)
    monkeypatch.setattr(registry.neural_controls, "fit_neural_control_development", forbidden)
    monkeypatch.setattr(registry.classical, "fit_b4_target", forbidden)
    monkeypatch.setattr(registry.classical, "fit_b5_target", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    before = capture_rng(cuda=False)
    restored = load(f)
    assert_rng_same(before)
    actual = f["ctx"].predict(restored, f["inputs"])
    assert torch.equal(expected.logits, actual.logits)
    assert torch.equal(expected.probabilities, actual.probabilities)
    assert torch.equal(expected.labels, actual.labels)
    assert restored.metadata_sha256 == f["fitted"].metadata_sha256
    original_tensors = artifacts._serialize(f["fitted"])[2]
    restored_tensors = artifacts._serialize(restored)[2]
    assert original_tensors.keys() == restored_tensors.keys()
    for name, original in original_tensors.items():
        rebuilt = restored_tensors[name]
        assert (rebuilt.shape, rebuilt.dtype, rebuilt.stride()) == (original.shape, original.dtype, original.stride())
        assert torch.equal(rebuilt, original)
    assert restored.support_role == role and artifacts._prefix(restored.payload) == (f["support"].event_uids, f["support"].acquired_event_uids)
    if method == "B2":
        assert not any(m.training for m in restored.payload.target_model.modules())
        assert not any(p.requires_grad for p in restored.payload.target_model.parameters())
        assert "encoder.SOFTWARE_NONPERSISTENT_BUFFER" in dict(restored.payload.target_model.named_buffers())
        assert module_state_sha256(restored.payload.target_model) == module_state_sha256(f["fitted"].payload.target_model)
    if method in {"B4", "B5"}:
        assert np.any(restored.payload.arrays["coef"] != 0)  # Restore actual target coefficients, not zero source prior.
    f["ctx"].source.assert_unchanged()


@pytest.mark.parametrize("method", ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "P", "B8", "B10", "B11", "P+PCT"])
def test_missing_class_fallback_roundtrips_without_inventing_target_parameters(setup, monkeypatch, method):
    f = fitted_fixture(setup, monkeypatch, method, both_classes=False)
    restored = load(f)
    expected, actual = f["ctx"].predict(f["fitted"], f["inputs"]), f["ctx"].predict(restored, f["inputs"])
    assert torch.equal(expected.logits, actual.logits)
    if method not in {"B4", "B5"}: assert restored.payload.fallback_reason == "insufficient_class_support"


def test_file_manifest_contains_only_explicit_tensor_json_and_sidecar_no_module_signal_labels_rng(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch, "B2")
    descriptors = artifacts.artifact_files(f["root"], f["descriptor"])
    assert {Path(row["path"]).name for row in descriptors} == {"receipt.json", "parameters.pt", "parameters.pt.sha256.json"}
    receipt = json.loads((f["root"] / f["descriptor"]["path"]).read_text())
    assert receipt["synthetic"] is True and receipt["identity"]["final_protocol_freeze_sha256"] is None
    assert receipt["source_or_query_signals_stored"] is receipt["labels_stored"] is receipt["rng_stored_or_restored"] is False
    state = torch.load(f["root"] / receipt["parameters"]["path"], map_location="cpu", weights_only=True)
    assert set(state) == {"schema_version", "identity_sha256", "tensors"}
    assert all(type(t) is torch.Tensor for t in state["tensors"].values())
    assert not any("support.x" in name or "support.y" in name for name in state["tensors"])
    with pytest.raises(ValueError, match="real/software"):
        load(f, software_test_double=False)


@pytest.mark.parametrize("violation", ["method", "configuration", "role", "paid_prefix", "valid_prefix", "candidate", "signal_name", "false_W7"])
def test_expected_candidate_and_full_paid_prefix_cannot_be_relabelled(setup, monkeypatch, violation):
    f = fitted_fixture(setup, monkeypatch)
    kwargs = {}
    if violation == "method": kwargs["expected_method"] = "P"
    elif violation == "configuration": kwargs["expected_configuration"] = {"dimension": 2, "radius": .15}
    elif violation == "role": kwargs["expected_support_role"] = "calibration_prefix"
    elif violation == "paid_prefix": kwargs["expected_acquired_event_uids"] = tuple(f"other{i}" for i in range(20))
    elif violation == "valid_prefix": kwargs["expected_support_event_uids"] = f["support"].event_uids[:2]
    else:
        cid = dict(f["candidate"])
        if violation == "candidate": cid["episode_plan_sha256"] = "f" * 64
        elif violation == "signal_name": cid["input_contract_sha256"] = cid.pop("signal_contract_sha256")
        else: cid["final_protocol_freeze_sha256"] = f["ctx"].signal_contract_sha256
        kwargs["candidate_identity"] = cid
    with pytest.raises((ValueError, PermissionError)):
        load(f, **kwargs)


@pytest.mark.parametrize("violation", ["payload_paid", "payload_valid", "payload_updates", "tensor_bytes", "sidecar_bytes", "source", "reference", "code", "extra_tensor"])
def test_payload_uid_update_state_code_and_byte_corruptions_fail_closed(setup, monkeypatch, violation):
    f = fitted_fixture(setup, monkeypatch)
    path = f["root"] / f["descriptor"]["path"]
    receipt = json.loads(path.read_text())
    if violation.startswith("payload_"):
        if violation == "payload_paid": receipt["payload"]["acquired_event_uids"][-1] = "other-paid-event"
        elif violation == "payload_valid": receipt["payload"]["support_event_uids"] = receipt["payload"]["support_event_uids"][:2]
        else: receipt["payload"]["updates"] = 0
        path.write_text(json.dumps(receipt))
        f["descriptor"] = {"path": f["descriptor"]["path"], "sha256": file_digest(path)}
    elif violation in {"tensor_bytes", "sidecar_bytes"}:
        target = receipt["parameters" if violation == "tensor_bytes" else "parameters_sidecar"]["path"]
        with (f["root"] / target).open("ab") as stream: stream.write(b"SOFTWARE_TEST_DOUBLE_CHANGED")
    elif violation == "source": (f["root"] / f["ctx"].source_receipt_path).write_text("CHANGED SOFTWARE SOURCE")
    elif violation == "reference": (f["root"] / f["ctx"].reference_receipt_path).write_text("CHANGED SOFTWARE REF")
    elif violation == "code": (f["root"] / "p3work/adapt.py").write_text("# CHANGED SOFTWARE CODE")
    else:
        state = torch.load(f["root"] / receipt["parameters"]["path"], weights_only=True)
        state["tensors"]["unrecognized_query_data"] = torch.zeros(1)
        torch.save(state, f["root"] / receipt["parameters"]["path"])
    with pytest.raises((ValueError, RuntimeError)):
        load(f)


def test_paid_qc_failed_event_still_cannot_be_scored_as_query_after_restore(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch)
    restored = load(f)
    inputs = PredictionInputs(f["inputs"].x[:1], ("s19",), f["ctx"].signal_contract_sha256)
    with pytest.raises(ValueError, match="overlap"):
        f["ctx"].predict(restored, inputs)


def test_direct_context_raw_fit_unbound_descriptor_overwrite_and_symlinks_rejected(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch)
    with pytest.raises(PermissionError):
        artifacts.load_registered_fit(registry.Context(), f["descriptor"], expected_method="B9", expected_configuration=f["row"],
            expected_support_role=f["support"].role, expected_acquired_event_uids=f["support"].acquired_event_uids,
            expected_support_event_uids=f["support"].event_uids, candidate_identity=f["candidate"])
    with pytest.raises(PermissionError):
        artifacts.save_registered_fit(f["ctx"], registry.RegisteredFit(), "runs/raw", candidate_identity=f["candidate"])
    with pytest.raises(ValueError):
        artifacts.save_registered_fit(f["ctx"], f["fitted"], "runs/fit-SOFTWARE_TEST_DOUBLE", candidate_identity=f["candidate"])
    original = f["descriptor"]
    f["descriptor"] = "unbound.json"
    with pytest.raises(ValueError): load(f)
    f["descriptor"] = original
    target = f["root"] / f["descriptor"]["path"]
    alias = f["root"] / "runs/link.json"; alias.symlink_to(target)
    original = f["descriptor"]
    f["descriptor"] = {"path": "runs/link.json", "sha256": original["sha256"]}
    with pytest.raises(ValueError, match="symlink"): load(f)


def test_b2_non_tensor_architecture_drift_rejected_before_checkpoint_write(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch, "B2")
    f["fitted"].payload.target_model.encoder[2].eps = .1
    with pytest.raises(ValueError, match="parameters/metadata/source context changed"):
        artifacts.save_registered_fit(f["ctx"], f["fitted"], "runs/changed-BN-SOFTWARE_TEST_DOUBLE",
            candidate_identity=f["candidate"], software_test_double=True)


def test_b2_value_preserving_stride_drift_rejected_before_predict_or_save(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch, "B2")
    target = f["fitted"].payload.target_model
    value = next(p for p in target.parameters() if p.ndim == 2)
    numerical_state = module_state_sha256(target)
    value.data = _different_dense_layout(value)
    assert module_state_sha256(target) == numerical_state
    with pytest.raises(ValueError, match="context or parameters changed"):
        f["ctx"].predict(f["fitted"], f["inputs"])
    with pytest.raises(ValueError, match="parameters/metadata/source context changed"):
        artifacts.save_registered_fit(f["ctx"], f["fitted"], "runs/changed-stride-SOFTWARE_TEST_DOUBLE",
            candidate_identity=f["candidate"], software_test_double=True)
    assert not (f["root"] / "runs/changed-stride-SOFTWARE_TEST_DOUBLE").exists()


def test_b2_column_major_source_template_roundtrips_exact_strides_and_logits(setup, monkeypatch):
    # A legitimate pre-issuance source layout remains supported, not coerced to C.
    for value in setup["source"].parameters():
        if value.ndim == 2:
            value.data = _different_dense_layout(value)
    f = fitted_fixture(setup, monkeypatch, "B2")
    expected = f["ctx"].predict(f["fitted"], f["inputs"])
    source_layouts = {name: value.stride() for name, value in f["ctx"].source.named_parameters()}
    restored = load(f)
    assert {name: value.stride() for name, value in restored.payload.target_model.named_parameters()} == source_layouts
    assert torch.equal(expected.logits, f["ctx"].predict(restored, f["inputs"]).logits)


def test_b2_renewed_file_checksums_cannot_change_the_issued_parameter_layout(setup, monkeypatch):
    f = fitted_fixture(setup, monkeypatch, "B2")
    receipt_path = f["root"] / f["descriptor"]["path"]
    receipt = json.loads(receipt_path.read_text())
    tensor_path = f["root"] / receipt["parameters"]["path"]
    record = torch.load(tensor_path, map_location="cpu", weights_only=True)
    name = "target.parameter.encoder.1.weight"
    record["tensors"][name] = _different_dense_layout(record["tensors"][name])
    value = record["tensors"][name]
    receipt["tensor_manifest"][name] = {"shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
    torch.save(record, tensor_path)
    receipt["parameters"].update(sha256=file_digest(tensor_path), bytes=tensor_path.stat().st_size)
    sidecar_path = f["root"] / receipt["parameters_sidecar"]["path"]
    sidecar_path.write_text(json.dumps({**receipt["parameters"], "identity_sha256": receipt["identity_sha256"]}))
    receipt["parameters_sidecar"].update(sha256=file_digest(sidecar_path), bytes=sidecar_path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    f["descriptor"] = {"path": f["descriptor"]["path"], "sha256": file_digest(receipt_path)}
    with pytest.raises(ValueError, match="layout differs from its verified source template"):
        load(f)


@pytest.mark.parametrize("method,field", [("B9", "dictionary_sha256"), ("B6", "fitted_weights_sha256"),
    ("B2", "fitted_state_sha256"), ("B3", "fitted_state_sha256")])
def test_payload_internal_parameter_hashes_are_verified_before_returning_a_callable_fit(setup, monkeypatch, method, field):
    f = fitted_fixture(setup, monkeypatch, method)
    # Deliberate in-memory mutation is a negative fixture, never a real fit.
    object.__setattr__(f["fitted"].payload, field, "f" * 64)
    with pytest.raises(ValueError):
        artifacts.save_registered_fit(f["ctx"], f["fitted"], "runs/corrupt-metadata-SOFTWARE_TEST_DOUBLE",
            candidate_identity=f["candidate"], software_test_double=True)


@pytest.mark.parametrize("method,tensor_name", [("B9", "weights.theta"), ("B0", "weights.theta"),
    ("B1", "weights.theta"), ("B0", "weights.delta_w"), ("B6", "weights.v"),
    ("B2", "target.buffer.encoder.2.running_mean")])
def test_method_constraints_reject_even_renewed_tensor_checksums(setup, monkeypatch, method, tensor_name):
    f = fitted_fixture(setup, monkeypatch, method)
    receipt_path = f["root"] / f["descriptor"]["path"]
    receipt = json.loads(receipt_path.read_text())
    tensor_path = f["root"] / receipt["parameters"]["path"]
    record = torch.load(tensor_path, map_location="cpu", weights_only=True)
    record["tensors"][tensor_name].fill_(10)
    value = record["tensors"][tensor_name]
    receipt["tensor_manifest"][tensor_name] = {"shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
    torch.save(record, tensor_path)
    receipt["parameters"].update(sha256=file_digest(tensor_path), bytes=tensor_path.stat().st_size)
    sidecar_path = f["root"] / receipt["parameters_sidecar"]["path"]
    sidecar = {**receipt["parameters"], "identity_sha256": receipt["identity_sha256"]}
    sidecar_path.write_text(json.dumps(sidecar))
    receipt["parameters_sidecar"].update(sha256=file_digest(sidecar_path), bytes=sidecar_path.stat().st_size)
    receipt_path.write_text(json.dumps(receipt))
    f["descriptor"] = {"path": f["descriptor"]["path"], "sha256": file_digest(receipt_path)}
    with pytest.raises(ValueError, match="method/radius|registered radius|frozen source BN"):
        load(f)
