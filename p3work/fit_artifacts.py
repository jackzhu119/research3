"""Typed registered adaptation checkpoints; tensors and JSON only, never modules.

Loading requires a live verified registry context, exact candidate/prefix, and
all original learned/source-bank capabilities. It never refits or restores RNG.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from . import method_registry as registry
from .adapt import FastWeights, FitResult, module_state_sha256
from .contracts import SOURCE_IDS, digest, file_digest
from .methods import classical, lowrank, neural_controls
from .spatial_geometry import initialize_dictionary, tensor_sha256
from .training_state import atomic_json, local_path

SCHEMA = "paper3-registered-fit-artifact-v1"
CODE_FILES = ("p3work/fit_artifacts.py", "p3work/method_registry.py", "p3work/adapt.py",
    "p3work/spatial_geometry.py", "p3work/methods/neural_controls.py", "p3work/methods/lowrank.py",
    "p3work/methods/classical.py", "p3work/meta_artifacts.py", "p3work/classical_training.py")
_NEURAL_TENSORS = ("alignment", "delta_w", "delta_b", "target_covariance", "target_shrunk_covariance")


def _context(ctx):
    if type(ctx) is not registry.Context:
        raise TypeError("A verified registry Context is required")
    ctx._assert_current()


def _prefix(payload):
    if type(payload) is classical.ClassicalModel:
        return tuple(payload.receipt["support_event_uids"]), tuple(payload.receipt["acquired_event_uids"])
    return payload.support_event_uids, payload.acquired_event_uids


def _prefix_check(valid, acquired):
    if (type(valid) is not tuple or type(acquired) is not tuple or len(acquired) not in {0, 4, 8, 12, 16, 20}
            or any(not isinstance(uid, str) or not uid for uid in valid + acquired)
            or len(set(valid)) != len(valid) or len(set(acquired)) != len(acquired)
            or tuple(uid for uid in acquired if uid in set(valid)) != valid):
        raise ValueError("Exact ordered paid-prefix and valid support subsequence required")


def _candidate(ctx, method, row, role, value):
    if type(value) is not dict:
        raise TypeError("An explicit JSON candidate identity required")
    result = json.loads(json.dumps(value, allow_nan=False))
    expected = {"method": method, "pipeline_seed": ctx.pipeline_seed, "configuration": row,
        "signal_contract_sha256": ctx.signal_contract_sha256, "source_context_identity_sha256": ctx.identity_sha256,
        "final_protocol_freeze_sha256": None}
    if any(key not in result or result[key] != item for key, item in expected.items()):
        raise ValueError("Candidate method/configuration/seed/input/source context differs from the fit")
    allowed = {"meta_validation_support": {"development_evaluation", "calibration_engineering_profile"},
               "calibration_prefix": {"target_inference", "target_evaluation"}}
    if role not in allowed or result.get("purpose") not in allowed[role]:
        raise PermissionError("Candidate purpose cannot change the fit's target/development capability")
    return result


def _descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}


def _read_artifact(root, descriptor):
    if type(descriptor) is not dict or not {"path", "sha256"} <= set(descriptor):
        raise ValueError("An independently SHA-bound fitted artifact descriptor required")
    path = local_path(root, descriptor["path"])
    if (not path.is_file() or file_digest(path) != descriptor["sha256"]
            or "bytes" in descriptor and descriptor["bytes"] != path.stat().st_size):
        raise ValueError("Fitted artifact full SHA/bytes mismatch")
    return path


def _dependencies(ctx, method, row, learned_descriptor, classical_descriptor, dictionary, payload):
    """Reissue source capabilities from their real loaders; no raw-matrix proof."""
    if method in registry.LEARNED_METHODS:
        _read_artifact(ctx.root, learned_descriptor)
        learned = ctx._learned(method, row, learned_descriptor["path"])
        if learned.receipt_sha256 != learned_descriptor["sha256"]:
            raise ValueError("The independent learned-method receipt changed")
        if dictionary is None or not torch.equal(dictionary.cpu(), learned.matrix.cpu()):
            raise ValueError("Persisted learned parameter differs from its verified selected artifact")
        expected_initial = learned.matrix
    else:
        if learned_descriptor is not None:
            raise ValueError("Random/fixed controls cannot claim a learned artifact")
        expected_initial = None
    if method in registry.CLASSICAL_METHODS:
        _read_artifact(ctx.root, classical_descriptor)
        bank = registry.load_verified_source_bank(ctx.root, classical_descriptor, method=method,
            expected_m=row.get("m"), expected_shrink=row.get("shrinkage"),
            input_sha=ctx.signal_contract_sha256, cache_sha=ctx.source_cache_sha256)
        if (bank.method != method or bank.receipt.get("synthetic") is not False
                or bank.receipt.get("bank_training_uid_set_sha256") != ctx.reference.events_sha256
                or bank.receipt.get("n_times") != ctx.contract["n_times"]
                or bank.receipt.get("signal_contract_sha256") != ctx.signal_contract_sha256
                or bank.receipt.get("feature_mode") != ("log_normalized_variance" if method == "B4" else "log_Euclidean_TS_float64")
                or payload.receipt.get("source_state_sha256") != bank.state_sha256):
            raise ValueError("The fitted classical model needs its matching real source-bank capability")
    elif classical_descriptor is not None:
        raise ValueError("Neural/spatial methods cannot use a classical source capability")
    if method in {"B0", "B1", "B9"}:
        fixed = initialize_dictionary(row.get("dimension", 2), ctx.pipeline_seed, device="cpu")
        if dictionary is None or not torch.equal(dictionary.cpu(), fixed):
            raise ValueError("The fixed/random dictionary must be generated by the complete pipeline seed")
    if method in {"B6", "B7", "B10", "B11"}:
        initial = expected_initial if expected_initial is not None else lowrank.initialize_lowrank(row["rank"], ctx.pipeline_seed)
        if payload.initial_u_sha256 != tensor_sha256(initial):
            raise ValueError("The low-rank initial parameter differs from its fixed/learned capability")
    if method in {"B2", "B3", "B4", "B5", "B6", "B7"} and dictionary is not None:
        raise ValueError("This method does not serialize a generator dictionary")


def _validate_fit(ctx, fitted):
    _context(ctx)
    if type(fitted) is not registry.RegisteredFit or getattr(fitted, "_seal", None) is not registry._FIT_SEAL:
        raise PermissionError("Only a context-issued RegisteredFit can be persisted")
    row = registry._configuration(fitted.method, fitted.configuration)
    payload_type = (classical.ClassicalModel if fitted.method in registry.CLASSICAL_METHODS else
        neural_controls.NeuralControlFit if fitted.method in {"B2", "B3"} else
        lowrank.LowRankFitResult if fitted.method in {"B6", "B7", "B10", "B11"} else FitResult)
    if type(fitted.payload) is not payload_type:
        raise ValueError("Registered method and fitted payload type disagree")
    if (fitted.context_identity_sha256 != ctx.identity_sha256
            or registry._weight_fingerprint(fitted.payload) != fitted.fitted_parameter_sha256
            or registry._fit_identity(fitted) != fitted.metadata_sha256
            or fitted.payload.method != fitted.method
            or fitted.scientific_meta_training_claim is not (fitted.method in registry.LEARNED_METHODS)):
        raise ValueError("Registered fitted parameters/metadata/source context changed")
    valid, acquired = _prefix(fitted.payload)
    _prefix_check(valid, acquired)
    if fitted.support_role not in {"calibration_prefix", "meta_validation_support"}:
        raise PermissionError("Only explicit target or development fits can be saved")
    if type(fitted.payload) is classical.ClassicalModel:
        p = fitted.payload.receipt
        if (p.get("support_role") != fitted.support_role or p.get("signal_contract_sha256") != ctx.signal_contract_sha256
                or p.get("n_times") != ctx.contract["n_times"]):
            raise ValueError("Classical fit signal/window/role changed")
    else:
        p = fitted.payload
        if p.protocol_freeze_sha != ctx.signal_contract_sha256 or p.source_state_sha256 != ctx.source_state_sha256:
            raise ValueError("Fitted signal/source differs from the verified context")
        if hasattr(p, "geometry_sha256") and p.geometry_sha256 != ctx.geometry_sha256:
            raise ValueError("Fitted geometry differs from its verified reference")
        if type(p) is neural_controls.NeuralControlFit and (p.source is not ctx.source or p.support_role != fitted.support_role):
            raise ValueError("Neural control source/role differs from its verified context")
        if type(p) is neural_controls.NeuralControlFit and p.target_model is not None:
            interface = lambda module: (repr(module.encoder), repr(module.source_head), module.input_layout, module.feature_dim)
            if interface(p.target_model) != interface(ctx.source):
                raise ValueError("B2 target architecture/non-tensor settings differ from its verified source")
        expected_count = (sum(value.numel() for value in ctx.source.parameters()) if fitted.method == "B2" else
            ctx.source.feature_dim + 1 if fitted.method in {"B1", "B3"} else
            20 * row["rank"] + ctx.source.feature_dim + 1 if type(p) is lowrank.LowRankFitResult else
            0 if fitted.method == "B0" else row["dimension"] + ctx.source.feature_dim + 1)
        if (p.fallback_reason not in {None, "insufficient_class_support"}
                or p.updates != (0 if p.fallback_reason or fitted.method == "B0" else 10)
                or p.trainable_parameters != expected_count):
            raise ValueError("Persisted update/fallback/parameter-count metadata violates the method contract")
        if type(p) is FitResult and p.dictionary_sha256 != tensor_sha256(fitted.dictionary):
            raise ValueError("Generator payload dictionary SHA differs from its actual dictionary")
        if type(p) is lowrank.LowRankFitResult and p.fitted_weights_sha256 != registry._weight_fingerprint(p):
            raise ValueError("Low-rank payload fitted-weight SHA differs from actual weights")
        if type(p) is neural_controls.NeuralControlFit:
            if fitted.method == "B2":
                state = module_state_sha256(p.target_model) if p.target_model is not None else None
            else:
                state = (digest({name: tensor_sha256(getattr(p, name)) for name in ("alignment", "delta_w", "delta_b")})
                         if p.alignment is not None else None)
            if p.fitted_state_sha256 != state:
                raise ValueError("Neural payload fitted-state SHA differs from actual parameters")
            if fitted.method == "B3":
                authorization = neural_controls.SourceReferenceAuthorization(ctx.reference, ctx.signal_contract_sha256, tuple(sorted(SOURCE_IDS)))
                if p.source_reference_authorization_sha256 != authorization.validate(ctx.geometry, ctx.signal_contract_sha256):
                    raise ValueError("B3 reference authorization differs from its actual source context")
    _dependencies(ctx, fitted.method, row, fitted.learned_receipt, fitted.classical_receipt, fitted.dictionary, fitted.payload)
    return row, valid, acquired


def _serialize(fitted):
    payload, tensors = fitted.payload, {}
    if fitted.dictionary is not None:
        tensors["dictionary"] = fitted.dictionary
    if type(payload) is FitResult:
        kind, skip = "generator", {"weights"}
        tensors.update({"weights." + name: getattr(payload.weights, name) for name in ("theta", "delta_w", "delta_b")})
    elif type(payload) is lowrank.LowRankFitResult:
        kind, skip = "lowrank", {"weights", "config"}
        tensors.update({"weights." + name: getattr(payload.weights, name) for name in ("u", "v", "delta_w", "delta_b")})
    elif type(payload) is neural_controls.NeuralControlFit:
        kind, skip = "neural", {"source", "target_model", "config", *_NEURAL_TENSORS}
        for name in _NEURAL_TENSORS:
            if getattr(payload, name) is not None:
                tensors[name] = getattr(payload, name)
        if payload.target_model is not None:
            payload.target_model.assert_buffers_unchanged()
            if any(m.training for m in payload.target_model.modules()) or any(p.requires_grad for p in payload.target_model.parameters()):
                raise ValueError("B2 target must remain completely eval/frozen before saving")
            tensors.update({"target.parameter." + name: value for name, value in payload.target_model.named_parameters()})
            tensors.update({"target.buffer." + name: value for name, value in payload.target_model.named_buffers()})
    elif type(payload) is classical.ClassicalModel:
        kind = "classical"
        tensors.update({"classical." + name: torch.from_numpy(value.copy(order="K")) for name, value in payload.arrays.items()})
        return kind, {"receipt": copy.deepcopy(payload.receipt), "state_sha256": payload.state_sha256}, tensors
    else:
        raise TypeError("Unsupported registered payload")
    metadata = {field.name: getattr(payload, field.name) for field in fields(payload) if field.name not in skip}
    if hasattr(payload, "config"):
        metadata["config"] = asdict(payload.config)
    return kind, metadata, tensors


def _copy_preserving_layout(value, *, device):
    """Copy values without silently selecting a different reduction layout.

    The registered implementations produce dense tensors, including QR's
    column-major outputs. Unsupported non-dense views fail before a checkpoint
    is issued instead of being silently compacted by clone/device transfer.
    """
    copied = value.detach().to(device=device).clone(memory_format=torch.preserve_format)
    if registry._tensor_layout(copied) != registry._tensor_layout(value):
        raise ValueError("Fitted numerical layout cannot be preserved by this checkpoint copy")
    return copied


def save_registered_fit(ctx, fitted, directory, *, candidate_identity, software_test_double=False):
    """Persist actual issued parameters; neither support signals nor labels saved."""
    if type(software_test_double) is not bool:
        raise TypeError("Explicit boolean software scope required")
    row, valid, acquired = _validate_fit(ctx, fitted)
    candidate = _candidate(ctx, fitted.method, row, fitted.support_role, candidate_identity)
    output = local_path(ctx.root, directory)
    if Path(directory).parts[0] != "runs" or output.exists():
        raise ValueError("Use a new immutable fit directory under runs/")
    kind, metadata, tensors = _serialize(fitted)
    # Preserve dense strides too: QR bases can be column-major, and forcing
    # contiguous storage changes a BLAS reduction path despite identical values.
    tensors = {name: _copy_preserving_layout(value, device="cpu") for name, value in tensors.items()}
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Nonfinite fitted parameters cannot be persisted")
    identity = {"method": fitted.method, "configuration": row, "support_role": fitted.support_role,
        "pipeline_seed": ctx.pipeline_seed, "context_identity_sha256": ctx.identity_sha256,
        "signal_contract_sha256": ctx.signal_contract_sha256, "split_sha256": ctx.split_sha256,
        "final_protocol_freeze_sha256": None,
        "source_state_sha256": ctx.source_state_sha256, "source_interface_sha256": ctx.source_interface_sha256,
        "geometry_sha256": ctx.geometry_sha256, "source_cache_sha256": ctx.source_cache_sha256,
        "source_receipt": {"path": ctx.source_receipt_path, "sha256": ctx.source_receipt_sha256},
        "reference_receipt": {"path": ctx.reference_receipt_path, "sha256": ctx.reference_receipt_sha256},
        "support_event_uids": list(valid), "acquired_event_uids": list(acquired),
        "candidate_identity": candidate, "candidate_identity_sha256": digest(candidate),
        "registered_metadata_sha256": fitted.metadata_sha256, "fitted_parameter_sha256": fitted.fitted_parameter_sha256,
        "registered_layout_sha256": registry._fit_layout_sha256(fitted),
        "payload_kind": kind, "payload_metadata_sha256": digest(metadata),
        "learned_receipt": fitted.learned_receipt, "classical_receipt": fitted.classical_receipt,
        "scientific_meta_training_claim": fitted.scientific_meta_training_claim,
        "code": {name: file_digest(ctx.root / name) for name in CODE_FILES}}
    identity_sha = digest(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".partial-", dir=output.parent))
    try:
        tensor_path = temporary / "parameters.pt"
        with tensor_path.open("wb") as stream:
            torch.save({"schema_version": SCHEMA, "identity_sha256": identity_sha, "tensors": tensors}, stream)
            stream.flush(); os.fsync(stream.fileno())
        tensor_descriptor = {"path": (output / tensor_path.name).relative_to(ctx.root).as_posix(),
                             "sha256": file_digest(tensor_path), "bytes": tensor_path.stat().st_size}
        sidecar_path = temporary / "parameters.pt.sha256.json"
        atomic_json(sidecar_path, {**tensor_descriptor, "identity_sha256": identity_sha})
        receipt = {"schema_version": SCHEMA, "status": "REGISTERED_FITTED_PARAMETERS_SAVED", "synthetic": software_test_double,
            "scope": "SOFTWARE_TEST_DOUBLE" if software_test_double else "REGISTERED_ADAPTATION_PARAMETERS",
            "identity": identity, "identity_sha256": identity_sha, "kind": kind, "payload": metadata,
            "parameters": tensor_descriptor,
            "parameters_sidecar": {"path": (output / sidecar_path.name).relative_to(ctx.root).as_posix(),
                                   "sha256": file_digest(sidecar_path), "bytes": sidecar_path.stat().st_size},
            "tensor_manifest": {name: {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype), "sha256": tensor_sha256(value)}
                                for name, value in tensors.items()},
            "source_or_query_signals_stored": False, "labels_stored": False, "rng_stored_or_restored": False}
        atomic_json(temporary / "receipt.json", receipt)
        if output.exists():
            raise ValueError("Existing fitted artifact cannot be replaced")
        os.rename(temporary, output)
        return _descriptor(ctx.root, output / "receipt.json")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _take(tensors, name, shape, dtype, *, device):
    value = tensors.pop(name)
    if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
        raise ValueError("Fitted tensor shape/dtype/finite contract mismatch: " + name)
    return _copy_preserving_layout(value, device=device)


def _restore(ctx, method, row, kind, metadata, tensors):
    tensors = dict(tensors)
    dictionary = tensors.pop("dictionary", None)
    if dictionary is not None:
        dictionary = _copy_preserving_layout(dictionary, device=ctx.geometry.q.device)
        if method in {"P", "P+PCT", "B8", "B9", "B0", "B1"}:
            ctx.geometry.validate_dictionary(dictionary)
    p = copy.deepcopy(metadata)
    for key in ("support_event_uids", "acquired_event_uids"):
        if key in p:
            p[key] = tuple(p[key])
    template = next(ctx.source.parameters())
    if kind == "generator" and method in {"P", "P+PCT", "B8", "B9", "B0", "B1"}:
        r = row.get("dimension", 2)
        weights = FastWeights(_take(tensors, "weights.theta", (r,), torch.float64, device=ctx.geometry.q.device),
            _take(tensors, "weights.delta_w", (ctx.source.feature_dim,), template.dtype, device=template.device),
            _take(tensors, "weights.delta_b", (), template.dtype, device=template.device))
        if ((method in {"B0", "B1"} and torch.any(weights.theta != 0))
                or torch.linalg.vector_norm(weights.theta) > row.get("radius", .15) + 1e-10
                or method == "B0" and (torch.any(weights.delta_w != 0) or bool(weights.delta_b != 0))):
            raise ValueError("Persisted generator/head parameters violate the fixed method/radius")
        payload = FitResult(weights=weights, **p)
    elif kind == "lowrank" and method in {"B6", "B7", "B10", "B11"}:
        config = lowrank.LowRankConfig(**p.pop("config"))
        expected = lowrank.LowRankConfig(**row, pct_weight=.1 if method in {"B7", "B11"} else 0)
        if config != expected:
            raise ValueError("Persisted low-rank optimizer configuration changed")
        weights = lowrank.LowRankWeights(*(_take(tensors, "weights." + name, (10, row["rank"]), torch.float64,
            device=ctx.geometry.q.device) for name in ("u", "v")),
            _take(tensors, "weights.delta_w", (ctx.source.feature_dim,), template.dtype, device=template.device),
            _take(tensors, "weights.delta_b", (), template.dtype, device=template.device))
        if torch.linalg.vector_norm(weights.u @ weights.v.T) > row["radius"] + 1e-10:
            raise ValueError("Persisted low-rank update exceeds its registered radius")
        payload = lowrank.LowRankFitResult(weights=weights, config=config, **p)
    elif kind == "neural" and method in {"B2", "B3"}:
        config_type = neural_controls.FineTuneConfig if method == "B2" else neural_controls.AlignmentConfig
        config = config_type(**p.pop("config"))
        if config != config_type(**row):
            raise ValueError("Persisted neural optimizer configuration changed")
        target = None
        if method == "B2" and p["fallback_reason"] is None:
            target = neural_controls._EvalSourceCopy(ctx.source)
            with torch.no_grad():
                for prefix, entries in (("target.parameter.", target.named_parameters()), ("target.buffer.", target.named_buffers())):
                    for name, value in entries:
                        stored = _take(tensors, prefix + name, value.shape, value.dtype, device=value.device)
                        if registry._tensor_layout(stored) != registry._tensor_layout(value):
                            raise ValueError("B2 stored parameter/buffer layout differs from its verified source template")
                        if prefix == "target.buffer." and not torch.equal(stored, value):
                            raise ValueError("B2 checkpoint changed a frozen source BN/buffer")
                        value.copy_(stored)
            for value in target.parameters():
                value.requires_grad_(False)
            target.eval(); target.assert_buffers_unchanged()
        extra = {name: None for name in _NEURAL_TENSORS}
        if method == "B3":
            authorization = neural_controls.SourceReferenceAuthorization(ctx.reference, ctx.signal_contract_sha256, tuple(sorted(SOURCE_IDS)))
            if p["source_reference_authorization_sha256"] != authorization.validate(ctx.geometry, ctx.signal_contract_sha256):
                raise ValueError("B3 reference capability changed")
            if p["fallback_reason"] is None:
                for name, shape in (("alignment", (11, 11)), ("target_covariance", (10, 10)), ("target_shrunk_covariance", (10, 10))):
                    extra[name] = _take(tensors, name, shape, torch.float64, device=ctx.geometry.q.device)
                extra["delta_w"] = _take(tensors, "delta_w", (ctx.source.feature_dim,), template.dtype, device=template.device)
                extra["delta_b"] = _take(tensors, "delta_b", (), template.dtype, device=template.device)
        payload = neural_controls.NeuralControlFit(source=ctx.source, target_model=target, config=config, **extra, **p)
    elif kind == "classical" and method in {"B4", "B5"}:
        arrays = {name.removeprefix("classical."): value.cpu().numpy().copy(order="K") for name, value in tensors.items()
                  if name.startswith("classical.")}
        if len(arrays) != len(tensors) or not arrays:
            raise ValueError("A complete explicit classical numerical array state required")
        tensors.clear()
        payload = classical._model(method, arrays, p["receipt"])
        if payload.state_sha256 != p["state_sha256"]:
            raise ValueError("Classical fitted parameters/receipt changed")
    else:
        raise ValueError("Persisted payload type does not match the registered method")
    if tensors:
        raise ValueError("Unrecognized tensors in fitted artifact")
    return payload, dictionary


def load_registered_fit(ctx, descriptor, *, expected_method, expected_configuration, expected_support_role,
                        expected_acquired_event_uids, expected_support_event_uids, candidate_identity,
                        software_test_double=False):
    """Reconstruct a callable registered fit from safe typed state, never a refit."""
    _context(ctx)
    row = registry._configuration(expected_method, expected_configuration)
    candidate = _candidate(ctx, expected_method, row, expected_support_role, candidate_identity)
    valid, acquired = tuple(expected_support_event_uids), tuple(expected_acquired_event_uids)
    _prefix_check(valid, acquired)
    path = _read_artifact(ctx.root, descriptor)
    receipt = json.loads(path.read_text())
    if (type(software_test_double) is not bool or receipt.get("schema_version") != SCHEMA
            or receipt.get("status") != "REGISTERED_FITTED_PARAMETERS_SAVED" or receipt.get("synthetic") is not software_test_double
            or receipt.get("scope") != ("SOFTWARE_TEST_DOUBLE" if software_test_double else "REGISTERED_ADAPTATION_PARAMETERS")
            or receipt.get("source_or_query_signals_stored") is not False or receipt.get("labels_stored") is not False
            or receipt.get("rng_stored_or_restored") is not False):
        raise ValueError("A matching real/software scoped fitted-parameter receipt required")
    identity = receipt["identity"]
    expected = {"method": expected_method, "configuration": row, "support_role": expected_support_role,
        "pipeline_seed": ctx.pipeline_seed, "context_identity_sha256": ctx.identity_sha256,
        "signal_contract_sha256": ctx.signal_contract_sha256, "split_sha256": ctx.split_sha256,
        "final_protocol_freeze_sha256": None,
        "source_state_sha256": ctx.source_state_sha256, "source_interface_sha256": ctx.source_interface_sha256,
        "geometry_sha256": ctx.geometry_sha256, "source_cache_sha256": ctx.source_cache_sha256,
        "source_receipt": {"path": ctx.source_receipt_path, "sha256": ctx.source_receipt_sha256},
        "reference_receipt": {"path": ctx.reference_receipt_path, "sha256": ctx.reference_receipt_sha256},
        "support_event_uids": list(valid), "acquired_event_uids": list(acquired),
        "candidate_identity": candidate, "candidate_identity_sha256": digest(candidate)}
    if receipt.get("identity_sha256") != digest(identity) or any(key not in identity or identity[key] != value for key, value in expected.items()):
        raise ValueError("Fitted artifact method/configuration/role/candidate/prefix/source identity mismatch")
    if identity.get("payload_kind") != receipt.get("kind") or identity.get("payload_metadata_sha256") != digest(receipt.get("payload")):
        raise ValueError("Fitted payload metadata differs from the checkpoint identity")
    if set(identity.get("code", {})) != set(CODE_FILES):
        raise ValueError("Complete fitted-parameter dependency code manifest required")
    for name, sha in identity["code"].items():
        if file_digest(local_path(ctx.root, name)) != sha:
            raise ValueError("Fitted-method implementation changed after adaptation")
    tensor_path = _read_artifact(ctx.root, receipt["parameters"])
    if tensor_path.parent != path.parent or tensor_path.name != "parameters.pt":
        raise ValueError("Parameter tensors must stay in the immutable receipt directory")
    sidecar_path = _read_artifact(ctx.root, receipt["parameters_sidecar"])
    if sidecar_path != tensor_path.with_suffix(".pt.sha256.json"):
        raise ValueError("Parameter sidecar must stay beside its parameter file")
    sidecar = json.loads(sidecar_path.read_text())
    if (sidecar.get("sha256") != receipt["parameters"]["sha256"] or sidecar.get("bytes") != tensor_path.stat().st_size
            or sidecar.get("identity_sha256") != digest(identity)):
        raise ValueError("Fitted parameter tensor full SHA/bytes/identity sidecar mismatch")
    record = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if record.get("schema_version") != SCHEMA or record.get("identity_sha256") != digest(identity):
        raise ValueError("Fitted tensor record identity mismatch")
    tensors, manifest = record.get("tensors"), receipt.get("tensor_manifest")
    if type(tensors) is not dict or type(manifest) is not dict or set(tensors) != set(manifest):
        raise ValueError("Complete explicit fitted tensor manifest required")
    for name, tensor in tensors.items():
        if (type(tensor) is not torch.Tensor or not torch.isfinite(tensor).all()
                or manifest[name] != {"shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype), "sha256": tensor_sha256(tensor)}):
            raise ValueError("Fitted tensor numeric SHA/shape/dtype mismatch")
    payload, dictionary = _restore(ctx, expected_method, row, receipt["kind"], receipt["payload"], tensors)
    if _prefix(payload) != (valid, acquired):
        raise ValueError("Fitted payload and expected complete prefix/subsequence differ")
    fitted = object.__new__(registry.RegisteredFit)
    for name, value in {"method": expected_method, "configuration": row, "support_role": expected_support_role,
        "payload": payload, "dictionary": dictionary, "context_identity_sha256": ctx.identity_sha256,
        "fitted_parameter_sha256": identity["fitted_parameter_sha256"], "learned_receipt": identity["learned_receipt"],
        "classical_receipt": identity["classical_receipt"], "scientific_meta_training_claim": identity["scientific_meta_training_claim"],
        "metadata_sha256": identity["registered_metadata_sha256"], "_seal": registry._FIT_SEAL}.items():
        object.__setattr__(fitted, name, value)
    _validate_fit(ctx, fitted)
    if identity.get("registered_layout_sha256") != registry._fit_layout_sha256(fitted):
        raise ValueError("Restored numerical layouts differ from the issued fitted artifact")
    return fitted


def artifact_files(root, descriptor):
    """List the three SHA-checked files; this alone is never a fit capability."""
    root = Path(root).resolve(strict=True)
    receipt_path = _read_artifact(root, descriptor)
    receipt = json.loads(receipt_path.read_text())
    result = [_descriptor(root, receipt_path)]
    for key in ("parameters", "parameters_sidecar"):
        path = _read_artifact(root, receipt[key])
        if path.parent != receipt_path.parent:
            raise ValueError("Fitted artifact manifest member escaped its directory")
        result.append(_descriptor(root, path))
    return tuple(result)
