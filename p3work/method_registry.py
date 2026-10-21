"""Formal method dispatch, separate from unverified mathematical kernels.

Creating a context requires real source/reference artifact loaders. This module
does not fetch datasets, open evaluation labels, start training jobs, or grant
external-data access. B4/B5 require independently SHA-bound, newly fitted
source-bank artifacts; historical core models cannot serve as their priors.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
from itertools import product
from pathlib import Path

import torch

from . import adapt, meta_artifacts
from .adapt import CalibrationPrefix, FastWeights, FitResult, PredictionInputs, PredictionResult, module_state_sha256
from .classical_training import load_verified_source_bank
from .contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from .f0_bridge import load_verified_source
from .methods import classical, lowrank, neural_controls
from .source_reference_io import load_verified_reference
from .spatial_geometry import SpatialGeometry, initialize_dictionary, tensor_sha256
from .training_state import local_path

METHOD_IDS = tuple(f"B{i}" for i in range(12)) + ("P", "P+PCT")
LEARNED_METHODS = frozenset({"P", "P+PCT", "B8", "B10", "B11"})
CLASSICAL_METHODS = frozenset({"B4", "B5"})
_CONTEXT_SEAL = object()
_FIT_SEAL = object()


def registered_grid(method):
    """Return fresh dictionaries for the exact predeclared <=9-setting grids."""
    if method == "B0":
        return ({},)
    if method == "B1":
        return tuple({"head_lr": lr, "head_l2": l2} for lr, l2 in product((.003, .01, .03), (0., .001, .01)))
    if method == "B2":
        return tuple({"learning_rate": lr, "l2_to_source": l2} for lr, l2 in product((.0001, .0003, .001), (0., .0001, .001)))
    if method == "B3":
        return tuple({"shrinkage": shrink, "head_learning_rate": lr} for shrink, lr in product((.01, .05, .1), (.003, .01, .03)))
    if method == "B4":
        return tuple({"alpha": alpha, "m": m} for alpha, m in product((0., .5, 1.), (1, 2, 3)))
    if method == "B5":
        return tuple({"shrinkage": shrink, "l2_to_source": l2} for shrink, l2 in product((.01, .05, .1), (.01, .1, 1.)))
    if method in {"B6", "B7", "B10", "B11"}:
        return tuple({"rank": k, "radius": rho} for k, rho in product((1, 2, 4), (.05, .15, .35)))
    if method in {"P", "P+PCT", "B8", "B9"}:
        # Generator dimension is not the rank of exp(M)-I.
        return tuple({"dimension": r, "radius": rho} for r, rho in product((2, 4, 8), (.05, .15, .35)))
    raise ValueError("Unknown registered Paper3 method")


def registry_status():
    return {method: {"registered_configurations": len(registered_grid(method)),
                     "formal_dispatch": "IMPLEMENTED_NOT_A_COMPLETED_FIT",
                     "requires_independent_learned_artifact": method in LEARNED_METHODS,
                     "requires_verified_new_classical_source_bank": method in CLASSICAL_METHODS}
            for method in METHOD_IDS}


def _configuration(method, configuration):
    if type(configuration) is not dict or any(type(v) not in {int, float} for v in configuration.values()):
        raise ValueError("A registered configuration dictionary with numeric values required")
    for key in ("dimension", "rank", "m"):
        if key in configuration and type(configuration[key]) is not int:
            raise ValueError("Generator dimension/CSP count/LoRA rank must be integers")
    for row in registered_grid(method):
        if configuration == row:
            return row.copy()
    raise ValueError("Configuration is outside this method's predeclared grid; no caller-supplied B/U0/seed")


def _weight_fingerprint(value):
    if type(value) is classical.ClassicalModel:
        value.assert_unchanged()
        return value.state_sha256
    if type(value) is FitResult:
        return digest({"theta": tensor_sha256(value.weights.theta), "delta_w": tensor_sha256(value.weights.delta_w),
                       "delta_b": tensor_sha256(value.weights.delta_b), "method": value.method})
    if type(value) is lowrank.LowRankFitResult:
        return hashlib.sha256("/".join(tensor_sha256(v) for v in
            (value.weights.u, value.weights.v, value.weights.delta_w, value.weights.delta_b)).encode()).hexdigest()
    if type(value) is neural_controls.NeuralControlFit:
        data = {"method": value.method, "fitted_state_sha256": value.fitted_state_sha256}
        if value.target_model is not None:
            data["actual_target_state"] = module_state_sha256(value.target_model)
            data["actual_target_interface"] = _source_interface(value.target_model)
            data["target_parameter_requires_grad"] = {name: p.requires_grad for name, p in value.target_model.named_parameters()}
            data["target_module_training"] = {name: m.training for name, m in value.target_model.named_modules()}
        for name in ("alignment", "delta_w", "delta_b", "target_covariance", "target_shrunk_covariance"):
            tensor = getattr(value, name)
            if tensor is not None:
                data[name] = tensor_sha256(tensor)
        return digest(data)
    raise TypeError("Unknown fit payload; raw tensors are not registered fitted methods")


@dataclass(frozen=True, init=False)
class RegisteredFit:
    method: str
    configuration: dict
    support_role: str
    payload: object
    dictionary: torch.Tensor | None
    context_identity_sha256: str
    fitted_parameter_sha256: str
    learned_receipt: dict | None
    classical_receipt: dict | None
    scientific_meta_training_claim: bool
    metadata_sha256: str
    _seal: object


def _fit_identity(value):
    return digest({"method": value.method, "configuration": value.configuration, "support_role": value.support_role,
        "context_identity_sha256": value.context_identity_sha256,
        "fitted_parameter_sha256": value.fitted_parameter_sha256,
        "dictionary_sha256": tensor_sha256(value.dictionary) if value.dictionary is not None else None,
        "parameter_layout_sha256": _fit_layout_sha256(value),
        "learned_receipt": value.learned_receipt, "classical_receipt": value.classical_receipt,
        "payload_metadata_sha256": _payload_metadata_sha256(value.payload),
        "scientific_meta_training_claim": value.scientific_meta_training_claim})


def _tensor_layout(value):
    """Portable numerical layout; device moves do not change this identity."""
    if type(value) not in {torch.Tensor, torch.nn.Parameter} or value.layout != torch.strided:
        raise TypeError("A strided numerical tensor is required")
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "layout": str(value.layout)}


def _module_layout(module):
    return {kind + "." + name: _tensor_layout(value)
            for kind, entries in (("parameter", module.named_parameters()), ("buffer", module.named_buffers()))
            for name, value in entries}


def _fit_layout_sha256(fitted):
    payload = fitted.payload
    if type(payload) is classical.ClassicalModel:
        layouts = {name: {"shape": list(value.shape), "stride_bytes": list(value.strides), "dtype": str(value.dtype)}
                   for name, value in payload.arrays.items()}
    elif type(payload) in {FitResult, lowrank.LowRankFitResult}:
        layouts = {field.name: _tensor_layout(getattr(payload.weights, field.name)) for field in fields(payload.weights)}
    elif type(payload) is neural_controls.NeuralControlFit:
        layouts = {name: _tensor_layout(getattr(payload, name)) for name in
                   ("alignment", "delta_w", "delta_b", "target_covariance", "target_shrunk_covariance")
                   if getattr(payload, name) is not None}
        if payload.target_model is not None:
            layouts["target_model"] = _module_layout(payload.target_model)
    else:
        raise TypeError("Unknown registered fit payload")
    return digest({"payload": layouts,
                   "dictionary": _tensor_layout(fitted.dictionary) if fitted.dictionary is not None else None})


def _context_layout_sha256(reference, geometry):
    return digest({"reference": _tensor_layout(reference.matrix),
        "geometry": {name: _tensor_layout(getattr(geometry, name))
                     for name in ("q", "rref_sqrt", "rref_invsqrt", "e")}})


def _payload_metadata_sha256(payload):
    """Seal operational metadata as well as numerical adaptation parameters."""
    if type(payload) is classical.ClassicalModel:
        return digest({"method": payload.method, "receipt": payload.receipt, "state_sha256": payload.state_sha256})
    if type(payload) not in {FitResult, lowrank.LowRankFitResult, neural_controls.NeuralControlFit}:
        raise TypeError("Unknown registered fit payload")
    skip = {"weights", "source", "target_model", "alignment", "delta_w", "delta_b", "target_covariance", "target_shrunk_covariance"}
    data = {f.name: (asdict(payload.config) if f.name == "config" else getattr(payload, f.name))
            for f in fields(payload) if f.name not in skip}
    if type(payload) is neural_controls.NeuralControlFit:
        data["actual_source_state_sha256"] = payload.source.frozen_state_sha256
        data["actual_source_interface_sha256"] = _source_interface(payload.source)
    return digest(data)


@dataclass(frozen=True, init=False)
class Context:
    """Receipt-issued source context; direct construction cannot issue a seal."""
    root: Path
    source: object
    reference: object
    geometry: SpatialGeometry
    contract: dict
    pipeline_seed: int
    signal_contract_sha256: str
    split_sha256: str
    source_receipt_path: str
    source_receipt_sha256: str
    reference_receipt_path: str
    reference_receipt_sha256: str
    source_state_sha256: str
    source_interface_sha256: str
    source_cache_sha256: str
    geometry_sha256: str
    reference_array_sha256: str
    numerical_layout_sha256: str
    identity_sha256: str
    _seal: object

    @classmethod
    def from_verified(cls, root, source_receipt, reference_receipt, contract):
        root = Path(root).resolve(strict=True)
        signal_sha = validate_signal_contract(contract)
        source_path, ref_path = local_path(root, source_receipt), local_path(root, reference_receipt)
        # Neither loader runs a fit; preserve CPU RNG used by bridge allocation.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            source, source_info = load_verified_source(root, source_receipt, contract)
            reference, ref_info = load_verified_reference(root, reference_receipt, source_receipt, contract)
        seed = source_info.get("pipeline_seed")
        if type(seed) is not int or seed not in {20261018, 20261019, 20261020, 20261021, 20261022} or source_info.get("seed") != seed:
            raise ValueError("The context needs one complete predeclared pipeline seed")
        if (source_info.get("signal_contract_sha256") != signal_sha or ref_info.get("signal_contract_sha256") != signal_sha
                or ref_info.get("source_f0_receipt_sha256") != file_digest(source_path)
                or ref_info.get("source_cache_manifest", {}).get("sha256") != source_info.get("source_cache_manifest_sha256")
                or not source_info.get("source_cache_manifest_sha256")
                or set(ref_info.get("training_dataset_ids", [])) != SOURCE_IDS
                or ref_info.get("training_uid_set_sha256") != reference.events_sha256):
            raise ValueError("Source/reference/pipeline/input artifacts do not agree")
        geometry = SpatialGeometry.from_reference(reference.matrix)
        source.assert_unchanged()
        values = {"root": root, "source": source, "reference": reference, "geometry": geometry,
                  "contract": copy.deepcopy(contract), "pipeline_seed": seed, "signal_contract_sha256": signal_sha,
                  "split_sha256": contract["window_audit_receipt"]["split_sha256"],
                  "source_receipt_path": str(source_receipt), "source_receipt_sha256": file_digest(source_path),
                  "reference_receipt_path": str(reference_receipt), "reference_receipt_sha256": file_digest(ref_path),
                  "source_state_sha256": source.frozen_state_sha256, "source_interface_sha256": _source_interface(source),
                  "source_cache_sha256": source_info["source_cache_manifest_sha256"],
                  "geometry_sha256": geometry.sha256(),
                  "reference_array_sha256": tensor_sha256(reference.matrix),
                  "numerical_layout_sha256": _context_layout_sha256(reference, geometry), "_seal": _CONTEXT_SEAL}
        values["identity_sha256"] = digest({k: values[k] for k in ("pipeline_seed", "signal_contract_sha256", "split_sha256",
            "source_receipt_sha256", "reference_receipt_sha256", "source_state_sha256", "source_interface_sha256", "source_cache_sha256",
            "geometry_sha256", "reference_array_sha256", "numerical_layout_sha256")})
        context = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(context, name, value)
        return context

    def _assert_current(self):
        if getattr(self, "_seal", None) is not _CONTEXT_SEAL:
            raise PermissionError("Only Context.from_verified can issue a production method context")
        self.source.assert_unchanged()
        if (validate_signal_contract(self.contract) != self.signal_contract_sha256
                or self.source.frozen_state_sha256 != self.source_state_sha256
                or _source_interface(self.source) != self.source_interface_sha256
                or self.geometry.sha256() != self.geometry_sha256
                or tensor_sha256(self.reference.matrix) != self.reference_array_sha256
                or _context_layout_sha256(self.reference, self.geometry) != self.numerical_layout_sha256
                or file_digest(local_path(self.root, self.source_receipt_path)) != self.source_receipt_sha256
                or file_digest(local_path(self.root, self.reference_receipt_path)) != self.reference_receipt_sha256):
            raise ValueError("Verified method context source/reference/input changed")

    def _prefix(self, support, *, development):
        self._assert_current()
        if type(support) is not CalibrationPrefix:
            raise TypeError("An acquired calibration-prefix capability is required")
        support.__post_init__()
        expected = "meta_validation_support" if development else "calibration_prefix"
        if support.role != expected:
            raise PermissionError("Use the explicitly matching target or development prefix entry")
        if support.protocol_freeze_sha != self.signal_contract_sha256:
            raise ValueError("Support and verified source input protocols differ")
        self._signal_shape(support.x)

    def _signal_shape(self, x):
        if x.shape[1:] != (11, self.contract["n_times"]):
            raise ValueError("Trials must match the verified clinic11 frozen window")
        if x.dtype not in {torch.float32, torch.float64} or not torch.isfinite(x).all():
            raise ValueError("Finite floating point pre-RMS signals required")

    def _learned(self, method, row, receipt):
        if not isinstance(receipt, str) or not receipt:
            raise ValueError("This method requires its own verified independent learned artifact receipt")
        learned = meta_artifacts.load_learned_parameter(self.root, receipt, expected_method=method,
            expected_seed=self.pipeline_seed, input_contract_sha256=self.signal_contract_sha256,
            split_sha256=self.split_sha256, source=self.source, geometry=self.geometry)
        expected_dimension = row.get("dimension", row.get("rank"))
        if learned.job.dimension != expected_dimension or learned.job.radius != row["radius"]:
            raise ValueError("Learned artifact dimension/radius differ from the requested configuration")
        return learned

    def fit(self, method, support, configuration, *, learned_receipt=None, classic_model=None):
        self._prefix(support, development=False)
        return self._fit(method, support, configuration, learned_receipt=learned_receipt,
                         classic_model=classic_model, development=False)

    def fit_development(self, method, support, configuration, *, learned_receipt=None, classic_model=None):
        self._prefix(support, development=True)
        return self._fit(method, support, configuration, learned_receipt=learned_receipt,
                         classic_model=classic_model, development=True)

    def _fit(self, method, support, configuration, *, learned_receipt, classic_model, development):
        row = _configuration(method, configuration)
        if method not in CLASSICAL_METHODS and classic_model is not None:
            raise ValueError("Classical artifacts cannot enter neural/spatial methods")
        if method not in LEARNED_METHODS and learned_receipt is not None:
            raise ValueError("Fixed/random controls do not accept learned artifacts or caller generators")
        learned, dictionary, classical_receipt = None, None, None
        if method in LEARNED_METHODS:
            learned = self._learned(method, row, learned_receipt)
            dictionary = learned.matrix.detach().clone().to(self.geometry.q.device)
        if method in CLASSICAL_METHODS:
            if (type(classic_model) is not dict or not {"path", "sha256"} <= set(classic_model)
                    or set(classic_model) - {"path", "sha256", "bytes"}):
                raise ValueError("An independently SHA-bound new source-bank descriptor required; legacy/raw classical models are forbidden")
            bank = load_verified_source_bank(self.root, classic_model, method=method,
                expected_m=row["m"] if method == "B4" else None,
                expected_shrink=row["shrinkage"] if method == "B5" else None,
                input_sha=self.signal_contract_sha256, cache_sha=self.source_cache_sha256)
            expected_mode = "log_normalized_variance" if method == "B4" else "log_Euclidean_TS_float64"
            if (type(bank) is not classical.ClassicalModel or bank.method != method
                    or bank.receipt.get("synthetic") is not False
                    or bank.receipt.get("feature_mode") != expected_mode
                    or bank.receipt.get("bank_training_uid_set_sha256") != self.reference.events_sha256
                    or bank.receipt.get("signal_contract_sha256") != self.signal_contract_sha256
                    or bank.receipt.get("n_times") != self.contract["n_times"]
                    or (bank.receipt.get("m") != row["m"] if method == "B4" else bank.receipt.get("shrink") != row["shrinkage"])):
                raise ValueError("New classical source bank differs from the source f0/reference/input/configuration")
            classical_receipt = copy.deepcopy(classic_model)
            if method == "B4":
                entry = classical.fit_b4_development if development else classical.fit_b4_target
                payload = entry(bank, support, alpha=row["alpha"], m=row["m"])
            else:
                entry = classical.fit_b5_development if development else classical.fit_b5_target
                payload = entry(bank, support, l2=row["l2_to_source"])
        elif method in {"B2", "B3"}:
            config = neural_controls.FineTuneConfig(**row) if method == "B2" else neural_controls.AlignmentConfig(**row)
            authorization = neural_controls.SourceReferenceAuthorization(self.reference, self.signal_contract_sha256, tuple(sorted(SOURCE_IDS)))
            if development:
                payload = neural_controls.fit_neural_control_development(self.source, support, method=method,
                    geometry=self.geometry, source_reference=authorization, config=config)
            elif method == "B2":
                payload = neural_controls.fit_full_finetune(self.source, support, config=config)
            else:
                payload = neural_controls.fit_source_covariance_alignment(self.source, self.geometry, support,
                    source_reference=authorization, config=config)
        elif learned is not None and not development:
            payload = meta_artifacts.fit_verified_meta_method(self.root, learned, self.source, self.geometry, support)
        elif method in {"B6", "B7", "B10", "B11"}:
            config = lowrank.LowRankConfig(**row, pct_weight=.1 if method in {"B7", "B11"} else 0)
            if development:
                u0 = dictionary if learned is not None else lowrank.initialize_lowrank(row["rank"], self.pipeline_seed, device=self.geometry.q.device)
                initial_sha = tensor_sha256(u0)
                weights, updates, fallback = lowrank.inner_unroll_lowrank(self.source, self.geometry, u0, support, config=config)
                frozen = lowrank.LowRankWeights(*(t.detach().clone() for t in (weights.u, weights.v, weights.delta_w, weights.delta_b)))
                payload = lowrank.LowRankFitResult(frozen, updates, fallback, method, 20 * row["rank"] + self.source.feature_dim + 1,
                    support.event_uids, support.acquired_event_uids, support.protocol_freeze_sha,
                    self.source.frozen_state_sha256, self.geometry.sha256(), initial_sha, _weight_fingerprint_lowrank(frozen), config)
            else:
                payload = lowrank.fit_lowrank_target(self.source, self.geometry, support, method=method, config=config, seed=self.pipeline_seed)
        else:
            if dictionary is None:
                dictionary = initialize_dictionary(row.get("dimension", 2), self.pipeline_seed, device=self.geometry.q.device)
            config = adapt.InnerConfig(radius=row.get("radius", .15), head_lr=row.get("head_lr", .01),
                                       head_l2=row.get("head_l2", .001), pct_weight=.1 if method == "P+PCT" else 0)
            if development:
                if method == "B0":
                    weights, updates, fallback = adapt.zero_fast_weights(self.source, dictionary.shape[1], device=dictionary.device), 0, None
                else:
                    weights, updates, fallback = adapt.inner_unroll(self.source, self.geometry, dictionary, support,
                        config=config, family="additive" if method == "B8" else "exp", head_only=method == "B1")
                frozen = FastWeights(*(t.detach().clone() for t in (weights.theta, weights.delta_w, weights.delta_b)))
                count = 0 if method == "B0" else self.source.feature_dim + 1 + (0 if method == "B1" else dictionary.shape[1])
                payload = FitResult(frozen, updates, fallback, method, count, support.event_uids, support.acquired_event_uids,
                    support.protocol_freeze_sha, self.source.frozen_state_sha256, tensor_sha256(dictionary), self.geometry.sha256())
            else:
                payload = adapt.fit_target(self.source, self.geometry, dictionary, support, method=method, config=config)
        self._assert_current()
        fitted = object.__new__(RegisteredFit)
        values = {"method": method, "configuration": row, "support_role": support.role, "payload": payload,
            "dictionary": dictionary, "context_identity_sha256": self.identity_sha256,
            "fitted_parameter_sha256": _weight_fingerprint(payload),
            "learned_receipt": {"path": learned.receipt_path, "sha256": learned.receipt_sha256} if learned else None,
            "classical_receipt": classical_receipt,
            "scientific_meta_training_claim": learned is not None, "_seal": _FIT_SEAL}
        for name, value in values.items():
            object.__setattr__(fitted, name, value)
        object.__setattr__(fitted, "metadata_sha256", _fit_identity(fitted))
        return fitted

    def predict(self, fitted, inputs):
        self._assert_current()
        if type(fitted) is not RegisteredFit or type(inputs) is not PredictionInputs:
            raise TypeError("RegisteredFit and label-free PredictionInputs required")
        if getattr(fitted, "_seal", None) is not _FIT_SEAL:
            raise PermissionError("Only a verified context can issue a registered fitted method")
        inputs.__post_init__()
        self._signal_shape(inputs.x)
        if inputs.protocol_freeze_sha != self.signal_contract_sha256:
            raise ValueError("Prediction input differs from the verified signal contract")
        if (fitted.context_identity_sha256 != self.identity_sha256
                or _weight_fingerprint(fitted.payload) != fitted.fitted_parameter_sha256
                or _fit_identity(fitted) != fitted.metadata_sha256):
            raise ValueError("Fitted method context or parameters changed")
        if fitted.learned_receipt and file_digest(local_path(self.root, fitted.learned_receipt["path"])) != fitted.learned_receipt["sha256"]:
            raise ValueError("The learned method evidence receipt changed after fitting")
        if fitted.classical_receipt and file_digest(local_path(self.root, fitted.classical_receipt["path"])) != fitted.classical_receipt["sha256"]:
            raise ValueError("The classical source-bank evidence receipt changed after fitting")
        if type(fitted.payload) is classical.ClassicalModel:
            result = classical.predict_classical(fitted.payload, inputs)
            return PredictionResult(torch.from_numpy(result["logits"]), torch.from_numpy(result["probability"]),
                torch.from_numpy(result["prediction"]), result["event_uids"], inputs.protocol_freeze_sha)
        if type(fitted.payload) is neural_controls.NeuralControlFit:
            if fitted.payload.source is not self.source:
                raise ValueError("Neural fitted source is not the verified context capability")
            return neural_controls.predict_neural_control(fitted.payload, inputs)
        if type(fitted.payload) is lowrank.LowRankFitResult:
            return lowrank.predict_lowrank(self.source, self.geometry, fitted.payload, inputs)
        return adapt.predict(self.source, self.geometry, fitted.dictionary, fitted.payload, inputs)


def _weight_fingerprint_lowrank(weights):
    return hashlib.sha256("/".join(tensor_sha256(v) for v in
        (weights.u, weights.v, weights.delta_w, weights.delta_b)).encode()).hexdigest()


def _source_interface(source):
    # A weight/buffer hash alone does not cover layout, feature count, or such
    # non-tensor module settings as BatchNorm eps and the encoder activation.
    return digest({"feature_dim": source.feature_dim, "input_layout": source.input_layout,
                   "module_structure": repr(source), "numerical_layout": _module_layout(source)})
