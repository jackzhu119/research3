"""B2/B3 support-only neural controls from BASELINE_SPEC, independent of legacy.

No training job, query loader, or dataset access is started by this module.
Synthetic CPU tests verify software; they are not real EEG fits or pilots.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re

import torch
from torch import Tensor, nn

from ..adapt import (CalibrationPrefix, FrozenSourceModel, PredictionInputs,
                     PredictionResult, balanced_bce, module_state_sha256)
from ..contracts import SOURCE_IDS, digest
from ..spatial_geometry import ReferenceCovariance, SpatialGeometry, helmert_rows, joint_trial_rms, spd_roots, tensor_sha256


@dataclass(frozen=True)
class FineTuneConfig:
    learning_rate: float = 0.0003
    l2_to_source: float = 0.0001
    steps: int = 10

    def __post_init__(self):
        if self.learning_rate not in {0.0001, 0.0003, 0.001} or self.l2_to_source not in {0, 0.0001, 0.001} or self.steps != 10:
            raise ValueError("B2 requires its registered 3x3 grid and exactly 10 full-support steps")


@dataclass(frozen=True)
class AlignmentConfig:
    shrinkage: float = 0.05
    head_learning_rate: float = 0.01
    head_l2: float = 0.001
    steps: int = 10

    def __post_init__(self):
        if (self.shrinkage not in {0.01, 0.05, 0.1} or self.head_learning_rate not in {0.003, 0.01, 0.03}
                or self.head_l2 != 0.001 or self.steps != 10):
            raise ValueError("B3 requires its registered shrinkage/head-lr 3x3 grid and fixed head regularizer/10 steps")


@dataclass(frozen=True)
class SourceReferenceAuthorization:
    """Bind Rref to the source-pretraining training role and input protocol.

    The issuer must substantiate the four-source event manifest in a verified
    source receipt. This type checks provenance consistency, not whether the
    caller actually downloaded EEG. It never grants query or validation access.
    """
    reference: ReferenceCovariance
    protocol_freeze_sha: str
    dataset_ids: tuple[str, ...]
    role: str = "source_pretrain_train"

    def validate(self, geometry: SpatialGeometry, protocol_sha: str) -> str:
        if self.role != "source_pretrain_train" or len(self.dataset_ids) != 4 or set(self.dataset_ids) != SOURCE_IDS:
            raise PermissionError("B3 Rref must come only from the four registered source-pretrain training splits")
        if re.fullmatch(r"[0-9a-f]{64}", self.protocol_freeze_sha) is None or self.protocol_freeze_sha != protocol_sha:
            raise ValueError("B3 source reference and calibration input protocol differ")
        ref = self.reference
        if type(ref) is not ReferenceCovariance or ref.shrinkage != 0.05 or ref.weighting != "dataset_equal/subject_equal/trial_equal":
            raise ValueError("B3 must reuse the fixed hierarchical source Rref with shrinkage 0.05")
        if not ref.event_uids or tuple(sorted(set(ref.event_uids))) != ref.event_uids:
            raise ValueError("Source Rref needs an actual unique sorted source-event manifest")
        events_sha = hashlib.sha256("\n".join(ref.event_uids).encode()).hexdigest()
        if events_sha != ref.events_sha256:
            raise ValueError("Source Rref event-manifest digest mismatch")
        geometry.validate()
        if not torch.allclose(geometry.q, helmert_rows(device=geometry.q.device), atol=1e-14, rtol=0):
            raise ValueError("B3 requires the fixed clinic11 Helmert coordinate array")
        matrix = ref.matrix
        if matrix.dtype != torch.float64 or matrix.shape != (10, 10):
            raise ValueError("Source Rref must be float64 10x10")
        root, inverse = spd_roots(matrix)
        if (not torch.allclose(torch.trace(matrix), matrix.new_tensor(1.0), atol=1e-10, rtol=0)
                or float(torch.linalg.eigvalsh(matrix).min()) < 0.005 - 1e-12):
            raise ValueError("Source Rref must retain unit trace and the fixed source shrinkage floor")
        for actual, expected in ((ref.sqrt, root), (ref.invsqrt, inverse),
                                 (geometry.rref_sqrt, root.to(geometry.q.device)),
                                 (geometry.rref_invsqrt, inverse.to(geometry.q.device))):
            if not torch.allclose(actual, expected, atol=1e-10, rtol=1e-10):
                raise ValueError("B3 geometry does not match the authorized source Rref")
        return digest({"matrix": tensor_sha256(matrix), "events": events_sha,
                       "datasets": sorted(self.dataset_ids), "role": self.role,
                       "protocol_freeze_sha": protocol_sha})


def _no_amp():
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("First v2 implementation does not use mixed precision")


def _support(source, support, *, role="calibration_prefix"):
    _no_amp()
    if type(source) is not FrozenSourceModel or type(support) is not CalibrationPrefix:
        raise TypeError("FrozenSourceModel and CalibrationPrefix capabilities required")
    support.__post_init__()
    if role not in {"calibration_prefix", "meta_validation_support"} or support.role != role:
        raise PermissionError("B2/B3 prefix capability does not match the explicitly authorized target/development entry")
    if support.x.shape[1] != 11:
        raise ValueError("B2/B3 require the same pre-RMS clinic11 input")
    source.assert_unchanged()


def _has_both_classes(support):
    return bool((support.y == 0).any() and (support.y == 1).any())


def _buffers_sha(module):
    h = hashlib.sha256()
    for name, value in sorted(module.named_buffers()):
        value = value.detach().cpu().contiguous()
        h.update(f"{name}/{tuple(value.shape)}/{value.dtype}".encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class _EvalSourceCopy(nn.Module):
    """Private B2 trainable weights with permanently frozen BN/dropout mode."""
    def __init__(self, source):
        super().__init__()
        self.encoder, self.source_head = copy.deepcopy((source.encoder, source.source_head))
        self.input_layout, self.feature_dim = source.input_layout, source.feature_dim
        for value in self.parameters():
            value.requires_grad_(True)
        self.eval()
        self.buffer_sha256 = _buffers_sha(self)

    def train(self, mode=True):
        return super().train(False)

    def assert_buffers_unchanged(self):
        if any(module.training for module in self.modules()) or _buffers_sha(self) != self.buffer_sha256:
            raise RuntimeError("B2 target copy changed frozen BN/dropout/buffer state")

    def forward(self, x):
        self.assert_buffers_unchanged()
        template = next(self.parameters())
        normalized = joint_trial_rms(x).to(device=template.device, dtype=template.dtype)
        if self.input_layout == "N1CT":
            normalized = normalized.unsqueeze(1)
        features = self.encoder(normalized)
        if features.shape != (len(x), self.feature_dim):
            raise ValueError("B2 encoder feature shape changed")
        logits = self.source_head(features)
        if logits.shape != (len(x), 2):
            raise ValueError("B2 must retain the original double-logit head")
        self.assert_buffers_unchanged()
        return logits


def normalized_source_l2(parameters, source_parameters):
    """B2 ||psi-psi0||² / Ntrainable, including EVERY bias, excluding buffers."""
    parameters, source_parameters = tuple(parameters), tuple(source_parameters)
    if not parameters or len(parameters) != len(source_parameters):
        raise ValueError("A complete source parameter copy is required")
    if any(p.shape != p0.shape for p, p0 in zip(parameters, source_parameters)):
        raise ValueError("Source and target parameter shapes differ")
    count = sum(p.numel() for p in parameters)
    return sum((p - p0).square().sum() for p, p0 in zip(parameters, source_parameters)) / count


@dataclass(frozen=True)
class NeuralControlFit:
    method: str
    source: FrozenSourceModel
    target_model: nn.Module | None
    alignment: Tensor | None
    delta_w: Tensor | None
    delta_b: Tensor | None
    updates: int
    fallback_reason: str | None
    trainable_parameters: int
    support_event_uids: tuple[str, ...]
    acquired_event_uids: tuple[str, ...]
    protocol_freeze_sha: str
    source_state_sha256: str
    fitted_state_sha256: str | None
    source_reference_authorization_sha256: str | None
    target_covariance: Tensor | None
    target_shrunk_covariance: Tensor | None
    support_role: str
    config: FineTuneConfig | AlignmentConfig


def _result(source, support, method, config, *, target_model=None, alignment=None, delta_w=None, delta_b=None,
            updates=0, fallback=None, count=0, authorization_sha=None, ctarget=None, rt=None):
    state_sha = module_state_sha256(target_model) if target_model is not None else (
        digest({"alignment": tensor_sha256(alignment), "delta_w": tensor_sha256(delta_w),
                "delta_b": tensor_sha256(delta_b)}) if alignment is not None else None)
    source.assert_unchanged()
    return NeuralControlFit(method, source, target_model, alignment, delta_w, delta_b, updates, fallback,
                            count, support.event_uids, support.acquired_event_uids, support.protocol_freeze_sha,
                            source.frozen_state_sha256, state_sha, authorization_sha, ctarget, rt, support.role, config)


def _fit_full_finetune(source: FrozenSourceModel, support: CalibrationPrefix, *, config=None, role) -> NeuralControlFit:
    """B2: private full f0 copy, 10 simultaneous full-support SGD updates, eval BN.

    Optimizes Lbal + lambda*||psi-psi0||²/Ntrainable, including biases. No Adam,
    momentum, data augmentation, query statistics or stopping/selection is used.
    """
    _support(source, support, role=role)
    config = config or FineTuneConfig()
    if type(config) is not FineTuneConfig:
        raise TypeError("FineTuneConfig required")
    config.__post_init__()
    target = _EvalSourceCopy(source)
    parameters = tuple(target.parameters())
    count = sum(p.numel() for p in parameters)
    if not _has_both_classes(support):
        return _result(source, support, "B2", config, fallback="insufficient_class_support", count=count)
    origin = tuple(p.detach().clone() for p in parameters)
    for _ in range(config.steps):
        logits = target(support.x.detach())
        loss = balanced_bce(logits, support.y) + config.l2_to_source * normalized_source_l2(parameters, origin)
        if not torch.isfinite(loss):
            raise FloatingPointError("B2 support loss is nonfinite")
        gradients = torch.autograd.grad(loss, parameters)
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-config.learning_rate)
    target.assert_buffers_unchanged()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return _result(source, support, "B2", config, target_model=target, updates=config.steps, count=count)


def _support_alignment(geometry: SpatialGeometry, support: CalibrationPrefix, *, shrinkage=0.05, role):
    """B3's support-only covariance and exact target→source alignment direction.

    Equal valid-trial mean of trace-normalized QXX'Q', then Rt=(1-a)Ct+a*I/d.
    This is a fixed-source covariance matching control, not original two-sided EA.
    """
    if type(support) is not CalibrationPrefix or role not in {"calibration_prefix", "meta_validation_support"} or support.role != role:
        raise PermissionError("B3 covariance accepts the calibration-prefix capability only")
    support.__post_init__()
    geometry.validate()
    if shrinkage not in {0.01, 0.05, 0.1} or not len(support.x):
        raise ValueError("Registered B3 shrinkage and nonempty support required")
    x = support.x.detach().to(device=geometry.q.device, dtype=torch.float64)
    if x.shape[1] != 11:
        raise ValueError("B3 requires clinic11")
    z = geometry.q @ x
    covariances = z @ z.transpose(-1, -2)
    traces = covariances.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not torch.isfinite(traces).all() or torch.any(traces <= 0):
        raise ValueError("Degenerate CAR support covariance")
    ctarget = (covariances / traces[:, None, None]).mean(0)
    d = geometry.q.shape[0]
    rt = (1 - shrinkage) * ctarget + shrinkage * torch.eye(d, dtype=torch.float64, device=x.device) / d
    _, invsqrt = spd_roots(rt)
    alignment = geometry.q.T @ geometry.rref_sqrt @ invsqrt @ geometry.q
    return alignment.detach(), ctarget.detach(), rt.detach()


def _fit_source_covariance_alignment(source: FrozenSourceModel, geometry: SpatialGeometry,
                                    support: CalibrationPrefix, *, source_reference: SourceReferenceAuthorization,
                                    config=None, role) -> NeuralControlFit:
    """B3: support covariance matching plus original two-logit margin-head fitting."""
    _support(source, support, role=role)
    config = config or AlignmentConfig()
    if type(config) is not AlignmentConfig or type(source_reference) is not SourceReferenceAuthorization:
        raise TypeError("AlignmentConfig and explicit SourceReferenceAuthorization required")
    config.__post_init__()
    authorization_sha = source_reference.validate(geometry, support.protocol_freeze_sha)
    count = source.feature_dim + 1
    if not _has_both_classes(support):
        return _result(source, support, "B3", config, fallback="insufficient_class_support",
                       count=count, authorization_sha=authorization_sha)
    alignment, ctarget, rt = _support_alignment(geometry, support, shrinkage=config.shrinkage, role=role)
    with torch.no_grad():
        transformed = joint_trial_rms(alignment @ support.x.detach().to(device=alignment.device, dtype=torch.float64))
        features, base_logits = source.features_and_logits(transformed)
    dw = features.new_zeros(source.feature_dim, requires_grad=True)
    db = features.new_zeros((), requires_grad=True)
    for _ in range(config.steps):
        margin_increment = features @ dw + db
        logits = base_logits + torch.stack((-margin_increment / 2, margin_increment / 2), 1)
        loss = balanced_bce(logits, support.y) + config.head_l2 * (dw.square().sum() + db.square()) / count
        if not torch.isfinite(loss):
            raise FloatingPointError("B3 support loss is nonfinite")
        gw, gb = torch.autograd.grad(loss, (dw, db))
        dw = (dw - config.head_learning_rate * gw).detach().requires_grad_(True)
        db = (db - config.head_learning_rate * gb).detach().requires_grad_(True)
    return _result(source, support, "B3", config, alignment=alignment.detach().clone(),
                   delta_w=dw.detach().clone(), delta_b=db.detach().clone(), updates=config.steps,
                   count=count, authorization_sha=authorization_sha, ctarget=ctarget, rt=rt)


def fit_full_finetune(source: FrozenSourceModel, support: CalibrationPrefix, *, config=None) -> NeuralControlFit:
    """Formal B2 target entry; accepts only the target calibration-prefix role."""
    return _fit_full_finetune(source, support, config=config, role="calibration_prefix")


def support_alignment(geometry: SpatialGeometry, support: CalibrationPrefix, *, shrinkage=0.05):
    """Formal B3 target covariance entry; no development/query role promotion."""
    return _support_alignment(geometry, support, shrinkage=shrinkage, role="calibration_prefix")


def fit_source_covariance_alignment(source: FrozenSourceModel, geometry: SpatialGeometry,
                                    support: CalibrationPrefix, *, source_reference: SourceReferenceAuthorization,
                                    config=None) -> NeuralControlFit:
    """Formal B3 target entry with a separately authorized source-only Rref."""
    return _fit_source_covariance_alignment(source, geometry, support, source_reference=source_reference,
                                             config=config, role="calibration_prefix")


def fit_neural_control_development(source: FrozenSourceModel, support: CalibrationPrefix, *, method,
                                    geometry=None, source_reference=None, config=None) -> NeuralControlFit:
    """Validation-prefix adaptation for the registered B2/B3 development grid.

    Accepts meta_validation_support only. Query inputs/labels and model-selection
    metrics remain outside fitting; support keeps its role in the fit receipt.
    The high-level episode catalog still proves participant/session ownership.
    """
    if type(support) is not CalibrationPrefix or support.role != "meta_validation_support":
        raise PermissionError("A development fit needs meta_validation_support, not query/target/internal data")
    if method == "B2":
        return _fit_full_finetune(source, support, config=config, role="meta_validation_support")
    if method == "B3":
        return _fit_source_covariance_alignment(source, geometry, support, source_reference=source_reference,
                                                config=config, role="meta_validation_support")
    raise ValueError("This development entry implements B2 and B3 only")


def predict_neural_control(fitted: NeuralControlFit, inputs: PredictionInputs) -> PredictionResult:
    """Label-free fixed-model prediction; no query mean/covariance/early stopping."""
    _no_amp()
    if type(fitted) is not NeuralControlFit or type(inputs) is not PredictionInputs:
        raise TypeError("NeuralControlFit and label-free PredictionInputs required")
    inputs.__post_init__()
    source = fitted.source
    source.assert_unchanged()
    if source.frozen_state_sha256 != fitted.source_state_sha256 or inputs.protocol_freeze_sha != fitted.protocol_freeze_sha:
        raise ValueError("B2/B3 source or input protocol changed after fitting")
    if set(inputs.event_uids) & set(fitted.acquired_event_uids):
        raise ValueError("Query overlaps the acquired calibration prefix")
    with torch.no_grad():
        if fitted.fallback_reason:
            logits = source.source_logits(inputs.x)
        elif fitted.method == "B2":
            if module_state_sha256(fitted.target_model) != fitted.fitted_state_sha256:
                raise ValueError("B2 fitted parameters/buffers changed")
            logits = fitted.target_model(inputs.x)
        elif fitted.method == "B3":
            state = digest({"alignment": tensor_sha256(fitted.alignment), "delta_w": tensor_sha256(fitted.delta_w),
                            "delta_b": tensor_sha256(fitted.delta_b)})
            if state != fitted.fitted_state_sha256:
                raise ValueError("B3 alignment or head increment changed")
            normalized = joint_trial_rms(fitted.alignment @ inputs.x.to(device=fitted.alignment.device, dtype=torch.float64))
            features, base = source.features_and_logits(normalized)
            dm = features @ fitted.delta_w + fitted.delta_b
            logits = base + torch.stack((-dm / 2, dm / 2), 1)
        else:
            raise ValueError("Unknown neural control")
        probabilities, labels = logits.softmax(1), logits.argmax(1)
    source.assert_unchanged()
    return PredictionResult(logits, probabilities, labels, inputs.event_uids, inputs.protocol_freeze_sha)
