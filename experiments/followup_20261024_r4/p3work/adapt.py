"""Support-only Paper3 target fitting and differentiable inner updates.

This is software preparation, not a trained method or an EEG pilot. No target
query labels are accepted by fitting/prediction interfaces. Meta-development
query labels have a distinct capability type in methods.meta_train.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import re
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .spatial_geometry import SpatialGeometry, joint_trial_rms, project_ball, tensor_sha256


def _freeze_sha(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("a concrete protocol_freeze_sha is required")


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    # named_parameters/named_buffers also cover nonpersistent registered buffers.
    entries = [("parameter", n, v) for n, v in module.named_parameters()]
    entries += [("buffer", n, v) for n, v in module.named_buffers()]
    for kind, name, value in sorted(entries, key=lambda e: (e[0], e[1])):
        x = value.detach().cpu().contiguous()
        digest.update(f"{kind}/{name}/{tuple(x.shape)}/{x.dtype}".encode())
        digest.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationPrefix:
    """Authorized acquired-prefix view; x/y contain ONLY valid prefix events.

    acquired_event_uids includes failed-QC trials and preserves acquisition order;
    event_uids is its valid subsequence. Never replace failed prefix trials with
    later events. Constructor cannot prove upstream physical order: the event
    manifest builder must audit it before issuing this capability.
    """
    x: Tensor
    y: Tensor
    event_uids: tuple[str, ...]
    acquired_event_uids: tuple[str, ...]
    protocol_freeze_sha: str
    role: Literal["calibration_prefix", "meta_train_support", "meta_validation_support"] = "calibration_prefix"

    def __post_init__(self):
        _freeze_sha(self.protocol_freeze_sha)
        if self.role not in {"calibration_prefix", "meta_train_support", "meta_validation_support"}:
            raise PermissionError("only a calibration-prefix capability can be fitted")
        if self.x.ndim != 3 or self.y.shape != (self.x.shape[0],):
            raise ValueError("support expects N x C x T and one label per event")
        if len(self.event_uids) != len(self.y) or len(set(self.event_uids)) != len(self.event_uids):
            raise ValueError("duplicate or missing support event IDs")
        if len(set(self.acquired_event_uids)) != len(self.acquired_event_uids):
            raise ValueError("duplicate acquired event IDs")
        if len(self.acquired_event_uids) not in {0, 4, 8, 12, 16, 20}:
            raise ValueError("unsupported acquired calibration budget")
        valid_set = set(self.event_uids)
        if tuple(uid for uid in self.acquired_event_uids if uid in valid_set) != self.event_uids:
            raise ValueError("valid events must be an ordered acquired-prefix subsequence")
        if self.y.dtype not in {torch.int32, torch.int64} or not torch.isin(self.y, torch.tensor([0, 1], device=self.y.device)).all():
            raise ValueError("support labels must be integer 0/1")
        if not torch.isfinite(self.x).all():
            raise ValueError("upstream QC must exclude nonfinite support trials")
        if len(self.x) and torch.any(self.x.to(torch.float64).square().sum((-2, -1)) == 0):
            raise ValueError("upstream QC must exclude zero-energy support trials")


@dataclass(frozen=True)
class PredictionInputs:
    """No labels, covariance statistics, or selection feedback are present."""
    x: Tensor
    event_uids: tuple[str, ...]
    protocol_freeze_sha: str

    def __post_init__(self):
        _freeze_sha(self.protocol_freeze_sha)
        if self.x.ndim != 3 or len(self.x) != len(self.event_uids) or len(set(self.event_uids)) != len(self.event_uids):
            raise ValueError("prediction inputs require one unique UID per trial")


class FrozenSourceModel(nn.Module):
    """A private, permanently eval-mode copy of phi and the ORIGINAL two-logit h.

    requires_grad=False freezes source parameters, but input derivatives remain
    enabled so spatial inner and outer gradients pass through the encoder.
    """
    def __init__(self, encoder: nn.Module, source_head: nn.Module, *, feature_dim: int,
                 input_layout: Literal["NCT", "N1CT"] = "NCT"):
        super().__init__()
        if feature_dim < 1 or input_layout not in {"NCT", "N1CT"}:
            raise ValueError("invalid feature dimension or source input layout")
        self.encoder = copy.deepcopy(encoder)
        self.source_head = copy.deepcopy(source_head)
        self.feature_dim = feature_dim
        self.input_layout = input_layout
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        self.frozen_state_sha256 = module_state_sha256(self)

    def train(self, mode: bool = True):
        # Permanently freeze BN statistics and dropout even if a parent calls train().
        return super().train(False)

    def assert_unchanged(self):
        if any(m.training for m in self.modules()) or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("source parameters/BN/dropout freeze violated")
        if module_state_sha256(self) != self.frozen_state_sha256:
            raise RuntimeError("source parameter or buffer changed")

    def features_and_logits(self, normalized_x: Tensor) -> tuple[Tensor, Tensor]:
        self.assert_unchanged()
        parameter = next(self.encoder.parameters(), None)
        head_parameter = next(self.source_head.parameters(), None)
        template = parameter if parameter is not None else head_parameter
        x = normalized_x if template is None else normalized_x.to(device=template.device, dtype=template.dtype)
        if self.input_layout == "N1CT":
            x = x.unsqueeze(1)
        features = self.encoder(x)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("encoder must expose N x feature_dim before its source head")
        base_logits = self.source_head(features)
        if base_logits.shape != (len(features), 2):
            raise ValueError("the original source head must output N x 2 logits")
        return features, base_logits

    def source_logits(self, x: Tensor) -> Tensor:
        return self.features_and_logits(joint_trial_rms(x))[1]


@dataclass(frozen=True)
class InnerConfig:
    radius: float = 0.15
    steps: int = 10
    theta_lr: float = 0.05
    head_lr: float = 0.01
    theta_l2: float = 0.01
    head_l2: float = 0.001
    pct_weight: float = 0.0
    pct_beta: float = 5.0

    def __post_init__(self):
        if self.steps != 10:
            raise ValueError("v2 inner fitting has exactly 10 full-support updates")
        if self.radius <= 0 or min(self.theta_lr, self.head_lr) <= 0:
            raise ValueError("positive radius and learning rates required")
        if min(self.theta_l2, self.head_l2, self.pct_weight, self.pct_beta) < 0:
            raise ValueError("negative regularization is not permitted")

    def validate_primary(self, *, pct=False):
        if self.radius not in {0.05, 0.15, 0.35}:
            raise ValueError("registered primary radius grid is {0.05,0.15,0.35}")
        if (self.theta_lr, self.head_lr, self.theta_l2, self.head_l2) != (0.05, 0.01, 0.01, 0.001):
            raise ValueError("primary inner optimizer constants require a new frozen protocol to change")
        if self.pct_weight != (0.1 if pct else 0.0) or self.pct_beta != 5.0:
            raise ValueError("primary P/P+PCT regularizer settings differ from v2")


@dataclass(frozen=True)
class FastWeights:
    theta: Tensor
    delta_w: Tensor
    delta_b: Tensor


@dataclass(frozen=True)
class FitResult:
    weights: FastWeights
    updates: int
    fallback_reason: str | None
    method: str
    trainable_parameters: int
    support_event_uids: tuple[str, ...]
    acquired_event_uids: tuple[str, ...]
    protocol_freeze_sha: str
    source_state_sha256: str
    dictionary_sha256: str
    geometry_sha256: str
    # Fixed teacher is an adaptation-only implementation detail, never target query.


def balanced_bce(logits: Tensor, labels: Tensor) -> Tensor:
    labels = labels.to(device=logits.device)
    if logits.shape != (len(labels), 2) or not ((labels == 0).any() and (labels == 1).any()):
        raise ValueError("balanced BCE requires both classes and N x 2 logits")
    margin = logits[:, 1] - logits[:, 0]
    loss = F.binary_cross_entropy_with_logits(margin, labels.to(margin.dtype), reduction="none")
    return 0.5 * (loss[labels == 0].mean() + loss[labels == 1].mean())


def zero_fast_weights(source: FrozenSourceModel, rank: int, *, device) -> FastWeights:
    parameter = next(source.source_head.parameters(), None)
    dtype = torch.float32 if parameter is None else parameter.dtype
    return FastWeights(torch.zeros(rank, dtype=torch.float64, device=device, requires_grad=True),
                       torch.zeros(source.feature_dim, dtype=dtype, device=device, requires_grad=True),
                       torch.zeros((), dtype=dtype, device=device, requires_grad=True))


def adapted_logits(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
                   x: Tensor, weights: FastWeights, *, family="exp") -> Tensor:
    features, base = source.features_and_logits(geometry.transform(x, weights.theta, b, family=family))
    delta_margin = features @ weights.delta_w + weights.delta_b
    # Preserve the original source two-logit numerical path, including source bias.
    return base + torch.stack((-delta_margin / 2, delta_margin / 2), dim=1)


def inner_unroll(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
                 support: CalibrationPrefix, *, config: InnerConfig | None = None,
                 family="exp", head_only=False, create_graph=False) -> tuple[FastWeights, int, str | None]:
    """Functional 10-step simultaneous SGD; create_graph retains full B dependency.

    This internal primitive is also used by the separately authorized meta_train
    entry. It never accepts query inputs or query labels.
    """
    if type(support) is not CalibrationPrefix:
        raise TypeError("CalibrationPrefix capability required")
    config = config or InnerConfig()
    source.assert_unchanged()
    weights = zero_fast_weights(source, b.shape[1], device=b.device)
    if not ((support.y == 0).any() and (support.y == 1).any()):
        return weights, 0, "insufficient_class_support"
    teacher = None
    if config.pct_weight:
        with torch.no_grad():
            teacher = source.source_logits(support.x).detach()
        fixed_weight = 1 + config.pct_beta * (teacher.argmax(dim=1) == support.y.to(teacher.device))
    for _ in range(config.steps):
        logits = adapted_logits(source, geometry, b, support.x, weights, family=family)
        loss = balanced_bce(logits, support.y)
        loss = loss + config.theta_l2 * weights.theta.square().sum()
        loss = loss + config.head_l2 * (weights.delta_w.square().sum() + weights.delta_b.square()) / (source.feature_dim + 1)
        if teacher is not None:
            loss = loss + config.pct_weight * (fixed_weight * 0.5 * (logits - teacher).square().sum(dim=1)).mean()
        parameters = (weights.delta_w, weights.delta_b) if head_only else (weights.theta, weights.delta_w, weights.delta_b)
        gradients = torch.autograd.grad(loss, parameters, create_graph=create_graph)
        if head_only:
            theta = weights.theta
            dw, db = gradients
        else:
            gt, dw, db = gradients
            theta = project_ball(weights.theta - config.theta_lr * gt, config.radius)
        weights = FastWeights(theta, weights.delta_w - config.head_lr * dw, weights.delta_b - config.head_lr * db)
        if not create_graph:
            weights = FastWeights(*(v.detach().requires_grad_(True) for v in (weights.theta, weights.delta_w, weights.delta_b)))
    source.assert_unchanged()
    return weights, config.steps, None


def fit_target(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
               support: CalibrationPrefix, *, method="P", config: InnerConfig | None = None) -> FitResult:
    """Fit only an authorized target prefix; no query argument is available.

    P and B8 require a separately trained dictionary; this function does not
    claim that a supplied dictionary has been learned. B9 uses a fixed random
    dictionary generated from the pipeline seed. Full W5 comparisons remain a
    later gate and are not silently replaced by these implemented families.
    """
    if method not in {"P", "P+PCT", "B0", "B1", "B8", "B9", "full-traceless-exp", "symmetric-exp", "skew-exp"}:
        raise NotImplementedError(f"method {method} is not implemented by this kernel")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("v2 first implementation does not use mixed precision")
    config = config or InnerConfig(pct_weight=0.1 if method == "P+PCT" else 0.0)
    if method == "P" and config.pct_weight != 0:
        raise ValueError("P has lambda_PC=0; register P+PCT separately")
    if method in {"P", "P+PCT", "B8", "B9", "full-traceless-exp", "symmetric-exp", "skew-exp"}:
        config.validate_primary(pct=method == "P+PCT")
    if method == "B1" and (config.head_lr not in {0.003, 0.01, 0.03} or config.head_l2 not in {0.0, 0.001, 0.01} or config.pct_weight != 0):
        raise ValueError("B1 requires its registered 9-setting head grid without PCT")
    if type(support) is not CalibrationPrefix:
        raise TypeError("CalibrationPrefix capability required")
    support.__post_init__()
    geometry.validate_dictionary(b)
    if method in {"P", "P+PCT", "B8", "B9"} and (geometry.e.shape[1] != 99 or b.shape[1] not in {2, 4, 8}):
        raise ValueError("primary dictionary families require full 99-D coordinates and r in {2,4,8}")
    if method == "full-traceless-exp" and b.shape != (99, 99):
        raise ValueError("full-traceless-exp has 99 spatial coefficients")
    if method == "symmetric-exp" and geometry.e.shape[1] != 54:
        raise ValueError("symmetric ablation requires the symmetric-traceless E")
    if method == "skew-exp" and geometry.e.shape[1] != 45:
        raise ValueError("skew ablation requires the skew E")
    if support.role != "calibration_prefix":
        raise PermissionError("target fitting does not accept development query/support capabilities")
    if method == "B0":
        weights, updates, fallback = zero_fast_weights(source, b.shape[1], device=b.device), 0, None
    else:
        weights, updates, fallback = inner_unroll(source, geometry, b, support, config=config,
                                                family="additive" if method == "B8" else "exp", head_only=method == "B1")
    spatial_count = 0 if method in {"B0", "B1"} else b.shape[1]
    count = 0 if method == "B0" else spatial_count + source.feature_dim + 1
    frozen_weights = FastWeights(*(v.detach().clone() for v in (weights.theta, weights.delta_w, weights.delta_b)))
    return FitResult(frozen_weights, updates, fallback, method, count, support.event_uids,
                     support.acquired_event_uids, support.protocol_freeze_sha, source.frozen_state_sha256,
                     tensor_sha256(b), geometry.sha256())


@dataclass(frozen=True)
class PredictionResult:
    logits: Tensor
    probabilities: Tensor
    labels: Tensor
    event_uids: tuple[str, ...]
    protocol_freeze_sha: str


def predict(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
            fitted: FitResult, inputs: PredictionInputs) -> PredictionResult:
    if type(inputs) is not PredictionInputs:
        raise TypeError("label-free PredictionInputs capability required")
    inputs.__post_init__()
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("v2 first implementation does not use mixed precision")
    if inputs.protocol_freeze_sha != fitted.protocol_freeze_sha:
        raise ValueError("prediction and calibration protocol digests differ")
    if fitted.source_state_sha256 != source.frozen_state_sha256:
        raise ValueError("prediction source differs from fitted source")
    if fitted.dictionary_sha256 != tensor_sha256(b) or fitted.geometry_sha256 != geometry.sha256():
        raise ValueError("prediction dictionary or reference geometry differs from fit")
    if set(inputs.event_uids) & set(fitted.acquired_event_uids):
        raise ValueError("prediction events overlap acquired calibration prefix")
    with torch.no_grad():
        if fitted.fallback_reason or fitted.method == "B0":
            logits = source.source_logits(inputs.x)
        else:
            logits = adapted_logits(source, geometry, b, inputs.x, fitted.weights,
                                    family="additive" if fitted.method == "B8" else "exp")
        probabilities = logits.softmax(dim=1)
        labels = logits.argmax(dim=1)  # ties choose class 0, from the SAME logits
    source.assert_unchanged()
    return PredictionResult(logits, probabilities, labels, inputs.event_uids, inputs.protocol_freeze_sha)
