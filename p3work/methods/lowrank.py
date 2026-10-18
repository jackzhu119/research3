"""B6/B7 free spatial LoRA and B10/B11 Meta-LoRA CPU/GPU math kernels.

No data loading, scheduling, real fitting, or training loop is provided here.
Target fitting accepts only an acquired calibration-prefix capability; future
labels are accepted solely by the separate Stieger meta-train episode operator.
B10 and B11 require separately learned U0 artifacts: this module never labels a
seeded random initialization as a completed meta-trained model.
Fixed Helmert coordinates are checked here. Numerical SPD geometry checks do
not prove source-only Rref provenance; the production artifact loader must bind
that reference to its formal source-training evidence and receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor

from ..adapt import (CalibrationPrefix, FrozenSourceModel, PredictionInputs,
                     PredictionResult, balanced_bce)
from ..spatial_geometry import (SpatialGeometry, helmert_rows, joint_trial_rms, positive_qr,
                                stiefel_step, tensor_sha256)
from .meta_train import MetaTrainEpisode


METHODS = frozenset({"B6", "B7", "B10", "B11"})
META_METHODS = frozenset({"B10", "B11"})
PCT_METHODS = frozenset({"B7", "B11"})
LOWRANK_GRID = tuple((rank, radius) for rank in (1, 2, 4) for radius in (0.05, 0.15, 0.35))


@dataclass(frozen=True)
class LowRankConfig:
    rank: int = 1
    radius: float = 0.15
    steps: int = 10
    factor_lr: float = 0.05
    head_lr: float = 0.01
    spatial_l2: float = 0.01
    head_l2: float = 0.001
    pct_weight: float = 0.0
    pct_beta: float = 5.0

    def __post_init__(self):
        if type(self.rank) is not int or (self.rank, self.radius) not in LOWRANK_GRID:
            raise ValueError("registered low-rank grid is k={1,2,4}, rho={.05,.15,.35}")
        if type(self.steps) is not int or self.steps != 10:
            raise ValueError("B6/B7/B10/B11 have exactly 10 simultaneous inner updates")
        if (self.factor_lr, self.head_lr, self.spatial_l2, self.head_l2) != (0.05, 0.01, 0.01, 0.001):
            raise ValueError("low-rank optimizer constants require a newly frozen protocol to change")
        if self.pct_weight not in {0.0, 0.1} or self.pct_beta != 5.0:
            raise ValueError("PCT uses alpha=1, beta=5, lambda=.1; non-PCT has lambda=0")


def _configuration(config: LowRankConfig | None, method: str) -> LowRankConfig:
    if method not in METHODS:
        raise ValueError("low-rank kernel accepts B6/B7/B10/B11 only")
    config = config or LowRankConfig(pct_weight=0.1 if method in PCT_METHODS else 0.0)
    if type(config) is not LowRankConfig:
        raise TypeError("LowRankConfig required")
    config.__post_init__()
    if config.pct_weight != (0.1 if method in PCT_METHODS else 0.0):
        raise ValueError("method identity and PCT configuration disagree")
    return config


def _precision_guard():
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("v2 first low-rank implementation does not use mixed precision")


def initialize_lowrank(rank: int, seed: int, *, device=None) -> Tensor:
    """The fixed seed gives thin positive-diagonal QR U0; V0 is always zero.

    This uses a private CPU generator and does not consume the global RNG. Meta
    methods begin here too, but must learn their own U0 in an independent run.
    """
    if type(rank) is not int or rank not in {1, 2, 4} or type(seed) is not int:
        raise ValueError("registered rank and integer pipeline seed required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return positive_qr(torch.randn(10, rank, dtype=torch.float64, generator=generator)).to(device=device)


def _require_fixed_helmert(geometry: SpatialGeometry):
    if (geometry.q.shape != (10, 11) or geometry.q.dtype != torch.float64
            or not torch.equal(geometry.q, helmert_rows(11, device=geometry.q.device))):
        raise ValueError("registered low-rank controls require the fixed clinic11 Helmert Q")


def _validate_geometry(geometry: SpatialGeometry):
    geometry.validate()
    _require_fixed_helmert(geometry)


def _validate_u0(geometry: SpatialGeometry, u0: Tensor, rank: int):
    _validate_geometry(geometry)
    if (u0.shape != (10, rank) or u0.dtype != torch.float64 or not torch.isfinite(u0).all()
            or u0.device != geometry.q.device):
        raise ValueError("U0 must be finite float64 CAR10 x k on the geometry device")
    eye = torch.eye(rank, dtype=torch.float64, device=u0.device)
    if not torch.allclose(u0.T @ u0, eye, atol=1e-10, rtol=1e-10):
        raise ValueError("initial U0 must have orthonormal columns")


@dataclass(frozen=True)
class LowRankWeights:
    u: Tensor
    v: Tensor
    delta_w: Tensor
    delta_b: Tensor


def _weights_sha256(weights: LowRankWeights) -> str:
    return hashlib.sha256("/".join(tensor_sha256(v) for v in
        (weights.u, weights.v, weights.delta_w, weights.delta_b)).encode()).hexdigest()


def zero_lowrank_weights(source: FrozenSourceModel, u0: Tensor) -> LowRankWeights:
    source.assert_unchanged()
    parameter = next(source.source_head.parameters(), None)
    if parameter is None:
        parameter = next(source.encoder.parameters(), None)
    dtype = torch.float32 if parameter is None else parameter.dtype
    head_device = u0.device if parameter is None else parameter.device
    u = u0 if u0.requires_grad else u0.detach().clone().requires_grad_(True)
    return LowRankWeights(u, torch.zeros_like(u0, requires_grad=True),
        torch.zeros(source.feature_dim, dtype=dtype, device=head_device, requires_grad=True),
        torch.zeros((), dtype=dtype, device=head_device, requires_grad=True))


def lowrank_matrix(geometry: SpatialGeometry, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return A=Q^T R^.5 (I+UV^T) R^-.5 Q, its whitened T, and update D.

    U and V after inner updates are free factors. Only the initial U0 and meta
    outer U0 are Stiefel constrained; the inner U must not be QR-retracted.
    """
    _require_fixed_helmert(geometry)
    if (u.ndim != 2 or u.shape[0] != 10 or u.shape[1] not in {1, 2, 4} or v.shape != u.shape
            or any(t.dtype != torch.float64 or not torch.isfinite(t).all() for t in (u, v))
            or u.device != geometry.q.device or v.device != geometry.q.device):
        raise ValueError("finite matched float64 CAR10 x k low-rank factors required")
    update = u @ v.T
    transform = torch.eye(10, dtype=torch.float64, device=u.device) + update
    a = geometry.q.T @ geometry.rref_sqrt @ transform @ geometry.rref_invsqrt @ geometry.q
    return a, transform, update


def lowrank_logits(source: FrozenSourceModel, geometry: SpatialGeometry, x: Tensor,
                   weights: LowRankWeights) -> Tensor:
    """Pure differentiable logits; capability checks belong to fitting/scoring.

    Validation runners may use this after inner_unroll_lowrank on an authorized
    meta-validation support prefix, scoring their separately held label view.
    """
    _precision_guard()
    if type(weights) is not LowRankWeights or x.ndim != 3 or x.shape[1] != 11:
        raise ValueError("low-rank weights and N x 11 x T trials required")
    a, _, _ = lowrank_matrix(geometry, weights.u, weights.v)
    normalized = joint_trial_rms(a @ x.to(device=a.device, dtype=torch.float64))
    features, base_logits = source.features_and_logits(normalized)
    delta_margin = features @ weights.delta_w + weights.delta_b
    return base_logits + torch.stack((-delta_margin / 2, delta_margin / 2), dim=1)


def project_lowrank_update(u: Tensor, v: Tensor, radius: float) -> tuple[Tensor, Tensor]:
    """Project actual UV^T Frobenius norm by scaling V only, retaining U.

    The inactive branch returns the factors unchanged. Avoiding an unnecessary
    norm derivative at zero also keeps full second derivatives finite there.
    """
    if radius not in {0.05, 0.15, 0.35}:
        raise ValueError("registered low-rank radius required")
    norm = torch.linalg.vector_norm(u @ v.T)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite low-rank update")
    if bool(norm > radius):
        v = v * (radius / norm)
    return u, v


def inner_unroll_lowrank(source: FrozenSourceModel, geometry: SpatialGeometry, u0: Tensor,
                         support: CalibrationPrefix, *, config: LowRankConfig | None = None,
                         create_graph=False) -> tuple[LowRankWeights, int, str | None]:
    """Ten simultaneous full-support updates, retaining U0 dependency if asked.

    This internal operator accepts target, meta-train, and meta-validation
    support capabilities. It never accepts query inputs or query labels.
    """
    _precision_guard()
    if type(support) is not CalibrationPrefix:
        raise TypeError("CalibrationPrefix capability required")
    support.__post_init__()
    config = config or LowRankConfig()
    if type(config) is not LowRankConfig:
        raise TypeError("LowRankConfig required")
    config.__post_init__()
    _validate_u0(geometry, u0, config.rank)
    weights = zero_lowrank_weights(source, u0)
    if not ((support.y == 0).any() and (support.y == 1).any()):
        return weights, 0, "insufficient_class_support"
    teacher = fixed_weight = None
    if config.pct_weight:
        with torch.no_grad():
            teacher = source.source_logits(support.x).detach()
            fixed_weight = 1 + config.pct_beta * (teacher.argmax(1) == support.y.to(teacher.device))
    for _ in range(config.steps):
        logits = lowrank_logits(source, geometry, support.x, weights)
        loss = balanced_bce(logits, support.y)
        loss = loss + config.spatial_l2 * (weights.u @ weights.v.T).square().sum()
        loss = loss + config.head_l2 * (weights.delta_w.square().sum() + weights.delta_b.square()) / (source.feature_dim + 1)
        if teacher is not None:
            loss = loss + config.pct_weight * (fixed_weight * 0.5 * (logits - teacher).square().sum(1)).mean()
        # A single autograd call against the same old forward gives genuinely
        # simultaneous factor/head updates; no factor sees an already new head.
        gu, gv, gw, gb = torch.autograd.grad(loss,
            (weights.u, weights.v, weights.delta_w, weights.delta_b), create_graph=create_graph)
        u, v = project_lowrank_update(weights.u - config.factor_lr * gu,
                                      weights.v - config.factor_lr * gv, config.radius)
        weights = LowRankWeights(u, v, weights.delta_w - config.head_lr * gw,
                                 weights.delta_b - config.head_lr * gb)
        if not create_graph:
            weights = LowRankWeights(*(v.detach().requires_grad_(True) for v in
                (weights.u, weights.v, weights.delta_w, weights.delta_b)))
    source.assert_unchanged()
    return weights, config.steps, None


@dataclass(frozen=True)
class LowRankFitResult:
    weights: LowRankWeights
    updates: int
    fallback_reason: str | None
    method: str
    trainable_parameters: int
    support_event_uids: tuple[str, ...]
    acquired_event_uids: tuple[str, ...]
    protocol_freeze_sha: str
    source_state_sha256: str
    geometry_sha256: str
    initial_u_sha256: str
    fitted_weights_sha256: str
    config: LowRankConfig


def fit_lowrank_target(source: FrozenSourceModel, geometry: SpatialGeometry, support: CalibrationPrefix,
                       *, method="B6", config: LowRankConfig | None = None, seed=20261018,
                       u0: Tensor | None = None) -> LowRankFitResult:
    """Support-only target fitting; B10/B11 need their own learned U0 artifact.

    Existence and scientific provenance of a supplied learned U0 are the gated
    runner's responsibility; these math kernels do not declare it meta-trained.
    """
    _precision_guard()
    config = _configuration(config, method)
    if type(support) is not CalibrationPrefix:
        raise TypeError("CalibrationPrefix capability required")
    support.__post_init__()
    if support.role != "calibration_prefix":
        raise PermissionError("public target fit accepts calibration_prefix only")
    if method in META_METHODS:
        if u0 is None:
            raise ValueError("B10/B11 require their own independently learned U0 artifact")
    elif u0 is not None:
        raise ValueError("B6/B7 U0 is fixed by the pipeline seed, not a supplied learned artifact")
    else:
        u0 = initialize_lowrank(config.rank, seed, device=geometry.q.device)
    _validate_u0(geometry, u0, config.rank)
    initial_sha = tensor_sha256(u0)
    weights, updates, fallback = inner_unroll_lowrank(source, geometry, u0, support, config=config)
    frozen = LowRankWeights(*(t.detach().clone() for t in (weights.u, weights.v, weights.delta_w, weights.delta_b)))
    source.assert_unchanged()
    return LowRankFitResult(frozen, updates, fallback, method, 20 * config.rank + source.feature_dim + 1,
        support.event_uids, support.acquired_event_uids, support.protocol_freeze_sha,
        source.frozen_state_sha256, geometry.sha256(), initial_sha, _weights_sha256(frozen), config)


def predict_lowrank(source: FrozenSourceModel, geometry: SpatialGeometry, fitted: LowRankFitResult,
                    inputs: PredictionInputs) -> PredictionResult:
    _precision_guard()
    if type(inputs) is not PredictionInputs or type(fitted) is not LowRankFitResult:
        raise TypeError("label-free PredictionInputs and LowRankFitResult required")
    inputs.__post_init__()
    _configuration(fitted.config, fitted.method)
    if inputs.protocol_freeze_sha != fitted.protocol_freeze_sha:
        raise ValueError("prediction and calibration protocol digests differ")
    if fitted.source_state_sha256 != source.frozen_state_sha256 or fitted.geometry_sha256 != geometry.sha256():
        raise ValueError("prediction source or geometry differs from fit")
    if _weights_sha256(fitted.weights) != fitted.fitted_weights_sha256:
        raise ValueError("fitted low-rank weights changed after the fit receipt")
    if set(inputs.event_uids) & set(fitted.acquired_event_uids):
        raise ValueError("prediction overlaps the acquired calibration prefix")
    with torch.no_grad():
        logits = (source.source_logits(inputs.x) if fitted.fallback_reason else
                  lowrank_logits(source, geometry, inputs.x, fitted.weights))
        probabilities, labels = logits.softmax(1), logits.argmax(1)
    source.assert_unchanged()
    return PredictionResult(logits, probabilities, labels, inputs.event_uids, inputs.protocol_freeze_sha)


def lowrank_episode_objective(source: FrozenSourceModel, geometry: SpatialGeometry, u0: Tensor,
                              episode: MetaTrainEpisode, *, config: LowRankConfig | None = None,
                              method="B10") -> Tensor:
    """Full ten-step U0 -> future-query objective, exclusively meta_train labels."""
    _precision_guard()
    if method not in META_METHODS:
        raise ValueError("only B10/B11 may learn U0 with meta-training labels")
    config = _configuration(config, method)
    if type(episode) is not MetaTrainEpisode:
        raise TypeError("authorized MetaTrainEpisode capability required")
    episode.__post_init__()
    _validate_u0(geometry, u0, config.rank)
    weights, _, fallback = inner_unroll_lowrank(source, geometry, u0, episode.support,
                                              config=config, create_graph=True)
    if fallback:
        return balanced_bce(source.source_logits(episode.query_x), episode.query_y) + u0.sum() * 0
    return balanced_bce(lowrank_logits(source, geometry, episode.query_x, weights), episode.query_y)


def lowrank_outer_step(source: FrozenSourceModel, geometry: SpatialGeometry, u0: Tensor,
                       episodes: tuple[MetaTrainEpisode, ...], *, config: LowRankConfig | None = None,
                       method="B10") -> tuple[Tensor, dict]:
    """One independent 4-episode full-unroll Stiefel U0 step; no optimizer state."""
    _precision_guard()
    if method not in META_METHODS:
        raise ValueError("B6/B7 do not meta-train U0")
    config = _configuration(config, method)
    if len(episodes) != 4 or any(type(e) is not MetaTrainEpisode for e in episodes):
        raise ValueError("one authorized meta-batch contains exactly 4 MetaTrainEpisode objects")
    if u0.dtype != torch.float64 or not u0.requires_grad:
        raise ValueError("meta U0 must be float64 and require gradients")
    _validate_u0(geometry, u0, config.rank)
    for episode in episodes:
        episode.__post_init__()
    if (len({e.support.protocol_freeze_sha for e in episodes}) != 1
            or len({e.split_manifest_sha256 for e in episodes}) != 1):
        raise ValueError("mixed input or split versions in one low-rank meta-batch")
    loss = torch.stack([lowrank_episode_objective(source, geometry, u0, e, config=config, method=method)
                        for e in episodes]).mean()
    gradient, = torch.autograd.grad(loss, u0)
    next_u = stiefel_step(u0, gradient, learning_rate=0.01, max_norm=1.0).detach().requires_grad_(True)
    source.assert_unchanged()
    return next_u, {"method": method, "objective": float(loss.detach()),
        "euclidean_gradient_norm": float(gradient.detach().norm()), "episodes": 4, "inner_steps": 10,
        "full_unroll": True, "outer_learning_rate": 0.01, "gradient_norm_cap": 1.0,
        "outer_parameter": "U0_only", "outer_parameter_elements": u0.numel(),
        "v0_is_zero": True, "source_state_sha256": source.frozen_state_sha256,
        "initial_u_sha256": tensor_sha256(u0), "next_u_sha256": tensor_sha256(next_u)}
