"""Authorized development-only full-unroll objective and ONE pure outer step.

No data loader, dataset download, GPU scheduling, or 1000-update trainer is
included. A later stage-gated runner must enforce actual manifests, sampler,
checkpoint/validation cadence, RNG state, and resource budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import torch
from torch import Tensor

from ..adapt import CalibrationPrefix, FrozenSourceModel, InnerConfig, adapted_logits, balanced_bce, inner_unroll
from ..spatial_geometry import SpatialGeometry, stiefel_step


@dataclass(frozen=True)
class MetaTrainEpisode:
    support: CalibrationPrefix
    query_x: Tensor
    query_y: Tensor
    query_event_uids: tuple[str, ...]
    split_manifest_sha256: str
    temporal_audit_sha256: str
    dataset: str = "Stieger2021"
    role: str = "meta_train"

    def __post_init__(self):
        if self.role != "meta_train" or self.dataset != "Stieger2021":
            raise PermissionError("meta-gradient labels are limited to authorized Stieger meta_train")
        if type(self.support) is not CalibrationPrefix or self.support.role != "meta_train_support":
            raise PermissionError("meta_train_support capability required")
        if len(self.support.acquired_event_uids) != 20:
            raise ValueError("primary meta-training uses the fixed 20-acquired-trial prefix")
        if any(re.fullmatch(r"[0-9a-f]{64}", sha) is None
               for sha in (self.split_manifest_sha256, self.temporal_audit_sha256)):
            raise ValueError("actual split and temporal-audit SHA256 references required")
        if self.query_x.ndim != 3 or self.query_y.shape != (len(self.query_x),):
            raise ValueError("malformed meta query")
        if self.query_x.shape[1:] != self.support.x.shape[1:]:
            raise ValueError("support/query input versions differ")
        if len(self.query_event_uids) != len(self.query_x) or len(set(self.query_event_uids)) != len(self.query_x):
            raise ValueError("one unique meta query UID per original trial required")
        if set(self.query_event_uids) & set(self.support.acquired_event_uids):
            raise ValueError("meta query overlaps the entire acquired support prefix")
        if self.query_y.dtype not in {torch.int32, torch.int64} or not torch.isin(self.query_y, torch.tensor([0, 1], device=self.query_y.device)).all():
            raise ValueError("meta query labels must be integer 0/1")
        if len(self.query_y) < 20 or min(int((self.query_y == c).sum()) for c in (0, 1)) < 5:
            raise ValueError("episode query requires >=20 original trials and >=5/class")
        if not torch.isfinite(self.query_x).all() or torch.any(self.query_x.to(torch.float64).square().sum((-2, -1)) == 0):
            raise ValueError("invalid meta query; upstream QC required")


def episode_objective(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
                      episode: MetaTrainEpisode, *, config=None, family="exp") -> Tensor:
    """Full inner -> future-query loss; never accepts final target query labels."""
    if type(episode) is not MetaTrainEpisode:
        raise TypeError("authorized MetaTrainEpisode capability required")
    episode.__post_init__()
    config = config or InnerConfig()
    config.validate_primary(pct=config.pct_weight == 0.1)
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("v2 first implementation does not use mixed precision")
    weights, _, fallback = inner_unroll(source, geometry, b, episode.support,
                                      config=config, family=family, create_graph=True)
    if fallback:
        # No gradient is invented when support lacks a class: use exact frozen f0.
        logits = source.source_logits(episode.query_x)
        return balanced_bce(logits, episode.query_y) + b.sum() * 0
    logits = adapted_logits(source, geometry, b, episode.query_x, weights, family=family)
    return balanced_bce(logits, episode.query_y)


def outer_step(source: FrozenSourceModel, geometry: SpatialGeometry, b: Tensor,
               episodes: tuple[MetaTrainEpisode, ...], *, config=None, family="exp") -> tuple[Tensor, dict]:
    """One 4-episode meta-batch, lr .01, norm cap 1, no optimizer momentum.

    B8 uses family='additive' and must learn its own B in an independent run;
    this kernel does not reuse P's selected dictionary as an alleged fair B8.
    """
    if len(episodes) != 4:
        raise ValueError("the registered meta-batch has exactly 4 episodes")
    if b.dtype != torch.float64 or not b.requires_grad:
        raise ValueError("B must be float64 and require gradients")
    geometry.validate_dictionary(b)
    if len({e.support.protocol_freeze_sha for e in episodes}) != 1 or len({e.split_manifest_sha256 for e in episodes}) != 1:
        raise ValueError("mixed input or split versions in one meta-batch")
    loss = torch.stack([episode_objective(source, geometry, b, episode, config=config, family=family)
                        for episode in episodes]).mean()
    gradient, = torch.autograd.grad(loss, b)
    next_b = stiefel_step(b, gradient).detach().requires_grad_(True)
    source.assert_unchanged()
    return next_b, {"objective": float(loss.detach()), "euclidean_gradient_norm": float(gradient.norm()),
                    "episodes": 4, "inner_steps": 10, "full_unroll": True}
