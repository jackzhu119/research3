"""Three hypotheses: support-conditioned geometry, temporal risk, soft teacher trust.

This module consumes only tensors. Fit never accepts target query data. Geometry
and SGD constants match Paper3 at gates=(1,1). Evidence is detached, support-only.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from p3work.adapt import FastWeights, balanced_bce, zero_fast_weights
from p3work.spatial_geometry import project_ball, joint_trial_rms, stiefel_step

VARIANTS = ("P", "global", "scalar", "dual", "PCT", "softPCT", "temporal", "dual_temporal")
FEATURES = ("covariance_shift", "split_covariance_disagreement", "source_balanced_loss",
            "source_entropy", "class_imbalance", "valid_fraction")

@dataclass(frozen=True)
class Config:
    variant: str = "dual"
    rank: int = 4
    radius: float = .15
    inner_steps: int = 10
    theta_lr: float = .05
    head_lr: float = .01
    theta_l2: float = .01
    head_l2: float = .001
    outer_lr: float = .01
    temporal_weight: float = .25
    temporal_temperature: float = .5

    def __post_init__(self):
        if self.variant not in VARIANTS or self.rank not in (2, 4, 8):
            raise ValueError("Unknown method or rank")
        if self.radius <= 0 or self.temporal_temperature <= 0:
            raise ValueError("Positive radius and temperature required")

def init_gate(config, device):
    if config.variant == "global":
        shape = (2, 1)
    elif config.variant == "scalar":
        shape = (1, 7)
    elif config.variant in ("dual", "dual_temporal"):
        shape = (2, 7)
    else:
        shape = (0, 0)
    gate = torch.zeros(shape, dtype=torch.float64, device=device)
    if gate.numel():
        gate[:, -1] = math.log(9.)  # both gates initially .9, never selected from holdout
    return gate.requires_grad_(True)

def log_spd(matrix):
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * values.clamp_min(torch.finfo(values.dtype).eps).log()) @ vectors.T

@torch.no_grad()
def support_evidence(source, geometry, support):
    """Six bounded features. Disagreement is a proxy, NOT calibrated uncertainty."""
    x, y = support.x, support.y
    if not len(y) or not all(bool((y == c).any()) for c in (0, 1)):
        return torch.zeros(6, dtype=torch.float64, device=x.device)
    z = geometry.q @ x.double()
    cov = z @ z.transpose(-2, -1)
    cov = cov / cov.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    eye = torch.eye(10, dtype=torch.float64, device=x.device) / 10
    def mean_cov(ids):
        return .95 * cov[ids].mean(0) + .05 * eye
    def white_log(c):
        return log_spd(geometry.rref_invsqrt @ c @ geometry.rref_invsqrt)
    # Interleave within class: avoids confounding the two halves by class order.
    left, right = [], []
    for c in (0, 1):
        ids = (y == c).nonzero().flatten().tolist()
        left.extend(ids[::2]); right.extend(ids[1::2])
    shift = white_log(mean_cov(list(range(len(y))))).norm() / math.sqrt(10)
    disagreement = ((white_log(mean_cov(left)) - white_log(mean_cov(right))).norm()
                    / math.sqrt(10)) if left and right else shift.new_tensor(1.)
    logits = source.source_logits(x)
    probability = logits.double().softmax(1).clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(1).mean() / math.log(2)
    loss = balanced_bce(logits, y).double()
    # Fixed monotone scaling, no validation-fitted standardization.
    return torch.stack((shift / (1 + shift), disagreement / (1 + disagreement),
        loss / (1 + loss), entropy, abs((y == 0).double().mean() - .5) * 2,
        shift.new_tensor(len(y) / max(1, len(support.acquired_event_uids))))).detach()

def coefficients(gate, evidence):
    if not gate.numel():
        return evidence.new_ones(2)
    inputs = evidence.new_ones(1) if gate.shape[1] == 1 else torch.cat((evidence, evidence.new_ones(1)))
    alpha = torch.sigmoid(gate @ inputs)
    return alpha.expand(2) if len(alpha) == 1 else alpha

def gated_matrix(geometry, b, theta, alpha):
    generators = (geometry.e @ b).T.reshape(b.shape[1], 10, 10)
    m = torch.einsum("r,rij->ij", theta, generators)
    # This decomposition is invariant to changing the dictionary basis B -> BO.
    symmetric, skew = (m + m.T) / 2, (m - m.T) / 2
    return alpha[0] * symmetric + alpha[1] * skew

def logits(source, geometry, b, weights, alpha, x):
    m = gated_matrix(geometry, b, weights.theta, alpha)
    a = geometry.q.T @ geometry.rref_sqrt @ torch.matrix_exp(m) @ geometry.rref_invsqrt @ geometry.q
    features, base = source.features_and_logits(joint_trial_rms(a @ x.double()))
    delta = features @ weights.delta_w + weights.delta_b
    return base + torch.stack((-delta / 2, delta / 2), dim=1)

@torch.no_grad()
def teacher_weights(teacher, y, *, soft):
    correct = (teacher.argmax(1) == y).double()
    if not soft:
        return 1 + 5 * correct
    q = teacher.double().softmax(1).clamp_min(1e-12)
    entropy = -(q * q.log()).sum(1) / math.log(2)
    reliability = torch.empty_like(correct)
    for c in (0, 1):
        ids = y == c
        n = ids.sum()
        # Beta(1,1) posterior mean, leaving this trial's correctness out.
        reliability[ids] = (1 + correct[ids].sum() - correct[ids]) / (n + 1)
    true_probability = q.gather(1, y[:, None]).squeeze(1)
    return 1 + 5 * reliability * true_probability * (1 - entropy).clamp(0, 1)

def fit(source, geometry, b, gate, support, config, *, create_graph=False):
    weights = zero_fast_weights(source, b.shape[1], device=b.device)
    evidence = support_evidence(source, geometry, support)
    alpha = coefficients(gate, evidence)
    if not all(bool((support.y == c).any()) for c in (0, 1)):
        return weights, alpha, "insufficient_class_support"
    teacher = None
    if config.variant in ("PCT", "softPCT"):
        with torch.no_grad():
            teacher = source.source_logits(support.x).detach()
            trust = teacher_weights(teacher, support.y, soft=config.variant == "softPCT")
    for _ in range(config.inner_steps):
        pred = logits(source, geometry, b, weights, alpha, support.x)
        loss = balanced_bce(pred, support.y) + config.theta_l2 * weights.theta.square().sum()
        loss = loss + config.head_l2 * (weights.delta_w.square().sum() + weights.delta_b.square()) / (source.feature_dim + 1)
        if teacher is not None:
            # Preserve original PCT precision when reproducing P+PCT.
            w = trust.to(pred.dtype) if config.variant == "PCT" else trust
            loss = loss + .1 * (w * .5 * (pred - teacher).square().sum(1)).mean()
        gt, gw, gb = torch.autograd.grad(loss, (weights.theta, weights.delta_w, weights.delta_b),
                                       create_graph=create_graph, retain_graph=create_graph)
        weights = FastWeights(project_ball(weights.theta - config.theta_lr * gt, config.radius),
            weights.delta_w - config.head_lr * gw, weights.delta_b - config.head_lr * gb)
        if not create_graph:
            weights = FastWeights(*(p.detach().requires_grad_(True) for p in
                                    (weights.theta, weights.delta_w, weights.delta_b)))
            alpha = alpha.detach()
    return weights, alpha, None

def temporal_loss(pred, y, run_ids, config):
    mean = balanced_bce(pred, y)
    # Equal physical-run groups; no manufactured temporal windows.
    per_run = []
    for rid in sorted(set(run_ids)):
        mask = torch.tensor([r == rid for r in run_ids], device=y.device)
        if all(bool((y[mask] == c).any()) for c in (0, 1)):
            per_run.append(balanced_bce(pred[mask], y[mask]))
    if len(per_run) < 2:
        return mean  # reported in dataset coverage; keep the episode
    losses = torch.stack(per_run)
    tau = config.temporal_temperature
    tilted = tau * (torch.logsumexp(losses / tau, 0) - math.log(len(losses)))
    return (1 - config.temporal_weight) * mean + config.temporal_weight * tilted

def outer_update(source, geometry, b, gate, episodes, config):
    """Sequential accumulation releases each full inner graph; batch still exactly 4."""
    grad_b, grad_gate, objective = torch.zeros_like(b), torch.zeros_like(gate), 0.
    alphas = []
    for e in episodes:
        weights, alpha, fallback = fit(source, geometry, b, gate, e.support, config, create_graph=True)
        alphas.append(alpha.detach())
        pred = source.source_logits(e.query_x) if fallback else logits(source, geometry, b, weights, alpha, e.query_x)
        loss = (temporal_loss(pred, e.query_y, e.run_ids, config)
                if config.variant in ("temporal", "dual_temporal") else balanced_bce(pred, e.query_y))
        loss = loss + b.sum() * 0 + gate.sum() * 0
        gb, gg = torch.autograd.grad(loss / len(episodes), (b, gate))
        grad_b += gb; grad_gate += gg
        objective += float(loss.detach()) / len(episodes)
    if not torch.isfinite(grad_b).all() or not torch.isfinite(grad_gate).all():
        raise FloatingPointError("Non-finite meta-gradient; resume last committed step")
    next_b = stiefel_step(b, grad_b, learning_rate=config.outer_lr).detach().requires_grad_(True)
    next_gate = (gate - config.outer_lr * grad_gate / grad_gate.norm().clamp_min(1)).detach().requires_grad_(True)
    return next_b, next_gate, {"objective": objective, "gate_mean": torch.stack(alphas).mean(0).tolist(),
                              "gate_gradient_norm": float(grad_gate.norm().detach())}
