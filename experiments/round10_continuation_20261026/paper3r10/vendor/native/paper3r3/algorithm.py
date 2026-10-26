"""Support-only, nuisance-aware Lie calibration (independent round-three learner).

The primary learner optimizes a fixed identity-tangent logistic surrogate and
uses the actual exponential transform for prediction. It does not claim to
optimize the nonlinear inner problem exactly. Only first outer derivatives are
supported through the constrained quadratic solve.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from torch.nn import functional as F
from p3work.adapt import FastWeights, zero_fast_weights, balanced_bce
from p3work.spatial_geometry import joint_trial_rms, project_ball


VARIANTS = ("sgd10", "sgd100", "head_sgd", "head_newton", "tangent_diag",
            "tangent_joint", "tangent_profiled", "tangent_information",
            "tangent_protected", "information_protected", "relinearized")


@dataclass(frozen=True)
class Config:
    variant: str = "tangent_profiled"
    rank: int = 4
    radius: float = .15
    tangent_steps: int = 10
    theta_l2: float = .01
    head_l2: float = .001
    information_scale: float = 1.
    theta_lr: float = .05
    head_lr: float = .01
    damping: float = 1e-4
    kl_epsilon: float = .01
    line_search_steps: int = 16

    def __post_init__(self):
        if self.variant not in VARIANTS or self.rank not in (2, 4, 8):
            raise ValueError("Unknown round-three variant or dictionary rank")
        if self.radius <= 0 or self.theta_l2 <= 0 or self.head_l2 <= 0:
            raise ValueError("Positive radius and both prior penalties required")
        if self.tangent_steps < 1 or self.line_search_steps < 1:
            raise ValueError("Positive finite iteration budgets required")
        if self.damping < 0 or self.kl_epsilon <= 0:
            raise ValueError("Nonnegative damping and positive KL budget required")
        if not math.isfinite(self.information_scale) or self.information_scale <= 0:
            raise ValueError("Positive finite information scale required")


@dataclass
class Fitted:
    weights: FastWeights
    diagnostics: dict = field(default_factory=dict)
    fallback_reason: str | None = None
    protection_fraction: float = 1.


def _assert_frozen(source):
    check = getattr(source, "assert_unchanged", None)
    if check is not None:
        check()
    elif any(m.training for m in source.modules()) or any(p.requires_grad for p in source.parameters()):
        raise RuntimeError("Source parameters, dropout and BN must remain frozen in eval mode")


def _class_weights(y):
    counts = torch.stack([(y == c).sum() for c in (0, 1)]).to(dtype=torch.float64)
    return (.5 / counts)[y]


def _augmented(features):
    f = features.double()
    return torch.cat((f, torch.ones((len(f), 1), device=f.device, dtype=f.dtype)), 1)


def support_tangent(source, geometry, support):
    """Return detached (source margin, 99-D Lie derivative, frozen features).

    The derivative includes the CAR projection, recoloring and joint-trial RMS.
    For an eval-mode encoder with independent trials, a gradient of the sum of
    margins yields every trial's input derivative in a single reverse pass.
    No source parameters or BN statistics are changed. B is absent, so these
    detached quantities give the exact derivative of the *fixed* surrogate.
    """
    _assert_frozen(source)
    with torch.enable_grad():
        x = support.x.double()
        left = geometry.q.T @ geometry.rref_sqrt
        z = geometry.rref_invsqrt @ geometry.q @ x
        y0 = left @ z
        rms = y0.square().mean((-2, -1), keepdim=True).sqrt()
        if len(x) and bool((rms == 0).any()):
            raise ValueError("Zero-energy support trial after CAR")
        normalized = (y0 / rms).detach().requires_grad_(True)
        features, base = source.features_and_logits(normalized)
        margin = base[:, 1] - base[:, 0]
        if not len(x):
            return margin.detach().double(), x.new_empty((0, geometry.e.shape[1])), features.detach()
        grad = torch.autograd.grad(margin.sum(), normalized, create_graph=False)[0]
        tangent_grad = (grad - y0 * (grad * y0).sum((-2, -1), keepdim=True)
                        / y0.square().sum((-2, -1), keepdim=True)) / rms
        # G_i = L^T v_i z_i^T; <G_i, E_k> is row-major basis derivative.
        generator_gradient = torch.einsum("ca,nct,nbt->nab", left, tangent_grad, z)
        coordinates = generator_gradient.flatten(1) @ geometry.e
    return margin.detach().double(), coordinates.detach(), features.detach()


class _BallQuadratic(torch.autograd.Function):
    """argmin 1/2 x^T H x - b^T x, ||x|| <= radius, H symmetric SPD.

    The active boundary backward differentiates the coupled KKT equations,
    including the radius. Differentiating a detached bisection multiplier as a
    constant would give an incorrect first meta-gradient on the boundary.
    """
    @staticmethod
    def forward(ctx, hessian, rhs, radius):
        h = (hessian + hessian.T) / 2
        # Only the rank-2/4/8 system crosses to CPU. Keeping the scalar root
        # there avoids 80 CUDA solve/norm/host-sync cycles per Newton step.
        # Calibration timing must include these small transfers. The backward
        # below still differentiates the exact KKT equations on input device.
        h_cpu = h.detach().to(device="cpu", dtype=torch.float64).numpy()
        rhs_cpu = rhs.detach().to(device="cpu", dtype=torch.float64).numpy()
        radius_cpu = float(radius.detach().cpu())
        values, vectors = np.linalg.eigh(h_cpu)
        if not np.isfinite(values).all() or not np.all(values > 0):
            raise FloatingPointError("Quadratic-ball Hessian must be finite positive definite")
        projected = vectors.T @ rhs_cpu
        x_cpu = vectors @ (projected / values)
        active = bool(np.linalg.norm(x_cpu) > radius_cpu)
        multiplier_cpu = 0.
        if active:
            # Since lambda_min(H)>0, nu=||rhs||/radius is a valid upper bound.
            lower, upper = 0., float(np.linalg.norm(rhs_cpu) / radius_cpu)
            for _ in range(80):
                midpoint = (lower + upper) / 2
                if np.linalg.norm(projected / (values + midpoint)) > radius_cpu:
                    lower = midpoint
                else:
                    upper = midpoint
            multiplier_cpu = upper
            x_cpu = vectors @ (projected / (values + multiplier_cpu))
        x = torch.from_numpy(x_cpu.copy()).to(device=h.device, dtype=h.dtype)
        multiplier = h.new_tensor(multiplier_cpu)
        ctx.active = active
        ctx.save_for_backward(h, x, radius, multiplier)
        return x

    @staticmethod
    def backward(ctx, output_gradient):
        h, x, radius, multiplier = ctx.saved_tensors
        a = h + multiplier * torch.eye(len(x), device=h.device, dtype=h.dtype)
        u = torch.linalg.solve(a, output_gradient)
        radius_gradient = radius.new_zeros(())
        if ctx.active:
            normal = torch.linalg.solve(a, x)
            dual = torch.dot(x, u) / torch.dot(x, normal)
            u = u - normal * dual
            radius_gradient = radius * dual
        gh = -(torch.outer(u, x) + torch.outer(x, u)) / 2
        return gh, u, radius_gradient


def ball_quadratic(hessian, rhs, radius):
    """Differentiable constrained SPD quadratic; radius may be a tensor."""
    radius = torch.as_tensor(radius, dtype=hessian.dtype, device=hessian.device)
    if bool(radius <= 0):
        # A zero observable radius means no spatial adaptation.
        return rhs * 0 + radius * 0
    return _BallQuadratic.apply(hessian, rhs, radius)


def _objective(margin, y, theta, head, config):
    loss = (_class_weights(y) * F.binary_cross_entropy_with_logits(
        margin, y.to(margin), reduction="none")).sum()
    return (loss + config.theta_l2 * theta.square().sum()
            + config.head_l2 * head.square().sum() / len(head))


def _blocks(j, z, margin, y, theta, head, config):
    probability = margin.sigmoid()
    a = _class_weights(y)
    residual = a * (probability - y.to(margin))
    curvature = a * probability * (1 - probability)
    lt, lh = 2 * config.theta_l2, 2 * config.head_l2 / len(head)
    gt = j.T @ residual + lt * theta
    gh = z.T @ residual + lh * head
    htt = j.T @ (curvature[:, None] * j)
    hth = j.T @ (curvature[:, None] * z)
    hhh = z.T @ (curvature[:, None] * z)
    eye_t = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
    eye_h = torch.eye(z.shape[1], device=z.device, dtype=z.dtype)
    return gt, gh, htt + (lt + config.damping) * eye_t, hth, hhh + (lh + config.damping) * eye_h


def information(j, z, margin, y, config):
    """Penalized-head data Schur information, excluding spatial prior/damping."""
    a = _class_weights(y)
    p = margin.sigmoid()
    d = a * p * (1 - p)
    raw = j.T @ (d[:, None] * j)
    cross = j.T @ (d[:, None] * z)
    head = z.T @ (d[:, None] * z)
    head = head + (2 * config.head_l2 / z.shape[1]) * torch.eye(
        z.shape[1], device=z.device, dtype=z.dtype)
    schur = raw - cross @ torch.linalg.solve(head, cross.T)
    schur = (schur + schur.T) / 2
    eye = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
    effective = torch.trace(torch.linalg.solve(
        schur + 2 * config.theta_l2 * config.information_scale * eye, schur))
    effective = effective.clamp(0, j.shape[1])
    return schur, effective


def _propose(j, z, margin, y, theta, head, radius, config, *, mode):
    gt, gh, htt, hth, hhh = _blocks(j, z, margin, y, theta, head, config)
    if mode == "diag":
        return project_ball(theta - gt / htt.diagonal(), float(radius.detach())), head - gh / hhh.diagonal()
    if mode == "joint":
        full = torch.cat((torch.cat((htt, hth), 1), torch.cat((hth.T, hhh), 1)), 0)
        delta = -torch.linalg.solve(full, torch.cat((gt, gh)))
        # Deliberate naive clipping control; head does not account for clipping.
        return project_ball(theta + delta[:len(theta)], float(radius.detach())), head + delta[len(theta):]
    inverse_cross = torch.linalg.solve(hhh, hth.T)
    schur = htt - hth @ inverse_cross
    schur = (schur + schur.T) / 2
    reduced_gradient = gt - hth @ torch.linalg.solve(hhh, gh)
    next_theta = ball_quadratic(schur, schur @ theta - reduced_gradient, radius)
    delta_theta = next_theta - theta
    next_head = head - torch.linalg.solve(hhh, gh + hth.T @ delta_theta)
    return next_theta, next_head


def _surrogate_fit(m0, coordinates, features, b, y, config):
    j, z = coordinates @ b, _augmented(features)
    theta = b.new_zeros(b.shape[1])
    head = b.new_zeros(z.shape[1])
    schur_data, effective = information(j, z, m0, y, config)
    radius = b.new_tensor(config.radius)
    if config.variant in ("tangent_information", "information_protected"):
        # The tiny floor only regularizes sqrt's derivative at exactly zero.
        ratio = effective / b.shape[1]
        radius = radius * torch.where(ratio > 0, ratio.clamp_min(1e-24).sqrt(), ratio * 0)
    mode = {"tangent_diag": "diag", "tangent_joint": "joint"}.get(config.variant, "profiled")
    fractions = []
    for _ in range(config.tangent_steps):
        margin = m0 + j @ theta + z @ head
        candidate_theta, candidate_head = _propose(j, z, margin, y, theta, head, radius, config, mode=mode)
        before = _objective(margin, y, theta, head, config)
        fraction = 0.
        for power in range(config.line_search_steps):
            trial_fraction = 2. ** -power
            tt = theta + trial_fraction * (candidate_theta - theta)
            hh = head + trial_fraction * (candidate_head - head)
            value = _objective(m0 + j @ tt + z @ hh, y, tt, hh, config)
            if bool(value.detach() <= before.detach() + 1e-12):
                theta, head, fraction = tt, hh, trial_fraction
                break
        fractions.append(fraction)
    diagnostics = {"surrogate": "fixed_identity_tangent_balanced_logistic",
        "optimizer_updates": config.tangent_steps, "surrogate_line_search_fractions": fractions,
        "effective_dimension": float(effective.detach()), "effective_radius": float(radius.detach()),
        "schur_data_eigenvalues": torch.linalg.eigvalsh(schur_data.detach()).tolist(),
        "surrogate_loss": float(_objective(m0 + j @ theta + z @ head, y, theta, head, config).detach())}
    return FastWeights(theta, head[:-1], head[-1]), diagnostics


def _actual(source, geometry, b, weights, x):
    m = (geometry.e @ b @ weights.theta).reshape(10, 10)
    a = geometry.q.T @ geometry.rref_sqrt @ torch.matrix_exp(m) @ geometry.rref_invsqrt @ geometry.q
    features, base = source.features_and_logits(joint_trial_rms(a @ x.double()))
    features = features.to(weights.delta_w.dtype)
    base = base.to(weights.delta_w.dtype)
    delta = features @ weights.delta_w + weights.delta_b
    return base + torch.stack((-delta / 2, delta / 2), 1), features


def _sgd_fit(source, geometry, b, support, config, create_graph):
    weights = zero_fast_weights(source, b.shape[1], device=b.device)
    count = 100 if config.variant == "sgd100" else 10
    head_only = config.variant == "head_sgd"
    for _ in range(count):
        pred, _ = _actual(source, geometry, b, weights, support.x)
        loss = balanced_bce(pred, support.y) + config.theta_l2 * weights.theta.square().sum()
        loss = loss + config.head_l2 * (weights.delta_w.square().sum() + weights.delta_b.square()) / (source.feature_dim + 1)
        parameters = (weights.delta_w, weights.delta_b) if head_only else (
            weights.theta, weights.delta_w, weights.delta_b)
        grad = torch.autograd.grad(loss, parameters, create_graph=create_graph)
        if head_only:
            theta, gw, gb = weights.theta, *grad
        else:
            gt, gw, gb = grad
            theta = project_ball(weights.theta - config.theta_lr * gt, config.radius)
        weights = FastWeights(theta, weights.delta_w - config.head_lr * gw, weights.delta_b - config.head_lr * gb)
        if not create_graph:
            weights = FastWeights(*(v.detach().requires_grad_(True) for v in
                (weights.theta, weights.delta_w, weights.delta_b)))
    return weights, {"optimizer_updates": count, "surrogate": "actual_nonlinear_model"}


def _head_newton(m0, features, b, y, config):
    z = _augmented(features)
    head = b.new_zeros(z.shape[1])
    theta = b.new_zeros(b.shape[1])
    for _ in range(config.tangent_steps):
        margin = m0 + z @ head
        p = margin.sigmoid(); a = _class_weights(y)
        precision = 2 * config.head_l2 / len(head)
        gradient = z.T @ (a * (p - y.to(p))) + precision * head
        curvature = z.T @ ((a * p * (1 - p))[:, None] * z)
        curvature = curvature + (precision + config.damping) * torch.eye(len(head), dtype=head.dtype, device=head.device)
        direction = -torch.linalg.solve(curvature, gradient)
        before = _objective(margin, y, theta, head, config)
        for power in range(config.line_search_steps):
            trial = head + (2. ** -power) * direction
            value = _objective(m0 + z @ trial, y, theta, trial, config)
            if bool(value.detach() <= before.detach() + 1e-12):
                head = trial
                break
    return FastWeights(theta, head[:-1], head[-1]), {
        "optimizer_updates": config.tangent_steps, "surrogate": "exact_head_logistic"}


def _relinearized(source, geometry, b, support, config):
    weights = FastWeights(b.new_zeros(b.shape[1], requires_grad=True),
        b.new_zeros(source.feature_dim, requires_grad=True), b.new_zeros((), requires_grad=True))
    accepted = []
    for _ in range(config.tangent_steps):
        pred, features = _actual(source, geometry, b, weights, support.x)
        margin = pred[:, 1] - pred[:, 0]
        j = torch.stack([torch.autograd.grad(margin[i], weights.theta, retain_graph=True)[0]
                         for i in range(len(margin))]).detach()
        head = torch.cat((weights.delta_w, weights.delta_b[None]))
        z = _augmented(features).detach()
        tt, hh = _propose(j, z, margin.detach(), support.y, weights.theta, head,
                         b.new_tensor(config.radius), config, mode="profiled")
        before = _objective(margin, support.y, weights.theta, head, config)
        fraction = 0.
        for power in range(config.line_search_steps):
            alpha = 2. ** -power
            trial_t = weights.theta + alpha * (tt - weights.theta)
            trial_h = head + alpha * (hh - head)
            trial = FastWeights(trial_t, trial_h[:-1], trial_h[-1])
            p, _ = _actual(source, geometry, b, trial, support.x)
            value = _objective(p[:, 1] - p[:, 0], support.y, trial_t, trial_h, config)
            if bool(value.detach() <= before.detach() + 1e-12):
                weights, fraction = trial, alpha
                break
        weights = FastWeights(*(v.detach().requires_grad_(True) for v in
            (weights.theta, weights.delta_w, weights.delta_b)))
        accepted.append(fraction)
    return weights, {"optimizer_updates": config.tangent_steps,
        "surrogate": "relinearized_nonlinear_GGN_posthoc", "nonlinear_line_search_fractions": accepted}


def _bernoulli_kl(reference_margin, candidate_margin):
    probability = reference_margin.sigmoid()
    return (probability * (F.logsigmoid(reference_margin) - F.logsigmoid(candidate_margin))
        + (1 - probability) * (F.logsigmoid(-reference_margin) - F.logsigmoid(-candidate_margin)))


def _protect(source, geometry, b, weights, support, config, baseline):
    if baseline is None:
        with torch.no_grad():
            baseline = source.source_logits(support.x)
    if baseline.shape == (len(support.y), 2):
        reference = (baseline[:, 1] - baseline[:, 0]).detach().double()
    elif baseline.shape == (len(support.y),):
        reference = baseline.detach().double()
    else:
        raise ValueError("Protected baseline must contain support-only margins or two logits")
    # Shrink the entire fitted update from identity, not the last Newton step.
    # The selected discrete fraction is constant in the local outer derivative.
    for power in range(25):
        fraction = 2. ** -power
        trial = FastWeights(*(v * fraction for v in (weights.theta, weights.delta_w, weights.delta_b)))
        pred, _ = _actual(source, geometry, b, trial, support.x)
        kl = (_class_weights(support.y) * _bernoulli_kl(reference, (pred[:, 1] - pred[:, 0]).double())).sum()
        if bool(kl.detach() <= config.kl_epsilon):
            return trial, fraction, float(kl.detach())
    trial = FastWeights(*(v * 0 for v in (weights.theta, weights.delta_w, weights.delta_b)))
    pred, _ = _actual(source, geometry, b, trial, support.x)
    kl = (_class_weights(support.y) * _bernoulli_kl(reference, (pred[:, 1] - pred[:, 0]).double())).sum()
    if bool(kl.detach() > config.kl_epsilon + 1e-10):
        raise ValueError("Supplied protection baseline disagrees with identity source; zero update infeasible")
    return trial, 0., float(kl.detach())


def fit(source, geometry, b, support, config, *, create_graph=False, protected_baseline=None):
    """Fit only the acquired support. Query tensors and labels are not accepted."""
    if b.shape != (99, config.rank):
        raise ValueError("B must match the full 99-dimensional Lie basis and configured rank")
    if config.variant == "relinearized" and create_graph:
        raise ValueError("relinearized is an explicit posthoc reference, not an outer-trained learner")
    _assert_frozen(source)
    if not all(bool((support.y == c).any()) for c in (0, 1)):
        return Fitted(zero_fast_weights(source, b.shape[1], device=b.device),
            {"optimizer_updates": 0}, "insufficient_class_support")
    with torch.enable_grad():
        if config.variant in ("sgd10", "sgd100", "head_sgd"):
            weights, diagnostics = _sgd_fit(source, geometry, b, support, config, create_graph)
        elif config.variant == "relinearized":
            weights, diagnostics = _relinearized(source, geometry, b, support, config)
        else:
            m0, coordinates, features = support_tangent(source, geometry, support)
            if config.variant == "head_newton":
                weights, diagnostics = _head_newton(m0, features, b, support.y, config)
            else:
                weights, diagnostics = _surrogate_fit(m0, coordinates, features, b, support.y, config)
        fraction = 1.
        if config.variant in ("tangent_protected", "information_protected"):
            weights, fraction, kl = _protect(source, geometry, b, weights, support, config, protected_baseline)
            diagnostics.update(protection_fraction=fraction, actual_source_support_kl=kl)
        diagnostics.update(theta_norm=float(weights.theta.detach().norm()), variant=config.variant)
        if not all(bool(torch.isfinite(v).all()) for v in (weights.theta, weights.delta_w, weights.delta_b)):
            raise FloatingPointError("Nonfinite adaptation state")
        if not create_graph:
            weights = FastWeights(*(v.detach() for v in (weights.theta, weights.delta_w, weights.delta_b)))
    _assert_frozen(source)
    return Fitted(weights, diagnostics, protection_fraction=fraction)


def predict(source, geometry, b, fitted, query_x):
    """Actual exponential-transform prediction; no query labels or updates."""
    if fitted.fallback_reason:
        return source.source_logits(query_x)
    return _actual(source, geometry, b, fitted.weights, query_x)[0]
