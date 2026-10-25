"""Class-balanced support objectives for source-relative risk exploration.

All references are frozen support predictions. The exponential-regret bound
concerns reweightings of these same finite support points, not unseen queries,
physical drift, label-noise immunity, or a statistical confidence interval.
"""
import math

import torch


KINDS = ("ce", "absolute_dro", "relative_dro", "variance_regret")


def class_balanced_weights(y, *, dtype=torch.float64):
    """Give each observed binary class mass1/2, retaining trial multiplicity."""
    if not isinstance(y, torch.Tensor) or y.ndim != 1 or not len(y):
        raise ValueError("Nonempty one-dimensional binary support labels required")
    if not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("Binary support labels0/1 required")
    counts = torch.stack([(y == c).sum() for c in (0, 1)]).to(dtype=dtype)
    if not bool((counts > 0).all()):
        raise ValueError("Both support classes required for balanced risk")
    weights = (.5 / counts)[y.long()]
    return weights / weights.sum()


def pointwise_losses(logits, y):
    """True log-domain binary NLL without a probability clipping ceiling."""
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape != (len(y), 2) or logits.device != y.device):
        raise ValueError("Same-device N x2 logits and binary labels required")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite support logits")
    log_probabilities = logits.double().log_softmax(1)
    return -log_probabilities.gather(1, y.long()[:, None]).squeeze(1)


def _temperature(value, reference):
    temperature = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if temperature.numel() != 1 or not bool(torch.isfinite(temperature).all()) or bool(temperature <= 0):
        raise ValueError("Finite positive risk temperature required")
    return temperature.reshape(())


def _parts(logits, source_logits, y):
    weights = class_balanced_weights(y)
    candidate = pointwise_losses(logits, y)
    if not isinstance(source_logits, torch.Tensor) or source_logits.shape != logits.shape:
        raise ValueError("Frozen support reference logits must match candidate shape")
    reference = pointwise_losses(source_logits.detach().to(device=logits.device), y)
    return candidate, reference, weights.to(candidate)


def entropic_risk(values, weights, temperature):
    """Weighted log-MGF with stable large-temperature and extreme-loss limits.

    Max-centering prevents exponential overflow. Near mass1, expm1/log1p
    prevents multiplying logsumexp's tiny normalization roundoff by a very
    large temperature. The detached center does not alter the exact first or
    higher derivatives, because an additive shift cancels from log-MGF.
    """
    tau = _temperature(temperature, values)
    center = values.max().detach()
    offsets = (values - center) / tau
    changed_mass = (weights * torch.expm1(offsets)).sum()
    if bool(changed_mass.detach() > -.5):
        log_mass = torch.log1p(changed_mass)
    else:
        log_mass = torch.logsumexp(weights.log() + offsets, dim=0)
    return center + tau * log_mass


def support_risk(logits, source_logits, y, kind="relative_dro", temperature=.5):
    """Return a scalar data objective; the caller adds its scientific ridge.

    ce: ordinary class-balanced NLL.
    absolute_dro: entropic absolute NLL (KL-penalized finite reweighting).
    relative_dro: entropic NLL difference against the frozen source.
    variance_regret: mean difference plus its weighted variance/(2*tau).

    This function never accepts query samples or labels. Reference logits are
    detached even when supplied by an outer differentiable computation.
    """
    if kind not in KINDS:
        raise ValueError("Unknown support risk kind")
    candidate, reference, weights = _parts(logits, source_logits, y)
    tau = _temperature(temperature, candidate)
    if kind == "ce":
        return (weights * candidate).sum()
    if kind == "absolute_dro":
        return entropic_risk(candidate, weights, tau)
    regret = candidate - reference
    if kind == "relative_dro":
        return entropic_risk(regret, weights, tau)
    mean = (weights * regret).sum()
    variance = (weights * (regret - mean).square()).sum()
    return mean + variance / (2 * tau)


def risk_diagnostics(logits, source_logits, y, *, temperature=.5, ridge=0., kl_radius=0.):
    """JSON-safe audit; a positive risk is recorded, never a startup gate.

    Donsker--Varadhan gives E_p[regret] <= R_tau + tau*KL(p||q).
    If R_tau + ridge <=0, the sharper <= tau*KL(p||q)-ridge follows.
    ``earned_kl_radius`` remains signed: a negative value is not certified.
    """
    with torch.no_grad():
        candidate, reference, weights = _parts(logits, source_logits, y)
        tau = _temperature(temperature, candidate)
        penalty = float(torch.as_tensor(ridge, dtype=torch.float64).detach())
        radius = float(torch.as_tensor(kl_radius, dtype=torch.float64).detach())
        if not math.isfinite(penalty) or penalty < 0 or not math.isfinite(radius) or radius < 0:
            raise ValueError("Finite nonnegative ridge and requested KL radius required")
        regret = candidate - reference
        mean = (weights * regret).sum()
        variance = (weights * (regret - mean).square()).sum()
        robust = entropic_risk(regret, weights, tau)
        log_adversary = weights.log() + (regret - regret.max()) / tau
        adversary = log_adversary.softmax(0)
        robust_value, tau_value = float(robust), float(tau)
        total = robust_value + penalty
        certified = total <= 0.
        earned = -robust_value / tau_value
        return {
            "risk_scope": "same_finite_labeled_support_reweighting_only",
            "support_only": True,
            "future_risk_guarantee": False,
            "label_noise_robustness_guarantee": False,
            "source_reference_detached": True,
            "source_relative_entropic_risk": robust_value,
            "weighted_mean_regret": float(mean),
            "weighted_regret_variance": float(variance),
            "log_exponential_moment": robust_value / tau_value,
            "temperature": tau_value,
            "ridge": penalty,
            "regularized_relative_objective": total,
            "support_certificate_holds": certified,
            "earned_kl_radius": earned,
            "earned_kl_radius_nonnegative": earned >= 0.,
            "requested_kl_radius": radius,
            "variational_regret_bound_at_requested_kl": robust_value + tau_value * radius,
            "certificate_bound_if_nonpositive": tau_value * radius - penalty if certified else None,
            "adversarial_max_weight": float(adversary.max()),
            "adversarial_effective_sample_size": float(adversary.square().sum().reciprocal()),
            "adversarial_class0_mass": float(adversary[y == 0].sum()),
            "adversarial_class1_mass": float(adversary[y == 1].sum()),
            "numerical_certificate_tolerance": 0.,
        }
