"""Support-only covariance transport with stable first-order SPD derivatives.

The frozen source reference and per-trial unit-trace CAR covariances match the
preceding support-EA control. No target query is accepted by the fitting API.
The Bures map is the classical minimum-displacement covariance map, not a new
optimal-transport theorem. Partial/conditional transport are testable learners.
"""
from dataclasses import dataclass
import math
import torch
from torch.autograd.function import once_differentiable


def _sym(a):
    return (a + a.transpose(-2, -1)) / 2


class _SPDRoot(torch.autograd.Function):
    """Frechet root derivative, finite even for repeated eigenvalues.

    The backward solves S X + X S = dA in the eigenbasis of S=A**.5.
    Only first derivatives are supported/needed for the direct outer learner.
    Differentiating eigenvectors with a raw eigh backward is intentionally
    avoided: eigenvector derivatives need not exist at repeated eigenvalues.
    """
    @staticmethod
    def forward(ctx, matrix, inverse):
        values, vectors = torch.linalg.eigh(_sym(matrix))
        if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
            raise FloatingPointError("Covariance root requires a positive definite matrix")
        roots = values.sqrt()
        ctx.save_for_backward(roots, vectors)
        ctx.inverse = inverse
        diagonal = roots.reciprocal() if inverse else roots
        return _sym((vectors * diagonal.unsqueeze(-2)) @ vectors.transpose(-2, -1))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        roots, vectors = ctx.saved_tensors
        denominator = roots.unsqueeze(-1) + roots.unsqueeze(-2)
        coefficient = denominator.reciprocal()
        if ctx.inverse:
            coefficient = -coefficient / (roots.unsqueeze(-1) * roots.unsqueeze(-2))
        local = vectors.transpose(-2, -1) @ _sym(grad) @ vectors
        answer = vectors @ (coefficient * local) @ vectors.transpose(-2, -1)
        return _sym(answer), None


def spd_sqrt(matrix):
    return _SPDRoot.apply(matrix, False)


def spd_invsqrt(matrix):
    return _SPDRoot.apply(matrix, True)


@dataclass(frozen=True)
class TransportConfig:
    kind: str = "recolor"
    alpha: float = .5
    shrinkage: float = .05
    covariance_weighting: str = "pooled"
    parameterization: str = "fixed"
    minimum_shrinkage: float = .001
    beta: float = .1
    beta_rule: str = "fixed"
    fiber_control: str = "aligned"
    fiber_seed: int = 20261025

    def __post_init__(self):
        if self.kind not in ("identity", "ea", "recolor", "bures", "metric_bures", "fiber"):
            raise ValueError("Unknown transport kind")
        if self.parameterization not in ("fixed", "global", "evidence"):
            raise ValueError("Unknown transport parameterization")
        if self.covariance_weighting not in ("pooled", "class_balanced"):
            raise ValueError("Unknown support covariance weighting")
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("Transport strength must lie in [0, 1]")
        if not math.isfinite(self.shrinkage) or not 0 <= self.shrinkage <= 1:
            raise ValueError("Covariance shrinkage must lie in [0, 1]")
        if not math.isfinite(self.minimum_shrinkage) or not 0 < self.minimum_shrinkage < 1:
            raise ValueError("Learned covariance shrinkage floor must lie in (0, 1)")
        if self.kind in ("identity", "ea") and self.parameterization != "fixed":
            raise ValueError("Frozen-source/EA controls have no learned meta-parameters")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Fiber rotation magnitude must be finite and nonnegative")
        if self.beta_rule not in ("fixed", "bounded"):
            raise ValueError("Unknown fiber step rule")
        if self.fiber_control not in ("aligned", "reversed", "random"):
            raise ValueError("Unknown covariance-fiber control")
        if isinstance(self.fiber_seed, bool) or not isinstance(self.fiber_seed, int):
            raise ValueError("Fiber control seed must be an integer")


@dataclass
class TransportFit:
    transform: torch.Tensor
    diagnostics: dict
    fallback_reason: str | None = None


def parameter_shape(config):
    return {"fixed": (0,), "global": (2,), "evidence": (2, 4)}[config.parameterization]


def _logit(value):
    value = min(1 - 1e-6, max(1e-6, float(value)))
    return math.log(value / (1 - value))


def init_meta_parameters(config, device=None):
    """Initialize global/gated strengths; no target label-derived initialization."""
    if config.parameterization == "fixed":
        return None
    eta_unit = (config.shrinkage - config.minimum_shrinkage) / (1 - config.minimum_shrinkage)
    logits = torch.tensor([_logit(config.alpha), _logit(eta_unit)], dtype=torch.float64, device=device)
    if config.parameterization == "global":
        return logits.requires_grad_(True)
    parameters = torch.zeros((2, 4), dtype=torch.float64, device=device)
    parameters[:, -1] = logits
    return parameters.requires_grad_(True)


def trial_covariances(q, x):
    """Each acquired valid trial contributes one unit-trace covariance."""
    z = q @ x.double()
    covariance = z @ z.transpose(-2, -1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not bool(torch.isfinite(covariance).all()) or bool((trace <= 0).any()):
        raise FloatingPointError("Nonfinite or zero-energy support covariance")
    return _sym(covariance / trace[:, None, None])


def support_features(covariances, reference):
    """Three bounded, label-free support descriptors plus an intercept.

    Dispersion is a descriptive finite-prefix variation statistic, not a
    calibrated sampling uncertainty under dependent EEG trials.
    """
    count = len(covariances)
    mean = covariances.mean(0)
    reference = reference / reference.trace()
    relative_shift = (mean - reference).norm() / reference.norm().clamp_min(1e-12)
    dispersion = ((covariances - mean).square().sum((-2, -1)).mean()
                  / mean.square().sum().clamp_min(1e-12)).sqrt()
    return torch.stack((relative_shift / (1 + relative_shift),
                        dispersion / (1 + dispersion),
                        mean.new_tensor(count / (count + 20)), mean.new_tensor(1.))).detach()


def covariance_mean(covariances, labels, weighting):
    """Class-balanced control uses acquired labels only, retaining one-class data."""
    if weighting == "class_balanced" and all(bool((labels == c).any()) for c in (0, 1)):
        return .5 * sum(covariances[labels == c].mean(0) for c in (0, 1)), "class_balanced"
    reason = "pooled_one_class_support" if weighting == "class_balanced" else "pooled"
    return covariances.mean(0), reason


def covariance_map(covariance, reference, kind):
    """Return T with T C T' = R, for each full-strength covariance map."""
    c_inverse = spd_invsqrt(covariance)
    if kind in ("recolor", "ea"):
        return spd_sqrt(reference) @ c_inverse
    if kind == "bures":
        c_root = spd_sqrt(covariance)
        middle = spd_sqrt(_sym(c_root @ reference @ c_root))
        return _sym(c_inverse @ middle @ c_inverse)
    if kind == "metric_bures":
        r_root = spd_sqrt(reference)
        r_inverse = spd_invsqrt(reference)
        white_covariance = _sym(r_inverse @ covariance @ r_inverse)
        return r_root @ spd_invsqrt(white_covariance) @ r_inverse
    if kind == "identity":
        return torch.eye(len(covariance), dtype=covariance.dtype, device=covariance.device)
    raise ValueError("Unknown covariance map")


def fiber_map(covariance, reference, class_contrast, class_reference, config):
    """Label-aware orthogonal freedom invisible to the pooled covariance.

    class_reference is a fixed training-only class contrast in source-whitened
    coordinates. Alignment increases to first order along [Ds,Dt]; a finite
    rotation need not improve prediction or alignment. This is an experimental
    learner, not an identification of the true physiological transform.
    """
    if class_reference is None or tuple(class_reference.shape) != tuple(reference.shape):
        raise ValueError("Fiber transport requires a source-whitened training-only class reference")
    ds = _sym(class_reference.detach().to(reference))
    if not bool(torch.isfinite(ds).all()):
        raise ValueError("Class reference contains nonfinite values")
    c_inverse = spd_invsqrt(covariance)
    dt = _sym(c_inverse @ class_contrast @ c_inverse)
    commutator = ds @ dt - dt @ ds
    norm = commutator.norm()
    # A vanishing commutator has no preferred local rotation. It is not a
    # claim that either global alignment or classification is optimized.
    direction = torch.where(norm > 1e-12, commutator / norm.clamp_min(1e-12),
                            torch.zeros_like(commutator))
    # For F(t)=tr(Ds exp(tK) Dt exp(-tK)), a uniform second-derivative
    # bound is 4||Ds||F ||Dt||F ||K||op^2. Use the aligned K to set the
    # magnitude for every control, so reverse/random keep the same beta.
    slope = (commutator * direction).sum()
    lipschitz = 4 * ds.norm() * dt.norm() * torch.linalg.matrix_norm(direction, ord=2).square()
    if config.beta_rule == "bounded":
        beta = torch.minimum(norm.new_tensor(config.beta), slope / lipschitz.clamp_min(1e-30))
    else:
        beta = norm.new_tensor(config.beta)
    if config.fiber_control == "reversed":
        direction = -direction
    elif config.fiber_control == "random":
        generator = torch.Generator(device="cpu").manual_seed(config.fiber_seed)
        random = torch.randn(tuple(reference.shape), generator=generator, dtype=torch.float64).to(reference)
        random = random - random.T
        # Keep exactly the same active direction norm including the zero-null.
        direction = random / random.norm().clamp_min(1e-12) * direction.norm()
    rotation = torch.matrix_exp(beta * direction)
    mapped_contrast = rotation @ dt @ rotation.T
    full = spd_sqrt(reference) @ rotation @ c_inverse
    with torch.no_grad():
        diagnostics = {"fiber_beta": float(beta.detach()), "fiber_beta_max": config.beta,
            "fiber_beta_rule": config.beta_rule, "fiber_control": config.fiber_control,
            "fiber_commutator_norm": float(norm.detach()),
            "fiber_direction_norm": float(direction.detach().norm()),
            "fiber_local_direction_unidentified": bool(norm.detach() < 1e-12),
            "fiber_aligned_initial_slope": float(slope.detach()),
            "fiber_aligned_second_derivative_bound": float(lipschitz.detach()),
            "fiber_alignment_gain_lower_bound": float((beta*slope-.5*lipschitz*beta.square()).detach())
                if config.fiber_control == "aligned" else None,
            "fiber_bound_scope": "class_contrast_alignment_on_regularized_support_covariance_only",
            "fiber_alignment_before": float(torch.trace(ds @ dt)),
            "fiber_alignment_after": float(torch.trace(ds @ mapped_contrast)),
            "fiber_contrast_distance_before": float((ds-dt).square().sum()),
            "fiber_contrast_distance_after": float((ds-mapped_contrast).square().sum()),
            "fiber_rotation_orthogonality_error": float((rotation @ rotation.T -
                torch.eye(len(rotation), device=rotation.device, dtype=rotation.dtype)).norm())}
    return full, diagnostics


def _coefficients(config, features, meta_parameters):
    if config.parameterization == "fixed":
        return features.new_tensor(config.alpha), features.new_tensor(config.shrinkage)
    if meta_parameters is None or tuple(meta_parameters.shape) != parameter_shape(config):
        raise ValueError("Transport meta-parameters do not match the declared model")
    if not bool(torch.isfinite(meta_parameters).all()):
        raise FloatingPointError("Nonfinite learned transport parameters")
    values = meta_parameters if config.parameterization == "global" else meta_parameters @ features
    probabilities = values.sigmoid()
    return probabilities[0], config.minimum_shrinkage + (1 - config.minimum_shrinkage) * probabilities[1]


def fit_transport(source, geometry, support, config, meta_parameters=None):
    """Fit a per-session map from the acquired support; no query argument exists.

    source is accepted for a common learner interface, but this covariance fit
    never updates or calls the source model. Predictions retain autograd paths
    to meta-parameters while diagnostics are detached scalar observations.
    """
    del source
    metadata = meta_parameters if isinstance(meta_parameters, dict) else {}
    learned = metadata.get("learned") if isinstance(meta_parameters, dict) else meta_parameters
    q = geometry.q
    eye = torch.eye(q.shape[0], dtype=torch.float64, device=q.device)
    diagnostics = {"kind": config.kind, "support_trials": len(support.x),
        "covariance_normalization": "per_trial_unit_trace", "source_reference_recoloring": True,
        "parameterization": config.parameterization, "covariance_weighting": config.covariance_weighting,
        "uses_support_labels": config.covariance_weighting == "class_balanced" or config.kind == "fiber",
        "head_parameters": 0, "target_optimization_steps": 0,
        "query_used_for_fitting": False}
    if config.kind == "identity":
        return TransportFit(eye, dict(diagnostics, alpha=0., shrinkage=0.))
    if not len(support.x):
        return TransportFit(eye, diagnostics, "empty_support")
    try:
        covariances = trial_covariances(q, support.x)
        reference = _sym(geometry.rref_sqrt @ geometry.rref_sqrt.T)
        features = support_features(covariances, reference)
        mean, actual_weighting = covariance_mean(covariances, support.y, config.covariance_weighting)
        alpha, shrinkage = _coefficients(config, features, learned)
        if config.kind == "ea":
            alpha, shrinkage = eye.new_tensor(1.), eye.new_tensor(.05)
        covariance = _sym((1 - shrinkage) * mean + shrinkage * eye / len(eye))
        if config.kind == "fiber":
            two_classes = all(bool((support.y == c).any()) for c in (0, 1))
            contrast = (covariances[support.y == 1].mean(0) - covariances[support.y == 0].mean(0)
                        if two_classes else torch.zeros_like(covariance))
            full, fiber_diagnostics = fiber_map(covariance, reference, contrast,
                                               metadata.get("class_reference"), config)
            diagnostics.update(fiber_diagnostics, fiber_two_support_classes=two_classes)
        else:
            full = covariance_map(covariance, reference, config.kind)
        transform = eye + alpha * (full - eye)
        with torch.no_grad():
            values = torch.linalg.eigvalsh(covariance)
            fitted_covariance = transform @ covariance @ transform.T
            target_residual = (fitted_covariance - reference).norm() / reference.norm()
            full_residual = (full @ covariance @ full.T - reference).norm() / reference.norm()
            displacement = torch.trace((transform-eye) @ covariance @ (transform-eye).T)
            normalized_full = spd_invsqrt(reference) @ full @ spd_sqrt(covariance)
            diagnostics.update(alpha=float(alpha.detach()), shrinkage=float(shrinkage.detach()),
                actual_covariance_weighting=actual_weighting,
                support_features=features.cpu().tolist(),
                covariance_min_eigenvalue=float(values.min()),
                covariance_condition=float(values.max() / values.min()),
                transformed_regularized_covariance_residual=float(target_residual),
                full_map_covariance_residual=float(full_residual),
                covariance_weighted_displacement=float(displacement),
                polar_rotation_distance=float((normalized_full-eye).norm()),
                transport_asymmetry=float((transform-transform.T).norm()),
                regularized_covariance_trace=float(covariance.trace()),
                source_reference_trace=float(reference.trace()))
        return TransportFit(transform, diagnostics)
    except (FloatingPointError, torch.linalg.LinAlgError) as error:
        # A numeric failure remains an observed source-model fallback, never a
        # silently removed participant or a change to the acquired prefix.
        diagnostics.update(numeric_fallback=True, reason=str(error))
        return TransportFit(eye, diagnostics, "invalid_support_covariance:" + type(error).__name__)


def transform_signal(geometry, fitted, x):
    """Identity extension preserves any tiny floating-point CAR residual."""
    raw = x.double()
    q = geometry.q
    eye = torch.eye(q.shape[0], dtype=q.dtype, device=q.device)
    return raw + q.T @ (fitted.transform - eye) @ q @ raw


def predict_transport(source, geometry, fitted, x):
    """Apply a frozen per-session transform to any batch without query labels."""
    if fitted.fallback_reason is not None or fitted.diagnostics.get("kind") == "identity":
        return source.source_logits(x)
    signal = transform_signal(geometry, fitted, x)
    rms = signal.square().mean((-2, -1), keepdim=True).sqrt()
    if not bool(torch.isfinite(signal).all()) or bool((rms <= 0).any()):
        raise FloatingPointError("Nonfinite or zero-energy transformed query")
    return source.features_and_logits(signal / rms)[1]
