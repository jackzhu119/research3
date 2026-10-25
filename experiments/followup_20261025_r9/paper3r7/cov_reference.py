"""Support-only zero-lag covariance classifier for an exact temporal null.

This reference has no neural or temporal feature dependence. It is not a
filter-bank CSP implementation and does not claim invariance after arbitrary
filters that fail to commute with a time transformation.
"""
from dataclasses import dataclass
import torch
from . import core


@dataclass
class CovarianceState:
    coefficients: torch.Tensor | None
    constant_margin: torch.Tensor | None = None
    shrinkage: float = .1


def covariance_features(geometry, x, shrinkage=.1):
    """Trace-normalized native-CAR covariance and log-Euclidean coordinates."""
    if not 0 < shrinkage < 1:
        raise ValueError('Positive fixed covariance shrinkage below one required')
    y = geometry.q.double() @ x.double()
    covariance = y @ y.transpose(-1, -2) / y.shape[-1]
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not bool(torch.isfinite(covariance).all()) or bool((trace <= 0).any()):
        raise FloatingPointError('Finite nonzero trial covariance required')
    eye = torch.eye(10, dtype=y.dtype, device=y.device)
    covariance = (1-shrinkage)*covariance/trace[:, None, None] + shrinkage*eye/10
    values, vectors = torch.linalg.eigh((covariance+covariance.transpose(-1, -2))/2)
    logarithm = (vectors * values.log()[:, None, :]) @ vectors.transpose(-1, -2)
    indices = torch.triu_indices(10, 10, device=y.device)
    features = logarithm[:, indices[0], indices[1]]
    factors = torch.where(indices[0] == indices[1], 1., 2.**.5).to(features)
    features = features*factors
    return torch.cat((features, features.new_ones((len(x), 1))), 1)


def fit_covariance(geometry, support, *, head_l2=.1, iterations=10, shrinkage=.1):
    """Balanced ridge-logistic fit; all features depend on XX^T only."""
    x, y = support.x, support.y
    if not all(bool((y == label).any()) for label in (0, 1)):
        right = (y == 1).sum().double()
        probability = (right+1)/(len(y)+2)
        margin = torch.logit(probability).to(device=x.device)
        return CovarianceState(None, margin, shrinkage), {
            'uses_support_labels': True, 'feature_family': 'zero_lag_covariance_only',
            'fallback_reason': 'single_class_constant_probability',
            'temporal_orthogonal_invariance': True, 'head_parameters': 1}
    features = covariance_features(geometry, x, shrinkage)
    q = core._class_weights(y)
    weights = features.new_zeros(features.shape[1])
    ridge = 2*head_l2/len(weights)
    identity = torch.eye(len(weights), dtype=features.dtype, device=features.device)
    def objective(v):
        losses = torch.nn.functional.binary_cross_entropy_with_logits(features@v, y.to(features), reduction='none')
        return (q*losses).sum()+head_l2*v.square().sum()/len(v)
    for _ in range(iterations):
        probability = (features@weights).sigmoid()
        gradient = features.T @ (q*(probability-y.to(features))) + ridge*weights
        curvature = features.T @ ((q*probability*(1-probability))[:, None]*features) + ridge*identity
        delta = torch.linalg.solve(curvature, gradient)
        previous = objective(weights)
        for power in range(16):
            candidate = weights - delta*2.**-power
            if bool(objective(candidate) <= previous+1e-12):
                weights = candidate
                break
    return CovarianceState(weights.detach(), shrinkage=shrinkage), {
        'uses_support_labels': True, 'feature_family': 'zero_lag_covariance_only',
        'fallback_reason': None, 'temporal_orthogonal_invariance': True,
        'head_parameters': len(weights), 'covariance_shrinkage': shrinkage,
        'support_balanced_loss_plus_ridge': float(objective(weights)),
        'optimizer_updates': iterations, 'not_filter_bank_invariance': True}


def covariance_logits(geometry, state, x):
    if state.coefficients is None:
        margin = state.constant_margin.expand(len(x))
    else:
        margin = covariance_features(geometry, x, state.shrinkage) @ state.coefficients
    return torch.stack((-margin/2, margin/2), 1)
