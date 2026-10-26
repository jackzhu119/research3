"""Matched query-profile mechanism controls; neither certifies nonlinear J.

Both controls reuse the main fit, frozen geometry, class-balanced likelihood,
UID nugget and numerical anchor. The variance gate is R9-*like*, not the native
R9 ordinary-likelihood baseline. No query labels or batch-fitted statistics are
accepted. NumPy-only CPU calculations do not claim future-risk guarantees.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import algorithm as core

METHODS = ('profile_quadratic', 'profile_variance_gate')
VERSION = 'paper3-r10-matched-mechanism-ablations-v1'


def _finish(state, method, probabilities, rows, started):
    counts = {}
    for row in rows:
        counts[row['reason']] = counts.get(row['reason'], 0)+1
    return probabilities, {
        'version': VERSION, 'method': method, 'fit_identity': state.fit_identity,
        'not_profile_certified': True,
        'nonlinear_model_set_certificate': False,
        'queries': rows,
        'summary': {
            'queries': len(rows), 'reason_counts': counts,
            'nonzero_updates': sum(bool(row.get('prediction_changed')) for row in rows),
            'prediction_seconds': time.perf_counter()-started,
            'query_labels_seen': 0, 'query_batch_statistics_updated': False,
            'true_future_risk_or_coverage_guaranteed': False,
            'nonlinear_model_set_certificate': False,
        }}


def _row(reason, **details):
    return dict(reason=reason, correction=0., prediction_changed=False,
                certificate='not_profile_certified', **details)


def predict_ablation(state, query_features, query_logits, *, method, query_ids=None):
    """Predict one matched ablation without changing any fitted-state array."""
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError('Unknown R10 mechanism ablation')
    config = state.config.validate()
    raw = np.asarray(query_features, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != state.features.shape[1]:
        raise ValueError('Ablation query feature dimension differs from frozen fit')
    margins = core._margins(query_logits, config.probability_floor)
    if margins.shape != (len(raw),):
        raise ValueError('Ablation query logits and features must align')
    ids = core._identifiers(query_ids, len(raw), 'query-')
    if set(ids).intersection(state.support_ids):
        raise ValueError('Future query and acquired support event identifiers overlap')
    anchor = core.anchor_probabilities(query_logits, config.probability_floor)
    result = anchor.copy()
    if state.status != 'FIT_COMPLETE':
        return _finish(state, method, result,
                       [_row(state.fallback_reason or 'fit_fallback') for _ in raw], started)
    if method == 'profile_quadratic' and config.evidence_fraction == 1:
        # An explicit common anchor-limit control, not a theorem about Q(0).
        return _finish(state, method, result,
                       [_row('explicit_c1_anchor_control') for _ in raw], started)
    if not len(raw):
        return _finish(state, method, result, [], started)
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            features = (raw-state.reference['mean']) @ state.reference['whitener']
            cross = core._kernel(state.features, features, config, state.reference)
            diagonal = core._kernel_diagonal(features, config)
        valid = (np.isfinite(raw).all(axis=1) & np.isfinite(features).all(axis=1)
                 & np.isfinite(cross).all(axis=0) & np.isfinite(diagonal))
        coefficients = np.zeros_like(cross)
        if valid.any():
            coefficients[:, valid] = np.linalg.solve(state.factor, cross[:, valid])
        value, gradient, hessian = state._objective().evaluate(state.map_parameter, hessian=True)
        np.linalg.cholesky(hessian)
        inverse_gradient = np.linalg.solve(hessian, gradient)
        center = state.map_parameter-inverse_gradient
        quadratic_minimum = value-.5*float(gradient@inverse_gradient)
        inverse_coefficients = np.zeros_like(coefficients)
        if valid.any():
            inverse_coefficients[:, valid] = np.linalg.solve(hessian, coefficients[:, valid])
        if not (math.isfinite(value) and math.isfinite(quadratic_minimum)
                and np.isfinite(center).all()):
            raise FloatingPointError('Nonfinite shared curvature state')
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as error:
        return _finish(state, method, result,
                       [_row('ablation_projection_failure:'+type(error).__name__) for _ in raw], started)
    nodes = weights = None
    if method == 'profile_variance_gate':
        nodes, weights = np.polynomial.hermite.hermgauss(20)
        weights = weights/np.sqrt(np.pi)
    rows = []
    for i, coefficient in enumerate(coefficients.T):
        if not valid[i]:
            rows.append(_row('ablation_query_geometry_overflow'))
            continue
        mu = float(coefficient@state.map_parameter)
        projection = float(coefficient@coefficient)
        innovation = float(diagonal[i]-projection)
        scale = 1+abs(float(diagonal[i]))+projection
        tolerance = 64*np.finfo(float).eps*scale
        if (not np.isfinite(coefficient).all() or not math.isfinite(mu)
                or not math.isfinite(innovation) or innovation < -tolerance):
            rows.append(_row('ablation_invalid_joint_kernel'))
            continue
        # The same declared UID-nugget geometry as the main implementation.
        innovation = max(0., innovation)
        projected_covariance = float(coefficient@inverse_coefficients[:, i])
        if not math.isfinite(projected_covariance) or projected_covariance < -tolerance:
            rows.append(_row('ablation_invalid_curvature_variance'))
            continue
        projected_covariance = max(0., projected_covariance)
        details = _row('pending', map_correction=mu, kernel_innovation=innovation,
                       support_projection_energy=projection,
                       class_weighting='n/(2*n_class)' if config.balanced else 'ordinary_unit_weights')
        try:
            if method == 'profile_quadratic':
                slack = state.threshold-quadratic_minimum
                slack_tolerance = 64*np.finfo(float).eps*(1+abs(state.threshold)+abs(quadratic_minimum))
                if not math.isfinite(slack) or slack < -slack_tolerance:
                    raise FloatingPointError('Invalid quadratic level set')
                functional_variance = projected_covariance+innovation/config.regularization
                radius = math.sqrt(2*max(0., slack)*functional_variance)
                central = float(coefficient@center)
                low, high = central-radius, central+radius
                correction = low if low > 0 else high if high < 0 else 0.
                if not all(math.isfinite(x) for x in (central, low, high, correction)):
                    raise FloatingPointError('Nonfinite quadratic query interval')
                probability = float(core._sigmoid(np.array([margins[i]+correction]))[0])
                details.update(reason='quadratic_profile_approximation', correction=correction,
                               quadratic_center=central, quadratic_minimum=quadratic_minimum,
                               quadratic_slack=slack, quadratic_functional_variance=functional_variance,
                               interval_lower=low, interval_upper=high,
                               approximation='local_second_order_not_nonlinear_outer_bound')
            else:
                prior = float(diagonal[i])/config.regularization
                variance = innovation/config.regularization+projected_covariance
                variance_tolerance = 64*np.finfo(float).eps*(1+abs(prior)+abs(variance))
                if (not math.isfinite(prior) or not math.isfinite(variance)
                        or prior <= 0 or variance < -variance_tolerance
                        or variance > prior+variance_tolerance):
                    raise FloatingPointError('Inconsistent matched Laplace variance')
                variance = min(prior, max(0., variance))
                information = float(np.clip((prior-variance)/prior, 0., 1.))
                gate = .75*information
                latent = margins[i]+mu+np.sqrt(2*variance)*nodes
                posterior_probability = float(weights@core._sigmoid(latent))
                probability = (1-gate)*float(anchor[i, 1])+gate*posterior_probability
                details.update(reason='matched_r9_like_variance_gate', correction=mu,
                               posterior_variance=variance, prior_variance=prior,
                               information_fraction=information, mixture_weight=gate,
                               posterior_predictive_probability1=posterior_probability,
                               quadrature_points=20, anchor_cap=.75,
                               likelihood_scope='matched_R10_weights_not_native_R9_baseline')
            probability = float(np.clip(probability, config.probability_floor, 1-config.probability_floor))
            if not math.isfinite(probability):
                raise FloatingPointError('Nonfinite ablation probability')
            result[i] = (1-probability, probability)
            if method == 'profile_variance_gate':
                # Mixture probabilities are not obtained by applying raw mu
                # to the anchor. Keep mu separately; report the actual shift.
                effective = math.log(probability)-math.log1p(-probability)-float(margins[i])
                details['correction'] = effective
                details['effective_logit_correction'] = effective
            details['prediction_changed'] = bool(probability != anchor[i, 1])
            rows.append(details)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as error:
            rows.append(_row('ablation_prediction_numerical_failure:'+type(error).__name__))
    return _finish(state, method, result, rows, started)
