"""Gain-retaining, query-functional conservative EEG calibration.

The frozen anchor, kernel, and numerical geometry are supplied by the caller.
Only support labels enter ``fit``. Prediction computes an analytic *outer*
bound on a logistic-objective-compatible query correction set. This is neither
a GP posterior interval nor a future-risk/coverage guarantee. See R10_MATH.md.

NumPy-only small linear algebra intentionally stays on CPU (n <= 20). Frozen
EEGNet feature extraction is handled separately by the scientific worker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np


VERSION = "paper3-r10-query-profile-v1"


@dataclass(frozen=True)
class Config:
    kernel: str = "linear"
    amplitude: float = .3
    regularization: float = 1.
    evidence_fraction: float = .5
    nugget: float = 1e-10
    include_intercept: bool = True
    balanced: bool = True
    mode: str = "robust"
    use_null_completion: bool = True
    lengthscale: float = 1.
    max_support: int = 20
    max_newton: int = 60
    max_root: int = 12
    max_bracket: int = 32
    gradient_tolerance: float = 1e-10
    root_tolerance: float = 1e-7
    certification_padding: float = 1e-10
    probability_floor: float = 1e-12

    def validate(self):
        if self.kernel not in ("linear", "rbf") or self.mode not in ("robust", "map"):
            raise ValueError("Unknown residual kernel or prediction mode")
        for name in ("amplitude", "regularization", "evidence_fraction", "nugget",
                     "lengthscale", "gradient_tolerance", "root_tolerance",
                     "certification_padding", "probability_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("Finite numeric configuration required: " + name)
        if not 0 <= self.amplitude <= 100:
            raise ValueError("Kernel amplitude must be in [0,100]")
        if not 1e-8 <= self.regularization <= 1e8:
            raise ValueError("Positive coercive regularization required")
        if not 0 <= self.evidence_fraction <= 1:
            raise ValueError("Evidence fraction must be in [0,1]")
        if not 0 < self.nugget < .01 or self.lengthscale <= 0:
            raise ValueError("Explicit positive UID nugget and lengthscale required")
        if not 0 < self.probability_floor < 1e-3:
            raise ValueError("Invalid common anchor probability floor")
        if min(self.gradient_tolerance, self.root_tolerance, self.certification_padding) <= 0:
            raise ValueError("Positive solver tolerances required")
        for name in ("max_support", "max_newton", "max_root", "max_bracket"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer solver/work budget required")
        if self.max_support > 20:
            raise ValueError("R10 maximum calibration acquisition budget is 20")
        return self


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1. / (1. + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    out[~positive] = e / (1. + e)
    return out


def _margins(logits, floor):
    value = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Finite frozen anchor logits required")
    if value.ndim == 2 and value.shape[1] == 2:
        # Opposite finite 1e308 logits can have an overflowing difference;
        # clipping that signed margin has a well-defined bounded probability.
        with np.errstate(over="ignore"):
            value = value[:, 1] - value[:, 0]
    elif value.ndim != 1:
        raise ValueError("Binary logits [n,2] or binary margins [n] required")
    bound = math.log((1. - floor) / floor)
    return np.clip(value, -bound, bound)


def anchor_probabilities(logits, probability_floor=1e-12):
    """One shared numerical anchor for EA baseline and every R10 fallback."""
    if not 0 < probability_floor < 1e-3:
        raise ValueError("Invalid anchor floor")
    p = np.clip(_sigmoid(_margins(logits, probability_floor)),
                probability_floor, 1. - probability_floor)
    return np.column_stack((1. - p, p))


def _matrix(value, name, dimension=None):
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or (dimension is not None and x.shape[1] != dimension):
        raise ValueError(name + " must be a feature matrix of the frozen dimension")
    if x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError(name + " must contain finite nonempty feature coordinates")
    return np.ascontiguousarray(x)


def _reference(reference, dimension):
    if reference is None:
        return {"mean": np.zeros(dimension), "whitener": np.eye(dimension),
                "lengthscale": 1., "provenance": {"role": "identity_software_fixture"}}
    if not isinstance(reference, dict):
        raise ValueError("Explicit frozen reference dictionary required")
    mean = np.asarray(reference["mean"], dtype=np.float64)
    whitener = np.asarray(reference["whitener"], dtype=np.float64)
    lengthscale = float(reference.get("lengthscale", 1.))
    provenance = json.loads(json.dumps(reference.get("provenance", {}), allow_nan=False))
    role = provenance.get("role")
    if role not in ("meta_train", "source_pretrain_train", "identity_software_fixture"):
        raise ValueError("Residual geometry must be frozen from meta_train/source training")
    if (mean.shape != (dimension,) or whitener.shape != (dimension, dimension)
            or not np.isfinite(mean).all() or not np.isfinite(whitener).all()
            or not math.isfinite(lengthscale) or lengthscale <= 0):
        raise ValueError("Frozen reference shape or numerical values differ")
    return {"mean": mean.copy(), "whitener": whitener.copy(),
            "lengthscale": lengthscale, "provenance": provenance}


def _transform(features, reference):
    with np.errstate(over="ignore", invalid="ignore"):
        z = (features - reference["mean"]) @ reference["whitener"]
    if not np.isfinite(z).all():
        raise ValueError("Frozen feature whitening overflowed")
    # Exactly the supplied frozen whitening, without fitting query statistics
    # or adding a new feature saturation mechanism to the matched ablations.
    return z


def _kernel(x, z, config, reference):
    d = x.shape[1]
    if config.kernel == "linear":
        value = x @ z.T / d
    else:
        distance = np.maximum(0., ((x*x).sum(1)[:, None] +
                                  (z*z).sum(1)[None, :] - 2*x@z.T) / d)
        length = config.lengthscale * reference["lengthscale"]
        value = np.exp(-distance / (2*length*length))
    if config.include_intercept:
        value = value + 1.
    return config.amplitude**2 * value


def _kernel_diagonal(x, config):
    diagonal = np.mean(x*x, axis=1) if config.kernel == "linear" else np.ones(len(x))
    return config.amplitude**2 * (diagonal + float(config.include_intercept)) + config.nugget


def _identifiers(value, n, prefix):
    ids = tuple(prefix + str(i) for i in range(n)) if value is None else tuple(value)
    if len(ids) != n or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != n:
        raise ValueError("Unique nonempty event identifiers required")
    return ids


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _array_sha(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(_canonical(list(value.shape)) + value.tobytes()).hexdigest()


class _Objective:
    def __init__(self, factor, margin, labels, weights, regularization):
        self.B, self.margin, self.y, self.weights = factor, margin, labels, weights
        self.regularization = regularization

    def evaluate(self, v, tilt=None, hessian=False):
        z = self.margin + self.B @ v
        probability = _sigmoid(z)
        penalty = self.regularization
        value = float(np.dot(self.weights, np.logaddexp(0., z) - self.y*z) +
                      .5*penalty*np.dot(v, v))
        gradient = self.B.T @ (self.weights * (probability-self.y)) + penalty*v
        if tilt is not None:
            value -= float(np.dot(tilt, v))
            gradient = gradient-tilt
        matrix = None
        if hessian:
            curvature = self.weights*probability*(1-probability)
            matrix = (self.B.T*curvature) @ self.B + penalty*np.eye(len(v))
        return value, gradient, matrix


def _newton(objective, config, *, tilt=None, start=None):
    n = len(objective.margin)
    v = np.zeros(n) if start is None else np.array(start, dtype=np.float64, copy=True)
    evaluations, converged, reason = 0, False, "iteration_budget"
    for iteration in range(config.max_newton):
        value, gradient, hessian = objective.evaluate(v, tilt, True)
        evaluations += 1
        if not math.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite regularized logistic objective")
        if np.linalg.norm(gradient) <= config.gradient_tolerance*(1 + (0. if tilt is None else np.linalg.norm(tilt))):
            converged, reason = True, "gradient_tolerance"
            break
        direction = -np.linalg.solve(hessian, gradient)
        slope = float(np.dot(gradient, direction))
        if not math.isfinite(slope) or slope >= 0:
            raise FloatingPointError("Newton direction was not a descent direction")
        step, accepted = 1., False
        for _ in range(40):
            candidate = v + step*direction
            following = objective.evaluate(candidate, tilt)[0]
            evaluations += 1
            # A tiny roundoff allowance can end line search without inventing
            # a convergence certificate: the final residual remains in duals.
            allowance = 8*np.finfo(float).eps*(1+abs(value))
            if math.isfinite(following) and following <= value + 1e-4*step*slope + allowance:
                v, accepted = candidate, True
                break
            step *= .5
        if not accepted:
            reason = "line_search_budget"
            break
    value, gradient, _ = objective.evaluate(v, tilt)
    residual_squared = float(np.dot(gradient, gradient))
    if not math.isfinite(value) or not math.isfinite(residual_squared):
        raise FloatingPointError("Nonfinite tilted objective certificate")
    return {"v": v, "value": value, "gradient": gradient,
            "lower_bound": value-residual_squared/(2*objective.regularization),
            "gradient_norm": math.sqrt(residual_squared), "converged": converged,
            "iterations": iteration+1, "evaluations": evaluations, "reason": reason}


@dataclass
class FittedState:
    config: Config
    features: np.ndarray
    margins: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    factor: np.ndarray
    map_parameter: np.ndarray
    reference: dict
    support_ids: tuple
    fit_identity: str
    status: str
    fallback_reason: str | None
    objective_at_zero: float
    objective_at_fit: float
    gain: float
    threshold: float
    diagnostics: dict

    def __post_init__(self):
        for value in (self.features, self.margins, self.labels, self.weights,
                      self.factor, self.map_parameter, self.reference["mean"],
                      self.reference["whitener"]):
            value.setflags(write=False)

    @property
    def support_count(self):
        return len(self.margins)

    def _objective(self):
        return _Objective(self.factor, self.margins, self.labels, self.weights,
                          self.config.regularization)


def _state_identity(config, features, margins, labels, reference, support_ids):
    body = {"version": VERSION, "config": asdict(config),
            "support_ids": list(support_ids), "features_sha256": _array_sha(features),
            "margins_sha256": _array_sha(margins), "labels_sha256": _array_sha(labels),
            "reference_mean_sha256": _array_sha(reference["mean"]),
            "reference_whitener_sha256": _array_sha(reference["whitener"]),
            "reference_lengthscale": reference["lengthscale"],
            "reference_provenance": reference["provenance"]}
    return hashlib.sha256(_canonical(body)).hexdigest()


def fit(support_features, support_logits, support_labels, config=None, *, reference=None,
        support_ids=None):
    """Fit support-only residual state; ``reference`` must already be frozen.

    Balanced weights sum to n: w_i=n/(2*n_class). This is an existing
    class-balanced likelihood convention, not part of the novelty claim.
    """
    started = time.perf_counter()
    config = (config or Config()).validate()
    features = _matrix(support_features, "support_features")
    n, dimension = features.shape
    if n > config.max_support:
        raise ValueError("Support exceeds the original acquired-position budget")
    margins = _margins(support_logits, config.probability_floor)
    labels = np.asarray(support_labels)
    if (margins.shape != (n,) or labels.shape != (n,)
            or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Support features, binary labels and margins must align")
    labels = labels.astype(np.float64)
    reference = _reference(reference, dimension)
    geometry_failure = None
    try:
        features = _transform(features, reference)
    except (FloatingPointError, OverflowError, ValueError) as error:
        # Keep finite raw coordinates only for a reproducible fallback-state
        # identity. No residual model or transformed coordinates are claimed.
        features = features.copy()
        geometry_failure = "support_geometry_failure:"+type(error).__name__
    ids = _identifiers(support_ids, n, "support-")
    identity = _state_identity(config, features, margins, labels, reference, ids)
    weights = np.ones(n)
    classes = np.bincount(labels.astype(np.int64), minlength=2)
    if config.balanced and min(classes, default=0) > 0:
        weights = n / (2*classes[labels.astype(np.int64)])
    fallback = ("empty_support" if n == 0 else "insufficient_support" if n < 2 else
                "one_class_support" if np.count_nonzero(classes) < 2 else
                "zero_kernel_amplitude" if config.amplitude == 0 else geometry_failure)
    factor, vhat = np.empty((0, 0)), np.zeros(n)
    f0 = float(np.dot(weights, np.logaddexp(0., margins)-labels*margins))
    fhat, gain, threshold, details = f0, 0., f0, {}
    if fallback is None:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                kernel = _kernel(features, features, config, reference)
            if not np.isfinite(kernel).all():
                raise FloatingPointError("Support kernel overflowed")
            kernel = (kernel+kernel.T)/2 + config.nugget*np.eye(n)
            factor = np.linalg.cholesky(kernel)
            objective = _Objective(factor, margins, labels, weights, config.regularization)
            solution = _newton(objective, config)
            vhat, fhat = solution["v"], solution["value"]
            if fhat > f0 or not np.isfinite(vhat).all():
                fallback = "no_verified_objective_gain"
            else:
                gain = max(0., f0-fhat)
                threshold = fhat+config.evidence_fraction*gain
            details = {k: solution[k] for k in ("converged", "gradient_norm", "iterations", "evaluations", "reason")}
            details.update(kernel_dimension=n, all_spd_directions_retained=True,
                           declared_uid_nugget=config.nugget,
                           minimum_kernel_eigenvalue=float(np.linalg.eigvalsh(kernel)[0]))
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
            fallback = "fit_numerical_failure:"+type(error).__name__
    if fallback is not None:
        gain, threshold = 0., f0
    details.update(fit_seconds=time.perf_counter()-started,
                   weighting="n/(2*n_class)" if config.balanced else "ordinary_unit_weights",
                   effective_support_rows=n, fit_queries_seen=0,
                   conditional_model_set_statement_only=True)
    return FittedState(config, features, margins, labels, weights, factor, vhat,
                       reference, ids, identity,
                       "ANCHOR_FALLBACK" if fallback else "FIT_COMPLETE", fallback,
                       f0, fhat, gain, threshold, details)


def _direction_bound(state, h, innovation):
    """Conservative max(h^T v + sqrt(s)t) bound at several valid dual tilts."""
    config, objective = state.config, state._objective()
    regularization, threshold = config.regularization, state.threshold
    kappa = threshold-state.objective_at_fit
    norm_squared = float(np.dot(h, h))
    if norm_squared+innovation <= 0 or kappa <= 0:
        return {"upper": math.inf, "reason": "degenerate_tilt", "tilts": 0, "converged": False}
    zeta = math.sqrt(2*regularization*kappa/max(norm_squared+innovation, np.finfo(float).tiny))
    zeta = min(max(zeta, 1e-12), 1e12)
    best, best_zeta, tilts, failures = math.inf, None, 0, 0
    v = state.map_parameter.copy()

    def evaluate(tilt):
        nonlocal best, best_zeta, tilts, failures, v
        solution = _newton(objective, config, tilt=tilt*h, start=v)
        v = solution["v"]
        tilts += 1
        failures += int(not solution["converged"])
        lower = solution["lower_bound"]
        # This lower bound is analytic for every approximate iterate; no
        # Newton convergence flag is used as a substitute for the residual.
        bound = (threshold-lower)/tilt + tilt*innovation/(2*regularization)
        padding = config.certification_padding*(1+abs(bound)+
                    abs(threshold)/tilt+abs(lower)/tilt+tilt*innovation/regularization)
        bound += padding
        if math.isfinite(bound) and bound < best:
            best, best_zeta = bound, tilt
        original_value = objective.evaluate(v)[0]
        energy = original_value + tilt*tilt*innovation/(2*regularization)
        return energy-threshold

    low, high = 0., None
    for _ in range(config.max_bracket):
        residual = evaluate(zeta)
        if residual >= 0:
            high = zeta
            break
        low, zeta = zeta, zeta*2
        if not math.isfinite(zeta) or zeta > 1e16:
            break
    if high is None:
        return {"upper": math.inf, "reason": "tilt_bracket_budget", "tilts": tilts,
                "unconverged_inner_solves": failures, "converged": False}
    for _ in range(config.max_root):
        middle = (low+high)/2
        residual = evaluate(middle)
        if residual >= 0:
            high = middle
        else:
            low = middle
        if high-low <= config.root_tolerance*(1+high):
            break
    return {"upper": best, "zeta": best_zeta, "reason": "dual_outer_bound",
            "tilts": tilts, "unconverged_inner_solves": failures,
            "root_iterations_budget":config.max_root,
            "root_bracket_width":high-low,
            "dual_certificate_valid_despite_root_budget":math.isfinite(best),
            "converged": math.isfinite(best)}


def predict(state, query_features, query_logits, *, query_ids=None):
    """Predict without accepting labels, fitting on queries or batch statistics."""
    started = time.perf_counter()
    config = state.config.validate()
    raw = _matrix(query_features, "query_features", state.features.shape[1])
    margins = _margins(query_logits, config.probability_floor)
    if margins.shape != (len(raw),):
        raise ValueError("Query frozen logits and features must align")
    ids = _identifiers(query_ids, len(raw), "query-")
    if set(ids).intersection(state.support_ids):
        raise ValueError("Future query and acquired support event identifiers overlap")
    anchor = anchor_probabilities(query_logits, config.probability_floor)
    result = anchor.copy()
    rows = []
    if state.status != "FIT_COMPLETE":
        rows = [{"reason": state.fallback_reason, "correction": 0., "certificate": "anchor_identity"}
                for _ in raw]
    elif config.evidence_fraction == 1 and config.mode == "robust":
        rows = [{"reason": "zero_function_in_set", "correction": 0., "certificate": "anchor_feasible"}
                for _ in raw]
    elif state.gain <= config.certification_padding:
        rows = [{"reason": "negligible_support_gain", "correction": 0., "certificate": "anchor_identity"}
                for _ in raw]
    else:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                features = (raw-state.reference["mean"]) @ state.reference["whitener"]
                cross = _kernel(state.features, features, config, state.reference)
                diagonal = _kernel_diagonal(features, config)
            valid = np.isfinite(features).all(axis=1) & np.isfinite(cross).all(axis=0) & np.isfinite(diagonal)
            coefficients = np.zeros_like(cross)
            if valid.any():
                coefficients[:,valid] = np.linalg.solve(state.factor, cross[:,valid])
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError) as error:
            reason = "query_projection_failure:"+type(error).__name__
            return anchor, {"version": VERSION, "fit_identity": state.fit_identity,
                "queries": [{"reason":reason, "correction":0., "certificate":"anchor_identity"}
                            for _ in raw],
                "summary": {"queries":len(raw),"reason_counts":{reason:len(raw)},
                    "nonzero_updates":0,"prediction_seconds":time.perf_counter()-started,
                    "query_labels_seen":0,"query_batch_statistics_updated":False,
                    "true_future_risk_or_coverage_guaranteed":False}}
        for i, a in enumerate(coefficients.T):
            if not valid[i]:
                rows.append({"reason":"query_geometry_or_kernel_overflow", "correction":0.,
                             "certificate":"anchor_identity"})
                continue
            mu = float(a @ state.map_parameter)
            projected = float(a @ a)
            innovation = float(diagonal[i]-projected)
            numerical_scale = 1+abs(diagonal[i])+projected
            allowed_negative = 64*np.finfo(float).eps*numerical_scale
            details = {"map_correction": mu, "kernel_innovation": innovation,
                       "support_projection_energy": projected, "correction": 0.}
            if not np.isfinite(a).all() or not math.isfinite(mu) or innovation < -allowed_negative:
                details.update(reason="invalid_joint_kernel_certificate", certificate="anchor_identity")
                rows.append(details)
                continue
            innovation = max(0., innovation) if config.use_null_completion else 0.
            details["declared_profile_innovation"] = innovation
            if config.mode == "map" or config.evidence_fraction == 0:
                correction = mu
                details.update(reason="map_ablation", certificate="not_profile_certified")
            elif mu == 0 or projected == 0:
                correction = 0.
                details.update(reason="zero_query_cross_information", certificate="anchor_feasible")
            elif (innovation > 0 and config.regularization*mu*mu/(2*innovation)
                  <= config.evidence_fraction*state.gain):
                correction = 0.
                details.update(reason="support_null_completion_witness", certificate="anchor_feasible",
                               null_witness_cost=config.regularization*mu*mu/(2*innovation))
            else:
                try:
                    sign = 1. if mu > 0 else -1.
                    bound = _direction_bound(state, -sign*a, innovation)
                    details.update(bound)
                    if not bound["converged"]:
                        correction = 0.
                        details.update(certificate="anchor_identity")
                    else:
                        extent = max(0., -bound["upper"])
                        # MAP is a feasible witness. A bound claiming a nearer
                        # endpoint beyond MAP is inconsistent, not permission
                        # to silently replace the certified endpoint by MAP.
                        if extent > abs(mu):
                            correction = 0.
                            details.update(reason="inconsistent_dual_bound", certificate="anchor_identity")
                        else:
                            correction = sign*extent
                            details.update(reason="certified_nonzero_update" if extent > 0 else
                                           "outer_interval_contains_anchor",
                                           certificate="analytic_dual_outer_bound_float64_guarded")
                except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                    correction = 0.
                    details.update(reason="prediction_numerical_failure:"+type(error).__name__,
                                   certificate="anchor_identity")
            if correction:
                p = float(np.clip(_sigmoid(np.array([margins[i]+correction]))[0],
                                  config.probability_floor, 1-config.probability_floor))
                result[i] = (1-p, p)
            details["correction"] = correction
            rows.append(details)
    reasons = {}
    for row in rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0)+1
    return result, {"version": VERSION, "fit_identity": state.fit_identity,
        "queries": rows, "summary": {"queries": len(rows), "reason_counts": reasons,
        "nonzero_updates": sum(row["correction"] != 0 for row in rows),
        "prediction_seconds": time.perf_counter()-started,
        "query_labels_seen": 0, "query_batch_statistics_updated": False,
        "true_future_risk_or_coverage_guaranteed": False}}


def save_state(state, path):
    """Atomic pickle-free scientific fit state, independently hashable by worker."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"version": VERSION, "config": asdict(state.config),
        "support_ids": state.support_ids, "fit_identity": state.fit_identity,
        "status": state.status, "fallback_reason": state.fallback_reason,
        "objective_at_zero": state.objective_at_zero, "objective_at_fit": state.objective_at_fit,
        "gain": state.gain, "threshold": state.threshold, "diagnostics": state.diagnostics,
        "reference_lengthscale": state.reference["lengthscale"],
        "reference_provenance": state.reference["provenance"],
        "derived_array_sha256": {key: _array_sha(getattr(state,key)) for key in
            ("weights", "factor", "map_parameter")}}
    fd, temporary = tempfile.mkstemp(prefix=".r10-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, metadata=np.array(_canonical(metadata).decode()),
                features=state.features, margins=state.margins, labels=state.labels,
                weights=state.weights, factor=state.factor, map_parameter=state.map_parameter,
                reference_mean=state.reference["mean"], reference_whitener=state.reference["whitener"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_state(path, *, expected_fit_identity=None):
    with np.load(path, allow_pickle=False) as archive:
        value = json.loads(str(archive["metadata"].item()))
        if value.get("version") != VERSION:
            raise ValueError("R10 fit state version changed")
        config = Config(**value["config"]).validate()
        arrays = {k: np.array(archive[k], dtype=np.float64, copy=True) for k in
                  ("features", "margins", "labels", "weights", "factor", "map_parameter")}
        reference = {"mean": np.array(archive["reference_mean"], copy=True),
            "whitener": np.array(archive["reference_whitener"], copy=True),
            "lengthscale": value["reference_lengthscale"],
            "provenance": value["reference_provenance"]}
    identity = _state_identity(config, arrays["features"], arrays["margins"], arrays["labels"],
                               reference, tuple(value["support_ids"]))
    if identity != value["fit_identity"] or (expected_fit_identity is not None and identity != expected_fit_identity):
        raise ValueError("R10 saved support/geometry/config identity changed")
    for key in ("weights", "factor", "map_parameter"):
        if _array_sha(arrays[key]) != value.get("derived_array_sha256", {}).get(key):
            raise ValueError("R10 derived fit-state bytes changed: "+key)
    n = len(arrays["margins"])
    if (arrays["features"].ndim != 2 or arrays["features"].shape[0] != n
            or arrays["labels"].shape != (n,) or arrays["weights"].shape != (n,)
            or arrays["map_parameter"].shape != (n,)
            or not all(np.isfinite(x).all() for x in arrays.values())):
        raise ValueError("R10 saved fit state is malformed or nonfinite")
    if value["status"] == "FIT_COMPLETE":
        if arrays["factor"].shape != (n,n):
            raise ValueError("R10 support factor shape changed")
        objective = _Objective(arrays["factor"], arrays["margins"], arrays["labels"],
                               arrays["weights"], config.regularization)
        recomputed = objective.evaluate(arrays["map_parameter"])[0]
        if not math.isclose(recomputed,value["objective_at_fit"],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("R10 fitted objective metadata changed")
        f0 = objective.evaluate(np.zeros(n))[0]
        gain = f0-recomputed
        threshold = recomputed+config.evidence_fraction*gain
        for key,actual in (("objective_at_zero",f0),("gain",gain),("threshold",threshold)):
            if not math.isclose(actual,value[key],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("R10 compatible-set metadata changed: "+key)
    elif value["status"] != "ANCHOR_FALLBACK":
        raise ValueError("R10 saved fit status is unknown")
    return FittedState(config, arrays["features"], arrays["margins"], arrays["labels"], arrays["weights"],
        arrays["factor"], arrays["map_parameter"], reference, tuple(value["support_ids"]), identity,
        value["status"], value["fallback_reason"], value["objective_at_zero"], value["objective_at_fit"],
        value["gain"], value["threshold"], value["diagnostics"])
