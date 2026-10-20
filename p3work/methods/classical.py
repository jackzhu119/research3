"""Paper3 B4/B5: authorized source fitting and support-only calibration.

This module implements formulas, not completed EEG fits. Source fitting accepts
only the frozen source training partition (or one explicit source LODO fold).
Target fitting accepts the same acquired-prefix capability as neural methods.
Neither fit API has a query argument. Saved models contain JSON and NPY only.

B4 target/source LDA is exactly sklearn lsqr/auto with equal class priors.
As in the supplied legacy source recipe, source class covariance means are
hierarchically balanced, whereas the source LDA classifier fit is unweighted
(sklearn LDA has no sample_weight). This distinction is explicit in receipts;
we do not introduce a new weighted shrinkage estimator. B5 retains the legacy
log-Euclidean reference, weighted
source StandardScaler, and C=1 source logistic classifier; it is not an
affine-invariant-mean tangent-space method.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..adapt import CalibrationPrefix, PredictionInputs
from ..contracts import SOURCE_IDS, digest, file_digest, validate_signal_contract
from ..spatial_geometry import helmert_rows


IMPLEMENTATION_VERSION = "paper3-classical-v2.0.3"
B4_GRID = tuple((alpha, m) for alpha in (0.0, 0.5, 1.0) for m in (1, 2, 3))
B5_GRID = tuple((shrink, l2) for shrink in (0.01, 0.05, 0.1) for l2 in (0.01, 0.1, 1.0))
_Q = helmert_rows().numpy().copy()  # exactly the main method's saved coordinates


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} requires a concrete SHA256")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(f"{array.dtype.str}/{array.shape}".encode())
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def _numpy(value) -> np.ndarray:
    # Torch capability inputs may reside on a GPU; classical fitting is CPU.
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _readonly(value) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite model state")
    result.flags.writeable = False
    return result


def _trials(x, *, allow_empty=False) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1] != 11 or x.shape[2] < 2 or (not allow_empty and not len(x)):
        raise ValueError("expected nonempty N x 11 x T pre-RMS trials")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("trial QC must reject nonfinite/nonnumeric values")
    # Input is the protocol's CAR/time-demeaned cache; do not silently repair it.
    scale = np.max(np.abs(x), axis=(1, 2)) if len(x) else np.empty(0)
    if (scale <= 0).any():
        raise ValueError("zero-energy trial")
    # Episode capabilities promote the frozen float32 cache to float64. That
    # promotion preserves the original rounding; it cannot restore exact CAR
    # or temporal means. Recognize only an exact float32 round trip, retaining
    # the strict float64 check for genuinely higher-precision inputs. No signal
    # is rounded, rereferenced or demeaned here.
    cache_precision = x.dtype.itemsize <= 4
    if not cache_precision and np.issubdtype(x.dtype, np.floating):
        with np.errstate(over="ignore", invalid="ignore"):
            restored = x.astype(np.float32).astype(x.dtype)
        cache_precision = np.array_equal(restored, x)
    tolerance = scale * (2e-6 if cache_precision else 1e-10)
    if len(x) and ((np.max(np.abs(np.mean(x, axis=1)), axis=1) > tolerance).any()
                   or (np.max(np.abs(np.mean(x, axis=2)), axis=1) > tolerance).any()):
        raise ValueError("input must already follow CAR11 and temporal demeaning")
    return x


def trace_covariances(x, *, shrink=0.0) -> np.ndarray:
    """Trace-one CAR10 covariance; only prescribed isotropic SPD shrinkage."""
    if not 0 <= shrink < 1:
        raise ValueError("covariance shrinkage must be in [0,1)")
    x = _trials(x)
    # Normalize each trial by a common positive gain before the product solely
    # to avoid overflow/underflow; trace normalization cancels this exactly.
    normalized = x.astype(np.float64) / np.max(np.abs(x), axis=(1, 2))[:, None, None]
    z = np.einsum("dc,nct->ndt", _Q, normalized)
    covariance = z @ np.swapaxes(z, -1, -2)
    traces = np.trace(covariance, axis1=1, axis2=2)
    if not np.isfinite(traces).all() or (traces <= 0).any():
        raise ValueError("degenerate CAR covariance")
    covariance /= traces[:, None, None]
    return (1 - shrink) * covariance + shrink * np.eye(10)[None] / 10


def matrix_function(matrix, kind: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (10, 10) or not np.isfinite(matrix).all():
        raise ValueError("expected finite CAR10 matrix")
    eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.swapaxes(matrix, -1, -2)) / 2)
    if kind in {"log", "inverse_root"} and (eigenvalues <= 0).any():
        raise ValueError("non-SPD covariance; no eigenvalue flooring is allowed")
    function = {"log": np.log, "exp": np.exp, "inverse_root": lambda w: w ** -0.5}.get(kind)
    if function is None:
        raise ValueError("unknown symmetric matrix function")
    return (eigenvectors * function(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)


def tangent_features(covariances, inverse_root) -> np.ndarray:
    """Whitened matrix log, upper triangle, sqrt(2) on off-diagonals."""
    logs = matrix_function(inverse_root @ covariances @ inverse_root, "log")
    i, j = np.triu_indices(10)
    return logs[:, i, j] * np.where(i == j, 1.0, np.sqrt(2.0))


def _subject(row) -> str:
    value = row.get("subject_id", row.get("subject"))
    if value is None or str(value) == "":
        raise ValueError("missing source subject identity")
    if row.get("subject") is not None and row.get("subject_id") is not None and str(row["subject"]) != str(row["subject_id"]):
        raise ValueError("conflicting source subject identities")
    return str(value)


def hierarchical_weights(rows, labels) -> np.ndarray:
    """Equal dataset, participant, class, trial-within-class mass, as legacy."""
    datasets = {row["dataset"] for row in rows}
    people = {dataset: {_subject(row) for row in rows if row["dataset"] == dataset}
              for dataset in datasets}
    cells = Counter((row["dataset"], _subject(row), int(label))
                    for row, label in zip(rows, labels, strict=True))
    classes = {person: {label for dataset, subject, label in cells if (dataset, subject) == person}
               for person in {(dataset, subject) for dataset, subject, _ in cells}}
    weights = np.array([1 / (len(datasets) * len(people[row["dataset"]])
                            * len(classes[(row["dataset"], _subject(row))])
                            * cells[(row["dataset"], _subject(row), int(label))])
                        for row, label in zip(rows, labels, strict=True)])
    return weights / weights.sum()


def hierarchical_class_means(covariances, rows, labels) -> np.ndarray:
    """Within each class: trial -> participant -> source dataset equal means."""
    means = []
    for label in (0, 1):
        dataset_means = []
        for dataset in sorted({row["dataset"] for row in rows}):
            subject_means = []
            for subject in sorted({_subject(row) for row in rows if row["dataset"] == dataset}):
                indices = [i for i, row in enumerate(rows)
                           if row["dataset"] == dataset and _subject(row) == subject and labels[i] == label]
                if indices:
                    subject_means.append(covariances[indices].mean(axis=0))
            if not subject_means:
                raise ValueError("each source dataset must provide both classes")
            dataset_means.append(np.mean(subject_means, axis=0))
        means.append(np.mean(dataset_means, axis=0))
    return np.stack(means)


@dataclass(frozen=True)
class SourceTrainingInputs:
    """Caller-issued source split capability; no target/query role is accepted.

    held_out_dataset=None requires all four registered sources. An explicit
    held-out source requires exactly the other three. synthetic=True labels
    software fixtures honestly and is retained in every derived receipt.
    """
    x: np.ndarray
    y: np.ndarray
    rows: tuple[dict, ...]
    signal_contract: dict
    source_manifest_sha256: str
    held_out_dataset: str | None = None
    synthetic: bool = False

    def validate(self) -> tuple[np.ndarray, np.ndarray, str]:
        if self.held_out_dataset is not None and self.held_out_dataset not in SOURCE_IDS:
            raise PermissionError("LODO must hold out one registered source")
        expected = SOURCE_IDS - ({self.held_out_dataset} if self.held_out_dataset else set())
        # Validate permissions before inspecting any signal payload.
        if any(row.get("role") != "source_train" or row.get("dataset") not in expected for row in self.rows):
            raise PermissionError("classical source fitting accepts authorized source_train only")
        if {row["dataset"] for row in self.rows} != expected:
            raise ValueError("source training membership differs from declared full/LODO fold")
        signal_sha = validate_signal_contract(self.signal_contract)
        _sha(self.source_manifest_sha256, "source_manifest_sha256")
        x = _trials(self.x)
        y = np.asarray(self.y)
        if x.shape[2] != self.signal_contract["n_times"]:
            raise ValueError("source trials do not match the frozen signal window")
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or set(y.tolist()) != {0, 1}:
            raise ValueError("source fitting requires integer labels with both classes")
        if len(self.rows) != len(x) or any(not row.get("event_uid") for row in self.rows):
            raise ValueError("source event provenance is incomplete")
        for row, label in zip(self.rows, y, strict=True):
            _subject(row)
            if "label" in row and row["label"] != int(label):
                raise ValueError("source array label disagrees with its event provenance")
        uids = [row["event_uid"] for row in self.rows]
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate source event UID")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic flag must be explicit boolean")
        return x, y, signal_sha


@dataclass(frozen=True)
class ClassicalModel:
    method: str
    arrays: dict[str, np.ndarray]
    receipt: dict
    state_sha256: str

    def assert_unchanged(self) -> None:
        if digest({"method": self.method, "receipt": self.receipt,
                   "arrays": {key: array_sha256(value) for key, value in sorted(self.arrays.items())}}) != self.state_sha256:
            raise RuntimeError("classical model state or provenance was mutated")


def _model(method, arrays, receipt) -> ClassicalModel:
    arrays = {key: _readonly(value) for key, value in arrays.items()}
    identity = digest({"method": method, "receipt": receipt,
                       "arrays": {key: array_sha256(value) for key, value in sorted(arrays.items())}})
    return ClassicalModel(method, arrays, receipt, identity)


def _source_receipt(inputs, x, y, signal_sha, started) -> dict:
    return {
        "schema_version": 1, "implementation": IMPLEMENTATION_VERSION, "kind": "source_model",
        "implementation_sha256": file_digest(Path(__file__)),
        "synthetic": inputs.synthetic, "signal_contract_sha256": signal_sha,
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "held_out_dataset": inputs.held_out_dataset,
        "source_datasets": sorted({row["dataset"] for row in inputs.rows}),
        "source_event_uids_sha256": digest([row["event_uid"] for row in inputs.rows]),
        "source_rows_sha256": digest(inputs.rows), "source_x_sha256": array_sha256(x),
        "source_y_sha256": array_sha256(y), "source_trials": len(x),
        "n_times": int(x.shape[2]),
        "participants_by_source": {dataset: len({_subject(row) for row in inputs.rows if row["dataset"] == dataset})
                                   for dataset in sorted({row["dataset"] for row in inputs.rows})},
        "source_statistics_weighting": "dataset_equal_then_subject_equal_then_class_equal_then_trial_equal",
        "q_sha256": array_sha256(_Q), "wall_seconds": time.monotonic() - started,
        "query_accessed": False,
    }


def _canonical_subspace(eigenvectors) -> np.ndarray:
    """Deterministic coordinate basis for an exactly tied eigenspace."""
    projector = eigenvectors @ np.linalg.pinv(eigenvectors)
    result = []
    for candidate in projector.T:
        candidate = candidate.copy()
        for basis in result:
            candidate -= basis * np.dot(basis, candidate)
        length = np.linalg.norm(candidate)
        if length > 1e-10:
            result.append(candidate / length)
        if len(result) == eigenvectors.shape[1]:
            return np.stack(result, axis=1)
    raise ValueError("cannot construct deterministic tied eigenspace")


def csp_filters(class_means, m: int) -> tuple[np.ndarray, np.ndarray]:
    if m not in (1, 2, 3) or np.shape(class_means) != (2, 10, 10):
        raise ValueError("B4 expects two CAR10 class means and m in {1,2,3}")
    means = 0.9 * np.asarray(class_means) + 0.1 * np.eye(10)[None] / 10
    composite = means.sum(axis=0)
    values, vectors = eigh(means[1], composite, check_finite=True)
    # scipy returns ascending eigenvalues. Exact/roundoff ties share a fixed
    # coordinate-space Gram--Schmidt basis rather than arbitrary eigensolver axes.
    start = 0
    while start < 10:
        stop = start + 1
        while stop < 10 and abs(values[stop] - values[start]) <= 1e-12:
            stop += 1
        if stop - start > 1:
            block = _canonical_subspace(vectors[:, start:stop])
            for column in range(block.shape[1]):
                # Re-orthogonalize in the generalized metric.
                for previous in range(column):
                    block[:, column] -= block[:, previous] * (block[:, previous] @ composite @ block[:, column])
                block[:, column] /= np.sqrt(block[:, column] @ composite @ block[:, column])
            vectors[:, start:stop] = block
        start = stop
    for column in range(10):
        pivot = np.argmax(np.abs(vectors[:, column]))  # first index resolves ties
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    # Smallest ascending, then largest descending: order and signs are fixed.
    indices = list(range(m)) + list(range(9, 9 - m, -1))
    return vectors[:, indices].T, values[indices]


def csp_features(x, filters) -> np.ndarray:
    x = _trials(x)
    z = np.einsum("dc,nct->ndt", _Q, x.astype(np.float64)
                  / np.max(np.abs(x), axis=(1, 2))[:, None, None])
    projected = np.einsum("kd,ndt->nkt", filters, z)
    variance = np.mean(projected * projected, axis=-1)
    denominator = variance.sum(axis=1)
    if (variance <= 0).any() or (denominator <= 0).any() or not np.isfinite(variance).all():
        raise ValueError("CSP log-variance is undefined; no implicit epsilon")
    return np.log(variance / denominator[:, None])


def fit_b4_source(inputs: SourceTrainingInputs, *, m=1) -> ClassicalModel:
    if type(inputs) is not SourceTrainingInputs:
        raise PermissionError("source fitting requires SourceTrainingInputs capability")
    started = time.monotonic()
    x, y, signal_sha = inputs.validate()
    covariances = trace_covariances(x)
    class_means = hierarchical_class_means(covariances, inputs.rows, y)
    filters, eigenvalues = csp_filters(class_means, m)
    features = csp_features(x, filters)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(features, y)
    receipt = _source_receipt(inputs, x, y, signal_sha, started)
    receipt.update({"status": "SOURCE_CLASSICAL_COMPLETE", "method": "B4", "m": m,
                    "class_covariance_weighting": "within_class_dataset_then_subject_then_trial_equal",
                    "trial_covariance_shrink": 0.0, "class_mean_shrink": 0.1,
                    "source_classifier": "sklearn_LDA_lsqr_auto_priors_0.5_0.5",
                    "source_classifier_weighting": "unweighted_trials_legacy_recipe_sklearn_LDA_has_no_sample_weight",
                    "source_optimizer_updates": 0, "source_exact_solves": 1,
                    "source_fit_tasks": 1, "source_trainable_parameters": int(filters.size + classifier.coef_.size + 1)})
    return _model("B4", {"q": _Q, "class_means": class_means, "filters": filters,
                         "eigenvalues": eigenvalues, "coef": classifier.coef_.reshape(-1),
                         "bias": classifier.intercept_[0]}, receipt)


def fit_b5_source(inputs: SourceTrainingInputs, *, shrink=0.05) -> ClassicalModel:
    if type(inputs) is not SourceTrainingInputs:
        raise PermissionError("source fitting requires SourceTrainingInputs capability")
    if shrink not in (0.01, 0.05, 0.1):
        raise ValueError("B5 source shrink must match a registered grid value")
    started = time.monotonic()
    x, y, signal_sha = inputs.validate()
    weights = hierarchical_weights(inputs.rows, y)
    covariances = trace_covariances(x, shrink=shrink)
    reference = matrix_function(np.average(matrix_function(covariances, "log"), axis=0, weights=weights), "exp")
    inverse_root = matrix_function(reference, "inverse_root")
    features = tangent_features(covariances, inverse_root)
    scaler = StandardScaler().fit(features, sample_weight=weights)
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(scaler.transform(features), y, sample_weight=weights * len(y))
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in recorded)
    receipt = _source_receipt(inputs, x, y, signal_sha, started)
    receipt.update({"status": "SOURCE_CLASSICAL_COMPLETE" if converged else "SOURCE_CLASSICAL_NOT_CONVERGED",
                    "method": "B5", "shrink": shrink, "reference": "weighted_log_Euclidean_not_affine_invariant",
                    "source_classifier": "sklearn_logistic_C1_lbfgs_maxiter2000_weight_times_N",
                    "source_optimizer_updates": int(classifier.n_iter_[0]), "source_fit_tasks": 1,
                    "source_trainable_parameters": 56, "converged": converged})
    return _model("B5", {"q": _Q, "reference": reference, "inverse_root": inverse_root,
                         "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                         "coef": classifier.coef_.reshape(-1), "bias": classifier.intercept_[0]}, receipt)


def _authorized_source(source: ClassicalModel, method: str) -> None:
    if type(source) is not ClassicalModel or source.method != method:
        raise ValueError("incorrect source model type")
    source.assert_unchanged()
    if source.receipt.get("kind") != "source_model" or source.receipt.get("status") != "SOURCE_CLASSICAL_COMPLETE":
        raise ValueError("source model must have a complete source-fit receipt")
    # A source LODO model cannot serve as the deployed all-source target prior.
    if source.receipt.get("held_out_dataset") is not None or set(source.receipt.get("source_datasets", [])) != SOURCE_IDS:
        raise ValueError("target calibration requires the all-four-source prior")


def _support(source: ClassicalModel, support: CalibrationPrefix, *, allowed_role: str):
    if type(support) is not CalibrationPrefix:
        raise PermissionError("target fitting accepts CalibrationPrefix only")
    # Dataclasses contain mutable tensors, so revalidate at the actual fit edge.
    support.__post_init__()
    if support.role != allowed_role:
        raise PermissionError(f"this fitting entry accepts {allowed_role} only")
    if support.protocol_freeze_sha != source.receipt["signal_contract_sha256"]:
        raise ValueError("target and source signal versions differ")
    x = _trials(_numpy(support.x), allow_empty=True)
    if x.shape[2] != source.receipt["n_times"]:
        raise ValueError("support and frozen source window lengths differ")
    return x, _numpy(support.y)


def _target_receipt(source, support, started, **extra):
    development = support.role == "meta_validation_support"
    return {"schema_version": 1, "implementation": IMPLEMENTATION_VERSION,
            "kind": "development_model" if development else "target_model",
            "implementation_sha256": file_digest(Path(__file__)),
            "support_role": support.role,
            "fit_purpose": "development_model_selection" if development else "target_inference",
            "method": source.method, "synthetic": source.receipt["synthetic"],
            "source_state_sha256": source.state_sha256,
            "signal_contract_sha256": support.protocol_freeze_sha,
            "n_times": source.receipt["n_times"],
            "acquired_event_uids": list(support.acquired_event_uids),
            "support_event_uids": list(support.event_uids), "acquired_trials": len(support.acquired_event_uids),
            "valid_support_trials": len(support.event_uids), "query_accessed": False,
            "wall_seconds": time.monotonic() - started, **extra}


def _fallback(source, support, started, **config):
    receipt = _target_receipt(source, support, started, status="TARGET_FALLBACK",
                              fallback_reason="insufficient_class_support", target_optimizer_updates=0,
                              target_trainable_parameters=0, **config)
    return _model(source.method, source.arrays, receipt)


def fit_b4_target(source: ClassicalModel, support: CalibrationPrefix, *, alpha=0.5, m=None) -> ClassicalModel:
    """Public target fitting accepts only the acquired calibration_prefix."""
    return _fit_b4_support(source, support, alpha=alpha, m=m, allowed_role="calibration_prefix")


def fit_b4_development(source: ClassicalModel, support: CalibrationPrefix, *, alpha=0.5, m=None) -> ClassicalModel:
    """Validation-grid calibration from its support; no query labels accepted."""
    return _fit_b4_support(source, support, alpha=alpha, m=m, allowed_role="meta_validation_support")


def _fit_b4_support(source, support, *, alpha, m, allowed_role):
    _authorized_source(source, "B4")
    m = source.receipt["m"] if m is None else m
    if (alpha, m) not in B4_GRID or m != source.receipt["m"]:
        raise ValueError("B4 target grid must match its frozen source m")
    started = time.monotonic()
    x, y = _support(source, support, allowed_role=allowed_role)
    if set(y.tolist()) != {0, 1}:
        return _fallback(source, support, started, alpha=alpha, m=m)
    covariance = trace_covariances(x)
    target_means = np.stack([covariance[y == label].mean(axis=0) for label in (0, 1)])
    mixed = (1 - alpha) * target_means + alpha * source.arrays["class_means"]
    filters, eigenvalues = csp_filters(mixed, m)
    features = csp_features(x, filters)
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]).fit(features, y)
    except (ValueError, np.linalg.LinAlgError) as error:
        # Do not exclude/replace the episode or invent another class estimator.
        receipt = _target_receipt(source, support, started, status="TARGET_FIT_FAILED", alpha=alpha, m=m,
                                  fallback_reason="lda_numerical_fit_failed", failure_type=type(error).__name__,
                                  target_optimizer_updates=0, target_trainable_parameters=0)
        return _model("B4", source.arrays, receipt)
    receipt = _target_receipt(source, support, started, status="TARGET_FIT_COMPLETE", alpha=alpha, m=m,
                              target_optimizer_updates=0, target_exact_solves=1,
                              target_trainable_parameters=int((filters.size if alpha < 1 else 0) + 2 * m + 1),
                              parameter_count_definition="target_dependent_filter_scalars_plus_LDA_margin_parameters",
                              target_gradient_trainable_parameters=0, target_covariance_used=alpha < 1,
                              fallback_reason=None,
                              target_classifier="sklearn_LDA_lsqr_auto_priors_0.5_0.5",
                              class_mean_shrink=0.1, fit_warning_categories=[item.category.__name__ for item in recorded])
    return _model("B4", {"q": _Q, "filters": filters, "eigenvalues": eigenvalues,
                         "coef": classifier.coef_.reshape(-1), "bias": classifier.intercept_[0]}, receipt)


def balanced_logistic_objective(parameters, features, y, source_parameters, l2):
    """Analytic value/gradient, bias included in the /(q+1) source penalty."""
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(y)
    parameters = np.asarray(parameters, dtype=np.float64)
    source_parameters = np.asarray(source_parameters, dtype=np.float64)
    if (features.ndim != 2 or parameters.shape != (features.shape[1] + 1,)
            or source_parameters.shape != parameters.shape or y.shape != (len(features),)
            or set(y.tolist()) != {0, 1} or l2 < 0):
        raise ValueError("invalid class-balanced logistic objective inputs")
    weights = np.array([0.5 / np.count_nonzero(y == label) for label in y])
    margin = features @ parameters[:-1] + parameters[-1]
    delta = parameters - source_parameters
    residual = weights * (expit(margin) - y)
    value = np.dot(weights, np.logaddexp(0, margin) - y * margin) + l2 * np.dot(delta, delta) / len(delta)
    gradient = np.concatenate((features.T @ residual, [residual.sum()])) + 2 * l2 * delta / len(delta)
    return float(value), gradient


def fit_b5_target(source: ClassicalModel, support: CalibrationPrefix, *, l2=0.1) -> ClassicalModel:
    """Public target fitting accepts only the acquired calibration_prefix."""
    return _fit_b5_support(source, support, l2=l2, allowed_role="calibration_prefix")


def fit_b5_development(source: ClassicalModel, support: CalibrationPrefix, *, l2=0.1) -> ClassicalModel:
    """Validation-grid calibration from its support; no query labels accepted."""
    return _fit_b5_support(source, support, l2=l2, allowed_role="meta_validation_support")


def _fit_b5_support(source, support, *, l2, allowed_role):
    _authorized_source(source, "B5")
    shrink = source.receipt["shrink"]
    if (shrink, l2) not in B5_GRID:
        raise ValueError("B5 target hyperparameters must match the registered grid")
    started = time.monotonic()
    x, y = _support(source, support, allowed_role=allowed_role)
    if set(y.tolist()) != {0, 1}:
        return _fallback(source, support, started, shrink=shrink, l2=l2)
    covariance = trace_covariances(x, shrink=shrink)
    features = (tangent_features(covariance, source.arrays["inverse_root"])
                - source.arrays["scaler_mean"]) / source.arrays["scaler_scale"]
    initial = np.concatenate((source.arrays["coef"], [source.arrays["bias"].item()]))
    result = minimize(balanced_logistic_objective, initial.copy(),
                      args=(features, y, initial, l2), jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "gtol": 1e-8, "ftol": 0.0, "maxls": 50})
    gradient_inf = float(np.max(np.abs(result.jac)))
    converged = bool(result.success and gradient_inf <= 1e-8)
    receipt = _target_receipt(source, support, started, status="TARGET_FIT_COMPLETE", shrink=shrink, l2=l2,
                              target_optimizer_updates=int(result.nit), function_evaluations=int(result.nfev),
                              target_trainable_parameters=56, fallback_reason=None, converged=converged,
                              optimizer_success=bool(result.success), optimizer_status=int(result.status),
                              optimizer_message=str(result.message), gradient_inf=gradient_inf,
                              optimizer="scipy_L-BFGS-B_unconstrained", max_iterations=200,
                              gradient_tolerance=1e-8, function_tolerance=0.0,
                              objective="class_balanced_BCE_plus_lambda_source_L2_including_bias_over_q_plus_1",
                              final_objective=float(result.fun), source_scaler_frozen=True)
    arrays = dict(source.arrays)
    arrays.update(coef=result.x[:-1], bias=result.x[-1])
    source.assert_unchanged()
    return _model("B5", arrays, receipt)


def predict_classical(model: ClassicalModel, inputs: PredictionInputs) -> dict:
    """Pure per-trial prediction; never estimates reference/scaler from query."""
    if type(inputs) is not PredictionInputs:
        raise PermissionError("prediction requires an unlabeled PredictionInputs capability")
    inputs.__post_init__()
    model.assert_unchanged()
    if model.receipt["signal_contract_sha256"] != inputs.protocol_freeze_sha:
        raise ValueError("prediction and fitted model signal versions differ")
    if set(inputs.event_uids) & set(model.receipt.get("acquired_event_uids", [])):
        # Includes every acquired prefix UID, even failed-QC events absent from
        # the valid support. Applies equally to a fit, failed fit, and fallback.
        raise ValueError("query overlaps the acquired calibration prefix")
    x = _trials(_numpy(inputs.x))
    if x.shape[2] != model.receipt["n_times"]:
        raise ValueError("prediction and frozen model window lengths differ")
    if model.method == "B4":
        features = csp_features(x, model.arrays["filters"])
    elif model.method == "B5":
        covariance = trace_covariances(x, shrink=model.receipt["shrink"])
        features = (tangent_features(covariance, model.arrays["inverse_root"])
                    - model.arrays["scaler_mean"]) / model.arrays["scaler_scale"]
    else:
        raise ValueError("unsupported classical model")
    margin = features @ model.arrays["coef"] + model.arrays["bias"]
    logits = np.stack((np.zeros_like(margin), margin), axis=1)
    return {"event_uids": inputs.event_uids, "logits": logits, "probability": softmax(logits, axis=1),
            "prediction": np.argmax(logits, axis=1), "model_state_sha256": model.state_sha256}


def save_classical_model(model: ClassicalModel, directory) -> dict:
    """Persist a new immutable numerical artifact with full per-file identities."""
    model.assert_unchanged()
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("model output directory must not already exist")
    directory.mkdir(parents=True)
    files = []
    for key, value in sorted(model.arrays.items()):
        path = directory / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        files.append({"path": path.name, "sha256": file_digest(path), "bytes": path.stat().st_size})
    receipt = {"schema_version": 1, "method": model.method, "state_sha256": model.state_sha256,
               "fit_receipt": model.receipt, "files": files, "array_format": "NPY_allow_pickle_false"}
    (directory / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return receipt


def load_classical_model(directory, *, expected_receipt_sha256: str) -> ClassicalModel:
    """Restore only with an independently bound receipt SHA, without pickle."""
    _sha(expected_receipt_sha256, "expected_receipt_sha256")
    directory = Path(directory)
    path = directory / "receipt.json"
    if path.is_symlink() or file_digest(path) != expected_receipt_sha256:
        raise ValueError("model receipt SHA mismatch or symlink")
    receipt = json.loads(path.read_text())
    arrays = {}
    for entry in receipt["files"]:
        name = entry["path"]
        if not re.fullmatch(r"[a-z_]+\.npy", name):
            raise ValueError("unsafe numerical artifact path")
        path = directory / name
        if path.is_symlink() or path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError("numerical artifact hash/size mismatch or symlink")
        if name[:-4] in arrays:
            raise ValueError("duplicate model artifact")
        arrays[name[:-4]] = np.load(path, allow_pickle=False)
    model = _model(receipt["method"], arrays, receipt["fit_receipt"])
    if model.state_sha256 != receipt["state_sha256"]:
        raise ValueError("restored model state identity mismatch")
    return model
