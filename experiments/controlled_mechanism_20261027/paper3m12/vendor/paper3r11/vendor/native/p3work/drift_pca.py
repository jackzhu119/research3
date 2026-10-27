"""Source drift-PCA software scaffolding; no scientific estimator is enabled.

METHOD_SPEC §11 specifies class-balanced session covariance logs, within-person
centering, trace removal and Frobenius PCA on Stieger meta_train40. It does not
freeze the choices listed in UNRESOLVED_FIELDS. This module supplies explicitly
configured software math, a pending parameter contract and real input gates.
It never emits a scientific PCA checkpoint/COMPLETE receipt, writes artifacts,
downloads data, accesses evaluation labels, or starts GPU/R2 work.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import digest, file_digest, validate_signal_contract
from .spatial_geometry import helmert_rows, traceless_basis


SCHEMA = "paper3-drift-pca-parameter-contract-v1"
SOFTWARE_SCOPE = "SOFTWARE_TEST_DOUBLE_DRIFT_PCA_NOT_RESEARCH_EVIDENCE"
UNRESOLVED_FIELDS = (
    "covariance_coordinates", "trial_covariance_normalization", "covariance_shrinkage", "shrinkage_stage",
    "class_balance", "session_trial_scope", "single_class_session_policy", "within_subject_centering",
    "pca_observation_weighting", "global_center_after_subject_centering", "insufficient_subject_sessions_policy",
)


class DriftPCAError(ValueError):
    pass


def pending_parameter_contract():
    """No B5/Rref shrinkage, class/session weight or coordinate default is borrowed."""
    return {"schema_version": SCHEMA, "status": "UNRESOLVED_NOT_REGISTERED", "dataset": "Stieger2021",
            "role": "meta_train", "participant_count": 40, "generator_dimensions": [2, 4, 8],
            "synthetic": False, "parameters": {name: None for name in UNRESOLVED_FIELDS},
            "input_contract_sha256": None, "split_sha256": None, "implementation_amendment": None,
            "scientific_estimation_started": False}


def unresolved_parameters(contract):
    parameters = contract.get("parameters", {})
    return tuple(name for name in UNRESOLVED_FIELDS if parameters.get(name) is None)


def validate_parameters(contract, *, formal=False):
    if (contract.get("schema_version") != SCHEMA or contract.get("dataset") != "Stieger2021"
            or contract.get("role") != "meta_train" or contract.get("participant_count") != 40
            or contract.get("generator_dimensions") != [2, 4, 8]):
        raise DriftPCAError("Drift-PCA is limited to the registered Stieger meta_train40/dimension grid")
    missing = unresolved_parameters(contract)
    if missing:
        raise DriftPCAError("Drift-PCA implementation parameters remain unresolved: " + ", ".join(missing))
    value = contract["parameters"]
    choices = {
        "covariance_coordinates": {"car10", "source_whitened_car10"},
        "trial_covariance_normalization": {"per_trial_trace_after_coordinate_map"},
        "shrinkage_stage": {"per_trial", "after_balanced_session_mean"},
        "class_balance": {"equal_class_then_trial"},
        "session_trial_scope": {"all_valid_lr_trials_of_session", "B20_support_and_eligible_future_query"},
        "single_class_session_policy": {"error_stop"},
        "within_subject_centering": {"equal_sessions", "valid_trial_count"},
        "pca_observation_weighting": {"subject_equal_then_session_equal", "session_equal"},
        "insufficient_subject_sessions_policy": {"error_stop"},
    }
    for name, allowed in choices.items():
        if value[name] not in allowed:
            raise DriftPCAError(f"Unimplemented explicit drift-PCA parameter: {name}")
    if (type(value["covariance_shrinkage"]) not in (float, int)
            or not np.isfinite(value["covariance_shrinkage"]) or not 0 <= value["covariance_shrinkage"] < 1
            or type(value["global_center_after_subject_centering"]) is not bool):
        raise DriftPCAError("Explicit finite shrinkage in [0,1) and explicit global-centering policy required")
    if formal:
        if contract.get("status") != "FROZEN_IMPLEMENTATION_CONTRACT" or contract.get("synthetic") is not False:
            raise DriftPCAError("Formal drift-PCA needs an independently registered non-synthetic implementation contract")
        from .episodes import _sha
        _sha(contract.get("input_contract_sha256"))
        _sha(contract.get("split_sha256"))
        if not isinstance(contract.get("implementation_amendment"), dict):
            raise DriftPCAError("The under-specified protocol requires a SHA-bound implementation amendment")
    elif contract.get("status") != "SOFTWARE_TEST_DOUBLE_PARAMETER_CONTRACT" or contract.get("synthetic") is not True:
        raise DriftPCAError("Unregistered PCA mathematics is exposed only as an explicit SOFTWARE_TEST_DOUBLE")
    return digest(contract)


def preparation_plan(root=None, *, parameter_contract=None):
    contract = pending_parameter_contract() if parameter_contract is None else parameter_contract
    authority = {"method": "protocol/METHOD_SPEC.md §11 Source drift-PCA exp",
                 "work": "protocol/WORK_TASKS.md W5/W6"}
    if root is not None:
        root = Path(root).resolve(strict=True)
        authority["method_file_sha256"] = file_digest(root / "protocol/METHOD_SPEC.md")
        authority["work_file_sha256"] = file_digest(root / "protocol/WORK_TASKS.md")
    missing = unresolved_parameters(contract)
    return {"status": "BLOCKED_PENDING_DRIFT_PCA_PARAMETER_FREEZE" if missing else "BLOCKED_FORMAL_ESTIMATOR_NOT_IMPLEMENTED",
            "authority": authority,
            "parameter_contract": contract, "unresolved_parameters": list(unresolved_parameters(contract)),
            "scientific_estimation_started": False, "scientific_PCA_estimated": False,
            "formal_estimator_implemented": False, "writes": False, "gpu_execution": False,
            "minimum_required_evidence": ["frozen implementation amendment resolving the listed choices",
                "real W1 and complete W3 meta_train40 cache/role/window audit",
                "verified matching four-source f0 and Rref receipts"]}


@dataclass(frozen=True)
class VerifiedDriftInputs:
    """Verified metadata/capabilities only; construction does not estimate PCA."""
    catalog: object
    reference: object
    parameter_contract_sha256: str
    input_contract_sha256: str
    split_sha256: str
    implementation_amendment_sha256: str
    source_f0_receipt_sha256: str
    reference_receipt_sha256: str
    scientific_PCA_estimated: bool = False


def verify_drift_inputs(root, parameter_contract, *, source_receipt, reference_receipt):
    """Read only authorized meta_train data, after parameter/stage/receipt gates.

    This is an input verifier, not a formal estimator. The scientific fitting
    adapter remains pending until the scientific parameter amendment exists.
    """
    contract_sha = validate_parameters(parameter_contract, formal=True)
    from .episodes import _artifact, _local, load_episode_catalog
    root = Path(root).resolve(strict=True)
    tasks = json.loads(_local(root, "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE" or tasks.get("W3", {}).get("status") != "COMPLETE":
        raise DriftPCAError("Real complete W1 and W3 are required before drift-PCA input access")
    signal = json.loads(_local(root, "contracts/input_contract.json").read_text())
    signal_sha = validate_signal_contract(signal)
    if (parameter_contract["input_contract_sha256"] != signal_sha
            or parameter_contract["split_sha256"] != signal["window_audit_receipt"]["split_sha256"]):
        raise DriftPCAError("PCA parameters and the frozen input/split differ")
    amendment_file = _artifact(root, parameter_contract["implementation_amendment"], suffix=".json")
    amendment = json.loads(amendment_file.read_text())
    if (amendment.get("status") != "DRIFT_PCA_IMPLEMENTATION_AMENDMENT_FROZEN"
            or amendment.get("parameter_values_sha256") != digest(parameter_contract["parameters"])
            or amendment.get("input_contract_sha256") != signal_sha
            or amendment.get("split_sha256") != parameter_contract["split_sha256"]
            or amendment.get("allowed_dataset") != "Stieger2021" or amendment.get("allowed_role") != "meta_train"
            or amendment.get("protocol_method_sha256") != file_digest(root / "protocol/METHOD_SPEC.md")
            or amendment.get("protocol_work_tasks_sha256") != file_digest(root / "protocol/WORK_TASKS.md")
            or amendment.get("registered_before_internal_or_external_query") is not True):
        raise DriftPCAError("An explicit protocol-bound drift-PCA implementation amendment is required")
    if not source_receipt or not reference_receipt:
        raise DriftPCAError("Matching source f0 and source-only Rref receipts are required")
    cache_manifest = json.loads(_artifact(root, tasks["W3"].get("cache_manifest"), suffix=".json").read_text())
    if (cache_manifest.get("synthetic") is not False or any(cache_manifest.get(key)
            for key in ("test_double_not_research_evidence", "fixture_scope", "test_double"))):
        raise DriftPCAError("Real W3 cache evidence is required; software-double statuses cannot issue formal PCA inputs")
    from .source_reference_io import load_verified_reference
    # Its implementation rechecks real W1/W2 source caches, actual f0 receipt,
    # source UID allowlist and numerical reference artifacts; no source fit.
    reference, _ = load_verified_reference(root, reference_receipt, source_receipt, signal)
    catalog = load_episode_catalog(root, role="meta_train", purpose="meta_gradient")
    if (catalog.role != "meta_train" or len(catalog.partition_people) != 40
            or catalog.input_contract_sha256 != signal_sha or catalog.split_sha256 != parameter_contract["split_sha256"]):
        raise DriftPCAError("Drift-PCA requires exactly the frozen authorized meta_train40")
    return VerifiedDriftInputs(catalog, reference, contract_sha, signal_sha, catalog.split_sha256,
        file_digest(amendment_file), file_digest(_local(root, source_receipt)),
        file_digest(_local(root, reference_receipt)))


def _software_only(software_test_double):
    if software_test_double is not True:
        raise PermissionError("PCA math is unregistered SOFTWARE_TEST_DOUBLE only; no scientific estimator is enabled")


def symmetric_log(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (10, 10) or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise DriftPCAError("Finite symmetric CAR10 covariance required")
    values, vectors = np.linalg.eigh(matrix)
    if (values <= 0).any():
        raise DriftPCAError("Session covariance is not SPD; no eigenvalue floor or implicit shrinkage")
    return (vectors * np.log(values)) @ vectors.T


def session_covariance_math(x, y, parameter_contract, *, inverse_source_root=None, software_test_double=False):
    """Explicitly configured class-balanced covariance, synthetic math only."""
    _software_only(software_test_double)
    validate_parameters(parameter_contract)
    p = parameter_contract["parameters"]
    x, y = np.asarray(x), np.asarray(y)
    if (x.ndim != 3 or x.shape[1] != 11 or x.shape[2] < 2 or y.shape != (len(x),)
            or y.dtype.kind not in "iu" or set(y.tolist()) != {0, 1} or not np.isfinite(x).all()):
        raise DriftPCAError("Valid LR synthetic trials with both classes are required; no missing-class substitution")
    scale = np.max(np.abs(x), axis=(1, 2))
    if (scale <= 0).any():
        raise DriftPCAError("Zero-energy synthetic trial")
    z = np.einsum("dc,nct->ndt", helmert_rows().numpy(), x.astype(np.float64) / scale[:, None, None])
    if p["covariance_coordinates"] == "source_whitened_car10":
        root = np.asarray(inverse_source_root, dtype=np.float64)
        if root.shape != (10, 10) or not np.isfinite(root).all() or not np.allclose(root, root.T):
            raise DriftPCAError("Explicit finite symmetric source inverse root required for whitened coordinates")
        if np.linalg.eigvalsh(root).min() <= 0:
            raise DriftPCAError("Source inverse root must be SPD")
        z = root @ z
    elif inverse_source_root is not None:
        raise DriftPCAError("Unwhitened coordinate recipe cannot silently consume a source whitening matrix")
    covariance = z @ np.swapaxes(z, -1, -2)
    trace = np.trace(covariance, axis1=1, axis2=2)
    if (trace <= 0).any() or not np.isfinite(trace).all():
        raise DriftPCAError("Degenerate CAR-space trial")
    covariance /= trace[:, None, None]
    shrink = p["covariance_shrinkage"]
    if p["shrinkage_stage"] == "per_trial":
        covariance = (1 - shrink) * covariance + shrink * np.eye(10)[None] / 10
    balanced = sum(covariance[y == label].mean(axis=0) for label in (0, 1)) / 2
    if p["shrinkage_stage"] == "after_balanced_session_mean":
        balanced = (1 - shrink) * balanced + shrink * np.eye(10) / 10
    symmetric_log(balanced)  # Fail at a singular no-shrink recipe, rather than silently changing it.
    return balanced


@dataclass(frozen=True)
class SoftwareSessionCovariance:
    participant_id: str
    session_id: str
    covariance: np.ndarray
    valid_trial_count: int
    dataset: str = "Stieger2021"
    role: str = "meta_train"
    scope: str = SOFTWARE_SCOPE


@dataclass(frozen=True)
class DriftPCAMathResult:
    generators: np.ndarray
    eigenvalues: np.ndarray
    centered_traceless_logs: np.ndarray
    observation_weights: np.ndarray
    participant_ids: tuple[str, ...]
    session_keys: tuple[tuple[str, str], ...]
    parameter_contract_sha256: str
    status: str = "SOFTWARE_TEST_DOUBLE_MATH_ONLY"
    synthetic: bool = True
    scientific_PCA_estimated: bool = False


def _canonical_tied_space(vectors):
    projector = vectors @ vectors.T
    columns = []
    for candidate in projector.T:
        candidate = candidate.copy()
        for previous in columns:
            candidate -= previous * np.dot(previous, candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-10:
            columns.append(candidate / norm)
        if len(columns) == vectors.shape[1]:
            return np.stack(columns, axis=1)
    raise DriftPCAError("Cannot identify a deterministic tied PCA subspace")


def fit_drift_pca_math(sessions, parameter_contract, *, rank, software_test_double=False):
    """Within-person centered symmetric traceless Frobenius PCA, software only.

    No source/f0/Rref estimator or real-data receipt is emitted. Eigenvalue ties
    use fixed-coordinate projected Gram--Schmidt, then positive max-entry sign.
    """
    _software_only(software_test_double)
    contract_sha = validate_parameters(parameter_contract)
    if type(rank) is not int or rank not in {2, 4, 8}:
        raise DriftPCAError("The registered generator coefficient dimensions are {2,4,8}")
    sessions = tuple(sessions)
    if not sessions:
        raise DriftPCAError("Explicit synthetic session covariance observations required")
    if any(type(s) is not SoftwareSessionCovariance or s.dataset != "Stieger2021" or s.role != "meta_train"
           or s.scope != SOFTWARE_SCOPE for s in sessions):
        raise PermissionError("Neither validation/internal/external nor unlabelled real session data can enter the software PCA kernel")
    keys = tuple((s.participant_id, s.session_id) for s in sessions)
    if len(set(keys)) != len(keys) or any(not p or not s for p, s in keys):
        raise DriftPCAError("Unique participant/session identities required")
    if any(type(s.valid_trial_count) is not int or s.valid_trial_count <= 0 for s in sessions):
        raise DriftPCAError("Each session needs its explicit valid-trial count")
    groups = {person: np.array([i for i, s in enumerate(sessions) if s.participant_id == person])
              for person in sorted({s.participant_id for s in sessions})}
    if any(len(indices) < 2 for indices in groups.values()):
        raise DriftPCAError("Explicit prototype policy stops when any person lacks two session observations")
    p = parameter_contract["parameters"]
    logs = np.stack([symmetric_log(s.covariance) for s in sessions])
    centered, weights = np.empty_like(logs), np.empty(len(sessions))
    for indices in groups.values():
        center_weights = (np.ones(len(indices)) if p["within_subject_centering"] == "equal_sessions"
                          else np.array([sessions[i].valid_trial_count for i in indices]))
        centered[indices] = logs[indices] - np.average(logs[indices], axis=0, weights=center_weights)
        weights[indices] = (1 / (len(groups) * len(indices)) if p["pca_observation_weighting"] == "subject_equal_then_session_equal"
                            else 1 / len(sessions))
    centered -= np.trace(centered, axis1=1, axis2=2)[:, None, None] * np.eye(10)[None] / 10
    if p["global_center_after_subject_centering"]:
        centered -= np.average(centered, axis=0, weights=weights)
    # E_s is orthonormal for the FULL Frobenius metric, so off-diagonal terms
    # receive exactly their two-entry norm, never an unscaled upper triangle.
    basis = traceless_basis(10, kind="symmetric").numpy()
    features = centered.reshape(len(sessions), 100) @ basis
    scatter = (features * weights[:, None]).T @ features
    eigenvalues, vectors = np.linalg.eigh(scatter)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if eigenvalues[0] <= 0:
        raise DriftPCAError("No observed within-person drift; directions are not invented")
    # A relative numerical-rank check, not an undocumented absolute EEG/drift
    # amplitude cutoff. Physical/effect thresholds would need separate freezing.
    threshold = np.finfo(float).eps * float(eigenvalues[0]) * len(eigenvalues) * 32
    if eigenvalues[rank - 1] <= threshold:
        raise DriftPCAError("Insufficient observed drift dimension; zero/random generators are not invented")
    start = 0
    while start < len(eigenvalues):
        stop = start + 1
        while stop < len(eigenvalues) and abs(eigenvalues[stop] - eigenvalues[start]) <= threshold:
            stop += 1
        if stop - start > 1:
            vectors[:, start:stop] = _canonical_tied_space(vectors[:, start:stop])
        start = stop
    generators = (basis @ vectors[:, :rank]).T.reshape(rank, 10, 10)
    for generator in generators:
        flat = generator.reshape(-1)
        if flat[np.argmax(np.abs(flat))] < 0:
            generator *= -1
    return DriftPCAMathResult(generators, eigenvalues[:rank], centered, weights,
                             tuple(groups), keys, contract_sha)
