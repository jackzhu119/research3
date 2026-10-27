"""METHOD_SPEC §§5/6/7/11/12 diagnostics on explicitly supplied CPU arrays.

These functions neither load weights/data nor establish that a dictionary was
learned, source-only, or useful. They do not compute performance/clinical
conclusions. Numerical validation uses the geometry module's 1e-10 tolerance;
boundary comparisons use a reported float64 roundoff scale, not a scientific
"near boundary" threshold. All returned matrix ranks are numerical SVD ranks.
"""
from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor

from .spatial_geometry import helmert_rows, spd_roots


SCOPE = "SUPPLIED_ARRAY_MATHEMATICS_ONLY"
NUMERICAL_ATOL = 1e-10  # Existing SpatialGeometry validation tolerance.


def _tensor(value, name, ndim):
    if (not isinstance(value, Tensor) or value.ndim != ndim or value.dtype != torch.float64 or
            value.device.type != "cpu" or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite float64 CPU tensor of dimension {ndim}")
    return value.detach()


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or
            not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _columns(value, name):
    value = _tensor(value, name, 2)
    n, r = value.shape
    if not 1 <= r <= n:
        raise ValueError(f"{name} must have nonempty orthonormal basis columns")
    gram = value.T @ value
    if not torch.allclose(gram, torch.eye(r, dtype=torch.float64), atol=NUMERICAL_ATOL, rtol=NUMERICAL_ATOL):
        raise ValueError(f"{name} basis columns must already be orthonormal; no silent QR/rank reduction")
    return value


def _generators(value):
    value = _tensor(value, "generators", 3)
    r, d, width = value.shape
    if d < 2 or width != d or not 1 <= r <= d * d - 1:
        raise ValueError("A nonempty r x d x d traceless dictionary is required")
    traces = value.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not torch.allclose(traces, torch.zeros_like(traces), atol=NUMERICAL_ATOL, rtol=0):
        raise ValueError("Generators must be traceless")
    _columns(value.flatten(1).T, "vectorized generators")
    return value


def _scope():
    return {"scope": SCOPE, "data_or_learned_weight_provenance_verified": False,
            "real_mechanism_conclusion": False}


def _rank(value, *, subtraction_reference_scale=None):
    singular_values = torch.linalg.svdvals(value)
    rtol = max(value.shape) * torch.finfo(torch.float64).eps
    cutoff = float(singular_values[0]) * rtol
    if subtraction_reference_scale is not None:
        cutoff = max(cutoff, subtraction_reference_scale * rtol)
    return {"numerical_rank": int((singular_values > cutoff).sum()), "absolute_cutoff": cutoff,
            "relative_tolerance": rtol, "subtraction_reference_scale": subtraction_reference_scale,
            "definition": "count(sigma > max(sigma_max, supplied_pre_subtraction_operator_scale) * max(shape) * float64_eps)"}


def generator_energy(generators):
    """Whole-dictionary energy; invariant to sign/permutation/orthogonal mixing.

    Frobenius squared norms of sym(G) and skew(G) add to total=r. Individual
    G_k labels are deliberately not interpreted as cross-seed components.
    """
    g = _generators(generators)
    symmetric = (g + g.mT) / 2
    skew = (g - g.mT) / 2
    sym_energy, skew_energy = float(symmetric.square().sum()), float(skew.square().sum())
    total = sym_energy + skew_energy
    return {**_scope(), "coefficient_count_r": g.shape[0], "effective_dimension_d": g.shape[1],
        "symmetric_energy": sym_energy, "skew_energy": skew_energy, "total_energy": total,
        "symmetric_fraction": sym_energy / total, "skew_fraction": skew_energy / total,
        "definition": "sum_k ||sym(G_k)||_F^2 and sum_k ||skew(G_k)||_F^2",
        "generator_orthogonality_max_abs_error": float((g.flatten(1) @ g.flatten(1).T -
            torch.eye(g.shape[0], dtype=torch.float64)).abs().max()), "input_numerical_atol": NUMERICAL_ATOL}


def subspace_principal_angles(left_basis_columns, right_basis_columns):
    """Compare columns in the SAME explicit ambient coordinate frame.

    Supports unequal dimensions. Principal angles number min(r_left,r_right);
    projection distance additionally detects unmatched dimensions. No alignment
    of individual columns or silent QR preprocessing is performed.
    """
    left = _columns(left_basis_columns, "left")
    right = _columns(right_basis_columns, "right")
    if left.shape[0] != right.shape[0]:
        raise ValueError("Subspaces must share the same ambient coordinate dimension")
    small, large = (left, right) if left.shape[1] <= right.shape[1] else (right, left)
    u, cosine, _ = torch.linalg.svd(small.T @ large, full_matrices=False)
    cosine = cosine.clamp(0, 1)
    principal_vectors = small @ u
    residual = principal_vectors - large @ (large.T @ principal_vectors)
    sine = torch.linalg.vector_norm(residual, dim=0).clamp(0, 1)
    # atan2 avoids arccos' loss of precision near identical subspaces.
    angles = torch.atan2(sine, cosine)
    left_projector, right_projector = left @ left.T, right @ right.T
    return {**_scope(), "ambient_dimension": left.shape[0], "left_dimension": left.shape[1],
        "right_dimension": right.shape[1], "unmatched_dimensions": abs(left.shape[1] - right.shape[1]),
        "principal_angles_radians": angles.tolist(), "principal_angle_cosines": cosine.tolist(),
        "projector_frobenius_distance": float(torch.linalg.vector_norm(left_projector - right_projector)),
        "squared_subspace_overlap": float(cosine.square().sum()),
        "left_projector": left_projector, "right_projector": right_projector,
        "invariance": "basis sign, permutation and orthogonal column rotation",
        "input_numerical_atol": NUMERICAL_ATOL}


def dictionary_subspace_diagnostics(left_generators, right_generators):
    """Learned/seed/PCA arrays share row-major vec(d x d) ambient coordinates.

    This common frame also permits symmetric PCA versus unrestricted G. It
    does not establish that either input was learned or came from permitted data.
    """
    left, right = _generators(left_generators), _generators(right_generators)
    if left.shape[1:] != right.shape[1:]:
        raise ValueError("Generator matrix coordinate frames must have identical dimensions")
    result = subspace_principal_angles(left.flatten(1).T, right.flatten(1).T)
    result["ambient_coordinate_definition"] = "row_major_vec_of_same_d_by_d_matrix_coordinates"
    return result


def coefficient_diagnostics(generators, theta, rho):
    """Actual M Frobenius/operator norms and coefficient-vector boundary counts.

    theta may be r or N x r. Rates weight supplied coefficient vectors equally,
    not participants/sessions. The raw radius gap and exact equality count are
    returned alongside a roundoff-only comparison eps*r*max(1,rho).
    """
    g = _generators(generators)
    if not isinstance(theta, Tensor) or theta.ndim not in {1, 2}:
        raise ValueError("theta must be one coefficient vector or a nonempty row matrix")
    values = _tensor(theta, "theta", theta.ndim)
    values = values[None] if values.ndim == 1 else values
    if values.shape[0] < 1 or values.shape[1] != g.shape[0]:
        raise ValueError("theta rows must match the dictionary coefficient count")
    radius = _positive(rho, "rho")
    norm = torch.linalg.vector_norm(values, dim=1)
    if not torch.isfinite(norm).all():
        raise ValueError("Coefficient norms overflowed float64")
    comparison_atol = torch.finfo(torch.float64).eps * g.shape[0] * max(1.0, radius)
    if bool(torch.any(norm > radius + comparison_atol)):
        raise ValueError("Supplied coefficients exceed the registered radius beyond float64 roundoff")
    m = torch.einsum("nr,rij->nij", values, g)
    frobenius = torch.linalg.vector_norm(m, dim=(1, 2))
    operator = torch.linalg.svdvals(m)[:, 0]
    if not torch.isfinite(frobenius).all() or not torch.isfinite(operator).all():
        raise ValueError("Generator diagnostics overflowed float64")
    gap = radius - norm
    exact_hits, numerical_hits = norm == radius, gap.abs() <= comparison_atol
    return {**_scope(), "coefficient_count_r": g.shape[0], "coefficient_vector_count": values.shape[0],
        "rho": radius, "theta_l2_norms": norm.tolist(), "M_frobenius_norms": frobenius.tolist(),
        "M_operator_2_norms": operator.tolist(), "M_frobenius_minus_theta_l2": (frobenius - norm).tolist(),
        "radius_minus_theta_l2": gap.tolist(), "exact_boundary_hits": exact_hits.tolist(),
        "exact_boundary_hit_rate": float(exact_hits.double().mean()),
        "roundoff_boundary_hits": numerical_hits.tolist(), "roundoff_boundary_hit_rate": float(numerical_hits.double().mean()),
        "boundary_comparison_atol": comparison_atol, "boundary_comparison_definition": "float64_eps*r*max(1,rho)",
        "rate_unit": "equally_weighted_supplied_coefficient_vectors_not_biological_participants",
        "M_matrices": m, "coefficient_count_is_matrix_update_rank": False}


def exp_transform_diagnostics(generators, theta, rho, reference_covariance, q=None):
    """Separate exp(M)'s whitened bounds from the sensor-space CAR restriction.

    A is structurally rank d on R^(d+1), hence its full condition number is
    infinite and determinant zero. R^(1/2) T R^(-1/2) gives A's CAR restriction;
    its Euclidean singular values generally differ from T's. RMS is nonlinear
    and has no volume-preservation interpretation supplied by this function.
    """
    g = _generators(generators)
    values = _tensor(theta, "theta", 1)
    coefficients = coefficient_diagnostics(g, values, rho)
    d = g.shape[1]
    reference = _tensor(reference_covariance, "reference covariance", 2)
    if reference.shape != (d, d):
        raise ValueError("Reference covariance must match the generator coordinates")
    root, inverse = spd_roots(reference)
    fixed_q = helmert_rows(d + 1)
    q = fixed_q if q is None else _tensor(q, "Q", 2)
    if not torch.equal(q, fixed_q):
        raise ValueError("The fixed Helmert CAR coordinate convention is required")
    m = coefficients["M_matrices"][0]
    t = torch.matrix_exp(m)
    sensor_restriction = root @ t @ inverse
    sensor = q.T @ sensor_restriction @ q
    if any(not torch.isfinite(value).all() for value in (t, sensor_restriction, sensor)):
        raise ValueError("Transformation diagnostics overflowed float64")
    singular_t, singular_sensor = torch.linalg.svdvals(t), torch.linalg.svdvals(sensor_restriction)
    if singular_t[-1] <= 0 or singular_sensor[-1] <= 0:
        raise ValueError("Transformation is numerically singular in effective coordinates")
    condition_t, condition_sensor = float(singular_t[0] / singular_t[-1]), float(singular_sensor[0] / singular_sensor[-1])
    if not math.isfinite(condition_t) or not math.isfinite(condition_sensor):
        raise ValueError("Transformation condition number overflowed float64")
    radius = coefficients["rho"]
    log_bound = 2 * radius if radius <= torch.finfo(torch.float64).max / 2 else None
    bound = math.exp(log_bound) if log_bound is not None and log_bound < math.log(torch.finfo(torch.float64).max) else None
    sign, logdet = torch.linalg.slogdet(t)
    h = q.T @ q
    return {**_scope(), "coefficients": coefficients, "M": m, "T_whitened": t,
        "sensor_CAR_restriction": sensor_restriction, "A_sensor": sensor,
        "whitened_singular_values": singular_t.tolist(),
        "whitened_condition_number": condition_t,
        "whitened_log_condition_upper_bound": log_bound, "whitened_condition_upper_bound": bound,
        "whitened_condition_upper_bound_expression": "exp(2*rho)",
        "whitened_logabsdet": float(logdet), "whitened_determinant_sign": float(sign),
        "whitened_determinant": float(torch.linalg.det(t)), "expected_whitened_determinant": 1.0,
        "sensor_CAR_restriction_condition_number": condition_sensor,
        "sensor_full_condition_number": None, "sensor_full_condition_definition": "infinite_structurally_singular_due_to_CAR",
        "sensor_structural_rank": d, "sensor_structural_determinant": 0.0,
        "sensor_numerical_rank": _rank(sensor), "M_numerical_rank": _rank(m),
        "whitened_update_numerical_rank": _rank(t - torch.eye(d, dtype=torch.float64),
            subtraction_reference_scale=max(1.0, float(singular_t[0]))),
        "sensor_change_from_CAR_numerical_rank": _rank(sensor - h,
            subtraction_reference_scale=max(1.0, float(singular_sensor[0]))),
        "whitened_singular_value_bound_applies_to_sensor_euclidean_coordinates": False,
        "post_RMS_volume_preservation_claim": False,
        "sensor_common_reference_annihilation_error": float(torch.linalg.vector_norm(sensor @ torch.ones(d + 1, dtype=torch.float64))),
        "interpretation": "model input transform; not brain connectivity, electrode displacement or clinical recovery"}


def parameter_count_table(feature_dim, r, k, *, d=10, generator_space="full", learned_dictionary=True):
    """Exact declared target counts, storage counts and distinct rank notions.

    Main target: r+p+1. Free/Meta-LoRA target: 2dk+p+1, including both factors;
    the rank-k matrix manifold dimension 2dk-k^2 is not the parameter count.
    Stiefel and Grassmann dimensions describe constrained/gauge degrees of
    freedom, not the number of entries optimized or stored in B.
    """
    p, r, k, d = (_integer(v, name) for v, name in
                  ((feature_dim, "feature_dim"), (r, "r"), (k, "k"), (d, "d")))
    if d < 2 or type(learned_dictionary) is not bool or generator_space not in {"full", "symmetric", "skew"}:
        raise ValueError("Valid matrix dimension, generator space and explicit dictionary flag required")
    n = {"full": d * d - 1, "symmetric": d * (d + 1) // 2 - 1, "skew": d * (d - 1) // 2}[generator_space]
    if r > n or k > d:
        raise ValueError("Coefficient/rank dimensions exceed the declared matrix space")
    head = p + 1
    return {**_scope(), "feature_dimension_p": p, "effective_dimension_d": d, "generator_ambient_dimension": n,
        "generator": {"target_spatial_parameters": r, "target_head_parameters": head, "target_total_parameters": r + head,
            "coefficient_count_r": r, "matrix_update_rank_upper_bound": d, "coefficient_count_is_matrix_update_rank": False,
            "dictionary_tensor_entries": n * r, "meta_trainable_dictionary_tensor_entries": n * r if learned_dictionary else 0,
            "Stiefel_intrinsic_dimension": n * r - r * (r + 1) // 2,
            "Grassmann_subspace_dimension": r * (n - r), "orthogonal_basis_gauge_dimension": r * (r - 1) // 2},
        "free_or_meta_lora": {"target_spatial_parameters": 2 * d * k, "target_head_parameters": head,
            "target_total_parameters": 2 * d * k + head, "factor_rank_k": k, "matrix_update_rank_upper_bound": k,
            "rank_exactly_k_matrix_manifold_dimension": 2 * d * k - k * k,
            "factorization_redundancy_dimension_when_both_factors_full_rank": k * k},
        "head_only_target_parameters": head, "f0_target_parameters": 0,
        "equal_target_parameter_count": r == 2 * d * k,
        "no_update_rank_inferred_from_parameter_count": True}
