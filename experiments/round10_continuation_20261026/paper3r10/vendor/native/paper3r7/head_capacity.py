"""Support-only head capacity and local spatial identifiability diagnostics.

The numerical projector is detached: it is computed from the frozen source's
identity features on the acquired support, never from labels or query samples.
The diagnostics distinguish data-only projected information from ridge-induced
curvature. They do not certify a physical mixing matrix or stop a worker.
"""
from functools import lru_cache
import math

import torch


def _integer(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(name + (' must be a positive integer' if positive else ' must be a nonnegative integer'))


def _rank(singular, rows, columns, reference=None):
    maximum = float(singular.max()) if singular.numel() else 0.
    scale = maximum if reference is None else float(reference)
    tolerance = max(rows, columns, 1) * torch.finfo(torch.float64).eps * scale
    return int((singular > tolerance).sum()), tolerance


@lru_cache(maxsize=16)
def _random_basis(dimension, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.where(r.diagonal() < 0, -1., 1.)
    return q.detach()


def projector(features, policy, rank, head_rank=4, head_max_rank=16, *, basis='pca', seed=20261024):
    """Return a p x h float64 orthonormal head basis and JSON metadata.

    ``full`` is exactly I_p. ``fixed`` selects at most head_rank directions;
    ``budget`` requests max(0,min(head_max_rank,n_valid-1-rank)). ``none`` is
    the bias-only control. Reduced heads also respect centered support rank,
    so constant features cannot create fictitious independent head directions.
    Random directions use the same nested seed-fixed QR basis at every budget;
    only their count can depend on the support's numerical rank.
    """
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError('Expected an acquired-support feature matrix')
    if features.shape[1] < 1 or not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError('Support features must be finite floating values with positive feature dimension')
    if policy not in ('full', 'fixed', 'budget', 'none') or basis not in ('pca', 'random'):
        raise ValueError('Unknown head-capacity policy or basis')
    _integer(rank, 'spatial rank', positive=True)
    _integer(head_rank, 'head rank')
    _integer(head_max_rank, 'maximum head rank')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Random head basis seed must be an integer')
    n, p = features.shape
    with torch.no_grad():
        work = features.detach().to(device='cpu', dtype=torch.float64)
        centered = work - work.mean(0, keepdim=True) if n else work
        if n:
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            observed, tolerance = _rank(singular, n, p)
        else:
            singular, vh, observed, tolerance = work.new_empty(0), work.new_empty((0, p)), 0, 0.
        if policy == 'full':
            requested, h = p, p
            u = torch.eye(p, dtype=torch.float64)
        else:
            requested = (head_rank if policy == 'fixed' else
                         max(0, min(head_max_rank, n - 1 - rank)) if policy == 'budget' else 0)
            h = min(requested, p, max(0, n - 1), observed)
            if not h:
                u = work.new_empty((p, 0))
            elif basis == 'random':
                u = _random_basis(p, seed)[:, :h].clone()
            else:
                u = vh[:h].T.clone()
                # Fix isolated singular-vector signs; repeated singular spaces
                # still have ordinary SVD basis ambiguity, documented below.
                pivot = u.abs().argmax(0)
                signs = torch.where(u[pivot, torch.arange(h)] < 0, -1., 1.)
                u = u * signs
        u = u.to(device=features.device)
    metadata = {
        'head_policy': policy, 'head_basis': 'identity' if policy == 'full' else basis,
        'valid_support_count': n, 'full_head_feature_dim': p,
        'requested_head_rank': requested, 'effective_head_rank': h,
        'head_parameter_count_including_bias': h + 1,
        'centered_support_feature_rank': observed,
        'centered_support_singular_values': singular.tolist(),
        'feature_rank_tolerance': tolerance,
        'head_l2_effective_multiplier': (h + 1) / (p + 1),
        'head_prior_normalization_dim_including_bias': p + 1,
        'projector_scope': 'frozen_source_identity_features_on_acquired_support_only',
        'projector_uses_labels': False, 'projector_uses_queries': False,
        'projector_differentiated': False,
        'basis_rotation_seed': seed if basis == 'random' and policy != 'full' else None,
        'pca_repeated_singular_space_has_basis_ambiguity': basis == 'pca' and policy != 'full',
        'dimension_necessary_condition_only': n - (h + 1) >= rank,
        'identifiability_claimed': False,
    }
    return u, metadata


def project_weights(delta_w, u):
    """Orthogonally project full head coefficients, preserving their graph."""
    if (not isinstance(delta_w, torch.Tensor) or delta_w.ndim not in (1, 2)
            or not isinstance(u, torch.Tensor) or u.ndim != 2 or delta_w.shape[0] != u.shape[0]):
        raise ValueError('Head coefficients and projector dimensions differ')
    u = u.to(device=delta_w.device, dtype=delta_w.dtype)
    return u @ (u.T @ delta_w)


def effective_head_l2(head_l2, projected_dim, full_feature_dim):
    """Keep lambda ||delta_w||^2/(p+1) after a reduced h-dimensional fit."""
    if not math.isfinite(head_l2) or head_l2 < 0:
        raise ValueError('Head prior must be finite and nonnegative')
    _integer(projected_dim, 'projected head dimension')
    _integer(full_feature_dim, 'full feature dimension', positive=True)
    if projected_dim > full_feature_dim:
        raise ValueError('Projected dimension exceeds original feature dimension')
    return head_l2 * (projected_dim + 1) / (full_feature_dim + 1)


def identifiability(j, features, y, margin, head_l2, theta_l2, *, full_head_dim=None):
    """Return diagnostics, without changing parameters or enforcing a gate.

    ``features`` contains the actually allowed head coordinates, before its
    intercept is appended. ``full_head_dim`` is the original source feature
    dimension p (excluding intercept), required to retain the same physical
    head ridge after restriction. W uses class-balanced model-logistic
    curvature at the supplied identity-surrogate margin, not label-score
    outer products and not the exact nonlinear-network Hessian.
    """
    tensors = (j, features, y, margin)
    if not all(isinstance(v, torch.Tensor) for v in tensors):
        raise ValueError('Diagnostic inputs must be tensors')
    if j.ndim != 2 or features.ndim != 2 or y.ndim != 1 or margin.ndim != 1:
        raise ValueError('Invalid diagnostic matrix/vector ranks')
    n, r = j.shape
    if r < 1 or any(len(v) != n for v in (features, y, margin)):
        raise ValueError('Diagnostic support dimensions differ')
    if not all(bool(torch.isfinite(v).all()) for v in tensors) or not bool(torch.isin(y, y.new_tensor([0, 1])).all()):
        raise ValueError('Nonfinite or nonbinary support diagnostic data')
    h = features.shape[1]
    full_head_dim = h if full_head_dim is None else full_head_dim
    _integer(full_head_dim, 'original feature dimension')
    if full_head_dim < h or any(not math.isfinite(v) or v < 0 for v in (head_l2, theta_l2)):
        raise ValueError('Invalid original dimension or prior penalty')
    with torch.no_grad():
        jt, f, labels, m = [v.detach().to(device='cpu', dtype=torch.float64) for v in tensors]
        z = torch.cat((f, f.new_ones((n, 1))), 1)
        counts = torch.stack([(labels == c).sum() for c in (0, 1)]).double()
        classes = int((counts > 0).sum())
        q = (1. / (max(classes, 1) * counts.clamp_min(1)))[labels.long()]
        curvature = q * torch.sigmoid(m) * torch.sigmoid(-m)
        root = curvature.sqrt()
        zw, jw = root[:, None] * z, root[:, None] * jt
        if n:
            left, singular, _ = torch.linalg.svd(zw, full_matrices=False)
            head_rank, head_tolerance = _rank(singular, n, h + 1)
        else:
            left, singular, head_rank, head_tolerance = zw.new_empty((0, 0)), zw.new_empty(0), 0, 0.
        basis = left[:, :head_rank]
        residual = jw - basis @ (basis.T @ jw)
        if head_rank == n:
            # In this case P=I exactly. Avoid interpreting SVD roundoff as
            # empirical information in otherwise wholly absorbed directions.
            residual = torch.zeros_like(jw)
        data_info = residual.T @ residual
        data_info = (data_info + data_info.T) / 2
        j_singular = torch.linalg.svdvals(jw) if n else jw.new_empty(0)
        residual_singular = torch.linalg.svdvals(residual) if n else jw.new_empty(0)
        spatial_rank, spatial_tolerance = _rank(residual_singular, n, r,
            float(j_singular.max()) if j_singular.numel() else 0.)
        positive_rows = int((curvature > 0).sum())
        rank_bound = min(r, max(0, positive_rows - head_rank))
        spatial_rank = min(spatial_rank, rank_bound)
        eta = 2 * head_l2 / (full_head_dim + 1)
        if eta == 0:
            schur = data_info.clone()
        else:
            # SVD is the stable n-dimensional equivalent of the penalized-head
            # Schur complement. It stays usable as eta approaches zero without
            # inverting a large, almost singular (h+1)-dimensional matrix.
            projected = left.T @ jw
            remainder = jw - left @ projected
            if left.shape[1] == n:
                remainder = torch.zeros_like(jw)
            attenuation = eta / (singular.square() + eta)
            schur = remainder.T @ remainder + projected.T @ (attenuation[:, None] * projected)
            schur = (schur + schur.T) / 2
        total = schur + (2 * theta_l2) * torch.eye(r, dtype=torch.float64)
        energy = float(jw.square().sum())
        residual_energy = float(residual.square().sum())
        absorbed_fraction = min(1., max(0., 1 - residual_energy / energy)) if energy else 0.
        return {
            'identifiability_diagnostic_scope': 'local_fixed_identity_tangent_surrogate_only',
            'identifiability_claimed': False, 'diagnostic_is_training_gate': False,
            'valid_support_count': n, 'classes_present': classes, 'positive_curvature_rows': positive_rows,
            'allowed_head_feature_dim': h, 'original_head_feature_dim': full_head_dim,
            'weighted_head_span_rank': head_rank, 'weighted_head_singular_values': singular.tolist(),
            'weighted_head_rank_tolerance': head_tolerance,
            'data_only_spatial_info': data_info.tolist(),
            'data_only_spatial_eigenvalues': torch.linalg.eigvalsh(data_info).clamp_min(0).tolist(),
            'data_only_spatial_info_rank': spatial_rank,
            'data_only_spatial_info_rank_bound': rank_bound,
            'data_only_spatial_rank_tolerance': spatial_tolerance,
            'weighted_spatial_singular_values': j_singular.tolist(),
            'residual_spatial_singular_values': residual_singular.tolist(),
            'head_absorbed_spatial_fraction': absorbed_fraction,
            'head_absorption_residual_norm': math.sqrt(residual_energy),
            'local_data_spatial_full_rank': spatial_rank == r,
            'necessary_observation_rank_condition': rank_bound >= r,
            'head_prior_precision_eta': eta,
            'head_prior_normalization_dim_including_bias': full_head_dim + 1,
            'penalized_head_spatial_schur': schur.tolist(),
            'penalized_head_spatial_eigenvalues': torch.linalg.eigvalsh(schur).clamp_min(0).tolist(),
            'head_ridge_induced_trace': float(torch.trace(schur - data_info)),
            'spatial_prior_precision': 2 * theta_l2,
            'total_penalized_spatial_schur': total.tolist(),
            'total_penalized_spatial_eigenvalues': torch.linalg.eigvalsh(total).clamp_min(0).tolist(),
            'penalized_curvature_is_data_identifiability': False,
            'solver_damping_included_in_information': False,
        }
