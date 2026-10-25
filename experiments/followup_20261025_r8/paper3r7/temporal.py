"""Finite-window spatial/temporal Lie maps and their identity derivatives.

The primary temporal generator uses zero-boundary differences projected away
from the constant-time mode. Its exponential is a noncausal operation on the
already observed trial, not an online physiological delay correction.

Production never constructs the (channels*times)-square superoperator. A fixed
Hermitian time decomposition reduces its action to times many channel blocks.
"""
from functools import lru_cache
import math

import torch

from . import core


MODES = ('static', 'rotation', 'temporal', 'spatiotemporal',
         'split_left', 'split_right', 'split_symmetric')
OPERATORS = ('zero_boundary', 'circular')


def _check_kind(n_times, kind):
    if isinstance(n_times, bool) or not isinstance(n_times, int) or n_times < 3:
        raise ValueError('At least three finite-window time points required')
    if kind not in OPERATORS:
        raise ValueError('Unknown fixed temporal operator')


@lru_cache(maxsize=16)
def _cpu_operator(n_times, kind):
    _check_kind(n_times, kind)
    shift = torch.zeros(n_times, n_times, dtype=torch.float64)
    shift[1:, :-1] = torch.eye(n_times - 1, dtype=torch.float64)
    if kind == 'circular':
        shift[0, -1] = 1.
    projection = (torch.eye(n_times, dtype=torch.float64)
                  - torch.ones(n_times, n_times, dtype=torch.float64) / n_times)
    raw = projection @ ((shift - shift.T) / 2) @ projection
    raw = (raw - raw.T) / 2
    norm = torch.linalg.vector_norm(raw)
    if not bool(norm > 0):
        raise ValueError('Temporal generator has no nonconstant skew component')
    # This is a superoperator Frobenius budget, not an operator-norm bound.
    return (raw * (math.sqrt(n_times) / norm)).detach()


def make_time_operator(n_times, kind='zero_boundary', device=None, dtype=torch.float64):
    """Return a private D copy: D^T=-D, D1=0 and ||D||_F=sqrt(T)."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Temporal operators require a real floating dtype')
    return _cpu_operator(n_times, kind).to(device=device, dtype=dtype).clone()


@lru_cache(maxsize=16)
def _cpu_spectrum(n_times, kind):
    d = _cpu_operator(n_times, kind)
    # iD is Hermitian. If iD=U diag(w) U^H, then D^T=iU diag(w) U^H.
    values, vectors = torch.linalg.eigh(1j * d)
    return values.detach(), vectors.detach()


@lru_cache(maxsize=32)
def _spectrum(n_times, kind, device_name, dtype):
    values, vectors = _cpu_spectrum(n_times, kind)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return (values.to(device=device_name, dtype=dtype),
            vectors.to(device=device_name, dtype=complex_dtype))


def _parts(m):
    return (m - m.T) / 2, (m + m.T) / 2


def _matrix_exp(value):
    """Double-precision scaled Taylor action for the small channel blocks.

    The selected Torch2.5.1 polynomial branch can leave ~1e-10 discrepancies
    between equivalent small-norm split maps. A fixed degree18 series with
    one-norm <=.5 has a negligible truncation remainder in float64 and a
    regular derivative at zero; eigendecomposing a trainable skew matrix would
    instead have undefined eigenvector derivatives at repeated eigenvalues.
    Only the integer numerical scaling branch is detached.
    """
    norm = float(torch.linalg.matrix_norm(value.detach(), ord=1).max().cpu())
    scaling = max(0, math.ceil(math.log2(norm / .5))) if norm > .5 else 0
    a = value / (2 ** scaling)
    identity = torch.eye(value.shape[-1], device=value.device, dtype=value.dtype)
    term, result = identity, identity
    for order in range(1, 19):
        term = term @ a / order
        result = result + term
    for _ in range(scaling):
        result = result @ result
    return result


def _block_action(y, k, s, operator):
    values, vectors = _spectrum(y.shape[-1], operator, str(y.device), y.dtype)
    blocks = k.to(vectors.dtype)[None] + 1j * values[:, None, None] * s
    exponential = _matrix_exp(blocks)
    transformed = y.to(vectors.dtype) @ vectors
    identity = torch.eye(k.shape[-1], device=k.device, dtype=vectors.dtype)
    transformed = torch.einsum('tcd,...td->...tc', exponential - identity, transformed.movedim(-1, -2))
    result = transformed.movedim(-1, -2) @ vectors.conj().T
    # Conjugate paired eigenmodes yield a real result. This discard is a
    # standard complex implementation of an exactly real finite operator.
    return y + result.real


def _rotation_action(y, k):
    identity = torch.eye(k.shape[-1], device=k.device, dtype=k.dtype)
    return y + (_matrix_exp(k) - identity) @ y


def apply_white(y, m, mode='spatiotemporal', operator='zero_boundary'):
    """Apply the map to real (...,channels,time) whitened trial tensors.

    H(Y)=KY+SYD^T, so its column-vectorized generator is I_T kron K+D kron S.
    Modes share the same M and target parameter coordinates. Split names refer
    to the matrix product: split_left applies temporal then exp(K).
    """
    if mode not in MODES:
        raise ValueError('Unknown spatial/temporal mode')
    _check_kind(y.shape[-1], operator)
    if (y.ndim < 2 or m.ndim != 2 or m.shape[0] != m.shape[1]
            or m.shape[0] != y.shape[-2] or y.is_complex() or m.is_complex()):
        raise ValueError('Real channel-square generator and channel-time trials required')
    dtype = torch.promote_types(y.dtype, m.dtype)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('Floating point trial/generator coordinates required')
    y, m = y.to(dtype), m.to(device=y.device, dtype=dtype)
    if mode == 'static':
        return _rotation_action(y, m)
    k, s = _parts(m)
    if mode == 'rotation':
        return _rotation_action(y, k)
    if mode == 'spatiotemporal':
        return _block_action(y, k, s, operator)
    if mode == 'temporal':
        return _block_action(y, torch.zeros_like(k), s, operator)
    if mode == 'split_left':
        return _rotation_action(_block_action(y, torch.zeros_like(k), s, operator), k)
    if mode == 'split_right':
        return _block_action(_rotation_action(y, k), torch.zeros_like(k), s, operator)
    return _rotation_action(_block_action(_rotation_action(y, k / 2), torch.zeros_like(k), s, operator), k / 2)


def actual(source, geometry, b, weights, x, mode='spatiotemporal', operator='zero_boundary'):
    """Actual finite-window map followed by the unchanged residual head."""
    m = (geometry.e @ b @ weights.theta).reshape(10, 10)
    white = geometry.rref_invsqrt @ geometry.q @ x.double()
    mapped = apply_white(white, m, mode, operator)
    # Extend the CAR-subspace map by identity on its orthogonal complement.
    # Residual evaluation is mathematically equal on exact CAR inputs and
    # keeps theta=0 exactly equal to the original source input in floating point.
    signal = x.double() + geometry.q.T @ geometry.rref_sqrt @ (mapped - white)
    features, base = source.features_and_logits(core.joint_trial_rms(signal))
    features, base = features.to(weights.delta_w.dtype), base.to(weights.delta_w.dtype)
    delta = features @ weights.delta_w + weights.delta_b
    return base + torch.stack((-delta / 2, delta / 2), 1), features


def support_tangent(source, geometry, support, mode='spatiotemporal', operator='zero_boundary'):
    """Detached exact identity margin derivative for the chosen finite map.

    Source parameters/BN remain frozen. The derivative includes CAR,
    recoloring, joint RMS and the observed trial's finite temporal operator.
    No future query values or labels are accepted.
    """
    if mode not in MODES:
        raise ValueError('Unknown spatial/temporal mode')
    _check_kind(support.x.shape[-1], operator)
    core._assert_frozen(source)
    with torch.enable_grad():
        x = support.x.double()
        left = geometry.q.T @ geometry.rref_sqrt
        white = geometry.rref_invsqrt @ geometry.q @ x
        signal = x
        rms = signal.square().mean((-2, -1), keepdim=True).sqrt()
        if len(x) and bool((rms == 0).any()):
            raise ValueError('Zero-energy support trial after CAR')
        normalized = (signal / rms).detach().requires_grad_(True)
        features, base = source.features_and_logits(normalized)
        margin = base[:, 1] - base[:, 0]
        if not len(x):
            return margin.detach().double(), x.new_empty((0, geometry.e.shape[1])), features.detach()
        gradient = torch.autograd.grad(margin.sum(), normalized, create_graph=False)[0]
        raw_gradient = (gradient - signal * (gradient * signal).sum((-2, -1), keepdim=True)
                        / signal.square().sum((-2, -1), keepdim=True)) / rms
        white_gradient = left.T @ raw_gradient
        spatial = white_gradient @ white.transpose(-2, -1)
        skew = (spatial - spatial.transpose(-2, -1)) / 2
        if mode == 'static':
            effective = spatial
        elif mode == 'rotation':
            effective = skew
        else:
            d = _cpu_operator(x.shape[-1], operator).to(device=x.device)
            temporal = white_gradient @ d @ white.transpose(-2, -1)
            symmetric = (temporal + temporal.transpose(-2, -1)) / 2
            effective = symmetric if mode == 'temporal' else skew + symmetric
        coordinates = effective.flatten(1) @ geometry.e
    return margin.detach().double(), coordinates.detach(), features.detach()


def common_time_shift(x, t, operator='zero_boundary'):
    """Right-multiply all channels by exp(tD), preserving raw XX^T and RMS.

    This is a finite-window orthogonal mechanism control, not a causal shift.
    Its covariance invariance does not imply invariance after arbitrary filters.
    """
    _check_kind(x.shape[-1], operator)
    if x.is_complex() or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('Real floating channel-time trials required')
    values, vectors = _spectrum(x.shape[-1], operator, str(x.device), x.dtype)
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    multiplier = torch.expm1(-1j * t * values)
    return x + ((x.to(vectors.dtype) @ vectors * multiplier) @ vectors.conj().T).real


def diagnostics(m, n_times=128, mode='spatiotemporal', operator='zero_boundary'):
    """Small detached scalar audit; never serialize a dense time operator."""
    if mode not in MODES:
        raise ValueError('Unknown spatial/temporal mode')
    d = _cpu_operator(n_times, operator)
    k, s = _parts(m.detach().double())
    if mode == 'static':
        budget = m.detach().double().norm()
    elif mode == 'rotation':
        budget = k.norm()
    elif mode == 'temporal':
        budget = s.norm()
    else:
        budget = (k.square().sum() + s.square().sum()).sqrt()
    return {'time_points': n_times,
            'time_generator_frobenius_norm': float(d.norm()),
            'time_generator_operator_norm': float(torch.linalg.matrix_norm(d, ord=2)),
            'time_skew_residual': float((d + d.T).norm()),
            'time_constant_residual': float((d @ torch.ones(n_times, dtype=d.dtype)).norm()),
            'joint_generator_scaled_frobenius_norm': float(budget),
            'spatial_skew_generator_norm': float(k.norm()),
            'temporal_symmetric_generator_norm': float(s.norm()),
            'uses_periodic_boundary': operator == 'circular',
            'finite_trial_noncausal': True}
