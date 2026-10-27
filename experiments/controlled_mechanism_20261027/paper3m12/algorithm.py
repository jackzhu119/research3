"""Support-only controlled spatial interventions; no scheduler or file writes.

Fisher spectrum is a local, source-curvature diagnostic, not a confidence
statement. Every reference dose uses the model's separate reference_margin
when available; future query inputs and labels are absent from fitting.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

PROBABILITY_FLOOR = 1e-12


class NumericalFitError(ValueError):
    """A finite scientific update could not be constructed."""


def _notice(progress, **fields):
    if progress is not None:
        try:
            progress(**fields)
        except Exception:
            # Telemetry is deliberately unable to interrupt a scientific fit.
            pass


def _options(config):
    config = config or {}
    optimizer = config.get('optimizer', {})
    model = config.get('model', {})
    matching = config.get('matching', {})
    return dict(primary_steps=int(optimizer.get('primary_steps', 100)),
                secondary_steps=int(optimizer.get('secondary_steps', 10)),
                lr=float(optimizer.get('theta_learning_rate', .05)),
                l2=float(optimizer.get('theta_l2', .01)),
                radius=float(optimizer.get('theta_radius', .15)),
                draws=int(model.get('random_draws', 5)),
                abs_threshold=float(model.get('eigen_threshold_absolute', 1e-10)),
                rel_threshold=float(model.get('eigen_threshold_relative', 1e-6)),
                tolerance=float(matching.get('absolute_match_tolerance', 1e-5)),
                episode_identity=config.get('episode_identity'))


def _device_dtype(model):
    return torch.device(getattr(model, 'device', 'cpu')), getattr(model, 'dtype', torch.float64)


def _inputs(model, support_x, support_y, reference_x):
    device, dtype = _device_dtype(model)
    x = torch.as_tensor(support_x, device=device, dtype=dtype)
    labels = torch.as_tensor(support_y, device=device)
    if labels.is_complex() or not (((labels == 0) | (labels == 1)).all()):
        raise ValueError('Support labels must be exact binary values before integer conversion')
    y = labels.to(dtype=torch.long)
    ref = torch.as_tensor(reference_x, device=device, dtype=dtype)
    if x.ndim < 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('Support inputs require matching trial and label dimensions')
    if ref.ndim != x.ndim or not len(ref) or tuple(ref.shape[1:]) != tuple(x.shape[1:]):
        raise ValueError('A nonempty fixed reference input with the same signal shape is required')
    if not torch.isfinite(x).all() or not torch.isfinite(ref).all():
        raise ValueError('Invalid nonfinite scientific inputs cannot become a successful fallback')
    if not torch.isin(y, torch.tensor([0, 1], device=device)).all():
        raise ValueError('Support labels must be binary')
    return x, y, ref


def _margin(model, theta, x, *, reference=False, head_w=None, head_b=None):
    function = getattr(model, 'reference_margin', model.margin) if reference else model.margin
    value = (function(theta, x, head_w=head_w, head_b=head_b)
             if head_w is not None or head_b is not None else function(theta, x))
    if not isinstance(value, torch.Tensor) or value.shape != (len(x),):
        raise ValueError('Model margin must return one Tensor margin per input trial')
    if not torch.isfinite(value).all():
        raise NumericalFitError('Nonfinite model margins')
    return value


def balanced_loss(margin, y):
    """Balanced-mean binary cross entropy, using the common numerical path."""
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError('Balanced support loss requires both classes')
    losses = F.softplus(margin.double()) - y.double() * margin.double()
    return .5 * (losses[y == 0].mean() + losses[y == 1].mean())


def _canonical_columns(matrix):
    if not matrix.shape[1]:
        return matrix
    indices = matrix.abs().argmax(dim=0)
    signs = matrix[indices, torch.arange(matrix.shape[1], device=matrix.device)].sign()
    return matrix * torch.where(signs == 0, torch.ones_like(signs), signs)


def support_spectrum(model, x, y, theta=None, *, abs_threshold=1e-10, rel_threshold=1e-6):
    """Derivative at theta, weighted by the common zero-update anchor curvature."""
    device, dtype = _device_dtype(model)
    dimension = int(model.dimension)
    at = (torch.zeros(dimension, device=device, dtype=dtype) if theta is None
          else torch.as_tensor(theta, device=device, dtype=dtype).detach().clone())
    at.requires_grad_(True)
    margin = _margin(model, at, x)
    derivatives = []
    for i in range(len(margin)):
        derivative = (torch.autograd.grad(margin[i], at, retain_graph=i+1 < len(margin),
                                         allow_unused=True)[0] if margin[i].requires_grad else None)
        derivatives.append(torch.zeros_like(at) if derivative is None else derivative)
    jacobian = (torch.stack(derivatives) if derivatives
                else torch.empty((0, dimension), dtype=dtype, device=device))
    # Curvature remains the common EA anchor's p0, including when the
    # derivative is re-evaluated at a fitted update for component deletion.
    with torch.no_grad():
        anchor_margin = (margin.detach() if theta is None else _margin(model,
            torch.zeros(dimension,device=device,dtype=dtype),x))
    probability = torch.sigmoid(anchor_margin.double()).clamp(PROBABILITY_FLOOR, 1-PROBABILITY_FLOOR)
    counts = torch.stack([(y == 0).sum(), (y == 1).sum()]).double()
    weight = probability * (1-probability) / (2*counts[y])
    weighted = weight.sqrt().unsqueeze(1) * jacobian.double()
    if not torch.isfinite(weighted).all():
        raise NumericalFitError('Nonfinite weighted support Jacobian')
    _, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    threshold = max(float(abs_threshold), float(rel_threshold)*float(singular[0]) if len(singular) else 0.)
    rank = int((singular > threshold).sum().item())
    basis = _canonical_columns(vh[:rank].T).to(device=device, dtype=dtype)
    q = min(2, rank//2)
    top = basis[:, :q]
    weak = basis[:, rank-q:rank] if q else basis[:, :0]
    gap = (float(singular[q-1]/singular[rank-q]) if q else None)
    return dict(jacobian=jacobian.detach(), singular_values=singular.detach(),
                positive_basis=basis.detach(), top=top.detach(), weak=weak.detach(),
                positive_rank=rank, q=q, threshold=threshold, gap=gap)


def _seed(seed, budget, identity, phase, draw):
    text = json.dumps([20261027, int(seed), int(budget), identity, phase, int(draw)],
                      sort_keys=True, separators=(',', ':'), default=str)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2**63-1)


def random_subspace(positive_basis, q, seed):
    """Local generator: no global Torch/NumPy RNG sequence is consumed."""
    if q == 0:
        return positive_basis[:, :0]
    generator = torch.Generator(device='cpu'); generator.manual_seed(int(seed))
    sample = torch.randn((positive_basis.shape[1], q), generator=generator, dtype=torch.float64)
    orthogonal, r = torch.linalg.qr(sample, mode='reduced')
    signs = torch.diag(r).sign(); signs[signs == 0] = 1
    orthogonal = orthogonal*signs
    return positive_basis @ orthogonal.to(device=positive_basis.device, dtype=positive_basis.dtype)


def _tensor_identity(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _raw_fit(model, x, y, basis, steps, options, *, name, identity, raw_cache,
             checkpoint, progress):
    key = name + '-' + hashlib.sha256(json.dumps(dict(identity=identity,
        basis=_tensor_identity(basis), steps=steps, lr=options['lr'], l2=options['l2'],
        radius=options['radius']), sort_keys=True).encode()).hexdigest()[:20]
    record = raw_cache.get(key)
    if record is not None:
        if not isinstance(record, dict) or record.get('cache_identity') != key:
            raise ValueError('Raw-fit checkpoint identity mismatch')
        completed = int(record['completed_steps'])
        coefficient = torch.as_tensor(record['coefficient'], device=basis.device, dtype=basis.dtype)
        snapshots = {str(k): list(v) for k, v in record['snapshots'].items()}
        if (coefficient.shape != (basis.shape[1],) or not torch.isfinite(coefficient).all()
                or not 0 <= completed <= steps):
            raise ValueError('Invalid raw-fit checkpoint coefficients or step counter')
    else:
        completed = 0
        coefficient = torch.zeros(basis.shape[1], device=basis.device, dtype=basis.dtype)
        snapshots = {'0': [0.]*basis.shape[0]}
    if basis.shape[1] == 0:
        return dict(theta=[0.]*basis.shape[0], snapshots={str(k):[0.]*basis.shape[0]
                    for k in {options['primary_steps'], options['secondary_steps']}},
                    fit_status='FALLBACK_ZERO_RANK', completed_steps=0,
                    fallback_reason='positive_support_rank_below_two')
    try:
        for step in range(completed, steps):
            variable = coefficient.detach().requires_grad_(True)
            theta = basis @ variable
            loss = balanced_loss(_margin(model, theta, x), y) + options['l2']*theta.square().sum()
            gradient = torch.autograd.grad(loss, variable)[0]
            if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                raise NumericalFitError('Nonfinite support SGD objective or gradient')
            coefficient = (variable-options['lr']*gradient).detach()
            norm = torch.linalg.vector_norm(coefficient)
            coefficient = coefficient / (norm/options['radius']).clamp_min(1.)
            done = step+1
            if done in {options['primary_steps'], options['secondary_steps']}:
                snapshots[str(done)] = (basis @ coefficient).detach().cpu().tolist()
            if done % 10 == 0 or done == steps:
                record = dict(cache_identity=key, completed_steps=done,
                    coefficient=coefficient.detach().cpu().tolist(), snapshots=snapshots,
                    status='RAW_FIT_COMPLETE' if done == steps else 'RAW_FIT_CHECKPOINT')
                raw_cache[key] = record
                if checkpoint is not None and (done == options['secondary_steps'] or done % 50 == 0 or done == steps):
                    checkpoint(key, record)
                _notice(progress, stage='support_sgd', raw_condition=name, step=done,
                        total_steps=steps, support_objective=float(loss.detach()), raw_fit_reused=False)
        if str(steps) not in snapshots:
            snapshots[str(steps)] = (basis @ coefficient).detach().cpu().tolist()
        return dict(theta=(basis @ coefficient).detach().cpu().tolist(), snapshots=snapshots,
                    fit_status='FITTED', completed_steps=steps, cache_identity=key)
    except NumericalFitError as error:
        return dict(theta=[0.]*basis.shape[0], snapshots={str(k):[0.]*basis.shape[0]
                    for k in {options['primary_steps'], options['secondary_steps']}},
                    fit_status='FALLBACK_NONFINITE', completed_steps=0,
                    fallback_reason=type(error).__name__+': '+str(error))


def _reference_rms(model, theta, reference, anchor):
    with torch.no_grad():
        difference = _margin(model, theta, reference, reference=True).double()-anchor
        value = torch.sqrt(difference.square().mean())
    if not torch.isfinite(value):
        raise NumericalFitError('Nonfinite reference response')
    return float(value)


def match_reference_rms(model, theta, reference, anchor, target, *, tolerance=1e-5):
    """A bracketed continuous root; only [0,1] scaling, never amplification."""
    if not math.isfinite(target) or target < 0:
        raise ValueError('Reference RMS target must be finite and nonnegative')
    raw = _reference_rms(model, theta, reference, anchor)
    if target <= 0 or raw <= 0:
        return torch.zeros_like(theta), 0., 0.
    if target > raw+tolerance:
        raise ValueError('Dose matching may not amplify a weak update')
    if abs(raw-target) <= tolerance:
        return theta.clone(), 1., raw
    low, high = 0., 1.
    best = (theta.clone(), 1., raw)
    for _ in range(48):
        scale = (low+high)/2
        value = _reference_rms(model, theta*scale, reference, anchor)
        if abs(value-target) < abs(best[2]-target):
            best = (theta*scale, scale, value)
        if abs(value-target) <= tolerance:
            break
        if value < target:
            low = scale
        else:
            high = scale
    if abs(best[2]-target) > tolerance:
        raise NumericalFitError('Reference RMS could not be matched within the frozen tolerance')
    return best


def _state(model, theta, reference, anchor, *, method, layer, steps, scale=1., status='FITTED',
           fallback_reason=None, **extras):
    return dict(theta=theta.detach().cpu().tolist(), method=method, layer=layer,
        steps=int(steps), scale=float(scale), fit_status=status,
        fallback_reason=fallback_reason, reference_rms=_reference_rms(model, theta, reference, anchor),
        parameter_norm=float(torch.linalg.vector_norm(theta)), **extras)


def _serialize_spectrum(spectrum):
    singular = spectrum['singular_values'].cpu().tolist()
    rank, q = spectrum['positive_rank'], spectrum['q']
    # This is a numerical non-uniqueness flag, not a statistical exclusion.
    # The frozen positive-value tolerance also defines a near-tied boundary.
    boundaries = sorted({q, rank-q}) if q else []
    near_tied = [index for index in boundaries if 0 < index < rank
                 and abs(singular[index-1]-singular[index]) <= spectrum['threshold']]
    return dict(singular_values=spectrum['singular_values'].cpu().tolist(),
        positive_rank=spectrum['positive_rank'], q=spectrum['q'],
        positive_threshold=spectrum['threshold'], top_weak_singular_gap=spectrum['gap'],
        positive_basis=spectrum['positive_basis'].cpu().tolist(),
        top_basis=spectrum['top'].cpu().tolist(), weak_basis=spectrum['weak'].cpu().tolist(),
        near_tied_subspace_boundaries=near_tied, direction_separation_insufficient=bool(near_tied),
        tied_episode_retained=True)


def fit_episode(model, support_x, support_y, reference_x, *, budget, seed, config=None,
                tau=None, progress=None, raw_cache=None, checkpoint=None):
    """Fit A/B interventions using support and frozen reference inputs only.

    ``tau=None`` is an explicitly labelled pilot scale observation; frozen
    validation/target stages must pass their previously sealed tau_B.
    ``checkpoint`` owns durability; its errors are not disguised as successful
    fits. In contrast, exceptions from telemetry ``progress`` are ignored.
    """
    options = _options(config)
    if tau is None and not (config or {}).get('allow_pilot_tau_observation', False):
        raise ValueError('Frozen tau_B is required outside an explicit support-only pilot fixture')
    if tau is not None and (not math.isfinite(float(tau)) or float(tau) < 0):
        raise ValueError('Frozen tau_B must be finite and nonnegative')
    if (min(options['primary_steps'], options['secondary_steps']) < 1
            or options['draws'] < 1 or min(options['lr'], options['radius'], options['tolerance']) <= 0
            or options['l2'] < 0):
        raise ValueError('Invalid controlled adaptation settings')
    if budget not in (4, 8, 12, 16, 20):
        raise ValueError('Only frozen acquired budgets are allowed')
    x, y, reference = _inputs(model, support_x, support_y, reference_x)
    device, dtype = _device_dtype(model); dimension = int(model.dimension)
    if dimension < 1:
        raise ValueError('Positive spatial dimension required')
    zero = torch.zeros(dimension, device=device, dtype=dtype)
    with torch.no_grad():
        anchor_reference = _margin(model, zero, reference, reference=True).double().detach()
    states = {}; diagnostics = dict(valid_support_trials=len(y), acquired_budget=int(budget),
        source_seed=int(seed), source_head_frozen=True, query_used_for_fitting=False,
        reference_uses_separate_mapping=hasattr(model, 'reference_margin'),
        tau_is_pilot_observation=tau is None, randomized_metrics_are_not_ensembles=True)
    raw_cache = {} if raw_cache is None else raw_cache
    both_classes = bool((y == 0).any() and (y == 1).any())
    step_values = sorted({options['primary_steps'], options['secondary_steps']}, reverse=True)
    if not both_classes:
        reason = 'empty_support' if not len(y) else 'single_class_support'
        for steps in step_values:
            suffix = '' if steps == options['primary_steps'] else '_steps'+str(steps)
            names = ['support_ea', 'full_sgd', 'top_q', 'weak_q', 'scalar_shrink_full',
                     'top_q_weak_match', 'drop_weak', 'drop_scalar_shrink_full']
            names += ['random_q_'+str(i) for i in range(options['draws'])]
            names += ['drop_random_'+str(i) for i in range(options['draws'])]
            for name in names:
                states[name+suffix] = _state(model, zero, reference, anchor_reference,
                    method=name, layer='anchor_fallback', steps=steps, scale=0.,
                    status='FALLBACK_SUPPORT', fallback_reason=reason)
        diagnostics.update(fallback_reason=reason, positive_rank=0, q=0, weak_deleted_reference_rms=0.)
        return dict(states=states, diagnostics=diagnostics, tau_observation=0.)
    initial = support_spectrum(model, x, y, abs_threshold=options['abs_threshold'],
                               rel_threshold=options['rel_threshold'])
    diagnostics['initial_spectrum'] = _serialize_spectrum(initial)
    identity = dict(support_x=_tensor_identity(x), support_y=_tensor_identity(y),
        model=getattr(model, 'cache_identity', getattr(model, 'identity', type(model).__name__)),
        source_seed=int(seed), budget=int(budget), episode=options['episode_identity'])
    subspaces = dict(full_sgd=torch.eye(dimension, device=device, dtype=dtype),
                     top_q=initial['top'], weak_q=initial['weak'])
    for draw in range(options['draws']):
        subspaces['random_q_'+str(draw)] = random_subspace(initial['positive_basis'], initial['q'],
            _seed(seed, budget, options['episode_identity'] or identity['support_x'], 'initial', draw))
    fits = {}
    maximum_steps = max(step_values)
    for name, basis in subspaces.items():
        fits[name] = _raw_fit(model, x, y, basis, maximum_steps, options, name=name,
            identity=identity, raw_cache=raw_cache, checkpoint=checkpoint, progress=progress)
    tau_steps = options['secondary_steps']
    tau_theta = torch.as_tensor(fits['full_sgd']['snapshots'][str(tau_steps)], device=device, dtype=dtype)
    tau_observation = _reference_rms(model, tau_theta, reference, anchor_reference)
    if tau is None:
        tau_value = tau_observation
    else:
        tau_value = float(tau)
        if not math.isfinite(tau_value) or tau_value < 0:
            raise ValueError('Frozen tau_B must be finite and nonnegative')
    diagnostics['tau_used'] = tau_value
    diagnostics['tau_observation'] = tau_observation
    diagnostics['fit_status_by_condition'] = {k:v['fit_status'] for k,v in fits.items()}
    diagnostics['steps'] = {}
    for steps in step_values:
        suffix = '' if steps == options['primary_steps'] else '_steps'+str(steps)
        vectors = {name:torch.as_tensor(fit['snapshots'][str(steps)], device=device, dtype=dtype)
                   for name,fit in fits.items()}
        for name, theta in vectors.items():
            base = _state(model, theta, reference, anchor_reference, method=name,
                layer='raw', steps=steps, status=fits[name]['fit_status'],
                fallback_reason=fits[name].get('fallback_reason'))
            states[name+'_raw'+suffix] = base
        states['support_ea'+suffix] = _state(model, zero, reference, anchor_reference,
            method='support_ea', layer='anchor', steps=0, status='ANCHOR')
        states['full_sgd'+suffix] = dict(states['full_sgd_raw'+suffix])
        primary = ['top_q'] + ['random_q_'+str(i) for i in range(options['draws'])] + ['full_sgd']
        target = min([tau_value]+[_reference_rms(model, vectors[name], reference, anchor_reference)
                                   for name in primary]) if initial['q'] else 0.
        for name in primary:
            matched, alpha, actual = match_reference_rms(model, vectors[name], reference,
                anchor_reference, target, tolerance=options['tolerance'])
            output = 'scalar_shrink_full' if name == 'full_sgd' else name
            states[output+suffix] = _state(model, matched, reference, anchor_reference,
                method=output, layer='reference_rms', steps=steps, scale=alpha,
                status=fits[name]['fit_status'] if initial['q'] else 'FALLBACK_ZERO_RANK',
                fallback_reason=(fits[name].get('fallback_reason') if initial['q'] else
                                 'positive_support_rank_below_two'), target_reference_rms=target,
                match_absolute_error=abs(actual-target))
        weak_target = min(target, _reference_rms(model, vectors['weak_q'], reference, anchor_reference))
        for name, output in [('weak_q','weak_q'), ('top_q','top_q_weak_match')]:
            matched, alpha, actual = match_reference_rms(model, vectors[name], reference,
                anchor_reference, weak_target, tolerance=options['tolerance'])
            states[output+suffix] = _state(model, matched, reference, anchor_reference,
                method=output, layer='weak_reference_rms', steps=steps, scale=alpha,
                status=fits[name]['fit_status'], fallback_reason=fits[name].get('fallback_reason'),
                target_reference_rms=weak_target, match_absolute_error=abs(actual-weak_target))
        common_parameter_norm = min(float(torch.linalg.vector_norm(vectors[n])) for n in primary)
        for name in primary:
            value = float(torch.linalg.vector_norm(vectors[name]))
            alpha = min(1., common_parameter_norm/value) if value > 0 else 0.
            states[name+'_parameter_norm'+suffix] = _state(model, vectors[name]*alpha, reference,
                anchor_reference, method=name, layer='parameter_norm', steps=steps, scale=alpha,
                status=fits[name]['fit_status'] if initial['q'] else 'FALLBACK_ZERO_RANK',
                fallback_reason=(fits[name].get('fallback_reason') if initial['q'] else
                                 'positive_support_rank_below_two'),
                target_parameter_norm=common_parameter_norm)
        full = vectors['full_sgd']
        final = support_spectrum(model, x, y, full, abs_threshold=options['abs_threshold'],
                                 rel_threshold=options['rel_threshold'])
        weak_part = final['weak'] @ (final['weak'].T @ full)
        removals = {'drop_weak': full-weak_part, 'drop_scalar_shrink_full': full}
        for draw in range(options['draws']):
            basis = random_subspace(final['positive_basis'], final['q'],
                _seed(seed, budget, options['episode_identity'] or identity['support_x'], 'removal'+str(steps), draw))
            removals['drop_random_'+str(draw)] = full-basis@(basis.T@full)
        removal_target = min([tau_value]+[_reference_rms(model, value, reference, anchor_reference)
                                          for value in removals.values()]) if final['q'] else 0.
        removal_status = ('DERIVED_FROM_REAL_UPDATE' if fits['full_sgd']['fit_status'] == 'FITTED'
                          else fits['full_sgd']['fit_status'])
        for name, value in removals.items():
            states[name+'_raw'+suffix] = _state(model, value, reference, anchor_reference,
                method=name, layer='raw', steps=steps, status=removal_status,
                fallback_reason=fits['full_sgd'].get('fallback_reason'))
            matched, alpha, actual = match_reference_rms(model, value, reference, anchor_reference,
                removal_target, tolerance=options['tolerance'])
            states[name+suffix] = _state(model, matched, reference, anchor_reference, method=name,
                layer='removal_reference_rms', steps=steps, scale=alpha,
                status=removal_status if final['q'] else 'FALLBACK_ZERO_RANK',
                fallback_reason=(fits['full_sgd'].get('fallback_reason') if final['q'] else
                                 'final_positive_support_rank_below_two'), target_reference_rms=removal_target,
                match_absolute_error=abs(actual-removal_target))
        with torch.no_grad():
            actual_support_change = _margin(model, full, x)-_margin(model, zero, x)
            linear_change = initial['jacobian']@full
            nonlinear = float(torch.linalg.vector_norm(actual_support_change.double()-linear_change.double())
                /torch.linalg.vector_norm(actual_support_change.double()).clamp_min(1e-12))
            support_losses = {name:float(balanced_loss(_margin(model, value, x), y))
                              for name,value in vectors.items()}
        diagnostics['steps'][str(steps)] = dict(primary_reference_rms_target=target,
            weak_reference_rms_target=weak_target, removal_reference_rms_target=removal_target,
            parameter_norm_target=common_parameter_norm, final_spectrum=_serialize_spectrum(final),
            real_weak_component_parameter_energy=float(weak_part.square().sum()/full.square().sum().clamp_min(1e-24)),
            real_weak_component_norm=float(torch.linalg.vector_norm(weak_part)),
            real_weak_deletion_reference_rms=float(torch.sqrt((
                _margin(model, full, reference, reference=True).detach().double()
                -_margin(model, full-weak_part, reference, reference=True).detach().double()).square().mean())),
            support_loss_by_raw_condition=support_losses,
            source_support_loss=float(balanced_loss(_margin(model, zero, x).detach(), y)),
            nonlinear_support_remainder_fraction=nonlinear)
        _notice(progress, stage='episode_interventions_complete', steps=steps,
                q=initial['q'], reference_rms_target=target, tau_observation=tau_observation)
    # Ensure all metadata/state is valid JSON without permissive NaN encoding.
    result = dict(states=states, diagnostics=diagnostics, tau_observation=tau_observation)
    json.dumps(result, allow_nan=False)
    diagnostics['q'] = initial['q']
    diagnostics['positive_rank'] = initial['positive_rank']
    diagnostics['weak_deleted_reference_rms'] = diagnostics['steps'][str(options['primary_steps'])]['real_weak_deletion_reference_rms']
    return result


def fit_full(model, support_x, support_y, *, steps=10, config=None, progress=None,
             raw_cache=None, checkpoint=None, budget=20, seed=0):
    """Fast support-only full-space SGD; no reference/query labels or prediction."""
    options = _options(config)
    options['primary_steps'] = options['secondary_steps'] = int(steps)
    device, dtype = _device_dtype(model)
    x = torch.as_tensor(support_x, device=device, dtype=dtype)
    labels = torch.as_tensor(support_y, device=device)
    if labels.is_complex() or not (((labels == 0) | (labels == 1)).all()):
        raise ValueError('Support labels must be exact binary values before integer conversion')
    y = labels.to(dtype=torch.long)
    dimension = int(model.dimension)
    if (x.ndim < 2 or y.shape != (len(x),) or not torch.isfinite(x).all()
            or not torch.isin(y, torch.tensor([0, 1], device=device)).all()):
        raise ValueError('Invalid full-SGD support input')
    if int(steps) < 1:
        raise ValueError('Positive SGD step count required')
    if not ((y == 0).any() and (y == 1).any()):
        return dict(theta=[0.]*dimension, method='full_sgd', layer='raw', steps=0,
                    fit_status='FALLBACK_SUPPORT', fallback_reason='empty_or_single_class_support')
    identity = dict(support_x=_tensor_identity(x), support_y=_tensor_identity(y),
        model=getattr(model, 'cache_identity', getattr(model, 'identity', type(model).__name__)),
        source_seed=int(seed), budget=int(budget), episode=options['episode_identity'])
    fitted = _raw_fit(model, x, y, torch.eye(dimension, device=device, dtype=dtype),
        int(steps), options, name='calibration_full_sgd', identity=identity,
        raw_cache={} if raw_cache is None else raw_cache, checkpoint=checkpoint, progress=progress)
    with torch.no_grad():
        theta = torch.as_tensor(fitted['theta'], device=device, dtype=dtype)
        loss = float(balanced_loss(_margin(model, theta, x), y))
    return dict(theta=fitted['theta'], method='full_sgd', layer='raw', steps=int(steps),
                fit_status=fitted['fit_status'], fallback_reason=fitted.get('fallback_reason'),
                support_loss=loss)


def reference_rms(model, state, reference_x):
    device, dtype = _device_dtype(model)
    reference = torch.as_tensor(reference_x, device=device, dtype=dtype)
    if not len(reference) or not torch.isfinite(reference).all():
        raise ValueError('Nonempty finite fixed reference input required')
    theta = torch.as_tensor(state['theta'], device=device, dtype=dtype)
    zero = torch.zeros(int(model.dimension), device=device, dtype=dtype)
    with torch.no_grad():
        anchor = _margin(model, zero, reference, reference=True).double()
    if 'head_w' in state or 'head_b' in state:
        head_w=(torch.as_tensor(state['head_w'],device=device,dtype=dtype) if 'head_w' in state else None)
        head_b=torch.as_tensor(state.get('head_b',0.),device=device,dtype=dtype)
        return _joint_reference_rms(model,theta,head_w,head_b,reference,anchor)
    return _reference_rms(model, theta, reference, anchor)


def calibrate_tau(model, support_x, support_y, reference_x, *, budget, seed, config=None,
                  progress=None, raw_cache=None, checkpoint=None):
    """One Full-SGD10 observation; callers seal pooled support-only medians."""
    state = fit_full(model, support_x, support_y, steps=10, config=config, progress=progress,
                     raw_cache=raw_cache, checkpoint=checkpoint, budget=budget, seed=seed)
    value = reference_rms(model, state, reference_x)
    return dict(tau_observation=value, state=state,
                diagnostics=dict(query_used_for_fitting=False, support_only_scale=True,
                                 reference_uses_separate_mapping=hasattr(model, 'reference_margin')))


def predict(model, state, x, batch_size=64):
    """Predict each query batch independently; no state is fitted here."""
    device, dtype = _device_dtype(model)
    values = torch.as_tensor(x, device=device, dtype=dtype)
    theta = torch.as_tensor(state['theta'], device=device, dtype=dtype)
    if theta.shape != (int(model.dimension),) or not torch.isfinite(theta).all():
        raise ValueError('Prediction state has invalid spatial coefficients')
    if batch_size < 1:
        raise ValueError('Prediction batch size must be positive')
    head_w=(torch.as_tensor(state['head_w'],device=device,dtype=dtype) if 'head_w' in state else None)
    head_b=(torch.as_tensor(state['head_b'],device=device,dtype=dtype) if 'head_b' in state else None)
    probabilities = []
    with torch.no_grad():
        for offset in range(0, len(values), batch_size):
            margin = _margin(model, theta, values[offset:offset+batch_size],head_w=head_w,head_b=head_b)
            p = torch.sigmoid(margin.double()).clamp(PROBABILITY_FLOOR, 1-PROBABILITY_FLOOR)
            probabilities.append(torch.stack((1-p, p), dim=1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 2), dtype=np.float64)


def fit_shift_controls(model, support_x, support_y, reference_x, *, budget, seed,
                       config=None, tau=None, progress=None, raw_cache=None, checkpoint=None):
    """Lightweight known-shift positive controls, not primary dose comparisons.

    Only EA, actual full-space SGD and Top-q SGD are fitted. These diagnostic
    controls deliberately retain raw update amplitudes; they cannot replace
    the separately matched A/B primary comparisons.
    """
    options = _options(config)
    x, y, reference = _inputs(model, support_x, support_y, reference_x)
    device, dtype = _device_dtype(model); dimension = int(model.dimension)
    zero = torch.zeros(dimension, device=device, dtype=dtype)
    with torch.no_grad():anchor = _margin(model, zero, reference, reference=True).double()
    states = {'support_ea':_state(model, zero, reference, anchor, method='support_ea',
                                 layer='anchor', steps=0, status='ANCHOR')}
    if not ((y == 0).any() and (y == 1).any()):
        for name in ('full_sgd','top_q'):
            states[name] = _state(model, zero, reference, anchor, method=name,
                layer='anchor_fallback', steps=0, status='FALLBACK_SUPPORT',
                fallback_reason='empty_or_single_class_support')
        return dict(states=states, diagnostics=dict(query_used_for_fitting=False,
            matching_scope='unscaled_known_shift_positive_controls', q=0), tau_observation=0.)
    spectrum = support_spectrum(model, x, y, abs_threshold=options['abs_threshold'],
                                rel_threshold=options['rel_threshold'])
    identity = dict(support_x=_tensor_identity(x), support_y=_tensor_identity(y),
        model=getattr(model,'cache_identity',getattr(model,'identity',type(model).__name__)),
        seed=seed,budget=budget,episode=options['episode_identity'],block='known_shift')
    cache={} if raw_cache is None else raw_cache
    for name,basis in [('full_sgd',torch.eye(dimension,device=device,dtype=dtype)),
                       ('top_q',spectrum['top'])]:
        result=_raw_fit(model,x,y,basis,max(options['primary_steps'],options['secondary_steps']),
            options,name='shift_'+name,identity=identity,raw_cache=cache,
            checkpoint=checkpoint,progress=progress)
        theta=torch.as_tensor(result['snapshots'][str(options['primary_steps'])],device=device,dtype=dtype)
        states[name]=_state(model,theta,reference,anchor,method=name,layer='shift_raw_control',
            steps=options['primary_steps'],status=result['fit_status'],
            fallback_reason=result.get('fallback_reason'))
    return dict(states=states,diagnostics=dict(query_used_for_fitting=False,
        matching_scope='unscaled_known_shift_positive_controls',initial_spectrum=_serialize_spectrum(spectrum),
        q=spectrum['q'],reference_uses_separate_mapping=hasattr(model,'reference_margin')),
        tau_observation=0.)


def _joint_reference_rms(model, theta, head_w, head_b, reference, anchor):
    with torch.no_grad():
        change=_margin(model,theta,reference,reference=True,head_w=head_w,head_b=head_b).double()-anchor
        value=torch.sqrt(change.square().mean())
    if not torch.isfinite(value):raise NumericalFitError('Nonfinite joint reference response')
    return float(value)


def _joint_state(model, theta, head_w, head_b, reference, anchor, *, method, layer,
                 steps, scale=1., status='FITTED', **fields):
    return dict(theta=theta.detach().cpu().tolist(),head_w=head_w.detach().cpu().tolist(),
        head_b=float(head_b),method=method,layer=layer,steps=int(steps),scale=float(scale),
        fit_status=status,reference_rms=_joint_reference_rms(model,theta,head_w,head_b,reference,anchor),
        parameter_norm=float(torch.linalg.vector_norm(theta)),
        head_parameter_norm=float(torch.sqrt(head_w.square().sum()+head_b.square())),**fields)


def _match_joint_reference(model, values, reference, anchor, target, tolerance):
    theta,head_w,head_b=values
    raw=_joint_reference_rms(model,*values,reference,anchor)
    if target<=0 or raw<=0:return (theta*0,head_w*0,head_b*0),0.,0.
    if target>raw+tolerance:raise ValueError('Joint matching may never amplify an update')
    if abs(raw-target)<=tolerance:return values,1.,raw
    low,high=0.,1.;best=(values,1.,raw)
    for _ in range(48):
        scale=(low+high)/2;scaled=tuple(x*scale for x in values)
        actual=_joint_reference_rms(model,*scaled,reference,anchor)
        if abs(actual-target)<abs(best[2]-target):best=(scaled,scale,actual)
        if abs(actual-target)<=tolerance:break
        if actual<target:low=scale
        else:high=scale
    if abs(best[2]-target)>tolerance:raise NumericalFitError('Joint reference RMS matching failed')
    return best


def _joint_fit(model,x,y,spatial_basis,head_basis,train_bias,options,*,name,identity,
               raw_cache,checkpoint,progress):
    steps=options['primary_steps'];device,dtype=_device_dtype(model)
    feature_dimension=head_basis.shape[0]
    key=name+'-'+hashlib.sha256(json.dumps(dict(identity=identity,
        spatial=_tensor_identity(spatial_basis),head=_tensor_identity(head_basis),
        train_bias=train_bias,options={k:options[k] for k in ['primary_steps','lr','l2','radius']}),
        sort_keys=True).encode()).hexdigest()[:20]
    saved=raw_cache.get(key)
    if saved is not None:
        if not isinstance(saved,dict) or saved.get('cache_identity')!=key:
            raise ValueError('Joint raw checkpoint identity mismatch')
        a=torch.as_tensor(saved['coefficient'],device=device,dtype=dtype)
        h=torch.as_tensor(saved['head_coefficient'],device=device,dtype=dtype)
        b=torch.as_tensor(saved['head_b'],device=device,dtype=dtype)
        done=int(saved['completed_steps'])
        if (a.shape!=(spatial_basis.shape[1],) or h.shape!=(head_basis.shape[1],)
            or b.shape!=() or not all(torch.isfinite(t).all() for t in (a,h,b)) or not 0<=done<=steps):
            raise ValueError('Invalid joint raw checkpoint state')
    else:
        a=torch.zeros(spatial_basis.shape[1],device=device,dtype=dtype)
        h=torch.zeros(head_basis.shape[1],device=device,dtype=dtype)
        b=torch.zeros((),device=device,dtype=dtype);done=0
    try:
        for step in range(done,steps):
            av=a.detach().requires_grad_(True);hv=h.detach().requires_grad_(True)
            bv=b.detach().requires_grad_(train_bias)
            theta=spatial_basis@av;head_w=head_basis@hv
            loss=(balanced_loss(_margin(model,theta,x,head_w=head_w,head_b=bv),y)
                +options['l2']*theta.square().sum()+.001*(head_w.square().sum()+bv.square())/65.)
            params=[av,hv]+([bv] if train_bias else [])
            gradients=torch.autograd.grad(loss,params,allow_unused=True)
            gradients=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gradients)]
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
                raise NumericalFitError('Nonfinite joint support gradient')
            a=(av-options['lr']*gradients[0]).detach()
            a=a/(torch.linalg.vector_norm(a)/options['radius']).clamp_min(1.)
            h=(hv-.01*gradients[1]).detach()
            b=(bv-.01*gradients[2]).detach() if train_bias else bv.detach()
            completed=step+1
            if completed%10==0 or completed==steps:
                record=dict(cache_identity=key,coefficient=a.cpu().tolist(),
                    head_coefficient=h.cpu().tolist(),head_b=float(b),completed_steps=completed,
                    status='RAW_FIT_COMPLETE' if completed==steps else 'RAW_FIT_CHECKPOINT')
                raw_cache[key]=record
                if checkpoint is not None and (completed == 10 or completed % 50 == 0 or completed == steps):
                    checkpoint(key,record)
                _notice(progress,stage='head_competition_fit',raw_condition=name,step=completed,
                        total_steps=steps,support_objective=float(loss.detach()))
        return (spatial_basis@a,head_basis@h,b),'FITTED',None
    except NumericalFitError as error:
        return (torch.zeros(int(model.dimension),device=device,dtype=dtype),
                torch.zeros(feature_dimension,device=device,dtype=dtype),
                torch.zeros((),device=device,dtype=dtype)),'FALLBACK_NONFINITE',str(error)


def fit_head_controls(model,support_x,support_y,reference_x,*,budget,seed,tau,
                      config=None,progress=None,raw_cache=None,checkpoint=None):
    """Predeclared head-competition controls, with total function-dose matching.

    Frozen, bias-only and full feature heads share one reference-RMS family;
    no head capacity is interpreted as new observed support information.
    """
    if tau is None or not math.isfinite(float(tau)) or float(tau)<0:
        raise ValueError('Head controls require sealed finite tau_B')
    if not hasattr(model,'features'):
        raise ValueError('Head controls require frozen model features')
    options=_options(config);x,y,reference=_inputs(model,support_x,support_y,reference_x)
    device,dtype=_device_dtype(model);dimension=int(model.dimension)
    zero=torch.zeros(dimension,device=device,dtype=dtype)
    feature_function=getattr(model,'reference_features',model.features)
    with torch.no_grad():
        reference_features=feature_function(zero,reference).double()
        anchor=_margin(model,zero,reference,reference=True).double()
    if reference_features.ndim!=2 or reference_features.shape[0]!=len(reference):
        raise ValueError('Reference features must be Nxfeature_dimension')
    feature_dimension=reference_features.shape[1]
    if not torch.isfinite(reference_features).all():raise ValueError('Nonfinite frozen reference features')
    empty_head=torch.empty((feature_dimension,0),device=device,dtype=dtype)
    full_head=torch.eye(feature_dimension,device=device,dtype=dtype)
    centered=reference_features-reference_features.mean(0)
    _,_,vh=torch.linalg.svd(centered,full_matrices=False)
    low_head=_canonical_columns(vh[:min(4,len(vh))].T).to(device=device,dtype=dtype)
    empty_spatial=torch.empty((dimension,0),device=device,dtype=dtype)
    states={};cache={} if raw_cache is None else raw_cache
    zero_head=torch.zeros(feature_dimension,device=device,dtype=dtype);zero_bias=torch.zeros((),device=device,dtype=dtype)
    states['support_ea']=_joint_state(model,zero,zero_head,zero_bias,reference,anchor,
        method='support_ea',layer='anchor',steps=0,status='ANCHOR',head_mode='frozen')
    if not ((y==0).any() and (y==1).any()):
        for mode in ('frozen','bias','full'):
            for direction in ['top_q']+['random_q_'+str(i) for i in range(options['draws'])]:
                name=direction+'_head_'+mode
                states[name]=_joint_state(model,zero,zero_head,zero_bias,reference,anchor,
                    method=name,layer='anchor_fallback',steps=0,status='FALLBACK_SUPPORT',
                    head_mode=mode,fallback_reason='empty_or_single_class_support')
        for mode in ('bias','rank4','full'):
            name='head_only_'+mode
            states[name]=_joint_state(model,zero,zero_head,zero_bias,reference,anchor,
                method=name,layer='anchor_fallback',steps=0,status='FALLBACK_SUPPORT',head_mode=mode)
        return dict(states=states,diagnostics=dict(q=0,source_head_parameters_frozen=True,
            delta_head_controls=True,query_used_for_fitting=False),tau_observation=0.)
    spectrum=support_spectrum(model,x,y,abs_threshold=options['abs_threshold'],rel_threshold=options['rel_threshold'])
    identity=dict(support_x=_tensor_identity(x),support_y=_tensor_identity(y),source_seed=int(seed),
        budget=int(budget),episode=options['episode_identity'],block='head_competition',
        model=getattr(model,'cache_identity',getattr(model,'identity',type(model).__name__)))
    bases={'top_q':spectrum['top']}
    for draw in range(options['draws']):
        bases['random_q_'+str(draw)]=random_subspace(spectrum['positive_basis'],spectrum['q'],
            _seed(seed,budget,options['episode_identity'] or identity['support_x'],'head_competition',draw))
    values={};statuses={};reasons={};modes={}
    for mode in ('frozen','bias','full'):
        head_basis=full_head if mode=='full' else empty_head
        for direction,basis in bases.items():
            name=direction+'_head_'+mode
            if spectrum['q']:
                values[name],statuses[name],reasons[name]=_joint_fit(model,x,y,basis,head_basis,mode!='frozen',
                    options,name=name,identity=identity,raw_cache=cache,checkpoint=checkpoint,progress=progress)
            else:
                values[name]=(zero,zero_head,zero_bias);statuses[name]='FALLBACK_ZERO_RANK';reasons[name]='zero_direction_rank'
            modes[name]=mode
    target=min([float(tau)]+[_joint_reference_rms(model,*value,reference,anchor) for value in values.values()])
    for name,value in values.items():
        states[name+'_raw']=_joint_state(model,*value,reference,anchor,method=name,
            layer='head_raw',steps=options['primary_steps'],status=statuses[name],head_mode=modes[name],
            fallback_reason=reasons[name])
        matched,alpha,actual=_match_joint_reference(model,value,reference,anchor,target,options['tolerance'])
        states[name]=_joint_state(model,*matched,reference,anchor,method=name,
            layer='head_reference_rms',steps=options['primary_steps'],scale=alpha,status=statuses[name],
            head_mode=modes[name],target_reference_rms=target,match_absolute_error=abs(actual-target),
            head_physical_penalty_denominator=65,fallback_reason=reasons[name])
    for mode,basis in [('bias',empty_head),('rank4',low_head),('full',full_head)]:
        name='head_only_'+mode
        value,status,reason=_joint_fit(model,x,y,empty_spatial,basis,True,options,name=name,
            identity=identity,raw_cache=cache,checkpoint=checkpoint,progress=progress)
        states[name]=_joint_state(model,*value,reference,anchor,method=name,layer='head_only_raw',
            steps=options['primary_steps'],status=status,head_mode=mode,fallback_reason=reason,
            head_physical_penalty_denominator=65)
    result=dict(states=states,diagnostics=dict(q=spectrum['q'],source_head_parameters_frozen=True,
        delta_head_controls=True,query_used_for_fitting=False,initial_spectrum=_serialize_spectrum(spectrum),
        head_family_reference_rms_target=target,head_physical_penalty_denominator=65,
        head_feature_dimension=feature_dimension,reference_uses_separate_mapping=hasattr(model,'reference_margin'),
        matching_scope='independent head-competition family, not primary A/B family'),tau_observation=0.)
    json.dumps(result,allow_nan=False)
    return result
