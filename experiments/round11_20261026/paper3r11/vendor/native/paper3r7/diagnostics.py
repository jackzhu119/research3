"""Predeclared synthetic stress tests. Never overwrite cached EEG or true labels."""
from dataclasses import replace
import math
from types import SimpleNamespace
import torch
from .io import digest

LEGACY_SCENARIOS = ('label_noise10', 'label_noise20', 'gain10', 'gain25', 'rotation10', 'rotation25')
COUPLED_SCENARIOS = ('coupled_ks15', 'coupled_sk15', 'coupled_null15',
                     'coupled_ks35', 'coupled_sk35', 'coupled_null35')
TIME_SCENARIOS = ('time_cov_plus35', 'time_cov_minus35')
SCENARIOS = LEGACY_SCENARIOS + COUPLED_SCENARIOS + TIME_SCENARIOS


def coupled_generators(plan):
    """Return session-fixed K_non, S, K_null and seed in native CAR axes.

    Directions are generated on CPU without touching global RNG state.  They
    depend on participant/session only, never the pipeline seed, source
    reference, support budget, observed data or labels.  The null uses exactly
    the same S and an isospectral nonzero rotation generator; only its rotation
    plane changes from unequal-gain axes to the repeated-gain eigenspace.
    """
    seed = int(digest(['paper3-round5-coupled-drift-v1', plan.participant_id,
                       plan.session_id])[:15], 16)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn(10, 10, generator=generator, dtype=torch.float64)
    rotation, r = torch.linalg.qr(raw)
    signs = torch.where(torch.diagonal(r) < 0, -1., 1.)
    rotation = rotation * signs[None]
    k = torch.zeros(10, 10, dtype=torch.float64)
    k[0, 2], k[2, 0] = -1., 1.
    k = k / k.norm()
    k_null = torch.zeros_like(k)
    k_null[0, 1], k_null[1, 0] = -1., 1.
    k_null = k_null / k_null.norm()
    diagonal = torch.randn(10, generator=generator, dtype=torch.float64)
    diagonal[0], diagonal[1], diagonal[2] = -.75, -.75, .75
    diagonal -= diagonal.mean()
    s = torch.diag(diagonal / diagonal.norm())
    return (rotation @ k @ rotation.T, rotation @ s @ rotation.T,
            rotation @ k_null @ rotation.T, seed)


def coupled_matrix(k, s, strength, order):
    """Build a controlled two-factor map with ||a(K+S)||_F=strength.

    ``commuting`` is a canonical null construction, not an additional learned
    calibration model.  KS and SK retain the same rotation and gain spectrum;
    rightmost factors act first on column-vector signals.
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Finite nonnegative coupled-drift strength required')
    a = float(strength) / math.sqrt(2)
    if order == 'ks':
        return torch.matrix_exp(a * k) @ torch.matrix_exp(a * s)
    if order == 'sk':
        return torch.matrix_exp(a * s) @ torch.matrix_exp(a * k)
    if order == 'commuting':
        return torch.matrix_exp(a * (k + s))
    raise ValueError('Unknown coupled-drift order: ' + str(order))


def coupled_operator(plan, scenario):
    """Return the native-CAR 10-D operator and an auditable synthetic recipe."""
    if scenario not in COUPLED_SCENARIOS:
        raise ValueError('Unknown coupled diagnostic scenario: ' + str(scenario))
    k, s, k_null, seed = coupled_generators(plan)
    strength = .15 if scenario.endswith('15') else .35
    null = scenario.startswith('coupled_null')
    order = 'commuting' if null else 'ks' if scenario.startswith('coupled_ks') else 'sk'
    k = k_null if null else k
    operator = coupled_matrix(k, s, strength, order)
    a = strength / math.sqrt(2)
    singular = torch.linalg.svdvals(operator)
    metadata = {'synthetic_stress': True, 'scenario': scenario, 'direction_seed': seed,
        'scope': 'same_shift_support_and_future_query', 'support_changed': True,
        'query_changed': True, 'labels_changed': False, 'source_reference_used': False,
        'coordinates': 'native_10d_CAR_not_source_whitened', 'order': order,
        'commuting_null': null, 'severity': strength,
        'car_generator_norm': float((a * (k + s)).norm()),
        'rotation_generator_norm': float((a * k).norm()),
        'gain_generator_norm': float((a * s).norm()),
        'commutator_norm': float((a * a * (k @ s - s @ k)).norm()),
        'car_transform_deviation': float((operator - torch.eye(10, dtype=operator.dtype)).norm()),
        'car_condition': float(singular[0] / singular[-1]),
        'car_determinant': float(torch.linalg.det(operator)),
        'ks_sk_gain_spectrum_matched': True,
        'gain_spectrum_matched_to_noncommuting': True,
        'rotation_spectrum_matched_to_noncommuting': True,
        'null_gain_spectrum_matched_to_noncommuting': True,
        'null_transform_displacement_matched_to_noncommuting': False,
        'matching_scope': 'same_gain_generator_and_rotation_angle_spectrum; relative_eigenaxes_change',
        'units': 'native_CAR_log_generator_Frobenius_norm_not_percent_gain',
        'claim': 'controlled offline counterfactual shift, not a measured acquisition mechanism'}
    return operator, metadata

def perturb_episode(episode, geometry, plan, scenario):
    if scenario == 'clean':
        return episode, {"synthetic_stress": False}
    if scenario not in SCENARIOS:
        raise ValueError('Unknown diagnostic scenario: ' + scenario)
    if scenario in TIME_SCENARIOS:
        from .temporal import common_time_shift
        amount = (.35 if scenario == 'time_cov_plus35' else -.35) / math.sqrt(10)
        before_support, before_query = episode.support.x.double(), episode.query_x.double()
        support = common_time_shift(before_support, amount)
        query = common_time_shift(before_query, amount)
        def covariance_error(before, after):
            first, second = before@before.transpose(-1,-2), after@after.transpose(-1,-2)
            return float((second-first).norm()/first.norm().clamp_min(1e-300)) if len(before) else 0.
        values = dict(vars(episode))
        values['support'] = replace(episode.support, x=support)
        values['query_x'] = query
        return SimpleNamespace(**values), {
            'synthetic_stress': True, 'scenario': scenario,
            'scope': 'same_common_time_orthogonal_map_support_and_future_query',
            'support_changed': True, 'query_changed': True, 'labels_changed': False,
            'source_reference_used': False, 'pipeline_seed_used': False,
            'severity': .35, 'signed_time_parameter': amount,
            'zero_lag_covariance_invariant': True, 'joint_RMS_invariant': True,
            'support_covariance_relative_error': covariance_error(before_support, support),
            'query_covariance_relative_error': covariance_error(before_query, query),
            'operator': 'zero_boundary', 'finite_trial_noncausal': True,
            'filter_bank_covariance_invariance_claimed': False,
            'claim': 'offline counterfactual mechanism; not a measured acquisition shift'}
    if scenario in COUPLED_SCENARIOS:
        transform, metadata = coupled_operator(plan, scenario)
        # The same native-CAR shift is applied before the model's joint RMS to
        # both acquired support and fixed future queries. No Rref enters here.
        q = geometry.q.double()
        operator = q.T @ transform.to(q) @ q
        values = dict(vars(episode))
        values['support'] = replace(episode.support, x=operator @ episode.support.x.double())
        values['query_x'] = operator @ episode.query_x.double()
        return SimpleNamespace(**values), metadata
    kind = 'labels' if scenario.startswith('label_noise') else 'gain' if scenario.startswith('gain') else 'rotation'
    # The same direction/permutation is shared across severities, models and pipeline seeds.
    seed = int(digest(['paper3-followup-stress-v1', plan.participant_id, plan.session_id, kind])[:15],16)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    values = dict(vars(episode))
    metadata = {'synthetic_stress': True, 'scenario': scenario, 'direction_seed': seed}
    if kind == 'labels':
        fraction = .1 if scenario.endswith('10') else .2
        y = episode.support.y.clone()
        indices = torch.randperm(len(y), generator=generator)[:round(len(y)*fraction)].to(y.device)
        y[indices] = 1-y[indices]
        values['support'] = replace(episode.support, y=y)
        metadata.update(label_flip_fraction_requested=fraction, label_flips_actual=len(indices),
                        affected_support_ids=[episode.support.event_uids[i] for i in indices.cpu().tolist()],
                        query_changed=False)
    else:
        strength = .1 if scenario.endswith('10') else .25
        if kind == 'gain':
            d = torch.randn(10, generator=generator, dtype=torch.float64)
            d -= d.mean(); direction = torch.diag(d)
        else:
            raw = torch.randn(10,10,generator=generator,dtype=torch.float64)
            direction = (raw-raw.T)/2
        direction *= strength/direction.norm()
        # Apply a common CAR-coordinate shift after calibration, independent of each f0's Rref.
        operator = geometry.q.T @ torch.matrix_exp(direction.to(geometry.q)) @ geometry.q
        values['query_x'] = operator @ episode.query_x.double()
        metadata.update(log_generator_frobenius_norm=strength, support_changed=False,
                        query_changed=True, units='CAR_subspace_log_transform_norm_not_percent_gain')
    return SimpleNamespace(**values), metadata
