"""Predeclared synthetic stress tests. Never overwrite cached EEG or true labels."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from .io import digest

SCENARIOS = ('label_noise10', 'label_noise20', 'gain10', 'gain25', 'rotation10', 'rotation25')

def perturb_episode(episode, geometry, plan, scenario):
    if scenario == 'clean':
        return episode, {"synthetic_stress": False}
    if scenario not in SCENARIOS:
        raise ValueError('Unknown diagnostic scenario: ' + scenario)
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
