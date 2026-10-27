"""CPU-only integration of actual spatial kernels, receipts, DAG and reporting.

All signals are synthetic and explicitly marked; no school Slurm, GPU or
network is contacted and no scientific completion is manufactured by fixtures.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper3m12 import common, native_data, runtime, worker

SEEDS = (20261018, 20261019, 20261020)


class FrozenSyntheticSource(torch.nn.Module):
    """A differentiable frozen64-feature source in the real native wrapper."""
    feature_dim = 64

    def __init__(self, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(4, 64, dtype=torch.float64,
                                                     generator=generator)*.2, requires_grad=False)
        self.head = torch.nn.Parameter(torch.randn(64, 2, dtype=torch.float64,
                                                   generator=generator)*.2, requires_grad=False)
        self.eval()

    def features_and_logits(self, x):
        features = torch.tanh((x.mean(-1)+.2*x.square().mean(-1)) @ self.weight)
        return features, features @ self.head


@dataclass(frozen=True)
class Support:
    x: object
    y: object
    event_uids: tuple
    acquired_event_uids: tuple
    signal_sha: str = 'explicit-synthetic-integration'
    role: str = 'meta_train_support'


class NativeSyntheticProvider:
    """Synthetic access boundary around genuine exp(M), EA and frozen sources."""
    def __init__(self, output):
        self.output = Path(output)
        geometry_module = native_data._data_module().native('p3work.spatial_geometry')
        self.geometry = geometry_module.SpatialGeometry.from_reference(torch.eye(3, dtype=torch.float64)/3,
                                                                       channels=4)
        generator = torch.Generator().manual_seed(91)
        dimension = self.geometry.e.shape[1]
        self.basis = torch.linalg.qr(torch.randn(dimension, 4, dtype=torch.float64,
                                                generator=generator), mode='reduced').Q
        self.x = torch.randn(20, 4, 8, dtype=torch.float64, generator=generator)
        self.x -= self.x.mean(1, keepdim=True)
        self.y = torch.tensor([0, 1]*10, dtype=torch.long)
        self.q = torch.randn(4, 4, 8, dtype=torch.float64, generator=generator)
        self.q -= self.q.mean(1, keepdim=True)
        # Identical inputs with balanced labels give BA=.5 for every model,
        # exercising a genuine prespecified negative decision deterministically.
        self.q = self.q[:1].expand(4, -1, -1).clone()
        self.c = torch.randn(8, 4, 8, dtype=torch.float64, generator=generator)
        self.c -= self.c.mean(1, keepdim=True)
        self.sources = {seed: FrozenSyntheticSource(seed) for seed in SEEDS}

    def participants(self, role):
        return ['P1'] if role == 'pilot' else ['V1']

    def sessions(self, role, participant):
        return [SimpleNamespace(participant_id=participant, session_id='s1', role=role,
                                identity=role+'-'+participant+'-sealed-query-identity')]

    def support(self, session, budget):
        uids = tuple(session.identity+'-s'+str(i) for i in range(budget))
        return Support(self.x[:budget], self.y[:budget], uids, uids)

    def model(self, seed):
        return native_data.NativeMarginModel(self.sources[seed], self.geometry, self.basis,
                                             identity={'seed': seed, 'synthetic_fixture': True})

    def episode_model(self, seed, support):
        alignment, error = native_data._fit_ea(self.geometry, support.x)
        assert error is None
        return native_data.NativeMarginModel(self.sources[seed], self.geometry, self.basis,
            alignment=alignment, identity={'seed': seed, 'synthetic_fixture': True,
                                           'support_uid_sha256': common.digest(support.event_uids)})

    def reference(self, seed):
        return dict(x=self.c, event_uids=tuple('source-reference-'+str(i) for i in range(len(self.c))),
                    identity={'reference': 'source-only-synthetic', 'source_seed': seed},
                    reference_transform=np.eye(3))

    def query_inputs(self, session):
        return SimpleNamespace(x=self.q, event_uids=tuple(session.identity+'-q'+str(i) for i in range(4)),
                               identity=session.identity)

    def evaluate_labels(self, session, path, expected):
        # Assert the real worker has fixed and sealed predictions before labels.
        assert common.sha_file(path) == expected['sha256']
        assert (Path(path).parent/'prediction-seal.json').is_file()
        assert (Path(path).parent/'states.json').is_file()
        with np.load(path, allow_pickle=False) as values:
            assert np.isfinite(values['all_probabilities']).all()
            assert str(values['session_identity'].item()) == session.identity
        return np.array([0, 1, 0, 1])

    def perturb(self, seed, x, generator, amplitude):
        theta = torch.zeros(4, dtype=torch.float64); theta[generator] = amplitude
        model = self.model(seed)
        m = torch.einsum('r,rij->ij', theta, model._generators)
        transform = model._left @ (torch.matrix_exp(m)-torch.eye(m.shape[0], dtype=m.dtype)) @ model._right
        return x+transform @ x


def _scientific_process(plan_path, slot):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['SLURM_JOB_ID'] = 'synthetic-'+str(slot)
    _, plan = common.load_plan(plan_path)
    provider = NativeSyntheticProvider(plan['output'])
    def science(plan, task, **kwargs):
        return worker.run_task(plan, task, provider_factory=lambda *_: provider, **kwargs)
    raise SystemExit(runtime.worker_loop(plan_path, slot, run_task=science, device='cpu'))


def _plan(tmp_path):
    config = common.read_json(ROOT/'EXPERIMENT_CONFIG.json')
    config['data']['acquired_budgets'] = [20]
    config['optimizer'].update(primary_steps=100, secondary_steps=2)
    config['model']['random_draws'] = 2
    config['statistics'].update(bootstrap_resamples=16, large_n_signflip_draws=16)
    tasks = []
    for seed in SEEDS:
        tasks.append(dict(key='calibrate-'+str(seed), kind='calibration', seed=seed,
                          role='reference', dependencies=[]))
    calibrations = [row['key'] for row in tasks]
    tasks.append(dict(key='freeze-tau', kind='freeze_tau', dependencies=calibrations))
    for seed in SEEDS:
        tasks.append(dict(key='paired-pilot-P1-s'+str(seed), kind='paired_block', role='pilot',
                          participant='P1', seed=seed, dependencies=['freeze-tau']))
    paired = [row['key'] for row in tasks if row['kind'] == 'paired_block']
    tasks.append(dict(key='pilot-decision', kind='pilot_decision', dependencies=paired))
    for kind in ('head_block', 'shift_block'):
        tasks.append(dict(key=kind+'-pilot-P1', kind=kind, role='pilot', participant='P1', seed=SEEDS[0],
                          dependencies=['pilot-decision', 'freeze-tau']))
    gate = dict(task='pilot-decision', field='decision', equals='GO')
    tasks.append(dict(key='gated-validation', kind='paired_block', role='meta_validation',
                      participant='V1', seed=SEEDS[0], dependencies=['pilot-decision'], gate=gate))
    tasks.append(dict(key='development-freeze', kind='development_freeze',
                      dependencies=['gated-validation', 'pilot-decision'], gate=gate))
    tasks.append(dict(key='final-report', kind='final_report', dependencies=[row['key'] for row in tasks],
                      allow_skipped_dependencies=True, allow_blocked_dependencies=True))
    plan = dict(root=str(tmp_path), output=str(tmp_path/'output'), tasks=tasks,
                synthetic_fixture=True, start_sidecars=False, recipe_sha256=common.digest(config),
                config=config, participants={'pilot': ['P1'], 'meta_validation': ['V1']})
    path = tmp_path/'plan.json'; common.atomic_json(path, plan)
    return path, common.load_plan(path)[1]


def test_actual_native_science_four_process_dag_tau_decision_secondary_and_report(tmp_path):
    path, plan = _plan(tmp_path)
    ctx = multiprocessing.get_context('spawn')
    processes = [ctx.Process(target=_scientific_process, args=(str(path), slot)) for slot in range(4)]
    for process in processes:
        process.start()
    deadline = time.monotonic()+90
    try:
        for process in processes:
            process.join(max(.1, deadline-time.monotonic()))
        status = runtime.status(plan)
        assert all(not process.is_alive() for process in processes), status
        assert all(process.exitcode == 0 for process in processes), status
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()  # Only this test-owned CPU fixture process.
                process.join(3)
    states = runtime.task_states(plan)
    assert not any(row['phase'].startswith('BLOCKED') for row in states.values()), states
    completed = [(task, common.verify_result(plan, task, deep=True)) for task in plan['tasks']]
    done = {task['key']: result for task, result in completed if result is not None}
    assert all(result['synthetic_fixture'] is True for result in done.values())
    assert all(row.get('attempt', 0) == 1 for row in states.values() if row['phase'] == 'COMPLETE')
    tau = common.read_json(Path(plan['output'])/'frozen_tau.json')
    assert tau['support_only'] is True and tau['observation_counts'] == {'20': 3}
    assert done['pilot-decision']['decision'] == 'STOP'
    assert states['gated-validation']['phase'] == 'SKIPPED_DECISION'
    assert states['development-freeze']['phase'] == 'SKIPPED_DECISION'
    assert done['head_block-pilot-P1']['completed_units'] == 1
    assert done['shift_block-pilot-P1']['completed_units'] == 9
    assert done['final-report']['independent_data_status'] == 'DECLARED_NOT_AVAILABLE'
    assert status['status'] == 'STOPPED_BY_PREDECLARED_RESEARCH_RULE'
    # Numerical progress from actual science, distinct from a queue RUNNING flag.
    for task in plan['tasks']:
        if task['kind'] == 'paired_block' and task['role'] == 'pilot':
            result = done[task['key']]
            with (Path(plan['output'])/result['metrics_csv']['path']).open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) > 10
            assert all(0 <= float(row['BA']) <= 1 for row in rows)
            assert common.read_json(common.task_folder(plan, task)/'progress.json')['completed_units'] == 1
    # A repeat worker observes receipts without constructing science or refitting.
    assert runtime.worker_loop(path, 0, device='cpu',
        run_task=lambda *_a, **_kw: pytest.fail('scientific fit reran after completed DAG')) == 0


def test_receipts_are_bound_to_frozen_recipe(tmp_path):
    path, plan = _plan(tmp_path)
    task = plan['tasks'][0]
    result = dict(status='COMPLETE', task_key=task['key'], task_identity=common.digest(task),
                  recipe_sha256='a-different-frozen-recipe', files=[])
    common.atomic_json(common.task_folder(plan, task)/'result.json', result)
    assert common.verify_result(plan, task) is None
    result['recipe_sha256'] = plan['recipe_sha256']
    common.atomic_json(common.task_folder(plan, task)/'result.json', result)
    assert common.verify_result(plan, task) == result
    assert runtime._receipt(plan, task) == result
    assert runtime._receipt(dict(plan, recipe_sha256='changed-recipe'), task) is None
