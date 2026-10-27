from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from paper3m12 import algorithm, worker
from paper3m12.common import atomic_json, digest, read_json, sha_file
from paper3m12.metrics import session_metrics


@dataclass(frozen=True)
class Support:
    x: object
    y: object
    event_uids: tuple
    acquired_event_uids: tuple
    signal_sha: str = 'fixture'
    role: str = 'meta_train_support'


class Model:
    dimension = 4
    device = 'cpu'
    dtype = torch.float64
    feature_dim = 4

    def __init__(self, offset=None):
        self.offset = torch.zeros(4, dtype=self.dtype) if offset is None else offset
        self.identity = dict(source='fixed-synthetic-source', offset=self.offset.tolist())
        self.cache_identity = digest(self.identity)

    def features(self, theta, x):
        return torch.as_tensor(x, dtype=self.dtype)-self.offset

    def reference_features(self, theta, x):
        return torch.as_tensor(x, dtype=self.dtype)

    def _value(self, theta, x, head_w, head_b, reference=False):
        z = self.reference_features(theta, x) if reference else self.features(theta, x)
        fixed = torch.tensor([.7, -.3, .2, .1], dtype=self.dtype)
        value = z@fixed + z@theta + .04*(z.square()@theta.square())
        if head_w is not None:
            value = value + z@torch.as_tensor(head_w, dtype=self.dtype)
        if head_b is not None:
            value = value + head_b
        return value

    def margin(self, theta, x, head_w=None, head_b=None):
        return self._value(theta, x, head_w, head_b)

    def reference_margin(self, theta, x, head_w=None, head_b=None):
        return self._value(theta, x, head_w, head_b, reference=True)


class Provider:
    def __init__(self, *, single_class=False, fail_labels_once=False):
        rng = np.random.default_rng(18)
        self.x = torch.tensor(rng.normal(size=(20, 4)), dtype=torch.float64)
        self.y = torch.tensor(np.zeros(20) if single_class else np.tile([0, 1], 10), dtype=torch.long)
        self.q = torch.tensor(rng.normal(size=(8, 4)), dtype=torch.float64)
        self.labels = np.tile([0, 1], 4)
        self.c = torch.tensor(rng.normal(size=(12, 4)), dtype=torch.float64)
        self.session = SimpleNamespace(participant_id='P1', session_id='S1', role='pilot',
                                       identity='fixed-session-uid')
        self.query_calls = 0; self.label_calls = 0; self.fail_labels_once = fail_labels_once

    def participants(self, role):
        return ['P1']

    def sessions(self, role, person):
        assert person == 'P1'
        return [self.session]

    def support(self, session, budget):
        return Support(self.x[:budget], self.y[:budget], tuple('s'+str(i) for i in range(budget)),
                       tuple('s'+str(i) for i in range(budget)))

    def query_inputs(self, session):
        self.query_calls += 1
        return SimpleNamespace(x=self.q, event_uids=tuple('q'+str(i) for i in range(len(self.q))),
                               identity=session.identity)

    def evaluate_labels(self, session, path, expectedidentity):
        self.label_calls += 1
        assert Path(path).is_file()
        assert sha_file(path) == expectedidentity['sha256']
        with np.load(path, allow_pickle=False) as saved:
            assert saved['probabilities'].shape == (8, 2)
            assert saved['all_probabilities'].shape[0] == len(saved['state_keys'])
            assert str(saved['session_identity'].item()) == session.identity
            assert np.isfinite(saved['all_probabilities']).all()
        assert (Path(path).parent/'states.json').is_file()
        if self.fail_labels_once:
            self.fail_labels_once = False
            raise OSError('simulated evaluator IO after immutable predictions')
        return self.labels.copy()

    def episode_model(self, seed, support):
        return Model(support.x.mean(0) if len(support.x) else None)

    def model(self, seed):
        return Model()

    def reference(self, seed):
        return {'x': self.c, 'event_uids': tuple('c'+str(i) for i in range(len(self.c))),
                'identity': {'reference': 'fixed-source-only'}, 'reference_transform': np.eye(4)}

    def perturb(self, seed, x, generator, amplitude):
        result = x.clone(); result[:, generator] += amplitude
        return result


class FastAlgorithm:
    """Deterministic fitter fixture; scientific kernels have independent tests."""
    def __init__(self, fail_budget_once=None, crash_after_raw=False):
        self.fits = []; self.predict_calls = 0
        self.fail_budget_once = fail_budget_once; self.crash_after_raw = crash_after_raw
        self.resumed_raw = False

    def fit_episode(self, model, x, y, reference, *, budget, seed, config, tau,
                    progress=None, raw_cache=None, checkpoint=None):
        self.fits.append(budget)
        if self.fail_budget_once == budget:
            self.fail_budget_once = None
            raise RuntimeError('simulated independent budget fit failure')
        if self.crash_after_raw:
            self.crash_after_raw = False
            checkpoint('fixture-raw', {'completed_steps': 10, 'theta': [.01, 0, 0, 0]})
            raise RuntimeError('simulated worker failure after durable raw checkpoint')
        if raw_cache and 'fixture-raw' in raw_cache:
            self.resumed_raw = raw_cache['fixture-raw']['completed_steps'] == 10
        if progress:
            progress(stage='actual_fit_advance', outer_update=100, budget=budget)
        states = {
            'support_ea': dict(theta=[0.]*4, method='support_ea', layer='anchor', steps=0,
                               fit_status='ANCHOR', reference_rms=0.),
            'top_q': dict(theta=[.02, -.01, 0, 0], method='top_q', layer='reference_rms',
                          steps=100, fit_status='FITTED', reference_rms=.01,
                          target_reference_rms=.01, match_absolute_error=0.),
            'full_sgd': dict(theta=[.03, -.02, .01, .01], method='full_sgd', layer='raw',
                            steps=100, fit_status='FITTED', reference_rms=.03)}
        for draw in range(5):
            states['random_q_'+str(draw)] = dict(theta=[.01, .002*draw, -.01, 0],
                method='random_q_'+str(draw), layer='reference_rms', steps=100,
                fit_status='FITTED', reference_rms=.01, target_reference_rms=.01, match_absolute_error=0.)
        return dict(states=states, diagnostics=dict(q=2, positive_rank=4, tau_used=tau,
            source_head_frozen=True, weak_deleted_reference_rms=.002), tau_observation=.02)

    def fit_head_controls(self, *args, **kwargs):
        result = self.fit_episode(*args, **kwargs)
        result['diagnostics']['source_head_frozen'] = False
        return result

    def fit_shift_controls(self, *args, **kwargs):
        result = self.fit_episode(*args, **kwargs)
        result['states'] = {key: result['states'][key] for key in ('support_ea', 'full_sgd', 'top_q')}
        return result

    def calibrate_tau(self, model, x, y, reference, *, budget, seed, config, progress=None,
                      raw_cache=None, checkpoint=None):
        assert len(x) == budget and len(y) == budget
        checkpoint('calibration-'+str(budget), {'completed_steps': 10, 'theta': [0.]*4})
        return dict(tau_observation=.01+budget*.001, state={'theta': [0.]*4}, diagnostics={'support_only': True})

    def predict(self, model, state, x):
        self.predict_calls += 1
        return algorithm.predict(model, state, x)


def plan_task(tmp_path, budgets=(4, 8), kind='paired_block'):
    plan = dict(root=str(tmp_path), output=str(tmp_path/'output'), synthetic_fixture=True,
                recipe_sha256='a'*64, config={'data': {'acquired_budgets': list(budgets)}})
    task = dict(key='pilot-P1-s18-'+kind, kind=kind, role='pilot', participant='P1', seed=20261018)
    atomic_json(tmp_path/'output/frozen_tau.json', {'tau_by_budget': {str(b): .02 for b in budgets}})
    return plan, task


def run(plan, task, provider, scientific, **kwargs):
    return worker.run_task(plan, task, device='cpu', provider_factory=lambda *_: provider,
                           algorithm_module=scientific, **kwargs)


def read_metrics(plan, result):
    with (Path(plan['output'])/result['metrics_csv']['path']).open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_real_algorithm_seals_all_predictions_before_labels_and_reuses_completed_task(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); plan['config'].update(
        optimizer={'primary_steps': 12, 'secondary_steps': 4}, model={'random_draws': 2})
    provider = Provider()
    result = run(plan, task, provider, algorithm)
    assert result['status'] == 'COMPLETE' and result['completed_units'] == 1
    assert provider.label_calls == 1
    rows = read_metrics(plan, result)
    assert len(rows) > 10 and all(np.isfinite(float(row['BA'])) for row in rows)
    assert all(row['query_identity'] == provider.session.identity for row in rows)
    before = provider.query_calls, provider.label_calls
    reused = run(plan, task, provider, SimpleNamespace(fit_episode=lambda *a, **k: pytest.fail('completed fit rerun')))
    assert reused == result and (provider.query_calls, provider.label_calls) == before


def test_telemetry_failure_never_stops_durable_scientific_work(tmp_path, monkeypatch):
    plan, task = plan_task(tmp_path); provider = Provider(); scientific = FastAlgorithm()
    original = worker.atomic_json
    def failed_telemetry(path, value):
        if Path(path).name == 'progress.json':
            raise FileNotFoundError('simulated shared-filesystem telemetry staging removal')
        return original(path, value)
    monkeypatch.setattr(worker, 'atomic_json', failed_telemetry)
    result = run(plan, task, provider, scientific,
                 progress=lambda **fields: (_ for _ in ()).throw(MemoryError('reporter fixture failure')))
    assert result['status'] == 'COMPLETE' and scientific.fits == [4, 8]
    assert provider.label_calls == 2 and (Path(plan['output'])/'tasks'/task['key']/'result.json').is_file()


def test_partial_block_keeps_completed_budget_and_retries_only_missing_unit(tmp_path):
    plan, task = plan_task(tmp_path); provider = Provider(); scientific = FastAlgorithm(fail_budget_once=8)
    with pytest.raises(RuntimeError, match='budget fit failure'):
        run(plan, task, provider, scientific)
    assert provider.label_calls == 1
    assert not (Path(plan['output'])/'tasks'/task['key']/'result.json').exists()
    result = run(plan, task, provider, scientific)
    assert result['status'] == 'COMPLETE'
    assert scientific.fits == [4, 8, 8]
    assert provider.label_calls == 2  # B4 evaluation is also reused.


def test_raw_fit_checkpoint_is_reused_after_fit_process_failure(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); provider = Provider(); scientific = FastAlgorithm(crash_after_raw=True)
    with pytest.raises(RuntimeError, match='durable raw checkpoint'):
        run(plan, task, provider, scientific)
    assert provider.query_calls == provider.label_calls == 0
    result = run(plan, task, provider, scientific)
    assert result['status'] == 'COMPLETE' and scientific.resumed_raw


def test_prediction_seal_resume_does_not_refit_or_repredict(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); provider = Provider(fail_labels_once=True); scientific = FastAlgorithm()
    with pytest.raises(OSError, match='evaluator IO'):
        run(plan, task, provider, scientific)
    count = scientific.predict_calls
    result = run(plan, task, provider, scientific)
    # Query predictions reuse the immutable bundle; support scoring runs only
    # after evaluation resumes and is expected to make new small predictions.
    assert result['status'] == 'COMPLETE' and scientific.fits == [4]
    assert scientific.predict_calls == count+len(read_metrics(plan, result))


def test_tampered_completed_artifact_cannot_become_retrained_success(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); provider = Provider(); scientific = FastAlgorithm()
    result = run(plan, task, provider, scientific)
    (Path(plan['output'])/result['metrics_csv']['path']).write_text('corrupt completed science\n')
    with pytest.raises(worker.ScientificArtifactError, match='SHA or size changed'):
        run(plan, task, provider, scientific)
    assert scientific.fits == [4]


def test_single_class_fallback_remains_in_primary_metric_layer(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); provider = Provider(single_class=True)
    result = run(plan, task, provider, algorithm)
    rows = read_metrics(plan, result)
    top = next(row for row in rows if row['method'] == 'top_q' and row['steps'] == '100')
    assert top['layer'] == 'reference_rms' and top['state_layer'] == 'anchor_fallback'
    assert top['fallback_reason'] == 'single_class_support'
    assert top['support_BCE'] == '' and result['status'] == 'COMPLETE'


def test_calibration_never_opens_query_inputs_or_labels(tmp_path):
    plan, task = plan_task(tmp_path, (4, 8), 'calibration'); provider = Provider(); scientific = FastAlgorithm()
    result = run(plan, task, provider, scientific)
    assert result['tau_observations'] == {'4': [.014], '8': [.018000000000000002]}
    assert provider.query_calls == provider.label_calls == 0
    assert result['completed_units'] == 2


def test_known_shift_oracle_predictions_stay_clean_and_resume_with_same_order(tmp_path):
    plan, task = plan_task(tmp_path, (4,), 'shift_block'); provider = Provider(); scientific = FastAlgorithm()
    result = run(plan, task, provider, scientific)
    rows = read_metrics(plan, result)
    oracle = [row for row in rows if row['base_method'] == 'oracle_clean_ea']
    assert len(oracle) == 9 and len({row['BA'] for row in oracle}) == 1
    assert len({row['balanced_NLL'] for row in oracle}) == 1
    assert all(row['inference_scope'] == 'synthetic_intervention_on_real_development_only' for row in rows)
    assert result['completed_units'] == 9
    result_path = Path(plan['output'])/'tasks'/task['key']/'result.json'
    result_path.unlink()  # Simulate lost final aggregation after all episode units committed.
    before = scientific.fits.copy(), provider.label_calls
    again = run(plan, task, provider, scientific)
    assert again['n_rows'] == result['n_rows'] and (scientific.fits, provider.label_calls) == before


def test_secondary_interventions_cannot_run_on_target_holdout(tmp_path):
    plan, task = plan_task(tmp_path, (4,), 'head_block'); task['role'] = 'holdout'
    with pytest.raises(ValueError, match='development-only'):
        run(plan, task, Provider(), FastAlgorithm())


def test_zero_statistical_weight_distortion_from_query_class_imbalance():
    labels = np.asarray([0, 0, 0, 1])
    predicted = np.asarray([[.9, .1], [.9, .1], [.9, .1], [.9, .1]])
    result = session_metrics(labels, predicted, predicted)
    assert result['BA'] == .5
    assert result['corrected_vs_EA'] == result['introduced_vs_EA'] == 0


def test_unknown_kind_is_not_a_completed_noop(tmp_path):
    plan, task = plan_task(tmp_path); task['kind'] = 'not-a-scientific-task'
    with pytest.raises(ValueError, match='Unknown registered scientific'):
        run(plan, task, Provider(), FastAlgorithm())
    assert not (Path(plan['output'])/'tasks'/task['key']/'result.json').exists()


def test_graceful_pause_waits_for_complete_unit_then_reuses_it(tmp_path):
    plan, task = plan_task(tmp_path); provider = Provider(); scientific = FastAlgorithm()
    stop = threading.Event()
    def completed(**event):
        if event.get('stage') == 'paired_unit_complete':
            stop.set()
    with pytest.raises(worker.PauseRequested):
        run(plan, task, provider, scientific, progress=completed, stop=stop)
    assert scientific.fits == [4] and provider.label_calls == 1
    assert not (Path(plan['output'])/'tasks'/task['key']/'result.json').exists()
    stop.clear()
    result = run(plan, task, provider, scientific, stop=stop)
    assert result['status'] == 'COMPLETE' and scientific.fits == [4, 8]
    assert provider.label_calls == 2


def test_completed_result_is_bound_to_recipe_and_never_silently_reused(tmp_path):
    plan, task = plan_task(tmp_path, (4,)); provider = Provider(); scientific = FastAlgorithm()
    run(plan, task, provider, scientific)
    plan['recipe_sha256'] = 'b'*64
    with pytest.raises(worker.ScientificArtifactError, match='identity/status differs'):
        run(plan, task, provider, scientific)
    assert scientific.fits == [4]


def test_real_head_competition_and_compact_diagnostics_are_sealed(tmp_path):
    plan, task = plan_task(tmp_path, (4,), 'head_block')
    plan['config'].update(optimizer={'primary_steps': 3, 'secondary_steps': 1}, model={'random_draws': 2})
    provider = Provider(); result = run(plan, task, provider, algorithm)
    rows = read_metrics(plan, result)
    assert any(row['method'] == 'head_only_full' for row in rows)
    full = [row for row in rows if row['method'] == 'top_q_head_full']
    assert full and all(row['source_head_parameters_frozen'] == 'True' for row in full)
    assert all(row['source_head_frozen'] == 'False' for row in full)
    assert all(row['task_kind'] == 'head_block' for row in rows)
    assert result['primary_eligible'] is False and provider.label_calls == 1
    diagnostics = read_json(Path(plan['output'])/result['mechanism_diagnostics']['path'])
    assert len(diagnostics['episodes']) == 1
    assert diagnostics['episodes'][0]['diagnostics']['delta_head_controls'] is True
    assert result['mechanism_diagnostics'] in result['files']
    # Frozen source and support-EA use distinct actual model mappings.
    unit = next((Path(plan['output'])/'tasks'/task['key']/'units').glob('*/B4/clean'))
    with np.load(unit/'predictions.npz', allow_pickle=False) as saved:
        keys = saved['state_keys'].tolist(); values = saved['all_probabilities']
        assert not np.array_equal(values[keys.index('frozen')], values[keys.index('support_ea')])


def test_known_shift_clean_control_reuses_sealed_raw_pilot_fit_only(tmp_path):
    class RawFixture(FastAlgorithm):
        def fit_episode(self, *args, **kwargs):
            result = super().fit_episode(*args, **kwargs)
            result['states']['top_q_raw'] = dict(result['states']['top_q'], layer='raw')
            return result
    plan, pilot = plan_task(tmp_path, (4,)); provider = Provider(); scientific = RawFixture()
    shift = dict(pilot, key='shift-P1-s18', kind='shift_block')
    plan['tasks'] = [pilot, shift]
    run(plan, pilot, provider, scientific)
    scientific.fits.clear()
    result = run(plan, shift, provider, scientific)
    # Clean control reuses sealed support states; eight perturbations fit anew.
    assert scientific.fits == [4]*8 and result['completed_units'] == 9
    diagnostics = read_json(Path(plan['output'])/result['mechanism_diagnostics']['path'])
    clean = next(episode for episode in diagnostics['episodes'] if episode['intervention'] == 'clean')
    assert clean['diagnostics']['query_metrics_used_for_reuse'] is False
    assert clean['diagnostics']['reused_clean_support_fit']['path'].startswith('tasks/'+pilot['key'])
