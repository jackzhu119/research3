from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper3m12 import native_data as n


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return n.describe(path, path.parents[0])


@pytest.fixture
def metadata_study(tmp_path, monkeypatch):
    root = tmp_path/'project'; out = root/'runs/m12'; root.mkdir()
    split = {'partitions': {'meta_train': ['S'+str(i) for i in range(40)],
                           'meta_validation': ['V'+str(i) for i in range(11)],
                           'internal_holdout': ['H'+str(i) for i in range(11)]},
             'split_sha256': 'b'*64}
    split_path = root/'cache/split.json'; write(split_path, split)
    manifest = {'status': 'STIEGER_CACHE_COMPLETE', 'input_contract_sha256': 'a'*64,
                'participant_split': n.describe(split_path, root), 'sessions': []}
    manifest_path = root/'cache/manifest.json'; write(manifest_path, manifest)
    identity = {'manifest_sha256': n.sha(manifest_path), 'split_sha256': 'b'*64,
                'signal_contract_sha256': 'a'*64}
    registry = {'schema_version': 'paper3-r11-study-registry-v1', 'root': str(root),
                'internal_cache_manifest': n.describe(manifest_path, root),
                'internal_data_identity': identity, 'source_by_seed': {}}
    r7out = root/'runs/followup_20261024_r7'; r7out.mkdir(parents=True)
    tasks = [{'family': 'basis', 'key': f'basis-static_ce-r4-seed{seed}', 'seed': seed,
              'config': {'rank': 4, 'mode': 'static', 'objective': 'ce'}, 'updates': 1000}
             for seed in n.SEEDS]
    r7plan = {'root': str(root), 'output': str(r7out), 'recipe_sha256': 'c'*64, 'tasks': tasks}
    write(r7out/'plan.json', r7plan)
    for seed, task in zip(n.SEEDS, tasks):
        folder = root/'runs/source_training_v2'; folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder/f'f0-{seed}.pt'; checkpoint.write_bytes(b'fixed model bytes')
        source = {'status': 'SOURCE_F0_COMPLETE', 'pipeline_seed': seed,
                  'signal_contract_sha256': 'a'*64, 'checkpoint': n.describe(checkpoint, root)}
        source_path = folder/f'f0_receipt_seed-{seed}.json'; write(source_path, source)
        reference_folder = root/f'runs/training_campaign_v1/references/seed-{seed}'
        reference_folder.mkdir(parents=True)
        array = reference_folder/'reference.npy'; array.write_bytes(b'fixed small reference bytes')
        reference = {'status': 'SOURCE_REFERENCE_COMPLETE', 'source_f0_receipt_sha256': n.sha(source_path),
                     'signal_contract_sha256': 'a'*64,
                     'arrays': {'reference': n.describe(array, root)}}
        reference_path = reference_folder/'receipt.json'; write(reference_path, reference)
        registry['source_by_seed'][str(seed)] = {'source_receipt': n.describe(source_path, root),
                                               'reference_receipt': n.describe(reference_path, root)}
        source_identity = {'source_receipt_sha256': n.sha(source_path),
                           'reference_receipt_sha256': n.sha(reference_path),
                           'source_checkpoint_sha256': source['checkpoint']['sha256'],
                           'signal_contract_sha256': 'a'*64}
        folder = r7out/task['key']; folder.mkdir()
        basis_path = folder/'selected.pt'; basis_path.write_bytes(b'fixed selected dictionary bytes')
        selected = {'status': 'COMPLETE', 'task': task, 'source_identity': source_identity,
                    'data_identity': identity, 'holdout_used_for_selection': False,
                    'identity': n._basis_identity(r7plan, task, source_identity, identity),
                    'selected_checkpoint': n.describe(basis_path, root)}
        write(folder/'result.json', selected)
    monkeypatch.setattr(n, '_existing_registry', lambda path: (copy.deepcopy(registry), None))
    return root, out, registry


def test_freeze_inputs_only_small_metadata_and_resume_without_rediscovery(metadata_study, monkeypatch):
    root, out, _ = metadata_study
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    value = n.freeze_inputs(root, out)
    assert value['schema_version'] == n.SCHEMA
    assert len(value['people']['reference']) == 32 and len(value['people']['pilot']) == 8
    assert not set(value['people']['reference']) & set(value['people']['pilot'])
    assert value['discovery_policy']['imports_torch'] is False
    assert value['discovery_policy']['reads_EEG_arrays'] is False
    assert {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file() and 'm12' not in p.parts} == before
    monkeypatch.setattr(n, '_existing_registry', lambda *_: pytest.fail('resume rediscovered old inputs'))
    again = n.freeze_inputs(root, out)
    assert again == value


def test_metadata_freeze_recovers_transient_staging_loss(metadata_study, monkeypatch):
    root, out, _ = metadata_study
    from paper3m12 import common
    replace_original = common.os.replace
    attempts = []
    def replace_once_lost(source, target):
        attempts.append(Path(source))
        if len(attempts) == 1:
            Path(source).unlink()
            raise FileNotFoundError(2, 'Transient shared-filesystem staging loss', str(source))
        return replace_original(source, target)
    monkeypatch.setattr(common.os, 'replace', replace_once_lost)
    result = n.freeze_inputs(root, out)
    assert len(attempts) == 2
    assert all(path.parent.name == '.m12-atomic' for path in attempts)
    assert n.read(out/'frozen-inputs.json')['input_identity'] == result['input_identity']


def test_fixed_dictionary_not_chosen_by_performance(metadata_study):
    root, out, _ = metadata_study
    r7 = root/'runs/followup_20261024_r7'
    value = n.freeze_inputs(root, out)
    for seed, descriptor in value['dictionary_by_seed'].items():
        assert descriptor['task_key'] == f'basis-static_ce-r4-seed{seed}'
        assert descriptor['rank'] == 4 and descriptor['map'] == 'full_generator_static_expM'
    assert not (r7/'new-results').exists()


def test_completed_basis_with_foreign_source_is_refused(metadata_study):
    root, out, _ = metadata_study
    receipt = root/f'runs/followup_20261024_r7/basis-static_ce-r4-seed{n.SEEDS[0]}/result.json'
    value = n.read(receipt); value['source_identity']['source_checkpoint_sha256'] = 'f'*64
    write(receipt, value)
    with pytest.raises(ValueError, match='does not bind'):
        n.freeze_inputs(root, out)
    assert not (out/'frozen-inputs.json').exists()


def test_frozen_input_tamper_refuses_resume(metadata_study):
    root, out, _ = metadata_study
    n.freeze_inputs(root, out)
    path = out/'frozen-inputs.json'; value = n.read(path)
    value['people']['pilot'][0] = value['people']['reference'][0]
    write(path, value)
    with pytest.raises(ValueError, match='different study'):
        n.freeze_inputs(root, out)


def test_worker_runtime_is_deterministic_without_reseeding():
    state = torch.get_rng_state().clone()
    device, actual = n._scientific_runtime('cpu')
    assert device == 'cpu' and actual['device_name'] == 'CPU'
    assert actual['torch_cpu_threads'] == 16
    assert actual['deterministic_algorithms'] and actual['cudnn_deterministic']
    assert not actual['cudnn_benchmark'] and not actual['cuda_matmul_tf32'] and not actual['cudnn_tf32']
    assert actual['cublas_workspace_config'] == ':4096:8' and not actual['rng_reseeded']
    assert torch.equal(state, torch.get_rng_state())


def test_requested_cuda_never_silently_falls_back_or_overrides_driver_check(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requested.*unavailable'):
        n._scientific_runtime('cuda:0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    with pytest.raises(RuntimeError, match='outside.*allocation'):
        n._scientific_runtime('cuda:9')


def native_module():
    return n._data_module()


class TinySource(torch.nn.Module):
    feature_dim = 64
    def __init__(self):
        super().__init__()
        gen = torch.Generator().manual_seed(97)
        self.weight = torch.nn.Parameter(torch.randn(11, 64, dtype=torch.float64, generator=gen)*.1,
                                         requires_grad=False)
        self.head = torch.nn.Parameter(torch.randn(64, 2, dtype=torch.float64, generator=gen)*.1,
                                       requires_grad=False)
        self.eval()
    def features_and_logits(self, x):
        features = torch.tanh((x.mean(-1)+.2*x.square().mean(-1)) @ self.weight)
        return features, features @ self.head


@pytest.fixture
def real_native_geometry():
    module = native_module()
    geo_module = module.native('p3work.spatial_geometry')
    geometry = geo_module.SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64)/10)
    gen = torch.Generator().manual_seed(91)
    basis = torch.linalg.qr(torch.randn(99, 4, dtype=torch.float64, generator=gen), mode='reduced').Q
    source = TinySource()
    return module, geo_module, geometry, basis, source


def test_EA_alignment_matches_original_exact_baseline_and_source_stays_frozen(real_native_geometry):
    module, geo, geometry, basis, source = real_native_geometry
    torch.manual_seed(3); x = torch.randn(6, 11, 16, dtype=torch.float64)
    support = SimpleNamespace(x=x, y=torch.tensor([0, 1, 0, 1, 0, 1]), acquired_event_uids=tuple(range(6)))
    baseline = module.native('paper3r7.baseline')
    native, _ = baseline.fit_reference(source, geometry, support, 'support_EA')
    ours, fallback = n._fit_ea(geometry, x)
    assert fallback is None and torch.equal(ours, native['transform'])
    before = [p.detach().clone() for p in source.parameters()]
    model = n.NativeMarginModel(source, geometry, basis, alignment=ours)
    theta = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    margin = model.margin(theta, x)
    expected = baseline.reference_logits(source, geometry, native, x)
    assert torch.allclose(margin, expected[:, 1]-expected[:, 0], atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(margin.sum(), theta)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(torch.equal(a, b) for a, b in zip(before, source.parameters()))
    assert all(not p.requires_grad for p in source.parameters())


def test_static_full_generator_input_gradient_and_reference_EA_separation(real_native_geometry):
    _, _, geometry, basis, source = real_native_geometry
    gen = torch.Generator().manual_seed(14)
    x = torch.randn(4, 11, 16, dtype=torch.float64, generator=gen)
    ea, _ = n._fit_ea(geometry, x)
    model = n.NativeMarginModel(source, geometry, basis, alignment=ea)
    value = torch.tensor([.01, -.02, .03, -.01], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(model.margin(value, x).sum(), value)[0]
    finite = []
    for coordinate in range(4):
        step = torch.zeros(4, dtype=torch.float64); step[coordinate] = 1e-5
        finite.append((model.margin(value.detach()+step, x).sum()
                       - model.margin(value.detach()-step, x).sum())/2e-5)
    assert torch.allclose(gradient, torch.stack(finite), atol=1e-7, rtol=1e-5)
    other = n.NativeMarginModel(source, geometry, basis, alignment=None)
    assert torch.equal(model.reference_margin(value, x), other.margin(value, x))
    assert not torch.allclose(model.margin(value, x), model.reference_margin(value, x), atol=1e-6)


@pytest.fixture
def provider_fixture(tmp_path, monkeypatch, real_native_geometry):
    module, _, geometry, basis, source = real_native_geometry
    root = tmp_path/'root'; root.mkdir()
    state_identity = 'd'*64; path = root/'basis.pt'
    torch.save({'identity': state_identity, 'b': basis}, path)
    source_identity = {'source_checkpoint_sha256': 'a'*64}
    people = {'meta_train': ['P'] + ['R'+str(i) for i in range(32)], 'pilot': ['P'],
              'reference': ['R'+str(i) for i in range(32)], 'meta_validation': ['V'],
              'internal_holdout': ['H']}
    payload = {'schema_version': n.SCHEMA, 'root': str(root), 'registry': {}, 'people': people,
               'source_by_seed': {str(n.SEEDS[0]): {'identity': source_identity}},
               'dictionary_by_seed': {str(n.SEEDS[0]): {'checkpoint': n.describe(path, root),
                    'native_identity': state_identity, 'field': 'b'}}}
    payload['input_identity'] = n.digest(payload)
    gen = np.random.default_rng(7); query_allowed = {'value': False, 'reads': 0}
    class GuardedRow(dict):
        def __getitem__(self, key):
            if key == 'label' and self.get('is_query'):
                query_allowed['reads'] += 1
                if not query_allowed['value']:
                    raise AssertionError('Target query label was opened before prediction seal')
            return super().__getitem__(key)
    plans = []; arrays = {}
    for person in people['meta_train']:
        rows = {}
        for i in range(40):
            uid = f'{person}:event:{i}'
            row = dict(valid=True, cache_index=i, label=i % 2, physical_run_id=str(i//20), is_query=i>=20)
            rows[uid] = GuardedRow(row) if person == 'P' else row
        session = SimpleNamespace(participant_id=person, session_id='1', rows=rows,
            audit={'acquired_prefix_uids': [f'{person}:event:{i}' for i in range(20)],
                   'query_uids': [f'{person}:event:{i}' for i in range(20, 40)]}, role='meta_train')
        plans.append(session); arrays[person] = gen.normal(size=(40, 11, 16))
    catalog = SimpleNamespace(role='meta_train', plans=plans, people=people['meta_train'],
            by_person={p: [s for s in plans if s.participant_id == p] for p in people['meta_train']},
            signal_sha='f'*64, identity={'cache': 'same'})
    catalog.array = lambda session: arrays[session.participant_id]
    calls = {'load': 0}
    def load(*args, **kwargs):
        calls['load'] += 1
        return source, geometry, source_identity
    adapter = SimpleNamespace(SessionAccess=module.SessionAccess,
                              native=module.native, load_source=load,
                              open_catalog=lambda *args, **kwargs: catalog,
                              frozen_source_signature=module.frozen_source_signature,
                              assert_source_unchanged=module.assert_source_unchanged)
    monkeypatch.setattr(n, '_data_module', lambda: adapter)
    provider = n.DataProvider({'root': str(root), 'native_inputs': payload}, 'cpu')
    return provider, query_allowed, calls, root, arrays


def test_support_query_capabilities_and_sealed_prediction_labels(provider_fixture):
    provider, guard, _, root, _ = provider_fixture
    session = provider.sessions('pilot', 'P')[0]
    assert not hasattr(session, 'rows')
    support = provider.support(session, 4)
    assert len(support.event_uids) == 4 and len(support.acquired_event_uids) == 4
    query = provider.query_inputs(session)
    assert not hasattr(query, 'y') and guard['reads'] == 0
    path = root/'sealed.npz'
    with pytest.raises(FileNotFoundError):
        provider.evaluate_labels(session, path, {'sha256':'0'*64, 'session_identity': query.identity})
    assert guard['reads'] == 0
    np.savez(path, event_uids=np.asarray(query.event_uids), session_identity=np.asarray(query.identity),
             probabilities=np.tile([.5, .5], (len(query.event_uids), 1)))
    guard['value'] = True
    labels = provider.evaluate_labels(session, path, {'sha256': n.sha(path), 'session_identity': query.identity})
    assert labels.shape == (20,) and guard['reads'] == 20
    with pytest.raises(ValueError, match='seal SHA'):
        provider.evaluate_labels(session, path, '0'*64)


def test_reference64_disjoint_from_pilot_and_source_cache_reused(provider_fixture):
    provider, guard, calls, _, _ = provider_fixture
    reference = provider.reference(n.SEEDS[0], provider.participants('reference'))
    assert reference['x'].shape == (64, 11, 16)
    assert len(set(reference['event_uids'])) == 64
    assert all(not uid.startswith('P:') for uid in reference['event_uids'])
    assert reference['identity']['target_query_labels_read'] is False and guard['reads'] == 0
    assert provider.reference(n.SEEDS[0]) is reference
    session = provider.sessions('pilot', 'P')[0]
    for budget in (4, 8, 20):
        support = provider.support(session, budget)
        model = provider.episode_model(n.SEEDS[0], support)
        result = model.reference_margin(torch.zeros(4, dtype=torch.float64), reference['x'])
        assert result.shape == (64,) and torch.isfinite(result).all()
    assert calls['load'] == 1 and guard['reads'] == 0


def test_worker_real_adapter_and_spatial_algorithm_seal_before_evaluation(provider_fixture, monkeypatch):
    """Exercise the assembled worker API, not a hand-built prediction bundle."""
    from paper3m12 import worker
    provider, guard, calls, root, _ = provider_fixture
    task = dict(key='native-worker-smoke', kind='paired_block', role='pilot',
                participant='P', seed=n.SEEDS[0])
    plan = dict(root=str(root), output=str(root/'runs/native-worker-smoke'),
                synthetic_fixture=True, recipe_sha256='a'*64, tasks=[task],
                tau_by_budget={'4': .02}, config={
                    'data': {'acquired_budgets': [4]},
                    'optimizer': {'primary_steps': 1, 'secondary_steps': 1},
                    'model': {'random_draws': 1}})
    original = provider.evaluate_labels
    def labels_after_archive(session, path, identity):
        assert Path(path).is_file() and Path(path).with_name('prediction-seal.json').is_file()
        assert Path(path).with_name('states.json').is_file()
        assert guard['reads'] == 0
        guard['value'] = True
        return original(session, path, identity)
    monkeypatch.setattr(provider, 'evaluate_labels', labels_after_archive)
    result = worker.run_task(plan, task, device='cpu', provider_factory=lambda *_: provider)
    assert result['status'] == 'COMPLETE' and result['synthetic_fixture'] is True
    assert guard['reads'] == 20 and calls['load'] == 1
    again = worker.run_task(plan, task, device='cpu', provider_factory=lambda *_: provider)
    assert again == result and guard['reads'] == 20


def test_known_spatial_shift_is_invertible_on_CAR_and_leaves_source_input_unchanged(provider_fixture):
    provider, _, _, _, _ = provider_fixture
    session = provider.sessions('pilot', 'P')[0]; support = provider.support(session, 8)
    before = support.x.clone()
    shifted = provider.perturb(n.SEEDS[0], support.x, 0, .05)
    recovered = provider.perturb(n.SEEDS[0], shifted, 0, -.05)
    geo = provider._source(n.SEEDS[0])[1]
    normal = native_module().native('p3work.spatial_geometry').joint_trial_rms
    expected = normal((geo.q.T @ geo.q) @ support.x)
    assert torch.allclose(recovered, expected, atol=1e-10, rtol=1e-10)
    assert torch.equal(before, support.x)
    with pytest.raises(ValueError, match='predeclared'):
        provider.perturb(n.SEEDS[0], support.x, 3, .05)


def test_cached_source_mutation_is_detected_without_full_source_reload(provider_fixture):
    provider, _, calls, _, _ = provider_fixture
    source, _, _ = provider._source(n.SEEDS[0])
    with torch.no_grad():
        next(source.parameters()).add_(.001)
    with pytest.raises(ValueError, match='mutated'):
        provider.model(n.SEEDS[0])
    assert calls['load'] == 1


def test_actual_compact_eegnet_float32_source_retains_spatial_gradients(real_native_geometry):
    module, _, geometry, basis, _ = real_native_geometry
    factory = module.native('p3work.f0_bridge').CompactEEGNet
    frozen = module.native('paper3r7.data').FrozenSource(factory(11, 128, .25))
    frozen.eval()
    gen = torch.Generator().manual_seed(27)
    support_x = torch.randn(4, 11, 128, dtype=torch.float64, generator=gen)
    alignment, _ = n._fit_ea(geometry, support_x)
    model = n.NativeMarginModel(frozen, geometry, basis, alignment=alignment)
    theta = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    margin = model.margin(theta, support_x)
    baseline = module.native('paper3r7.baseline')
    direct = baseline.reference_logits(frozen, geometry,
        {'variant': 'support_EA', 'fallback_reason': None, 'transform': alignment}, support_x)
    assert torch.equal(margin, (direct[:, 1]-direct[:, 0]).double())
    gradient = torch.autograd.grad(margin.sum(), theta)[0]
    assert margin.shape == (4,) and torch.isfinite(margin).all()
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert frozen.feature_dim == 64
    assert not any(p.requires_grad for p in frozen.parameters())
    assert not any(m.training for m in frozen.modules())
