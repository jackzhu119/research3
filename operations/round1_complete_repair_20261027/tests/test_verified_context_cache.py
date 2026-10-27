"""CPU snapshot isolation tests; no school jobs or real EEG evidence."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
_local_native = Path('/workspace/paper3-r11-repo')
SOURCE = Path(os.environ.get('PAPER3_R1_NATIVE_ROOT', str(_local_native if _local_native.is_dir() else '/public/home/zhuziyuan/projects/paper3')))
sys.path.insert(0, str(SOURCE))
import verified_context_cache as repair
from p3work.methods import classical


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path/'project'; root.mkdir()
    write(root/'status/tasks.json', {'W1': {'status': 'COMPLETE'}, 'W2': {'status': 'COMPLETE'}})
    write(root/'contracts/input_contract.json', {'scope': 'CPU_SOFTWARE_TEST_ONLY'})
    write(root/'cache/manifest.json', {'source': 'fixture'})
    write(root/'cache/deep-event.json', {'event_uid': 'source/event/1'})
    np.save(root/'cache/X.npy', np.arange(128, dtype=np.float64))
    receipt = root/'bank/fit_receipt.json'; write(receipt, {'status': 'CPU_FIXTURE'})
    model = classical._model('B4', {'filters': np.eye(10)[:4], 'coef': np.arange(4)*.1,
                                  'bias': np.array(.2)}, {'scope': 'CPU_SOFTWARE_TEST_ONLY'})
    calls = {'context': 0, 'bank': 0, 'publish': 0}

    def verified_context(root):
        calls['context'] += 1
        data = np.load(Path(root)/'cache/X.npy', mmap_mode='r', allow_pickle=False)
        assert np.isfinite(data).all()
        paths = ['cache/manifest.json', 'cache/X.npy', 'cache/deep-event.json']
        capability = types.SimpleNamespace(root=root, manifest_path='cache/manifest.json',
            arrays=[data], labels=np.array([0, 1]), rows=[{'event_uid': 'source/event/1'}],
            locations=[(0, 0)], artifact_sha256={p: hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths},
            upstream_acceptance_sha256={})
        return capability, {'scope': 'CPU_SOFTWARE_TEST_ONLY'}

    training = types.ModuleType('p3work._r1_snapshot_test_training')
    training.verified_context = verified_context
    def load_verified_source_bank(root, receipt_path, *, method, expected_m=None,
                                  expected_shrink=None, input_sha, cache_sha,
                                  expected_receipt_sha256=None):
        calls['bank'] += 1
        training.verified_context(root)
        descriptor = receipt_path if isinstance(receipt_path, dict) else {
            'path': receipt_path, 'sha256': expected_receipt_sha256}
        assert hashlib.sha256((Path(root)/descriptor['path']).read_bytes()).hexdigest() == descriptor['sha256']
        if method != 'B4' or expected_m not in (1, 2, 3):
            raise ValueError('invalid configuration')
        return model
    training.load_verified_source_bank = load_verified_source_bank
    development = types.ModuleType('p3work._r1_snapshot_test_development')
    def publish(*args, **kwargs):
        calls['publish'] += 1
        write(root/'receipt.json', {'status': 'COMPLETE_SOFTWARE_FIXTURE'})
        return 'published'
    development._publish_completion = publish
    alias = types.ModuleType('p3work._r1_snapshot_test_alias')
    alias.load_verified_source_bank = load_verified_source_bank
    for module in (training, development, alias):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    descriptor = {'path': 'bank/fit_receipt.json', 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
    options = dict(method='B4', expected_m=2, input_sha='a'*64, cache_sha='b'*64)
    return types.SimpleNamespace(root=root, training=training, development=development,
        alias=alias, model=model, descriptor=descriptor, options=options, calls=calls,
        original_context=verified_context, original_bank=load_verified_source_bank, original_publish=publish)


def scope(env):
    return repair.active(env.root, {'kind': 'development', 'key': 'B4-fixture'},
        training=env.training, development=env.development, emit=lambda *_a, **_k: None)


def bank(env, **options):
    return env.training.load_verified_source_bank(env.root, env.descriptor, **{**env.options, **options})


def test_original_full_verifier_once_for_500_support_dependency_loads(env):
    with scope(env) as state:
        for _ in range(500):
            candidate = bank(env)
            assert candidate.state_sha256 == env.model.state_sha256
        assert env.calls == {'context': 1, 'bank': 1, 'publish': 0}
        assert state.stats['bank_hits'] == 499
        assert state.stats['context_verifications'] == 1
    assert env.training.load_verified_source_bank is env.original_bank
    assert env.alias.load_verified_source_bank is env.original_bank


def test_private_source_model_and_receipt_cannot_be_mutated_through_a_hit(env):
    with scope(env):
        original = bank(env)
        original.receipt['caller_extra'] = 'attempt-one'
        hit = bank(env)
        assert 'caller_extra' not in hit.receipt
        hit.receipt['caller_extra'] = 'attempt-two'
        with pytest.raises(ValueError):
            hit.arrays['coef'][0] = 999
        another = bank(env)
        assert 'caller_extra' not in another.receipt
        assert not np.shares_memory(another.arrays['coef'], hit.arrays['coef'])
        another.assert_unchanged()


def test_two_source_configurations_share_only_fully_verified_context(env):
    with scope(env) as state:
        bank(env, expected_m=1)
        bank(env, expected_m=2)
        bank(env, expected_m=1)
        assert env.calls['bank'] == 2 and env.calls['context'] == 1
        assert state.stats['context_hits'] == 1 and state.stats['bank_hits'] == 1


def test_each_action_has_a_fresh_validation_not_a_persistent_capability(env):
    with scope(env):
        bank(env); bank(env)
    with scope(env):
        bank(env); bank(env)
    assert env.calls['bank'] == 2 and env.calls['context'] == 2


@pytest.mark.parametrize('anchor', ['status/tasks.json', 'contracts/input_contract.json', 'cache/manifest.json',
                                  'bank/fit_receipt.json'])
def test_identity_anchor_change_rejects_next_hit_without_silent_reload(env, anchor):
    with pytest.raises(repair.FrozenDependencyChanged):
        with scope(env):
            bank(env)
            (env.root/anchor).write_text('changed dependency identity')
            bank(env)
    assert env.calls['bank'] == 1
    assert env.training.load_verified_source_bank is env.original_bank


def test_deep_dependency_change_blocks_original_completion_before_receipt(env):
    with pytest.raises(repair.FrozenDependencyChanged):
        with scope(env):
            bank(env)
            (env.root/'cache/deep-event.json').write_text('changed during action')
            env.development._publish_completion()
    assert env.calls['publish'] == 0
    assert not (env.root/'receipt.json').exists()


def test_valid_completion_calls_original_function_and_source_bytes_unchanged(env):
    paths = [p for p in env.root.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in paths}
    with scope(env):
        bank(env)
        assert env.development._publish_completion() == 'published'
    assert env.calls['publish'] == 1
    assert before == {p: p.read_bytes() for p in paths}
    assert env.development._publish_completion is env.original_publish


def test_unrelated_stage_status_update_does_not_change_frozen_source_identity(env):
    with scope(env):
        bank(env)
        path = env.root/'status/tasks.json'
        current = json.loads(path.read_text()); current['W99'] = {'status': 'RUNNING'}
        write(path, current)
        bank(env)
        env.development._publish_completion()
    assert env.calls['bank'] == 1
    assert env.calls['publish'] == 1


def test_failed_original_verification_is_not_memoized(env):
    with scope(env):
        for _ in range(2):
            with pytest.raises(ValueError, match='invalid configuration'):
                bank(env, expected_m=999)
    assert env.calls['bank'] == 2


def test_existing_action_failure_not_masked_by_exit_dependency_guard(env):
    with pytest.raises(RuntimeError, match='actual solver failed'):
        with scope(env):
            bank(env)
            (env.root/'cache/X.npy').unlink()
            raise RuntimeError('actual solver failed')
    assert env.training.verified_context is env.original_context


def test_non_development_action_is_not_patched(env):
    with repair.active(env.root, {'kind': 'meta_screening'}, training=env.training,
                       development=env.development) as state:
        assert state is None
        assert env.training.load_verified_source_bank is env.original_bank


def test_cached_model_preserves_native_classical_predictions_and_rng(env):
    import random
    import torch
    from p3work.adapt import PredictionInputs
    from scipy.linalg import helmert
    model = classical._model('B4', {'filters': np.eye(10)[:4], 'coef': np.arange(4)*.1,
                                  'bias': np.array(.2)},
                            {'signal_contract_sha256': 'a'*64, 'n_times': 64})
    # Native predictor exercises CAR coordinates, CSP log variance, logits and
    # class predictions. This is a software fixture, never an EEG result.
    rng = np.random.default_rng(42); z = rng.normal(size=(12, 10, 64)); z -= z.mean(-1, keepdims=True)
    signals = np.einsum('cd,ndt->nct', helmert(11, full=False).T, z)
    query = PredictionInputs(torch.tensor(signals), tuple('query/%d'%i for i in range(12)), 'a'*64)
    expected = classical.predict_classical(model, query)
    python_rng = random.getstate(); numpy_rng = np.random.get_state(); torch_rng = torch.get_rng_state().clone()
    for _ in range(20):
        duplicate = repair._clone_bank(model)
        actual = classical.predict_classical(duplicate, query)
        for key in ('probability', 'logits', 'prediction'):
            np.testing.assert_array_equal(actual[key], expected[key])
        assert actual['model_state_sha256'] == expected['model_state_sha256']
    assert random.getstate() == python_rng
    assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_rng))
    assert torch.equal(torch.get_rng_state(), torch_rng)


@pytest.mark.parametrize('changed', ['core-artifact', 'source-code'])
def test_registered_bank_tree_dependencies_checked_before_final_publish(env, changed):
    output = env.root/'runs/source_classical_v2'
    selected = {'job_id': 'bank/ALL_SOURCE/B4/m-2', 'group': 'bank'}
    core = {'job_id': 'lodo/SOURCE/core-csp', 'group': 'core'}
    code = env.root/'p3work/classical_training.py'
    code.parent.mkdir(parents=True); code.write_text('# actual validated code fixture\n')
    write(output/'identity.json', {'budget': {'path': 'runs/source_classical_v2/budget.json'}})
    write(output/'budget.json', {'max_fit_attempts': 16})
    write(output/'plan.json', {'jobs': [core, selected],
        'code_policy': {'code': {'p3work/classical_training.py': hashlib.sha256(code.read_bytes()).hexdigest()}}})
    for job in (core, selected):
        folder = output/job['job_id']; folder.mkdir(parents=True, exist_ok=True)
        np.save(folder/'weights.npy', np.arange(8, dtype=np.float64))
        manifest = folder/'artifact_manifest.json'
        write(manifest, {'files': [{'path': (folder/'weights.npy').relative_to(env.root).as_posix()}]})
        write(folder/'status.json', {'artifact_manifest': {'path': manifest.relative_to(env.root).as_posix()}})
    selected_receipt = output/selected['job_id']/'fit_receipt.json'
    write(selected_receipt, {'job': selected})
    env.descriptor = {'path': selected_receipt.relative_to(env.root).as_posix(),
        'sha256': hashlib.sha256(selected_receipt.read_bytes()).hexdigest()}
    with pytest.raises(repair.FrozenDependencyChanged):
        with scope(env):
            bank(env)
            target = output/core['job_id']/'weights.npy' if changed == 'core-artifact' else code
            target.write_bytes(b'changed after actual successful source verification')
            bank(env)  # Same immutable capability; no full tree walk per support fit.
            env.development._publish_completion()
    assert env.calls['publish'] == 0
    assert not (env.root/'receipt.json').exists()


@pytest.mark.parametrize('method', ['B4', 'B5'])
def test_native_support_fit_parameters_and_future_predictions_equal_for_cloned_bank(method):
    import torch
    from p3work.adapt import CalibrationPrefix, PredictionInputs
    from p3work.contracts import SOURCE_IDS
    from scipy.linalg import helmert
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(1267)
    z = rng.normal(size=(32, 10, 64)); z -= z.mean(-1, keepdims=True)
    signals = np.einsum('cd,ndt->nct', helmert(11, full=False).T, z)
    receipt = dict(kind='source_model', status='SOURCE_CLASSICAL_COMPLETE', held_out_dataset=None,
        source_datasets=sorted(SOURCE_IDS), signal_contract_sha256='a'*64, n_times=64, synthetic=True)
    if method == 'B4':
        means = np.stack((np.diag(np.arange(1,11)/55), np.diag(np.arange(10,0,-1)/55)))
        filters, _ = classical.csp_filters(means, 2)
        source = classical._model(method, dict(class_means=means, filters=filters,
            coef=np.zeros(4), bias=np.array(0.)), dict(receipt, m=2))
        function = lambda bank, support: classical.fit_b4_target(bank, support, alpha=.5, m=2)
    else:
        source = classical._model(method, dict(inverse_root=np.eye(10), scaler_mean=np.zeros(55),
            scaler_scale=np.ones(55), coef=np.zeros(55), bias=np.array(0.)), dict(receipt, shrink=.1))
        function = lambda bank, support: classical.fit_b5_target(bank, support, l2=.1)
    acquired = tuple('support/%d'%i for i in range(20))
    support = CalibrationPrefix(torch.tensor(signals[:20]), torch.tensor([i%2 for i in range(20)]),
        acquired, acquired, 'a'*64)
    query = PredictionInputs(torch.tensor(signals[20:]), tuple('query/%d'%i for i in range(12)), 'a'*64)
    with threadpool_limits(limits=1):
        expected = function(source, support)
        actual = function(repair._clone_bank(source), support)
    for key in expected.arrays:
        np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])
    before = classical.predict_classical(expected, query)
    after = classical.predict_classical(actual, query)
    for key in ('logits', 'probability', 'prediction'):
        np.testing.assert_array_equal(before[key], after[key])
    assert {k:v for k,v in actual.receipt.items() if k != 'wall_seconds'} == {
        k:v for k,v in expected.receipt.items() if k != 'wall_seconds'}
