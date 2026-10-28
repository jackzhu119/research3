"""Finalization software fixtures only; no school data or GPU execution."""
from contextlib import contextmanager
import ast
import builtins
import io
import json
import os
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalization_cache as cache


class SourceSnapshot:
    instances = []

    def __init__(self, root, action, *, emit=None):
        self.root, self.action = root, action
        self.checks = 0
        self.closed = None
        type(self).instances.append(self)

    def install(self, training, development):
        return self

    def checkpoint(self):
        self.checks += 1

    def close(self, *, successful):
        self.closed = successful


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'runs').mkdir()
    (root / 'nested').mkdir()
    (root / 'nested' / 'prediction.npy').write_bytes(b'actual prediction bytes')
    (root / 'receipt.json').write_text('{"score":0.625,"synthetic":false}')
    (root / 'selection.json').write_text('{"method":"B4"}')
    calls = dict(receipt=0, selection=0, writer=0)
    development = types.ModuleType('p3work._finalization_cache_fixture')
    alias = types.ModuleType('p3work._finalization_cache_alias')

    def verify_receipt(root, descriptor):
        calls['receipt'] += 1
        # Exercise Path.read_text/io.open and builtins.open, as native JSON,
        # NumPy and torch loaders do. No synthetic scientific completion seal
        # is ever accepted by the production implementation in these tests.
        value = json.loads((Path(root) / descriptor['path']).read_text())
        with open(Path(root) / 'nested' / 'prediction.npy', 'rb') as handle:
            value['payload'] = handle.read().decode()
        if descriptor.get('invalid'):
            raise ValueError('original verification failed')
        return value

    def verify_selection(root, descriptor):
        calls['selection'] += 1
        selected = json.loads((Path(root) / descriptor['path']).read_text())
        selected['verified'] = development.verify_development_receipt(root, {'path': 'receipt.json'})
        return selected

    def put(path, value):
        calls['writer'] += 1
        Path(path).write_text(json.dumps(value))
        return 'original immutable writer called'

    development.verify_development_receipt = verify_receipt
    development.verify_configuration_receipt = verify_selection
    development._put = put
    alias.verify_development_receipt = verify_receipt
    monkeypatch.setitem(sys.modules, development.__name__, development)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    action = dict(kind='strongest_control', key='strongest-control',
                  expected_receipt='runs/strongest_control.json')
    SourceSnapshot.instances = []
    return types.SimpleNamespace(root=root, development=development, alias=alias,
                                 training=types.SimpleNamespace(), action=action,
                                 old=types.SimpleNamespace(ActionSnapshot=SourceSnapshot), calls=calls,
                                 originals=(verify_receipt, verify_selection, put))


def active(env, **options):
    return cache.active(env.root, env.action, env.old, training=env.training,
                        development=env.development, emit=lambda *_a, **_kw: None, **options)


def test_native_validation_once_private_results_and_alias_restore(env):
    original_open, original_io = builtins.open, io.open
    with active(env) as scope:
        first = env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
        first['method'] = 'caller mutation'
        for _ in range(20):
            result = env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
            assert result['method'] == 'B4'
            result['verified']['score'] = 999
        result = env.alias.verify_development_receipt(env.root, {'path': 'receipt.json'})
        assert result['score'] == .625
        assert env.calls == dict(receipt=1, selection=1, writer=0)
        assert scope.stats['reused_verifications'] == 21
    assert builtins.open is original_open and io.open is original_io
    assert env.alias.verify_development_receipt is env.originals[0]
    assert env.development.verify_configuration_receipt is env.originals[1]
    assert SourceSnapshot.instances[-1].closed is True


def test_nested_cached_dependencies_are_captured_by_outer_verification(env):
    with pytest.raises(cache.FinalizationDependencyChanged):
        with active(env):
            env.development.verify_development_receipt(env.root, {'path': 'receipt.json'})
            env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
            (env.root / 'nested/prediction.npy').write_bytes(b'changed prediction bytes')
            env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
    assert env.calls['receipt'] == 1 and env.calls['selection'] == 1


@pytest.mark.parametrize('relative', ['receipt.json', 'selection.json', 'nested/prediction.npy'])
def test_deep_change_blocks_original_writer_before_publication(env, relative):
    with pytest.raises(cache.FinalizationDependencyChanged):
        with active(env):
            env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
            (env.root / relative).write_bytes(b'changed dependency')
            env.development._put(env.root / env.action['expected_receipt'],
                                  {'status': cache.STATUS['strongest_control']})
    assert env.calls['writer'] == 0
    assert not (env.root / env.action['expected_receipt']).exists()


def test_native_output_and_writer_preserved_exactly(env):
    with active(env) as scope:
        env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
        value = {'status': cache.STATUS['strongest_control'], 'selected': 'B4', 'scores': [.5, .625]}
        assert env.development._put(env.root / env.action['expected_receipt'], value) == 'original immutable writer called'
        assert scope.stats['publications_guarded'] == 1
    assert json.loads((env.root / env.action['expected_receipt']).read_text()) == value
    assert env.calls['writer'] == 1


def test_invalid_native_result_is_never_cached(env):
    with active(env):
        for _ in range(2):
            with pytest.raises(ValueError, match='original verification failed'):
                env.development.verify_development_receipt(env.root, {'path': 'receipt.json', 'invalid': True})
    assert env.calls['receipt'] == 2


def test_different_exact_arguments_run_native_validation(env):
    with active(env):
        env.development.verify_development_receipt(env.root, {'path': 'receipt.json'})
        env.development.verify_development_receipt(env.root, {'path': 'receipt.json', 'extra_identity': 'different'})
    assert env.calls['receipt'] == 2


def test_fresh_action_has_no_cross_action_cache(env):
    for _ in range(2):
        with active(env):
            env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
            env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
    assert env.calls['receipt'] == env.calls['selection'] == 2


def test_same_size_same_mtime_replacement_is_detected_by_inode_or_ctime(env):
    with pytest.raises(cache.FinalizationDependencyChanged):
        with active(env):
            env.development.verify_development_receipt(env.root, {'path': 'receipt.json'})
            path = env.root / 'nested/prediction.npy'
            old = path.stat()
            replacement = path.with_suffix('.new')
            replacement.write_bytes(b'x' * old.st_size)
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
            replacement.replace(path)
            env.development.verify_development_receipt(env.root, {'path': 'receipt.json'})


def test_original_exception_not_masked_by_exit_guard(env):
    with pytest.raises(RuntimeError, match='native numeric failure'):
        with active(env):
            env.development.verify_development_receipt(env.root, {'path': 'receipt.json'})
            (env.root / 'nested/prediction.npy').unlink()
            raise RuntimeError('native numeric failure')
    assert SourceSnapshot.instances[-1].closed is False
    assert env.development.verify_development_receipt is env.originals[0]


def test_unsupported_fitting_action_is_unchanged(env):
    env.action['kind'] = 'internal'
    with active(env) as scope:
        assert scope is None
        assert env.development.verify_development_receipt is env.originals[0]
    assert SourceSnapshot.instances == []


def test_freeze_has_its_own_native_publication_identity(env):
    env.action = dict(kind='freeze', key='freeze', expected_receipt='runs/pre_internal_freeze.json')
    with active(env) as scope:
        env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
        env.development._put(env.root / env.action['expected_receipt'], {'status': cache.STATUS['freeze']})
        assert scope.stats['publications_guarded'] == 1


def test_wrong_result_path_or_status_refuses_publication(env):
    with pytest.raises(cache.FinalizationDependencyChanged, match='PUBLICATION_IDENTITY'):
        with active(env):
            env.development._put(env.root / 'runs/wrong.json', {'status': cache.STATUS['strongest_control']})
    assert env.calls['writer'] == 0


def test_install_is_scope_local_idempotent_and_reversible(env):
    @contextmanager
    def previous(root, action, **kwargs):
        yield 'native scope'
    startup = types.SimpleNamespace(active=previous)
    result = cache.install(startup, env.old, env.root, training=env.training,
                           development=env.development, emit=lambda *_a, **_kw: None)
    assert result['status'] == 'R1_FINALIZATION_REUSE_INSTALLED'
    assert cache.install(startup, env.old, env.root)['status'] == 'ALREADY_INSTALLED'
    with startup.active(env.root, env.action) as value:
        assert value == 'native scope'
        env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
        env.development.verify_configuration_receipt(env.root, {'path': 'selection.json'})
    assert env.calls['selection'] == env.calls['receipt'] == 1
    assert cache.restore(startup) is True and startup.active is previous


def test_root_escape_rejected_without_touching_outside(env, tmp_path):
    env.action['expected_receipt'] = str(tmp_path / 'outside.json')
    with pytest.raises(cache.FinalizationDependencyChanged, match='OUTSIDE_RUNS'):
        with active(env):
            pass
    assert not (tmp_path / 'outside.json').exists()


def test_actual_native_strongest_control_selection_and_output_are_identical(env, monkeypatch):
    """Execute the unchanged native aggregate/selector, replacing only I/O.

    Generated rows are software fixtures, never school experimental evidence.
    The original selector's eleven methods, three seeds, timing tie rule and
    native strongest-control output are compared byte-for-byte as JSON.
    """
    import hashlib
    import importlib.util

    native = Path('/workspace/paper3-r11-repo/p3work')
    selection_spec = importlib.util.spec_from_file_location('_r1_native_selection_fixture', native / 'selection.py')
    selection = importlib.util.module_from_spec(selection_spec)
    selection_spec.loader.exec_module(selection)
    original_source = (native / 'development_evaluation.py').read_text()
    function = next(node for node in ast.parse(original_source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'select_strongest_control')
    methods = tuple('B%d' % index for index in range(1, 12))
    seeds = (20261018, 20261019, 20261020)
    configurations, confirmations, rows, frozen = {}, [], {}, {}
    for index, method in enumerate(methods):
        config_ref = {'path': 'configs/%s.json' % method}
        configurations[method] = config_ref
        selected_ref = {'path': 'confirmed/%s-%s.json' % (method, seeds[0])}
        frozen[method] = {'method': method, 'configuration_index': index,
                          'configuration': {'setting': index},
                          'selected_screening_receipt': selected_ref}
        for seed in seeds:
            ref = {'path': 'confirmed/%s-%s.json' % (method, seed)}
            confirmations.append(ref)
            rows[ref['path']] = {
                'method': method, 'seed': seed,
                'configuration': {'setting': index},
                'job': {'configuration_index': index},
                'inputs': {'configuration_receipt': config_ref,
                           'device_identity_sha256': 'a' * 64,
                           'dependencies': {'source_f0': {'seed': seed}}},
                'catalog': {'identity': 'same registered query catalog'},
                'timing_protocol': 'same registered measured timing protocol',
                'participant_mean_ba': .8 - index * .001,
                'calibration_median_seconds': 20 - index}
    calls = dict(configuration=0, development=0)

    def verify_configuration(root, descriptor):
        calls['configuration'] += 1
        return json.loads(json.dumps(frozen[Path(descriptor['path']).stem]))

    def verify_development(root, descriptor):
        calls['development'] += 1
        row = json.loads(json.dumps(rows[descriptor['path']]))
        # Real production confirmation verification recursively re-verifies
        # its frozen selection; emulate that call graph here.
        env.development.verify_configuration_receipt(root, row['inputs']['configuration_receipt'])
        return row

    monkeypatch.setattr(env.development, 'verify_configuration_receipt', verify_configuration)
    monkeypatch.setattr(env.development, 'verify_development_receipt', verify_development)
    globals_map = vars(env.development)
    globals_map.update(Path=Path, CONTROLS=methods, SEEDS=seeds, SCHEMA='native fixture schema',
                       DevelopmentGateError=ValueError,
                       choose_strongest_control=selection.choose_strongest_control,
                       digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<unchanged native select_strongest_control>', 'exec'), globals_map)
    arguments = dict(configuration_receipts=configurations, confirmation_receipts=confirmations)
    original = env.development.select_strongest_control(env.root, **arguments)
    original_calls = dict(calls)
    calls.update(configuration=0, development=0)
    with active(env):
        reused = env.development.select_strongest_control(env.root, **arguments)
        again = env.development.select_strongest_control(env.root, **arguments)
    assert reused == again == original
    assert json.dumps(reused, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert original_calls == {'configuration': 44, 'development': 33}
    assert calls == {'configuration': 11, 'development': 33}
