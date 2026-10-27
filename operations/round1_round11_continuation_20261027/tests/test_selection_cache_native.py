"""CPU/software fixtures for future selection caching, not EEG evidence."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/workspace/paper3-r11-repo')
import selection_cache as repair
from p3work import development_evaluation as native


def load_old():
    path = Path('/workspace/paper3-r1-full-repair/verified_context_cache.py')
    spec = importlib.util.spec_from_file_location('_selection_test_old_cache', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD = load_old()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path/'project'; root.mkdir()
    write(root/'status/tasks.json', {'W1':{'status':'COMPLETE'}, 'W2':{'status':'COMPLETE'}})
    write(root/'contracts/input_contract.json', {'fixture':'SOFTWARE_ONLY'})
    write(root/'cache/manifest.json', {'source':'SOFTWARE_ONLY'})
    write(root/'cache/deep.json', {'source':'untouched'})
    np.save(root/'cache/X.npy', np.arange(32, dtype=np.float64))
    calls = dict(context=0, bank=0, verify=0, native_put=0)

    def context(root):
        calls['context'] += 1
        paths = ['cache/manifest.json', 'cache/deep.json', 'cache/X.npy']
        for name in paths:
            assert (Path(root)/name).is_file()
        cache = types.SimpleNamespace(root=root, manifest_path='cache/manifest.json',
            artifact_sha256={p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in paths},
            upstream_acceptance_sha256={}, arrays=[np.load(Path(root)/'cache/X.npy', mmap_mode='r')],
            rows=[], locations=[], labels=np.array([0,1]))
        return cache, {'fixture':'SOFTWARE_ONLY'}

    def bank(*_args, **_kwargs):
        calls['bank'] += 1
        raise AssertionError('This software selection fixture never needs a model fit')

    training = types.ModuleType('p3work._selection_test_training')
    training.verified_context = context
    training.load_verified_source_bank = bank
    monkeypatch.setitem(sys.modules, training.__name__, training)
    events = []
    def emit(event, **fields):
        events.append(dict(event=event, **fields))

    @contextmanager
    def startup(root, action, **kwargs):
        yield 'ORIGINAL_STARTUP_RESULT'
    startup_module = types.SimpleNamespace(active=startup)
    expected = 'runs/training_campaign_v1/development_selections/B1.json'
    action = dict(kind='development_selection', key='selection-B1', method='B1',
                  expected_receipt=expected, serial=True, affinity_slot=0)
    refs = ['fixture-ref-%d'%i for i in range(len(native.registry.registered_grid('B1')))]
    rows = {}
    for i, ref in enumerate(refs):
        job = native.DevelopmentJob('B1', i)
        rows[ref] = dict(job=dict(method=job.method, configuration_index=job.configuration_index,
                                 seed=job.seed, phase=job.phase),
                         participant_mean_ba=.5+i*.01, participant_mean_bce=.7,
                         configuration=job.configuration,
                         catalog={'fixture':'same_software_catalog'},
                         inputs=dict(device_identity_sha256='a'*64, dependencies={'source_f0':'same'},
                                     selected_meta_outer_update=None, code={'fixture':'SOFTWARE_ONLY'}))

    def verify(root, descriptor):
        calls['verify'] += 1
        for _ in range(6):
            training.verified_context(root)
        return copy.deepcopy(rows[descriptor])

    original_put = native._put
    def counting_put(path, value):
        calls['native_put'] += 1
        return original_put(path, value)
    monkeypatch.setattr(native, 'verify_development_receipt', verify)
    monkeypatch.setattr(native, '_put', counting_put)
    return types.SimpleNamespace(root=root, training=training, development=native,
        calls=calls, events=events, emit=emit, original_context=context,
        original_bank=bank, native_put=counting_put, original_verify=verify,
        startup=startup_module, startup_original=startup, action=action,
        refs=refs, expected=root/expected)


def scope(e):
    return repair.active(e.root,e.action,OLD,training=e.training,
                         development=e.development,emit=e.emit)


def freeze(e):
    return native.freeze_screening_configuration(e.root, method='B1',
        screening_receipts=e.refs, output=e.action['expected_receipt'], execute=True)


def test_actual_native_freeze_runs_every_receipt_verifier_and_native_writer(env):
    # The actual native selection function and immutable writer execute. The
    # receipt verifier is a software fixture, never claimed as production EEG.
    with scope(env) as state:
        result = freeze(env)
        assert env.calls['verify'] == len(env.refs)
        assert env.calls['context'] == 1
        assert env.calls['native_put'] == 1
        assert state.stats['context_hits'] == 6*len(env.refs)-1
        assert native.verify_development_receipt is env.original_verify
    assert json.loads(env.expected.read_text()) == result
    assert result['configuration_index'] == len(env.refs)-1
    assert env.training.verified_context is env.original_context
    assert native._put is env.native_put
    assert OLD.ENABLED_KINDS == frozenset(('calibration_profile','development','internal',
                                         'extension_development','confirmation'))
    guards = [x for x in env.events if x['event']=='R1_SELECTION_SOURCE_DEPENDENCIES_CHECKED_BEFORE_PUBLICATION']
    assert len(guards)==1


def test_native_selection_result_bytes_and_rng_equal_with_and_without_cache(env):
    import random
    import torch
    reference = env.root/'runs/unwrapped-selection.json'
    native.freeze_screening_configuration(env.root,method='B1',
        screening_receipts=env.refs,output=str(reference.relative_to(env.root)),execute=True)
    before_context = env.calls['context']
    assert before_context == 6*len(env.refs)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    with scope(env):
        freeze(env)
    assert env.calls['context']-before_context == 1
    assert env.expected.read_bytes() == reference.read_bytes()
    assert random.getstate() == python_rng
    assert all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),numpy_rng))
    assert torch.equal(torch_rng,torch.get_rng_state())


@pytest.mark.parametrize('target',['cache/deep.json','cache/X.npy','contracts/input_contract.json'])
def test_changed_source_blocks_actual_native_selection_publication(env, monkeypatch, target):
    original_verify = native.verify_development_receipt
    def changing(root, descriptor):
        result = original_verify(root,descriptor)
        if descriptor == env.refs[-1]:
            (env.root/target).write_bytes(b'changed after successful source verification')
        return result
    monkeypatch.setattr(native,'verify_development_receipt',changing)
    with pytest.raises(OLD.FrozenDependencyChanged):
        with scope(env):
            freeze(env)
    assert not env.expected.exists()
    assert env.calls['native_put']==0
    assert native._put is env.native_put
    assert env.training.verified_context is env.original_context


def test_changed_source_status_blocks_publication(env, monkeypatch):
    original_verify = native.verify_development_receipt
    def changing(root, descriptor):
        result = original_verify(root,descriptor)
        if descriptor==env.refs[-1]:
            write(env.root/'status/tasks.json',{'W1':{'status':'CHANGED'},'W2':{'status':'COMPLETE'}})
        return result
    monkeypatch.setattr(native,'verify_development_receipt',changing)
    with pytest.raises(OLD.FrozenDependencyChanged):
        with scope(env):
            freeze(env)
    assert env.calls['native_put']==0
    assert not env.expected.exists()


@pytest.mark.parametrize('kind',['meta_selection','meta_screening','development','calibration_profile'])
def test_other_action_kinds_are_not_patched(env,kind):
    action = dict(env.action,kind=kind)
    with repair.active(env.root,action,OLD,training=env.training,
                       development=env.development,emit=env.emit) as state:
        assert state is None
        assert env.training.verified_context is env.original_context
        assert native._put is env.native_put


def test_each_action_establishes_a_fresh_source_snapshot(env):
    for _ in range(2):
        with scope(env):
            freeze(env)
    assert env.calls['context']==2
    assert env.calls['verify']==2*len(env.refs)
    assert env.calls['native_put']==2


def test_original_verification_error_is_preserved_and_restores_everything(env,monkeypatch):
    def failure(*args,**kwargs):
        raise RuntimeError('original verifier failure')
    monkeypatch.setattr(native,'verify_development_receipt',failure)
    with pytest.raises(RuntimeError,match='original verifier failure'):
        with scope(env):
            freeze(env)
    assert not env.expected.exists()
    assert native._put is env.native_put
    assert env.training.verified_context is env.original_context


@pytest.mark.parametrize('changes',[{'method':'B2'}, {'status':'WRONG'}])
def test_output_identity_guard_rejects_wrong_method_or_status(env,changes):
    with pytest.raises(repair.SelectionPublicationGuardError):
        with scope(env):
            env.training.verified_context(env.root)
            native._put(env.expected,{'status':repair.SELECTION_STATUS,'method':'B1',**changes})
    assert not env.expected.exists()
    assert env.calls['native_put']==0


def test_output_guard_rejects_different_selection_target(env):
    wrong = env.root/'runs/other-selection.json'
    with pytest.raises(repair.SelectionPublicationGuardError):
        with scope(env):
            env.training.verified_context(env.root)
            native._put(wrong,{'status':repair.SELECTION_STATUS,'method':'B1'})
    assert not wrong.exists()


def test_install_idempotence_and_original_startup_context_preserved(env):
    one=repair.install(env.startup,OLD,env.root,training=env.training,
                       development=env.development,emit=env.emit)
    wrapper=env.startup.active
    two=repair.install(env.startup,OLD,env.root,training=env.training,
                       development=env.development,emit=env.emit)
    assert one['status']=='R1_SELECTION_SOURCE_CACHE_INSTALLED'
    assert two['status']=='ALREADY_INSTALLED' and env.startup.active is wrapper
    with env.startup.active(env.root,env.action,traceback_interval=0) as value:
        assert value=='ORIGINAL_STARTUP_RESULT'
        freeze(env)
    assert repair.restore(env.startup)
    assert env.startup.active is env.startup_original


def test_root_mismatch_rejected_before_original_startup(env,tmp_path):
    other=tmp_path/'other';other.mkdir()
    repair.install(env.startup,OLD,env.root,training=env.training,
                   development=env.development,emit=env.emit)
    with pytest.raises(repair.SelectionPublicationGuardError):
        with env.startup.active(other,env.action):
            raise AssertionError('must not enter scientific action')
    assert native._put is env.native_put


def test_reinstall_with_different_root_rejected(env,tmp_path):
    other=tmp_path/'other';other.mkdir()
    repair.install(env.startup,OLD,env.root,training=env.training,
                   development=env.development,emit=env.emit)
    with pytest.raises(repair.SelectionPublicationGuardError):
        repair.install(env.startup,OLD,other)


def test_old_cache_enabled_kinds_overlap_refused(env):
    changed=types.SimpleNamespace(ENABLED_KINDS={'development_selection'})
    with pytest.raises(repair.SelectionPublicationGuardError):
        repair.install(env.startup,changed,env.root)
    assert env.startup.active is env.startup_original


def test_later_writer_owner_not_overwritten(env):
    later=lambda *args,**kwargs: None
    with scope(env):
        env.training.verified_context(env.root)
        native._put=later
    assert native._put is later


def test_observability_failure_never_changes_native_result(env):
    def bad_emit(*args,**kwargs):
        raise OSError('output unavailable')
    with repair.active(env.root,env.action,OLD,training=env.training,
                       development=env.development,emit=bad_emit):
        result=freeze(env)
    assert result['status']==repair.SELECTION_STATUS
    assert env.expected.exists()
