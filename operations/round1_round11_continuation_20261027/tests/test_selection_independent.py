"""Independent native selection/verification boundary review, CPU fixtures only."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from test_selection_cache_native import env, OLD, repair, native, scope, write


# Capture the actual native public verifier before the imported fixture replaces
# its expensive production evidence body with a software test callback.
REAL_VERIFY_DEVELOPMENT = native.verify_development_receipt


@pytest.fixture
def real_descriptor_evidence(env, monkeypatch):
    """Real descriptor SHA/type gates, native freeze and configuration verifier.

    The inner scientific production audit is deliberately a software callback;
    these fixtures are never claimed as EEG evidence or scientific completion.
    """
    refs=[]
    for index in range(len(env.refs)):
        job=native.DevelopmentJob('B1',index)
        value=dict(
            schema_version=native.SCHEMA,
            status='DEVELOPMENT_EVALUATION_COMPLETE',
            synthetic=False, scientific_fit_completed=True,
            measured=True, role='meta_validation',
            job=dict(method='B1',configuration_index=index,
                     seed=job.seed,phase='screening'),
            participant_mean_ba=.51+index*.01,
            participant_mean_bce=.7,
            configuration=job.configuration,
            catalog={'fixture':'same_software_catalog'},
            inputs=dict(
                device_identity_sha256='a'*64,
                dependencies={'source_f0':'same'},
                selected_meta_outer_update=None,
                code={'fixture':'SOFTWARE_ONLY'}))
        path=env.root/'runs/software_screen'/('B1-%d.json'%index)
        write(path,value)
        refs.append(dict(path=path.relative_to(env.root).as_posix(),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    env.refs=refs
    env.calls['production_verify']=0

    def production(root,descriptor,value):
        env.calls['production_verify']+=1
        for _ in range(4):
            env.training.verified_context(root)
        return copy.deepcopy(value)

    monkeypatch.setattr(native,'verify_development_receipt',REAL_VERIFY_DEVELOPMENT)
    monkeypatch.setattr(native,'_verify_development_production',production)
    return env


def test_native_configuration_roundtrip_rebuild_is_read_only_and_keeps_receipt_gates(real_descriptor_evidence):
    e=real_descriptor_evidence
    with scope(e) as snapshot:
        frozen=native.freeze_screening_configuration(
            e.root,method='B1',screening_receipts=e.refs,
            output=e.action['expected_receipt'],execute=True)
        original_bytes=e.expected.read_bytes()
        descriptor=dict(path=e.action['expected_receipt'],
                        sha256=hashlib.sha256(original_bytes).hexdigest())
        # This calls the native freeze function again with execute=False. It
        # must rerun every ordinary receipt verifier without another _put.
        rebuilt=native.verify_configuration_receipt(e.root,descriptor)
        assert rebuilt==frozen
        assert e.expected.read_bytes()==original_bytes
        assert e.calls['production_verify']==2*len(e.refs)
        assert e.calls['context']==1
        assert e.calls['native_put']==1
        assert snapshot.stats['context_hits']==8*len(e.refs)-1
        assert native.verify_development_receipt is REAL_VERIFY_DEVELOPMENT
    assert native._put is e.native_put
    assert e.training.verified_context is e.original_context


def test_cached_source_never_bypasses_changed_screening_receipt_descriptor(real_descriptor_evidence):
    e=real_descriptor_evidence
    original_production=native._verify_development_production

    def corrupt_next(root,descriptor,value):
        result=original_production(root,descriptor,value)
        if descriptor==e.refs[0]:
            # The source context is now verified/cached. Change the next
            # screening receipt without changing its descriptor identity.
            path=e.root/e.refs[1]['path']
            path.write_bytes(path.read_bytes()+b' ')
        return result

    native._verify_development_production=corrupt_next
    try:
        with pytest.raises(ValueError):
            with scope(e):
                native.freeze_screening_configuration(
                    e.root,method='B1',screening_receipts=e.refs,
                    output=e.action['expected_receipt'],execute=True)
    finally:
        native._verify_development_production=original_production
    assert e.calls['context']==1
    assert e.calls['production_verify']==1
    assert e.calls['native_put']==0
    assert not e.expected.exists()
    assert native._put is e.native_put


def test_first_context_validation_failure_is_not_reused(env):
    e=env
    original=e.training.verified_context
    attempts=[]

    def flake(root):
        attempts.append(1)
        if len(attempts)==1:
            raise ValueError('initial full source verification failed')
        return original(root)

    e.training.verified_context=flake
    with scope(e) as snapshot:
        with pytest.raises(ValueError,match='initial full'):
            e.training.verified_context(e.root)
        e.training.verified_context(e.root)
        e.training.verified_context(e.root)
        assert snapshot.stats['context_verifications']==1
        assert snapshot.stats['context_hits']==1
    assert len(attempts)==2
    assert e.training.verified_context is flake
    assert not e.expected.exists()


def test_cached_context_metadata_and_labels_are_private_to_each_caller(env):
    e=env
    with scope(e):
        first,contract=e.training.verified_context(e.root)
        first.rows.append({'caller':'modified'})
        first.locations.append((99,99))
        first.artifact_sha256['caller.json']='0'*64
        first.arrays.append(np.zeros(3))
        first.labels[0]=99
        contract['caller_modified']=True
        second,second_contract=e.training.verified_context(e.root)
        assert second.rows==[] and second.locations==[]
        assert 'caller.json' not in second.artifact_sha256
        assert len(second.arrays)==1
        np.testing.assert_array_equal(second.labels,[0,1])
        assert not second.labels.flags.writeable
        assert 'caller_modified' not in second_contract
    assert e.calls['context']==1


def test_guard_does_not_stop_unrelated_native_diagnostic_publication(env):
    e=env
    path=e.root/'runs/software_diagnostic.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with scope(e):
        e.training.verified_context(e.root)
        native._put(path,{'status':'CALIBRATION_METADATA','method':'B1'})
    assert json.loads(path.read_text())['status']=='CALIBRATION_METADATA'
    assert e.calls['native_put']==1
    assert not e.expected.exists()
