"""Synthetic EDF structure/math tests only: no scientific fit or extra EEG access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from p3work.source_extensions.ma2022 import (
    Ma2022Error, audit_pair, create_development_cache, descriptor, prepare_context,
)

NAMES = ['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6',
         'Cz','C3','C4','T3','T4','A1','A2','CP1','CP2','CP5','CP6',
         'Pz','P3','P4','T5','T6','PO3','PO4','Oz','O1','O2']


def fixed(value, n):
    return str(value).ljust(n).encode('ascii')


def fixture(tmp_path, *, suffix=0, unit='uV', event_clock='sample'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    common = (fixed('0',8)+fixed('fixture',80)+fixed('fixture',80)+fixed('01.01.01',8)
              +fixed('00.00.00',8)+fixed(8448,8)+fixed('',44)+fixed(8,8)+fixed(1,8)+fixed(32,4))
    fields = [(NAMES,16),(['']*32,80),([unit]*32,8),([-100]*32,8),([100]*32,8),
              ([-32768]*32,8),([32767]*32,8),(['unknown']*32,80),([250]*32,8),(['']*32,32)]
    header = common+b''.join(fixed(v,width) for values,width in fields for v in values)
    t=np.arange(2000)/250
    signal=np.stack([15000*np.sin(2*np.pi*(9+i/10)*t+i) for i in range(32)]).astype('<i2')
    if suffix:
        signal[:,375:1000]=suffix
        signal[:,1375:2000]=suffix
    native=signal.reshape(32,8,250).transpose(1,0,2).copy()
    edf=tmp_path/'sample.edf';edf.write_bytes(header+native.tobytes())
    events=tmp_path/'sample.tsv'
    if event_clock=='sample':
        events.write_text('onset\tduration\ttrial_type\tresponse_time\tsample\tvalue\n1\t1000\tleft\tn/a\t1\t1\n1001\t1000\tright\tn/a\t1001\t2\n')
    else:
        events.write_text('onset\tduration\ttrial_type\tresponse_time\tsample\tvalue\n0\t4000\tleft\tn/a\t0\t1\n4000\t4000\tright\tn/a\t1000\t2\n')
    return edf,events


def args(edf,events):
    return dict(edf_sha256=descriptor(edf)['sha256'],events_sha256=descriptor(events)['sha256'])


def test_native_units_sample_clock_and_non_B20_scope(tmp_path):
    edf,events=fixture(tmp_path)
    result=audit_pair(edf,events,**args(edf,events))
    assert result['header']['clinic11_present']==['C3','Cz','C4']
    assert len(result['header']['clinic11_missing'])==8
    assert result['header']['n_samples']==2000
    assert result['event_clock_generic_sidecar_conflict'] is True
    assert result['raw_acquisition_prefix_eligible'] is False
    assert result['actual_fits']==0 and result['training_ready'] is False
    assert all(e['original_acquisition_ordinal'] is None for e in result['events'])
    assert [e['analysis_anchor_sample'] for e in result['events']]==[0,1000]


def test_wrong_generic_ms_clock_cannot_silently_reanchor(tmp_path):
    edf,events=fixture(tmp_path,event_clock='ms')
    with pytest.raises(Ma2022Error,match='1based_sample_clock'):
        audit_pair(edf,events,**args(edf,events))


def test_native_calibration_units_and_hash_are_mandatory(tmp_path):
    edf,events=fixture(tmp_path,unit='mV')
    with pytest.raises(Ma2022Error,match='microvolt'):
        audit_pair(edf,events,**args(edf,events))
    edf,events=fixture(tmp_path,unit='uV')
    bad=args(edf,events);bad['edf_sha256']='f'*64
    with pytest.raises(Ma2022Error,match='SHA256_mismatch'):
        audit_pair(edf,events,**bad)
    with pytest.raises(Ma2022Error,match='monopolar'):
        audit_pair(edf,events,reference_evidence='bipolar_guess',**args(edf,events))


def test_development_cache_never_filters_beyond_1p5_context(tmp_path):
    one=fixture(tmp_path/'one')
    two=fixture(tmp_path/'two',suffix=30000)
    a=create_development_cache(*one,tmp_path/'out1',**args(*one))
    b=create_development_cache(*two,tmp_path/'out2',**args(*two))
    x=np.load(tmp_path/'out1/X.npy',allow_pickle=False)
    np.testing.assert_array_equal(x,np.load(tmp_path/'out2/X.npy',allow_pickle=False))
    assert x.shape==(2,3,128) and x.dtype==np.float32
    np.testing.assert_allclose(x.mean(axis=1),0,atol=1e-11)
    np.testing.assert_allclose(x.mean(axis=2),0,atol=1e-11)
    np.testing.assert_array_equal(np.load(tmp_path/'out1/y.npy'),[0,1])
    assert a['training_ready'] is False and a['actual_fits']==0
    assert a['source_acquisition_order_recovered'] is False
    assert a['independent_external_people_added']==0
    assert a['normalization_applied'] is False
    with pytest.raises(Ma2022Error,match='no_overwrite'):
        create_development_cache(*one,tmp_path/'out1',**args(*one))


def test_zero_energy_and_bad_context_fail():
    with pytest.raises(Ma2022Error,match='zero_energy'):
        prepare_context(np.zeros((3,375)))
    with pytest.raises(Ma2022Error,match='3_by_375'):
        prepare_context(np.ones((3,1000)))
