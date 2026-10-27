from __future__ import annotations
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from paper3m12 import algorithm as a


class NonlinearModel:
    dimension=4; device='cpu'; dtype=torch.float64
    cache_identity={'source':'synthetic-nonlinear-identity-v1'}
    def margin(self, theta, x, head_w=None, head_b=None):
        z=x@theta
        out=.08*x.sum(1)+z+.15*z.square()
        if head_w is not None:out=out+self.features(theta,x)@head_w
        if head_b is not None:out=out+head_b
        return out
    def reference_margin(self, theta, x, head_w=None, head_b=None):
        return self.margin(theta,x*1.7,head_w,head_b)
    def features(self,theta,x):
        base=torch.cat([x, x.square(), torch.sin(x), torch.cos(x)],1)
        return base.repeat(1,4)+.02*(x@theta).unsqueeze(1)
    def reference_features(self,theta,x):return self.features(theta,x*1.7)


def sample():
    generator=torch.Generator().manual_seed(17)
    x=torch.randn((16,4),dtype=torch.float64,generator=generator)*torch.tensor([1.,.7,.3,.05])
    y=(x[:,0]+.2*x[:,1]>0).long()
    ref=torch.randn((24,4),dtype=torch.float64,generator=generator)
    return x,y,ref


@pytest.fixture(scope='module')
def fitted():
    x,y,ref=sample();model=NonlinearModel()
    return model,x,y,ref,a.fit_episode(model,x,y,ref,budget=20,seed=20261018,tau=.025)


def test_support_jacobian_agrees_with_finite_differences():
    model=NonlinearModel();x,y,ref=sample()
    at=torch.tensor([.01,-.015,.02,.003],dtype=torch.float64)
    result=a.support_spectrum(model,x,y,at)
    for column in range(4):
        step=torch.zeros(4,dtype=torch.float64);step[column]=1e-6
        expected=(model.margin(at+step,x)-model.margin(at-step,x))/(2e-6)
        torch.testing.assert_close(result['jacobian'][:,column],expected,rtol=1e-7,atol=1e-9)
    assert result['positive_rank']==4 and result['q']==2
    torch.testing.assert_close(result['top'].T@result['weak'],torch.zeros(2,2,dtype=torch.float64),atol=1e-12,rtol=0)


def test_actual_nonlinear_reference_doses_match_and_never_amplify(fitted):
    model,x,y,ref,result=fitted
    states=result['states']
    primary=['top_q','scalar_shrink_full']+['random_q_'+str(i) for i in range(5)]
    target=result['diagnostics']['steps']['100']['primary_reference_rms_target']
    assert target>0
    for key in primary:
        state=states[key]
        assert 0<=state['scale']<=1
        assert state['reference_rms']==pytest.approx(target,abs=1e-5)
        assert a.reference_rms(model,state,ref)==pytest.approx(target,abs=1e-5)
        assert state['parameter_norm']<=.15+1e-12
    assert states['full_sgd']['layer']=='raw'
    assert states['weak_q']['layer']=='weak_reference_rms'
    assert states['top_q_weak_match']['reference_rms']==pytest.approx(states['weak_q']['reference_rms'],abs=1e-5)
    # Fixed C deliberately differs from target support mapping.
    theta=torch.tensor(states['top_q']['theta'],dtype=torch.float64)
    wrong=torch.sqrt((model.margin(theta,ref)-model.margin(torch.zeros(4,dtype=torch.float64),ref)).square().mean())
    assert abs(float(wrong)-target)>1e-4


def test_random_subspaces_are_in_positive_rowspace_and_preserve_rng():
    model=NonlinearModel();x,y,ref=sample();s=a.support_spectrum(model,x,y)
    before=torch.random.get_rng_state().clone()
    basis=a.random_subspace(s['positive_basis'],2,33)
    assert torch.equal(torch.random.get_rng_state(),before)
    torch.testing.assert_close(basis.T@basis,torch.eye(2,dtype=torch.float64),atol=1e-12,rtol=0)
    projection=s['positive_basis']@s['positive_basis'].T
    torch.testing.assert_close(projection@basis,basis,atol=1e-12,rtol=0)


def test_fit_does_not_consume_global_rng_or_accept_query_arguments():
    model=NonlinearModel();x,y,ref=sample()
    before=torch.random.get_rng_state().clone()
    a.fit_episode(model,x,y,ref,budget=20,seed=5,tau=.01)
    assert torch.equal(torch.random.get_rng_state(),before)
    assert not any('query' in name for name in inspect.signature(a.fit_episode).parameters)


def test_true_update_deletion_is_derived_from_actual_final_support_spectrum(fitted):
    model,x,y,ref,result=fitted;states=result['states']
    theta=torch.tensor(states['full_sgd']['theta'],dtype=torch.float64)
    final=a.support_spectrum(model,x,y,theta)
    expected=theta-final['weak']@(final['weak'].T@theta)
    torch.testing.assert_close(torch.tensor(states['drop_weak_raw']['theta'],dtype=torch.float64),expected)
    assert states['drop_weak']['fit_status']=='DERIVED_FROM_REAL_UPDATE'
    targets=[states[name]['reference_rms'] for name in ['drop_weak','drop_scalar_shrink_full']+
             ['drop_random_'+str(i) for i in range(5)]]
    assert max(targets)-min(targets)<=2e-5


def test_secondary_step10_snapshot_and_parameter_norm_controls_are_distinct(fitted):
    model,x,y,ref,result=fitted;states=result['states']
    direct=a.fit_full(model,x,y,steps=10)
    np.testing.assert_allclose(states['full_sgd_steps10']['theta'],direct['theta'],rtol=0,atol=1e-14)
    names=['top_q','full_sgd']+['random_q_'+str(i) for i in range(5)]
    norms=[states[name+'_parameter_norm']['parameter_norm'] for name in names]
    assert max(norms)-min(norms)<1e-12
    assert states['top_q_parameter_norm']['layer']=='parameter_norm'
    json.dumps(result,allow_nan=False)


def test_interrupted_raw_fit_resumes_exact_sgd_state_without_repeating_completed_steps(fitted):
    model,x,y,ref,expected=fitted;cache={};events=[]
    class StopAfterCheckpoint(Exception):pass
    def checkpoint(name,record):
        cache[name]=json.loads(json.dumps(record))
        if record['completed_steps']==10:raise StopAfterCheckpoint()
    with pytest.raises(StopAfterCheckpoint):
        a.fit_episode(model,x,y,ref,budget=20,seed=20261018,tau=.025,raw_cache=cache,checkpoint=checkpoint)
    actual=a.fit_episode(model,x,y,ref,budget=20,seed=20261018,tau=.025,raw_cache=cache,
                         progress=lambda **fields:events.append(fields))
    assert actual['states']==expected['states']
    full=[e for e in events if e.get('raw_condition')=='full_sgd']
    assert full and min(e['step'] for e in full)==20


def test_telemetry_errors_cannot_interrupt_science(fitted):
    model,x,y,ref,expected=fitted
    def broken(**fields):raise OSError('simulated monitor unavailable')
    actual=a.fit_episode(model,x,y,ref,budget=20,seed=20261018,tau=.025,progress=broken)
    assert actual['states']==expected['states']


@pytest.mark.parametrize('empty',[False,True])
def test_single_class_and_empty_support_return_explicit_common_anchor(empty):
    model=NonlinearModel();x,y,ref=sample()
    if empty:x,y=x[:0],y[:0]
    else:y=torch.zeros(len(y),dtype=torch.long)
    result=a.fit_episode(model,x,y,ref,budget=20,seed=1,tau=.01)
    assert result['diagnostics']['fallback_reason'] in ('empty_support','single_class_support')
    assert all(state['theta']==[0.]*4 for state in result['states'].values())
    assert all(state['fit_status']=='FALLBACK_SUPPORT' for state in result['states'].values())


def test_zero_derivative_rank_is_reported_without_inventing_information():
    class Constant(NonlinearModel):
        def margin(self,theta,x,head_w=None,head_b=None):return .05*x.sum(1)
    model=Constant();x,y,ref=sample()
    result=a.fit_episode(model,x,y,ref,budget=20,seed=1,tau=.02)
    assert result['diagnostics']['initial_spectrum']['positive_rank']==0
    assert result['states']['top_q']['theta']==[0.]*4


def test_rank_one_retains_raw_full_but_all_matched_direction_deletions_use_anchor():
    class RankOne(NonlinearModel):
        def margin(self,theta,x,head_w=None,head_b=None):return .08*x.sum(1)+theta[0]*x[:,0]
    model=RankOne();x,y,ref=sample()
    result=a.fit_episode(model,x,y,ref,budget=20,seed=1,tau=.02)
    assert result['diagnostics']['positive_rank']==1 and result['diagnostics']['q']==0
    assert result['states']['full_sgd']['parameter_norm']>0
    for state in result['states'].values():
        if state['layer'] in ('reference_rms','weak_reference_rms','removal_reference_rms'):
            assert state['theta']==[0.]*4
            assert state['fit_status']=='FALLBACK_ZERO_RANK'
            assert state['fallback_reason']


def test_numerical_failure_status_survives_all_derived_output_layers(monkeypatch):
    def fail(model,x,y,basis,steps,options,**kwargs):
        return dict(theta=[0.]*4,snapshots={str(k):[0.]*4 for k in
            (options['primary_steps'],options['secondary_steps'])},
            fit_status='FALLBACK_NONFINITE',fallback_reason='injected nonfinite gradient',completed_steps=0)
    monkeypatch.setattr(a,'_raw_fit',fail)
    model=NonlinearModel();x,y,ref=sample()
    result=a.fit_episode(model,x,y,ref,budget=20,seed=1,tau=.02)
    for state in result['states'].values():
        if state['method']=='support_ea':continue
        assert state['fit_status']=='FALLBACK_NONFINITE'
        assert state['fallback_reason']=='injected nonfinite gradient'


def test_tied_support_spectrum_is_flagged_and_retained():
    class Linear(NonlinearModel):
        def margin(self,theta,x,head_w=None,head_b=None):return x@theta
    model=Linear();x=torch.eye(4,dtype=torch.float64);y=torch.tensor([0,0,1,1])
    spectrum=a._serialize_spectrum(a.support_spectrum(model,x,y))
    assert spectrum['positive_rank']==4 and spectrum['q']==2
    assert spectrum['near_tied_subspace_boundaries']==[2]
    assert spectrum['direction_separation_insufficient'] is True
    assert spectrum['tied_episode_retained'] is True


def test_prediction_is_query_batch_independent_and_does_not_mutate_state(fitted):
    model,x,y,ref,result=fitted;state=json.loads(json.dumps(result['states']['top_q']))
    saved=json.loads(json.dumps(state));p=a.predict(model,state,ref,batch_size=5)
    np.testing.assert_array_equal(p,np.concatenate([a.predict(model,state,ref[:7]),a.predict(model,state,ref[7:])]))
    np.testing.assert_allclose(p.sum(1),1,atol=1e-15)
    assert state==saved
    assert a.predict(model,state,ref[:0]).shape==(0,2)


def test_target_stage_requires_previously_frozen_tau():
    model=NonlinearModel();x,y,ref=sample()
    with pytest.raises(ValueError,match='Frozen tau_B'):
        a.fit_episode(model,x,y,ref,budget=20,seed=1)
    observed=a.calibrate_tau(model,x,y,ref,budget=20,seed=1)
    assert observed['state']['steps']==10 and observed['tau_observation']>0
    assert observed['diagnostics']['support_only_scale'] is True


def test_invalid_inputs_are_not_reported_as_successful_scientific_fallback():
    model=NonlinearModel();x,y,ref=sample();x[0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        a.fit_episode(model,x,y,ref,budget=20,seed=1,tau=.01)


def test_known_shift_controls_fit_only_ea_full_top_and_keep_raw_dose(fitted):
    model,x,y,ref,base=fitted
    result=a.fit_shift_controls(model,x,y,ref,budget=20,seed=20261018,tau=.025)
    assert set(result['states'])=={'support_ea','full_sgd','top_q'}
    assert result['states']['top_q']['layer']=='shift_raw_control'
    np.testing.assert_array_equal(result['states']['full_sgd']['theta'],base['states']['full_sgd']['theta'])
    np.testing.assert_array_equal(result['states']['top_q']['theta'],base['states']['top_q_raw']['theta'])
    assert result['diagnostics']['matching_scope']=='unscaled_known_shift_positive_controls'


@pytest.fixture(scope='module')
def head_fitted():
    model=NonlinearModel();x,y,ref=sample()
    result=a.fit_head_controls(model,x,y,ref,budget=20,seed=20261018,tau=.025)
    return model,x,y,ref,result


def test_head_controls_share_actual_reference_dose_and_keep_head_frozen_mode_zero(head_fitted):
    model,x,y,ref,result=head_fitted
    states=result['states'];target=result['diagnostics']['head_family_reference_rms_target']
    assert target>0
    for mode in ('frozen','bias','full'):
        for name in ['top_q']+['random_q_'+str(i) for i in range(5)]:
            state=states[name+'_head_'+mode]
            assert state['reference_rms']==pytest.approx(target,abs=1e-5)
            assert a.reference_rms(model,state,ref)==pytest.approx(target,abs=1e-5)
            assert 0<=state['scale']<=1
            assert state['head_physical_penalty_denominator']==65
            if mode=='frozen':
                assert state['head_w']==[0.]*64 and state['head_b']==0
            if mode=='bias':assert state['head_w']==[0.]*64
    assert states['head_only_full']['theta']==[0.]*4
    assert states['head_only_rank4']['theta']==[0.]*4
    assert states['head_only_bias']['head_w']==[0.]*64
    assert abs(states['head_only_full']['head_b'])+sum(abs(z) for z in states['head_only_full']['head_w'])>0


def test_head_predictions_use_delta_head_and_are_query_batch_independent(head_fitted):
    model,x,y,ref,result=head_fitted;state=result['states']['head_only_full']
    predicted=a.predict(model,state,ref,batch_size=3)
    theta=torch.tensor(state['theta'],dtype=torch.float64)
    w=torch.tensor(state['head_w'],dtype=torch.float64);b=torch.tensor(state['head_b'],dtype=torch.float64)
    expected=torch.sigmoid(model.margin(theta,ref,w,b)).numpy()
    np.testing.assert_allclose(predicted[:,1],expected,rtol=0,atol=1e-15)
    np.testing.assert_allclose(predicted,np.concatenate([a.predict(model,state,ref[:4]),a.predict(model,state,ref[4:])]),rtol=0,atol=2e-15)
    source=a.predict(model,result['states']['support_ea'],ref)
    assert np.max(abs(predicted-source))>1e-3


def test_head_raw_fit_resume_does_not_retrain_completed_conditions(head_fitted):
    model,x,y,ref,expected=head_fitted;cache={};events=[]
    class Interruption(Exception):pass
    def checkpoint(name,record):
        cache[name]=json.loads(json.dumps(record))
        if record['completed_steps']==10:raise Interruption()
    with pytest.raises(Interruption):
        a.fit_head_controls(model,x,y,ref,budget=20,seed=20261018,tau=.025,
                            raw_cache=cache,checkpoint=checkpoint)
    actual=a.fit_head_controls(model,x,y,ref,budget=20,seed=20261018,tau=.025,
        raw_cache=cache,progress=lambda **fields:events.append(fields))
    assert actual['states']==expected['states']
    assert min(e['step'] for e in events if e.get('raw_condition')=='top_q_head_frozen')==20


def test_head_calibration_controls_do_not_consume_global_rng():
    model=NonlinearModel();x,y,ref=sample();before=torch.random.get_rng_state().clone()
    result=a.fit_head_controls(model,x,y,ref,budget=20,seed=2,tau=.02)
    assert torch.equal(before,torch.random.get_rng_state())
    json.dumps(result,allow_nan=False)


def test_actual_float32_EEGNet_with_float64_static_spatial_input_gradients():
    """Use the shipped native architecture/geometry, not the analytic fixture."""
    from paper3m12 import native_data
    data=native_data._data_module()
    source_module=data.native('paper3r7.data')
    geometry_module=data.native('p3work.spatial_geometry')
    old_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            source=source_module.FrozenSource(source_module.CompactEEGNet(11,128,.25))
            generator=torch.Generator().manual_seed(18)
            basis=torch.linalg.qr(torch.randn(99,4,dtype=torch.float64,generator=generator),mode='reduced').Q
            geometry=geometry_module.SpatialGeometry.from_reference(torch.eye(10,dtype=torch.float64)/10)
            x=torch.randn(8,11,128,dtype=torch.float64,generator=generator)
            ref=torch.randn(12,11,128,dtype=torch.float64,generator=generator)
            ea,_=native_data._fit_ea(geometry,x)
            model=native_data.NativeMarginModel(source,geometry,basis,alignment=ea,identity={'fixture':'real-native-EEGNet'})
            y=torch.tensor([0,1]*4)
            before=[p.detach().clone() for p in source.parameters()]
            config={'optimizer':{'primary_steps':3,'secondary_steps':2},'model':{'random_draws':1}}
            result=a.fit_episode(model,x,y,ref,budget=8,seed=4,tau=.001,config=config)
            assert result['diagnostics']['q']>0
            assert all(np.isfinite(a.predict(model,state,ref)).all() for state in result['states'].values())
            head=a.fit_head_controls(model,x,y,ref,budget=8,seed=4,tau=.001,config=config)
            assert head['diagnostics']['head_feature_dimension']==64
            assert a.predict(model,head['states']['head_only_full'],ref).shape==(12,2)
            assert all(torch.equal(old,new) for old,new in zip(before,source.parameters()))
            assert all(not parameter.requires_grad for parameter in source.parameters())
    finally:torch.set_num_threads(old_threads)


def test_fractional_labels_are_not_silently_cast_into_valid_classes():
    model=NonlinearModel();x,y,ref=sample();labels=y.double();labels[0]=.5
    with pytest.raises(ValueError,match='exact binary'):
        a.fit_episode(model,x,labels,ref,budget=20,seed=1,tau=.01)
    with pytest.raises(ValueError,match='exact binary'):
        a.fit_full(model,x,labels)
