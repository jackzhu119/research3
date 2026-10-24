"""Actual finite-window support fits on synthetic EEG, not effect evidence."""
from dataclasses import replace
import inspect

import pytest
import torch

from test_algorithm import fixture
from paper3r7 import algorithm as alg
from paper3r7 import temporal, head_capacity, risk


@pytest.mark.parametrize('mode',alg.R7_MODES)
def test_new_maps_have_exact_source_identity_including_float32_source(mode):
    source,g,s,b,q = fixture(double=False)
    cfg = alg.Config(variant='r7_refined',mode=mode,head_l2=.1)
    weights = alg._zero_r7(source,b)
    actual = alg.actual_r7(source,g,b,weights,s.x,cfg)[0]
    assert torch.equal(actual,source.source_logits(s.x).double())
    signal = alg.transform_signal(g,b,weights,s.x,cfg)
    assert torch.equal(signal,s.x.double())


@pytest.mark.parametrize('mode',['static','spatiotemporal','global_time'])
@pytest.mark.parametrize('objective',risk.KINDS)
def test_actual_objectives_are_finite_and_nonincreasing(mode,objective):
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',mode=mode,objective=objective,
                     head_l2=.1,tangent_steps=2,inner_steps=2)
    original = {k:v.clone() for k,v in source.state_dict().items()}
    answer = alg.fit(source,g,b,s,cfg)
    assert answer.r7_config is cfg and not answer.fallback_reason
    assert torch.isfinite(alg.predict(source,g,b,answer,q)).all()
    history = answer.diagnostics['actual_support_objective_history']
    assert all(y<=x for x,y in zip(history,history[1:]))
    assert answer.diagnostics['final_actual_support_objective'] <= answer.diagnostics['identity_actual_support_objective']
    assert answer.weights.theta.norm() <= cfg.radius+1e-12
    assert answer.diagnostics['head_prior_original_dimension'] == source.feature_dim+1
    if objective == 'relative_dro':
        assert answer.diagnostics['identity_actual_support_objective'] == 0.
        assert answer.diagnostics['support_certificate_holds']
        assert answer.diagnostics['regularized_relative_objective'] <= 0.
    if mode == 'global_time':
        assert torch.equal(answer.weights.theta[1:],answer.weights.theta[1:]*0)
        assert answer.diagnostics['spatial_parameter_count'] == 1
    assert all(torch.equal(v,source.state_dict()[k]) for k,v in original.items())


@pytest.mark.parametrize('policy,h',[('full',64),('budget',15),('fixed',4),('none',0)])
def test_head_subspace_and_original_dimension_prior_are_respected(policy,h):
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',mode='spatiotemporal',objective='relative_dro',
                     head_policy=policy,head_rank=4,head_l2=.1,tangent_steps=2,inner_steps=2)
    answer = alg.fit(source,g,b,s,cfg)
    u = answer.head_projector
    assert u.shape == (source.feature_dim,h)
    torch.testing.assert_close(answer.weights.delta_w,u@(u.T@answer.weights.delta_w),atol=1e-12,rtol=1e-11)
    assert answer.diagnostics['head_prior_reduced_initializer'] == pytest.approx(.1*(h+1)/65)
    assert answer.diagnostics['head_prior_original_dimension'] == 65
    assert answer.diagnostics['head_subspace_residual'] < 1e-12
    if policy == 'full':
        assert answer.diagnostics['data_only_spatial_info_rank'] == 0
        assert answer.diagnostics['head_absorbed_spatial_fraction'] == 1.


def test_short_paid_budget_can_be_unidentified_without_blocking_fit():
    source,g,s,b,q = fixture(n=4)
    cfg = alg.Config(variant='r7_refined',head_policy='budget',head_l2=.1,
                     tangent_steps=1,inner_steps=1)
    answer = alg.fit(source,g,b,s,cfg)
    assert answer.head_projector.shape == (64,0)
    assert answer.diagnostics['dimension_necessary_condition_only'] is False
    assert answer.diagnostics['necessary_observation_rank_condition'] is False
    assert answer.fallback_reason is None
    assert torch.isfinite(alg.predict(source,g,b,answer,q)).all()


def test_reduced_initializer_uses_the_same_original_head_norm_penalty():
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',mode='static',head_policy='fixed',head_rank=4,
                     head_l2=.1,tangent_steps=2,inner_steps=1)
    m0,c,f = temporal.support_tangent(source,g,s,mode='static')
    u,_ = head_capacity.projector(f,'fixed',4,head_rank=4)
    small_cfg = replace(cfg,variant='tangent_profiled',head_l2=.1*5/65)
    small,_ = alg.core._surrogate_fit(m0,c,f.double()@u,b,s.y,small_cfg)
    big = alg.core.FastWeights(small.theta,u@small.delta_w,small.delta_b)
    reduced_penalty = small_cfg.head_l2*(small.delta_w.square().sum()+small.delta_b.square())/5
    full_penalty = cfg.head_l2*(big.delta_w.square().sum()+big.delta_b.square())/65
    torch.testing.assert_close(reduced_penalty,full_penalty,atol=1e-16,rtol=1e-14)


def test_same_fitted_prediction_is_query_batch_independent_and_fit_has_no_query_argument():
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',head_policy='budget',head_l2=.1,
                     tangent_steps=1,inner_steps=1)
    answer = alg.fit(source,g,b,s,cfg)
    first = alg.predict(source,g,b,answer,q)
    separate = torch.cat([alg.predict(source,g,b,answer,x[None]) for x in q])
    torch.testing.assert_close(first,separate,atol=2e-12,rtol=2e-10)
    assert 'query' not in inspect.signature(alg.fit).parameters
    assert not answer.diagnostics['query_adaptation_performed']


def test_missing_support_class_falls_back_to_frozen_source():
    source,g,s,b,q = fixture()
    s = replace(s,y=s.y*0)
    cfg = alg.Config(variant='r7_refined')
    answer = alg.fit(source,g,b,s,cfg)
    assert answer.fallback_reason == 'insufficient_class_support'
    assert torch.equal(alg.predict(source,g,b,answer,q),source.source_logits(q))


def test_old_round_six_control_is_numerically_unchanged():
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='uq_joint',head_l2=.1,tangent_steps=2)
    expected = alg.geometry6.fit(source,g,b,s,cfg)
    actual = alg.fit(source,g,b,s,cfg)
    for key in ('theta','delta_w','delta_b'):
        assert torch.equal(getattr(expected.weights,key),getattr(actual.weights,key))
    assert torch.equal(alg.predict(source,g,b,actual,q),alg.geometry6.predict(source,g,b,expected,q))


@pytest.mark.parametrize('kwargs',[
    {'mode':'unknown'},{'objective':'unknown'},{'temperature':0.},
    {'head_rank':-1},{'head_policy':'unknown'},{'head_basis':'unknown'},
    {'time_operator':'unknown'},{'head_seed':True},
])
def test_unknown_scientific_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        alg.Config(variant='r7_refined',**kwargs)
