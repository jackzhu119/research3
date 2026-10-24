"""Support-only cross-fitting/robust calibration checks on synthetic signals."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest
import torch

from test_algorithm import fixture
from paper3r6 import algorithm as alg


def test_two_member_simplex_has_exact_closed_form_shrink_coefficient():
    torch.manual_seed(3)
    y = torch.arange(20) % 2
    p0 = torch.rand(20, dtype=torch.float64) * .5 + .2
    pa = (y.double() * .7 + .15) + torch.randn(20, dtype=torch.float64) * .02
    alpha = alg.shrink_weight(pa, p0, y, .07)
    mix = alg.simplex_brier_weights(torch.stack((p0, pa), 1), y, .07)
    torch.testing.assert_close(mix, torch.stack((1-alpha, alpha)), atol=1e-12, rtol=1e-12)
    assert 0 < alpha < 1


def test_simplex_three_member_solution_satisfies_kkt_and_probability_constraints():
    torch.manual_seed(18)
    y = torch.arange(16) % 2
    p = torch.rand(16, 3, dtype=torch.float64)
    ridge = .05
    weights = alg.simplex_brier_weights(p, y, ridge)
    assert bool((weights >= 0).all())
    torch.testing.assert_close(weights.sum(), weights.new_tensor(1.))
    prior = weights.new_tensor([1., 0., 0.])
    gradient = 2 * p.T @ (alg._class_weights(y) * (p @ weights - y.double())) + ridge * (weights-prior)
    active = weights > 1e-9
    torch.testing.assert_close(gradient[active], gradient[active].mean().expand_as(gradient[active]), atol=1e-10, rtol=1e-9)
    assert bool((gradient[~active] >= gradient[active].mean()-1e-10).all())
    mixed = p @ weights
    assert bool((mixed >= p.min(1).values-1e-12).all())
    assert bool((mixed <= p.max(1).values+1e-12).all())


def test_simplex_selected_face_first_derivative_matches_numerical_difference():
    torch.manual_seed(8)
    probabilities = (torch.rand(18, 3, dtype=torch.float64)*.7+.15).requires_grad_(True)
    labels = torch.arange(18) % 2
    assert torch.autograd.gradcheck(lambda value: alg.simplex_brier_weights(value, labels, .08),
                                   (probabilities,), eps=1e-6, atol=2e-6, rtol=1e-4)


def test_source_centered_ridge_uses_source_when_components_identical():
    y = torch.arange(8) % 2
    p = torch.linspace(.1, .8, 8, dtype=torch.float64)
    assert alg.shrink_weight(p, p, y, .01) == 0
    mix = alg.simplex_brier_weights(p[:, None].expand(-1,3), y, .01)
    torch.testing.assert_close(mix, mix.new_tensor([1.,0.,0.]), atol=1e-12,rtol=0)


def test_blocked_folds_use_original_paid_positions_after_failed_qc():
    _, _, support, _, _ = fixture()
    indices = torch.tensor([0] + list(range(5,20)))
    modified = replace(support, x=support.x[indices], y=support.y[indices],
                       event_uids=tuple(support.event_uids[i] for i in indices.tolist()))
    folds = alg.support_folds(modified, 5, 'blocked')
    assert [fold.tolist() for fold in folds] == [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
    assert sorted(torch.cat(folds).tolist()) == list(range(16))
    assert len(alg.support_folds(modified, 5, 'loo')) == 16


def test_oof_training_never_sees_held_labels_and_cache_is_label_free():
    source, geometry, support, b, _ = fixture()
    cfg = alg.Config(variant='xfit_shrink', tangent_steps=2, head_l2=.1)
    cached = alg.support_tangent(source, geometry, support)
    folds = alg.support_folds(support)
    first = alg._oof(source, geometry, b, support, cached, cfg, folds)[2]
    changed_y = support.y.clone(); changed_y[folds[0]] = 1-changed_y[folds[0]]
    changed = replace(support, y=changed_y)
    second = alg._oof(source, geometry, b, changed, cached, cfg, folds)[2]
    # Other folds train on the changed labels, but this held block must agree.
    torch.testing.assert_close(first[folds[0]], second[folds[0]], atol=1e-12, rtol=1e-10)


def test_one_class_training_fold_falls_back_to_source_without_abort():
    source, geometry, support, b, _ = fixture()
    labels = torch.zeros_like(support.y); labels[:4] = 1
    support = replace(support, y=labels)
    cfg = alg.Config(variant='xfit_shrink', tangent_steps=1, head_l2=.1)
    cached = alg.support_tangent(source,geometry,support)
    folds = alg.support_folds(support)
    p0, _, ps, failures, count = alg._oof(source,geometry,b,support,cached,cfg,folds)
    assert failures == 1 and count == 5
    torch.testing.assert_close(ps[:4],p0[:4])
    result = alg.fit(source,geometry,b,support,cfg)
    assert result.fallback_reason is None
    assert result.diagnostics['cv']['fold_fallbacks'] == 1
    assert result.diagnostics['cv']['valid_folds'] == 4


@pytest.mark.parametrize('variant', ['xfit_shrink','xfit_stack','xfit_select','xfit_select_stack'])
def test_tangent_is_computed_once_and_all_selection_stays_on_support(variant,monkeypatch):
    source,geometry,support,b,query = fixture()
    original = alg.core.support_tangent
    calls=[]
    def audited(*args):
        assert args[2] is support
        calls.append(len(args[2].y))
        return original(*args)
    monkeypatch.setattr(alg.core,'support_tangent',audited)
    cfg=alg.Config(variant=variant,tangent_steps=1,head_l2=.1,cv_head_grid=(.01,.1),cv_radius_grid=(.05,.15))
    result=alg.fit(source,geometry,b,support,cfg)
    assert calls == [20]
    assert result.diagnostics['cv']['selection_labels']=='acquired_support_only'
    assert result.diagnostics['cv']['n_candidate_fits'] > 5
    assert 'query' not in inspect.signature(alg.fit).parameters
    assert 'query_y' not in inspect.signature(alg.predict).parameters
    before=result.diagnostics.copy()
    assert torch.isfinite(alg.predict(source,geometry,b,result,query)).all()
    assert result.diagnostics==before


def test_selection_one_se_tie_prefers_small_radius_and_larger_head_prior(monkeypatch):
    source,geometry,support,b,_=fixture()
    def same_prediction(source,geometry,b,support,cached,config,folds,**kwargs):
        p=cached[0].sigmoid()
        return p,p,p,0,len(folds)
    monkeypatch.setattr(alg,'_oof',same_prediction)
    cfg=alg.Config(variant='xfit_select',tangent_steps=1,head_l2=.1)
    result=alg.fit(source,geometry,b,support,cfg)
    chosen=result.diagnostics['cv']['selected_config']
    assert chosen['head_l2']==1 and chosen['radius']==.05
    assert result.diagnostics['cv']['candidate_count']==6
    assert result.diagnostics['cv']['selection_rule']=='one_se'


@pytest.mark.parametrize('variant,expected', [('fixed_half',[.5,.5]),('uniform_stack',[1/3]*3)])
def test_fixed_controls_do_not_run_cross_fitting(variant,expected,monkeypatch):
    source,geometry,support,b,_=fixture()
    monkeypatch.setattr(alg,'_oof',lambda *a,**k:pytest.fail('Fixed control ran OOF'))
    result=alg.fit(source,geometry,b,support,alg.Config(variant=variant,tangent_steps=1,head_l2=.1))
    torch.testing.assert_close(result.mixture_weights,result.mixture_weights.new_tensor(expected))
    assert result.diagnostics['cv']['cv_scheme']=='none'


def test_final_selected_head_refit_matches_selected_oof_head_prior(monkeypatch):
    source,geometry,support,b,_=fixture()
    original=alg._cached_fit
    records=[]
    def audited(*args,**kwargs):
        if len(args[3].y)==20 and kwargs.get('variant')=='head_newton':
            records.append(args[5].head_l2)
        return original(*args,**kwargs)
    monkeypatch.setattr(alg.round4,'_cached_fit',audited)
    cfg=alg.Config(variant='xfit_select_stack',tangent_steps=1,cv_head_grid=(1.,),cv_radius_grid=(.05,))
    result=alg.fit(source,geometry,b,support,cfg)
    assert records==[1.]
    assert result.diagnostics['cv']['selected_config']['head_l2']==1.


def test_probabilities_are_actual_component_convex_average():
    source,geometry,support,b,query=fixture()
    cfg=alg.Config(variant='xfit_stack',tangent_steps=2,head_l2=.1)
    result=alg.fit(source,geometry,b,support,cfg)
    probabilities=alg.predict(source,geometry,b,result,query).softmax(1)[:,1]
    component=torch.stack([alg._probability(source,geometry,b,state,query) for state in result.components],1)
    torch.testing.assert_close(probabilities,component@result.mixture_weights,atol=1e-12,rtol=1e-10)
    assert not result.mixture_weights.requires_grad
    assert not result.weights.theta.requires_grad


@pytest.mark.parametrize('q',[.3,.7,1.])
def test_gce_true_target_objective_is_monotonic_not_only_weighted_ce(q):
    source,geometry,support,b,_=fixture()
    result=alg.fit(source,geometry,b,support,alg.Config(variant='gce_profiled',gce_q=q,tangent_steps=5,head_l2=.1))
    history=result.diagnostics['gce_objective_history']
    assert len(history)==6
    assert all(second<=first+1e-12 for first,second in zip(history,history[1:]))
    assert result.diagnostics['surrogate']=='fixed_identity_tangent_nonconvex_GCE'
    assert 0<result.diagnostics['robust_target_probability_weight_mean']<=1


def test_gce_zero_limit_matches_original_profiled_ce_exactly():
    source,geometry,support,b,query=fixture()
    config=alg.Config(variant='gce_profiled',gce_q=0,tangent_steps=3,head_l2=.1)
    robust=alg.fit(source,geometry,b,support,config)
    ce=alg.fit(source,geometry,b,support,replace(config,variant='tangent_profiled'))
    for name in ('theta','delta_w','delta_b'):
        torch.testing.assert_close(getattr(robust.weights,name),getattr(ce.weights,name),atol=0,rtol=0)
    torch.testing.assert_close(alg.predict(source,geometry,b,robust,query),alg.predict(source,geometry,b,ce,query),atol=0,rtol=0)


@pytest.mark.parametrize('variant',['xfit_shrink','xfit_stack','gce_profiled'])
def test_complete_meta_gradient_through_oof_fit_mixture_and_actual_query(variant):
    source,geometry,support,b,query=fixture()
    config=alg.Config(variant=variant,tangent_steps=2,head_l2=.1,radius=.005,cv_folds=3,stacking_l2=.001)
    def value(dictionary):
        state=alg.fit(source,geometry,dictionary,support,config,create_graph=True)
        return alg.predict(source,geometry,dictionary,state,query).square().mean()
    gradient=torch.autograd.grad(value(b),b)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm()) > 1e-10
    direction=torch.randn_like(b);direction/=direction.norm()
    epsilon=2e-5
    difference=(value(b+epsilon*direction).detach()-value(b-epsilon*direction).detach())/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(),difference,atol=2e-8,rtol=3e-3)


@pytest.mark.parametrize('variant',['xfit_shrink','xfit_stack','xfit_select_stack','gce_profiled'])
def test_single_class_fallback_is_source_and_does_not_abort(variant):
    source,geometry,support,b,query=fixture()
    support=replace(support,y=torch.zeros_like(support.y))
    result=alg.fit(source,geometry,b,support,alg.Config(variant=variant))
    assert result.fallback_reason=='insufficient_class_support'
    torch.testing.assert_close(alg.predict(source,geometry,b,result,query),source.source_logits(query))


@pytest.mark.parametrize('variant',['xfit_shrink','xfit_stack','xfit_select','xfit_select_stack'])
def test_no_valid_training_fold_reverts_to_source_without_failing_task(variant):
    source,geometry,support,b,query=fixture()
    # Two valid trials remain in the paid B20 prefix; all held fold fits lack
    # both classes. Selection cannot pretend those OOF fits identify a learner.
    support=replace(support,x=support.x[:2],y=support.y[:2],event_uids=support.event_uids[:2])
    result=alg.fit(source,geometry,b,support,alg.Config(variant=variant,tangent_steps=1))
    assert result.fallback_reason=='cv_no_valid_training_fold'
    assert result.diagnostics['cv']['valid_folds']==0
    assert result.diagnostics['cv']['mix_source']==1
    torch.testing.assert_close(alg.predict(source,geometry,b,result,query),source.source_logits(query))
