"""Actual chart calibration and complete first meta gradients on synthetic EEG."""
from dataclasses import replace
import inspect

import pytest
import torch

from test_algorithm import fixture
from paper3r6 import algorithm as alg
from paper3r6.maps import matrix_map

CHARTS=('single','left_rotation','right_rotation','symmetric')


@pytest.mark.parametrize('chart',CHARTS)
def test_refinement_uses_its_own_actual_support_objective_and_preserves_source(chart):
    source,geometry,support,b,query=fixture()
    before={name:value.detach().clone() for name,value in source.state_dict().items()}
    config=alg.Config(variant='geometry_refined',geometry=chart,tangent_steps=3,
                      inner_steps=5,head_l2=.1,radius=.15)
    result=alg.fit(source,geometry,b,support,config)
    history=result.diagnostics['actual_support_loss_history']
    assert len(history)==6 and result.diagnostics['actual_fit_steps']==5
    assert all(second <= first+2e-12 for first,second in zip(history,history[1:]))
    actual=alg._geometry_loss(source,geometry,b,support,result.weights,config)
    assert float(actual.detach())==pytest.approx(history[-1],abs=2e-12)
    expected=alg.actual_geometry(source,geometry,b,result.weights,query,chart)[0]
    torch.testing.assert_close(alg.predict(source,geometry,b,result,query),expected,atol=0,rtol=0)
    assert result.geometry==chart
    assert result.diagnostics['optimizer_scope']=='actual_chart_balanced_CE_plus_ridge'
    assert result.diagnostics['actual_nonlinear_optimization_performed']
    assert not result.weights.theta.requires_grad
    assert torch.isfinite(expected).all()
    for name,value in source.state_dict().items():
        torch.testing.assert_close(value,before[name],atol=0,rtol=0)


def test_common_tangent_probe_really_has_identical_parameters_but_actual_different_charts():
    source,geometry,support,b,query=fixture()
    answers={chart:alg.fit(source,geometry,b,support,alg.Config(variant='geometry_tangent',
             geometry=chart,tangent_steps=4,head_l2=.1,radius=.35)) for chart in alg.GEOMETRIES}
    baseline=answers['single']
    for answer in answers.values():
        assert answer.diagnostics['actual_fit_steps']==0
        assert answer.diagnostics['optimizer_scope']=='common_identity_tangent_probe'
        assert not answer.diagnostics['actual_nonlinear_optimization_performed']
        for key in ('theta','delta_w','delta_b'):
            torch.testing.assert_close(getattr(answer.weights,key),getattr(baseline.weights,key),atol=0,rtol=0)
    left=alg.predict(source,geometry,b,answers['left_rotation'],query)
    right=alg.predict(source,geometry,b,answers['right_rotation'],query)
    assert float((left-right).abs().max()) > 1e-8


def test_first_zero_initialized_update_is_identical_under_all_primary_charts():
    source,geometry,support,b,_=fixture()
    answers=[alg.fit(source,geometry,b,support,alg.Config(variant='geometry_sgd_zero',
             geometry=chart,inner_steps=1,head_l2=.1)) for chart in CHARTS]
    for answer in answers[1:]:
        for name in ('theta','delta_w','delta_b'):
            torch.testing.assert_close(getattr(answer.weights,name),getattr(answers[0].weights,name),
                                       atol=2e-14,rtol=1e-10)


@pytest.mark.parametrize('chart',CHARTS)
def test_complete_outer_gradient_through_warm_start_and_actual_map_refinement(chart):
    source,geometry,support,b,query=fixture()
    config=alg.Config(variant='geometry_refined',geometry=chart,tangent_steps=2,
                      inner_steps=2,head_l2=.1,radius=.035)
    def value(dictionary):
        result=alg.fit(source,geometry,dictionary,support,config,create_graph=True)
        return alg.predict(source,geometry,dictionary,result,query).square().mean()
    gradient=torch.autograd.grad(value(b),b)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm())>1e-10
    direction=torch.randn_like(b);direction/=direction.norm()
    epsilon=2e-5
    difference=(value(b+epsilon*direction).detach()-value(b-epsilon*direction).detach())/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(),difference,atol=3e-8,rtol=4e-3)


@pytest.mark.parametrize('variant',alg.GEOMETRY_VARIANTS)
def test_single_class_geometry_fallback_preserves_source_predictions(variant):
    source,geometry,support,b,query=fixture()
    support=replace(support,y=torch.zeros_like(support.y))
    result=alg.fit(source,geometry,b,support,alg.Config(variant=variant,geometry='symmetric'))
    assert result.fallback_reason=='insufficient_class_support'
    torch.testing.assert_close(alg.predict(source,geometry,b,result,query),source.source_logits(query),atol=0,rtol=0)


def test_optional_matched_white_amplitude_budget_is_feasible_without_aborting():
    source,geometry,support,b,_=fixture()
    result=alg.fit(source,geometry,b,support,alg.Config(variant='geometry_refined',
        geometry='symmetric',tangent_steps=2,inner_steps=2,head_l2=.1,max_transform_deviation=.001))
    assert result.diagnostics['white_transform_deviation']<=.001+1e-11
    assert result.diagnostics['final_actual_support_loss']<=result.diagnostics['initial_actual_support_loss']+2e-12


def test_chart_selection_and_refinement_accept_no_future_query_information():
    parameters=inspect.signature(alg.fit).parameters
    assert not any('query' in name for name in parameters)
    with pytest.raises(ValueError,match='BCH2'):
        alg.Config(variant='geometry_refined',geometry='bch2')
    with pytest.raises(ValueError,match='Unknown nonlinear geometry'):
        alg.Config(geometry='unregistered')
    with pytest.raises(ValueError,match='Common-tangent'):
        alg.Config(variant='geometry_tangent',max_transform_deviation=.1)
