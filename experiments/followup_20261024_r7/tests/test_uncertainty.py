"""Curvature stencil identities and mathematical nulls on synthetic EEG."""
from dataclasses import replace
import inspect

import pytest
import torch

from test_algorithm import fixture
from paper3r7 import algorithm as alg


def config(**changes):
    return alg.Config(variant='uq_joint',head_l2=.1,tangent_steps=3,**changes)


def test_full_block_inverse_equals_schur_conditional_covariance():
    torch.manual_seed(601)
    j,z=torch.randn(20,4,dtype=torch.float64),torch.randn(20,7,dtype=torch.float64)
    margin=torch.randn(20,dtype=torch.float64)
    y=torch.arange(20)%2
    a,c,d,s,sinv,dinv,k=alg.local_precision(j,z,margin,y,config())
    full=torch.cat((torch.cat((a,c),1),torch.cat((c.T,d),1)),0)
    inverse=torch.linalg.inv(full)
    torch.testing.assert_close(inverse[:4,:4],sinv,atol=3e-12,rtol=3e-11)
    torch.testing.assert_close(inverse[:4,4:],-sinv @ k.T,atol=3e-12,rtol=3e-11)
    torch.testing.assert_close(inverse[4:,4:],dinv+k @ sinv @ k.T,atol=3e-12,rtol=3e-11)
    torch.testing.assert_close(s,a-c @ torch.linalg.solve(d,c.T))


def test_covariance_excludes_solver_damping_and_normalizes_only_spatial_trace():
    source,geometry,support,b,_=fixture()
    m0,coordinates,features=alg.support_tangent(source,geometry,support)
    j,z=coordinates @ b,alg.core._augmented(features)
    answers=[alg.local_precision(j,z,m0,support.y,config(damping=d)) for d in (0.,10.)]
    for first,second in zip(*answers):
        torch.testing.assert_close(first,second,atol=0,rtol=0)
    for variant in ('uq_joint','uq_isotropic','uq_diagonal'):
        covariance,tau=alg.covariance_shape(answers[0][4],.15,.25,variant)
        assert float(covariance.trace().detach())==pytest.approx(.25*.15**2,abs=1e-15)
        assert float(tau.detach())>0
        assert torch.linalg.eigvalsh(covariance).min()>0


@pytest.mark.parametrize('center_weight',[0.,.5])
def test_unprojected_positive_stencil_exactly_matches_first_two_moments(center_weight):
    theta=torch.tensor([.1,-.03,.04,.02],dtype=torch.float64)
    factor=torch.tensor([[.04,0,0,0],[.02,.03,0,0],[0,.01,.02,0],[.01,0,.01,.025]],dtype=torch.float64)
    covariance=factor @ factor.T
    points,weights,raw=alg.sigma_stencil(theta,covariance,10.,center_weight)
    mean=(weights[:,None]*points).sum(0)
    centered=points-mean
    actual=centered.T @ (weights[:,None]*centered)
    torch.testing.assert_close(mean,theta,atol=1e-16,rtol=1e-15)
    torch.testing.assert_close(actual,covariance,atol=2e-18,rtol=1e-14)
    assert weights.min()>=0 and float(weights.sum())==pytest.approx(1.)
    torch.testing.assert_close(points,raw,atol=0,rtol=0)


def test_projection_changes_mean_and_moments_and_is_explicitly_recorded():
    theta=torch.tensor([.15,0,0,0],dtype=torch.float64)
    covariance=torch.eye(4,dtype=torch.float64)*.25*.15**2/4
    points,weights,raw=alg.sigma_stencil(theta,covariance,.15)
    assert (points.norm(dim=1)<=.15+1e-14).all()
    assert (raw.norm(dim=1)>.15+1e-12).any()
    mean=(weights[:,None]*points).sum(0)
    assert mean[0]<theta[0]
    centered=points-mean
    assert (centered.T @ (weights[:,None]*centered)).trace()<covariance.trace()


def test_fixed_stencil_rotation_preserves_raw_moments_without_changing_rng():
    theta=torch.zeros(4,dtype=torch.float64)
    covariance=torch.diag(theta.new_tensor([.001,.002,.003,.004]))
    before=torch.get_rng_state().clone()
    first=alg.sigma_stencil(theta,covariance,10.,rotation=721)
    second=alg.sigma_stencil(theta,covariance,10.,rotation=721)
    torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
    torch.testing.assert_close(first[0],second[0],atol=0,rtol=0)
    points,weights,_=first
    torch.testing.assert_close(points.T @ (weights[:,None]*points),covariance,atol=5e-18,rtol=1e-13)
    assert not torch.equal(points,alg.sigma_stencil(theta,covariance,10.)[0])


@pytest.mark.parametrize('order',[5,9,17])
def test_logistic_normal_symmetric_noise_preserves_exact_labels_even_near_ties(order):
    margin=torch.tensor([-100.,-1.,-1e-20,0.,1e-20,1.,100.],dtype=torch.float64)
    variance=torch.tensor([0.,.1,1.,10.,1.,.1,0.],dtype=torch.float64)
    logits=alg._symmetric_noise_logits(margin,variance,order)
    assert torch.equal(logits.argmax(1),(margin>0).long())
    assert torch.isfinite(logits).all()
    assert logits[3,0]==logits[3,1]
    probability=logits.softmax(1)[:,1]
    torch.testing.assert_close(probability,1-probability.flip(0),atol=1e-12,rtol=1e-12)


def test_symmetric_noise_at_zero_retains_positive_continuous_derivative():
    margin=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    variance=torch.tensor([0.,1.,10.],dtype=torch.float64)
    logits=alg._symmetric_noise_logits(margin,variance,9)
    derivative=torch.autograd.grad((logits[:,1]-logits[:,0]).sum(),margin)[0]
    assert (derivative>0).all()
    assert derivative[0]==pytest.approx(1.)


def test_log_domain_quadrature_does_not_artificially_cap_extreme_nll():
    margin=torch.tensor([-1000.,1000.],dtype=torch.float64)
    logprob=alg.normal_logistic_logprob(margin,torch.zeros_like(margin),9)
    expected=torch.stack((torch.nn.functional.logsigmoid(-margin),
                          torch.nn.functional.logsigmoid(margin)),1)
    torch.testing.assert_close(logprob,expected,atol=3e-13,rtol=1e-13)
    assert float(-logprob[0,1])==pytest.approx(1000.)


@pytest.mark.parametrize('variant',alg.UQ_VARIANTS)
def test_uq_variants_share_common_center_and_bound_every_spatial_point(variant):
    source,geometry,support,b,query=fixture()
    cfg=replace(config(),variant=variant)
    answer=alg.fit(source,geometry,b,support,cfg)
    common=alg.fit(source,geometry,b,support,replace(cfg,variant='uq_map'))
    for name in ('theta','delta_w','delta_b'):
        torch.testing.assert_close(getattr(answer.weights,name),getattr(common.weights,name),atol=0,rtol=0)
    if variant!='uq_map':
        assert (answer.sigma_theta.norm(dim=1)<=cfg.radius+1e-12).all()
        assert float(answer.point_weights.sum())==pytest.approx(1.)
        expected=alg._head_vector(answer.weights)[None]-(answer.sigma_theta-answer.weights.theta) @ answer.head_compensation.T
        torch.testing.assert_close(answer.sigma_heads,expected,atol=1e-14,rtol=1e-12)
        assert answer.diagnostics['precision_solver_damping']==0.
        assert not answer.diagnostics['exact_bayesian_posterior']
    with torch.no_grad():
        logits=alg.predict(source,geometry,b,answer,query)
    assert logits.shape==(len(query),2) and torch.isfinite(logits).all()
    assert not answer.weights.theta.requires_grad


@pytest.mark.parametrize('variant',['uq_head','uq_linear'])
def test_head_and_linear_predictors_leave_actual_map_hard_labels_unchanged(variant):
    source,geometry,support,b,query=fixture()
    answer=alg.fit(source,geometry,b,support,replace(config(),variant=variant))
    actual=alg.geometry5.actual_geometry(source,geometry,b,answer.weights,query,'single')[0]
    with torch.no_grad():
        logits=alg.predict(source,geometry,b,answer,query)
    assert torch.equal(logits.argmax(1),actual.argmax(1))
    assert float((logits.softmax(1)-actual.softmax(1)).abs().max().detach())>1e-10


@pytest.mark.parametrize('variant',alg.UQ_VARIANTS)
def test_zero_variance_limit_is_exact_actual_common_center(variant):
    source,geometry,support,b,query=fixture()
    answer=alg.fit(source,geometry,b,support,replace(config(variance_fraction=0.),variant=variant))
    expected=alg.geometry5.actual_geometry(source,geometry,b,answer.weights,query,'single')[0]
    torch.testing.assert_close(alg.predict(source,geometry,b,answer,query),expected,atol=0,rtol=0)


def test_mean_control_matches_projected_parameter_centroid_exactly():
    source,geometry,support,b,query=fixture()
    answer=alg.fit(source,geometry,b,support,replace(config(),variant='uq_mean'))
    theta=(answer.point_weights[:,None]*answer.sigma_theta).sum(0)
    head=(answer.point_weights[:,None]*answer.sigma_heads).sum(0)
    margin,_=alg._actual_components(source,geometry,b,theta,head,query)
    expected=torch.stack((-margin/2,margin/2),1)
    torch.testing.assert_close(alg.predict(source,geometry,b,answer,query),expected,atol=0,rtol=0)
    assert answer.diagnostics['centroid_theta_shift']>0


def test_independent_pairing_is_exact_product_with_same_finite_marginals(monkeypatch):
    source,geometry,support,b,query=fixture()
    answer=alg.fit(source,geometry,b,support,replace(config(),variant='uq_independent'))
    expected=query.new_zeros(len(query))
    for i,theta in enumerate(answer.sigma_theta):
        for j,head in enumerate(answer.sigma_heads):
            margin,z=alg._actual_components(source,geometry,b,theta,head,query)
            p=alg.normal_logistic_probability(margin,alg._head_variance(z,answer),9)
            expected=expected+answer.point_weights[i]*answer.point_weights[j]*p
    calls=[]
    actual_components=alg.geometry6._actual_components
    def record(*args,**kwargs):
        calls.append(1)
        return actual_components(*args,**kwargs)
    monkeypatch.setattr(alg.geometry6,'_actual_components',record)
    actual=alg.predict(source,geometry,b,answer,query).softmax(1)[:,1]
    torch.testing.assert_close(actual,expected,atol=3e-15,rtol=1e-13)
    assert len(calls)==len(answer.sigma_theta)==9
    product=answer.point_weights[:,None]*answer.point_weights[None]
    torch.testing.assert_close(product.sum(0),answer.point_weights)
    torch.testing.assert_close(product.sum(1),answer.point_weights)


@pytest.mark.parametrize('variant',['uq_joint','uq_independent','uq_linear'])
def test_query_predictions_are_batch_independent_and_preserve_source(variant):
    source,geometry,support,b,query=fixture()
    state={k:v.clone() for k,v in source.state_dict().items()}
    answer=alg.fit(source,geometry,b,support,replace(config(),variant=variant))
    together=alg.predict(source,geometry,b,answer,query)
    apart=torch.cat([alg.predict(source,geometry,b,answer,item[None]) for item in query])
    torch.testing.assert_close(together,apart,atol=3e-13,rtol=1e-11)
    for k,v in source.state_dict().items():
        torch.testing.assert_close(v,state[k],atol=0,rtol=0)


def test_uq_support_api_has_no_query_and_invalid_config_is_explicit():
    assert not any('query' in name for name in inspect.signature(alg.fit).parameters)
    for changes in ({'variance_fraction':-.1},{'head_quadrature':7},
                    {'sigma_center_weight':1.},{'quadrature_rotation':True},
                    {'geometry':'symmetric'}):
        with pytest.raises(ValueError):
            config(**changes)


@pytest.mark.parametrize('variant',['uq_joint','uq_head','uq_linear'])
def test_single_class_fallback_retains_source(variant):
    source,geometry,support,b,query=fixture()
    support=replace(support,y=torch.zeros_like(support.y))
    answer=alg.fit(source,geometry,b,support,replace(config(),variant=variant))
    assert answer.fallback_reason=='insufficient_class_support'
    torch.testing.assert_close(alg.predict(source,geometry,b,answer,query),source.source_logits(query),atol=0,rtol=0)
