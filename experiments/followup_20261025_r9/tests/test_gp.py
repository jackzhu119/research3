import math
from dataclasses import replace
import pytest
import torch
from paper3r9.gp import (GPConfig,GPFit,fit_gp,fit_reference_geometry,
                        ReferenceGeometry,_quadrature_probability)


def fixture(seed=19,n=20,d=64):
    generator=torch.Generator().manual_seed(seed)
    x=torch.randn(n,d,generator=generator,dtype=torch.float64)
    f=torch.randn(n,generator=generator,dtype=torch.float64)
    y=(torch.arange(n)%2).long()
    q=torch.randn(13,d,generator=generator,dtype=torch.float64)
    fq=torch.randn(13,generator=generator,dtype=torch.float64)
    return x,f,y,q,fq


def test_zero_prior_and_missing_class_are_exact_source():
    x,f,y,q,fq=fixture()
    logits=torch.stack((-fq/2,fq/2),1)
    for config,labels,reason in ((GPConfig(tau=0),y,'zero_residual_prior'),
        (GPConfig(),torch.zeros_like(y),'insufficient_class_support')):
        fitted=fit_gp(x,f,labels,config)
        assert fitted.diagnostics['fallback_reason']==reason
        for mode in ('map','laplace','anchored_laplace'):
            assert torch.equal(fitted.predict_proba(q,logits,mode),torch.softmax(logits,1))


def test_zero_linear_kernel_falls_back_and_finite_support_rank():
    x,f,y,q,fq=fixture()
    fitted=fit_gp(torch.zeros_like(x),f,y)
    assert fitted.diagnostics['fallback_reason']=='zero_support_kernel'
    fitted=fit_gp(x,f,y,GPConfig(tau=1))
    assert fitted.diagnostics['support_kernel_rank']<=20
    assert 0<=fitted.diagnostics['effective_df']<=20
    assert fitted.diagnostics['final_gradient_max']<1e-6


@pytest.mark.parametrize('kernel',['linear','rbf'])
def test_psd_variance_probability_and_anchor_loss_bound(kernel):
    x,f,y,q,fq=fixture()
    config=GPConfig(kernel=kernel,tau=2,anchor_cap=.75)
    fitted=fit_gp(x,f,y,config)
    mean,variance,information,prior=fitted.posterior_moments(q)
    assert bool((variance>=0).all()) and bool((variance<=prior).all())
    assert bool((information>=0).all()) and bool((information<=1).all())
    source=torch.stack((torch.sigmoid(-fq),torch.sigmoid(fq)),1)
    raw=fitted.predict_proba(q,fq,'laplace')
    anchored=fitted.predict_proba(q,fq,'anchored_laplace')
    gate=config.anchor_cap*information
    torch.testing.assert_close(anchored,(1-gate[:,None])*source+gate[:,None]*raw)
    assert bool((anchored>0).all())
    torch.testing.assert_close(anchored.sum(1),torch.ones(len(q),dtype=q.dtype))
    assert bool(((anchored-source).abs().max(1).values<=gate+1e-12).all())
    for label in (0,1):
        loss_increase=-anchored[:,label].log()+source[:,label].log()
        assert bool((loss_increase<=-torch.log1p(-gate)+1e-12).all())


def test_unseen_linear_direction_and_far_rbf_preserve_source():
    x=torch.tensor([[1.,0],[-1.,0],[2.,0],[-2.,0]],dtype=torch.float64)
    y=torch.tensor([1,0,1,0]);f=torch.zeros(4,dtype=torch.float64)
    q=torch.tensor([[0.,8.],[0.,-8.]],dtype=torch.float64)
    fq=torch.tensor([3.,-3.],dtype=torch.float64)
    fit=fit_gp(x,f,y,GPConfig(tau=2,anchor_cap=.75))
    source=torch.stack((torch.sigmoid(-fq),torch.sigmoid(fq)),1)
    torch.testing.assert_close(fit.predict_proba(q,fq,'anchored_laplace'),source,rtol=0,atol=0)
    # Plain posterior averaging does move toward 0.5 in unseen directions.
    assert bool((fit.predict_proba(q,fq,'laplace')[:,1]-source[:,1]).abs().min()>.1)
    far=fit_gp(x,f,y,GPConfig(kernel='rbf',tau=2,lengthscale=.2))
    torch.testing.assert_close(far.predict_proba(q*1e5,fq,'anchored_laplace'),source,rtol=0,atol=0)


def test_zero_gate_cap_is_exact_source():
    x,f,y,q,fq=fixture()
    fitted=fit_gp(x,f,y,GPConfig(anchor_cap=0))
    source=torch.stack((torch.sigmoid(-fq),torch.sigmoid(fq)),1)
    assert torch.equal(fitted.predict_proba(q,fq),source)


def test_map_and_laplace_share_identical_mode_and_state_roundtrip():
    x,f,y,q,fq=fixture()
    fitted=fit_gp(x,f,y,GPConfig(tau=1))
    before=fitted.mode.clone()
    fitted.predict_proba(q,fq,'map')
    fitted.predict_proba(q,fq,'laplace')
    assert torch.equal(before,fitted.mode)
    clone=GPFit.from_state_dict(fitted.state_dict())
    torch.testing.assert_close(clone.predict_proba(q,fq),fitted.predict_proba(q,fq),rtol=0,atol=0)


def test_kernel_permutation_invariance_and_logits_gauge():
    x,f,y,q,fq=fixture()
    permutation=torch.arange(len(x)-1,-1,-1)
    original=fit_gp(x,f,y,GPConfig(kernel='rbf',tau=.8))
    permuted=fit_gp(x[permutation],f[permutation],y[permutation],GPConfig(kernel='rbf',tau=.8))
    torch.testing.assert_close(original.predict_proba(q,fq),permuted.predict_proba(q,fq),atol=1e-9,rtol=1e-8)
    binary=torch.stack((-f/2+7,f/2+7),1)
    other=fit_gp(x,binary,y,GPConfig(kernel='rbf',tau=.8))
    logits=torch.stack((-fq/2-4,fq/2-4),1)
    torch.testing.assert_close(original.predict_proba(q,fq),other.predict_proba(q,logits),atol=1e-9,rtol=1e-8)


def test_one_dimensional_map_solves_direct_scalar_problem():
    x=torch.tensor([[1.],[2.],[-.5],[-1.]],dtype=torch.float64)
    f=torch.tensor([.3,-.1,.4,-.8],dtype=torch.float64)
    y=torch.tensor([1,0,0,1])
    config=GPConfig(tau=.7)
    fit=fit_gp(x,f,y,config)
    assert fit.diagnostics['support_kernel_rank']==1
    # Direct primal prior g(x)=tau*x*v, independent of GP eigenspace signs.
    v=torch.tensor(0.,dtype=torch.float64)
    for _ in range(30):
        p=torch.sigmoid(f+config.tau*x[:,0]*v)
        grad=v+(config.tau*x[:,0]*(p-y)).sum()
        hess=1+(config.tau*x[:,0]).square().mul(p*(1-p)).sum()
        v=v-grad/hess
    q=torch.tensor([[.4],[-2.]],dtype=torch.float64)
    shift,var,info,prior=fit.posterior_moments(q)
    torch.testing.assert_close(shift,config.tau*q[:,0]*v,atol=1e-8,rtol=1e-8)
    p=torch.sigmoid(f+config.tau*x[:,0]*v)
    hess=1+(config.tau*x[:,0]).square().mul(p*(1-p)).sum()
    torch.testing.assert_close(var,(config.tau*q[:,0]).square()/hess,atol=1e-8,rtol=1e-8)


def test_quadrature_matches_high_order_reference_in_regular_range():
    mean=torch.tensor([-4.,-1.,0.,.3,2.,4.],dtype=torch.float64)
    var=torch.tensor([.1,.4,1.,2.,3.,.2],dtype=torch.float64)
    value=_quadrature_probability(mean,var,20)
    reference=_quadrature_probability(mean,var,100)
    torch.testing.assert_close(value,reference,atol=2e-6,rtol=2e-6)


def test_weighted_reference_is_training_only_and_serializes():
    x,_,_,q,_=fixture(n=25,d=8)
    weights=torch.arange(1,len(x)+1,dtype=x.dtype)
    ref=fit_reference_geometry(x,weights=weights,provenance={'split':'meta_train','test_labels_used':False})
    expected=(x*weights[:,None]).sum(0)/weights.sum()
    torch.testing.assert_close(ref.mean,expected)
    assert ref.provenance['split']=='meta_train'
    assert ref.lengthscale>0
    copy=ReferenceGeometry.from_state_dict(ref.state_dict())
    torch.testing.assert_close(copy.transform(q),ref.transform(q))
    assert ref.diagnostics['reference_rows']==25


def test_reference_fit_matches_explicit_pretransform():
    x,f,y,q,fq=fixture(d=8)
    ref=fit_reference_geometry(torch.cat((x,x*.2)),provenance={'split':'meta_train'})
    config=GPConfig(kernel='rbf',tau=.5,lengthscale=ref.lengthscale)
    direct=fit_gp(ref.transform(x),f,y,config)
    implicit=fit_gp(x,f,y,config,reference=ref)
    torch.testing.assert_close(direct.predict_proba(ref.transform(q),fq),implicit.predict_proba(q,fq))


def test_nonfinite_input_is_explicit_source_fallback():
    x,f,y,q,fq=fixture()
    x[0,0]=float('nan')
    fit=fit_gp(x,f,y)
    assert fit.diagnostics['fallback_reason']=='nonfinite_support'
    p=fit.predict_proba(q,fq)
    assert bool(torch.isfinite(p).all())


def test_constant_mix_and_anchored_map_share_state():
    x,f,y,q,fq=fixture()
    fit=fit_gp(x,f,y)
    source=torch.stack((torch.sigmoid(-fq),torch.sigmoid(fq)),1)
    posterior=fit.predict_proba(q,fq,'laplace')
    constant=fit.predict_proba(q,fq,'constant_laplace')
    torch.testing.assert_close(constant,.25*source+.75*posterior)
    _,_,r,_=fit.posterior_moments(q)
    map_probability=fit.predict_proba(q,fq,'map')
    anchored=fit.predict_proba(q,fq,'anchored_map')
    torch.testing.assert_close(anchored,(1-.75*r[:,None])*source+.75*r[:,None]*map_probability)


def test_nonfinite_query_row_retains_other_rows_and_source():
    x,f,y,q,fq=fixture()
    fit=fit_gp(x,f,y)
    original=fit.predict_proba(q,fq)
    bad=q.clone();bad[3,5]=float('nan')
    for mode in ('map','laplace','constant_laplace','anchored_laplace'):
        before=fit.predict_proba(q,fq,mode)
        after=fit.predict_proba(bad,fq,mode)
        assert fit.last_prediction_diagnostics['nonfinite_query_feature_fallback_rows']==1
        mask=torch.arange(len(q))!=3
        torch.testing.assert_close(before[mask],after[mask])
        torch.testing.assert_close(after[3],torch.stack((torch.sigmoid(-fq[3]),torch.sigmoid(fq[3]))))
