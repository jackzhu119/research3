"""Finite real temporal maps, gradients and invariants on synthetic tensors."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from paper3r7 import temporal
from p3work.adapt import CalibrationPrefix, FastWeights
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary, joint_trial_rms


torch.set_num_threads(1)


def tensors(channels=3, times=7):
    generator = torch.Generator().manual_seed(7701 + times)
    y = torch.randn(4, channels, times, generator=generator, dtype=torch.float64)
    y -= y.mean(-1, keepdim=True)
    m = torch.randn(channels, channels, generator=generator, dtype=torch.float64) * .07
    m -= torch.eye(channels, dtype=m.dtype) * m.trace() / channels
    return y, m


def dense_reference(y, m, kind='zero_boundary'):
    c, n = y.shape[-2:]
    d = temporal.make_time_operator(n, kind)
    k, s = (m-m.T)/2, (m+m.T)/2
    # vec uses column-major channel stacking: flatten Y.T.
    h = torch.kron(torch.eye(n, dtype=m.dtype), k.contiguous()) + torch.kron(d.contiguous(), s.contiguous())
    transformed = (torch.matrix_exp(h) @ y.transpose(-2, -1).flatten(-2).unsqueeze(-1)).squeeze(-1)
    return transformed.reshape(*y.shape[:-2], n, c).transpose(-2, -1)


@pytest.mark.parametrize('kind', temporal.OPERATORS)
@pytest.mark.parametrize('n', (3, 7, 128))
def test_fixed_time_generator_zero_mean_skew_and_frobenius_budget(kind, n):
    d = temporal.make_time_operator(n, kind)
    torch.testing.assert_close(d+d.T, torch.zeros_like(d), atol=2e-14, rtol=0)
    torch.testing.assert_close(d @ torch.ones(n, dtype=d.dtype), torch.zeros(n, dtype=d.dtype), atol=3e-14, rtol=0)
    assert float(d.norm()) == pytest.approx(n**.5, rel=2e-15)
    d.zero_()
    assert float(temporal.make_time_operator(n, kind).norm()) == pytest.approx(n**.5)


@pytest.mark.parametrize('kind', temporal.OPERATORS)
def test_spectral_action_is_exact_dense_kronecker_without_production_dense_operator(kind):
    y, m = tensors()
    expected = dense_reference(y, m, kind)
    actual = temporal.apply_white(y, m, operator=kind)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=3e-14)


@pytest.mark.parametrize('mode', temporal.MODES[1:])
@pytest.mark.parametrize('kind', temporal.OPERATORS)
def test_energy_and_time_zero_mean_are_preserved(mode, kind):
    y, m = tensors(times=128)
    result = temporal.apply_white(y, m, mode, kind)
    torch.testing.assert_close(result.square().sum((-2, -1)), y.square().sum((-2, -1)), atol=5e-11, rtol=2e-13)
    torch.testing.assert_close(result.mean(-1), torch.zeros_like(result.mean(-1)), atol=2e-14, rtol=0)


@pytest.mark.parametrize('mode', ('static', 'rotation', 'temporal', 'spatiotemporal', 'split_symmetric'))
def test_parameter_negation_is_inverse_for_registered_reversible_modes(mode):
    y, m = tensors()
    result = temporal.apply_white(temporal.apply_white(y, m, mode), -m, mode)
    torch.testing.assert_close(result, y, atol=2e-14, rtol=2e-14)


def test_ordered_splits_have_opposite_order_inverse_not_false_same_order_claim():
    y, m = tensors()
    result = temporal.apply_white(temporal.apply_white(y, m, 'split_left'), -m, 'split_right')
    torch.testing.assert_close(result, y, atol=2e-14, rtol=2e-14)
    same = temporal.apply_white(temporal.apply_white(y, m, 'split_left'), -m, 'split_left')
    assert float((same-y).abs().max()) > 1e-4


def test_superoperator_frobenius_isometry_with_same_traceless_generator_budget():
    y, m = tensors()
    c,n=y.shape[-2:]
    d=temporal.make_time_operator(n)
    k,s=(m-m.T)/2,(m+m.T)/2
    h=torch.kron(torch.eye(n,dtype=m.dtype),k.contiguous())+torch.kron(d.contiguous(),s.contiguous())
    assert float(h.square().sum()/n)==pytest.approx(float(m.square().sum()), rel=1e-14)
    assert float((h+h.T).abs().max())<2e-15


def test_time_sign_and_symmetric_sign_cancel_exactly():
    y,m=tensors();k,s=(m-m.T)/2,(m+m.T)/2
    # D's eigenvalue sign can equivalently be carried by S. This independent
    # dense check guards the iD versus D-transpose spectral convention.
    d=temporal.make_time_operator(y.shape[-1]);n=y.shape[-1]
    h=torch.kron(torch.eye(n,dtype=m.dtype),k.contiguous())+torch.kron((-d).contiguous(),(-s).contiguous())
    expected=(torch.matrix_exp(h)@y.transpose(-2,-1).flatten(-2).unsqueeze(-1)).squeeze(-1).reshape(4,n,3).transpose(-2,-1)
    torch.testing.assert_close(temporal.apply_white(y,m),expected,atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize('case', ('pure_skew','pure_symmetric','commuting'))
def test_split_zero_effects_are_algebraic_not_empirical(case):
    y,m=tensors()
    if case=='pure_skew':m=(m-m.T)/2
    elif case=='pure_symmetric':m=(m+m.T)/2
    else:
        k=torch.zeros_like(m);k[0,1]=.1;k[1,0]=-.1
        m=k+torch.diag(torch.tensor([.05,.05,-.1],dtype=m.dtype))
    results=[temporal.apply_white(y,m,mode) for mode in ('spatiotemporal','split_left','split_right','split_symmetric')]
    for result in results[1:]:torch.testing.assert_close(result,results[0],atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize('kind', temporal.OPERATORS)
def test_common_time_transform_preserves_raw_covariance_joint_rms_and_inverse(kind):
    y,_=tensors(channels=11,times=128)
    result=temporal.common_time_shift(y,.2,kind)
    torch.testing.assert_close(result @ result.transpose(-2,-1), y @ y.transpose(-2,-1),atol=5e-12,rtol=2e-13)
    torch.testing.assert_close(result.square().mean((-2,-1)),y.square().mean((-2,-1)),atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(temporal.common_time_shift(result,-.2,kind),y,atol=2e-14,rtol=2e-14)


class TinyFrozen(torch.nn.Module):
    def __init__(self, times=7):
        super().__init__();torch.manual_seed(7702)
        self.encoder=torch.nn.Linear(11*times,6,dtype=torch.float64)
        self.source_head=torch.nn.Linear(6,2,dtype=torch.float64)
        self.feature_dim=6
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
    def features_and_logits(self,x):
        features=torch.tanh(self.encoder(x.flatten(1)))
        return features,self.source_head(features)
    def source_logits(self,x):return self.features_and_logits(joint_trial_rms(x))[1]


def source_fixture():
    torch.manual_seed(7703)
    model=TinyFrozen();factor=torch.randn(10,10,dtype=torch.float64)
    geometry=SpatialGeometry.from_reference(factor @ factor.T+torch.eye(10,dtype=torch.float64))
    x=torch.randn(4,11,7,dtype=torch.float64)
    x-=x.mean(1,keepdim=True);x-=x.mean(2,keepdim=True)
    ids=tuple('prefix-'+str(i) for i in range(len(x)))
    support=CalibrationPrefix(x,torch.arange(len(x))%2,ids,ids,'b'*64,'meta_train_support')
    b=initialize_dictionary(4,7703).requires_grad_(True)
    return model,geometry,support,b


@pytest.mark.parametrize('mode', temporal.MODES)
def test_identity_tangent_matches_actual_full_autograd_including_rms_and_recolor(mode):
    source,geometry,support,b=source_fixture()
    m0,coordinates,features=temporal.support_tangent(source,geometry,support,mode)
    def margins(theta):
        w=FastWeights(theta,theta.new_zeros(6),theta.new_zeros(()))
        logits=temporal.actual(source,geometry,b,w,support.x,mode)[0]
        return logits[:,1]-logits[:,0]
    theta=torch.zeros(4,dtype=torch.float64,requires_grad=True)
    jac=torch.autograd.functional.jacobian(margins,theta)
    torch.testing.assert_close(jac,coordinates @ b,atol=2e-13,rtol=2e-11)
    torch.testing.assert_close(margins(theta),m0,atol=2e-14,rtol=2e-14)
    assert features.shape==(4,6)
    assert not coordinates.requires_grad and not features.requires_grad and not m0.requires_grad
    assert all(not p.requires_grad for p in source.parameters())


@pytest.mark.parametrize('mode', ('temporal','spatiotemporal','split_left','split_symmetric'))
def test_complex_spectral_actual_meta_dictionary_gradient_is_nonzero_and_correct(mode):
    source,geometry,support,b=source_fixture()
    weights=FastWeights(torch.tensor([.04,-.07,.02,.01],dtype=torch.float64),
                        torch.linspace(-.1,.1,6,dtype=torch.float64),torch.tensor(.02,dtype=torch.float64))
    def value(dictionary):return temporal.actual(source,geometry,dictionary,weights,support.x,mode)[0].square().mean()
    grad=torch.autograd.grad(value(b),b)[0]
    assert torch.isfinite(grad).all() and float(grad.norm())>1e-8
    direction=torch.randn_like(b);direction/=direction.norm();eps=1e-5
    finite=(value(b+eps*direction).detach()-value(b-eps*direction).detach())/(2*eps)
    torch.testing.assert_close((grad*direction).sum(),finite,atol=3e-10,rtol=3e-6)


def test_actual_preserves_car_and_uses_same_head_at_zero():
    source,geometry,support,b=source_fixture()
    zero=FastWeights(torch.zeros(4,dtype=torch.float64),torch.zeros(6,dtype=torch.float64),torch.zeros((),dtype=torch.float64))
    before={k:v.clone() for k,v in source.state_dict().items()}
    for mode in temporal.MODES:
        logits,_=temporal.actual(source,geometry,b,zero,support.x,mode)
        torch.testing.assert_close(logits,source.source_logits(support.x),atol=2e-14,rtol=2e-14)
    white=geometry.rref_invsqrt @ geometry.q @ support.x
    mapped=geometry.q.T @ geometry.rref_sqrt @ temporal.apply_white(white,(geometry.e @ b @ torch.ones(4,dtype=b.dtype)*.03).reshape(10,10))
    torch.testing.assert_close(mapped.mean(-2),torch.zeros_like(mapped.mean(-2)),atol=3e-14,rtol=0)
    assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())


def test_invalid_modes_and_small_time_axes_fail_with_clear_configuration_error():
    y,m=tensors()
    with pytest.raises(ValueError,match='Unknown spatial/temporal'):temporal.apply_white(y,m,'unknown')
    with pytest.raises(ValueError,match='Unknown fixed'):temporal.make_time_operator(7,'unknown')
    with pytest.raises(ValueError,match='three'):temporal.make_time_operator(2)
    with pytest.raises(ValueError,match='real floating'):temporal.make_time_operator(7,dtype=torch.complex128)


def test_diagnostics_are_small_scalar_values_and_not_dense_operator_artifacts():
    _,m=tensors()
    result=temporal.diagnostics(m,7)
    assert result['time_points']==7 and result['finite_trial_noncausal']
    assert not result['uses_periodic_boundary']
    assert all(isinstance(v,(bool,int,float)) for v in result.values())
    assert result['joint_generator_scaled_frobenius_norm']==pytest.approx(float(m.norm()))


@pytest.mark.parametrize('mode', temporal.MODES)
def test_residual_identity_is_exact_for_non_car_inputs_and_has_correct_nearby_derivative(mode):
    source,geometry,support,b=source_fixture()
    # Preserve even a deliberately non-CAR component; the numerical extension
    # is identity on that component, rather than silently projecting at zero.
    support=replace(support,x=support.x + .03*torch.arange(7,dtype=torch.float64)[None,None,:])
    zero=FastWeights(torch.zeros(4,dtype=torch.float64),torch.zeros(6,dtype=torch.float64),torch.zeros((),dtype=torch.float64))
    actual=temporal.actual(source,geometry,b,zero,support.x,mode)[0]
    torch.testing.assert_close(actual,source.source_logits(support.x),atol=0,rtol=0)
    m0,coordinates,_=temporal.support_tangent(source,geometry,support,mode)
    torch.testing.assert_close(m0,actual[:,1]-actual[:,0],atol=0,rtol=0)
    def margin(theta):
        w=FastWeights(theta,zero.delta_w,zero.delta_b)
        out=temporal.actual(source,geometry,b,w,support.x,mode)[0]
        return out[:,1]-out[:,0]
    jac=torch.autograd.functional.jacobian(margin,zero.theta)
    torch.testing.assert_close(jac,coordinates @ b,atol=2e-13,rtol=2e-11)
    direction=torch.tensor([.3,-.2,.1,.4],dtype=b.dtype);eps=1e-5
    finite=(margin(eps*direction)-margin(-eps*direction))/(2*eps)
    torch.testing.assert_close(finite,(coordinates @ b) @ direction,atol=5e-11,rtol=2e-7)


@pytest.mark.parametrize('kind', temporal.OPERATORS)
def test_raw_and_white_zero_maps_are_bitwise_identity_without_parameter_branches(kind):
    x,m=tensors(times=128)
    for mode in temporal.MODES:
        assert torch.equal(temporal.apply_white(x,torch.zeros_like(m),mode,kind),x)
    assert torch.equal(temporal.common_time_shift(x,0.,kind),x)


@pytest.mark.parametrize('size',(3,10))
@pytest.mark.parametrize('norm',(.015,.05,.35,2.))
def test_stable_small_block_exponential_matches_independent_scipy(size,norm):
    from scipy.linalg import expm
    generator=torch.Generator().manual_seed(7704+size)
    a=torch.randn(size,size,generator=generator,dtype=torch.float64)
    a=(a-a.T)/2;a=a*(norm/a.norm())
    real=temporal._matrix_exp(a)
    expected=torch.from_numpy(expm(a.numpy()))
    torch.testing.assert_close(real,expected,atol=3e-14,rtol=5e-14)
    assert float((real.T @real-torch.eye(size,dtype=a.dtype)).norm())<3e-14
