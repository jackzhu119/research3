"""Exact map geometry on synthetic matrices; no EEG efficacy claims."""
import json

import pytest
import torch

from paper3r6.maps import split_generator, matrix_map, transform_diagnostics


CHARTS = ('single', 'left_rotation', 'right_rotation', 'symmetric', 'bch2')


def generator(scale=.4):
    random = torch.Generator(device='cpu').manual_seed(819)
    value = torch.randn(10, 10, generator=random, dtype=torch.float64)
    value = value - value.trace() * torch.eye(10, dtype=value.dtype) / 10
    return value * (scale / value.norm())


def test_split_is_unique_orthogonal_skew_plus_symmetric_traceless_decomposition():
    m = generator()
    k, s = split_generator(m)
    torch.testing.assert_close(k+s, m, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(k.T, -k, atol=1e-15, rtol=0)
    torch.testing.assert_close(s.T, s, atol=1e-15, rtol=0)
    assert abs(float(s.trace())) < 1e-15
    assert abs(float((k*s).sum())) < 1e-15
    torch.testing.assert_close(m.square().sum(), k.square().sum()+s.square().sum(), atol=1e-15, rtol=1e-14)
    commutator = k@s-s@k
    torch.testing.assert_close(commutator.T, commutator, atol=1e-15, rtol=0)
    assert abs(float(commutator.trace())) < 1e-15


@pytest.mark.parametrize('chart', CHARTS)
def test_each_map_has_identity_origin_same_first_tangent_and_unit_determinant(chart):
    m = generator()
    eye = torch.eye(10, dtype=m.dtype)
    torch.testing.assert_close(matrix_map(m*0, chart), eye, atol=1e-15, rtol=0)
    _, derivative = torch.autograd.functional.jvp(lambda x: matrix_map(x, chart), m*0, m)
    torch.testing.assert_close(derivative, m, atol=2e-14, rtol=2e-13)
    transform = matrix_map(m, chart)
    torch.testing.assert_close(torch.linalg.det(transform), m.new_tensor(1.), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('kind', ['pure_skew', 'pure_symmetric', 'commuting_nonzero'])
def test_all_charts_agree_when_the_commutator_vanishes(kind):
    k, s = split_generator(generator())
    if kind == 'pure_skew':
        m = k
    elif kind == 'pure_symmetric':
        m = s
    else:
        k = torch.zeros(10,10,dtype=torch.float64)
        k[0,1], k[1,0] = .21, -.21
        diagonal = torch.linspace(-.2,.3,10,dtype=torch.float64)
        diagonal[:2] = diagonal[:2].mean()
        diagonal -= diagonal.mean()
        s = torch.diag(diagonal)
        m = k+s
        torch.testing.assert_close(k@s, s@k, atol=1e-16, rtol=1e-15)
    expected = matrix_map(m, 'single')
    for chart in CHARTS[1:]:
        torch.testing.assert_close(matrix_map(m, chart), expected, atol=2e-13, rtol=2e-13)


def test_same_parameter_splits_are_orthogonally_similar_with_identical_spectra_and_deviation():
    m = generator(.7)
    k, s = split_generator(m)
    left = matrix_map(m, 'left_rotation')
    eye = torch.eye(10,dtype=m.dtype)
    expected_sv = torch.linalg.eigvalsh(s).exp().sort(descending=True).values
    for chart, tau in [('left_rotation',1.), ('right_rotation',0.), ('symmetric',.5)]:
        actual = matrix_map(m,chart)
        conjugator = torch.matrix_exp((tau-1)*k)
        torch.testing.assert_close(conjugator.T@conjugator,eye,atol=2e-14,rtol=2e-14)
        torch.testing.assert_close(actual,conjugator@left@conjugator.T,atol=2e-13,rtol=2e-13)
        sv = torch.linalg.svdvals(actual)
        torch.testing.assert_close(sv,expected_sv,atol=2e-13,rtol=2e-13)
        torch.testing.assert_close(sv[0]/sv[-1],torch.linalg.svdvals(left)[0]/torch.linalg.svdvals(left)[-1],atol=3e-13,rtol=3e-13)
        for order in ('fro',2,'nuc'):
            torch.testing.assert_close(torch.linalg.matrix_norm(actual-eye,ord=order),
                torch.linalg.matrix_norm(left-eye,ord=order),atol=2e-13,rtol=2e-13)
        u,_,vh = torch.linalg.svd(actual)
        torch.testing.assert_close(u@vh,torch.matrix_exp(k),atol=3e-13,rtol=3e-13)


def test_symmetric_and_single_are_sign_reversible_and_ordered_maps_swap_on_inverse():
    m=generator(.7)
    eye=torch.eye(10,dtype=m.dtype)
    for chart in ('single','symmetric'):
        torch.testing.assert_close(matrix_map(-m,chart)@matrix_map(m,chart),eye,atol=3e-13,rtol=3e-13)
    torch.testing.assert_close(matrix_map(-m,'left_rotation')@matrix_map(m,'right_rotation'),eye,atol=3e-13,rtol=3e-13)
    torch.testing.assert_close(matrix_map(-m,'right_rotation')@matrix_map(m,'left_rotation'),eye,atol=3e-13,rtol=3e-13)
    assert float((matrix_map(-m,'left_rotation')@matrix_map(m,'left_rotation')-eye).norm()) > 1e-4


def test_bch_second_order_term_and_symmetric_third_order_error_have_expected_scaling():
    m=generator(1.)
    k,s=split_generator(m)
    commutator=k@s-s@k
    ordered_errors, symmetric_errors, bch_errors = [],[],[]
    for strength in (.16,.08,.04):
        x=m*strength
        joint=matrix_map(x,'single')
        left=matrix_map(x,'left_rotation')
        right=matrix_map(x,'right_rotation')
        ordered_errors.append(float((left-joint).norm()))
        symmetric_errors.append(float((matrix_map(x,'symmetric')-joint).norm()))
        bch_errors.append(float((matrix_map(x,'bch2')-left).norm()))
        if strength==.04:
            torch.testing.assert_close((left-joint)/(strength**2), .5*commutator,atol=1.5e-3,rtol=1.5e-2)
            torch.testing.assert_close((left-right)/(strength**2),commutator,atol=1.5e-3,rtol=1.5e-2)
    for a,b in zip(ordered_errors,ordered_errors[1:]):
        assert 3.6 < a/b < 4.5
    for errors in (symmetric_errors,bch_errors):
        for a,b in zip(errors,errors[1:]):
            assert 7.0 < a/b < 9.0


def test_white_equal_magnitude_does_not_imply_equal_car_euclidean_magnitude():
    m=generator(.7)
    root=torch.diag(torch.logspace(-1,1,10,dtype=m.dtype))
    inverse=torch.linalg.inv(root)
    eye=torch.eye(10,dtype=m.dtype)
    values=[]
    for chart in ('left_rotation','right_rotation','symmetric'):
        t=matrix_map(m,chart)
        values.append(float((root@t@inverse-eye).norm()))
    assert max(values)-min(values)>1e-2
    white=[float((matrix_map(m,c)-eye).norm()) for c in ('left_rotation','right_rotation','symmetric')]
    assert max(white)-min(white)<1e-12


@pytest.mark.parametrize('chart',CHARTS)
def test_map_first_and_second_derivatives_are_finite_and_match_finite_difference(chart):
    random=torch.Generator().manual_seed(831)
    basis=torch.randn(4,10,10,generator=random,dtype=torch.float64)
    basis-=basis.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]*torch.eye(10,dtype=basis.dtype)/10
    basis/=basis.norm(dim=(-2,-1),keepdim=True)
    x=torch.randn(10,3,generator=random,dtype=torch.float64)
    theta=torch.tensor([.1,-.05,.07,.03],dtype=torch.float64,requires_grad=True)
    def objective(parameters):
        m=torch.einsum('r,rij->ij',parameters,basis)
        return (matrix_map(m,chart)@x).sin().square().sum()
    assert torch.autograd.gradcheck(objective,(theta,),eps=1e-6,atol=1e-6,rtol=1e-5)
    assert torch.autograd.gradgradcheck(objective,(theta,),eps=1e-6,atol=2e-6,rtol=2e-5)


def test_diagnostics_are_json_finite_and_keep_white_and_car_metrics_distinct():
    m=generator(.5)
    root=torch.diag(torch.logspace(-1,1,10,dtype=m.dtype))
    inverse=torch.linalg.inv(root)
    out=transform_diagnostics(m,'left_rotation',rref_sqrt=root,rref_invsqrt=inverse)
    json.dumps(out,allow_nan=False)
    t=matrix_map(m,'left_rotation');eye=torch.eye(10,dtype=m.dtype)
    k,s=split_generator(m)
    assert out['white_transform_deviation']==pytest.approx(float((t-eye).norm()),rel=1e-12)
    assert out['car_transform_deviation']==pytest.approx(float((root@t@inverse-eye).norm()),rel=1e-12)
    sv=torch.linalg.svdvals(t)
    assert out['white_condition']==pytest.approx(float(sv[0]/sv[-1]),rel=1e-12)
    assert out['commutator_norm']==pytest.approx(float((k@s-s@k).norm()),rel=1e-12)
