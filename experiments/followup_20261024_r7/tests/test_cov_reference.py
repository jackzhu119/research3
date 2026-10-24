"""Covariance classifier is exactly blind to the declared time-orthogonal shift."""
from types import SimpleNamespace
import torch
import pytest
from paper3r7.cov_reference import covariance_features, fit_covariance, covariance_logits
from paper3r7.temporal import common_time_shift
from p3work.spatial_geometry import SpatialGeometry


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    generator = torch.Generator().manual_seed(712)
    x = torch.randn(30, 11, 32, generator=generator, dtype=torch.float64)
    x -= x.mean(1, keepdim=True); x -= x.mean(2, keepdim=True)
    geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64)/10)
    return geometry, x, torch.arange(30)%2


def test_whole_fit_and_predictions_invariant_under_common_time_orthogonal_action():
    geometry,x,y=fixture(); shifted=common_time_shift(x, .35/(10**.5))
    torch.testing.assert_close(covariance_features(geometry,x),covariance_features(geometry,shifted),atol=1e-12,rtol=1e-12)
    first,meta=fit_covariance(geometry,SimpleNamespace(x=x[:20],y=y[:20]))
    second,_=fit_covariance(geometry,SimpleNamespace(x=shifted[:20],y=y[:20]))
    a,b=covariance_logits(geometry,first,x[20:]),covariance_logits(geometry,second,shifted[20:])
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    assert torch.equal(a.argmax(1),b.argmax(1))
    assert meta['feature_family']=='zero_lag_covariance_only'


def test_single_class_fallback_remains_covariance_only_and_input_independent():
    geometry,x,y=fixture();y[:20]=1
    state,meta=fit_covariance(geometry,SimpleNamespace(x=x[:20],y=y[:20]))
    assert meta['fallback_reason']=='single_class_constant_probability'
    assert torch.equal(covariance_logits(geometry,state,x[:3]),covariance_logits(geometry,state,x[4:7]*9))


def test_covariance_readout_is_positive_scale_invariant_and_rejects_zero_energy():
    geometry,x,_=fixture()
    torch.testing.assert_close(covariance_features(geometry,x),covariance_features(geometry,x*3),atol=1e-12,rtol=1e-12)
    with pytest.raises(FloatingPointError):covariance_features(geometry,torch.zeros_like(x))
