"""Full first meta derivatives through actual R7 support adaptation."""
import pytest
import torch

from test_algorithm import fixture
from paper3r7 import algorithm as alg


@pytest.mark.parametrize('mode,objective,policy',[
    ('static','ce','full'),
    ('spatiotemporal','ce','full'),
    ('spatiotemporal','relative_dro','full'),
    ('spatiotemporal','relative_dro','budget'),
    ('static','absolute_dro','budget'),
    ('spatiotemporal','variance_regret','fixed'),
])
def test_complete_meta_dictionary_gradient_matches_directional_difference(mode,objective,policy):
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',mode=mode,objective=objective,head_policy=policy,
                     head_l2=.1,tangent_steps=2,inner_steps=2)
    def value(dictionary):
        fitted = alg.fit(source,g,dictionary,s,cfg,create_graph=True)
        logits = alg.predict(source,g,dictionary,fitted,q)
        return -logits.log_softmax(1)[torch.arange(len(q)),torch.arange(len(q))%2].mean()
    gradient = torch.autograd.grad(value(b),b)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>1e-10
    torch.manual_seed(837)
    direction = torch.randn_like(b);direction /= direction.norm()
    eps = 2e-5
    numerical = (value(b+eps*direction).detach()-value(b-eps*direction).detach())/(2*eps)
    torch.testing.assert_close((gradient*direction).sum(),numerical,atol=2e-7,rtol=5e-3)


def test_global_time_identity_tangent_matches_actual_finite_difference():
    source,g,s,b,q = fixture()
    cfg = alg.Config(variant='r7_refined',mode='global_time')
    m0,j,_ = alg._global_tangent(source,g,s,cfg.rank,cfg.time_operator)
    theta = b.new_zeros(4,requires_grad=True)
    def margins(t):
        weights = alg.core.FastWeights(t,t.new_zeros(source.feature_dim),t.new_zeros(()))
        logits = alg.actual_r7(source,g,b,weights,s.x,cfg)[0]
        return logits[:,1]-logits[:,0]
    derivative = torch.autograd.functional.jacobian(margins,theta)
    torch.testing.assert_close(derivative,j,atol=2e-12,rtol=2e-9)
    assert torch.equal(derivative[:,1:],derivative[:,1:]*0)
    torch.testing.assert_close(margins(theta),m0,atol=2e-12,rtol=2e-10)


def test_actual_float32_frozen_encoder_supports_double_spatiotemporal_meta_gradient():
    source,g,s,b,q = fixture(double=False)
    cfg = alg.Config(variant='r7_refined',mode='spatiotemporal',
                     objective='relative_dro',head_policy='budget',
                     head_l2=.1,tangent_steps=1,inner_steps=1)
    fitted = alg.fit(source,g,b,s,cfg,create_graph=True)
    prediction = alg.predict(source,g,b,fitted,q)
    loss = -prediction.log_softmax(1)[torch.arange(len(q)),torch.arange(len(q))%2].mean()
    gradient = torch.autograd.grad(loss,b)[0]
    assert prediction.dtype == fitted.weights.delta_w.dtype == torch.float64
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert fitted.diagnostics['support_certificate_holds']
