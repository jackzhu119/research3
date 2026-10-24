"""Full first meta derivatives through center, shape and actual averaging."""
from dataclasses import replace
import pytest
import torch

from test_algorithm import fixture
from paper3r6 import algorithm as alg


@pytest.mark.parametrize('variant',['uq_joint','uq_spatial','uq_mean_head','uq_isotropic','uq_independent','uq_linear'])
def test_actual_outer_gradient_matches_directional_difference(variant):
    source,geometry,support,b,query=fixture()
    cfg=alg.Config(variant=variant,head_l2=.1,tangent_steps=2,radius=.15)
    def value(dictionary):
        result=alg.fit(source,geometry,dictionary,support,cfg,create_graph=True)
        logits=alg.predict(source,geometry,dictionary,result,query)
        return -logits.log_softmax(1)[torch.arange(len(query)),torch.arange(len(query))%2].mean()
    gradient=torch.autograd.grad(value(b),b)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm())>1e-9
    direction=torch.randn_like(b);direction/=direction.norm()
    epsilon=2e-5
    difference=(value(b+epsilon*direction).detach()-value(b-epsilon*direction).detach())/(2*epsilon)
    torch.testing.assert_close((gradient*direction).sum(),difference,atol=8e-8,rtol=5e-3)


def test_center_and_stencil_state_are_durable_torch_weights_only(tmp_path):
    source,geometry,support,b,query=fixture()
    cfg=alg.Config(variant='uq_joint',head_l2=.1,tangent_steps=2)
    answer=alg.fit(source,geometry,b,support,cfg)
    path=tmp_path/'uq-tensor-state.pt'
    names=('sigma_theta','sigma_heads','point_weights','head_inverse','spatial_covariance','head_compensation','temperature')
    torch.save({name:getattr(answer,name) for name in names},path)
    restored=torch.load(path,weights_only=True)
    for name in names:
        torch.testing.assert_close(restored[name],getattr(answer,name),atol=0,rtol=0)
    clone=replace(answer,**restored)
    torch.testing.assert_close(alg.predict(source,geometry,b,clone,query),
                               alg.predict(source,geometry,b,answer,query),atol=0,rtol=0)
