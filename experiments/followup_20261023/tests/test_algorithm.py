from types import SimpleNamespace
import pytest
import torch
from p3work.adapt import CalibrationPrefix, FrozenSourceModel, InnerConfig, inner_unroll, adapted_logits
from p3work.f0_bridge import CompactEEGNet
from p3work.spatial_geometry import SpatialGeometry, initialize_dictionary, project_ball
from paper3v2.algorithm import Config, VARIANTS, fit, logits, init_gate, outer_update, gated_matrix, teacher_weights, temporal_loss
from paper3v2.data import FrozenSource

def fixture():
    torch.manual_seed(18)
    model = CompactEEGNet(11, 128)
    native = FrozenSourceModel(torch.nn.Sequential(model.features, torch.nn.Flatten(1)), model.classifier,
                               feature_dim=64, input_layout="N1CT")
    source = FrozenSource(model)
    geometry = SpatialGeometry.from_reference(torch.eye(10, dtype=torch.float64) / 10)
    x = torch.randn(40, 11, 128, dtype=torch.float64)
    x -= x.mean(1, keepdim=True); x -= x.mean(2, keepdim=True)
    ids = tuple(f"s{i}" for i in range(20))
    support = CalibrationPrefix(x[:20], torch.arange(20) % 2, ids, ids, "a"*64, "meta_train_support")
    e = SimpleNamespace(support=support, query_x=x[20:], query_y=torch.arange(20)%2,
                        run_ids=tuple("run1" if i < 10 else "run2" for i in range(20)))
    b = initialize_dictionary(4, 20261018).requires_grad_(True)
    return native, source, geometry, e, b

@pytest.mark.parametrize("variant,pct", [("P", 0.), ("PCT", .1)])
def test_reproduces_native_calibration(variant, pct):
    native, source, geometry, e, b = fixture()
    expected, _, _ = inner_unroll(native, geometry, b, e.support, config=InnerConfig(pct_weight=pct))
    actual, alpha, _ = fit(source, geometry, b, init_gate(Config(variant=variant), "cpu"), e.support, Config(variant=variant))
    for name in ("theta", "delta_w", "delta_b"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(logits(source, geometry, b, actual, alpha, e.query_x),
        adapted_logits(native, geometry, b, e.query_x, expected), atol=1e-7, rtol=1e-6)

@pytest.mark.parametrize("variant", VARIANTS)
def test_full_unroll_finite_and_source_frozen(variant):
    _, source, geometry, e, b = fixture()
    prior = {k: v.clone() for k,v in source.state_dict().items()}
    cfg = Config(variant=variant)
    gate = init_gate(cfg, "cpu")
    nb, ng, _ = outer_update(source, geometry, b, gate, [e], cfg)
    assert torch.isfinite(nb).all() and torch.isfinite(ng).all()
    torch.testing.assert_close(nb.T @ nb, torch.eye(4, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    for k, v in prior.items():
        assert torch.equal(v, source.state_dict()[k])

def test_geometry_bound_volume_and_basis_invariance():
    _, _, geometry, _, b = fixture()
    theta = project_ball(torch.randn(4, dtype=torch.float64), .15)
    a = torch.tensor([.2, .8], dtype=torch.float64)
    m = gated_matrix(geometry, b, theta, a)
    assert float(m.norm().detach()) <= .15 + 1e-12
    assert abs(float(m.trace().detach())) < 1e-12
    assert abs(float(torch.det(torch.matrix_exp(m)).detach()) - 1) < 1e-12
    o, _ = torch.linalg.qr(torch.randn(4,4,dtype=torch.float64))
    torch.testing.assert_close(m, gated_matrix(geometry, b @ o, o.T @ theta, a), atol=1e-12, rtol=1e-12)

def test_gate_gradient_reaches_full_inner_adaptation():
    _, source, geometry, e, b = fixture()
    cfg = Config(variant="dual"); gate = init_gate(cfg, "cpu")
    weights, alpha, _ = fit(source, geometry, b, gate, e.support, cfg, create_graph=True)
    loss = logits(source, geometry, b, weights, alpha, e.query_x).square().mean()
    gb, gg = torch.autograd.grad(loss, (b, gate))
    assert gb.norm() > 0 and gg.norm() > 0

def test_bad_confident_teacher_is_not_given_high_trust():
    y = torch.tensor([0,0,1,1])
    correct = torch.tensor([[5.,-5.],[5.,-5.],[-5.,5.],[-5.,5.]])
    assert (teacher_weights(correct,y,soft=True) > teacher_weights(-correct,y,soft=True)).all()

def test_temporal_risk_emphasizes_worst_run_and_falls_back():
    pred = torch.tensor([[2.,-2.],[-2.,2.],[-2.,2.],[2.,-2.]])
    y = torch.tensor([0,1,0,1]); cfg = Config(variant="temporal")
    from p3work.adapt import balanced_bce
    assert temporal_loss(pred,y,("a","a","b","b"),cfg) > balanced_bce(pred,y)
    assert temporal_loss(pred,y,("a",)*4,cfg) == balanced_bce(pred,y)
