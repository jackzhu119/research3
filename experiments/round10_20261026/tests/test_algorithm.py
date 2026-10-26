import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from paper3r10 import algorithm as alg


def signal(config=None):
    x = np.column_stack((np.array([-2., -1.5, -1., -.5, .5, 1., 1.5, 2.]), np.zeros(8)))
    y = (x[:,0] > 0).astype(int)
    config = config or alg.Config(amplitude=2., include_intercept=False)
    return alg.fit(x, np.zeros(len(x)), y, config)


def test_balanced_objective_gradient_and_hessian_finite_differences():
    rng = np.random.default_rng(12)
    b = rng.normal(size=(5,5))*.2
    obj = alg._Objective(b, rng.normal(size=5), np.array([0,0,0,1,1]),
                         np.array([5/6]*3+[5/4]*2), .7)
    v, tilt, epsilon = rng.normal(size=5), rng.normal(size=5)*.3, 1e-5
    _, gradient, hessian = obj.evaluate(v, tilt, True)
    numerical_g, numerical_h = [], []
    for i in range(5):
        change = np.eye(5)[i]*epsilon
        numerical_g.append((obj.evaluate(v+change,tilt)[0]-obj.evaluate(v-change,tilt)[0])/(2*epsilon))
        numerical_h.append((obj.evaluate(v+change,tilt)[1]-obj.evaluate(v-change,tilt)[1])/(2*epsilon))
    np.testing.assert_allclose(gradient,numerical_g,atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(hessian,np.array(numerical_h).T,atol=1e-9,rtol=1e-8)
    assert np.linalg.eigvalsh(hessian).min() >= .7-1e-12


def test_fit_is_monotone_feasible_and_weights_sum_to_n():
    x = np.arange(20,dtype=float).reshape(10,2)/10
    state = alg.fit(x,np.zeros(10),[0]*7+[1]*3)
    assert state.status == "FIT_COMPLETE"
    assert state.objective_at_fit <= state.objective_at_zero
    assert state.gain == state.objective_at_zero-state.objective_at_fit
    assert state.threshold == state.objective_at_fit+.5*state.gain
    assert state.weights.sum() == pytest.approx(10.)
    np.testing.assert_allclose(state.weights[:7],10/14)
    np.testing.assert_allclose(state.weights[7:],10/6)
    assert state.diagnostics["all_spd_directions_retained"]


def test_actual_newton_residual_makes_global_tilt_lower_bound():
    state = signal()
    objective = state._objective()
    h = np.arange(8,dtype=float)/8
    config = dataclasses.replace(state.config,max_newton=1)
    solution = alg._newton(objective,config,tilt=h)
    assert not solution["converged"]
    rng = np.random.default_rng(44)
    values = [objective.evaluate(rng.normal(size=8)*2,h)[0] for _ in range(100)]
    assert min(values) >= solution["lower_bound"]-1e-12


def test_same_support_strong_identifiable_query_gets_nonzero_update():
    state = signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.05))
    p,diag = alg.predict(state,[[1.,0]],np.zeros(1))
    assert .5 < p[0,1] < 1
    assert diag["queries"][0]["reason"] == "certified_nonzero_update"
    assert diag["queries"][0]["correction"] <= diag["queries"][0]["map_correction"]


def test_null_completion_proves_zero_despite_nonzero_map_and_kernel_cross():
    state = signal()
    p,diag = alg.predict(state,[[.2,20.]],np.zeros(1))
    row = diag["queries"][0]
    assert row["map_correction"] > 0 and row["support_projection_energy"] > 0
    assert row["kernel_innovation"] > 0
    assert row["reason"] == "support_null_completion_witness"
    np.testing.assert_array_equal(p,alg.anchor_probabilities(np.zeros(1)))
    assert row["null_witness_cost"] <= state.config.evidence_fraction*state.gain


def test_null_completion_ablation_is_a_different_declared_function_class():
    state = signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.05,
                                      use_null_completion=False))
    p,diag = alg.predict(state,[[.2,20.]],np.zeros(1))
    assert p[0,1] > .5
    assert diag["queries"][0]["declared_profile_innovation"] == 0


def test_zero_kernel_cross_returns_exact_anchor():
    state = signal()
    logits = np.array([[-.4,.7]])
    p,diag = alg.predict(state,[[0,1]],logits)
    np.testing.assert_array_equal(p,alg.anchor_probabilities(logits))
    assert diag["queries"][0]["reason"] == "zero_query_cross_information"


@pytest.mark.parametrize("kernel",["linear","rbf"])
def test_fraction_one_is_exact_anchor_and_zero_is_declared_map_ablation(kernel):
    base=alg.Config(amplitude=2.,kernel=kernel)
    one=signal(dataclasses.replace(base,evidence_fraction=1.))
    zero=signal(dataclasses.replace(base,evidence_fraction=0.))
    query=np.array([[1.,0],[-1.,0],[0,3.]])
    logits=np.array([-.2,.3,.1])
    p,d=alg.predict(one,query,logits)
    np.testing.assert_array_equal(p,alg.anchor_probabilities(logits))
    p0,d0=alg.predict(zero,query,logits)
    assert all(r["certificate"]=="not_profile_certified" for r in d0["queries"])
    assert np.max(np.abs(p0-alg.anchor_probabilities(logits))) > .01


def test_more_ambiguity_never_increases_query_update_with_solved_certificates():
    query=np.array([[.5,0],[1,0],[-1,0],[.3,.7],[.2,10]])
    deviations=[]
    for fraction in (.02,.1,.3,.6,.9,1.):
        state=signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=fraction))
        p,_=alg.predict(state,query,np.zeros(5))
        deviations.append(abs(p[:,1]-.5))
    assert (np.diff(deviations,axis=0) <= 2e-7).all()


def test_innovation_increase_reduces_or_removes_update():
    state=signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.1))
    queries=np.column_stack((np.ones(5),[0,.1,.3,1,10]))
    p,_=alg.predict(state,queries,np.zeros(5))
    assert (np.diff(abs(p[:,1]-.5)) <= 2e-7).all()
    assert p[-1,1] == .5


def test_dual_outer_bound_dominates_dense_primal_grid_including_completion():
    cfg=alg.Config(amplitude=1.,regularization=.7,evidence_fraction=.3,include_intercept=False)
    state=alg.fit([[-1.,0],[1.,0]],np.zeros(2),[0,1],cfg)
    q=np.array([[1.,.4]])
    cross=alg._kernel(state.features,q,cfg,state.reference)
    a=np.linalg.solve(state.factor,cross)[:,0]
    innovation=float(alg._kernel_diagonal(q,cfg)[0]-a@a)
    objective=state._objective()
    radius=np.sqrt(2*state.threshold/cfg.regularization)
    axis=np.linspace(-radius,radius,501)
    grid=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    logits=state.margins[None,:]+grid@state.factor.T
    costs=(state.weights[None,:]*(np.logaddexp(0,logits)-state.labels[None,:]*logits)).sum(1)
    costs+=.5*cfg.regularization*(grid*grid).sum(1)
    eligible=costs<=state.threshold
    candidates=grid[eligible]
    free=np.sqrt(2*innovation*(state.threshold-costs[eligible])/cfg.regularization)
    for sign in (-1,1):
        attained=(sign*candidates@a+free).max()
        certified=alg._direction_bound(state,sign*a,innovation)
        assert certified["converged"]
        assert attained <= certified["upper"]+1e-9
        assert certified["upper"]-attained < .025


def test_minimax_logloss_relative_anchor_on_entire_declared_interval():
    state=signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.1))
    p,diag=alg.predict(state,[[1.,0]],np.array([.3]))
    row=diag["queries"][0]
    anchor=alg.anchor_probabilities(np.array([.3]))[0,1]
    # Certified lower endpoint is below every compatible eta. It is enough to
    # check that endpoint and 1 because relative CE is affine in eta.
    lower=p[0,1]
    assert row["correction"] > 0 and lower > anchor
    for eta in np.linspace(lower,1,101):
        regret=-eta*np.log(lower/anchor)-(1-eta)*np.log((1-lower)/(1-anchor))
        assert regret <= 1e-12


def test_inconsistent_dual_certificate_must_anchor_instead_of_clipping_to_map(monkeypatch):
    state=signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.01))
    monkeypatch.setattr(alg,"_direction_bound",lambda *args:{"upper":-1e6,"converged":True})
    p,d=alg.predict(state,[[1.,0]],np.zeros(1))
    np.testing.assert_array_equal(p,alg.anchor_probabilities(np.zeros(1)))
    assert d["queries"][0]["reason"] == "inconsistent_dual_bound"


def test_projection_failure_anchors_entire_batch(monkeypatch):
    state=signal()
    monkeypatch.setattr(np.linalg,"solve",lambda *a: (_ for _ in ()).throw(np.linalg.LinAlgError("fixture")))
    p,d=alg.predict(state,[[1.,0],[-1.,0]],np.zeros(2))
    np.testing.assert_array_equal(p,alg.anchor_probabilities(np.zeros(2)))
    assert d["summary"]["nonzero_updates"] == 0
    assert all(r["reason"].startswith("query_projection_failure") for r in d["queries"])


@pytest.mark.parametrize("reason,x,y",[
    ("empty_support",np.empty((0,2)),[]),
    ("insufficient_support",[[1,0]],[0]),
    ("one_class_support",[[1,0],[2,0]],[1,1]),
])
def test_degenerate_support_returns_anchor(reason,x,y):
    state=alg.fit(x,np.zeros(len(x)),y)
    assert state.fallback_reason == reason
    p,d=alg.predict(state,[[1,2]],np.array([.4]))
    np.testing.assert_array_equal(p,alg.anchor_probabilities(np.array([.4])))


def test_duplicate_features_with_unique_uids_retain_every_spd_direction():
    state=alg.fit(np.ones((6,3)),np.zeros(6),[0,1,0,1,0,1],support_ids=[str(i) for i in range(6)])
    assert state.status == "FIT_COMPLETE"
    assert state.factor.shape == (6,6)
    assert state.diagnostics["minimum_kernel_eigenvalue"] > 0
    assert not state.features.flags.writeable


def test_zero_amplitude_uses_no_uid_nugget_scientific_update():
    state=signal(alg.Config(amplitude=0))
    assert state.fallback_reason == "zero_kernel_amplitude"
    p,_=alg.predict(state,[[1,0]],np.zeros(1))
    assert p[0,1] == .5


def test_queries_have_no_label_parameter_and_batch_composition_does_not_fit():
    assert "query_labels" not in inspect.signature(alg.predict).parameters
    state=signal()
    with pytest.raises(TypeError):alg.predict(state,[[1,0]],np.zeros(1),query_labels=[1])
    p,_=alg.predict(state,[[1,0]],np.zeros(1))
    both,_=alg.predict(state,[[1,0],[100,-100]],np.zeros(2))
    np.testing.assert_allclose(p[0],both[0],atol=2e-14,rtol=0)


def test_uids_overlap_and_bad_reference_roles_are_rejected():
    state=signal()
    with pytest.raises(ValueError,match="overlap"):
        alg.predict(state,[[1,0]],np.zeros(1),query_ids=[state.support_ids[0]])
    with pytest.raises(ValueError,match="geometry"):
        alg.fit([[0,1],[1,0]],np.zeros(2),[0,1],reference={
            "mean":[0,0],"whitener":np.eye(2),"provenance":{"role":"internal_holdout"}})


def test_extreme_finite_logits_and_query_kernel_overflow_return_finite_anchor():
    logits=np.array([[-1e308,1e308],[1e308,-1e308],[0,0]])
    p=alg.anchor_probabilities(logits)
    assert np.isfinite(p).all() and (p>0).all() and (p<1).all()
    state=signal()
    p,d=alg.predict(state,[[1e300,1e300]],np.zeros(1))
    assert p[0,1] == .5
    assert d["summary"]["nonzero_updates"] == 0


def test_overflow_query_does_not_change_another_querys_prediction():
    state=signal(dataclasses.replace(alg.Config(),amplitude=2.,include_intercept=False,evidence_fraction=.05))
    single,_=alg.predict(state,[[1.,0]],np.zeros(1))
    combined,d=alg.predict(state,[[1.,0],[1e300,1e300]],np.zeros(2))
    np.testing.assert_array_equal(single[0],combined[0])
    assert combined[1,1] == .5
    assert d["queries"][1]["reason"] == "query_geometry_or_kernel_overflow"


def test_support_whitening_overflow_is_explicit_reproducible_anchor_state(tmp_path):
    reference={"mean":[0.,0.],"whitener":np.eye(2)*1e300,
               "provenance":{"role":"meta_train"}}
    state=alg.fit([[1e300,0],[-1e300,0]],np.zeros(2),[1,0],reference=reference)
    assert state.fallback_reason.startswith("support_geometry_failure")
    alg.save_state(state,tmp_path/'fallback.npz')
    restored=alg.load_state(tmp_path/'fallback.npz')
    p,_=alg.predict(restored,[[1.,1.]],np.zeros(1))
    assert p[0,1] == .5


def test_serialization_exact_predictions_and_identity_then_derived_corruption_rejected(tmp_path):
    state=signal()
    path=tmp_path/'fit.npz'
    alg.save_state(state,path)
    restored=alg.load_state(path,expected_fit_identity=state.fit_identity)
    q=np.array([[1,0],[-1,0],[.2,20]])
    p,_=alg.predict(state,q,np.zeros(3))
    p2,_=alg.predict(restored,q,np.zeros(3))
    np.testing.assert_array_equal(p,p2)
    with pytest.raises(ValueError,match="identity"):
        alg.load_state(path,expected_fit_identity='wrong')
    with np.load(path,allow_pickle=False) as a:
        contents={k:np.array(a[k],copy=True) for k in a.files}
    contents['map_parameter'][0]+=.01
    with path.open('wb') as f:np.savez_compressed(f,**contents)
    with pytest.raises(ValueError,match="derived"):
        alg.load_state(path)


@pytest.mark.parametrize("kwargs",[
    {"regularization":0},{"nugget":0},{"evidence_fraction":1.1},
    {"max_support":21},{"max_newton":0},{"probability_floor":0},{"kernel":"unknown"},
])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):alg.Config(**kwargs).validate()
