"""Independent PFST mathematics and software verification; synthetic CPU only."""
from dataclasses import asdict,replace
from itertools import combinations
import inspect
import json

import numpy as np
import pytest
from paper3r11 import algorithm as a


def reference(d=2,**extra):
    return dict(mean=np.zeros(d),whitener=np.eye(d),eigenvectors=np.eye(d),role='meta_train',**extra)


def rotation_pairs():
    support=np.array([[-1.,0.],[-1.,0.],[1.,0.],[1.,0.]])
    future=np.array([[0.,-1.],[0.,-1.],[0.,1.],[0.,1.]])
    return [dict(support_features=support,support_logits=np.zeros((4,2)),support_y=np.array([0,0,1,1]),
       future_features=future,future_logits=np.zeros((4,2)),future_y=np.array([0,0,1,1]),
       participant_id='train1',session_id='s1',training_role='meta_train',
       support_ids=['s0','s1','s2','s3'],future_ids=['q0','q1','q2','q3'])]


def cfg(**kwargs):return a.Config(**dict(feature_rank=2,include_intercept=False,ridge=.03,**kwargs))


def train(config=None):return a.train_operator(rotation_pairs(),config or cfg(),reference=reference())


@pytest.mark.parametrize('n,n1',[(4,2),(6,2),(6,3)])
def test_exact_conditional_permutation_covariance_by_full_enumeration(n,n1):
    rng=np.random.default_rng(114+n);phi=rng.normal(size=(n,3));z=rng.normal(size=n)
    scores=[];base=np.zeros(n,dtype=int);base[:n1]=1
    for ones in combinations(range(n),n1):
        y=np.zeros(n,dtype=int);y[list(ones)]=1
        g,noise,_=a.support_moments(phi,z,y,cfg())
        scores.append(g)
    scores=np.array(scores);centered=scores-scores.mean(0)
    observed=centered.T@centered/len(scores)
    np.testing.assert_allclose(noise,observed,rtol=2e-13,atol=2e-15)
    assert np.linalg.eigvalsh(noise).min()>-1e-12


def test_support_score_is_actual_balanced_logistic_negative_gradient():
    phi=np.array([[1.,0.],[-1.,.1],[.2,1.],[.5,-.4]])
    z=np.array([.2,.7,-.8,.1]);y=np.array([0,0,0,1]);v=np.array([1/6,1/6,1/6,1/2]);eps=1e-6
    g,_,_=a.support_moments(phi,z,y,cfg())
    def f(theta):
        logits=z+phi@theta;return v@(np.logaddexp(0,logits)-y*logits)
    numerical=np.array([(f(eps*np.eye(2)[j])-f(-eps*np.eye(2)[j]))/(2*eps) for j in range(2)])
    np.testing.assert_allclose(g,-numerical,atol=1e-10)


def test_directed_nonzero_rotation_beats_positive_semidefinite_on_known_synthetic_pair():
    directed=train();psd=train(replace(cfg(),operator_mode='psd',max_iter=10000,tolerance=1e-8))
    pair=rotation_pairs()[0]
    def score(operator):
        state=a.adapt(operator,pair['support_features'],pair['support_logits'],pair['support_y'])
        probs,_=a.predict(state,pair['future_features'],pair['future_logits'])
        return -np.log(probs[np.arange(4),pair['future_y']]).mean()
    np.testing.assert_allclose(directed.A[1,0],.25/(.25*(.25+1/12)+.03),atol=1e-12)
    assert directed.A[1,0]>.1 and abs(directed.A[0,1])<1e-12
    assert score(directed)<score(psd)<np.log(2)
    assert directed.diagnostics['majorizer_value']<0
    assert directed.diagnostics['majorizer_dominates_observed_erm']
    assert np.linalg.eigvalsh(psd.A).min()>-1e-10


@pytest.mark.parametrize('mode',['directed','diagonal','symmetric','psd','identity'])
def test_operator_constraints_and_training_majorizer_nonincrease(mode):
    operator=train(replace(cfg(),operator_mode=mode,max_iter=10000,tolerance=1e-8))
    assert operator.diagnostics['objective_value']<=1e-12
    assert operator.diagnostics['observed_training_erm_difference']<=operator.diagnostics['majorizer_value']+1e-10
    assert operator.diagnostics['future_risk_guaranteed'] is False
    if mode in ('symmetric','psd'):np.testing.assert_allclose(operator.A,operator.A.T,atol=1e-12)
    if mode=='diagonal':np.testing.assert_allclose(operator.A,np.diag(np.diag(operator.A)))
    if mode=='identity':np.testing.assert_allclose(operator.A,np.eye(2)*operator.A[0,0])
    if mode=='psd':assert np.linalg.eigvalsh(operator.A).min()>-1e-10


def test_logistic_global_quadratic_majorizer_for_large_arbitrary_updates():
    rng=np.random.default_rng(88)
    for _ in range(100):
        margin=rng.normal(size=40)*8;y=rng.integers(0,2,40);delta=rng.normal(size=40)*20
        actual=np.logaddexp(0,margin+delta)-y*(margin+delta)-np.logaddexp(0,margin)+y*margin
        bound=(a._sigmoid(margin)-y)*delta+.125*delta**2
        assert np.all(actual<=bound+1e-10)


def test_objective_directional_gradient_and_hessian_are_independent_finite_differences():
    rng=np.random.default_rng(23);d=3;rows=[]
    for _ in range(4):
        phi=rng.normal(size=(7,d));g=rng.normal(size=d);noise=rng.normal(size=(d,d));noise=noise@noise.T
        rows.append(dict(weight=.25,M=phi.T@phi/7,S=np.outer(g,g)+noise,r=rng.normal(size=d),g=g))
    objective=a._Objective(rows,cfg());x=rng.normal(size=(d,d));v=rng.normal(size=(d,d));eps=1e-6
    finite=(objective.majorizer(x+eps*v)-objective.majorizer(x-eps*v))/(2*eps)
    analytic=np.sum((objective.hessian(x)-objective.b)*v)
    np.testing.assert_allclose(finite,analytic,atol=2e-8)
    dense=np.column_stack([objective.hessian(np.eye(d*d)[i].reshape(d,d)).ravel() for i in range(d*d)])
    np.testing.assert_allclose(dense,dense.T,atol=1e-12)
    assert np.linalg.eigvalsh(dense).min()>=cfg().ridge-1e-10
    solved,diag=a._cg(objective,'directed',1000,1e-12)
    np.testing.assert_allclose(solved.ravel(),np.linalg.solve(dense,objective.b.ravel()),atol=1e-10)
    assert diag['converged']


def test_meta_train_only_and_target_fit_api_excludes_future_labels():
    pairs=rotation_pairs();pairs[0]['training_role']='meta_validation'
    with pytest.raises(ValueError,match='meta_train'):a.train_operator(pairs,cfg(),reference())
    with pytest.raises(ValueError,match='meta_train'):a.train_operator(rotation_pairs(),cfg(),dict(reference(),role='holdout'))
    assert 'future_y' not in inspect.signature(a.adapt).parameters
    assert 'query_y' not in inspect.signature(a.predict).parameters


def test_identity_overlap_and_duplicate_support_rejected():
    pair=rotation_pairs()[0];pair['future_ids'][0]='s0'
    with pytest.raises(ValueError,match='overlap'):a.train_operator([pair],cfg(),reference())
    op=train()
    with pytest.raises(ValueError,match='duplicated'):a.adapt(op,np.ones((4,2)),np.zeros(4),[0,0,1,1],support_ids=['same']*4)
    cal=a.adapt(op,np.ones((4,2)),np.zeros(4),[0,0,1,1],support_ids=['a','b','c','d'])
    with pytest.raises(ValueError,match='overlap'):a.predict(cal,np.ones((2,2)),np.zeros(2),query_ids=['a','new'])


def test_one_class_and_empty_support_preserve_anchor_exactly():
    op=train();query=np.array([[0.,1.],[0.,-1.]]);logits=np.array([[4.,-3.],[-2.,1.]])
    for features,labels in ((np.ones((3,2)),np.ones(3,dtype=int)),(np.zeros((0,2)),np.zeros(0,dtype=int))):
        cal=a.adapt(op,features,np.zeros((len(features),2)),labels)
        probs,diag=a.predict(cal,query,logits)
        np.testing.assert_array_equal(probs,a.anchor_probabilities(logits))
        assert diag['summary']['nonzero_updates']==0


def test_query_predictions_are_pointwise_and_batch_statistics_are_never_updated():
    pair=rotation_pairs()[0];op=train();cal=a.adapt(op,pair['support_features'],pair['support_logits'],pair['support_y'])
    query=np.array([[0.,1.],[2.,1.],[0.,-1.]]);z=np.array([.2,-.8,.4])
    whole,diag=a.predict(cal,query,z)
    singles=np.concatenate([a.predict(cal,query[i:i+1],z[i:i+1])[0] for i in range(3)])
    np.testing.assert_allclose(whole,singles,atol=2e-16,rtol=2e-16)
    assert diag['summary']['query_labels_seen']==0 and not diag['summary']['query_batch_statistics_updated']


def test_bounded_frozen_eigenbasis_map_and_large_finite_coordinates():
    ref=reference();ref['eigenvectors']=np.array([[0.,-1.],[1.,0.]])
    config=replace(cfg(),feature_rank=1,include_intercept=True)
    mapped=a.feature_map(np.array([[2.,3.],[2e100,3e100]]),a._reference(ref),config)
    np.testing.assert_allclose(mapped[0],np.array([1.,1.])/np.sqrt(2))
    np.testing.assert_array_equal(mapped[:,-1],np.full(2,1/np.sqrt(2)))
    assert np.max(np.linalg.norm(mapped,axis=1))<=1+1e-15


@pytest.mark.parametrize('noise',['permutation','feature','fisher','none'])
def test_noise_controls_are_explicit_and_psd(noise):
    pair=rotation_pairs()[0];g,n,diag=a.support_moments(pair['support_features'],pair['support_logits'],pair['support_y'],replace(cfg(),noise_type=noise))
    assert np.linalg.eigvalsh(n).min()>-1e-12
    if noise=='none':assert not n.any()
    op=train(replace(cfg(),noise_type=noise));assert op.diagnostics['objective_value']<=1e-12


def test_ridge_limit_shrinks_operator_without_retraining_target():
    small=train(replace(cfg(),ridge=.01));large=train(replace(cfg(),ridge=1e6))
    assert np.linalg.norm(large.A)<np.linalg.norm(small.A)*1e-5


def test_pickle_free_operator_and_calibration_resume_tamper_detection(tmp_path):
    pair=rotation_pairs()[0];op=train();cal=a.adapt(op,pair['support_features'],pair['support_logits'],pair['support_y'])
    expected,_=a.predict(cal,pair['future_features'],pair['future_logits'])
    path=tmp_path/'operator.npz';a.save_operator(op,path)
    restored=a.load_operator(path,expected_operator_identity=op.operator_identity)
    assert restored.operator_identity==op.operator_identity
    assert not restored.A.flags.writeable
    cpath=tmp_path/'calibration.npz';a.save_state(cal,cpath)
    resumed=a.load_state(cpath,expected_fit_identity=cal.fit_identity)
    actual,_=a.predict(resumed,pair['future_features'],pair['future_logits'])
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match='identity'):a.load_operator(path,expected_operator_identity='0'*64)
    with np.load(path,allow_pickle=False) as saved:arrays={k:np.asarray(saved[k]).copy() for k in saved.files}
    arrays['A'][0,0]+=.1;np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='SHA256'):a.load_operator(path)


def test_tiny_iteration_budget_accepts_nonzero_feasible_partial_iterate_without_optimality_claim():
    rng=np.random.default_rng(199);pair=rotation_pairs()[0]
    pair['support_features']=rng.normal(size=(4,2));pair['future_features']=rng.normal(size=(4,2))
    op=a.train_operator([pair],replace(cfg(),max_iter=1,tolerance=1e-20),reference())
    assert op.status=='OPERATOR_TRAINED' and op.A.any()
    assert not op.diagnostics['fallback_zero_operator']
    assert op.diagnostics['accepted_feasible_iterate'] and not op.diagnostics['optimality_claimed']
    assert op.diagnostics['objective_value']<0 and op.diagnostics['majorizer_nonpositive_certificate']
    assert op.diagnostics['observed_training_balanced_nll_difference']<0


def test_erm_control_is_separate_bounded_training_loss():
    op=train(replace(cfg(),training_loss='erm',max_iter=10000,tolerance=1e-8))
    assert op.diagnostics['objective_value']<0
    assert op.config.training_loss=='erm' and not op.diagnostics['future_risk_guaranteed']


def test_constant_postmap_offset_balanced_permutation_covariance_and_feature_equivalence():
    rng=np.random.default_rng(71);x=rng.normal(size=(6,2));config=replace(cfg(),include_intercept=True)
    phi=a.feature_map(x,a._reference(reference()),config);z=rng.normal(size=6);y=np.array([0,0,0,1,1,1])
    np.testing.assert_array_equal(phi[:,-1],np.full(6,1/np.sqrt(2)))
    _,noise,_=a.support_moments(phi,z,y,config)
    centered=phi-phi.mean(0)
    np.testing.assert_allclose(noise,centered.T@centered/(4*6*5),atol=2e-17)
    np.testing.assert_allclose(noise[-1],0,atol=1e-17)


def test_unrestricted_scalar_control_can_reverse_direction():
    pair=rotation_pairs()[0];pair['future_features']=pair['support_features'].copy()
    pair['future_y']=1-pair['support_y']
    op=a.train_operator([pair],replace(cfg(),operator_mode='identity'),reference())
    assert op.A[0,0]<0 and op.diagnostics['objective_value']<0
    assert op.diagnostics['parameter_count']==1


def test_singular_permutation_noise_infinite_strength_does_not_imply_ea():
    # Constant features and a miscalibrated anchor: all fixed-count label
    # permutations have the same score, hence N=0 but the score is nonzero.
    pair=rotation_pairs()[0]
    pair['support_features']=np.zeros((4,2));pair['future_features']=np.zeros((4,2))
    pair['support_logits']=np.full(4,1.5);pair['future_logits']=np.full(4,1.5)
    low=replace(cfg(),include_intercept=True,noise_strength=0.)
    high=replace(low,noise_strength=1e20)
    a0=a.train_operator([pair],low,reference());a1=a.train_operator([pair],high,reference())
    np.testing.assert_allclose(a0.A,a1.A,atol=1e-12)
    assert np.linalg.norm(a1.A)>.1
    assert a1.diagnostics['observed_training_balanced_nll_difference']<0


def test_scientific_operator_identity_repeats_despite_runtime_timer():
    first=train();second=train()
    np.testing.assert_array_equal(first.A,second.A)
    assert first.operator_identity==second.operator_identity


def test_full_sixtyfour_feature_operator_is_finite_and_uses_no_dense_4225_hessian():
    rng=np.random.default_rng(301);pairs=[]
    for k in range(3):
        sx=rng.normal(size=(20,64));qx=rng.normal(size=(40,64))
        pairs.append(dict(support_features=sx,support_logits=np.zeros(20),support_y=np.tile([0,1],10),
            future_features=qx,future_logits=np.zeros(40),future_y=(qx[:,0]>.2).astype(int),
            participant_id=f'person{k}',session_id='s',training_role='meta_train'))
    op=a.train_operator(pairs,a.Config(feature_rank=64,ridge=1e-6),reference(64))
    assert op.A.shape==(65,65) and op.diagnostics['parameter_count']==4225
    assert np.isfinite(op.A).all() and op.diagnostics['objective_value']<=0
    assert op.diagnostics['solver']['solver']=='matrix_free_cg'
    assert op.diagnostics['majorizer_nonpositive_certificate']


def test_matched_feature_control_equals_balanced_permutation_noise_including_constant_offset():
    rng=np.random.default_rng(508);phi=a.feature_map(rng.normal(size=(6,2)),a._reference(reference()),
                                             replace(cfg(),include_intercept=True))
    z=np.array([-2.,-.7,.2,.8,1.5,2.1]);balanced=np.array([0,1,0,1,0,1])
    _,permutation,_=a.support_moments(phi,z,balanced,replace(cfg(),noise_type='permutation'))
    _,feature,_=a.support_moments(phi,z,balanced,replace(cfg(),noise_type='feature'))
    np.testing.assert_array_equal(permutation,feature)
    np.testing.assert_allclose(feature[-1],0,atol=1e-17)
    unbalanced=np.array([0,0,0,0,1,1])
    _,unbalanced_perm,_=a.support_moments(phi,z,unbalanced,replace(cfg(),noise_type='permutation'))
    _,unbalanced_feature,_=a.support_moments(phi,z,unbalanced,replace(cfg(),noise_type='feature'))
    np.testing.assert_array_equal(unbalanced_feature,feature)
    assert not np.allclose(unbalanced_perm,unbalanced_feature,atol=1e-7,rtol=1e-7)
