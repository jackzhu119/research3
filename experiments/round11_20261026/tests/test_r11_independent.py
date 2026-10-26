"""Independent, deterministic R11 counterexamples and protocol checks.

No GPU, Slurm, real EEG, remote writes, or production-source mutations.
"""
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import pytest

from paper3r11 import algorithm, study
from paper3r11.platform import common


def reference(d=2):
    return {'mean':np.zeros(d),'whitener':np.eye(d),'eigenvectors':np.eye(d),
            'provenance':{'role':'meta_train','source_identity':{'fixture':True}}}


def exact_scores(phi, margins, positives):
    """Compute the definition using an independent fixed-count enumeration."""
    n=len(phi);prob=1/(1+np.exp(-np.asarray(margins)))
    scores=[]
    for subset in combinations(range(n), positives):
        y=np.zeros(n,dtype=int);y[list(subset)]=1
        w=np.where(y==1,1/(2*positives),1/(2*(n-positives)))
        scores.append(np.sum(phi*(w*(y-prob))[:,None],axis=0))
    return np.array(scores)


def test_centered_randomization_bound_matches_full_unbalanced_enumeration():
    # Includes a true constant offset; unbalanced counts make E[g_pi] != 0.
    phi=np.array([[-.6,.2,.4],[.5,.1,.4],[-.2,-.3,.4],[.3,.7,.4],[.4,-.2,.4]])
    z=np.array([-.8,1.2,.3,-.2,1.7]);labels=np.array([1,0,1,0,0])
    cfg=algorithm.Config(feature_rank=2,include_intercept=True,noise_strength=2.7,ridge=.01)
    g,n,_=algorithm.support_moments(phi,z,labels,cfg)
    scores=exact_scores(phi,z,2);centered=scores-scores.mean(0)
    assert np.linalg.norm(scores.mean(0))>.01
    np.testing.assert_allclose(n,centered.T@centered/len(scores),rtol=2e-13,atol=1e-15)
    future_phi=np.array([[.2,-.5,.4],[-.4,.3,.4],[.3,.2,.4]])
    future_z=np.array([-.4,.7,.8]);future_y=np.array([0,1,0]);weight=np.array([.25,.5,.25])
    prob=1/(1+np.exp(-future_z))
    r=np.sum(future_phi*(weight*(future_y-prob))[:,None],axis=0)
    m=(future_phi.T*weight)@future_phi
    a=np.array([[.4,1.1,-.6],[-.9,.2,.3],[.1,-.4,.2]])
    expected_change=[]
    for xi in centered:
        delta=future_phi@(a@(g+np.sqrt(cfg.noise_strength)*xi))
        before=np.logaddexp(0,future_z)-future_y*future_z
        after=np.logaddexp(0,future_z+delta)-future_y*(future_z+delta)
        expected_change.append(weight@(after-before))
    upper=-r@(a@g)+.125*np.trace(m@a@(np.outer(g,g)+cfg.noise_strength*n)@a.T)
    assert np.mean(expected_change)<=upper+1e-14
    # Omitting centering changes the expected linear score term: a different object.
    raw_second=scores.T@scores/len(scores)
    assert np.linalg.norm(raw_second-n)>1e-4


def test_balanced_permutation_is_geometric_not_anchor_uncertainty():
    phi=np.array([[-.7,.2,.5],[.2,-.3,.5],[.6,.1,.5],[-.1,.4,.5]])
    y=np.array([0,1,0,1]);cfg=algorithm.Config(feature_rank=2)
    expected=np.cov(phi,rowvar=False,ddof=1)/(4*len(phi))
    previous=None
    for z in (np.zeros(4),np.array([-12.,8.,6.,-9.])):
        _,n,_=algorithm.support_moments(phi,z,y,cfg)
        np.testing.assert_allclose(n,expected,atol=1e-16)
        _,control,_=algorithm.support_moments(phi,z,y,replace(cfg,noise_type='feature'))
        np.testing.assert_array_equal(n,control)
        assert np.all(n[-1]==0.) and np.all(n[:,-1]==0.)
        if previous is not None:np.testing.assert_array_equal(previous,n)
        previous=n


def rotation_pairs():
    pairs=[]
    # The two tasks require opposite off-diagonal signs; symmetric A cannot
    # satisfy both. A single rotation pair would not isolate this requirement.
    for k,(u,v) in enumerate(((np.array([1.,0.]),np.array([0.,1.])),
                              (np.array([0.,1.]),np.array([-1.,0.])))):
        pairs.append({'training_role':'meta_train','participant_id':'P'+str(k),'session_id':'1',
          'support_features':np.stack((-u,u)),'support_logits':np.zeros((2,2)),
          'support_y':np.array([0,1]),'future_features':np.stack((-v,v)),
          'future_logits':np.zeros((2,2)),'future_y':np.array([0,1]),
          'support_ids':('s'+str(k)+'a','s'+str(k)+'b'),
          'future_ids':('q'+str(k)+'a','q'+str(k)+'b')})
    return pairs


def test_two_role_rotation_requires_skew_operator_not_just_indefinite_symmetric():
    cfg=algorithm.Config(feature_rank=2,include_intercept=False,ridge=.02,noise_strength=1.)
    pairs=rotation_pairs()
    directed=algorithm.train_operator(pairs,cfg,reference())
    symmetric=algorithm.train_operator(pairs,replace(cfg,operator_mode='symmetric'),reference())
    assert directed.A[1,0]>.1 and directed.A[0,1]<-.1
    np.testing.assert_allclose(symmetric.A,np.zeros((2,2)),atol=1e-12)
    assert np.linalg.norm(directed.A-directed.A.T)>.1
    for pair in pairs:
        cal=algorithm.adapt(directed,pair['support_features'],pair['support_logits'],pair['support_y'],
                            support_ids=pair['support_ids'])
        p,_=algorithm.predict(cal,pair['future_features'],pair['future_logits'],query_ids=pair['future_ids'])
        assert np.all(p[np.arange(2),pair['future_y']]>.5)
    assert directed.diagnostics['observed_training_balanced_nll_difference']<0
    assert directed.diagnostics['future_risk_guaranteed'] is False


def test_finite_partial_solution_retains_actual_feasible_update_and_resume(tmp_path):
    rng=np.random.default_rng(116);pairs=rotation_pairs()
    for pair in pairs:
        pair['support_features']=rng.normal(size=(2,2))
        pair['future_features']=rng.normal(size=(2,2))
    cfg=algorithm.Config(feature_rank=2,include_intercept=False,max_iter=1,tolerance=1e-30,ridge=.001)
    op=algorithm.train_operator(pairs,cfg,reference())
    assert op.A.any() and not op.diagnostics['optimality_claimed']
    assert op.diagnostics['accepted_feasible_iterate']
    assert op.diagnostics['objective_value']<=0.
    path=tmp_path/'partial.npz';algorithm.save_operator(op,path)
    loaded=algorithm.load_operator(path,expected_operator_identity=op.operator_identity)
    np.testing.assert_array_equal(op.A,loaded.A)
    assert not loaded.diagnostics['optimality_claimed']


def prepared_registry(tmp_path):
    root=tmp_path/'school';root.mkdir();package=tmp_path/'package'
    (package/'paper3r11').mkdir(parents=True);(package/'configs').mkdir()
    (package/'START.py').write_text('# software fixture\n')
    (package/'paper3r11/fixture.py').write_text('VALUE=1\n')
    (package/'configs/profile_r10.default.json').write_bytes(
        (Path(study.PACKAGE)/'configs/profile_r10.default.json').read_bytes())
    p=root/'small.json';common.atomic(p,{'fixture_only':True});d=study.describe(p,root)
    registry={'schema_version':'paper3-r11-study-registry-v1','root':str(root),
      'source_by_seed':{str(s):{'source_receipt':d,'reference_receipt':d} for s in study.SEEDS},
      'external_manifests':{'pan2023':d,'weibo2014':d},'baseline_artifacts':{},
      'profile_reference_by_seed':{},'profile_raw_reference_by_seed':{},
      'previously_used_participants':['OLD:P0']}
    p=root/'registry.json';common.atomic(p,registry)
    return root,package,study.describe(p,root)


def test_full_campaign_bound_and_optional_history_do_not_block_core_dag(tmp_path):
    root,package,descriptor=prepared_registry(tmp_path)
    plan=study.build_plan(root,descriptor,package=package)
    tasks=plan['tasks'];lookup={t['key']:t for t in tasks}
    assert len(tasks)==10270 and len(lookup)==10270
    assert plan['resources']=={'jobs':4,'gpus_per_job':1,'cpus_per_job':16,
        'memory_per_job':'64G','partition':'hpc_gpu','time_limit':'5-03:00:00'}
    selection=next(t for t in tasks if t['family']=='selection')
    assert len(selection['requires_terminal'])==1656
    assert all(lookup[k]['role']=='meta_validation' and lookup[k]['family']=='evaluate'
               for k in selection['requires_terminal'])
    assert all(t['role']=='meta_train' for t in tasks if t['family']=='operator_train')
    historical=[t for t in tasks if t['method'] in study.NEEDS_ARTIFACTS]
    assert historical and all(t['optional_baseline'] for t in historical)
    assert all(t.get('availability_at_preparation',{}).get('status')=='UNAVAILABLE' for t in historical)
    for task in tasks:
        if task['family']=='evaluate' and task['role'] in ('internal_holdout','external_exploratory'):
            assert selection['key'] in task['requires']
    assert not plan['target_query_labels_used_for_selection']
    p=study.write_plan(plan,root/'r11-campaign-plan.json')
    assert p.stat().st_size<64*1024*1024
    assert common.load_plan(p)[1]['recipe_sha256']==plan['recipe_sha256']


def independent_confirmation(root):
    contract={'n_times':128,'fixture_only':True}
    (root/'contracts').mkdir();common.atomic(root/'contracts/input_contract.json',contract)
    signals=root/'new.npy';np.save(signals,np.zeros((40,11,128),dtype=np.float32))
    events=[{'participant_id':'NEW:P1','session_id':'1','tasknumber':1,
      'event_uid':'new-'+str(i),'trial_uid':'new-'+str(i),'original_trial_index':i,
      'label':i%2,'valid':True,'cache_index':i,
      'physical_run_id':'prefix' if i<20 else 'future','physical_run_complete_verified':True}
      for i in range(40)]
    common.atomic(root/'events.json',events)
    common.atomic(root/'independence-review.json',{'status':'REVIEWED','prior_design_history_reviewed':True,
      'previously_used_for_algorithm_design':False,'reviewer':'software-fixture-only',
      'reviewed_utc':'2026-10-26T00:00:00Z','dataset_identity':{'id':'new-fixture'},'participants':['NEW:P1']})
    manifest={'schema_version':'paper3-r11-independent-cache-v1','status':'COMPLETE',
      'cohort_kind':'independent_confirmation','acquisition_order_verified':True,
      'physical_run_id_available':True,'previously_used_for_algorithm_design':False,
      'signal_contract_sha256':common.digest(contract),'dataset_identity':{'id':'new-fixture'},
      'participants':['NEW:P1'],'independence_review':study.describe(root/'independence-review.json',root),
      'sessions':[{'participant_id':'NEW:P1','session_id':'1','acquisition_order_verified':True,
         'X':study.describe(signals,root),'events':study.describe(root/'events.json',root)}]}
    path=root/'confirmation.json';common.atomic(path,manifest)
    return path


def test_independent_confirmation_metadata_rejected_when_only_an_empty_label(tmp_path):
    root,package,descriptor=prepared_registry(tmp_path)
    bad=root/'confirmation.json';common.atomic(bad,{'status':'COMPLETE','cohort_kind':'independent_confirmation'})
    with pytest.raises(ValueError,match='Independent confirmation'):
        study.build_plan(root,descriptor,package=package,stage='confirmatory',
             grid={'feature_ranks':[2],'noise_strengths':[1.],'ridges':[.01]},
             seeds=(study.SEEDS[0],),confirmation=study.describe(bad,root))


def test_confirmation_can_contain_development_selection_before_independent_targets(tmp_path):
    root,package,descriptor=prepared_registry(tmp_path);manifest=independent_confirmation(root)
    plan=study.build_plan(root,descriptor,package=package,stage='confirmatory',
          grid={'feature_ranks':[2],'noise_strengths':[1.],'ridges':[.01]},
          seeds=(study.SEEDS[0],),confirmation=study.describe(manifest,root))
    assert plan['confirmation_metadata_audit']['eligible_sessions'][0]['query_count']==20
    dev=[t for t in plan['tasks'] if t['role']=='meta_validation']
    assert dev and all(t['phase']=='development' for t in dev)
    targets=[t for t in plan['tasks'] if t['role']=='independent_confirmation']
    assert targets and all(t['phase']=='confirmatory' for t in targets)
    assert all(t['selection_dependency'] in {x['key'] for x in dev if x['family']=='selection'}
               for t in targets if t['method'].startswith('operator'))
    path=study.write_plan(plan,root/'r11-confirmatory-plan.json')
    assert common.load_plan(path)[1]['recipe_sha256']==plan['recipe_sha256']
    # Same cache row under a distinct acquired/future UID cannot fake independence.
    events=common.read(root/'events.json');events[-1]['cache_index']=0;common.atomic(root/'events.json',events)
    value=common.read(manifest);value['sessions'][0]['events']=study.describe(root/'events.json',root);common.atomic(manifest,value)
    with pytest.raises(ValueError,match='alias'):
        study.build_plan(root,descriptor,package=package,stage='confirmatory',
             grid={'feature_ranks':[2],'noise_strengths':[1.],'ridges':[.01]},
             seeds=(study.SEEDS[0],),confirmation=study.describe(manifest,root))


def test_unrestricted_scalar_control_can_reverse_an_informative_prefix_gradient():
    pair=rotation_pairs()[0]
    pair['future_features']=-pair['support_features']
    config=algorithm.Config(feature_rank=2,include_intercept=False,operator_mode='identity',ridge=.02)
    op=algorithm.train_operator([pair],config,reference())
    assert np.trace(op.A)<0.
    np.testing.assert_allclose(op.A,np.eye(2)*op.A[0,0],atol=1e-12)
    cal=algorithm.adapt(op,pair['support_features'],pair['support_logits'],pair['support_y'],support_ids=pair['support_ids'])
    p,_=algorithm.predict(cal,pair['future_features'],pair['future_logits'],query_ids=pair['future_ids'])
    assert np.all(p[np.arange(2),pair['future_y']]>.5)


@pytest.mark.parametrize('stage', ['exploratory', 'confirmatory'])
def test_actual_formal_dag_runs_its_development_selection_without_target_rows(tmp_path, monkeypatch, stage):
    """A legal formal plan still must execute its in-DAG development selection.

    Compact loading and optimization are isolated here so the test directly
    covers the formerly inconsistent plan-stage/selection-stage boundary.
    A poisoned target row also checks the capability passed to selection.
    """
    from paper3r11 import selection
    root, package, descriptor = prepared_registry(tmp_path)
    extra = {}
    if stage == 'confirmatory':
        extra['confirmation'] = study.describe(independent_confirmation(root), root)
    plan = study.build_plan(root, descriptor, package=package, stage=stage,
        grid={'feature_ranks': [2], 'noise_strengths': [1.], 'ridges': [.01]},
        seeds=(study.SEEDS[0],), **extra)
    path = study.write_plan(plan, Path(plan['output']) / 'plan.json')
    task = next(t for t in plan['tasks'] if t['family'] == 'selection')
    development = {'role': 'meta_validation', 'cohort': 'meta_validation',
                   'method': 'support_ea', 'budget': 20, 'candidate_id': ''}
    poison = {'role': 'independent_confirmation', 'cohort': 'confirmation',
              'method': 'TARGET_LABELS_MUST_NOT_REACH_SELECTION', 'budget': 20,
              'candidate_id': ''}
    monkeypatch.setattr(selection, 'load_completed',
        lambda _: (plan, [development, poison], {}, []))
    seen = []
    def choose(_plan, index, rows, method):
        assert rows == [development]
        assert all(key[0] == 'meta_validation' for key in index.groups)
        seen.append(method)
        return {'status': 'DEFAULT_UNVALIDATED', 'configuration': study.default_configuration(),
                'candidate_id': 'fixture-default', 'operator_artifacts': {},
                'candidate_summaries': [], 'excluded_candidates': [],
                'lopo_diagnostic': {'folds': []}}
    monkeypatch.setattr(selection, '_select_method', choose)
    result = selection.run_selection_task(plan, task, Path(plan['output']) / task['key'])
    assert set(seen) == set(selection.SELECTED_METHODS)
    assert result['source_plan_sha256'] == common.sha(path)
    assert result['target_query_labels_used_for_selection'] is False
    assert result['status'] == 'DEFAULT_UNVALIDATED'
