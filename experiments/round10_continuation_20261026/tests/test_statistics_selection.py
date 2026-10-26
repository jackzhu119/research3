import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from paper3r10.algorithm import Config
from paper3r10.platform import common
from paper3r10.report import (PairingError,generate_report,load_completed,paired_vectors,
                             participant_records,write_csv)
from paper3r10.selection import development_decision,freeze_development
from paper3r10.statistics import (PRIMARY_FAMILY,bootstrap_mean_interval,holm_fixed,
                                 normalized_aubc,paired_effect,signflip_pvalue)


def make_plan(tmp_path,*,development=False):
    root=tmp_path/'study';root.mkdir()
    package=tmp_path/'package';package.mkdir();(package/'fixture.py').write_text('VALUE = 1\n')
    registry=root/'study-registry.json';common.atomic(registry,{'schema_version':'SOFTWARE_TEST_ONLY','root':str(root)})
    tasks=[]
    if development:
        for amplitude in (.3,1.,3.):
            for fraction in (.01,.1,.5):
                candidate=f'a{amplitude}-c{fraction}'
                tasks.append({'key':'profile-'+candidate,'family':'mechanism','method':'profile',
                    'candidate_id':candidate,'role':'meta_validation','cohort':'meta_validation','seed':20261018,
                    'config':asdict(Config(amplitude=amplitude,evidence_fraction=fraction)), 'requires':[]})
        tasks.append({'key':'ea','family':'mechanism','method':'support_ea','role':'meta_validation',
                      'cohort':'meta_validation','seed':20261018,'config':{},'requires':[]})
    else:
        for seed in (20261018,20261019):
            for method in ('profile','support_ea','sgd100','r9_ea_anchored','profile_map',
                           'profile_no_completion','profile_quadratic','profile_variance_gate','profile_frozen_anchor'):
                tasks.append({'key':method+'-'+str(seed),'family':'evaluate','method':method,
                    'role':'holdout','cohort':'holdout','seed':seed,
                    'config':asdict(Config()) if method.startswith('profile') else {},'requires':[]})
    plan={'schema_version':common.PLAN_SCHEMA,'round':10,'stage':'mechanism' if development else 'formal_exploratory',
        'synthetic_fixture':True,
        'root':str(root),'package':str(package),'output':str(root/'runs/r10-stat-test'),
        'code_manifest':{'fixture.py':common.sha(package/'fixture.py')},'tasks':tasks,
        'comparison_family':copy.deepcopy(PRIMARY_FAMILY),'budgets':[4,8,12,16,20],
        'study_registry':{'path':registry.name,'sha256':common.sha(registry),'bytes':registry.stat().st_size},
        'reference_strategy':'build_meta_train_prefix',
        'cohorts':['meta_validation'] if development else ['holdout']}
    if not development:
        freeze=root/'freeze.json';common.atomic(freeze,{
            'status':'FROZEN','target_query_labels_used_for_selection':False,
            'scientific_scope':'SOFTWARE_TEST_ONLY',
            'development_query_labels_used':True,'selection_roles':['meta_train','meta_validation'],
            'selected_configuration':asdict(Config()),'comparison_family':PRIMARY_FAMILY})
        plan['freeze']={'path':freeze.name,'sha256':common.sha(freeze),'bytes':freeze.stat().st_size}
        tasks[0]['is_primary_algorithm']=True
    plan=common.sign_plan(plan);path=tmp_path/'plan.json';common.atomic(path,plan)
    return path,plan


def cells(task,*,development=False):
    source={'checkpoint':'source-'+str(task['seed'])};data={'cohort':task['cohort']}
    rows=[]
    ordered=sorted(['S'+str(i) for i in range(8)],key=lambda p:common.digest(['r10-mechanism-v1','participant',p]))
    for person in range(8 if development else 4):
        for session in ('1','2'):
            for budget in (4,8,12,16,20):
                base=.5+.005*person+.002*int(session)
                gain=.005 if development else .04
                score=base+(gain if task['method']=='profile' else 0.)
                rows.append({'participant_id':'S'+str(person),'session_id':session,'seed':task['seed'],
                    'budget':budget,'cohort':task['cohort'],'role':task['role'],'method':task['method'],
                    'selection_split':'selection' if 'S'+str(person) in ordered[:4] else 'check','n_query':100,
                    'n_class0':50,'n_class1':50,'ba':score,'balanced_nll':.68-(gain if task['method']=='profile' else 0),
                    'balanced_brier':.24,'n_valid_support':budget,'elapsed_seconds':.5,
                    'query_signature':common.digest([person,session]),
                    'source_identity_sha256':common.digest(source),'data_identity_sha256':common.digest(data),
                    'updated_query_fraction':.02 if task['method']=='profile' else 0.,
                    'certificate_violation_count':0,'numerical_fallback_count':0,
                    'incremental_prediction_seconds':.1,'fit_seconds':.001})
    return rows,source,data


def complete(plan,task,rows=None,*,development=False):
    standard,source,data=cells(task,development=development)
    rows=standard if rows is None else rows
    folder=common.task_folder(plan,task);folder.mkdir(parents=True,exist_ok=True)
    write_csv(folder/'session_metrics.csv',rows)
    payload=folder/'predictions.npz';payload.write_bytes(b'SYNTHETIC sealed payload; never opened by report')
    artifacts=[{'path':p.name,'sha256':common.sha(p),'bytes':p.stat().st_size}
               for p in (folder/'session_metrics.csv',payload)]
    result={'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],'task_key':task['key'],
        'scientific_scope':'SOFTWARE_TEST_ONLY',
        'task_identity':common.task_identity(plan,task),'artifacts':artifacts,'synthetic':True,
        'source_identity':source,'catalog_identity':data,'registry_sha256':plan['study_registry']['sha256'],'sessions':len(rows)//5,
        'participants':len({r['participant_id'] for r in rows}),
        'completed_session_budget_cells':len(rows)}
    common.atomic(folder/'result.json',result);common.seal_result(plan,task)


def fill(plan,development=False,modify=None):
    for task in plan['tasks']:
        rows,_,_=cells(task,development=development)
        if modify:modify(task,rows)
        complete(plan,task,rows,development=development)


def test_signflip_exact_and_monte_carlo_plus_one_are_participant_tests():
    assert signflip_pvalue([1,1])['pvalue']==.5
    assert signflip_pvalue([1,1,1,1])['pvalue']==.125
    assert signflip_pvalue([0,0])['pvalue']==1.
    value=signflip_pvalue(np.ones(21),monte_carlo=1000)
    assert value['method']=='monte_carlo_sign_flip_plus_one' and value['pvalue']>0
    assert value==signflip_pvalue(np.ones(21),monte_carlo=1000)


def test_paired_bootstrap_and_holm_preserve_missing_family_members():
    assert bootstrap_mean_interval([1,1,1])==[1.,1.]
    assert holm_fixed([.01,None,.04,None])==[.04,None,.12,None]
    result=paired_effect([.7,.8],[.6,.9])
    assert result['n_participants']==2 and result['negative_transfer_fraction']==.5
    assert 'not inverse' in result['ci_method']
    assert 'negative_transfer_fraction' not in paired_effect([.3,.4],[.5,.6],metric='balanced_nll')


def test_budget_auc_is_actual_acquired_position_trapezoid():
    assert normalized_aubc([.5,.6,.7,.8,.9])==pytest.approx(.7)
    with pytest.raises(ValueError):normalized_aubc([.5,.6])


def test_aggregation_sessions_then_seeds_then_participants_not_trial_weighted():
    rows=[]
    for seed,values in [(1,[0.,1.]),(2,[1.])]:
        for session,value in enumerate(values):
            rows.append({'cohort':'c','method':'m','candidate_id':'','configuration_id':'a',
                'participant_id':'p','seed':seed,'session_id':str(session),'budget':20,
                'n_query':1 if value==0 else 10000,'ba':value,'balanced_nll':1.,'balanced_brier':.2})
    value=participant_records(rows)[0]
    assert value['ba']==.75 and value['n_seeds']==2


def test_report_complete_actual_coverage_primary_holm_and_secondary_labels(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan)
    report=generate_report(path)
    assert report['status']=='COMPLETE' and report['coverage']['sealed_complete_tasks']==18
    assert report['scientific_scope']=='SOFTWARE_TEST_ONLY'
    with pytest.raises(ValueError,match='Synthetic'):common.load_plan(path)
    assert report['primary_family_size']==4 and report['primary_family_complete']
    assert report['methods'] and report['comparisons']
    primary=[r for r in report['comparisons'] if r.get('holm_family_name')=='primary']
    assert len(primary)==4 and all(r['holm_family_size']==4 for r in primary)
    assert all(r['n_participants']==4 and r['n_seeds']==2 for r in primary)
    assert report['secondary_ablation_family_size']==8
    assert report['secondary_ablation_family_complete']
    assert all(r['inference_scope']=='secondary_unadjusted_not_primary' for r in report['comparisons'][12:])
    status=common.read(Path(plan['output'])/'reports/report-status.json')
    assert status['complete_task_keys_sha256']==common.digest(sorted(t['key'] for t in plan['tasks']))
    assert status['report_sha256']==common.sha(Path(plan['output'])/'reports/report.json')


def test_missing_task_keeps_family_size_four_and_report_partial(tmp_path):
    path,plan=make_plan(tmp_path)
    for task in plan['tasks']:
        if task['method']!='r9_ea_anchored':complete(plan,task)
    report=generate_report(path)
    assert report['status']=='PARTIAL' and report['primary_family_size']==4
    assert not report['primary_family_complete']
    primary=report['comparisons'][:4]
    assert primary[2]['status']=='NOT_READY' and primary[2]['holm_pvalue'] is None
    assert all(not x['significant_holm_05'] for x in primary)


def test_reporting_rehashes_compact_metrics_not_full_prediction_archives(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan)
    task=plan['tasks'][0];folder=common.task_folder(plan,task)
    (folder/'predictions.npz').write_bytes(b'archive storage corruption; compact historical metrics still sealed')
    assert not common.verify_result(plan,task)
    report=generate_report(path)
    assert report['status']=='COMPLETE'
    assert not report['coverage']['full_prediction_archives_rehashed_for_report']
    (folder/'session_metrics.csv').write_bytes(b'compact corruption')
    report=generate_report(path)
    assert report['status']=='PARTIAL' and report['coverage']['invalid_compact_metrics']


def test_pairing_mismatch_never_silently_drops_unmatched_cells(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan)
    _,rows,_,_=load_completed(path)
    candidate=next(r for r in rows if r['method']=='profile' and r['budget']==20)
    rows.remove(candidate)
    with pytest.raises(PairingError,match='intersection'):
        paired_vectors(rows,'holdout','profile','support_ea')


@pytest.mark.parametrize('field',['query_signature','source_identity_sha256','data_identity_sha256'])
def test_comparison_blocks_changed_paired_identities(tmp_path,field):
    path,plan=make_plan(tmp_path);fill(plan)
    _,rows,_,_=load_completed(path)
    candidate=next(r for r in rows if r['method']=='profile' and r['budget']==20)
    candidate[field]='changed'
    with pytest.raises(PairingError,match='identity'):
        paired_vectors(rows,'holdout','profile','support_ea')


def test_development_one_se_conservative_selection_and_immutable_pass_freeze(tmp_path):
    path,plan=make_plan(tmp_path,development=True);fill(plan,True)
    decision=development_decision(path)
    assert decision['status']=='PASS',decision.get('stop_reasons')
    assert decision['selected_configuration']['evidence_fraction']==.5
    assert decision['selected_configuration']['amplitude']==.3
    assert len(decision['selection_participant_ids'])==len(decision['check_participant_ids'])==4
    target=Path(plan['root'])/'r10-freeze.json'
    descriptor=freeze_development(path,target);before=target.read_bytes()
    assert freeze_development(path,target)==descriptor and target.read_bytes()==before
    receipt=common.read(target)
    assert receipt['status']=='FROZEN' and receipt['allowed_ablations']['profile_map']['mode']=='map'
    assert receipt['allowed_ablations']['profile_map']['evidence_fraction']==0
    assert receipt['target_query_labels_used_for_selection'] is False


def test_check_group_can_stop_but_never_reranks_the_selected_configuration(tmp_path):
    path,plan=make_plan(tmp_path,development=True)
    def modify(task,rows):
        if task.get('candidate_id')=='a0.3-c0.5':
            for row in rows:
                if row['selection_split']=='check':row['ba']-=.02
    fill(plan,True,modify)
    decision=development_decision(path)
    assert decision['status']=='STOP' and decision['selected_candidate_id']=='a0.3-c0.5'
    assert 'check_ba_noninferiority' in decision['stop_reasons']
    with pytest.raises(ValueError,match='STOP'):freeze_development(path,Path(plan['root'])/'freeze.json')


def test_freeze_cannot_silently_reaccept_changed_honesty_or_statistical_metadata(tmp_path):
    path,plan=make_plan(tmp_path,development=True);fill(plan,True)
    target=Path(plan['root'])/'r10-freeze.json';freeze_development(path,target)
    receipt=common.read(target);receipt['target_query_labels_used_for_selection']=True
    common.atomic(target,receipt)
    with pytest.raises(ValueError,match='immutable'):
        freeze_development(path,target)


@pytest.mark.parametrize('field,value,gate',[
    ('updated_query_fraction',0.,'nontrivial_updates'),
    ('certificate_violation_count',1,'certificate_violations'),
    ('numerical_fallback_count',2,'global_numerical_fallback'),
    ('incremental_prediction_seconds',1.,'incremental_prediction_cost'),
])
def test_mechanism_and_cost_failures_stop_formal_expansion(tmp_path,field,value,gate):
    path,plan=make_plan(tmp_path,development=True)
    def modify(task,rows):
        if task['method']=='profile':
            for row in rows:row[field]=value
    fill(plan,True,modify)
    decision=development_decision(path)
    assert decision['status']=='STOP' and gate in decision['stop_reasons']


def test_selection_requires_all_nine_candidates_and_no_partial_sealed_tasks(tmp_path):
    path,plan=make_plan(tmp_path,development=True)
    for task in plan['tasks'][:-1]:complete(plan,task,development=True)
    decision=development_decision(path)
    assert decision['status']=='STOP' and not decision['gates']['pairing_and_identity']


def test_missing_entire_seed_for_both_methods_does_not_claim_a_ready_family(tmp_path):
    path,plan=make_plan(tmp_path)
    for task in plan['tasks']:
        if task['seed']==20261018:complete(plan,task)
    report=generate_report(path)
    assert report['status']=='PARTIAL'
    assert not report['primary_family_complete']
    assert all(r['status']=='NOT_READY' for r in report['comparisons'][:4])


def test_posthoc_swapping_check_and_selection_markers_is_rejected(tmp_path):
    path,plan=make_plan(tmp_path,development=True)
    def modify(task,rows):
        for row in rows:row['selection_split']='check' if row['selection_split']=='selection' else 'selection'
    fill(plan,True,modify)
    decision=development_decision(path)
    assert decision['status']=='STOP' and 'metadata hash' in decision['stop_reasons'][0]


def test_declared_split_role_mismatch_is_not_overwritten_by_report_enrichment(tmp_path):
    path,plan=make_plan(tmp_path,development=True)
    def modify(task,rows):
        if task['method']=='profile':
            rows[0]['role']='internal_holdout'
    fill(plan,True,modify)
    report=generate_report(path)
    assert report['status']=='PARTIAL'
    assert any('split role' in x['reason'] for x in report['coverage']['invalid_compact_metrics'])
    assert development_decision(path)['status']=='STOP'


def test_target_plan_cannot_be_used_to_select_hyperparameters(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan)
    with pytest.raises(ValueError,match='forbidden'):development_decision(path)
