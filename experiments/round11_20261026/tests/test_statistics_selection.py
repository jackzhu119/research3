import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from paper3r11.algorithm import Config
from paper3r11.platform import common
from paper3r11.report import (MetricIndex,PairingError,generate_report,load_completed,
    paired_vectors,participant_records,write_csv)
from paper3r11.selection import (development_decision,execute_selection,run_selection_task,
    _lopo,_one_se)
from paper3r11.statistics import (PRIMARY_FAMILY,SECONDARY_ABLATION_FAMILY,
    OPERATOR_CONTROLS,bootstrap_mean_interval,holm_fixed,normalized_aubc,paired_effect,signflip_pvalue)

SEEDS=(20261018,20261019,20261020)
BUDGETS=(4,8,12,16,20)
PEOPLE=('P0','P1','P2','P3','P4')


def make_plan(tmp_path,stage='development'):
    root=tmp_path/'study';root.mkdir()
    package=tmp_path/'package';package.mkdir();(package/'fixture.py').write_text('VALUE=1\n')
    registry=root/'registry.json';common.atomic(registry,{'schema_version':'SOFTWARE_TEST_ONLY','root':str(root)})
    config=asdict(Config());tasks=[]
    def add(key,family,method,seed,role,cohort,cfg=None,**extra):
        value={'key':key,'family':family,'method':method,'seed':seed,'role':role,'cohort':cohort,
               'config':cfg or {},'requires':[],'budgets':list(BUDGETS),**extra};tasks.append(value);return value
    if stage in ('development','campaign'):
        for method in ('operator','operator_no_noise_tuned','operator_erm_tuned'):
            defaults=dict(config)
            if method!='operator':defaults.update(noise_strength=0.,noise_type='none',training_loss='erm' if method=='operator_erm_tuned' else 'majorizer')
            choices=[('default',defaults),('small',dict(defaults,feature_rank=4,ridge=.1))]
            for name,cfg in choices:
                candidate=method+'-'+name
                for seed in SEEDS:
                    for b in BUDGETS:
                        add('train-'+candidate+'-'+str(seed)+'-B'+str(b),'operator_train',method,seed,
                            'meta_train','meta_train',cfg,budget=b,budgets=[b],candidate_id=candidate)
                    add('dev-'+candidate+'-'+str(seed),'evaluate',method,seed,'meta_validation','meta_validation',cfg,candidate_id=candidate)
        for seed in SEEDS:add('dev-ea-'+str(seed),'evaluate','support_ea',seed,'meta_validation','meta_validation')
        add('selection','selection','selection',SEEDS[0],'meta_validation','meta_validation')
    if stage in ('campaign','exploratory','confirmatory'):
        methods=['operator','support_ea','sgd100','r9_raw_anchored','r9_ea_anchored','profile','operator_erm_tuned',*OPERATOR_CONTROLS]
        for method in methods:
            for seed in SEEDS:
                add('target-'+method+'-'+str(seed),'evaluate',method,seed,
                    'independent_confirmation' if stage=='confirmatory' else 'internal_holdout',
                    'confirmation' if stage=='confirmatory' else 'holdout',config if method.startswith('operator') else {})
    plan={'schema_version':common.PLAN_SCHEMA,'round':11,'stage':stage,'synthetic_fixture':True,
        'root':str(root),'package':str(package),'output':str(root/'runs/r11-stat-test'),
        'code_manifest':{'fixture.py':common.sha(package/'fixture.py')},'tasks':tasks,
        'study_registry':{'path':'registry.json','sha256':common.sha(registry),'bytes':registry.stat().st_size},
        'reference_strategy':'reuse_r9_ea','comparison_family':copy.deepcopy(PRIMARY_FAMILY),
        'secondary_ablation_family':copy.deepcopy(SECONDARY_ABLATION_FAMILY),
        'default_configuration':config,'cohorts':['meta_validation'] if stage=='development' else ['confirmation' if stage=='confirmatory' else 'meta_validation','holdout'],
        'budgets':list(BUDGETS)}
    if stage=='confirmatory':plan['cohorts']=['confirmation']
    plan=common.sign_plan(plan);path=root/'plan.json';common.atomic(path,plan)
    out=Path(plan['output']);out.mkdir(parents=True);common.atomic(out/'plan.json',plan)
    return path,plan


def cells(task,gain=None):
    src={'checkpoint':'source-'+str(task['seed'])};data={'cohort':task['cohort']};rows=[]
    if gain is None:gain=.02 if task['method'].startswith('operator') else 0.
    for j,p in enumerate(PEOPLE):
        for session in ('1','2'):
            for b in BUDGETS:
                g=gain(j,b) if callable(gain) else gain
                base=.52+j*.025+int(session)*.002
                rows.append({'participant_id':p,'session_id':session,'seed':task['seed'],'budget':b,
                    'cohort':task['cohort'],'role':task['role'],'method':task['method'],'selection_split':'development',
                    'n_query':100,'n_class0':50,'n_class1':50,'ba':base+g,'balanced_nll':.70-g,
                    'balanced_brier':.24,'n_valid_support':b,'elapsed_seconds':.2,'fit_seconds':.01,
                    'incremental_prediction_seconds':.02,'query_signature':common.digest([p,session]),
                    'source_identity_sha256':common.digest(src),'data_identity_sha256':common.digest(data)})
    return rows,src,data


def complete(plan,task,gain=None,mutate=None):
    folder=common.task_folder(plan,task);folder.mkdir(parents=True,exist_ok=True)
    source={'checkpoint':'source-'+str(task['seed'])};data={'cohort':task['cohort']}
    fields={};paths=[]
    if task['family']=='operator_train':
        for name in ('operator.npz','operator.json'):
            p=folder/name;p.write_bytes(b'SOFTWARE fixture only' if name.endswith('npz') else b'{"software_fixture":true}');paths.append(p)
        fields.update(operator_identity=common.digest([task['candidate_id'],task['seed'],task['budget']]))
    elif task['family']=='selection':
        p=folder/'selection.json';common.atomic(p,{'fixture':True});paths=[p]
    else:
        rows,source,data=cells(task,gain)
        if mutate:mutate(rows)
        p=folder/'session_metrics.csv';write_csv(p,rows);paths.append(p)
        p=folder/'predictions.npz';p.write_bytes(b'SOFTWARE sealed predictions, report never opens payload');paths.append(p)
        fields.update(completed_session_budget_cells=len(rows),sessions=len(rows)//5,
                      participants=len({r['participant_id'] for r in rows}),resolved_configuration=task['config'])
    result={'schema_version':'paper3-r11-scientific-result-v1','status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],
        'task_key':task['key'],'task_identity':common.task_identity(plan,task),'scientific_scope':'SOFTWARE_TEST_ONLY',
        'source_identity':source,'catalog_identity':data,'registry_sha256':plan['study_registry']['sha256'],
        'artifacts':[{'path':p.name,'sha256':common.sha(p),'bytes':p.stat().st_size} for p in paths],**fields}
    common.atomic(folder/'result.json',result)
    from paper3r11.platform.runtime import finish_task
    finish_task(plan,task,0)


def fill(plan,*,gain=None,skip=None):
    for task in plan['tasks']:
        if skip and skip(task):continue
        complete(plan,task,gain=gain(task) if gain else None)


def test_participant_randomization_and_fixed_missing_holm():
    assert signflip_pvalue([1,1])['pvalue']==.5
    assert signflip_pvalue([1,1,1,1])['pvalue']==.125
    assert signflip_pvalue([0,0])['pvalue']==1.
    assert signflip_pvalue(np.ones(21),monte_carlo=1000)['pvalue']>0
    assert holm_fixed([.01,None,.04,None])==[.04,None,.12,None]
    assert bootstrap_mean_interval([1,1,1])==[1.,1.]
    assert paired_effect([.7,.8],[.6,.9])['negative_transfer_fraction']==.5
    assert 'negative_transfer_fraction' not in paired_effect([.3,.4],[.5,.6],metric='balanced_nll')


def test_actual_budget_trapezoid_not_plain_mean():
    assert normalized_aubc([.5,.6,.7,.8,.9])==pytest.approx(.7)
    assert normalized_aubc([1.,0.,0.,0.,0.])==.125
    with pytest.raises(ValueError):normalized_aubc([.5,.6])


def test_sessions_then_seeds_then_participants_no_query_weighting():
    rows=[]
    for seed,values in ((1,[0.,1.]),(2,[1.])):
        for session,v in enumerate(values):
            rows.append(dict(cohort='x',method='m',candidate_id='',configuration_id='cfg',
                participant_id='P',seed=seed,session_id=str(session),budget=20,n_query=1 if v==0 else 10000,
                ba=v,balanced_nll=1.,balanced_brier=.2))
    value=participant_records(rows)[0];assert value['ba']==.75 and value['n_seeds']==2


def test_report_complete_formal_families_and_full_csv(tmp_path):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan);r=generate_report(path)
    assert r['status']=='COMPLETE' and r['round']==11
    assert r['primary_family_size']==6 and r['secondary_ablation_family_size']==18
    primary=[x for x in r['comparisons'] if x.get('holm_family_name')=='primary']
    assert len(primary)==6 and all(x['n_participants']==5 and x['n_seeds']==3 for x in primary)
    assert all(x['holm_family_size']==6 for x in primary)
    assert r['primary_family_complete'] and r['secondary_ablation_family_complete']
    import csv
    rows=list(csv.DictReader((Path(plan['output'])/'reports/session_metrics.csv').open()))
    assert len(rows)==len(plan['tasks'])*len(PEOPLE)*2*5
    assert all('incremental_prediction_seconds_per1000' in x for x in r['methods'])


def test_missing_primary_method_does_not_shrink_family(tmp_path):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan,skip=lambda t:t['method']=='r9_ea_anchored')
    r=generate_report(path);assert r['status']=='PARTIAL' and r['primary_family_size']==6
    assert not r['primary_family_complete']
    assert all(not x['significant_holm_05'] for x in r['comparisons'] if x.get('holm_family_name')=='primary')


def test_report_never_rehashes_full_prediction_payload(tmp_path):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan)
    folder=common.task_folder(plan,plan['tasks'][0]);(folder/'predictions.npz').write_bytes(b'archive corruption')
    assert not common.verify_result(plan,plan['tasks'][0])
    assert generate_report(path)['status']=='COMPLETE'
    (folder/'session_metrics.csv').write_bytes(b'compact corruption')
    r=generate_report(path);assert r['status']=='PARTIAL' and r['coverage']['invalid_compact_metrics']


def test_pairing_rejects_missing_cell_identity_and_missing_seed(tmp_path):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan);_,rows,_,_=load_completed(path)
    target=next(r for r in rows if r['method']=='operator' and r['budget']==20)
    rows.remove(target)
    with pytest.raises(PairingError):paired_vectors(rows,'holdout','operator','support_ea')
    _,rows,_,_=load_completed(path);next(r for r in rows if r['method']=='operator')['query_signature']='0'*64
    with pytest.raises(PairingError):paired_vectors(rows,'holdout','operator','support_ea',budget=4)
    _,rows,_,_=load_completed(path);rows=[r for r in rows if r['seed']!=SEEDS[-1]]
    with pytest.raises(PairingError):paired_vectors(MetricIndex(rows),'holdout','operator','support_ea',expected_seeds=SEEDS)


def test_development_uses_five_participants_three_seeds_all_budgets_and_no_gate(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan,gain=lambda t:-.01 if t['method'].startswith('operator') else 0.)
    d=development_decision(path)
    assert d['status']=='VALIDATED_SELECTION' and not d['performance_gate_applied']
    assert not d['check_population_split_applied'] and d['selected_configuration']['noise_strength']>0
    for method,x in d['selected_by_method'].items():
        assert x['participant_ids']==list(PEOPLE) and x['eligible_candidates']==2
        assert x['lopo_diagnostic']['status']=='DESCRIPTIVE_LOPO'
        assert len(x['lopo_diagnostic']['folds'])==5
        assert x['lopo_diagnostic']['pvalue'] is None
        assert set(x['operator_artifacts'])=={str(s) for s in SEEDS}
        assert all(set(v)=={str(b) for b in BUDGETS} for v in x['operator_artifacts'].values())
    assert d['selected_by_method']['operator_erm_tuned']['configuration']['training_loss']=='erm'
    assert d['selected_by_method']['operator_no_noise_tuned']['configuration']['noise_strength']==0


def test_one_se_uses_paired_curve_uncertainty_and_conservative_ties():
    records=[{'candidate_id':'small','configuration':{'feature_rank':4,'ridge':1.,'noise_strength':2.}},
             {'candidate_id':'large','configuration':{'feature_rank':32,'ridge':.001,'noise_strength':.1}}]
    selected,band=_one_se(records,[[0.,0.,0.,0.,0.],[-.1,.1,-.1,.1,.05]])
    assert selected==0 and band['best_candidate_id']=='large'


def test_lopo_choice_does_not_read_held_out_participant():
    records=[{'candidate_id':'A','configuration':{'feature_rank':4,'ridge':1.,'noise_strength':2.}},
             {'candidate_id':'B','configuration':{'feature_rank':32,'ridge':.001,'noise_strength':.1}}]
    d=_lopo(records,[[.01,.01,.01,.01,.01],[1.,-.2,-.2,-.2,-.2]],list(PEOPLE))
    assert d['folds'][0]['selected_candidate_id']=='A'
    assert d['folds'][0]['held_out_paired_gain']==.01
    assert d['independent_confirmation'] is False and d['ci95'] is None


def test_all_missing_records_default_and_retains_ready_training_artifacts(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan,skip=lambda t:t['family']!='operator_train')
    d=development_decision(path);assert d['status']=='DEFAULT_UNVALIDATED'
    assert d['selected_configuration']==asdict(Config()) and not d['performance_gate_applied']
    assert d['selected_operator_artifacts'] and len(d['selected_by_method']['operator']['excluded_candidates'])==2


def test_incomplete_candidate_excluded_as_whole_even_if_best_on_visible_subset(tmp_path):
    path,plan=make_plan(tmp_path)
    fill(plan,skip=lambda t:t.get('candidate_id')=='operator-small' and t['family']=='evaluate' and t['seed']==SEEDS[-1],
         gain=lambda t:.2 if t.get('candidate_id')=='operator-small' else .01 if t['method'].startswith('operator') else 0.)
    d=development_decision(path);main=d['selected_by_method']['operator']
    assert main['eligible_candidates']==1 and main['candidate_id']=='operator-default'
    assert any(x['candidate_id']=='operator-small' and x['whole_candidate_excluded'] for x in main['excluded_candidates'])


def test_target_outcomes_cannot_change_development_choice_or_enter_primary_grid(tmp_path):
    path,plan=make_plan(tmp_path,'campaign');fill(plan)
    first=development_decision(path)
    for t in plan['tasks']:
        if t['role']=='internal_holdout':complete(plan,t,gain=.3 if t['method']=='operator' else -.1)
    second=development_decision(path)
    assert second['selected_configuration']==first['selected_configuration']
    assert all(x['task_key'].startswith('dev-') for x in second['source_result_receipts'])
    r=generate_report(path)
    assert r['primary_family_size']==6
    assert all(x['cohort']=='holdout' for x in r['comparisons'] if x.get('holm_family_name')=='primary')
    assert r['coverage']['metric_expected_tasks']==sum(t['family']=='evaluate' for t in plan['tasks'])
    assert r['status']=='COMPLETE'


def test_selection_artifact_is_immutable_reused_and_has_actual_training_descriptors(tmp_path):
    path,plan=make_plan(tmp_path);fill(plan);task=next(t for t in plan['tasks'] if t['family']=='selection')
    folder=common.task_folder(plan,task)
    # Remove the generic completion fixture so the actual selection writer runs.
    for name in ('selection.json','result.json','completion.json'):(folder/name).unlink(missing_ok=True)
    first=run_selection_task(plan,task,folder);stamp=(folder/'selection.json').stat().st_mtime_ns
    again=run_selection_task(plan,task,folder)
    assert again==first and (folder/'selection.json').stat().st_mtime_ns==stamp
    entry=first['selected_operator_artifacts'][str(SEEDS[0])]['20']
    assert common.sha(entry['path'])==entry['sha256']
    assert Path(entry['training_result']['path']).is_file() and entry['source_identity']
    bad=common.read(folder/'selection.json');bad['selected_configuration']['ridge']=99.;common.atomic(folder/'selection.json',bad)
    with pytest.raises(ValueError,match='changed bytes'):run_selection_task(plan,task,folder)


def test_selection_never_allowed_to_choose_on_exploratory_plan(tmp_path):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan)
    with pytest.raises(ValueError,match='development'):development_decision(path)


def test_selected_training_join_ignores_same_config_matched_or_wrong_family(tmp_path):
    from paper3r11.selection import _training_artifacts
    path,plan=make_plan(tmp_path);fill(plan)
    config=asdict(Config());original=next(t for t in plan['tasks'] if t.get('candidate_id')=='operator-default' and t['family']=='operator_train')
    wrong=dict(original,key='matched-shuffled',method='operator_shuffled_pairs',candidate_id='matched',phase='exploratory')
    # This deliberately shares configuration/seed/budget while being a different
    # paired-future experiment. It must never replace the selected model.
    plan['tasks'].append(wrong);complete(plan,wrong)
    artifacts,_=_training_artifacts(plan,config,'operator','operator-default')
    assert artifacts[str(original['seed'])][str(original['budget'])]['task_key']==original['key']
    assert all(v['task_key']!='matched-shuffled' for bybudget in artifacts.values() for v in bybudget.values())


def test_unvalidated_selection_label_reaches_report_and_comparisons(tmp_path):
    path,plan=make_plan(tmp_path,'campaign')
    fill(plan,skip=lambda t:t['role']=='meta_validation')
    task=next(t for t in plan['tasks'] if t['family']=='selection');folder=common.task_folder(plan,task)
    decision=run_selection_task(plan,task,folder)
    assert decision['status']=='DEFAULT_UNVALIDATED'
    artifact={'path':'selection.json','sha256':common.sha(folder/'selection.json'),'bytes':(folder/'selection.json').stat().st_size}
    common.atomic(folder/'result.json',{'status':'COMPLETE','recipe_sha256':plan['recipe_sha256'],'task_key':task['key'],
        'task_identity':common.task_identity(plan,task),'scientific_scope':'SOFTWARE_TEST_ONLY','artifacts':[artifact],
        'selection_status':decision['status']})
    from paper3r11.platform.runtime import finish_task
    finish_task(plan,task,0)
    report=generate_report(path)
    assert report['selection']['status']=='DEFAULT_UNVALIDATED' and report['no_validated_selection']
    assert report['selection']['selection_evidence']=='NO_VALIDATED_SELECTION'
    assert all(x['method_selection_evidence_status']=='DEFAULT_UNVALIDATED' for x in report['comparisons'] if x.get('method')=='operator')


def test_completed_cache_avoids_reverification_of_all_planned_tasks(tmp_path,monkeypatch):
    path,plan=make_plan(tmp_path,'exploratory');fill(plan)
    import paper3r11.report as module
    # There is no selection task in this fixture. Reporting can use the durable
    # completion cache and must not reopen every task's completion seal.
    monkeypatch.setattr(module,'verify_result',lambda *a,**k:(_ for _ in ()).throw(AssertionError('all-task verification repeated')))
    report=generate_report(path)
    assert report['status']=='COMPLETE'


def test_confirmation_campaign_does_not_relabel_development_metrics_as_confirmation(tmp_path):
    from paper3r11.report import _task_rows
    path,plan=make_plan(tmp_path);task=next(t for t in plan['tasks'] if t['family']=='evaluate')
    complete(plan,task);result=common.read(common.task_folder(plan,task)/'result.json')
    # Exercise semantic annotation with real-data scope; role, not the name of
    # a mixed campaign, determines whether these are development observations.
    header=dict(plan,stage='confirmatory',synthetic_fixture=False)
    rows,_=_task_rows(header,task,result)
    assert rows and {r['analysis_scope'] for r in rows}=={'development'}
