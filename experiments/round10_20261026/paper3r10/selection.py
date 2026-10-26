"""Predeclared, development-only R10 selection and immutable PASS freeze."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from .algorithm import Config
from .platform.common import atomic, digest, locked, read, sha, utc
from .report import PairingError, load_completed, paired_vectors
from .statistics import PRIMARY_FAMILY,SECONDARY_ABLATION_FAMILY

CRITERIA={
    'selection_participants':4,'check_participants':4,
    'selection_ba_gain_minimum':.0025,
    'check_ba_gain_minimum':0.,'check_nll_excess_maximum':.01,
    'minimum_changed_query_fraction':.01,
    'maximum_certificate_violations':0,
    'maximum_global_numerical_fallback_fraction':.01,
    'maximum_incremental_prediction_seconds_per1000':5.,
    'query_probability_change_threshold':1e-4,
}


def _candidate_tasks(plan):
    candidates={}
    for task in plan['tasks']:
        if task.get('method')!='profile':continue
        key=task.get('candidate_id')
        if not isinstance(key,str) or not key:raise PairingError('Development profile task needs a fixed candidate_id')
        config=asdict(Config(**task['config']).validate())
        if (config['amplitude'] not in (.3,1.,3.) or config['evidence_fraction'] not in (.01,.1,.5)
                or config['mode']!='robust' or config['use_null_completion'] is not True):
            raise PairingError('Development candidate is outside the fixed nine-point grid')
        if key in candidates and candidates[key]!=config:
            raise PairingError('A development candidate_id maps to different configurations')
        candidates[key]=config
    pairs={(v['amplitude'],v['evidence_fraction']) for v in candidates.values()}
    if len(candidates)!=9 or pairs!={(a,c) for a in (.3,1.,3.) for c in (.01,.1,.5)}:
        raise PairingError('Exactly nine predeclared profile candidates are required')
    other={digest({k:v for k,v in config.items() if k not in ('amplitude','evidence_fraction')})
           for config in candidates.values()}
    if len(other)!=1:
        raise PairingError('Development grid changed an undeclared additional hyperparameter')
    return candidates


def _people(rows,plan):
    mapping={}
    for row in rows:
        if row['cohort']!='meta_validation' or row['role']!='meta_validation':
            raise PairingError('Selection may read only registered meta_validation outcomes')
        group=row.get('selection_split')
        if group not in ('selection','check'):
            raise PairingError('Every development participant needs its predetermined hash split')
        person=row['participant_id']
        if person in mapping and mapping[person]!=group:
            raise PairingError('A participant changes development selection/check assignment')
        mapping[person]=group
    selection=sorted(p for p,g in mapping.items() if g=='selection')
    check=sorted(p for p,g in mapping.items() if g=='check')
    if len(selection)!=4 or len(check)!=4 or set(selection)&set(check):
        raise PairingError('Exactly four selection and four disjoint check participants are required')
    salts={str(t.get('session_subset',plan.get('session_subset',{})).get('salt','r10-mechanism-v1'))
           for t in plan['tasks'] if t.get('method') in ('profile','support_ea')}
    if len(salts)!=1:raise PairingError('Development tasks use different participant hash salts')
    salt=next(iter(salts))
    ordered=sorted(mapping,key=lambda person:digest([salt,'participant',person]))
    if set(selection)!=set(ordered[:4]) or set(check)!=set(ordered[4:]):
        raise PairingError('Development selection/check marker differs from its predetermined metadata hash')
    return selection,check


def _finite(row,key,*,minimum=0.,maximum=None):
    value=row.get(key)
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<minimum or (maximum is not None and value>maximum):
        raise PairingError('Required finite development mechanism metric missing or invalid: '+key)
    return float(value)


def _global_mechanisms(rows):
    if not rows:raise PairingError('No profile mechanism records exist')
    queries=sum(r['n_query'] for r in rows)
    violations=sum(_finite(r,'certificate_violation_count') for r in rows)
    failures=sum(_finite(r,'numerical_fallback_count') for r in rows)
    seconds=sum(_finite(r,'incremental_prediction_seconds') for r in rows)
    if failures>queries or violations>queries:
        raise PairingError('Mechanism event counters exceed the evaluated query count')
    return {'queries':queries,'certificate_violations':violations,
            'numerical_fallback_count':failures,'numerical_fallback_fraction':failures/queries,
            'incremental_prediction_seconds':seconds,
            'incremental_prediction_seconds_per1000':1000*seconds/queries}


def _registry_binding(plan):
    descriptor=plan.get('study_registry',plan.get('input_registry'))
    if (not isinstance(descriptor,dict) or type(descriptor.get('bytes')) is not int
            or not 0<=descriptor['bytes']<=16*1024*1024
            or len(str(descriptor.get('sha256','')))!=64):
        raise PairingError('Development freeze requires an explicit bounded study-registry descriptor')
    path=Path(descriptor['path'])
    if not path.is_absolute():
        path=Path(plan['root'])/path
        if not path.resolve().is_relative_to(Path(plan['root']).resolve()):
            raise PairingError('Study registry escaped its declared root')
    if not path.is_file() or path.stat().st_size!=descriptor['bytes'] or sha(path)!=descriptor['sha256']:
        raise PairingError('Study-registry bytes changed after development planning')
    return descriptor['sha256']


def development_decision(plan_path):
    plan,rows,coverage,ledger=load_completed(plan_path)
    if plan['stage']!='mechanism':
        raise ValueError('Model selection is forbidden on target/exploratory/confirmation plans')
    folder=Path(plan['output'])/'reports';folder.mkdir(parents=True,exist_ok=True)
    result={'schema_version':'paper3-r10-development-decision-v1','status':'STOP','updated_utc':utc(),
        'scientific_scope':'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
        'source_plan_recipe_sha256':plan['recipe_sha256'],'source_plan_sha256':sha(plan_path),
        'reference_strategy':plan.get('reference_strategy'),
        'criteria':CRITERIA,'coverage':coverage,'source_result_receipts':ledger,
        'selection_roles':['meta_train','meta_validation'],
        'target_query_labels_used_for_selection':False,'development_query_labels_used':True,
        'rule':'one-SE band on selection-four B20 participant BA; c descending, amplitude ascending, candidate_id lexical',
        'check_role':'PASS/STOP only; never rerank candidates',
        'comparison_family':plan.get('comparison_family',PRIMARY_FAMILY),
        'secondary_ablation_family':plan.get('secondary_ablation_family',SECONDARY_ABLATION_FAMILY),
        'true_future_risk_guaranteed':False}
    try:
        result['study_registry_sha256']=_registry_binding(plan)
        if (coverage['sealed_complete_tasks']!=coverage['total_tasks'] or
                coverage['metric_valid_tasks']!=coverage['metric_expected_tasks'] or
                coverage['invalid_compact_metrics']):
            raise PairingError('All development tasks and compact identities must be sealed before selection')
        candidates=_candidate_tasks(plan)
        outcomes=[r for r in rows if r['method']=='profile' or r['method']=='support_ea']
        select_people,check_people=_people(outcomes,plan)
        summaries=[]
        for key,config in sorted(candidates.items()):
            all_people,_,_,_=paired_vectors(outcomes,'meta_validation','profile','support_ea',
                                           'ba',20,candidate=key)
            if all_people!=sorted(select_people+check_people):
                raise PairingError('A candidate omitted a predetermined development participant')
            people,a,b,_=paired_vectors(outcomes,'meta_validation','profile','support_ea',
                                       'ba',20,candidate=key,participants=select_people)
            if people!=select_people:raise PairingError('A candidate omitted selection participants')
            # Check-role identities are validated above, but their outcome
            # values are never used to order or rank these candidates.
            summaries.append({'candidate_id':key,'configuration':config,'mean_ba':float(a.mean()),
                'se_ba':float(a.std(ddof=1)/np.sqrt(len(a))),
                'paired_ba_gain':float((a-b).mean()),'n_participants':len(a)})
        best=min(summaries,key=lambda x:(-x['mean_ba'],x['candidate_id']))
        cutoff=best['mean_ba']-best['se_ba']
        eligible=[x for x in summaries if x['mean_ba']>=cutoff-1e-12]
        selected=min(eligible,key=lambda x:(-x['configuration']['evidence_fraction'],
                                           x['configuration']['amplitude'],x['candidate_id']))
        candidate=selected['candidate_id']
        people,a,b,_=paired_vectors(outcomes,'meta_validation','profile','support_ea',
                                   'ba',20,candidate=candidate,participants=check_people)
        if people!=check_people:raise PairingError('The frozen candidate omitted check participants')
        _,nll,nll0,_=paired_vectors(outcomes,'meta_validation','profile','support_ea',
                                   'balanced_nll',20,candidate=candidate,participants=check_people)
        selected_b20=[r for r in outcomes if r['method']=='profile' and r['candidate_id']==candidate and r['budget']==20]
        denominator=sum(r['n_query'] for r in selected_b20)
        changed=sum(r['n_query']*_finite(r,'updated_query_fraction',maximum=1.) for r in selected_b20)/denominator
        mechanisms=_global_mechanisms([r for r in outcomes if r['method']=='profile'])
        check_gain=float((a-b).mean());check_nll=float((nll-nll0).mean())
        gates={
            'selection_ba_gain':selected['paired_ba_gain']>=CRITERIA['selection_ba_gain_minimum']-1e-12,
            'check_ba_noninferiority':check_gain>=CRITERIA['check_ba_gain_minimum']-1e-12,
            'check_nll_guard':check_nll<=CRITERIA['check_nll_excess_maximum']+1e-12,
            'nontrivial_updates':changed>=CRITERIA['minimum_changed_query_fraction']-1e-12,
            'certificate_violations':mechanisms['certificate_violations']==0,
            'global_numerical_fallback':mechanisms['numerical_fallback_fraction']<=CRITERIA['maximum_global_numerical_fallback_fraction']+1e-12,
            'incremental_prediction_cost':mechanisms['incremental_prediction_seconds_per1000']<=CRITERIA['maximum_incremental_prediction_seconds_per1000']+1e-12,
            'pairing_and_identity':True}
        result.update(status='PASS' if all(gates.values()) else 'STOP',gates=gates,
            selection_participant_ids=select_people,check_participant_ids=check_people,
            candidates=summaries,best_selection_candidate=best['candidate_id'],one_se_cutoff=cutoff,
            selected_candidate_id=candidate,selected_configuration=selected['configuration'],
            selected_selection_ba_gain=selected['paired_ba_gain'],selected_check_ba_gain=check_gain,
            selected_check_nll_excess=check_nll,updated_query_fraction=changed,
            global_mechanisms=mechanisms,
            stop_reasons=[key for key,value in gates.items() if not value])
    except (PairingError,KeyError,ValueError) as error:
        result.update(stop_reasons=[str(error)],gates={'pairing_and_identity':False})
    identity_body={k:v for k,v in result.items() if k not in ('updated_utc','decision_identity')}
    result['decision_identity']=digest(identity_body)
    atomic(folder/'development-decision.json',result)
    return result


def freeze_development(plan_path,output=None):
    decision=development_decision(plan_path)
    if decision['status']!='PASS':
        raise ValueError('R10 development decision is STOP; no formal algorithm may be frozen')
    plan,_,_,_=load_completed(plan_path)
    root=Path(plan['root']).resolve()
    path=Path(output or root/'runs/round10_preparation/freeze.json').expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError('Immutable R10 freeze must be stored under the declared study root')
    selected=decision['selected_configuration']
    allowed={
        'profile_map':{**selected,'mode':'map','evidence_fraction':0.},
        'profile_no_completion':{**selected,'use_null_completion':False},
        'profile_quadratic':dict(selected),
        'profile_variance_gate':dict(selected),
        'profile_frozen_anchor':dict(selected)}
    receipt={'schema_version':'paper3-r10-development-freeze-v1','status':'FROZEN','created_utc':utc(),
        'scientific_scope':decision['scientific_scope'],
        'decision_identity':decision['decision_identity'],
        'source_plan_recipe_sha256':plan['recipe_sha256'],'source_plan_sha256':sha(plan_path),
        'source_code_manifest_sha256':plan['code_manifest_sha256'],
        'code_manifest_sha256':plan['code_manifest_sha256'],
        'study_registry_sha256':decision['study_registry_sha256'],
        'reference_strategy':plan.get('reference_strategy'),
        'source_result_receipts':decision['source_result_receipts'],
        'selected_candidate_id':decision['selected_candidate_id'],'selected_configuration':selected,
        'allowed_ablations':allowed,'comparison_family':decision['comparison_family'],
        'secondary_ablation_family':decision['secondary_ablation_family'],
        'selection_roles':['meta_train','meta_validation'],'development_query_labels_used':True,
        'target_query_labels_used_for_selection':False,'independent_confirmation_has_not_run':True,
        'exploratory_cohorts':['holdout','pan2023','weibo2014'],
        'selection_participant_ids':decision['selection_participant_ids'],
        'check_participant_ids':decision['check_participant_ids'],
        'development_gate_criteria':CRITERIA,'development_gates':decision['gates'],
        'true_future_risk_guaranteed':False}
    with locked(path.with_name(path.name+'.lock')):
        if path.exists():
            original=read(path,{})
            if ({k:v for k,v in original.items() if k!='created_utc'} !=
                    {k:v for k,v in receipt.items() if k!='created_utc'}):
                raise ValueError('An existing immutable R10 freeze differs; create a new research round')
        else:
            atomic(path,receipt)
    return {'path':str(path.relative_to(root)),'sha256':sha(path),'bytes':path.stat().st_size}
