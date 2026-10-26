"""Fresh, complete compact-metric reports; never open full prediction archives."""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

from .platform.common import (atomic, atomic_bytes, digest, load_plan, read, sha,
                              task_folder, utc, verify_descriptor, verify_result)
from .statistics import (BUDGETS, PRIMARY_FAMILY, SECONDARY_ABLATION_FAMILY, holm_fixed, normalized_aubc,
                         paired_effect)

METRICS=('ba','balanced_nll','balanced_brier')
IDENTITY_FIELDS=('query_signature','source_identity_sha256','data_identity_sha256')


class PairingError(ValueError):
    pass


def write_csv(path, rows):
    fields=sorted({key for row in rows for key in row})
    handle=io.StringIO(newline='')
    writer=csv.DictWriter(handle,fields);writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(list,dict)) else v
                         for k,v in row.items()})
    atomic_bytes(path,handle.getvalue().encode())


def _cast(row):
    output={}
    for key,value in row.items():
        if (key in ('participant_id','session_id','method','cohort','role','selection_split','candidate_id')
                or 'identity' in key or key.endswith(('sha256','signature'))):
            output[key]=value
        elif value in ('True','False','true','false'):
            output[key]=value.lower()=='true'
        else:
            try: output[key]=float(value)
            except (TypeError,ValueError): output[key]=value
    for key in ('seed','budget','n_query','n_class0','n_class1'):
        if key in output:
            if not isinstance(output[key],(int,float)) or not math.isfinite(output[key]) or int(output[key])!=output[key]:
                raise PairingError('Integer compact-metric field required: '+key)
            output[key]=int(output[key])
    return output


def _task_rows(plan, task, result):
    folder=task_folder(plan,task)
    artifacts=[x for x in result.get('artifacts',[]) if x.get('path')=='session_metrics.csv']
    if len(artifacts)!=1:
        raise PairingError('Evaluation result needs exactly one sealed session_metrics.csv')
    path=verify_descriptor(folder,artifacts[0])
    with path.open(newline='') as handle:
        rows=[_cast(row) for row in csv.DictReader(handle)]
    if not rows or len(rows)!=result.get('completed_session_budget_cells'):
        raise PairingError('Compact session count differs from scientific receipt')
    if (not isinstance(result.get('source_identity'),dict) or not result['source_identity']
            or not isinstance(result.get('catalog_identity'),dict) or not result['catalog_identity']):
        raise PairingError('Scientific receipt lacks a nonempty source/catalog identity')
    registry=plan.get('study_registry',plan.get('input_registry'))
    if isinstance(registry,dict) and result.get('registry_sha256')!=registry.get('sha256'):
        raise PairingError('Scientific result used a different registered study identity')
    source=digest(result['source_identity'])
    data=digest(result['catalog_identity'])
    cohort=task.get('cohort',task.get('role'))
    config=task.get('config',{})
    candidate=task.get('candidate_id','')
    config_id=digest(config)
    budgets=task.get('budgets',plan.get('budgets',list(BUDGETS)))
    seen=set();cells=defaultdict(set);query_sets=defaultdict(set)
    for row in rows:
        if (row.get('method')!=task.get('method') or row.get('seed')!=int(task['seed'])
                or row.get('budget') not in budgets or not row.get('participant_id')
                or not row.get('session_id')):
            raise PairingError('Session method, seed, population or budget differs from task')
        if row.get('cohort',cohort)!=cohort:
            raise PairingError('Compact metrics changed declared cohort')
        if row.get('role',task.get('role',cohort))!=task.get('role',cohort):
            raise PairingError('Compact metrics changed declared split role')
        for metric in METRICS:
            value=row.get(metric)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (metric!='balanced_nll' and value>1+1e-12):
                raise PairingError('Invalid compact outcome metric: '+metric)
        if min(row.get('n_class0',0),row.get('n_class1',0))<1 or row.get('n_query')!=row['n_class0']+row['n_class1']:
            raise PairingError('Balanced query metrics require declared nonempty class counts')
        if row.get('source_identity_sha256')!=source or row.get('data_identity_sha256')!=data:
            raise PairingError('Compact metric source/data identity differs from sealed task result')
        if not isinstance(row.get('query_signature'),str) or len(row['query_signature'])!=64:
            raise PairingError('Compact metrics need the fixed query UID SHA256')
        key=(row['participant_id'],row['session_id'],row['seed'],row['budget'])
        if key in seen: raise PairingError('Duplicated participant/session/seed/budget cell')
        seen.add(key)
        cells[key[:3]].add(row['budget']);query_sets[key[:3]].add(row['query_signature'])
        row.update(task_key=task['key'],cohort=cohort,role=task.get('role',cohort),
                   candidate_id=candidate,configuration_id=config_id,
                   analysis_scope='software_fixture' if plan.get('synthetic_fixture') else
                                  'independent_confirmation' if plan['stage']=='formal_confirmatory' else
                                  'exploratory_protocol_amendment' if plan.get('amendment') else
                                  'development' if plan['stage']=='mechanism' else 'exploratory_reused_cohort')
        if plan.get('amendment'):
            row['protocol_amendment_sha256']=plan['amendment']['sha256']
            row['selection_was_retrospectively_revised']=True
    if any(value!=set(budgets) for value in cells.values()) or any(len(value)!=1 for value in query_sets.values()):
        raise PairingError('Budget curve is incomplete or uses a changing future query set')
    if (len(cells)!=result.get('sessions') or len({x['participant_id'] for x in rows})!=result.get('participants')):
        raise PairingError('Compact population coverage differs from sealed scientific receipt')
    return rows, {'task_key':task['key'],'result_sha256':sha(folder/'result.json'),
                  'compact_metrics':artifacts[0],'rows':len(rows),'source_identity_sha256':source,
                  'data_identity_sha256':data}


def load_completed(plan_path):
    """Trust the immutable completion seal, then rehash only compact metrics.

    Full scientific artifacts were verified when sealing. Reporting does not
    rehash hundreds of prediction/checkpoint files, and never claims to do so.
    """
    # Read-only stage-1 reporting tests may exercise formal-shaped statistics
    # with an explicit recipe-bound SOFTWARE_TEST_ONLY fixture. This does not
    # authorize runtime/Slurm: its normal load_plan still rejects such plans.
    header=read(plan_path,{})
    software_report=bool(header.get('synthetic_fixture') and
                         header.get('stage') in ('formal_exploratory','formal_confirmatory'))
    _,plan=load_plan(plan_path,validate_science_stage=not software_report)
    rows=[];ledger=[];complete=[];missing=[];invalid=[]
    for task in plan['tasks']:
        if not verify_result(plan,task,sealed=True,verify_artifacts=False):
            missing.append(task['key']);continue
        complete.append(task['key'])
        if task.get('family')=='feature_reference':
            ledger.append({'task_key':task['key'],'result_sha256':sha(task_folder(plan,task)/'result.json'),
                           'family':'feature_reference'})
            continue
        result=read(task_folder(plan,task)/'result.json',{})
        try:
            new,proof=_task_rows(plan,task,result)
            rows.extend(new);ledger.append(proof)
        except (OSError,ValueError,KeyError,TypeError) as error:
            invalid.append({'task_key':task['key'],'reason':str(error)})
    coverage={'total_tasks':len(plan['tasks']),'sealed_complete_tasks':len(complete),
        'metric_expected_tasks':sum(t.get('family')!='feature_reference' for t in plan['tasks']),
        'metric_valid_tasks':sum('compact_metrics' in x for x in ledger),
        'complete_task_keys':sorted(complete),'complete_task_keys_sha256':digest(sorted(complete)),
        'missing_or_unsealed_task_keys':missing,'invalid_compact_metrics':invalid,
        'compact_session_rows':len(rows),'full_prediction_archives_rehashed_for_report':False}
    return plan,rows,coverage,ledger


def _method_rows(rows,cohort,method,budget,candidate=None):
    selected=[r for r in rows if r['cohort']==cohort and r['method']==method and r['budget']==budget
              and (candidate is None or r.get('candidate_id','')==candidate)]
    if len({r['configuration_id'] for r in selected})>1:
        raise PairingError('A comparison cannot mix multiple configurations of '+method)
    return selected


def paired_vectors(rows,cohort,method,reference,metric='ba',budget=20,*,candidate=None,
                   participants=None,expected_seeds=None):
    a=_method_rows(rows,cohort,method,budget,candidate)
    b=_method_rows(rows,cohort,reference,budget)
    if participants is not None:
        people=set(participants);a=[r for r in a if r['participant_id'] in people];b=[r for r in b if r['participant_id'] in people]
    def key(row): return row['participant_id'],row['session_id'],row['seed']
    left,right={key(r):r for r in a},{key(r):r for r in b}
    if len(left)!=len(a) or len(right)!=len(b):
        raise PairingError('Duplicated paired session cells')
    if not left or set(left)!=set(right):
        raise PairingError('Full participant/session/seed pairing is missing; intersection dropping is forbidden')
    for k in left:
        if any(left[k].get(field)!=right[k].get(field) for field in IDENTITY_FIELDS):
            raise PairingError('Paired query/source/data identity mismatch')
    people=sorted({k[0] for k in left}); av=[];bv=[]
    actual_seeds={k[2] for k in left}
    if expected_seeds is not None and actual_seeds!=set(expected_seeds):
        raise PairingError('Planned paired source seeds are missing or changed')
    expected_seeds=actual_seeds
    for person in people:
        seeds={k[2] for k in left if k[0]==person}
        if seeds!=expected_seeds:
            raise PairingError('Every participant must have all paired registered seeds')
        aa=[];bb=[]
        for seed in sorted(seeds):
            keys=[k for k in sorted(left) if k[0]==person and k[2]==seed]
            aa.append(float(np.mean([left[k][metric] for k in keys])))
            bb.append(float(np.mean([right[k][metric] for k in keys])))
        av.append(float(np.mean(aa)));bv.append(float(np.mean(bb)))
    return people,np.array(av),np.array(bv),{'session_seed_pairs':len(left),'n_seeds':len(expected_seeds)}


def participant_records(rows,*,per_seed=False):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['cohort'],row['method'],row['candidate_id'],row['configuration_id'],
                 row['participant_id'],row['seed'],row['budget'])].append(row)
    seed_rows=[]
    for keys,values in sorted(grouped.items()):
        cohort,method,candidate,config,person,seed,budget=keys
        numeric=set.intersection(*[{k for k,v in r.items() if isinstance(v,(int,float,bool)) and
            k not in ('seed','budget','n_query','n_class0','n_class1')} for r in values])
        negative_fields=[key for key in numeric if key.startswith('negative_transfer_vs_')]
        numeric-=set(negative_fields)
        seed_rows.append({'cohort':cohort,'method':method,'candidate_id':candidate,
            'configuration_id':config,'participant_id':person,'seed':seed,'budget':budget,
            'n_sessions':len(values),'n_query_total':sum(r['n_query'] for r in values),
            **{k:float(np.mean([r[k] for r in values])) for k in sorted(numeric)},
            **{key.replace('negative_transfer_vs_','negative_transfer_session_fraction_vs_'):
               float(np.mean([r[key] for r in values])) for key in negative_fields}})
        amendments={r.get('protocol_amendment_sha256') for r in values if r.get('protocol_amendment_sha256')}
        if amendments:
            if len(amendments)!=1 or any(r.get('protocol_amendment_sha256') not in amendments for r in values):
                raise PairingError('Participant records cannot mix original and amended protocols')
            seed_rows[-1].update(protocol_amendment_sha256=next(iter(amendments)),
                                 selection_was_retrospectively_revised=True)
    if per_seed: return seed_rows
    grouped=defaultdict(list)
    for row in seed_rows:
        grouped[(row['cohort'],row['method'],row['candidate_id'],row['configuration_id'],
                 row['participant_id'],row['budget'])].append(row)
    result=[]
    for keys,values in sorted(grouped.items()):
        cohort,method,candidate,config,person,budget=keys
        numeric=set.intersection(*[{k for k,v in r.items() if isinstance(v,(int,float,bool)) and
            k not in ('seed','budget','n_sessions','n_query_total')} for r in values])
        entry={'cohort':cohort,'method':method,'candidate_id':candidate,'configuration_id':config,
            'participant_id':person,'budget':budget,'n_seeds':len(values),
            'seeds':[r['seed'] for r in values],
            **{k:float(np.mean([r[k] for r in values])) for k in sorted(numeric)}}
        if 'anchor_ea_ba' in entry:
            entry['negative_transfer_participant_vs_ea']=int(entry['ba']<entry['anchor_ea_ba']-1e-12)
        amendments={r.get('protocol_amendment_sha256') for r in values if r.get('protocol_amendment_sha256')}
        if amendments:
            if len(amendments)!=1 or any(r.get('protocol_amendment_sha256') not in amendments for r in values):
                raise PairingError('Participant seeds cannot mix original and amended protocols')
            entry.update(protocol_amendment_sha256=next(iter(amendments)),
                         selection_was_retrospectively_revised=True)
        result.append(entry)
    return result


def _comparison(rows,cohort,spec,metric=None,budget=None,candidate=None,expected_seeds=None):
    metric=metric or spec['metric'];budget=budget or spec['budget']
    record={'name':spec['name'],'cohort':cohort,'method':spec['method'],
            'reference':spec['reference'],'metric':metric,'budget':budget,
            'benefit_direction':'higher' if metric=='ba' else 'lower',
            'candidate_id':candidate or '', 'status':'NOT_READY','n_participants':0}
    try:
        people,a,b,counts=paired_vectors(rows,cohort,spec['method'],spec['reference'],metric,budget,
                                       candidate=candidate,expected_seeds=expected_seeds)
        if len(people)<2:raise PairingError('At least two independent participants are required for inference')
        record.update(paired_effect(a,b,metric=metric),**counts,status='READY',participant_ids=people)
        if metric=='ba':record.update(effect_pp=100*record['effect_mean'],ci95_pp=[100*x for x in record['ci95']],
            difference_sd_pp=100*record['difference_sd'],difference_se_pp=100*record['difference_se'])
    except (PairingError,KeyError) as error:
        record['not_ready_reason']=str(error)
    return record


def _aubc(rows,cohort,spec,expected_seeds=None):
    records=[]
    try:
        by_budget=[];people0=None
        for budget in BUDGETS:
            people,a,b,counts=paired_vectors(rows,cohort,spec['method'],spec['reference'],'ba',budget,
                                           expected_seeds=expected_seeds)
            if people0 is not None and people!=people0:raise PairingError('AUBC population changed across budgets')
            people0=people;by_budget.append((a,b))
        aa=np.array([x[0] for x in by_budget]).T;bb=np.array([x[1] for x in by_budget]).T
        a=[normalized_aubc(x) for x in aa];b=[normalized_aubc(x) for x in bb]
        if len(a)<2:raise PairingError('AUBC needs at least two paired participants')
        return {'name':spec['name'],'cohort':cohort,'method':spec['method'],'reference':spec['reference'],
                'metric':'normalized_BA_AUBC','budget':'4_to_20','status':'READY',
                'inference_scope':'secondary_unadjusted_not_primary',**paired_effect(a,b)}
    except (PairingError,KeyError) as error:
        return {'name':spec['name'],'cohort':cohort,'metric':'normalized_BA_AUBC','budget':'4_to_20',
                'status':'NOT_READY','not_ready_reason':str(error)}


def generate_report(plan_path):
    plan,rows,coverage,ledger=load_completed(plan_path)
    folder=Path(plan['output'])/'reports';folder.mkdir(parents=True,exist_ok=True)
    cohorts=plan.get('cohorts') or sorted({t.get('cohort',t.get('role')) for t in plan['tasks']
                                         if t.get('family')!='feature_reference'})
    comparisons=[];primary=[];secondary_ablations=[]
    if plan['stage']=='mechanism':
        for cohort in cohorts:
            candidates=sorted({r['candidate_id'] for r in rows if r['cohort']==cohort and r['method']=='profile'})
            for candidate in candidates:
                spec={'name':'development_profile_vs_ea','method':'profile','reference':'support_ea','metric':'ba','budget':20}
                value=_comparison(rows,cohort,spec,candidate=candidate)
                value['inference_scope']='development_descriptive_not_confirmatory'
                comparisons.append(value)
        family=[]
        ablation_family=[]
    else:
        family=plan.get('comparison_family',PRIMARY_FAMILY)
        for cohort in cohorts:
            for spec in family:
                expected_seeds={int(t['seed']) for t in plan['tasks'] if
                    t.get('cohort',t.get('role'))==cohort and t.get('method') in (spec['method'],spec['reference'])}
                record=_comparison(rows,cohort,spec,expected_seeds=expected_seeds)
                record['inference_scope']=('confirmatory_primary' if plan['stage']=='formal_confirmatory' else
                    'exploratory_amended_protocol_reused_cohort' if plan.get('amendment') else
                    'prespecified_exploratory_reused_cohort')
                primary.append(record)
                for budget in BUDGETS:
                    for metric in METRICS:
                        if metric==spec['metric'] and budget==spec['budget']:continue
                        secondary=_comparison(rows,cohort,spec,metric,budget,expected_seeds=expected_seeds)
                        secondary['inference_scope']='secondary_unadjusted_not_primary'
                        comparisons.append(secondary)
                comparisons.append(_aubc(rows,cohort,spec,expected_seeds))
        adjusted=holm_fixed([r.get('pvalue') if r['status']=='READY' else None for r in primary])
        family_complete=all(r['status']=='READY' for r in primary)
        for record,value in zip(primary,adjusted):
            record.update(holm_pvalue=value,holm_family_size=len(primary),
                          holm_family_name='primary',
                          family_complete=family_complete,
                          significant_holm_05=bool(family_complete and value is not None and value<.05))
        ablation_family=plan.get('secondary_ablation_family',SECONDARY_ABLATION_FAMILY)
        for cohort in cohorts:
            for spec in ablation_family:
                expected_seeds={int(t['seed']) for t in plan['tasks'] if
                    t.get('cohort',t.get('role'))==cohort and t.get('method') in (spec['method'],spec['reference'])}
                record=_comparison(rows,cohort,spec,expected_seeds=expected_seeds)
                record['inference_scope']=('exploratory_amended_secondary_ablation_not_primary'
                    if plan.get('amendment') else 'prespecified_secondary_ablation_not_primary')
                secondary_ablations.append(record)
        adjusted=holm_fixed([r.get('pvalue') if r['status']=='READY' else None for r in secondary_ablations])
        complete=all(r['status']=='READY' for r in secondary_ablations)
        for record,value in zip(secondary_ablations,adjusted):
            record.update(holm_pvalue=value,holm_family_size=len(secondary_ablations),
                          holm_family_name='secondary_ablation',family_complete=complete,
                          significant_holm_05=bool(complete and value is not None and value<.05))
        comparisons=primary+secondary_ablations+comparisons
    participant=participant_records(rows)
    method_groups=defaultdict(list)
    for row in participant:
        method_groups[(row['cohort'],row['method'],row['candidate_id'],row['configuration_id'],row['budget'])].append(row)
    methods=[]
    for keys,values in sorted(method_groups.items()):
        cohort,method,candidate,configuration,budget=keys
        entry={'cohort':cohort,'method':method,'candidate_id':candidate,'configuration_id':configuration,
               'budget':budget,'n_participants':len(values)}
        for metric in METRICS:
            entry[metric]=float(np.mean([v[metric] for v in values]))
        for key in ('elapsed_seconds','fit_seconds','updated_query_fraction','n_valid_support',
                    'balanced_corrected_vs_ea','balanced_introduced_vs_ea',
                    'negative_transfer_session_fraction_vs_ea','negative_transfer_participant_vs_ea'):
            if all(key in v for v in values):
                label='negative_transfer_participant_fraction_vs_ea' if key=='negative_transfer_participant_vs_ea' else key
                entry[label]=float(np.mean([v[key] for v in values]))
        cells=[r for r in rows if (r['cohort'],r['method'],r['candidate_id'],r['configuration_id'],r['budget'])==keys]
        if cells and all(isinstance(r.get('incremental_prediction_seconds'),(int,float)) for r in cells):
            entry['incremental_prediction_seconds_per1000']=1000*sum(r['incremental_prediction_seconds'] for r in cells)/sum(r['n_query'] for r in cells)
        methods.append(entry)
    if plan.get('amendment'):
        for record in methods+comparisons:
            record.update(protocol_amendment_sha256=plan['amendment']['sha256'],
                          selection_was_retrospectively_revised=True)
    ready=(coverage['sealed_complete_tasks']==coverage['total_tasks'] and
           coverage['metric_valid_tasks']==coverage['metric_expected_tasks'] and
           not coverage['invalid_compact_metrics'] and all(r['status']=='READY' for r in primary+secondary_ablations))
    report={'schema_version':'paper3-r10-report-v1','round':10,'updated_utc':utc(),
        'status':'COMPLETE' if ready else 'PARTIAL','stage':plan['stage'],
        'scientific_scope':'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
        'completion_scope':'This signed stage plan only, not all R10 stages or independent confirmation',
        'recipe_sha256':plan['recipe_sha256'],'coverage':coverage,
        'primary_family':family,'primary_family_size':len(primary),
        'primary_family_complete':all(r['status']=='READY' for r in primary),
        'secondary_ablation_family':ablation_family,'secondary_ablation_family_size':len(secondary_ablations),
        'secondary_ablation_family_complete':all(r['status']=='READY' for r in secondary_ablations),
        'metric_unit':'sessions equal within seed, seeds equal within participant; participants independent',
        'cohorts':cohorts,'methods':methods,'comparisons':comparisons,'source_receipts':ledger,
        'bootstrap_signflip_ci_not_dual':True,
        'target_query_labels_used_for_model_selection':False,
        'exploratory_cohorts_reused_for_algorithm_design':plan['stage']!='formal_confirmatory',
        'true_future_risk_guarantee_claimed':False}
    if plan.get('amendment'):
        report.update(protocol_amendment=dict(plan['amendment']),
            retrospective_protocol_revision=True,independent_confirmation=False,
            original_mechanism_results_preserved=True,
            legacy_mechanism_rerun_tasks=plan.get('legacy_mechanism_rerun_tasks',0),
            selection_lineage=dict(plan.get('selection_lineage',{})),
            performance_gates_control_training=False)
    write_csv(folder/'session_metrics.csv',rows)
    write_csv(folder/'participant_seed_metrics.csv',participant_records(rows,per_seed=True))
    write_csv(folder/'participant_metrics.csv',participant)
    write_csv(folder/'method_summary.csv',methods)
    write_csv(folder/'comparisons.csv',comparisons)
    atomic(folder/'report.json',report)
    atomic(folder/'report-status.json',{'status':report['status'],'updated_utc':report['updated_utc'],
        'recipe_sha256':plan['recipe_sha256'],'report_sha256':sha(folder/'report.json'),
        'complete_task_keys_sha256':coverage['complete_task_keys_sha256'],
        'sealed_complete_tasks':coverage['sealed_complete_tasks'],'total_tasks':coverage['total_tasks'],
        'primary_family_complete':report['primary_family_complete']})
    return report
