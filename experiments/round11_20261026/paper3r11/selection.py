"""Development-only R11 grid selection; no performance veto or target tuning.

Every candidate is paired on the entire available registered meta-validation
population, all five budgets and all planned seeds. A missing candidate is
excluded as a whole. If none is complete, a predeclared default is recorded and
unrelated scientific work continues. LOPO is a selection-optimism diagnostic,
not independent confirmation or a future-risk certificate.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np

from .algorithm import Config
from .platform.common import atomic,digest,load_plan,locked,read,sha,task_folder,utc,verify_descriptor,verify_result
from .report import MetricIndex,PairingError,load_completed,paired_vectors,write_csv
from .statistics import BUDGETS,PRIMARY_FAMILY,SECONDARY_ABLATION_FAMILY,normalized_aubc

SELECTED_METHODS=('operator','operator_no_noise_tuned','operator_erm_tuned')
DEVELOPMENT_ROLES=('meta_train','meta_validation')


def _canonical(config):
    return asdict(Config(**dict(config)).validate())


def _default(plan,method):
    spec=plan.get('selection_config',{})
    config=spec.get('default_by_method',{}).get(method)
    if config is None:
        config=spec.get('default_configuration',plan.get('default_configuration',asdict(Config())))
        config=_canonical(config)
        if method!='operator':
            config=dict(config,noise_strength=0.,noise_type='none',training_loss='erm' if method=='operator_erm_tuned' else 'majorizer')
    return _canonical(config)


def _candidate_tasks(plan,method):
    candidates={};invalid={}
    for task in plan['tasks']:
        if task.get('family') not in ('evaluate','evaluation','mechanism') or task.get('method')!=method:
            continue
        if task.get('role')!='meta_validation' or task.get('cohort',task.get('role'))!='meta_validation':
            continue
        candidate=task.get('candidate_id')
        if not isinstance(candidate,str) or not candidate:
            invalid[task['key']]='Development operator evaluation requires a declared candidate_id';continue
        try:
            config=_canonical(task.get('config',{}))
            expected_noise='permutation' if method=='operator' else 'none'
            if config['operator_mode']!='directed' or config['noise_type']!=expected_noise:
                raise ValueError('Tuned candidate changed its declared directed operator or noise family')
            if method=='operator' and (config['noise_strength']<=0 or config['training_loss']!='majorizer'):
                raise ValueError('Main candidates require positive noise and the majorizer objective')
            if method!='operator' and (config['noise_strength']!=0 or
                    config['training_loss']!=('erm' if method=='operator_erm_tuned' else 'majorizer')):
                raise ValueError('Independent unpenalized control candidate changed its declared objective')
            if candidate in candidates and candidates[candidate]!=config:
                raise ValueError('Candidate identifier maps to inconsistent seed configurations')
            candidates[candidate]=config
        except (TypeError,ValueError) as error:
            invalid[candidate]=str(error)
    for candidate in invalid:candidates.pop(candidate,None)
    return candidates,invalid


def _preference(record):
    c=record['configuration']
    return (c['feature_rank'],-c['ridge'],-c['noise_strength'],record['candidate_id'])


def _one_se(records,differences):
    """Maximize participant mean paired nAUBC, then use the best paired SE band."""
    values=np.asarray(differences,dtype=np.float64)
    means=values.mean(1)
    errors=values.std(1,ddof=1)/np.sqrt(values.shape[1]) if values.shape[1]>1 else np.zeros(len(records))
    best=min(range(len(records)),key=lambda i:(-means[i],records[i]['candidate_id']))
    cutoff=float(means[best]-errors[best])
    eligible=[i for i in range(len(records)) if means[i]>=cutoff-1e-12]
    chosen=min(eligible,key=lambda i:_preference(records[i]))
    return chosen,{'best_candidate_id':records[best]['candidate_id'],'best_mean_paired_gain':float(means[best]),
                   'best_paired_standard_error':float(errors[best]),'one_se_cutoff':cutoff,
                   'eligible_candidate_ids':[records[i]['candidate_id'] for i in eligible]}


def _curve(index,candidate,method,expected_seeds):
    people0=None;left=[];right=[];counts=[]
    for budget in BUDGETS:
        people,a,b,count=paired_vectors(index,'meta_validation',method,'support_ea','ba',budget,
                                      candidate=candidate,expected_seeds=expected_seeds)
        if people0 is not None and people!=people0:
            raise PairingError('A candidate changes the participant population across budgets')
        people0=people;left.append(a);right.append(b);counts.append(count)
    if not people0:raise PairingError('No paired development participants')
    a=np.asarray(left).T;b=np.asarray(right).T
    av=np.asarray([normalized_aubc(x) for x in a]);bv=np.asarray([normalized_aubc(x) for x in b])
    return people0,av,bv,counts


def _lopo(records,difference,people):
    if len(people)<3:
        return {'status':'NOT_AVAILABLE','reason':'LOPO needs at least three development participants',
                'independent_confirmation':False,'folds':[]}
    folds=[];d=np.asarray(difference)
    for j,person in enumerate(people):
        selected,band=_one_se(records,np.delete(d,j,axis=1))
        folds.append({'held_out_participant':person,'selected_candidate_id':records[selected]['candidate_id'],
            'training_mean_paired_gain':float(np.delete(d[selected],j).mean()),
            'held_out_paired_gain':float(d[selected,j]),'one_se_cutoff':band['one_se_cutoff']})
    selected,_=_one_se(records,d)
    heldout=float(np.mean([f['held_out_paired_gain'] for f in folds]))
    return {'status':'DESCRIPTIVE_LOPO','independent_confirmation':False,'folds':folds,
        'mean_held_out_paired_gain':heldout,'full_selection_apparent_gain':float(d[selected].mean()),
        'apparent_minus_lopo_gain':float(d[selected].mean()-heldout),
        'pvalue':None,'ci95':None,
        'limitation':'Folds share training participants; no independent-confirmation p value or future-risk guarantee'}


def _training_artifacts(plan,configuration,method,candidate):
    result={};missing=[]
    declared={key for task in plan['tasks'] if task.get('family') in ('evaluate','evaluation','mechanism')
              and task.get('role')=='meta_validation' and task.get('method')==method
              and task.get('candidate_id')==candidate
              for key in task.get('operator_dependencies',{}).values()}
    for task in plan['tasks']:
        if (task.get('family')!='operator_train' or task.get('method')!=method
                or task.get('candidate_id')!=candidate or (declared and task['key'] not in declared)):
            continue
        try:
            if _canonical(task.get('config',{}))!=configuration:continue
        except (TypeError,ValueError):continue
        budget=task.get('budget')
        if budget is None:
            values=task.get('budgets',[])
            budget=values[0] if len(values)==1 else None
        if budget not in BUDGETS:continue
        seed=str(task['seed']);key=str(budget)
        if not verify_result(plan,task,sealed=True,verify_artifacts=False):
            missing.append(task['key']);continue
        folder=task_folder(plan,task)
        receipt=read(folder/'result.json',{})
        checkpoints=[a for a in receipt.get('artifacts',[]) if a.get('path')=='operator.npz']
        metadata=[a for a in receipt.get('artifacts',[]) if a.get('path')=='operator.json']
        if len(checkpoints)!=1 or len(metadata)!=1:missing.append(task['key']);continue
        checkpoint=verify_descriptor(folder,checkpoints[0]);meta=verify_descriptor(folder,metadata[0])
        entry={'path':str(checkpoint),'sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size,
            'task_key':task['key'],'operator_identity':receipt.get('operator_identity'),
            'source_identity':receipt.get('source_identity'),'registry_sha256':receipt.get('registry_sha256'),
            'budget':budget,'seed':int(seed),
            'metadata':{'path':str(meta),'sha256':sha(meta),'bytes':meta.stat().st_size},
            'training_result':{'path':str(folder/'result.json'),'sha256':sha(folder/'result.json'),
                               'bytes':(folder/'result.json').stat().st_size}}
        previous=result.setdefault(seed,{}).get(key)
        if previous and previous['operator_identity']!=entry['operator_identity']:
            raise PairingError('Two different trained operator identities map to one selected seed/budget')
        result[seed][key]=entry
    return result,missing


def _select_method(plan,index,rows,method):
    candidates,invalid=_candidate_tasks(plan,method)
    seeds={int(t['seed']) for t in plan['tasks'] if t.get('role')=='meta_validation'
           and t.get('method') in (method,'support_ea')}
    records=[];difference=[];people0=None;excluded=[{'candidate_id':k,'reason':v} for k,v in invalid.items()]
    for candidate,config in sorted(candidates.items()):
        try:
            people,a,b,counts=_curve(index,candidate,method,seeds)
            if people0 is not None and people!=people0:
                raise PairingError('A complete candidate differs from the fixed EA participant registry')
            people0=people;delta=a-b
            records.append({'candidate_id':candidate,'configuration':config,'n_participants':len(people),
                'mean_normalized_ba_aubc':float(a.mean()),'baseline_normalized_ba_aubc':float(b.mean()),
                'mean_paired_gain':float(delta.mean()),
                'paired_standard_error':float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 0.,
                'completed_seeds':sorted(seeds),'paired_cells_by_budget':counts})
            difference.append(delta)
        except (PairingError,ValueError,KeyError,TypeError) as error:
            excluded.append({'candidate_id':candidate,'reason':str(error),'whole_candidate_excluded':True})
    if records:
        choice,band=_one_se(records,difference);selected=records[choice]
        config=selected['configuration'];candidate=selected['candidate_id'];status='VALIDATED_SELECTION'
        diagnostic=_lopo(records,difference,people0)
    else:
        config=_default(plan,method)
        candidate=next((k for k,v in sorted(candidates.items()) if v==config),
                       plan.get('selection_config',{}).get('default_candidate_ids',{}).get(method,'default-unvalidated-'+method))
        status='DEFAULT_UNVALIDATED';band={};people0=[]
        diagnostic={'status':'NOT_AVAILABLE','reason':'No complete paired candidate','independent_confirmation':False,'folds':[]}
    artifacts,missing=_training_artifacts(plan,config,method,candidate)
    return {'status':status,'configuration':config,'candidate_id':candidate,
        'operator_artifacts':artifacts,'missing_selected_training_tasks':missing,
        'candidate_summaries':records,'excluded_candidates':excluded,'planned_candidates':len(candidates)+len(invalid),
        'eligible_candidates':len(records),'participant_ids':people0,'selection_band':band,'lopo_diagnostic':diagnostic,
        'performance_gate_applied':False,'negative_development_gain_does_not_stop_campaign':True}


def _registry_binding(plan):
    descriptor=plan.get('study_registry',plan.get('input_registry'))
    if not isinstance(descriptor,dict):raise PairingError('A selected model requires its source/data registry descriptor')
    path=verify_descriptor(plan['root'],descriptor)
    return sha(path)


def development_decision(plan_path):
    plan,rows,coverage,ledger=load_completed(plan_path)
    if plan['stage'] not in ('development','campaign','exploratory','confirmatory'):
        raise ValueError('Model selection requires a predeclared research plan')
    if plan['stage'] in ('exploratory','confirmatory'):
        declared_development=[t for t in plan['tasks'] if t.get('role')=='meta_validation'
            and t.get('cohort')=='meta_validation' and t.get('phase')=='development']
        if (not any(t.get('family')=='selection' for t in declared_development)
                or not any(t.get('family') in ('evaluate','evaluation','mechanism') for t in declared_development)):
            raise ValueError('Model selection requires a declared development evaluation and selection DAG')
    development=[r for r in rows if r['role']=='meta_validation' and r['cohort']=='meta_validation']
    # Target rows may already exist in a resumed campaign; they are excluded
    # before every candidate computation, diagnostic and source-receipt join.
    index=MetricIndex(development)
    selected={m:_select_method(plan,index,development,m) for m in SELECTED_METHODS}
    main=selected['operator']
    development_keys={t['key'] for t in plan['tasks'] if t.get('role')=='meta_validation'
                      and t.get('family') in ('evaluate','evaluation','mechanism')}
    result={'schema_version':'paper3-r11-development-selection-v1','status':main['status'],'updated_utc':utc(),
        'scientific_scope':'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
        'source_plan_recipe_sha256':plan['recipe_sha256'],'source_plan_sha256':sha(plan_path),
        'source_code_manifest_sha256':plan['code_manifest_sha256'],'code_manifest_sha256':plan['code_manifest_sha256'],
        'study_registry_sha256':_registry_binding(plan),'reference_strategy':plan.get('reference_strategy'),
        'selected_configuration':main['configuration'],'selected_candidate_id':main['candidate_id'],
        'selected_operator_artifacts':main['operator_artifacts'],'selected_by_method':selected,
        'selection_roles':list(DEVELOPMENT_ROLES),'target_query_labels_used_for_selection':False,
        'development_query_labels_used':True,'check_population_split_applied':False,
        'selection_metric':'participant_mean_paired_normalized_BA_AUBC_B4_to_B20',
        'rule':'Best mean paired nAUBC minus that candidate paired SE; prefer lower feature_rank, higher ridge, higher noise_strength, lexical candidate id',
        'performance_gate_applied':False,'no_complete_candidate_action':'Use the predeclared default; retain DEFAULT_UNVALIDATED label; other methods continue',
        'coverage':{k:v for k,v in coverage.items() if k not in ('complete_task_keys','missing_or_unsealed_task_keys')},
        'source_result_receipts':[x for x in ledger if x['task_key'] in development_keys],
        'comparison_family':plan.get('comparison_family',PRIMARY_FAMILY),
        'secondary_ablation_family':plan.get('secondary_ablation_family',SECONDARY_ABLATION_FAMILY),
        'true_future_risk_guaranteed':False,'independent_confirmation':False}
    result['decision_identity']=digest({k:v for k,v in result.items() if k!='updated_utc'})
    folder=Path(plan['output'])/'reports';folder.mkdir(parents=True,exist_ok=True)
    atomic(folder/'development-decision.json',result)
    for method,entry in selected.items():
        write_csv(folder/('selection-candidates-'+method+'.csv'),entry['candidate_summaries'])
        write_csv(folder/('selection-exclusions-'+method+'.csv'),entry['excluded_candidates'])
        write_csv(folder/('selection-lopo-'+method+'.csv'),entry['lopo_diagnostic']['folds'])
    return result


def run_selection_task(plan,task,folder):
    """Write a resumable immutable selection artifact; worker seals its result."""
    if task.get('family')!='selection' or task.get('role','meta_validation') not in ('meta_validation','selection'):
        raise ValueError('Only the predeclared selection task may choose operator configurations')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'selection.json'
    with locked(folder/'selection.lock'):
        if path.is_file():
            prior=read(path,{})
            expected=digest({k:v for k,v in prior.items() if k not in ('updated_utc','decision_identity')})
            if (prior.get('source_plan_recipe_sha256')!=plan['recipe_sha256'] or prior.get('decision_identity')!=expected):
                raise ValueError('An existing selection artifact belongs to another plan or changed bytes')
            return prior
        source=Path(plan['output'])/'plan.json'
        if not source.is_file():
            # Tests and direct worker use may supply the original plan path.
            source=Path(plan.get('plan_path',''))
        if not source.is_file():
            raise ValueError('The exact prepared plan.json is needed to record model selection')
        decision=development_decision(source)
        atomic(path,decision)
        return decision


def execute_selection(plan_path,task=None,output=None):
    from .platform.common import load_plan
    _,plan=load_plan(plan_path)
    selected=task or next((t for t in plan['tasks'] if t.get('family')=='selection'),None)
    if selected is None:raise ValueError('No predeclared selection task exists')
    folder=Path(output or task_folder(plan,selected))
    return run_selection_task(plan,selected,folder)


def freeze_development(plan_path,output=None):
    """Compatibility API: choose/freeze without a performance PASS/STOP gate."""
    decision=execute_selection(plan_path)
    _,plan=load_plan(plan_path)
    task=next(t for t in plan['tasks'] if t.get('family')=='selection')
    path=task_folder(plan,task)/'selection.json'
    if output is not None:
        path=Path(output).expanduser().resolve()
        with locked(path.with_name(path.name+'.lock')):
            if path.is_file() and read(path,{})!=decision:
                raise ValueError('An existing immutable selection copy differs')
            if not path.exists():atomic(path,decision)
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'selection_status':decision['status']}
