"""Finite, extensible R11 research DAG; standard-library-only preparation."""
from __future__ import annotations
import copy
from pathlib import Path
from .platform.common import atomic_bytes,digest,encoded,read,sha,sign_plan

PACKAGE=Path(__file__).resolve().parents[1]
SEEDS=(20261018,20261019,20261020)
BUDGETS=(4,8,12,16,20)
RANKS=(4,8,16,32,64)
NOISE_STRENGTHS=(.03,.1,.3,1.,3.,10.,30.,100.)
RIDGES=(1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,.003,.01,.03,.1)
BASELINES=('frozen','support_ea','head_bias','head_rank4','head_full','sgd10','sgd100',
 'r3_protected','r4_xfit_select_stack','r8_fiber_bounded','r9_raw_anchored','r9_ea_anchored','profile')
MATCHED_CONTROLS=('operator_no_noise','operator_symmetric','operator_psd','operator_diagonal',
 'operator_identity','operator_feature_noise','operator_fisher_noise','operator_shuffled_pairs','operator_erm')
TUNED_CONTROLS=('operator_no_noise_tuned','operator_erm_tuned')
METHODS=BASELINES+('operator',)+MATCHED_CONTROLS+TUNED_CONTROLS
NEEDS_ARTIFACTS=frozenset(('sgd10','sgd100','r3_protected','r4_xfit_select_stack',
 'r8_fiber_bounded','r9_raw_anchored','r9_ea_anchored'))
PRIMARY_FAMILY=[{'name':'operator_vs_'+m,'method':'operator','reference':m,'metric':'ba','budget':20}
 for m in ('support_ea','sgd100','r9_raw_anchored','r9_ea_anchored','profile','operator_erm_tuned')]
SECONDARY_FAMILY=[{'name':'operator_vs_'+m,'method':'operator','reference':m,'metric':metric,'budget':20}
 for m in MATCHED_CONTROLS for metric in ('ba','balanced_nll')]


def describe(path,base=None):
 path=Path(path).expanduser().resolve(strict=True)
 name=str(path.relative_to(Path(base).resolve())) if base and path.is_relative_to(Path(base).resolve()) else str(path)
 return {'path':name,'sha256':sha(path),'bytes':path.stat().st_size}


def resolve_descriptor(root,item):
 p=Path(item['path']).expanduser();p=p if p.is_absolute() else Path(root)/p
 if not p.is_file() or p.stat().st_size!=item.get('bytes') or sha(p)!=item.get('sha256'):
  raise ValueError('Declared input is absent or changed: '+str(p))
 return p.resolve()


def production_manifest(package=PACKAGE):
 package=Path(package)
 paths=list((package/'paper3r11').rglob('*.py'))
 paths += [package/n for n in ('START.py','paper3r11/baseline_registry.json',
                             'configs/algorithm.default.json','configs/profile_r10.default.json')]
 return {p.relative_to(package).as_posix():sha(p) for p in sorted(paths)
         if p.is_file() and '__pycache__' not in p.parts}


def default_configuration():
 return read(PACKAGE/'configs/algorithm.default.json')


def development_candidates(grid=None):
 grid=grid or {'feature_ranks':RANKS,'noise_strengths':NOISE_STRENGTHS,'ridges':RIDGES}
 ranks=tuple(grid.get('feature_ranks',RANKS));etas=tuple(grid.get('noise_strengths',NOISE_STRENGTHS));ridges=tuple(grid.get('ridges',RIDGES))
 if (not ranks or not etas or not ridges or any(type(x) is not int or not 1<=x<=64 for x in ranks)
     or any(not isinstance(x,(int,float)) or x<=0 for x in etas+ridges)):
  raise ValueError('A finite positive rank/noise/ridge development grid is required')
 candidates=[]
 for rank in ranks:
  for ridge in ridges:
   for eta in etas:
    cfg=dict(default_configuration(),feature_rank=rank,ridge=float(ridge),noise_strength=float(eta))
    candidates.append({'method':'operator','candidate_family':'main','config':cfg})
   for method,loss in (('operator_no_noise_tuned','majorizer'),('operator_erm_tuned','erm')):
    cfg=dict(default_configuration(),feature_rank=rank,ridge=float(ridge),noise_strength=0.,noise_type='none',training_loss=loss)
    candidates.append({'method':method,'candidate_family':method,'config':cfg})
 for item in candidates:item['candidate_id']=item['method']+'-'+digest(item['config'])[:14]
 if len({v['candidate_id'] for v in candidates})!=len(candidates):
  raise ValueError('Development grid contains duplicate configurations')
 return candidates


def _descriptors(value):
 if isinstance(value,dict):
  if {'path','bytes','sha256'}<=set(value):return [value]
  return [x for y in value.values() for x in _descriptors(y)]
 if isinstance(value,list):return [x for y in value for x in _descriptors(y)]
 return []


def _task(registry,registry_sha,*,family,method,seed,cohort,role,config=None,budgets=BUDGETS,
          requires=(),reference=None,phase='development',candidate=None,operators=None,selection=None):
 cfg=copy.deepcopy(config or {})
 value={'family':family,'kind':family,'method':method,'seed':int(seed),'cohort':cohort,'role':role,
        'phase':phase,'config':cfg,'method_config':copy.deepcopy(cfg),'budgets':list(budgets),
        'requires':list(requires),'registry_sha256':registry_sha,
        'input_identities':{'registry_sha256':registry_sha}}
 if reference:value['reference_dependency']=reference
 if selection:value['selection_dependency']=selection
 if operators:value['operator_dependencies']={str(k):v for k,v in operators.items()}
 if candidate:
  value['candidate_id']=candidate['candidate_id'];value['candidate_family']=candidate['candidate_family']
 source=registry.get('source_by_seed',{}).get(str(seed),{})
 value['input_identities'].update({k+'_sha256':v['sha256'] for k,v in source.items()
                                  if isinstance(v,dict) and v.get('sha256')})
 if method in NEEDS_ARTIFACTS:
  from .registry import baseline_availability
  artifacts=registry.get('baseline_artifacts',{}).get(str(seed),{}).get(method,{})
  value['baseline_artifacts']=_descriptors(artifacts);value['requires_baseline_artifacts']=True
  value['optional_baseline']=True
  availability=baseline_availability(registry,method,seed)
  if availability['status']!='AVAILABLE':value['availability_at_preparation']=availability
 if method=='operator' and phase!='development':value['is_primary_algorithm']=True
 value['key']=family+'-'+cohort+'-'+method+'-s'+str(seed)+'-'+digest(value)[:12]
 return value


def build_plan(root,registry,*,stage='campaign',package=PACKAGE,output=None,python=None,
               grid=None,seeds=SEEDS,cohorts=None,confirmation=None,**unused):
 root=Path(root).expanduser().resolve();package=Path(package).expanduser().resolve()
 inputs=read(resolve_descriptor(root,registry),{})
 if inputs.get('schema_version')!='paper3-r11-study-registry-v1' or Path(inputs.get('root','')).resolve()!=root:
  raise ValueError('Prepared R11 source/cache registry root differs')
 if stage not in ('campaign','development','exploratory','confirmatory'):
  raise ValueError('Unknown R11 research stage')
 seeds=tuple(int(s) for s in seeds)
 if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Fixed unique pipeline seeds required')
 for seed in seeds:
  if str(seed) not in inputs.get('source_by_seed',{}):raise ValueError('Missing frozen source/reference for seed '+str(seed))
 if stage=='confirmatory':
  if not confirmation:raise ValueError('Independent confirmation requires actual new registered data')
  cohorts=('confirmation',)
 elif stage=='development':cohorts=()
 else:cohorts=tuple(cohorts or ('holdout','pan2023','weibo2014'))
 for name in cohorts:
  if name in ('pan2023','weibo2014') and name not in inputs.get('external_manifests',{}):
   raise ValueError('Actual external cache missing: '+name)
 confirmation_audit=None
 if confirmation:
  from .confirmation import validate_confirmation_metadata
  confirmation_audit=validate_confirmation_metadata({'root':str(root),
    'confirmation_manifest':confirmation,'previously_used_participants':inputs.get('previously_used_participants',[])},inputs)
 candidates=development_candidates(grid)
 references={};tasks=[];training={};evaluations=[]
 reuse=all(str(s) in inputs.get('profile_reference_by_seed',{}) and
           str(s) in inputs.get('profile_raw_reference_by_seed',{}) for s in seeds)
 strategy='reuse_r9_ea' if reuse else 'build_meta_train_prefix'
 for seed in seeds:
  task=_task(inputs,registry['sha256'],family='feature_reference',method='feature_reference',seed=seed,
    cohort='meta_train',role='meta_train',config={'preprocessing':'ea','reference_strategy':strategy},budgets=(20,))
  references[seed]=task['key'];tasks.append(task)
 for candidate in candidates:
  for seed in seeds:
   deps={}
   for budget in BUDGETS:
    task=_task(inputs,registry['sha256'],family='operator_train',method=candidate['method'],seed=seed,
      cohort='meta_train',role='meta_train',config=candidate['config'],budgets=(budget,),
      requires=(references[seed],),reference=references[seed],candidate=candidate)
    deps[budget]=task['key'];tasks.append(task)
   training[(candidate['candidate_id'],seed)]=deps
   task=_task(inputs,registry['sha256'],family='evaluate',method=candidate['method'],seed=seed,
     cohort='meta_validation',role='meta_validation',config=candidate['config'],
     requires=tuple(deps.values()),reference=references[seed],operators=deps,candidate=candidate)
   evaluations.append(task['key']);tasks.append(task)
 for seed in seeds:
  for method in ('frozen','support_ea'):
   task=_task(inputs,registry['sha256'],family='evaluate',method=method,seed=seed,
      cohort='meta_validation',role='meta_validation')
   evaluations.append(task['key']);tasks.append(task)
 selected=_task(inputs,registry['sha256'],family='selection',method='development_selection',seed=seeds[0],
   cohort='meta_validation',role='meta_validation',config={'rule':'participant_paired_AUBC_oneSE'},budgets=BUDGETS)
 selected['requires_terminal']=evaluations
 selection_key=selected['key'];tasks.append(selected)
 for seed in seeds:
  control_deps={}
  for method in MATCHED_CONTROLS:
   deps={}
   for budget in BUDGETS:
    task=_task(inputs,registry['sha256'],family='operator_train',method=method,seed=seed,
     cohort='meta_train',role='meta_train',phase='exploratory',budgets=(budget,),
     config={'selection_dependency':selection_key,'variant':method},requires=(selection_key,references[seed]),
     reference=references[seed],selection=selection_key)
    if cohorts:tasks.append(task)
    deps[budget]=task['key']
   control_deps[method]=deps
  for cohort in cohorts:
   role='internal_holdout' if cohort=='holdout' else 'independent_confirmation' if cohort=='confirmation' else 'external_exploratory'
   for method in METHODS:
    cfg={};deps=[selection_key];opdeps=None;ref=None
    if method.startswith('operator'):
     cfg={'selection_dependency':selection_key,'variant':method};ref=references[seed]
     if method in MATCHED_CONTROLS:
      opdeps=control_deps[method];deps.extend(opdeps.values())
    elif method=='profile':cfg=read(package/'configs/profile_r10.default.json');ref=references[seed];deps.append(ref)
    task=_task(inputs,registry['sha256'],family='evaluate',method=method,seed=seed,cohort=cohort,role=role,
       phase='confirmatory' if cohort=='confirmation' else 'exploratory',config=cfg,requires=deps,
       reference=ref,selection=selection_key if method.startswith('operator') else None,operators=opdeps)
    tasks.append(task)
 out=Path(output or root/'runs'/('r11_'+stage)).expanduser().resolve()
 plan={'schema_version':'paper3-r11-plan-v1','round':11,'stage':stage,'root':str(root),'package':str(package),
  'output':str(out),'tasks':tasks,'study_registry':copy.deepcopy(registry),'study_registry_sha256':registry['sha256'],
  'reference_strategy':strategy,'code_manifest':production_manifest(package),
  'resources':{'jobs':4,'gpus_per_job':1,'cpus_per_job':16,'memory_per_job':'64G','partition':'hpc_gpu','time_limit':'5-03:00:00'},
  'budgets':list(BUDGETS),'pipeline_seeds':list(seeds),'methods':list(METHODS),'expected_total_tasks':len(tasks),
  'default_configuration':default_configuration(),'development_candidates':candidates,'selection_task':selection_key,
  'selection_rule':{'metric':'normalized_ba_aubc','comparison':'support_ea','unit':'participant',
   'rule':'one_SE_on_paired_gain','ties':['feature_rank_ascending','ridge_descending','noise_strength_descending','candidate_id_lexical'],
   'diagnostic':'participant_leave_one_out_selection','performance_stop_gate':False},
  'comparison_family':copy.deepcopy(PRIMARY_FAMILY),'secondary_ablation_family':copy.deepcopy(SECONDARY_FAMILY),
  'scientific_scope':'development_and_exploratory_reused_cohorts' if cohorts and stage!='confirmatory' else
    'independent_confirmation' if stage=='confirmatory' else 'development_only',
  'development_query_labels_used_for_selection':True,'target_query_labels_used_for_selection':False,
  'historical_target_results_used_for_algorithm_design':True,
  'previously_used_participants':inputs.get('previously_used_participants',[]),
  'training_failure_policy':'isolate_task_and_failed_dependencies_continue_other_tasks',
  'candidate_failure_policy':'exclude_whole_candidate_preserve_failure_ledger_and_fixed_comparison_family',
  'all_development_configurations_predeclared':True,'independent_confirmation_claimed':False}
 if python:plan['python']=str(Path(python).absolute())
 if confirmation:
  plan['confirmation_manifest']=copy.deepcopy(confirmation)
  plan['confirmation_metadata_audit']=confirmation_audit
 return sign_plan(plan)


def write_plan(plan,path):
 path=Path(path).expanduser().resolve();data=encoded(plan)
 if path.is_file() and path.read_bytes()!=data:
  raise ValueError('Existing R11 plan is immutable; resume it or use another explicit output path')
 if not path.is_file():atomic_bytes(path,data)
 return path
