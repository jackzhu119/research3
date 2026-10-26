"""Finite R10 stage plans. No numerical, credential, network or Slurm imports."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .platform.common import (atomic_bytes, digest, encoded, read, sha,
                              sign_plan, verify_descriptor)

PACKAGE = Path(__file__).resolve().parents[1]
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (4, 8, 12, 16, 20)
METHODS = ('frozen', 'support_ea', 'head_bias', 'head_rank4', 'head_full',
           'sgd10', 'sgd100', 'r3_protected', 'r4_xfit_select_stack',
           'r8_fiber_bounded', 'r9_raw_anchored', 'r9_ea_anchored',
           'profile', 'profile_map', 'profile_no_completion', 'profile_frozen_anchor',
           'profile_quadratic','profile_variance_gate')
NEEDS_ARTIFACTS = frozenset(('sgd10','sgd100','r3_protected','r4_xfit_select_stack',
                            'r8_fiber_bounded','r9_raw_anchored','r9_ea_anchored'))
PRIMARY_FAMILY = [
    {'name':'profile_vs_ea','method':'profile','reference':'support_ea','metric':'ba','budget':20},
    {'name':'profile_vs_spatial_sgd100','method':'profile','reference':'sgd100','metric':'ba','budget':20},
    {'name':'profile_vs_r9_ea_anchored','method':'profile','reference':'r9_ea_anchored','metric':'ba','budget':20},
    {'name':'profile_vs_matched_map','method':'profile','reference':'profile_map','metric':'ba','budget':20}]
SECONDARY_FAMILY = [
    {'name':'profile_vs_'+reference,'method':'profile','reference':reference,'metric':metric,'budget':20}
    for reference in ('profile_no_completion','profile_quadratic','profile_variance_gate','profile_frozen_anchor')
    for metric in ('ba','balanced_nll')]
MECHANISM_SUBSET = {'max_participants':8,'max_sessions_per_participant':1,
                    'max_sessions':8,'salt':'r10-mechanism-v1',
                    'required_participants':8,'selection_participants':4,'check_participants':4}
REFERENCE_SUBSET = {'max_participants':4,'max_sessions_per_participant':2,
                    'max_sessions':8,'salt':'r10-reference-v1'}


def describe(path, base=None):
    path = Path(path).expanduser().resolve(strict=True)
    value = str(path)
    if base is not None and path.is_relative_to(Path(base).resolve()):
        value = str(path.relative_to(Path(base).resolve()))
    return {'path':value,'sha256':sha(path),'bytes':path.stat().st_size}


def resolve_descriptor(root, item):
    path = Path(item['path']).expanduser()
    path = path if path.is_absolute() else Path(root)/path
    if (not path.is_file() or path.stat().st_size!=item.get('bytes')
            or sha(path)!=item.get('sha256')):
        raise ValueError('Declared R10 input changed: '+str(path))
    return path.resolve()


def production_manifest(package=PACKAGE):
    package = Path(package).resolve()
    paths = sorted((package/'paper3r10').rglob('*.py'))
    paths += [p for p in (package/'START.py', package/'paper3r10/baseline_registry.json',
                          package/'configs/algorithm.default.json') if p.is_file()]
    return {str(p.relative_to(package)):sha(p) for p in paths if '__pycache__' not in p.parts}


def default_configuration():
    return read(PACKAGE/'configs/algorithm.default.json')


def ablation_configurations(selected):
    return {'profile_map':dict(selected,mode='map',evidence_fraction=0.0),
            'profile_no_completion':dict(selected,use_null_completion=False),
            'profile_frozen_anchor':dict(selected),
            'profile_quadratic':dict(selected),
            'profile_variance_gate':dict(selected)}


def _artifact_descriptors(value):
    if isinstance(value,dict):
        if {'path','sha256','bytes'}<=set(value):
            return [value]
        return [x for v in value.values() for x in _artifact_descriptors(v)]
    if isinstance(value,list):
        return [x for v in value for x in _artifact_descriptors(v)]
    return []


def _task(method, seed, cohort, role, config, registry, registry_sha, *, family='evaluate',
          budgets=BUDGETS, reference=None, subset=None, candidate=None):
    scientific = {'method':method,'seed':seed,'cohort':cohort,'role':role,
                  'method_config':copy.deepcopy(config),'budgets':list(budgets),
                  'family':family,'registry_sha256':registry_sha}
    key = f'{family}-{cohort}-{method}-seed{seed}-{digest(scientific)[:12]}'
    value = dict(scientific, key=key, kind=family, config=copy.deepcopy(config),
                 requires=[reference] if reference else [],
                 input_identities={'registry_sha256':registry_sha})
    if reference: value['reference_dependency']=reference
    if subset: value['session_subset']=copy.deepcopy(subset)
    if candidate: value['candidate_id']=candidate
    source = registry.get('source_by_seed',{}).get(str(seed),{})
    value['input_identities'].update({k+'_sha256':v.get('sha256') for k,v in source.items()
                                     if isinstance(v,dict) and v.get('sha256')})
    if method in NEEDS_ARTIFACTS:
        artifacts = registry.get('baseline_artifacts',{}).get(str(seed),{}).get(method,{})
        value['baseline_artifacts']=_artifact_descriptors(artifacts)
        value['requires_baseline_artifacts']=True
    if method=='profile': value['is_primary_algorithm']=True
    if method in ('profile_map','profile_no_completion','profile_frozen_anchor',
                  'profile_quadratic','profile_variance_gate'):
        value['ablation']=method
    if method.startswith('profile'):
        value['predictor_anchor']='frozen' if method=='profile_frozen_anchor' else 'support_ea'
    return value


def build_plan(root, registry, *, stage='mechanism', package=PACKAGE, output=None,
               freeze=None, confirmation=None, amendment=None, python=None, require_artifacts=True):
    """Build only. The caller must separately request submission with --submit."""
    root, package = Path(root).expanduser().resolve(), Path(package).expanduser().resolve()
    registry_path = resolve_descriptor(root,registry)
    inputs = read(registry_path)
    if not isinstance(inputs,dict) or inputs.get('schema_version')!='paper3-r10-study-registry-v1':
        raise ValueError('A prepared, SHA-bound R10 study registry is required')
    if Path(inputs.get('root','')).resolve()!=root:
        raise ValueError('Study registry project root differs')
    if stage not in ('mechanism','formal_exploratory','formal_confirmatory'):
        raise ValueError('Unknown R10 stage')
    seeds = (SEEDS[0],) if stage=='mechanism' else SEEDS
    for seed in seeds:
        if str(seed) not in inputs.get('source_by_seed',{}):
            raise ValueError('Missing frozen source/reference receipts for seed '+str(seed))
    selected = default_configuration()
    frozen = None
    amended = None
    if amendment:
        if stage!='formal_exploratory' or freeze or confirmation:
            raise ValueError('An exploratory amendment cannot authorize original or confirmatory protocols')
        amended = read(verify_descriptor(root,amendment),{})
        if (amended.get('schema_version')!='paper3-r10-exploratory-amendment-v1'
                or amended.get('status')!='EXPLORATORY_SELECTED'
                or amended.get('retrospective_amendment') is not True
                or amended.get('performance_gates_are_diagnostics') is not True
                or amended.get('target_query_labels_used_for_selection') is not False):
            raise ValueError('Explicit development-only exploratory amendment required')
        if (amended.get('study_registry_sha256')!=registry['sha256']
                or amended.get('code_manifest_sha256')!=digest(production_manifest(package))):
            raise ValueError('Amendment source registry or new execution code differs')
        if amended.get('legacy_mechanism_completed_tasks')!=15:
            raise ValueError('The completed original fifteen-task mechanism must be explicitly referenced')
        from .algorithm import Config
        from dataclasses import asdict
        selected = asdict(Config(**amended['selected_configuration']).validate())
    if stage!='mechanism':
        if not amended:
            if not freeze: raise ValueError('Formal planning awaits a passing development freeze')
            frozen = read(verify_descriptor(root,freeze),{})
            if frozen.get('status')!='FROZEN' or frozen.get('scientific_scope')!='REAL_DATA':
                raise ValueError('Development did not pass its prespecified gates; formal work is stopped')
            selected = dict(frozen['selected_configuration'])
            if frozen.get('study_registry_sha256')!=registry['sha256']:
                raise ValueError('Formal work must retain the frozen study/source registry')
    if stage=='formal_confirmatory':
        if not confirmation:
            raise ValueError('No genuinely new independent confirmation data registered')
        manifest=read(resolve_descriptor(root,confirmation),{})
        if (manifest.get('cohort_kind')!='independent_confirmation'
                or manifest.get('status')!='COMPLETE'
                or manifest.get('previously_used_for_algorithm_design') is not False
                or manifest.get('acquisition_order_verified') is not True
                or manifest.get('physical_run_id_available') is not True
                or not manifest.get('participants')):
            raise ValueError('Confirmation data lack reviewed independence and physical chronology')
        if set(manifest['participants']) & set(inputs.get('previously_used_participants',[])):
            raise ValueError('Confirmation participants overlap previously used participants')
    refs={};tasks=[]
    # Reference geometry must not switch when the active stage expands from
    # one development seed to three formal seeds. Derive from the full fixed
    # seed registry in every stage, including development.
    reuse=all(str(s) in inputs.get('profile_reference_by_seed',{})
              and str(s) in inputs.get('profile_raw_reference_by_seed',{}) for s in SEEDS)
    strategy='reuse_r9_ea' if reuse else 'build_meta_train_prefix'
    if frozen and frozen.get('reference_strategy')!=strategy:
        raise ValueError('Feature-reference strategy differs from development freeze')
    if amended and amended.get('reference_strategy')!=strategy:
        raise ValueError('Feature-reference strategy differs from the reused development experiment')
    for seed in seeds:
        task=_task('feature_reference',seed,'meta_train','meta_train',
                   {'preprocessing':'ea','reference_strategy':strategy},inputs,registry['sha256'],
                   family='feature_reference',budgets=(20,),subset=REFERENCE_SUBSET)
        refs[seed]=task['key']; tasks.append(task)
    if stage=='mechanism':
        for amplitude in (.3,1.,3.):
            for c in (.01,.1,.5):
                cfg=dict(selected,amplitude=amplitude,evidence_fraction=c)
                tasks.append(_task('profile',SEEDS[0],'meta_validation','meta_validation',cfg,
                    inputs,registry['sha256'],reference=refs[SEEDS[0]],budgets=(4,8,20),
                    subset=MECHANISM_SUBSET,candidate=f'a{amplitude:g}-c{c:g}'))
        for amplitude in (.3,1.,3.):
            cfg=dict(selected,amplitude=amplitude,evidence_fraction=0.,mode='map')
            tasks.append(_task('profile_map',SEEDS[0],'meta_validation','meta_validation',cfg,
                inputs,registry['sha256'],reference=refs[SEEDS[0]],budgets=(4,8,20),subset=MECHANISM_SUBSET))
        for method in ('frozen','support_ea'):
            tasks.append(_task(method,SEEDS[0],'meta_validation','meta_validation',{},
                inputs,registry['sha256'],budgets=(4,8,20),subset=MECHANISM_SUBSET))
    else:
        cohorts=('holdout','pan2023','weibo2014') if stage=='formal_exploratory' else ('confirmation',)
        ablations=ablation_configurations(selected)
        for cohort in cohorts:
            role='internal_holdout' if cohort=='holdout' else ('independent_confirmation'
                if cohort=='confirmation' else 'external_exploratory')
            if cohort in ('pan2023','weibo2014') and cohort not in inputs.get('external_manifests',{}):
                raise ValueError('Declared exploratory cohort cache is absent: '+cohort)
            for seed in seeds:
                for method in METHODS:
                    cfg = selected if method=='profile' else ablations.get(method,{})
                    tasks.append(_task(method,seed,cohort,role,cfg,inputs,registry['sha256'],
                                       reference=refs[seed] if method.startswith('profile') else None))
    missing=[f"{t['method']}/seed{t['seed']}" for t in tasks
             if t.get('requires_baseline_artifacts') and not t.get('baseline_artifacts')]
    if missing and require_artifacts:
        raise ValueError('Required historical baseline artifacts are absent: '+', '.join(sorted(set(missing))))
    if amended:
        for task in tasks:
            if task.get('requires_baseline_artifacts'):
                task['optional_baseline']=True
                task['missing_artifact_policy']='isolate_this_task_keep_fixed_comparison_family'
    output=Path(output or root/'runs'/('r10_'+stage)).expanduser().resolve()
    plan={'schema_version':'paper3-r10-plan-v1','round':10,'stage':stage,
          'root':str(root),'package':str(package),'output':str(output),'tasks':tasks,
          'study_registry':dict(registry),'study_registry_sha256':registry['sha256'],
          'reference_strategy':strategy,'resources':{'jobs':4,'gpus_per_job':1,
              'cpus_per_job':16,'memory_per_job':'64G' if amended else '68G','partition':'hpc_gpu','time_limit':'5-03:00:00'},
          'comparison_family':copy.deepcopy(PRIMARY_FAMILY),
          'secondary_ablation_family':copy.deepcopy(SECONDARY_FAMILY),
          'budgets':list((4,8,20) if stage=='mechanism' else BUDGETS),
          'scientific_scope':'development_only' if stage=='mechanism' else
              ('exploratory_protocol_amendment_reused_cohorts' if amended else
               'exploratory_reused_cohorts' if stage=='formal_exploratory' else 'independent_confirmation'),
          'target_query_labels_used_for_selection':False,
          'historical_target_results_used_for_algorithm_design':True,
          'development_query_labels_used':stage=='mechanism',
          'selection_split':{'rule':'hash_participant_then_first8_split4_4',
              'salt':'r10-mechanism-v1','selection_count':4,'check_count':4},
          'previously_used_participants':inputs.get('previously_used_participants',[]),
          'code_manifest':production_manifest(package),'expected_total_tasks':len(tasks),
          'all_stages_submitted_now':False}
    if python: plan['python']=str(Path(python).absolute())
    if freeze: plan['freeze']=dict(freeze)
    if amended:
        plan['amendment']=dict(amendment)
        plan['retrospective_amendment']=True
        plan['performance_gate_controls_training']=False
        plan['legacy_mechanism_reused']=True
        plan['legacy_mechanism_rerun_tasks']=0
        plan['selection_lineage']={k:amended.get(k) for k in (
            'selection_policy','selected_candidate_id','source_plan_sha256','source_report_sha256',
            'legacy_mechanism_completed_tasks','selection_participant_ids','check_participant_ids')}
    if confirmation: plan['confirmation_manifest']=dict(confirmation)
    if stage=='formal_confirmatory':
        from .confirmation import validate_confirmation_metadata
        plan['confirmation_metadata_audit']=validate_confirmation_metadata(plan,inputs)
    if frozen and frozen.get('code_manifest_sha256')!=digest(plan['code_manifest']):
        raise ValueError('Code changed after development; the frozen selection cannot be transferred silently')
    return sign_plan(plan)


def write_plan(plan, path):
    path=Path(path).expanduser().resolve();data=encoded(plan)
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('An existing stage plan is immutable; use a new explicit output path')
    if not path.exists(): atomic_bytes(path,data)
    return path
