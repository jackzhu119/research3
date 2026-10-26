"""R10 continuation plan and honest scientific provenance regressions."""
import copy
import csv

import pytest

from paper3r10 import study
from paper3r10.platform.common import atomic, digest, sign_plan
from paper3r10.report import generate_report
from test_study_cli import prepared
from test_statistics_selection import make_plan, complete


def amendment(root, package, descriptor, **changes):
    value = {
        'schema_version': 'paper3-r10-exploratory-amendment-v1',
        'status': 'EXPLORATORY_SELECTED',
        'retrospective_amendment': True,
        'performance_gates_are_diagnostics': True,
        'target_query_labels_used_for_selection': False,
        'study_registry_sha256': descriptor['sha256'],
        'code_manifest_sha256': digest(study.production_manifest(package)),
        'legacy_mechanism_completed_tasks': 15,
        'source_plan_sha256': '1'*64,
        'source_report_sha256': '2'*64,
        'reference_strategy': 'reuse_r9_ea',
        'selected_configuration': study.default_configuration(),
        'selection_policy': 'paired_gain_one_se_selection_only',
        'selected_candidate_id': 'a0.3-c0.1',
    }
    value.update(changes)
    path=root/'amendment.json'; atomic(path,value)
    return study.describe(path,root)


def test_amendment_launches_formal_165_with_four_64g_jobs_and_no_mechanism_evaluations(prepared):
    root,package,registry,path,desc=prepared
    plan=study.build_plan(root,desc,stage='formal_exploratory',package=package,
                         amendment=amendment(root,package,desc))
    assert len(plan['tasks'])==165
    assert sum(t['family']=='feature_reference' for t in plan['tasks'])==3
    evaluations=[t for t in plan['tasks'] if t['family']=='evaluate']
    assert len(evaluations)==162
    assert {t['method'] for t in evaluations}==set(study.METHODS)
    assert {t['cohort'] for t in evaluations}=={'holdout','pan2023','weibo2014'}
    assert all(t['budgets']==[4,8,12,16,20] for t in evaluations)
    assert all(t['role']!='meta_validation' for t in evaluations)
    assert plan['resources']['jobs']==4 and plan['resources']['gpus_per_job']==1
    assert plan['resources']['cpus_per_job']==16 and plan['resources']['memory_per_job']=='64G'
    assert plan['legacy_mechanism_rerun_tasks']==0 and not plan['performance_gate_controls_training']
    assert plan['comparison_family']==study.PRIMARY_FAMILY
    assert 'amendment' in plan['scientific_scope'] and 'freeze' not in plan


@pytest.mark.parametrize('change',[
    {'target_query_labels_used_for_selection':True},
    {'retrospective_amendment':False},
    {'performance_gates_are_diagnostics':False},
    {'study_registry_sha256':'9'*64},
    {'code_manifest_sha256':'9'*64},
    {'legacy_mechanism_completed_tasks':14},
])
def test_invalid_amendment_is_not_silently_accepted(prepared,change):
    root,package,_,_,desc=prepared
    with pytest.raises(ValueError):
        study.build_plan(root,desc,stage='formal_exploratory',package=package,
                         amendment=amendment(root,package,desc,**change))


def test_amendment_does_not_authorize_independent_confirmation(prepared):
    root,package,_,_,desc=prepared
    with pytest.raises(ValueError,match='confirmatory'):
        study.build_plan(root,desc,stage='formal_confirmatory',package=package,
                         amendment=amendment(root,package,desc))


def test_missing_one_historical_artifact_preserves_other_tasks_and_fixed_family(prepared):
    root,package,registry,path,_=prepared
    registry['baseline_artifacts']['20261019'].pop('sgd100');atomic(path,registry)
    desc=study.describe(path,root)
    plan=study.build_plan(root,desc,stage='formal_exploratory',package=package,require_artifacts=False,
                         amendment=amendment(root,package,desc))
    missing=[t for t in plan['tasks'] if t.get('requires_baseline_artifacts') and not t['baseline_artifacts']]
    assert len(missing)==3 and all(t['method']=='sgd100' and t['seed']==20261019 for t in missing)
    assert all(t['optional_baseline'] for t in missing)
    assert len(plan['tasks'])==165 and plan['comparison_family']==study.PRIMARY_FAMILY


def test_report_and_all_analysis_tables_keep_amendment_and_missing_comparisons(tmp_path):
    path,plan=make_plan(tmp_path)
    record={'path':'amendment.json','sha256':'a'*64,'bytes':10}
    plan.pop('freeze',None)
    plan.update(amendment=record,retrospective_amendment=True,legacy_mechanism_rerun_tasks=0,
                selection_lineage={'selection_policy':'paired_gain_one_se_selection_only'})
    plan=sign_plan(plan);atomic(path,plan)
    for task in plan['tasks']:
        if task['method']!='r9_ea_anchored': complete(plan,task)
    report=generate_report(path)
    assert report['status']=='PARTIAL' and report['primary_family_size']==4
    assert report['retrospective_protocol_revision'] and report['independent_confirmation'] is False
    assert report['protocol_amendment']==record
    assert report['original_mechanism_results_preserved']
    assert all(r['protocol_amendment_sha256']=='a'*64 for r in report['methods']+report['comparisons'])
    primary=report['comparisons'][:4]
    assert primary[2]['status']=='NOT_READY' and primary[2]['holm_pvalue'] is None
    assert all(r['inference_scope']=='exploratory_amended_protocol_reused_cohort' for r in primary)
    assert not any(r['significant_holm_05'] for r in primary)
    for name in ('session_metrics.csv','participant_seed_metrics.csv','participant_metrics.csv','method_summary.csv','comparisons.csv'):
        with (path.parent/'study/runs/r10-stat-test/reports'/name).open() as stream:
            rows=list(csv.DictReader(stream))
        assert rows and all(r['protocol_amendment_sha256']=='a'*64 for r in rows)
