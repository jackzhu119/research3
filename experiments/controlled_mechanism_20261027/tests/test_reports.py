from __future__ import annotations
import csv
import io
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from paper3m12 import reports as r
from paper3m12.common import atomic_json, digest, sha_file


def seal(plan, task, payload, files=()):
    folder = Path(plan['output'])/'tasks'/task['key'];folder.mkdir(parents=True, exist_ok=True)
    result = dict(status='COMPLETE', task=task, task_key=task['key'], task_identity=digest(task),
                  files=list(files), recipe_sha256=plan.get('recipe_sha256'), **payload)
    atomic_json(folder/'result.json', result)
    return result


def metrics(plan, task, rows):
    folder = Path(plan['output'])/'tasks'/task['key'];folder.mkdir(parents=True, exist_ok=True)
    path = folder/'metrics.csv'
    fields = sorted({name for row in rows for name in row})
    stream = io.StringIO(newline='');writer=csv.DictWriter(stream, fields);writer.writeheader();writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())
    item = dict(path=path.relative_to(plan['output']).as_posix(), bytes=path.stat().st_size, sha256=sha_file(path))
    seal(plan, task, dict(metrics_csv=item), [item])


def row(task, method, *, session='s1', draw=None, ba=.65, nll=.5, support=.4, layer='reference_rms', steps=100):
    return dict(role=task['role'], participant=task['participant'], session=session, seed=task['seed'],
        budget=20, method=method, draw_index=draw, layer=layer, steps=steps, BA=ba,
        balanced_NLL=nll, Brier=.2, support_BCE=support, reference_RMS=.02,
        weak_deleted_reference_RMS=.01, match_absolute_error=0., target_reference_rms=.02,
        query_identity=task['participant']+'/'+session+'/'+str(task['seed']), n_query=20, n_valid_support=15)


def all_rows(task, sessions=('s1',)):
    out=[]
    for session in sessions:
        out.extend([row(task,'support_ea',session=session,ba=.60,nll=.60,support=.7,layer='anchor',steps=0),
                    row(task,'full_sgd',session=session,ba=.55,nll=.75,support=.4,layer='raw'),
                    row(task,'top_q',session=session),
                    row(task,'scalar_shrink_full',session=session,ba=.60,nll=.60),
                    row(task,'drop_weak',session=session,ba=.64,nll=.51,layer='removal_reference_rms')])
        for draw in range(5):
            out.append(row(task,'random_q',session=session,draw=draw,ba=.55+draw*.001,nll=.60))
            out.append(row(task,'drop_random',session=session,draw=draw,ba=.55,nll=.60,layer='removal_reference_rms'))
    return out


@pytest.fixture
def study(tmp_path):
    tasks=[dict(key=f'pilot-{person}-{seed}',kind='paired_block',role='pilot',participant=person,seed=seed)
           for person in ['p1','p2'] for seed in [20261018,20261019,20261020]]
    plan=dict(output=str(tmp_path/'output'),recipe_sha256='example',tasks=tasks,
              config={'model':{'source_seeds':[20261018,20261019,20261020],'random_draws':5},
                      'statistics':{'bootstrap_resamples':64,'large_n_signflip_draws':100},
                      'matching':{'absolute_match_tolerance':1e-5}})
    for task in tasks:metrics(plan,task,all_rows(task))
    return plan


def test_exact_paired_statistics_and_holm_keep_full_family():
    result=r.paired_statistics([1.,1.,1.],bootstrap_repeats=20)
    assert result['pvalue']==.25 and result['n_participants']==3
    assert result['ci95']==[1.,1.]
    assert r.paired_statistics([0.,0.],bootstrap_repeats=20)['pvalue']==1.
    assert r.holm([.01,.04,.03,.9,.4,.05])==pytest.approx([.06,.16,.15,.9,.8,.16])


def test_participants_equal_sessions_seeds_and_random_metrics_not_pooled(study):
    for task in study['tasks']:
        rows=all_rows(task, ('s1','s2') if task['participant']=='p1' else ('s1',))
        for item in rows:
            if item['method']=='top_q':item['BA']=(.8 if item['session']=='s1' else .2) if task['participant']=='p1' else .9
            if item['method']=='random_q':item['BA']=.3+item['draw_index']*.05
        metrics(study,task,rows)
    report=r.build_report(study)
    comparison=report['roles']['pilot']['comparisons'][0]
    assert comparison['n_participants']==2
    assert comparison['participant_differences']==pytest.approx([.1,.5])
    assert comparison['effect_mean']==pytest.approx(.3)
    assert comparison['family_complete'] is True
    assert report['scientific_completion_claimed'] is False
    assert report['independent_confirmation_available'] is False


def test_missing_source_seed_never_reduces_family_or_claims_completion(study):
    (Path(study['output'])/'tasks'/study['tasks'][0]['key']/'result.json').unlink()
    report=r.build_report(study)
    assert report['status']=='PARTIAL'
    assert all(row['holm_pvalue'] is None for row in report['roles']['pilot']['comparisons'])
    assert not report['roles']['pilot']['family_complete']
    with pytest.raises(RuntimeError,match='incomplete'):
        r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))


def test_missing_random_draw_is_reported_not_silently_average_four(study):
    task=study['tasks'][0]
    rows=[item for item in all_rows(task) if not(item['method']=='random_q' and item['draw_index']==4)]
    metrics(study,task,rows)
    report=r.build_report(study)
    assert report['status']=='PARTIAL'
    assert report['issues'][0]['code']=='random_draw_coverage_incomplete'
    assert all(row['pvalue'] is None for row in report['roles']['pilot']['comparisons'])


def test_conflicting_duplicate_metrics_or_changed_query_are_refused(study):
    task=study['tasks'][0];rows=all_rows(task)
    duplicate=dict(rows[2],BA=.1);rows.append(duplicate);metrics(study,task,rows)
    with pytest.raises(ValueError,match='Conflicting duplicate'):r.build_report(study)
    rows=all_rows(task)
    for item in rows:
        if item['method']=='random_q' and item['draw_index']==0:item['query_identity']='different query'
    metrics(study,task,rows)
    with pytest.raises(ValueError,match='different future queries'):r.build_report(study)


def test_pilot_go_is_prespecified_point_gate_not_significance(study):
    result=r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))
    assert result['decision']=='GO'
    assert all(result['criteria'].values())
    saved=r.read_json(Path(study['output'])/'pilot-decision.json')
    assert saved['statistical_significance_claimed'] is False
    assert saved['future_risk_guarantee'] is False
    # A valid negative development result is a completed scientific STOP,
    # not an operational failure and not a request to change any old round.
    for task in study['tasks']:
        rows=all_rows(task)
        for item in rows:
            if item['method']=='top_q':item['BA']=.4
        metrics(study,task,rows)
    result=r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))
    assert result['status']=='COMPLETE' and result['decision']=='STOP'


def test_one_bad_random_matching_error_is_not_hidden_by_average(study):
    task=study['tasks'][0];rows=all_rows(task)
    for item in rows:
        if item['method']=='random_q' and item['draw_index']==0:item['match_absolute_error']=4e-5
    metrics(study,task,rows)
    result=r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))
    assert result['decision']=='STOP' and not result['criteria']['numeric_reference_matching_stable']


def test_tau_freeze_reads_support_only_all_seeds_and_never_revises(tmp_path):
    seeds=[20261018,20261019,20261020]
    tasks=[dict(key=f'calibrate-{seed}',kind='calibration',seed=seed) for seed in seeds]
    plan=dict(output=str(tmp_path/'output'),tasks=tasks,recipe_sha256='example',config={'model':{'source_seeds':seeds}})
    for index,task in enumerate(tasks):seal(plan,task,dict(tau_observations={str(b):[.01*(index+1)] for b in [4,8,12,16,20]},query_labels_used=False))
    freeze=dict(key='freeze-tau',kind='freeze_tau',dependencies=[task['key'] for task in tasks])
    result=r.run_control_task(plan,freeze)
    assert result['tau_by_budget']['20']==pytest.approx(.02)
    path=Path(plan['output'])/'frozen_tau.json';before=path.read_bytes()
    assert r.run_control_task(plan,freeze)['tau_sha256']==sha_file(path)
    assert path.read_bytes()==before
    seal(plan,tasks[0],dict(tau_observations={str(b):[.9] for b in [4,8,12,16,20]},query_labels_used=False))
    with pytest.raises(ValueError,match='cannot be revised'):r.run_control_task(plan,freeze)


def test_tau_future_labels_or_unsealed_metrics_are_not_accepted(study,tmp_path):
    task=study['tasks'][0]
    path=Path(study['output'])/'tasks'/task['key']/'metrics.csv';path.write_text('corrupted')
    with pytest.raises(ValueError,match='size differs'):r.build_report(study)


def test_development_freeze_records_variance_without_retuning(study):
    for task in study['tasks']:
        task['role']='meta_validation';metrics(study,task,all_rows(task))
    atomic_json(Path(study['output'])/'frozen_tau.json',dict(tau_by_budget={'20':.02}))
    result=r.run_control_task(study,dict(key='development-freeze',kind='development_freeze',role='meta_validation'))
    assert result['decision']=='GO'
    saved=r.read_json(Path(study['output'])/'development-freeze.json')
    assert saved['planned_new_evaluable_participants_unchanged']==90
    assert saved['independent_data_status']=='DECLARED_NOT_AVAILABLE'
    assert saved['method_selection'].startswith('Predeclared methods retained')


def test_compact_sealed_summary_equals_csv_and_avoids_reparsing_completed_blocks(study,monkeypatch):
    expected=r.build_report(study)['roles']['pilot']['comparisons'][0]['effect_mean']
    for task in study['tasks']:
        folder=Path(study['output'])/'tasks'/task['key'];result=r.read_json(folder/'result.json')
        summary=r.summarize_task_metrics(study,task,result['metrics_csv'])
        path=folder/'participant_summary.json';atomic_json(path,summary)
        descriptor=dict(path=path.relative_to(study['output']).as_posix(),bytes=path.stat().st_size,sha256=sha_file(path))
        result['participant_summary']=descriptor;result['files'].append(descriptor)
        atomic_json(folder/'result.json',result)
    monkeypatch.setattr(r.csv,'DictReader',lambda *a,**k:pytest.fail('Completed block CSV was parsed again'))
    report=r.build_report(study)
    assert report['roles']['pilot']['comparisons'][0]['effect_mean']==pytest.approx(expected)
    assert report['roles']['pilot']['family_complete'] is True
    assert any(Path(item['path']).name == 'session_metrics_index.csv' for item in report['artifacts'])
    decision=r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))
    assert decision['decision']=='GO'


def test_undefined_support_loss_is_scientific_stop_without_dropping_participant(study):
    task=study['tasks'][0]
    for task in [item for item in study['tasks'] if item['participant']=='p1']:
        rows=all_rows(task)
        for item in rows:item['support_BCE']=''
        metrics(study,task,rows)
    result=r.run_control_task(study,dict(key='pilot-decision',kind='pilot_decision',role='pilot'))
    assert result['status']=='COMPLETE' and result['decision']=='STOP'
    body=r.read_json(Path(study['output'])/'pilot-decision.json')
    assert body['decision_status']=='STOP_INSUFFICIENT_EVIDENCE'
    assert body['evidence']['support_loss_participants_available']==1
    assert r.build_report(study)['roles']['pilot']['comparisons'][0]['n_participants']==2


def test_final_report_artifacts_are_immutable_across_background_refresh(study):
    result=r.run_control_task(study,dict(key='final',kind='final_report'))
    prior={item['path']:item['sha256'] for item in result['artifacts']}
    r.build_report(study, role='pilot')
    task=study['tasks'][0];rows=all_rows(task)
    for row_ in rows:
        if row_['method']=='top_q':row_['BA']=.7
    metrics(study,task,rows)
    later=r.build_report(study)
    assert later['snapshot_sha256']
    assert all(sha_file(Path(study['output'])/path)==sha for path,sha in prior.items())
    assert all(path!='reports/report.json' for path in prior)


def test_head_and_shift_random_draws_remain_separate_conditions(study):
    task=study['tasks'][0]
    rows=[]
    for suffix in ('_head_A', '__clean'):
        rows.extend(row(task,'random_q_'+str(draw)+suffix,draw=draw,ba=.5+draw*.01) for draw in range(5))
    metrics(study,task,rows)
    summary=r.summarize_task_metrics(study,task,r.read_json(Path(study['output'])/'tasks'/task['key']/'result.json')['metrics_csv'])
    assert {item['method'] for item in summary['seed_rows']}=={'random_q_head_A','random_q__clean'}
    assert not summary['issues']


def test_secondary_diagnostics_have_separate_tables_and_required_coverage(study):
    secondaries=[]
    for task in study['tasks']:
        extra=dict(task, key='head-'+task['key'], kind='head_block')
        secondaries.append(extra)
        rows=[dict(item, method=item['method']+'_head_A') for item in all_rows(extra)]
        metrics(study,extra,rows)
    study['tasks'].extend(secondaries)
    report=r.build_report(study)
    assert report['status']=='COMPLETE'
    assert report['secondary']['head_block']['complete_tasks']==6
    assert report['secondary']['head_block']['primary_family_included'] is False
    assert report['roles']['pilot']['comparisons'][0]['n_participants']==2
    assert any('/secondary/head_block/participant_metrics.csv' in item['path'] for item in report['artifacts'])
    (Path(study['output'])/'tasks'/secondaries[0]['key']/'result.json').unlink()
    assert r.build_report(study)['status']=='PARTIAL'
    assert r.build_report(study,role='pilot')['status']=='COMPLETE'
