"""Round-seven external endpoint family: generated outputs, never EEG accuracy."""
import copy
from pathlib import Path

import numpy as np
import pytest

from paper3r7 import plan as scientific, report
from paper3r7.io import digest, put


def value(person, session, *, candidate=False, confidence=.8):
    y=[0]*5+[1]*5
    base=[0,0,0,1,1,1,1,1,0,0]
    pred=[0,0,0,0,1,1,1,1,1,0] if candidate else base
    probs=lambda seq:[[confidence,1-confidence] if p==0 else [1-confidence,confidence] for p in seq]
    ids=[f'{person}-{session}-{i}' for i in range(10)]
    return {'participant_id':person,'session_id':session,'budget':20,'labels':y,'event_uids':ids,
        'query_signature':digest({'uids':ids,'labels':y}), 'predictions':pred,
        'f0_predictions':base,'probabilities':probs(pred),'f0_probabilities':probs(base),
        'calibration_seconds':.1,'prediction_seconds':.2,'prediction_seconds_per_trial':.02,
        'supportdiagnostics_seconds':.03}


def fixed_plan(tmp_path, *, people=14):
    groups=sorted({v for pair in scientific.COMPARISONS for v in pair})
    return {'output':str(tmp_path),'recipe_sha256':'synthetic-repeated-external-family',
        'tasks':[{'key':f'{group.replace(":","-")}-{seed}','family':'transfer','group':group,
            'seed':seed,'variant':group,'partition':scientific.PAN_PARTITION,'budgets':[20],
            'scenarios':['clean']} for group in groups for seed in scientific.SEEDS],
        'seeds':list(scientific.SEEDS),'comparisons':list(scientific.COMPARISONS),
        'budgets':[20],'primary_metrics':['nll_balanced','ba'],'primary_holm_family_size':24,
        'primary_partition':scientific.PAN_PARTITION,'primary_budget':20,'primary_scenario':'clean',
        'primary_expected_participants':people,'primary_expected_sessions':2*people,
        'interpretation':'repeated external and internal exploratory evaluation',
        'exploratory_internal_replication':True,'exploratory_external_replication':True,
        'fresh_confirmatory':False}


def save(p, task, *, people=14, sessions=('D1','D2')):
    root=Path(p['output'])/task['key']
    rows=[value(f'P{person:02}',session,candidate=':st:' in task['group'])
          for person in range(1,people+1) for session in sessions]
    folder=root/'evaluations'/task['partition']/'B20'/'clean'
    for row in rows:put(folder/f'{row["participant_id"]}-{row["session_id"]}.json',row)
    put(folder/'summary.json',{'budget':20,'expected_sessions':[
        {k:r[k] for k in ('participant_id','session_id','query_signature')} for r in rows]})
    put(root/'result.json',{'status':'COMPLETE','recipe_sha256':p['recipe_sha256']})


def test_joint_24_endpoint_family_and_ba_compatibility_table_use_one_holm(tmp_path):
    p=fixed_plan(tmp_path)
    for task in p['tasks']:save(p,task)
    result=report.build_report(p)
    assert result['status']=='COMPLETE' and result['primary_family_complete']
    assert result['planned_primary_hypothesis_count']==24
    table=result['primary_multimetric_comparisons']
    assert len(table)==24 and {r['metric'] for r in table}=={'ba','nll_balanced'}
    assert all(r['holm_family_size']==24 and r['matched_session_grid'] for r in table)
    assert all(r['n_participants']==14 and r['n_paired_session_seed_rows']==84 for r in table)
    assert [r['p_holm'] for r in table]==pytest.approx(report.holm([r['p'] for r in table]))
    nll=next(r for r in table if r['metric']=='nll_balanced' and r['baseline']=='static_ce:static:ce:full' and r['candidate']=='static_ce:st:ce:full')
    assert nll['delta']>0 and nll['baseline_mean']>nll['candidate_mean']
    assert nll['direction']=='baseline_minus_candidate'
    for legacy in result['primary_comparisons']:
        paired=next(r for r in table if r['metric']=='ba' and r['candidate']==legacy['candidate'] and r['baseline']==legacy['baseline'])
        assert legacy['p_holm']==paired['p_holm'] and legacy['holm_family_size']==24
    assert {r['metric'] for r in result['timings']} >= {
        'calibration_seconds','prediction_seconds','prediction_seconds_per_trial','supportdiagnostics_seconds'}
    assert result['fresh_confirmatory'] is False
    assert result['exploratory_external_replication'] is True
    assert all(row['inference']=='exploratory_repeated_external_multimetric_family' for row in table)


def test_missing_one_primary_seed_withholds_all_twentyfour_pvalues(tmp_path):
    p=fixed_plan(tmp_path,people=2)
    for task in p['tasks'][:-1]:save(p,task,people=2)
    result=report.build_report(p)
    assert not result['primary_family_complete'] and result['status']=='PARTIAL'
    assert len(result['primary_multimetric_comparisons'])==24
    assert all(r['p'] is None and r['p_holm'] is None for r in result['primary_multimetric_comparisons'])
    assert all(r['p'] is None for r in result['primary_comparisons'])


def test_incomplete_external_cohort_does_not_pass_merely_because_every_result_says_complete(tmp_path):
    p=fixed_plan(tmp_path,people=14)
    for task in p['tasks']:save(p,task,people=2)
    result=report.build_report(p)
    assert result['status']=='COMPLETE'  # Execution complete, inference family incomplete.
    assert not result['primary_family_complete']
    assert all(not r['expected_cohort_ready'] and r['p'] is None for r in result['primary_multimetric_comparisons'])


def test_session_grid_must_match_across_seeds_not_only_within_each_pair(tmp_path):
    p=fixed_plan(tmp_path,people=2)
    for task in p['tasks']:
        sessions=('D1','D2') if task['seed']!=scientific.SEEDS[-1] else ('D1','D3')
        save(p,task,people=2,sessions=sessions)
    result=report.build_report(p)
    assert not result['primary_family_complete']
    assert all(not r['matched_session_grid'] and r['p'] is None for r in result['primary_multimetric_comparisons'])


def test_nll_uses_participant_not_pooled_trials_or_session_counts():
    a,b={1:{},2:{}},{1:{},2:{}}
    for seed,sessions in ((1,('1','2','3')),(2,('1',))):
        for sess in sessions:
            a[seed][('P1',sess)]=value('P1',sess,candidate=seed==1)
            b[seed][('P1',sess)]=value('P1',sess)
        a[seed][('P2','1')]=value('P2','1')
        b[seed][('P2','1')]=value('P2','1')
    d,_=report._paired_metric(a,b,'nll_balanced')
    single=report.row_metrics(value('P1','1',candidate=True))['nll_balanced']-report.row_metrics(value('P1','1'))['nll_balanced']
    assert d['P1']==pytest.approx(single/2) and d['P2']==pytest.approx(0)
    assert np.mean(list(d.values()))==pytest.approx(single/4)


def test_true_log_probabilities_preserve_extreme_balanced_nll_without_clipping():
    # Four binary rows with margins +/-1000. Half are correct, half wrong;
    # float64 probability underflow is legitimate, not a loss ceiling.
    labels=[0,1,0,1]
    logp=np.array([[0.,-1000.],[-1000.,0.],[-1000.,0.],[0.,-1000.]])
    p=np.exp(logp)
    result=report._probability_metrics(p, labels, log_values=logp)
    assert result['nll_balanced']==pytest.approx(500.)
    assert result['brier_balanced']==pytest.approx(.5)
    assert report._probability_metrics(p, labels)['nll_balanced']<14.  # Old-row compatibility only.
    ids=['extreme-'+str(i) for i in range(4)]
    row={'participant_id':'P1','session_id':'D1','labels':labels,'event_uids':ids,
         'probabilities':p.tolist(),'log_probabilities':logp.tolist(),
         'f0_probabilities':p.tolist(),'f0_log_probabilities':logp.tolist()}
    metrics=report.row_metrics(row)
    assert metrics['nll_balanced']==metrics['f0_nll_balanced']==pytest.approx(500.)


@pytest.mark.parametrize('bad',[
    [[0.,-1000.],[0.,-1000.]],  # exp disagreement with second row
    [[1.,-1000.],[-1000.,0.]],  # positive/unnormalized log probability
    [[-1.,-1000.],[-1000.,0.]], # not logsumexp-normalized
    [[float('-inf'),0.],[-1000.,0.]], # new worker tails must be finite
])
def test_log_probability_schema_rejects_inconsistent_or_nonfinite_logs(bad):
    with pytest.raises(ValueError,match='[Ll]og probabil'):
        report._probability_metrics([[1.,0.],[0.,1.]],[0,1],log_values=bad)
