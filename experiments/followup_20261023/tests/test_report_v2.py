"""Reporting tests use synthetic predictions, never scientific EEG results."""
from pathlib import Path
import copy
import numpy as np
import pytest
from paper3v2.io import put, read, digest
from paper3v2.report import aggregate, row_metrics


def plan(root, variants=('P','dual'), seeds=(1,2,3)):
    return {'output':str(root), 'seeds':list(seeds), 'variants':list(variants), 'reference_variants':[],
        'budgets':[20,4], 'comparisons':[['dual','P']],
        'tasks':[{'key':f'{v}-seed{s}','seed':s,'arm':'core','config':{'variant':v,'rank':4,'radius':.15}}
                 for v in variants for s in seeds]}


def row(person,session='1', *, budget=20, improved=False):
    y=[0]*5+[1]*5
    base=[0,0,0,1,1,1,1,1,0,0]
    pred=[0,0,0,0,1,1,1,1,1,0] if improved else base
    probs=lambda p:[[.8,.2] if c==0 else [.2,.8] for c in p]
    ids=[f'{person}-{session}-{i}' for i in range(10)]
    return {'participant_id':person,'session_id':session,'budget':budget,'event_uids':ids,'labels':y,
        'query_signature':digest({'uids':ids,'labels':y}), 'ba':.8 if improved else .6,'bce':.5,
        'f0_ba':.6,'negative_flip_rate':0.,'corrected_error_rate':.2 if improved else 0.,
        'predictions':pred,'f0_predictions':base,'probabilities':probs(pred),'f0_probabilities':probs(base),
        'run_ids':['a','b','a','b','a','a','b','a','b','a'], 'future_runs_with_both_classes':2,
        'calibration_seconds':.1,'fallback':None}


def save_eval(folder, rows):
    for r in rows: put(folder/f'{r["participant_id"]}-{r["session_id"]}.json',r)
    put(folder/'summary.json',{'budget':rows[0]['budget'], 'expected_sessions':[
        {k:r[k] for k in ('participant_id','session_id','query_signature')} for r in rows]})


def save_task(p,task):
    root=Path(p['output'])/task['key']
    put(root/'result.json',{'status':'COMPLETE'})
    for budget in task.get('budgets',p['budgets']):
        save_eval(root/f'holdout/B{budget}',[row('S1',budget=budget,improved=task['config']['variant']=='dual'),
                                           row('S2',budget=budget,improved=task['config']['variant']=='dual')])


def test_probability_metrics_are_class_balanced_and_flip_accounting_matches():
    r=row('S1',improved=True)
    m=row_metrics(r)
    assert m['ba']==pytest.approx(.8) and m['f0_ba']==pytest.approx(.6)
    assert m['ba']-m['f0_ba']==pytest.approx(m['corrected_error_rate']-m['negative_flip_rate'])
    assert m['brier_balanced']==pytest.approx(.8*.2**2+.2*.8**2)
    assert m['ece_balanced_10bins']==pytest.approx(0.)
    assert m['f0_ece_balanced_10bins']==pytest.approx(.2)
    assert m['run_ba_min']<=m['run_ba_p10']
    invalid=copy.deepcopy(r);invalid['probabilities'][0]=[.9,.9]
    with pytest.raises(ValueError,match='probability'):row_metrics(invalid)


def test_core_holm_waits_for_all_core_without_discarding_partial_data(tmp_path):
    p=plan(tmp_path,variants=('P','dual','global'))
    p['comparisons']=[['dual','P'],['dual','global']]
    for t in p['tasks']:
        if t['config']['variant']!='global':save_task(p,t)
    partial=aggregate(p)
    assert partial['status']=='PARTIAL' and not partial['core_complete']
    assert partial['planned_comparison_count']==2 and len(partial['comparisons'])==1
    assert partial['comparisons'][0]['p'] is None and partial['comparisons'][0]['positive_evidence'] is None
    assert partial['budget_curves']
    for t in p['tasks']:
        if t['config']['variant']=='global':save_task(p,t)
    final=aggregate(p)
    assert final['core_complete'] and len(final['comparisons'])==2
    assert all(c['p_holm'] is not None for c in final['comparisons'])
    assert all(c['n_participants']==2 for c in final['comparisons'])


def test_query_mismatch_across_budgets_is_detected_even_if_local_summary_matches(tmp_path):
    p=plan(tmp_path)
    for t in p['tasks']:save_task(p,t)
    target=tmp_path/'dual-seed1/holdout/B4'
    rows=[row('S1',budget=4,improved=True),row('S2',budget=4,improved=True)]
    rows[0]['event_uids'][0]='a-different-query-trial'
    rows[0]['query_signature']=digest({'uids':rows[0]['event_uids'],'labels':rows[0]['labels']})
    save_eval(target,rows)
    with pytest.raises(ValueError,match='Query trial'):aggregate(p)


def test_missing_same_person_in_every_method_cannot_silently_shrink_denominator(tmp_path):
    p=plan(tmp_path)
    for t in p['tasks']:
        save_task(p,t)
        for b in p['budgets']:(tmp_path/t['key']/f'holdout/B{b}/S2-1.json').unlink()
    with pytest.raises(ValueError,match='Missing or extra'):aggregate(p)


def test_duplicate_session_not_silently_overwritten(tmp_path):
    p=plan(tmp_path)
    for t in p['tasks']:save_task(p,t)
    folder=tmp_path/'P-seed1/holdout/B20'
    put(folder/'duplicate.json',read(folder/'S1-1.json'))
    with pytest.raises(ValueError,match='Duplicate participant/session'):aggregate(p)


def test_reference_and_sensitivity_tasks_do_not_expand_primary_holm_family(tmp_path):
    p=plan(tmp_path)
    p['reference_variants']=['head_only']
    for seed in p['seeds']:
        p['tasks'].append({'key':f'head_only-seed{seed}','seed':seed,'arm':'reference','config':{'variant':'head_only'}})
        for variant in ('P','dual'):
            p['tasks'].append({'key':f'{variant}-r2-seed{seed}','seed':seed,'arm':'sensitivity',
                'setting':'r2-rho0p15','budgets':[20],'config':{'variant':variant,'rank':2,'radius':.15}})
    for t in p['tasks']:save_task(p,t)
    result=aggregate(p)
    assert len(result['comparisons'])==1 and result['planned_comparison_count']==1
    assert len(result['sensitivities'])==1 and result['sensitivities'][0]['p'] is None
    curves={r['variant'] for r in result['budget_curves']}
    assert curves=={'P','dual','head_only','f0'}
    assert result['sensitivities'][0]['n_participants']==2


def test_diagnostics_are_paired_exploratory_and_do_not_change_clean_result(tmp_path):
    p=plan(tmp_path)
    for t in p['tasks']:
        save_task(p,t)
        save_eval(tmp_path/t['key']/'diagnostics/gain10',[row('S1',improved=False),row('S2',improved=False)])
    result=aggregate(p)
    assert len(result['diagnostics'])==2 and len(result['diagnostic_comparisons'])==1
    assert result['diagnostic_comparisons'][0]['p'] is None
    assert result['comparisons'][0]['delta_ba_pp']==pytest.approx(20.)
    dual=[r for r in result['diagnostics'] if r['variant']=='dual'][0]
    assert dual['change_from_own_clean']['delta_ba_pp']==pytest.approx(-20.)
