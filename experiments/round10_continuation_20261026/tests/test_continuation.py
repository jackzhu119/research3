"""Isolated metadata fixtures. No EEG/model execution or scientific claim."""
import copy,csv,io,json
from pathlib import Path
import pytest
from paper3r10 import continuation,study,cli
from paper3r10.platform import common,runtime
from test_study_cli import prepared


def legacy_fixture(prepared):
    root,package,registry,registry_path,desc=prepared
    plan=study.build_plan(root,desc,package=package,output=root/'runs/r10_mechanism')
    output=Path(plan['output']);output.mkdir(parents=True)
    path=study.write_plan(plan,output/'plan.json')
    people=[f'P{i}' for i in range(8)];ordered=sorted(people,key=lambda p:common.digest(['r10-mechanism-v1','participant',p]))
    ledger=[];allrows=[]
    for task in plan['tasks']:
        folder=output/task['key'];folder.mkdir();source={'source':'frozen-fixture'};data={'cache':'fixture'}
        artifacts=[];rows=[]
        if task['family']=='feature_reference':
            artifact=folder/'reference.npz';artifact.write_bytes(b'software fixture only; never decoded')
            metadata=folder/'reference.json';common.atomic(metadata,{'software_fixture_only':True})
            artifacts=[study.describe(artifact,folder),study.describe(metadata,folder)]
        else:
            for person in people:
                for budget in task['budgets']:
                    ea=.55+.02*people.index(person)
                    gain=.01 if task.get('candidate_id')=='a0.3-c0.1' else .011 if task.get('candidate_id')=='a0.3-c0.01' else 0.
                    rows.append(dict(participant_id=person,session_id='1',seed=task['seed'],budget=budget,
                        method=task['method'],cohort='meta_validation',role='meta_validation',
                        selection_split='selection' if person in ordered[:4] else 'check',
                        ba=ea+gain,balanced_nll=.6,balanced_brier=.2,updated_query_fraction=.1 if gain else 0.,
                        n_query=40,n_class0=20,n_class1=20,certificate_violation_count=0,numerical_fallback_count=0,
                        incremental_prediction_seconds=.001,query_signature=common.digest([person,'query']),
                        source_identity_sha256=common.digest(source),data_identity_sha256=common.digest(data)))
            f=io.StringIO(newline='');writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            artifact=folder/'session_metrics.csv';artifact.write_text(f.getvalue());artifacts=[study.describe(artifact,folder)]
            for row in rows:row.update(task_key=task['key'],candidate_id=task.get('candidate_id',''),configuration_id=common.digest(task['config']))
            allrows.extend(rows)
        result=dict(status='COMPLETE',scientific_scope='REAL_DATA',recipe_sha256=plan['recipe_sha256'],
            task_key=task['key'],task_identity=common.task_identity(plan,task),artifacts=artifacts,paper_artifacts=artifacts,
            registry_sha256=desc['sha256'],source_identity=source,catalog_identity=data,
            completed_session_budget_cells=len(rows),sessions=8,participants=8)
        common.atomic(folder/'result.json',result);proof=common.seal_result(plan,task)
        receipt=dict(task_key=task['key'],result_sha256=proof['result_sha256'])
        if rows:receipt.update(compact_metrics=artifacts[0],rows=len(rows),source_identity_sha256=common.digest(source),data_identity_sha256=common.digest(data))
        else:receipt.update(family='feature_reference')
        ledger.append(receipt)
    keys=sorted(t['key'] for t in plan['tasks']);coverage=dict(total_tasks=15,sealed_complete_tasks=15,
        metric_expected_tasks=14,metric_valid_tasks=14,complete_task_keys=keys,complete_task_keys_sha256=common.digest(keys),
        missing_or_unsealed_task_keys=[],invalid_compact_metrics=[],compact_session_rows=len(allrows))
    report=dict(status='COMPLETE',scientific_scope='REAL_DATA',recipe_sha256=plan['recipe_sha256'],coverage=coverage,source_receipts=ledger)
    reports=output/'reports';reports.mkdir();common.atomic(reports/'report.json',report)
    common.atomic(reports/'report-status.json',dict(status='COMPLETE',recipe_sha256=plan['recipe_sha256'],report_sha256=common.sha(reports/'report.json')))
    return dict(root=root,package=package,plan=plan,path=path,expected_recipe=plan['recipe_sha256'],expected_plan_sha256=common.sha(path))


def inspect(f):
    return continuation.inspect_legacy(f['path'],root=f['root'],expected_recipe=f['expected_recipe'],expected_plan_sha256=f['expected_plan_sha256'])


def test_legacy_inspection_ignores_physical_old_code_but_verifies_all_compact_seals(prepared):
    f=legacy_fixture(prepared);(f['package']/'paper3r10/algorithm.py').write_text('unrelated physical legacy package changed')
    old={p:p.read_bytes() for p in Path(f['plan']['output']).rglob('*') if p.is_file()}
    proof=inspect(f)
    assert len(proof['rows'])==336 and proof['proof']['legacy_mechanism_completed_tasks']==15
    assert proof['proof']['legacy_physical_code_rehashed'] is False
    assert all(p.read_bytes()==b for p,b in old.items())


@pytest.mark.parametrize('kind',['csv','result','seal','report','registry','plan'])
def test_legacy_tampering_rejected_before_target_work(prepared,kind):
    f=legacy_fixture(prepared);folder=Path(f['plan']['output']);evaltask=next(t for t in f['plan']['tasks'] if t['family']=='evaluate')
    targets={'csv':folder/evaltask['key']/'session_metrics.csv','result':folder/evaltask['key']/'result.json',
        'seal':folder/evaltask['key']/'completion.json','report':folder/'reports/report.json',
        'registry':f['root']/'registry.json','plan':f['path']}
    path=targets[kind]
    if kind=='seal':
        value=json.loads(path.read_text());value['result_sha256']='0'*64;common.atomic(path,value)
    else:path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):inspect(f)
    assert not (f['root']/'runs/r10_formal_exploratory_optimized').exists()


def test_continuation_reuses_old15_and_prepares165_once_without_submit_or_imports(prepared,monkeypatch):
    f=legacy_fixture(prepared);forbidden=[]
    monkeypatch.setattr(runtime.subprocess,'Popen',lambda *a,**k:forbidden.append('Popen'))
    monkeypatch.setattr(runtime.subprocess,'run',lambda *a,**k:forbidden.append('run'))
    resultpath,result=continuation.prepare_continuation(f['root'],source_plan=f['path'],package=f['package'],
        expected_recipe=f['expected_recipe'],expected_plan_sha256=f['expected_plan_sha256'])
    new=common.read(resultpath);assert len(new['tasks'])==165 and new['resources']['memory_per_job']=='64G'
    assert new['resources']['jobs']==4 and result['submitted_jobs']==0 and not forbidden
    oldeval={t['key'] for t in f['plan']['tasks'] if t['family']=='evaluate'}
    assert not oldeval & {t['key'] for t in new['tasks']}
    amendment=common.read(Path(new['root'])/new['amendment']['path'])
    assert amendment['status']=='EXPLORATORY_SELECTED' and amendment['retrospective_amendment']
    assert amendment['performance_gates_are_diagnostics'] and amendment['legacy_tasks_rerun']==0
    assert sum('legacy_reference_reuse' in t for t in new['tasks'])==1
    assert all(t.get('optional_baseline') for t in new['tasks'] if t.get('requires_baseline_artifacts'))
    assert (Path(new['output'])/'reports/legacy_evidence/plan.json').read_bytes()==f['path'].read_bytes()
    assert not (runtime.runtime_dir(new)/'SUBMIT_AUTHORIZATION.json').exists()
    stamps={p:p.stat().st_mtime_ns for p in Path(new['output']).rglob('*') if p.is_file()}
    # Resume loads its exact new plan, without re-reading or reranking development.
    monkeypatch.setattr(continuation,'inspect_legacy',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No repeat old audit')))
    path2,r2=continuation.prepare_continuation(f['root'],package=f['package'],expected_recipe=f['expected_recipe'],expected_plan_sha256=f['expected_plan_sha256'])
    assert path2==resultpath and r2['reused_continuation_plan']
    assert all(p.stat().st_mtime_ns==ts for p,ts in stamps.items() if '/reports/' in str(p) or p.name=='plan.json')


def test_cli_continue_submit_is_explicit_and_calls_no_gpu_itself(tmp_path,monkeypatch,capsys):
    path=tmp_path/'plan.json';calls=[]
    monkeypatch.setattr(continuation,'prepare_continuation',lambda *a,**k:(path,dict(status='PREPARED_NOT_SUBMITTED')))
    monkeypatch.setattr(runtime,'start',lambda p,submit=False:calls.append((p,submit)) or dict(status='DETACHED_CONTROLLER_STARTED'))
    assert cli.main(['continue','--root',str(tmp_path)])==0 and not calls
    assert cli.main(['continue','--root',str(tmp_path),'--submit'])==0 and calls==[(path,True)]


@pytest.mark.parametrize('failure_point',['build','prepare'])
def test_partial_preparation_recovers_without_deleting_or_changing_amendment(prepared,monkeypatch,failure_point):
    from paper3r10 import amendment_selection
    f=legacy_fixture(prepared);original_select=amendment_selection.amended_development_decision;calls=[0]
    def timed(*a,**k):
        result=original_select(*a,**k);calls[0]+=1;result['updated_utc']='fixture-time-'+str(calls[0]);return result
    monkeypatch.setattr(amendment_selection,'amended_development_decision',timed)
    original_build=study.build_plan;original_prepare=runtime.prepare
    def fail(*a,**k):raise RuntimeError('Injected recoverable local preparation failure')
    monkeypatch.setattr(study if failure_point=='build' else runtime,'build_plan' if failure_point=='build' else 'prepare',fail)
    options=dict(source_plan=f['path'],package=f['package'],expected_recipe=f['expected_recipe'],expected_plan_sha256=f['expected_plan_sha256'])
    with pytest.raises(RuntimeError):continuation.prepare_continuation(f['root'],**options)
    amendment=f['root']/'runs/r10_formal_exploratory_optimized/reports/development_amendment.json'
    before=amendment.read_bytes()
    monkeypatch.setattr(study,'build_plan',original_build);monkeypatch.setattr(runtime,'prepare',original_prepare)
    plan,result=continuation.prepare_continuation(f['root'],**options)
    assert result['status']=='PREPARED_NOT_SUBMITTED' and len(common.read(plan)['tasks'])==165
    assert amendment.read_bytes()==before
