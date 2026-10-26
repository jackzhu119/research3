"""Read sealed R10 development evidence and prepare an explicit amended continuation.

The legacy reader never imports the old package or passes its plan through the
new executable-code gate. It verifies original plan/task identities and only
rehashes compact evidence required for the development decision, once.
"""
from __future__ import annotations
import csv
import io
import json
import math
from pathlib import Path
from .platform.common import atomic_bytes,digest,encoded,local,read,sha,utc

SOURCE_RECIPE='d9bbaad2f1af0159da345b4c1e73a90bf0aa05268f9466bc01150a5e72cf0574'
SOURCE_PLAN_SHA256='a8e414b8db28ffe8b4d73bbc579764908921e8903c2b5c20e963d6197032ab13'


def immutable(path,data):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('An immutable continuation record already differs: '+str(path))
    else:atomic_bytes(path,data)
    return path


def immutable_record(path,value):
    """Reuse original record bytes on retry; timestamps do not change science."""
    path=Path(path)
    if path.exists():
        previous=json.loads(stable_bytes(path))
        def scientific(record):
            return {k:v for k,v in record.items() if k not in ('updated_utc','created_utc')}
        if scientific(previous)!=scientific(value):
            raise ValueError('An immutable continuation scientific decision already differs')
        return path
    return immutable(path,encoded(value))


def stable_bytes(path,limit=16*1024*1024):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('Required compact legacy evidence is absent: '+str(path))
    a=path.stat()
    if a.st_size>limit:raise ValueError('Compact legacy evidence exceeds its bounded size')
    data=path.read_bytes();b=path.stat()
    if (a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns):
        raise ValueError('Legacy evidence changed during capture')
    return data


def _descriptor_bytes(folder,item):
    if not isinstance(item,dict) or type(item.get('bytes')) is not int:raise ValueError('Legacy artifact descriptor is invalid')
    p=local(folder,item['path']);data=stable_bytes(p)
    import hashlib
    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item.get('sha256'):
        raise ValueError('Legacy compact evidence SHA256 differs: '+item['path'])
    return data


def inspect_legacy(source_plan,*,root=None,expected_recipe=SOURCE_RECIPE,expected_plan_sha256=SOURCE_PLAN_SHA256):
    """Validate original metadata without executing, rewriting or rehashing EEG/predictions."""
    import hashlib
    source_plan=Path(source_plan).expanduser().resolve();plan_bytes=stable_bytes(source_plan)
    plan_sha=hashlib.sha256(plan_bytes).hexdigest();plan=json.loads(plan_bytes)
    if expected_plan_sha256 and plan_sha!=expected_plan_sha256:raise ValueError('The registered original R10 plan SHA256 differs')
    if (plan.get('schema_version')!='paper3-r10-plan-v1' or plan.get('round')!=10
            or plan.get('stage')!='mechanism' or plan.get('synthetic_fixture')):
        raise ValueError('Continuation requires the original real-data R10 mechanism plan')
    recipe=plan.get('recipe_sha256')
    if recipe!=digest({k:v for k,v in plan.items() if k not in ('recipe_sha256','operational_metadata')}):
        raise ValueError('The original R10 plan content differs from its signed recipe')
    if expected_recipe and recipe!=expected_recipe:raise ValueError('Unexpected original R10 mechanism recipe')
    project=Path(plan['root']).resolve();output=Path(plan['output']).resolve()
    if root is not None and project!=Path(root).expanduser().resolve():raise ValueError('Legacy project root differs')
    if not output.is_relative_to(project) or output==project:raise ValueError('Legacy scientific output escaped the project')
    tasks=plan.get('tasks',[]);keys=[t.get('key') for t in tasks]
    if len(tasks)!=15 or len(set(keys))!=15:raise ValueError('All original 15 mechanism tasks are required')
    if any(t.get('role') not in ('meta_train','meta_validation') for t in tasks):raise ValueError('Legacy selection evidence contains target roles')
    registry=plan.get('study_registry',{});rp=Path(registry.get('path',''))
    rp=rp if rp.is_absolute() else project/rp
    registry_bytes=_descriptor_bytes(rp.parent,dict(registry,path=rp.name))
    inputs=json.loads(registry_bytes)
    if Path(inputs.get('root','')).resolve()!=project:raise ValueError('Legacy registry root differs')
    report_bytes=stable_bytes(output/'reports/report.json');report=json.loads(report_bytes)
    report_sha=hashlib.sha256(report_bytes).hexdigest();report_status_bytes=stable_bytes(output/'reports/report-status.json')
    report_status=json.loads(report_status_bytes);coverage=report.get('coverage',{})
    if (report.get('status')!='COMPLETE' or report.get('recipe_sha256')!=recipe
            or report_status.get('status')!='COMPLETE' or report_status.get('report_sha256')!=report_sha
            or report_status.get('recipe_sha256')!=recipe or report.get('scientific_scope')!='REAL_DATA'
            or coverage.get('complete_task_keys')!=sorted(keys) or coverage.get('sealed_complete_tasks')!=15
            or coverage.get('total_tasks')!=15 or coverage.get('invalid_compact_metrics')
            or coverage.get('missing_or_unsealed_task_keys')
            or coverage.get('complete_task_keys_sha256')!=digest(sorted(keys))):
        raise ValueError('Legacy report does not cover the exact sealed mechanism plan')
    sources={'plan.json':plan_bytes,'registry.json':registry_bytes,'reports/report.json':report_bytes,
             'reports/report-status.json':report_status_bytes};rows=[];ledger=[];bindings=[]
    for task in tasks:
        key=task['key'];folder=local(output,key);rb=stable_bytes(folder/'result.json');cb=stable_bytes(folder/'completion.json')
        result=json.loads(rb);completion=json.loads(cb);result_sha=hashlib.sha256(rb).hexdigest()
        identity=digest({'schema':'paper3-r10-task-v1','recipe_sha256':recipe,'stage':'mechanism','task':task})
        if (result.get('status')!='COMPLETE' or result.get('scientific_scope')!='REAL_DATA'
                or result.get('task_identity')!=identity or completion.get('task_identity')!=identity
                or result.get('recipe_sha256')!=recipe or completion.get('recipe_sha256')!=recipe
                or result.get('task_key')!=key or completion.get('task_key')!=key
                or completion.get('schema_version')!='paper3-r10-completion-v1'
                or completion.get('scientific_result_verified') is not True
                or completion.get('result_sha256')!=result_sha or completion.get('artifacts')!=result.get('artifacts')):
            raise ValueError('Legacy task completion seal differs: '+key)
        if result.get('registry_sha256')!=registry['sha256']:raise ValueError('Legacy task registry identity differs: '+key)
        sources[key+'/result.json']=rb;sources[key+'/completion.json']=cb
        bindings.append(dict(task_key=key,task_identity=identity,result_sha256=result_sha,
                             completion_sha256=hashlib.sha256(cb).hexdigest()))
        if task.get('family')=='feature_reference':
            ledger.append(dict(task_key=key,result_sha256=result_sha,family='feature_reference'));continue
        descriptors=[x for x in result.get('artifacts',[]) if x.get('path')=='session_metrics.csv']
        if len(descriptors)!=1:raise ValueError('Legacy task lacks one sealed compact metrics artifact')
        data=_descriptor_bytes(folder,descriptors[0]);sources[key+'/session_metrics.csv']=data
        compact=list(csv.DictReader(io.StringIO(data.decode('utf-8'),newline='')))
        if len(compact)!=result.get('completed_session_budget_cells') or not compact:raise ValueError('Legacy compact metrics coverage differs')
        source_identity=digest(result['source_identity']);data_identity=digest(result['catalog_identity']);seen=set();cells={}
        for row in compact:
            if (row.get('method')!=task['method'] or int(row['seed'])!=int(task['seed'])
                    or int(row['budget']) not in task['budgets'] or row.get('role')!='meta_validation'
                    or row.get('cohort')!='meta_validation' or not row.get('participant_id') or not row.get('session_id')
                    or row.get('source_identity_sha256')!=source_identity or row.get('data_identity_sha256')!=data_identity
                    or len(row.get('query_signature',''))!=64):
                raise ValueError('Legacy compact metrics method/source/data/role identity differs')
            for metric in ('ba','balanced_nll','balanced_brier'):
                value=float(row[metric])
                if not math.isfinite(value) or value<0 or (metric!='balanced_nll' and value>1+1e-12):raise ValueError('Legacy metric is invalid')
            cell=(row['participant_id'],row['session_id'],int(row['seed']),int(row['budget']))
            if cell in seen:raise ValueError('Legacy compact metrics duplicate a session/budget cell')
            seen.add(cell);c=cells.setdefault(cell[:3],{'budgets':set(),'query':set()});c['budgets'].add(int(row['budget']));c['query'].add(row['query_signature'])
            row.update(task_key=key,candidate_id=task.get('candidate_id',''),configuration_id=digest(task.get('config',{})))
        if (any(c['budgets']!=set(task['budgets']) or len(c['query'])!=1 for c in cells.values())
                or len(cells)!=result.get('sessions') or len({x['participant_id'] for x in compact})!=result.get('participants')):
            raise ValueError('Legacy session, participant or fixed-query budget coverage differs')
        rows.extend(compact);ledger.append(dict(task_key=key,result_sha256=result_sha,compact_metrics=descriptors[0],
            rows=len(compact),source_identity_sha256=source_identity,data_identity_sha256=data_identity))
    if len(rows)!=coverage.get('compact_session_rows') or len(ledger)!=15:raise ValueError('Legacy report row ledger differs')
    expected_ledger={x['task_key']:x['result_sha256'] for x in report.get('source_receipts',[])}
    if expected_ledger!={x['task_key']:x['result_sha256'] for x in ledger}:raise ValueError('Legacy report source-result ledger differs')
    proof=dict(schema_version='paper3-r10-legacy-evidence-proof-v1',scientific_scope='REAL_DATA',source_plan=str(source_plan),
        source_plan_sha256=plan_sha,source_recipe_sha256=recipe,source_code_manifest_sha256=plan.get('code_manifest_sha256'),
        source_report_sha256=report_sha,registry_sha256=registry['sha256'],legacy_mechanism_completed_tasks=15,
        exact_task_bindings=bindings,legacy_physical_code_imported=False,legacy_physical_code_rehashed=False,
        full_predictions_rehashed=False,original_scientific_outputs_modified=False)
    return dict(plan=plan,rows=rows,coverage=coverage,ledger=ledger,proof=proof,sources=sources,registry=registry)


def prepare_continuation(root,*,source_plan=None,output=None,package=None,python=None,
                         expected_recipe=SOURCE_RECIPE,expected_plan_sha256=SOURCE_PLAN_SHA256):
    from .study import PACKAGE,build_plan,describe,production_manifest,write_plan
    from .platform.runtime import prepare
    root=Path(root).expanduser().resolve();package=Path(package or PACKAGE).expanduser().resolve()
    output=Path(output or root/'runs/r10_formal_exploratory_optimized').expanduser().resolve()
    if output==root/'runs/r10_mechanism' or not output.is_relative_to(root):raise ValueError('Continuation requires a new output inside this project')
    plan_path=output/'plan.json'
    if plan_path.is_file():
        previous=read(plan_path,{})
        if (previous.get('stage')!='formal_exploratory' or previous.get('retrospective_amendment') is not True
                or not previous.get('amendment') or len(previous.get('tasks',[]))!=165
                or previous.get('legacy_mechanism_rerun_tasks')!=0 or previous.get('legacy_mechanism_reused') is not True
                or Path(previous.get('root','')).resolve()!=root or Path(previous.get('output','')).resolve()!=output
                or any(previous.get('resources',{}).get(k)!=v for k,v in
                    (('jobs',4),('gpus_per_job',1),('cpus_per_job',16),('memory_per_job','64G')))):
            raise ValueError('Existing output is not this explicit amended165-task continuation')
        from .platform.common import verify_descriptor
        amended=read(verify_descriptor(root,previous['amendment']),{})
        evidence=amended.get('legacy_evidence',{})
        if (amended.get('legacy_mechanism_completed_tasks')!=15 or amended.get('legacy_tasks_rerun')!=0
                or (expected_plan_sha256 and amended.get('source_plan_sha256')!=expected_plan_sha256)
                or (expected_recipe and evidence.get('source_recipe_sha256')!=expected_recipe)):
            raise ValueError('Existing continuation has a different legacy evidence binding')
        result=prepare(plan_path);result.update(reused_continuation_plan=True,legacy_tasks_rerun=0)
        return plan_path,result
    original=inspect_legacy(source_plan or root/'runs/r10_mechanism/plan.json',root=root,
        expected_recipe=expected_recipe,expected_plan_sha256=expected_plan_sha256)
    if output==Path(original['plan']['output']).resolve():raise ValueError('Cannot overwrite original R10 mechanism output')
    from .amendment_selection import amended_development_decision
    decision=amended_development_decision(original['plan'],original['rows'],original['coverage'],original['ledger'],
        registry_sha256=original['registry']['sha256'],source_plan_sha256=original['proof']['source_plan_sha256'],
        source_report_sha256=original['proof']['source_report_sha256'])
    if decision.get('status')!='EXPLORATORY_SELECTED':raise ValueError('Development evidence identity/coverage is invalid; no target work was prepared')
    code_sha=digest(production_manifest(package))
    amendment=dict(decision,schema_version='paper3-r10-exploratory-amendment-v1',retrospective_amendment=True,
        performance_gates_are_diagnostics=True,study_registry_sha256=original['registry']['sha256'],
        reference_strategy=original['plan'].get('reference_strategy'),code_manifest_sha256=code_sha,
        legacy_mechanism_completed_tasks=15,legacy_evidence=original['proof'],
        target_query_labels_used_for_selection=False,source_plan_sha256=original['proof']['source_plan_sha256'],
        source_report_sha256=original['proof']['source_report_sha256'],
        continuation_scope='Exploratory amended selection after mechanism data; not independent confirmation',
        original_records_preserved=True,legacy_tasks_rerun=0)
    reports=output/'reports';evidence=reports/'legacy_evidence'
    for relative,data in original['sources'].items():immutable(evidence/relative,data)
    immutable(evidence/'PROOF.json',encoded(original['proof']))
    amendment_path=immutable_record(reports/'development_amendment.json',amendment)
    plan=build_plan(root,original['registry'],stage='formal_exploratory',package=package,output=output,
        amendment=describe(amendment_path,root),python=python,require_artifacts=False)
    # The original seed18 geometry is imported as exact sealed bytes, without
    # executing any original mechanism task or reopening its model/catalog.
    old_ref=next(t for t in original['plan']['tasks'] if t.get('family')=='feature_reference')
    old_folder=Path(original['plan']['output'])/old_ref['key']
    old_result=json.loads(original['sources'][old_ref['key']+'/result.json'])
    ref_artifacts={a['path']:a for a in old_result['artifacts']}
    def old_descriptor(path,data=None):
        path=Path(path);data=stable_bytes(path) if data is None else data
        import hashlib
        return dict(path=str(path.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    ref_reuse=dict(schema_version='paper3-r10-legacy-reference-reuse-v1',
        source_recipe_sha256=original['plan']['recipe_sha256'],task_key=old_ref['key'],
        task_identity=old_result['task_identity'],registry_sha256=old_result['registry_sha256'],
        source_identity=old_result['source_identity'],catalog_identity=old_result['catalog_identity'],
        source_plan=old_descriptor(original['proof']['source_plan'],original['sources']['plan.json']),
        result=old_descriptor(old_folder/'result.json',original['sources'][old_ref['key']+'/result.json']),
        completion=old_descriptor(old_folder/'completion.json',original['sources'][old_ref['key']+'/completion.json']))
    for field,name in (('reference_json','reference.json'),('reference_npz','reference.npz')):
        if name not in ref_artifacts:raise ValueError('Original reference lacks its sealed geometry artifact: '+name)
        ref_reuse[field]=dict(ref_artifacts[name],path=str((old_folder/name).relative_to(root)))
    matched=[t for t in plan['tasks'] if t.get('family')=='feature_reference' and t.get('seed')==old_ref['seed']]
    if len(matched)!=1:raise ValueError('Exactly one matching legacy reference import is required')
    matched[0]['legacy_reference_reuse']=ref_reuse
    matched[0]['input_identities']['legacy_reference_result_sha256']=ref_reuse['result']['sha256']
    from .platform.common import sign_plan
    plan=sign_plan(plan)
    if len(plan['tasks'])!=165:raise ValueError('The amended R10 formal plan must contain exactly165 new tasks')
    if any(t['key'] in {x['key'] for x in original['plan']['tasks'] if x.get('family')!='feature_reference'} for t in plan['tasks']):
        raise ValueError('Old mechanism evaluation tasks cannot enter the continuation execution plan')
    write_plan(plan,plan_path);result=prepare(plan_path)
    result.update(development_amendment=describe(amendment_path,root),legacy_mechanism_completed_tasks=15,
        legacy_tasks_rerun=0,retrospective_amendment=True,performance_gates_are_diagnostics=True)
    return plan_path,result
