"""Versioned read-only W3 review with explicit descriptor-consumer history.

No raw native parse, network, GPU, fitting, predictions, internal episode load,
status mutation or repair. Static fork of the immutable historical verifier.
Default reports PENDING; --execute writes a new audit only after real completion.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work.contracts import file_digest
from p3work.episodes import load_episode_catalog
from p3work.window_acceptance import verify_frozen_window
from tools import prepare_stieger_cache as cache
from tools.r2_restore import load_verified_receipt
from tools import build_data_plan_deploy_manifest as deploy
from tools import build_data_plan_deploy_manifest_v2 as deploy_v2

HISTORICAL_VERIFIER='audit/data_plan_20261019/stieger/audit_completed_w3.py'
HISTORICAL_VERIFIER_SHA='56c8445f5476b41450fd2ce1bd9142e8bbe985dc5648552bc81f46523b7875f5'
PRECISION_AMENDMENT='plans/data_plan_20261019/classical_precision_amendment.json'
PRECISION_AMENDMENT_SHA='ce5e553833f54a6ef93c0c4938d3e651172c7be1dc7fa395f6f6cd04ec7da295'
FROZEN_CLOSURE='tools/build_data_plan_deploy_manifest.py'
FROZEN_CLOSURE_SHA='bf8e5664b445d2108db1586b655b70a57d04985e9601f62321ac1d69b64e588a'
PREVIOUS_VERIFIER='tools/audit_completed_stieger_w3.py'
PREVIOUS_VERIFIER_SHA='0f1dffe53ad7f2ebd00ce49d109a575e663b46d5462c51d35dc141500a397f70'
EPISODE_SOURCE='p3work/episodes.py'
EPISODE_OLD_SHA='348604283251b8d50a34c5b0545a895afb7bbaa024ff82d7c62be11e30504a65'
PREDICTION_SOURCE='p3work/prediction_store.py'
PREDICTION_OLD_SHA='e1af634b84bc234bc71394c6fff222d4cd67b2d4ce7f88bc2da5a40b4abde5ab'
OUTPUT='audit/data_plan_20261020/stieger_complete_quality_review_v2'


def pinned(root, path, sha):
    value=deploy.descriptor(root,path)
    if value['sha256']!=sha:raise RuntimeError('pinned_verifier_or_amendment_changed')
    return value


def bound(root, value):
    if (not isinstance(value,dict) or not {'path','sha256'}<=set(value)
            or set(value)-{'path','sha256','bytes'}):
        raise RuntimeError('exact_local_artifact_descriptor_required')
    actual=deploy.descriptor(root,value['path'])
    if any(actual[k]!=v for k,v in value.items()):
        raise RuntimeError('bound_artifact_SHA_or_bytes_mismatch')
    return actual


def software_integrity(root):
    """Verify 114 historical files and exactly three declared current fixes.

    The original 117-file inventory and both old verifiers remain immutable.
    No old snapshot is substituted for a current canonical consumer.
    """
    historical=pinned(root,HISTORICAL_VERIFIER,HISTORICAL_VERIFIER_SHA)
    previous=pinned(root,PREVIOUS_VERIFIER,PREVIOUS_VERIFIER_SHA)
    original=pinned(root,'audit/software_code_manifest.json',deploy.ORIGINAL_SOFTWARE_MANIFEST_SHA)
    amendment_item=pinned(root,PRECISION_AMENDMENT,PRECISION_AMENDMENT_SHA)
    amendment=json.loads((root/PRECISION_AMENDMENT).read_bytes())
    if (amendment.get('schema_version')!='paper3-classical-precision-compatibility-amendment-v1'
            or amendment.get('P1_P2_code_modified') is not False
            or amendment.get('legacy_baseline_relabelled_as_new_algorithm') is not False
            or amendment.get('old_fixed_archives_unchanged') is not True
            or amendment.get('original_protocol_52_unchanged') is not True
            or amendment.get('GPU_actions')!=0 or amendment.get('scientific_fits')!=0):
        raise RuntimeError('approved_precision_amendment_scope_required')
    closure=deploy.Closure(root)
    closure.register_precision_fix()
    if len(closure.precision_history)!=1:raise RuntimeError('exact_approved_current_precision_fix_required')
    history=closure.precision_history[0]
    expected={'changed_Paper3_code':history['current_implementation'],
        'original_code':history['archive_snapshot'],'original_117_manifest':history['original_117_manifest'],
        'native_read_only_regression':history['native_numerical_regression']}
    if any(amendment.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('precision_amendment_snapshot_or_native_proof_binding_changed')
    compatibility_closure=deploy_v2.Closure(root)
    compatibility=compatibility_closure.register_episode_history(defer_walk=True)
    if not isinstance(compatibility,dict):
        raise RuntimeError('exact_descriptor_consumer_compatibility_history_required')
    bound(root,compatibility.get('software_fix_contract'))
    bound(root,compatibility.get('actual_new_software_acceptance'))
    compatibility_inventory=bound(root,compatibility.get('original_117_manifest'))
    if compatibility_inventory['sha256']!=original['sha256'] or compatibility_inventory['bytes']!=original['bytes']:
        raise RuntimeError('compatibility_original_117_inventory_mismatch')
    consumers={EPISODE_SOURCE:(compatibility,EPISODE_OLD_SHA,43688),
        PREDICTION_SOURCE:(compatibility.get('prediction_store_history'),PREDICTION_OLD_SHA,32085)}
    for path,(record,old_sha,old_bytes) in consumers.items():
        if (not isinstance(record,dict) or record.get('historical_source_path')!=path
                or record.get('historical_source_sha256')!=old_sha
                or record.get('historical_source_bytes')!=old_bytes):
            raise RuntimeError('exact_two_descriptor_consumer_histories_required')
        archived=bound(root,record.get('archive_snapshot'))
        current=bound(root,record.get('current_implementation'))
        if (archived['sha256']!=old_sha or archived['bytes']!=old_bytes
                or current['path']!=path or current['sha256']==old_sha):
            raise RuntimeError('descriptor_consumer_original_and_current_binding_mismatch')
    old=json.loads((root/original['path']).read_bytes())
    rows=old.get('files',[])
    if len(rows)!=117 or len({r.get('path') for r in rows})!=117:
        raise RuntimeError('exact_unique_historical_117_manifest_required')
    unchanged=0
    for row in rows:
        if row['path']==deploy.PRECISION_SOURCE:
            if (row.get('sha256')!=deploy.PRECISION_OLD_SHA
                    or row.get('bytes')!=history['archive_snapshot']['bytes']):
                raise RuntimeError('original_precision_row_snapshot_mismatch')
        elif row['path'] in consumers:
            _,old_sha,old_bytes=consumers[row['path']]
            if row!={'path':row['path'],'sha256':old_sha,'bytes':old_bytes}:
                raise RuntimeError('original_descriptor_consumer_row_snapshot_mismatch')
        else:
            bound(root,row);unchanged+=1
    if unchanged!=114:raise RuntimeError('exact_114_historical_files_required')
    return {'historical_verifier':historical,'historical_117_manifest':original,
        'previous_0f1_verifier':previous,
        'historical_unchanged_files':114,'approved_modified_files':3,
        'precision_amendment':amendment_item,'precision_implementation_history':history,
        'descriptor_consumer_compatibility_history':compatibility,
        'historical_inventory_is_not_a_claim_of_117_current_unchanged_files':True}


def finite_nonnegative(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def session_accounting(root, state):
    """Recompute charged bodies from closed transfers; never open native raw."""
    if 'transfer_intent' in state:raise RuntimeError('closed_session_has_open_transfer_intent')
    totals={'official_provider_body_bytes':0,'r2_restore_body_bytes':0}
    verified={k.replace('body_bytes','verified_body_bytes'):0 for k in totals}
    history=state.get('transfer_history',[])
    if not isinstance(history,list):raise RuntimeError('closed_transfer_history_required')
    for item in history:
        kind=item.get('kind');reserve=item.get('reserved_bytes')
        if (kind not in {'official_get','r2_restore'} or type(reserve) is not int or reserve<0
                or item.get('file_id')!=state.get('file_id') or item.get('raw_path')!=state.get('raw_path')):
            raise RuntimeError('transfer_history_identity_or_reservation_invalid')
        key='official_provider_body_bytes' if kind=='official_get' else 'r2_restore_body_bytes'
        if item.get('status')=='SUCCESS_ACTUAL_BODY_SETTLED':
            amount=item.get('actual_verified_bytes')
            if type(amount) is not int or not 0<=amount<=reserve:
                raise RuntimeError('successful_transfer_actual_body_invalid')
            verified[key.replace('body_bytes','verified_body_bytes')]+=amount
        elif item.get('status') in {'FAILURE_FULL_RESERVATION_CONSUMED','HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED'}:
            amount=reserve
        else:raise RuntimeError('transfer_history_not_conservatively_closed')
        totals[key]+=amount
    for key,value in {**totals,**verified}.items():
        if type(state.get(key,0)) is not int or state.get(key,0)!=value:
            raise RuntimeError('session_transfer_conservation_mismatch')
    artifacts=state.get('artifacts',[])
    if (len(artifacts)!=3 or len({a['path'] for a in artifacts})!=3
            or {Path(a['path']).name for a in artifacts}!={'X.npy','events.json','metadata.json'}):
        raise RuntimeError('exact_three_session_artifacts_required')
    size=sum(bound(root,a)['bytes'] for a in artifacts)
    if type(state.get('cache_bytes')) is not int or state['cache_bytes']!=size:
        raise RuntimeError('session_cache_bytes_mismatch')
    receipt_path=cache.checked(root,state['cache_backup_receipt'])
    rows,_=load_verified_receipt(receipt_path)
    expected={(a['path'],a['sha256'],(root/a['path']).stat().st_size,
        'audit' if Path(a['path']).name=='metadata.json' else 'result','Stieger2021') for a in artifacts}
    if {tuple(r[k] for k in ('path','sha256','bytes','role','dataset')) for r in rows}!=expected:
        raise RuntimeError('exact_session_cache_receipt_bytes_and_roles_required')
    return {**totals,'cache_bytes':size}


def cpu_completion(root, plan, task, manifest_path, states):
    """Bind explicit cross-boot reconciliation, closed caps and actual full GET."""
    pinned(root,FROZEN_CLOSURE,FROZEN_CLOSURE_SHA)
    progress_path=cache.local(root,plan['output']+'/progress.json')
    progress=json.loads(progress_path.read_bytes())
    if (progress.get('status')!='W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY'
            or progress.get('completed_sessions')!=598 or len(states)!=598
            or progress.get('plan_sha256')!=plan['plan_sha256'] or 'wall_intent' in progress
            or progress.get('gpu_jobs_started')!=0 or progress.get('remote_jobs_started')!=0
            or progress.get('external_payload_accessed') is not False):
        raise RuntimeError('all598_CPU_progress_must_be_closed_without_GPU_remote_or_external')
    amendment=progress.get('explicit_runtime_reconciliation')
    if not isinstance(amendment,dict) or task.get('cpu_operational_recovery')!=amendment:
        raise RuntimeError('explicit_CPU_reconciliation_must_match_final_task')
    amendment_item=bound(root,amendment)
    amendment_value=json.loads((root/amendment_item['path']).read_bytes())
    if (Path(amendment_item['path']).name!='runtime_reconciliation_amendment.json'
            or not amendment_item['path'].startswith('audit/data_plan_20261020/runtime_recovery/')):
        raise RuntimeError('explicit_CPU_transaction_namespace_required')
    transaction=(Path(amendment_item['path']).parent/'transaction.json').as_posix()
    closure=deploy_v2.Closure(root);closure.register_precision_fix()
    closure.register_episode_history(defer_walk=True)
    closure.load_runtime_recovery(transaction)
    if (closure.errors or len(closure.runtime_recovery_history)!=1
            or closure.runtime_recovery_history[0]['amendment']!=amendment_item):
        raise RuntimeError('CPU_reconciliation_complete_evidence_closure_required')
    if amendment_value.get('Stieger_plan')!=deploy.descriptor(root,'audit/data_plan_20261019/stieger/cache_plan.json'):
        raise RuntimeError('reconciliation_original_Stieger_plan_binding_mismatch')
    completion=task.get('cpu_operational_completion');op_receipt=task.get('cpu_operational_R2_receipt')
    completion_item=bound(root,completion);op_item=bound(root,op_receipt)
    value=json.loads((root/completion_item['path']).read_bytes())
    cap=amendment_value.get('effective_Stieger_CPU_wall_cap_seconds')
    manifest=deploy.descriptor(root,manifest_path.relative_to(root).as_posix())
    if (task.get('cache_manifest')!=manifest or progress.get('cache_manifest')!=manifest
            or value.get('cache_manifest')!=manifest
            or progress.get('cache_r2_receipt')!=task.get('cache_r2_receipt')
            or value.get('original_global_cache_GET_receipt')!=task.get('cache_r2_receipt')
            or progress.get('cpu_operational_completion')!=completion
            or progress.get('cpu_operational_R2_receipt')!=op_receipt
            or value.get('schema_version')!='paper3-recovered-Stieger-CPU-completion-v1'
            or value.get('amendment')!=amendment
            or value.get('scientific_plan_sha256')!=plan['plan_sha256']
            or value.get('original_body_caps_unchanged') is not True
            or any(v.get('effective_CPU_wall_cap_seconds')!=cap for v in (value,task,progress))
            or value.get('original_plan_wall_cap_seconds')!=plan['max_wall_seconds']
            or progress.get('original_plan_wall_cap_seconds')!=plan['max_wall_seconds']
            or any(value.get(k)!=v for k,v in {'scientific_fits':0,'GPU_actions':0,
                'remote_server_actions':False,'external_data_access':False}.items())):
        raise RuntimeError('CPU_completion_scientific_identity_or_original_caps_changed')
    builder=amendment_value['implementation']
    if value.get('operational_builder')!=builder or progress.get('operational_builder')!=builder:
        raise RuntimeError('CPU_operational_builder_binding_changed')
    rows,_=load_verified_receipt(root/op_item['path'])
    required={(d['path'],d['sha256'],d['bytes'],role,'Stieger2021') for d,role in
        ((completion_item,'audit'),(amendment_item,'audit'),(bound(root,builder),'code'))}
    actual={tuple(r[k] for k in ('path','sha256','bytes','role','dataset')) for r in rows}
    if actual!=required or len(rows)!=3:raise RuntimeError('exact_three_operational_full_GET_objects_required')
    elapsed=progress.get('active_wall_elapsed_s')
    before=value.get('cumulative_wall_seconds_before_operational_backup')
    floor=amendment_value['conservative_accounting']['new_wall_high_water_seconds']['Stieger2021']
    if (not all(finite_nonnegative(x) for x in (elapsed,before,floor))
            or type(cap) is not int or not floor<=before<=elapsed<cap):
        raise RuntimeError('closed_CPU_wall_must_preserve_reconciled_high_water_and_strict_cap')
    totals={'official_provider_body_bytes':0,'r2_restore_body_bytes':0,'cache_bytes':0}
    for state in states:
        charges=session_accounting(root,state)
        for key in totals:totals[key]+=charges[key]
    for key,limit in (('official_provider_body_bytes','max_download_bytes'),
                      ('r2_restore_body_bytes','max_r2_restore_bytes'),('cache_bytes','max_cache_bytes')):
        if (type(progress.get(key)) is not int or progress[key]!=totals[key] or totals[key]>plan[limit]
                or (key!='cache_bytes' and value.get(key)!=totals[key])):
            raise RuntimeError('closed_CPU_body_or_cache_accounting_must_match_original_caps')
    return {'closed_progress':cache.descriptor(root,progress_path),'reconciliation':amendment_item,
        'recovery_transaction':closure.runtime_recovery_history[0]['transaction'],
        'completion':completion_item,'operational_full_GET_receipt':op_item,
        'operational_R2_objects':3,'reconciliation_conservative_accounting_recomputed':True,
        'cumulative_wall_seconds':elapsed,'effective_CPU_wall_cap_seconds':cap,
        'original_plan_wall_cap_seconds':plan['max_wall_seconds'],'charged_bodies_and_cache':totals,
        'original_body_caps_unchanged':True,'current_boot_not_used_to_reinterpret_closed_history':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve(strict=True)
    directory=root/'audit/data_plan_20261019/stieger'
    plan_path=directory/'cache_plan.json';plan=json.loads(plan_path.read_text())
    tasks=json.loads((root/'status/tasks.json').read_text())
    complete=tasks.get('W1',{}).get('status')=='COMPLETE' and tasks.get('W3',{}).get('status')=='COMPLETE'
    if not args.execute or not complete:
        print(json.dumps({'status':'PENDING_REAL_COMPLETED_W3_READ_ONLY_AUDIT' if not complete else 'PENDING_EXPLICIT_READ_ONLY_AUDIT',
            'required_W1':'COMPLETE','required_W3':'COMPLETE','completion_report_written':False,
            'current_W3':tasks.get('W3',{}).get('status'),'expected_sessions':598,
            'episode_roles_to_load':['meta_train','meta_validation'],'internal_episode_loader_allowed':False,
            'network_operations':0,'GPU_actions':0,'actual_fits':0},indent=2))
        return 0 if not args.execute else 2
    output=cache.local(root,OUTPUT)
    if output.exists():raise RuntimeError('preexisting_audit_preserved_no_overwrite')
    software=software_integrity(root)
    _,split=cache.validate_plan(root,plan,execute=True)
    window=verify_frozen_window(root)
    if tasks['W3']['plan_sha256']!=plan['plan_sha256']:raise RuntimeError('W3_plan_binding_mismatch')
    manifest_path=cache.checked(root,tasks['W3']['cache_manifest'])
    manifest=json.loads(manifest_path.read_text())
    if (manifest.get('status')!='STIEGER_CACHE_COMPLETE' or manifest.get('session_count')!=598
            or manifest.get('participant_count')!=62 or manifest.get('synthetic') is not False
            or manifest.get('old_npz_reused') is not False or manifest.get('training_started') is not False
            or manifest.get('internal_holdout_evaluated') is not False
            or manifest.get('external_payload_accessed') is not False):
        raise RuntimeError('real_all598_unfitted_cache_manifest_required')
    by_name={r['source_record_id']:r for r in manifest['sessions']}
    if len(by_name)!=598:raise RuntimeError('session_registry_not_unique')
    states=[];raw_receipts={};cache_receipts={};partition_counts=Counter();physical_source_bytes=0
    for entry in plan['files']:
        state_path=root/plan['output']/entry['partition']/entry['provider_subject']/f"session-{entry['session_ordinal']}"/'state.json'
        state=json.loads(state_path.read_text())
        if not cache.verify_session_state(state,root,plan):raise RuntimeError('incomplete_session_state')
        if state['session_record']!=by_name[entry['name']]:raise RuntimeError('state_manifest_session_record_mismatch')
        if (state.get('file_id')!=entry['id'] or state.get('name')!=entry['name']
                or state.get('partition')!=entry['partition']):
            raise RuntimeError('state_not_exact_official_session_identity')
        expected_artifacts={str(state_path.parent.relative_to(root)/name) for name in ('X.npy','events.json','metadata.json')}
        if ({a['path'] for a in state['artifacts']}!=expected_artifacts
                or {(a['path'],a['sha256']) for a in state['artifacts']}
                    !={(state['session_record'][k]['path'],state['session_record'][k]['sha256']) for k in ('X','events','metadata')}):
            raise RuntimeError('state_artifacts_not_exact_session_record_and_directory')
        if state['raw_integrity']['bytes']!=entry['size']:raise RuntimeError('official_size_mismatch')
        raw_receipts[state['raw_backup_receipt']['path']]=state['raw_backup_receipt']['sha256']
        cache_receipts[state['cache_backup_receipt']['path']]=state['cache_backup_receipt']['sha256']
        states.append(state);partition_counts[entry['partition']]+=1;physical_source_bytes+=entry['size']
    final_receipt=tasks['W3']['cache_r2_receipt']
    global_receipt_path=cache.checked(root,final_receipt)
    global_rows,_=load_verified_receipt(global_receipt_path)
    required={manifest_path.relative_to(root).as_posix(),manifest['duplicate_audit']['path'],
        *(plan[k]['path'] for k in ('official_inventory','input_contract','participant_split')),
        *cache_receipts}
    if len(global_rows)!=len(required) or {r['path'] for r in global_rows}!=required:
        raise RuntimeError('global_full_GET_receipt_does_not_cover_exact_manifest_and_all_session_receipts')
    expected_roles={manifest_path.relative_to(root).as_posix():'manifest',manifest['duplicate_audit']['path']:'audit',
        **{plan[k]['path']:'protocol' for k in ('official_inventory','input_contract','participant_split')},
        **{path:'audit' for path in cache_receipts}}
    for row in global_rows:
        path=cache.local(root,row['path'])
        if (file_digest(path)!=row['sha256'] or path.stat().st_size!=row['bytes']
                or row['role']!=expected_roles[row['path']] or row['dataset']!='Stieger2021'):
            raise RuntimeError('global_backed_metadata_changed')
    cpu=cpu_completion(root,plan,tasks['W3'],manifest_path,states)
    duplicate_path=cache.checked(root,manifest['duplicate_audit'])
    duplicate=json.loads(duplicate_path.read_text())
    if duplicate.get('status')!='NO_CROSS_SPLIT_EXACT_SIGNAL_DUPLICATES':
        raise RuntimeError('cross_partition_signal_duplicate_gate_failed')
    role_reports={};session_reports=[];blocking=[]
    for role in ('meta_train','meta_validation'):
        catalog=load_episode_catalog(root,role=role)
        people={p:{'sessions':0,'query_eligible_sessions':0,'fallback_sessions':0,
            'original_LR_trials':0,'valid_support_trials':0,'valid_future_query_trials':0,
            'invalid_prefix_trials':0,'invalid_future_query_trials':0} for p in catalog.partition_people}
        query_reasons=Counter();fallback_reasons=Counter()
        for episode in catalog.plans:
            audit=json.loads(episode.temporal_audit_json)
            summary=people[episode.participant_id]
            summary['sessions']+=1;summary['query_eligible_sessions']+=episode.query_eligible
            summary['fallback_sessions']+=episode.fallback_reason is not None
            summary['original_LR_trials']+=audit['original_lr_trials']
            summary['valid_support_trials']+=len(episode.support_event_uids)
            summary['valid_future_query_trials']+=len(episode.query_event_uids)
            summary['invalid_prefix_trials']+=audit['invalid_prefix_trials']
            summary['invalid_future_query_trials']+=len(episode.query_excluded_uids)
            if episode.fallback_reason is not None:fallback_reasons[episode.fallback_reason]+=1
            query_reasons['eligible' if episode.query_eligible else 'insufficient_future_query_or_class_count']+=1
            if (set(episode.acquired_event_uids)&set(episode.query_event_uids)
                    or set(episode.support_event_uids)&set(episode.query_event_uids)
                    or set(episode.unused_boundary_run_uids)&set(episode.query_event_uids)
                    or set(episode.incomplete_future_run_uids)&set(episode.query_event_uids)):
                raise RuntimeError('support_query_boundary_UID_overlap')
            session_reports.append({'participant_id':episode.participant_id,'session_id':episode.session_id,'role':role,
                'original_LR_trials':audit['original_lr_trials'],'acquired_prefix_cost':len(episode.acquired_event_uids),
                'valid_support_trials':len(episode.support_event_uids),'valid_future_query_trials':len(episode.query_event_uids),
                'query_eligible':episode.query_eligible,'fallback_reason':episode.fallback_reason,
                'unused_boundary_run_trials':len(episode.unused_boundary_run_uids),
                'incomplete_future_run_trials':len(episode.incomplete_future_run_uids),
                'invalid_future_query_trials':len(episode.query_excluded_uids),
                'query_class_counts':audit['query_class_counts'],'query_rule':episode.query_rule,
                'temporal_audit_sha256':episode.temporal_audit_sha256})
        missing=[p for p,v in people.items() if v['query_eligible_sessions']==0]
        if missing:blocking.append({'role':role,'people_without_eligible_future_query':missing,
            'policy':'preserve frozen identities; do not replace/drop participants or fabricate episodes'})
        role_reports[role]={'participant_count':len(people),'session_count':len(catalog.plans),
            'query_eligibility_summary':dict(query_reasons),'fallback_reason_counts':dict(fallback_reasons),
            'participants':people,'catalog_identity_sha256':catalog.identity_sha256,
            'scientific_fit_or_prediction_performed':False}
    # Recheck all current code and the exact three declared histories again.
    # The original 117-file inventory remains historical evidence.
    if software_integrity(root)!=software:raise RuntimeError('software_changed_during_read_only_audit')
    if cache.descriptor(root,cache.local(root,plan['output']+'/progress.json'))!=cpu['closed_progress']:
        raise RuntimeError('closed_progress_changed_during_read_only_audit')
    report={'schema_version':'paper3-full-W3-independent-quality-review-v3',
        'status':'PASS_REAL_W3_ARTIFACTS_AND_TRAIN_VALIDATION_EPISODE_QUALITY' if not blocking else 'BLOCKED_FROZEN_PEOPLE_WITHOUT_ELIGIBLE_QUERY',
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'cache_manifest':tasks['W3']['cache_manifest'],
        'global_full_GET_receipt':final_receipt,'plan_sha256':plan['plan_sha256'],
        'window_gate_status':window['status'],'all598_session_state_and_artifact_SHA_passed':True,
        'session_count':len(states),'source_raw_bytes':physical_source_bytes,'partition_session_counts':dict(partition_counts),
        'original_raw_R2_receipts_verified':len(raw_receipts),'per_session_cache_R2_receipts_verified':len(cache_receipts),
        'global_metadata_R2_objects_verified':len(global_rows),'cross_split_exact_signal_duplicates':0,
        'within_partition_exact_signal_duplicates':len(duplicate['within_partition_duplicates']),
        'train_validation_episode_quality':role_reports,'blocking_people':blocking,
        'internal_cache_artifact_bytes_read_for_integrity_hash_only':True,'internal_episode_loader_invoked':False,
        'internal_training_or_evaluation_performed':False,'query_materialization_or_label_capabilities_invoked':False,
        'scientific_fits':0,'GPU_actions':0,'network_operations':0,'sealed_external_payloads_accessed':False,
        'historical_114_code_SHA_unchanged':True,'approved_original_file_change_count':3,
        'approved_precision_modified_file_count':1,'approved_descriptor_consumer_modified_file_count':2,
        'software_integrity':software,'CPU_operational_completion':cpu,
        'verifier':cache.descriptor(root,Path(__file__).resolve())}
    output.mkdir(parents=True)
    cache.write_once_json(output/'session_candidate_quality.json',session_reports)
    report['session_candidate_quality']=cache.descriptor(root,output/'session_candidate_quality.json')
    cache.write_once_json(output/'independent_quality_review.json',report)
    print(json.dumps({'status':report['status'],'report':cache.descriptor(root,output/'independent_quality_review.json'),
        'blocking_people':blocking,'actual_fits':0,'GPU_actions':0,'network_operations':0},indent=2))
    return 0 if not blocking else 2


if __name__=='__main__':
    raise SystemExit(main())
