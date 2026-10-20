#!/usr/bin/env python3
"""Inspect all frozen source EDF structures before another scientific parser freeze.

This is a source-only metadata qualification, never a training/cache admission.
The sole root coordinator may execute it beside the existing Yang/Stieger jobs
after the Dreyer V4 worker has stopped. New raw objects stay owned here; no raw
cleanup, EEG array read, fabricated stop event, fit, GPU or external access.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import source_extension_coordinator as core
from p3work import source_extension_io as storage

ENTRY='tools/qualify_dreyer_native_structures.py'
PARENT='971c1a1272fdb25b0f21e7d69c4e7349593d352341bdaa2eaa0725b600dbc1d1'
PARENT_PLAN='audit/source_extensions_20261019/Dreyer2023/plan_after_acquisition_control_review.json'
BASE='audit/source_extensions_20261019/Dreyer2023/native_structure_qualification'
SCHEMA='paper3-dreyer-native-structure-qualification-v1'
CODE=(ENTRY,'tools/prepare_dreyer_source_cpu_v4.py','p3work/source_extension_coordinator.py',
      'p3work/source_extension_io.py','p3work/source_extensions/dreyer2023.py','tools/r2_backup.py','tools/r2_restore.py')

def read(path):return json.loads(Path(path).read_bytes())

def private_module(root,path,name):
    spec=importlib.util.spec_from_file_location(name,core.local(root,path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class MetadataNumericMarkers:
    """Capture unknown numeric triggers in a private metadata reader only."""
    def __contains__(self,value):
        return isinstance(value,str) and value.isdecimal() and 0<=int(value)<=65535

def native_structure(root,raw):
    raw=raw if raw.is_absolute() else root/raw
    common={'source_file':core.descriptor(root,raw),'EEG_array_samples_read':False,
            'annotations_modified_or_synthesized':False,'scientific_admission_granted':False}
    module=private_module(root,'p3work/source_extensions/dreyer2023.py','_p3_native_qualification_only')
    scientific_markers=set(module.ALLOWED_MARKERS)
    module.ALLOWED_MARKERS=MetadataNumericMarkers()
    try:
        native=module._read_native_metadata(raw)
    except ValueError as exc:
        return dict(common,status='NATIVE_METADATA_REVIEW_REQUIRED',error_type='ValueError',error_code=str(exc))
    counts=dict(Counter(r['description'] for r in native['annotations']))
    unknown=sorted(set(counts)-scientific_markers)
    outside=[r['original_annotation_index'] for r in native['annotations']
             if r['onset']+r['duration']>Fraction(native['n_times'],512)]
    rows=[{k:r[k] for k in ('original_annotation_index','data_record','onset_token','duration_token','description')}
          for r in native['annotations']]
    try:
        if unknown:raise ValueError('Unknown numeric triggers require review under the frozen V4 metadata gate')
        if outside:raise ValueError('Original annotation lies outside the physical EDF clock')
        trials=module._trials(native)
        gate={'status':'EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS','native_trial_count':len(trials)}
    except ValueError as exc:
        gate={'status':'EXISTING_V4_GATE_REQUIRES_REVIEW','error_code':str(exc)}
    return {**common,'status':'NATIVE_TAL_STRUCTURE_CAPTURED_FOR_REVIEW','native_marker_counts':counts,
            'unknown_numeric_markers':unknown,'annotations_outside_physical_clock':outside,
            'native_n_times':native['n_times'],'native_record_count':native['record_count'],
            'native_signal_order':native['signal_labels'],'native_signal_units':native['signal_units'],
            'native_annotations':rows,'existing_V4_gate':gate,'annotations_modified_or_synthesized':False,
            'EEG_array_samples_read':False,'scientific_admission_granted':False}

def make_plan(root):
    wrapper=private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_qualification_parent')
    parent,_,_=wrapper.fixed_plans(root)
    audit=f'audit/source_extensions_20261019/Dreyer2023/{PARENT}'
    claim=read(root/audit/'parallel_worker_claim.json')
    if claim.get('active_invocation') is not False:raise core.PreparationError('stop_Dreyer_parent_before_qualification')
    ledger_path=root/audit/'budget_ledger.json';ledger=read(ledger_path)
    charged,wall=wrapper.closed_history(ledger)
    budgets={k:parent['budgets'][k]-v for k,v in charged.items()}
    budgets['wall_seconds']=min(10800,math.floor(parent['budgets']['wall_seconds']-wall))
    if any(type(v)is not int or v<=0 for v in budgets.values()):raise core.PreparationError('parent_remaining_budget_exhausted')
    reused={}
    for record in parent['records']:
        path=root/audit/record['record_id']/'state.json'
        if path.is_file() and read(path).get('status')=='COMPLETE':
            state=read(path)
            if not core._state_verified(root,state,parent,record):raise core.PreparationError('prior_complete_proofs_required')
            reused[record['record_id']]=core.descriptor(root,path)
    value={'schema_version':SCHEMA,'dataset':'Dreyer2023','role':'source_native_metadata_qualification_only',
           'parent_plan':core.descriptor(root,root/PARENT_PLAN),'parent_closed_ledger':core.descriptor(root,ledger_path),
           'parent_claim':core.descriptor(root,root/audit/'parallel_worker_claim.json'),
           'parent_actual_charged':charged,'parent_actual_elapsed_seconds':wall,'budgets':budgets,
           'record_count':520,'records':parent['records'],'reused_complete_parent_states':reused,
           'code':{p:core.descriptor(root,root/p) for p in CODE},'maximum_owned_raw_bytes':6*1024**3,
           'minimum_free_disk_bytes':4*1024**3,'raw_cleanup_allowed':False,
           'EEG_array_samples_read':False,'scientific_admission_granted':False,'fits':0,'GPU_actions':0,
           'sealed_external_payload_access_allowed':False}
    value['plan_sha256']=core.digest(value)
    return value

def validate(root,plan):
    if (plan.get('schema_version')!=SCHEMA or plan.get('record_count')!=520 or len(plan.get('records',[]))!=520
            or core.digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan.get('plan_sha256')
            or plan.get('scientific_admission_granted') is not False or plan.get('raw_cleanup_allowed') is not False):
        raise core.PreparationError('fixed_qualification_plan_required')
    policy={'dataset':'Dreyer2023','role':'source_native_metadata_qualification_only',
            'maximum_owned_raw_bytes':6*1024**3,'minimum_free_disk_bytes':4*1024**3,
            'EEG_array_samples_read':False,'fits':0,'GPU_actions':0,'sealed_external_payload_access_allowed':False}
    if any(type(plan.get(k))is not type(v) or plan.get(k)!=v for k,v in policy.items()):
        raise core.PreparationError('exact_metadata_only_qualification_policy_required')
    if set(plan.get('code',{}))!=set(CODE):raise core.PreparationError('exact_frozen_qualification_code_set_required')
    for name,item in plan['code'].items():
        if item.get('path')!=name:raise core.PreparationError('qualification_code_path_mismatch')
        core.checked(root,item)
    parent=read(core.checked(root,plan['parent_plan']))
    ledger=read(core.checked(root,plan['parent_closed_ledger']))
    claim=read(core.checked(root,plan['parent_claim']))
    wrapper=private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_validate_qualification_parent')
    frozen_parent,_,_=wrapper.fixed_plans(root)
    if parent!=frozen_parent:raise core.PreparationError('exact_frozen_parent_semantic_identity_required')
    charged,wall=wrapper.closed_history(ledger)
    if (parent.get('plan_sha256')!=PARENT or claim.get('active_invocation')is not False
            or parent['records']!=plan['records'] or charged!=plan['parent_actual_charged']
            or wall!=plan['parent_actual_elapsed_seconds']):raise core.PreparationError('parent_closed_identity_changed')
    if (set(plan.get('budgets',{}))!={'provider_body_bytes','r2_get_body_bytes','wall_seconds'}
            or any(type(v)is not int or v<=0 for v in plan['budgets'].values())
            or plan['budgets']['wall_seconds']>10800):raise core.PreparationError('positive_bounded_qualification_budgets_required')
    for key,value in charged.items():
        if plan['budgets'][key]+value>parent['budgets'][key]:raise core.PreparationError('body_budget_must_not_reset')
    if plan['budgets']['wall_seconds']+wall>parent['budgets']['wall_seconds']:
        raise core.PreparationError('wall_budget_must_not_reset')
    by_id={r['record_id']:r for r in parent['records']}
    for record_id,item in plan['reused_complete_parent_states'].items():
        if record_id not in by_id or item.get('path')!=f'audit/source_extensions_20261019/Dreyer2023/{PARENT}/{record_id}/state.json':
            raise core.PreparationError('reused_parent_state_namespace_changed')
        if not core._state_verified(root,read(core.checked(root,item)),parent,by_id[record_id]):
            raise core.PreparationError('prior_complete_proofs_required')
    return parent

def verified_state(root,state,plan,record,parent):
    if state.get('status')!='STRUCTURE_QUALIFICATION_RECORDED' or state.get('plan_sha256')!=plan['plan_sha256']:
        return False
    if state.get('record_id')!=record['record_id']:raise core.PreparationError('qualification_record_identity_changed')
    if state.get('scientific_admission_granted')is not False or state.get('EEG_array_samples_read')is not False:
        raise core.PreparationError('qualification_state_must_remain_metadata_only')
    if state.get('reused_parent_state'):
        if state['reused_parent_state']!=plan['reused_complete_parent_states'].get(record['record_id']):
            raise core.PreparationError('qualification_parent_state_binding_changed')
        prior=read(core.checked(root,state['reused_parent_state']))
        if not core._state_verified(root,prior,parent,record):raise core.PreparationError('prior_complete_proofs_required')
        expected=f'audit/source_extensions_20261019/Dreyer2023/{PARENT}/{record["record_id"]}/native_metadata.json'
        actual=core.descriptor(root,root/expected)
        if state.get('native_structure')!=actual or actual not in prior.get('artifacts',[]):
            raise core.PreparationError('reused_native_metadata_exact_cache_backed_descriptor_required')
        native=read(core.checked(root,actual))
        if state.get('native_marker_counts')!=native.get('native_marker_counts') or state.get('existing_V4_gate')!={'status':'EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS'}:
            raise core.PreparationError('reused_native_metadata_summary_changed')
    else:
        core.verify_raw_receipt(root,state['raw_backup_receipt'],parent,record)
        core.checked(root,state['native_structure'])
        rows=core.receipt_rows(root,state['metadata_backup_receipt'])
        expected=dict(state['native_structure'],dataset='Dreyer2023',role='audit')
        if len(rows)!=1 or any(rows[0].get(k)!=expected[k] for k in storage.ENTRY_FIELDS):
            raise core.PreparationError('qualification_native_metadata_full_GET_required')
        native=read(core.checked(root,state['native_structure']))
        if (native.get('source_file')!={k:record['raw_files'][0][k] for k in ('path','sha256','bytes')}
                or native.get('EEG_array_samples_read')is not False or native.get('scientific_admission_granted')is not False
                or native.get('annotations_modified_or_synthesized')is not False):
            raise core.PreparationError('qualification_native_metadata_source_and_scope_required')
        expected_gate=native.get('existing_V4_gate',{'status':'NATIVE_METADATA_REVIEW_REQUIRED','error_code':native.get('error_code')})
        if state.get('native_marker_counts')!=native.get('native_marker_counts') or state.get('existing_V4_gate')!=expected_gate:
            raise core.PreparationError('qualification_native_metadata_summary_changed')
    return True

def final_backup_verified(root,plan,states,summary_item,receipt_item):
    expected=[dict(core.descriptor(root,root/BASE/'plan.json'),dataset='Dreyer2023',role='audit'),
              dict(summary_item,dataset='Dreyer2023',role='audit'),
              *[dict(r,dataset='Dreyer2023',role='audit') for r in states]]
    rows=core.receipt_rows(root,receipt_item);keys=storage.ENTRY_FIELDS
    if len(rows)!=len(expected) or {tuple(r[k] for k in keys) for r in rows}!={tuple(r[k] for k in keys) for r in expected}:
        raise core.PreparationError('exact_all520_qualification_final_full_GET_required')

def completed_qualification(root,plan,parent,folder):
    path=folder/'qualification_status.json'
    if not path.exists():return None
    result=read(path)
    if result.get('status')!='ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY' or result.get('scientific_admission_granted')is not False:
        raise core.PreparationError('existing_qualification_completion_scope_invalid')
    summary=read(core.checked(root,result['summary']))
    states=[];required_r2=0;required_provider=0
    for record in plan['records']:
        state_path=folder/record['record_id']/'state.json';state=read(state_path)
        if not verified_state(root,state,plan,record,parent):raise core.PreparationError('all520_completed_qualification_states_required')
        states.append(core.descriptor(root,state_path))
        if not state.get('reused_parent_state'):
            required_r2+=sum(r['bytes'] for r in record['raw_files'])+state['native_structure']['bytes']
        if state.get('owned_raw'):required_provider+=state['owned_raw']['bytes']
    if (summary.get('record_states')!=states or summary.get('records')!=520
            or summary.get('plan')!=core.descriptor(root,root/BASE/'plan.json')
            or summary.get('schema_version')!=SCHEMA
            or summary.get('status')!='ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY'
            or summary.get('EEG_array_samples_read')is not False or summary.get('fits')!=0 or summary.get('GPU_actions')!=0
            or summary.get('scientific_admission_granted')is not False):
        raise core.PreparationError('qualification_final_summary_binding_changed')
    final_backup_verified(root,plan,states,result['summary'],result['backup_receipt'])
    ledger=read(folder/'budget_ledger.json')
    wrapper=private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_completed_qualification_budget')
    charged,wall=wrapper.closed_history(ledger)
    if ledger.get('plan_sha256')!=plan['plan_sha256'] or any(charged[k]>plan['budgets'][k] for k in charged) or wall>plan['budgets']['wall_seconds']:
        raise core.PreparationError('qualification_completion_requires_closed_conserved_budget')
    required_r2+=sum(r['bytes'] for r in core.receipt_rows(root,result['backup_receipt']))
    if charged['r2_get_body_bytes']<required_r2 or charged['provider_body_bytes']<required_provider:
        raise core.PreparationError('qualification_completion_charges_below_actual_full_body_proofs')
    return result

def execute(root,plan,*,max_new_records=None):
    parent=validate(root,plan)
    wrapper=private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_qualification_dataset_owner')
    with wrapper.dataset_lock(root):
        return _execute_dataset_owned(root,plan,parent,max_new_records=max_new_records)

def _execute_dataset_owned(root,plan,parent,*,max_new_records=None):
    import requests
    sha=plan['plan_sha256'];folder=root/BASE/sha;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise core.PreparationError('one_qualification_worker_only') from None
    ledger=None
    try:
        complete=completed_qualification(root,plan,parent,folder)
        if complete is not None:return complete
        ledger=core.Ledger(folder/'budget_ledger.json',plan);session=requests.Session();states=[];new=0;owned_bytes=0
        for record in plan['records']:
            audit=folder/record['record_id'];audit.mkdir(exist_ok=True);state_path=audit/'state.json'
            state=read(state_path) if state_path.is_file() else {'plan_sha256':sha,'record_id':record['record_id']}
            if state.get('plan_sha256')!=sha or state.get('record_id')!=record['record_id']:
                raise core.PreparationError('existing_qualification_state_identity_changed')
            if 'owned_raw' in state:
                expected_owner={k:record['raw_files'][0][k] for k in ('path','sha256','bytes')}
                expected_owner['created_by']=ENTRY
                if state['owned_raw']!=expected_owner:raise core.PreparationError('qualification_owned_raw_identity_changed')
            if state.get('owned_raw'):owned_bytes+=record['raw_files'][0]['bytes']
            if verified_state(root,state,plan,record,parent):states.append(core.descriptor(root,state_path));continue
            if max_new_records is not None and new>=max_new_records:break
            ledger.checkpoint()
            prior=plan['reused_complete_parent_states'].get(record['record_id'])
            if prior:
                original=read(core.checked(root,prior))
                if not core._state_verified(root,original,parent,record):raise core.PreparationError('prior_complete_proofs_required')
                native_path=root/f'audit/source_extensions_20261019/Dreyer2023/{PARENT}'/record['record_id']/'native_metadata.json'
                native=read(native_path)
                state.update(reused_parent_state=prior,native_structure=core.descriptor(root,native_path),
                    native_marker_counts=native['native_marker_counts'],existing_V4_gate={'status':'EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS'})
            else:
                item=record['raw_files'][0];raw=core.local(root,item['path'])
                if not raw.exists():
                    if shutil.disk_usage(root).free<item['bytes']+plan['minimum_free_disk_bytes']:
                        raise core.PreparationError('qualification_raw_disk_reserve_required')
                    if not state.get('owned_raw'):
                        if owned_bytes+item['bytes']>plan['maximum_owned_raw_bytes']:raise core.PreparationError('qualification_owned_raw_cap')
                        if raw.with_name(raw.name+'.owned.partial').exists():raise core.PreparationError('unowned_partial_preserved')
                        state['owned_raw']={'path':item['path'],'sha256':item['sha256'],'bytes':item['bytes'],'created_by':ENTRY}
                        core.atomic_json(state_path,state);owned_bytes+=item['bytes']
                    core.download_component(item,raw,session=session,ledger=ledger)
                core.verify_raw_file(raw,item)
                if state.get('raw_backup_receipt'):
                    core.verify_raw_receipt(root,state['raw_backup_receipt'],parent,record)
                if not state.get('raw_backup_receipt'):
                    state['raw_backup_receipt']=core.verified_backup(root,audit/'raw_backup',dataset='Dreyer2023',
                        files=[(raw,'source_development_raw')],prefix=f'paper3/source-native-qualification-20261020/Dreyer2023/{sha}/{record["record_id"]}/raw/',
                        ledger=ledger,backup_fn=storage.run_backup)
                    core.atomic_json(state_path,state)
                native_path=audit/'native_structure.json'
                if not native_path.exists():core.atomic_json(native_path,native_structure(root,raw),immutable=True)
                state['native_structure']=core.descriptor(root,native_path)
                if not state.get('metadata_backup_receipt'):
                    state['metadata_backup_receipt']=core.verified_backup(root,audit/'metadata_backup',dataset='Dreyer2023',
                        files=[(native_path,'audit')],prefix=f'paper3/source-native-qualification-20261020/Dreyer2023/{sha}/{record["record_id"]}/metadata/',
                        ledger=ledger,backup_fn=storage.run_backup)
                native=read(native_path);state.update(native_marker_counts=native.get('native_marker_counts'),
                    existing_V4_gate=native.get('existing_V4_gate',{'status':'NATIVE_METADATA_REVIEW_REQUIRED','error_code':native.get('error_code')}))
            state.update(status='STRUCTURE_QUALIFICATION_RECORDED',scientific_admission_granted=False,EEG_array_samples_read=False)
            core.atomic_json(state_path,state)
            if not verified_state(root,state,plan,record,parent):raise core.PreparationError('qualification_proof_incomplete')
            states.append(core.descriptor(root,state_path));new+=1
            print(json.dumps({'record':record['record_id'],'native_gate':state['existing_V4_gate']['status'],'records':len(states),'total':520}),flush=True)
        if len(states)!=520:return {'status':'PAUSED_STRUCTURE_QUALIFICATION_ONLY','records':len(states),'total':520,'scientific_admission_granted':False}
        summary={'schema_version':SCHEMA,'status':'ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY',
            'plan':core.descriptor(root,root/BASE/'plan.json'),'record_states':states,'records':520,
            'scientific_admission_granted':False,'EEG_array_samples_read':False,'fits':0,'GPU_actions':0}
        summary_path=folder/'all_structure_summary.json';core.atomic_json(summary_path,summary,immutable=True)
        existing_receipt=folder/'final_backup_receipt.json'
        if existing_receipt.is_file():
            receipt=core.descriptor(root,existing_receipt)
            final_backup_verified(root,plan,states,core.descriptor(root,summary_path),receipt)
        else:
            receipt=core.verified_backup(root,folder/'final_backup',dataset='Dreyer2023',
                files=[(root/BASE/'plan.json','audit'),(summary_path,'audit'),*[(core.checked(root,r),'audit') for r in states]],
                prefix=f'paper3/source-native-qualification-20261020/Dreyer2023/{sha}/final/',ledger=ledger,backup_fn=storage.run_backup)
        ledger.checkpoint(close=True)
        result={'status':summary['status'],'summary':core.descriptor(root,summary_path),'backup_receipt':receipt,'scientific_admission_granted':False}
        core.atomic_json(folder/'qualification_status.json',result,immutable=True);return result
    finally:
        if ledger is not None:ledger.checkpoint(close=True)
        lock.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--write-plan',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--max-new-records',type=int)
    a=p.parse_args(argv)
    try:
        root=a.root.resolve(strict=True);path=root/BASE/'plan.json'
        if a.write_plan and a.execute:raise core.PreparationError('freeze_qualification_plan_separately')
        if a.execute and not path.is_file():raise core.PreparationError('frozen_qualification_plan_required_before_execution')
        if a.max_new_records is not None and a.max_new_records<1:raise core.PreparationError('positive_limit_required')
        plan=read(path) if path.exists() else make_plan(root)
        validate(root,plan)
        if a.write_plan:core.atomic_json(path,plan,immutable=True)
        result=execute(root,plan,max_new_records=a.max_new_records) if a.execute else {'status':'DRY_RUN_SOURCE_NATIVE_QUALIFICATION_ONLY',
            'plan_sha256':plan['plan_sha256'],'reused_parent_complete_records':len(plan['reused_complete_parent_states']),
            'budgets':plan['budgets'],'scientific_admission_granted':False,'network_actions':0}
        print(json.dumps(result,indent=2));return 0
    except Exception as exc:
        print(json.dumps({'status':'STOPPED_SOURCE_QUALIFICATION_PRESERVE_ALL','error_type':type(exc).__name__,
            'error_code':str(exc) if isinstance(exc,core.PreparationError) else 'qualification_failed_details_suppressed',
            'scientific_admission_granted':False,'GPU_actions':0,'fits':0}));return 1

if __name__=='__main__':raise SystemExit(main())
