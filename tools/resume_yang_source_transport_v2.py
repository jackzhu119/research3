#!/usr/bin/env python3
"""One explicit same-plan Yang transport call per invocation; default read-only.

This operational amendment never edits the scientific plan, coordinator, URL,
raw identities, budgets or clock. Eight separate execute calls are the maximum;
there is no automatic whole-job retry loop. Unknown calls remain blocked.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import resume_yang_source_transport as previous
from p3work import source_extension_coordinator as core

SCHEMA='paper3-yang-sameplan-no-cache-transport-recovery-v2'
ENTRY='tools/resume_yang_source_transport_v2.py'
PLAN_SHA=previous.PLAN_SHA
PLAN_PATH=previous.PLAN_PATH
AMENDMENT_PATH='plans/data_plan_20261019/yang_transport_recovery_v2_amendment.json'
HISTORY='audit/data_plan_20261020/yang_transport_recovery_v2/no_cache_wrapper_v2'
PROBE_PATH='audit/data_plan_20261020/yang_transport_recovery_v2/diagnostic01/probe.json'
PARENT_ENTRY='tools/resume_yang_source_transport.py'
PARENT_SHA='932ef34b5e9880ef54bf046cfae3e31ffdcb7ad8dcda32d9feb08de52d294f1e'
PARENT_TERMINAL_SHA='9d5998a4cf90f8ada2b7eb227029cee3704edbf40c5010e25c2445784bd4dbf3'
PROBE_SHA='0c11f5d5d3cfe97ddee6bc7f9d3bfa8ecae20eff1b828770798b4c972eef16a6'
MAX_CALLS=8
HEADERS={'Cache-Control':'no-cache','Pragma':'no-cache'}
STATUS='AUTHORIZED_SINGLE_CALL_SAMEPLAN_TRANSPORT_V2'
TERMINALS={None,'SOURCE_EXTENSION_CACHE_COMPLETE','STOPPED_NON_TRANSPORT_ERROR_NO_RETRY','V2_EXPLICIT_CALL_LIMIT_EXHAUSTED'}
AUTHORIZATION={'authorized_by':'root_coordinator_20261020_under_user_Paper3_CPU_preparation_authorization',
 'scope':'eight_additional_separate_explicit_sameplan_Yang_transport_calls_after_actual_thirty_byte_probe',
 'prior_V1_whole_job_recoveries_including_inherited':16,'max_additional_independent_execute_calls':MAX_CALLS,
 'one_coordinator_prepare_call_per_explicit_execute':True,'no_automatic_whole_job_retry_loop':True,
 'original_scientific_plan_URL_CRC_SHA_signals_body_caps_and_clock_unchanged':True,
 'no_GPU_fit_RunPod_external_query_or_credential_change':True}


class TransportRecoveryError(ValueError):pass
read=previous.read
path=previous.path
ref=previous.ref
checked=previous.checked
immutable_bytes=previous.immutable_bytes


def code(root):
 filename=path(root,ENTRY)
 if filename.read_bytes()!=Path(__file__).read_bytes():raise TransportRecoveryError('executed_V2_wrapper_differs_from_root')
 return ref(root,filename)


def load_plan(root,plan_path=PLAN_PATH):
 root,plan,item=previous.load_plan(root,plan_path)
 if ref(root,path(root,PARENT_ENTRY))['sha256']!=PARENT_SHA:raise TransportRecoveryError('frozen_V1_code_changed')
 return root,plan,item


def _closed_value(value,plan):
 if (value.get('schema_version')!='paper3-source-extension-budget-ledger-v1'
     or value.get('plan_sha256')!=plan['plan_sha256'] or value.get('wall_intent') is not None
     or value.get('active_transfer') is not None or not isinstance(value.get('history'),list)):
  raise TransportRecoveryError('original_ledger_must_be_closed_no_reset')
 wall=value.get('elapsed_seconds');counts={'provider_body_bytes':0,'r2_get_body_bytes':0}
 if type(wall) not in (int,float) or not math.isfinite(wall) or not 0<=wall<=plan['budgets']['wall_seconds']:
  raise TransportRecoveryError('original_wall_charge_invalid_or_exhausted')
 for row in value['history']:
  kind=row.get('kind');reserved=row.get('reserved_bytes')
  if kind not in counts or type(reserved) is not int or reserved<0:raise TransportRecoveryError('invalid_original_transfer_history')
  if row.get('status')=='SUCCESS_FULL_BODY_VERIFIED':
   charge=row.get('verified_bytes')
   if type(charge) is not int or not 0<=charge<=reserved:raise TransportRecoveryError('invalid_verified_transfer_charge')
  elif row.get('status') in {'FAILURE_FULL_RESERVATION_CONSUMED','HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED'}:charge=reserved
  else:raise TransportRecoveryError('unclosed_original_transfer_history')
  counts[kind]+=charge
 if value.get('charged')!=counts or any(count>plan['budgets'][key] for key,count in counts.items()):
  raise TransportRecoveryError('original_body_charge_history_or_cap_disagrees')
 return value


def closed_ledger(root,plan,filename=None):
 filename=filename or previous.ledger_path(root,plan);body=Path(filename).read_bytes()
 return _closed_value(json.loads(body),plan),body


def _prefix(before,after):
 previous.ledger_prefix(before,after)


@contextmanager
def owner(root):
 folder=path(root,HISTORY);folder.mkdir(parents=True,exist_ok=True)
 with (folder/'owner.lock').open('a') as handle:
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise TransportRecoveryError('V2_wrapper_already_active_no_duplicate_call') from None
  yield


def _source_snapshot_map(root,inventory,closure):
 if inventory.get('schema_version')!='paper3-Yang-V2-opaque-parent-snapshot-inventory-v1':
  raise TransportRecoveryError('exact_V1_snapshot_inventory_required')
 result={}
 for row in inventory.get('files',[]):
  original=row.get('source_at_snapshot');item=row.get('opaque_snapshot')
  if not isinstance(original,str) or original in result or not isinstance(item,dict) or not item.get('path','').endswith('.txt'):
   raise TransportRecoveryError('unique_opaque_parent_snapshot_required')
  filename=checked(root,item);closure[item['path']]=item['sha256'];result[original]=filename
 return result


def _snapshot_json(snapshots,item):
 original=item['path'];filename=snapshots.get(original)
 if filename is None:raise TransportRecoveryError('parent_snapshot_dependency_missing')
 raw=filename.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=item['sha256'] or ('bytes' in item and len(raw)!=item['bytes']):
  raise TransportRecoveryError('parent_snapshot_dependency_SHA_changed')
 return json.loads(raw)


def _verify_inherited(root,plan,inventory,post_probe_ref,probe_ref,closure):
 snapshots=_source_snapshot_map(root,inventory,closure)
 def original(name):
  if name not in snapshots:raise TransportRecoveryError('required_V1_snapshot_missing')
  return json.loads(snapshots[name].read_bytes())
 parent_code=snapshots.get(PARENT_ENTRY)
 if parent_code is None or hashlib.sha256(parent_code.read_bytes()).hexdigest()!=PARENT_SHA:
  raise TransportRecoveryError('original_V1_code_snapshot_required')
 parent_amendment=original(previous.AMENDMENT_PATH)
 parent_state=original(previous.HISTORY+'/recovery_ledger.json')
 if hashlib.sha256(snapshots[previous.HISTORY+'/recovery_ledger.json'].read_bytes()).hexdigest()!=PARENT_TERMINAL_SHA:
  raise TransportRecoveryError('exact_actual_V1_terminal_ledger_required')
 if (parent_state.get('attempts_used_including_inherited')!=16 or parent_state.get('active_attempt') is not None
     or parent_state.get('terminal')!='TRANSPORT_RECOVERY_LIMIT_EXHAUSTED'
     or len(parent_state.get('attempt_receipts',[]))!=15):raise TransportRecoveryError('actual_V1_sixteen_exhausted_closed_calls_required')
 expected_identity={'schema_version':previous.SCHEMA,'amendment':parent_state.get('identity',{}).get('amendment'),
                    'plan_sha256':PLAN_SHA}
 if (parent_state.get('identity')!=expected_identity or original(previous.HISTORY+'/claim.json')!=expected_identity
     or parent_amendment.get('schema_version')!=previous.SCHEMA
     or parent_amendment.get('authorization')!=previous.AUTHORIZATION
     or parent_amendment.get('wrapper_code',{}).get('sha256')!=PARENT_SHA):
  raise TransportRecoveryError('V1_original_authorization_or_identity_changed')
 _snapshot_json(snapshots,parent_state['identity']['amendment'])
 before=_closed_value(_snapshot_json(snapshots,parent_amendment['parent_closed_ledger_snapshot']),plan)
 for sequence,item in enumerate(parent_state['attempt_receipts'],2):
  receipt=_snapshot_json(snapshots,item);intent=_snapshot_json(snapshots,receipt['intent'])
  if (receipt.get('attempt')!=sequence or receipt.get('amendment')!=expected_identity['amendment']
      or receipt.get('retryable_transport_error') is not True or receipt.get('error_type') is None
      or intent.get('attempt')!=sequence or intent.get('amendment')!=expected_identity['amendment']
      or intent.get('headers')!=HEADERS or intent.get('HTTP_adapter_retries')!=0):
   raise TransportRecoveryError('V1_actual_closed_transport_history_identity_changed')
  start=_closed_value(_snapshot_json(snapshots,receipt['before_closed_ledger']),plan)
  finish=_closed_value(_snapshot_json(snapshots,receipt['after_closed_ledger']),plan)
  _prefix(before,start);_prefix(start,finish);before=finish
 probe_file=checked(root,probe_ref);probe_raw=probe_file.read_bytes();closure[probe_ref['path']]=probe_ref['sha256']
 if hashlib.sha256(probe_raw).hexdigest()!=PROBE_SHA:raise TransportRecoveryError('exact_actual_thirty_byte_probe_required')
 probe=json.loads(probe_raw);record=next((r for r in plan['records'] if r['record_id']==probe.get('record_id')),None)
 downloads=[] if record is None else [r['download'] for r in record['raw_files']]
 offsets=[d for d in downloads if d.get('header_offset')==probe.get('offset') and d.get('url')==previous.OFFICIAL_URL]
 if len(offsets)!=1:raise TransportRecoveryError('probe_must_bind_original_frozen_member_header_offset')
 archive_bytes=offsets[0]['archive_bytes'];offset=probe['offset'];response=probe.get('actual_safe_response',{})
 if (probe.get('same_plan_sha256')!=PLAN_SHA or probe.get('requested_bytes')!=30
     or probe.get('status')!='EXACT_SOURCE_LOCAL_HEADER_THIRTY_BYTES_VERIFIED'
     or probe.get('PK_local_header_magic') is not True or probe.get('original_official_URL_unchanged') is not True
     or probe.get('failed_reservations_retained') is not True or not re.fullmatch('[a-f0-9]{64}',str(probe.get('body_sha256')))
     or response!={'status_code':206,'Content-Length':'30',
        'Content-Range':f'bytes {offset}-{offset+29}/{archive_bytes}','Content-Encoding':'identity'}):
  raise TransportRecoveryError('actual_exact_source_probe_fields_invalid')
 post_file=checked(root,post_probe_ref);post_body=post_file.read_bytes();closure[post_probe_ref['path']]=post_probe_ref['sha256']
 if (hashlib.sha256(post_body).hexdigest()!=probe['ledger_after_probe']['sha256']
     or len(post_body)!=probe['ledger_after_probe']['bytes']):raise TransportRecoveryError('post_probe_snapshot_does_not_match_original_probe')
 post=_closed_value(json.loads(post_body),plan);_prefix(before,post)
 delta=post['history'][len(before['history']):]
 expected_delta={'kind':'provider_body_bytes','reserved_bytes':30,'identity':f'range:{offset}:30',
                 'verified_bytes':30,'status':'SUCCESS_FULL_BODY_VERIFIED'}
 if (delta!=[expected_delta] or post['charged']['provider_body_bytes']!=before['charged']['provider_body_bytes']+30
     or post['charged']['r2_get_body_bytes']!=before['charged']['r2_get_body_bytes']):
  raise TransportRecoveryError('new_probe_thirty_bytes_must_be_fully_charged_without_refund')
 return post,{'record_id':probe['record_id'],'offset':offset,'requested_bytes':30,'body_sha256':probe['body_sha256'],
              'response':response,'original_budget_delta_provider_bytes':30,'inherited_V1_recoveries':16}


def _amendment_value(root,plan_ref,inventory_ref,post_ref,probe_ref,parsed):
 return {'schema_version':SCHEMA,'status':STATUS,'authorization':AUTHORIZATION,'frozen_plan':plan_ref,
  'wrapper_code':code(root),'parent_wrapper_code':{'original_relative_path':PARENT_ENTRY,'frozen_SHA256':PARENT_SHA},
  'historical_snapshot_inventory':inventory_ref,'post_probe_closed_ledger_snapshot':post_ref,
  'diagnostic_probe_snapshot':probe_ref,'probe_parsed_evidence':parsed,
  'session_headers':HEADERS,'requests_adapter_retries':0,'session_trust_env':'requests_default_preserved',
  'original_kernel_is_only_transfer_and_wall_budget_authority':True,'max_independent_execute_calls':MAX_CALLS,
  'no_tight_retry':True,'non_transport_errors_retry_forbidden':True,
  'training_started':False,'GPU_actions':0,'external_payload_accessed':False}


def write_amendment(root,plan_path=PLAN_PATH):
 root,plan,plan_ref=load_plan(root,plan_path);filename=path(root,AMENDMENT_PATH)
 if filename.exists():
  _,item=verify_static_authority(root,plan);return {'status':STATUS,'amendment':item,'execute_started':False}
 with owner(root):
  with previous.coordinator_lock(root):
   v1lock=path(root,previous.HISTORY+'/owner.lock')
   if not v1lock.is_file():raise TransportRecoveryError('original_V1_owner_lock_evidence_missing')
   with v1lock.open('rb') as handle:
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise TransportRecoveryError('V1_worker_still_active_no_snapshot_or_supersession') from None
    current,body=closed_ledger(root,plan)
    probe_source=path(root,PROBE_PATH);probe=read(probe_source)
    if ref(root,probe_source)['sha256']!=PROBE_SHA or ref(root,previous.ledger_path(root,plan))!=probe.get('ledger_after_probe'):
     raise TransportRecoveryError('original_closed_ledger_must_equal_actual_post_probe_before_V2_authorization')
    originals=[path(root,PARENT_ENTRY),path(root,previous.AMENDMENT_PATH)]
    originals.extend(sorted(p for p in path(root,previous.HISTORY).rglob('*') if p.is_file() and p.suffix!='.lock'))
    rows=[]
    for original in originals:
     relative=original.relative_to(root).as_posix();path(root,relative)
     snapshot=path(root,HISTORY+'/inherited_opaque/'+relative+'.txt')
     immutable_bytes(snapshot,original.read_bytes());rows.append({'source_at_snapshot':relative,'opaque_snapshot':ref(root,snapshot)})
    probe_file=path(root,HISTORY+'/diagnostic_probe_original.json.txt');immutable_bytes(probe_file,probe_source.read_bytes())
    post_file=path(root,HISTORY+'/post_probe_original_closed_ledger.json');immutable_bytes(post_file,body)
    inventory={'schema_version':'paper3-Yang-V2-opaque-parent-snapshot-inventory-v1','files':rows,
               'V1_terminal_and_all_parent_original_bytes_preserved':True,'old_JSON_never_used_as_live_budget_descriptors':True}
    inventory_file=path(root,HISTORY+'/historical_snapshot_inventory.json');core.atomic_json(inventory_file,inventory,immutable=True)
    closure={};_,parsed=_verify_inherited(root,plan,inventory,ref(root,post_file),ref(root,probe_file),closure)
    value=_amendment_value(root,plan_ref,ref(root,inventory_file),ref(root,post_file),ref(root,probe_file),parsed)
    core.atomic_json(filename,value,immutable=True)
 return {'status':STATUS,'amendment':ref(root,filename),'execute_started':False,'scientific_ledger_modified':False}


def verify_static_authority(root,plan):
 """Pure reads/hashes only: no live scientific ledger, locks or IO clients."""
 root=Path(root).resolve(strict=True);_,fixed,plan_ref=load_plan(root)
 if plan!=fixed:raise TransportRecoveryError('only_exact_original_scientific_plan_allowed')
 filename=path(root,AMENDMENT_PATH);value=read(filename)
 closure={};inventory=read(checked(root,value['historical_snapshot_inventory']))
 _,parsed=_verify_inherited(root,plan,inventory,value['post_probe_closed_ledger_snapshot'],value['diagnostic_probe_snapshot'],closure)
 expected=_amendment_value(root,plan_ref,value['historical_snapshot_inventory'],value['post_probe_closed_ledger_snapshot'],value['diagnostic_probe_snapshot'],parsed)
 if value!=expected:raise TransportRecoveryError('V2_operational_amendment_or_implementation_changed')
 return value,ref(root,filename)


def _history(root,plan,amendment_ref,*,initialize=False):
 folder=path(root,HISTORY);claim=folder/'claim.json';filename=folder/'recovery_ledger.json'
 identity={'schema_version':SCHEMA,'amendment':amendment_ref,'plan_sha256':PLAN_SHA}
 if claim.exists()!=filename.exists():raise TransportRecoveryError('issued_V2_call_history_missing_no_reset')
 if not filename.exists():
  if any(folder.glob('call-*')):raise TransportRecoveryError('V2_call_directories_without_counter_no_reset')
  if not initialize:return None
  core.atomic_json(claim,identity,immutable=True)
  core.atomic_json(filename,{'identity':identity,'calls_used':0,'active_call':None,'terminal':None,'call_receipts':[]})
 state=read(filename)
 if read(claim)!=identity or state.get('identity')!=identity or type(state.get('calls_used')) is not int or not 0<=state['calls_used']<=MAX_CALLS:
  raise TransportRecoveryError('V2_persistent_call_identity_or_counter_changed')
 if state.get('active_call') is not None:raise TransportRecoveryError('unknown_active_V2_call_explicit_audit_required_no_retry')
 if state.get('terminal') not in TERMINALS or not isinstance(state.get('call_receipts'),list):raise TransportRecoveryError('V2_call_terminal_or_receipts_invalid')
 n=state['calls_used']
 if len(state['call_receipts'])!=n or sorted(p.name for p in folder.glob('call-*'))!=[f'call-{i:02d}' for i in range(1,n+1)]:
  raise TransportRecoveryError('V2_call_count_or_history_rolled_back')
 amendment=read(path(root,AMENDMENT_PATH));last,_=closed_ledger(root,plan,checked(root,amendment['post_probe_closed_ledger_snapshot']))
 for sequence,item in enumerate(state['call_receipts'],1):
  receipt=read(checked(root,item));intent=read(checked(root,receipt['intent']))
  if (receipt.get('schema_version')!=SCHEMA or intent.get('schema_version')!=SCHEMA
      or receipt.get('call_number')!=sequence or receipt.get('amendment')!=amendment_ref
      or receipt.get('whole_job_prepare_calls')!=1 or intent.get('call_number')!=sequence
      or intent.get('amendment')!=amendment_ref or intent.get('plan')!=amendment['frozen_plan']
      or intent.get('before_closed_ledger')!=receipt.get('before_closed_ledger') or intent.get('session_headers')!=HEADERS
      or intent.get('requests_adapter_retries')!=0 or intent.get('one_prepare_call_only') is not True):
   raise TransportRecoveryError('V2_immutable_call_receipt_or_intent_changed')
  before,_=closed_ledger(root,plan,checked(root,receipt['before_closed_ledger']))
  after,_=closed_ledger(root,plan,checked(root,receipt['after_closed_ledger']));_prefix(last,before);_prefix(before,after);last=after
  required=None
  if receipt.get('error_type') is not None and receipt.get('retryable_transport_error') is not True:required='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
  elif receipt.get('error_type') is None and (receipt.get('coordinator_result') or {}).get('status')=='SOURCE_EXTENSION_CACHE_COMPLETE':required='SOURCE_EXTENSION_CACHE_COMPLETE'
  elif sequence==MAX_CALLS:required='V2_EXPLICIT_CALL_LIMIT_EXHAUSTED'
  if required is not None and sequence!=n:raise TransportRecoveryError('V2_terminal_history_rolled_back_or_replayed')
  if sequence==n and state.get('terminal')!=required:raise TransportRecoveryError('V2_terminal_history_disagrees_with_final_immutable_receipt')
 if n==0 and state.get('terminal') is not None:raise TransportRecoveryError('V2_empty_history_cannot_be_terminal')
 return folder,filename,state,last


def _safe_classify(error):
 retryable,code_value=previous.classify(error)
 if not retryable and (not isinstance(code_value,str) or not re.fullmatch('[a-z0-9_]{1,180}',code_value)):
  code_value='non_transport_'+type(error).__name__
 return retryable,code_value


def verify_complete_transport_history(root,plan):
 """Pure offline153/5GET+static ancestry+closed calls; never acquire a lock."""
 root=Path(root).resolve(strict=True)
 audit=path(root,f'audit/source_extensions_20261019/Yang2025/{PLAN_SHA}')
 status_file=audit/'coordinator_status.json'
 if not status_file.is_file() or read(status_file).get('status')!='SOURCE_EXTENSION_CACHE_COMPLETE':return None
 closure={};history=None
 if path(root,HISTORY+'/claim.json').exists() or path(root,HISTORY+'/recovery_ledger.json').exists():
  amendment,amendment_ref=verify_static_authority(root,plan)
  result=_history(root,plan,amendment_ref)
  if result is None:raise TransportRecoveryError('V2_claimed_completion_missing_call_history')
  folder,filename,state,last=result
  final_receipt=read(checked(root,state['call_receipts'][-1])) if state['calls_used'] else {}
  if (state.get('terminal')!='SOURCE_EXTENSION_CACHE_COMPLETE' or state['calls_used']<1
      or final_receipt.get('error_type') is not None
      or (final_receipt.get('coordinator_result') or {}).get('status')!='SOURCE_EXTENSION_CACHE_COMPLETE'):
   raise TransportRecoveryError('V2_complete_transport_requires_actual_successful_final_call_receipt')
  inventory_ref=amendment['historical_snapshot_inventory'];closure[inventory_ref['path']]=inventory_ref['sha256']
  _verify_inherited(root,plan,read(checked(root,inventory_ref)),amendment['post_probe_closed_ledger_snapshot'],amendment['diagnostic_probe_snapshot'],closure)
  for p in [path(root,AMENDMENT_PATH),path(root,ENTRY),folder/'claim.json',filename]:closure[p.relative_to(root).as_posix()]=ref(root,p)['sha256']
  for item in state['call_receipts']:
   closure[item['path']]=item['sha256'];receipt=read(checked(root,item))
   for key in ['intent','before_closed_ledger','after_closed_ledger']:closure[receipt[key]['path']]=receipt[key]['sha256']
  current,_=closed_ledger(root,plan);_prefix(last,current)
  history={'amendment':amendment_ref,'recovery_ledger':ref(root,filename),'calls_used':state['calls_used'],
           'provenance_closure':closure}
 status=read(status_file);states=[]
 if status.get('record_count')!=153:raise TransportRecoveryError('complete_status_requires_all153_records')
 for record in plan['records']:
  state=read(audit/record['record_id']/'state.json')
  if not core._state_verified(root,state,plan,record):raise TransportRecoveryError('claimed_complete_source_has_unverified_record')
  states.append(state)
 base=core.assemble_manifest(root,plan,states);manifest_file=checked(root,status['manifest']);manifest=read(manifest_file)
 if (any(manifest.get(k)!=v for k,v in base.items()) or set(manifest)!=set(base)|{'native_inventory','unit_evidence','role_assignment'}):
  raise TransportRecoveryError('complete_manifest_not_exact_verified153_record_assembly')
 from p3work.source_extension_bridge import _completed_preparation
 final_closure={};_completed_preparation(root,manifest,manifest_file,final_closure)
 for filename,sha in final_closure.items():
  if filename in closure and closure[filename]!=sha:raise TransportRecoveryError('completion_provenance_SHA_conflict')
  closure[filename]=sha
 closed_ledger(root,plan)
 return {'status':'SOURCE_EXTENSION_CACHE_COMPLETE','already_complete_offline_verified':True,'verified_records':153,
  'manifest':status['manifest'],'backup_receipt':status['backup_receipt'],
  'original_closed_ledger':ref(root,previous.ledger_path(root,plan)),
  'transport_history':history,'provenance_closure':closure,'coordinator_calls_this_invocation':0,'new_HTTP_or_R2_requests':0,
  'training_started':False,'GPU_actions':0,'external_payload_accessed':False}


def run(root,plan_path=PLAN_PATH,*,execute=False,max_new_records=None):
 if max_new_records is not None and (type(max_new_records) is not int or max_new_records<1):raise TransportRecoveryError('positive_record_limit_required')
 root,plan,plan_ref=load_plan(root,plan_path)
 if not execute:return {'schema_version':SCHEMA,'status':'DRY_RUN_SINGLE_EXPLICIT_CALL_NO_NETWORK','frozen_plan':plan_ref,
  'amendment_path':AMENDMENT_PATH,'session_headers':HEADERS,'max_independent_execute_calls':MAX_CALLS,
  'one_prepare_call_only':True,'original_scientific_budget_or_clock_changed':False,'training_started':False,'GPU_actions':0}
 complete=verify_complete_transport_history(root,plan)
 if complete is not None:return complete
 with owner(root):
  amendment,amendment_ref=verify_static_authority(root,plan)
  folder,filename,state,last=_history(root,plan,amendment_ref,initialize=True)
  if state['terminal'] is not None:return {'status':state['terminal'],'calls_used':state['calls_used'],
    'recovery_ledger':ref(root,filename),'coordinator_calls_this_invocation':0,'training_started':False,'GPU_actions':0}
  if state['calls_used']>=MAX_CALLS:raise TransportRecoveryError('V2_call_limit_exhausted_no_new_dispatch')
  sequence=state['calls_used']+1;attempt=folder/f'call-{sequence:02d}'
  with previous.coordinator_lock(root):
   before,body=closed_ledger(root,plan);_prefix(last,before)
   immutable_bytes(attempt/'before_closed_ledger.json',body)
  before_ref=ref(root,attempt/'before_closed_ledger.json')
  intent={'schema_version':SCHEMA,'call_number':sequence,'amendment':amendment_ref,'plan':plan_ref,
   'before_closed_ledger':before_ref,'session_headers':HEADERS,'requests_adapter_retries':0,
   'one_prepare_call_only':True,'max_new_records':max_new_records,'started_clock_for_audit_only':core.clock_snapshot()}
  core.atomic_json(attempt/'intent.json',intent,immutable=True)
  state.update(calls_used=sequence,active_call=sequence);core.atomic_json(filename,state)
  result=None;error=None;retryable=False;error_code=None;client=None
  try:
   client=previous.session()
   result=core.prepare(root,plan,execute=True,max_new_records=max_new_records,session=client)
  except Exception as caught:error=caught;retryable,error_code=_safe_classify(caught)
  finally:
   if client is not None:client.close()
  # Core owns settlement. An unknown close leaves active_call set and forbids
  # another call; no exception here can refund a byte or reset elapsed time.
  with previous.coordinator_lock(root):
   after,body=closed_ledger(root,plan);_prefix(before,after);immutable_bytes(attempt/'after_closed_ledger.json',body)
  receipt={'schema_version':SCHEMA,'call_number':sequence,'amendment':amendment_ref,'intent':ref(root,attempt/'intent.json'),
   'before_closed_ledger':before_ref,'after_closed_ledger':ref(root,attempt/'after_closed_ledger.json'),
   'finished_clock_for_audit_only':core.clock_snapshot(),'error_type':type(error).__name__ if error else None,
   'error_code':error_code,'retryable_transport_error':retryable,'coordinator_result':result,
   'whole_job_prepare_calls':1,'wrapper_modified_original_budget_or_clock':False,'GPU_actions':0,'training_started':False}
  receipt_file=attempt/'receipt.json';core.atomic_json(receipt_file,receipt,immutable=True)
  state['call_receipts'].append(ref(root,receipt_file));state['active_call']=None
  if error is not None and not retryable:state['terminal']='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
  elif error is None and (result or {}).get('status')=='SOURCE_EXTENSION_CACHE_COMPLETE':state['terminal']='SOURCE_EXTENSION_CACHE_COMPLETE'
  elif sequence==MAX_CALLS:state['terminal']='V2_EXPLICIT_CALL_LIMIT_EXHAUSTED'
  core.atomic_json(filename,state)
  if state['terminal']=='SOURCE_EXTENSION_CACHE_COMPLETE':
   try:proof=verify_complete_transport_history(root,plan)
   except Exception:
    raise TransportRecoveryError('claimed_complete_requires_actual_offline153_and_five_GET_proof_no_new_call') from None
   if proof is None:raise TransportRecoveryError('claimed_complete_missing_offline_proof_no_new_call')
  return {'status':state['terminal'] or ((result or {}).get('status') if error is None else 'PAUSED_TRANSPORT_ERROR_REQUIRES_NEW_EXPLICIT_EXECUTE'),
   'coordinator_result':result,'calls_used':sequence,'recovery_ledger':ref(root,filename),
   'last_call_receipt':ref(root,receipt_file),'coordinator_calls_this_invocation':1,'error_code':error_code,
   'automatic_retry_performed':False,'training_started':False,'GPU_actions':0,'external_payload_accessed':False}


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--plan',default=PLAN_PATH)
 action=p.add_mutually_exclusive_group();action.add_argument('--write-amendment',action='store_true');action.add_argument('--execute',action='store_true')
 p.add_argument('--max-new-records',type=int);a=p.parse_args(argv)
 try:
  result=write_amendment(a.root,a.plan) if a.write_amendment else run(a.root,a.plan,execute=a.execute,max_new_records=a.max_new_records)
  print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
  return 2 if result['status'] in {'STOPPED_NON_TRANSPORT_ERROR_NO_RETRY','V2_EXPLICIT_CALL_LIMIT_EXHAUSTED','PAUSED_TRANSPORT_ERROR_REQUIRES_NEW_EXPLICIT_EXECUTE'} else 0
 except Exception as error:
  print(json.dumps({'status':'BLOCKED_PRESERVE_ALL_ORIGINAL_BUDGET_AND_HISTORY','error_type':type(error).__name__,
   'error_code':str(error) if isinstance(error,(TransportRecoveryError,core.PreparationError,previous.TransportRecoveryError)) else 'local_dependency_or_evidence_failed',
   'training_started':False,'GPU_actions':0,'external_payload_accessed':False}));return 1


if __name__=='__main__':raise SystemExit(main())
