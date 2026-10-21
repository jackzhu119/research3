#!/usr/bin/env python3
"""Bounded same-plan Yang HTTP transport recovery; default is read-only.

The frozen coordinator remains the sole transfer, wall, native and R2 authority.
This wrapper adds explicit no-cache request headers and a permanent, finite
whole-job recovery history. It never changes URLs, CRC/SHA, windows or signals.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import source_extension_coordinator as core

SCHEMA='paper3-yang-sameplan-no-cache-transport-recovery-v1'
PLAN_SHA='ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609'
PLAN_PATH=f'audit/source_extensions_20261019/Yang2025/{PLAN_SHA}/frozen_plan.json'
AMENDMENT_PATH='plans/data_plan_20261019/yang_transport_recovery_amendment.json'
HISTORY='audit/data_plan_20261020/yang_transport_recovery/no_cache_wrapper_v1'
REASON_PATH='audit/data_plan_20261020/yang_transport_recovery/attempt01/resume_reason.json'
OFFICIAL_URL='https://ndownloader.figshare.com/files/51001884'
MAX_RECOVERIES=16
INHERITED_RECOVERIES=1
HEADERS={'Cache-Control':'no-cache','Pragma':'no-cache'}
RETRY_POLICY=['provider_did_not_honor_exact_range','requests_ConnectionError_except_SSL_and_Proxy',
    'requests_ConnectTimeout','requests_ReadTimeout','requests_Timeout','requests_ChunkedEncodingError']
AUTHORIZATION={'authorized_by':'root_coordinator_20261020_under_user_Paper3_CPU_preparation_authorization',
    'scope':'same_official_Yang_source_plan_HTTP_transport_only',
    'provider_URL_CRC_SHA_native_window_and_signal_unchanged':True,
    'max_whole_job_transport_recoveries_including_inherited':MAX_RECOVERIES,
    'no_GPU_training_server_start_or_credential_change':True}


class TransportRecoveryError(ValueError):pass


def read(path):return json.loads(Path(path).read_bytes())


def path(root,relative):return core.local(Path(root),relative)


def ref(root,filename):return core.descriptor(Path(root),Path(filename))


def checked(root,item):return core.checked(Path(root),item)


@contextmanager
def coordinator_lock(root):
    """Read/copy a settled ledger only while the original owner lock is free."""
    filename=path(root,'audit/source_extensions_20261019/.source-extension-singleton.lock')
    filename.parent.mkdir(parents=True,exist_ok=True)
    with filename.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise TransportRecoveryError('original_coordinator_active_no_ledger_snapshot_or_duplicate_dispatch') from None
        yield


def immutable_bytes(filename,body):
    """Byte-exact parent snapshot, never an SHA reference to a mutable ledger."""
    filename=Path(filename)
    if any(p.is_symlink() for p in (filename,*filename.parents)):raise TransportRecoveryError('symlink_forbidden')
    filename.parent.mkdir(parents=True,exist_ok=True)
    if filename.exists():
        if filename.read_bytes()!=body:raise TransportRecoveryError('immutable_history_differs_no_overwrite')
        return
    fd,name=tempfile.mkstemp(prefix='.yang-transport-',dir=filename.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.link(name,filename);core._sync_parent(filename)
    finally:Path(name).unlink(missing_ok=True)


def load_plan(root,plan_path=PLAN_PATH):
    root=Path(root).resolve(strict=True);filename=path(root,plan_path);plan=read(filename)
    if (plan.get('dataset')!='Yang2025' or plan.get('plan_sha256')!=PLAN_SHA
            or plan.get('record_count')!=153):raise TransportRecoveryError('exact_original_Yang_153_record_plan_required')
    core.validate_plan(root,plan)
    urls={raw['download']['url'] for record in plan['records'] for raw in record['raw_files']}
    if urls!={OFFICIAL_URL}:raise TransportRecoveryError('same_official_archive_URL_required')
    return root,plan,ref(root,filename)


def ledger_path(root,plan):
    return path(root,f"audit/source_extensions_20261019/Yang2025/{plan['plan_sha256']}/budget_ledger.json")


def closed_ledger(root,plan,filename=None):
    filename=filename or ledger_path(root,plan);body=Path(filename).read_bytes();value=json.loads(body)
    if (value.get('schema_version')!='paper3-source-extension-budget-ledger-v1'
            or value.get('plan_sha256')!=plan['plan_sha256'] or value.get('wall_intent') is not None
            or value.get('active_transfer') is not None or not isinstance(value.get('history'),list)):
        raise TransportRecoveryError('original_coordinator_ledger_must_be_closed_no_reset')
    elapsed=value.get('elapsed_seconds')
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not 0<=elapsed<=plan['budgets']['wall_seconds']:
        raise TransportRecoveryError('original_cumulative_wall_invalid_or_exhausted')
    charged=value.get('charged')
    if not isinstance(charged,dict) or set(charged)!={'provider_body_bytes','r2_get_body_bytes'}:
        raise TransportRecoveryError('original_transfer_budget_invalid')
    for key,count in charged.items():
        if type(count) is not int or not 0<=count<=plan['budgets'][key]:raise TransportRecoveryError('original_body_budget_invalid_or_exhausted')
    return value,body


def ledger_prefix(previous,current):
    if (previous['plan_sha256']!=current['plan_sha256']
            or current['elapsed_seconds']<previous['elapsed_seconds']
            or any(current['charged'][k]<v for k,v in previous['charged'].items())
            or current['history'][:len(previous['history'])]!=previous['history']):
        raise TransportRecoveryError('original_budget_or_transfer_history_rolled_back')


def code(root):
    filename=path(root,'tools/resume_yang_source_transport.py')
    if filename.read_bytes()!=Path(__file__).read_bytes():raise TransportRecoveryError('executed_transport_wrapper_differs_from_root')
    return ref(root,filename)


def inherited_evidence(root,plan):
    reason_file=path(root,REASON_PATH);reason=read(reason_file)
    if (reason.get('dataset')!='Yang2025' or reason.get('same_plan_sha256')!=plan['plan_sha256']
            or reason.get('bounded_resume_attempt')!=INHERITED_RECOVERIES
            or reason.get('previous_error')!='provider_did_not_honor_exact_range'
            or reason.get('budget_and_clock_not_reset') is not True):
        raise TransportRecoveryError('actual_inherited_sameplan_recovery_evidence_required')
    probe_ref=reason['actual_same_official_archive_range_probe'];probe=read(checked(root,probe_ref))
    if (probe.get('same_source_plan_sha256')!=plan['plan_sha256']
            or probe.get('status')!='EXACT_THIRTY_BYTE_SOURCE_RANGE_VERIFIED'
            or probe.get('body_reservation_kept_in_original_ledger') is not True):
        raise TransportRecoveryError('actual_original_bounded_probe_evidence_required')
    checked(root,probe['parent_ledger'])
    return {'reason':ref(root,reason_file),'probe':probe_ref,'inherited_recoveries':INHERITED_RECOVERIES}


def write_amendment(root,plan_path=PLAN_PATH):
    root,plan,plan_ref=load_plan(root,plan_path);wrapper=code(root)
    filename=path(root,AMENDMENT_PATH)
    if filename.exists():
        _,item=verify_amendment(root,plan,plan_ref)
        return {'status':'AUTHORIZED_SAMEPLAN_TRANSPORT_AMENDMENT','amendment':item,'execute_started':False}
    evidence=inherited_evidence(root,plan)
    with coordinator_lock(root):
        ledger,body=closed_ledger(root,plan)
        snapshot=path(root,HISTORY+'/parent_closed_ledger.json');immutable_bytes(snapshot,body)
    value={'schema_version':SCHEMA,'status':'AUTHORIZED_SAMEPLAN_TRANSPORT_AMENDMENT',
        'authorization':AUTHORIZATION,'frozen_plan':plan_ref,'wrapper_code':wrapper,
        'parent_closed_ledger_snapshot':ref(root,snapshot),'inherited_evidence':evidence,
        'session_headers':HEADERS,'requests_adapter_retries':0,'session_trust_env':'requests_default_preserved',
        'original_kernel_is_only_transfer_and_wall_budget_authority':True,
        'retryable_errors':RETRY_POLICY,
        'non_transport_errors_retry_forbidden':True,'Dreyer_v4_peer_guard_requires_separate_scheduling':True,
        'training_started':False,'GPU_actions':0,'external_payload_accessed':False}
    core.atomic_json(filename,value,immutable=True)
    return {'status':value['status'],'amendment':ref(root,filename),'execute_started':False}


def verify_amendment(root,plan,plan_ref):
    filename=path(root,AMENDMENT_PATH);value=read(filename)
    expected={'schema_version':SCHEMA,'status':'AUTHORIZED_SAMEPLAN_TRANSPORT_AMENDMENT',
        'authorization':AUTHORIZATION,'frozen_plan':plan_ref,'wrapper_code':code(root),'session_headers':HEADERS,
        'requests_adapter_retries':0,'session_trust_env':'requests_default_preserved',
        'inherited_evidence':inherited_evidence(root,plan),
        'parent_closed_ledger_snapshot':ref(root,path(root,HISTORY+'/parent_closed_ledger.json')),
        'original_kernel_is_only_transfer_and_wall_budget_authority':True,
        'retryable_errors':RETRY_POLICY,'Dreyer_v4_peer_guard_requires_separate_scheduling':True,
        'non_transport_errors_retry_forbidden':True,'training_started':False,'GPU_actions':0,'external_payload_accessed':False}
    if value!=expected:raise TransportRecoveryError('reviewed_transport_amendment_identity_changed')
    previous,_=closed_ledger(root,plan,checked(root,value['parent_closed_ledger_snapshot']))
    with coordinator_lock(root):
        current,_=closed_ledger(root,plan);ledger_prefix(previous,current)
    return value,ref(root,filename)


def classify(error):
    import requests
    if isinstance(error,core.PreparationError) and str(error)=='provider_did_not_honor_exact_range':
        return True,'provider_did_not_honor_exact_range'
    candidate=error;prefix=''
    # The frozen process_record deliberately suppresses non-gate error text.
    # Python retains its actual exception object in __context__ even for
    # ``raise ... from None``. Inspect just that one exact wrapper; never
    # infer a transport failure from native/CRC/SHA/budget message text.
    if isinstance(error,core.PreparationError) and str(error)=='operation_failed_details_suppressed':
        candidate=error.__context__;prefix='coordinator_wrapped_'
    if isinstance(candidate,(requests.exceptions.SSLError,requests.exceptions.ProxyError)):
        return False,prefix+'requests_'+type(candidate).__name__
    if isinstance(candidate,(requests.exceptions.Timeout,requests.exceptions.ConnectionError,requests.exceptions.ChunkedEncodingError)):
        return True,prefix+'requests_'+type(candidate).__name__
    return False,('coordinator_'+str(error) if isinstance(error,core.PreparationError) else 'non_transport_'+type(error).__name__)


def session():
    import requests
    from requests.adapters import HTTPAdapter
    result=requests.Session();result.headers.update(HEADERS)
    result.mount('https://',HTTPAdapter(max_retries=0));result.mount('http://',HTTPAdapter(max_retries=0))
    return result


def _history_state(root,plan,amendment_ref):
    folder=path(root,HISTORY);identity={'schema_version':SCHEMA,'amendment':amendment_ref,'plan_sha256':plan['plan_sha256']}
    claim=folder/'claim.json';filename=folder/'recovery_ledger.json'
    if claim.exists() and not filename.is_file():raise TransportRecoveryError('issued_recovery_ledger_missing_no_reset')
    core.atomic_json(claim,identity,immutable=True)
    if filename.exists():state=read(filename)
    else:
        if any(folder.glob('attempt-*')):raise TransportRecoveryError('attempt_history_without_recovery_ledger_no_reset')
        state={'identity':identity,'attempts_used_including_inherited':INHERITED_RECOVERIES,
            'active_attempt':None,'terminal':None,'attempt_receipts':[]}
        core.atomic_json(filename,state)
    if (state.get('identity')!=identity or type(state.get('attempts_used_including_inherited')) is not int
            or not INHERITED_RECOVERIES<=state['attempts_used_including_inherited']<=MAX_RECOVERIES
            or not isinstance(state.get('attempt_receipts'),list)):
        raise TransportRecoveryError('persistent_transport_recovery_identity_or_count_invalid')
    if state.get('active_attempt') is not None:raise TransportRecoveryError('uncertain_recovery_attempt_preserve_evidence_no_automatic_retry')
    expected=state['attempts_used_including_inherited']-INHERITED_RECOVERIES
    directories=sorted(p.name for p in folder.glob('attempt-*'))
    if directories!=[f'attempt-{n:02d}' for n in range(INHERITED_RECOVERIES+1,state['attempts_used_including_inherited']+1)] or len(state['attempt_receipts'])!=expected:
        raise TransportRecoveryError('recovery_attempt_history_or_count_rolled_back')
    previous,_=closed_ledger(root,plan,checked(root,read(path(root,AMENDMENT_PATH))['parent_closed_ledger_snapshot']))
    for sequence,item in enumerate(state['attempt_receipts'],INHERITED_RECOVERIES+1):
        receipt=read(checked(root,item))
        if receipt.get('attempt')!=sequence or receipt.get('amendment')!=amendment_ref:
            raise TransportRecoveryError('immutable_recovery_receipt_identity_changed')
        intent=read(checked(root,receipt['intent']))
        if (intent.get('attempt')!=sequence or intent.get('amendment')!=amendment_ref
                or intent.get('headers')!=HEADERS or intent.get('HTTP_adapter_retries')!=0):
            raise TransportRecoveryError('immutable_recovery_dispatch_intent_changed')
        before,_=closed_ledger(root,plan,checked(root,receipt['before_closed_ledger']))
        after,_=closed_ledger(root,plan,checked(root,receipt['after_closed_ledger']))
        ledger_prefix(previous,before);ledger_prefix(before,after);previous=after
    if state['attempt_receipts']:
        last=read(checked(root,state['attempt_receipts'][-1]));required=None
        if last['error_type'] is not None and not last['retryable_transport_error']:required='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
        elif last['error_type'] is None and last['coordinator_result'].get('status')=='SOURCE_EXTENSION_CACHE_COMPLETE':required='SOURCE_EXTENSION_CACHE_COMPLETE'
        if required is not None and state.get('terminal')!=required:raise TransportRecoveryError('terminal_transport_history_rolled_back_no_retry')
    with coordinator_lock(root):
        current,_=closed_ledger(root,plan);ledger_prefix(previous,current)
    return folder,filename,state


def offline_completed(root,plan):
    """No mutable Ledger, prepare, HTTP/R2 session, or new completion receipt."""
    audit=path(root,f"audit/source_extensions_20261019/Yang2025/{plan['plan_sha256']}")
    status_file=audit/'coordinator_status.json'
    if not status_file.is_file() or read(status_file).get('status')!='SOURCE_EXTENSION_CACHE_COMPLETE':return None
    with coordinator_lock(root):
        status=read(status_file)
        if status.get('status')!='SOURCE_EXTENSION_CACHE_COMPLETE':raise TransportRecoveryError('completion_status_changed_under_owner_lock')
        states=[]
        for record in plan['records']:
            state=read(audit/record['record_id']/'state.json')
            if not core._state_verified(root,state,plan,record):raise TransportRecoveryError('claimed_completion_has_unverified_source_record')
            states.append(state)
        base=core.assemble_manifest(root,plan,states)
        manifest_file=checked(root,status['manifest']);manifest=read(manifest_file)
        if (any(manifest.get(k)!=v for k,v in base.items())
                or set(manifest)!=set(base)|{'native_inventory','unit_evidence','role_assignment'}):
            raise TransportRecoveryError('actual_complete_manifest_differs_from_all_verified_states')
        from p3work.source_extension_bridge import _completed_preparation
        closure={};_completed_preparation(root,manifest,manifest_file,closure)
        closed_ledger(root,plan)
        return {'status':'SOURCE_EXTENSION_CACHE_COMPLETE','already_complete_offline_verified':True,
            'manifest':status['manifest'],'backup_receipt':status['backup_receipt'],
            'verified_records':len(states),'original_closed_ledger':ref(root,ledger_path(root,plan)),
            'coordinator_calls_this_invocation':0,'new_HTTP_or_R2_requests':0,
            'training_started':False,'GPU_actions':0,'external_payload_accessed':False}


def run(root,plan_path=PLAN_PATH,*,execute=False,max_new_records=None):
    if max_new_records is not None and (type(max_new_records) is not int or max_new_records<1):raise TransportRecoveryError('positive_record_limit_required')
    root,plan,plan_ref=load_plan(root,plan_path)
    if not execute:
        return {'schema_version':SCHEMA,'status':'DRY_RUN_SAMEPLAN_TRANSPORT_ONLY_NO_NETWORK',
            'frozen_plan':plan_ref,'amendment_path':AMENDMENT_PATH,'session_headers':HEADERS,
            'max_whole_job_recoveries':MAX_RECOVERIES,'inherited_recoveries':INHERITED_RECOVERIES,
            'original_kernel_budget_not_reset':True,'training_started':False,'GPU_actions':0,'external_payload_accessed':False}
    complete=offline_completed(root,plan)
    if complete is not None:return complete
    amendment,amendment_ref=verify_amendment(root,plan,plan_ref)
    folder=path(root,HISTORY);folder.mkdir(parents=True,exist_ok=True);lock=(folder/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise TransportRecoveryError('sameplan_transport_wrapper_already_running') from None
    try:
        folder,state_file,state=_history_state(root,plan,amendment_ref)
        if state['terminal']=='SOURCE_EXTENSION_CACHE_COMPLETE':
            raise TransportRecoveryError('prior_completed_whole_source_proof_missing_no_refit')
        if state['terminal'] is not None:return {'status':state['terminal'],'recovery_ledger':ref(root,state_file),'new_attempts':0}
        while state['attempts_used_including_inherited']<MAX_RECOVERIES:
            sequence=state['attempts_used_including_inherited']+1;attempt=folder/f'attempt-{sequence:02d}'
            with coordinator_lock(root):
                previous_item=(read(checked(root,state['attempt_receipts'][-1]))['after_closed_ledger']
                    if state['attempt_receipts'] else amendment['parent_closed_ledger_snapshot'])
                previous,_=closed_ledger(root,plan,checked(root,previous_item))
                before,body=closed_ledger(root,plan);ledger_prefix(previous,before)
                immutable_bytes(attempt/'before_closed_ledger.json',body)
            before_ref=ref(root,attempt/'before_closed_ledger.json')
            intent={'schema_version':SCHEMA,'attempt':sequence,'amendment':amendment_ref,'plan':plan_ref,
                'before_closed_ledger':before_ref,'headers':HEADERS,'HTTP_adapter_retries':0,
                'started_clock_for_audit_only':core.clock_snapshot(),'max_new_records':max_new_records}
            core.atomic_json(attempt/'intent.json',intent,immutable=True)
            state.update(attempts_used_including_inherited=sequence,active_attempt=sequence)
            core.atomic_json(state_file,state)
            result=None;error=None;retryable=False;error_code=None;client=None
            try:
                client=session()
                result=core.prepare(root,plan,execute=True,max_new_records=max_new_records,session=client)
            except Exception as caught:error=caught;retryable,error_code=classify(caught)
            finally:
                if client is not None:client.close()
            if error is None and result.get('status')=='SOURCE_EXTENSION_CACHE_COMPLETE':
                try:
                    if offline_completed(root,plan) is None:raise TransportRecoveryError('coordinator_completion_missing_offline_whole_cache_proof')
                except Exception as caught:error=caught;retryable,error_code=False,'offline_complete_proof_'+type(caught).__name__
            # A failed close, native exception or unknown owner never grants a
            # replacement attempt. Only the unchanged coordinator settles time.
            try:
                with coordinator_lock(root):
                    after,body=closed_ledger(root,plan);ledger_prefix(before,after)
                    immutable_bytes(attempt/'after_closed_ledger.json',body)
            except (ValueError,OSError,KeyError,TypeError) as audit_error:
                core.atomic_json(attempt/'unsettled_error.json',{'schema_version':SCHEMA,'attempt':sequence,
                    'core_error_type':type(error).__name__ if error else None,'core_error_code':error_code,
                    'audit_error_type':type(audit_error).__name__,'retry_forbidden_until_explicit_audit':True},immutable=True)
                raise
            receipt={'schema_version':SCHEMA,'attempt':sequence,'amendment':amendment_ref,'intent':ref(root,attempt/'intent.json'),
                'before_closed_ledger':before_ref,'after_closed_ledger':ref(root,attempt/'after_closed_ledger.json'),
                'finished_clock_for_audit_only':core.clock_snapshot(),'error_type':type(error).__name__ if error else None,
                'error_code':error_code,'retryable_transport_error':retryable,'coordinator_result':result,
                'wrapper_did_not_modify_original_budgets_or_clock':True,'training_started':False,'GPU_actions':0}
            receipt_file=attempt/'receipt.json';core.atomic_json(receipt_file,receipt,immutable=True)
            state['attempt_receipts'].append(ref(root,receipt_file));state['active_attempt']=None
            if error is not None and not retryable:state['terminal']='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
            elif error is None and result.get('status')=='SOURCE_EXTENSION_CACHE_COMPLETE':state['terminal']='SOURCE_EXTENSION_CACHE_COMPLETE'
            core.atomic_json(state_file,state)
            if state['terminal'] is not None or error is None:
                return {'status':state['terminal'] or result['status'],'coordinator_result':result,
                    'attempts_used_including_inherited':sequence,'recovery_ledger':ref(root,state_file),
                    'last_attempt_receipt':ref(root,receipt_file),'error_code':error_code,'training_started':False,'GPU_actions':0}
        state['terminal']='TRANSPORT_RECOVERY_LIMIT_EXHAUSTED';core.atomic_json(state_file,state)
        return {'status':state['terminal'],'recovery_ledger':ref(root,state_file),
            'attempts_used_including_inherited':MAX_RECOVERIES,'training_started':False,'GPU_actions':0}
    finally:lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--plan',default=PLAN_PATH)
    action=parser.add_mutually_exclusive_group();action.add_argument('--write-amendment',action='store_true');action.add_argument('--execute',action='store_true')
    parser.add_argument('--max-new-records',type=int);args=parser.parse_args(argv)
    try:
        if args.max_new_records is not None and args.max_new_records<1:raise TransportRecoveryError('positive_record_limit_required')
        result=write_amendment(args.root,args.plan) if args.write_amendment else run(args.root,args.plan,execute=args.execute,max_new_records=args.max_new_records)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        return 2 if result['status'] in {'STOPPED_NON_TRANSPORT_ERROR_NO_RETRY','TRANSPORT_RECOVERY_LIMIT_EXHAUSTED'} else 0
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'BLOCKED_PRESERVE_ORIGINAL_LEDGER','error_type':type(error).__name__,
            'error_code':str(error) if isinstance(error,(TransportRecoveryError,core.PreparationError)) else 'local_dependency_or_evidence_failed',
            'training_started':False,'GPU_actions':0,'external_payload_accessed':False}))
        return 1


if __name__=='__main__':raise SystemExit(main())
