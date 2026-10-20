#!/usr/bin/env python3
"""One explicitly authorized no-cache continuation of frozen Dreyer qualification.

No automatic whole-job retry. The old qualifier alone owns transfer and wall
budgets, source metadata, R2 proofs and its dataset lock. This dedicated CLI
process temporarily injects a Session subclass and restores/closes it on exit.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import source_extension_coordinator as core

SCHEMA='paper3-dreyer-fixed-qualification-no-cache-continuation-v1'
PLAN_SHA='f39a24a3d26c6eb3de6e286e2efb8afc032397b91626f84fd647ae2c182b5efd'
ENTRY='tools/qualify_dreyer_native_structures.py'
QUALIFIER_SHA='4efb897c380ab6e2d26a95a9d11c5f3a44fc2fd32155024afa38ebb301a9d01d'
NATIVE='p3work/source_extensions/dreyer2023.py'
NATIVE_SHA='ec2f7dbc10903c308cd30564c30b04c4653c64173fdb0331ede318fd1c3b1489'
BASE='audit/source_extensions_20261019/Dreyer2023/native_structure_qualification'
PLAN_PATH=BASE+'/plan.json'
HISTORY='audit/data_plan_20261020/dreyer_qualification_network_recovery/no_cache_wrapper_v1'
AMENDMENT='plans/data_plan_20261019/dreyer_qualification_transport_amendment.json'
HEADERS={'Cache-Control':'no-cache','Pragma':'no-cache'}
PRIOR_CALLS=('pilot','full','resume01','resume02')
MAX_NEW_CALLS=4
COMPLETE='ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY'
AUTHORIZATION={'authorized_by':'root_20261020_under_user_Paper3_CPU_preparation_authorization',
    'scope':'fixed_Dreyer_f39a_source_native_metadata_qualification_transport_only',
    'prior_actual_whole_job_calls':list(PRIOR_CALLS),'maximum_new_whole_job_calls':MAX_NEW_CALLS,
    'one_whole_job_call_per_explicit_execute':True,'automatic_retry_loop':False,
    'original_budget_clock_raw_SHA_native_parser_and_scientific_gates_unchanged':True,
    'scientific_admission_granted':False,'GPU_actions':0,'fits':0,'server_actions':False}

class RecoveryError(ValueError):pass

def path(root,name):return core.local(Path(root),name)
def read(filename):return json.loads(Path(filename).read_bytes())
def ref(root,filename):return core.descriptor(Path(root),Path(filename))
def checked(root,item):return core.checked(Path(root),item)

def immutable_bytes(filename,body):
    filename=Path(filename)
    if any(p.is_symlink() for p in (filename,*filename.parents)):raise RecoveryError('symlink_forbidden')
    filename.parent.mkdir(parents=True,exist_ok=True)
    if filename.exists():
        if filename.read_bytes()!=body:raise RecoveryError('immutable_transport_evidence_differs')
        return
    fd,name=tempfile.mkstemp(prefix='.dreyer-qualification-',dir=filename.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.link(name,filename);core._sync_parent(filename)
    finally:Path(name).unlink(missing_ok=True)

def load(root):
    root=Path(root).resolve(strict=True)
    if ref(root,path(root,ENTRY))['sha256']!=QUALIFIER_SHA:raise RecoveryError('exact_frozen_qualifier_4ef_required')
    if ref(root,path(root,NATIVE))['sha256']!=NATIVE_SHA:raise RecoveryError('canonical_ec2f_must_remain_unchanged_during_qualification')
    plan=read(path(root,PLAN_PATH))
    if plan.get('plan_sha256')!=PLAN_SHA:raise RecoveryError('exact_frozen_f39a_qualification_plan_required')
    spec=importlib.util.spec_from_file_location('_p3_fixed_dreyer_qualification_transport',path(root,ENTRY))
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    parent=q.validate(root,plan)
    return root,q,plan,parent

@contextmanager
def dataset(q,root):
    wrapper=q.private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_qualification_transport_dataset_lock')
    with wrapper.dataset_lock(root):yield

def ledger_file(root):return path(root,BASE+'/'+PLAN_SHA+'/budget_ledger.json')

def closed(root,q,plan,filename=None):
    filename=filename or ledger_file(root);body=Path(filename).read_bytes();value=json.loads(body)
    if (value.get('plan_sha256')!=PLAN_SHA or 'wall_intent' in value or 'active_transfer' in value):
        raise RecoveryError('original_qualification_ledger_must_be_closed_no_reset')
    wrapper=q.private_module(root,'tools/prepare_dreyer_source_cpu_v4.py','_p3_qualification_transport_closed_history')
    charged,wall=wrapper.closed_history(value)
    if (set(charged)!={'provider_body_bytes','r2_get_body_bytes'}
            or any(type(v)is not int or not 0<=v<=plan['budgets'][k] for k,v in charged.items())
            or type(wall) not in (int,float) or not math.isfinite(wall) or not 0<=wall<=plan['budgets']['wall_seconds']):
        raise RecoveryError('original_qualification_budget_invalid_or_exhausted')
    return value,body

def prefix(before,after):
    if (before['plan_sha256']!=after['plan_sha256'] or after['elapsed_seconds']<before['elapsed_seconds']
            or any(after['charged'][k]<v for k,v in before['charged'].items())
            or after['history'][:len(before['history'])]!=before['history']):
        raise RecoveryError('qualification_budget_history_rolled_back')

def completed(root,q,plan,parent):
    with dataset(q,root):
        result=q.completed_qualification(root,plan,parent,root/BASE/PLAN_SHA)
        if result is None:return None
        closed(root,q,plan)
        return {'status':COMPLETE,'already_completed_offline_verified':True,'original_result':result,
            'new_whole_job_calls':0,'new_HTTP_or_R2_requests':0,'scientific_admission_granted':False,'fits':0,'GPU_actions':0}

def wrapper_code(root):
    filename=path(root,'tools/resume_dreyer_qualification_transport.py')
    if filename.read_bytes()!=Path(__file__).read_bytes():raise RecoveryError('executed_wrapper_differs_from_root')
    return ref(root,filename)

def prior_evidence(root):
    values=[]
    for ordinal in (1,2):
        filename=path(root,f'audit/data_plan_20261020/dreyer_qualification_network_recovery/attempt{ordinal:02d}/HEAD_probe_and_resume_reason.json')
        value=read(filename)
        if value.get('same_qualification_plan_sha256',value.get('same_plan_sha256'))!=PLAN_SHA:
            raise RecoveryError('prior_qualification_resume_plan_mismatch')
        if ordinal==1:
            if value.get('closed_ledger')is not True or value.get('no_budget_or_wall_reset')is not True:
                raise RecoveryError('prior_resume01_must_preserve_closed_budget')
            checked(root,value['prior_ledger'])
        else:
            if (value.get('bounded_resume_attempt')!=2 or value.get('prior_failed_reservation_retained')is not True
                    or value.get('budget_or_clock_reset')is not False):raise RecoveryError('prior_resume02_must_preserve_closed_budget')
            checked(root,value['closed_ledger_snapshot'])
        values.append(ref(root,filename))
    return {'prior_actual_call_names':list(PRIOR_CALLS),'prior_actual_call_count':4,
        'pilot_and_full_basis':'root_observed_actual_calls_in_task_authorization','resume01_and_resume02_evidence':values}

def amendment_value(root,plan):
    return {'schema_version':SCHEMA,'authorization':AUTHORIZATION,'frozen_plan':ref(root,path(root,PLAN_PATH)),
        'qualifier_code':ref(root,path(root,ENTRY)),'native_code':ref(root,path(root,NATIVE)),
        'wrapper_code':wrapper_code(root),'prior_calls':prior_evidence(root),
        'parent_closed_ledger_snapshot':ref(root,path(root,HISTORY+'/parent_closed_ledger.json')),
        'session_headers':HEADERS,'requests_adapter_retries':0,'trust_env':'requests_default_preserved',
        'original_qualifier_is_only_transfer_and_wall_budget_authority':True}

def write_amendment(root):
    root,q,plan,parent=load(root);done=completed(root,q,plan,parent)
    if done is not None:return done
    filename=path(root,AMENDMENT)
    if filename.exists():verify_amendment(root,q,plan);return {'status':'QUALIFICATION_TRANSPORT_AMENDMENT_PRESENT','amendment':ref(root,filename)}
    with dataset(q,root):
        _,body=closed(root,q,plan);immutable_bytes(path(root,HISTORY+'/parent_closed_ledger.json'),body)
    core.atomic_json(filename,amendment_value(root,plan),immutable=True)
    return {'status':'QUALIFICATION_TRANSPORT_AMENDMENT_WRITTEN','amendment':ref(root,filename),'execute_started':False}

def verify_amendment(root,q,plan):
    filename=path(root,AMENDMENT);value=read(filename)
    if value!=amendment_value(root,plan):raise RecoveryError('reviewed_qualification_transport_amendment_changed')
    with dataset(q,root):
        old,_=closed(root,q,plan,checked(root,value['parent_closed_ledger_snapshot']))
        now,_=closed(root,q,plan);prefix(old,now)
    return value,ref(root,filename)

def continuation_allowed(error):
    import requests
    if isinstance(error,core.PreparationError) and str(error)=='direct_get_status_or_declared_size_mismatch':return True
    if isinstance(error,(requests.exceptions.SSLError,requests.exceptions.ProxyError)):return False
    return isinstance(error,(requests.exceptions.Timeout,requests.exceptions.ConnectionError,requests.exceptions.ChunkedEncodingError))

def safe_error(error):
    if isinstance(error,core.PreparationError):
        code=str(error)
        if code.isidentifier() and len(code)<=120:return code
    return 'details_suppressed_'+type(error).__name__

def dedicated_process():
    if threading.active_count()!=1:raise RecoveryError('dedicated_non_threaded_qualification_process_required')

@contextmanager
def no_cache_session_factory(statistics_file):
    """Scope is one dedicated, non-threaded qualification CLI process."""
    import requests
    from requests.adapters import HTTPAdapter
    dedicated_process()
    original=requests.Session;instances=[]
    stats={'schema_version':SCHEMA,'GET_calls':0,'HTTP_status_counts':{},'declared_content_length_counts':{},
        'content_encoding_counts':{},'exception_type_counts':{},'URL_query_body_or_full_headers_recorded':False}
    def save():core.atomic_json(statistics_file,stats)
    def count(field,key):stats[field][str(key)]=stats[field].get(str(key),0)+1
    class NoCacheSession(original):
        def __init__(self):
            super().__init__();self.headers.update(HEADERS)
            self.mount('https://',HTTPAdapter(max_retries=0));self.mount('http://',HTTPAdapter(max_retries=0));instances.append(self)
        def get(self,*args,**kwargs):
            stats['GET_calls']+=1;save()
            try:response=super().get(*args,**kwargs)
            except Exception as error:count('exception_type_counts',type(error).__name__);save();raise
            status=response.status_code
            count('HTTP_status_counts',status if type(status)is int and 100<=status<=599 else 'invalid')
            length=response.headers.get('Content-Length')
            length=(int(length) if isinstance(length,str) and length.isascii() and length.isdecimal() and len(length)<24 else 'missing_or_invalid')
            encoding=response.headers.get('Content-Encoding','identity')
            encoding=encoding.lower() if isinstance(encoding,str) else 'other'
            count('declared_content_length_counts',length)
            count('content_encoding_counts',encoding if encoding in {'identity','gzip','br','deflate','zstd','compress'} else 'other')
            save();return response
    save();requests.Session=NoCacheSession
    try:yield stats
    finally:
        requests.Session=original
        close_error=False
        for instance in instances:
            try:instance.close()
            except Exception:close_error=True
        save()
        if close_error:raise RecoveryError('injected_Session_cleanup_failed_no_replay')

def state(root,q,plan,amendment_ref):
    folder=path(root,HISTORY);claim={'schema_version':SCHEMA,'amendment':amendment_ref,'plan_sha256':PLAN_SHA}
    filename=folder/'continuation_ledger.json'
    if (folder/'claim.json').exists() and not filename.is_file():raise RecoveryError('issued_continuation_ledger_missing_no_reset')
    core.atomic_json(folder/'claim.json',claim,immutable=True)
    if filename.exists():value=read(filename)
    else:
        if any(folder.glob('call-*')):raise RecoveryError('immutable_call_history_without_counter_no_reset')
        value={'claim':claim,'prior_actual_calls':4,'new_calls':0,'active_call':None,'terminal':None,'receipts':[]};core.atomic_json(filename,value)
    if (value.get('claim')!=claim or value.get('prior_actual_calls')!=4 or type(value.get('new_calls'))is not int
            or not 0<=value['new_calls']<=MAX_NEW_CALLS or not isinstance(value.get('receipts'),list)):
        raise RecoveryError('qualification_continuation_identity_or_count_invalid')
    if value.get('active_call')is not None:raise RecoveryError('unknown_prior_whole_call_no_automatic_replay')
    if (sorted(p.name for p in folder.glob('call-*'))!=[f'call-{i:02d}' for i in range(1,value['new_calls']+1)]
            or len(value['receipts'])!=value['new_calls']):raise RecoveryError('immutable_continuation_history_or_count_rolled_back')
    previous,_=closed(root,q,plan,checked(root,read(path(root,AMENDMENT))['parent_closed_ledger_snapshot']))
    for number,item in enumerate(value['receipts'],1):
        receipt=read(checked(root,item));intent=read(checked(root,receipt['intent']))
        if receipt.get('new_call')!=number or receipt.get('amendment')!=amendment_ref or intent.get('actual_whole_call_ordinal')!=4+number:
            raise RecoveryError('continuation_receipt_or_dispatch_identity_changed')
        before,_=closed(root,q,plan,checked(root,receipt['before_closed_ledger']))
        after,_=closed(root,q,plan,checked(root,receipt['after_closed_ledger']))
        checked(root,receipt['closed_HTTP_statistics']);prefix(previous,before);prefix(before,after);previous=after
        if receipt['error_type']is not None and not receipt['transport_continuation_eligible'] and value.get('terminal')!='STOPPED_NON_TRANSPORT_NO_RETRY':
            raise RecoveryError('terminal_non_transport_history_rolled_back')
    with dataset(q,root):now,_=closed(root,q,plan);prefix(previous,now)
    return folder,filename,value

def run(root,*,execute=False):
    root,q,plan,parent=load(root)
    if not execute:return {'status':'DRY_RUN_FIXED_QUALIFICATION_TRANSPORT_ONLY','plan_sha256':PLAN_SHA,
        'prior_actual_whole_calls':list(PRIOR_CALLS),'max_new_whole_calls':MAX_NEW_CALLS,
        'whole_calls_this_invocation':0,'network_actions':0,'scientific_admission_granted':False,'fits':0,'GPU_actions':0}
    done=completed(root,q,plan,parent)
    if done is not None:return done
    dedicated_process()
    amendment,amendment_ref=verify_amendment(root,q,plan)
    folder=path(root,HISTORY);folder.mkdir(parents=True,exist_ok=True);lock=(folder/'owner.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:lock.close();raise RecoveryError('one_qualification_transport_owner_only') from None
    try:
        folder,state_file,current=state(root,q,plan,amendment_ref)
        if current['terminal']is not None:return {'status':current['terminal'],'whole_calls_this_invocation':0,'continuation_ledger':ref(root,state_file)}
        if current['new_calls']>=MAX_NEW_CALLS:return {'status':'NEW_WHOLE_CALL_LIMIT_EXHAUSTED','whole_calls_this_invocation':0,'continuation_ledger':ref(root,state_file)}
        number=current['new_calls']+1;call=folder/f'call-{number:02d}'
        with dataset(q,root):
            prior_item=(read(checked(root,current['receipts'][-1]))['after_closed_ledger'] if current['receipts'] else amendment['parent_closed_ledger_snapshot'])
            previous,_=closed(root,q,plan,checked(root,prior_item));before,body=closed(root,q,plan);prefix(previous,before)
            immutable_bytes(call/'before_closed_ledger.json',body)
        intent={'schema_version':SCHEMA,'new_call':number,'actual_whole_call_ordinal':4+number,'amendment':amendment_ref,
            'frozen_plan':amendment['frozen_plan'],'before_closed_ledger':ref(root,call/'before_closed_ledger.json'),
            'session_headers':HEADERS,'requests_adapter_retries':0,'automatic_retry_loop':False,'clock_for_audit_only':core.clock_snapshot()}
        core.atomic_json(call/'intent.json',intent,immutable=True);current.update(new_calls=number,active_call=number);core.atomic_json(state_file,current)
        result=None;error=None
        try:
            with no_cache_session_factory(call/'HTTP_statistics_live.json'):
                result=q.execute(root,plan)
        except Exception as caught:error=caught
        with dataset(q,root):
            after,body=closed(root,q,plan);prefix(before,after);immutable_bytes(call/'after_closed_ledger.json',body)
        immutable_bytes(call/'HTTP_statistics_closed.json',(call/'HTTP_statistics_live.json').read_bytes())
        if error is None and result.get('status')==COMPLETE:
            if completed(root,q,plan,parent)is None:raise RecoveryError('returned_completion_missing_actual_offline_qualification_proof')
        eligible=error is not None and continuation_allowed(error)
        receipt={'schema_version':SCHEMA,'new_call':number,'amendment':amendment_ref,'intent':ref(root,call/'intent.json'),
            'before_closed_ledger':ref(root,call/'before_closed_ledger.json'),'after_closed_ledger':ref(root,call/'after_closed_ledger.json'),
            'closed_HTTP_statistics':ref(root,call/'HTTP_statistics_closed.json'),'error_type':type(error).__name__ if error else None,
            'error_code':safe_error(error) if error else None,'transport_continuation_eligible':eligible,'qualifier_result':result,
            'automatic_retry_performed':False,'original_budget_or_clock_reset':False,'scientific_admission_granted':False,'fits':0,'GPU_actions':0}
        receipt_file=call/'receipt.json';core.atomic_json(receipt_file,receipt,immutable=True)
        current['receipts'].append(ref(root,receipt_file));current['active_call']=None
        if error is not None and not eligible:current['terminal']='STOPPED_NON_TRANSPORT_NO_RETRY'
        core.atomic_json(state_file,current)
        return {'status':('STOPPED_TRANSPORT_NO_AUTOMATIC_RETRY' if eligible else current['terminal']) if error else result['status'],
            'whole_calls_this_invocation':1,'new_calls_used':number,'prior_actual_calls':4,'new_calls_limit':MAX_NEW_CALLS,
            'receipt':ref(root,receipt_file),'qualifier_result':result,'scientific_admission_granted':False,'fits':0,'GPU_actions':0}
    finally:lock.close()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    action=p.add_mutually_exclusive_group();action.add_argument('--write-amendment',action='store_true');action.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    try:
        result=write_amendment(a.root) if a.write_amendment else run(a.root,execute=a.execute)
        print(json.dumps(result,indent=2,allow_nan=False));return 3 if result['status'].startswith(('STOPPED_','NEW_WHOLE_CALL_LIMIT')) else 0
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'BLOCKED_PRESERVE_QUALIFICATION_LEDGER','error_type':type(error).__name__,
            'error_code':str(error) if isinstance(error,(RecoveryError,core.PreparationError)) else 'local_dependency_or_evidence_failed',
            'scientific_admission_granted':False,'fits':0,'GPU_actions':0,'network_actions_this_error_handler':0}));return 2

if __name__=='__main__':raise SystemExit(main())
