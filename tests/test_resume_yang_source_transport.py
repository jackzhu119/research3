"""Synthetic CPU-only transport-control tests; no provider/R2/native EEG calls.

The fake plan validator/prepare statuses below supply no real data authority.
Only local JSON journals, locks, HTTP request construction and error policy run.
"""
import copy
import fcntl
import json
from pathlib import Path

import pytest
import requests

from tools import resume_yang_source_transport as work


@pytest.fixture
def fixture_root(tmp_path,monkeypatch):
    target=tmp_path/'tools/resume_yang_source_transport.py';target.parent.mkdir()
    target.write_bytes(Path(work.__file__).read_bytes())
    plan={'dataset':'Yang2025','plan_sha256':work.PLAN_SHA,'record_count':153,
        'budgets':{'provider_body_bytes':100000,'r2_get_body_bytes':100000,'wall_seconds':172800},
        'records':[{'record_id':f'SYNTHETIC-{n:03d}','raw_files':[{'download':{'url':work.OFFICIAL_URL}}]} for n in range(153)],
        'fixture_scope':'NO_REAL_PLAN_OR_DATA_AUTHORITY'}
    filename=work.path(tmp_path,work.PLAN_PATH);work.core.atomic_json(filename,plan)
    monkeypatch.setattr(work.core,'validate_plan',lambda *args: {})
    ledger={'schema_version':'paper3-source-extension-budget-ledger-v1','plan_sha256':work.PLAN_SHA,
        'charged':{'provider_body_bytes':30,'r2_get_body_bytes':0},'elapsed_seconds':1.,
        'history':[{'kind':'provider_body_bytes','reserved_bytes':30,'identity':'SYNTHETIC_ONLY',
            'verified_bytes':None,'status':'FAILURE_FULL_RESERVATION_CONSUMED'}]}
    ledger_file=work.ledger_path(tmp_path,plan);work.core.atomic_json(ledger_file,ledger)
    before=work.path(tmp_path,'audit/data_plan_20261020/yang_transport_recovery/attempt01/ledger_before_probe.json')
    work.immutable_bytes(before,ledger_file.read_bytes())
    probe_file=before.with_name('probe.json')
    probe={'same_source_plan_sha256':work.PLAN_SHA,'status':'EXACT_THIRTY_BYTE_SOURCE_RANGE_VERIFIED',
        'body_reservation_kept_in_original_ledger':True,'parent_ledger':work.ref(tmp_path,before),
        'fixture_scope':'SYNTHETIC_NO_ACTUAL_HTTP_PROBE'}
    work.core.atomic_json(probe_file,probe)
    reason={'dataset':'Yang2025','same_plan_sha256':work.PLAN_SHA,'bounded_resume_attempt':1,
        'previous_error':'provider_did_not_honor_exact_range','budget_and_clock_not_reset':True,
        'actual_same_official_archive_range_probe':work.ref(tmp_path,probe_file)}
    work.core.atomic_json(work.path(tmp_path,work.REASON_PATH),reason)
    monkeypatch.setattr(requests.Session,'get',lambda *a,**kw:pytest.fail('unit test attempted actual network'))
    return tmp_path,plan,ledger_file


def authorize(fixture_root):
    root,plan,filename=fixture_root;work.write_amendment(root)
    return root,plan,filename


def fake_prepare(monkeypatch,fixture_root,errors):
    root,plan,filename=fixture_root;calls=[];finished=[]
    real_offline=work.offline_completed
    def offline(root,plan):
        if finished:return {'status':'SOURCE_EXTENSION_CACHE_COMPLETE','new_attempts':0,
            'fixture_scope':'EXPLICIT_SOFTWARE_COMPLETE_PROOF_SUBSTITUTE_NO_REAL_DATA'}
        return real_offline(root,plan)
    monkeypatch.setattr(work,'offline_completed',offline)
    def prepare(actual_root,actual_plan,*,execute,max_new_records,session):
        assert actual_root==root and actual_plan==plan and execute is True
        assert dict((key,session.headers[key]) for key in work.HEADERS)==work.HEADERS
        assert session.trust_env is True
        assert session.adapters['https://'].max_retries.total==0
        calls.append(max_new_records)
        ledger=work.core.Ledger(filename,plan)
        try:
            ledger.reserve('provider_body_bytes',30,'SYNTHETIC_RESERVED_REQUEST_NO_NETWORK')
            error=errors[len(calls)-1] if len(calls)<=len(errors) else None
            if error is not None:
                ledger.finish()
                raise error
            ledger.finish(verified_bytes=30)
            finished.append(True)
            return {'status':'SOURCE_EXTENSION_CACHE_COMPLETE','fixture_scope':'SYNTHETIC_CONTROL_FLOW_ONLY'}
        finally:ledger.checkpoint(close=True)
    monkeypatch.setattr(work.core,'prepare',prepare)
    return calls


def test_default_is_read_only_and_does_not_construct_session_or_prepare(fixture_root,monkeypatch):
    root,_,_=fixture_root;before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(work,'session',lambda:pytest.fail('dryrun constructed session'))
    monkeypatch.setattr(work.core,'prepare',lambda *a,**kw:pytest.fail('dryrun called prepare'))
    assert work.run(root)['status']=='DRY_RUN_SAMEPLAN_TRANSPORT_ONLY_NO_NETWORK'
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_amendment_copies_exact_closed_bytes_and_stays_immutable(fixture_root):
    root,plan,ledger_file=fixture_root;body=ledger_file.read_bytes()
    result=work.write_amendment(root);amendment=work.read(work.checked(root,result['amendment']))
    snapshot=work.checked(root,amendment['parent_closed_ledger_snapshot'])
    assert snapshot!=ledger_file and snapshot.read_bytes()==body
    assert amendment['authorization']==work.AUTHORIZATION
    assert amendment['wrapper_code']==work.code(root)
    value=work.read(ledger_file);value['elapsed_seconds']+=1;work.core.atomic_json(ledger_file,value)
    assert snapshot.read_bytes()==body
    assert work.write_amendment(root)['amendment']==result['amendment']


@pytest.mark.parametrize('field',['wall_intent','active_transfer'])
def test_open_original_ledger_cannot_authorize_or_reset(fixture_root,field):
    root,_,filename=fixture_root;value=work.read(filename);value[field]={'SYNTHETIC_OPEN':True}
    work.core.atomic_json(filename,value);body=filename.read_bytes()
    with pytest.raises(work.TransportRecoveryError,match='must_be_closed'):work.write_amendment(root)
    assert filename.read_bytes()==body and not work.path(root,work.AMENDMENT_PATH).exists()


def test_original_global_owner_blocks_amendment(fixture_root):
    root,_,_=fixture_root;filename=work.path(root,'audit/source_extensions_20261019/.source-extension-singleton.lock')
    with filename.open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(work.TransportRecoveryError,match='original_coordinator_active'):work.write_amendment(root)


@pytest.mark.parametrize('mutation',['plan','url','code'])
def test_only_original_plan_official_URL_and_executed_code_authorize(fixture_root,mutation):
    root,plan,_=fixture_root
    if mutation=='code':work.path(root,'tools/resume_yang_source_transport.py').write_text('CHANGED')
    else:
        value=copy.deepcopy(plan)
        if mutation=='plan':value['plan_sha256']='0'*64
        else:value['records'][0]['raw_files'][0]['download']['url']='https://example.org/other'
        work.core.atomic_json(work.path(root,work.PLAN_PATH),value)
    with pytest.raises(work.TransportRecoveryError):work.write_amendment(root)


def test_session_preserves_request_Range_identity_URL_and_has_no_adapter_retry():
    with work.session() as client:
        request=client.prepare_request(requests.Request('GET',work.OFFICIAL_URL,
            headers={'Range':'bytes=30-59','Accept-Encoding':'identity'}))
        assert request.url==work.OFFICIAL_URL
        assert request.headers['Range']=='bytes=30-59' and request.headers['Accept-Encoding']=='identity'
        assert all(request.headers[k]==v for k,v in work.HEADERS.items())
        assert client.trust_env is True and client.adapters['https://'].max_retries.total==0


def test_two_exact_range_failures_preserve_reservations_and_then_complete(fixture_root,monkeypatch):
    root,plan,filename=authorize(fixture_root);body=filename.read_bytes()
    calls=fake_prepare(monkeypatch,fixture_root,[work.core.PreparationError('provider_did_not_honor_exact_range')]*2)
    result=work.run(root,execute=True)
    assert result['status']=='SOURCE_EXTENSION_CACHE_COMPLETE' and len(calls)==3
    assert result['attempts_used_including_inherited']==4
    current=work.read(filename);assert current['charged']['provider_body_bytes']==120
    assert current.get('wall_intent') is None and current.get('active_transfer') is None
    amendment=work.read(work.path(root,work.AMENDMENT_PATH))
    assert work.checked(root,amendment['parent_closed_ledger_snapshot']).read_bytes()==body
    receipts=sorted(work.path(root,work.HISTORY).glob('attempt-*/receipt.json'))
    assert len(receipts)==3 and all(work.read(p)['after_closed_ledger'] for p in receipts)
    assert work.run(root,execute=True)['new_attempts']==0 and len(calls)==3


@pytest.mark.parametrize('error',[
    requests.exceptions.ReadTimeout('PRIVATE_SENTINEL'),requests.exceptions.ConnectTimeout('PRIVATE_SENTINEL'),
    requests.exceptions.Timeout('PRIVATE_SENTINEL'),requests.exceptions.ConnectionError('PRIVATE_SENTINEL'),
    requests.exceptions.ChunkedEncodingError('PRIVATE_SENTINEL')])
def test_explicit_transport_errors_are_finite_and_redacted(fixture_root,monkeypatch,error):
    root,_,_=authorize(fixture_root);calls=fake_prepare(monkeypatch,fixture_root,[error])
    assert work.run(root,execute=True)['status']=='SOURCE_EXTENSION_CACHE_COMPLETE' and len(calls)==2
    assert 'PRIVATE_SENTINEL' not in ''.join(p.read_text() for p in work.path(root,work.HISTORY).rglob('*.json'))


@pytest.mark.parametrize('error',[
    work.core.PreparationError('native_sha_or_size_mismatch'),
    work.core.PreparationError('zip_eof_crc_or_decoded_size_mismatch'),
    work.core.PreparationError('range_body_short'),work.core.PreparationError('cumulative_body_budget_exhausted'),
    requests.exceptions.SSLError('PRIVATE_SENTINEL'),requests.exceptions.ProxyError('PRIVATE_SENTINEL'),
    requests.exceptions.HTTPError('PRIVATE_SENTINEL'),requests.exceptions.InvalidURL('PRIVATE_SENTINEL'),
    ValueError('PRIVATE_SENTINEL')])
def test_native_integrity_budget_unknown_and_nontransport_errors_never_retry(fixture_root,monkeypatch,error):
    root,_,_=authorize(fixture_root);calls=fake_prepare(monkeypatch,fixture_root,[error])
    assert work.run(root,execute=True)['status']=='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
    assert len(calls)==1 and work.run(root,execute=True)['new_attempts']==0
    assert 'PRIVATE_SENTINEL' not in ''.join(p.read_text() for p in work.path(root,work.HISTORY).rglob('*.json'))


def test_sixteen_total_includes_prior_root_resume_and_survives_restart(fixture_root,monkeypatch):
    root,_,filename=authorize(fixture_root)
    calls=fake_prepare(monkeypatch,fixture_root,[requests.exceptions.ReadTimeout()] * 20)
    result=work.run(root,execute=True)
    assert result['status']=='TRANSPORT_RECOVERY_LIMIT_EXHAUSTED' and len(calls)==15
    assert work.read(filename)['charged']['provider_body_bytes']==30+15*30
    assert work.run(root,execute=True)['new_attempts']==0 and len(calls)==15


def test_unknown_interruption_keeps_attempt_intent_and_does_not_replay(fixture_root,monkeypatch):
    root,_,_=authorize(fixture_root);calls=fake_prepare(monkeypatch,fixture_root,[KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):work.run(root,execute=True)
    assert len(calls)==1
    with pytest.raises(work.TransportRecoveryError,match='uncertain_recovery_attempt'):work.run(root,execute=True)
    assert len(calls)==1


@pytest.mark.parametrize('mutation',['counter','terminal','history','charges'])
def test_recovery_or_original_ledger_rollback_cannot_grant_retry(fixture_root,monkeypatch,mutation):
    root,_,filename=authorize(fixture_root)
    calls=fake_prepare(monkeypatch,fixture_root,[work.core.PreparationError('native_sha_or_size_mismatch')])
    work.run(root,execute=True)
    state_file=work.path(root,work.HISTORY+'/recovery_ledger.json');state=work.read(state_file)
    if mutation=='counter':state['attempts_used_including_inherited']=1;state['attempt_receipts']=[]
    elif mutation=='terminal':state['terminal']=None
    else:
        value=work.read(filename)
        if mutation=='history':value['history'][0]['identity']='TAMPERED'
        else:value['charged']['provider_body_bytes']=0
        work.core.atomic_json(filename,value)
    work.core.atomic_json(state_file,state)
    with pytest.raises(work.TransportRecoveryError):work.run(root,execute=True)
    assert len(calls)==1


def offline_complete_fixture(monkeypatch,fixture_root):
    """Explicit JSON-only stand-ins isolate the completed re-entry path."""
    from p3work import source_extension_bridge as bridge
    root,plan,_=fixture_root;audit=work.path(root,f'audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}')
    for record in plan['records']:
        work.core.atomic_json(audit/record['record_id']/'state.json',{'status':'COMPLETE','record_id':record['record_id'],
            'fixture_scope':'SYNTHETIC_STATE_NO_DATA_AUTHORITY'})
    base={'status':'SOURCE_EXTENSION_CACHE_COMPLETE','dataset':'Yang2025','fixture_scope':'SYNTHETIC_MANIFEST_ONLY'}
    manifest_file=work.path(root,f'cache/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/source_cache.json')
    manifest={**base,'native_inventory':{},'unit_evidence':{},'role_assignment':{}}
    work.core.atomic_json(manifest_file,manifest)
    work.core.atomic_json(audit/'coordinator_status.json',{'status':'SOURCE_EXTENSION_CACHE_COMPLETE',
        'manifest':work.ref(root,manifest_file),'backup_receipt':{'fixture_scope':'NO_REAL_GET_PROOF'}})
    checks=[]
    def state_verified(root,state,plan,record):checks.append(record['record_id']);return True
    def assemble(root,plan,states):assert len(states)==153;return base
    proofs=[]
    def final_proof(root,manifest,path,closure):proofs.append(path)
    monkeypatch.setattr(work.core,'_state_verified',state_verified)
    monkeypatch.setattr(work.core,'assemble_manifest',assemble)
    monkeypatch.setattr(bridge,'_completed_preparation',final_proof)
    return checks,proofs,manifest_file


def test_completed_reentry_verifies_all_states_and_final_proof_without_any_write_or_GET(fixture_root,monkeypatch):
    root,_,_=fixture_root;checks,proofs,manifest=offline_complete_fixture(monkeypatch,fixture_root)
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    # The original coordinator lock is an existing completion dependency.
    lock=work.path(root,'audit/source_extensions_20261019/.source-extension-singleton.lock');lock.touch()
    before[lock.relative_to(root)]=b''
    monkeypatch.setattr(work,'session',lambda:pytest.fail('completed re-entry constructed HTTP session'))
    monkeypatch.setattr(work.core,'prepare',lambda *a,**kw:pytest.fail('completed re-entry called prepare'))
    result=work.run(root,execute=True)
    assert result['already_complete_offline_verified'] is True
    assert len(checks)==153 and proofs==[manifest]
    assert result['new_HTTP_or_R2_requests']==0 and result['coordinator_calls_this_invocation']==0
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not work.path(root,work.HISTORY).exists() and not work.path(root,work.AMENDMENT_PATH).exists()


def test_claimed_completed_cache_missing_actual_final_GET_proof_never_falls_back_to_prepare(fixture_root,monkeypatch):
    from p3work import source_extension_bridge as bridge
    root,_,_=fixture_root;offline_complete_fixture(monkeypatch,fixture_root)
    monkeypatch.setattr(bridge,'_completed_preparation',lambda *a,**kw:(_ for _ in ()).throw(ValueError('SYNTHETIC_BAD_GET_PROOF')))
    monkeypatch.setattr(work.core,'prepare',lambda *a,**kw:pytest.fail('bad completed proof retriggered prepare'))
    with pytest.raises(ValueError,match='SYNTHETIC_BAD_GET_PROOF'):work.run(root,execute=True)


def test_completed_cache_still_requires_every_actual_state(fixture_root,monkeypatch):
    root,_,_=fixture_root;offline_complete_fixture(monkeypatch,fixture_root)
    monkeypatch.setattr(work.core,'_state_verified',lambda *a:False)
    monkeypatch.setattr(work.core,'prepare',lambda *a,**kw:pytest.fail('bad completed state retriggered prepare'))
    with pytest.raises(ValueError,match='unverified_source_record'):work.run(root,execute=True)


def actual_process_record_error(fixture_root,error):
    """Run the frozen coordinator's real exception wrapper; no download occurs."""
    root,plan,filename=fixture_root;plan=copy.deepcopy(plan);plan['code']={};plan['budgets']['disk_reserve_bytes']=0
    record={'record_id':'SYNTHETIC_WRAPPED_TRANSPORT','subject_id':'SYNTHETIC_ONLY','session_id':'SYNTHETIC_ONLY',
        'raw_files':[{'path':'data/source_extensions_20261019/Yang2025/SYNTHETIC/data.bdf',
            'bytes':1,'sha256':'0'*64}]}
    ledger=work.core.Ledger(filename,plan)
    def fail_before_download(item,target,*,session,ledger):
        ledger.reserve('provider_body_bytes',30,'SYNTHETIC_WRAPPED_REQUEST_NO_NETWORK')
        raise error
    def forbidden(*a,**kw):pytest.fail('synthetic error test reached native or R2')
    try:
        work.core.process_record(root,plan,record,native_profile={},ledger=ledger,session=None,
            materialize=forbidden,backup_fn=forbidden,download_fn=fail_before_download)
    except work.core.PreparationError as caught:return caught
    finally:ledger.checkpoint(close=True)
    pytest.fail('synthetic exception unexpectedly disappeared')


@pytest.mark.parametrize('error',[requests.exceptions.ReadTimeout('PRIVATE_SENTINEL'),
    requests.exceptions.ConnectTimeout('PRIVATE_SENTINEL'),requests.exceptions.ConnectionError('PRIVATE_SENTINEL'),
    requests.exceptions.ChunkedEncodingError('PRIVATE_SENTINEL')])
def test_real_frozen_process_record_transport_exception_wrapper_still_recovers(fixture_root,monkeypatch,error):
    caught=actual_process_record_error(fixture_root,error)
    assert str(caught)=='operation_failed_details_suppressed'
    assert caught.__context__ is error and work.classify(caught)[0] is True
    root,_,_=authorize(fixture_root);calls=fake_prepare(monkeypatch,fixture_root,[caught])
    assert work.run(root,execute=True)['status']=='SOURCE_EXTENSION_CACHE_COMPLETE' and len(calls)==2
    assert 'PRIVATE_SENTINEL' not in ''.join(p.read_text() for p in work.path(root,work.HISTORY).rglob('*.json'))


@pytest.mark.parametrize('error',[ValueError('PRIVATE_SENTINEL'),requests.exceptions.SSLError('PRIVATE_SENTINEL'),
    requests.exceptions.ProxyError('PRIVATE_SENTINEL'),requests.exceptions.HTTPError('PRIVATE_SENTINEL'),
    work.core.PreparationError('native_sha_or_size_mismatch')])
def test_real_frozen_process_record_native_or_nontransport_wrapper_remains_terminal(fixture_root,error):
    caught=actual_process_record_error(fixture_root,error)
    assert work.classify(caught)[0] is False


def test_budget_exception_context_cannot_be_misclassified_as_transport():
    error=work.core.PreparationError('cumulative_wall_budget_exhausted')
    error.__context__=requests.exceptions.ReadTimeout()
    assert work.classify(error)[0] is False


def test_live_ledger_rollback_between_attempts_blocks_before_another_dispatch(fixture_root,monkeypatch):
    root,_,filename=authorize(fixture_root)
    calls=fake_prepare(monkeypatch,fixture_root,[requests.exceptions.ReadTimeout()])
    original_atomic=work.core.atomic_json
    def atomic(filename_to_write,value,**kwargs):
        original_atomic(filename_to_write,value,**kwargs)
        if (Path(filename_to_write).name=='recovery_ledger.json' and value.get('active_attempt') is None
                and len(value.get('attempt_receipts',[]))==1):
            amendment=work.read(work.path(root,work.AMENDMENT_PATH))
            rolled_back=work.read(work.checked(root,amendment['parent_closed_ledger_snapshot']))
            original_atomic(filename,rolled_back)
    monkeypatch.setattr(work.core,'atomic_json',atomic)
    with pytest.raises(work.TransportRecoveryError,match='history_rolled_back'):work.run(root,execute=True)
    assert len(calls)==1
