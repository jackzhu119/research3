"""JSON/lock/HTTP-factory software fixtures only; no real source or R2 requests.

Fake qualification results below carry no scientific/native admission authority.
The frozen qualifier and parser are never modified or executed on real data.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tools import resume_dreyer_qualification_transport as work
REAL_LOAD=work.load


@pytest.fixture
def software(tmp_path,monkeypatch):
    script=tmp_path/'tools/resume_dreyer_qualification_transport.py';script.parent.mkdir()
    script.write_bytes(Path(work.__file__).read_bytes())
    for name in (work.ENTRY,work.NATIVE):
        destination=tmp_path/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((work.ROOT/name).read_bytes())
    plan={'plan_sha256':work.PLAN_SHA,'budgets':{'provider_body_bytes':10000,'r2_get_body_bytes':10000,'wall_seconds':10800},
        'fixture_scope':'NO_REAL_QUALIFICATION_PLAN_AUTHORITY'}
    work.core.atomic_json(work.path(tmp_path,work.PLAN_PATH),plan)
    ledger={'schema_version':'paper3-source-extension-budget-ledger-v1','plan_sha256':work.PLAN_SHA,
        'charged':{'provider_body_bytes':30,'r2_get_body_bytes':0},'elapsed_seconds':1.,'history':[
            {'kind':'provider_body_bytes','identity':'SYNTHETIC_INITIAL_FAILED_REQUEST','reserved_bytes':30,
             'verified_bytes':None,'status':'FAILURE_FULL_RESERVATION_CONSUMED'}]}
    ledger_file=work.ledger_file(tmp_path);work.core.atomic_json(ledger_file,ledger)
    for ordinal in (1,2):
        base=work.path(tmp_path,f'audit/data_plan_20261020/dreyer_qualification_network_recovery/attempt{ordinal:02d}')
        snapshot=base/'prior_closed_ledger.json';work.immutable_bytes(snapshot,ledger_file.read_bytes())
        reason=({'same_qualification_plan_sha256':work.PLAN_SHA,'closed_ledger':True,'no_budget_or_wall_reset':True,
                 'prior_ledger':work.ref(tmp_path,snapshot)} if ordinal==1 else
                {'same_plan_sha256':work.PLAN_SHA,'bounded_resume_attempt':2,'prior_failed_reservation_retained':True,
                 'budget_or_clock_reset':False,'closed_ledger_snapshot':work.ref(tmp_path,snapshot)})
        work.core.atomic_json(base/'HEAD_probe_and_resume_reason.json',reason)
    @contextmanager
    def dataset_lock(root):
        filename=root/'audit/SYNTHETIC_DATASET_LOCK';filename.parent.mkdir(parents=True,exist_ok=True)
        with filename.open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);yield
    def closed_history(value):
        assert 'wall_intent' not in value and 'active_transfer' not in value
        counted={'provider_body_bytes':0,'r2_get_body_bytes':0}
        for row in value['history']:
            counted[row['kind']]+=row['verified_bytes'] if row.get('status')=='SUCCESS_FULL_BODY_VERIFIED' else row['reserved_bytes']
        assert counted==value['charged']
        return value['charged'],value['elapsed_seconds']
    wrapper=SimpleNamespace(dataset_lock=dataset_lock,closed_history=closed_history)
    finished=[];calls=[];errors=[];parent={'fixture_scope':'NO_REAL_PARENT_AUTHORITY'}
    q=SimpleNamespace(private_module=lambda *a:wrapper,
        completed_qualification=lambda *a:({'status':work.COMPLETE,'scientific_admission_granted':False,
            'fixture_scope':'SOFTWARE_COMPLETION_PROOF_SUBSTITUTE'} if finished else None))
    def fake_get(self,*a,**kw):
        assert self.headers['Cache-Control']=='no-cache' and self.headers['Pragma']=='no-cache'
        assert self.trust_env is True and self.adapters['https://'].max_retries.total==0
        return SimpleNamespace(status_code=200,headers={'Content-Length':'10','Content-Encoding':'identity',
            'Authorization':'PRIVATE_HEADER_VALUE','PRIVATE_HEADER_KEY':'PRIVATE_HEADER_VALUE'})
    monkeypatch.setattr(requests.Session,'get',fake_get)
    def execute(root,current_plan):
        # Acquiring the original dataset lock proves snapshot locks were released.
        with wrapper.dataset_lock(root):
            calls.append(1);ledger=work.core.Ledger(ledger_file,current_plan)
            try:
                one=requests.Session();two=requests.Session()
                one.get('https://example.invalid/PRIVATE_URL?PRIVATE_QUERY',stream=True)
                ledger.reserve('provider_body_bytes',30,'SYNTHETIC_RESERVED_REQUEST_NO_NETWORK')
                error=errors[len(calls)-1] if len(calls)<=len(errors) else None
                if error is not None:ledger.finish();raise error
                ledger.finish(verified_bytes=30)
                return {'status':'PAUSED_STRUCTURE_QUALIFICATION_ONLY','scientific_admission_granted':False,
                    'fixture_scope':'SOFTWARE_CONTROL_FLOW_ONLY'}
            finally:ledger.checkpoint(close=True)
    q.execute=execute
    monkeypatch.setattr(work,'load',lambda root:(Path(root).resolve(),q,plan,parent))
    return SimpleNamespace(root=tmp_path,plan=plan,q=q,parent=parent,ledger=ledger_file,
        finished=finished,calls=calls,errors=errors,wrapper=wrapper)


def authorize(software):work.write_amendment(software.root);return software


def test_default_is_read_only_and_never_installs_factory_or_calls_qualifier(software,monkeypatch):
    before={p.relative_to(software.root):p.read_bytes() for p in software.root.rglob('*') if p.is_file()}
    monkeypatch.setattr(work,'no_cache_session_factory',lambda *a:pytest.fail('dryrun installed factory'))
    result=work.run(software.root)
    assert result['status']=='DRY_RUN_FIXED_QUALIFICATION_TRANSPORT_ONLY' and not software.calls
    assert before=={p.relative_to(software.root):p.read_bytes() for p in software.root.rglob('*') if p.is_file()}


def test_amendment_binds_four_prior_calls_and_exact_parent_snapshot(software):
    body=software.ledger.read_bytes();result=work.write_amendment(software.root)
    value=work.read(work.checked(software.root,result['amendment']))
    assert value['prior_calls']['prior_actual_call_names']==['pilot','full','resume01','resume02']
    assert value['authorization']['maximum_new_whole_job_calls']==4
    snapshot=work.checked(software.root,value['parent_closed_ledger_snapshot'])
    assert snapshot!=software.ledger and snapshot.read_bytes()==body
    assert work.write_amendment(software.root)['amendment']==result['amendment']


def test_unamended_execution_cannot_call_qualifier(software):
    with pytest.raises(FileNotFoundError):work.run(software.root,execute=True)
    assert not software.calls


@pytest.mark.parametrize('field',['wall_intent','active_transfer'])
def test_open_original_ledger_never_authorizes_or_resets(software,field):
    value=work.read(software.ledger);value[field]={'SYNTHETIC_UNKNOWN':True};work.core.atomic_json(software.ledger,value)
    body=software.ledger.read_bytes()
    with pytest.raises(ValueError,match='must_be_closed'):work.write_amendment(software.root)
    assert software.ledger.read_bytes()==body and not software.calls


def test_each_explicit_execute_calls_qualifier_once_and_stops_on_transport_error(software):
    authorize(software);software.errors.extend([work.core.PreparationError('direct_get_status_or_declared_size_mismatch')]*5)
    for number in range(1,5):
        result=work.run(software.root,execute=True)
        assert result['status']=='STOPPED_TRANSPORT_NO_AUTOMATIC_RETRY'
        assert result['whole_calls_this_invocation']==1 and result['new_calls_used']==number
        assert len(software.calls)==number
    assert work.run(software.root,execute=True)['status']=='NEW_WHOLE_CALL_LIMIT_EXHAUSTED'
    assert len(software.calls)==4
    assert work.read(software.ledger)['charged']['provider_body_bytes']==30+4*30


@pytest.mark.parametrize('error',[work.core.PreparationError('native_sha_or_size_mismatch'),
    work.core.PreparationError('zip_eof_crc_or_decoded_size_mismatch'),
    work.core.PreparationError('cumulative_body_budget_exhausted'),requests.exceptions.SSLError('PRIVATE_SECRET'),
    requests.exceptions.ProxyError('PRIVATE_SECRET'),ValueError('PRIVATE_SECRET')])
def test_native_integrity_budget_or_nontransport_error_is_terminal(software,error):
    authorize(software);software.errors.append(error)
    assert work.run(software.root,execute=True)['status']=='STOPPED_NON_TRANSPORT_NO_RETRY'
    assert work.run(software.root,execute=True)['whole_calls_this_invocation']==0 and len(software.calls)==1
    assert 'PRIVATE_SECRET' not in ''.join(p.read_text() for p in work.path(software.root,work.HISTORY).rglob('*.json'))


@pytest.mark.parametrize('error',[requests.exceptions.ReadTimeout('PRIVATE_SECRET'),requests.exceptions.ConnectTimeout(),
    requests.exceptions.ConnectionError(),requests.exceptions.ChunkedEncodingError()])
def test_requests_transport_errors_allow_only_a_later_explicit_call(software,error):
    authorize(software);software.errors.append(error)
    assert work.run(software.root,execute=True)['status']=='STOPPED_TRANSPORT_NO_AUTOMATIC_RETRY'
    assert len(software.calls)==1
    assert work.run(software.root,execute=True)['whole_calls_this_invocation']==1 and len(software.calls)==2


def test_session_factory_restored_and_every_instance_closed_after_exception(software,monkeypatch):
    authorize(software);original=requests.Session;closed=[]
    old_close=original.close
    def close(instance):closed.append(id(instance));old_close(instance)
    monkeypatch.setattr(original,'close',close);software.errors.append(requests.exceptions.ReadTimeout('PRIVATE_SECRET'))
    result=work.run(software.root,execute=True)
    assert requests.Session is original and len(closed)==2 and len(set(closed))==2
    receipt=work.read(work.checked(software.root,result['receipt']))
    stats=work.read(work.checked(software.root,receipt['closed_HTTP_statistics']))
    assert stats['GET_calls']==1 and stats['HTTP_status_counts']=={'200':1}
    assert stats['declared_content_length_counts']=={'10':1} and stats['content_encoding_counts']=={'identity':1}
    assert all(secret not in json.dumps(stats) for secret in ('PRIVATE_URL','PRIVATE_QUERY','PRIVATE_HEADER_VALUE','PRIVATE_HEADER_KEY'))


def test_unknown_interruption_restores_factory_and_blocks_replay(software,monkeypatch):
    authorize(software);original=requests.Session;software.errors.append(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):work.run(software.root,execute=True)
    assert requests.Session is original
    with pytest.raises(ValueError,match='unknown_prior_whole_call'):work.run(software.root,execute=True)
    assert len(software.calls)==1


@pytest.mark.parametrize('mutation',['counter','terminal','original_charges'])
def test_history_rollback_never_grants_another_call(software,mutation):
    authorize(software);software.errors.append(work.core.PreparationError('native_sha_or_size_mismatch'))
    work.run(software.root,execute=True)
    filename=work.path(software.root,work.HISTORY+'/continuation_ledger.json');state=work.read(filename)
    if mutation=='counter':state['new_calls']=0;state['receipts']=[]
    elif mutation=='terminal':state['terminal']=None
    else:
        value=work.read(software.ledger);value['charged']['provider_body_bytes']=0;work.core.atomic_json(software.ledger,value)
    work.core.atomic_json(filename,state)
    with pytest.raises((ValueError,AssertionError)):work.run(software.root,execute=True)
    assert len(software.calls)==1


def test_completed_reentry_uses_offline_qualifier_proof_and_creates_no_attempt(software,monkeypatch):
    software.finished.append(True)
    before={p.relative_to(software.root):p.read_bytes() for p in software.root.rglob('*') if p.is_file()}
    # Original dataset lock file already exists in every real previously run job.
    filename=software.root/'audit/SYNTHETIC_DATASET_LOCK';filename.touch();before[filename.relative_to(software.root)]=b''
    original=requests.Session
    result=work.run(software.root,execute=True)
    assert result['already_completed_offline_verified'] is True and result['new_HTTP_or_R2_requests']==0
    assert requests.Session is original and not software.calls and not work.path(software.root,work.HISTORY).exists()
    assert before=={p.relative_to(software.root):p.read_bytes() for p in software.root.rglob('*') if p.is_file()}


def test_bad_completed_proof_cannot_fall_back_to_a_new_call(software,monkeypatch):
    def bad(*a):raise work.core.PreparationError('qualification_final_summary_binding_changed')
    software.q.completed_qualification=bad
    with pytest.raises(ValueError,match='qualification_final_summary_binding_changed'):work.run(software.root,execute=True)
    assert not software.calls


@pytest.mark.parametrize('changed,reason',[(work.ENTRY,'exact_frozen_qualifier_4ef_required'),
    (work.NATIVE,'canonical_ec2f_must_remain_unchanged')])
def test_real_load_rejects_changed_frozen_qualifier_and_native_bytes(software,changed,reason):
    # The real pre-import fixed-SHA gate rejects changed code without needing
    # a fabricated 520-record parent proof hierarchy.
    for name,expected in ((work.ENTRY,work.QUALIFIER_SHA),(work.NATIVE,work.NATIVE_SHA)):
        assert work.ref(software.root,work.path(software.root,name))['sha256']==expected
    work.path(software.root,changed).write_text('CHANGED')
    with pytest.raises(work.RecoveryError,match=reason):REAL_LOAD(software.root)


def test_real_direct_download_header_failure_keeps_full_original_reservation(software,monkeypatch):
    """Exercise the unchanged download kernel behind the actual factory hook."""
    authorize(software)
    def get(instance,*a,**kw):
        assert instance.headers['Cache-Control']=='no-cache' and instance.headers['Pragma']=='no-cache'
        assert instance.adapters['https://'].max_retries.total==0
        return SimpleNamespace(status_code=200,headers={'Content-Length':'9','Content-Encoding':'identity'},
            close=lambda:None,iter_content=lambda *a:pytest.fail('header failure must not read any body'))
    monkeypatch.setattr(requests.Session,'get',get)
    item={'path':'data/source_extensions_20261019/Dreyer2023/SYNTHETIC_ONLY.edf','bytes':10,'sha256':'0'*64,
        'download':{'kind':'direct','url':'https://data.nemar.org/SYNTHETIC_ONLY.edf'}}
    def execute(root,plan):
        with software.wrapper.dataset_lock(root):
            software.calls.append(1);ledger=work.core.Ledger(software.ledger,plan)
            try:work.core.download_component(item,root/item['path'],session=requests.Session(),ledger=ledger)
            finally:ledger.checkpoint(close=True)
    software.q.execute=execute
    result=work.run(software.root,execute=True)
    assert result['status']=='STOPPED_TRANSPORT_NO_AUTOMATIC_RETRY' and len(software.calls)==1
    original=work.read(software.ledger)
    assert original['charged']['provider_body_bytes']==40
    assert original['history'][-1]['status']=='FAILURE_FULL_RESERVATION_CONSUMED'
    assert original['history'][-1]['reserved_bytes']==10 and 'wall_intent' not in original


def test_threaded_process_is_rejected_before_a_call_intent(software,monkeypatch):
    authorize(software);monkeypatch.setattr(work.threading,'active_count',lambda:2)
    with pytest.raises(ValueError,match='dedicated_non_threaded'):work.run(software.root,execute=True)
    assert not software.calls and not list(work.path(software.root,work.HISTORY).glob('call-*'))
