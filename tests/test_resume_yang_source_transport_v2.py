"""Synthetic CPU-only control fixtures. No original ledger, HTTP, R2 or fit."""
import copy
import fcntl
import hashlib
from pathlib import Path

import pytest
import requests

from tools import resume_yang_source_transport_v2 as work
from tools import resume_yang_source_transport as old
from tests import test_resume_yang_source_transport as parent_tests


@pytest.fixture
def fixture_root(tmp_path,monkeypatch):
 root,plan,ledger_file=parent_tests.fixture_root.__wrapped__(tmp_path,monkeypatch)
 plan['records'][0]['raw_files'][0]['download'].update(header_offset=100,archive_bytes=10000)
 old.core.atomic_json(old.path(root,old.PLAN_PATH),plan)
 (root/work.ENTRY).write_bytes(Path(work.__file__).read_bytes())
 old.write_amendment(root)
 parent_tests.fake_prepare(monkeypatch,(root,plan,ledger_file),[requests.exceptions.ReadTimeout()]*20)
 assert old.run(root,execute=True)['status']=='TRANSPORT_RECOVERY_LIMIT_EXHAUSTED'
 parent_state=old.path(root,old.HISTORY+'/recovery_ledger.json')
 monkeypatch.setattr(work,'PARENT_TERMINAL_SHA',hashlib.sha256(parent_state.read_bytes()).hexdigest())
 ledger=work.core.Ledger(ledger_file,plan)
 try:ledger.reserve('provider_body_bytes',30,'range:100:30');ledger.finish(verified_bytes=30)
 finally:ledger.checkpoint(close=True)
 probe={'record_id':plan['records'][0]['record_id'],'same_plan_sha256':work.PLAN_SHA,'offset':100,'requested_bytes':30,
  'original_official_URL_unchanged':True,'scientific_fits':0,'GPU_actions':0,
  'status':'EXACT_SOURCE_LOCAL_HEADER_THIRTY_BYTES_VERIFIED','body_sha256':'b'*64,'PK_local_header_magic':True,
  'actual_safe_response':{'status_code':206,'Content-Length':'30','Content-Range':'bytes 100-129/10000','Content-Encoding':'identity'},
  'ledger_after_probe':old.ref(root,ledger_file),'failed_reservations_retained':True,
  'fixture_scope':'SYNTHETIC_NO_REAL_PROVIDER_PROBE_OR_DATA_AUTHORITY'}
 probe_file=work.path(root,work.PROBE_PATH);work.core.atomic_json(probe_file,probe)
 monkeypatch.setattr(work,'PROBE_SHA',work.ref(root,probe_file)['sha256'])
 return root,plan,ledger_file


def authorize(fixture_root):
 root,plan,filename=fixture_root;work.write_amendment(root);return root,plan,filename


def single_prepare(monkeypatch,fixture_root,error=None):
 root,plan,filename=fixture_root;calls=[]
 def prepare(actual_root,actual_plan,*,execute,max_new_records,session):
  assert actual_root==root and actual_plan==plan and execute
  assert session.trust_env is True
  assert all(session.headers[k]==v for k,v in work.HEADERS.items())
  assert session.adapters['https://'].max_retries.total==0
  calls.append(max_new_records);ledger=work.core.Ledger(filename,plan)
  try:
   ledger.reserve('provider_body_bytes',30,'SYNTHETIC_ONLY_NO_NETWORK')
   if error is not None:ledger.finish();raise error
   ledger.finish(verified_bytes=30)
   return {'status':'PAUSED_EXPLICIT_RECORD_LIMIT','fixture_scope':'NO_REAL_CACHE_COMPLETE'}
  finally:ledger.checkpoint(close=True)
 monkeypatch.setattr(work.core,'prepare',prepare)
 return calls


def test_default_readonly_no_session_no_call_or_history_write(fixture_root,monkeypatch):
 root,_,_=fixture_root;before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
 monkeypatch.setattr(old,'session',lambda:pytest.fail('dry constructed Session'))
 assert work.run(root)['status']=='DRY_RUN_SINGLE_EXPLICIT_CALL_NO_NETWORK'
 assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_authorization_preserves_V1_probe_exactbytes_and_uses_opaque_snapshots(fixture_root):
 root,plan,ledger=fixture_root;before=ledger.read_bytes();probe=work.path(root,work.PROBE_PATH).read_bytes()
 result=work.write_amendment(root);value,item=work.verify_static_authority(root,plan)
 assert item==result['amendment'] and work.checked(root,value['post_probe_closed_ledger_snapshot']).read_bytes()==before
 assert work.checked(root,value['diagnostic_probe_snapshot']).read_bytes()==probe
 assert value['diagnostic_probe_snapshot']['path'].endswith('.txt')
 assert value['max_independent_execute_calls']==8 and value['no_tight_retry']
 assert ledger.read_bytes()==before


def test_retryable_error_makes_only_one_call_per_explicit_execute(fixture_root,monkeypatch):
 root,_,filename=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root,requests.exceptions.ReadTimeout('PRIVATE_SENTINEL'))
 first=work.run(root,execute=True)
 assert len(calls)==1 and first['status']=='PAUSED_TRANSPORT_ERROR_REQUIRES_NEW_EXPLICIT_EXECUTE'
 assert work.run(root,execute=True)['calls_used']==2 and len(calls)==2
 assert 'PRIVATE_SENTINEL' not in ''.join(p.read_text() for p in work.path(root,work.HISTORY).rglob('call-*/*.json'))
 assert work.read(filename)['charged']['provider_body_bytes']==570


def test_eight_distinct_calls_are_finite_and_not_one_tight_loop(fixture_root,monkeypatch):
 root,_,_=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root,requests.exceptions.ReadTimeout())
 for number in range(1,9):
  result=work.run(root,execute=True);assert len(calls)==number and result['calls_used']==number
 assert result['status']=='V2_EXPLICIT_CALL_LIMIT_EXHAUSTED'
 assert work.run(root,execute=True)['coordinator_calls_this_invocation']==0 and len(calls)==8


@pytest.mark.parametrize('error',[work.core.PreparationError('native_SHA_mismatch'),
 work.core.PreparationError('zip_eof_crc_or_decoded_size_mismatch'),work.core.PreparationError('cumulative_body_budget_exhausted'),
 requests.exceptions.SSLError('PRIVATE_SENTINEL'),requests.exceptions.ProxyError('PRIVATE_SENTINEL'),ValueError('PRIVATE_SENTINEL')])
def test_nontransport_never_grants_another_call(fixture_root,monkeypatch,error):
 root,_,_=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root,error)
 assert work.run(root,execute=True)['status']=='STOPPED_NON_TRANSPORT_ERROR_NO_RETRY'
 assert work.run(root,execute=True)['coordinator_calls_this_invocation']==0 and len(calls)==1


def test_unknown_active_call_is_never_automatically_replayed(fixture_root,monkeypatch):
 root,_,_=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root,KeyboardInterrupt())
 with pytest.raises(KeyboardInterrupt):work.run(root,execute=True)
 with pytest.raises(work.TransportRecoveryError,match='unknown_active'):work.run(root,execute=True)
 assert len(calls)==1


@pytest.mark.parametrize('field',['wall_intent','active_transfer'])
def test_open_original_budget_does_not_authorize_V2(fixture_root,field):
 root,_,filename=fixture_root;value=work.read(filename);value[field]={'fixture':True};work.core.atomic_json(filename,value)
 with pytest.raises(work.TransportRecoveryError):work.write_amendment(root)
 assert not work.path(root,work.AMENDMENT_PATH).exists()


def test_original_global_owner_blocks_snapshot_and_dispatch(fixture_root):
 root,_,_=fixture_root;lockfile=work.path(root,'audit/source_extensions_20261019/.source-extension-singleton.lock')
 with lockfile.open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(old.TransportRecoveryError,match='original_coordinator_active'):work.write_amendment(root)
 assert not work.path(root,work.AMENDMENT_PATH).exists()


def test_probe_charge_and_original_history_cannot_be_rolled_back(fixture_root):
 root,_,filename=fixture_root;value=work.read(filename);value['charged']['provider_body_bytes']-=30
 work.core.atomic_json(filename,value)
 with pytest.raises(work.TransportRecoveryError):work.write_amendment(root)


def test_static_authority_never_reads_live_original_ledger_or_takes_lock(fixture_root,monkeypatch):
 root,plan,filename=authorize(fixture_root)
 original=Path.read_bytes
 def guarded(self):
  if self==filename:pytest.fail('static verifier read live scientific ledger')
  return original(self)
 monkeypatch.setattr(Path,'read_bytes',guarded)
 monkeypatch.setattr(old,'coordinator_lock',lambda *a:pytest.fail('static verifier tried lock'))
 monkeypatch.setattr(old,'session',lambda:pytest.fail('static verifier constructed Session'))
 monkeypatch.setattr(work.core,'Ledger',lambda *a,**k:pytest.fail('static verifier constructed Ledger'))
 value,_=work.verify_static_authority(root,plan);assert value['status']==work.STATUS


def test_later_closed_ledger_changes_do_not_invalidate_opaque_probe(fixture_root,monkeypatch):
 root,plan,filename=authorize(fixture_root);single_prepare(monkeypatch,fixture_root)
 work.run(root,execute=True)
 assert work.verify_static_authority(root,plan)[0]['status']==work.STATUS


def test_counter_or_history_deletion_cannot_reset_calls(fixture_root,monkeypatch):
 root,_,_=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root)
 work.run(root,execute=True);filename=work.path(root,work.HISTORY+'/recovery_ledger.json')
 value=work.read(filename);value['calls_used']=0;value['call_receipts']=[];work.core.atomic_json(filename,value)
 with pytest.raises(work.TransportRecoveryError,match='history_rolled_back'):work.run(root,execute=True)
 assert len(calls)==1


def test_already_complete_is_pure_offline_no_session_attempt_lock_or_ledger_constructor(fixture_root,monkeypatch):
 root,plan,_=fixture_root
 checks,proofs,_=parent_tests.offline_complete_fixture(monkeypatch,fixture_root)
 status_file=work.path(root,f'audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/coordinator_status.json')
 status=work.read(status_file);status['record_count']=153;work.core.atomic_json(status_file,status)
 before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
 monkeypatch.setattr(old,'coordinator_lock',lambda *a:pytest.fail('completed verifier took a lock'))
 monkeypatch.setattr(old,'session',lambda:pytest.fail('completed verifier constructed Session'))
 monkeypatch.setattr(work.core,'Ledger',lambda *a,**k:pytest.fail('completed verifier constructed Ledger'))
 monkeypatch.setattr(work.core,'prepare',lambda *a,**k:pytest.fail('completed verifier dispatched'))
 result=work.run(root,execute=True)
 assert result['already_complete_offline_verified'] and result['coordinator_calls_this_invocation']==0
 assert len(checks)==153 and len(proofs)==1
 assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_session_preserves_default_proxy_CA_environment_and_original_range():
 with old.session() as client:
  request=client.prepare_request(requests.Request('GET',old.OFFICIAL_URL,headers={'Range':'bytes=100-129','Accept-Encoding':'identity'}))
  assert request.url==old.OFFICIAL_URL and request.headers['Range']=='bytes=100-129'
  assert client.trust_env is True and client.adapters['https://'].max_retries.total==0


@pytest.mark.parametrize('terminal',['SOURCE_EXTENSION_CACHE_COMPLETE','V2_EXPLICIT_CALL_LIMIT_EXHAUSTED'])
def test_terminal_cannot_be_forged_after_only_retryable_error(fixture_root,monkeypatch,terminal):
 root,_,_=authorize(fixture_root);calls=single_prepare(monkeypatch,fixture_root,requests.exceptions.ReadTimeout())
 work.run(root,execute=True);filename=work.path(root,work.HISTORY+'/recovery_ledger.json')
 state=work.read(filename);state['terminal']=terminal;work.core.atomic_json(filename,state)
 with pytest.raises(work.TransportRecoveryError,match='final_immutable_receipt'):work.run(root,execute=True)
 assert len(calls)==1


def test_external_data_complete_cannot_convert_retryable_transport_history_into_V2_success(fixture_root,monkeypatch):
 root,plan,_=authorize(fixture_root);single_prepare(monkeypatch,fixture_root,requests.exceptions.ReadTimeout())
 work.run(root,execute=True)
 parent_tests.offline_complete_fixture(monkeypatch,fixture_root)
 status_file=work.path(root,f'audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/coordinator_status.json')
 value=work.read(status_file);value['record_count']=153;work.core.atomic_json(status_file,value)
 with pytest.raises(work.TransportRecoveryError,match='successful_final_call'):work.verify_complete_transport_history(root,plan)


def test_zero_V2_calls_cannot_be_turned_into_complete_transport_chain(fixture_root,monkeypatch):
 root,plan,_=authorize(fixture_root)
 _,ref=work.verify_static_authority(root,plan);work._history(root,plan,ref,initialize=True)
 parent_tests.offline_complete_fixture(monkeypatch,fixture_root)
 status_file=work.path(root,f'audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/coordinator_status.json')
 value=work.read(status_file);value['record_count']=153;work.core.atomic_json(status_file,value)
 with pytest.raises(work.TransportRecoveryError,match='successful_final_call'):work.verify_complete_transport_history(root,plan)


def test_completed_offline_proof_returns_final_scientific_provenance(fixture_root,monkeypatch):
 from p3work import source_extension_bridge as bridge
 root,plan,_=fixture_root;parent_tests.offline_complete_fixture(monkeypatch,fixture_root)
 status_file=work.path(root,f'audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/coordinator_status.json')
 value=work.read(status_file);value['record_count']=153;work.core.atomic_json(status_file,value)
 monkeypatch.setattr(bridge,'_completed_preparation',lambda r,m,p,closure:closure.update({'fixture_only_final_proof.json':'c'*64}))
 result=work.verify_complete_transport_history(root,plan)
 assert result['provenance_closure']=={'fixture_only_final_proof.json':'c'*64}
