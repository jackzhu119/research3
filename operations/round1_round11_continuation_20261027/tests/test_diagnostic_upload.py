import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('diagnostic_upload_under_test', HERE/'diagnostic_upload.py')
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)
TOKEN = 'fixture-private-token-must-never-appear'
HEAD = 'a'*40
COMMIT = 'b'*40


class FakeGitHub:
    def __init__(self, *, branch=True, conflicts=0, corrupted=False, create_conflict=False):
        self.branch = branch
        self.conflicts = conflicts
        self.corrupted = corrupted
        self.create_conflict = create_conflict
        self.calls = []
        self.content = None

    def __call__(self, endpoint, token, body, method, timeout):
        assert token == TOKEN
        assert 0 < timeout <= 8
        self.calls.append((endpoint, body, method))
        if endpoint == 'git/ref/heads/' + u.BRANCH:
            if not self.branch:
                raise u.DeliveryError('HTTP_ERROR', 404)
            return {'object':{'sha':HEAD}}
        if endpoint == 'git/ref/heads/' + u.BASE_BRANCH:
            return {'object':{'sha':HEAD}}
        if endpoint == 'git/refs':
            assert body == {'ref':'refs/heads/' + u.BRANCH, 'sha':HEAD}
            self.branch = True
            if self.create_conflict:
                raise u.DeliveryError('HTTP_ERROR', 422)
            return {'object':{'sha':HEAD}}
        if method == 'PUT':
            assert endpoint == 'contents/' + u.REPORT_PATH
            assert body['branch'] == u.BRANCH
            if self.conflicts:
                self.conflicts -= 1
                raise u.DeliveryError('HTTP_ERROR', 409)
            self.content = base64.b64decode(body['content'])
            return {'commit':{'sha':COMMIT}, 'content':{'sha':'c'*40}}
        if endpoint.endswith('?ref=' + u.BRANCH):
            if self.content is None:
                raise u.DeliveryError('HTTP_ERROR', 404)
            return {'sha':'c'*40}
        if endpoint.endswith('?ref=' + COMMIT):
            raw = b'{}' if self.corrupted else self.content
            return {'encoding':'base64', 'content':base64.b64encode(raw).decode()}
        raise AssertionError(endpoint)


def publish(fake, value=None, **kwargs):
    data = json.dumps(value or {'round':1, 'diagnostic_only':True}).encode()
    events=[]
    result = u.publish(data, '/unused-home', _request=fake,
        _token=lambda *args:TOKEN, emit=events.append, **kwargs)
    assert TOKEN not in json.dumps(result)
    assert TOKEN not in json.dumps(events)
    return result, events


def test_verified_upload_uses_dedicated_path_and_pinned_full_get():
    fake=FakeGitHub()
    result, events=publish(fake)
    assert result['status']=='DIAGNOSTIC_UPLOADED_VERIFIED'
    assert result['complete_GET_SHA256_verified'] is True
    assert result['commit_sha']==COMMIT
    assert fake.calls[-1][0].endswith('?ref='+COMMIT)
    assert all('monitoring/paper3-round1/status.json' not in c[0] for c in fake.calls)
    assert [e['step'] for e in events]==[
        'CREDENTIAL_DISCOVERY','BRANCH_OBSERVATION','DIAGNOSTIC_PUBLISH_ATTEMPT_1','PINNED_FULL_GET_VERIFY']


def test_creates_only_diagnostic_branch_from_original_progress_branch():
    fake=FakeGitHub(branch=False)
    result,_=publish(fake)
    assert result['status']=='DIAGNOSTIC_UPLOADED_VERIFIED'
    assert any(c[0]=='git/ref/heads/'+u.BASE_BRANCH for c in fake.calls)
    assert sum(c[2]=='POST' for c in fake.calls)==1


def test_branch_creation_lost_ack_reobserves():
    fake=FakeGitHub(branch=False,create_conflict=True)
    result,_=publish(fake)
    assert result['status']=='DIAGNOSTIC_UPLOADED_VERIFIED'
    assert sum(c[0]=='git/ref/heads/'+u.BRANCH for c in fake.calls)==2


def test_cas_conflicts_bounded_and_retried():
    fake=FakeGitHub(conflicts=2)
    result,_=publish(fake)
    assert result['status']=='DIAGNOSTIC_UPLOADED_VERIFIED'
    assert sum(c[2]=='PUT' for c in fake.calls)==3
    fake=FakeGitHub(conflicts=10)
    result,_=publish(fake)
    assert result['status']=='RETRYING'
    assert sum(c[2]=='PUT' for c in fake.calls)==3


def test_corrupt_full_get_never_claims_verified_upload():
    result,_=publish(FakeGitHub(corrupted=True))
    assert result['status']=='RETRYING'
    assert result['reason_code']=='REMOTE_SHA256_MISMATCH'
    assert 'complete_GET_SHA256_verified' not in result


@pytest.mark.parametrize('data', [b'not json',b'[]',b'null',b'{"value":NaN}',b'{"value":Infinity}', b' '*(u.MAX_REPORT_BYTES+1), 'not bytes'])
def test_invalid_input_never_reaches_credentials_or_network(data):
    def forbidden(*args):pytest.fail('Invalid input reached network or credentials')
    result=u.publish(data,'/unused',_token=forbidden,_request=forbidden,emit=lambda e:None)
    assert result['status']=='UNAVAILABLE'
    assert 'complete_GET_SHA256_verified' not in result


def test_credential_in_report_refused_without_request():
    result=u.publish(json.dumps({'secret':TOKEN}).encode(),'/unused',
        _token=lambda *a:TOKEN,_request=lambda *a:pytest.fail('Secret report published'),emit=lambda e:None)
    assert result['reason_code']=='CREDENTIAL_IN_REPORT_REFUSED'


def test_missing_credentials_never_attempts_network():
    result=u.publish(b'{}','/unused',_token=lambda *a:None,
        _request=lambda *a:pytest.fail('Network called'),emit=lambda e:None)
    assert result['status']=='UNAVAILABLE'
    assert result['reason_code']=='CREDENTIAL_UNAVAILABLE'


def test_deadline_stops_new_requests():
    times=iter([0.,0.,0.,0.,100.,100.])
    result,_=publish(FakeGitHub(),_clock=lambda:next(times))
    assert result['status']=='RETRYING'
    assert result['reason_code']=='UPLOAD_DEADLINE'


def test_callback_failure_does_not_fail_upload():
    def broken(event):raise OSError('log unavailable')
    result=u.publish(b'{}','/unused',_token=lambda *a:TOKEN,_request=FakeGitHub(),emit=broken)
    assert result['status']=='DIAGNOSTIC_UPLOADED_VERIFIED'


def test_discovery_uses_only_token_key_and_does_not_change_files(tmp_path,monkeypatch):
    for key in ('PAPER3_GITHUB_TOKEN','GITHUB_TOKEN','GH_TOKEN','PAPER3_GITHUB_CREDENTIALS'):
        monkeypatch.delenv(key,raising=False)
    p=tmp_path/'.config/paper3/github.json';p.parent.mkdir(parents=True)
    raw=json.dumps({'token':TOKEN,'unused':'not copied'}).encode();p.write_bytes(raw)
    assert u._discover_token(tmp_path,None)==TOKEN
    assert p.read_bytes()==raw


def test_credential_environment_priority(monkeypatch,tmp_path):
    monkeypatch.setenv('PAPER3_GITHUB_TOKEN',TOKEN)
    assert u._discover_token(tmp_path,tmp_path/'missing.json')==TOKEN


def test_authenticated_redirect_refused_without_following():
    handler=u._NoRedirect()
    with pytest.raises(u.DeliveryError,match='REDIRECT_REFUSED'):
        handler.redirect_request(None,None,302,'moved',{},'https://evil.invalid/')


def test_unsafe_endpoint_is_refused_before_network():
    for endpoint in ('https://evil.invalid/','../other','/repos/evil/x'):
        with pytest.raises(u.DeliveryError):
            u._wire_request({'endpoint':endpoint,'token':TOKEN,'method':'GET','timeout':1})


def test_isolated_credential_worker_returns_only_private_pipe_result(tmp_path,monkeypatch):
    monkeypatch.setenv('PAPER3_GITHUB_TOKEN',TOKEN)
    assert u._token_provider(tmp_path,None,3)==TOKEN


def test_timeout_stops_only_the_newly_created_helper(monkeypatch):
    class Child:
        returncode=None
        killed=False
        calls=0
        def communicate(self,*args,**kwargs):
            self.calls+=1
            if self.calls==1:
                raise subprocess.TimeoutExpired('owned-private-helper',.01)
            return b'',None
        def kill(self):
            self.killed=True
    child=Child()
    def popen(args,**kwargs):
        assert args[-1]=='--private-worker'
        assert TOKEN not in repr(args)
        return child
    monkeypatch.setattr(u.subprocess,'Popen',popen)
    with pytest.raises(u.DeliveryError,match='UPLOAD_DEADLINE'):
        u._isolated({'operation':'request','token':TOKEN},.01)
    assert child.killed and child.calls==2


def test_no_training_or_sibling_process_api_in_source():
    source=(HERE/'diagnostic_upload.py').read_text()
    for forbidden in ('os.kill(', 'killpg(', 'scancel', 'sbatch', 'srun', 'import torch', 'import numpy'):
        assert forbidden not in source
