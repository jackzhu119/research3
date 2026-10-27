from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.request

import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/'experiments/round11_20261026'))
import network as n
from paper3r11.platform.uploads import FullVerifiedGitHub
from paper3r11.platform.vendor.lossless_publish import VerifiedGitHub
from paper3r11.platform.vendor.paper3_result_transport import ApiError


class Response(io.BytesIO):
    status=200


class FastStop:
    def __init__(self):self.delays=[]
    def is_set(self):return False
    def wait(self,value):self.delays.append(value);return False


def client(tmp_path,base=VerifiedGitHub,**kwargs):
    return n.make_bounded_client_class(base)('test-secret-token',lambda *a,**k:None,
        cache_path=tmp_path/'git-blobs.json',**kwargs)


def test_network_retry_is_bounded_and_raises_sanitized_retry(tmp_path):
    stop=FastStop();gh=client(tmp_path,stop=stop);calls=[]
    def offline(*args,**kwargs):
        calls.append(1);raise urllib.error.URLError('https://test-secret-token@example.invalid/private')
    gh._request=offline
    with pytest.raises(n.TransportRetryLater) as error:gh.api('repos/x/y')
    assert len(calls)==4 and len(stop.delays)==3
    assert 'test-secret-token' not in str(error.value)


def test_publish_cas_conflicts_have_bounded_non_retryable_escape(tmp_path):
    gh=client(tmp_path);calls=[]
    def conflict(url,data=None,method=None,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(url,409,'conflict',{},io.BytesIO(b'private body'))
    gh._request=conflict
    for _ in range(3):
        with pytest.raises(ApiError):gh.api('repos/x/y/git/refs/heads/paper3-results',{},'PATCH')
    with pytest.raises(n.TransportRetryLater):
        gh.api('repos/x/y/git/refs/heads/paper3-results',{},'PATCH')
    assert len(calls)==4


def test_cdn_redirect_strips_normal_and_unredirected_authorization():
    req=urllib.request.Request('https://api.github.com/releases/assets/1',
                               headers={'Authorization':'Bearer secret'})
    req.add_unredirected_header('Authorization','Bearer another-secret')
    redirected=n.SafeRedirect().redirect_request(req,None,302,'found',{},
        'https://release-assets.githubusercontent.com/asset?signature=secret')
    assert all(key.lower()!='authorization' for mapping in (redirected.headers,redirected.unredirected_hdrs)
               for key in mapping)


@pytest.mark.parametrize('url',[
    'http://release-assets.githubusercontent.com/asset',
    'https://githubusercontent.com.evil.invalid/asset',
    'https://example.com/asset'])
def test_unknown_or_insecure_redirect_is_rejected(url):
    req=urllib.request.Request('https://api.github.com/releases/assets/1')
    with pytest.raises(n.TransportRetryLater):n.SafeRedirect().redirect_request(req,None,302,'found',{},url)


def test_cross_host_write_redirect_is_rejected():
    req=urllib.request.Request('https://uploads.github.com/assets',data=b'bytes',method='POST')
    with pytest.raises(n.TransportRetryLater):
        n.SafeRedirect().redirect_request(req,None,302,'found',{},'https://release-assets.githubusercontent.com/asset')


def test_large_legacy_cache_is_streamed_bounded_and_preserved(tmp_path,monkeypatch):
    cache=tmp_path/'git-blobs.json'
    cache.write_text(json.dumps({format(i,'040x'):format(i,'040x') for i in range(18000)}))
    before=cache.read_bytes();original=Path.read_text
    def guarded(self,*args,**kwargs):
        if self==cache:pytest.fail('legacy cache was fully read_text loaded')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    gh=client(tmp_path)
    assert len(gh.blob_cache)==8192
    assert format(17999,'040x') in gh.blob_cache
    assert cache.read_bytes()==before


def test_binary_git_blob_full_remote_get_and_append_cache_round_trip(tmp_path):
    gh=client(tmp_path);data=b'\x00\xffnot UTF8\x01';expected=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    calls=[]
    def request(url,body=None,method=None,**kwargs):
        calls.append((url,method))
        if method=='POST':
            import base64
            assert base64.b64decode(json.loads(body)['content'])==data
            return Response(json.dumps({'sha':expected}).encode())
        assert kwargs['accept']=='application/vnd.github.raw+json'
        return Response(data)
    gh._request=request
    assert gh._blob(data)==expected
    assert gh._blob(data)==expected
    assert len(calls)==2
    assert not (tmp_path/'git-blobs.json').exists()
    journal=tmp_path/'git-blobs.json.recovery-v2.jsonl'
    assert json.loads(journal.read_text())['git_blob_sha']==expected
    restarted=client(tmp_path)
    assert expected in restarted.blob_cache
    assert 'test-secret-token' not in journal.read_text()


def test_full_asset_hash_requires_actual_complete_bytes_even_with_digest(tmp_path):
    gh=client(tmp_path,FullVerifiedGitHub);payload=b'actual archive bytes';calls=[]
    def request(*args,**kwargs):calls.append(1);return Response(payload)
    gh._request=request
    result=gh._asset_hash(dict(id=1,size=len(payload),digest='sha256:'+'0'*64))
    assert result==hashlib.sha256(payload).hexdigest() and calls==[1]
    with pytest.raises(n.TransportRetryLater):gh._asset_hash(dict(id=1,size=len(payload)+1))


def test_stop_interrupts_management_between_asset_chunks(tmp_path):
    stop=threading.Event();gh=client(tmp_path,FullVerifiedGitHub,stop=stop)
    class StopsOnRead(Response):
        def read(self,size=-1):value=super().read(size);stop.set();return value
    gh._request=lambda *a,**k:StopsOnRead(b'archive')
    with pytest.raises(n.TransportStopped):gh._asset_hash(dict(id=1,size=7))


def test_tls_verification_preserved_and_fixed_error_codes(tmp_path):
    gh=client(tmp_path)
    assert gh.context.check_hostname and gh.context.verify_mode==ssl.CERT_REQUIRED
    assert gh.timeout==120
    assert n.error_code(urllib.error.URLError(socket.gaierror('private host')))=='DNS_LOOKUP_FAILED'
    assert n.error_code(urllib.error.URLError(TimeoutError('private URL')))=='NETWORK_TIMEOUT'
    assert n.error_code(urllib.error.URLError(ssl.SSLCertVerificationError('private certificate')))=='TLS_CERTIFICATE_VERIFICATION_FAILED'


def test_broken_progress_sink_does_not_fail_transport(tmp_path):
    def progress(**kwargs):raise MemoryError('monitor only')
    gh=client(tmp_path,progress=progress)
    gh._notice(status='anything')
    assert gh._token=='test-secret-token'


def test_slow_school_asset_post_has_long_send_deadline_but_get_idle_is_bounded(tmp_path):
    gh=client(tmp_path);calls=[]
    class Opener:
        def open(self,request,timeout):calls.append((request.get_method(),timeout));return Response(b'{}')
    gh.opener=Opener()
    gh._request('https://uploads.github.com/repos/x/y/releases/1/assets',b'asset','POST',content_type='application/octet-stream')
    gh._request('https://api.github.com/repos/x/y/releases/assets/1',accept='application/octet-stream')
    assert calls==[('POST',900),('GET',120)]
