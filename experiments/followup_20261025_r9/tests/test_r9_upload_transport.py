import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request

from paper3_result_transport import GitHub, ApiError, _TransportRedirect


class PartHub(GitHub):
    def __init__(self, data, *, lost_ack=False, digest=True, existing=False, damaged=False):
        self.data, self.lost_ack, self.has_digest = data, lost_ack, digest
        self.assets, self.posts, self.downloads, self.deletes = [], 0, 0, []
        self.list_calls=0
        self._token = 'SECRET_NEVER_PRINT'
        self.repo, self.prefix, self.branch = 'o/r', 'repos/o/r', 'results'
        self.emit = lambda *a, **k: None
        self._delay = lambda *a, **k: None
        if existing:
            self.assets.append(self.asset(damaged))

    def asset(self, damaged=False):
        value = {'id': 1, 'name': 'part', 'size': len(self.data), 'state': 'uploaded',
                 'browser_download_url': 'https://github.com/x'}
        if self.has_digest:
            value['digest'] = 'sha256:' + ('0'*64 if damaged else hashlib.sha256(self.data).hexdigest())
        return value

    def api(self, path, body=None, method=None, allow404=False):
        if method == 'DELETE':
            self.deletes.append(path)
            self.assets.clear()
            return {}
        if '/assets?' in path:
            self.list_calls+=1
            return list(self.assets)
        raise AssertionError(path)

    def _request(self, url, data=None, method=None, **kwargs):
        if method == 'POST':
            self.posts += 1
            self.assets[:] = [self.asset()]
            if self.lost_ack:
                self.lost_ack = False
                raise urllib.error.URLError('Connection lost after durable POST')
            return io.BytesIO(json.dumps(self.assets[0]).encode())
        self.downloads += 1
        return io.BytesIO(self.data)


class RefHub(GitHub):
    def __init__(self):
        self.branch, self.repo, self.prefix = 'results', 'o/r', 'repos/o/r'
        self.emit = lambda *a, **k: None
        self._delay = lambda *a, **k: None
        self.head, self.patch_count, self.bases = 'headA', 0, []
        self.entries = None
    def ensure_branch(self): return self.head
    def _blob(self, content): return hashlib.sha1(content).hexdigest()
    def api(self, path, body=None, method=None, allow404=False):
        if '/git/ref/' in path:
            return {'object': {'sha': self.head}}
        if '/git/commits/' in path:
            return {'tree': {'sha': {'headA':'treeA', 'otherRoundHead':'treeOther'}[path.rsplit('/',1)[-1]]}}
        if path.endswith('/git/trees'):
            self.bases.append(body['base_tree'])
            self.entries = body['tree']
            return {'sha': 'treeNew'}
        if path.endswith('/git/commits'):
            return {'sha': 'newCommit' + body['parents'][0]}
        if method == 'PATCH':
            self.patch_count += 1
            if self.patch_count == 1:
                self.head = 'otherRoundHead'
                raise ApiError(422)
            assert body['force'] is False
            self.head = body['sha']
            return {'object': {'sha':self.head}}
        raise AssertionError(path)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'part'
        self.data = b'complete compressed scientific archive data' * 30
        self.path.write_bytes(self.data)
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.release = {'id': 7, 'upload_url': 'https://uploads.github.com/repos/o/r/releases/7/assets{?name,label}'}
    def tearDown(self): self.temp.cleanup()

    def test_lost_ack_rechecks_asset_and_never_reposts(self):
        hub = PartHub(self.data, lost_ack=True)
        result = hub.upload_part(self.release, self.path, 'part', self.sha)
        self.assertEqual(hub.posts, 1)
        self.assertTrue(result['reused'])
        self.assertEqual(hub.list_calls,2)

    def test_asset_cache_prevents_relisting_successful_parts(self):
        hub=PartHub(self.data)
        hub.upload_part(self.release,self.path,'part',self.sha)
        hub.upload_part(self.release,self.path,'part',self.sha)
        self.assertEqual(hub.posts,1);self.assertEqual(hub.list_calls,1)

    def test_no_server_digest_downloads_before_skip(self):
        hub = PartHub(self.data, digest=False, existing=True)
        result = hub.upload_part(self.release, self.path, 'part', self.sha)
        self.assertEqual((hub.posts, hub.downloads), (0, 1))
        self.assertTrue(result['reused'])

    def test_damaged_exact_asset_replaced_only_in_its_release(self):
        hub = PartHub(self.data, existing=True, damaged=True)
        hub.upload_part(self.release, self.path, 'part', self.sha)
        self.assertEqual(hub.deletes, ['repos/o/r/releases/assets/1'])
        self.assertEqual(hub.posts, 1)

    def test_local_hash_mismatch_prevents_upload(self):
        hub = PartHub(self.data)
        with self.assertRaises(ValueError):
            hub.upload_part(self.release, self.path, 'part', '0'*64)
        self.assertEqual(hub.posts, 0)

    def test_concurrent_other_round_is_preserved_by_rebase(self):
        hub = RefHub()
        result = hub.git_publish({'results/round02/LATEST.json': b'{}'}, 'Archive')
        self.assertEqual(hub.bases, ['treeA', 'treeOther'])
        self.assertEqual(result, 'newCommitotherRoundHead')
        self.assertEqual(len(hub.entries), 1)
        self.assertEqual(hub.entries[0]['path'], 'results/round02/LATEST.json')

    def test_redirect_strips_token_on_asset_host(self):
        request = urllib.request.Request('https://api.github.com/x', headers={'Authorization':'Bearer SECRET'})
        target = _TransportRedirect().redirect_request(request, None, 302, 'Moved', {},
            'https://release-assets.githubusercontent.com/result')
        self.assertFalse(any(k.lower()=='authorization' for k in target.headers))
        self.assertFalse(any(k.lower()=='authorization' for k in target.unredirected_hdrs))

    def test_permission_error_never_contains_token_or_body(self):
        hub = GitHub.__new__(GitHub)
        hub._token, hub.emit = 'SECRET_NEVER_PRINT', lambda *a, **k: None
        hub._delay = lambda *a, **k: self.fail('401 must not retry internally')
        def fail(*a, **k):
            raise urllib.error.HTTPError('https://api.github.com/x', 401, 'SECRET_NEVER_PRINT', {},
                io.BytesIO(b'{"message":"SECRET_NEVER_PRINT"}'))
        hub._request = fail
        with self.assertRaises(ApiError) as caught: hub.api('repos/o/r')
        self.assertEqual(caught.exception.status, 401)
        self.assertNotIn('SECRET', str(caught.exception))

    def test_429_is_retried_then_succeeds(self):
        hub = GitHub.__new__(GitHub)
        hub.emit = lambda *a, **k: None
        calls, delays = [], []
        hub._delay = lambda *a, **k: delays.append(a)
        def request(*a, **k):
            calls.append(1)
            if len(calls)==1:
                raise urllib.error.HTTPError('https://api.github.com/x',429,'Too many',{},io.BytesIO(b'{}'))
            return io.BytesIO(b'{"ok":true}')
        hub._request = request
        self.assertEqual(hub.api('repos/o/r'), {'ok':True})
        self.assertEqual((len(calls),len(delays)), (2,1))


if __name__ == '__main__': unittest.main()
