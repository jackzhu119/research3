"""Offline software fixtures for lossless report publication; no service calls."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

from transport import lossless_publish as mod


def silent(*args, **kwargs):
    pass


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


class RemoteState:
    def __init__(self):
        self.blobs = {}
        self.posts = {}
        self.full_gets = {}
        self.head = 'head-initial'
        self.trees = {'tree-initial': {'results/paper3/round-08/LATEST.json': 'full-archive-pointer'}}
        self.commits = {self.head: {'tree': {'sha': 'tree-initial'}, 'parents': []}}
        self.tree_batches = []
        self.fail_at = None
        self.post_attempts = 0
        self.corrupt_get = None
        self.concurrent_once = False


class RawResponse(io.BytesIO):
    status = 200
    def __init__(self, data):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


class FakeGitHub(mod.VerifiedGitHub):
    """Keep the real blob cache, verification and publisher over a local API double."""
    def __init__(self, state, cache_path):
        super().__init__('SOFTWARE_FIXTURE_SECRET', silent, repo='owner/repo',
                         branch='paper3-results', cache_path=cache_path)
        self.state = state
        self._delay = lambda *args, **kwargs: None

    def ensure_branch(self):
        return self.state.head

    def api(self, path, body=None, method=None, allow404=False):
        state = self.state
        if path.endswith('/git/blobs') and method == 'POST':
            data = base64.b64decode(body['content'], validate=True)
            sha = git_blob(data)
            state.post_attempts += 1
            if state.fail_at == state.post_attempts:
                raise OSError('SOFTWARE_FIXTURE_SECRET interruption')
            state.posts[sha] = state.posts.get(sha, 0) + 1
            state.blobs[sha] = data
            return {'sha': sha}
        if '/git/blobs/' in path:
            sha = path.rsplit('/', 1)[-1]
            if sha not in state.blobs:
                raise mod.ApiError(404)
            return {'sha': sha, 'encoding': 'base64',
                    'content': base64.b64encode(state.blobs[sha]).decode(),
                    'size': len(state.blobs[sha])}
        if '/git/ref/heads/' in path:
            return {'object': {'sha': state.head}}
        if path.endswith('/git/trees') and method == 'POST':
            entries = body['tree']
            state.tree_batches.append(entries)
            tree = dict(state.trees[body['base_tree']])
            tree.update({record['path']: record['sha'] for record in entries})
            sha = 'tree-' + hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
            state.trees[sha] = tree
            return {'sha': sha}
        if path.endswith('/git/commits') and method == 'POST':
            sha = 'commit-' + str(len(state.commits))
            state.commits[sha] = {'sha': sha, 'tree': {'sha': body['tree']},
                                  'parents': [{'sha': p} for p in body['parents']]}
            return {'sha': sha}
        if '/git/commits/' in path:
            return state.commits[path.rsplit('/', 1)[-1]]
        if '/git/refs/heads/' in path and method == 'PATCH':
            self.assert_nonforce(body)
            if state.concurrent_once:
                state.concurrent_once = False
                previous = state.commits[state.head]['tree']['sha']
                tree = dict(state.trees[previous])
                tree['results/paper3/round-08/full-channel-new.json'] = 'independent-full-channel'
                state.trees['tree-concurrent'] = tree
                state.commits['head-concurrent'] = {'tree': {'sha': 'tree-concurrent'},
                                                    'parents': [{'sha': state.head}]}
                state.head = 'head-concurrent'
                raise mod.ApiError(422)
            commit = state.commits[body['sha']]
            if commit['parents'][0]['sha'] != state.head:
                raise mod.ApiError(422)
            state.head = body['sha']
            return {'object': {'sha': state.head}}
        if '/contents/' in path:
            parsed = urllib.parse.urlsplit(path)
            relative = urllib.parse.unquote(parsed.path.split('/contents/', 1)[1])
            ref = urllib.parse.parse_qs(parsed.query)['ref'][0]
            tree = state.trees[state.commits[ref]['tree']['sha']]
            sha = tree[relative]
            return {'sha': sha, 'size': len(state.blobs[sha]), 'type': 'file'}
        raise AssertionError('Unexpected fake API route: ' + path)

    @staticmethod
    def assert_nonforce(body):
        if body.get('force') is not False:
            raise AssertionError('Result channels must never force-push')

    def _request(self, url, data=None, method=None, **kwargs):
        path = urllib.parse.urlsplit(url).path
        if '/git/blobs/' not in path or method not in (None, 'GET'):
            raise AssertionError('Only full blob GET is expected')
        sha = path.rsplit('/', 1)[-1]
        if sha not in self.state.blobs:
            raise mod.ApiError(404)
        block = self.state.blobs[sha]
        self.state.full_gets[sha] = self.state.full_gets.get(sha, 0) + 1
        if self.state.corrupt_get == sha:
            block = b'!' + block[1:]
        return RawResponse(block)


class LosslessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        self.cache = self.base / 'cache'
        self.work = self.base / 'work'

    def tearDown(self):
        self.temporary.cleanup()

    def stage(self, relative='reports/report.json', raw=b'{"value":1}', chunk_bytes=1024):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return mod.stage_file(self.source, relative, self.cache,
                              chunk_bytes=chunk_bytes, notify=silent)

    def assert_exact(self, record, original):
        result = hashlib.sha256()
        position = 0
        for number, part in enumerate(record['parts'], 1):
            data = Path(part['local_path']).read_bytes()
            self.assertEqual(part['number'], number)
            self.assertEqual(part['offset_bytes'], position)
            self.assertEqual(part['bytes'], len(data))
            self.assertEqual(part['sha256'], hashlib.sha256(data).hexdigest())
            data.decode('utf-8')
            result.update(data)
            position += len(data)
        self.assertEqual(position, original.stat().st_size)
        self.assertEqual(record['source_bytes'], position)
        self.assertEqual(record['source_sha256'], result.hexdigest())
        self.assertEqual(digest(original), result.hexdigest())

    def test_real_report_over_64_mib_restores_exact_multibyte_bytes(self):
        original = self.source / 'report.json'
        unit = ('中文😀' * 1024).encode()
        with original.open('wb') as handle:
            handle.write(b'{"payload":"')
            for _ in range(7200):
                handle.write(unit)
            handle.write(b'","counts":[1,2,3]}')
        self.assertGreater(original.stat().st_size, 64 * 1024**2)
        record = mod.stage_file(self.source, 'report.json', self.cache, notify=silent)
        self.assert_exact(record, original)
        self.assertTrue(all(p['bytes'] <= 1024**2 + 3 for p in record['parts']))

    def test_csv_keeps_bom_crlf_quoted_multiline_and_original_offsets(self):
        raw = b'\xef\xbb\xbfid,note\r\n1,"line1\r\nline2"\r\n2,"' + ('中文😀' * 20).encode() + b'"\r\n'
        record = self.stage('reports/session_metrics.csv', raw, chunk_bytes=17)
        self.assert_exact(record, self.source / 'reports/session_metrics.csv')
        self.assertEqual(b''.join(Path(p['local_path']).read_bytes() for p in record['parts']), raw)

    def test_no_recursive_source_scan(self):
        original = self.source / 'plan.json'
        original.write_bytes(b'{"tasks":[]}')
        with patch.object(Path, 'glob', side_effect=AssertionError('recursive discovery forbidden')), \
             patch.object(Path, 'rglob', side_effect=AssertionError('recursive discovery forbidden')), \
             patch.object(os, 'walk', side_effect=AssertionError('recursive discovery forbidden')):
            record = mod.stage_file(self.source, 'plan.json', self.cache, notify=silent)
        self.assert_exact(record, original)

    def test_symlink_source_or_parent_is_rejected(self):
        target = self.base / 'outside.json'
        target.write_bytes(b'{"outside":true}')
        (self.source / 'linked.json').symlink_to(target)
        with self.assertRaises((OSError, ValueError)):
            mod.stage_file(self.source, 'linked.json', self.cache, notify=silent)
        (self.source / 'reports').symlink_to(self.base, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            mod.stage_file(self.source, 'reports/outside.json', self.cache, notify=silent)

    def test_cached_parts_are_reused_without_reading_source_body(self):
        record = self.stage(raw=b'{"payload":"' + b'abcdefg' * 1000 + b'"}')
        expected_inode = (self.source / 'reports/report.json').stat().st_ino
        original_read = os.read
        original_fdopen = os.fdopen

        def no_source_read(fd, size):
            if os.fstat(fd).st_ino == expected_inode:
                raise AssertionError('stable cached source body was read again')
            return original_read(fd, size)

        def no_source_stream(fd, *args, **kwargs):
            # Buffered stream.read uses C I/O, bypassing the Python os.read name.
            if os.fstat(fd).st_ino == expected_inode:
                raise AssertionError('stable cached source entered the copy stream again')
            return original_fdopen(fd, *args, **kwargs)

        with patch.object(os, 'read', side_effect=no_source_read), \
             patch.object(os, 'fdopen', side_effect=no_source_stream):
            repeated = mod.stage_file(self.source, 'reports/report.json', self.cache,
                                      chunk_bytes=1024, notify=silent)
        self.assertEqual(record['source_sha256'], repeated['source_sha256'])
        self.assertEqual([p['local_path'] for p in record['parts']],
                         [p['local_path'] for p in repeated['parts']])

    def test_source_mutation_during_copy_cannot_create_complete_staging_manifest(self):
        original = self.source / 'report.json'
        original.write_bytes(b'{"payload":"' + b'abcdefg' * 100 + b'"}')
        changed = False

        def mutate(event, **fields):
            nonlocal changed
            if event == 'SOURCE_STAGE_PROGRESS' and not changed:
                changed = True
                with original.open('ab') as handle:
                    handle.write(b' ')

        with self.assertRaisesRegex(mod.ValidationError, 'source_changed_during_copy'):
            mod.stage_file(self.source, 'report.json', self.cache, chunk_bytes=31, notify=mutate)
        self.assertTrue(changed)
        self.assertEqual(list(self.cache.glob('*/manifest.json')), [])
        self.assertEqual(list(self.work.glob('*UPLOADED*.json')), [])

    def test_corrupt_staging_cache_is_repaired_from_unchanged_source(self):
        raw = b'{"payload":"' + b'abcdefg' * 100 + b'"}'
        record = self.stage(raw=raw, chunk_bytes=31)
        Path(record['parts'][0]['local_path']).write_bytes(b'corrupt')
        repaired = mod.stage_file(self.source, 'reports/report.json', self.cache,
                                  chunk_bytes=31, notify=silent)
        self.assert_exact(repaired, self.source / 'reports/report.json')
        self.assertEqual((self.source / 'reports/report.json').read_bytes(), raw)

    def test_private_paths_cannot_enter_a_snapshot(self):
        for relative in ('private/github.json', '.env', 'config/.env.production', '.aws/credentials'):
            with self.subTest(relative=relative):
                with self.assertRaises(mod.ValidationError):
                    mod.stage_file(self.source, relative, self.cache, notify=silent)

    def make_hub(self, state=None):
        self.work.mkdir(exist_ok=True)
        return FakeGitHub(state or RemoteState(), self.work / 'git-blobs.json')

    def test_two_channels_survive_nonforce_rebase_and_all_parts_get_verified(self):
        record = self.stage(raw=json.dumps({'rows': list(range(1500))}).encode(), chunk_bytes=100)
        state = RemoteState()
        state.concurrent_once = True
        gh = self.make_hub(state)
        result = mod.publish_snapshot(gh, 8, [record], self.work,
                                      metadata={'reports_complete': True, 'warnings': []})
        tree = state.trees[state.commits[result['commit']]['tree']['sha']]
        self.assertEqual(tree['results/paper3/round-08/LATEST.json'], 'full-archive-pointer')
        self.assertEqual(tree['results/paper3/round-08/full-channel-new.json'], 'independent-full-channel')
        self.assertIn('results/paper3/round-08/PAPER_DATA_LATEST.json', tree)
        self.assertTrue(all(len(batch) <= 32 for batch in state.tree_batches))
        for part in record['parts']:
            sha = git_blob(Path(part['local_path']).read_bytes())
            self.assertGreater(state.full_gets.get(sha, 0), 0)

    def test_upload_interruption_preserves_partial_blobs_without_complete_pointer(self):
        record = self.stage(raw=json.dumps({'rows': list(range(1500))}).encode(), chunk_bytes=100)
        state = RemoteState()
        state.fail_at = 35
        gh = self.make_hub(state)
        with self.assertRaises(OSError):
            mod.publish_snapshot(gh, 8, [record], self.work, metadata={})
        tree = state.trees[state.commits[state.head]['tree']['sha']]
        self.assertNotIn('results/paper3/round-08/PAPER_DATA_LATEST.json', tree)
        self.assertFalse(list(self.work.glob('*UPLOADED*.json')))
        prior_posts = dict(state.posts)
        state.fail_at = None
        result = mod.publish_snapshot(self.make_hub(state), 8, [record], self.work, metadata={})
        for sha, count in prior_posts.items():
            self.assertEqual(state.posts[sha], count)
        self.assertTrue(result['commit'])
        self.assertTrue(list(self.work.glob('*UPLOADED*.json')))

    def test_corrupt_local_part_cannot_publish_latest(self):
        record = self.stage(raw=b'{"payload":"' + b'abcd' * 1000 + b'"}')
        Path(record['parts'][0]['local_path']).write_bytes(b'damaged')
        state = RemoteState()
        with self.assertRaises(ValueError):
            mod.publish_snapshot(self.make_hub(state), 8, [record], self.work, metadata={})
        tree = state.trees[state.commits[state.head]['tree']['sha']]
        self.assertNotIn('results/paper3/round-08/PAPER_DATA_LATEST.json', tree)
        self.assertFalse(list(self.work.glob('*UPLOADED*.json')))

    def test_corrupt_remote_blob_cannot_publish_latest(self):
        record = self.stage(raw=b'{"payload":"original"}')
        state = RemoteState()
        state.corrupt_get = git_blob(Path(record['parts'][0]['local_path']).read_bytes())
        with self.assertRaises(ValueError):
            mod.publish_snapshot(self.make_hub(state), 8, [record], self.work, metadata={})
        tree = state.trees[state.commits[state.head]['tree']['sha']]
        self.assertNotIn('results/paper3/round-08/PAPER_DATA_LATEST.json', tree)
        self.assertFalse(list(self.work.glob('*UPLOADED*.json')))

    def test_persistent_blob_ack_cache_is_not_a_remote_get_proof(self):
        state = RemoteState()
        first = self.make_hub(state)
        data = b'original report bytes'
        sha = first._blob(data)
        self.assertEqual(state.posts[sha], 1)
        self.assertEqual(state.full_gets[sha], 1)
        second = self.make_hub(state)
        self.assertEqual(second._blob(data), sha)
        self.assertEqual(state.posts[sha], 1)
        self.assertEqual(state.full_gets[sha], 2)
        del state.blobs[sha]
        third = self.make_hub(state)
        self.assertEqual(third._blob(data), sha)
        self.assertEqual(state.posts[sha], 2)
        self.assertEqual(state.full_gets[sha], 3)

    def test_partial_http_body_never_counts_as_verified_blob(self):
        record = self.stage(raw=b'{"report":"complete source"}')
        state = RemoteState()
        gh = self.make_hub(state)
        original_request = gh._request

        def partial(*args, **kwargs):
            response = original_request(*args, **kwargs)
            response.status = 206
            return response

        gh._request = partial
        with self.assertRaisesRegex(mod.ValidationError, 'HTTP_200_required'):
            mod.publish_snapshot(gh, 8, [record], self.work, metadata={})
        tree = state.trees[state.commits[state.head]['tree']['sha']]
        self.assertNotIn('results/paper3/round-08/PAPER_DATA_LATEST.json', tree)
        self.assertFalse(list(self.work.glob('*UPLOADED*.json')))

    def test_required_missing_stays_partial_despite_successful_file_upload(self):
        record = self.stage('plan.json', b'{"tasks":[]}')
        state = RemoteState()
        receipt = mod.publish_snapshot(self.make_hub(state), 8, [record], self.work,
            metadata={'collection_status': 'PARTIAL', 'required_missing': ['reports/report.json']})
        self.assertEqual(receipt['status'], 'PAPER_DATA_PARTIAL_FILES_VERIFIED')
        self.assertFalse(receipt['scientific_completion_claimed'])
        self.assertFalse(receipt['full_raw_results_uploaded'])
        tree = state.trees[state.commits[receipt['commit']]['tree']['sha']]
        self.assertEqual(tree['results/paper3/round-08/LATEST.json'], 'full-archive-pointer')


if __name__ == '__main__':
    unittest.main()
