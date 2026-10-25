import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import report_supplement as mod
from restore_report import restore


def silent(*args, **kwargs):
    pass


class FakeGitHub:
    branch = 'paper3-results'
    prefix = 'repos/jackzhu119/research3'
    def __init__(self, *, fail_at=None, cache=None):
        self.cache = cache if cache is not None else {}
        self.uploads = 0
        self.blob_calls = 0
        self.entries = None
        self.fail_at = fail_at
        self.emits = []

    def _blob(self, content):
        self.blob_calls += 1
        expected = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if expected not in self.cache:
            self.uploads += 1
            if self.fail_at == self.uploads:
                raise OSError('simulated lost network')
            self.cache[expected] = content
        return expected

    def emit(self, *args, **kwargs):
        self.emits.append((args, kwargs))

    def ensure_branch(self):
        return 'HEAD'

    def api(self, path, body=None, method=None):
        if '/git/ref/heads/' in path:
            return {'object': {'sha': 'HEAD'}}
        if '/git/commits/' in path:
            return {'tree': {'sha': 'BASE'}}
        if path.endswith('/git/trees'):
            self.entries = body['tree']
            return {'sha': 'TREE'}
        if path.endswith('/git/commits'):
            return {'sha': 'COMMIT'}
        if '/git/refs/heads/' in path:
            return {'object': {'sha': 'COMMIT'}}
        raise AssertionError(path)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'root'
        self.work = Path(self.tmp.name) / 'work'
        self.source = self.root / mod.SOURCE_REL
        self.source.parent.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, content):
        self.source.write_bytes(content)

    def test_unicode_chunks_exact_and_bounded(self):
        raw = ('{"中文":"😀中国\\\"\\n", "a":[{"b":1}]}').encode()
        parts = list(mod.utf8_chunks(io.BytesIO(raw), 8))
        self.assertEqual(b''.join(parts), raw)
        self.assertTrue(all(len(x) <= 11 for x in parts))
        for p in parts:
            p.decode()

    def test_report_over_64_mib_exact_reconstruction(self):
        # An actual >64MiB report, not a mocked size. Valid JSON and mixed UTF-8.
        unit = ('中文😀' * 1000).encode()
        with self.source.open('wb') as handle:
            handle.write(b'{"payload":"')
            for _ in range(6800):
                handle.write(unit)
            handle.write(b'","counts":[1,2,3]}')
        self.assertGreater(self.source.stat().st_size, 64 * 1024**2)
        cache, manifest = mod.prepare(self.root, self.work, notify=silent)
        output = Path(self.tmp.name) / 'restored.json'
        result = restore(manifest, output, lambda p: (cache / p['local_name']).read_bytes())
        digest = hashlib.sha256()
        with self.source.open('rb') as handle:
            for block in iter(lambda: handle.read(1024**2), b''):
                digest.update(block)
        self.assertEqual(result['sha256'], digest.hexdigest())
        self.assertEqual(result['bytes'], self.source.stat().st_size)
        self.assertEqual([p['key'] for p in manifest['structural_index']['fields']], ['payload', 'counts'])
        self.assertTrue(manifest['structural_index']['balanced_container_depth'])

    def test_unchanged_cache_and_source_is_read_only(self):
        self.write(b'{"x":[1,2,3]}')
        initial = self.source.read_bytes()
        cache, manifest = mod.prepare(self.root, self.work, chunk_bytes=8, notify=silent)
        with patch.object(mod, 'utf8_chunks', side_effect=AssertionError('reread source')):
            again, second = mod.prepare(self.root, self.work, chunk_bytes=8, notify=silent)
        self.assertEqual(again, cache)
        self.assertEqual(manifest, second)
        self.assertEqual(self.source.read_bytes(), initial)

    def test_source_change_does_not_create_complete_manifest(self):
        self.write(b'{"x": "' + b'a' * 100 + b'"}')
        changed = False
        def notify(event, **fields):
            nonlocal changed
            if event == 'R7_REPORT_COPY_PROGRESS' and not changed:
                changed = True
                with self.source.open('ab') as handle:
                    handle.write(b' ')
        with self.assertRaises(RuntimeError):
            mod.prepare(self.root, self.work, chunk_bytes=16, notify=notify)
        self.assertEqual(list(self.work.glob('source-cache/*/manifest.json')), [])
        self.assertFalse((self.work / 'R7-REPORT-JSON-UPLOADED.json').exists())

    def test_upload_resume_reuses_existing_blobs_and_final_pointer(self):
        self.write(json.dumps({'payload': list(range(1200))}).encode())
        cache, manifest = mod.prepare(self.root, self.work, chunk_bytes=1000, notify=silent)
        gh = FakeGitHub(fail_at=3)
        with self.assertRaises(OSError):
            mod.publish(gh, cache, manifest, self.work, notify=silent)
        self.assertEqual(len(gh.cache), 2)
        self.assertFalse((self.work / 'R7-REPORT-JSON-UPLOADED.json').exists())
        gh2 = FakeGitHub(cache=gh.cache)
        result = mod.publish(gh2, cache, manifest, self.work, notify=silent)
        self.assertEqual(gh2.uploads, len(manifest['parts']) - 2 + 4)
        self.assertTrue(result['all_original_report_bytes_uploaded'])
        paths = [e['path'] for e in gh2.entries]
        self.assertIn(mod.PREFIX + '/REPORT_JSON_LATEST.json', paths)
        self.assertNotIn(mod.PREFIX + '/PAPER_DATA_LATEST.json', paths)
        self.assertTrue(all('/round-07/' in p for p in paths))
        self.assertEqual(json.loads((self.work / 'status.json').read_text())['status'], 'COMPLETE')

    def test_corrupt_cache_never_published(self):
        self.write(b'{"a":"abcdefghij"}')
        cache, manifest = mod.prepare(self.root, self.work, chunk_bytes=8, notify=silent)
        (cache / manifest['parts'][0]['local_name']).write_bytes(b'corrupt')
        gh = FakeGitHub()
        with self.assertRaises(ValueError):
            mod.publish(gh, cache, manifest, self.work, notify=silent)
        self.assertIsNone(gh.entries)

    def test_restore_wrong_hash_preserves_prior_destination(self):
        self.write(b'{"a": 1}')
        cache, manifest = mod.prepare(self.root, self.work, notify=silent)
        output = Path(self.tmp.name) / 'existing.json'
        output.write_bytes(b'prior')
        with self.assertRaises(ValueError):
            restore(manifest, output, lambda p: b'wrong')
        self.assertEqual(output.read_bytes(), b'prior')

    def test_source_versions_get_distinct_caches(self):
        self.write(b'{"v":1}')
        first, m1 = mod.prepare(self.root, self.work, notify=silent)
        self.write(b'{"v":2}')
        second, m2 = mod.prepare(self.root, self.work, notify=silent)
        self.assertNotEqual(first, second)
        self.assertNotEqual(m1['source_sha256'], m2['source_sha256'])

    def test_prepare_never_scans_task_trees(self):
        self.write(b'{"summary":{"tasks":273},"details":[1,2]}')
        with patch.object(Path, 'glob', side_effect=AssertionError('glob forbidden')), \
             patch.object(Path, 'rglob', side_effect=AssertionError('rglob forbidden')), \
             patch.object(mod.os, 'walk', side_effect=AssertionError('walk forbidden')):
            cache, manifest = mod.prepare(self.root, self.work, notify=silent)
        self.assertEqual(manifest['source_path'], mod.SOURCE_REL)
        self.assertEqual(len(manifest['parts']), 1)


if __name__ == '__main__':
    unittest.main()
