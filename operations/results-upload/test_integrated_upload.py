import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import tarfile
import unittest


HERE = Path(__file__).resolve().parent
COMBINED = '\n\n'.join((HERE / name).read_text() for name in
    ('paper3_result_pack.py', 'paper3_result_transport.py', 'upload_main.py'))
NS = {'__name__': 'paper3_upload_integrated_test', '__file__': str(HERE/'combined.py')}
exec(compile(COMBINED, str(HERE/'combined.py'), 'exec'), NS)


class FakeGitHub:
    def __init__(self):
        self.files = {'results/paper3/round-02/previous/old/status.json': b'{"old": true}',
                      'results/paper3/round-05/previous/old/status.json': b'{"r5": true}'}
        self.releases = {}
        self.parts = {}
        self.events = []
        self.fail_next_part = False
        self.upload_posts = 0
        self.commits = []

    def ensure_release(self, tag, title, body):
        self.events.append(('release', tag))
        if tag not in self.releases:
            self.releases[tag] = {'id': len(self.releases)+1,
                                  'tag_name': tag, 'html_url': 'https://github.com/example/releases/'+tag}
        return self.releases[tag]

    def upload_part(self, release, path, name, sha256):
        self.events.append(('part', release['tag_name'], name))
        if self.fail_next_part:
            self.fail_next_part = False
            raise OSError('simulated interrupted upload')
        data = Path(path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha256
        key = release['tag_name'], name
        if key not in self.parts:
            self.upload_posts += 1
            self.parts[key] = data
        return {'id': list(self.parts).index(key)+1, 'url': 'https://github.com/example/assets/'+name}

    def git_publish(self, files, message):
        self.events.append(('git', tuple(sorted(files))))
        # Check ordering: there is no LATEST publication until every archive part
        # declared in this very transaction is already stored remotely.
        for path, data in files.items():
            if path.endswith('/archive.json'):
                archive = json.loads(data)
                for part in archive['parts']:
                    value = self.parts[(archive['release_tag'], part['name'])]
                    assert len(value) == part['bytes']
                    assert hashlib.sha256(value).hexdigest() == part['sha256']
        self.files.update(files)
        self.commits.append(message)
        return 'commit-%d'%len(self.commits)


def emit(*args, **kwargs):
    pass


def inventory(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


class IntegratedUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root, self.work = self.base/'source', self.base/'work'
        self.work.mkdir()
        self.events = []
        self.emit_orig = NS['upload_emit']
        NS['upload_emit'] = lambda event, **fields: self.events.append((event, fields))
        self.sources = {}
        rng = random.Random(123)
        for n in (2,3,7):
            out = self.root/'runs'/('followup_20261023_v2' if n==2 else 'followup_20261024_r%d'%n)
            out.mkdir(parents=True)
            tasks = [{'key': 'basis-legacy-seed1', 'family': 'basis', 'arm': 'core'},
                     {'key': 'core-legacy-seed1', 'family': 'core', 'arm': 'reference'}]
            (out/'plan.json').write_text(json.dumps({'output': str(out), 'recipe_sha256':'fixture%d'%n, 'tasks':tasks}))
            (out/'report.json').write_text(json.dumps({'status':'COMPLETE', 'tasks':2}))
            (out/'comparisons.csv').write_text('candidate,baseline,delta\na,b,0.01\n')
            for t in tasks:
                folder = out/t['key']
                folder.mkdir()
                (folder/'result.json').write_text(json.dumps({'status':'COMPLETE','label':[0,1], 'prediction':[0,1]}))
                (folder/'progress.json').write_text(json.dumps({'status':'COMPLETE','outer_update':1000}))
                (folder/'selected.json').write_text('{"score":0.75}')
                (folder/'checkpoint.pt').write_bytes(bytes(rng.randrange(256) for _ in range(9000)))
                (folder/'sessions').mkdir()
                (folder/'sessions'/'trial.json').write_text('{"predictions":[0,1],"labels":[0,1]}')
            (out/'runtime').mkdir()
            (out/'runtime'/'state.json').write_text('{"status":"FINISHED"}')
            (out/'private').mkdir()
            (out/'private'/'github.json').write_text('{"token":"SECRET_MUST_NEVER_UPLOAD"}')
            self.sources[n] = out
        self.source_before = inventory(self.root)

    def tearDown(self):
        NS['upload_emit'] = self.emit_orig
        self.temp.cleanup()

    def prepare(self,n):
        return NS['prepare_round'](self.root,n,self.work,emit,part_bytes=4096)

    def test_all_rounds_archive_reconstruct_and_source_is_read_only(self):
        hub = FakeGitHub()
        for n in (2,3,7):
            packed = self.prepare(n)
            self.assertEqual(packed['status'],'READY')
            self.assertEqual((packed['complete_tasks'],packed['total_tasks']),(2,2))
            self.assertGreater(len(packed['parts']),1)
            latest = NS['upload_remote_round'](hub,packed,self.work)
            self.assertEqual(latest['status'],'COMPLETE_OUTPUT_UPLOADED')
            path = 'results/paper3/round-%02d/LATEST.json'%n
            self.assertEqual(json.loads(hub.files[path])['snapshot_id'],packed['snapshot_id'])
            tag = 'paper3-results-r%02d-%s'%(n,packed['snapshot_id'])
            joined = b''.join(hub.parts[(tag,p['name'])] for p in packed['parts'])
            with tarfile.open(fileobj=io.BytesIO(joined),mode='r:gz') as archive:
                extracted = {m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
            self.assertEqual(len(extracted),len(packed['manifest']['files']))
            for record in packed['manifest']['files']:
                data = extracted[record['path']]
                self.assertEqual(len(data),record['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
                self.assertEqual(data,(self.sources[n]/record['path']).read_bytes())
            self.assertNotIn('runtime/state.json',extracted)
            self.assertNotIn('private/github.json',extracted)
            self.assertFalse(any(b'SECRET_MUST_NEVER_UPLOAD' in v for v in extracted.values()))
        self.assertIn('results/paper3/round-02/previous/old/status.json',hub.files)
        self.assertIn('results/paper3/round-05/previous/old/status.json',hub.files)
        self.assertEqual(self.source_before,inventory(self.root))

    def test_failed_part_never_publishes_latest_then_rerun_reuses_parts(self):
        packed = self.prepare(3)
        hub = FakeGitHub()
        hub.fail_next_part = True
        with self.assertRaises(OSError):
            NS['upload_remote_round'](hub,packed,self.work)
        self.assertNotIn('results/paper3/round-03/LATEST.json',hub.files)
        self.assertEqual(hub.commits,[])
        latest = NS['upload_remote_round'](hub,packed,self.work)
        self.assertEqual(latest['status'],'COMPLETE_OUTPUT_UPLOADED')
        posts = hub.upload_posts
        packed2 = self.prepare(3)
        self.assertTrue(packed2['reused'])
        self.assertEqual(packed2['snapshot_id'],packed['snapshot_id'])
        NS['upload_remote_round'](hub,packed2,self.work)
        self.assertEqual(hub.upload_posts,posts)
        self.assertEqual(len(hub.releases),1)
        self.assertIn('results/paper3/round-02/previous/old/status.json',hub.files)

    def test_incomplete_round_is_not_uploadable(self):
        (self.sources[7]/'core-legacy-seed1'/'result.json').write_text('{"status":"EVALUATING"}')
        packed = self.prepare(7)
        self.assertEqual(packed['status'],'PENDING')
        self.assertEqual((packed['complete_tasks'],packed['total_tasks']),(1,2))
        self.assertNotIn('parts',packed)


if __name__ == '__main__':
    unittest.main()
