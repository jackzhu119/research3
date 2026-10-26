import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_channel as channel
import paper3_result_pack as pack


class FakeGitHub:
    def __init__(self):
        self.files = {}
        self.parts = {}
        self.publishes = []

    def api(self, endpoint, allow404=False):
        path = endpoint.split('/contents/', 1)[1].split('?ref=', 1)[0]
        data = self.files.get(path)
        if data is None:
            return None
        blob = hashlib.sha1(('blob %d\0' % len(data)).encode()+data).hexdigest()
        return {'encoding': 'base64', 'content': base64.b64encode(data).decode(), 'sha': blob}

    def ensure_release(self, tag, title, body):
        return {'id': len(self.parts)+1, 'tag_name': tag,
                'html_url': 'https://example.test/releases/'+tag}

    def ensure_branch(self):
        return 'a'*40

    def upload_part(self, release, path, name, sha256):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise AssertionError('Part content SHA must match')
        self.parts[(release['tag_name'], name)] = data
        return {'id': len(self.parts), 'url': 'https://example.test/assets/'+name}

    def git_publish(self, files, message):
        # Completion indices must not appear until every declared archive part exists.
        for name, value in files.items():
            if name.endswith('/archive.json'):
                archive = json.loads(value)
                for part in archive['parts']:
                    self.assert_part(archive['release_tag'], part)
        self.files.update(files)
        self.publishes.append(message)
        return 'a'*40

    def assert_part(self, tag, part):
        actual = self.parts[(tag, part['name'])]
        if len(actual) != part['bytes'] or hashlib.sha256(actual).hexdigest() != part['sha256']:
            raise AssertionError('Published archive must reference verified parts')


class FullChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root, self.work, self.home = self.base/'project', self.base/'work', self.base/'home'
        self.root.mkdir()
        self.home.mkdir()
        self.events = []
        self.gh = FakeGitHub()

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def fixture(self, number, missing=False):
        out = channel.output_directory(self.root, number)
        out.mkdir(parents=True)
        tasks = [{'key': 'basis-fixture', 'family': 'basis'}, {'key': 'evaluation-fixture', 'family': 'core'}]
        plan = {'tasks': tasks, 'output': str(out), 'recipe_sha256': str(number)*64}
        (out/'plan.json').write_text(json.dumps(plan))
        for index, task in enumerate(tasks):
            p = out/task['key']
            p.mkdir()
            if not missing or index == 0:
                (p/'result.json').write_text(json.dumps({'status':'COMPLETE'}))
            (p/'predictions.csv').write_text('participant,trial,label,prediction\n1,1,0,1\n')
            (p/'checkpoint.pt').write_bytes(b'checkpoint-bytes'+bytes([number]))
        private = out/'basis-fixture/private'
        private.mkdir()
        (private/'github.json').write_text('must-not-be-read-or-uploaded')
        reports = out/'reports'
        reports.mkdir()
        (reports/'report.json').write_text(json.dumps({'status':'COMPLETE','total_tasks':2}))
        return out

    def worker(self, numbers, **kwargs):
        return channel.FullArchiveWorker(self.root, self.work, numbers, self.gh,
                                         self.emit, lock_home=self.home, **kwargs)

    def test_all_four_mappings_pack_and_restore_prediction_checkpoint_bytes(self):
        for number in (4,5,6,8):
            with self.subTest(round=number):
                out = self.fixture(number)
                before = {str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns)
                          for p in out.rglob('*') if p.is_file()}
                packed = pack.prepare_round(self.root,number,self.work,self.emit,part_bytes=1024)
                self.assertEqual(packed['status'],'READY')
                archive = b''.join(Path(p['path']).read_bytes() for p in packed['parts'])
                with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
                    self.assertEqual(tar.extractfile('evaluation-fixture/checkpoint.pt').read(),
                                     b'checkpoint-bytes'+bytes([number]))
                    self.assertEqual(tar.extractfile('evaluation-fixture/predictions.csv').read(),
                                     before['evaluation-fixture/predictions.csv'][0])
                    self.assertFalse(any('private' in name for name in tar.getnames()))
                after = {str(p.relative_to(out)):(p.read_bytes(),p.stat().st_mtime_ns)
                         for p in out.rglob('*') if p.is_file()}
                self.assertEqual(before,after)
                self.assertEqual(out.name,'followup_20261025_r8' if number==8 else 'followup_20261024_r%d'%number)

    def test_missing_declared_task_result_stays_pending_without_archive(self):
        for number in (4,5,6,8):
            self.fixture(number,missing=True)
            value=pack.prepare_round(self.root,number,self.work,self.emit)
            self.assertEqual(value['status'],'PENDING')
            self.assertEqual(value['complete_tasks'],1)
            self.assertEqual(value['total_tasks'],2)
            self.assertFalse((self.work/('round-%02d'%number)/'prepared.json').exists())

    def test_remote_complete_skips_before_any_recursive_scientific_scan(self):
        out=self.fixture(4)
        snapshot='c'*64
        prefix='results/paper3/round-04/final/'+snapshot
        latest={'round':4,'status':'COMPLETE_OUTPUT_UPLOADED','complete_tasks':2,'total_tasks':2,
                'snapshot_id':snapshot,'archive':prefix+'/archive.json'}
        archive={**latest,'status':'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED','parts':[{'name':'part'}]}
        self.gh.files.update({'results/paper3/round-04/LATEST.json':json.dumps(latest).encode(),
                             prefix+'/archive.json':json.dumps(archive).encode(),
                             prefix+'/files/plan.json':(out/'plan.json').read_bytes()})
        def forbidden(*args,**kwargs):raise AssertionError('No packing after complete remote pointer')
        with patch.object(pack.os,'walk',side_effect=forbidden):
            w=self.worker([4],prepare=forbidden,publish=forbidden)
            state=w.cycle()
        self.assertEqual(state['status'],'COMPLETE')
        self.assertEqual(state['completed_rounds'],[4])
        self.assertTrue(any(e=='ROUND_REMOTE_ALREADY_COMPLETE' for e,_ in self.events))

    def test_remote_pointer_matching_counts_but_wrong_plan_is_not_skipped(self):
        self.fixture(4)
        snapshot='d'*64;prefix='results/paper3/round-04/final/'+snapshot
        latest={'round':4,'status':'COMPLETE_OUTPUT_UPLOADED','complete_tasks':2,'total_tasks':2,
                'snapshot_id':snapshot,'archive':prefix+'/archive.json'}
        self.gh.files.update({'results/paper3/round-04/LATEST.json':json.dumps(latest).encode(),
            prefix+'/archive.json':json.dumps({**latest,'status':'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED','parts':[{}]}).encode(),
            prefix+'/files/plan.json':b'{"different":"plan"}'})
        w=self.worker([4]);w.cycle()
        self.assertEqual(w.todo,[])
        self.assertEqual(w.states['4'],'COMPLETE')
        self.assertEqual(len(self.gh.publishes),1)

    def test_active_legacy_r6_full_upload_does_not_block_other_rounds(self):
        for number in (4,5,6,8):self.fixture(number)
        legacy=self.root/'runs/followup_20261024_r6/runtime/three_jobs/result-channels/full/worker.lock'
        legacy.parent.mkdir(parents=True)
        with legacy.open('a+b') as h:
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            before=legacy.stat().st_mtime_ns
            w=self.worker([4,5,6,8]);state=w.cycle()
            self.assertEqual(state['completed_rounds'],[4,5,8])
            self.assertEqual(state['remaining_rounds'],[6])
            self.assertEqual(state['round_states']['6'],'WAITING_LEGACY_R6_FULL_UPLOAD')
            self.assertEqual(legacy.stat().st_mtime_ns,before)
            self.assertFalse((self.work/'round-06').exists())
        self.assertEqual(w.cycle()['status'],'COMPLETE')
        self.assertTrue((self.work/'round-06-UPLOADED.json').is_file())

    def test_global_per_round_lock_prevents_duplicate_different_workdirs(self):
        self.fixture(4)
        worker=self.worker([4])
        lock=worker.locks/'round-04-completion-export.lock'
        lock.parent.mkdir(parents=True)
        with lock.open('a+b') as h:
            fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=worker.cycle()
            self.assertEqual(state['remaining_rounds'],[4])
            self.assertEqual(state['round_states']['4'],'WAITING_OTHER_FULL_EXPORT')
            self.assertEqual(self.gh.publishes,[])
        self.assertEqual(worker.cycle()['status'],'COMPLETE')

    def test_completed_rounds_237_are_never_accepted(self):
        for number in (2,3,7):
            with self.assertRaises(ValueError):self.worker([number])
            with self.assertRaises(ValueError):pack.prepare_round(self.root,number,self.work,self.emit)

    def test_large_manifest_exact_byte_fragments_restore_without_git_size_block(self):
        files={}
        data=json.dumps({'files':['数据/'+str(n) for n in range(30)]},ensure_ascii=False).encode()
        storage=channel.store_manifest(files,'results/test',data,inline_limit=32,part_bytes=17)
        self.assertEqual(storage['storage'],'exact-byte-fragments')
        index=json.loads(files[storage['path']])
        restored=channel.restore_manifest_bytes(index,files.__getitem__)
        self.assertEqual(restored,data)
        self.assertEqual(hashlib.sha256(restored).hexdigest(),storage['sha256'])
        self.assertEqual(json.loads(restored)['files'],['数据/'+str(n) for n in range(30)])
        self.assertTrue(all(len(files[p['path']])<=17 for p in index['parts']))
        files[index['parts'][0]['path']]=b'corrupt'
        with self.assertRaises(ValueError):channel.restore_manifest_bytes(index,files.__getitem__)

    def test_publication_routes_large_manifest_locator_and_keeps_paper_link(self):
        self.fixture(8)
        packed=pack.prepare_round(self.root,8,self.work,self.emit,part_bytes=1024)
        original=channel.store_manifest
        def force_chunks(files,prefix,data):
            return original(files,prefix,data,inline_limit=32,part_bytes=128)
        with patch.object(channel,'store_manifest',new=force_chunks):
            latest=channel.upload_remote_round(self.gh,packed,self.work,self.emit)
        self.assertEqual(latest['manifest_storage'],'exact-byte-fragments')
        locator=json.loads(self.gh.files[latest['manifest']])
        manifest_bytes=channel.restore_manifest_bytes(locator,self.gh.files.__getitem__)
        self.assertEqual(json.loads(manifest_bytes),packed['manifest'])
        archive=json.loads(self.gh.files[latest['archive']])
        self.assertEqual(archive['manifest_path'],latest['manifest'])
        self.assertEqual(archive['manifest_sha256'],hashlib.sha256(manifest_bytes).hexdigest())
        self.assertEqual(archive['manifest_index_sha256'],hashlib.sha256(self.gh.files[latest['manifest']]).hexdigest())
        self.assertIn(b'PAPER_DATA_ZH.md',self.gh.files['results/paper3/round-08/README_ZH.md'])

    def test_finished_local_archive_is_reused_after_transient_upload_failure(self):
        self.fixture(5)
        attempts=[]
        def publisher(gh,packed,work):
            attempts.append(packed)
            if len(attempts)==1:raise OSError('transient')
            return channel.upload_remote_round(gh,packed,work,self.emit)
        w=self.worker([5],publish=publisher)
        self.assertEqual(w.cycle()['remaining_rounds'],[5])
        self.assertEqual(w.cycle()['status'],'COMPLETE')
        self.assertEqual(attempts[0]['snapshot_id'],attempts[1]['snapshot_id'])
        self.assertTrue(attempts[1]['reused'])
        self.assertTrue(any(e=='LOCAL_ARCHIVE_REUSED' for e,_ in self.events))

    def test_initial_credential_discovery_failure_retries_in_same_worker(self):
        import paper_credentials
        import paper3_result_transport
        sleeps=[]
        def no_real_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps)>1:raise KeyboardInterrupt('Unexpected repeated retry')
        fake_worker=__import__('unittest.mock',fromlist=['MagicMock']).MagicMock()
        fake_worker.return_value.run.return_value=0
        with patch.object(paper_credentials,'upload_discover_token',side_effect=[RuntimeError('not yet present'),'fixture-token']) as credentials, \
             patch.object(paper3_result_transport,'GitHub',return_value=self.gh) as github, \
             patch.object(channel,'FullArchiveWorker',new=fake_worker), \
             patch.object(channel.time,'sleep',side_effect=no_real_sleep), \
             patch.object(channel,'upload_emit') as emitted:
            result=channel.main(['--worker','--root',str(self.root),'--work-dir',str(self.work),'--rounds','4'])
        self.assertEqual(result,0)
        self.assertEqual(credentials.call_count,2)
        self.assertEqual(github.call_count,1)
        self.assertEqual(fake_worker.return_value.run.call_count,1)
        self.assertEqual(sleeps,[30])
        self.assertEqual(json.loads((self.work/'worker.json').read_text())['pid'],os.getpid())
        self.assertTrue(any(c.args[0]=='FULL_CHANNEL_INITIALIZATION_OR_AUTH_RETRY' for c in emitted.call_args_list))

    def test_auth_failure_refreshes_credentials_and_preserves_same_background_worker(self):
        import paper_credentials
        import paper3_result_transport
        self.fixture(4)
        class AuthError(Exception):
            status=401
        class ExpiredGitHub(FakeGitHub):
            def api(self,*args,**kwargs):raise AuthError('sensitive message must not be logged')
        expired=ExpiredGitHub()
        sleeps=[]
        def no_real_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps)>1:raise KeyboardInterrupt('Auth error was not refreshed')
        with patch.object(paper_credentials,'upload_discover_token',side_effect=['expired-fixture','fresh-fixture']) as credentials, \
             patch.object(paper3_result_transport,'GitHub',side_effect=[expired,self.gh]) as github, \
             patch.object(channel.time,'sleep',side_effect=no_real_sleep), \
             patch.object(channel.Path,'home',return_value=self.home), \
             patch.object(channel,'upload_emit') as emitted:
            result=channel.main(['--worker','--root',str(self.root),'--work-dir',str(self.work),'--rounds','4'])
        self.assertEqual(result,0)
        self.assertEqual(credentials.call_count,2)
        self.assertEqual(github.call_count,2)
        self.assertEqual(sleeps,[30])
        status=json.loads((self.work/'status.json').read_text())
        self.assertEqual(status['status'],'COMPLETE')
        self.assertEqual(status['completed_rounds'],[4])
        self.assertEqual(json.loads((self.work/'worker.json').read_text())['pid'],os.getpid())
        self.assertTrue((self.work/'round-04-UPLOADED.json').is_file())
        self.assertTrue(any(c.args[0]=='FULL_CHANNEL_AUTH_REFRESH_REQUIRED' for c in emitted.call_args_list))
        self.assertNotIn('sensitive message',str(emitted.call_args_list))


if __name__=='__main__':unittest.main()
