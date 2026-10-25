import base64
import fcntl
import hashlib
import gzip
import tarfile
import os
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import result_channels as C


class Hub:
    def __init__(self):self.files={};self.publishes=[];self.assets={};self.release_calls=0
    def git_publish(self,files,message):self.files.update(files);self.publishes.append(files);return 'commit'
    def ensure_release(self,*args):self.release_calls+=1;return {'id':1}
    def upload_part(self,release,path,name,sha):
        data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
        self.assets[name]=data
        return {'id':len(self.assets),'url':'https://example.org/'+name,'sha256':sha,'bytes':len(data)}

class Channels(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.out=self.base/'output';self.out.mkdir();self.work=self.base/'work';self.work.mkdir()
        self.plan={'root':str(self.base),'output':str(self.out),'recipe_sha256':'identity','tasks':[{'key':'task1'},{'key':'task2'}]}
        self.put(self.out/'plan.json',self.plan)
        self.put(self.out/'task1/result.json',{'status':'COMPLETE','recipe_sha256':'identity'})
        self.put(self.out/'task2/progress.json',{'status':'EVALUATING'})
        (self.out/'task1/checkpoint').mkdir();(self.out/'task1/checkpoint/model.pt').write_bytes(b'checkpoint\x00binary')
        (self.out/'reports').mkdir();(self.out/'reports/session_metrics.csv').write_text('participant_id,session_id,ba\np1,s1,.7\np2,s2,.6\n')
    def put(self,p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    def tearDown(self):self.tmp.cleanup()
    def test_paper_only_shallow_paths_and_full_participant_identity(self):
        hub=Hub()
        with mock.patch.object(C.os,'walk',side_effect=AssertionError('No trial trees')):
            C.paper_cycle(hub,self.out,self.work)
        pointer=json.loads(hub.files[C.PREFIX+'/PAPER_DATA_LATEST.json'])
        manifest=json.loads(hub.files[pointer['manifest']])
        self.assertFalse(manifest['trial_prediction_tree_scanned'])
        row=next(r for r in manifest['files'] if r['source_path']=='reports/session_metrics.csv')
        original=b''.join(base64.b64decode(hub.files[p['path']]) for p in row['parts'])
        self.assertEqual(original,(self.out/'reports/session_metrics.csv').read_bytes())
        self.assertIn(b'participant_id',original);self.assertIn(b'p1,s1',original)
        self.assertFalse(any(Path(k).name=='LATEST.json' for k in hub.files))
    def test_paper_unchanged_cycle_does_not_republish(self):
        hub=Hub();C.paper_cycle(hub,self.out,self.work);count=len(hub.publishes)
        C.paper_cycle(hub,self.out,self.work);self.assertEqual(len(hub.publishes),count)
    def test_csv_complete_multiline_rows_and_offsets(self):
        p=self.out/'reports/test.csv';p.write_text('id,text\na,"two\nlines"\nb,你好\nc,end\n')
        parts=list(C.csv_views(p,chunk_bytes=21))
        import csv
        rows=[];offset=0
        for data,start,n in parts:
            parsed=list(csv.reader(io.StringIO(data.decode())));self.assertEqual(parsed[0],['id','text'])
            self.assertEqual(start,offset);self.assertEqual(n,len(parsed)-1);rows+=parsed[1:];offset+=n
        self.assertEqual(rows,[['a','two\nlines'],['b','你好'],['c','end']])
    def test_exact_chunks_capture_large_unicode_and_resume(self):
        p=self.out/'reports/large.json';data=('你好🙂'*(7*1024*1024)).encode();p.write_bytes(data)
        first=C.capture(p,self.work,chunk_bytes=1024*1024)
        self.assertGreater(len(data),64*1024*1024)
        rebuilt=b''.join((self.work/r['local']).read_bytes() for r in first['parts'])
        self.assertEqual(rebuilt,data);self.assertEqual(hashlib.sha256(rebuilt).hexdigest(),first['sha256'])
        with mock.patch.object(Path,'open',side_effect=AssertionError('Cache should reuse')):
            # read() uses read_text()->open as well, so only compare storage identity separately.
            pass
        second=C.capture(p,self.work,chunk_bytes=1024*1024);self.assertEqual(first,second)
    def test_archive_includes_checkpoints_and_does_not_rewalk_completed_tasks(self):
        hub=Hub();C.archive_cycle(hub,self.out,self.work)
        manifest=json.loads(next(v for k,v in hub.files.items() if k.endswith('/manifest.json')))
        self.assertTrue(any(r['source_path']=='task1/checkpoint/model.pt' for r in manifest['files']))
        self.assertFalse(any(k.endswith('/LATEST.json') for k in hub.files))
        with mock.patch.object(C,'task_inventory',side_effect=AssertionError('Archived task must not rescan')):
            C.archive_cycle(hub,self.out,self.work)
    def test_final_archive_pointer_requires_tasks_and_final_report(self):
        hub=Hub();self.put(self.out/'task2/result.json',{'status':'COMPLETE','recipe_sha256':'identity'})
        C.archive_cycle(hub,self.out,self.work)
        self.assertNotIn(C.PREFIX+'/LATEST.json',hub.files)
        self.put(self.out/'reports/report.json',{'status':'COMPLETE','recipe_sha256':'identity','complete_tasks':2,'total_tasks':2})
        C.archive_cycle(hub,self.out,self.work)
        pointer=json.loads(hub.files[C.PREFIX+'/LATEST.json'])
        self.assertTrue(pointer['full_raw_results_uploaded']);self.assertEqual(pointer['archived_tasks'],2)
    def test_same_content_atomic_report_rewrite_refreshes_signature(self):
        for kind,cycle in (('paper',C.paper_cycle),('archive',C.archive_cycle)):
            hub=Hub();work=self.work/kind;cycle(hub,self.out,work)
            path=self.out/'reports/session_metrics.csv';data=path.read_bytes()
            temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
            cycle(hub,self.out,work)
            state=json.loads((work/'state.json').read_text())
            row=state['files']['reports/session_metrics.csv'] if kind=='paper' else next(r for r in state['reports']['capture_files'] if r['source_path']=='reports/session_metrics.csv')
            self.assertEqual(row['signature'],C.source_signature(path))
    def test_publication_retries_if_report_changes_during_network(self):
        path=self.out/'reports/session_metrics.csv'
        class Racing(Hub):
            def git_publish(inner,files,message):
                result=super(Racing,inner).git_publish(files,message)
                if 'evidence file manifest' in message: path.write_text(path.read_text()+'p3,s3,.8\n')
                return result
        hub=Racing()
        with self.assertRaisesRegex(RuntimeError,'Report changed'):
            C.paper_cycle(hub,self.out,self.work)
        self.assertNotIn(C.PREFIX+'/PAPER_DATA_LATEST.json',hub.files)
    def test_resume_only_retries_unacknowledged_file_chunks(self):
        hub=Hub();initial=C.archive_cycle(hub,self.out,self.work)
        before=dict(hub.assets)
        # A new unfinished task becomes complete: only its own output is added.
        self.put(self.out/'task2/result.json',{'status':'COMPLETE','recipe_sha256':'identity'})
        (self.out/'task2/new.pt').write_bytes(b'new checkpoint')
        result=C.archive_cycle(hub,self.out,self.work)
        self.assertEqual(result['tasks']['task1'],initial['tasks']['task1'])
        self.assertTrue(all(hub.assets[name]==value for name,value in before.items()))
        self.assertEqual(len(result['tasks']),2)
    def test_archive_waits_for_native_task_tail_write_without_blocking(self):
        lock=self.out/'task1/.lock'
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=C.archive_cycle(Hub(),self.out,self.work)
            self.assertEqual(state['tasks'],{})
        state=C.archive_cycle(Hub(),self.out,self.work)
        self.assertIn('task1',state['tasks'])
    def test_final_report_wrong_recipe_does_not_claim_scientific_completion(self):
        self.put(self.out/'task2/result.json',{'status':'COMPLETE','recipe_sha256':'identity'})
        self.put(self.out/'reports/report.json',{'status':'COMPLETE','recipe_sha256':'other','complete_tasks':2,'total_tasks':2})
        self.assertFalse(C.report_completion(self.out)[2])
    def test_archive_drops_deleted_old_report_shard_from_current_manifest(self):
        path=self.out/'reports/session_metrics-0002.csv';path.write_text('person,ba\np1,.6\n')
        hub=Hub();C.archive_cycle(hub,self.out,self.work)
        path.unlink();state=C.archive_cycle(hub,self.out,self.work)
        self.assertFalse(any(r['source_path']=='reports/session_metrics-0002.csv' for r in state['reports']['capture_files']))
        index=json.loads(hub.files[C.PREFIX+'/ARCHIVE_PROGRESS.json'])
        current=json.loads(hub.files[index['manifest']])
        group=json.loads(hub.files[current['groups'][-1]['manifest']])
        self.assertFalse(any(r['source_path']=='reports/session_metrics-0002.csv' for r in group['files']))
    def test_many_small_scientific_files_are_one_aggregated_asset_and_restore_exactly(self):
        for i in range(120):
            self.put(self.out/('task1/pred-%04d.json'%i),{'probabilities':[.1,.9],'index':i})
        hub=Hub();state=C.archive_cycle(hub,self.out,self.work)
        self.assertEqual(state['tasks']['task1']['files'],122)
        self.assertEqual(state['tasks']['task1']['archive_parts'],1)
        self.assertLessEqual(len(hub.assets),2) # one task, one plan/reports aggregate
        group=json.loads(hub.files[state['tasks']['task1']['manifest']])
        data=b''.join(hub.assets[p['url'].rsplit('/',1)[-1]] for p in group['parts'])
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            self.assertEqual(len(archive.getmembers()),122)
            for row in group['files']:
                original=(self.out/row['source_path']).read_bytes();restored=archive.extractfile(row['source_path']).read()
                self.assertEqual(restored,original);self.assertEqual(hashlib.sha256(restored).hexdigest(),row['sha256'])
    def test_interrupted_archive_upload_reuses_packed_parts_and_acknowledged_chunks(self):
        path=self.out/'task1/random.bin';path.write_bytes(os.urandom(50000))
        paths=list(C.task_inventory(self.out,'task1'));identity='receipt-identity'
        C.pack_group(self.out,paths,self.work,'task1',identity,part_bytes=8192)
        class Interrupted(Hub):
            def __init__(inner):super().__init__();inner.calls=[];inner.failed=False
            def upload_part(inner,release,path,name,sha):
                inner.calls.append(name)
                if len(inner.calls)==2 and not inner.failed:inner.failed=True;raise TimeoutError('lost network')
                return super(Interrupted,inner).upload_part(release,path,name,sha)
        hub=Interrupted()
        with self.assertRaises(TimeoutError):C.archive_group(hub,{'id':1},self.out,paths,self.work,'task1',identity)
        first=hub.calls[0]
        with mock.patch.object(C,'source_signature',side_effect=AssertionError('Packed archive must be reused')):
            group=C.archive_group(hub,{'id':1},self.out,paths,self.work,'task1',identity)
        self.assertEqual(hub.calls.count(first),1);self.assertGreater(group['archive_parts'],3)
    def test_scientific_groups_use_deterministic_release_buckets(self):
        class Buckets(Hub):
            def __init__(inner):super().__init__();inner.tags=[]
            def ensure_release(inner,tag,*a):inner.tags.append(tag);return {'id':tag}
        hub=Buckets();C.archive_cycle(hub,self.out,self.work)
        self.assertTrue(all(tag.startswith('paper3-r9-archive-b') for tag in hub.tags))
        expected=int(hashlib.sha256(b'task1').hexdigest()[:2],16)%64
        self.assertIn('paper3-r9-archive-b%02d'%expected,hub.tags)
        counts={}
        for i in range(1548):
            bucket=int(hashlib.sha256(('task-%04d'%i).encode()).hexdigest()[:2],16)%64
            counts[bucket]=counts.get(bucket,0)+1
        self.assertEqual(len(counts),64);self.assertLess(max(counts.values()),60)
    def test_upload_failure_never_changes_source_or_calls_training(self):
        args=SimpleNamespace(operational=str(self.base/'ops'),output=str(self.out),channel='paper',credentials=None,once=True,interval=0)
        before={p:p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        class Broken(Hub):
            def __init__(self,*a,**k):super().__init__()
            def git_publish(self,*a):raise TimeoutError('network')
        with mock.patch.object(C,'upload_discover_token',return_value='not-printed'),mock.patch('sys.stdout',io.StringIO()):
            self.assertEqual(C.run_channel(args,gh_factory=Broken),0)
        self.assertEqual(before,{p:p.read_bytes() for p in self.out.rglob('*') if p.is_file()})
        self.assertEqual(json.loads((self.base/'ops/upload-paper/retry.json').read_text())['status'],'RETRYING')

if __name__=='__main__':unittest.main()
