import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHub:
    def __init__(self):
        self.files = {'results/paper3/round-02/previous/existing/status.json':b'{"keep":true}',
                      'results/paper3/round-07/final/existing/manifest.json':b'{"keep":true}'}
        self.calls = []
    def git_publish(self, files, message):
        self.calls.append((dict(files),message))
        self.files.update(files)
        return 'commit-%d'%len(self.calls)


class AutoSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = load_module('auto_summary_tested','auto_summary.py')
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'project';self.root.mkdir()
        self.op=self.root/'runs/training_campaign_parallel_v1';self.op.mkdir(parents=True)
        self.events=[]
        self.hub=FakeHub()
        self.credential=Path(self.tmp.name)/'private/github.json';self.credential.parent.mkdir()
        self.credential.write_text(json.dumps({'token':'UNIT_TEST_TOKEN_DO_NOT_PUBLISH','branch':'wrong-old-monitor'}))
        self.meta=self.put('runs/meta/m1/receipt.json',{'status':'META_TRAINING_COMPLETE',
            'method':'P','outer_updates_completed':1000,'summary':{'ba':.71,'n_participants':11},
            'participant_ids':['SECRET_PERSON'],'probabilities':[[.2,.8]],'token':'SECRET_IN_FILE'})
        self.put('runs/training_campaign_v1/meta_selections/P.json',{'status':'META_SELECTION_COMPLETE','configuration':{'dimension':4,'radius':.15}})
        self.put('runs/training_campaign_v1/selections/P.json',{'status':'DEVELOPMENT_CONFIGURATION_FROZEN','configuration':{'dimension':4,'radius':.15}})
        self.put('runs/training_campaign_v1/profiles/P/receipt.json',{'status':'CALIBRATION_PROFILE_COMPLETE','summary':{'seconds':.05}})
        self.put('runs/internal_validation_once/receipt.json',{'status':'ONE_TIME_INTERNAL_CHECK_COMPLETE','summary':{'ba':.7}})
        self.put('runs/source_extension_downstream/BASE4_PLUS_YANG/pre_internal_development_freeze.json',{'status':'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN','arm':'BASE4_PLUS_YANG'})
        self.snapshot={'status':'RUNNING','updated_utc':'2026-10-25T00:00:00Z','tasks':[
            {'key':'m1','kind':'meta_screening','method':'P','status':'COMPLETE','expected_receipt':'runs/meta/m1/receipt.json'}]}
        for number in (5,6):
            prefix='runs/followup_20261024_r%d/'%number
            self.put(prefix+'plan.json',{'recipe_sha256':'recipe%d'%number,'tasks':[
                {'key':'basis1','family':'basis','seed':20261018},{'key':'core1','family':'core','seed':20261018}]})
            self.put(prefix+'basis1/result.json',{'status':'COMPLETE','summary':{'ba':.71,'n_participants':11},
                'participant_id':'SECRET_PERSON','labels':[1,0],'predictions':[0,0]})
            self.put(prefix+'reports/report.json',{'status':'PARTIAL','complete_tasks':1,'total_tasks':2,
                'tables':{'primary_comparisons':[{'candidate':'a','baseline':'b','delta_ba_pp':.1,'p_holm':1.,'participant_deltas_ba':['SECRET_DELTA']}]}})
            csv=self.root/(prefix+'reports/primary_comparisons.csv');csv.parent.mkdir(parents=True,exist_ok=True)
            csv.write_text('candidate,baseline,delta_ba_pp,p_holm\na,b,0.1,1.0\n')
            # A full prediction file is intentionally outside the summary source set.
            self.put(prefix+'basis1/evaluations/session1.json',{'labels':['RAW_TRIAL_DO_NOT_READ'],'probabilities':[.9,.1]})
        self.source_before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
    def tearDown(self):self.tmp.cleanup()
    def put(self,relative,value):
        p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def emit(self,event,**fields):self.events.append((event,fields))
    def bundle(self,number=1,**kwargs):
        op=self.op if number==1 else self.root/('runs/followup_20261024_r%d/runtime/three_jobs'%number)
        op.mkdir(parents=True,exist_ok=True)
        return self.M.collect_summary(self.root,number,op,self.snapshot if number==1 else None,**kwargs)
    def assert_source_unchanged(self):
        for p,(data,mtime) in self.source_before.items():
            self.assertEqual(p.read_bytes(),data,p)
            self.assertEqual(p.stat().st_mtime_ns,mtime,p)
    def test_primary_receipts_and_existing_selections_are_summarized_without_trials(self):
        bundle=self.bundle(1)
        joined=b'\n'.join(bundle['files'].values()).decode()
        self.assertIn('META_TRAINING_COMPLETE',joined)
        self.assertIn('META_SELECTION_COMPLETE',joined)
        self.assertIn('SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN',joined)
        for secret in ('SECRET_PERSON','SECRET_IN_FILE','RAW_TRIAL_DO_NOT_READ','UNIT_TEST_TOKEN_DO_NOT_PUBLISH'):
            self.assertNotIn(secret,joined)
        self.assert_source_unchanged()
    def test_round56_existing_partial_reports_and_csv_are_kept_truthfully(self):
        for number in (5,6):
            bundle=self.bundle(number)
            keys=list(bundle['files'])
            self.assertTrue(any(k.endswith('plan.json') for k in keys))
            self.assertTrue(any(k.endswith('primary_comparisons.csv') for k in keys))
            reports=[json.loads(data) for k,data in bundle['files'].items() if k.endswith('report.json')]
            self.assertEqual(reports[0]['status'],'PARTIAL')
            self.assertNotIn('RAW_TRIAL_DO_NOT_READ',str(bundle['files']))
            self.assertNotIn('SECRET_PERSON',str(bundle['files']))
            self.assertNotIn('SECRET_DELTA',str(bundle['files']))
        self.assert_source_unchanged()
    def test_content_identity_ignores_heartbeat_and_mtime_only_changes(self):
        first=self.bundle(1)
        for p in self.source_before:
            st=p.stat();os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+10_000_000_000))
        self.snapshot['updated_utc']='2026-10-25T00:03:00Z'
        second=self.bundle(1)
        self.assertEqual(first['content_hash'],second['content_hash'])
    def test_only_summary_latest_is_written_and_prior_data_is_retained(self):
        before=dict(self.hub.files)
        bundle=self.bundle(1)
        self.M.publish_summary(bundle,1,self.hub,emit=self.emit)
        self.assertEqual(len(self.hub.calls),1)
        files=self.hub.calls[0][0]
        self.assertTrue(any(k.endswith('/AUTO_SUMMARY_LATEST.json') for k in files))
        self.assertFalse(any(Path(k).name=='LATEST.json' for k in files))
        self.assertTrue(all(k.startswith('results/paper3/round-01/') for k in files))
        for k,v in before.items():self.assertEqual(self.hub.files[k],v)
        pointer=json.loads(next(v for k,v in files.items() if k.endswith('/AUTO_SUMMARY_LATEST.json')))
        self.assertNotEqual(pointer.get('status'),'COMPLETE_OUTPUT_UPLOADED')
        self.assert_source_unchanged()
    def test_large_existing_report_has_locator_without_uploading_its_payload(self):
        path=self.root/'runs/followup_20261024_r5/reports/oversized.csv';path.write_text('x'*2048)
        bundle=self.bundle(5,max_file_bytes=1024)
        self.assertFalse(any(k.endswith('oversized.csv') for k in bundle['files']))
        self.assertIn('oversized.csv',json.dumps(bundle['manifest']))
    def test_daemon_single_flight_does_not_block_status_and_reuses_unchanged_summary(self):
        entered,release=threading.Event(),threading.Event()
        bundle=self.bundle(1)
        calls=[]
        def slow_collector(*args,**kwargs):
            entered.set();release.wait(3);return bundle
        dispatcher=self.M.SummaryDispatcher(self.root,1,self.op,credentials=self.credential,
            client_factory=lambda token:self.hub,emit=self.emit,collector=slow_collector)
        started=time.monotonic();self.assertTrue(dispatcher.kick(self.snapshot))
        self.assertLess(time.monotonic()-started,.2)
        self.assertTrue(entered.wait(1));self.assertTrue(dispatcher.thread.daemon)
        self.assertFalse(dispatcher.kick(self.snapshot))
        # A fresh status publish remains free to complete while summary I/O is blocked.
        calls.append('status-published');self.assertEqual(calls,['status-published'])
        release.set();dispatcher.thread.join(3);self.assertFalse(dispatcher.thread.is_alive())
        self.assertEqual(len(self.hub.calls),1)
        self.assertTrue(dispatcher.kick(self.snapshot));dispatcher.thread.join(3)
        self.assertEqual(len(self.hub.calls),1)
        self.assert_source_unchanged()
    def test_summary_network_failure_is_logged_without_propagating_to_training(self):
        class FailingHub(FakeHub):
            def git_publish(self,*args,**kwargs):raise TimeoutError('UNIT_TEST_TOKEN_DO_NOT_PUBLISH')
        dispatcher=self.M.SummaryDispatcher(self.root,1,self.op,credentials=self.credential,
            client_factory=lambda token:FailingHub(),emit=self.emit)
        self.assertTrue(dispatcher.kick(self.snapshot));dispatcher.thread.join(3)
        self.assertFalse(dispatcher.thread.is_alive())
        log=json.dumps(self.events)
        self.assertNotIn('UNIT_TEST_TOKEN_DO_NOT_PUBLISH',log)
        self.assert_source_unchanged()
    def test_real_status_loop_returns_while_summary_collection_is_blocked(self):
        monitor=load_module('monitor_summary_hook_tested','monitor.py')
        round56='round_number' in inspect.signature(monitor.run_monitor).parameters
        number=5 if round56 else 1
        op=self.root/'runs/followup_20261024_r5/runtime/three_jobs' if round56 else self.op
        op.mkdir(parents=True,exist_ok=True)
        bundle=self.bundle(number)
        entered,release=threading.Event(),threading.Event()
        def slow_collector(*a,**k):
            entered.set();release.wait(3);return bundle
        dispatcher=self.M.SummaryDispatcher(self.root,number,op,credentials=self.credential,
            client_factory=lambda token:self.hub,emit=self.emit,collector=slow_collector)
        calls=[]
        class StatusClient:
            def publish(self,snapshot):calls.append(snapshot)
        def queue(*a,**k):return mock.Mock(returncode=0,stdout='')
        started=time.monotonic()
        with mock.patch.object(monitor,'ProgressGitHub',return_value=StatusClient()),mock.patch('sys.stdout',io.StringIO()):
            if round56:
                result=monitor.run_monitor(self.root,number,op,credentials=self.credential,
                    once=True,runner=queue,summary_dispatcher=dispatcher)
            else:
                result=monitor.run_monitor(self.root,op,credentials=self.credential,
                    once=True,runner=queue,summary_dispatcher=dispatcher)
        self.assertLess(time.monotonic()-started,.3)
        self.assertEqual((result,len(calls)),(0,1))
        self.assertTrue(entered.wait(1));self.assertTrue(dispatcher.thread.is_alive())
        release.set();dispatcher.thread.join(3)
        self.assertEqual(len(self.hub.calls),1)


class ConcurrentGitTransportTests(unittest.TestCase):
    def test_summary_ref_conflict_rebases_on_other_round_instead_of_overwriting(self):
        T=load_module('summary_transport_tested','summary_transport.py')
        class RefHub(T.GitHub):
            def __init__(self):
                self.branch,self.repo,self.prefix='paper3-results','o/r','repos/o/r'
                self.emit=lambda *a,**k:None;self._delay=lambda *a,**k:None
                self.head,self.patches,self.bases='headA',0,[]
            def ensure_branch(self):return self.head
            def _blob(self,content):return hashlib.sha1(content).hexdigest()
            def api(self,path,body=None,method=None,allow404=False):
                if '/git/ref/' in path:return {'object':{'sha':self.head}}
                if '/git/commits/' in path:return {'tree':{'sha':{'headA':'treeA','headOther':'treeOther'}[path.rsplit('/',1)[-1]]}}
                if path.endswith('/git/trees'):
                    self.bases.append(body['base_tree']);self.entries=body['tree'];return {'sha':'treeNew'}
                if path.endswith('/git/commits'):return {'sha':'commit-'+body['parents'][0]}
                if method=='PATCH':
                    self.patches+=1
                    if self.patches==1:self.head='headOther';raise T.ApiError(422)
                    assert not body['force'];self.head=body['sha'];return {'object':{'sha':self.head}}
                raise AssertionError(path)
        hub=RefHub();sha=hub.git_publish({'results/paper3/round-05/AUTO_SUMMARY_LATEST.json':b'{}'},'summary')
        self.assertEqual(hub.bases,['treeA','treeOther'])
        self.assertEqual(sha,'commit-headOther')
        self.assertEqual(hub.entries[0]['path'],'results/paper3/round-05/AUTO_SUMMARY_LATEST.json')


if __name__=='__main__':unittest.main()
