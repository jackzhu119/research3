import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import urllib.error
from unittest import mock

SPEC = importlib.util.spec_from_file_location('r1_monitor', Path(__file__).with_name('monitor.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Response:
    def __init__(self, value):
        self.value = json.dumps(value).encode()
    def read(self): return self.value
    def __enter__(self): return self
    def __exit__(self, *args): pass


class QueueResult:
    returncode = 0
    stdout = ('9001|paper3r1-test|RUNNING|paper3r1-worker0|hpc_gpu1|1:00|gpu4|gpu4\n'
              '9002|paper3r1-test|PENDING|paper3r1-worker1|hpc_gpu1|0:00|(AssocGrpGRES)|\n'
              '9999|paper3r4-unrelated|RUNNING|paper3r4-unrelated|hpc_gpu|1:00|gpu2|gpu2\n')


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.op = self.root / M.DEFAULT_OPERATIONAL
        self.op.mkdir(parents=True)
        self.put('runs/training_campaign_v1/campaign_progress.json', {
            'status':'INTERRUPTED_PRESERVE_SAME_JOB', 'actions':[{'kind':'meta_screening'}]*32,
            'active':{'key':'old-stale'}, 'private':{'token':'NEVER_PUBLISH'}})
        self.put('runs/source_training_v2/source_training_progress.json', {
            'status':'NEURAL_SOURCE_COMPLETE_CLASSICAL_PENDING','actual_neural_fits_complete':15})
        self.put('runs/source_classical_v2/receipt.json', {'status':'SOURCE_CLASSICAL_CORE_COMPLETE'})
        self.put('runs/meta/m1/progress.json', {'status':'RUNNING','outer_updates_completed':777,
            'inner_updates_actual':31080,'participant_id':'do-not-publish','predictions':[9],
            'token':'NEVER_PUBLISH'})
        self.put(M.DEFAULT_OPERATIONAL+'/progress.json', {
            'status':'RUNNING','phase':'meta_screening','updated_utc':'2026-10-25T05:00:00Z',
            'completed_by_kind':{'meta_screening':32}, 'planned_counts':{'meta_screening':45},
            'tasks':[{'key':'m0','status':'REUSED_COMPLETE','kind':'meta_screening'},
                     {'key':'m1','status':'RUNNING','kind':'meta_screening','method':'P',
                      'job':{'dimension':4,'radius':.35,'seed':20261018,'token':'NEVER_PUBLISH'},
                      'expected_receipt':'runs/meta/m1/receipt.json','job_id':'9001','slot':0}],
            'jobs':[{'slot':0,'job_id':'9001','state':'RUNNING','cpus':12,'memory':'43G'},
                    {'slot':1,'job_id':'9002','state':'PENDING','cpus':12,'memory':'43G'}],
            'token':'NEVER_PUBLISH'})
        self.before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*.json')}
    def tearDown(self): self.temporary.cleanup()
    def put(self, relative, value):
        p=self.root/relative; p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def run_queue(self, command, **kwargs):
        self.assertEqual(command[0], 'squeue')
        self.assertEqual(kwargs['timeout'], 15)
        return QueueResult()
    def test_real_counters_and_queue_are_independent_of_old_campaign(self):
        snapshot=M.collect(self.root,runner=self.run_queue)
        self.assertEqual(snapshot['coordinator']['status'],'RUNNING')
        self.assertEqual(snapshot['parallel_plan_complete_tasks'],1)
        self.assertEqual(snapshot['completed_by_kind']['meta_screening'],32)
        self.assertEqual(snapshot['historical_campaign']['completed_stage_counts']['meta_screening'],32)
        task=next(r for r in snapshot['tasks'] if r['key']=='m1')
        self.assertEqual(task['observations'][0]['values']['outer_updates_completed'],777)
        self.assertEqual(snapshot['jobs'][1]['live_state'],'PENDING')
        self.assertEqual(snapshot['source_classical']['status'],'SOURCE_CLASSICAL_CORE_COMPLETE')
        self.assertEqual(len(snapshot['live_queue']['jobs']),2)
        self.assertFalse(snapshot['scientific_completion_claimed'])
    def test_no_sensitive_payload_or_other_round_data_published(self):
        payload=json.dumps(M.collect(self.root,runner=self.run_queue))
        for term in ('NEVER_PUBLISH','participant_id','"predictions":','paper3r4-unrelated'):
            self.assertNotIn(term,payload)
        for p,(data,mtime) in self.before.items():
            self.assertEqual(p.read_bytes(),data); self.assertEqual(p.stat().st_mtime_ns,mtime)
    def test_progress_path_cannot_read_private_or_outside_root(self):
        private=self.put('private/github.json',{'status':'TOKEN_PRIVATE','token':'NEVER_PUBLISH'})
        task={'progress_path':str(private),'expected_receipt':'../../outside/receipt.json'}
        self.assertEqual(M.progress_for_task(self.root,task),[])
    def test_queue_failure_is_observation_only(self):
        def runner(*a,**k): raise subprocess.TimeoutExpired('squeue',15)
        snapshot=M.collect(self.root,runner=runner)
        self.assertEqual(snapshot['live_queue']['status'],'UNAVAILABLE')
        self.assertEqual(snapshot['coordinator']['status'],'RUNNING')
    def test_actual_parallel_marker_is_observed_before_first_progress_write(self):
        (self.op/'progress.json').unlink()
        rows = ('772076|paper3r1p-84c2da292f724ce9|PENDING|paper3r1p-84c2da292f724ce9-s0-a1|hpc_gpu|0:00|(AssocGrpGRES)|\n'
                '772077|(null)|PENDING|paper3r1p-84c2da292f724ce9-s1-a1|hpc_gpu|0:00|(Resources)|\n'
                '772078|paper3r1-legacy|RUNNING|old-known-prefix|hpc_gpu|1:00|gpu2|gpu2\n'
                '772079|(null)|PENDING|paper3-r1-legacy|hpc_gpu|0:00|(Resources)|\n'
                '772080|paper3r5p-other|RUNNING|paper3r5p-other-s0-a1|hpc_gpu1|1:00|gpu4|gpu4\n')
        def runner(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 0, stdout=rows)
        snapshot = M.collect(self.root, runner=runner)
        self.assertEqual(snapshot['jobs'], [])
        self.assertEqual(snapshot['live_queue']['status'], 'OBSERVED')
        self.assertEqual([row['job_id'] for row in snapshot['live_queue']['jobs']],
                         ['772076', '772077', '772078', '772079'])
        self.assertEqual(snapshot['live_queue']['jobs'][0]['state'], 'PENDING')
        self.assertEqual(snapshot['live_queue']['jobs'][0]['reason_or_node'], '(AssocGrpGRES)')
    def test_bounded_json_and_missing_progress_do_not_scan_predictions(self):
        p=self.root/'small.json';p.write_bytes(b'x'*20)
        self.assertEqual(M.read_json(p,limit=10)[1],'JSON_TOO_LARGE')
        self.assertEqual(M.progress_for_task(self.root,{'expected_receipt':'absent/receipt.json'}),[])
    def test_transient_publish_failure_preserves_local_observation(self):
        class Client:
            def publish(self,snapshot): raise TimeoutError('SECRET_NEVER_LOG')
        out=io.StringIO()
        with mock.patch('sys.stdout',out):
            snapshot,client,ok=M.monitor_cycle(self.root,self.op,client=Client(),runner=self.run_queue)
        self.assertFalse(ok)
        self.assertNotIn('SECRET_NEVER_LOG',out.getvalue())
        self.assertEqual(json.loads((self.op/'monitor_snapshot.json').read_text())['coordinator']['status'],'RUNNING')
        for p,(data,mtime) in self.before.items():
            self.assertEqual(p.read_bytes(),data);self.assertEqual(p.stat().st_mtime_ns,mtime)
    def test_first_cycle_pushes_immediately_and_has_no_experiment_import(self):
        calls=[]
        class Client:
            def publish(self,snapshot): calls.append(snapshot)
        self.put('private/github.json',{'token':'unit-test-token'})
        with mock.patch.object(M,'ProgressGitHub',return_value=Client()),mock.patch('sys.stdout',io.StringIO()):
            M.run_monitor(self.root,credentials=self.root/'private/github.json',once=True,runner=self.run_queue,summary_dispatcher=False)
        self.assertEqual(len(calls),1)
        self.assertFalse(calls[0]['report_recomputed'])
        self.assertFalse(calls[0]['trial_predictions_read'])
    def test_fixed_destination_and_conflict_retry(self):
        class Hub(M.ProgressGitHub):
            def __init__(self):
                self.branch_ready=True;self.readme_ready=True;self.puts=[];self.lookups=0
            def api(self,path,value=None,method=None,allow404=False):
                self.asserted_paths=getattr(self,'asserted_paths',[])+[path]
                if method=='PUT':
                    self.puts.append((path,value))
                    if len(self.puts)==1:
                        raise urllib.error.HTTPError('https://api.github.com/mock',409,'Conflict',{},None)
                    return {'content':{'sha':'new'}}
                self.lookups+=1;return {'sha':'old%d'%self.lookups}
        hub=Hub();snapshot=M.collect(self.root,runner=self.run_queue);hub.publish(snapshot)
        self.assertEqual(len(hub.puts),2)
        self.assertEqual(hub.puts[-1][1]['branch'],'paper3-r1-progress')
        self.assertEqual(hub.puts[-1][0],'contents/monitoring/paper3-round1/status.json')
        decoded=json.loads(base64.b64decode(hub.puts[-1][1]['content']))
        self.assertEqual(decoded['round'],1)
    def test_mainline_completion_never_claims_two_extension_arms_finished(self):
        p=self.op/'progress.json';value=json.loads(p.read_text())
        value.update(mainline_complete=True,scientific_completion_claimed=True)
        value['tasks'] += [
            {'key':'init-yang','kind':'extension_initialize','arm':'BASE4_PLUS_YANG',
             'track':'extension','status':'COMPLETE'},
            {'key':'develop-yang','kind':'extension_development','arm':'BASE4_PLUS_YANG',
             'track':'extension','status':'RUNNING','expected_receipt':
                'runs/source_extension_downstream/BASE4_PLUS_YANG/pre_internal_development_freeze.json'},
            {'key':'init-dreyer','kind':'extension_initialize','arm':'BASE4_PLUS_DREYER',
             'track':'extension','status':'PENDING'}]
        p.write_text(json.dumps(value))
        base='runs/source_extension_downstream/BASE4_PLUS_YANG/'
        self.put(base+'catalog.json',{'source_f0':{'seed1':{},'seed2':{}},
            'meta':{'m0':{},'m1':{}},'validation':{},'token':'NEVER_PUBLISH'})
        self.put(base+'progress.json',{'status':'RUNNING','outer_updates_completed':444,'token':'NEVER_PUBLISH'})
        snapshot=M.collect(self.root,runner=self.run_queue)
        self.assertTrue(snapshot['mainline_complete'])
        self.assertFalse(snapshot['all_original_complete'])
        self.assertFalse(snapshot['scientific_completion_claimed'])
        self.assertEqual(snapshot['mainline']['observed_actions'],2)
        yang=snapshot['extension_arms']['BASE4_PLUS_YANG']
        self.assertEqual(yang['status'],'RUNNING')
        self.assertEqual(yang['native_completed_artifact_counts']['source_f0'],2)
        self.assertEqual(yang['native_completed_artifact_counts']['meta'],2)
        self.assertEqual(yang['completed_observed_actions'],1)
        self.assertEqual(yang['active_actions'][0]['track'],'extension')
        self.assertEqual(yang['active_actions'][0]['observations'][0]['values']['outer_updates_completed'],444)
        dreyer=snapshot['extension_arms']['BASE4_PLUS_DREYER']
        self.assertEqual(dreyer['status'],'PENDING_OR_PAUSED_ACTIONS')
        self.assertNotIn('NEVER_PUBLISH',json.dumps(snapshot))
    def test_new_target_is_hpc_gpu_and_old_partition_cannot_override_new_launch(self):
        old=self.root/'runs/training_campaign_v1/campaign_progress.json'
        old_value=json.loads(old.read_text());old_value['partition']='hpc_gpu1'
        old.write_text(json.dumps(old_value))
        first=M.collect(self.root,runner=self.run_queue)
        self.assertEqual(first['resources_target']['partition'],'hpc_gpu')
        self.assertEqual(first['resources_target']['gpu_model_requested_by_group'],'A100-PCIe-40GB')
        self.assertEqual(first['resources_target']['memory_total'],'128G')
        self.assertEqual(first['resources_target']['memory_per_job'],['43G','43G','42G'])
        p=self.op/'progress.json';v=json.loads(p.read_text())
        v.update(partition='hpc_gpu1',requested_resources={'partition':'hpc_gpu1'})
        p.write_text(json.dumps(v))
        self.put(M.DEFAULT_OPERATIONAL+'/launch.json',{'partition':'hpc_gpu'})
        current=M.collect(self.root,runner=self.run_queue)
        self.assertEqual(current['resources_target']['partition'],'hpc_gpu')
        self.assertEqual(current['resources_target']['configuration_source'],'new_launch')


if __name__=='__main__': unittest.main()
