import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
def module(name):
    spec=importlib.util.spec_from_file_location('r9_test_'+name,RUNTIME/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
M=module('monitor')

class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.output=self.root/'runs/followup_20261025_r9';self.op=self.output/'runtime/three_jobs'
        self.op.mkdir(parents=True)
        self.put(self.output/'plan.json',{'recipe_sha256':'r9-recipe','tasks':[{'key':'first'},{'key':'second'}]})
        self.put(self.output/'first/result.json',{'status':'COMPLETE','recipe_sha256':'r9-recipe',
                 'summary':{'balanced_accuracy':.64},'predictions':['KEEP_LOCAL'],'token':'KEEP_LOCAL'})
        self.put(self.output/'second/progress.json',{'status':'EVALUATING','session':12,'sessions_total':105,
                 'optimizer_updates':14,'outer_update':14,'token':'KEEP_LOCAL'})
        self.put(self.op/'progress.json',{'round':9,'status':'RUNNING','complete_tasks':1,'total_tasks':2,
                 'tasks':[{'index':0,'key':'first','family':'core','status':'COMPLETE'},
                          {'index':1,'key':'second','family':'core','status':'RUNNING',
                           'progress_path':str(self.output/'second/progress.json')}],
                 'jobs':[{'job_id':'80001','slot':0},{'job_id':'80002','slot':1}],
                 'slots':{'0':{'status':'RUNNING','task_key':'second','job_id':'80001'}}})
    def tearDown(self):self.tmp.cleanup()
    def put(self,path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    def queue(self,*a,**k):
        return subprocess.CompletedProcess(a,0,
          '80001|paper3r9p-fixture|RUNNING|paper3r9p-fixture-s0-a1|hpc_gpu|1:00|gpu1|gpu1\n'
          '80002|paper3r9p-fixture|PENDING|paper3r9p-fixture-s1-a1|hpc_gpu|0:00|(AssocGrpGRES)|\n'
          '770043|paper3-direct-old|RUNNING|paper3-train-direct|hpc_gpu1|2-10:00|gpu4|gpu4\n','')
    def test_round9_resources_and_status_destination(self):
        s=M.collect(self.root,9,runner=self.queue)
        self.assertEqual(s['progress_branch'],'paper3-r9-progress')
        self.assertEqual(s['progress_path'],'monitoring/paper3-round9/three-jobs-status.json')
        self.assertEqual(s['requested_resources']['partition'],'hpc_gpu')
        self.assertEqual(s['requested_resources']['gpu_model_requested_by_group'],'A100-PCIe-40GB')
        self.assertEqual(s['requested_resources']['cpus_per_job'],16)
        self.assertEqual(s['requested_resources']['cpus_total'],48)
        self.assertEqual(s['requested_resources']['memory_total'],'192G')
        self.assertEqual(s['jobs'][1]['live_state'],'PENDING')
        self.assertEqual(len(s['live_queue']['jobs']),2)
        self.assertNotIn('KEEP_LOCAL',json.dumps(s))
        self.assertEqual(s['tasks'][1]['observations'][0]['values']['session'],12)
    def test_network_failure_does_not_change_training_files(self):
        before={p:p.read_bytes() for p in self.output.rglob('*.json')}
        class Broken:
            def publish(self,s):raise TimeoutError('DO_NOT_PRINT_CREDENTIAL')
        with mock.patch('sys.stdout',io.StringIO()) as output:
            snapshot,client,synced=M.monitor_cycle(self.root,9,self.op,client=Broken(),runner=self.queue)
        self.assertFalse(synced);self.assertNotIn('DO_NOT_PRINT_CREDENTIAL',output.getvalue())
        self.assertEqual(snapshot['status'],'RUNNING')
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)
    def test_publish_readme_has_current_round_resources(self):
        class Hub(M.ProgressGitHub):
            def __init__(self):
                self.round_number=9;self.branch,self.status_path,self.readme_path=M.destination(9)
                self.branch_ready=True;self.readme_ready=False;self.puts={}
            def put_file(self,path,data,message):self.puts[path]=data
        hub=Hub();hub.publish(M.collect(self.root,9,runner=self.queue))
        readme=hub.puts['monitoring/paper3-round9/THREE_JOBS_ZH.md'].decode()
        self.assertIn('16CPU/64G',readme);self.assertIn('48CPU/192G',readme)
        self.assertIn('A100-PCIe-40GB',readme)

if __name__=='__main__':unittest.main()
