import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import urllib.error
from unittest import mock

SPEC=importlib.util.spec_from_file_location('r56_monitor',Path(__file__).with_name('monitor.py'))
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.ops={}
        for number in (5,6):
            out=self.root/('runs/followup_20261024_r%d'%number)
            op=out/'runtime/three_jobs';op.mkdir(parents=True);self.ops[number]=op
            self.put(out/'runtime/state.json',{'controller_status':'QUEUED_OLD_SINGLE_ALLOCATION','token':'NEVER_PUBLISH'})
            self.put(out/'task1/progress.json',{'status':'EVALUATING','session':16,'sessions_total':105,
                'outer_update':0,'budget':20,'labels':[0,1],'token':'NEVER_PUBLISH'})
            self.put(op/'progress.json',{'round':number,'recipe_sha256':'fixture%d'%number,
                'status':'RUNNING','phase':'TRAINING','updated_utc':'2026-10-25T06:00:00Z',
                'complete_tasks':1,'total_tasks':2,'tasks':[
                    {'index':0,'key':'task0','family':'basis','status':'COMPLETED','slot':None},
                    {'index':1,'key':'task1','family':'core','status':'RUNNING','slot':0,'job_id':'%d001'%number,
                     'progress_path':str(out/'task1/progress.json')}],
                'jobs':[{'slot':0,'job_id':'%d001'%number,'status':'RUNNING','cpus':12,'memory':'64G'},
                        {'slot':1,'job_id':'%d002'%number,'status':'PENDING','cpus':12,'memory':'64G'}],
                'slots':{'0':{'status':'RUNNING','task_key':'task1','worker_progress':
                    {'status':'EVALUATING','session':15,'sessions_total':105,'labels':[0,1]}}}})
        self.before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.rglob('*.json')}
    def tearDown(self):self.tmp.cleanup()
    def put(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def queue(self,command,**kwargs):
        self.assertEqual(command[0],'squeue');self.assertEqual(kwargs['timeout'],15)
        result=mock.Mock(returncode=0)
        result.stdout=('5001|paper3r5p-fixed|RUNNING|paper3r5p-slot0|hpc_gpu|1:00|gpu2|gpu2\n'
            '5002|paper3r5p-fixed|PENDING|paper3r5p-slot1|hpc_gpu|0:00|(AssocGrpGRES)|\n'
            '6001|paper3r6p-fixed|RUNNING|paper3r6p-slot0|hpc_gpu|1:00|gpu1|gpu1\n'
            '6002|paper3r6p-fixed|PENDING|paper3r6p-slot1|hpc_gpu|0:00|(Priority)|\n'
            '9999|paper3r5-OLD|PENDING|paper3r5-old|hpc_gpu|0:00|(AssocGrpGRES)|\n')
        return result
    def test_each_round_has_separate_new_path_and_fixed_resources(self):
        for n in (5,6):
            s=M.collect(self.root,n,runner=self.queue)
            self.assertEqual(s['experiment'],'paper3-round%d'%n)
            self.assertEqual(s['progress_branch'],'paper3-r%d-progress'%n)
            self.assertEqual(s['progress_path'],'monitoring/paper3-round%d/three-jobs-status.json'%n)
            self.assertEqual(s['requested_resources']['cpus_total'],36)
            self.assertEqual(s['requested_resources']['memory_total'],'192G')
            self.assertEqual(s['requested_resources']['partition'],'hpc_gpu1')
            self.assertEqual(s['allocation_mode'],'THREE_INDEPENDENT_SINGLE_GPU_JOBS')
    def test_real_updates_and_queue_are_new_stream_only(self):
        s=M.collect(self.root,5,runner=self.queue)
        self.assertEqual((s['complete_tasks'],s['total_tasks']),(1,2))
        self.assertEqual(s['families']['basis'],{'complete':1,'total':1})
        self.assertEqual(s['tasks'][1]['observations'][0]['values']['session'],16)
        self.assertEqual(s['gpu_slots']['0']['worker_progress']['session'],15)
        self.assertEqual(s['jobs'][1]['live_state'],'PENDING')
        self.assertEqual(len(s['live_queue']['jobs']),2)
        serialized=json.dumps(s)
        self.assertNotIn('paper3r5-OLD',serialized)
        self.assertNotIn('QUEUED_OLD_SINGLE_ALLOCATION',serialized)
        self.assertNotIn('NEVER_PUBLISH',serialized)
        self.assertNotIn('"labels":',serialized)
    def test_queue_failure_does_not_change_training_status(self):
        def fail(*a,**k):raise subprocess.TimeoutExpired('squeue',15)
        s=M.collect(self.root,6,runner=fail)
        self.assertEqual(s['status'],'RUNNING');self.assertEqual(s['live_queue']['status'],'UNAVAILABLE')
    def test_no_progress_is_waiting_not_old_allocation_state(self):
        (self.ops[5]/'progress.json').unlink()
        s=M.collect(self.root,5,runner=self.queue)
        self.assertEqual(s['status'],'WAITING_FOR_THREE_JOB_CONTROLLER')
        self.assertEqual(s['total_tasks'],0)
    def test_local_snapshot_and_network_failure_do_not_modify_source(self):
        class Client:
            def publish(self,s):raise TimeoutError('NEVER_PUBLISH')
        text=io.StringIO()
        with mock.patch('sys.stdout',text):
            s,c,ok=M.monitor_cycle(self.root,5,self.ops[5],client=Client(),runner=self.queue)
        self.assertFalse(ok);self.assertNotIn('NEVER_PUBLISH',text.getvalue())
        self.assertTrue((self.ops[5]/'monitor_snapshot.json').is_file())
        for path,(data,mtime) in self.before.items():
            self.assertEqual(path.read_bytes(),data);self.assertEqual(path.stat().st_mtime_ns,mtime)
    def test_first_cycle_is_immediate_and_lock_is_independent(self):
        calls=[]
        class Client:
            def publish(self,s):calls.append(s)
        cred=self.root/'private/github.json';self.put(cred,{'token':'fake-unit-test'})
        with mock.patch.object(M,'ProgressGitHub',return_value=Client()),mock.patch('sys.stdout',io.StringIO()):
            M.run_monitor(self.root,6,credentials=cred,once=True,runner=self.queue,summary_dispatcher=False)
        self.assertEqual(len(calls),1)
        self.assertTrue((self.ops[6]/'monitor.lock').is_file())
        self.assertFalse((self.ops[6].parent/'monitor.lock').exists())
    def test_publish_never_overwrites_legacy_status_or_other_round(self):
        class Hub(M.ProgressGitHub):
            def __init__(self,n):
                self.round_number=n;self.branch,self.status_path,self.readme_path=M.destination(n)
                self.branch_ready=True;self.readme_ready=False;self.puts=[]
            def put_file(self,path,data,message):self.puts.append((path,data))
        for n in (5,6):
            h=Hub(n);h.publish(M.collect(self.root,n,runner=self.queue))
            self.assertEqual(h.puts[0][0],'monitoring/paper3-round%d/three-jobs-status.json'%n)
            self.assertEqual(h.puts[1][0],'monitoring/paper3-round%d/THREE_JOBS_ZH.md'%n)
            self.assertNotEqual(h.puts[0][0],'monitoring/paper3-round%d/status.json'%n)
            with self.assertRaises(ValueError):h.publish({'round':11-n})
    def test_explicit_progress_paths_cannot_read_private_config(self):
        p=self.root/'private/github.json';self.put(p,{'status':'NEVER_PUBLISH'})
        self.assertEqual(M.progress_for_task(self.root,{'progress_path':str(p)}),[])
    def test_new_hpc_gpu1_target_ignores_old_launch_and_prefers_new_resources(self):
        for n in (5,6):
            op=self.ops[n]
            self.put(op.parent/'launch.json',{'partition':'hpc_gpu','slots':3})
            initial=M.collect(self.root,n,runner=self.queue)
            self.assertEqual(initial['requested_resources']['partition'],'hpc_gpu1')
            self.assertEqual(initial['requested_resources']['gpu_model_requested_by_group'],'A100-SXM4-80GB')
            p=op/'progress.json';value=json.loads(p.read_text())
            value.update(partition='hpc_gpu',resources={'partition':'hpc_gpu1'})
            p.write_text(json.dumps(value))
            nested=M.collect(self.root,n,runner=self.queue)
            self.assertEqual(nested['requested_resources']['partition'],'hpc_gpu1')
            value['resources']['partition']='hpc_gpu';p.write_text(json.dumps(value))
            self.put(op/'launch.json',{'partition':'hpc_gpu1'})
            fresh=M.collect(self.root,n,runner=self.queue)
            self.assertEqual(fresh['requested_resources']['partition'],'hpc_gpu1')
            self.assertEqual(fresh['requested_resources']['configuration_source'],'new_launch')


if __name__=='__main__':unittest.main()
