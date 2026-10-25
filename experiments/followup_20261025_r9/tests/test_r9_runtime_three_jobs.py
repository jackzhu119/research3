import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import three_jobs as runtime
import backend_entry


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.out=self.base/'output';self.out.mkdir()
        self.op=self.out/'runtime/three_jobs';self.op.mkdir(parents=True)
        self.plan={'root':str(self.base),'output':str(self.out),'recipe_sha256':'unchanged-recipe',
                   'tasks':[{'key':'basis','family':'basis','requires':[]},
                            {'key':'eval-a','family':'core','requires':['basis']},
                            {'key':'eval-b','family':'core','requires':['basis']}]}
        self.path=self.out/'plan.json';self.path.write_text(json.dumps(self.plan))
        self.original=self.path.read_bytes();self.original_mtime=self.path.stat().st_mtime_ns
    def tearDown(self):self.temp.cleanup()
    def complete(self,index):
        folder=self.out/self.plan['tasks'][index]['key'];folder.mkdir(exist_ok=True)
        (folder/'result.json').write_text(json.dumps({'status':'COMPLETE','recipe_sha256':'unchanged-recipe'}))
    def test_parallel_claims_are_unique_and_dependencies_wait(self):
        a=runtime.claim_task(self.plan,self.out,self.op,0,'700')
        self.assertEqual(a[0],0)
        self.assertIsNone(runtime.claim_task(self.plan,self.out,self.op,1,'701'))
        self.complete(0);runtime.finish_task(self.plan,self.out,self.op,0,0);a[1].close()
        b=runtime.claim_task(self.plan,self.out,self.op,1,'701')
        c=runtime.claim_task(self.plan,self.out,self.op,2,'702')
        self.assertEqual({b[0],c[0]},{1,2});b[1].close();c[1].close()
    def test_foreign_or_unidentified_receipt_is_not_reused(self):
        folder=self.out/'basis';folder.mkdir()
        for receipt in ({'status':'COMPLETE'}, {'status':'COMPLETE','recipe_sha256':'other'}):
            (folder/'result.json').write_text(json.dumps(receipt))
            self.assertFalse(runtime.result_complete(self.out,self.plan['tasks'][0],'unchanged-recipe'))
    def test_live_native_child_is_not_duplicated_after_parent_lease_loss(self):
        folder=self.out/'basis';folder.mkdir();handle=(folder/'.lock').open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertIsNone(runtime.claim_task(self.plan,self.out,self.op,1,'701'))
        handle.close();claim=runtime.claim_task(self.plan,self.out,self.op,1,'701')
        self.assertEqual(claim[0],0);claim[1].close()
    def test_completed_task_reused_and_source_plan_never_changed(self):
        self.complete(0)
        before=(self.out/'basis/result.json').stat().st_mtime_ns
        claim=runtime.claim_task(self.plan,self.out,self.op,1,'701');self.assertIn(claim[0],(1,2));claim[1].close()
        self.assertEqual(self.path.read_bytes(),self.original)
        self.assertEqual(self.path.stat().st_mtime_ns,self.original_mtime)
        self.assertEqual((self.out/'basis/result.json').stat().st_mtime_ns,before)
    def test_retry_preserves_task_and_checkpoint(self):
        folder=self.out/'basis';folder.mkdir();checkpoint=folder/'state-0.pt';checkpoint.write_bytes(b'original-state')
        claim=runtime.claim_task(self.plan,self.out,self.op,0,'700');claim[1].close()
        phase=runtime.finish_task(self.plan,self.out,self.op,0,2,clock=lambda:100)
        self.assertEqual(phase,'RETRYING');self.assertEqual(checkpoint.read_bytes(),b'original-state')
        self.assertIsNone(runtime.claim_task(self.plan,self.out,self.op,1,'701',clock=lambda:101))
        nxt=runtime.claim_task(self.plan,self.out,self.op,1,'701',clock=lambda:131)
        self.assertEqual(nxt[2],2);nxt[1].close()
    def test_actual_completed_result_overrides_stale_progress(self):
        self.complete(0)
        (self.out/'basis/progress.json').write_text('{"status":"TRAINING"}')
        value=runtime.publish_progress(self.path,self.op,round_number=9)
        self.assertEqual(value['complete_tasks'],1);self.assertEqual(value['tasks'][0]['status'],'COMPLETE')
        self.assertEqual(value['resources']['memory_total'],'192G')
    def test_single_job_gpu_mapping_keeps_allocated_device(self):
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'5','PAPER3R9_STOP_REQUESTED':'1'},clear=True):
            env=runtime.child_environment(9,2,'cuda:0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'5');self.assertEqual(env['PAPER3R9_GPU_SLOT'],'2');self.assertEqual(env['OMP_NUM_THREADS'],'16');self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertNotIn('PAPER3R9_STOP_REQUESTED',env)
        with mock.patch.dict(os.environ,{'SLURM_JOB_GPUS':'5-5'},clear=True):
            env=runtime.child_environment(9,0,'cuda:0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'5')
    def test_native_worker_cli_and_isolated_status_entry(self):
        command=runtime.worker_command('/school/python','/original/package',9,self.path,7)
        self.assertEqual(Path(command[5]).name,'backend_entry.py')
        self.assertIn('/original/package/paper3r9/entry.py',command)
        self.assertEqual(command[-2:],['--device','cuda:0'])
        r=subprocess.run([sys.executable,'-I','-S','-B',str(Path(runtime.__file__)),
            'status','--plan',str(self.path),'--round','9'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['total_tasks'],3)
    def test_supervisor_real_isolated_child_dag_and_reuses_checkpoint(self):
        package=self.base/'package';(package/'paper3r9').mkdir(parents=True)
        entry=package/'paper3r9/entry.py'
        entry.write_text('''import json,pathlib,sys
a=sys.argv;plan=json.loads(pathlib.Path(a[a.index("--plan")+1]).read_text());i=int(a[a.index("--task")+1])
out=pathlib.Path(plan["output"])/plan["tasks"][i]["key"];out.mkdir(exist_ok=True)
checkpoint=out/"state-0.pt"
if checkpoint.exists():assert checkpoint.read_bytes()==b"existing-native-state"
(out/"result.json").write_text(json.dumps({"status":"COMPLETE","recipe_sha256":plan["recipe_sha256"]}))
''')
        folder=self.out/'basis';folder.mkdir();checkpoint=folder/'state-0.pt';checkpoint.write_bytes(b'existing-native-state')
        before=checkpoint.stat().st_mtime_ns
        args=argparse.Namespace(plan=str(self.path),runtime=str(self.op),slot=0,round_number=9,
             package=str(package),python=sys.executable,device='cpu',poll_seconds=.01)
        with mock.patch.object(runtime,'emit',lambda *a,**k:None):
            self.assertEqual(runtime.worker_main(args),0)
        value=runtime.publish_progress(self.path,self.op,round_number=9)
        self.assertEqual(value['complete_tasks'],3);self.assertEqual(value['status'],'COMPLETE')
        self.assertEqual(checkpoint.read_bytes(),b'existing-native-state')
        self.assertEqual(checkpoint.stat().st_mtime_ns,before)
        self.assertEqual(self.path.read_bytes(),self.original)
    def test_existing_node_cache_reused_from_metadata_only(self):
        identity={'python':sys.version,'executable':sys.executable,'files':{'not-opened.so':'recorded-old-hash'}}
        sha=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        cache=self.base/'cache'/sha;(cache/'stdlib').mkdir(parents=True);(cache/'site').mkdir()
        (cache/'manifest.json').write_text(json.dumps(identity))
        receipt=self.base/'.paper3-hpc/jobs/769863/node_cache.json';receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps({'status':'NODE_LOCAL_CACHE_VERIFIED','cache_root':str(cache),
            'python_executable':sys.executable,'manifest_sha256':sha}))
        self.assertEqual(backend_entry.verified_cache_paths(self.base),[str(cache/'stdlib'),str(cache/'site')])
        self.assertEqual(backend_entry.verified_cache_paths(self.base,version='wrong-python'),[])


if __name__=='__main__':unittest.main()
