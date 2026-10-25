import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r1_worker_tested',HERE/'worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


FAKE_BRIDGE='''import json,os,pathlib,time
def action_complete(root,action):
 try:return json.loads((pathlib.Path(root)/action['expected_receipt']).read_text()).get('status')=='COMPLETE'
 except FileNotFoundError:return False
def perform_action(root,action,payload_manifest,**kwargs):
 root=pathlib.Path(root)
 with (root/'calls.log').open('a') as f:f.write(action['key']+'\\n')
 if action.get('mode')=='fail':raise ValueError('fake scientific failure remains visible')
 time.sleep(action.get('startup_delay',0))
 kwargs['on_ready']()
 (root/('ready-'+action['key'])).write_text('ready')
 if action.get('mode')=='hold':
  try:
   while True:time.sleep(.01)
  except KeyboardInterrupt:
   (root/'checkpoint-saved.json').write_text('{"update": 50}')
   raise
 time.sleep(.1)
 p=root/action['expected_receipt'];p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text('{"status":"COMPLETE"}')
 return {'status':'COMPLETE','same_original_action':True}
'''

FAKE_DIRECT='''import os
def reuse_node_cache(root,prior_job):return []
def child_environment(root,manifest,seconds,cache_paths=None):
 env=os.environ.copy();env.pop('PYTHONHOME',None);env['PYTHONUNBUFFERED']='1';return env
'''


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.root=self.base/'project';self.root.mkdir()
        self.runtime=self.root/'runs/training_campaign_parallel_v1';self.runtime.mkdir(parents=True)
        self.package=self.base/'package';self.package.mkdir()
        for name in ('worker.py','scheduler.py'):shutil.copyfile(HERE/name,self.package/name)
        (self.package/'kernel_bridge.py').write_text(FAKE_BRIDGE)
        self.direct=self.base/'old-direct';self.direct.mkdir();(self.direct/'direct_runner.py').write_text(FAKE_DIRECT)
        self.manifest=self.base/'payload_manifest.json';self.manifest.write_text('{}')
        self.processes=[];self.handles=[]
    def tearDown(self):
        for p in self.processes:
            if p.poll() is None:
                p.send_signal(signal.SIGUSR1)
                try:p.wait(timeout=4)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in self.handles:h.close()
        self.temp.cleanup()
    def action(self,key,**extra):
        return {'kind':'meta_screening','key':key,'expected_receipt':'results/'+key+'/receipt.json',
                'affinity_slot':None,'serial':False,**extra}
    def queue(self,actions,status='RUNNING'):
        worker.atomic_json(self.runtime/'queue.json',{'status':status,'ready_actions':actions,'phase':'FAKE'})
    def spawn(self,slot):
        env=os.environ.copy();env.update(SLURM_JOB_ID=str(9000+slot),PAPER3_R1_MARKER='fake-r1',
                                        PAPER3_R1_ATTEMPT_NAME='fake-r1-s%d-a1'%slot)
        log=self.base/('worker-%d.log'%slot);handle=log.open('wb');self.handles.append(handle)
        cmd=[sys.executable,'-I','-S','-B','-u',str(self.package/'worker.py'),'--root',str(self.root),
             '--runtime-dir',str(self.runtime),'--slot',str(slot),'--package',str(self.package),
             '--payload-manifest',str(self.manifest),'--direct-package',str(self.direct),
             '--allocation-seconds','600']
        process=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT)
        self.processes.append(process);return process,log
    def wait_for(self,predicate,timeout=8):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            if predicate():return
            time.sleep(.02)
        logs='\n'.join(p.read_text() for p in self.base.glob('worker-*.log'))
        self.fail('timed out\n'+logs)
    def test_three_processes_never_execute_one_action_twice(self):
        a=self.action('only-one');self.queue([a])
        processes=[self.spawn(i)[0] for i in range(3)]
        receipt=self.root/a['expected_receipt']
        self.wait_for(receipt.exists)
        self.queue([],status='COMPLETE')
        for p in processes:self.assertEqual(p.wait(timeout=8),0)
        self.assertEqual((self.root/'calls.log').read_text().splitlines(),['only-one'])
        self.assertEqual(worker.read_json(self.runtime/'claims.json',{}),{})
    def test_usr1_waits_for_ready_and_saves_original_checkpoint(self):
        a=self.action('holding',mode='hold');self.queue([a]);process,log=self.spawn(0)
        self.wait_for(lambda:'R1_KERNEL_READY_OBSERVED' in log.read_text())
        process.send_signal(signal.SIGUSR1)
        self.assertEqual(process.wait(timeout=8),75)
        self.assertEqual(json.loads((self.root/'checkpoint-saved.json').read_text())['update'],50)
        self.assertEqual(worker.read_json(self.runtime/'claims.json',{}),{})
        self.assertIn('R1_KERNEL_CHECKPOINT_REQUESTED',log.read_text())
        results=list((self.runtime/'action-outcomes').glob('*.json'))
        self.assertEqual(len(results),1)
        self.assertEqual(json.loads(results[0].read_text())['status'],'PAUSED')
    def test_usr1_during_startup_is_delayed_until_kernel_ready(self):
        a=self.action('slow-start',mode='hold',startup_delay=.5)
        self.queue([a]);process,log=self.spawn(0)
        self.wait_for((self.root/'calls.log').exists)
        process.send_signal(signal.SIGUSR1)
        self.assertEqual(process.wait(timeout=8),75)
        self.assertTrue((self.root/'checkpoint-saved.json').exists())
        text=log.read_text()
        self.assertLess(text.index('R1_KERNEL_READY_OBSERVED'),text.index('R1_KERNEL_CHECKPOINT_REQUESTED'))
    def test_stalled_progress_observer_does_not_block_stdout_or_child_exit(self):
        unblock=threading.Event();stop=threading.Event()
        script='import sys;print("KERNEL_READY drain-token",flush=True);sys.stdout.write("x"*1048576);sys.stdout.flush()'
        started=time.monotonic()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code=worker.run_child([sys.executable,'-u','-c',script],env=os.environ.copy(),
                    cwd=self.root,log=self.base/'drain.log',stop=stop,
                    observe=lambda pid:unblock.wait(10),token='drain-token',deadline=time.monotonic()+10)
            self.assertEqual(code,0)
            self.assertLess(time.monotonic()-started,3)
            self.assertGreaterEqual((self.base/'drain.log').stat().st_size,1048576)
        finally:
            unblock.set()
            if worker._observer_gate['thread'] is not None:worker._observer_gate['thread'].join(timeout=1)
    def test_one_scientific_error_is_preserved_and_next_action_still_runs(self):
        bad=self.action('bad',mode='fail');good=self.action('good')
        self.queue([bad,good]);process,log=self.spawn(0)
        self.wait_for((self.root/good['expected_receipt']).exists)
        self.queue([],status='COMPLETE');self.assertEqual(process.wait(timeout=8),0)
        self.assertEqual((self.root/'calls.log').read_text().splitlines(),['bad','good'])
        outcomes=[json.loads(p.read_text()) for p in (self.runtime/'action-outcomes').glob('*.json')]
        failure=next(x for x in outcomes if x['action_key']=='bad')
        self.assertEqual(failure['returned']['error_type'],'ValueError')
        self.assertIn('fake scientific failure',failure['returned']['message'])
        self.assertTrue(worker.read_json(self.runtime/'action-backoff.json')['bad']['retry_after']>time.time())
    def test_complete_action_is_not_reexecuted(self):
        a=self.action('old-complete');p=self.root/a['expected_receipt'];p.parent.mkdir(parents=True)
        p.write_text('{"status":"COMPLETE"}')
        before=p.stat().st_mtime_ns
        self.queue([a]);process,log=self.spawn(1)
        self.wait_for(lambda:(self.runtime/'worker-status/slot-1.json').exists())
        self.queue([],status='COMPLETE');self.assertEqual(process.wait(timeout=8),0)
        self.assertFalse((self.root/'calls.log').exists())
        self.assertEqual(p.stat().st_mtime_ns,before)
    def test_other_workers_claim_is_never_overwritten(self):
        a=self.action('claimed');self.queue([a]);existing={'claimed':{'slot':1,'job_id':'111','pid':222,
                                                     'token':'other-owner','claimed_utc':0}}
        worker.atomic_json(self.runtime/'claims.json',existing)
        fake=type('Bridge',(),{'action_complete':staticmethod(lambda r,a:False)})()
        value,reason=worker.claim_next(self.root,self.runtime,0,'9000',fake)
        self.assertIsNone(value)
        self.assertEqual(worker.read_json(self.runtime/'claims.json'),existing)
    def test_affinity_and_serial_claims_are_honored(self):
        only_slot1=self.action('only-slot1',affinity_slot=1)
        serial=self.action('serial',serial=True)
        self.queue([only_slot1,serial])
        fake=type('Bridge',(),{'action_complete':staticmethod(lambda r,a:False)})()
        value,_=worker.claim_next(self.root,self.runtime,1,'9001',fake)
        self.assertEqual(value[0]['key'],'only-slot1')
        other,reason=worker.claim_next(self.root,self.runtime,0,'9000',fake)
        self.assertIsNone(other)
        action,claim,lock=value;worker.release_claim(self.runtime,action,claim);lock.close()
        value,_=worker.claim_next(self.root,self.runtime,0,'9000',fake)
        self.assertEqual(value[0]['key'],'serial')
        other,_=worker.claim_next(self.root,self.runtime,1,'9001',fake);self.assertIsNone(other)
        action,claim,lock=value;worker.release_claim(self.runtime,action,claim);lock.close()


if __name__=='__main__':unittest.main()
