import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from scheduler import ThreeJobScheduler, _atomic


class Clock:
    def __init__(self): self.value=1792890000
    def __call__(self): return self.value
    def advance(self,seconds): self.value+=seconds


class Slurm:
    def __init__(self):
        self.queue={}
        self.accounting={}
        self.calls=[]
        self.next_id=800001
        self.next_submit=None
        self.fail_queue=False
        self.fail_accounting=False
    def __call__(self,command,**kwargs):
        self.calls.append(command)
        if command[0]=='squeue':
            if self.fail_queue: return subprocess.CompletedProcess(command,1,'','temporarily unavailable')
            rows=['|'.join([str(j),r['marker'],r['state'],r['name'],r.get('reason','gpu1'),r.get('node','gpu1')])
                  for j,r in self.queue.items()]
            return subprocess.CompletedProcess(command,0,'\n'.join(rows),'')
        if command[0]=='sacct':
            if self.fail_accounting:return subprocess.CompletedProcess(command,1,'','temporary database lag')
            rows=['|'.join([str(j),r['state'],r.get('code','0:0'),r.get('marker',''),r['name']])
                  for j,r in self.accounting.items()]
            return subprocess.CompletedProcess(command,0,'\n'.join(rows),'')
        if command[0]=='sbatch':
            mode,self.next_submit=self.next_submit,None
            if mode=='qos':
                return subprocess.CompletedProcess(command,1,'','sbatch: error: QOSMaxSubmitJobPerUserLimit')
            if mode=='uncertain':
                raise subprocess.TimeoutExpired(command,45)
            name=next(s.split('=',1)[1] for s in command if s.startswith('--job-name='))
            marker=next(s.split('=',1)[1] for s in command if s.startswith('--comment='))
            job=self.next_id;self.next_id+=1
            self.queue[job]={'name':name,'marker':marker,'state':'PENDING','reason':'(Resources)','node':''}
            if mode=='lost_ack': raise subprocess.TimeoutExpired(command,45)
            return subprocess.CompletedProcess(command,0,str(job)+';school','')
        raise AssertionError(command)
    def finish(self,job,code=75,state='COMPLETED',comment=True):
        row=self.queue.pop(job)
        self.accounting[job]={**row,'state':state,'code':str(code)+':0'}
        if not comment:self.accounting[job]['marker']=''
    @property
    def submits(self): return [c for c in self.calls if c[0]=='sbatch']


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.runtime=Path(self.temp.name)/'runtime'
        self.clock=Clock();self.slurm=Slurm();self.events=[]
        self.launch={'worker_command':['/venv/python','-I','-S','-u','/package/worker.py',
                     '--root','/root/project','--slot','{slot}']}
        self.scheduler=ThreeJobScheduler(self.runtime,self.launch,self.slurm,self.clock,
                                        lambda event,**fields:self.events.append((event,fields)))
    def tearDown(self): self.temp.cleanup()
    def test_three_independent_jobs_exact_resources(self):
        summary=self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(summary['inflight_jobs'],3)
        for slot,command in enumerate(self.slurm.submits):
            for value in ['--partition=hpc_gpu','--nodes=1','--ntasks=1','--cpus-per-task=12','--gres=gpu:1']:
                self.assertIn(value,command)
            self.assertFalse(any(s.startswith('--exclude=') for s in command))
            self.assertIn('--mem='+['43G','43G','42G'][slot],command)
            self.assertNotIn('--array', ' '.join(command))
            self.assertNotIn('--hold',command)
            self.assertEqual(command[-1],str(slot))
        self.scheduler.step();self.assertEqual(len(self.slurm.submits),3)
        held=ThreeJobScheduler(Path(self.temp.name)/'held-runtime',
                               {**self.launch,'submit_hold':True},self.slurm,self.clock)
        held._load();attempt=held._new_attempt(0)
        self.assertTrue(attempt['held_requested'])
        self.assertIn('--hold',held.build_sbatch_command(0,attempt))
    def test_lost_ack_is_adopted_without_duplicate(self):
        self.slurm.next_submit='lost_ack'
        summary=self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(summary['controller_status'],'AWAITING_SUBMISSION_EVIDENCE')
        summary=self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(summary['jobs'][0]['job_id'],'800001')
    def test_unknown_intent_reserves_slot_indefinitely(self):
        self.slurm.next_submit='uncertain';self.scheduler.step()
        self.assertEqual(len(self.slurm.queue),2)
        for _ in range(3):
            self.clock.advance(10000);self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(self.scheduler.summary()['inflight_jobs'],3)
    def test_qos_rejection_retries_without_stopping_or_storm(self):
        self.slurm.next_submit='qos';summary=self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),1)
        self.assertEqual(summary['jobs'][0]['status'],'REJECTED')
        self.clock.advance(31);self.scheduler.step()
        self.assertEqual(len(self.slurm.queue),3)
        self.assertEqual(len(self.slurm.submits),4)
    def test_checkpoint_resume_with_delayed_accounting(self):
        self.scheduler.step();job=800001
        row=self.slurm.queue.pop(job)
        self.clock.advance(1000);self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.slurm.accounting[job]={**row,'state':'FAILED','code':'75:0'}
        self.scheduler.step();self.assertEqual(len(self.slurm.submits),3)
        self.clock.advance(31);self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),4)
        self.assertEqual(len(self.slurm.queue),3)
    def test_local_outcome_covers_accounting_lag_and_never_frees_live_job(self):
        self.scheduler.step();a=self.scheduler.state['slots']['0']['attempts'][-1]
        _atomic(self.runtime/'scheduler-outcomes'/'800001.json',{
            'slot':0,'job_id':'800001','exit_code':75,'name':a['name'],'marker':self.scheduler.marker})
        self.scheduler.step();self.assertEqual(len(self.slurm.submits),3)
        self.slurm.queue.pop(800001);self.scheduler.step()
        self.clock.advance(31);self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),4)
    def test_current_marker_only_never_counts_other_round_or_old_campaign(self):
        self.slurm.queue[770043]={'name':'paper3-train-direct','marker':'paper3-direct-old','state':'RUNNING'}
        self.slurm.queue[772011]={'name':'paper3r4-other-a2','marker':'paper3r4-other','state':'RUNNING'}
        self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(self.scheduler.summary()['inflight_jobs'],3)
        self.assertFalse(any(c[0]=='scancel' for c in self.slurm.calls))
    def test_null_account_comment_accepted_only_with_known_exact_name(self):
        self.scheduler.step();self.slurm.finish(800001,75,comment=False)
        self.scheduler.step();self.clock.advance(31);self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),4)
    def test_reduce_desired_workers_does_not_cancel_or_replace_slot_zero(self):
        self.scheduler.step()
        for job in list(self.slurm.queue):self.slurm.finish(job,0)
        summary=self.scheduler.step(keep_workers=[1,2])
        self.assertEqual(len(self.slurm.submits),5)
        self.assertEqual(summary['jobs'][0]['job_id'],'800001')
        self.assertEqual(summary['jobs'][0]['status'],'FINISHED')
        for job in list(self.slurm.queue):self.slurm.finish(job,0)
        self.assertEqual(self.scheduler.step(False)['controller_status'],'FINISHED')
        self.assertEqual(len(self.slurm.submits),5)
    def test_terminal_historical_jobs_are_not_readopted_into_new_attempts(self):
        self.scheduler.step();self.slurm.finish(800001,0)
        self.scheduler.step(keep_workers=[1,2])
        for _ in range(3):self.scheduler.step(keep_workers=[1,2])
        self.assertEqual(len(self.slurm.submits),3)
    def test_observation_failure_is_retry_and_does_not_interrupt_existing_workers(self):
        self.scheduler.step();self.slurm.fail_queue=True
        summary=self.scheduler.step()
        self.assertEqual(summary['controller_status'],'OBSERVATION_RETRY')
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(len(self.slurm.queue),3)
    def test_restart_recovers_durable_attempt_receipts(self):
        self.scheduler.step()
        self.scheduler.state_path.unlink()
        replacement=ThreeJobScheduler(self.runtime,self.launch,self.slurm,self.clock)
        replacement.step()
        self.assertEqual(len(self.slurm.submits),3)
        self.assertEqual(replacement.summary()['inflight_jobs'],3)
    def test_nested_lock_is_reentrant(self):
        with self.scheduler.acquire_lock():
            self.scheduler.step()
        self.assertEqual(len(self.slurm.submits),3)
    def test_batch_usr1_reaches_child_and_waits_for_checkpoint_exit(self):
        helper=Path(self.temp.name)/'checkpoint_child.py'
        ready=Path(self.temp.name)/'ready'
        checkpoint=Path(self.temp.name)/'saved-checkpoint'
        helper.write_text('''import pathlib,signal,sys,time
stopping=False
def stop(*args):
 global stopping
 stopping=True
signal.signal(signal.SIGUSR1,stop)
pathlib.Path(sys.argv[1]).write_text('ready')
while not stopping:time.sleep(.01)
pathlib.Path(sys.argv[2]).write_text('checkpoint_saved')
raise SystemExit(75)
''')
        process=subprocess.Popen([str(self.scheduler.batch_path),sys.executable,str(helper),str(ready),str(checkpoint)],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            while not ready.exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGUSR1)
            stdout,stderr=process.communicate(timeout=5)
            self.assertEqual(process.returncode,75,(stdout,stderr))
            self.assertEqual(checkpoint.read_text(),'checkpoint_saved')
        finally:
            if process.poll() is None:
                process.kill();process.wait()


if __name__=='__main__':unittest.main()
