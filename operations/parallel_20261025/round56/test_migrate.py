import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import fcntl

from migrate import HeldQueueMigration,MigrationOwnershipError,MigrationTransient


class Scheduler:
    def __init__(self,round_number,old_marker,new_marker):
        self.round=round_number;self.old_marker=old_marker;self.new_marker=new_marker
        self.old_job={5:'772012',6:'772013'}[round_number]
        self.old_state='PENDING';self.old_terminal=False;self.race=False
        self.bad_memory=False;self.queue_failure=False;self.calls=[]
        self.new={str(900001+s):dict(slot=s,held=True,state='PENDING') for s in range(3)}
    def __call__(self,command,**kwargs):
        self.calls.append(command)
        if command[0]=='squeue':
            if self.queue_failure:return SimpleNamespace(returncode=1,stdout='',stderr='controller temporarily down')
            rows=[]
            if not self.old_terminal:
                rows.append(f'{self.old_job}|u|{self.old_marker}|{self.old_state}|{self.old_marker}-a2|hpc_gpu|(AssocGrpGRES)')
            for job,row in self.new.items():
                reason='(JobHeldUser)' if row['held'] else '(AssocGrpGRES)'
                rows.append(f"{job}|u|{self.new_marker}|{row['state']}|{self.new_marker}-s{row['slot']}-a1|hpc_gpu1|{reason}")
            # Foreign round/application must never be signaled or released.
            rows.append('800000|u|paper2-other|PENDING|paper2|hpc_gpu|(AssocGrpGRES)')
            return SimpleNamespace(returncode=0,stdout='\n'.join(rows)+'\n',stderr='')
        if command[0]=='sacct':
            state='CANCELLED' if self.old_terminal else self.old_state
            return SimpleNamespace(returncode=0,stdout=f'{self.old_job}|u|{state}|0:0|{self.old_marker}-a2|hpc_gpu|{self.old_marker}\n',stderr='')
        if command[0]=='scontrol' and '--oneliner' in command:
            job=command[-1];row=self.new[job];memory='32G' if self.bad_memory else '64G'
            reason='JobHeldUser' if row['held'] else 'AssocGrpGRES';priority=0 if row['held'] else 100
            text=(f"JobId={job} JobName={self.new_marker}-s{row['slot']}-a1 UserId=u(1000) "
                  f"JobState={row['state']} Reason={reason} Priority={priority} Comment={self.new_marker} "
                  f"Partition=hpc_gpu1 NumCPUs=12 NumNodes=1 TRES=cpu=12,mem={memory},node=1,gres/gpu=1")
            return SimpleNamespace(returncode=0,stdout=text,stderr='')
        if command[0]=='scancel':
            assert command[-1]==self.old_job
            if '--state=PENDING' in command:
                if self.race:self.old_state='RUNNING';self.race=False
                else:self.old_terminal=True
            elif '--signal=USR1' in command:self.old_terminal=True
            else:raise AssertionError('Unconditional cancellation!')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if command[:2]==['scontrol','release']:
            self.new[command[-1]]['held']=False
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        raise AssertionError(command)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.output=self.base/'round5';self.output.mkdir()
        self.plan={'output':str(self.output),'root':str(self.base),'recipe_sha256':'a'*64,
                   'tasks':[{'key':'already-complete'},{'key':'unfinished'}],'prepared_at':1}
        self.path=self.output/'plan.json';self.path.write_text(json.dumps(self.plan))
        self.original_plan=self.path.read_bytes()
        original=self.output/'runtime';original.mkdir()
        marker='paper3r5-'+hashlib.sha256(json.dumps(['a'*64,str(self.output)],separators=(',',':')).encode()).hexdigest()[:16]
        self.old_marker=marker;self.new_marker='paper3r5p-newrecipe'
        old={'marker':marker,'attempts':[{'sequence':2,'job_id':'772012','name':marker+'-a2','status':'ACKNOWLEDGED','marker':marker}]}
        (original/'state.json').write_text(json.dumps(old));(original/'launch.json').write_text('{"slots":3}')
        for key in ('already-complete','unfinished'):(self.output/key).mkdir()
        (self.output/'already-complete/result.json').write_text('{"status":"COMPLETE"}')
        self.ckpt=self.output/'unfinished/state-0.pt';self.ckpt.write_bytes(b'PRESERVE_CHECKPOINT')
        self.scheduler=Scheduler(5,marker,self.new_marker)
        self.runtime=original/'three_jobs'
        self.launch={'account':'u','partition':'hpc_gpu1','old_partition':'hpc_gpu','cpus':12,'memories':['64G']*3,'submit_hold':True}
        self.migration=HeldQueueMigration(self.path,self.runtime,5,self.launch,runner=self.scheduler,clock=lambda:100000,account='u')
        self.summary={'marker':self.new_marker,'jobs':[{'slot':s,'job_id':str(900001+s)} for s in range(3)]}
    def tearDown(self):self.temp.cleanup()
    def test_three_real_held_jobs_before_old_cancel_then_release(self):
        first=self.migration.step(self.summary)
        self.assertEqual(first['status'],'WAITING_OLD_ALLOCATION_TERMINAL')
        self.assertFalse((self.runtime/'HANDOFF_READY.json').exists())
        self.assertFalse(any(c[:2]==['scontrol','release'] for c in self.scheduler.calls))
        value=self.migration.step(self.summary)
        self.assertEqual(value['status'],'READY')
        self.assertTrue((self.runtime/'HANDOFF_READY.json').is_file())
        self.assertTrue((self.output/'runtime/STOP').is_file())
        self.assertTrue((self.runtime/'old-allocation-history/STOP').is_file())
        calls=self.scheduler.calls
        cancel=next(i for i,c in enumerate(calls) if c[0]=='scancel')
        self.assertEqual(sum(c[0]=='scontrol' and '--oneliner' in c for c in calls[:cancel]),3)
        self.assertEqual(sum(c[:2]==['scontrol','release'] for c in calls),3)
        self.assertEqual(self.path.read_bytes(),self.original_plan)
        self.assertEqual(self.ckpt.read_bytes(),b'PRESERVE_CHECKPOINT')
    def test_submit_rejection_or_missing_ack_does_not_touch_old(self):
        bad=copy.deepcopy(self.summary);bad['jobs'][2]['job_id']=None
        result=self.migration.step(bad)
        self.assertEqual(result['status'],'WAITING_THREE_HELD_REPLACEMENTS')
        self.assertFalse((self.output/'runtime/STOP').exists())
        self.assertFalse(any(c[0]=='scancel' for c in self.scheduler.calls))
    def test_pending_running_race_gets_checkpoint_signal(self):
        self.scheduler.race=True;self.migration.step(self.summary)
        self.migration.step(self.summary)
        self.assertIn(['scancel','--state=PENDING','772012'],self.scheduler.calls)
        self.assertIn(['scancel','--batch','--signal=USR1','772012'],self.scheduler.calls)
        self.assertEqual(self.migration.step(self.summary)['status'],'READY')
    def test_old_allocation_lock_waits(self):
        self.migration.step(self.summary)
        with (self.output/'runtime/allocation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(self.migration.step(self.summary)['status'],'WAITING_OLD_CONTROLLER_AND_ALLOCATION_LOCKS')
            self.assertFalse((self.runtime/'HANDOFF_READY.json').exists())
    def test_wrong_resource_does_not_touch_old(self):
        self.scheduler.bad_memory=True
        with self.assertRaises(MigrationOwnershipError):self.migration.step(self.summary)
        self.assertFalse((self.output/'runtime/STOP').exists())
        self.assertFalse(any(c[0]=='scancel' for c in self.scheduler.calls))
    def test_scheduler_failure_is_not_empty_queue(self):
        self.scheduler.queue_failure=True
        with self.assertRaises(MigrationTransient):self.migration.step(self.summary)
        self.assertFalse((self.output/'runtime/STOP').exists())
    def test_ready_releases_a_new_held_replacement(self):
        self.migration.step(self.summary);self.migration.step(self.summary)
        self.scheduler.new['900001']['held']=True
        count=sum(c[:2]==['scontrol','release'] for c in self.scheduler.calls)
        self.assertEqual(self.migration.step(self.summary)['status'],'READY')
        self.assertEqual(sum(c[:2]==['scontrol','release'] for c in self.scheduler.calls),count+1)


if __name__=='__main__':unittest.main()
