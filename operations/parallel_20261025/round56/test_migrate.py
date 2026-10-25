import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
import fcntl

from migrate import HeldQueueMigration,MigrationOwnershipError,MigrationTransient

# Exact login1 Slurm 20.11 scontrol outputs supplied on 2026-10-25.
SCHOOL_SCONTROL = {
    5: "JobId=772070 JobName=paper3r5p-2a492b1bc14432d0-s0-a1 UserId=zhuziyuan(2461) GroupId=zhuziyuan(11017) MCS_label=N/A Priority=0 Nice=0 Account=zhuziyuan QOS=user_zhuziyuan JobState=PENDING Reason=JobHeldUser Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:00:00 TimeLimit=5-03:00:00 TimeMin=N/A SubmitTime=2026-10-25T11:46:20 EligibleTime=Unknown AccrueTime=Unknown StartTime=Unknown EndTime=Unknown Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-25T11:46:20 Partition=hpc_gpu1 AllocNode:Sid=login1:814290 ReqNodeList=(null) ExcNodeList=gpu3 NodeList=(null) NumNodes=1-1 NumCPUs=12 NumTasks=1 CPUs/Task=12 ReqB:S:C:T=0:0:*:* TRES=cpu=12,mem=64G,node=1,billing=12,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=12 MinMemoryNode=64G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/runtime/three_jobs/single-gpu-worker.sbatch /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /public/home/zhuziyuan/paper3-parallel-migrations/fa00be482b1440d3fae7fff1/round56/three_jobs.py worker --plan /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/plan.json --runtime /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/runtime/three_jobs --round 5 --slot 0 --package /public/home/zhuziyuan/paper3-round5-hpcgpu/02955e93c9180dadeb6e --python /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python WorkDir=/tmp Comment=paper3r5p-2a492b1bc14432d0  StdErr=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/runtime/three_jobs/logs/slot-0-772070.err StdIn=/dev/null StdOut=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r5/runtime/three_jobs/logs/slot-0-772070.out Power= TresPerNode=gpu:1 NtasksPerTRES:0",
    6: "JobId=772073 JobName=paper3r6p-1223a296dd91e758-s0-a1 UserId=zhuziyuan(2461) GroupId=zhuziyuan(11017) MCS_label=N/A Priority=0 Nice=0 Account=zhuziyuan QOS=user_zhuziyuan JobState=PENDING Reason=JobHeldUser Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:00:00 TimeLimit=5-03:00:00 TimeMin=N/A SubmitTime=2026-10-25T11:46:22 EligibleTime=Unknown AccrueTime=Unknown StartTime=Unknown EndTime=Unknown Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-25T11:46:22 Partition=hpc_gpu1 AllocNode:Sid=login1:814369 ReqNodeList=(null) ExcNodeList=gpu3 NodeList=(null) NumNodes=1-1 NumCPUs=12 NumTasks=1 CPUs/Task=12 ReqB:S:C:T=0:0:*:* TRES=cpu=12,mem=64G,node=1,billing=12,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=12 MinMemoryNode=64G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6/runtime/three_jobs/single-gpu-worker.sbatch /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /public/home/zhuziyuan/paper3-parallel-migrations/fa00be482b1440d3fae7fff1/round56/three_jobs.py worker --plan /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6/plan.json --runtime /public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6/runtime/three_jobs --round 6 --slot 0 --package /public/home/zhuziyuan/paper3-round6-hpcgpu/45fb15d5c45bf0af0372 --python /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python WorkDir=/tmp Comment=paper3r6p-1223a296dd91e758  StdErr=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6/runtime/three_jobs/logs/slot-0-772073.err StdIn=/dev/null StdOut=/public/home/zhuziyuan/projects/paper3/runs/followup_20261024_r6/runtime/three_jobs/logs/slot-0-772073.out Power= TresPerNode=gpu:1 NtasksPerTRES:0",
}


class Scheduler:
    def __init__(self,round_number,old_marker,new_marker):
        self.round=round_number;self.old_marker=old_marker;self.new_marker=new_marker
        self.old_job={5:'772012',6:'772013'}[round_number]
        self.old_state='PENDING';self.old_terminal=False;self.race=False
        self.bad_memory=False;self.queue_failure=False;self.calls=[];self.nodes='1';self.cpus='12'
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
                  f"Partition=hpc_gpu1 NumCPUs={self.cpus} NumNodes={self.nodes} TRES=cpu=12,mem={memory},node=1,gres/gpu=1")
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
    def test_slurm_20_11_fixed_one_node_range_completes_handoff(self):
        self.scheduler.nodes='1-1'
        self.assertEqual(self.migration.step(self.summary)['status'],'WAITING_OLD_ALLOCATION_TERMINAL')
        self.assertEqual(self.migration.step(self.summary)['status'],'READY')
    def test_exact_school_r5_and_r6_held_job_fixtures(self):
        for round_number,text in SCHOOL_SCONTROL.items():
            with self.subTest(round=round_number):
                first=int(re.search(r'JobId=(\d+)',text)[1])
                marker=re.search(r'(?:^|\s)Comment=([^\s]+)',text)[1]
                account=re.search(r'UserId=([^\s(]+)',text)[1]
                output=self.base/('school-round'+str(round_number));output.mkdir()
                plan=dict(self.plan,output=str(output))
                path=output/'plan.json';path.write_text(json.dumps(plan))
                rows=[dict(slot=s,job_id=str(first+s),user=account,marker=marker,
                    state='PENDING',name=marker+'-s'+str(s)+'-a1',
                    partition='hpc_gpu1',reason='(JobHeldUser)') for s in range(3)]
                def query(command,**kwargs):
                    self.assertEqual(command[:4],['scontrol','--oneliner','show','job'])
                    slot=int(command[-1])-first
                    value=text.replace(f'JobId={first}',f'JobId={first+slot}')
                    value=value.replace('-s0-a1',f'-s{slot}-a1')
                    return SimpleNamespace(returncode=0,stdout=value,stderr='')
                migration=HeldQueueMigration(path,output/'runtime/three_jobs',round_number,
                    dict(self.launch,account=account),runner=query,account=account)
                summary={'marker':marker,'jobs':[{'slot':r['slot'],'job_id':r['job_id']} for r in rows]}
                verified=migration._new_jobs(summary,rows)
                self.assertEqual(len(verified),3)
                self.assertTrue(all(row['nodes']==1 and row['gpus']==1 for row in verified))
                self.assertFalse((output/'runtime/STOP').exists())
    def test_other_node_counts_or_ranges_are_rejected_before_old_mutation(self):
        events=[]
        self.migration.emit=lambda event,**fields:events.append((event,fields))
        for nodes in ('1-2','2','0','0-1','01','1-01','2-2'):
            with self.subTest(nodes=nodes):
                self.scheduler.nodes=nodes
                with self.assertRaises(MigrationOwnershipError) as caught:
                    self.migration.step(self.summary)
                self.assertIn('"numnodes": "'+nodes+'"',str(caught.exception))
                self.assertIn('"numcpus": "12"',str(caught.exception))
                self.assertEqual(events[-1][0],'REPLACEMENT_VERIFICATION_MISMATCH')
                self.assertEqual(events[-1][1]['observed']['numnodes'],nodes)
        self.assertFalse((self.output/'runtime/STOP').exists())
        self.assertFalse(any(c[0]=='scancel' for c in self.scheduler.calls))
    def test_fixed_node_range_does_not_bypass_cpu_or_memory_mismatch(self):
        self.scheduler.nodes='1-1';self.scheduler.cpus='24'
        with self.assertRaises(MigrationOwnershipError):self.migration.step(self.summary)
        self.scheduler.cpus='12';self.scheduler.bad_memory=True
        with self.assertRaises(MigrationOwnershipError):self.migration.step(self.summary)
        self.assertFalse((self.output/'runtime/STOP').exists())
        self.assertFalse(any(c[0]=='scancel' for c in self.scheduler.calls))
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
