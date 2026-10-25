import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import START as start

# Exact login1 Slurm 20.11 scontrol output supplied on 2026-10-25.
SCHOOL_SCONTROL = "JobId=772076 JobName=paper3r1p-84c2da292f724ce9-s0-a1 UserId=zhuziyuan(2461) GroupId=zhuziyuan(11017) MCS_label=N/A Priority=0 Nice=0 Account=zhuziyuan QOS=user_zhuziyuan JobState=PENDING Reason=JobHeldUser Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=0:0 RunTime=00:00:00 TimeLimit=5-03:00:00 TimeMin=N/A SubmitTime=2026-10-25T11:46:31 EligibleTime=Unknown AccrueTime=Unknown StartTime=Unknown EndTime=Unknown Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-25T11:46:31 Partition=hpc_gpu AllocNode:Sid=login1:814457 ReqNodeList=(null) ExcNodeList=(null) NodeList=(null) NumNodes=1-1 NumCPUs=12 NumTasks=1 CPUs/Task=12 ReqB:S:C:T=0:0:*:* TRES=cpu=12,mem=43G,node=1,billing=12,gres/gpu=1 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=12 MinMemoryNode=43G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null) Command=/public/home/zhuziyuan/projects/paper3/runs/training_campaign_parallel_v1/single-gpu-worker.sbatch /public/home/zhuziyuan/.venvs/paper3-hpc-cu121-offline/bin/python -I -S -B -u /public/home/zhuziyuan/paper3-parallel-migrations/fa00be482b1440d3fae7fff1/round1/worker.py --root /public/home/zhuziyuan/projects/paper3 --runtime-dir /public/home/zhuziyuan/projects/paper3/runs/training_campaign_parallel_v1 --slot 0 --package /public/home/zhuziyuan/paper3-parallel-migrations/fa00be482b1440d3fae7fff1/round1 --payload-manifest /public/home/zhuziyuan/paper3-runtime-repairs/6336246671678df5a4ae5b28b266412c594d95a13a785a717eab23e42f04c129/payload_manifest.json --direct-package /public/home/zhuziyuan/paper3-direct-train/656cb8e704b87c4e69e2fcee840093beff03a3f8df964272dd6b63789ae48528 --prior-job 769863 --allocation-seconds 442800 WorkDir=/tmp Comment=paper3r1p-84c2da292f724ce9  StdErr=/public/home/zhuziyuan/projects/paper3/runs/training_campaign_parallel_v1/logs/slot-0-772076.err StdIn=/dev/null StdOut=/public/home/zhuziyuan/projects/paper3/runs/training_campaign_parallel_v1/logs/slot-0-772076.out Power= TresPerNode=gpu:1 NtasksPerTRES:0"


def completed(stdout='', returncode=0):
    return SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)


def jobs():
    return [{'slot':n, 'job_id':str(800001+n), 'memory':mem,
             'slurm_state':'PENDING', 'held_requested':True}
            for n,mem in enumerate(('43G','43G','42G'))]


def frontier(complete=False):
    return {'phase':'all_complete' if complete else 'meta_screening',
            'phase_actions':[], 'ready_actions':[], 'phase_completed':45 if complete else 32,
            'phase_total':45, 'mainline_complete':complete, 'all_complete':complete,
            'blocked':[]}


class StartTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'project'
        self.root.mkdir()
        self.runtime=self.root/start.OPERATIONAL
        self.runtime.mkdir(parents=True)
        self.launch={'root':str(self.root), 'old_job':'770043', 'direct_package':str(Path(self.temp.name)/'direct'),
                     'partition':'hpc_gpu','worker_command':['python','worker.py','{slot}']}

    def resource_query(self, *, reason='JobHeldUser', queue_state='PENDING', memories=None,
                       nodes='1',cpus='12'):
        memories=memories or ('43G','43G','42G')
        def run(argv):
            if argv[0]=='squeue':
                return completed('\n'.join('%s|mine|%s|mine-s%d-a1|hpc_gpu'%(800001+n,queue_state,n) for n in range(3)))
            if argv[:3]==['scontrol','show','job']:
                n=int(argv[-1])-800001
                return completed('JobId=%s Partition=hpc_gpu NumNodes=%s NumCPUs=%s MinMemoryNode=%s '
                                 'TresPerNode=gres:gpu:1 TRES=cpu=12,mem=%s,gres/gpu=1 Reason=%s Priority=0'
                                 %(argv[-1],nodes,cpus,memories[n],memories[n],reason))
            raise AssertionError(argv)
        return run

    def test_three_distinct_held_jobs_exact_resources(self):
        result=start.confirm_three_held({'jobs':jobs(),'marker':'mine'},run=self.resource_query())
        self.assertTrue(result['ready'])
        self.assertEqual(sum(start.memory_mib(v['memory']) for v in result['jobs']),128*1024)
        self.assertEqual(sum(v['cpus'] for v in result['jobs']),36)
        self.assertEqual(sum(v['gpus'] for v in result['jobs']),3)

    def test_slurm_20_11_fixed_one_node_range_is_accepted(self):
        result=start.confirm_three_held({'jobs':jobs(),'marker':'mine'},
                                       run=self.resource_query(nodes='1-1'))
        self.assertTrue(result['ready'])

    def test_exact_school_held_job_fixture(self):
        first=int(re.search(r'JobId=(\d+)',SCHOOL_SCONTROL)[1])
        marker=re.search(r'(?:^|\s)Comment=([^\s]+)',SCHOOL_SCONTROL)[1]
        rows=[dict(row,job_id=str(first+row['slot'])) for row in jobs()]
        def query(argv):
            if argv[0]=='squeue':
                return completed('\n'.join(
                    f"{row['job_id']}|{marker}|PENDING|{marker}-s{row['slot']}-a1|hpc_gpu"
                    for row in rows))
            self.assertEqual(argv[:3],['scontrol','show','job'])
            slot=int(argv[-1])-first
            text=SCHOOL_SCONTROL.replace(f'JobId={first}',f'JobId={first+slot}')
            text=text.replace('-s0-a1',f'-s{slot}-a1')
            text=text.replace('mem=43G','mem='+rows[slot]['memory'])
            text=text.replace('MinMemoryNode=43G','MinMemoryNode='+rows[slot]['memory'])
            return completed(text)
        self.assertTrue(start.confirm_three_held({'jobs':rows,'marker':marker},run=query)['ready'])

    def test_other_node_counts_or_ranges_are_rejected_with_observed_fields(self):
        for nodes in ('1-2','2','0','0-1','01','1-01','2-2'):
            with self.subTest(nodes=nodes):
                with self.assertRaises(ValueError) as caught:
                    start.confirm_three_held({'jobs':jobs(),'marker':'mine'},
                                             run=self.resource_query(nodes=nodes))
                self.assertIn('"NumNodes": "'+nodes+'"',str(caught.exception))
                self.assertIn('"NumCPUs": "12"',str(caught.exception))

    def test_fixed_node_range_does_not_bypass_cpu_or_memory_mismatch(self):
        for query in (self.resource_query(nodes='1-1',cpus='24'),
                      self.resource_query(nodes='1-1',memories=('64G','43G','42G'))):
            with self.assertRaises(ValueError) as caught:
                start.confirm_three_held({'jobs':jobs(),'marker':'mine'},run=query)
            self.assertIn('observed=',str(caught.exception))
            self.assertIn('expected=',str(caught.exception))

    def test_pending_but_not_user_held_is_not_handoff_ready(self):
        result=start.confirm_three_held({'jobs':jobs(),'marker':'mine'},
                                       run=self.resource_query(reason='Priority'))
        self.assertFalse(result['ready'])

    def test_running_replacement_is_not_an_held_replacement(self):
        result=start.confirm_three_held({'jobs':jobs(),'marker':'mine'},
                                       run=self.resource_query(queue_state='RUNNING'))
        self.assertFalse(result['ready'])

    def test_duplicate_job_id_is_not_three_jobs(self):
        rows=jobs()
        for row in rows:row.update(job_id='800001',memory='43G')
        result=start.confirm_three_held({'jobs':rows,'marker':'mine'},run=self.resource_query())
        self.assertFalse(result['ready'])

    def test_prepare_is_readonly_no_signals_or_submissions(self):
        args=SimpleNamespace(root=str(self.root),old_job='770043',direct_package=None,
                             payload_manifest=None,python='/school/venv/bin/python')
        ctx={'direct_package':'/school/direct','payload_manifest':'/school/repair/payload_manifest.json'}
        with patch.object(start,'plan_ready',return_value=frontier()), \
             patch.object(start,'discover_context',return_value=ctx), \
             patch.object(start.subprocess,'Popen') as spawn, \
             patch.object(start.os,'kill') as kill, patch.object(start,'emit'):
            _,_,_,launch=start.prepare(args)
        spawn.assert_not_called();kill.assert_not_called()
        self.assertEqual(launch['partition'],'hpc_gpu')
        self.assertEqual(launch['memories'],['43G','43G','42G'])
        self.assertEqual(launch['cpus'],12)
        self.assertEqual(launch['gres'],'gpu:1')

    def controller_case(self, ready, complete=False, mainline=False):
        events=[]
        snap={'jobs':jobs(),'marker':'mine','inflight_jobs':0 if complete else 3,
              'terminal_job_ids':[]}
        class Scheduler:
            def __init__(_self,*a,**kw):_self.launch=dict(self.launch)
            def step(_self,keep_workers=True):
                events.append(('scheduler',keep_workers));return snap
        count=[0]
        def stop():count[0]+=1;return count[0]>1
        def verify(snapshot):events.append(('verify-held',));return {'ready':ready,'reason':'fixture'}
        def retire(*a,**kw):events.append(('retire',));return {'status':'READY'}
        def handoff(*a,**kw):events.append(('handoff',));return {'status':'READY','quiescence':{}}
        def run(argv):events.append(('command',tuple(argv)));return completed()
        phase=frontier(complete)
        if mainline:phase['mainline_complete']=True
        with patch.object(start,'ThreeJobScheduler',Scheduler), \
             patch.object(start,'plan_ready',return_value=phase), \
             patch.object(start,'confirm_three_held',side_effect=verify), \
             patch.object(start,'retire_legacy_monitor',side_effect=retire), \
             patch.object(start,'checkpoint_handoff_step',side_effect=handoff), \
             patch.object(start,'publish_progress',return_value={'phase':'fixture','status':'RUNNING',
                           'completed_by_kind':{},'jobs':snap['jobs']}), \
             patch.object(start,'command',side_effect=run), \
             patch.object(start,'emit'), patch.object(start.time,'sleep'):
            result=start.controller_loop(self.root,self.runtime,self.launch,stop_requested=stop)
        return events,result

    def test_old_job_not_retired_or_signalled_before_three_held_verified(self):
        events,_=self.controller_case(False)
        self.assertNotIn(('retire',),events)
        self.assertNotIn(('handoff',),events)
        self.assertFalse((self.runtime/'HANDOFF_READY.json').exists())

    def test_handoff_order_after_real_queue_verification(self):
        events,_=self.controller_case(True)
        self.assertLess(events.index(('verify-held',)),events.index(('retire',)))
        self.assertLess(events.index(('retire',)),events.index(('handoff',)))
        self.assertTrue((self.runtime/'HANDOFF_READY.json').is_file())

    def test_initial_already_complete_never_submits_or_signals(self):
        events,_=self.controller_case(True,complete=True)
        self.assertFalse(any(e[0]=='scheduler' and bool(e[1]) for e in events))
        self.assertNotIn(('retire',),events)
        self.assertNotIn(('handoff',),events)
        self.assertFalse(any(e[0]=='command' for e in events))

    def test_initial_extensions_still_prepares_three_held_before_handoff(self):
        events,_=self.controller_case(False,mainline=True)
        self.assertEqual(next(e for e in events if e[0]=='scheduler'),('scheduler',True))
        self.assertNotIn(('handoff',),events)

    def test_changed_legacy_current_job_requires_exact_history_comment(self):
        path=self.root/'.paper3-hpc/direct-train/state.json'
        path.parent.mkdir(parents=True)
        comment='paper3-direct-'+'a'*32
        path.write_text(json.dumps({'current_job_id':'772999',
            'history':[{'job_id':'772999','submission':{'comment':comment}}]}))
        target=start.legacy_target(self.root,self.launch,{'status':'WAITING','retirement_complete':True})
        self.assertEqual(target,{'job_id':'772999','comment':comment,'requested_old_job':'770043'})
        self.assertIsNone(start.legacy_target(self.root,self.launch,{'status':'WAITING'}))

    def test_ambiguous_legacy_submission_intent_is_not_handed_off(self):
        path=self.root/'.paper3-hpc/direct-train/state.json'
        path.parent.mkdir(parents=True)
        comment='paper3-direct-'+'b'*32
        state={'current_job_id':'770043','intent':{'comment':comment,'started_utc':'2026-10-25T00:00:00Z'}}
        path.write_text(json.dumps(state))
        queue='772991|%s|paper3-train-direct\n772992|%s|paper3-train-direct'%(comment,comment)
        with patch.object(start,'command',return_value=completed(queue)):
            target=start.legacy_target(self.root,self.launch,{'status':'WAITING','retirement_complete':True})
        self.assertIsNone(target)
        self.assertEqual(json.loads(path.read_text()),state)

    def test_progress_keeps_original_completed_actions_and_state_unmodified(self):
        old=self.root/'runs/training_campaign_v1/campaign_progress.json'
        old.parent.mkdir(parents=True)
        old.write_text(json.dumps({'actions':[{'key':'pilot','kind':'pilot','receipt':{'path':'runs/pilot/receipt.json'}},
                                            {'key':'meta-old','kind':'meta_screening','receipt':{'path':'runs/meta/old/receipt.json'}}]}))
        before=old.read_bytes()
        current={'tasks':{'completed-extra':{'key':'completed-extra','kind':'meta_screening','status':'COMPLETE'}}}
        with patch.object(start,'action_complete',return_value=False):
            progress=start.publish_progress(self.root,self.runtime,frontier(),{'jobs':[]},current)
        self.assertEqual(progress['completed_by_kind'],{'pilot':1,'meta_screening':2})
        self.assertEqual(old.read_bytes(),before)
        self.assertEqual(progress['requested_resources']['memory_total'],'128G')

    def test_actual_isolated_stdlib_help(self):
        result=subprocess.run([sys.executable,'-I','-S','-B',str(Path(start.__file__)),'--help'],
                              text=True,capture_output=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--execute',result.stdout)
        self.assertIn('--old-job',result.stdout)


if __name__=='__main__':
    unittest.main()
