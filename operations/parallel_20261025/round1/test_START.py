import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import START as start


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

    def resource_query(self, *, reason='JobHeldUser', queue_state='PENDING', memories=None):
        memories=memories or ('43G','43G','42G')
        def run(argv):
            if argv[0]=='squeue':
                return completed('\n'.join('%s|mine|%s|mine-s%d-a1|hpc_gpu'%(800001+n,queue_state,n) for n in range(3)))
            if argv[:3]==['scontrol','show','job']:
                n=int(argv[-1])-800001
                return completed('JobId=%s Partition=hpc_gpu NumNodes=1 NumCPUs=12 MinMemoryNode=%s '
                                 'TresPerNode=gres:gpu:1 TRES=cpu=12,mem=%s,gres/gpu=1 Reason=%s Priority=0'
                                 %(argv[-1],memories[n],memories[n],reason))
            raise AssertionError(argv)
        return run

    def test_three_distinct_held_jobs_exact_resources(self):
        result=start.confirm_three_held({'jobs':jobs(),'marker':'mine'},run=self.resource_query())
        self.assertTrue(result['ready'])
        self.assertEqual(sum(start.memory_mib(v['memory']) for v in result['jobs']),128*1024)
        self.assertEqual(sum(v['cpus'] for v in result['jobs']),36)
        self.assertEqual(sum(v['gpus'] for v in result['jobs']),3)

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
