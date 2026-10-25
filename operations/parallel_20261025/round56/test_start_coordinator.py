import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import coordinator
import START


class StartCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.root=self.base/'project';self.root.mkdir()
        self.packages={}
        for n in (5,6):
            output=self.root/('runs/followup_20261024_r%d'%n);(output/'runtime').mkdir(parents=True)
            package=self.base/('native%d'%n);(package/('paper3r%d'%n)).mkdir(parents=True)
            (package/('paper3r%d/entry.py'%n)).write_text('# SOFTWARE FIXTURE ONLY\n')
            self.packages[n]=package
            plan={'root':str(self.root),'output':str(output),'recipe_sha256':str(n)*64,
                  'tasks':[{'key':'done','family':'basis'},{'key':'pending','family':'core','requires':['done']}]}
            (output/'plan.json').write_text(json.dumps(plan))
            (output/'done').mkdir();(output/'done/result.json').write_text('{"status":"COMPLETE"}')
            (output/'pending').mkdir();(output/'pending/state-0.pt').write_bytes(b'PRESERVE_NATIVE_CHECKPOINT')
            (output/'runtime/launch.json').write_text(json.dumps({'package':str(package),'partition':'hpc_gpu','account':'u'}))
    def tearDown(self):self.temp.cleanup()
    def context(self,n=5):return coordinator.prepare_context(self.root,n,python=sys.executable)
    def test_prepare_both_rounds_uses_existing_worker_and_new_partition(self):
        for n in (5,6):
            context=self.context(n);launch=context['launch']
            self.assertEqual(launch['partition'],'hpc_gpu1');self.assertEqual(launch['old_partition'],'hpc_gpu')
            self.assertEqual(launch['memories'],['64G']*3);self.assertEqual(launch['cpus'],12)
            self.assertEqual(launch['package'],str(self.packages[n]))
            self.assertIn(str(self.packages[n]),launch['worker_command'])
            self.assertEqual(context['complete_tasks'],1)
    def test_prepare_cli_under_isolated_stdlib_has_no_background_or_slurm(self):
        for n in (5,6):
            output=self.root/('runs/followup_20261024_r%d'%n)
            before=(output/'plan.json').read_bytes(),(output/'pending/state-0.pt').read_bytes()
            command=[sys.executable,'-I','-S','-B','-u',str(Path(START.__file__)),
                     '--round',str(n),'--root',str(self.root)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('ROUND_56_THREE_JOBS_PREPARED',result.stdout)
            self.assertIn('"partition": "hpc_gpu1"',result.stdout)
            self.assertFalse((output/'runtime/STOP').exists())
            self.assertFalse((output/'runtime/three_jobs/controller-process.json').exists())
            self.assertEqual(before,((output/'plan.json').read_bytes(),(output/'pending/state-0.pt').read_bytes()))
    def test_prepare_does_not_spawn(self):
        with patch.object(START,'start_process') as spawn:
            self.assertEqual(START.main(['--round','5','--root',str(self.root)]),0)
            spawn.assert_not_called()
    def test_execute_starts_independent_controller_and_monitor(self):
        called=[]
        def spawn(context,kind,credentials=None):
            called.append((kind,credentials));return {'kind':kind,'pid':1}
        with patch.object(START,'start_process',side_effect=spawn):
            self.assertEqual(START.main(['--round','5','--root',str(self.root),'--execute']),0)
        self.assertEqual([x[0] for x in called],['controller','monitor'])
        self.assertFalse((self.root/'runs/followup_20261024_r5/runtime/STOP').exists())
    def test_scheduler_runs_before_migration_and_completion_skips_old_handoff(self):
        context=self.context();order=[]
        class Migration:
            def step(self,summary):order.append('migrate');return {'status':'WAITING_THREE_HELD_REPLACEMENTS'}
        def step(*args,**kwargs):
            order.append('scheduler');return {'marker':'new','jobs':[],'progress':{'complete_tasks':1,'total_tasks':2}}
        coordinator.controller_iteration(context,migration=Migration(),step_function=step,emit_callback=lambda *a,**k:None)
        self.assertEqual(order,['scheduler','migrate'])
        def complete(*args,**kwargs):
            return {'marker':'paper3r5p-test','jobs':[],'progress':{'complete_tasks':2,'total_tasks':2}}
        with patch.object(coordinator,'stop_owned_new_workers',return_value=True) as drain:
            result=coordinator.controller_iteration(context,migration=Migration(),step_function=complete,
                                                   emit_callback=lambda *a,**k:None)
            self.assertEqual(result['status'],'FINISHED');drain.assert_called_once()
        self.assertEqual(order,['scheduler','migrate'])
    def test_cleanup_only_cancels_exact_owned_new_jobs(self):
        context=self.context();calls=[];marker='paper3r5p-test'
        def run(command,**kwargs):
            calls.append(command)
            if command[0]=='squeue':
                return SimpleNamespace(returncode=0,stdout='910000|paper3r5p-test|PENDING|paper3r5p-test-s0-a1\n'
                    '772012|paper3r5-old|PENDING|paper3r5-old-a2\n'
                    '920000|paper3r6p-other|PENDING|paper3r6p-other-s0-a1\n',stderr='')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        self.assertFalse(coordinator.stop_owned_new_workers(context['runtime'],marker,'u',runner=run,
                                                           emit_callback=lambda *a,**k:None))
        self.assertEqual([c for c in calls if c[0]=='scancel'],[['scancel','--state=PENDING','910000']])
    def test_background_process_identity_reused(self):
        context=self.context();children=[]
        def fake_popen(command,**kwargs):
            script=command[command.index('-u')+1]
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',script,
                '--round','5','--root',str(self.root)],stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children.append(child);return child
        try:
            first=START.start_process(context,'controller',popen=fake_popen)
            second=START.start_process(context,'controller',popen=fake_popen)
            self.assertFalse(first['reused']);self.assertTrue(second['reused']);self.assertEqual(len(children),1)
        finally:
            for child in children:child.terminate();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
