import fcntl
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
sys.path.insert(0,str(RUNTIME))
from coordinator import prepare_context,controller_loop
from three_jobs import plan_context
from test_r8_runtime_scheduler import Slurm,Clock
spec=importlib.util.spec_from_file_location('r8_operational_start',RUNTIME/'START.py')
START=importlib.util.module_from_spec(spec);spec.loader.exec_module(START)

class StartTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.root=self.base/'project';self.root.mkdir()
        self.package=self.base/'package';(self.package/'paper3r8').mkdir(parents=True)
        (self.package/'paper3r8/entry.py').write_text('raise SystemExit(0)\n')
        self.out=self.root/'runs/followup_20261025_r8';self.out.mkdir(parents=True)
        self.path=self.out/'plan.json'
        self.plan={'root':str(self.root),'output':str(self.out),'recipe_sha256':'r8',
                   'tasks':[{'key':'a'},{'key':'b','requires':['a']} ]}
        self.path.write_text(json.dumps(self.plan))
        self.ctx=prepare_context(self.path,package=self.package,python=sys.executable)
    def tearDown(self):self.tmp.cleanup()
    def test_controller_submits_three_jobs_without_hold_or_handoff(self):
        slurm=Slurm();clock=Clock()
        with mock.patch('sys.stdout',io.StringIO()):
            self.assertEqual(controller_loop(self.ctx,runner=slurm,clock=clock,once=True),0)
            self.assertEqual(controller_loop(self.ctx,runner=slurm,clock=clock,once=True),0)
        self.assertEqual(len(slurm.submits),3)
        for c in slurm.submits:
            self.assertIn('--partition=hpc_gpu',c);self.assertIn('--cpus-per-task=16',c)
            self.assertIn('--nodes=1',c);self.assertIn('--mem=64G',c)
            self.assertNotIn('--hold',c);self.assertFalse(any(x.startswith('--exclude=') for x in c))
        self.assertFalse((self.ctx['runtime']/'HANDOFF_READY.json').exists())
        self.assertFalse(any(c[0]=='scancel' for c in slurm.calls))
    def test_start_adopts_existing_controller_lock(self):
        path=self.ctx['runtime']/'controller.lock'
        with path.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with mock.patch.object(START.subprocess,'Popen',side_effect=AssertionError('Must not spawn duplicate')):
                result=START.start_process(self.ctx,'controller',popen=START.subprocess.Popen)
        self.assertTrue(result['reused'])
    def test_start_prepare_is_stdlib_only_no_scheduler_actions(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(RUNTIME/'START.py'),
                '--plan',str(self.path),'--package',str(self.package)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        value=json.loads(r.stdout)
        self.assertEqual(value['cpus_total'],48);self.assertFalse(value['package_install'])
        self.assertFalse((self.ctx['runtime']/'scheduler-state.json').exists())
    def test_cycle_is_reported_before_submitting_jobs(self):
        self.plan['tasks'][0]['requires']=['b'];self.path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError,'cyclic'):
            plan_context(self.path)

if __name__=='__main__':unittest.main()
