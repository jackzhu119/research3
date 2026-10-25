import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
spec=importlib.util.spec_from_file_location('r9_test_report_dispatch',RUNTIME/'report_dispatch.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

class ReportDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.output=self.root/'runs/followup_20261025_r9';self.op=self.output/'runtime/three_jobs'
        self.op.mkdir(parents=True)
        self.package=self.root/'package';(self.package/'paper3r9').mkdir(parents=True)
        (self.package/'paper3r9/__init__.py').write_text('')
        (self.package/'paper3r9/report.py').write_text('''import json,pathlib
def build_report(plan,output=None):
 out=pathlib.Path(output)/'reports';out.mkdir(parents=True,exist_ok=True)
 report={'status':'PARTIAL','complete_tasks':1,'total_tasks':3}
 (out/'report.json').write_text(json.dumps(report))
 return report
''')
        self.plan=self.output/'plan.json'
        self.plan.write_text(json.dumps({'root':str(self.root),'output':str(self.output),
            'recipe_sha256':'r9','tasks':[{'key':'a'},{'key':'b'},{'key':'c'}]}))
        (self.op/'launch.json').write_text(json.dumps({'plan':str(self.plan),'package':str(self.package),'python':sys.executable}))
        self.snapshot={'recipe_sha256':'r9','complete_tasks':1,'total_tasks':3}
    def tearDown(self):self.temp.cleanup()
    def test_real_isolated_report_process_and_unchanged_count_skip(self):
        calls=[]
        def popen(*a,**k):calls.append(a[0]);return subprocess.Popen(*a,**k)
        d=D.ReportDispatcher(self.root,self.op,popen=popen)
        self.assertTrue(d.kick(self.snapshot));d.thread.join(5);self.assertFalse(d.thread.is_alive())
        self.assertEqual(json.loads((self.output/'reports/report.json').read_text())['complete_tasks'],1)
        self.assertEqual(json.loads((self.op/'report-state.json').read_text())['status'],'BUILT')
        d.next_at=0
        self.assertTrue(d.kick(self.snapshot));d.thread.join(5)
        self.assertEqual(len(calls),1)
        self.assertIn('-S',calls[0]);self.assertNotIn('worker',calls[0])
    def test_slow_report_does_not_block_monitor_and_single_flight(self):
        entered,release=threading.Event(),threading.Event()
        class Child:
            pid=424242
            def wait(self):entered.set();release.wait(3);return 0
        d=D.ReportDispatcher(self.root,self.op,popen=lambda *a,**k:Child())
        before=time.monotonic();self.assertTrue(d.kick(self.snapshot))
        self.assertLess(time.monotonic()-before,.2);self.assertTrue(entered.wait(1))
        self.assertFalse(d.kick(self.snapshot))
        release.set();d.thread.join(3)
    def test_count_change_is_throttled_but_final_report_can_start_promptly(self):
        clock=[100.0]
        class Child:
            pid=424242
            def wait(self):return 1
        d=D.ReportDispatcher(self.root,self.op,clock=lambda:clock[0],popen=lambda *a,**k:Child())
        self.assertTrue(d.kick(self.snapshot));d.thread.join(3)
        clock[0]+=30
        next_snapshot={**self.snapshot,'complete_tasks':2}
        self.assertFalse(d.kick(next_snapshot))
        final={**self.snapshot,'complete_tasks':3}
        self.assertTrue(d.kick(final));d.thread.join(3)
        self.assertFalse(d.kick(final))
        clock[0]+=601;self.assertTrue(d.kick(final));d.thread.join(3)
    def test_zero_completed_tasks_does_not_build_report(self):
        d=D.ReportDispatcher(self.root,self.op,popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError()))
        self.assertFalse(d.kick({**self.snapshot,'complete_tasks':0}))
        self.assertIsNone(d.thread)

if __name__=='__main__':unittest.main()
