import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import repair_manager as repair


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.root = self.home/'project'; self.root.mkdir()
        self.package = self.home/'fixed'; self.package.mkdir()
        self.account = __import__('getpass').getuser()
        self.killed = []; self.calls = []
        self.make_round(1)

    def make_round(self, number):
        self.number = number
        self.runtime = self.root / ('runs/training_campaign_parallel_v1' if number == 1 else
                         'runs/followup_20261024_r%d/runtime/three_jobs' % number)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.old = self.home/'old'/('round1' if number == 1 else 'round56')
        self.old.mkdir(parents=True, exist_ok=True)
        self.new = self.package/('round1' if number == 1 else 'round56')
        self.new.mkdir(parents=True, exist_ok=True)
        (self.new/'START.py').write_text('print("fixed starter acknowledged")\n')
        self.launch = {'worker_command': ['/python', str(self.old/('worker.py' if number == 1 else 'three_jobs.py'))],
                       'root':str(self.root), 'account':self.account, 'old_job':str({1:770043,5:772012,6:772013}[number]),
                       'package':str(self.old if number == 1 else self.home/'scientific'),
                       'direct_package':'/deployed/direct', 'payload_manifest':'/deployed/payload_manifest.json'}
        self.marker = 'paper3r%dp-'%number + hashlib.sha256(json.dumps([str(self.runtime),
            'original-campaign-v1' if number == 1 else 'round5-6-original-recipe'],separators=(',',':')).encode()).hexdigest()[:16]
        self.ids = ['772076','772077','772078'] if number == 1 else ['772070','772071','772072']
        state = {'marker':self.marker,'slots':{str(s):{'attempts':[{'job_id':job,'status':'ACKNOWLEDGED'}]}
                                               for s,job in enumerate(self.ids)}}
        (self.runtime/'launch.json').write_text(json.dumps(self.launch))
        (self.runtime/'scheduler-state.json').write_text(json.dumps(state))
        self.controller = self.process(400, 'controller')
        self.monitor = self.process(401, 'monitor')
        self.other = [self.process(402,'worker'), self.process(403,'uploader')]
        self.nodes = '1-1'; self.queue_state = 'PENDING'; self.gpu = '1'; self.cpu = '12'

    def process(self, pid, kind, fixed=False):
        folder = self.new if fixed else self.old
        if kind == 'controller':
            filename = 'START.py' if self.number == 1 else 'coordinator.py'
            argv = ['/python', str(folder/filename), '--root',str(self.root)]
            argv += ['--controller'] if self.number == 1 else ['--round',str(self.number),'--runtime',str(self.runtime)]
        elif kind == 'monitor':
            argv = ['/python',str(folder/'monitor.py'),'--root',str(self.root),'--operational',str(self.runtime)]
            if self.number != 1:argv += ['--round',str(self.number)]
        elif kind == 'worker':
            argv = ['/python',str(folder/'worker.py'),'--root',str(self.root),'--runtime-dir',str(self.runtime)]
        else:
            argv = ['/python','/home/uploader/UPLOAD-PAPER3-RESULTS.py','--worker','--root',str(self.root)]
        return {'pid':pid,'uid':os.getuid(),'argv':argv,'start_ticks':pid*10,'boot_id':'fixture-boot'}

    def runner(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == 'squeue':
            partition = 'hpc_gpu' if self.number == 1 else 'hpc_gpu1'
            text = '\n'.join('%s|%s|%s|%s|%s-s%d-a1|%s|(JobHeldUser)' %
                    (job,self.account,self.marker,self.queue_state,self.marker,s,partition) for s,job in enumerate(self.ids))
        elif argv[0] == 'scontrol':
            job = argv[-1]; slot = self.ids.index(job)
            mem = ('43G','43G','42G')[slot] if self.number == 1 else '64G'
            partition = 'hpc_gpu' if self.number == 1 else 'hpc_gpu1'
            text = ('JobId=%s JobName=%s-s%d-a1 UserId=%s(2461) Comment=%s Partition=%s '
                    'JobState=PENDING Priority=0 Reason=JobHeldUser NumNodes=%s NumCPUs=%s '
                    'TRES=cpu=12,mem=%s,node=1,gres/gpu=%s MinMemoryNode=%s TresPerNode=gpu:%s') % \
                    (job,self.marker,slot,self.account,self.marker,partition,self.nodes,self.cpu,mem,self.gpu,mem,self.gpu)
        else:
            self.assertIn('--execute', argv)
            self.assertEqual(Path(argv[5]), self.new/'START.py')
            text = 'background startup acknowledged\n'
        return subprocess.CompletedProcess(argv,0,text,'')

    def invoke(self, **changes):
        rows = [self.controller,self.monitor]+self.other
        options = dict(execute=True,runner=self.runner,killer=lambda pid,sig:self.killed.append((pid,sig)),
                       processes=lambda:rows,snapshot=lambda pid:next((p for p in rows if p['pid']==pid),None),
                       lock_check=lambda path:False)
        options.update(changes)
        return repair.repair_round(self.root,self.number,self.package,**options)

    def assert_no_slurm_mutation(self):
        self.assertFalse(any(c[0] in ('sbatch','scancel') or c[:2]==['scontrol','release'] for c in self.calls))

    def test_exact_school_one_one_all_three_rounds_retain_jobs_and_source_bytes(self):
        for number in (1,5,6):
            self.make_round(number);self.killed.clear();self.calls.clear()
            before = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in
                      (self.runtime/'launch.json',self.runtime/'scheduler-state.json')}
            result = self.invoke()
            self.assertEqual(result['status'],'FIXED_CONTROLLERS_STARTED_EXISTING_JOBS_PRESERVED')
            self.assertEqual(result['jobs'],self.ids)
            self.assertEqual(self.killed,[(400,signal.SIGKILL),(401,signal.SIGTERM)])
            self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in before})
            self.assert_no_slurm_mutation()
            self.assertTrue((Path(result['history'])/'before.json').is_file())

    def test_non_single_ranges_and_cpu_gpu_mismatch_never_signal(self):
        for nodes,cpu,gpu in [('1-2','12','1'),('2','12','1'),('0','12','1'),('1-1','24','1'),('1-1','12','2')]:
            self.nodes,self.cpu,self.gpu=nodes,cpu,gpu
            with self.assertRaises(RuntimeError):self.invoke()
            self.assertEqual(self.killed,[])

    def test_missing_held_worker_preserves_all_login_processes(self):
        self.ids.pop()
        with self.assertRaises(RuntimeError):self.invoke()
        self.assertEqual(self.killed,[])
        self.assert_no_slurm_mutation()

    def test_running_workers_are_never_stopped(self):
        self.queue_state='RUNNING'
        with self.assertRaises(RuntimeError):self.invoke()
        self.assertEqual(self.killed,[])

    def test_wrong_marker_or_account_is_not_adopted(self):
        self.marker='paper3r1p-wrong'
        with self.assertRaises(RuntimeError):self.invoke()
        self.assertEqual(self.killed,[])

    def test_pid_reuse_immediate_start_ticks_recheck_prevents_signal(self):
        def reused(pid):
            value=copy.deepcopy(self.controller);value['start_ticks']+=1;return value
        with self.assertRaisesRegex(RuntimeError,'identity_changed'):self.invoke(snapshot=reused)
        self.assertEqual(self.killed,[])

    def test_other_uid_and_workers_and_uploader_are_not_selected(self):
        fake=copy.deepcopy(self.controller);fake.update(pid=900,uid=os.getuid()+10000)
        self.other.append(fake)
        result=self.invoke()
        self.assertEqual([p['pid'] for p in result['processes']],[400,401])
        self.assert_no_slurm_mutation()

    def test_already_fixed_controller_repeat_does_not_query_or_signal(self):
        rows=[self.process(999,'controller',fixed=True)]
        result=self.invoke(processes=lambda:rows)
        self.assertEqual(result['status'],'ALREADY_FIXED_CONTROLLER_RUNNING')
        self.assertEqual(self.calls,[]);self.assertEqual(self.killed,[])

    def test_ready_handoff_is_observed_without_signals(self):
        (self.runtime/'HANDOFF_READY.json').write_text('{"status":"READY"}')
        result=self.invoke()
        self.assertEqual(result['status'],'OBSERVE_ONLY_HANDOFF_ALREADY_READY')
        self.assertEqual(self.killed,[]);self.assertEqual(self.calls,[])

    def test_diagnostic_mode_no_operational_writes_or_signal_or_start(self):
        before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in self.runtime.rglob('*') if p.is_file()}
        result=self.invoke(execute=False)
        after={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in self.runtime.rglob('*') if p.is_file()}
        self.assertEqual(result['status'],'REPAIR_PREPARED');self.assertEqual(before,after)
        self.assertEqual(self.killed,[])
        self.assertFalse(any('--execute' in c for c in self.calls))

    def test_independent_repair_lock_prevents_concurrent_signal_and_restart(self):
        import fcntl
        with (self.runtime/'repair.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=self.invoke()
        self.assertEqual(result['status'],'REPAIR_ALREADY_RUNNING')
        self.assertEqual(self.killed,[]);self.assertEqual(self.calls,[])

    def test_real_isolated_dummy_controller_lock_and_pid_replacement(self):
        script=self.old/'START.py'
        script.write_text('import pathlib,fcntl,time,sys\np=pathlib.Path(sys.argv[sys.argv.index("--root")+1])/"runs/training_campaign_parallel_v1/controller.lock"\nf=p.open("a");fcntl.flock(f,fcntl.LOCK_EX);print("READY",flush=True);time.sleep(60)\n')
        process=subprocess.Popen([sys.executable,str(script),'--controller','--root',str(self.root)],stdout=subprocess.PIPE,text=True)
        self.addCleanup(lambda:process.poll() is None and process.kill())
        self.addCleanup(process.stdout.close)
        self.assertEqual(process.stdout.readline().strip(),'READY')
        before=repair.proc_snapshot(process.pid)
        self.assertIsNotNone(before)
        def runner(argv,**kw):
            if argv[0] in ('squeue','scontrol'):return self.runner(argv,**kw)
            return subprocess.run(argv,**kw)
        value=repair.repair_round(self.root,1,self.package,execute=True,runner=runner,
                    processes=lambda:[before],snapshot=repair.proc_snapshot,killer=os.kill)
        process.wait(timeout=3)
        self.assertEqual(process.returncode,-signal.SIGKILL)
        self.assertIn('fixed starter acknowledged',value['starter_stdout'])
        self.assertEqual(value['jobs'],self.ids)
        self.assert_no_slurm_mutation()


if __name__=='__main__':unittest.main()
