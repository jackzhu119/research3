"""Installer and hook fixtures: no Torch, scheduler mutation or live GPU work."""
import base64
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

HERE = Path(__file__).resolve().parent
OLD = Path('/workspace/paper3-r1-hardware-fix')
BOOT = Path('/workspace/paper3-future-audit-20261025/r1/r1_spawn_bootstrap.py')
CACHE = Path('/workspace/paper3-r1-startup-cache-repair')


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.home=self.base/'home'
        self.root=self.home/'projects/paper3'
        self.runtime=self.root/'runs/training_campaign_parallel_v1'
        self.package=self.home/'paper3-r1-repairs/existing/active-round1/abcdef'
        self.package.mkdir(parents=True)
        for name in ('action_startup.py','audited_runtime.py'):
            (self.package/name).write_bytes((OLD/name).read_bytes())
        (self.package/'r1_spawn_bootstrap.py').write_bytes(BOOT.read_bytes())
        for name in ('worker.py','kernel_bridge.py','hardware_comparison.py','scheduler.py','START.py'):
            (self.package/name).write_text('original = '+repr(name)+'\n')
        (self.package/'action_startup.py').chmod(0o640)
        self.direct=self.home/'paper3-direct-train/existing'
        self.direct.mkdir(parents=True)
        (self.direct/'r1_spawn_bootstrap.py').write_bytes(BOOT.read_bytes())
        (self.direct/'campaign_recovery.py').write_text('original_recovery = True\n')
        save(self.runtime/'launch.json',{'package':str(self.package),'direct_package':str(self.direct),
            'worker_command':['python',str(self.package/'worker.py'),'--package',str(self.package)]})
        save(self.runtime/'scheduler-state.json',{'slots':{'0':{'attempts':[{'command':
            ['python','worker.py','--package',str(self.package)]}]}}})
        for relative in ('p3work/meta_training.py','p3work/source_training.py','p3work/contracts.py','tools/r2_backup.py'):
            p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original_science = True\n')
        self.checkpoint=self.root/'runs/meta/registered/last.pt'
        self.checkpoint.parent.mkdir(parents=True);self.checkpoint.write_bytes(b'original checkpoint\x00')
        self.other=self.root/'runs/followup_20261024_r7/result.json'
        save(self.other,{'status':'COMPLETE','untouched':True})
        self.module=load(HERE/'delivery_template.py','r1_resume_delivery_'+self.base.name)
        stub=lambda name:('import builtins\n'
            'builtins._r1_delivery_loads.append('+repr(name)+')\n'
            'def install(root):\n'
            '    builtins._r1_delivery_installs.append('+repr(name)+')\n')
        self.payload={'schema_version':self.module.SCHEMA,
            'helpers':{'r1_clock_recovery.py':stub('clock'),'r1_driver_compat.py':stub('driver'),
                       'node_refresh.py':(OLD/'node_refresh.py').read_text()},
            'lineage_sha256':{'action_startup.py':self.module.sha((OLD/'action_startup.py').read_bytes()),
                'audited_runtime.py':self.module.sha((OLD/'audited_runtime.py').read_bytes()),
                'r1_spawn_bootstrap.py':self.module.sha(BOOT.read_bytes())},
            'existing_digest_helper_sha256':self.module.sha((CACHE/'r1_digest_cache.py').read_bytes()),
            'startup_install':['r1_driver_compat.py','r1_clock_recovery.py'],
            'bootstrap_install':['r1_driver_compat.py','r1_clock_recovery.py'],
            'runtime_install':['r1_driver_compat.py']}
        self.configure()

    def configure(self):
        data=json.dumps(self.payload,sort_keys=True).encode()
        self.module.PAYLOAD_B85=base64.b85encode(zlib.compress(data)).decode()
        self.module.PAYLOAD_SHA256=hashlib.sha256(data).hexdigest()

    def apply(self, *flags):
        return self.module.main(['--home',str(self.home),'--root',str(self.root),*flags])

    def preserved(self):
        paths=[p for p in self.package.glob('*.py') if p.name not in self.module.MARKERS]
        paths += [self.direct/'campaign_recovery.py',self.runtime/'launch.json',self.runtime/'scheduler-state.json',
                  self.checkpoint,self.other]
        paths += list((self.root/'p3work').glob('*.py'))+[self.root/'tools/r2_backup.py']
        return {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}

    def add_digest_hooks(self):
        installer=load(CACHE/'delivery_template.py','old_digest_delivery_'+self.base.name)
        helper=self.home/'paper3-runtime-fixes/r1-digest-cache-v1/r1_digest_cache.py'
        helper.parent.mkdir(parents=True);helper.write_bytes((CACHE/'r1_digest_cache.py').read_bytes())
        p=self.package/'action_startup.py'
        p.write_bytes(installer.patched_startup(p.read_bytes(),helper))
        for p in (self.package/'r1_spawn_bootstrap.py',self.direct/'r1_spawn_bootstrap.py'):
            p.write_bytes(installer.patched_bootstrap(p.read_bytes(),helper))
        return helper

    def test_existing_digest_overlays_remain_and_apply_is_idempotent(self):
        self.add_digest_hooks()
        before=self.preserved()
        originals={p:p.read_bytes() for p in (self.package/'action_startup.py',self.package/'audited_runtime.py',
            self.package/'r1_spawn_bootstrap.py',self.direct/'r1_spawn_bootstrap.py')}
        self.assertEqual(self.apply(),0)
        after={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in originals}
        self.assertTrue(all(after[p][0].startswith(originals[p]) for p in originals))
        self.assertEqual(self.apply('--apply'),0)
        self.assertEqual(after,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in originals})
        self.assertEqual(before,self.preserved())
        self.assertEqual((self.package/'action_startup.py').stat().st_mode & 0o777,0o640)
        receipt=json.loads((self.home/'paper3-runtime-fixes/r1-resume-v3/APPLIED.json').read_text())
        self.assertEqual(receipt['owned_stage_bootstrap_files'],2)
        for k in ('scientific_files_changed','checkpoint_files_changed','plan_files_changed','jobs_submitted','jobs_cancelled','signals_sent','other_rounds_changed'):
            self.assertEqual(receipt[k],0)

    def test_all_entries_validated_before_any_existing_module_is_switched(self):
        original=(self.package/'action_startup.py').read_bytes()
        bad=self.direct/'r1_spawn_bootstrap.py';bad.write_bytes(bad.read_bytes()+b'\n# unknown operational change\n')
        with self.assertRaisesRegex(ValueError,'Unrecognized R1 operational source'):
            self.apply()
        self.assertEqual((self.package/'action_startup.py').read_bytes(),original)
        self.assertFalse((self.home/'paper3-runtime-fixes/r1-resume-v3'/self.module.PAYLOAD_SHA256[:20]).exists())

    def test_unknown_existing_digest_helper_is_not_silently_overlaid(self):
        helper=self.add_digest_hooks();original=(self.package/'action_startup.py').read_bytes()
        helper.write_bytes(helper.read_bytes()+b'\n# unrecognized\n')
        with self.assertRaisesRegex(ValueError,'digest helper lineage'):
            self.apply()
        self.assertEqual((self.package/'action_startup.py').read_bytes(),original)

    def test_launch_history_and_shallow_fallback_discover_original_packages(self):
        old=self.home/'paper3-parallel-migrations/old/round1';old.mkdir(parents=True)
        for name in ('worker.py','kernel_bridge.py','action_startup.py'):
            (old/name).write_bytes((self.package/name).read_bytes())
        save(self.runtime/'scheduler-state.json',{'slots':{'1':{'attempts':[{'command':['python','--package',str(old)]}]}}})
        self.assertEqual(self.module.packages(self.runtime,self.home),sorted([old,self.package]))
        (self.runtime/'launch.json').unlink();(self.runtime/'scheduler-state.json').unlink()
        self.assertEqual(self.module.packages(self.runtime,self.home),sorted([old,self.package]))

    def test_cached_helper_modules_load_once_across_runtime_startup_and_owned_hooks(self):
        self.apply()
        builtins._r1_delivery_loads=[];builtins._r1_delivery_installs=[]
        runtime=load(self.package/'audited_runtime.py','r1_runtime_fixture_'+self.base.name)
        transfer=types.SimpleNamespace(validate_transfer=lambda *a,**k:True)
        self.assertIsNone(runtime.install(self.root,transfer_module=transfer))
        startup=load(self.package/'action_startup.py','r1_startup_fixture_'+self.base.name)
        for _ in range(2):
            with startup.active(self.root,{'kind':'noop'},traceback_interval=0):
                pass
        bootstrap=load(self.direct/'r1_spawn_bootstrap.py','r1_boot_fixture_'+self.base.name)
        value={'status':'BOOTSTRAPPED'}
        bootstrap._paper3_r1_before_resume_owned_stage=lambda root:value
        self.assertIs(bootstrap.bootstrap_owned_stage(self.root),value)
        self.assertEqual(builtins._r1_delivery_loads,['driver','clock'])
        self.assertEqual(builtins._r1_delivery_installs,['driver','driver','clock','driver','clock','driver','clock'])

    def test_bootstrap_preserves_not_applicable_and_original_exception(self):
        self.apply();builtins._r1_delivery_loads=[];builtins._r1_delivery_installs=[]
        bootstrap=load(self.direct/'r1_spawn_bootstrap.py','r1_boot_result_fixture_'+self.base.name)
        value={'status':'NOT_APPLICABLE_NO_PARALLEL_LAUNCH'}
        bootstrap._paper3_r1_before_resume_owned_stage=lambda root:value
        self.assertIs(bootstrap.bootstrap_owned_stage(self.root),value)
        def fail(root):raise RuntimeError('original bootstrap error')
        bootstrap._paper3_r1_before_resume_owned_stage=fail
        with self.assertRaisesRegex(RuntimeError,'original bootstrap error'):
            bootstrap.bootstrap_owned_stage(self.root)
        self.assertEqual(builtins._r1_delivery_loads,[])

    def test_standalone_installer_never_imports_compute_network_or_scheduler_modules(self):
        self.add_digest_hooks();before=self.preserved()
        source=(HERE/'delivery_template.py').read_text().replace('__PAYLOAD_B85__',self.module.PAYLOAD_B85).replace('__PAYLOAD_SHA256__',self.module.PAYLOAD_SHA256)
        script=self.base/'FIX-R1.py';script.write_text(source)
        probe='''import runpy,sys
class Forbidden:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","numpy","subprocess","requests","socket") or fullname in ("urllib.request","http.client"):
   raise AssertionError("Forbidden installer import: "+fullname)
sys.meta_path.insert(0,Forbidden())
sys.argv=[sys.argv[1],"--home",sys.argv[2],"--root",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        result=subprocess.run([sys.executable,'-I','-S','-B','-u','-c',probe,str(script),str(self.home),str(self.root)],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('R1_RESUME_REPAIR_COMPLETE',result.stdout)
        self.assertEqual(before,self.preserved())

    def test_refresh_uses_existing_three_allocations_only_and_is_explicit(self):
        for slot in range(3):save(self.runtime/'worker-status'/('slot-%d.json'%slot),{'job_id':str(100+slot),'child_pid':200+slot})
        with patch.object(self.module,'refresh_current') as refresh:
            self.apply()
            refresh.assert_not_called()
        calls=[]
        def runner(command,**kwargs):
            calls.append((command,kwargs))
            return types.SimpleNamespace(returncode=0,stdout=json.dumps({'status':'CURRENT_CHILD_KEPT'}))
        values=self.module.refresh_current(self.runtime,self.base/'node_refresh.py',runner=runner)
        self.assertEqual(len(calls),3)
        self.assertTrue(all(v['status']=='CURRENT_CHILD_KEPT' for v in values))
        for command,kwargs in calls:
            self.assertEqual(command[0],'srun')
            self.assertIn('--gres=none',command)
            self.assertEqual(kwargs['timeout'],40)
            self.assertFalse(any(x in ('sbatch','scancel') for x in command))

    def test_proven_refresh_keeps_current_training_and_checks_pid_reuse(self):
        node=load(OLD/'node_refresh.py','r1_node_refresh_fixture_'+self.base.name)
        key='meta-task';token='a'*32;job='123';child=200;parent=100;claimed=2000
        expected=self.runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'action.json'
        action={'key':key,'expected_receipt':'runs/meta/registered/receipt.json'}
        save(expected,{'action':action,'claim':{'token':token,'job_id':job,'slot':0,'pid':parent,'claimed_utc':claimed}})
        save(self.runtime/'worker-status/slot-0.json',{'job_id':job,'slot':0,'pid':parent,'child_pid':child,'action_key':key,'claim_token':token,'expected_receipt':action['expected_receipt']})
        observed={'uid':os.getuid(),'ppid':parent,'start_ticks':10,'state':'S','argv':['python','worker.py','--child','--action-file',str(expected),'--runtime-dir',str(self.runtime),'--slot','0']}
        progress=self.checkpoint.parent/'progress.json';save(progress,{'status':'OUTER_ATTEMPT_DISPATCHED','outer_updates_completed':350})
        os.utime(progress,(2100,2100));signals=[]
        with patch.object(node,'process',return_value=observed):
            result=node.refresh(self.runtime,job,0,allocation_job=job,kill=lambda *x:signals.append(x))
        self.assertEqual(result['status'],'CURRENT_CHILD_KEPT');self.assertEqual(signals,[])
        os.utime(progress,(1000,1000));reused=copy.deepcopy(observed);reused['start_ticks']=999
        with patch.object(node,'process',side_effect=[observed,reused]):
            result=node.refresh(self.runtime,job,0,allocation_job=job,kill=lambda *x:signals.append(x))
        self.assertEqual(result['status'],'WAIT_NEXT_CHILD');self.assertEqual(signals,[])
        with patch.object(node,'process',return_value=observed):
            result=node.refresh(self.runtime,job,0,allocation_job=job,kill=lambda *x:signals.append(x))
        self.assertEqual(result['status'],'PRETRAIN_CHILD_REFRESH_REQUESTED')
        self.assertEqual(signals,[(child,node.signal.SIGINT)])

    def test_explicit_refresh_runs_only_after_verified_install_and_records_only_requested_signals(self):
        def refresh(runtime,node_helper):
            self.assertTrue((self.home/'paper3-runtime-fixes/r1-resume-v3/APPLIED.json').is_file())
            self.assertEqual(node_helper.read_bytes(),(OLD/'node_refresh.py').read_bytes())
            self.assertIn(self.module.MARKERS['action_startup.py'],(self.package/'action_startup.py').read_text())
            return [{'slot':0,'status':'PRETRAIN_CHILD_REFRESH_REQUESTED'},
                    {'slot':1,'status':'CURRENT_CHILD_KEPT'}, {'slot':2,'status':'CURRENT_CHILD_KEPT'}]
        before=self.preserved()
        with patch.object(self.module,'refresh_current',side_effect=refresh) as called:
            self.assertEqual(self.apply('--refresh-pretraining'),0)
            called.assert_called_once()
        receipt=json.loads((self.runtime/'resume-repair.json').read_text())
        self.assertEqual(receipt['signals_sent'],1)
        self.assertEqual(receipt['jobs_cancelled'],0)
        self.assertEqual(receipt['jobs_submitted'],0)
        self.assertEqual(before,self.preserved())
        bad=self.direct/'r1_spawn_bootstrap.py';bad.write_bytes(bad.read_bytes()+b'\n# unknown\n')
        with patch.object(self.module,'refresh_current') as called:
            with self.assertRaises(ValueError):self.apply('--refresh-pretraining')
            called.assert_not_called()


if __name__=='__main__':unittest.main()
