import json,os,tempfile,unittest,types
from pathlib import Path
from unittest.mock import patch
import apply_fix as fix

class RestartTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'project';self.runtime=self.root/'runs/training_campaign_parallel_v1';self.runtime.mkdir(parents=True)
  self.old=self.base/'combined/round1';self.old.mkdir(parents=True)
  for name in ('START.py','worker.py','kernel_bridge.py','audited_runtime.py','hardware_comparison.py','action_startup.py','scheduler.py'):(self.old/name).write_text('x = 1\n')
  self.marker='paper3r1p-'+'a'*16
  self.launch={'package':str(self.old),'root':str(self.root),'worker_command':['python',str(self.old/'worker.py'),'--package',str(self.old),'--root',str(self.root),'--slot','{slot}'],'cpus':12,'partition':'hpc_gpu','memories':['43G','43G','42G']}
  (self.runtime/'launch.json').write_text(json.dumps(self.launch));(self.runtime/'scheduler-state.json').write_text(json.dumps({'marker':self.marker,'slots':{}}))
 def tearDown(self):self.temp.cleanup()
 def standalone(self):return fix.standalone_launch(self.runtime,base=self.base/'dedicated')
 def test_standalone_changes_only_launch_and_copies_outside_combined(self):
  before=(self.runtime/'scheduler-state.json').read_bytes();p,packages,v=self.standalone()
  self.assertNotIn(str(self.base/'combined'),str(p));self.assertEqual(v['package'],str(p));self.assertEqual(fix.option(v['worker_command'],'--package'),str(p));self.assertIn(str(p/'worker.py'),v['worker_command']);self.assertEqual(v['partition'],'hpc_gpu');self.assertEqual(v['memories'],self.launch['memories']);self.assertEqual((self.runtime/'scheduler-state.json').read_bytes(),before)
  (self.old/'audited_runtime.py').write_text('restored_old = True\n');self.assertEqual((p/'audited_runtime.py').read_text(),'x = 1\n')
  again=self.standalone()[0];self.assertEqual(p,again)
 def test_only_owned_old_jobs_are_replaced_new_jobs_kept(self):
  p,packages,_=self.standalone();slots={}
  for slot,job,package in ((0,'100',self.old),(1,'101',self.old),(2,'104',p)):
   name=self.marker+'-s%d-a1'%slot;slots[str(slot)]={'attempts':[{'job_id':job,'name':name,'command':['worker','--package',str(package)]}]}
  (self.runtime/'scheduler-state.json').write_text(json.dumps({'marker':self.marker,'slots':slots}));calls=[]
  def run(command,**kwargs):
   calls.append(command)
   if command[0]=='squeue':return types.SimpleNamespace(returncode=0,stdout='\n'.join(['100|'+self.marker+'|RUNNING|'+self.marker+'-s0-a1','101|'+self.marker+'|PENDING|'+self.marker+'-s1-a1','102|paper3r6p-xx|RUNNING|other','104|'+self.marker+'|RUNNING|'+self.marker+'-s2-a1']))
   return types.SimpleNamespace(returncode=0,stdout='')
  self.assertEqual(fix.replace_old_jobs(self.runtime,p,packages,runner=run),['100','101']);self.assertEqual(calls[1],['scancel','--signal=USR1','--batch','100']);self.assertEqual(calls[2],['scancel','101']);self.assertEqual(len(calls),3)
 def test_remote_controller_lock_does_not_cancel_jobs(self):
  p,packages,_=self.standalone();signals=[]
  self.assertFalse(fix.ensure_standalone_controller(self.runtime,p,packages,held=lambda p:True,scanner=lambda *a:[],kill=lambda *a:signals.append(a),sleep=lambda n:None));self.assertFalse(signals)
 def test_old_controller_pid_reuse_not_signalled(self):
  p,packages,_=self.standalone();old={'pid':123,'uid':os.getuid(),'start_ticks':1,'argv':['python',str(self.old/'START.py'),'--controller','--root',str(self.root)]};signals=[]
  self.assertFalse(fix.ensure_standalone_controller(self.runtime,p,packages,held=lambda p:True,scanner=lambda *a:[old],snapshot=lambda p:{**old,'start_ticks':2},kill=lambda *a:signals.append(a),sleep=lambda n:None));self.assertFalse(signals)
 def test_verified_current_standalone_controller_is_reused(self):
  p,packages,_=self.standalone();row={'pid':123,'uid':os.getuid(),'start_ticks':1,'argv':['python',str(p/'START.py'),'--controller','--root',str(self.root)]}
  self.assertTrue(fix.ensure_standalone_controller(self.runtime,p,packages,held=lambda p:True,scanner=lambda *a:[row],sleep=lambda n:None))
 def test_bad_marker_does_not_signal_anything(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[]
   with self.assertRaises(ValueError):fix.checkpoint_restart(Path(d),runner=lambda *a,**k:calls.append(a))
   self.assertFalse(calls)
 def test_controller_confirmation_precedes_any_job_change(self):
  calls=[]
  with patch.object(fix,'ensure_standalone_controller',return_value=False),patch.object(fix,'replace_old_jobs',side_effect=lambda *a,**k:calls.append(a)):
   self.assertEqual(fix.checkpoint_restart(self.runtime,standalone_base=self.base/'dedicated'),[])
  self.assertFalse(calls)

if __name__=='__main__':unittest.main()
