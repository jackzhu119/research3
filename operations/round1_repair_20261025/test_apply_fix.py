import hashlib,json,os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import apply_fix as fix
import node_refresh as node

BRIDGE='''"""keep module"""\ndef activate_runtime(*a):return 1\ndef perform_action(root,action,payload_manifest):\n    """keep action doc"""\n    root = str(root)\n    campaign = activate_runtime(root,payload_manifest)\n    if action.get("done"):return 3\n    return campaign\nrun_action = perform_action\n'''

class ApplyTests(unittest.TestCase):
 def test_bridge_wraps_after_activation_and_is_idempotent(self):
  import ast
  value=fix.patch_bridge(BRIDGE.encode());self.assertEqual(fix.patch_bridge(value),value)
  tree=ast.parse(value);fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='perform_action')
  self.assertIsInstance(fun.body[-1],ast.With);self.assertEqual(len(fun.body[-1].body),2)
  self.assertIn('run_action = perform_action',value.decode())
 def test_backup_and_scientific_files_untouched(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);runtime=base/'project/runs/training_campaign_parallel_v1';runtime.mkdir(parents=True)
   package=base/'package/round1';package.mkdir(parents=True);(package/'kernel_bridge.py').write_text(BRIDGE);(package/'worker.py').write_text('parent worker unchanged\n');(package/'audited_runtime.py').write_text('old = 1\n');os.chmod(package/'audited_runtime.py',0o640)
   (runtime/'launch.json').write_text(json.dumps({'package':str(package)}))
   source=base/'payload';source.mkdir()
   for n in fix.MODULES:(source/n).write_text('fixed = 1\n')
   checkpoint=runtime.parent/'meta/checkpoint.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'original')
   out=fix.apply(runtime,source);self.assertEqual(len(out['files_changed']),4)
   self.assertEqual(checkpoint.read_bytes(),b'original');self.assertEqual((package/'worker.py').read_text(),'parent worker unchanged\n')
   backup=next(x for x in out['files_changed'] if x['path'].endswith('audited_runtime.py'))
   self.assertEqual(Path(backup['backup']).read_text(),'old = 1\n');self.assertEqual(os.stat(package/'audited_runtime.py').st_mode&0o777,0o640)
   self.assertEqual(fix.apply(runtime,source)['files_changed'],[])
 def test_missing_srun_leaves_existing_allocation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'worker-status').mkdir();(p/'worker-status/slot-0.json').write_text(json.dumps({'job_id':'123','child_pid':10}))
   def missing(*args,**kwargs):raise FileNotFoundError('srun')
   self.assertEqual(fix.refresh_current(p,p/'node_refresh.py',runner=missing)[0]['status'],'WAIT_NEXT_CHILD')
 def test_scheduler_record_selects_already_submitted_old_package(self):
  with tempfile.TemporaryDirectory() as d:
   b=Path(d);run=b/'r';run.mkdir();packs=[]
   for name in ('old','new'):
    p=b/name;p.mkdir();(p/'kernel_bridge.py').write_text(BRIDGE);(p/'worker.py').write_text('');packs.append(p)
   (run/'launch.json').write_text(json.dumps({'package':str(packs[1])}))
   (run/'scheduler-state.json').write_text(json.dumps({'slots':{'0':{'attempts':[{'command':['python','worker.py','--package',str(packs[0])]}]}}}))
   self.assertEqual(fix.discover_packages(run),sorted(packs))
 def test_refresh_failure_never_cancels_or_submits(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'worker-status').mkdir();(p/'worker-status/slot-0.json').write_text(json.dumps({'job_id':'123','child_pid':10}))
   calls=[]
   def runner(cmd,**kw):
    calls.append(cmd);raise TimeoutError()
   # Use the actual expected subprocess timeout class.
   import subprocess
   def timeout(cmd,**kw):calls.append(cmd);raise subprocess.TimeoutExpired(cmd,40)
   out=fix.refresh_current(p,p/'node_refresh.py',runner=timeout)
   self.assertEqual(out[0]['status'],'WAIT_NEXT_CHILD');self.assertEqual(calls[0][0],'srun')
   self.assertNotIn('scancel',' '.join(calls[0]));self.assertNotIn('sbatch',' '.join(calls[0]))

class RefreshTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.runtime=self.base/'project/runs/training_campaign_parallel_v1';self.runtime.mkdir(parents=True)
  self.key='meta-task';self.token='a'*32;self.job='123';self.child=200;self.parent=100;self.claimed=2000
  expected=self.runtime/'actions'/hashlib.sha256(self.key.encode()).hexdigest()/'attempts'/self.token/'action.json';expected.parent.mkdir(parents=True)
  action={'key':self.key,'expected_receipt':'runs/meta/abc/receipt.json'};claim={'token':self.token,'job_id':self.job,'slot':0,'pid':self.parent,'claimed_utc':self.claimed};expected.write_text(json.dumps({'action':action,'claim':claim}));self.expected=expected
  self.row={'job_id':self.job,'slot':0,'pid':self.parent,'child_pid':self.child,'action_key':self.key,'claim_token':self.token,'expected_receipt':action['expected_receipt']};(self.runtime/'worker-status').mkdir();(self.runtime/'worker-status/slot-0.json').write_text(json.dumps(self.row))
  self.proc={'uid':os.getuid(),'ppid':self.parent,'start_ticks':10,'state':'S','argv':['python','worker.py','--child','--action-file',str(expected),'--runtime-dir',str(self.runtime),'--slot','0']};self.signals=[]
 def tearDown(self):self.temp.cleanup()
 def call(self,**kw):
  with patch.object(node,'process',return_value=self.proc):return node.refresh(self.runtime,self.job,0,allocation_job=self.job,kill=lambda pid,sig:self.signals.append((pid,sig)),**kw)
 def test_old_checkpoint_allows_only_child_sigint(self):
  p=self.runtime.parents[1]/'runs/meta/abc/progress.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'INTERRUPTED_NOT_COMPLETE','outer_updates_completed':350}));os.utime(p,(1000,1000))
  result=self.call();self.assertEqual(result['status'],'PRETRAIN_CHILD_REFRESH_REQUESTED');self.assertEqual(self.signals,[(200,node.signal.SIGINT)])
 def test_real_current_training_is_not_interrupted(self):
  p=self.runtime.parents[1]/'runs/meta/abc/progress.json';p.parent.mkdir(parents=True);p.write_text(json.dumps({'status':'OUTER_ATTEMPT_DISPATCHED','outer_updates_completed':350}));os.utime(p,(2100,2100))
  self.assertEqual(self.call()['status'],'CURRENT_CHILD_KEPT');self.assertEqual(self.signals,[])
 def test_process_identity_mismatch_skips(self):
  self.proc['ppid']=999;self.assertEqual(self.call()['status'],'WAIT_NEXT_CHILD');self.assertEqual(self.signals,[])
 def test_missing_worker_status_skips(self):
  (self.runtime/'worker-status/slot-0.json').unlink();self.assertEqual(self.call()['status'],'WAIT_NEXT_CHILD')
 def test_process_state_change_does_not_hide_stable_identity(self):
  import copy
  other=copy.deepcopy(self.proc);other['state']='R'
  with patch.object(node,'process',side_effect=[self.proc,other]):
   out=node.refresh(self.runtime,self.job,0,allocation_job=self.job,kill=lambda p,s:self.signals.append((p,s)))
  self.assertEqual(out['status'],'PRETRAIN_CHILD_REFRESH_REQUESTED');self.assertEqual(self.signals,[(200,node.signal.SIGINT)])
 def test_pid_reuse_is_not_signalled(self):
  import copy
  other=copy.deepcopy(self.proc);other['start_ticks']=999
  with patch.object(node,'process',side_effect=[self.proc,other]):
   out=node.refresh(self.runtime,self.job,0,allocation_job=self.job,kill=lambda *a:self.signals.append(a))
  self.assertEqual(out['status'],'WAIT_NEXT_CHILD');self.assertFalse(self.signals)
 def test_mismatched_action_token_in_argv_is_not_signalled(self):
  self.proc['argv'][self.proc['argv'].index('--action-file')+1]=str(self.expected).replace(self.token,'b'*32)
  self.assertEqual(self.call()['status'],'WAIT_NEXT_CHILD');self.assertFalse(self.signals)
 def test_progress_transition_during_identity_check_keeps_child(self):
  progress=self.runtime.parents[1]/'runs/meta/abc/progress.json';progress.parent.mkdir(parents=True)
  progress.write_text(json.dumps({'status':'INTERRUPTED_NOT_COMPLETE'}));os.utime(progress,(1000,1000));count=[0]
  def read_process(*args):
   count[0]+=1
   if count[0]==2:
    progress.write_text(json.dumps({'status':'RUNNING'}));os.utime(progress,(2100,2100))
   return self.proc
  with patch.object(node,'process',side_effect=read_process):
   out=node.refresh(self.runtime,self.job,0,allocation_job=self.job,kill=lambda *a:self.signals.append(a))
  self.assertEqual(out['status'],'CURRENT_CHILD_KEPT');self.assertFalse(self.signals)
 def test_wrong_allocation_skips(self):
  result=node.refresh(self.runtime,self.job,0,allocation_job='999',kill=lambda *a:self.signals.append(a));self.assertEqual(result['status'],'WAIT_NEXT_CHILD');self.assertFalse(self.signals)

if __name__=='__main__':unittest.main()
