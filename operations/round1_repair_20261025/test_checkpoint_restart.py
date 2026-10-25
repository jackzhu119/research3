import fcntl,json,tempfile,unittest,types
from pathlib import Path
import apply_fix as fix

class RestartTests(unittest.TestCase):
 def test_only_owned_running_jobs_receive_checkpoint_signal(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);marker='paper3r1p-'+'a'*16;(root/'scheduler-state.json').write_text(json.dumps({'marker':marker}));(root/'launch.json').write_text('{}')
   lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   calls=[]
   def run(command,**kwargs):
    calls.append(command)
    if command[0]=='squeue':return types.SimpleNamespace(returncode=0,stdout='\n'.join(['100|'+marker+'|RUNNING|'+marker+'-s0-a1','101|'+marker+'|PENDING|'+marker+'-s1-a1','102|another|RUNNING|otherjob','103|'+marker+'|RUNNING|unowned-name']))
    return types.SimpleNamespace(returncode=0,stdout='')
   try:self.assertEqual(fix.checkpoint_restart(root,runner=run),['100'])
   finally:lock.close()
   self.assertEqual(calls[1],['scancel','--signal=USR1','--batch','100'])
   self.assertEqual(len(calls),2)
 def test_bad_marker_does_not_signal_anything(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[]
   with self.assertRaises(ValueError):fix.checkpoint_restart(Path(d),runner=lambda *a,**k:calls.append(a))
   self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
