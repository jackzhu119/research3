from pathlib import Path
from types import SimpleNamespace
import sys
import repair
from test_controller_fix import fixture


def args_for(tmp_path,path,submit=True):
 return SimpleNamespace(package=None,plan=path,work=tmp_path/'sidecars',python=sys.executable,credentials=None,submit=submit)


def test_ready_is_exact_child_and_nonce_and_venv_path_not_resolved(tmp_path,monkeypatch):
 c,env,path,_=fixture(tmp_path);args=args_for(tmp_path,path);venv=tmp_path/'venv-python';venv.symlink_to(sys.executable);args.python=str(venv)
 monkeypatch.setattr(repair,'load_original',lambda *a,**k:env);commands=[]
 class Child:
  def __init__(self,pid):self.pid=pid
  def poll(self):return None
 def popen(command,**kwargs):
  commands.append(command);kind=command[6];pid=1000+len(commands);work=Path(command[command.index('--work')+1]);token=command[command.index('--launch-token')+1]
  value=dict(pid=pid,recipe_sha256=env['header']['recipe_sha256'],launch_token=token,memory_patch_loaded=True,
   first_slurm_observation_completed=True,paper_patch_loaded=True,original_paper_lock_held=True)
  c.atomic(work/(kind+'-ready.json'),value);return Child(pid)
 result=repair.start_repair(args,popen=popen)
 assert result['status']=='OPERATIONAL_REPAIR_READY' and len(commands)==2
 assert all(command[0]==str(venv) for command in commands)
 assert '--submit' in commands[0] and '--submit' not in commands[1]
 assert result['scientific_workers_signalled']==0


def test_immediate_child_failure_cannot_be_reported_as_ready(tmp_path,monkeypatch):
 c,env,path,_=fixture(tmp_path);monkeypatch.setattr(repair,'load_original',lambda *a,**k:env)
 class Child:
  pid=321
  def poll(self):return 2
 result=repair.start_repair(args_for(tmp_path,path),popen=lambda *a,**k:Child())
 assert result['status']=='STARTUP_FAILED' and all(x['status']=='STARTUP_FAILED' for x in result['processes'])


def test_no_submit_and_unowned_paper_are_partial_and_never_signalled(tmp_path,monkeypatch):
 c,env,path,_=fixture(tmp_path);monkeypatch.setattr(repair,'load_original',lambda *a,**k:env)
 monkeypatch.setattr(repair,'lock_is_live',lambda p:Path(p).name=='paper-upload.lock')
 def no_spawn(*a,**k):raise AssertionError('No unowned replacement or implicit controller')
 result=repair.start_repair(args_for(tmp_path,path,submit=False),popen=no_spawn)
 assert result['status']=='PARTIAL_REPAIR'
 assert {r['status'] for r in result['processes']}=={'REQUIRES_EXPLICIT_SUBMIT_REAFFIRMATION','LIVE_PROCESS_OWNERSHIP_UNCONFIRMED_NO_SIGNAL'}


def test_pid_reuse_before_signal_refuses_and_no_other_process_receives_signal(tmp_path,monkeypatch):
 c,env,path,_=fixture(tmp_path);calls=[];count=[0]
 def identity(pid,env):
  count[0]+=1
  return dict(pid=pid,uid=1,start_ticks=count[0],args=['python','old-paper'])
 result=repair.stop_owned_paper(42,env,identity_fn=identity,signal_fn=lambda *a:calls.append(a))
 assert result['status']=='PROCESS_IDENTITY_CHANGED' and not calls
