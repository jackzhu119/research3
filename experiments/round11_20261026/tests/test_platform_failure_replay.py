"""Replay observed R5/R8/R9 startup failures using CPU or mocked boundaries."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from paper3r11.platform import common
from test_platform_contract import plan_fixture


@pytest.mark.parametrize('during_import', [False, True])
def test_isolated_entry_sets_cublas_before_science_and_handles_early_usr1(tmp_path,during_import):
    path,plan=plan_fixture(tmp_path)
    script='''import importlib,importlib.util,os,signal,sys
sys.path.insert(0,sys.argv[1])
from paper3r11.platform import entry
original=importlib.import_module
class Worker:
 @staticmethod
 def main(args):
  assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
  assert 'torch' not in sys.modules and 'numpy' not in sys.modules
  assert importlib.util.find_spec('numpy') is not None
  print('EXISTING_SITE_PATHS_READY_WITHOUT_TORCH')
  return 0
def load(name,*a,**k):
 if name=='paper3r11.worker':
  assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
  if sys.argv[3]=='1':os.kill(os.getpid(),signal.SIGUSR1)
  return Worker
 return original(name,*a,**k)
importlib.import_module=load
code=entry.main(['science','--plan',sys.argv[2],'--task-key','development-a'])
assert code==(75 if sys.argv[3]=='1' else 0)
if code==75:assert os.environ['PAPER3_R11_STOP_REQUESTED']=='1'
'''
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG='invalid-inherited-setting',PYTHONPATH='/ignored-untrusted')
    done=subprocess.run([sys.executable,'-I','-S','-B','-c',script,plan['package'],str(path),'1' if during_import else '0'],
                        env=env,capture_output=True,text=True,timeout=20)
    assert done.returncode==0,done.stderr
    if not during_import:assert 'EXISTING_SITE_PATHS_READY_WITHOUT_TORCH' in done.stdout


def test_balanced_loss_float_and_double_logits_accept_integer_support_labels():
    import torch
    from paper3r11.baselines import _native
    loss=_native()['adapt'].balanced_bce
    y=torch.tensor([0,1,0,1],dtype=torch.int64)
    for dtype in (torch.float32,torch.float64):
        logits=torch.tensor([[1.,0.],[0.,1.],[0.,1.],[1.,0.]],dtype=dtype,requires_grad=True)
        value=loss(logits,y);value.backward()
        assert value.dtype==dtype and bool(torch.isfinite(value))
        assert logits.grad.dtype==dtype and bool(torch.isfinite(logits.grad).all())


def test_strict_deterministic_reference_weight_cumsum_uses_cpu(monkeypatch):
    import torch
    from paper3r11.baselines import _native
    gp=_native()['gp'];original=torch.cumsum;observed=[]
    def checked(value,*args,**kwargs):
        observed.append(value.device.type)
        assert value.device.type=='cpu'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'cumsum',checked)
    old=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        x=torch.arange(32,dtype=torch.float32).reshape(8,4)
        reference=gp.fit_reference_geometry(x,weights=torch.arange(1,9,dtype=torch.float32),
                    provenance={'role':'meta_train_prefix20_only'})
        assert reference.mean.dtype==torch.float64 and reference.lengthscale>0
    finally:torch.use_deterministic_algorithms(old)
    assert observed==['cpu']
