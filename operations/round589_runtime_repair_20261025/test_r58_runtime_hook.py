"""Runtime corrections preserve original files and saved recipe identities."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import r58_runtime_hook as hook


def native_source(round_number):
    root = ('/workspace/paper3-round%d-hpcgpu' % round_number if round_number in (4, 5) else
            '/workspace/paper3-round%d' % round_number)
    path = Path(root) / ('paper3r%d/worker.py' % round_number)
    source = path.read_text()
    if round_number == 8:
        # The reviewed local reference contains the fix; restore original
        # published bytes in the test package, exactly and without changing it.
        source = source.replace(hook.NEW_BALANCED_LOSS, hook.OLD_BALANCED_LOSS, 1)
        source = source.replace(hook.NEW_TRAIN_WEIGHTS, hook.OLD_TRAIN_WEIGHTS, 1)
        assert hashlib.sha256(source.encode()).hexdigest() == (
            'f93174dd373f9b203890a98543d19f37b7fe6d30c96e9fd2a0fde774d1efb1e2')
    return source


@pytest.mark.parametrize('round_number,changes', [(4, 1), (5, 1), (8, 2), (9, 1)])
def test_only_exact_known_source_spans_are_changed_and_idempotent(round_number, changes):
    source = native_source(round_number)
    fixed, applied = hook.corrected_source('paper3r%d.worker' % round_number, source)
    assert len(applied) == changes
    assert hook.corrected_source('paper3r%d.worker' % round_number, fixed) == (fixed, [])
    compile(fixed, '<fixed-worker>', 'exec')
    assert hook.corrected_source('paper3r7.worker', source) == (source, [])
    malformed = (source.replace('"tangent_profiled"))', '"other_variant"))')
                 if round_number in (4, 5) else
                 source.replace('.5 / mask.sum()', '.6 / mask.sum()'))
    with pytest.raises(RuntimeError, match='Unexpected native'):
        hook.corrected_source('paper3r%d.worker' % round_number,
                              malformed)


def test_r5_environment_is_ready_before_torch_and_other_rounds_unaffected():
    script = r'''
import json,os,sys
import r58_runtime_hook as h
assert 'torch' not in sys.modules
os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
h.install('paper3r7')
assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
h.install('paper3r8')
assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
h.install('paper3r5')
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
assert 'torch' not in sys.modules
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
h.install('paper3r5')
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':16:8'
os.environ.pop('CUBLAS_WORKSPACE_CONFIG')
h.install('paper3r6')
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
assert 'paper3r6.worker' not in next(f for f in sys.meta_path if getattr(f,'paper3_runtime_correction',None)==h.VERSION).targets
assert 'torch' not in sys.modules
print('R5_ENVIRONMENT_PASS')
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'R5_ENVIRONMENT_PASS' in result.stdout


@pytest.mark.parametrize('round_number', [4, 5, 8, 9])
def test_real_native_worker_import_and_dtype_contract_preserve_source(tmp_path, round_number):
    source_root = Path('/workspace/paper3-round%d-hpcgpu' % round_number if round_number in (4, 5) else
                       '/workspace/paper3-round%d' % round_number)
    package = tmp_path / 'native'
    directories = ('p3work', 'paper3r%d' % round_number) if round_number in (4, 5) else (
        'p3work', 'paper3r7', 'paper3r%d' % round_number)
    for directory in directories:
        target = package / directory
        target.mkdir(parents=True, exist_ok=True)
        for path in (source_root / directory).glob('*.py'):
            shutil.copy2(path, target / path.name)
    worker_path = package / ('paper3r%d/worker.py' % round_number)
    original = native_source(round_number).encode()
    worker_path.write_bytes(original)
    expected_hash = hashlib.sha256(original).hexdigest()
    script = r'''
import hashlib,importlib,importlib.util,json,os,pathlib,sys
package,round_number,source_hash,test_root=sys.argv[1:]
sys.path.insert(0,package)
import r58_runtime_hook as h
assert 'torch' not in sys.modules
roundmodule='paper3r'+round_number
h.install(roundmodule);h.install(roundmodule)
assert 'torch' not in sys.modules
assert len([f for f in sys.meta_path if getattr(f,'paper3_runtime_correction',None)==h.VERSION])==1
w=importlib.import_module(roundmodule+'.worker')
import torch
from torch.utils._python_dispatch import TorchDispatchMode
torch.set_num_threads(1)
class Strict(TorchDispatchMode):
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  if func in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
   assert args[0].dtype==args[2].dtype,(args[0].dtype,args[2].dtype)
  return func(*args,**(kwargs or {}))
previous=torch.get_default_dtype()
for default in (() if round_number in ('4','5') else (torch.float32,torch.float64)):
 torch.set_default_dtype(default)
 for dtype in (torch.float32,torch.float64):
  y=torch.tensor([0,1,1,0,1],dtype=torch.long)
  logits=torch.tensor([[.2,-.5],[-.1,.7],[.3,.2],[1.,-.3],[.5,.8]],dtype=dtype,requires_grad=True)
  expected_logits=logits.detach().clone().requires_grad_(True)
  terms=torch.nn.functional.cross_entropy(expected_logits,y,reduction='none')
  expected=.5*(terms[y==0].mean()+terms[y==1].mean())
  with Strict():actual=w.balanced_loss(logits,y)
  torch.testing.assert_close(actual,expected)
  actual.backward();expected.backward()
  torch.testing.assert_close(logits.grad,expected_logits.grad)
torch.set_default_dtype(previous)
if round_number in ('4','5'):
 native_plan=importlib.import_module(roundmodule+'.plan')
 plan=native_plan.make_plan(package,pathlib.Path(test_root).parent/'planned-output')
 ea_tasks=[t for t in plan['tasks'] if t.get('variant')=='support_EA']
 assert len(ea_tasks)==3
 for task in ea_tasks:
  config,variant=w._configuration(task)
  assert variant=='support_EA' and config.variant=='head_newton'
 assert native_plan.make_plan(package,pathlib.Path(test_root).parent/'planned-output')['recipe_sha256']==plan['recipe_sha256']
 if round_number=='5':assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8',':16:8')
if round_number=='8':
 spec=importlib.util.spec_from_file_location('r8fixture','/workspace/paper3-round8/tests/test_worker.py')
 fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
 pathlib.Path(test_root).mkdir()
 with Strict():fixture.test_real_training_and_reference_evaluation_obey_cuda_indexing_contract(pathlib.Path(test_root))
file=pathlib.Path(w.__file__)
assert hashlib.sha256(file.read_bytes()).hexdigest()==source_hash
assert w.__paper3_runtime_correction__['source_sha256']==source_hash
assert w.__paper3_runtime_correction__['source_sha256']!=w.__paper3_runtime_correction__['executed_source_sha256']
assert w.__paper3_runtime_correction__['runtime_hook_sha256']==h.HOOK_SHA256
receipt_root=pathlib.Path(test_root).parent/'receipts';receipt_root.mkdir()
for name in ('result.json','runtime.json','error.json','latest.json'):
 value={'identity':'original-identity','task':{'key':'original-key'},'status':'CHECK'}
 w.put(receipt_root/name,value)
 saved=json.loads((receipt_root/name).read_text())
 assert saved['identity']==value['identity'] and saved['task']==value['task']
 assert 'runtime_corrections' not in value
 if name!='latest.json':assert saved['runtime_corrections']==w.__paper3_runtime_correction__
 else:assert saved==value
h.install(roundmodule)
assert importlib.import_module(roundmodule+'.worker') is w
assert hashlib.sha256(file.read_bytes()).hexdigest()==source_hash
print(json.dumps({'event':'RUNTIME_HOOK_NATIVE_WORKER_PASS','round':int(round_number),'receipt':w.__paper3_runtime_correction__},sort_keys=True))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(package),
        str(round_number), expected_hash, str(tmp_path / 'flow')],
        cwd=Path(__file__).parent, env=dict(os.environ, SLURM_CPUS_PER_TASK='1',
        PYTHONDONTWRITEBYTECODE='1'), text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RUNTIME_HOOK_NATIVE_WORKER_PASS' in result.stdout
    assert worker_path.read_bytes() == original
    assert not list(worker_path.parent.glob('__pycache__/worker.*.pyc'))
