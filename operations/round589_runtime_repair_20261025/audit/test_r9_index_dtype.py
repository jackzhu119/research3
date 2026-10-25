"""CPU regression of CUDA's indexed-assignment contract; no GPU claim."""
from pathlib import Path
import signal
import sys
sys.path.insert(0, '/workspace/paper3-round9')
sys.path.insert(0, '/workspace/paper3-round9/tests')
import pytest
import torch
from paper3r9 import worker
from paper3r7.io import read, put
from test_r9_worker_plan import fixture, SEEDS

class StrictCudaIndexedAssignment(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function in (torch.ops.aten.index_put_.default, torch.ops.aten.index_put.default):
            assert args[0].dtype == args[2].dtype, 'CUDA indexed assignment source/destination dtype mismatch'
        return function(*args, **(kwargs or {}))

def balanced_loss_fixed(logits, y):
    weights = logits.new_empty(len(y))
    for c in (0, 1):
        mask = y == c
        if not bool(mask.any()):
            raise ValueError('Both query classes are required')
        weights[mask] = .5 / mask.sum().to(dtype=logits.dtype)
    return (weights * torch.nn.functional.cross_entropy(logits, y, reduction='none')).sum()

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)}
    threads=torch.get_num_threads();det=torch.are_deterministic_algorithms_enabled()
    dtype=torch.get_default_dtype()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float32)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','1')
    monkeypatch.delenv('PAPER3R9_STOP_REQUESTED', raising=False)
    yield
    torch.set_default_dtype(dtype);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det)
    for sig,h in handlers.items():signal.signal(sig,h)

def test_original_r9_double_loss_violates_cuda_contract():
    logits=torch.tensor([[.2,-.1],[-.2,.1],[.5,.1]],dtype=torch.float64)
    y=torch.tensor([0,1,0])
    with pytest.raises(AssertionError,match='dtype mismatch'):
        with StrictCudaIndexedAssignment():worker.balanced_loss(logits,y)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('default',[torch.float32,torch.float64])
def test_fixed_loss_preserves_equation_and_gradient(dtype,default):
    torch.set_default_dtype(default)
    logits=torch.tensor([[.2,-.1],[-.2,.1],[.5,.1],[.3,-.2]],dtype=dtype,requires_grad=True)
    y=torch.tensor([0,1,0,0])
    expected_terms=torch.nn.functional.cross_entropy(logits,y,reduction='none')
    expected=.5*(expected_terms[y==0].mean()+expected_terms[y==1].mean())
    with StrictCudaIndexedAssignment():actual=balanced_loss_fixed(logits,y)
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual,logits,retain_graph=True)[0]
    e=torch.autograd.grad(expected,logits)[0]
    torch.testing.assert_close(a,e)

def test_actual_reference_selection_gp_and_head_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'balanced_loss',balanced_loss_fixed)
    plan,args=fixture(tmp_path)
    for variant in ('source_frozen','head_sgd10','head_sgd100','support_EA'):
        plan['tasks'].append({'key':variant,'family':'core','group':variant,'seed':SEEDS[0],
             'config':{'kernel':'rbf','preprocessing':'raw'},'requires':[],
             'variant':variant,'partition':'holdout','budgets':[20,4],'scenarios':['clean']})
    put(args.plan,plan)
    with StrictCudaIndexedAssignment():
        for i,task in enumerate(plan['tasks']):
            args.task=i
            assert worker.run(args)==0, task['key']
            assert read(Path(plan['output'])/task['key']/'result.json')['status']=='COMPLETE'
    root=Path(plan['output'])
    for variant in ('evaluate','source_frozen','head_sgd10','head_sgd100','support_EA'):
        for budget in (20,4):
            row=read(root/variant/f'evaluations/holdout/B{budget}/clean/S3-1.json')
            assert len(row['labels'])==80
