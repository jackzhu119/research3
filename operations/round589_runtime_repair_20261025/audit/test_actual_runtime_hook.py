"""Exercise the published correction hook against original scientific modules."""
from pathlib import Path
import subprocess
import sys
import pytest

CODE=r'''
import hashlib,json,os,signal,sys,tempfile
from pathlib import Path
round_number=int(sys.argv[1])
package=Path('/workspace/paper3-round9' if round_number==9 else '/workspace/paper3-round5')
native=package/f'paper3r{round_number}/worker.py'
before=native.read_bytes()
sys.path.insert(0,'/workspace/paper3-r5-r8-repair')
from r58_runtime_hook import install
os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
install(f'paper3r{round_number}')
sys.path.insert(0,str(package));sys.path.insert(0,str(package/'tests'))
sys.path.append(str(Path(sys.executable).absolute().parent.parent/'lib/python3.12/site-packages'))
import torch
assert round_number!=5 or os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
torch.set_num_threads(1)
os.environ['SLURM_CPUS_PER_TASK']='1'
class Strict(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self,function,types,args=(),kwargs=None):
        if function in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
            assert args[0].dtype==args[2].dtype,'Strict CUDA assignment mismatch'
        return function(*args,**(kwargs or {}))
with tempfile.TemporaryDirectory() as temporary:
    if round_number==9:
        from paper3r9 import worker
        from test_r9_worker_plan import fixture,SEEDS
        from paper3r7.io import put,read
        plan,args=fixture(Path(temporary))
        for variant in ('source_frozen','head_sgd10','head_sgd100','support_EA'):
            plan['tasks'].append({'key':variant,'family':'core','group':variant,
               'seed':SEEDS[0],'config':{'kernel':'rbf','preprocessing':'raw'},'requires':[],
               'variant':variant,'partition':'holdout','budgets':[20,4],'scenarios':['clean']})
        put(args.plan,plan)
        with Strict():
            for i,task in enumerate(plan['tasks']):
                args.task=i
                assert worker.run(args)==0,task['key']
                receipt=read(Path(plan['output'])/task['key']/'result.json')
                assert receipt['status']=='COMPLETE'
                assert receipt['runtime_corrections']['module']=='paper3r9.worker'
                assert receipt['runtime_corrections']['native_source_file_changed'] is False
        for variant in ('evaluate','source_frozen','head_sgd10','head_sgd100','support_EA'):
            for budget in (20,4):
                row=read(Path(plan['output'])/variant/f'evaluations/holdout/B{budget}/clean/S3-1.json')
                assert len(row['labels'])==80
        completed=7
    else:
        from paper3r5 import worker
        from paper3r5.plan import make_plan
        from test_worker import plan_fixture
        from paper3r5.io import put,read
        plan,args=plan_fixture(Path(temporary))
        actual=make_plan(temporary,str(Path(temporary)/'unused'))
        for task in actual['tasks']:
            assert worker._configuration(task)[1]==task['variant']
        plan['tasks']=[t for t in actual['tasks'] if t['variant']=='support_EA']
        for task in plan['tasks']:task['budgets']=[20,4]
        put(args.plan,plan)
        old_plan=Path(args.plan).read_bytes()
        def forbidden(*a,**kw):raise AssertionError('EA entered optimizer')
        worker.fit=forbidden;worker.train_loop=forbidden
        with Strict():
            for i,task in enumerate(plan['tasks']):
                args.task=i
                assert worker.run(args)==0,task['key']
                folder=Path(plan['output'])/task['key']
                receipt=read(folder/'result.json')
                assert receipt['runtime_corrections']['module']=='paper3r5.worker'
                assert receipt['runtime_corrections']['native_source_file_changed'] is False
                for budget in (20,4):
                    row=read(folder/f'evaluations/holdout/B{budget}/clean/S3-1.json')
                    assert row['variant']=='support_EA'
                    assert row['reference_fit']['uses_support_labels'] is False
                    assert row['reference_fit']['optimizer_updates']==0
        assert Path(args.plan).read_bytes()==old_plan
        completed=3
    assert native.read_bytes()==before
    print(json.dumps({'round':round_number,'complete_synthetic_tasks':completed,
         'native_source_sha256':hashlib.sha256(before).hexdigest(),
         'native_source_bytes_preserved':True,'cuda_indexed_contract_enforced_on_cpu':True,
         'actual_school_gpu_executed':False}))
'''

@pytest.mark.parametrize('round_number',[5,9])
def test_actual_hook_original_source_end_to_end(round_number):
    result=subprocess.run([sys.executable,'-I','-S','-c',CODE,str(round_number)],
        capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"native_source_bytes_preserved": true' in result.stdout
