"""Short actual-configuration execution audit; only synthetic software fixtures."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
import time

p=argparse.ArgumentParser();p.add_argument('round',type=int);p.add_argument('--only-list',action='store_true');a=p.parse_args()
base=Path('/workspace/paper3-round%d-hpcgpu'%a.round)
sys.path[:0]=[str(base),str(base/'tests'),'/workspace/paper3-r5-r8-repair']
import r58_runtime_hook
r58_runtime_hook.install('paper3r%d'%a.round)
import torch
from torch.utils._python_dispatch import TorchDispatchMode
torch.set_num_threads(1)
alg=importlib.import_module('paper3r%d.algorithm'%a.round)
worker=importlib.import_module('paper3r%d.worker'%a.round)
plans=importlib.import_module('paper3r%d.plan'%a.round)
test_worker=importlib.import_module('test_worker')
io=importlib.import_module('paper3r%d.io'%a.round)
baseline=importlib.import_module('paper3r%d.baseline'%a.round)
from p3work.spatial_geometry import initialize_dictionary
from p3work.adapt import balanced_bce

class CudaIndexContract(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
            if args[0].dtype!=args[2].dtype:
                raise RuntimeError('Strict CUDA index_put dtype mismatch: %s versus %s'%(args[0].dtype,args[2].dtype))
        return func(*args,**kwargs)

work=Path(tempfile.mkdtemp(prefix='paper3-r%d-config-audit-'%a.round))
plan=plans.make_plan(work,work/'out')
unique={}
for task in plan['tasks']:
    cfg,variant=worker._configuration(task)
    marker=json.dumps({'config':task['config'],'variant':variant},sort_keys=True)
    unique.setdefault(marker,(task,cfg,variant))
print(json.dumps({'round':a.round,'tasks':len(plan['tasks']),'configurations':len(unique),'variants':sorted({v for _,_,v in unique.values()})}),flush=True)
if a.only_list:raise SystemExit(0)
test_worker.write_project(work)
source,geometry,identity=worker.load_source(work,20261018,'cpu')
catalog=worker.Catalog(work,'meta_train',cache_gb=.1)
episode=catalog.materialize(catalog.plans[0],'cpu',budget=20)
results=[]
for task,cfg,variant in unique.values():
    started=time.monotonic()
    b=initialize_dictionary(cfg.rank,20261018).requires_grad_(True)
    try:
        with CudaIndexContract():
            if variant=='support_EA':
                fitted,_=baseline.fit_reference(source,geometry,episode.support,'support_EA')
                logits=baseline.reference_logits(source,geometry,fitted,episode.query_x[:4])
            else:
                fitted=alg.fit(source,geometry,b,episode.support,cfg,create_graph=task['family']=='basis')
                logits=alg.predict(source,geometry,b,fitted,episode.query_x[:4])
            assert logits.shape==(4,2) and bool(torch.isfinite(logits).all())
            grad=None
            if task['family']=='basis':
                loss=balanced_bce(logits,episode.query_y[:4])
                grad=torch.autograd.grad(loss,b,allow_unused=False)[0]
                assert bool(torch.isfinite(grad).all())
        result={'task':task['key'],'variant':variant,'geometry':getattr(cfg,'geometry',None),'status':'PASS','seconds':round(time.monotonic()-started,3),'basis_gradient_checked':grad is not None}
    except Exception as error:
        import traceback
        result={'task':task['key'],'variant':variant,'status':'FAIL','error_type':type(error).__name__,'reason':str(error),'traceback':traceback.format_exc()}
    results.append(result);print(json.dumps(result),flush=True)
target=Path('/workspace/paper3-future-audit-20261025/r456/configs-r%d.json'%a.round)
target.write_text(json.dumps({'round':a.round,'configurations':len(results),'passed':sum(x['status']=='PASS' for x in results),'results':results},indent=2)+'\n')
raise SystemExit(any(x['status']!='PASS' for x in results))
