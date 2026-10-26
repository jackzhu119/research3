"""Native worker resume test with396cached sessions; no CUDA execution claim."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys

import pytest


SCRIPT=r'''
from argparse import Namespace
from pathlib import Path
import hashlib,json,os,sys,time
stage,base=sys.argv[1:]
base=Path(base);package=Path('/workspace/paper3-round9')
if stage=='old':
    sys.path.insert(0,'/workspace/paper3-r5-r8-repair')
    from r58_runtime_hook import install
else:
    sys.path.insert(0,'/workspace/paper3-r9-cumsum-repair')
    from r9_runtime_hook import install
install('paper3r9')
sys.path.insert(0,str(package));sys.path.insert(0,str(package/'tests'))
import torch
torch.set_num_threads(1);os.environ['SLURM_CPUS_PER_TASK']='1'
from paper3r9 import worker
from paper3r9.plan import build,SEEDS
from paper3r7.io import read,put,sha
from synthetic_project import write_project
class Strict(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self,function,types,args=(),kwargs=None):
        if function in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
            assert args[0].dtype==args[2].dtype,'Strict CUDA indexed-assignment contract'
        return function(*args,**(kwargs or {}))
root=base/'project';planpath=base/'software-plan.json'
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if stage=='old':
    root.mkdir(parents=True);write_project(root)
    #396distinct session identities share one small synthetic source array.
    #This tests the real loop/cache layout without creating a large EEG corpus.
    original=read(root/'S1.json');cache=read(root/'cache.json')
    template=next(s for s in cache['sessions'] if s['partition']=='meta_train')
    sessions=[]
    for i in range(396):
        sid=str(i+1)
        rows=[dict(row,session_id=sid,event_uid=f'S1-{sid}-{j}',trial_uid=f'S1-{sid}-{j}')
              for j,row in enumerate(original)]
        path=root/'events'/f'S1-{sid}.json';put(path,rows)
        sessions.append(dict(template,session_id=sid,events={'path':str(path.relative_to(root)),'sha256':sha(path)}))
    cache['sessions']=sessions+[s for s in cache['sessions'] if s['partition']!='meta_train'];put(root/'cache.json',cache)
    tasks=read(root/'status/tasks.json');tasks['W3']['cache_manifest']['sha256']=sha(root/'cache.json');put(root/'status/tasks.json',tasks)
    full=build(root,root/'runs/r9')
    selected=[]
    for t in full['tasks']:
        if t['family'] in ('feature_reference','select') or (t['family']=='core'
             and t['partition']=='holdout' and t['group'] in ('rbf_anchored','ea_rbf_anchored')):
            t=dict(t)
            if t['family']=='core':t['budgets']=[20,4]
            selected.append(t)
    plan={**full,'tasks':selected,'recipe_sha256':'software-fixture-r9-reference396-cumsum-resume'}
    assert len(selected)==24
    put(planpath,plan)
    inputs=[package/'paper3r9/gp.py',package/'paper3r9/worker.py',planpath]
    inputs.extend(root/f'source-{seed}.pt' for seed in SEEDS)
    put(base/'inputs-before.json',{str(p):hashfile(p) for p in inputs})
    calls=[]
    def fail_after_extraction(features,**kwargs):
        calls.append(len(features));assert len(features)==396*19
        raise RuntimeError('INJECTED aggregation failure after396 cached sessions')
    worker.fit_reference_geometry=fail_after_extraction
    with Strict():assert worker.run(Namespace(plan=str(planpath),task=0,device='cpu'))==1
    assert calls==[396*19]
    folder=Path(plan['output'])/selected[0]['key']
    assert read(folder/'progress.json')['session']==395
    saved={str(p):hashfile(p) for p in (folder/'reference-sessions').glob('*.pt')}
    assert len(saved)==396
    put(base/'cached-before.json',saved)
    print('OLD_AGGREGATION_FAILED_AFTER_396_CACHE_COMMITTED')
else:
    plan=read(planpath);tasks=plan['tasks'];output=Path(plan['output'])
    original_feature=worker.feature_logits;original_prefix=worker._prefix
    def forbidden(*a,**kw):raise AssertionError('Previously cached396 prefixes re-extracted')
    worker.feature_logits=forbidden;worker._prefix=forbidden
    with Strict():assert worker.run(Namespace(plan=str(planpath),task=0,device='cpu'))==0
    worker.feature_logits=original_feature;worker._prefix=original_prefix
    saved=read(base/'cached-before.json')
    assert all(hashfile(p)==value for p,value in saved.items())
    result=read(output/tasks[0]['key']/'result.json')
    assert result['reference_scope']['sessions']==396
    assert result['reference_scope']['future_queries_read'] is False
    assert result['runtime_corrections']['version']=='paper3-r9-cumsum-fix-v2'
    gp=result['runtime_corrections']['dependency_corrections'][0]
    assert gp['module']=='paper3r9.gp'
    assert gp['corrections']==['weighted_median_deterministic_cpu_cumsum']
    assert gp['source_sha256']=='5956354a457c95db4503494c74bb0e1a6e6e2f3cfaf661c5c04ca1d49ee93ad7'
    assert gp['executed_source_sha256']=='bb3dbd077632251d8fd34a43f83645c8f411f2954d8110c8d907da9e840e17ee'
    assert gp['native_source_file_changed'] is False
    with Strict():
        for i,task in enumerate(tasks[1:],1):
            assert worker.run(Namespace(plan=str(planpath),task=i,device='cpu'))==0,task['key']
            assert read(output/task['key']/'result.json')['status']=='COMPLETE'
    refs=[t for t in tasks if t['family']=='feature_reference']
    assert len(refs)==6
    assert {(t['config']['preprocessing'],t['seed']) for t in refs}=={(p,s) for p in ('raw','ea') for s in SEEDS}
    assert all(read(output/t['key']/'result.json')['reference_scope']['sessions']==396 for t in refs)
    #Completion replay preserves already committed predictions and receipt bytes.
    last=output/tasks[-1]['key']/'result.json';data=last.read_bytes()
    assert worker.run(Namespace(plan=str(planpath),task=len(tasks)-1,device='cpu'))==0
    assert last.read_bytes()==data
    assert all(hashfile(p)==value for p,value in read(base/'inputs-before.json').items())
    assert torch.are_deterministic_algorithms_enabled()
    summary={'status':'PASS','cached_sessions_reused_without_extraction':396,'all_reference_sessions':6*396,
       'completed_refs':6,'completed_selectors':12,'completed_gp_evaluations':6,'complete_tasks':24,
       'original_source_plan_checkpoint_bytes_preserved':True,'strict_cuda_index_contract_on_cpu':True,
       'actual_gpu_executed':False,'python':sys.version.split()[0],'torch':str(torch.__version__)}
    put(base/'INTEGRATION_PROOF.json',summary)
    print(json.dumps(summary))
'''


def test_native396reference_cache_resume_then_all6refs_selection_and_gp(tmp_path):
    sites=[p for p in sys.path if p.endswith('site-packages')]
    old=subprocess.run([sys.executable,'-I','-S','-c',
        'import sys;sys.path.extend('+repr(sites)+');'+SCRIPT,
        'old',str(tmp_path)],capture_output=True,text=True,timeout=65)
    assert old.returncode==0,old.stdout+old.stderr
    assert 'OLD_AGGREGATION_FAILED_AFTER_396_CACHE_COMMITTED' in old.stdout
    new=subprocess.run([sys.executable,'-I','-S','-c',
        'import sys;sys.path.extend('+repr(sites)+');'+SCRIPT,
        'new',str(tmp_path)],capture_output=True,text=True,timeout=90)
    assert new.returncode==0,new.stdout+new.stderr
    proof=json.loads((tmp_path/'INTEGRATION_PROOF.json').read_text())
    assert proof['complete_tasks']==24
    target=Path(__file__).parent/'INTEGRATION_PROOF.json'
    target.write_text(json.dumps(proof,indent=2)+'\n')
