"""Execute original R8 workers with the already deployed correction hook."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time

BASE=Path('/workspace/paper3-round8')
EVIDENCE=Path(__file__).resolve().parent
WORK=Path(tempfile.mkdtemp(prefix='r8-future-paths-',dir=EVIDENCE))
PACKAGE=WORK/'native';PACKAGE.mkdir()
for directory in ('p3work','paper3r7','paper3r8'):
    target=PACKAGE/directory;target.mkdir()
    for file in (BASE/directory).glob('*.py'):shutil.copy2(file,target/file.name)
(PACKAGE/'data').symlink_to(BASE/'data',target_is_directory=True)
sys.path.insert(0,'/workspace/paper3-r5-r8-repair')
import r58_runtime_hook as hook
native=(PACKAGE/'paper3r8/worker.py').read_text().replace(hook.NEW_BALANCED_LOSS,hook.OLD_BALANCED_LOSS,1).replace(hook.NEW_TRAIN_WEIGHTS,hook.OLD_TRAIN_WEIGHTS,1)
assert hashlib.sha256(native.encode()).hexdigest()=='f93174dd373f9b203890a98543d19f37b7fe6d30c96e9fd2a0fde774d1efb1e2'
(PACKAGE/'paper3r8/worker.py').write_text(native)
sys.path.insert(0,str(PACKAGE));hook.install('paper3r8')
from paper3r8 import worker,plan
from paper3r8.transport import TransportConfig,init_meta_parameters
from paper3r7.data import Catalog,load_source
from paper3r7.io import put,read
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('original_r8_fixture',BASE/'tests/test_worker.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
os.environ['SLURM_CPUS_PER_TASK']='1';torch.set_num_threads(1)
class Strict(torch.utils._python_dispatch.TorchDispatchMode):
    writes=0
    def __torch_dispatch__(self,f,types,args=(),kwargs=None):
        if f in (torch.ops.aten.index_put_.default,torch.ops.aten.index_put.default):
            assert args[0].dtype==args[2].dtype,(f,args[0].dtype,args[2].dtype)
            self.writes+=1
        return f(*args,**(kwargs or {}))
root=WORK/'project';root.mkdir();fixture.write_project(root)
source,geometry,source_id=load_source(root,plan.SEEDS[0],'cpu')
cat=Catalog(root,'internal_holdout',cache_gb=.1)
full=plan.build(root,WORK/'full-output')
assert len(full['tasks'])==2835
unique={json.dumps(t.get('config',{}),sort_keys=True):t for t in full['tasks']}
results={'scope':'R8 future planned paths, original source + deployed hook','environment':{'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':np.__version__,'cuda_available':torch.cuda.is_available()},'plan_tasks':len(full['tasks']),'unique_configs':len(unique),'native_worker_sha256':hashlib.sha256(native.encode()).hexdigest(),'runtime_hook_sha256':hook.HOOK_SHA256,'checks':{},'failures':[]}
start=time.monotonic()
# Test every numerical configuration present in all2835tasks on the actual source.
ep=cat.materialize(cat.plans[0],'cpu',budget=20)
class_reference=torch.eye(10,dtype=torch.float64)*.01
with Strict() as strict:
    for i,t in enumerate(unique.values()):
        cfg=TransportConfig(**t.get('config',{}));params=init_meta_parameters(cfg,'cpu')
        params={'learned':params,'class_reference':class_reference} if cfg.kind=='fiber' else params
        fitted=worker.fit_transport(source,geometry,ep.support,cfg,params)
        assert fitted.fallback_reason is None,(t['key'],fitted.fallback_reason)
        logits=worker.predict_transport(source,geometry,fitted,ep.query_x[:7]);assert torch.isfinite(logits).all(),t['key']
        loss=worker.balanced_loss(logits.double(),ep.query_y[:7]);assert torch.isfinite(loss),t['key']
        if cfg.parameterization!='fixed':
            learner=params['learned'] if isinstance(params,dict) else params
            grad=torch.autograd.grad(loss,learner)[0];assert torch.isfinite(grad).all(),t['key']
    results['checks']['all_planned_numerical_configs']={'passed':len(unique),'strict_index_writes':strict.writes}
# Run all perturbation mechanisms with each structurally distinct fixed learner.
fixed=[('recolor',{}),('bures',{}),('metric_bures',{}),('fiber',{'beta_rule':'bounded'}),('fiber',{'fiber_control':'random'}),('fiber',{'fiber_control':'reversed'})]
checked=0
with Strict() as strict:
    for scenario in ('clean',*plan.SCENARIOS,'support_label_permutation'):
        changed,metadata=worker.perturb(ep,geometry,cat.plans[0],scenario)
        for kind,extra in fixed:
            cfg=TransportConfig(kind=kind,**extra)
            params={'class_reference':class_reference} if kind=='fiber' else None
            fitted=worker.fit_transport(source,geometry,changed.support,cfg,params)
            logits=worker.predict_transport(source,geometry,fitted,changed.query_x[:7]);assert torch.isfinite(logits).all(),(kind,scenario)
            worker.balanced_loss(logits.double(),changed.query_y[:7]);checked+=1
    results['checks']['scenario_by_learner_paths']={'passed':checked,'strict_index_writes':strict.writes}
# Whole-worker two-update train/select/dependency/evaluation across8plannedlearners.
train_tasks=[t for t in full['tasks'] if t['family']=='train' and t['seed']==plan.SEEDS[0]]
assert len(train_tasks)==8
for i,original in enumerate(train_tasks):
    tr=copy.deepcopy(original);tr.update(updates=2,validate_every=1,checkpoint_every=1,episodes_per_update=1,query_batch_size=17)
    ev=next(copy.deepcopy(t) for t in full['tasks'] if t.get('meta_key')==tr['key'] and t['partition']=='holdout' and t['family']=='core')
    ev.update(budgets=[20,4],scenarios=['clean','label_noise10'],query_batch_size=17)
    mini={'root':str(root),'output':str(WORK/f'learned-{i}'),'recipe_sha256':f'audit-learned-{i}','tasks':[tr,ev]}
    path=WORK/f'learned-{i}.json';put(path,mini)
    with Strict():
        for index in (0,1):
            assert worker.run(SimpleNamespace(plan=str(path),task=index,device='cpu'))==0,tr['key']
            receipt=read(Path(mini['output'])/mini['tasks'][index]['key']/'result.json')
            assert receipt['status']=='COMPLETE' and receipt['runtime_corrections']['runtime_hook_sha256']==hook.HOOK_SHA256
    print('LEARNED_AUDIT_COMPLETE',i,tr['key'],flush=True)
results['checks']['actual_meta_train_selection_dependency_eval']={'passed_train_tasks':8,'passed_evaluation_tasks':8,'updates_per_train':2,'evaluation_budgets':[20,4]}
# Whole-worker classreference -> eachfibercontrol, all5budgets andnull/stress.
ref=copy.deepcopy(full['tasks'][0]);fiber_tasks=[]
for group in ['fiber_zero','fiber_fixed','fiber_bounded','fiber_reversed','fiber_random']:
    item=next(copy.deepcopy(t) for t in full['tasks'] if t.get('group')==group and t['seed']==plan.SEEDS[0] and t['partition']=='holdout' and t['family']=='core')
    item['scenarios']=['clean','support_label_permutation','coupled_ks35','time_cov_plus35'];fiber_tasks.append(item)
mini={'root':str(root),'output':str(WORK/'fiber-dependencies'),'recipe_sha256':'audit-fiber','tasks':[ref,*fiber_tasks]}
path=WORK/'fiber.json';put(path,mini)
with Strict():
    for index in range(len(mini['tasks'])):
        assert worker.run(SimpleNamespace(plan=str(path),task=index,device='cpu'))==0,mini['tasks'][index]['key']
        assert read(Path(mini['output'])/mini['tasks'][index]['key']/'result.json')['status']=='COMPLETE'
results['checks']['class_reference_fiber_dependencies']={'reference_tasks_complete':1,'fiber_tasks_complete':5,'budgets':[20,4,8,12,16],'scenarios':['clean','support_label_permutation','coupled_ks35','time_cov_plus35']}
assert (PACKAGE/'paper3r8/worker.py').read_text()==native
results['native_source_unchanged']=True;results['seconds']=round(time.monotonic()-start,2);results['status']='PASS'
(EVIDENCE/'FINDINGS.json').write_text(json.dumps(results,indent=2)+'\n')
print('FINAL',json.dumps(results),flush=True)
# Rich learned/fiber cross-cohort dependencies using the bundled real cropped
# Pan/Weibo arrays; synthetic source weights remain software-test-only.
root2=WORK/'external-project';root2.mkdir();fixture.write_project(root2)
contract=read(BASE/'tests/fixtures/input_contract.json')
from paper3r7.io import digest,sha
contract_sha=digest(contract);put(root2/'contracts/input_contract.json',contract)
cache=read(root2/'cache.json');cache['input_contract_sha256']=contract_sha;put(root2/'cache.json',cache)
tasks_file=read(root2/'status/tasks.json');tasks_file['W3']['cache_manifest']['sha256']=sha(root2/'cache.json');put(root2/'status/tasks.json',tasks_file)
for seed in plan.SEEDS:
    p=root2/f'runs/source_training_v2/f0_receipt_seed-{seed}.json';value=read(p);value['signal_contract_sha256']=contract_sha;put(p,value)
    p2=root2/f'runs/training_campaign_v1/references/seed-{seed}/receipt.json';value=read(p2);value.update(signal_contract_sha256=contract_sha,source_f0_receipt_sha256=sha(p));put(p2,value)
full2=plan.build(root2,WORK/'external-dependencies');ref=copy.deepcopy(full2['tasks'][0])
learner=next(copy.deepcopy(t) for t in full2['tasks'] if t['family']=='train' and t['seed']==plan.SEEDS[0] and t['config']==plan.config('bures',parameterization='evidence',weighting='class_balanced'))
learner.update(updates=2,validate_every=1,checkpoint_every=1,episodes_per_update=1,query_batch_size=17)
ext_tasks=[]
for group in ['learned_bures_evidence_class_balanced','fiber_bounded','fiber_random']:
    for dataset in plan.DATASETS[1:]:
        item=next(copy.deepcopy(t) for t in full2['tasks'] if t['family']=='core' and t['seed']==plan.SEEDS[0] and t['group']==group and t['partition']==dataset)
        item.update(budgets=[20,4],scenarios=['clean','label_noise20','time_cov_minus35'],query_batch_size=17);ext_tasks.append(item)
mini={'root':str(root2),'output':str(WORK/'external-dependencies'),'recipe_sha256':'audit-r8-external-chain','tasks':[ref,learner,*ext_tasks]}
path=WORK/'external-chain.json';put(path,mini)
with Strict():
    assert worker.run(SimpleNamespace(plan=str(path),task=0,device='cpu'))==0
    original_train=worker.train_step
    def stop_after_update(*args,**kwargs):
        value=original_train(*args,**kwargs);os.kill(os.getpid(),signal.SIGUSR1);return value
    worker.train_step=stop_after_update
    assert worker.run(SimpleNamespace(plan=str(path),task=1,device='cpu'))==75
    learner_dir=Path(mini['output'])/learner['key'];assert read(learner_dir/'latest.json')['outer_update']==1
    worker.train_step=original_train
    assert worker.run(SimpleNamespace(plan=str(path),task=1,device='cpu'))==0
    assert read(learner_dir/'result.json')['outer_updates']==2
    count=0
    for index,task in enumerate(mini['tasks'][2:],2):
        assert worker.run(SimpleNamespace(plan=str(path),task=index,device='cpu'))==0,task['key']
        result=read(Path(mini['output'])/task['key']/'result.json');assert result['status']=='COMPLETE'
        expected_sessions=28 if task['partition'].startswith('pan') else 10
        for budget in (20,4):
            summary=read(Path(mini['output'])/task['key']/f'evaluations/{task["partition"]}/B{budget}/clean/summary.json')
            assert summary['sessions']==expected_sessions and summary['fallback_sessions']==0
            count+=summary['sessions']
results['checks']['bundled_external_learned_fiber_and_resume']={'train_paused_at_update':1,'resumed_complete_at_update':2,'external_tasks_complete':6,'clean_session_budget_rows':count,'budgets':[20,4],'scenarios':['clean','label_noise20','time_cov_minus35'],'network_downloads':0}
results['seconds']=round(time.monotonic()-start,2)
(EVIDENCE/'FINDINGS.json').write_text(json.dumps(results,indent=2)+'\n')
print('FINAL_WITH_EXTERNAL',json.dumps(results),flush=True)
