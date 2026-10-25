"""Finite result-informed R9 plan. Imports no Torch and submits no school jobs."""
from pathlib import Path
import hashlib
import json

PACKAGE=Path(__file__).resolve().parent.parent
SEEDS=(20261018,20261019,20261020)
DATASETS=('holdout','pan2023_published_order','weibo2014_published_order')
BUDGETS=(20,4,8,12,16)
SCENARIOS=('gain25','rotation25','label_noise10','label_noise20','time_cov_plus35','time_cov_minus35')
TAUS=(.03,.1,.3,1.,3.)
COMPARISONS=[['rbf_anchored','support_EA'],['rbf_anchored','rbf_pp'],
    ['rbf_anchored','rbf_global_mixture'],['ea_rbf_anchored','support_EA'],
    ['rbf_anchored','linear_anchored'],['rbf_pp','rbf_map']]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def reference_key(preprocessing,seed):return f'feature-reference-{preprocessing}-seed{seed}'
def select_key(preprocessing,kernel,seed):return f'select-{preprocessing}-{kernel}-seed{seed}'

def config(kernel='rbf',preprocessing='raw',predictive='anchored_laplace',**kwargs):
    return dict(kernel=kernel,preprocessing=preprocessing,predictive=predictive,tau=.3,lengthscale=1.,anchor_cap=.75,**kwargs)


def core_methods():
    methods=[('source_frozen','source_frozen',config(),False),
        ('support_EA','support_EA',config(),False),
        ('head_sgd10','head_sgd10',config(),False),('head_sgd100','head_sgd100',config(),False)]
    for temperature,label in ((.5,'T0p5'),(2.,'T2')):
        raw=config();raw['temperature']=temperature
        ea=config(preprocessing='ea');ea['temperature']=temperature
        methods.extend([('source_'+label,'source_temperature',raw,False),('EA_'+label,'ea_temperature',ea,False)])
    for preprocessing in ('raw','ea'):
        prefix='' if preprocessing=='raw' else 'ea_'
        for kernel in ('linear','rbf'):
            for name,mode in (('map','map'),('pp','laplace'),('anchored','anchored_laplace'),('global_mixture','constant_mixture')):
                methods.append((prefix+kernel+'_'+name,'gp',config(kernel,preprocessing,mode),True))
        methods.append((prefix+'rbf_fixed_prior','gp',config('rbf',preprocessing),False))
    return methods


def _evaluation(group,variant,cfg,selected,seed,dataset,family='core',budgets=BUDGETS,scenarios=('clean',)):
    item=dict(group=group,variant=variant,config=cfg,seed=seed,partition=dataset,
        family=family,budgets=list(budgets),scenarios=list(scenarios),requires=[])
    if variant=='gp':
        item['reference_key']=reference_key(cfg['preprocessing'],seed);item['requires'].append(item['reference_key'])
        if selected:
            item['selection_key']=select_key(cfg['preprocessing'],cfg['kernel'],seed);item['requires'].append(item['selection_key'])
    scientific=dict(item)
    if dataset!='holdout':
        external='pan2023' if dataset.startswith('pan') else 'weibo'
        item['external_manifest']=str(PACKAGE/'data'/external/'manifest.json')
        item['external_dataset']='Pan2023' if external=='pan2023' else 'Weibo2014'
    item['key']=f'{family}-{dataset.split("_")[0]}-{group}-seed{seed}-{digest(scientific)[:10]}'
    return item


def build(root,output=None):
    root=Path(root).expanduser().resolve();output=Path(output or root/'runs/followup_20261025_r9').resolve()
    tasks=[]
    for preprocessing in ('raw','ea'):
        for seed in SEEDS:
            tasks.append(dict(key=reference_key(preprocessing,seed),group='feature_reference',family='feature_reference',
                seed=seed,config=config(preprocessing=preprocessing),requires=[],
                scope='meta_train_prefix20_only_equal_person_session_trial_weighting'))
    methods=core_methods()
    for group,variant,cfg,selected in methods:
        if variant=='gp':continue
        for seed in SEEDS:
            for dataset in DATASETS:tasks.append(_evaluation(group,variant,cfg,selected,seed,dataset))
    for preprocessing in ('raw','ea'):
        for kernel in ('linear','rbf'):
            candidates=[dict(tau=tau,lengthscale=ell) for tau in TAUS for ell in ((1.,) if kernel=='linear' else (.5,1.,2.))]
            for seed in SEEDS:
                ref=reference_key(preprocessing,seed)
                tasks.append(dict(key=select_key(preprocessing,kernel,seed),group='select',family='select',
                    seed=seed,config=config(kernel,preprocessing),candidates=candidates,requires=[ref],reference_key=ref,
                    selection_role='meta_validation',selection_metric='participant_mean_balanced_NLL',
                    selection_predictive='anchored_laplace',anchor_cap_fixed=.75))
    for group,variant,cfg,selected in methods:
        if variant!='gp':continue
        for seed in SEEDS:
            for dataset in DATASETS:tasks.append(_evaluation(group,variant,cfg,selected,seed,dataset))
    stressed={'source_frozen','support_EA','rbf_map','rbf_pp','rbf_anchored','rbf_global_mixture','ea_rbf_anchored','linear_anchored'}
    nulled={'source_frozen','support_EA','rbf_pp','rbf_anchored','ea_rbf_anchored','linear_anchored'}
    for group,variant,cfg,selected in methods:
        if group in stressed:
            for seed in SEEDS:
                for dataset in DATASETS:
                    for scenario in SCENARIOS:tasks.append(_evaluation(group,variant,cfg,selected,seed,dataset,'stress',(20,),(scenario,)))
        if group in nulled:
            for seed in SEEDS:
                for dataset in DATASETS:tasks.append(_evaluation(group,variant,cfg,selected,seed,dataset,'label_null',(20,),('support_label_permutation',)))
    for preprocessing in ('raw','ea'):
        for kernel in ('linear','rbf'):
            for tau in TAUS:
                for ell in ((1.,) if kernel=='linear' else (.5,1.,2.)):
                    for predictive in ('map','anchored_laplace'):
                        # The same fixed RBF prior is already a core reference.
                        if kernel=='rbf' and tau==.3 and ell==1. and predictive=='anchored_laplace':continue
                        cfg=config(kernel,preprocessing,predictive);cfg.update(tau=tau,lengthscale=ell)
                        group=f'prior_{preprocessing}_{kernel}_t{tau}_l{ell}_{predictive}'.replace('.','p')
                        for seed in SEEDS:
                            for dataset in DATASETS:tasks.append(_evaluation(group,'gp',cfg,False,seed,dataset,'prior_sensitivity',(20,)))
            for cap in (.25,.5,1.):
                cfg=config(kernel,preprocessing);cfg['anchor_cap']=cap
                group=f'cap_{preprocessing}_{kernel}_{cap}'.replace('.','p')
                for seed in SEEDS:
                    for dataset in DATASETS:tasks.append(_evaluation(group,'gp',cfg,True,seed,dataset,'anchor_sensitivity',(20,)))
    hashes={}
    for directory in ('paper3r9','paper3r7','p3work'):
        for path in sorted((PACKAGE/directory).glob('*.py')):
            hashes[str(path.relative_to(PACKAGE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    identities=[]
    for task in tasks:
        value=dict(task)
        if 'external_manifest' in value:
            path=Path(value['external_manifest']);value['external_manifest']={'dataset':value['external_dataset'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        identities.append(value)
    scientific=dict(schema='paper3-round9-v1',tasks=identities,code=hashes,comparisons=COMPARISONS)
    from collections import Counter
    return dict(schema_version='paper3-round9-plan-v1',round=9,root=str(root),output=str(output),
        recipe_sha256=digest(scientific),code_sha256=hashes,tasks=tasks,seeds=list(SEEDS),budgets=list(BUDGETS),
        comparisons=COMPARISONS,primary_partitions=list(DATASETS[1:]),primary_metrics=['ba','nll_balanced'],
        expected_cohorts={'holdout':{'participants':11,'sessions':105},DATASETS[1]:{'participants':14,'sessions':28},DATASETS[2]:{'participants':10,'sessions':10}},
        resources={'partition':'hpc_gpu','jobs':3,'gpus_per_job':1,'cpus_per_job':16,'memory_per_job':'64G','hardware':'Full A100 40GB/80GB; actual device recorded without marketing-name gate'},
        counts=dict(Counter(t['family'] for t in tasks)),selection_scope='meta_train_whitening_meta_validation_anchored_NLL_finite_grid',
        full_test_at_start=False,compute_node_downloads=False,fixed_12h_total_limit=False,
        training_kind='Frozen encoder; prefix feature reference, finite meta-validation hyperparameter selection and per-prefix GP posterior',
        data_scope='Existing Stieger cache plus bundled Pan/Weibo cache; all reused exploratory cohorts',
        interpretation='Mechanism-seeking exploration, not new blinded confirmation; no guaranteed gain; seed/session repetitions do not enlarge participant n')
