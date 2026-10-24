"""Round 4: support cross-fitting, convex prediction pooling and robust loss."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
STRESS = ('label_noise10', 'label_noise20', 'gain10', 'gain25', 'rotation10', 'rotation25')
ADAPTERS = ('sgd10', 'head_newton', 'tangent_profiled', 'information_protected',
            'fixed_half', 'uniform_stack', 'insample_shrink', 'insample_stack',
            'xfit_shrink', 'xfit_stack', 'xfit_select', 'xfit_select_stack',
            'gce_profiled', 'xfit_stack_gce')
JOINT_BASES = {'matched_sgd': 'sgd10', 'xfit_shrink': 'xfit_shrink',
               'xfit_stack': 'xfit_stack', 'gce': 'gce_profiled'}
JOINT_ADAPTERS = ('tangent_profiled', 'xfit_shrink', 'xfit_stack', 'gce_profiled')
COMPARISONS = (
    ('legacy:xfit_shrink', 'legacy:fixed_half'),
    ('legacy:xfit_shrink', 'legacy:insample_shrink'),
    ('legacy:xfit_stack', 'legacy:uniform_stack'),
    ('legacy:xfit_stack', 'legacy:insample_stack'),
    ('legacy:xfit_stack', 'legacy:xfit_shrink'),
    ('legacy:xfit_select', 'legacy:tangent_profiled'),
    ('legacy:xfit_select_stack', 'legacy:xfit_stack'),
    ('legacy:gce_profiled', 'legacy:tangent_profiled'),
    ('xfit_stack:xfit_stack', 'matched_sgd:xfit_stack'),
    ('gce:gce_profiled', 'matched_sgd:gce_profiled'),
    ('xfit_shrink:xfit_shrink', 'matched_sgd:xfit_shrink'),
    ('legacy:xfit_stack_gce', 'legacy:xfit_stack'),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _code():
    names = ['algorithm.py','core.py','baseline.py','data.py','diagnostics.py','worker.py']
    files = [PACKAGE/'paper3r4'/name for name in names] + sorted((PACKAGE/'p3work').glob('*.py'))
    return {str(p.relative_to(PACKAGE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def basis_key(kind,rank,seed):
    return f'basis-{kind}-r{rank}-seed{seed}'


def configuration(variant,*,rank=4,radius=.15,legacy=False,**extra):
    if variant == 'xfit_stack_gce':
        variant,extra = 'xfit_stack',{'base_variant':'gce_profiled',**extra}
    return {'variant':variant,'rank':rank,'radius':radius,'tangent_steps':10,
            'theta_l2':.01,'head_l2':.001 if legacy else .1,'damping':1e-4,
            'information_scale':1.,'kl_epsilon':.01,'base_variant':'tangent_profiled',
            'cv_folds':5,'cv_scheme':'blocked','cv_rule':'one_se','stacking_l2':.01,
            'cv_head_grid':[.01,.1,1.],'cv_radius_grid':[.05,.15],'gce_q':.7,**extra}


def _basis(kind,rank,seed):
    variant = 'sgd10' if kind=='legacy' else JOINT_BASES[kind]
    return {'key':basis_key(kind,rank,seed),'family':'basis','group':f'basis:{kind}:r{rank}',
            'seed':seed,'variant':variant,'config':configuration(variant,rank=rank,legacy=kind=='legacy'),
            'updates':1000,'requires':[],'reuse_completed_earlier_P':kind=='legacy',
            'reuse_completed_round3':kind in ('legacy','matched_sgd')}


def _evaluation(kind,variant,seed,*,family='core',setting='main',config=None,
                group=None,baseline_group=None,scenarios=('clean',),budgets=BUDGETS):
    config = config or configuration(variant)
    group = group or f'{kind}:{variant}'
    return {'key':f"{family}-{kind}-{group.replace(':','-')}-{setting}-seed{seed}",
            'family':family,'group':group,'seed':seed,'variant':config['variant'],
            'basis_key':basis_key(kind,config['rank'],seed),'requires':[basis_key(kind,config['rank'],seed)],
            'config':config,'setting':setting,'baseline_group':baseline_group or 'legacy:tangent_profiled',
            'partition':'holdout','budgets':list(budgets),'scenarios':list(scenarios),'diagnostics_budget':20}


def make_plan(root,output,gpus=3,cpus=16):
    tasks = [_basis('legacy',4,seed) for seed in SEEDS]
    for seed in SEEDS:
        for variant in ADAPTERS:
            pressure = STRESS if variant in ('tangent_profiled','information_protected','xfit_shrink',
                                            'xfit_stack','gce_profiled','xfit_stack_gce') else ()
            tasks.append(_evaluation('legacy',variant,seed,scenarios=('clean',*pressure)))
    tasks += [_basis(kind,4,seed) for kind in JOINT_BASES for seed in SEEDS]
    for kind in JOINT_BASES:
        for seed in SEEDS:
            for variant in JOINT_ADAPTERS:
                tasks.append(_evaluation(kind,variant,seed,baseline_group=f'legacy:{variant}'))
    tasks += [_basis('legacy',rank,seed) for rank in (2,8) for seed in SEEDS]
    settings = []
    three = ('xfit_shrink','xfit_stack','xfit_select_stack')
    two = ('xfit_shrink','xfit_stack')
    for scheme in ('stratified','loo'):
        settings.append((f'cv-{scheme}',{'cv_scheme':scheme},three))
    for folds in (2,10):
        settings.append((f'folds{folds}',{'cv_folds':folds},two))
    for penalty in (.001,.1):
        settings.append((f'stackprior{penalty}',{'stacking_l2':penalty},two))
    settings.append(('protected-base',{'base_variant':'information_protected'},two))
    for rank in (2,8):
        settings.append((f'rank{rank}',{'rank':rank},two))
    for q in (.3,1.):
        settings.append((f'GCEq{q}',{'gce_q':q},('gce_profiled','xfit_stack_gce')))
    settings.append(('minimum-CV',{'cv_rule':'minimum'},('xfit_select','xfit_select_stack')))
    for radius in (.05,.35):
        settings.append((f'radius{radius}',{'radius':radius},two))
    for setting,changes,variants in settings:
        for variant in variants:
            for seed in SEEDS:
                tasks.append(_evaluation('legacy',variant,seed,family='sensitivity',setting=setting,
                    group=f'sensitivity:{variant}:{setting}',config=configuration(variant,**changes),
                    baseline_group=f'legacy:{variant}',budgets=(20,)))
    for seed in SEEDS:
        tasks.append({'key':f'reference-supportEA-seed{seed}','family':'reference','group':'support_EA',
            'variant':'support_EA','seed':seed,'requires':[],'config':configuration('head_newton'),
            'budgets':list(BUDGETS),'scenarios':['clean'],'partition':'holdout',
            'baseline_group':'legacy:head_newton'})
    counts = {family:sum(t['family']==family for t in tasks) for family in ('basis','core','sensitivity','reference')}
    assert counts == {'basis':21,'core':90,'sensitivity':90,'reference':3}
    assert len({t['key'] for t in tasks}) == len(tasks)
    spec = {'version':'paper3-round4-crossfit-robust-v1','seeds':list(SEEDS),'budgets':list(BUDGETS),
            'updates':1000,'episodes_per_update':4,'validate_every':50,'checkpoint_every':10,'outer_lr':.01,
            'comparisons':[list(pair) for pair in COMPARISONS],'tasks':tasks,
            'primary_partition':'holdout','primary_budget':20,'primary_scenario':'clean',
            'interpretation':'exploratory internal replication; cohort already used by earlier rounds',
            'counts':counts,'external_data_required_for_core':False,
            'adaptation_selection':'support-only blocked cross-fitting; future query never selects parameters or mixture'}
    return {**spec,'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),
            'gpus':gpus,'cpus':cpus,'recipe_sha256':_digest({'specification':spec,'code':_code()})}


def make_external_plan(root,output,external_manifest,basis_run,gpus=3,cpus=16):
    manifest = Path(external_manifest).expanduser().resolve()
    json.loads(manifest.read_text())
    basis_run = Path(basis_run).expanduser().resolve()
    base = json.loads((basis_run/'plan.json').read_text())
    groups = [('legacy',v) for v in ('tangent_profiled','information_protected','xfit_shrink',
                                    'xfit_stack','gce_profiled','xfit_stack_gce')]
    groups.append(('xfit_stack','xfit_stack'))
    tasks = []
    for kind,variant in groups:
        for seed in SEEDS:
            task = _evaluation(kind,variant,seed,family='transfer')
            task.update(requires=[],basis_output=str(basis_run),external_manifest=str(manifest),
                        partition='weibo_published_order')
            tasks.append(task)
    spec = {'version':'paper3-round4-weibo-published-order-v1','seeds':list(SEEDS),'tasks':tasks,
            'budgets':list(BUDGETS),'comparisons':[],'primary_partition':'weibo_published_order',
            'primary_budget':20,'primary_scenario':'clean','basis_recipe_sha256':base['recipe_sha256'],
            'external_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'interpretation':'descriptive external published-order LR-subtask robustness; physical acquisition clock unverified'}
    return {**spec,'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'gpus':gpus,'cpus':cpus,
            'external_manifest':str(manifest),'basis_output':str(basis_run),
            'recipe_sha256':_digest({'specification':spec,'code':_code()})}
