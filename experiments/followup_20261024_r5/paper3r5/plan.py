"""Focused rotation/gain coupling experiment; a new run never mutates prior rounds."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
CHARTS = ('single', 'left_rotation', 'right_rotation', 'symmetric')
PROBE_CHARTS = (*CHARTS, 'bch2')
STRESS = ('coupled_ks15', 'coupled_sk15', 'coupled_null15',
          'coupled_ks35', 'coupled_sk35', 'coupled_null35')
COMPARISONS = (
    ('single:left_rotation', 'single:single'),
    ('single:right_rotation', 'single:single'),
    ('single:symmetric', 'single:single'),
    ('single:symmetric', 'single:left_rotation'),
    ('single:symmetric', 'single:right_rotation'),
    ('left_rotation:left_rotation', 'single:left_rotation'),
    ('right_rotation:right_rotation', 'single:right_rotation'),
    ('symmetric:symmetric', 'single:symmetric'),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _code():
    names = ['algorithm.py', 'maps.py', 'round4.py', 'core.py', 'baseline.py', 'data.py', 'diagnostics.py', 'worker.py']
    files = [PACKAGE/'paper3r5'/name for name in names] + sorted((PACKAGE/'p3work').glob('*.py'))
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def basis_key(chart, rank, seed):
    return f'basis-{chart}-r{rank}-seed{seed}'


def configuration(chart='single', *, variant='geometry_refined', **extra):
    return {'variant':variant, 'geometry':chart, 'rank':4, 'radius':.15,
            'tangent_steps':10, 'inner_steps':5, 'initializer':'tangent',
            'theta_l2':.01, 'head_l2':.1, 'theta_lr':.05, 'head_lr':.01,
            'damping':1e-4, 'max_transform_deviation':None, **extra}


def _basis(chart, seed):
    return {'key':basis_key(chart,4,seed), 'family':'basis', 'group':f'basis:{chart}',
            'seed':seed, 'variant':'geometry_refined', 'config':configuration(chart),
            'updates':1000, 'requires':[], 'reuse_completed_earlier_P':False,
            'source_model_reused':True}


def _evaluation(basis, chart, seed, *, family='core', setting='main', config=None,
                group=None, scenarios=('clean',), budgets=BUDGETS):
    config = config or configuration(chart)
    group = group or f'{basis}:{chart}'
    return {'key':f"{family}-{group.replace(':','-')}-{setting}-seed{seed}", 'family':family,
            'group':group, 'seed':seed, 'variant':config['variant'], 'config':config,
            'basis_key':basis_key(basis,4,seed), 'requires':[basis_key(basis,4,seed)],
            'setting':setting, 'baseline_group':f'single:{chart}' if basis!='single' else 'single:single',
            'partition':'holdout', 'budgets':list(budgets), 'scenarios':list(scenarios),
            'diagnostics_budget':20}


def make_plan(root, output, gpus=3, cpus=12):
    tasks = [_basis('single',seed) for seed in SEEDS]
    for seed in SEEDS:
        for chart in CHARTS:
            tasks.append(_evaluation('single',chart,seed,scenarios=('clean',*STRESS)))
    tasks += [_basis(chart,seed) for chart in CHARTS[1:] for seed in SEEDS]
    for basis in CHARTS[1:]:
        for seed in SEEDS:
            for chart in CHARTS:
                tasks.append(_evaluation(basis,chart,seed))
    for radius in (.05,.35):
        for chart in CHARTS:
            for seed in SEEDS:
                tasks.append(_evaluation('single',chart,seed,family='sensitivity',setting=f'radius{radius}',
                    group=f'radius:{chart}:{radius}',config=configuration(chart,radius=radius),budgets=(20,)))
    for chart in PROBE_CHARTS:
        for seed in SEEDS:
            task = _evaluation('single',chart,seed,family='sensitivity',setting='shared-theta-probe',
                group=f'probe:{chart}',config=configuration(chart,variant='geometry_tangent'),budgets=(20,))
            task.update(mechanism_probe=True,baseline_group='probe:single',
                        interpretation='common identity-tangent theta/head; chart changes predictions only; not a deployment claim')
            tasks.append(task)
    for variant in ('head_newton','support_EA'):
        for seed in SEEDS:
            tasks.append({'key':f'reference-{variant}-seed{seed}','family':'reference','group':variant,
                'variant':variant,'seed':seed,'requires':[],
                'config':configuration(variant='head_newton'),'budgets':list(BUDGETS),
                'scenarios':['clean'],'partition':'holdout','baseline_group':'single:single'})
    counts = {family:sum(t['family']==family for t in tasks) for family in ('basis','core','sensitivity','reference')}
    assert counts=={'basis':12,'core':48,'sensitivity':39,'reference':6}
    assert len({t['key'] for t in tasks})==len(tasks)==105
    spec = {'version':'paper3-round5-reversible-spatial-coupling-v1','seeds':list(SEEDS),
            'budgets':list(BUDGETS),'updates':1000,'episodes_per_update':4,
            'validate_every':50,'checkpoint_every':10,'outer_lr':.01,
            'comparisons':[list(pair) for pair in COMPARISONS],'tasks':tasks,
            'primary_partition':'holdout','primary_budget':20,'primary_scenario':'clean',
            'interpretation':'exploratory internal replication; cohort already used by earlier rounds',
            'counts':counts,'task_breakdown':{'basis':12,'core':48,'radius_sensitivity':24,'shared_theta_probe':15,'reference':6},
            'external_data_required_for_core':False,
            'scientific_question':'Does reversible rotation/gain coupling improve fixed-prefix, fixed-rank spatial calibration?',
            'support_scope':'Only acquired prefix; future query excluded from fitting and model selection'}
    return {**spec,'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),
            'gpus':gpus,'cpus':cpus,'recipe_sha256':_digest({'specification':spec,'code':_code()})}


def make_external_plan(root, output, external_manifest, basis_run, gpus=3, cpus=12):
    manifest=Path(external_manifest).expanduser().resolve()
    json.loads(manifest.read_text())
    basis_run=Path(basis_run).expanduser().resolve()
    base=json.loads((basis_run/'plan.json').read_text())
    pairs=[('single',chart) for chart in CHARTS]
    pairs += [(chart,chart) for chart in CHARTS[1:]]
    tasks=[]
    for basis,chart in pairs:
        for seed in SEEDS:
            task=_evaluation(basis,chart,seed,family='transfer')
            task.update(requires=[],basis_output=str(basis_run),external_manifest=str(manifest),partition='weibo_published_order')
            tasks.append(task)
    spec={'version':'paper3-round5-weibo-published-order-v1','seeds':list(SEEDS),'tasks':tasks,
          'budgets':list(BUDGETS),'comparisons':[],'primary_partition':'weibo_published_order',
          'primary_budget':20,'primary_scenario':'clean','basis_recipe_sha256':base['recipe_sha256'],
          'external_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
          'interpretation':'descriptive external published-order LR-subtask robustness; physical acquisition clock unverified'}
    return {**spec,'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),'gpus':gpus,'cpus':cpus,
            'external_manifest':str(manifest),'basis_output':str(basis_run),
            'recipe_sha256':_digest({'specification':spec,'code':_code()})}
