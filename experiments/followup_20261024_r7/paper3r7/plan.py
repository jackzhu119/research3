"""Round seven: space-time Lie adaptation, head identifiability and relative risk.

This declaration never downloads datasets or starts jobs. One prepared DAG has
all dictionary, internal and repeated external tasks in one GPU allocation.
"""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
PAN_PARTITION = 'pan2023_published_order'
PRIMARY_METRICS = ('nll_balanced', 'ba')
COMMON_BASIS = 'static_ce'
MODES = {'static': 'static', 'st': 'spatiotemporal'}
OBJECTIVES = {'ce': 'ce', 'rel': 'relative_dro', 'abs': 'absolute_dro', 'var': 'variance_regret'}
HEADS = ('full', 'budget')
TIME_SCENARIOS = ('time_cov_plus35', 'time_cov_minus35')
NOISE_SCENARIOS = ('label_noise10', 'label_noise20')
BASIS_SETTINGS = {
    'static_ce': ('static', 'ce', 'full'),
    'st_ce': ('st', 'ce', 'full'),
    'static_rel': ('static', 'rel', 'full'),
    'st_rel': ('st', 'rel', 'full'),
    'st_rel_budget': ('st', 'rel', 'budget'),
}
BASIS_FAMILIES = tuple(BASIS_SETTINGS)


def group(basis, mode, objective, head):
    return f'{basis}:{mode}:{objective}:{head}'


def common(mode, objective, head):
    return group(COMMON_BASIS, mode, objective, head)


COMPARISONS = (
    (common('st', 'ce', 'full'), common('static', 'ce', 'full')),
    (common('st', 'rel', 'full'), common('st', 'ce', 'full')),
    (common('static', 'rel', 'full'), common('static', 'ce', 'full')),
    (common('st', 'rel', 'budget'), common('st', 'rel', 'full')),
    (common('static', 'rel', 'budget'), common('static', 'rel', 'full')),
    (common('st', 'rel', 'budget'), common('st', 'abs', 'budget')),
    (common('st', 'rel', 'budget'), common('st', 'var', 'budget')),
    (common('st', 'rel', 'budget'), common('static', 'rel', 'budget')),
    (common('st', 'rel', 'budget'), common('st', 'ce', 'budget')),
    (group('st_rel_budget', 'st', 'rel', 'budget'), common('st', 'rel', 'budget')),
    (group('st_ce', 'st', 'ce', 'full'), common('st', 'ce', 'full')),
    (group('static_rel', 'static', 'rel', 'full'), common('static', 'rel', 'full')),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _identity_spec(spec):
    """Hash numerical contents and producer identity, not installation paths."""
    value = json.loads(json.dumps(spec))
    for task in value.get('tasks', []):
        if 'external_manifest' in task:
            dataset = task.get('external_dataset', value.get('external_dataset', 'external'))
            manifest_sha = (value.get('pan_manifest_sha256') if dataset == 'Pan2023'
                            and 'pan_manifest_sha256' in value else value.get('external_manifest_sha256'))
            task['external_manifest'] = {'dataset': dataset, 'sha256': manifest_sha}
        if 'basis_output' in task:
            task['basis_output'] = {'basis_recipe_sha256': value.get('basis_recipe_sha256')}
    return value


def _code():
    names = ['algorithm.py', 'temporal.py', 'head_capacity.py', 'risk.py', 'cov_reference.py',
             'geometry6.py', 'geometry5.py', 'maps.py', 'round4.py', 'core.py',
             'baseline.py', 'data.py', 'external.py', 'pan2023.py', 'diagnostics.py', 'worker.py']
    files = [PACKAGE/'paper3r7'/name for name in names if (PACKAGE/'paper3r7'/name).is_file()]
    files += sorted((PACKAGE/'p3work').glob('*.py'))
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def basis_key(family, rank, seed):
    return f'basis-{family}-r{rank}-seed{seed}'


def configuration(variant='r7_refined', *, mode='static', objective='ce', head_policy='full', **extra):
    return {'variant': variant, 'geometry': 'single', 'rank': 4, 'radius': .15,
            'tangent_steps': 10, 'inner_steps': 5, 'initializer': 'tangent',
            'theta_l2': .01, 'head_l2': .1, 'theta_lr': .05, 'head_lr': .01,
            'damping': 1e-4, 'max_transform_deviation': None,
            'variance_fraction': .25, 'head_quadrature': 9,
            'sigma_center_weight': .5, 'quadrature_rotation': None,
            'mode': mode, 'objective': objective, 'head_policy': head_policy,
            'head_rank': 4, 'head_max_rank': 16, 'head_basis': 'pca', 'head_seed': 20261024,
            'temperature': .5, 'time_operator': 'zero_boundary', **extra}


def _setting(mode, objective, head, **extra):
    return configuration(mode=MODES.get(mode, mode), objective=OBJECTIVES.get(objective, objective),
                         head_policy=head, **extra)


def _basis(family, seed):
    mode, objective, head = BASIS_SETTINGS[family]
    return {'key': basis_key(family, 4, seed), 'family': 'basis', 'group': f'basis:{family}',
            'seed': seed, 'variant': 'r7_refined', 'config': _setting(mode, objective, head),
            'updates': 1000, 'requires': [], 'reuse_completed_earlier_P': False,
            'source_model_reused': True}


def _evaluation(basis, mode, objective, head, seed, *, family='core', setting='main',
                config=None, name=None, budgets=BUDGETS):
    config = config or _setting(mode, objective, head)
    name = name or group(basis, mode, objective, head)
    return {'key': f"{family}-{name.replace(':','-')}-{setting}-seed{seed}",
            'family': family, 'group': name, 'seed': seed,
            'variant': config['variant'], 'config': config,
            'basis_key': basis_key(basis, 4, seed), 'requires': [basis_key(basis, 4, seed)],
            'setting': setting, 'baseline_group': common(mode, objective, head),
            'partition': 'holdout', 'budgets': list(budgets), 'scenarios': ['clean']}


def _core():
    result = [_evaluation(COMMON_BASIS, mode, objective, head, seed)
              for mode in MODES for objective in OBJECTIVES for head in HEADS for seed in SEEDS]
    for family in BASIS_FAMILIES[1:]:
        mode, objective, head = BASIS_SETTINGS[family]
        for seed in SEEDS:
            result.append(_evaluation(family, mode, objective, head, seed))
            result.append(_evaluation(family, 'static', 'ce', 'full', seed, setting='cross-static-ce'))
    time_groups = {common('static', 'ce', 'full'), common('st', 'ce', 'full'),
                   common('st', 'rel', 'budget')}
    noise_groups = {common('st', objective, 'budget') for objective in OBJECTIVES}
    for task in result:
        scenarios = ['clean']
        if task['group'] in time_groups:
            scenarios.extend(TIME_SCENARIOS)
        if task['group'] in noise_groups:
            scenarios.extend(NOISE_SCENARIOS)
        task.update(scenarios=scenarios, diagnostics_budget=20)
    return result


def _sensitivity():
    result = []
    def add(name, mode='st', objective='rel', head='full', **extra):
        for seed in SEEDS:
            result.append(_evaluation(COMMON_BASIS, mode, objective, head, seed,
                family='sensitivity', setting=name, name=f'sensitivity:{name}', budgets=(20,),
                config=_setting(mode, objective, head, **extra)))
            task=result[-1]
            task['baseline_group']=common(mode if mode in MODES else 'st', objective,
                                          head if head in HEADS else 'full')
            if name=='head-random-rank4':
                task['baseline_group']='sensitivity:head-rank-4'
    for temperature in (.1, 1.):
        for mode in MODES:
            for objective in ('rel', 'abs'):
                add(f'temperature-{mode}-{objective}-{temperature}', mode, objective,
                    temperature=temperature)
    for rank in (0, 4, 8):
        add(f'head-rank-{rank}', head='fixed', head_rank=rank)
    for prior in (.01, 1.):
        add(f'head-prior-{prior}', head_l2=prior)
    for radius in (.05, .35):
        for mode in MODES:
            add(f'radius-{mode}-{radius}', mode=mode, radius=radius)
    add('time-circular', time_operator='circular')
    for mode in ('split_left', 'split_right', 'split_symmetric'):
        add(f'map-{mode}', mode=mode)
    for temperature in (.1, 1.):
        add(f'variance-temperature-{temperature}', objective='var', temperature=temperature)
    add('head-random-rank4', head='fixed', head_rank=4, head_basis='random')
    assert len(result) == 72
    return result


def _references():
    result = []
    for variant in ('head_newton', 'support_EA', 'covariance_only'):
        for seed in SEEDS:
            config = configuration('head_newton')
            result.append({'key': f'reference-{variant}-seed{seed}', 'family': 'reference',
                'group': variant, 'variant': variant, 'seed': seed, 'requires': [],
                'config': config, 'budgets': list(BUDGETS), 'scenarios': ['clean'],
                'partition': 'holdout', 'baseline_group': common('static', 'ce', 'full')})
    for seed in SEEDS:
        task = _evaluation(COMMON_BASIS, 'static', 'ce', 'full', seed, family='reference',
            config=configuration('uq_joint'), name='reference:uq_joint')
        result.append(task)
        for component in ('rotation', 'temporal'):
            task = _evaluation(COMMON_BASIS, component, 'ce', 'full', seed, family='reference',
                config=configuration(mode=component), name=f'reference:{component}')
            task.update(baseline_group=common('static', 'ce', 'full'),
                interpretation='component ablation changes active generator magnitude; not a pure causal factor isolation')
            result.append(task)
        result.append({'key': f'reference-global-time-seed{seed}', 'family': 'reference',
            'group': 'global_time', 'variant': 'r7_refined', 'seed': seed, 'requires': [],
            'config': configuration(mode='global_time'), 'budgets': list(BUDGETS),
            'scenarios': ['clean'], 'partition': 'holdout',
            'baseline_group': common('static', 'ce', 'full'), 'effective_spatial_parameters': 1,
            'interpretation': 'one active common temporal coefficient; rank4 container is not four active time parameters'})
    for task in result:
        if task['group'] in ('covariance_only', 'global_time'):
            task.update(scenarios=['clean', *TIME_SCENARIOS], diagnostics_budget=20)
    assert len(result) == 21
    return result


def _transfer(task, manifest, dataset, *, basis_output=None):
    task = dict(task)
    task.update(key=f'{dataset.lower()}-{task["key"]}', family='transfer',
                external_manifest=str(manifest), external_dataset=dataset,
                partition=PAN_PARTITION if dataset == 'Pan2023' else 'weibo_published_order')
    if basis_output is not None:
        task.update(requires=[], basis_output=str(basis_output))
    return task


def make_plan(root, output, gpus=3, cpus=12):
    bases = [_basis(family, seed) for family in BASIS_FAMILIES for seed in SEEDS]
    core, sensitive, references = _core(), _sensitivity(), _references()
    manifest = PACKAGE/'data/pan2023/manifest.json'
    pan = [_transfer(task, manifest, 'Pan2023') for task in core + references]
    tasks = bases + pan + core + references + sensitive
    counts = {family: sum(task['family'] == family for task in tasks)
              for family in ('basis', 'core', 'sensitivity', 'reference', 'transfer')}
    assert counts == {'basis': 15, 'core': 72, 'sensitivity': 72, 'reference': 21, 'transfer': 93}
    assert len(tasks) == len({task['key'] for task in tasks}) == 273
    spec = {'version': 'paper3-round7-spacetime-identifiability-relative-risk-v1',
        'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'updates': 1000, 'episodes_per_update': 4,
        'validate_every': 50, 'checkpoint_every': 10, 'outer_lr': .01,
        'comparisons': [list(pair) for pair in COMPARISONS], 'primary_metrics': list(PRIMARY_METRICS),
        'primary_holm_family_size': len(COMPARISONS)*len(PRIMARY_METRICS),
        'tasks': tasks, 'primary_partition': PAN_PARTITION, 'primary_budget': 20, 'primary_scenario': 'clean',
        'primary_expected_participants': 14, 'primary_expected_sessions': 28,
        'interpretation': 'Prespecified exploratory repeated Pan2023 external published-trial-prefix evaluation; both internal and external cohorts have prior-round exposure. No fresh blind, physical chronology or clinical guarantee.',
        'exploratory_internal_replication': True, 'exploratory_external_replication': True,
        'fresh_confirmatory': False, 'new_external_evaluation': False,
        'evidence_scope': 'reused_internal_and_external_cohorts_with_new_mathematical_hypotheses',
        'physical_acquisition_chronology_verified': False,
        'partition_interpretation': {'holdout': 'exploratory internal replication on reused participants',
            PAN_PARTITION: 'exploratory repeated external evaluation on Pan2023; provider trial order, not verified acquisition clock'},
        'counts': counts, 'internal_tasks': 180, 'included_pan_tasks': 93,
        'task_breakdown': {'basis': 15, 'core': 72, 'sensitivity': 72, 'reference': 21, 'pan_external': 93},
        'stress_breakdown': {'common_time_covariance_preserving_conditions': 60,
            'support_label_noise_conditions': 48, 'total_embedded_evaluation_conditions': 108,
            'budget': 20, 'extra_dictionary_training_tasks': 0},
        'external_data_required_for_core': False, 'pan_data_bundled': True,
        'pan_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        'scientific_question': 'Do finite-window space-time structure, limited head capacity and source-relative changes explain low-budget adaptation beyond static channel maps?',
        'support_scope': 'Only acquired/provider prefix; future query excluded from fitting and model selection'}
    return {**spec, 'root': str(Path(root).resolve()), 'output': str(Path(output).resolve()),
            'gpus': gpus, 'cpus': cpus,
            'requested_resources': {'partition': 'hpc_gpu', 'gpus': gpus, 'nodes': 1, 'slurm_jobs': 1,
                'cpus_total': gpus*cpus, 'host_memory_total': '128G', 'memory_scope': 'shared job total'},
            'recipe_sha256': _digest({'specification': _identity_spec(spec), 'code': _code()})}


def make_external_plan(root, output, external_manifest, basis_run, gpus=3, cpus=12):
    manifest = Path(external_manifest).expanduser().resolve()
    data = json.loads(manifest.read_text())
    dataset = 'Pan2023' if data.get('dataset_id') == 'Pan2023' else 'weibo'
    basis_run = Path(basis_run).expanduser().resolve()
    base = json.loads((basis_run/'plan.json').read_text())
    common_tasks = [task for task in _core() if task['basis_key'].startswith('basis-static_ce-')]
    candidates = common_tasks + [task for task in _references() if 'basis_key' not in task]
    tasks = [_transfer(task, manifest, dataset, basis_output=basis_run) for task in candidates]
    assert len(tasks) == 60
    spec = {'version': f'paper3-round7-{dataset.lower()}-published-order-v1', 'seeds': list(SEEDS),
        'tasks': tasks, 'budgets': list(BUDGETS), 'comparisons': [],
        'primary_partition': PAN_PARTITION if dataset == 'Pan2023' else 'weibo_published_order',
        'primary_budget': 20, 'primary_scenario': 'clean', 'basis_recipe_sha256': base['recipe_sha256'],
        'external_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'exploratory_internal_replication': False, 'exploratory_external_replication': True,
        'fresh_confirmatory': False,
        'interpretation': 'descriptive additional external published-order LR-prefix robustness; physical acquisition clock unverified'}
    return {**spec, 'root': str(Path(root).resolve()), 'output': str(Path(output).resolve()),
            'gpus': gpus, 'cpus': cpus, 'external_manifest': str(manifest),
            'external_dataset': dataset, 'basis_output': str(basis_run),
            'recipe_sha256': _digest({'specification': _identity_spec(spec), 'code': _code()})}
