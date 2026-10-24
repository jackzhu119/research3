"""Round six: bounded curvature-shaped predictive averaging plus fixed Pan external validation.

Plans are numerical declarations; make_plan never downloads data or starts jobs.
"""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
VARIANTS = ('uq_map', 'uq_head', 'uq_linear', 'uq_spatial', 'uq_joint',
            'uq_isotropic', 'uq_independent', 'uq_mean', 'uq_mean_head', 'uq_diagonal')
BASIS_FAMILIES = ('map', 'joint')
PAN_PARTITION = 'pan2023_published_order'
COMPARISONS = tuple(('map:uq_joint', 'map:' + variant) for variant in
                   ('uq_map', 'uq_mean_head', 'uq_isotropic', 'uq_independent',
                    'uq_spatial', 'uq_head', 'uq_linear')) + (('joint:uq_joint', 'map:uq_joint'),)
PRIMARY_METRICS = ('nll_balanced', 'ba')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()



def _identity_spec(spec):
    """Numerical identity binds input contents, never installation locations.

    Runtime tasks keep real absolute paths. Only this detached digest copy
    replaces them by the manifest content hash and producer recipe already
    bound by plan preparation.
    """
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
    names = ['algorithm.py', 'uncertainty.py', 'geometry5.py', 'maps.py', 'round4.py', 'core.py',
             'baseline.py', 'data.py', 'external.py', 'pan2023.py', 'diagnostics.py', 'worker.py']
    files = [PACKAGE/'paper3r6'/name for name in names if (PACKAGE/'paper3r6'/name).is_file()]
    files += sorted((PACKAGE/'p3work').glob('*.py'))
    return {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def basis_key(family, rank, seed):
    return f'basis-{family}-r{rank}-seed{seed}'


def configuration(variant='uq_joint', **extra):
    return {'variant': variant, 'geometry': 'single', 'rank': 4, 'radius': .15,
            'tangent_steps': 10, 'inner_steps': 5, 'initializer': 'tangent',
            'theta_l2': .01, 'head_l2': .1, 'theta_lr': .05, 'head_lr': .01,
            'damping': 1e-4, 'max_transform_deviation': None,
            'variance_fraction': .25, 'head_quadrature': 9,
            'sigma_center_weight': .5, 'quadrature_rotation': None, **extra}


def _basis(family, seed):
    variant = 'uq_map' if family == 'map' else 'uq_joint'
    return {'key': basis_key(family, 4, seed), 'family': 'basis', 'group': f'basis:{family}',
            'seed': seed, 'variant': variant, 'config': configuration(variant),
            'updates': 1000, 'requires': [], 'reuse_completed_earlier_P': False,
            'source_model_reused': True}


def _evaluation(basis, variant, seed, *, family='core', setting='main', config=None,
                group=None, budgets=BUDGETS):
    config = config or configuration(variant)
    group = group or f'{basis}:{variant}'
    return {'key': f"{family}-{group.replace(':','-')}-{setting}-seed{seed}",
            'family': family, 'group': group, 'seed': seed, 'variant': variant, 'config': config,
            'basis_key': basis_key(basis, 4, seed), 'requires': [basis_key(basis, 4, seed)],
            'setting': setting, 'baseline_group': 'map:uq_joint', 'partition': 'holdout',
            'budgets': list(budgets), 'scenarios': ['clean']}


def _references():
    tasks = []
    for variant in ('head_newton', 'support_EA'):
        for seed in SEEDS:
            tasks.append({'key': f'reference-{variant}-seed{seed}', 'family': 'reference',
                'group': variant, 'variant': variant, 'seed': seed, 'requires': [],
                'config': configuration('head_newton'), 'budgets': list(BUDGETS),
                'scenarios': ['clean'], 'partition': 'holdout', 'baseline_group': 'map:uq_joint'})
    for variant in ('geometry_refined', 'xfit_stack'):
        for seed in SEEDS:
            tasks.append(_evaluation('map', variant, seed, family='reference',
                group=f'reference:{variant}'))
    return tasks


def _transfer(task, manifest, dataset, *, basis_output=None):
    task = dict(task)
    task.update(key=f'{dataset.lower()}-{task["key"]}', family='transfer',
                external_manifest=str(manifest), external_dataset=dataset,
                partition=PAN_PARTITION if dataset == 'Pan2023' else 'weibo_published_order')
    if basis_output is not None:
        task.update(requires=[], basis_output=str(basis_output))
    return task


def make_plan(root, output, gpus=3, cpus=12):
    tasks = [_basis(family, seed) for family in BASIS_FAMILIES for seed in SEEDS]
    core = [_evaluation(basis, variant, seed) for basis in BASIS_FAMILIES
            for variant in VARIANTS for seed in SEEDS]
    tasks += core
    for fraction in (.0625, 1.):
        for variant in ('uq_joint', 'uq_isotropic', 'uq_mean_head'):
            for seed in SEEDS:
                tasks.append(_evaluation('map', variant, seed, family='sensitivity',
                    setting=f'variance{fraction}', group=f'variance:{variant}:{fraction}',
                    config=configuration(variant, variance_fraction=fraction), budgets=(20,)))
    for order in (5, 17):
        for seed in SEEDS:
            tasks.append(_evaluation('map', 'uq_joint', seed, family='sensitivity',
                setting=f'gh{order}', group=f'quadrature:gh{order}',
                config=configuration('uq_joint', head_quadrature=order), budgets=(20,)))
    for variant in ('uq_joint', 'uq_mean_head'):
        for seed in SEEDS:
            tasks.append(_evaluation('map', variant, seed, family='sensitivity',
                setting='centerweight0', group=f'quadrature:{variant}:center0',
                config=configuration(variant, sigma_center_weight=0.), budgets=(20,)))
            tasks.append(_evaluation('map', variant, seed, family='sensitivity',
                setting='rotated-stencil', group=f'quadrature:{variant}:rotated',
                config=configuration(variant, quadrature_rotation=20261024), budgets=(20,)))
    for radius in (.05, .35):
        for variant in ('uq_joint', 'uq_mean_head'):
            for seed in SEEDS:
                tasks.append(_evaluation('map', variant, seed, family='sensitivity',
                    setting=f'radius{radius}', group=f'radius:{variant}:{radius}',
                    config=configuration(variant, radius=radius), budgets=(20,)))
    references = _references()
    tasks += references
    internal_count = len(tasks)
    manifest = PACKAGE/'data'/'pan2023'/'manifest.json'
    # The delivered package contains the verified cache. Plan preparation only
    # declares that fixed input, and never fetches original MATs or a dataset.
    pan_tasks = [_transfer(task, manifest, 'Pan2023') for task in core + references]
    # Ready new external evidence precedes reused-internal sensitivity work.
    tasks = tasks[:6] + pan_tasks + core + references + [t for t in tasks if t['family'] == 'sensitivity']
    counts = {family: sum(t['family'] == family for t in tasks)
              for family in ('basis', 'core', 'sensitivity', 'reference', 'transfer')}
    assert counts == {'basis': 6, 'core': 60, 'sensitivity': 48, 'reference': 12, 'transfer': 72}
    assert internal_count == 126 and len(tasks) == len({t['key'] for t in tasks}) == 198
    spec = {'version': 'paper3-round6-curvature-predictive-averaging-v1', 'seeds': list(SEEDS),
        'budgets': list(BUDGETS), 'updates': 1000, 'episodes_per_update': 4,
        'validate_every': 50, 'checkpoint_every': 10, 'outer_lr': .01,
        'comparisons': [list(pair) for pair in COMPARISONS], 'primary_metrics': list(PRIMARY_METRICS),
        'primary_holm_family_size': len(COMPARISONS)*len(PRIMARY_METRICS),
        'tasks': tasks, 'primary_partition': PAN_PARTITION, 'primary_budget': 20, 'primary_scenario': 'clean',
        'primary_expected_participants': 14, 'primary_expected_sessions': 28,
        'interpretation': 'Prespecified-in-code new Pan2023 external published-trial-prefix evaluation; internal cohort is reused exploratory replication. No physical acquisition chronology claim.',
        'exploratory_internal_replication': True, 'fresh_confirmatory': False,
        'evidence_scope': 'mixed_existing_internal_and_new_external',
        'new_external_evaluation': True, 'physical_acquisition_chronology_verified': False,
        'partition_interpretation': {'holdout': 'exploratory internal replication on reused participants',
            PAN_PARTITION: 'new external evaluation fixed before Pan prediction results; provider trial order, not verified physical acquisition clock'},
        'counts': counts, 'internal_tasks': internal_count, 'included_pan_tasks': 72,
        'task_breakdown': {'basis': 6, 'core': 60, 'sensitivity': 48, 'reference': 12, 'pan_external': 72},
        'external_data_required_for_core': False, 'pan_data_bundled': True,
        'pan_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        'scientific_question': 'Does bounded curvature-shaped nonlinear predictive averaging add value beyond centroid shrinkage and symmetric logit smoothing?',
        'support_scope': 'Only acquired/provider prefix; future query excluded from fitting and model selection'}
    return {**spec, 'root': str(Path(root).resolve()), 'output': str(Path(output).resolve()),
            'gpus': gpus, 'cpus': cpus, 'recipe_sha256': _digest({'specification': _identity_spec(spec), 'code': _code()})}


def make_external_plan(root, output, external_manifest, basis_run, gpus=3, cpus=12):
    manifest = Path(external_manifest).expanduser().resolve()
    data = json.loads(manifest.read_text())
    dataset = 'Pan2023' if data.get('dataset_id') == 'Pan2023' else 'weibo'
    basis_run = Path(basis_run).expanduser().resolve()
    base = json.loads((basis_run/'plan.json').read_text())
    candidates = [_evaluation('map', variant, seed) for variant in VARIANTS for seed in SEEDS]
    candidates += [t for t in _references() if 'basis_key' not in t]
    tasks = [_transfer(t, manifest, dataset, basis_output=basis_run) for t in candidates]
    spec = {'version': f'paper3-round6-{dataset.lower()}-published-order-v1', 'seeds': list(SEEDS),
        'tasks': tasks, 'budgets': list(BUDGETS), 'comparisons': [],
        'primary_partition': PAN_PARTITION if dataset == 'Pan2023' else 'weibo_published_order',
        'primary_budget': 20, 'primary_scenario': 'clean', 'basis_recipe_sha256': base['recipe_sha256'],
        'external_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'exploratory_internal_replication': False, 'fresh_confirmatory': False,
        'interpretation': 'descriptive additional external published-order LR-prefix robustness; physical acquisition clock unverified'}
    return {**spec, 'root': str(Path(root).resolve()), 'output': str(Path(output).resolve()),
            'gpus': gpus, 'cpus': cpus, 'external_manifest': str(manifest),
            'external_dataset': dataset, 'basis_output': str(basis_run),
            'recipe_sha256': _digest({'specification': _identity_spec(spec), 'code': _code()})}
