"""Finite Round 8 experiment plan; no numerical/network imports."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parent.parent
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (20, 4, 8, 12, 16)
DATASETS = ('holdout', 'pan2023_published_order', 'weibo2014_published_order')
SCENARIOS = ('gain10', 'gain25', 'rotation10', 'rotation25', 'coupled_ks35',
             'coupled_sk35', 'coupled_null35', 'label_noise10', 'label_noise20',
             'time_cov_plus35', 'time_cov_minus35')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def config(kind='recolor', alpha=.5, shrinkage=.05, weighting='pooled', **kw):
    return {'kind': kind, 'alpha': alpha, 'shrinkage': shrinkage,
            'covariance_weighting': weighting, **kw}


def reference_key(seed):
    return f'class-reference-seed{seed}'


def meta_key(kind, mode, weighting, seed):
    return f'meta-{kind}-{mode}-{weighting}-seed{seed}'


def core_methods():
    methods = [
        ('source_frozen', 'source_frozen', config('identity'), None),
        ('support_EA', 'support_EA', config('ea', 1), None),
        ('head_sgd10', 'head_sgd10', config('identity'), None),
        ('head_sgd100', 'head_sgd100', config('identity'), None),
        ('full_bures', 'transport', config('bures', 1), None),
        ('full_metric_bures', 'transport', config('metric_bures', 1), None),
        ('partial_recolor', 'transport', config('recolor', .5), None),
        ('partial_bures', 'transport', config('bures', .5), None),
        ('balanced_EA', 'transport', config('recolor', 1, weighting='class_balanced'), None),
        ('balanced_bures', 'transport', config('bures', 1, weighting='class_balanced'), None),
    ]
    for kind in ('recolor', 'bures'):
        for mode in ('global', 'evidence'):
            for weighting in ('pooled', 'class_balanced'):
                group = f'learned_{kind}_{mode}_{weighting}'
                methods.append((group, 'transport', config(kind, parameterization=mode, weighting=weighting),
                                (kind, mode, weighting)))
    for name, beta, rule, control in (
        ('fiber_zero', 0., 'fixed', 'aligned'),
        ('fiber_fixed', .1, 'fixed', 'aligned'),
        ('fiber_bounded', .35, 'bounded', 'aligned'),
        ('fiber_reversed', .35, 'bounded', 'reversed'),
        ('fiber_random', .35, 'bounded', 'random'),
    ):
        methods.append((name, 'transport', config('fiber', 1, beta=beta, beta_rule=rule,
                                                 fiber_control=control), None))
    return methods


COMPARISONS = [
    ['full_bures', 'support_EA'],
    ['full_metric_bures', 'full_bures'],
    ['partial_bures', 'full_bures'],
    ['learned_bures_global_pooled', 'support_EA'],
    ['learned_bures_evidence_pooled', 'learned_bures_global_pooled'],
    ['learned_bures_evidence_class_balanced', 'learned_bures_evidence_pooled'],
    ['fiber_bounded', 'support_EA'],
    ['fiber_bounded', 'fiber_reversed'],
    ['fiber_bounded', 'fiber_random'],
    ['fiber_bounded', 'fiber_fixed'],
]


def _evaluation(group, variant, cfg, dep, seed, dataset, family='core', scenarios=('clean',), budgets=BUDGETS):
    scientific = {'group': group, 'variant': variant, 'config': cfg, 'seed': seed,
                  'partition': dataset, 'scenarios': list(scenarios), 'budgets': list(budgets)}
    key = f'{family}-{dataset.split("_")[0]}-{group}-seed{seed}-{digest(scientific)[:10]}'
    item = dict(scientific, key=key, family=family, requires=[])
    if dep:
        item['meta_key'] = meta_key(*dep, seed)
        item['requires'].append(item['meta_key'])
    if cfg['kind'] == 'fiber':
        item['reference_key'] = reference_key(seed)
        item['requires'].append(item['reference_key'])
    if dataset != 'holdout':
        external = 'pan2023' if dataset.startswith('pan') else 'weibo'
        item['external_dataset'] = 'Pan2023' if external == 'pan2023' else 'Weibo2014'
        item['external_manifest'] = str(PACKAGE / 'data' / external / 'manifest.json')
    return item


def build(root, output=None):
    """Prepare the predeclared experiment; repeated launch preserves its identity."""
    root = Path(root).expanduser().resolve()
    output = Path(output or root/'runs'/'followup_20261025_r8').resolve()
    tasks = []
    for seed in SEEDS:
        tasks.append({'key': reference_key(seed), 'group': 'class_reference', 'family': 'class_reference',
                      'seed': seed, 'requires': [], 'config': config('identity'),
                      'scope': 'meta_train_only_participant_then_session_equal_weight_prefix20'})
    # Immediate reference results appear before learned fits finish.
    methods = core_methods()
    for group, variant, cfg, dep in methods[:10]:
        for seed in SEEDS:
            for dataset in DATASETS:
                tasks.append(_evaluation(group, variant, cfg, dep, seed, dataset))
    for kind in ('recolor', 'bures'):
        for mode in ('global', 'evidence'):
            for weighting in ('pooled', 'class_balanced'):
                for seed in SEEDS:
                    tasks.append({'key': meta_key(kind, mode, weighting, seed), 'family': 'train',
                        'group': f'meta_{kind}_{mode}_{weighting}', 'seed': seed, 'requires': [],
                        'config': config(kind, parameterization=mode, weighting=weighting),
                        'updates': 1000, 'episodes_per_update': 4, 'checkpoint_every': 10,
                        'validate_every': 100, 'meta_lr': .01})
    for group, variant, cfg, dep in methods[10:]:
        for seed in SEEDS:
            for dataset in DATASETS:
                tasks.append(_evaluation(group, variant, cfg, dep, seed, dataset))
    # Paired mechanism experiments, same future query and same synthetic shift.
    selected = {'source_frozen', 'support_EA', 'full_bures', 'full_metric_bures',
                'learned_bures_evidence_pooled', 'fiber_bounded', 'fiber_reversed', 'fiber_random'}
    for group, variant, cfg, dep in methods:
        if group not in selected:
            continue
        for seed in SEEDS:
            for dataset in DATASETS:
                for scenario in SCENARIOS:
                    tasks.append(_evaluation(group, variant, cfg, dep, seed, dataset,
                        family='stress', scenarios=(scenario,), budgets=(20,)))
    # A fixed permutation preserves support class counts and pooled covariance.
    # Identity/EA predictions must remain unchanged, while fiber uses the labels.
    for group, variant, cfg, dep in methods:
        if group not in {'source_frozen', 'support_EA', 'fiber_bounded'}:
            continue
        for seed in SEEDS:
            for dataset in DATASETS:
                tasks.append(_evaluation(group, variant, cfg, dep, seed, dataset,
                    family='label_null', scenarios=('support_label_permutation',), budgets=(20,)))
    # Broad fixed response surfaces. Descriptive; not a winner-selection set.
    for kind in ('recolor', 'bures', 'metric_bures'):
        for alpha in (.125, .25, .5, .75, 1.):
            for eta in (.001, .01, .05, .15, .35, .75):
                for weighting in ('pooled', 'class_balanced'):
                    cfg = config(kind, alpha, eta, weighting)
                    # Identical core configurations are already evaluated at B20.
                    if any(variant == 'transport' and dep is None and existing == cfg
                           for _, variant, existing, dep in methods):
                        continue
                    group = f'grid_{kind}_a{str(alpha).replace(".","p")}_e{str(eta).replace(".","p")}_{weighting}'
                    for seed in SEEDS:
                        for dataset in DATASETS:
                            tasks.append(_evaluation(group, 'transport', cfg, None, seed, dataset,
                                                     family='sensitivity', budgets=(20,)))
    for weighting in ('pooled', 'class_balanced'):
        for beta in (.025, .05, .2, .5):
            for control in ('aligned', 'reversed', 'random'):
                group = f'fiber_grid_b{str(beta).replace(".","p")}_{weighting}_{control}'
                cfg = config('fiber', 1, weighting=weighting, beta=beta, beta_rule='fixed',
                             fiber_control=control, fiber_seed=20261025)
                for seed in SEEDS:
                    for dataset in DATASETS:
                        tasks.append(_evaluation(group, 'transport', cfg, None, seed, dataset,
                                                 family='fiber_sensitivity', budgets=(20,)))
    identities = []
    for task in tasks:
        value = dict(task)
        if 'external_manifest' in value:
            path = Path(value['external_manifest'])
            value['external_manifest'] = {'dataset': value['external_dataset'],
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        identities.append(value)
    source_files = sorted((PACKAGE/'paper3r8').glob('*.py'))+sorted((PACKAGE/'paper3r7').glob('*.py'))+sorted((PACKAGE/'p3work').glob('*.py'))
    hashes = {str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    scientific = {'schema': 'paper3-round8-v1', 'tasks': identities, 'code': hashes,
                  'comparisons': COMPARISONS}
    return {'schema_version': 'paper3-round8-plan-v1', 'round': 8, 'root': str(root),
        'output': str(output), 'recipe_sha256': digest(scientific), 'code_sha256': hashes,
        'tasks': tasks, 'seeds': list(SEEDS), 'comparisons': COMPARISONS,
        'primary_partitions': list(DATASETS[1:]), 'primary_metrics': ['ba', 'nll_balanced'],
        'expected_cohorts': {'holdout': {'participants': 11, 'sessions': 105},
            DATASETS[1]: {'participants': 14, 'sessions': 28}, DATASETS[2]: {'participants': 10, 'sessions': 10}},
        'resources': {'partition': 'hpc_gpu', 'jobs': 3, 'gpus_per_job': 1,
                      'cpus_per_job': 16, 'memory_per_job': '64G', 'hardware': 'A100-PCIe-40GB'},
        'full_test_at_start': False, 'compute_node_downloads': False,
        'data_scope': 'Existing Stieger cache and bundled Pan/Weibo cropped caches; repeated exploratory validation',
        'interpretation': 'Finite response-surface experiments; seeds/sessions are not independent participants; no guaranteed improvement'}
