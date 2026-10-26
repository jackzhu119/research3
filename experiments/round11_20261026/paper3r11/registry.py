"""Bounded, standard-library-only discovery of existing R3/R4/R8/R9 inputs.

Reads declared plans, small receipts, and selected small model artifacts only.
Never imports scientific modules, scans predictions/raw EEG, or starts jobs.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import tempfile

SCHEMA = 'paper3-r11-study-registry-v1'
SEEDS = (20261018, 20261019, 20261020)
ROUND_DIRS = {3: 'followup_20261024_r3', 4: 'followup_20261024_r4',
              8: 'followup_20261025_r8', 9: 'followup_20261025_r9'}
JSON_LIMIT = 16 * 1024 * 1024
MODEL_LIMIT = 64 * 1024 * 1024
GP_FIELDS = frozenset(('kernel', 'tau', 'lengthscale', 'jitter', 'predictive',
                       'gh_points', 'anchor_cap', 'max_iterations', 'tolerance'))
SPATIAL = ('sgd10', 'sgd100', 'r3_protected', 'r3_tangent_protected', 'r4_xfit_select_stack')
ALIASES = {'r9_raw_rbf_anchored': 'r9_raw_anchored',
           'r9_ea_linear_anchored': 'r9_ea_anchored'}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path):
    path = Path(path)
    if path.stat().st_size > JSON_LIMIT:
        raise ValueError('Metadata JSON exceeds bounded discovery size: ' + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Metadata JSON must be an object: ' + str(path))
    return value


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def describe(path, root, *, maximum=MODEL_LIMIT):
    """Bind selected small artifacts; explicit external paths stay absolute."""
    path, root = Path(path).resolve(), Path(root).resolve()
    size = path.stat().st_size
    if size > maximum:
        raise ValueError('Selected artifact exceeds bounded discovery size: ' + str(path))
    return {'path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            'bytes': size, 'sha256': _sha(path)}


def _path(value, root, relative=None):
    if isinstance(value, dict):
        value = value.get('path')
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    # Declared project-relative artifact paths use the original project root;
    # bare filename/./ plan references are resolved next to their launch file.
    base = relative if relative is not None else root
    return (Path(base) / path).resolve()


def _plan_references(value, root, launch_parent):
    found = []
    keys = frozenset(('plan', 'plan_path', 'backend_plan', 'backend_plan_path',
                      'worker_plan', 'task_plan', 'payload_plan', 'plan_file'))
    def visit(item, key=None, depth=0):
        if depth > 8:
            return
        if isinstance(item, dict):
            for k, child in item.items():
                if k in keys:
                    path = _path(child, root, launch_parent)
                    if path is not None:
                        found.append(path)
                visit(child, k, depth + 1)
        elif isinstance(item, list):
            # Native launch metadata sometimes retains command argv.
            for i, child in enumerate(item):
                if child == '--plan' and i + 1 < len(item):
                    path = _path(item[i + 1], root, launch_parent)
                    if path is not None:
                        found.append(path)
                visit(child, key, depth + 1)
        elif isinstance(item, str) and key in ('command', 'cmd'):
            try:
                parts = shlex.split(item)
            except ValueError:
                return
            for i, part in enumerate(parts[:-1]):
                if part == '--plan':
                    path = _path(parts[i + 1], root, launch_parent)
                    if path is not None:
                        found.append(path)
    visit(value)
    return found


def _discover_plans(root, audit):
    plans = {r: [] for r in ROUND_DIRS}
    seen = set()
    for number, dirname in ROUND_DIRS.items():
        folder = root / 'runs' / dirname
        candidates = [folder / name for name in (
            'plan.json', 'runtime/plan.json', 'runtime/three_jobs/plan.json',
            'runtime/three_jobs/backend-plan.json', 'launch.json',
            'runtime/launch.json', 'runtime/three_jobs/launch.json')]
        while candidates and len(seen) < 64:
            path = candidates.pop(0).resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                value = _read(path)
                candidates.extend(p for p in _plan_references(value, root, path.parent) if p not in seen)
                if not isinstance(value.get('tasks'), list):
                    continue
                if Path(value.get('root', root)).resolve() != root:
                    raise ValueError('Declared plan belongs to a different project root')
                output = _path(value.get('output', str(folder)), root)
                if output is None:
                    raise ValueError('Task plan lacks output directory')
                declared_round = int(value.get('round', number))
                if declared_round != number:
                    raise ValueError('Declared plan round differs from discovery slot')
                descriptor = describe(path, root, maximum=JSON_LIMIT)
                if any(p['descriptor']['sha256'] == descriptor['sha256'] for p in plans[number]):
                    continue
                plans[number].append({'value': value, 'path': path, 'output': output,
                                      'descriptor': descriptor})
                audit.append({'event': 'PLAN_DISCOVERED', 'round': number, **descriptor})
            except (OSError, ValueError, TypeError) as error:
                audit.append({'event': 'PLAN_REJECTED', 'round': number,
                              'path': str(path), 'reason': str(error)})
    return plans


def _task_key(task):
    key = task.get('key')
    return key if isinstance(key, str) and key not in ('.', '..') and Path(key).name == key else None


def _completed(plan, task, root, audit):
    key = _task_key(task)
    if key is None:
        return None
    folder = plan['output'] / key
    receipt = folder / 'result.json'
    if not receipt.is_file():
        return None
    try:
        result = _read(receipt)
        if result.get('status') != 'COMPLETE':
            return None
        if result.get('task') != task:
            raise ValueError('Completed task differs from declared task')
        if result.get('holdout_used_for_selection') is not False:
            raise ValueError('Completed learned artifact lacks holdout-exclusion proof')
        if result.get('recipe_sha256') is None and task.get('family') == 'basis':
            # R3/R4 encode recipe in their native identity, not a separate
            # result.recipe_sha256 field. Recompute that exact small identity.
            training = {'updates': task.get('updates', task.get('config', {}).get('updates',
                task.get('config', {}).get('outer_updates', plan['value'].get('updates', 1000)))),
                'validate_every': task.get('validate_every', plan['value'].get('validate_every', 50))}
            expected = _digest({'task': task, 'data': result.get('data_identity'),
                'source': result.get('source_identity'), 'recipe': plan['value'].get('recipe_sha256'),
                'basis': None, 'training': training})
            if result.get('identity') != expected:
                raise ValueError('Completed native basis identity differs from recipe/source/data/task')
        elif result.get('recipe_sha256') != plan['value'].get('recipe_sha256'):
            raise ValueError('Completed task recipe differs from declared plan')
        locator = result.get('selected_checkpoint', {})
        path = _path(locator, root, plan['output'])
        if path is None or not path.is_relative_to(folder.resolve()):
            raise ValueError('Selected checkpoint is outside the declared task folder')
        model = describe(path, root)
        if model['sha256'] != locator.get('sha256'):
            raise ValueError('Selected checkpoint SHA256 differs from completed receipt')
        return {'checkpoint': model, 'result': describe(receipt, root, maximum=JSON_LIMIT),
                'receipt': result, 'plan': plan, 'task': task}
    except (OSError, ValueError, TypeError) as error:
        audit.append({'event': 'SELECTED_ARTIFACT_REJECTED', 'path': str(receipt), 'reason': str(error)})
        return None


def _proof(selected, role):
    receipt = selected['receipt']
    development = selected['task'].get('family') in ('basis', 'select')
    proof = {'role': role, 'seed': selected['task']['seed'],
            'source_identity': receipt.get('source_identity'),
            'data_identity': receipt.get('data_identity'),
            'native_identity': receipt.get('identity'),
            'native_task_key': selected['task']['key'],
            'result': selected['result'], 'result_descriptor': selected['result'],
            'plan': selected['plan']['descriptor'],
            'target_query_labels_used': False, 'target_future_queries_read': False,
            'development_query_labels_used': development,
            'development_future_queries_read': development,
            'holdout_used_for_selection': False,
            'selection_role': 'meta_validation'}
    if selected['task'].get('family') == 'basis':
        proof['training_role'] = 'meta_train'
    if not development:
        proof.update(query_labels_used=False, future_queries_read=False)
    return proof


def _with_field(item, field):
    return dict(item, field=list(field))


def _core_basis(plans, round_number, seed, variant, root, audit):
    """Resolve the historical primary legacy main task's DECLARED dictionary.

    No choice among dictionaries is made using a holdout score. The historical
    primary controls all name their legacy learned basis explicitly.
    """
    for plan in plans[round_number]:
        tasks = plan['value']['tasks']
        candidates = [t for t in tasks if t.get('family') == 'core'
                      and t.get('seed') == seed and t.get('variant') == variant
                      and t.get('config', {}).get('rank', 4) == 4
                      and t.get('config', {}).get('radius', .15) == .15
                      and t.get('config', {}).get('head_l2', .1) == .1
                      and 'legacy' in str(t.get('key', ''))]
        for task in candidates:
            key = task.get('basis_key')
            basis_output = _path(task.get('basis_output', str(plan['output'])), root)
            source_plan = plan
            if basis_output != plan['output']:
                source_plan = next((p for candidates in plans.values() for p in candidates if p['output'] == basis_output), None)
            if source_plan is None:
                continue
            declared = next((t for t in source_plan['value']['tasks'] if t.get('key') == key), None)
            if not declared or declared.get('family') != 'basis' or declared.get('seed') != seed:
                continue
            if declared.get('config', {}).get('rank', 4) != 4 or declared.get('config', {}).get('radius', .15) != .15:
                continue
            selected = _completed(source_plan, declared, root, audit)
            if selected:
                selected['evaluation_config'] = task.get('config', {})
                selected['evaluation_task'] = task
                return selected
    # Explicit native plan may retain only dependency tasks: accept its fixed
    # original legacy key, never the best-looking protected/dictionary score.
    for plan in plans[round_number]:
        key = f'basis-legacy-r4-seed{seed}'
        task = next((t for t in plan['value']['tasks'] if t.get('key') == key
                     and t.get('family') == 'basis' and t.get('seed') == seed), None)
        if task and task.get('config', {}).get('rank', 4) == 4 and task.get('config', {}).get('radius', .15) == .15:
            selected = _completed(plan, task, root, audit)
            if selected:
                return selected
    return None


def _merge(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(left.get(key), dict):
            _merge(left[key], value)
        else:
            left[key] = copy.deepcopy(value)
    return left


def _internal_metadata(root, registry, audit):
    """Follow W3's declared small manifest/split; never open an EEG array."""
    try:
        tasks_path = root / 'status/tasks.json'
        tasks = _read(tasks_path)
        declared = tasks['W3']['cache_manifest']
        manifest_path = _path(declared, root)
        manifest = _read(manifest_path)
        manifest_desc = describe(manifest_path, root, maximum=JSON_LIMIT)
        if isinstance(declared, dict) and declared.get('sha256') != manifest_desc['sha256']:
            raise ValueError('W3 cache manifest SHA256 differs from its declaration')
        if manifest.get('status') != 'STIEGER_CACHE_COMPLETE':
            raise ValueError('W3 cache metadata is not complete')
        split_declared = manifest['participant_split']
        split_path = _path(split_declared, root)
        split = _read(split_path)
        split_desc = describe(split_path, root, maximum=JSON_LIMIT)
        if isinstance(split_declared, dict) and split_declared.get('sha256') != split_desc['sha256']:
            raise ValueError('Participant split file SHA256 differs from declared manifest')
        people = [p for group in split['partitions'].values() for p in group]
        if len(people) != len(set(people)):
            raise ValueError('Participant split overlaps')
        identity = {'manifest_sha256': manifest_desc['sha256'],
                    'split_sha256': split['split_sha256'],
                    'signal_contract_sha256': manifest['input_contract_sha256']}
        if any(not isinstance(v, str) or len(v) != 64 for v in identity.values()):
            raise ValueError('Internal catalog identity must contain three concrete SHA256 values')
        registry.update(internal_cache_manifest=manifest_desc,
                        participant_split=split_desc, internal_data_identity=identity)
        contract = root / 'contracts/input_contract.json'
        if contract.is_file():
            registry['input_contract'] = describe(contract, root, maximum=JSON_LIMIT)
    except (OSError, ValueError, TypeError, KeyError) as error:
        audit.append({'event': 'INTERNAL_DATA_METADATA_UNAVAILABLE', 'reason': str(error)})


def _validate_override_descriptors(value, root, checked=None):
    checked = {} if checked is None else checked
    if isinstance(value, dict):
        if 'path' in value and ('sha256' in value or 'bytes' in value):
            path = _path(value, root)
            actual = checked.setdefault(path, None)
            if actual is None:
                actual = checked[path] = describe(path, root)
            if any(value.get(k) != actual[k] for k in ('bytes', 'sha256')):
                raise ValueError('Explicit override artifact bytes/SHA256 differ: ' + str(path))
        for child in value.values():
            _validate_override_descriptors(child, root, checked)
    elif isinstance(value, list):
        for child in value:
            _validate_override_descriptors(child, root, checked)


def build_registry(root, home=None, overrides=None):
    root = Path(root).expanduser().resolve()
    home = Path(home).expanduser().resolve() if home is not None else Path.home()
    registry = {'schema_version': SCHEMA, 'root': str(root),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': list(SEEDS),
                'source_by_seed': {}, 'external_manifests': {}, 'baseline_artifacts': {},
                'profile_reference_by_seed': {}, 'profile_raw_reference_by_seed': {},
                'missing_artifacts': [], 'discovery_audit': [],
                'discovery_policy': {'imports_torch': False, 'reads_raw_eeg': False,
                    'recursive_scans': False, 'school_jobs_modified': False,
                    'baseline_artifacts_orientation': 'seed_then_method',
                    'home_used_only_for_explicit_override_resolution': str(home)}}
    audit = registry['discovery_audit']
    _internal_metadata(root, registry, audit)
    plans = _discover_plans(root, audit)
    for seed in SEEDS:
        slot = str(seed); source = {}
        for key, path in (
            ('source_receipt', root / f'runs/source_training_v2/f0_receipt_seed-{seed}.json'),
            ('reference_receipt', root / f'runs/training_campaign_v1/references/seed-{seed}/receipt.json')):
            try:
                _read(path)
                source[key] = describe(path, root, maximum=JSON_LIMIT)
            except (OSError, ValueError) as error:
                audit.append({'event': 'SOURCE_RECEIPT_UNAVAILABLE', 'seed': seed, 'path': str(path), 'reason': str(error)})
        if len(source) == 2:
            registry['source_by_seed'][slot] = source
        artifacts = registry['baseline_artifacts'].setdefault(slot, {})
        for method in SPATIAL:
            n = 4 if method == 'r4_xfit_select_stack' else 3
            variant = {'r3_protected': 'information_protected', 'r3_tangent_protected': 'tangent_protected',
                       'r4_xfit_select_stack': 'xfit_select_stack'}.get(method, method)
            selected = _core_basis(plans, n, seed, variant, root, audit)
            if selected is None and n == 4:
                selected = _core_basis(plans, 3, seed, 'sgd10', root, audit)
            if selected:
                proof = _proof(selected, 'meta_train')
                proof['native_evaluation_config'] = selected.get('evaluation_config')
                artifacts[method] = {'spatial_basis': _with_field(selected['checkpoint'], ['b']),
                                     'basis_provenance': proof}
        for plan in plans[8]:
            for task in plan['value']['tasks']:
                if task.get('family') == 'class_reference' and task.get('seed') == seed:
                    selected = _completed(plan, task, root, audit)
                    if selected and selected['receipt'].get('reference_scope') == 'meta_train_first20_acquired_prefix_only':
                        artifacts['r8_fiber_bounded'] = {
                            'class_reference': _with_field(selected['checkpoint'], ['class_reference']),
                            'class_reference_provenance': _proof(selected, 'meta_train')}
                        break
        for mode, profile_key in (('ea', 'profile_reference_by_seed'), ('raw', 'profile_raw_reference_by_seed')):
            ref = config = None
            representative_kernel = 'rbf' if mode == 'raw' else 'linear'
            for plan in plans[9]:
                for task in plan['value']['tasks']:
                    if task.get('seed') != seed or task.get('config', {}).get('preprocessing', 'raw') != mode:
                        continue
                    if task.get('family') == 'feature_reference':
                        candidate = _completed(plan, task, root, audit)
                        scope = candidate['receipt'].get('reference_scope') if candidate else None
                        if candidate and isinstance(scope, dict) and scope.get('role') == 'meta_train_prefix20_only' and scope.get('preprocessing') == mode and scope.get('future_queries_read') is False:
                            ref = candidate
                    elif task.get('family') == 'select' and task.get('config', {}).get('kernel') == representative_kernel and task.get('selection_role') == 'meta_validation':
                        candidate = _completed(plan, task, root, audit)
                        if candidate and isinstance(candidate['receipt'].get('selected_config'), dict):
                            cfg = candidate['receipt']['selected_config']
                            if cfg.get('kernel') == representative_kernel and cfg.get('preprocessing', 'raw') == mode and cfg.get('predictive') == 'anchored_laplace':
                                config = candidate
            if ref:
                registry[profile_key][slot] = {'checkpoint': ref['checkpoint'], 'result': ref['result']}
            if ref and config:
                values = {k: v for k, v in config['receipt']['selected_config'].items() if k in GP_FIELDS}
                artifacts['r9_' + mode + '_anchored'] = {
                    'r9_reference_' + mode: _with_field(ref['checkpoint'], ['reference']),
                    'r9_reference_provenance_' + mode: _proof(ref, 'meta_train'),
                    'r9_config_' + mode: values,
                    'r9_config_provenance_' + mode: _proof(config, 'meta_validation')}
    for plan in plans[9]:
        for task in plan['value']['tasks']:
            value = task.get('external_manifest', plan['value'].get('external_manifest'))
            if not value:
                continue
            kind = str(task.get('external_dataset', task.get('partition', ''))).lower()
            name = 'pan2023' if 'pan' in kind else 'weibo2014' if 'weibo' in kind else None
            if name is None or name in registry['external_manifests']:
                continue
            path = _path(value, root, plan['path'].parent)
            try:
                metadata = _read(path)
                if metadata.get('dataset_id') not in (None, 'Pan2023' if name == 'pan2023' else 'Weibo2014'):
                    raise ValueError('External manifest dataset identity differs')
                registry['external_manifests'][name] = describe(path, root, maximum=JSON_LIMIT)
            except (OSError, ValueError, TypeError) as error:
                audit.append({'event': 'EXTERNAL_MANIFEST_UNAVAILABLE', 'dataset': name, 'path': str(path), 'reason': str(error)})
    if overrides is not None:
        value = _read(overrides) if isinstance(overrides, (str, os.PathLike)) else overrides
        if not isinstance(value, dict) or value.get('schema_version', SCHEMA) != SCHEMA or Path(value.get('root', root)).resolve() != root:
            raise ValueError('Registry override schema/root differs')
        _validate_override_descriptors(value, root)
        _merge(registry, {k: v for k, v in value.items() if k not in ('schema_version', 'root', 'created_utc', 'missing_artifacts')})
        audit.append({'event': 'EXPLICIT_REGISTRY_OVERRIDE_APPLIED'})
    # Derive semantic availability after overrides; source/reference problems are
    # preparation diagnostics, never triggers to restart old training jobs.
    missing = []
    if not registry.get('internal_cache_manifest') or not registry.get('internal_data_identity'):
        missing.append({'scope': 'mechanism', 'kind': 'internal_catalog_metadata',
                        'expected': ['status/tasks.json:W3.cache_manifest', 'cache_manifest.participant_split']})
    for seed in registry['seeds']:
        slot = str(seed)
        if slot not in registry['source_by_seed']:
            missing.append({'seed': seed, 'scope': 'mechanism', 'kind': 'frozen_source_and_reference_receipts',
                            'expected': [f'runs/source_training_v2/f0_receipt_seed-{seed}.json', f'runs/training_campaign_v1/references/seed-{seed}/receipt.json']})
        if slot not in registry['profile_reference_by_seed']:
            missing.append({'seed': seed, 'scope': 'mechanism_reuse', 'kind': 'completed_R9_EA_feature_reference',
                            'expected_task': f'feature-reference-ea-seed{seed}'})
        for method in (*SPATIAL, 'r8_fiber_bounded', 'r9_raw_anchored', 'r9_ea_anchored'):
            if not registry['baseline_artifacts'].get(slot, {}).get(method):
                missing.append({'seed': seed, 'scope': 'formal', 'kind': 'baseline_artifact', 'method': method})
    for name in ('pan2023', 'weibo2014'):
        if name not in registry['external_manifests']:
            missing.append({'scope': 'external_exploratory', 'kind': 'external_manifest', 'dataset': name})
    registry['missing_artifacts'] = missing
    return registry


ARTIFACT_BASELINES = frozenset((*SPATIAL, 'r8_fiber_bounded',
                                'r9_raw_anchored', 'r9_ea_anchored'))


def baseline_availability(registry, method, seed):
    """Check one optional historical task without blocking unrelated methods."""
    method = ALIASES.get(method, method)
    seed = int(seed)
    row = {'seed': seed, 'method': method, 'scope': 'optional_baseline'}
    if method not in ARTIFACT_BASELINES:
        return dict(row, status='AVAILABLE', requires_historical_artifact=False)
    artifacts = registry.get('baseline_artifacts', {}).get(str(seed), {}).get(method)
    if not artifacts:
        return dict(row, status='UNAVAILABLE', kind='baseline_artifact',
                    reason='Verified historical artifacts are absent; this task alone is unavailable')
    try:
        _validate_override_descriptors(artifacts, Path(registry['root']).resolve())
    except (OSError, ValueError, TypeError, KeyError) as error:
        return dict(row, status='UNAVAILABLE', kind='artifact_identity_changed', reason=str(error))
    return dict(row, status='AVAILABLE', requires_historical_artifact=True)


def validate_registry(registry, *, stage='development', methods=None, seeds=None, cohorts=None):
    """Check required data; report optional historical tasks in a separate ledger."""
    seeds = {int(s) for s in (registry.get('seeds', SEEDS) if seeds is None else seeds)}
    methods = {ALIASES.get(m, m) for m in (methods or ())}
    cohorts = {'pan2023' if str(c).startswith('pan2023') else 'weibo2014' if str(c).startswith('weibo2014') else c for c in (cohorts or ())}
    # Match study.build_plan: stage expansion must not change the reference
    # geometry. Missing optional R9 reuse artifacts select the documented
    # meta-train-prefix construction, rather than blocking its availability.
    reuse = all(str(s) in registry.get('profile_reference_by_seed', {})
                and str(s) in registry.get('profile_raw_reference_by_seed', {}) for s in SEEDS)
    errors = []
    optional_methods = methods & ARTIFACT_BASELINES if methods else ARTIFACT_BASELINES
    unavailable = [value for seed in sorted(seeds) for method in sorted(optional_methods)
                   if (value := baseline_availability(registry, method, seed))['status'] != 'AVAILABLE']
    for row in registry.get('missing_artifacts', []):
        if 'seed' in row and row['seed'] not in seeds:
            continue
        if row['scope'] in ('mechanism','development','campaign') or (row['scope'] in ('mechanism_reuse','development_reuse') and reuse):
            errors.append(row)
        elif row['scope'] == 'external_exploratory' and row.get('dataset') in cohorts:
            errors.append(row)
    # Recheck selected small artifacts against their preparation descriptors so
    # a later changed/deleted dependency is rejected before a GPU allocation.
    root = Path(registry['root']).resolve()
    selected = {k: registry.get(k) for k in ('internal_cache_manifest', 'participant_split', 'input_contract')}
    for seed in seeds:
        slot = str(seed)
        selected['source_' + slot] = registry.get('source_by_seed', {}).get(slot)
        if reuse:
            selected['profile_' + slot] = registry.get('profile_reference_by_seed', {}).get(slot)
    selected['external'] = {k: v for k, v in registry.get('external_manifests', {}).items() if k in cohorts}
    try:
        _validate_override_descriptors(selected, root)
    except (OSError, ValueError, TypeError) as error:
        errors.append({'scope': stage, 'kind': 'artifact_identity_changed', 'reason': str(error)})
    return {'status': 'READY' if not errors else 'MISSING_INPUTS', 'stage': stage, 'missing_artifacts': errors,
            'unavailable_baselines': unavailable,
            'optional_baseline_policy': 'isolate_method_and_seed_keep_other_tasks_available',
            'reference_strategy': 'reuse_r9_ea' if reuse else 'build_meta_train_prefix',
            'torch_imported': False, 'gpu_allocated': False, 'jobs_submitted': 0}


def write_registry(registry, path=None, *, home=None, overrides=None):
    """Atomically write a NEW R11 preparation record, returning its descriptor."""
    if not isinstance(registry, dict):
        registry = build_registry(registry, home=home, overrides=overrides)
    root = Path(registry['root']).resolve()
    path = Path(path) if path is not None else root / 'runs/round11_preparation/study-registry.json'
    path = path.expanduser().resolve()
    if not path.is_relative_to(root / 'runs/round11_preparation'):
        raise ValueError('Registry writes are restricted to the new R11 preparation directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        previous = _read(path)
        before = {k: v for k, v in previous.items() if k != 'created_utc'}
        after = {k: v for k, v in registry.items() if k != 'created_utc'}
        if before == after:
            # A repeat preparation must not change the immutable study input
            # descriptor merely because the operator ran validation again.
            registry['created_utc'] = previous.get('created_utc')
            return describe(path, root, maximum=JSON_LIMIT)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(registry, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return describe(path, root, maximum=JSON_LIMIT)
