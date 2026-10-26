"""Isolated R10 scientific worker. Operational retries/uploads live elsewhere."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

# This precedes every Torch import, including lazy native/source imports.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

from . import algorithm
from .data import (SessionAccess, atomic_npz, cached_feature_batch, describe, feature_batch, input_artifact,
                   load_registry, load_source, native, open_catalog, selected_sessions,
                   weighted_reference)
from .metrics import participant_summaries, session_metrics
from .platform.common import (atomic, atomic_bytes, digest, emit, load_plan, locked,
                              read, sha, task_identity, utc, verify_descriptor, verify_result)

_STOP_REQUESTED = threading.Event()
_PROGRESS_LOCK = threading.RLock()
PROBABILITY_FLOOR = 1e-12


class PauseRequested(Exception):
    pass


def _check_stop(folder, output):
    if (_STOP_REQUESTED.is_set() or (folder / 'STOP_REQUESTED').exists()
            or (Path(output) / 'STOP_REQUESTED').exists()):
        raise PauseRequested('Stopped between atomic scientific session units; completed work retained')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(x) for x in value]
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy().tolist()
    return value


def progress(folder, task, **values):
    record = {'task_key': task['key'], 'updated_utc': utc(),
              'attempt_token': os.environ.get('PAPER3_R10_ATTEMPT_TOKEN'), **values}
    record['scientific_progress_utc'] = record['updated_utc']
    with _PROGRESS_LOCK:
        atomic(folder / 'progress.json', record)
    emit('R10_SCIENTIFIC_PROGRESS', **record)


def _csv(path, rows):
    fields = sorted({k for row in rows for k in row})
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    atomic_bytes(path, handle.getvalue().encode())


def _tensor_numpy(value):
    return value.detach().double().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _save_baseline_state(path, fitted):
    """Pickle-free replay archive; fitted model weights are never included."""
    arrays = {}
    def walk(value):
        if isinstance(value, np.ndarray) or hasattr(value, 'detach'):
            key = 'array_' + str(len(arrays))
            arrays[key] = value.detach().cpu().numpy() if hasattr(value, 'detach') else value.copy()
            return {'array': key}
        if isinstance(value, dict):
            return {str(k): walk(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [walk(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise ValueError('Unsupported baseline replay field: ' + type(value).__name__)
    tree = walk(fitted.state_dict())
    atomic_npz(path, metadata=np.asarray(json.dumps(tree, sort_keys=True, allow_nan=False)), **arrays)


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(1, keepdims=True)
    result = np.exp(logits)
    return result / result.sum(1, keepdims=True)


def _log_softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(1, keepdims=True))


def _bounded_binary_probability(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all()
            or np.any(value < 0) or np.any(value > 1)
            or not np.allclose(value.sum(1), 1., atol=1e-8, rtol=0)):
        raise ValueError('Finite binary native probabilities required before common scoring')
    bounded = np.clip(value, PROBABILITY_FLOOR, 1. - PROBABILITY_FLOOR)
    return bounded / bounded.sum(1, keepdims=True)


def _baseline_prediction(fitted, query_x, batch_size=64):
    """Identical frozen-encoder batch size for every method, no batch fitting."""
    values, logs, diagnostics = [], [], []
    for offset in range(0, len(query_x), batch_size):
        values.append(_tensor_numpy(fitted.predict_proba(query_x[offset:offset + batch_size])))
        logs.append(_tensor_numpy(fitted.last_log_probabilities))
        diagnostics.append(jsonable(fitted.diagnostics.get('prediction', {})))
    fitted.diagnostics['query_batch_policy'] = {'batch_size': batch_size,
                                              'normalization_statistics_updated': False,
                                              'prediction_batches': diagnostics}
    return np.concatenate(values), np.concatenate(logs)


def _artifact_references(plan, registry, method, seed, device, source_identity=None):
    """Load only explicitly registered historical fit/reference artifacts."""
    collection = registry.get('baseline_artifacts', {})
    collection = collection.get(str(seed), collection)
    entries = collection.get(method, {})
    aliases = {'frozen': 'source_frozen', 'support_ea': 'support_EA',
               'r9_raw_anchored': 'r9_raw_rbf_anchored',
               'r9_ea_anchored': 'r9_ea_linear_anchored'}
    if not entries:
        entries = collection.get(aliases.get(method, method), {})
    result = {}
    for key, item in entries.items():
        if key.endswith('provenance') or '_provenance_' in key:
            if not isinstance(item, dict) or int(item.get('seed', -1)) != seed:
                raise ValueError('Historical baseline provenance requires the exact source seed')
            if source_identity is not None and item.get('source_identity') != source_identity:
                raise ValueError('Historical baseline was fitted with another frozen source checkpoint')
            if item.get('target_query_labels_used') is True or item.get('holdout_used_for_selection') is True:
                raise ValueError('Historical baseline provenance cannot use target queries')
            if item.get('result'):
                result_path = input_artifact(plan['root'], item['result'])
                result_proof = json.loads(result_path.read_text())
                if (result_proof.get('status') != 'COMPLETE'
                        or result_proof.get('source_identity') != source_identity
                        or int(result_proof.get('task', {}).get('seed', -1)) != seed
                        or result_proof.get('task', {}).get('key') != item.get('native_task_key')
                        or result_proof.get('data_identity') != item.get('data_identity')
                        or (item.get('native_identity') is not None
                            and result_proof.get('identity') != item['native_identity'])):
                    raise ValueError('Historical baseline receipt/source/data/seed proof differs')
            if item.get('plan'):
                input_artifact(plan['root'], item['plan'])
        if not isinstance(item, dict) or 'path' not in item:
            # Pure-JSON configuration/provenance is itself SHA-bound by registry.
            result[key] = item
            continue
        path = input_artifact(plan['root'], item)
        proof_key = {'spatial_basis': 'basis_provenance',
                     'class_reference': 'class_reference_provenance'}.get(key)
        if key.startswith('r9_reference_') and not 'provenance' in key:
            proof_key = 'r9_reference_provenance_' + key.rsplit('_', 1)[1]
        proof = entries.get(proof_key, {}) if proof_key else {}
        if proof.get('result'):
            proof_result = json.loads(input_artifact(plan['root'], proof['result']).read_text())
            if proof_result.get('selected_checkpoint', {}).get('sha256') != item['sha256']:
                raise ValueError('Historical artifact differs from its declared native selected checkpoint')
        if path.suffix == '.json':
            value = json.loads(path.read_text())
            if item.get('json_key'):
                value = value[item['json_key']]
        elif path.suffix == '.npy':
            value = np.load(path, allow_pickle=False)
        elif path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as saved:
                array_key = item.get('array_key', key)
                if key.startswith('r9_reference_'):
                    value = {k: saved[k].copy() for k in ('mean', 'whitener', 'lengthscale')}
                    value['lengthscale'] = float(value['lengthscale'])
                    value['provenance'] = json.loads(str(saved['provenance'].item()))
                    value['diagnostics'] = json.loads(str(saved['diagnostics'].item()))
                else:
                    value = saved[array_key].copy()
        elif path.suffix in ('.pt', '.pth'):
            import torch
            value = torch.load(path, map_location='cpu', weights_only=True)
        else:
            raise ValueError('Historical artifacts require JSON/NPY/NPZ or Torch weights_only payloads')
        if (isinstance(value, dict) and 'identity' in value and proof.get('native_identity') is not None
                and value['identity'] != proof['native_identity']):
            raise ValueError('Historical checkpoint state and receipt identities differ')
        for field in item.get('field', []):
            value = value[field]
        if key in ('spatial_basis', 'class_reference'):
            import torch
            value = torch.as_tensor(value, dtype=torch.float64, device=device)
        if key.startswith('r9_reference_'):
            import torch
            value['mean'] = torch.as_tensor(value['mean'], dtype=torch.float64)
            value['whitener'] = torch.as_tensor(value['whitener'], dtype=torch.float64)
        result[key] = value
    return result


def _references(plan, task):
    dependency = task.get('reference_dependency')
    if dependency is None:
        dependencies = task.get('requires', [])
        dependency = dependencies[0] if dependencies else None
    if not dependency:
        raise ValueError('Real R10 fitting requires a frozen meta_train feature-reference task')
    reference_task = next((x for x in plan['tasks'] if x['key'] == dependency), None)
    if reference_task is None or reference_task.get('family') != 'feature_reference':
        raise ValueError('Unknown or non-reference R10 dependency')
    if int(reference_task['seed']) != int(task['seed']):
        raise ValueError('Feature reference and source model seeds differ')
    result = verify_result(plan, reference_task, sealed=True)
    if not result:
        raise ValueError('Feature reference dependency lacks verified complete receipt')
    base = Path(plan['output']) / dependency
    metadata = read(base / 'reference.json', {})
    descriptor = metadata['arrays']
    path = verify_descriptor(base, descriptor)
    references = {}
    with np.load(path, allow_pickle=False) as arrays:
        for variant in metadata['references']:
            references[variant] = {'mean': arrays[variant + '_mean'].copy(),
                'whitener': arrays[variant + '_whitener'].copy(),
                'lengthscale': float(arrays[variant + '_lengthscale']),
                'provenance': metadata['references'][variant]['provenance']}
    return references, metadata


def _write_result(plan, task, folder, artifacts, paper_artifacts, **fields):
    artifacts, paper_artifacts = list(artifacts), list(paper_artifacts)
    for name in ('runtime.json', 'import-runtime.json'):
        path = folder / name
        if path.is_file() and path not in artifacts:
            artifacts.append(path); paper_artifacts.append(path)
    result = {'schema_version': 'paper3-r10-scientific-result-v1', 'status': 'COMPLETE',
              'scientific_scope': 'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
              'recipe_sha256': plan['recipe_sha256'], 'task_key': task['key'],
              'task_identity': task_identity(plan, task), 'completed_utc': utc(),
              'artifacts': [describe(p, folder) for p in artifacts],
              'paper_artifacts': [str(p.relative_to(folder)) for p in paper_artifacts], **fields}
    atomic(folder / 'result.json', result)
    progress(folder, task, status='COMPLETE', stage='complete', **fields.get('progress', {}))
    return result


def feature_reference_task(plan, task, folder, source, geometry, catalog, source_identity, registry_sha, registry=None):
    if task.get('role') != 'meta_train' or catalog.role != 'meta_train':
        raise ValueError('Feature reference is restricted to meta_train support prefixes')
    strategy = plan.get('reference_strategy')
    if strategy == 'reuse_r9_ea':
        import torch
        entry = (registry or {}).get('profile_reference_by_seed', {}).get(str(task['seed']))
        if not entry:
            raise ValueError('Reuse strategy requires registered same-seed R9 EA reference artifacts')
        entries = {'ea': entry.get('ea', entry)}
        raw = entry.get('raw') or (registry or {}).get('profile_raw_reference_by_seed', {}).get(str(task['seed']))
        if raw:
            entries['raw'] = raw
        references = {}; reuse_audit = {}
        for variant, artifacts in entries.items():
            checkpoint = input_artifact(plan['root'], artifacts['checkpoint'])
            result_path = input_artifact(plan['root'], artifacts['result'])
            result = json.loads(result_path.read_text())
            if (result.get('status') != 'COMPLETE' or result.get('task', {}).get('family') != 'feature_reference'
                    or int(result.get('task', {}).get('seed', -1)) != int(task['seed'])
                    or result.get('source_identity') != source_identity
                    or result.get('data_identity') != catalog.identity
                    or result.get('holdout_used_for_selection') is not False
                    or result.get('selected_checkpoint', {}).get('sha256') != artifacts['checkpoint']['sha256']):
                raise ValueError('R9 reused reference receipt/source/data identity mismatch')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            reference = saved['reference']; proof = reference['provenance']
            if (saved['identity'] != result.get('identity')
                    or proof.get('role') != 'meta_train_prefix20_only'
                    or proof.get('preprocessing') != variant
                    or proof.get('data_identity') != catalog.identity
                    or proof.get('future_queries_read') is not False):
                raise ValueError('R9 reference must be meta_train-prefix-only with correct preprocessing')
            references[variant] = {'mean': _tensor_numpy(reference['mean']),
                                  'whitener': _tensor_numpy(reference['whitener']),
                                  'lengthscale': float(reference['lengthscale']),
                                  'provenance': {'role': 'meta_train', 'original_provenance': proof,
                                      'source_identity': source_identity, 'catalog_identity': catalog.identity,
                                      'registry_sha256': registry_sha, 'seed': int(task['seed']),
                                      'future_queries_read': False, 'query_labels_used': False,
                                      'reference_strategy': strategy, 'preprocessing': variant,
                                      'reused_checkpoint_sha256': artifacts['checkpoint']['sha256']}}
            reuse_audit[variant] = artifacts
        arrays = {name + '_' + key: reference[key] for name, reference in references.items()
                  for key in ('mean', 'whitener', 'lengthscale')}
        atomic_npz(folder / 'reference.npz', **arrays)
        metadata = {'schema_version': 'paper3-r10-feature-reference-v1', 'seed': int(task['seed']),
                    'role': 'meta_train', 'source_identity': source_identity, 'catalog_identity': catalog.identity,
                    'reference_strategy': strategy, 'reused_artifacts': reuse_audit,
                    'arrays': describe(folder / 'reference.npz', folder),
                    'references': {name: {'provenance': value['provenance']} for name, value in references.items()}}
        atomic(folder / 'reference.json', metadata)
        return _write_result(plan, task, folder, [folder / 'reference.npz', folder / 'reference.json'],
                             [folder / 'reference.json'], source_identity=source_identity,
                             catalog_identity=catalog.identity, registry_sha256=registry_sha)
    if strategy != 'build_meta_train_prefix':
        raise ValueError('Explicit reference_strategy build_meta_train_prefix or reuse_r9_ea required')
    sessions = selected_sessions(catalog, task.get('session_subset') or plan.get('reference_subset'))
    people = {s.participant_id for s in sessions}
    counts = {p: sum(s.participant_id == p for s in sessions) for p in people}
    features = {'raw': [], 'ea': []}; weights = []; prefix_records = []
    for index, session in enumerate(sessions):
        _check_stop(folder, plan['output'])
        progress(folder, task, status='BUILDING_REFERENCE', stage='meta_train_prefix20',
                 session=index, sessions_total=len(sessions), participant=session.participant_id)
        access = SessionAccess(catalog, session, str(next(source.parameters()).device))
        support = access.support(20)  # No query_inputs/evaluation_labels call in this path.
        if not len(support.event_uids):
            raise ValueError('Registered reference session has no valid acquired support')
        ea, _ = native('paper3r7.baseline').fit_reference(source, geometry, support, 'support_EA')
        features['raw'].append(feature_batch(source, geometry, support.x)[0])
        features['ea'].append(feature_batch(source, geometry, support.x, ea)[0])
        weights.extend([1. / (len(people) * counts[session.participant_id] * len(support.event_uids))]
                       * len(support.event_uids))
        prefix_records.append({'participant': session.participant_id, 'session': session.session_id,
                               'acquired_event_uids': list(support.acquired_event_uids),
                               'valid_event_uids': list(support.event_uids)})
    proof = {'source_identity': source_identity, 'catalog_identity': catalog.identity,
             'registry_sha256': registry_sha, 'seed': int(task['seed']),
             'prefix_records_sha256': digest(prefix_records), 'n_participants': len(people),
             'n_sessions': len(sessions), 'normalization': 'frozen_joint_trial_rms_no_query_statistics'}
    references = {name: weighted_reference(np.concatenate(values), weights,
                                           dict(proof, anchor='support_EA' if name == 'ea' else 'Frozen'))
                  for name, values in features.items()}
    arrays = {name + '_' + key: reference[key] for name, reference in references.items()
              for key in ('mean', 'whitener', 'lengthscale')}
    atomic_npz(folder / 'reference.npz', **arrays)
    metadata = {'schema_version': 'paper3-r10-feature-reference-v1', 'seed': int(task['seed']),
                'role': 'meta_train', 'source_identity': source_identity, 'catalog_identity': catalog.identity,
                'arrays': describe(folder / 'reference.npz', folder),
                'references': {name: {'provenance': value['provenance']} for name, value in references.items()},
                'prefix_records': prefix_records}
    atomic(folder / 'reference.json', metadata)
    return _write_result(plan, task, folder, [folder / 'reference.npz', folder / 'reference.json'],
                         [folder / 'reference.json'], source_identity=source_identity,
                         catalog_identity=catalog.identity, registry_sha256=registry_sha,
                         progress={'sessions_total': len(sessions), 'session': len(sessions)})


def _reusable_session(folder, stem, identity):
    receipt = read(folder / (stem + '.complete.json'), {})
    if receipt.get('status') != 'COMPLETE' or receipt.get('identity') != identity:
        return None
    try:
        for item in receipt['artifacts']:
            verify_descriptor(folder, item)
        return read(folder / receipt['metrics_path'])
    except (OSError, ValueError, KeyError):
        return None


def evaluate_task(plan, task, folder, source, geometry, catalog, source_identity, registry_sha, device, registry=None):
    method = task['method']; profile = method.startswith('profile')
    config = dict(task.get('method_config', task.get('config', {})))
    config.pop('cohort', None)
    reference, reference_metadata = (None, None)
    if profile:
        references, reference_metadata = _references(plan, task)
        variant = 'raw' if method == 'profile_frozen_anchor' else 'ea'
        if variant not in references:
            raise ValueError('Raw-anchor ablation requires an explicitly registered raw meta_train reference')
        reference = references[variant]
        if reference_metadata['source_identity'] != source_identity:
            raise ValueError('Feature reference/source checkpoint identity differs')
        if method == 'profile_map':
            config['mode'] = 'map'
        if method == 'profile_no_completion':
            config['use_null_completion'] = False
        algorithm_config = algorithm.Config(**config).validate()
        if algorithm_config.probability_floor != PROBABILITY_FLOOR:
            raise ValueError('All R10 methods must share the prespecified1e-12 probability floor')
    else:
        baseline_reference = _artifact_references(plan, registry=registry or {}, method=method,
                                                  seed=int(task['seed']), device=device,
                                                  source_identity=source_identity)
        registered_config = baseline_reference.get('basis_provenance', {}).get('native_evaluation_config')
        if registered_config:
            if not isinstance(registered_config, dict):
                raise ValueError('Historical baseline configuration must be a fixed dictionary')
            config = {**registered_config, **config}
    subset = task.get('session_subset') or plan.get('session_subset')
    sessions = selected_sessions(catalog, subset)
    people_order = sorted({s.participant_id for s in sessions},
                          key=lambda p: digest([(subset or {}).get(
                              'salt', 'r10-mechanism-v1'), 'participant', p]))
    if plan['stage'] == 'mechanism' and len(people_order) != 8:
        raise ValueError('Mechanism evaluation requires exactly8 metadata-fixed participants for4/4 selection/check')
    budgets = task.get('budgets', plan.get('budgets', [4, 8, 12, 16, 20]))
    if not budgets or any(b not in (4, 8, 12, 16, 20) for b in budgets):
        raise ValueError('Only registered acquired-position budgets4/8/12/16/20 allowed')
    rows = []; artifacts = []; session_count = len(sessions) * len(budgets)
    for session_index, session in enumerate(sessions):
        access = SessionAccess(catalog, session, device)
        query = None
        for budget in budgets:
            _check_stop(folder, plan['output'])
            identity = digest({'task': task_identity(plan, task), 'session': access.identity,
                               'budget': budget, 'source': source_identity, 'registry': registry_sha})
            stem = 'sessions/' + digest([session.participant_id, session.session_id, budget])[:24]
            existing = _reusable_session(folder, stem, identity)
            if existing:
                rows.append(existing)
                receipt = read(folder / (stem + '.complete.json'))
                artifacts.extend(folder / item['path'] for item in receipt['artifacts'])
                artifacts.append(folder / (stem + '.complete.json'))
                continue
            progress(folder, task, status='PREDICTING', stage='support_fit', budget=budget,
                     session=len(rows), sessions_total=session_count,
                     participant=session.participant_id, session_id=session.session_id,
                     outer_update=0, total_updates=1)
            prediction_path = folder / (stem + '.predictions.npz')
            prediction_meta = read(folder / (stem + '.prediction.json'), {})
            fitstate_path = folder / (stem + '.fit.npz')
            # Recover a sealed prediction after a metric-writing interruption.
            prediction_reusable = prediction_meta.get('identity') == identity
            try:
                if prediction_reusable:
                    verify_descriptor(folder, prediction_meta['prediction'])
                    if prediction_meta.get('fit_state'):
                        verify_descriptor(folder, prediction_meta['fit_state'])
            except (OSError, ValueError, KeyError):
                prediction_reusable = False
            if not prediction_reusable:
                started = time.perf_counter(); support = access.support(budget)
                if query is None:
                    query = access.query_inputs()
                ea_state, ea_diagnostics = native('paper3r7.baseline').fit_reference(
                    source, geometry, support, 'support_EA')
                cache_root = Path(plan['output']) / '_label_free_features'
                cache_identity = {'source': source_identity, 'data': catalog.identity, 'session': access.identity,
                                  'recipe_sha256': plan['recipe_sha256']}
                raw_query_features, raw_query_logits = cached_feature_batch(cache_root, source, geometry,
                    query.x, query.event_uids, dict(cache_identity, view='fixed_future_query', preprocessing='raw'))
                ea_query_features, ea_query_logits = cached_feature_batch(cache_root, source, geometry,
                    query.x, query.event_uids, dict(cache_identity, view='fixed_future_query', preprocessing='ea',
                    acquired_event_uids=list(support.acquired_event_uids)), ea_state)
                frozen_probability = _bounded_binary_probability(
                    algorithm.anchor_probabilities(raw_query_logits, PROBABILITY_FLOOR))
                ea_probability = _bounded_binary_probability(
                    algorithm.anchor_probabilities(ea_query_logits, PROBABILITY_FLOOR))
                fit_descriptor = None
                if profile:
                    use_raw = method == 'profile_frozen_anchor'
                    sx, sz = cached_feature_batch(cache_root, source, geometry, support.x,
                        support.event_uids, dict(cache_identity, view='acquired_support',
                        preprocessing='raw' if use_raw else 'ea',
                        acquired_event_uids=list(support.acquired_event_uids)), None if use_raw else ea_state)
                    fit_started = time.perf_counter()
                    state = algorithm.fit(sx, sz, _tensor_numpy(support.y), algorithm_config,
                                          reference=reference, support_ids=support.event_uids)
                    fit_seconds = time.perf_counter() - fit_started
                    algorithm.save_state(state, fitstate_path)
                    fitted = algorithm.load_state(fitstate_path, expected_fit_identity=state.fit_identity)
                    if method in ('profile_quadratic', 'profile_variance_gate'):
                        from .ablations import predict_ablation
                        probability, diagnostics = predict_ablation(
                            fitted, raw_query_features if use_raw else ea_query_features,
                            raw_query_logits if use_raw else ea_query_logits,
                            method=method, query_ids=query.event_uids)
                    else:
                        probability, diagnostics = algorithm.predict(
                            fitted, raw_query_features if use_raw else ea_query_features,
                            raw_query_logits if use_raw else ea_query_logits, query_ids=query.event_uids)
                    logs = np.log(np.maximum(probability, np.finfo(np.float64).tiny))
                    fit_descriptor = describe(fitstate_path, folder)
                    fit_diagnostics = {'fit_identity': state.fit_identity, 'fit_status': state.status,
                                       'fit_seconds': fit_seconds, 'query_diagnostics': jsonable(diagnostics),
                                       'config': asdict(algorithm_config)}
                    prediction_seconds = float(diagnostics.get('summary', {}).get('prediction_seconds', 0.))
                    reasons = diagnostics.get('summary', {}).get('reason_counts', {})
                    certificate_violations = int(reasons.get('invalid_joint_kernel_certificate', 0)
                                                 + reasons.get('inconsistent_dual_bound', 0))
                    numerical_fallbacks = sum(int(count) for reason, count in reasons.items()
                        if any(word in reason for word in ('numerical', 'geometry', 'overflow', 'projection_failure',
                                                           'factorization', 'invalid_joint', 'inconsistent_dual')))
                else:
                    from .baselines import fit_baseline
                    fitted = fit_baseline(method, source, geometry, support,
                                          reference=baseline_reference, config=config)
                    _save_baseline_state(fitstate_path, fitted)
                    fit_descriptor = describe(fitstate_path, folder)
                    prediction_started = time.perf_counter()
                    if method in ('support_ea', 'support_EA', 'ea'):
                        # Reuse the same true support-EA encoder outputs used
                        # by the R10 anchor; a different CUDA convolution batch
                        # shape cannot become an apparent algorithm gain.
                        probability, logs = ea_probability, _log_softmax(ea_query_logits)
                    elif method in ('frozen', 'source_frozen'):
                        probability, logs = frozen_probability, _log_softmax(raw_query_logits)
                    else:
                        probability, logs = _baseline_prediction(fitted, query.x)
                    prediction_seconds = time.perf_counter() - prediction_started
                    fit_diagnostics = jsonable(fitted.diagnostics)
                    fit_seconds = float(fitted.diagnostics.get('calibration_seconds', 0.))
                    certificate_violations = 0
                    numerical_fallbacks = int(bool(fitted.diagnostics.get('numerical_fallback', False)))
                # This file deliberately contains no evaluation labels.
                native_returned_logs = logs.copy()
                probability = _bounded_binary_probability(probability)
                logs = np.log(probability)
                atomic_npz(prediction_path, probabilities=probability, log_probabilities=logs,
                           native_returned_log_probabilities=native_returned_logs,
                           ea_probabilities=ea_probability, frozen_probabilities=frozen_probability,
                           event_uids=np.asarray(query.event_uids), run_ids=np.asarray(query.run_ids),
                           session_identity=np.asarray(access.identity), task_identity=np.asarray(identity))
                prediction_meta = {'identity': identity, 'prediction': describe(prediction_path, folder),
                                   'fit_state': fit_descriptor, 'fit_diagnostics': fit_diagnostics,
                                   'elapsed_seconds': time.perf_counter() - started,
                                   'acquired_event_uids': list(support.acquired_event_uids),
                                   'valid_support_event_uids': list(support.event_uids),
                                   'n_valid_support': len(support.event_uids), 'ea_fit': jsonable(ea_diagnostics)}
                prediction_meta.update(incremental_prediction_seconds=prediction_seconds,
                                       fit_seconds=fit_seconds,
                                       certificate_violation_count=certificate_violations,
                                       numerical_fallback_count=numerical_fallbacks)
                atomic(folder / (stem + '.prediction.json'), prediction_meta)
            progress(folder, task, status='EVALUATING', stage='sealed_prediction_evaluation',
                     budget=budget, session=len(rows), sessions_total=session_count, outer_update=1, total_updates=1)
            labels = access.evaluation_labels(prediction_path, prediction_meta['prediction']['sha256'])
            with np.load(prediction_path, allow_pickle=False) as saved:
                metric = session_metrics(labels, saved['probabilities'], log_probabilities=saved['log_probabilities'],
                    anchors={'ea': saved['ea_probabilities'], 'frozen': saved['frozen_probabilities']},
                    event_uids=tuple(saved['event_uids'].tolist()))
                ea_metric = session_metrics(labels, saved['ea_probabilities'])
                frozen_metric = session_metrics(labels, saved['frozen_probabilities'])
                metric.update({f'anchor_ea_{key}': ea_metric[key] for key in ('ba', 'balanced_nll', 'balanced_brier')})
                metric.update({f'anchor_frozen_{key}': frozen_metric[key] for key in ('ba', 'balanced_nll', 'balanced_brier')})
                metric['updated_query_fraction'] = float(np.mean(np.abs(
                    saved['probabilities'][:, 1] - saved['ea_probabilities'][:, 1]) >= 1e-4))
            row = {'participant_id': session.participant_id, 'session_id': str(session.session_id),
                   'seed': int(task['seed']), 'budget': budget, 'method': method,
                   'n_valid_support': prediction_meta['n_valid_support'],
                   'elapsed_seconds': prediction_meta['elapsed_seconds'], **metric}
            row.update(query_signature=digest(list(access.query_ids)),
                       source_identity_sha256=digest(source_identity), data_identity_sha256=digest(catalog.identity),
                       role=task.get('role'), cohort=task.get('cohort', task.get('role')),
                       selection_split=('selection' if people_order.index(session.participant_id) < 4 else 'check')
                                       if plan['stage'] == 'mechanism' else 'formal',
                       incremental_prediction_seconds=prediction_meta.get('incremental_prediction_seconds', 0.),
                       fit_seconds=prediction_meta.get('fit_seconds', 0.),
                       certificate_violation_count=prediction_meta.get('certificate_violation_count', 0),
                       numerical_fallback_count=prediction_meta.get('numerical_fallback_count', 0))
            metric_path = folder / (stem + '.metrics.json'); atomic(metric_path, row)
            evaluation_path = folder / (stem + '.evaluation.npz')
            atomic_npz(evaluation_path, event_uids=np.asarray(access.query_ids), labels=labels,
                       prediction_sha256=np.asarray(prediction_meta['prediction']['sha256']))
            saved_paths = [prediction_path, folder / (stem + '.prediction.json'), metric_path, evaluation_path]
            if prediction_meta.get('fit_state'):
                saved_paths.append(folder / prediction_meta['fit_state']['path'])
            receipt_path = folder / (stem + '.complete.json')
            atomic(receipt_path, {'status': 'COMPLETE', 'identity': identity,
                                  'metrics_path': str(metric_path.relative_to(folder)),
                                  'artifacts': [describe(path, folder) for path in saved_paths]})
            artifacts.extend(saved_paths + [receipt_path]); rows.append(row)
    _csv(folder / 'session_metrics.csv', rows)
    participants = participant_summaries(rows); _csv(folder / 'participant_metrics.csv', participants)
    summary = {'task_key': task['key'], 'method': method, 'seed': task['seed'],
               'scientific_scope': 'SOFTWARE_TEST_ONLY' if plan.get('synthetic_fixture') else 'REAL_DATA',
               'sessions': len(sessions), 'participants': len({x['participant_id'] for x in rows}),
               'budgets': budgets, 'query_policy': 'same_original_B20_future_set_at_all_budgets',
               'scope': 'exploratory' if task.get('cohort', task.get('role')) != 'confirmation' else 'independent_confirmation',
               'order_scope': getattr(catalog, 'order_scope', catalog.identity.get('order_scope')),
               'reference_dependency': task.get('reference_dependency'),
               'source_identity': source_identity, 'catalog_identity': catalog.identity,
               'registry_sha256': registry_sha, 'completed_session_budget_cells': len(rows)}
    summary.update(probability_floor=PROBABILITY_FLOOR,
                   nll_policy='balanced_log_of_common_bounded_binary_probability_all_methods',
                   native_returned_log_probabilities_archived_for_audit=True)
    atomic(folder / 'summary.json', summary)
    papers = [folder / 'session_metrics.csv', folder / 'participant_metrics.csv', folder / 'summary.json']
    return _write_result(plan, task, folder, artifacts + papers, papers, **summary,
                         progress={'sessions_total': session_count, 'session': session_count})


def runtime_observations(source, device):
    import torch
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, 'uuid', None)
    return {'schema_version': 'paper3-r10-actual-runtime-v1', 'observed_utc': utc(),
            'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'requested_device': str(device), 'device_name': properties.name,
            'device_uuid': None if uuid is None else str(uuid),
            'device_uuid_observed': uuid is not None,
            'device_total_memory_bytes': int(properties.total_memory),
            'device_compute_capability': [properties.major, properties.minor],
            'source_parameter_dtypes': sorted({str(p.dtype) for p in source.parameters()}),
            'scientific_kernel_dtype': 'numpy.float64_cpu', 'encoder_batch_size': 64,
            'probability_floor_all_methods': PROBABILITY_FLOOR,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
            'autocast': False, 'hardware_identity_is_observation_not_scientific_source_identity': True}


def validate_gpu_environment(device):
    """Quick declared-school compatibility gate; never compare prior device IDs/drivers."""
    import torch
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('School R10 science requires the validated Python3.12 environment')
    if torch.__version__ != '2.5.1+cu121' or torch.version.cuda != '12.1':
        raise RuntimeError('School R10 science requires validated Torch2.5.1+cu121/CUDA12.1')
    if np.__version__ not in ('2.2.6', '2.3.5'):
        raise RuntimeError('School R10 science requires validated NumPy2.2.6 or2.3.5')
    properties = torch.cuda.get_device_properties(device)
    if ('A100' not in properties.name or 'MIG' in properties.name.upper()
            or properties.total_memory < 35 * 1024 ** 3
            or (properties.major, properties.minor) != (8, 0)):
        raise RuntimeError('R10 requires a full allocated A10040/80GB; partial or other GPUs need separate validation')
    return properties


def run_task(plan, task, device):
    if device == 'cpu' or str(device).startswith('cpu:'):
        raise ValueError('Real-data R10 workers require an explicitly allocated CUDA device; CPU fixtures use tests')
    folder = Path(plan['output']) / task['key']; folder.mkdir(parents=True, exist_ok=True)
    with locked(folder / '.scientific.lock'):
        existing = verify_result(plan, task, sealed=False)
        if existing:
            emit('R10_SCIENTIFIC_RESULT_REUSED', task_key=task['key'])
            return existing
        _check_stop(folder, plan['output'])
        progress(folder, task, status='STARTING', stage='identity_validation')
        registry, registry_sha = load_registry(plan)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Allocated CUDA device is unavailable; no scientific CPU fallback')
        torch.cuda.set_device(device)
        validate_gpu_environment(device)
        torch.set_num_threads(max(1, min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '16')))))
        torch.manual_seed(int(task['seed'])); torch.cuda.manual_seed_all(int(task['seed']))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision('highest')
        cohort = task.get('cohort', task.get('config', {}).get('cohort', task.get('role')))
        if task.get('family') == 'feature_reference':
            cohort = 'meta_train'
        if plan['stage'] == 'mechanism' and cohort not in ('meta_train', 'meta_validation'):
            raise ValueError('Mechanism stage cannot open target holdout or external cohorts')
        source, geometry, source_identity = load_source(plan, registry, int(task['seed']), device)
        atomic(folder / 'runtime.json', runtime_observations(source, device))
        _check_stop(folder, plan['output'])
        catalog = open_catalog(plan, registry, cohort)
        if source_identity['signal_contract_sha256'] != catalog.signal_sha:
            raise ValueError('Frozen source and registered data signal contracts differ')
        if task.get('family') == 'feature_reference':
            return feature_reference_task(plan, task, folder, source, geometry, catalog, source_identity, registry_sha, registry)
        if task.get('family') not in ('evaluate', 'mechanism'):
            raise ValueError('Unknown R10 scientific task family')
        return evaluate_task(plan, task, folder, source, geometry, catalog, source_identity, registry_sha, device, registry)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--task-key', '--task', dest='task_key', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    _, plan = load_plan(args.plan)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _STOP_REQUESTED.set())
    if os.environ.get('PAPER3_R10_STOP_REQUESTED') == '1':
        _STOP_REQUESTED.set()
    task = next((x for x in plan['tasks'] if x['key'] == args.task_key), None)
    if task is None:
        parser.error('Task key absent from frozen plan')
    folder = Path(plan['output']) / task['key']
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(20):
            with _PROGRESS_LOCK:
                current = read(folder / 'progress.json', {})
                if current and current.get('attempt_token') == os.environ.get('PAPER3_R10_ATTEMPT_TOKEN'):
                    current.update(updated_utc=utc(), heartbeat_only=True,
                                   checkpoint_stop_requested=_STOP_REQUESTED.is_set())
                    atomic(folder / 'progress.json', current)
    thread = threading.Thread(target=heartbeat, name='r10-scientific-heartbeat', daemon=True)
    thread.start()
    try:
        run_task(plan, task, args.device)
        return 0
    except PauseRequested as error:
        progress(folder, task, status='PAUSED_CHECKPOINT_SAVED', stage='paused', reason=str(error))
        return 75
    except Exception as error:
        folder.mkdir(parents=True, exist_ok=True)
        atomic(folder / 'scientific_error.json', {'status': 'FAILED', 'utc': utc(),
            'task_key': task['key'], 'attempt_token': os.environ.get('PAPER3_R10_ATTEMPT_TOKEN'),
            'error_type': type(error).__name__, 'reason': str(error), 'traceback': traceback.format_exc()})
        progress(folder, task, status='FAILED', stage='failed', error_type=type(error).__name__)
        traceback.print_exc()
        return 1
    finally:
        finished.set(); thread.join(timeout=1)


if __name__ == '__main__':
    raise SystemExit(main())
