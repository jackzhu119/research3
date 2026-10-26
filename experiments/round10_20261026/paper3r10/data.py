"""Read frozen R1–R9 inputs through separate calibration/prediction/evaluation capabilities.

No catalog materialize() call is permitted here: that API returns query labels.
The scientific fitter receives SupportView only. Future labels can be opened
only after a matching prediction NPZ has been atomically persisted and hashed.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from .platform.common import atomic, digest, sha
from .platform.common import locked, read, verify_descriptor

NATIVE = Path(__file__).resolve().parent / 'vendor/native'


def native(name):
    path = str(NATIVE)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)


def input_artifact(root, item):
    """Declared read-only inputs may be explicit absolute external-cache paths."""
    if (not isinstance(item, dict) or type(item.get('bytes')) is not int
            or item['bytes'] < 0 or len(str(item.get('sha256', ''))) != 64):
        raise ValueError('Input artifact requires path, bytes and SHA256')
    root = Path(root).resolve()
    value = Path(item['path'])
    path = value if value.is_absolute() else root / value
    if not value.is_absolute() and not path.resolve().is_relative_to(root):
        raise ValueError('Relative input artifact escaped study root')
    if (not path.is_file() or path.stat().st_size != item['bytes']
            or sha(path) != item['sha256']):
        raise ValueError('Input artifact byte/SHA identity changed: ' + str(path))
    return path


def load_registry(plan):
    descriptor = plan.get('study_registry', plan.get('input_registry'))
    path = input_artifact(plan['root'], descriptor)
    registry = json.loads(path.read_text())
    if (registry.get('schema_version') != 'paper3-r10-study-registry-v1'
            or Path(registry.get('root', '')).resolve() != Path(plan['root']).resolve()):
        raise ValueError('R10 study registry/root identity mismatch')
    return registry, descriptor['sha256']


def load_source(plan, registry, seed, device):
    """Validate explicit same-seed receipts before invoking the frozen loader."""
    root = Path(plan['root'])
    entry = registry['source_by_seed'][str(seed)]
    expected = {'source_receipt': root / f'runs/source_training_v2/f0_receipt_seed-{seed}.json',
                'reference_receipt': root / f'runs/training_campaign_v1/references/seed-{seed}/receipt.json'}
    for key, path in expected.items():
        if input_artifact(root, entry[key]).resolve() != path.resolve():
            raise ValueError('Native loader must consume the registered same-seed receipt')
    source, geometry, identity = native('paper3r7.data').load_source(root, seed, device)
    source.eval()
    return source, geometry, identity


def open_catalog(plan, registry, cohort):
    root = Path(plan['root'])
    role = {'holdout': 'internal_holdout'}.get(cohort, cohort)
    if role in ('meta_train', 'meta_validation', 'internal_holdout'):
        catalog = native('paper3r7.data').Catalog(root, role, cache_gb=0)
        registered = registry.get('internal_data_identity')
        if not isinstance(registered, dict) or registered != catalog.identity:
            raise ValueError('Frozen internal cache/split/signal identity differs from study registry')
        catalog.order_scope = 'verified_physical_acquisition_order'
    elif cohort in ('pan2023', 'weibo2014'):
        path = input_artifact(root, registry['external_manifests'][cohort])
        cls = (native('paper3r7.pan2023').PanCatalog if cohort == 'pan2023'
               else native('paper3r7.external').ExternalCatalog)
        catalog = cls(root, path)
    elif cohort == 'confirmation':
        descriptor = plan.get('confirmation_manifest', registry.get('independent_confirmation'))
        manifest = input_artifact(root, descriptor)
        proof = json.loads(manifest.read_text())
        catalog = IndependentCatalog(plan, registry, manifest, proof)
    else:
        raise ValueError('Unknown registered cohort: ' + str(cohort))
    if not catalog.plans:
        raise ValueError('No eligible fixed-B20 future-query sessions')
    return catalog


class IndependentCatalog:
    """Explicit preprocessed new cohort; no downloads, format guessing or tuning.

    The review record attests experiment history, which software cannot infer
    from an EEG array. Chronology and participant disjointness are checked here.
    """
    role = 'independent_confirmation'
    order_scope = 'verified_physical_acquisition_order_independent_registry'

    def __init__(self, plan, registry, manifest_path, manifest):
        from .confirmation import validate_confirmation_metadata
        self.preparation_audit = validate_confirmation_metadata(plan, registry)
        required = {'schema_version': 'paper3-r10-independent-cache-v1', 'status': 'COMPLETE',
                    'cohort_kind': 'independent_confirmation',
                    'acquisition_order_verified': True, 'physical_run_id_available': True,
                    'previously_used_for_algorithm_design': False}
        if any(manifest.get(key) != value for key, value in required.items()):
            raise ValueError('Independent cache lacks new-data, chronology or complete-cache evidence')
        contract = json.loads((Path(plan['root']) / 'contracts/input_contract.json').read_text())
        self.signal_sha = digest(contract)
        if manifest.get('signal_contract_sha256') != self.signal_sha:
            raise ValueError('Independent preprocessing signal contract differs from frozen EEGNet')
        self.root = Path(manifest_path).parent
        review_path = input_artifact(self.root, manifest.get('independence_review'))
        review = json.loads(review_path.read_text())
        if (review.get('status') != 'REVIEWED' or review.get('prior_design_history_reviewed') is not True
                or review.get('previously_used_for_algorithm_design') is not False
                or not review.get('reviewer') or not review.get('reviewed_utc')):
            raise ValueError('Independent confirmation requires an explicit signed-history review record')
        self.people = manifest.get('participants')
        previous = set(plan.get('previously_used_participants', [])) | set(registry.get('previously_used_participants', []))
        if not previous:
            raise ValueError('Previous participant registry is required before independent confirmation')
        if (not isinstance(self.people, list) or not self.people or len(set(self.people)) != len(self.people)
                or previous.intersection(self.people)):
            raise ValueError('Independent participants missing, duplicated or previously used')
        if (review.get('dataset_identity') != manifest.get('dataset_identity')
                or set(review.get('participants', [])) != set(self.people)
                or not manifest.get('dataset_identity')):
            raise ValueError('Independence review must bind the exact dataset and participant set')
        self.identity = {'manifest_sha256': sha(manifest_path),
                         'dataset_identity_sha256': digest(manifest['dataset_identity']),
                         'signal_contract_sha256': self.signal_sha,
                         'independence_review_sha256': sha(review_path)}
        self.plans, self.excluded = [], []
        for item in manifest.get('sessions', []):
            if item.get('acquisition_order_verified') is not True or item.get('participant_id') not in self.people:
                raise ValueError('Independent session chronology/participant registration differs')
            events_path = input_artifact(self.root, item['events'])
            rows = json.loads(events_path.read_text())
            if (not isinstance(rows, list) or not rows
                    or any(not row.get('physical_run_id') or row.get('event_uid') != row.get('trial_uid')
                           or row.get('participant_id') != item['participant_id']
                           or str(row.get('session_id')) != str(item['session_id']) for row in rows)):
                raise ValueError('Independent events require physical run identity per acquired position')
            audit = native('p3work.splits').build_prefix_episode(rows, acquisition_order_verified=True, budget=20)
            if not audit['query_eligible'] or len(audit['acquired_prefix_uids']) != 20:
                self.excluded.append({'participant_id': item['participant_id'], 'session_id': item['session_id'],
                                      'reason': 'original_B20_future_query_ineligible'})
                continue
            mapped = {row['event_uid']: row for row in rows}
            if len(mapped) != len(rows):
                raise ValueError('Independent event UIDs must be unique')
            self.plans.append(SimpleNamespace(participant_id=item['participant_id'], session_id=str(item['session_id']),
                              array_descriptor=item['X'], rows=mapped, audit=audit, role=self.role))
        if {s.participant_id for s in self.plans} != set(self.people):
            raise ValueError('Each registered independent participant needs eligible complete future runs')
        keys = [(s.participant_id, s.session_id) for s in self.plans]
        if len(set(keys)) != len(keys):
            raise ValueError('Independent sessions are duplicated')
        self.plans.sort(key=lambda s: (s.participant_id, s.session_id))
        self.by_person = {p: [s for s in self.plans if s.participant_id == p] for p in self.people}
        self._verified_arrays = {}

    def array(self, session):
        key = (session.participant_id, session.session_id)
        if key not in self._verified_arrays:
            path = input_artifact(self.root, session.array_descriptor)
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.ndim != 3 or values.shape[1:] != (11, 128) or values.dtype.kind != 'f':
                raise ValueError('Independent signals must be pre-RMS Nx11x128 floating frozen-contract EEG')
            if any(type(row['cache_index']) is not int or row['cache_index'] < 0
                   or row['cache_index'] >= len(values) for row in session.rows.values() if row['valid']):
                raise ValueError('Independent valid event cache indices exceed the declared EEG array')
            self._verified_arrays[key] = values
        return self._verified_arrays[key]


def selected_sessions(catalog, subset=None):
    """Metadata hash only; never inspect labels, predictions or signal values."""
    plans = list(catalog.plans)
    if not subset:
        return plans
    salt = str(subset.get('salt', 'paper3-r10-mechanism-metadata-v1'))
    limit_people = int(subset.get('max_participants', 4))
    limit_sessions = int(subset.get('max_sessions', 16))
    if limit_people < 1 or limit_sessions < 1:
        raise ValueError('Session subset limits must be positive')
    people = sorted({p.participant_id for p in plans},
                    key=lambda p: digest([salt, 'participant', p]))[:limit_people]
    per_person = {p: sorted([s for s in plans if s.participant_id == p],
                           key=lambda s: digest([salt, p, s.session_id]))[
                           :int(subset.get('max_sessions_per_participant', len(plans)))] for p in people}
    # Round robin gives each selected participant a session before a second.
    selected = []
    while any(per_person.values()) and len(selected) < limit_sessions:
        for person in people:
            if per_person[person] and len(selected) < limit_sessions:
                selected.append(per_person[person].pop(0))
    return selected


@dataclass(frozen=True)
class SupportView:
    x: object
    y: object
    event_uids: tuple
    acquired_event_uids: tuple
    signal_sha: str
    role: str


@dataclass(frozen=True)
class QueryInputs:
    x: object
    event_uids: tuple
    run_ids: tuple
    identity: str


class SessionAccess:
    """Only this evaluator-owned object can resolve future labels."""
    def __init__(self, catalog, session, device):
        self._catalog, self._session, self.device = catalog, session, device
        self.query_ids = tuple(session.audit['query_uids'])
        acquired = tuple(session.audit['acquired_prefix_uids'])
        if len(acquired) != 20 or len(set(acquired)) != 20:
            raise ValueError('Exactly20 distinct original acquired prefix positions required')
        if (not self.query_ids or len(set(self.query_ids)) != len(self.query_ids)
                or set(acquired) & set(self.query_ids)):
            raise ValueError('Fixed future query overlaps acquired support')
        valid_uids = [u for u in acquired + self.query_ids if session.rows[u]['valid']]
        indices = [session.rows[u]['cache_index'] for u in valid_uids]
        if len(set(indices)) != len(indices):
            raise ValueError('Distinct support/query UIDs cannot alias the same cached EEG row')
        self.identity = digest({'catalog': catalog.identity, 'participant': session.participant_id,
                                'session': session.session_id, 'query_uids': self.query_ids,
                                'acquired_prefix_uids': acquired})

    def _x(self, ids):
        rows = self._session.rows
        values = self._catalog.array(self._session)
        array = np.array(values[[rows[u]['cache_index'] for u in ids]], dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite cached EEG')
        import torch
        return torch.from_numpy(array).to(self.device)

    def support(self, budget):
        if budget not in (4, 8, 12, 16, 20):
            raise ValueError('Budget is an original acquired-position count')
        rows = self._session.rows
        acquired = tuple(self._session.audit['acquired_prefix_uids'][:budget])
        valid = tuple(u for u in acquired if rows[u]['valid'])
        import torch
        labels = torch.tensor([rows[u]['label'] for u in valid], dtype=torch.long, device=self.device)
        role = {'meta_train': 'meta_train_support', 'meta_validation': 'meta_validation_support',
                'internal_holdout': 'calibration_prefix'}.get(self._catalog.role, 'calibration_prefix')
        return SupportView(self._x(valid), labels, valid, acquired, self._catalog.signal_sha, role)

    def query_inputs(self):
        rows = self._session.rows
        return QueryInputs(self._x(self.query_ids), self.query_ids,
                           tuple(str(rows[u].get('physical_run_id', 'unknown')) for u in self.query_ids),
                           self.identity)

    def evaluation_labels(self, prediction_path, prediction_sha256):
        """Read future labels only after a matching immutable prediction seal."""
        path = Path(prediction_path)
        if not path.is_file() or sha(path) != prediction_sha256:
            raise ValueError('Future labels require a persisted prediction SHA256 seal')
        with np.load(path, allow_pickle=False) as saved:
            if (tuple(saved['event_uids'].tolist()) != self.query_ids
                    or str(saved['session_identity'].item()) != self.identity
                    or saved['probabilities'].shape != (len(self.query_ids), 2)
                    or not np.isfinite(saved['probabilities']).all()):
                raise ValueError('Persisted prediction/query identity mismatch')
        return np.asarray([self._session.rows[u]['label'] for u in self.query_ids], dtype=np.int64)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def describe(path, base):
    path = Path(path)
    return {'path': str(path.relative_to(base)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def feature_batch(source, geometry, x, ea_state=None, batch_size=64):
    import torch
    normalize = native('p3work.spatial_geometry').joint_trial_rms
    features, logits = [], []
    with torch.no_grad():
        for offset in range(0, len(x), batch_size):
            chunk = x[offset:offset + batch_size]
            if ea_state is not None and not ea_state.get('fallback_reason'):
                chunk = ea_state['transform'].to(chunk.device) @ chunk.double()
            f, z = source.features_and_logits(normalize(chunk))
            features.append(f.detach().double().cpu().numpy())
            logits.append(z.detach().double().cpu().numpy())
    if not features:
        return np.empty((0, source.feature_dim)), np.empty((0, 2))
    return np.concatenate(features), np.concatenate(logits)


def cached_feature_batch(cache_root, source, geometry, x, event_uids, identity, ea_state=None):
    """Label-free cross-task cache, sealed by source/data/UID/transform identity."""
    transform = None
    if ea_state is not None and not ea_state.get('fallback_reason'):
        value = ea_state['transform'].detach().double().cpu().numpy()
        import hashlib
        transform = hashlib.sha256(value.tobytes(order='C')).hexdigest()
    proof = {'schema_version': 'paper3-r10-label-free-features-v1', 'input': identity,
             'event_uids': list(event_uids), 'transform_sha256': transform,
             'normalization': 'joint_trial_rms_frozen_encoder_eval_no_batch_updates'}
    key = digest(proof); folder = Path(cache_root) / key[:2]; folder.mkdir(parents=True, exist_ok=True)
    path, receipt_path = folder / (key + '.npz'), folder / (key + '.json')
    with locked(folder / (key + '.lock')):
        receipt = read(receipt_path, {})
        try:
            if receipt.get('status') != 'COMPLETE' or receipt.get('identity') != proof:
                raise ValueError('Feature-cache identity is absent')
            verified = verify_descriptor(folder, receipt['artifact'])
            with np.load(verified, allow_pickle=False) as saved:
                if tuple(saved['event_uids'].tolist()) != tuple(event_uids):
                    raise ValueError('Feature-cache UIDs differ')
                features, logits = saved['features'].copy(), saved['logits'].copy()
                if (features.shape != (len(x), source.feature_dim) or logits.shape != (len(x), 2)
                        or not np.isfinite(features).all() or not np.isfinite(logits).all()):
                    raise ValueError('Feature-cache shape/numerics differ')
                return features, logits
        except (ValueError, KeyError, OSError):
            pass
        features, logits = feature_batch(source, geometry, x, ea_state)
        if (features.shape != (len(x), source.feature_dim) or logits.shape != (len(x), 2)
                or not np.isfinite(features).all() or not np.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite/invalid frozen EEGNet features; cache not published')
        atomic_npz(path, features=features, logits=logits, event_uids=np.asarray(event_uids))
        atomic(receipt_path, {'status': 'COMPLETE', 'identity': proof, 'artifact': describe(path, folder),
                              'labels_in_cache': False, 'query_statistics_fitted': False})
        return features, logits


def weighted_reference(features, weights, provenance, shrinkage=.1):
    """Participant-equal support-only whitening; tiny metadata algebra on CPU."""
    values, w = np.asarray(features, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or w.shape != (len(values),) or not np.isfinite(values).all():
        raise ValueError('Finite meta-training prefix features required')
    if np.any(w <= 0) or not 0 < shrinkage <= 1:
        raise ValueError('Positive reference weights and shrinkage required')
    w = w / w.sum(); mean = w @ values
    center = values - mean; covariance = (center * w[:, None]).T @ center
    scale = max(float(np.trace(covariance) / values.shape[1]), 1e-12)
    covariance = (1 - shrinkage) * covariance + shrinkage * scale * np.eye(values.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    whitener = (eigenvectors / np.sqrt(np.maximum(eigenvalues, scale * 1e-10))) @ eigenvectors.T
    z = center @ whitener
    subset = z[np.linspace(0, len(z) - 1, min(256, len(z)), dtype=int)]
    indices = np.linspace(0, len(z) - 1, min(256, len(z)), dtype=int)
    distances = ((subset[:, None] - subset[None, :]) ** 2).sum(-1) / values.shape[1]
    left, right = np.triu_indices(len(subset), 1)
    distance, pair_weight = distances[left, right], w[indices[left]] * w[indices[right]]
    positive = distance > 1e-12
    distance, pair_weight = distance[positive], pair_weight[positive]
    if len(distance):
        order = np.argsort(distance, kind='stable')
        cumulative = np.cumsum(pair_weight[order])
        position = min(int(np.searchsorted(cumulative, cumulative[-1] * .5)), len(order) - 1)
        lengthscale = float(np.sqrt(distance[order[position]]))
    else:
        lengthscale = 1.
    return {'mean': mean, 'whitener': whitener, 'lengthscale': lengthscale,
            'provenance': dict(provenance, role='meta_train', future_queries_read=False,
                               query_labels_used=False, covariance_shrinkage=shrinkage,
                               feature_count=len(values), participant_equal_weighting=True)}
