"""Frozen native EEG inputs for M12, with support-only adaptation capabilities.

Preparation is standard-library-only. Scientific imports happen inside the
persistent DataProvider, and source/catalog/dictionary verification is cached
once per worker. No original result, checkpoint, runtime or source is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path

SCHEMA = 'paper3m12-native-inputs-v1'
SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (4, 8, 12, 16, 20)
_METADATA_LIMIT = 32 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    with Path(path).open('rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        if not 0 <= size <= _METADATA_LIMIT:
            raise ValueError('Native metadata exceeds bounded size: ' + str(path))
        data = handle.read(size + 1)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Native metadata object required: ' + str(path))
    return value


def resolve(root, value, *, relative=None):
    name = value.get('path') if isinstance(value, dict) else value
    if not isinstance(name, str) or not name:
        raise ValueError('Concrete native artifact path required')
    path = Path(name).expanduser()
    return path.resolve() if path.is_absolute() else (Path(relative or root) / path).resolve()


def describe(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return {'path': path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(root, item, *, relative=None):
    path = resolve(root, item, relative=relative)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if isinstance(item, dict):
        if item.get('bytes') is not None and path.stat().st_size != item['bytes']:
            raise ValueError('Native artifact byte identity changed: ' + str(path))
        if item.get('sha256') is not None and sha(path) != item['sha256']:
            raise ValueError('Native artifact SHA256 changed: ' + str(path))
    return path


def _write_immutable(path, value):
    from .common import atomic_bytes
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError('Frozen inputs already exist with a different identity: ' + str(path))
        return
    # Use the same private staging directory and narrow transient retry as
    # durable worker records; old broad temporary-file cleanup must not turn
    # a recoverable rename acknowledgement into a preparation failure.
    atomic_bytes(path, data)


def _data_module():
    return importlib.import_module('paper3m12.vendor.paper3r11.data')


def _existing_registry(root):
    for relative in ('runs/r11_campaign/plan.json',
                     'runs/round11_preparation/plan-campaign.json',
                     'runs/round11_preparation/plan.json'):
        path = root / relative
        if not path.is_file():
            continue
        plan = read(path)
        descriptor = plan.get('study_registry', plan.get('input_registry'))
        if isinstance(descriptor, dict):
            registry_path = checked(root, descriptor)
            registry = read(registry_path)
            if registry.get('schema_version') != 'paper3-r11-study-registry-v1':
                raise ValueError('Existing registry schema differs')
            if Path(registry.get('root', '')).resolve() != root:
                raise ValueError('Existing registry root differs')
            return registry, describe(registry_path, root)
    # This native registry builder reads known plans/small metadata only.
    builder = importlib.import_module('paper3m12.vendor.paper3r11.registry')
    registry = builder.build_registry(root)
    return registry, None


def _basis_identity(plan, task, source_identity, data_identity):
    training = {'updates': task.get('updates', task.get('config', {}).get('updates',
                    task.get('config', {}).get('outer_updates', plan.get('updates', 1000)))),
                'validate_every': task.get('validate_every', plan.get('validate_every', 50))}
    return digest({'task': task, 'data': data_identity, 'source': source_identity,
                   'recipe': plan['recipe_sha256'], 'basis': None, 'training': training})


def _r7_basis(root, seed, source_identity, data_identity, config):
    """Follow a fixed R7 static-CE basis task, never rank by target performance."""
    explicit = config.get('r7_plan') or config.get('inputs', {}).get('r7_plan')
    candidates = ([resolve(root, explicit)] if explicit else [])
    base = root / 'runs/followup_20261024_r7'
    candidates += [base / 'plan.json', base / 'runtime/plan.json',
                   base / 'runtime/three_jobs/backend-plan.json']
    for launch in (base/'launch.json', base/'runtime/launch.json', base/'runtime/three_jobs/launch.json'):
        if launch.is_file():
            record = read(launch)
            for name in ('plan', 'plan_path', 'backend_plan', 'backend_plan_path'):
                if record.get(name):
                    candidates.append(resolve(root, record[name], relative=launch.parent))
    key = 'basis-static_ce-r4-seed' + str(seed)
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path); plan = read(path)
        task = next((item for item in plan.get('tasks', []) if item.get('key') == key), None)
        if task is None:
            continue
        cfg = task.get('config', {})
        if (task.get('family') != 'basis' or int(task.get('seed', -1)) != seed
                or int(cfg.get('rank', -1)) != 4 or cfg.get('mode') != 'static'
                or cfg.get('objective') != 'ce'):
            raise ValueError('Registered R7 static rank4 dictionary task differs')
        output = resolve(root, plan['output'])
        receipt_path = output/key/'result.json'; receipt = read(receipt_path)
        if (receipt.get('status') != 'COMPLETE' or receipt.get('task') != task
                or receipt.get('source_identity') != source_identity
                or receipt.get('data_identity') != data_identity
                or receipt.get('identity') != _basis_identity(plan, task, source_identity, data_identity)):
            raise ValueError('Completed R7 dictionary does not bind source/data/task/recipe')
        if receipt.get('holdout_used_for_selection') is True:
            raise ValueError('Dictionary was selected on holdout')
        locator = receipt['selected_checkpoint']
        locator_name = Path(locator['path'])
        # Original R7 writes absolute locators; repacked registries may retain
        # project-relative runs/... descriptors or native output-relative names.
        base_for_locator = root if locator_name.parts[:1] == ('runs',) else output
        checkpoint = checked(root, locator, relative=base_for_locator)
        if not checkpoint.is_relative_to((output/key).resolve()):
            raise ValueError('R7 dictionary checkpoint is outside its declared task')
        return {'checkpoint': describe(checkpoint, root), 'receipt': describe(receipt_path, root),
                'plan': describe(path, root), 'field': 'b', 'rank': 4, 'task_key': key,
                'native_identity': receipt['identity'], 'source_identity': source_identity,
                'data_identity': data_identity, 'map': 'full_generator_static_expM'}
    raise FileNotFoundError('Completed registered R7 dictionary not found: ' + key)


def freeze_inputs(root, output, config=None, seedlist=SEEDS):
    """Freeze known small school identities once; no Torch, GPU or EEG reads."""
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    config = config or {}
    target = output/'frozen-inputs.json'
    if target.is_file():
        payload = read(target)
        identity = payload.pop('input_identity', None)
        if (payload.get('schema_version') != SCHEMA or Path(payload.get('root', '')).resolve() != root
                or identity != digest(payload)
                or set(payload.get('source_by_seed', {})) != {str(int(s)) for s in seedlist}):
            raise ValueError('Existing M12 frozen inputs belong to a different study')
        payload['input_identity'] = identity
        return dict(payload, manifest_descriptor=describe(target, root))
    registry, registry_origin = _existing_registry(root)
    manifest_path = checked(root, registry['internal_cache_manifest'])
    manifest = read(manifest_path)
    if manifest.get('status') != 'STIEGER_CACHE_COMPLETE':
        raise ValueError('Existing frozen Stieger cache is not complete')
    split_path = checked(root, manifest['participant_split']); split = read(split_path)
    people = {role: list(split['partitions'][role]) for role in
              ('meta_train', 'meta_validation', 'internal_holdout')}
    if [len(people[role]) for role in people] != [40, 11, 11]:
        raise ValueError('The registered40/11/11 participant split is required')
    flat = [person for group in people.values() for person in group]
    if len(set(flat)) != len(flat):
        raise ValueError('Frozen participant partitions overlap')
    ordered = sorted(people['meta_train'], key=lambda uid:
                     hashlib.sha256(('20261027|meta_train|' + str(uid)).encode()).hexdigest())
    people.update(reference=ordered[:32], pilot=ordered[32:])
    data_identity = {'manifest_sha256': sha(manifest_path), 'split_sha256': split['split_sha256'],
                     'signal_contract_sha256': manifest['input_contract_sha256']}
    if registry.get('internal_data_identity') != data_identity:
        raise ValueError('Existing registry and frozen input catalog differ')
    sources, dictionaries = {}, {}
    for seed in tuple(int(x) for x in seedlist):
        entry = registry['source_by_seed'][str(seed)]
        source_path = checked(root, entry['source_receipt']); reference_path = checked(root, entry['reference_receipt'])
        source = read(source_path); reference = read(reference_path)
        if source.get('status') != 'SOURCE_F0_COMPLETE' or int(source.get('pipeline_seed', -1)) != seed:
            raise ValueError('Frozen same-seed source is not complete')
        if (reference.get('status') != 'SOURCE_REFERENCE_COMPLETE'
                or reference.get('source_f0_receipt_sha256') != sha(source_path)
                or reference.get('signal_contract_sha256') != source.get('signal_contract_sha256')):
            raise ValueError('Frozen source/reference receipt lineage differs')
        checkpoint = checked(root, source['checkpoint'])
        reference_array = checked(root, reference['arrays']['reference'])
        identity = {'source_receipt_sha256': sha(source_path),
                    'reference_receipt_sha256': sha(reference_path),
                    'source_checkpoint_sha256': source['checkpoint']['sha256'],
                    'signal_contract_sha256': source['signal_contract_sha256']}
        if identity['signal_contract_sha256'] != data_identity['signal_contract_sha256']:
            raise ValueError('Frozen source window differs from target cache')
        sources[str(seed)] = {'source_receipt': describe(source_path, root),
            'reference_receipt': describe(reference_path, root), 'checkpoint': describe(checkpoint, root),
            'reference_array': describe(reference_array, root), 'identity': identity}
        dictionaries[str(seed)] = _r7_basis(root, seed, identity, data_identity, config)
    registry = json.loads(json.dumps(registry))
    payload = {'schema_version': SCHEMA, 'root': str(root), 'registry': registry,
               'registry_origin': registry_origin, 'source_by_seed': sources,
               'dictionary_by_seed': dictionaries, 'people': people,
               'internal_data_identity': data_identity,
               'reference_rule': '64events: one UID-hash-minimum valid training event per class per32reference participants',
               'reference_query_overlap': False,
               'discovery_policy': {'imports_torch': False, 'reads_EEG_arrays': False,
                                    'source_files_modified': False, 'school_jobs_started': 0}}
    payload['input_identity'] = digest(payload)
    _write_immutable(target, payload)
    return dict(payload, manifest_descriptor=describe(target, root))


@dataclass(frozen=True)
class SessionRef:
    participant_id: str
    session_id: str
    role: str
    identity: str
    audit: dict


class NativeMarginModel:
    """Full static exp(M) spatial adapter with separate target/reference EA."""
    dimension = 4

    def __init__(self, source, geometry, basis, *, alignment=None, identity=None):
        import torch
        self.source, self.geometry = source, geometry
        self.basis = basis.detach()
        self.device, self.dtype = basis.device, basis.dtype
        self.feature_dim = int(source.feature_dim)
        self.identity = identity or {}
        self.cache_identity = digest(self.identity)
        self.alignment = alignment
        self._generators = (geometry.e @ self.basis).T.reshape(4, geometry.q.shape[0], geometry.q.shape[0])
        self._left = geometry.q.T @ geometry.rref_sqrt
        self._right = geometry.rref_invsqrt @ geometry.q
        if basis.shape != (geometry.e.shape[1], 4):
            raise ValueError('M12 requires the completed4D full generator dictionary')
        if not torch.allclose(basis.T @ basis, torch.eye(4, device=basis.device, dtype=basis.dtype),
                              atol=1e-8, rtol=1e-8):
            raise ValueError('Native dictionary columns are not orthonormal')

    def _features_logits(self, theta, x, *, reference=False):
        import torch
        theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)
        x = x.to(device=self.device, dtype=self.dtype)
        if theta.shape != (4,):
            raise ValueError('Exactly4 static spatial parameters required')
        if not reference and self.alignment is not None:
            x = self.alignment @ x
        m = torch.einsum('r,rij->ij', theta, self._generators)
        delta = torch.matrix_exp(m) - torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
        # On the CAR subspace this is the original full static exp(M) map.
        # Writing its change from identity explicitly preserves the exact
        # fitted EA numerical path at theta=0 without dropping its derivative.
        transformed = x + self._left @ delta @ self._right @ x
        # The frozen model preserves input gradients; parameters/BN stay frozen.
        normalizer = _data_module().native('p3work.spatial_geometry').joint_trial_rms
        return self.source.features_and_logits(normalizer(transformed))

    def features(self, theta, x):
        return self._features_logits(theta, x)[0]

    def reference_features(self, theta, x):
        return self._features_logits(theta, x, reference=True)[0]

    def _margin(self, theta, x, head_w=None, head_b=None, *, reference=False):
        import torch
        features, logits = self._features_logits(theta, x, reference=reference)
        margin = (logits[:, 1]-logits[:, 0]).to(self.dtype)
        if head_w is not None:
            value = torch.as_tensor(head_w, device=features.device, dtype=features.dtype)
            margin = margin + (features @ value).to(self.dtype)
        if head_b is not None:
            margin = margin + torch.as_tensor(head_b, device=margin.device, dtype=margin.dtype)
        return margin

    def margin(self, theta, x, head_w=None, head_b=None):
        return self._margin(theta, x, head_w, head_b)

    def reference_margin(self, theta, x, head_w=None, head_b=None):
        return self._margin(theta, x, head_w, head_b, reference=True)


def _fit_ea(geometry, x):
    """Same fixed .05, per-trial-unit-trace Support-EA as native R7 baseline."""
    import torch
    if len(x) == 0:
        return None, 'empty_support'
    with torch.no_grad():
        z = geometry.q @ x.to(dtype=geometry.q.dtype, device=geometry.q.device)
        covariance = z @ z.transpose(-2, -1)
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
        if not torch.isfinite(covariance).all() or bool((trace <= 0).any()):
            return None, 'invalid_support_covariance'
        dimension = geometry.q.shape[0]
        mean = (covariance / trace[:, None, None]).mean(0)
        regularized = .95*mean + .05*torch.eye(dimension, dtype=mean.dtype, device=mean.device)/dimension
        regularized = (regularized+regularized.T)/2
        values, vectors = torch.linalg.eigh(regularized)
        inverse = (vectors*values.rsqrt()) @ vectors.T
        return geometry.q.T @ geometry.rref_sqrt @ inverse @ geometry.q, None


def _scientific_runtime(device):
    """Configure the actual persistent worker; preparation never calls this.

    Observed hardware/thread metadata is kept outside scientific input/cache
    identities, so replacing an interrupted worker does not erase valid fits.
    This function never seeds/reset RNG state or rewrites a driver check.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    requested = torch.device(device)
    if requested.type not in ('cpu', 'cuda'):
        raise ValueError('M12 native workers require an explicit CPU or CUDA device')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(16)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Torch permits this setting only once, before inter-op work begins.
        # The real observed value is recorded rather than falsified.
        pass
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested for a scientific worker but is unavailable')
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device is outside the worker allocation')
        torch.cuda.set_device(index)
        requested = torch.device('cuda', index)
        name = torch.cuda.get_device_name(requested)
    else:
        name = 'CPU'
    profile = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), device=str(requested), device_name=name,
        torch_cpu_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cuda_matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        rng_reseeded=False, separate_gpu_probe_job=False)
    return str(requested), profile


class DataProvider:
    """A persistent, bounded cache of immutable native inputs and source models."""
    def __init__(self, plan, device='cpu'):
        self.plan = plan
        self.root = Path(plan['root']).resolve()
        value = plan.get('native_inputs', plan.get('inputs'))
        if isinstance(value, dict) and value.get('schema_version') == SCHEMA:
            self.inputs = json.loads(json.dumps(value))
        else:
            descriptor = plan.get('inputs_descriptor', plan.get('input_manifest'))
            if descriptor is None and isinstance(value, dict) and 'path' in value:
                descriptor = value
            self.inputs = read(checked(self.root, descriptor))
        if self.inputs.get('schema_version') != SCHEMA or Path(self.inputs.get('root','')).resolve() != self.root:
            raise ValueError('M12 frozen native inputs/root identity differs')
        raw = {k: v for k, v in self.inputs.items() if k not in ('input_identity', 'manifest_descriptor')}
        if self.inputs.get('input_identity') != digest(raw):
            raise ValueError('Frozen native input identity differs')
        self.registry = self.inputs['registry']
        self.registry_sha = digest(self.registry)
        self.device, self.runtime_profile = _scientific_runtime(device)
        self._module = _data_module()
        self._sources, self._basis, self._catalogs, self._native_sessions, self._accesses = {}, {}, {}, {}, {}
        self._source_signatures = {}
        self._references = {}

    @staticmethod
    def _native_role(role):
        return {'pilot': 'meta_train', 'reference': 'meta_train', 'holdout': 'internal_holdout'}.get(role, role)

    def _catalog(self, role):
        native_role = self._native_role(role)
        if native_role not in self._catalogs:
            native_plan = {'root': str(self.root), 'study_registry': {'sha256': self.registry_sha}}
            self._catalogs[native_role] = self._module.open_catalog(native_plan, self.registry, native_role,
                                                                 registry_sha=self.registry_sha)
        return self._catalogs[native_role]

    def participants(self, role):
        names = self.inputs['people'].get(role)
        if names is None:
            names = self.inputs['people'].get(self._native_role(role))
        return list(names) if names is not None else list(self._catalog(role).people)

    def sessions(self, role, participant):
        if participant not in self.participants(role):
            raise ValueError('Participant is not in the frozen role')
        catalog = self._catalog(role)
        output = []
        for item in catalog.by_person[participant]:
            access = self._module.SessionAccess(catalog, item, self.device)
            key = (role, str(participant), str(item.session_id))
            self._native_sessions[key] = item
            self._accesses[key] = access
            output.append(SessionRef(str(participant), str(item.session_id), role, access.identity,
                                     {'query_uids': tuple(item.audit['query_uids']),
                                      'acquired_prefix_uids': tuple(item.audit['acquired_prefix_uids']),
                                      'query_eligible': True}))
        return output

    def _access(self, session):
        key = (session.role, session.participant_id, str(session.session_id))
        if key not in self._accesses:
            self.sessions(session.role, session.participant_id)
        access = self._accesses[key]
        if access.identity != session.identity:
            raise ValueError('Session capability identity changed')
        return access

    def support(self, session, budget):
        return self._access(session).support(budget)

    def query_inputs(self, session):
        return self._access(session).query_inputs()

    def evaluate_labels(self, session, prediction_path, expectedidentity):
        access = self._access(session); path = Path(prediction_path)
        actual = sha(path)
        if isinstance(expectedidentity, dict):
            if expectedidentity.get('session_identity', access.identity) != access.identity:
                raise ValueError('Prediction semantic identity changed')
            expected = expectedidentity.get('sha256')
        else:
            expected = expectedidentity
        if expected not in (actual, access.identity):
            raise ValueError('Prediction seal SHA/session identity differs')
        return access.evaluation_labels(path, actual)

    def _source(self, seed):
        seed = int(seed)
        if seed not in self._sources:
            source, geometry, identity = self._module.load_source(
                {'root': str(self.root)}, self.registry, seed, self.device, registry_sha=self.registry_sha)
            if identity != self.inputs['source_by_seed'][str(seed)]['identity']:
                raise ValueError('Loaded frozen source identity differs from preparation')
            self._sources[seed] = (source, geometry, identity)
            if hasattr(self._module, 'frozen_source_signature'):
                self._source_signatures[seed] = self._module.frozen_source_signature(source, geometry)
        elif seed in self._source_signatures:
            source, geometry, _ = self._sources[seed]
            self._module.assert_source_unchanged(source, geometry, self._source_signatures[seed])
        return self._sources[seed]

    def _dictionary(self, seed):
        seed = int(seed)
        if seed not in self._basis:
            import torch
            descriptor = self.inputs['dictionary_by_seed'][str(seed)]
            path = checked(self.root, descriptor['checkpoint'])
            with torch.random.fork_rng(devices=[], enabled=True):
                state = torch.load(path, map_location='cpu', weights_only=True)
            if state.get('identity') != descriptor['native_identity']:
                raise ValueError('Selected dictionary checkpoint native identity differs')
            value = state[descriptor.get('field', 'b')].detach().to(device=self.device, dtype=torch.float64)
            _, geometry, _ = self._source(seed)
            geometry.validate_dictionary(value)
            if value.shape[1] != 4:
                raise ValueError('Selected dictionary is not the frozen rank4 family')
            self._basis[seed] = value
        return self._basis[seed]

    def model(self, seed):
        source, geometry, identity = self._source(seed)
        descriptor = self.inputs['dictionary_by_seed'][str(int(seed))]
        return NativeMarginModel(source, geometry, self._dictionary(seed), identity={
            'seed': int(seed), 'source': identity, 'native_input_identity': self.inputs['input_identity'],
            'dictionary_sha256': descriptor['checkpoint']['sha256'], 'EA': 'none'})

    def episode_model(self, seed, support):
        source, geometry, identity = self._source(seed)
        transform, fallback = _fit_ea(geometry, support.x)
        descriptor = self.inputs['dictionary_by_seed'][str(int(seed))]
        support_fingerprint = hashlib.sha256(
            support.x.detach().cpu().contiguous().numpy().tobytes()
            + support.y.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        result = NativeMarginModel(source, geometry, self._dictionary(seed), alignment=transform,
            identity={'seed': int(seed), 'source': identity, 'native_input_identity': self.inputs['input_identity'],
                      'dictionary_sha256': descriptor['checkpoint']['sha256'],
                      'EA': 'support_only_unit_trace_shrinkage_0.05',
                      'EA_support_uids': list(support.event_uids),
                      'acquired_event_uids': list(support.acquired_event_uids),
                      'support_signal_and_label_sha256': support_fingerprint,
                      'EA_fallback_reason': fallback})
        result.ea_fallback_reason = fallback
        return result

    def perturb(self, seed, x, generator_index, amplitude):
        """Predeclared known input shift; no statistics, labels or file writes."""
        import torch
        _, geometry, _ = self._source(seed)
        basis = self._dictionary(seed)
        index = int(generator_index)
        if index not in (0, 1):
            raise ValueError('Only the two predeclared shift generators are permitted')
        value = float(amplitude)
        if value not in (-.15, -.05, .05, .15):
            raise ValueError('Only the predeclared shift amplitudes/inverses are permitted')
        with torch.no_grad():
            theta = torch.zeros(4, device=basis.device, dtype=basis.dtype)
            theta[index] = value
            matrix, _, _ = geometry.matrix(theta, basis, family='exp')
            result = matrix @ x.to(device=matrix.device, dtype=matrix.dtype)
            return self._module.native('p3work.spatial_geometry').joint_trial_rms(result)

    def reference(self, seed, reference_participants=None):
        """Freeze64 meta_train reference events, never target/future-role events."""
        import numpy as np
        import torch
        seed = int(seed)
        expected = self.inputs['people']['reference']
        if reference_participants is not None and list(reference_participants) != list(expected):
            raise ValueError('Reference participants differ from the frozen32-ID split')
        if seed in self._references:
            return self._references[seed]
        catalog = self._catalog('reference')
        chosen = []
        for person in expected:
            by_class = {0: [], 1: []}
            for session in catalog.by_person[person]:
                for uid, row in session.rows.items():
                    if row.get('valid') and row.get('label') in (0, 1):
                        by_class[int(row['label'])].append((uid, session, row))
            for label in (0, 1):
                if not by_class[label]:
                    raise ValueError('Reference participant lacks one training class: ' + person)
                chosen.append(min(by_class[label], key=lambda item: hashlib.sha256(item[0].encode()).hexdigest()))
        if len(chosen) != 64 or len({item[0] for item in chosen}) != 64:
            raise ValueError('Fixed64 distinct reference events required')
        values = []
        for uid, session, row in chosen:
            array = catalog.array(session)
            values.append(np.array(array[int(row['cache_index'])], dtype=np.float64, copy=True))
        raw = torch.from_numpy(np.stack(values)).to(self.device)
        _, geometry, source_identity = self._source(seed)
        transform, fallback = _fit_ea(geometry, raw)
        if fallback is not None:
            raise ValueError('Reference EA could not be fitted from frozen C')
        aligned = transform @ raw
        uids = tuple(item[0] for item in chosen)
        identity = {'source': source_identity, 'participant_ids': list(expected),
                    'event_uids_sha256': digest(uids), 'data': catalog.identity,
                    'reference_alignment_sha256': hashlib.sha256(
                        transform.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                    'target_query_inputs_read': False, 'target_query_labels_read': False}
        result = {'x': aligned.detach(), 'event_uids': uids, 'identity': identity,
                  'reference_transform': transform.detach()}
        self._references[seed] = result
        return result
