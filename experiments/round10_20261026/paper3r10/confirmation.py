"""Standard-library-only, pre-submission independent-cohort metadata audit.

Reads no EEG samples and imports neither NumPy nor Torch. Dataset history is
an explicit reviewer attestation; chronology and exact registry joins are
checked mechanically. EEG array bytes are SHA-checked later by the worker.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''): value.update(part)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _descriptor(base, item, *, hash_bytes=True):
    if (not isinstance(item, dict) or not re.fullmatch('[0-9a-f]{64}', str(item.get('sha256', '')))
            or type(item.get('bytes')) is not int or item['bytes'] < 0):
        raise ValueError('Independent input requires path, bytes and SHA256')
    base = Path(base).resolve(); relative = Path(item['path'])
    path = relative if relative.is_absolute() else base / relative
    if not relative.is_absolute() and not path.resolve().is_relative_to(base):
        raise ValueError('Independent relative input escaped manifest directory')
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise ValueError('Independent artifact is absent or its byte count differs')
    if hash_bytes and _sha(path) != item['sha256']:
        raise ValueError('Independent metadata SHA256 differs')
    return path


def _read(path):
    with Path(path).open('rb') as handle: data = handle.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024: raise ValueError('Independent metadata exceeds16MiB')
    return json.loads(data)


def _npy_header(path):
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY': raise ValueError('Independent EEG cache must use safe NPY')
        version = tuple(stream.read(2))
        if version == (1, 0): size = struct.unpack('<H', stream.read(2))[0]
        elif version in ((2, 0), (3, 0)): size = struct.unpack('<I', stream.read(4))[0]
        else: raise ValueError('Unknown independent NPY version')
        if size > 65536: raise ValueError('Independent NPY header exceeds64KiB')
        header = ast.literal_eval(stream.read(size).decode('utf-8' if version == (3, 0) else 'latin1'))
        payload_start = stream.tell()
    shape, dtype = header.get('shape'), header.get('descr')
    if (not isinstance(shape, tuple) or len(shape) != 3 or shape[1:] != (11, 128)
            or any(type(x) is not int or x < 1 for x in shape)
            or dtype not in ('<f4', '>f4', '<f8', '>f8', '=f4', '=f8')):
        raise ValueError('Independent EEG cache must contain pre-RMS Nx11x128 float32/64 signals')
    if payload_start + math.prod(shape) * int(dtype[-1]) != Path(path).stat().st_size:
        raise ValueError('Independent NPY header/payload byte count differs')
    return shape


def _episode_audit(rows):
    module_name = '_paper3r10_confirmation_native_splits'
    if module_name not in sys.modules:
        path = Path(__file__).resolve().parent / 'vendor/native/p3work/splits.py'
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].build_prefix_episode(rows, acquisition_order_verified=True, budget=20)


def validate_confirmation_metadata(plan, registry):
    root = Path(plan['root'])
    descriptor = plan.get('confirmation_manifest', registry.get('independent_confirmation'))
    manifest_path = _descriptor(root, descriptor); manifest = _read(manifest_path)
    required = {'schema_version': 'paper3-r10-independent-cache-v1', 'status': 'COMPLETE',
                'cohort_kind': 'independent_confirmation', 'acquisition_order_verified': True,
                'physical_run_id_available': True, 'previously_used_for_algorithm_design': False}
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Independent confirmation requires complete cache, new-data and chronology evidence')
    contract = _read(root / 'contracts/input_contract.json')
    if manifest.get('signal_contract_sha256') != _digest(contract):
        raise ValueError('Independent signal contract differs from frozen source')
    people = manifest.get('participants')
    previous = set(plan.get('previously_used_participants', [])) | set(registry.get('previously_used_participants', []))
    if (not isinstance(people, list) or not people or any(not isinstance(x, str) or not x for x in people)
            or len(set(people)) != len(people) or not previous or previous.intersection(people)):
        raise ValueError('Independent participant registry missing, duplicated, previously used or prior history absent')
    review_path = _descriptor(manifest_path.parent, manifest.get('independence_review'))
    review = _read(review_path)
    if (review.get('status') != 'REVIEWED' or review.get('prior_design_history_reviewed') is not True
            or review.get('previously_used_for_algorithm_design') is not False
            or not review.get('reviewer') or not review.get('reviewed_utc')
            or not manifest.get('dataset_identity')
            or review.get('dataset_identity') != manifest['dataset_identity']
            or set(review.get('participants', [])) != set(people)):
        raise ValueError('Independent confirmation requires an exact dataset/participant history review')
    eligible, excluded, seen, cache_rows_seen = [], [], set(), set()
    for item in manifest.get('sessions', []):
        key = (item.get('participant_id'), str(item.get('session_id')))
        if key in seen or key[0] not in people or item.get('acquisition_order_verified') is not True:
            raise ValueError('Independent session registration/chronology/uniqueness differs')
        seen.add(key)
        rows = _read(_descriptor(manifest_path.parent, item['events']))
        if (not isinstance(rows, list) or not rows or any(
                not row.get('physical_run_id') or row.get('event_uid') != row.get('trial_uid')
                or row.get('participant_id') != key[0] or str(row.get('session_id')) != key[1] for row in rows)):
            raise ValueError('Independent events require exact participant/session/UID and physical runs')
        shape = _npy_header(_descriptor(manifest_path.parent, item['X'], hash_bytes=False))
        if any(type(row.get('cache_index')) is not int or not 0 <= row['cache_index'] < shape[0]
               for row in rows if row.get('valid') is True):
            raise ValueError('Independent event cache indices exceed registered array shape')
        valid_indices = [row['cache_index'] for row in rows if row.get('valid') is True]
        storage_keys = {(item['X']['sha256'], index) for index in valid_indices}
        if len(storage_keys) != len(valid_indices) or storage_keys.intersection(cache_rows_seen):
            raise ValueError('Independent support/query events alias the same cached EEG row')
        cache_rows_seen.update(storage_keys)
        audit = _episode_audit(rows)
        target = eligible if audit['query_eligible'] and len(audit['acquired_prefix_uids']) == 20 else excluded
        target.append({'participant_id': key[0], 'session_id': key[1], 'query_count': len(audit['query_uids'])})
    if {x['participant_id'] for x in eligible} != set(people):
        raise ValueError('Each independent participant requires eligible complete future physical runs')
    return {'status': 'METADATA_VERIFIED', 'manifest_sha256': _sha(manifest_path),
            'independence_review_sha256': _sha(review_path), 'participants': people,
            'eligible_sessions': eligible, 'excluded_sessions': excluded,
            'reads_eeg_samples': False, 'imports_torch': False,
            'array_sha_verification': 'worker_on_first_open_before_scientific_fit'}
