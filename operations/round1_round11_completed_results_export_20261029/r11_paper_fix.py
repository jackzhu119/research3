"""Paper-process-only lossless binary transport repair for immutable R11 installs.

This module never changes scientific files, the original package, result seals,
plan identities, or the independent full-archive channel. Binary paper artifacts
are transported as explicitly named RFC4648 base64 ASCII files, with a complete
original/representation map and a strict decode/SHA proof.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import tempfile
import types
import urllib.parse

REPAIR_ID = 'paper3-r11-paper-binary-base64-sidecar-v1'
SCOPE = 'text raw; binary after lossless base64 decode'
BINARY_SUFFIXES = frozenset(('.npz', '.npy', '.pt', '.pth', '.bin', '.gz', '.zip',
                             '.tar', '.safetensors', '.png', '.jpg', '.jpeg', '.pdf'))
RESTORE = ('Concatenate the representation source-manifest fragments in number order without '
           'separators. Decode the resulting ASCII bytes with RFC4648 base64 and strict '
           'validation. Verify original_bytes and original_sha256 before writing the '
           'declared original_source_path under a chosen restore directory.')
VALIDATION_CODES = frozenset((
    'source_utf8_invalid', 'source_utf8_incomplete', 'source_changed_during_copy',
    'private_owned_staging_required', 'invalid_source_relative_path',
    'private_source_path_refused', 'directory_symlink_or_non_directory',
    'staging_symlink_refused', 'source_not_regular', 'invalid_chunk_bytes',
    'staged_manifest_invalid', 'staged_part_name_invalid', 'staged_part_identity_invalid',
    'staged_part_not_regular', 'staged_part_size_changed', 'staged_part_changed_during_read',
    'staged_part_SHA256_changed', 'staged_manifest_source_changed',
    'staged_part_offsets_invalid', 'staged_manifest_SHA256_changed',
    'staged_source_SHA256_changed', 'complete_remote_GET_verifier_required',
    'staged_files_required', 'snapshot_metadata_object_required', 'duplicate_source_path',
    'invalid_round_number', 'published_index_path_mismatch',
    'paper_artifact_not_sealed', 'paper_artifact_size_changed',
    'binary_base64_original_changed', 'binary_base64_decode_failed',
    'binary_base64_padding_before_end', 'binary_base64_original_sha_changed',
    'binary_base64_representation_sha_changed', 'paper_reserved_representation_path',
))
_FIXED_MESSAGES = {
    'Paper artifact SHA differs from scientific result': 'paper_artifact_sha_mismatch',
    'Cached completed-task receipt differs from scientific source': 'cached_receipt_changed',
    'Captured paper source differs from its staged fragments': 'captured_fragment_sha_mismatch',
    'Scientific source changed while capturing paper evidence': 'paper_source_changed_during_capture',
}
_ERROR_TYPES = frozenset(('ValidationError', 'ValueError', 'RuntimeError', 'OSError',
                         'FileNotFoundError', 'PermissionError', 'ApiError',
                         'ConnectionError', 'TimeoutError'))


class PaperRepairValidationError(ValueError):
    """Carries only one explicitly allowlisted fixed code."""


def diagnostic_code(error):
    value = str(error)
    if value in VALIDATION_CODES:
        return value
    return _FIXED_MESSAGES.get(value, 'paper_error_code_not_allowlisted')


def _require(condition, code):
    if not condition:
        raise PaperRepairValidationError(code)


def _binary(name):
    return Path(name).suffix.lower() in BINARY_SUFFIXES


def _raw_binary_capture(record):
    return any(_binary(item.get('source_path', '')) for item in record.get('files', ()))


def _scope_metadata(metadata, representations):
    result = dict(metadata)
    result.update(paper_sidecar_repair_id=REPAIR_ID, exact_original_bytes_scope=SCOPE,
                  original_scientific_files_modified=False,
                  original_scientific_sha256_verification_preserved=True,
                  binary_representations=list(representations),
                  transport_representations_changed=bool(representations))
    if representations:
        # The scientific originals are intact, but their transmitted files are
        # explicitly named encodings. Do not leave the legacy raw-byte flag true.
        result['declared_paper_artifacts_exact_unsanitized_bytes'] = False
    return result


def strict_decode_proof(path, original_bytes, original_sha256):
    """Verify strict RFC4648 decoding by streaming, including final padding."""
    digest = hashlib.sha256(); count = 0; padded = False
    with Path(path).open('rb') as source:
        while True:
            block = source.read(4 * 1024 * 1024)
            if not block:
                break
            _require(not padded, 'binary_base64_padding_before_end')
            try:
                decoded = base64.b64decode(block, validate=True)
            except (ValueError, base64.binascii.Error):
                raise PaperRepairValidationError('binary_base64_decode_failed') from None
            padded = b'=' in block
            digest.update(decoded); count += len(decoded)
    _require(count == original_bytes and digest.hexdigest() == original_sha256,
             'binary_base64_original_sha_changed')
    return dict(decoded_bytes=count, decoded_sha256=digest.hexdigest(), strict_base64_validated=True)


def _encode_binary(u, source, target, proof):
    """Only the owned transport copy is encoded; the scientific source is read-only."""
    source, target = Path(source), Path(target)
    before = u.signature(source); target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', dir=target.parent)
    original = hashlib.sha256(); encoded = hashlib.sha256(); count = encoded_count = 0
    try:
        with source.open('rb') as incoming, os.fdopen(descriptor, 'wb') as outgoing:
            # Three-byte blocks keep padding exclusively at the end of the file.
            for block in iter(lambda: incoming.read(3 * 1024 * 1024), b''):
                data = base64.b64encode(block)
                outgoing.write(data); original.update(block); encoded.update(data)
                count += len(block); encoded_count += len(data)
            outgoing.flush(); os.fsync(outgoing.fileno())
        _require(before == u.signature(source), 'binary_base64_original_changed')
        _require(count == proof['bytes'] and original.hexdigest() == proof['sha256'],
                 'binary_base64_original_sha_changed')
        decoded = strict_decode_proof(temporary, count, original.hexdigest())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(original_bytes=count, original_sha256=original.hexdigest(),
                encoded_bytes=encoded_count, encoded_sha256=encoded.hexdigest(),
                encoding='base64-rfc4648-ascii-no-newlines', reconstruction=RESTORE, **decoded)


def _capture_task(u, plan, key, row, stage, work):
    stage, work = Path(stage), Path(work)
    identity = u._capture_identity(row); relative = Path('per-task') / key / identity
    capture = stage / relative; folder = Path(plan['output']) / key
    saved = u.read(capture / 'capture.json', {}, max_bytes=u._STATE_BYTES)
    if (saved.get('identity') == identity and saved.get('recipe_sha256') == plan['recipe_sha256']
            and saved.get('task_key') == key and not _raw_binary_capture(saved)):
        return saved
    if (u.sha(folder / 'result.json') != row['result_sha256']
            or u.sha(folder / 'completion.json') != row['completion_sha256']):
        raise ValueError('Cached completed-task receipt differs from scientific source')
    result = u.read(folder / 'result.json', {})
    declared = result.get('paper_artifacts', [])
    sealed = {item['path']: item for item in result.get('artifacts', [])}
    sources = [('result.json', row['result_sha256'], None),
               ('completion.json', row['completion_sha256'], None)]
    for item in declared:
        name = item['path'] if isinstance(item, dict) else item
        u.safe_file(folder, folder / name)
        _require(not name.startswith('binary-base64/'), 'paper_reserved_representation_path')
        descriptor = sealed.get(name)
        _require(descriptor is not None, 'paper_artifact_not_sealed')
        sources.append((name, descriptor['sha256'], descriptor['bytes']))
    files = []; captured = []; paths = []; representations = []
    for name, expected, expected_bytes in sources:
        u.safe_file(folder, folder / name)
        proof = u._copy_stable(folder / name, capture / name, expected_sha256=expected)
        _require(expected_bytes is None or proof['bytes'] == expected_bytes, 'paper_artifact_size_changed')
        represented = name
        if _binary(name):
            represented = 'binary-base64/' + name + '.b64.txt'
            mapping = _encode_binary(u, capture / name, capture / represented, proof)
            mapping.update(original_source_path=key + '/' + name,
                           representation_source_path=key + '/' + represented,
                           original_task_identity=row['task_identity'],
                           original_result_sha256=row['result_sha256'],
                           original_completion_sha256=row['completion_sha256'])
            representations.append(mapping)
            staged = u.stage_file(capture, represented, work / 'staged-binary-base64-v1')
            _require(staged['source_bytes'] == mapping['encoded_bytes']
                     and staged['source_sha256'] == mapping['encoded_sha256'],
                     'binary_base64_representation_sha_changed')
            staged = dict(staged, binary_representation=mapping)
        else:
            staged = u.stage_file(capture, name, work / 'staged')
            if staged['source_sha256'] != proof['sha256']:
                raise ValueError('Captured paper source differs from its staged fragments')
        files.append(dict(staged, source_path=key + '/' + represented))
        paths.append((relative / represented).as_posix())
        if name not in ('result.json', 'completion.json'):
            item = dict(source_path=key + '/' + name, bytes=proof['bytes'], sha256=proof['sha256'])
            if _binary(name):
                item.update(transport_encoding='base64-rfc4648', representation_source_path=key + '/' + represented,
                            encoding_manifest_source_path=key + '/binary-base64/REPRESENTATIONS.json')
            captured.append(item)
    if representations:
        manifest_name = 'binary-base64/REPRESENTATIONS.json'
        manifest = dict(schema_version='paper3-r11-paper-binary-representations-v1',
                        repair_id=REPAIR_ID, task_key=key, recipe_sha256=plan['recipe_sha256'],
                        exact_original_bytes_scope=SCOPE, original_scientific_files_modified=False,
                        original_task_identity=row['task_identity'], representations=representations,
                        scientific_statistics_recomputed=False, reconstruction=RESTORE)
        u.atomic(capture / manifest_name, manifest)
        staged = u.stage_file(capture, manifest_name, work / 'staged-binary-base64-v1')
        files.append(dict(staged, source_path=key + '/' + manifest_name))
        paths.append((relative / manifest_name).as_posix())
    record = dict(identity=identity, recipe_sha256=plan['recipe_sha256'], task_key=key,
                  completion_sha256=row['completion_sha256'], files=files, paths=paths,
                  paper_sidecar_repair_id=REPAIR_ID, exact_original_bytes_scope=SCOPE,
                  binary_representations=representations,
                  coverage=dict(task_key=key, recipe_sha256=plan['recipe_sha256'],
                                captured_paper_artifacts=captured,
                                has_declared_paper_artifacts=bool(declared)))
    u.atomic(capture / 'capture.json', record)
    return record


def _collection(u, original, plan_path, work):
    _, plan = u.load_plan(plan_path, validate_science_stage=False)
    state = u._paper_capture_state(work, plan)
    bad = [key for key, record in state['tasks'].items() if _raw_binary_capture(record)]
    if bad:
        # Only invalid owned transport cache entries are rebuilt. Previously
        # verified text captures and their published receipts remain reusable.
        for key in bad:
            state['tasks'].pop(key)
        u.atomic(Path(work) / 'capture-index.json', state)
    stage, paths, metadata = original(plan_path, work)
    state = u._paper_capture_state(work, plan)
    active = {item['task_key'] for item in metadata['paper_task_coverage']}
    representations = [item for key in sorted(active)
                       for item in state['tasks'][key].get('binary_representations', ())]
    metadata = _scope_metadata(metadata, representations)
    u.atomic(stage / 'collection.json', dict(metadata=metadata,
             capture_identities={key:state['tasks'][key]['identity'] for key in sorted(active)}))
    return stage, paths, metadata


def _publish_scoped(u, gh, files, work, metadata, prefix, representations):
    metadata = _scope_metadata(metadata, representations)
    receipt = u.publish_snapshot(gh, 11, files, work, metadata=metadata, prefix_root=prefix)
    scoped = dict(receipt, exact_original_bytes_scope=SCOPE,
                  original_scientific_files_modified=False, transport_representations_changed=bool(representations),
                  binary_representations=list(representations), paper_sidecar_repair_id=REPAIR_ID)
    # Original publish_snapshot has already fully verified every transport blob
    # and source manifest. The final pointer adds explicit representation scope;
    # no byte proof, hash check or remote verification is disabled.
    pointer = {key:value for key,value in scoped.items()
               if key not in ('commit', 'published_utc', 'published_index_paths_verified')}
    pointer_path = prefix + '/PAPER_DATA_LATEST.json'
    content = u.encoded(pointer)
    scoped['commit'] = gh.git_publish({pointer_path:content},
                                     'Record R11 lossless binary representation scope')
    expected = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    observed = gh.api(gh.prefix + '/contents/' + urllib.parse.quote(pointer_path, safe='/')
                      + '?ref=' + urllib.parse.quote(scoped['commit'], safe=''))
    _require(observed.get('sha') == expected, 'published_index_path_mismatch')
    scoped['published_index_paths_verified'] = True
    return scoped


def _paper_cycle(u, plan_path, work, gh):
    stage, paths, metadata = u.paper_collection(plan_path, work); work = Path(work)
    _, plan = u.load_plan(plan_path, validate_science_stage=False)
    state = u._paper_capture_state(work, plan)
    saved = u.read(work / 'published-groups.json', {}, max_bytes=u._STATE_BYTES)
    if saved.get('recipe_sha256') != plan['recipe_sha256']:
        saved = dict(recipe_sha256=plan['recipe_sha256'], tasks={})
    active = {row['task_key'] for row in metadata['paper_task_coverage']}
    for key in sorted(active):
        capture = state['tasks'][key]; previous = saved['tasks'].get(key, {})
        mappings = capture.get('binary_representations', [])
        if (previous.get('identity') == capture['identity']
                and (not mappings or previous.get('receipt', {}).get('exact_original_bytes_scope') == SCOPE)):
            continue
        task_metadata = dict(stage=plan['stage'], recipe_sha256=plan['recipe_sha256'], task_key=key,
            collection_status='COMPLETE' if capture['coverage']['has_declared_paper_artifacts'] else 'PARTIAL',
            required_missing=[] if capture['coverage']['has_declared_paper_artifacts'] else [key+':paper_artifacts'],
            required_warnings=[], paper_task_coverage=[capture['coverage']], training_controlled=False)
        prefix = u.stage_prefix(plan) + '/per-task/' + key + '/' + capture['identity']
        receipt = _publish_scoped(u, gh, capture['files'], work, task_metadata, prefix, mappings)
        saved['tasks'][key] = dict(identity=capture['identity'], receipt=receipt)
        u.atomic(work / 'published-groups.json', saved)
    report_files = [record['file'] for _,record in sorted(state['reports'].items())]
    report_identity = u.digest([{name:record.get(name) for name in
        ('source_path','source_bytes','source_sha256','source_version')} for record in report_files])
    if saved.get('reports', {}).get('identity') != report_identity:
        receipt = _publish_scoped(u, gh, report_files, work, dict(stage=plan['stage'],
            recipe_sha256=plan['recipe_sha256'], collection_status=metadata['collection_status'],
            required_missing=metadata['required_missing'], required_warnings=[], training_controlled=False),
            u.stage_prefix(plan) + '/reports', [])
        saved['reports'] = dict(identity=report_identity, receipt=receipt)
        u.atomic(work / 'published-groups.json', saved)
    identity = u.digest(dict(tasks={key:saved['tasks'][key]['identity'] for key in sorted(active)},
                             reports=report_identity, metadata=metadata))
    previous = u.read(work / 'last-published.json', {})
    if previous.get('identity') == identity and previous.get('receipt'):
        return previous['receipt']
    groups = {key:saved['tasks'][key] for key in sorted(active)}
    manifest = dict(schema_version='paper3-r11-incremental-paper-manifest-v1', round=11,
        recipe_sha256=plan['recipe_sha256'], snapshot_id=identity, tasks=groups, reports=saved['reports'],
        metadata=metadata, exact_original_bytes=True, exact_original_bytes_scope=SCOPE,
        binary_representations=metadata['binary_representations'], original_scientific_files_modified=False,
        remote_complete_blob_GET_SHA256_verified=True, scientific_metrics_recomputed=False,
        paper_sidecar_repair_id=REPAIR_ID, training_controlled=False)
    root = u.stage_prefix(plan); manifest_path = root + '/paper-data/' + identity + '/manifest.json'
    receipts = [group['receipt'] for group in groups.values()] + [saved['reports']['receipt']]
    receipt = dict(schema_version='paper3-r11-incremental-paper-pointer-v1', round=11, stage=plan['stage'],
        recipe_sha256=plan['recipe_sha256'], snapshot_id=identity, manifest=manifest_path,
        status='PAPER_DATA_FILES_VERIFIED' if metadata['collection_status']=='COMPLETE' else 'PAPER_DATA_PARTIAL_FILES_VERIFIED',
        collection_status=metadata['collection_status'], complete_tasks=metadata['complete_tasks'],
        total_tasks=metadata['total_tasks'], files=sum(r.get('files',0) for r in receipts),
        bytes=sum(r.get('bytes',0) for r in receipts), incremental_immutable_task_captures=True,
        exact_original_bytes=True, exact_original_bytes_scope=SCOPE,
        binary_representations=metadata['binary_representations'], original_scientific_files_modified=False,
        transport_representations_changed=bool(metadata['binary_representations']),
        paper_sidecar_repair_id=REPAIR_ID, remote_complete_blob_GET_SHA256_verified=True,
        full_raw_results_uploaded=False, training_controlled=False)
    receipt['commit'] = gh.git_publish({manifest_path:u.encoded(manifest), root+'/PAPER_DATA_LATEST.json':u.encoded(receipt)},
                                     'Update R11 lossless paper representation index')
    u.publish_stage_index(gh, plan, 'paper', receipt)
    u.atomic(work / 'last-published.json', dict(identity=identity, receipt=receipt))
    return receipt


def install_patch(uploads_module, ready_callback=None):
    """Install only in an isolated paper sidecar process, after original plan checks."""
    u = uploads_module
    if ready_callback is not None:
        if not callable(ready_callback):
            raise TypeError('Paper ready_callback must be callable')
        u._paper_sidecar_ready_callback = ready_callback
    existing = getattr(u, '_paper_binary_sidecar_installation', None)
    if existing:
        return dict(existing)
    original_collection = u.paper_collection
    original_full = u.full_cycle
    original_monitor = u.monitor_cycle
    original_report = u.report_loop
    original_channel_loop = u.channel_loop
    def capture(plan, key, row, stage, work):
        return _capture_task(u, plan, key, row, stage, work)
    def collection(plan_path, work):
        return _collection(u, original_collection, plan_path, work)
    def cycle(gh, plan_path, work):
        try:
            return _paper_cycle(u, plan_path, work, gh)
        except Exception as error:
            kind = type(error).__name__
            diagnostic = dict(status='PAPER_SIDECAR_RETRY', repair_id=REPAIR_ID,
                fixed_code=diagnostic_code(error), error_type=kind if kind in _ERROR_TYPES else 'PaperSidecarError',
                updated_utc=u.utc(), training_controlled=False)
            u.atomic(Path(work) / 'paper-fix-diagnostic.json', diagnostic)
            u.emit('R11_PAPER_FIX_DIAGNOSTIC', **diagnostic)
            raise
    def channel_loop(plan_path, channel, **kwargs):
        if channel != 'paper':
            return original_channel_loop(plan_path, channel, **kwargs)
        _, plan = u.load_plan(plan_path, validate_science_stage=False)
        original_atomic = original_channel_loop.__globals__['atomic']
        sent = False
        def ready_atomic(path, value):
            nonlocal sent
            result = original_atomic(path, value)
            path = Path(path)
            if (not sent and path.name == 'process.json' and path.parent.name == 'upload-paper'
                    and isinstance(value, dict) and value.get('channel') == 'paper'
                    and value.get('pid') == os.getpid()):
                # This exact write is performed by the original loop only
                # after its nonblocking paper-upload.lock has been acquired.
                callback = getattr(u, '_paper_sidecar_ready_callback', None)
                if callback is not None:
                    callback(dict(status='PAPER_PROCESS_LOCK_ACQUIRED', pid=os.getpid(),
                        recipe_sha256=plan['recipe_sha256'], role='paper', lock_acquired=True,
                        patch_id=REPAIR_ID, remote_paper_upload_complete=False,
                        scientific_completion_claimed=False, training_controlled=False))
                sent = True
            return result
        namespace = dict(original_channel_loop.__globals__, atomic=ready_atomic,
                         paper_cycle=u.paper_cycle)
        isolated = types.FunctionType(original_channel_loop.__code__, namespace,
            original_channel_loop.__name__, original_channel_loop.__defaults__, original_channel_loop.__closure__)
        isolated.__kwdefaults__ = original_channel_loop.__kwdefaults__
        return isolated(plan_path, channel, **kwargs)
    u._capture_paper_task = capture
    u.paper_collection = collection
    u.paper_cycle = cycle
    u.channel_loop = channel_loop
    report = dict(repair_id=REPAIR_ID, patch_scope='paper_process_memory_only',
                  exact_original_bytes_scope=SCOPE, binary_suffixes=sorted(BINARY_SUFFIXES),
                  scientific_package_files_modified=False, original_plan_and_receipts_modified=False,
                  full_archive_channel_unchanged=(u.full_cycle is original_full),
                  monitor_channel_unchanged=(u.monitor_cycle is original_monitor),
                  reporter_unchanged=(u.report_loop is original_report))
    report['ready_callback_scope'] = 'after_original_paper_lock_and_process_receipt_before_network'
    u._paper_binary_sidecar_installation = report
    return dict(report)
