"""Tiny artificial IO fixtures; no real data, SDK network or scientific fit."""
from __future__ import annotations

import copy
import errno
import fcntl
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from hpc import stage_data as stage


def blob(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()


def descriptor(path, body):
    return {'path': path, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body)}


def receipt(entries, profile=False):
    prefix = 'paper3/source-extension/cache/' if profile else 'paper3/frozen-baseline/'
    result = {'schema_version': 1, 'operation': 'paper3_explicit_manifest_backup',
              'execute': True, 'status': 'verified', 'verification': 'full_get_body_sha256',
              'prefix': prefix, 'objects': []}
    if profile:
        result['source_extension_profile'] = stage.source_extension_io.profile()
    for row in entries:
        item = {key: row[key] for key in stage.bootstrap.ENTRY_FIELDS}
        item.update(key=prefix + row['path'], status='verified',
            put={'status': 'created', 'http_status': 200,
                 'request_body_bytes': row['bytes'], 'request_body_sha256': row['sha256']},
            get={'status': 'verified', 'http_status': 200, 'complete_body_read': True,
                 'bytes_read': row['bytes'], 'sha256': row['sha256'],
                 'declared_content_length': row['bytes']})
        result['objects'].append(item)
    return result


class FakeR2:
    def __init__(self, bodies):
        self.bodies, self.calls, self.streams = bodies, [], []

    def get_object(self, *, Bucket, Key):
        self.calls.append(Key)
        body = io.BytesIO(self.bodies[Key])
        self.streams.append(body)
        return {'Body': body, 'ContentLength': len(self.bodies[Key]),
                'ResponseMetadata': {'HTTPStatusCode': 200}}


@pytest.fixture
def fixed(tmp_path, monkeypatch, request):
    root = tmp_path / 'project'
    root.mkdir()
    incoming = root / '.hpc/packages'
    incoming.mkdir(parents=True)
    closure = '6' * 64
    bodies = {
        'cache/source_v2/BNCI2014_001/1/X.npy': b'fixture-cache-present',
        'cache/stieger_v2/Stieger2021/1/X.npy': b'fixture-baseline-missing',
        'cache/source_extensions_20261019/Yang2025/1/X.npy': b'fixture-extension-missing',
        'contracts/portable_python_312.json': b'fixed-runtime-contract',
        'audit/frozen/deploy.json': b'fixed-deploy-manifest',
        'audit/frozen/BASE4_PLUS_YANG.json': b'fixed-Yang-source-manifest',
        'audit/frozen/BASE4_PLUS_DREYER.json': b'fixed-Dreyer-source-manifest',
        'p3work/frozen_source.py': b'original frozen code',
    }
    rows = []
    for path, body in bodies.items():
        row = descriptor(path, body)
        row.update(role='result' if path.startswith('cache/') else ('code' if path.startswith('p3work/') else 'audit'),
                   dataset='Yang2025' if 'Yang2025/' in path else ('Stieger2021' if 'Stieger2021/' in path else 'Paper3'),
                   io_profile='source_extension' if 'Yang2025/' in path else 'baseline')
        rows.append(row)
    wheels = []
    for name in stage.bootstrap.SDK_NAMES:
        row = descriptor('data/dependency_wheels/cpu/' + name + '-1.0-py3-none-any.whl', b'wheel-not-required')
        wheels.append(row.copy())
        rows.append(dict(row, role='archive', dataset='Paper3', io_profile='baseline'))
    groups, r2_bodies = [], {}
    for profile in ('baseline', 'source_extension'):
        chosen = [row for row in rows if row['io_profile'] == profile]
        proof = receipt(chosen, profile == 'source_extension')
        relative = 'audit/frozen/' + profile + '-receipt.json'
        bodies[relative] = blob(proof)
        groups.append({'io_profile': profile, 'receipt': descriptor(relative, bodies[relative]),
                       'paths': [row['path'] for row in chosen]})
        for item in proof['objects']:
            if item['path'] in bodies:
                r2_bodies[item['key']] = bodies[item['path']]
    by_path = {row['path']: row for row in rows}
    index = {'schema_version': stage.bootstrap.SCHEMA, 'closure_sha256': closure,
             'deploy_manifest': descriptor('audit/frozen/deploy.json', bodies['audit/frozen/deploy.json']),
             'files': rows, 'restore_groups': groups,
             'runtime': {'path': stage.bootstrap.RUNTIME, 'sha256': stage.bootstrap.RUNTIME_SHA,
                         'bytes': stage.bootstrap.RUNTIME_BYTES},
             'runtime_contract': descriptor('contracts/portable_python_312.json', bodies['contracts/portable_python_312.json']),
             'requirements': descriptor('requirements-gpu.lock', b'unused-existing-environment'),
             'bootstrap_wheels': wheels,
             'extension_manifests': {arm: 'audit/frozen/' + arm + '.json' for arm in stage.bootstrap.ARMS},
             'raw_payloads_selected': 0, 'external_query_access_allowed': False,
             'scientific_fits_already_completed': 0, 'user_manual_GPU_boot_required': True,
             'thread_environment': {key: '1' for key in stage.bootstrap.THREAD_VARIABLES}}
    index['index_sha256'] = stage.bootstrap.digest(index)
    index_path = 'audit/frozen/bootstrap_index.json'
    bodies[index_path] = blob(index)
    release = {'schema_version': 'paper3-hpc-fixed-data-release-v1', 'closure_sha256': closure,
               'bootstrap_index': descriptor(index_path, bodies[index_path]), 'objects': {}}
    omitted = {'cache/stieger_v2/Stieger2021/1/X.npy',
               'cache/source_extensions_20261019/Yang2025/1/X.npy'}
    if getattr(request, 'param', None) == 'two_missing_baseline_objects':
        omitted.add('cache/source_v2/BNCI2014_001/1/X.npy')
    package_bodies = {path: body for path, body in bodies.items() if path not in omitted}
    def make_package(name, selected):
        payload = {path: package_bodies[path] for path in selected}
        manifest = {'schema_version': 'paper3-fixed-deployment-package-inputs-v1',
                    'status': 'LOCAL_EXACT_PAYLOAD_LIST_NOT_AN_ARCHIVE_OR_UPLOAD',
                    'closure_sha256': closure, 'bootstrap_index': release['bootstrap_index'],
                    'top_level_directory': 'paper3_data_plan_' + closure[:16],
                    'raw_EEG_embedded': False, 'full_wheelhouse_embedded': False,
                    'files': [descriptor(path, body) for path, body in payload.items()],
                    'file_count': len(payload), 'uncompressed_bytes': sum(map(len, payload.values()))}
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            for path, body in payload.items():
                info = tarfile.TarInfo(manifest['top_level_directory'] + '/' + path)
                info.size = len(body)
                info.mode = 0o755 if Path(path).suffix in {'.py', '.sh'} else 0o644
                archive.addfile(info, io.BytesIO(body))
        for kind, path, content in [('archive', 'archives/' + name + '.tar.gz', output.getvalue()),
                                    ('inputs', 'audit/frozen/' + name + '_inputs.json', blob(manifest))]:
            row = dict(descriptor(path, content), key='paper3/fixed/' + path)
            release['objects'][name + '_' + kind] = row
            r2_bodies[row['key']] = content
            target = incoming / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    make_package('main', list(package_bodies))
    make_package('control', ['audit/frozen/BASE4_PLUS_YANG.json'])
    ready_body = b'original historical ready proof'
    ready_row = dict(descriptor('audit/frozen/READY.json', ready_body), key='paper3/fixed/READY.json')
    release['objects']['ready'] = ready_row
    r2_bodies[ready_row['key']] = ready_body
    target = incoming / ready_row['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(ready_body)
    monkeypatch.setattr(stage, 'load_release', lambda: copy.deepcopy(release))
    return {'root': root, 'incoming': incoming, 'release': release, 'bodies': bodies,
            'index': index, 'omitted': omitted, 'client': FakeR2(r2_bodies)}


def test_original_release_descriptor_is_pinned(monkeypatch):
    monkeypatch.undo()
    value = stage.load_release()
    assert value['bootstrap_index']['sha256'] == 'ef220ccb0affd9ac7ce256284de6a1c6dbf2634d945eb4f538355cb86a67b3e7'
    assert len(value['objects']) == 5
    assert '/cache/cache/' not in value['objects']['main_archive']['key']


def test_materialize_reports_precise_missing_cache_without_overwriting_code(fixed):
    path = fixed['root'] / 'p3work/frozen_source.py'
    path.parent.mkdir()
    path.write_bytes(b'current compatibility code')
    result = stage.prepare_data(fixed['root'], mode='materialize')
    assert result['status'] == 'blocked'
    assert result['missing_cache_files'] == 2
    assert {row['path'] for row in result['missing_files']} == fixed['omitted']
    assert {group['io_profile'] for group in result['restore_groups']} == {'baseline', 'source_extension'}
    assert path.read_bytes() == b'current compatibility code'
    assert not (fixed['root'] / 'data').exists()
    assert not (fixed['root'] / 'runtime').exists()
    assert fixed['client'].calls == []
    assert result['scientific_fits'] == 0 and result['external_query_accessed'] is False


def test_explicit_fetch_uses_original_profile_keys_full_GET_and_is_resumable(fixed):
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'ready'
    assert len(fixed['client'].calls) == 2
    assert any('/cache/cache/' in key for key in fixed['client'].calls)
    assert all(stream.closed for stream in fixed['client'].streams)
    assert sum(proof['fresh_full_GETs'] for proof in result['group_restore_receipts']) == 2
    for path in fixed['omitted']:
        assert (fixed['root'] / path).read_bytes() == fixed['bodies'][path]
    second = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                               client=fixed['client'], bucket='fixture-bucket')
    assert second['status'] == 'ready'
    assert second['group_restore_receipts'] == []
    assert len(fixed['client'].calls) == 2
    checked = stage.prepare_data(fixed['root'], mode='check')
    assert checked['status'] == 'ready' and checked['missing_cache_files'] == 0
    assert len(stage.load_index(fixed['root'])['files']) == len(fixed['index']['files'])


@pytest.mark.parametrize('execute,network', [(False, False), (True, False), (False, True)])
def test_fetch_requires_both_explicit_flags(fixed, execute, network):
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=execute, network_host=network,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'failed'
    assert result['error_code'] == 'fetch_requires_execute_and_network_host'
    assert fixed['client'].calls == []


def test_compute_node_cannot_accidentally_fetch(fixed, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['error_code'] == 'fetch_is_separate_from_offline_compute_job'
    assert fixed['client'].calls == []


def test_missing_index_is_blocked_and_receipt_is_truthful(fixed):
    result = stage.prepare_data(fixed['root'])
    assert result['status'] == 'blocked' and result['error_code'] == 'missing_frozen_input'
    assert result['scientific_fits'] == 0
    assert fixed['client'].calls == []
    receipt_path = fixed['root'] / result['attempt'] / 'receipt.json'
    assert json.loads(receipt_path.read_bytes()) == result


def test_corrupt_local_cache_is_not_replaced(fixed):
    stage.prepare_data(fixed['root'], mode='materialize')
    path = fixed['root'] / next(iter(fixed['omitted']))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'wrong-cache')
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'failed'
    assert result['error_code'] == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert path.read_bytes() == b'wrong-cache' and fixed['client'].calls == []
    assert len(result['release_object_GETs']) == 5


def test_changed_original_receipt_blocks_before_network(fixed):
    stage.prepare_data(fixed['root'], mode='materialize')
    path = fixed['root'] / fixed['index']['restore_groups'][0]['receipt']['path']
    path.write_bytes(b'forged receipt')
    result = stage.prepare_data(fixed['root'], mode='check')
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert fixed['client'].calls == []


def test_wrong_fresh_GET_keeps_gate_closed(fixed):
    key = next(key for key in fixed['client'].bodies if key.endswith('Stieger2021/1/X.npy'))
    fixed['client'].bodies[key] = b'wrong-stream'
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'failed'
    assert result['error_code'] == 'original_profile_restore_failed_see_receipt'
    assert not (fixed['root'] / 'cache/stieger_v2/Stieger2021/1/X.npy').exists()
    assert result['group_restore_receipts'][0]['fresh_full_GETs'] == 0


def test_only_one_data_coordinator_can_run(fixed):
    path = fixed['root'] / '.paper3-hpc/data/stage.lock'
    path.parent.mkdir(parents=True)
    with path.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = stage.prepare_data(fixed['root'])
    assert result['error_code'] == 'data_stage_already_running'
    assert fixed['client'].calls == []


def test_symlink_destination_is_refused(fixed, tmp_path):
    (fixed['root'] / 'cache').symlink_to(tmp_path, target_is_directory=True)
    result = stage.prepare_data(fixed['root'], mode='materialize')
    assert result['status'] == 'failed'
    assert 'symlink' in result['error_code'].lower()
    assert not (tmp_path / 'source_v2').exists()


def test_archive_SHA_is_checked_before_materialization(fixed):
    archive = fixed['incoming'] / fixed['release']['objects']['main_archive']['path']
    archive.write_bytes(b'corrupt archive')
    result = stage.prepare_data(fixed['root'], mode='materialize')
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert not (fixed['root'] / 'cache').exists()


@pytest.mark.parametrize('path', ['data/raw.npy', 'cache/external/X.npy', 'cache/source_v2/sealed/X.npz',
                                  'p3work/current.py', 'tools/current.py', 'tests/current.py',
                                  'runtime/python.tar.gz', 'audit/raw.edf', 'requirements-gpu.lock'])
def test_data_staging_never_grants_software_or_query_scope(path):
    assert stage._allowed(path) is False


def test_credentials_are_never_exposed(fixed):
    class Broken:
        def get_object(self, **kwargs):
            raise RuntimeError('SECRET_MUST_NOT_BE_PRINTED')
    (fixed['incoming'] / fixed['release']['objects']['main_archive']['path']).unlink()
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=Broken(), bucket='fixture-bucket')
    assert result['status'] == 'failed'
    assert 'SECRET_MUST_NOT_BE_PRINTED' not in json.dumps(result)


def test_empty_network_host_fetch_reads_all_five_fixed_objects_before_data(fixed):
    for row in fixed['release']['objects'].values():
        (fixed['incoming'] / row['path']).unlink()
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
                                client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'ready'
    assert len(fixed['client'].calls) == 7
    fixed_keys = [row['key'] for row in fixed['release']['objects'].values()]
    assert fixed['client'].calls[:5] == fixed_keys
    assert all(proof['status'] == 'R2_FULL_GET_SHA_VERIFIED'
               and proof['complete_body_read'] is True for proof in result['release_object_GETs'])


def test_noninteractive_missing_configuration_stays_blocked(fixed, monkeypatch):
    from hpc import handoff
    seen = []
    def no_configuration(*, prompt=False):
        seen.append(prompt)
        raise handoff.HandoffError('existing_r2_configuration_required_on_network_allowed_node')
    monkeypatch.setattr(handoff, 'r2_client', no_configuration)
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True)
    assert result['status'] == 'blocked'
    assert seen == [False] and result['scientific_fits'] == 0
    assert fixed['client'].calls == []


def test_SHA_parallelism_is_bounded_by_Slurm_allocation(fixed, monkeypatch):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '24')
    result = stage.prepare_data(fixed['root'], mode='materialize')
    assert result['independent_SHA_IO_workers'] == 8
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '2')
    result = stage.prepare_data(fixed['root'], mode='check')
    assert result['independent_SHA_IO_workers'] == 2


def test_materialization_index_and_SHA_progress_is_persisted_and_scope_bounded(fixed, monkeypatch):
    ticks = iter(range(0, 100000, 2))
    monkeypatch.setattr(stage.time, 'monotonic', lambda: next(ticks))
    events, observations = [], []
    def observed(event):
        receipts = list((fixed['root'] / '.paper3-hpc/data').glob('attempt-*/receipt.json'))
        current = json.loads(max(receipts, key=lambda path: path.stat().st_mtime_ns).read_bytes())
        # Evidence must already be persisted before a display callback runs.
        observations.append((copy.deepcopy(current), copy.deepcopy(event)))
        events.append(copy.deepcopy(event))
    result = stage.prepare_data(fixed['root'], mode='materialize', progress=observed)
    assert result['status'] == 'blocked' and result['missing_cache_files'] == 2
    assert observations and all(current['active_transfer'] == event and current['status'] == 'running'
                                for current, event in observations)
    phases = [event['phase'] for event in events]
    for expected in ('archive_member_scan', 'archive_header_verification',
                     'materializing_verified_archive', 'index_descriptor_verification',
                     'index_receipt_binding_verification', 'index_contract_verification',
                     'frozen_scope_selection', 'frozen_file_SHA_verification'):
        assert expected in phases
    assert phases.index('index_receipt_binding_verification') < phases.index('frozen_file_SHA_verification')
    complete_archives = [event for event in events
        if event['phase'] == 'materializing_verified_archive' and event.get('completed')]
    assert {event['object'] for event in complete_archives} == {'main', 'control'}
    assert all(event['processed_files'] == event['total_files']
        == event['verified_files'] + event['excluded_files'] for event in complete_archives)
    bindings = next(event for event in events
        if event['phase'] == 'index_receipt_binding_verification' and event['completed'])
    assert bindings['processed_groups'] == bindings['total_groups'] == 2
    assert bindings['covered_files'] == len(fixed['index']['files'])
    inspected = [event for event in events if event['phase'] == 'frozen_file_SHA_verification']
    assert inspected[0]['processed_files'] == 0 and inspected[0]['completed'] is False
    assert inspected[-1]['completed'] is True and inspected[-1]['inspection'] == 'before_restore'
    assert inspected[-1]['processed_files'] == inspected[-1]['total_files'] == result['frozen_data_files']
    assert inspected[-1]['verified_files'] == result['locally_SHA_verified']
    assert inspected[-1]['missing_files_count'] == 2
    assert inspected[-1]['verified_bytes'] + result['missing_bytes'] == inspected[-1]['total_bytes']
    assert 'fixture-bucket' not in json.dumps(events)
    assert not any(event.get('status') == 'ready' for event in events)
    assert fixed['client'].calls == []


def test_after_restore_SHA_progress_is_distinct_and_matches_real_verified_bytes(fixed):
    events = []
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket', progress=events.append)
    assert result['status'] == 'ready'
    completed = [event for event in events
        if event['phase'] == 'frozen_file_SHA_verification' and event['completed']]
    assert [event['inspection'] for event in completed] == ['before_restore', 'after_restore']
    before, after = completed
    assert before['missing_files_count'] == 2 and after['missing_files_count'] == 0
    assert before['verified_files'] + 2 == after['verified_files'] == after['total_files']
    assert after['verified_bytes'] == after['total_bytes']
    assert after['verified_bytes'] - before['verified_bytes'] == sum(
        len(fixed['bodies'][path]) for path in fixed['omitted'])


def test_failed_local_SHA_reports_current_phase_without_false_completion(fixed, monkeypatch):
    stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket')
    (fixed['root'] / 'cache/stieger_v2/Stieger2021/1/X.npy').write_bytes(b'corrupt-software-fixture')
    ticks = iter(range(0, 100000, 6))
    monkeypatch.setattr(stage.time, 'monotonic', lambda: next(ticks))
    events = []
    result = stage.prepare_data(fixed['root'], mode='check', progress=events.append)
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    inspected = [event for event in events if event['phase'] == 'frozen_file_SHA_verification']
    assert inspected and not any(event['completed'] for event in inspected)
    assert any(event['verified_files'] == 1 for event in inspected)
    assert max(event['processed_files'] for event in inspected) < inspected[0]['total_files']
    assert result['active_transfer']['phase'] == 'frozen_file_SHA_verification'
    assert result['active_transfer']['completed'] is False
    assert result['scientific_fits'] == 0


def test_corrupt_original_receipt_reports_binding_phase_and_never_finishes_it(fixed):
    stage.prepare_data(fixed['root'], mode='materialize')
    receipt_path = fixed['root'] / fixed['index']['restore_groups'][0]['receipt']['path']
    receipt_path.write_bytes(b'corrupt-original-receipt-software-fixture')
    events = []
    result = stage.prepare_data(fixed['root'], mode='check', progress=events.append)
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    bindings = [event for event in events if event['phase'] == 'index_receipt_binding_verification']
    assert bindings and bindings[-1]['processed_groups'] == 0
    assert not any(event['completed'] for event in bindings)
    assert result['active_transfer']['phase'] == 'index_receipt_binding_verification'
    assert not any(event['phase'] == 'frozen_file_SHA_verification' for event in events)
    assert fixed['client'].calls == []


def test_scope_progress_is_throttled_except_start_and_actual_completion(monkeypatch):
    ticks = iter((0.0, 1.0, 4.9, 5.0, 5.1, 10.0, 10.1))
    monkeypatch.setattr(stage.time, 'monotonic', lambda: next(ticks))
    events = []
    progress = stage._Progress(events.append, 'software_fixture_only')
    for number in range(6):
        progress.emit(processed_files=number, total_files=6)
    progress.emit(processed_files=6, total_files=6, completed=True)
    assert [event['processed_files'] for event in events] == [0, 3, 5, 6]
    assert [event['completed'] for event in events] == [False, False, False, True]


def test_progress_receipt_write_failure_remains_a_failed_data_gate(fixed, monkeypatch):
    stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket')
    original = stage._atomic_json
    def quota_at_SHA_progress(root, relative, value):
        if (value.get('status') == 'running'
                and value.get('active_transfer', {}).get('phase') == 'frozen_file_SHA_verification'):
            raise OSError(errno.EDQUOT, 'SECRET_PROGRESS_RECEIPT_FAILURE')
        return original(root, relative, value)
    monkeypatch.setattr(stage, '_atomic_json', quota_at_SHA_progress)
    result = stage.prepare_data(fixed['root'], mode='check')
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_IO_EDQUOT'
    assert result['failure_detail']['phase'] == 'frozen_file_SHA_verification'
    assert result['active_transfer']['completed'] is False
    assert 'SECRET_PROGRESS_RECEIPT_FAILURE' not in json.dumps(result)


def test_broken_progress_display_does_not_remove_full_SHA_checks(fixed):
    def disconnected_display(event):
        raise BrokenPipeError('SECRET_DISPLAY_MESSAGE')
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket', progress=disconnected_display)
    assert result['status'] == 'ready'
    assert len(fixed['client'].calls) == 2
    assert result['locally_SHA_verified'] == result['frozen_data_files']
    assert result['scientific_fits'] == 0 and 'SECRET_DISPLAY_MESSAGE' not in json.dumps(result)


@pytest.mark.parametrize('malformation,code', [
    ('duplicate', 'archive_exact_member_set_mismatch'),
    ('bad_header', 'archive_regular_frozen_header_required'),
    ('trailing', 'archive_nonzero_trailing_data'),
])
def test_archive_scan_progress_preserves_original_full_archive_gates(fixed, malformation, code):
    row = fixed['release']['objects']['main_archive']
    path = fixed['incoming'] / row['path']
    with tarfile.open(fileobj=io.BytesIO(path.read_bytes()), mode='r:gz') as archive:
        contents = [(copy.deepcopy(member), archive.extractfile(member).read()) for member in archive.getmembers()]
    if malformation == 'duplicate':
        contents.append(copy.deepcopy(contents[0]))
    elif malformation == 'bad_header':
        contents[0][0].uid = 1
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for member, body in contents:
            archive.addfile(member, io.BytesIO(body))
    payload = output.getvalue()
    if malformation == 'trailing':
        payload = gzip.compress(gzip.decompress(payload) + b'forbidden-software-fixture-trailer')
    path.write_bytes(payload)
    # The fixture's outer SHA now matches; the unchanged inner archive gates
    # must still reject duplicate members, altered headers and nonzero tails.
    row.update(descriptor(row['path'], payload))
    events = []
    result = stage.prepare_data(fixed['root'], mode='materialize', progress=events.append)
    assert result['status'] == 'failed' and result['error_code'] == code
    assert not any(event['phase'] == 'materializing_verified_archive' and event.get('completed')
                   for event in events)
    assert not (fixed['root'] / 'cache').exists()
    assert result['scientific_fits'] == 0 and fixed['client'].calls == []


@pytest.fixture
def bounded_waits(monkeypatch):
    waits = []
    monkeypatch.setattr(stage.time, 'sleep', waits.append)
    monkeypatch.setattr(stage.hpc_fetch, 'SLEEP', waits.append)
    return waits


def test_fixed_object_progress_tracks_real_body_commit_and_final_verification(fixed, monkeypatch):
    row = fixed['release']['objects']['main_archive']
    (fixed['incoming'] / row['path']).unlink()
    clock = iter(range(0, 10000, 2))
    monkeypatch.setattr(stage.hpc_fetch, 'CLOCK', lambda: next(clock))
    events = []
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket', progress=events.append)
    assert result['status'] == 'ready' and fixed['client'].calls.count(row['key']) == 1
    main_events = [event for event in events if event.get('object') == 'main_archive']
    phases = [event['phase'] for event in main_events]
    assert phases.index('get_headers') < phases.index('body_read') < phases.index('commit') < phases.index('verified')
    body_event = next(event for event in main_events if event['phase'] == 'body_read')
    assert body_event['bytes_read'] == row['bytes'] and body_event['whole_body_verified'] is False
    assert all(not event['whole_body_verified'] for event in main_events if event['phase'] == 'get_headers')
    final = next(event for event in main_events if event['phase'] == 'verified')
    assert final['whole_body_verified'] is True and final['bytes_read'] == row['bytes']
    assert result['release_object_GETs'][0]['complete_body_read'] is True
    assert result['release_object_GETs'][0]['get_sha256'] == row['sha256']
    assert result['release_object_GETs'][1]['get_attempted'] is False
    receipt_path = fixed['root'] / result['attempt'] / 'receipt.json'
    assert json.loads(receipt_path.read_bytes()) == result
    assert result['active_transfer']['phase'] == 'finished'
    assert result['scientific_fits'] == 0 and result['GPU_checked'] is False


def test_release_ENOSPC_retains_safe_structural_diagnostics_and_closes_gate(fixed, monkeypatch, bounded_waits):
    row = fixed['release']['objects']['main_archive']
    target = fixed['incoming'] / row['path']
    target.unlink()
    secret = 'SECRET-EXCEPTION-PRIVATE-ENDPOINT-DO-NOT-PRINT'
    def disk_full(parent):
        raise OSError(errno.ENOSPC, secret, '/private/' + secret)
    monkeypatch.setattr(stage.hpc_fetch, '_new_temp', disk_full)
    events = []
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket', progress=events.append)
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_IO_ENOSPC'
    detail = result['failure_detail']
    assert detail['phase'] == 'temp_create' and detail['errno'] == errno.ENOSPC
    assert detail['errno_name'] == 'ENOSPC' and detail['retryable'] is False
    assert detail['bytes_read'] == 0 and detail['whole_body_verified'] is False
    assert fixed['client'].calls == [row['key']] and bounded_waits == []
    assert not target.exists() and 'quota' in result['minimal_gap']
    encoded = json.dumps({'result': result, 'events': events})
    assert secret not in encoded and 'fixture-bucket' not in encoded and row['key'] not in encoded
    assert not any(event['phase'] == 'verified' for event in events)
    assert json.loads((fixed['root'] / result['attempt'] / 'receipt.json').read_bytes()) == result
    assert all(stream.closed for stream in fixed['client'].streams)


def test_receipt_quota_failure_is_not_claimed_persisted_or_ready(fixed, monkeypatch):
    def no_receipt(*args):
        raise OSError(errno.EDQUOT, 'SECRET_STORAGE_MESSAGE')
    monkeypatch.setattr(stage, '_atomic_json', no_receipt)
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'failed' and result['receipt_persisted'] is False
    assert result['error_code'] == 'LOCAL_RECEIPT_WRITE_FAILED'
    assert result['failure_detail']['phase'] == 'stage_receipt_write'
    assert result['failure_detail']['errno_name'] == 'EDQUOT'
    assert fixed['client'].calls == [] and result['scientific_fits'] == 0
    assert 'SECRET_STORAGE_MESSAGE' not in json.dumps(result)


def test_materialization_directory_quota_failure_has_local_phase_not_R2_error(fixed, monkeypatch):
    def no_directory(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'SECRET_QUOTA_MESSAGE', '/private/SECRET_PATH')
    # Directory creation occurs while publishing an individual payload, after
    # the actual archive and headers have passed their original checks.
    monkeypatch.setattr(stage, '_copy_stream', no_directory)
    result = stage.prepare_data(fixed['root'], mode='materialize')
    assert result['status'] == 'failed' and result['error_code'] == 'LOCAL_IO_ENOSPC'
    assert result['failure_detail']['phase'] == 'materializing_verified_archive'
    assert result['failure_detail']['errno_name'] == 'ENOSPC'
    assert 'quota' in result['minimal_gap'] and 'SECRET' not in json.dumps(result)
    assert fixed['client'].calls == []


def test_cli_setup_io_failure_is_safe_json_not_raw_traceback(fixed, monkeypatch, capsys):
    def cannot_create(*args, **kwargs):
        raise OSError(errno.EACCES, 'SECRET_SETUP_MESSAGE', '/private/SECRET_PATH')
    monkeypatch.setattr(stage, 'prepare_data', cannot_create)
    assert stage.main(['fetch', '--root', str(fixed['root']), '--execute', '--network-host']) == 2
    printed = capsys.readouterr()
    value = json.loads(printed.out)
    assert value['status'] == 'failed' and value['receipt_persisted'] is False
    assert value['failure_detail']['errno_name'] == 'EACCES'
    assert 'SECRET' not in printed.out + printed.err


def test_bad_fixed_object_SHA_is_not_retried_or_published_ready(fixed, bounded_waits):
    row = fixed['release']['objects']['main_archive']
    target = fixed['incoming'] / row['path']
    target.unlink()
    original = fixed['client'].bodies[row['key']]
    fixed['client'].bodies[row['key']] = bytes([original[0] ^ 1]) + original[1:]
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=fixed['client'], bucket='fixture-bucket')
    assert result['status'] == 'failed' and result['error_code'] == 'R2_GET_BODY_SHA_OR_LENGTH_MISMATCH'
    assert result['failure_detail']['phase'] == 'body_verify'
    assert result['failure_detail']['retryable'] is False
    assert fixed['client'].calls == [row['key']] and bounded_waits == []
    assert not target.exists() and result['group_restore_receipts'] == []
    assert result['failure_detail']['whole_body_verified'] is False


@pytest.mark.parametrize('fixed', ['two_missing_baseline_objects'], indirect=True)
def test_original_profile_transient_retries_keep_receipts_without_reGET_of_success(fixed, monkeypatch, bounded_waits):
    class ReadTimeoutError(Exception):
        pass
    client = fixed['client']
    original = client.get_object
    bad_key = next(key for key in client.bodies if key.endswith('Stieger2021/1/X.npy'))
    good_key = next(key for key in client.bodies if key.endswith('BNCI2014_001/1/X.npy'))
    failures = [0]
    def sometimes(*, Bucket, Key):
        if Key == bad_key and failures[0] < 2:
            failures[0] += 1
            client.calls.append(Key)
            raise ReadTimeoutError('SECRET-RAW-SDK-EXCEPTION')
        return original(Bucket=Bucket, Key=Key)
    monkeypatch.setattr(client, 'get_object', sometimes)
    index_before = copy.deepcopy(fixed['index'])
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=client, bucket='fixture-bucket')
    assert result['status'] == 'ready' and client.calls.count(good_key) == 1 and client.calls.count(bad_key) == 3
    assert bounded_waits == [1, 2]
    proofs = [row for row in result['group_restore_receipts'] if row['ordinal'] == 0]
    assert [row['attempt'] for row in proofs] == [1, 2, 3]
    assert [row['status'] for row in proofs] == ['failed', 'failed', 'restored_verified']
    assert len({row['path'] for row in proofs}) == 3
    receipts = [json.loads((fixed['root'] / row['path']).read_bytes()) for row in proofs]
    assert receipts[0]['objects'][0]['status'] == 'restored_verified'
    assert all(value['objects'][0]['status'] == 'already_present_verified' for value in receipts[1:])
    assert 'SECRET-RAW-SDK-EXCEPTION' not in json.dumps(receipts + [result])
    assert stage.load_index(fixed['root']) == index_before
    profiled = result['group_restore_receipts'][-1]
    profile_receipt = json.loads((fixed['root'] / profiled['path']).read_bytes())
    assert profile_receipt['source_extension_profile'] == stage.source_extension_io.profile()
    assert all('/cache/cache/' in row['key'] for row in profile_receipt['objects'])
    assert all((fixed['root'] / row['path']).read_bytes() == fixed['bodies'][row['path']]
               for row in fixed['index']['files'] if row['path'] in fixed['omitted'])


@pytest.mark.parametrize('relative,ordinal', [
    ('cache/stieger_v2/Stieger2021/1/X.npy', 0),
    ('cache/source_extensions_20261019/Yang2025/1/X.npy', 1)])
def test_original_profile_persistent_network_failure_stops_after_three_saved_receipts(
        fixed, monkeypatch, bounded_waits, relative, ordinal):
    class ConnectionClosedError(Exception):
        pass
    client = fixed['client']
    original = client.get_object
    bad_key = next(key for key in client.bodies if key.endswith(relative))
    def blocked(*, Bucket, Key):
        if Key == bad_key:
            client.calls.append(Key)
            raise ConnectionClosedError('SECRET-BUCKET-PARAMETER')
        return original(Bucket=Bucket, Key=Key)
    monkeypatch.setattr(client, 'get_object', blocked)
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=client, bucket='fixture-bucket')
    assert result['status'] == 'failed' and result['error_code'] == 'original_profile_restore_failed_see_receipt'
    assert client.calls.count(bad_key) == 3 and bounded_waits == [1, 2]
    assert all(client.calls.count(key) == 1 for key in set(client.calls) - {bad_key})
    proofs = [row for row in result['group_restore_receipts'] if row['ordinal'] == ordinal]
    assert len(proofs) == 3 and [row['attempt'] for row in proofs] == [1, 2, 3]
    assert len({row['path'] for row in proofs}) == 3
    for proof in proofs:
        saved = json.loads((fixed['root'] / proof['path']).read_bytes())
        assert saved['status'] == 'failed' and saved['objects'][0]['complete_body_read'] is False
        assert 'SECRET-BUCKET-PARAMETER' not in json.dumps(saved)
    assert not (fixed['root'] / relative).exists()
    assert result['scientific_fits'] == 0 and result['external_query_accessed'] is False


@pytest.mark.parametrize('relative,ordinal', [
    ('cache/stieger_v2/Stieger2021/1/X.npy', 0),
    ('cache/source_extensions_20261019/Yang2025/1/X.npy', 1)])
def test_original_profile_bad_SHA_never_retries_or_relaxes_profile(fixed, bounded_waits, relative, ordinal):
    client = fixed['client']
    key = next(key for key in client.bodies if key.endswith(relative))
    raw = client.bodies[key]
    client.bodies[key] = bytes([raw[0] ^ 1]) + raw[1:]
    result = stage.prepare_data(fixed['root'], mode='fetch', execute=True, network_host=True,
        client=client, bucket='fixture-bucket')
    assert result['status'] == 'failed' and client.calls.count(key) == 1 and bounded_waits == []
    proofs = [row for row in result['group_restore_receipts'] if row['ordinal'] == ordinal]
    assert len(proofs) == 1
    saved = json.loads((fixed['root'] / proofs[0]['path']).read_bytes())
    assert saved['objects'][0]['error_code'] == 'remote_integrity_mismatch'
    assert saved['objects'][0]['complete_body_read'] is True
    assert saved['objects'][0]['get_sha256'] != saved['objects'][0]['sha256']
    assert saved['objects'][0]['status'] == 'failed'
    assert not (fixed['root'] / relative).exists()
    if ordinal:
        assert saved['source_extension_profile'] == stage.source_extension_io.profile()
