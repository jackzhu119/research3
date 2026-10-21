"""Tiny artificial IO fixtures; no real data, SDK network or scientific fit."""
from __future__ import annotations

import copy
import fcntl
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
def fixed(tmp_path, monkeypatch):
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
