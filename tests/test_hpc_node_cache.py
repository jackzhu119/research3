"""Byte-identical node-local software-cache fixtures; no installation or fit."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import pytest

from hpc import node_cache as cache


def allocated():
    return {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'cpu-fixture',
            'SLURM_JOB_PARTITION': 'cyhpc_1', 'SLURM_CPUS_PER_TASK': '16',
            'SLURM_MEM_PER_NODE': '65536'}


def digest(body):
    return 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b'=').decode()


@pytest.fixture
def software(tmp_path):
    site = tmp_path / 'installed' / 'site-packages'; site.mkdir(parents=True)
    standard = tmp_path / 'installed' / 'stdlib'; standard.mkdir()
    (standard / 'json').mkdir()
    (standard / 'json' / '__init__.py').write_bytes(b'# exact installed standard library\n')
    (standard / 'lib-dynload').mkdir()
    (standard / 'lib-dynload' / '_fixture.so').write_bytes(b'ELF SOFTWARE TEST DOUBLE NOT EXECUTABLE')
    entries = {
        'numpy/__init__.py': b'__version__ = "1.0"\n',
        'numpy/native.so': b'ELF PACKAGE SOFTWARE TEST DOUBLE NOT EXECUTABLE',
        'numpy-1.0.dist-info/METADATA': b'Metadata-Version: 2.1\nName: numpy\nVersion: 1.0\n',
        'numpy-1.0.dist-info/WHEEL': b'Wheel-Version: 1.0\n',
    }
    for relative, body in entries.items():
        path = site / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
    record = site / 'numpy-1.0.dist-info/RECORD'
    def update_record(extra=()):
        rows = [[name, digest(body), str(len(body))] for name, body in entries.items()]
        rows.extend(extra)
        rows.append(['numpy-1.0.dist-info/RECORD', '', ''])
        stream = io.StringIO(); csv.writer(stream).writerows(rows); record.write_text(stream.getvalue())
    update_record()
    scratch = tmp_path / 'node-local'; scratch.mkdir()
    output = tmp_path / 'project' / '.paper3-hpc/jobs/123/node_cache.json'
    def run(**kwargs):
        values = dict(environ=allocated(), hostname='cpu-fixture',
            distributions=list(importlib.metadata.distributions(path=[str(site)])),
            site_root=site, stdlib=standard, scratch=scratch,
            local_checker=lambda path: True, progress=None)
        values.update(kwargs)
        return cache.prepare_cache(output, **values)
    return dict(site=site, stdlib=standard, scratch=scratch, output=output,
                entries=entries, record=record, update_record=update_record, run=run)


def paths(result):
    return [Path(part) for part in result['pythonpath'].split(':') if part]


def cached_file(result, relative):
    values = [path / relative for path in paths(result) if (path / relative).is_file()]
    assert len(values) == 1
    return values[0]


def test_source_parent_validation_is_repeated_for_each_prepare_transaction(software):
    software['run']()
    package = software['site'] / 'numpy'
    original = software['site'] / 'original_numpy'
    package.rename(original)
    package.symlink_to(original, target_is_directory=True)
    with pytest.raises(cache.CacheError, match='symlink_forbidden'):
        software['run']()


def test_node_cache_copies_actual_bytes_and_records_verified_identity_without_install_or_fit(software):
    value = software['run']()
    assert value['files_verified'] >= 4
    assert value['bytes_verified'] >= len(software['entries']['numpy/native.so'])
    assert len(value['manifest_sha256']) == 64
    assert cached_file(value, 'numpy/__init__.py').read_bytes() == software['entries']['numpy/__init__.py']
    assert cached_file(value, 'numpy/native.so').read_bytes() == software['entries']['numpy/native.so']
    assert cached_file(value, 'json/__init__.py').read_bytes() == (software['stdlib'] / 'json/__init__.py').read_bytes()
    assert not any(path.name == 'lib-dynload' for path in paths(value))
    assert not any((path / 'lib-dynload/_fixture.so').exists() or (path / '_fixture.so').exists() for path in paths(value))
    assert (software['stdlib'] / 'lib-dynload/_fixture.so').read_bytes() == b'ELF SOFTWARE TEST DOUBLE NOT EXECUTABLE'
    assert all(path.is_relative_to(software['scratch']) for path in paths(value))
    assert json.loads(software['output'].read_text())['manifest_sha256'] == value['manifest_sha256']


@pytest.mark.parametrize('environment,host', [({}, 'cpu-fixture'),
    ({**allocated(), 'SLURMD_NODENAME': 'login1'}, 'cpu-fixture'),
    (allocated(), 'login1')])
def test_login_or_unallocated_execution_is_refused_before_writing(software, environment, host):
    before = list(software['scratch'].iterdir())
    with pytest.raises(ValueError):
        software['run'](environ=environment, hostname=host)
    assert list(software['scratch'].iterdir()) == before
    assert not software['output'].exists()


def test_record_changed_installed_software_is_never_cached(software):
    (software['site'] / 'numpy/__init__.py').write_bytes(b'changed installed software')
    with pytest.raises(ValueError):
        software['run']()
    assert not software['output'].exists()


def test_symlinked_installed_package_file_is_refused(software, tmp_path):
    source = software['site'] / 'numpy/native.so'; raw = source.read_bytes(); source.unlink()
    other = tmp_path / 'different-library.so'; other.write_bytes(raw); source.symlink_to(other)
    with pytest.raises(ValueError):
        software['run']()
    assert not software['output'].exists()


def test_software_cap_refuses_cache_without_false_ready_receipt(software):
    with pytest.raises(ValueError):
        software['run'](max_bytes=1)
    assert not software['output'].exists()


def test_nonlocal_scratch_is_refused(software):
    with pytest.raises(ValueError):
        software['run'](local_checker=lambda path: False)
    assert not software['output'].exists()


def test_symlink_scratch_is_refused_even_when_local_checker_claims_true(software, tmp_path):
    link = tmp_path / 'linked-scratch'; link.symlink_to(software['scratch'], target_is_directory=True)
    with pytest.raises(ValueError):
        software['run'](scratch=link)
    assert not software['output'].exists()


def test_environment_credentials_are_not_serialized(software, monkeypatch):
    secret = 'SOFTWARE_SECRET_NEVER_COPY_THIS_VALUE'
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', secret)
    events = []
    value = software['run'](environ={**allocated(), 'AWS_SECRET_ACCESS_KEY': secret}, progress=events.append)
    assert secret not in json.dumps(value)
    assert secret not in software['output'].read_text()
    assert secret not in json.dumps(events)
    assert all(secret.encode() not in file.read_bytes() for root in paths(value)
               for file in root.rglob('*') if file.is_file())


def test_untracked_private_paths_pth_and_installer_urls_are_not_copied(software, tmp_path):
    site = software['site']
    direct = site / 'numpy-1.0.dist-info/direct_url.json'
    direct.write_bytes(b'{"url":"https://private.example/DO_NOT_COPY_SECRET"}')
    hook = site / 'evil.pth'; hook.write_bytes(b'import NOT_AN_ALLOWED_RUNTIME_PACKAGE\n')
    external = site.parent / 'private-credentials'; external.write_bytes(b'PRIVATE_CREDENTIALS_NEVER_COPY')
    software['update_record']([
        ['numpy-1.0.dist-info/direct_url.json', digest(direct.read_bytes()), str(direct.stat().st_size)],
        ['evil.pth', digest(hook.read_bytes()), str(hook.stat().st_size)],
        ['../private-credentials', digest(external.read_bytes()), str(external.stat().st_size)],
    ])
    value = software['run']()
    files = [path for root in paths(value) for path in root.rglob('*') if path.is_file()]
    assert not any(path.name in {'direct_url.json', 'evil.pth', 'private-credentials'} for path in files)
    assert all(b'DO_NOT_COPY_SECRET' not in path.read_bytes() and b'PRIVATE_CREDENTIALS_NEVER_COPY' not in path.read_bytes() for path in files)


def test_standard_library_customization_and_site_packages_are_excluded(software):
    standard = software['stdlib']
    for relative in ['sitecustomize.py', 'usercustomize.py', 'site-packages/unselected.py', '__pycache__/unsafe.py']:
        path = standard / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'SHOULD_NOT_BE_COPIED')
    value = software['run']()
    assert not any(b'SHOULD_NOT_BE_COPIED' in path.read_bytes() for root in paths(value)
                   for path in root.rglob('*') if path.is_file())


def test_verified_existing_cache_reuses_same_file(software):
    value = software['run'](); path = cached_file(value, 'numpy/__init__.py'); inode = path.stat().st_ino
    again = software['run']()
    assert again['pythonpath'] == value['pythonpath']
    assert cached_file(again, 'numpy/__init__.py').stat().st_ino == inode
    assert cached_file(again, 'numpy/__init__.py').read_bytes() == software['entries']['numpy/__init__.py']


def test_corrupt_existing_cache_never_claims_verified_reuse(software):
    value = software['run'](); path = cached_file(value, 'numpy/__init__.py')
    path.chmod(0o600); path.write_bytes(b'corrupt local cache')
    with pytest.raises(ValueError):
        software['run']()


def test_unsigned_runtime_package_file_is_refused(software):
    rows = list(csv.reader(io.StringIO(software['record'].read_text())))
    for row in rows:
        if row[0] == 'numpy/__init__.py':
            row[1] = ''
    stream = io.StringIO(); csv.writer(stream).writerows(rows)
    software['record'].write_text(stream.getvalue())
    with pytest.raises(ValueError):
        software['run']()
    assert not software['output'].exists()


def test_insufficient_local_disk_is_refused_without_false_ready(software, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(cache.shutil, 'disk_usage', lambda path: SimpleNamespace(
        total=1024**3, used=1024**3, free=0))
    with pytest.raises(ValueError):
        software['run']()
    assert not software['output'].exists()


def test_unhashed_record_name_inside_runtime_package_is_not_metadata_exception(software):
    path = software['site'] / 'numpy/RECORD'; path.write_bytes(b'unsigned runtime payload')
    software['update_record']([['numpy/RECORD', '', str(path.stat().st_size)]])
    with pytest.raises(ValueError):
        software['run']()
    assert not software['output'].exists()


def test_unknown_slurm_memory_cannot_authorize_node_cache(software):
    env = {key: value for key, value in allocated().items() if key != 'SLURM_MEM_PER_NODE'}
    with pytest.raises(ValueError):
        software['run'](environ=env)
    assert not software['output'].exists()


@pytest.mark.parametrize('cpus,expected', [('2', 2), ('24', 8)])
def test_parallel_software_copy_never_exceeds_the_allocation_or_eight_workers(software, cpus, expected):
    value = software['run'](environ={**allocated(), 'SLURM_CPUS_PER_TASK': cpus})
    assert value['deployment_copy_workers'] == expected
    assert value['scientific_budgets_changed'] is False
