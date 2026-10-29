"""Offline checks of the exact embedded entrypoint; no uploads or science."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
FILES = (
    'delivery.py', 'group_archive.py', 'paper_credentials.py',
    'receipt_projection.py', 'r1_export.py', 'r11_export.py',
    'r11_paper_fix.py', 'transport/__init__.py',
    'transport/lossless_publish.py', 'transport/paper3_result_transport.py',
)


def copied_sources(tmp_path):
    folder = tmp_path / 'source'
    folder.mkdir()
    for name in (*FILES, 'build.py', 'bootstrap_template.py'):
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / name, destination)
    return folder


def run_python(*args, **kwargs):
    return subprocess.run(
        [sys.executable, '-I', '-S', '-B', '-u', *map(str, args)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=20, **kwargs,
    )


@pytest.fixture
def built(tmp_path):
    source = copied_sources(tmp_path)
    result = run_python(source / 'build.py')
    assert result.returncode == 0, result.stderr
    release = json.loads((source / 'RELEASE.json').read_text())
    entry = source / release['entrypoint']
    assert hashlib.sha256(entry.read_bytes()).hexdigest() == release['entrypoint_sha256']
    return source, entry, release


def install(entry, base):
    result = run_python(entry, '--install-base', base, '--install-only')
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]['event'] == 'R1_R11_EXPORT_BOOTSTRAP_STARTED'
    ready = rows[-1]
    assert ready['event'] == 'R1_R11_EXPORT_PACKAGE_READY'
    assert ready['scientific_jobs_changed'] == 0
    assert ready['torch_imported'] is False
    assert ready['downloads_performed'] is False
    return Path(ready['package'])


def test_exact_payload_install_and_repeat_reuse(built, tmp_path):
    source, entry, release = built
    package = install(entry, tmp_path / 'install')
    assert package.name == release['payload_sha256'][:20]
    assert set(release['files']) == set(FILES)
    assert {str(path.relative_to(package)) for path in package.rglob('*') if path.is_file()} == {
        *FILES, 'INSTALL_MANIFEST.json',
    }
    before = {}
    for name in FILES:
        path = package / name
        assert path.read_bytes() == (source / name).read_bytes()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == release['files'][name]
        before[name] = (path.stat().st_ino, path.stat().st_mtime_ns)
    assert install(entry, tmp_path / 'install') == package
    assert {name: ((package / name).stat().st_ino, (package / name).stat().st_mtime_ns)
            for name in FILES} == before


def test_tampered_installed_member_is_not_overwritten(built, tmp_path):
    _, entry, _ = built
    package = install(entry, tmp_path / 'install')
    changed = package / 'delivery.py'
    replacement = changed.read_bytes() + b'\n# Deliberately changed test source.\n'
    changed.write_bytes(replacement)
    result = run_python(entry, '--install-base', tmp_path / 'install', '--install-only')
    assert result.returncode != 0
    assert 'Installed uploader source differs' in result.stderr
    assert changed.read_bytes() == replacement


def test_embedded_readonly_check_does_not_load_science_or_network(built, tmp_path):
    _, entry, _ = built
    base = tmp_path / 'install'
    package = install(entry, base)
    root = tmp_path / 'science-root'
    root.mkdir()
    scientific_stop = root / 'STOP'
    scientific_stop.write_bytes(b'keep science stopped\n')
    work = tmp_path / 'uploader-work'
    # Execute the actual self-contained entrypoint while refusing imports of
    # scientific adapters, numeric libraries, and upload transports. A status
    # check must need none of them and must create no upload/runtime files.
    code = '''
import builtins, json, runpy, sys
original = builtins.__import__
blocked = ('torch', 'numpy', 'paper3r11', 'r1_export', 'r11_export',
           'paper_credentials', 'transport', 'urllib.request', 'http.client')
def guarded(name, *args, **kwargs):
    if any(name == value or name.startswith(value + '.') for value in blocked):
        raise AssertionError('Unexpected operational import during status: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--install-base', sys.argv[2], '--check',
            '--root', sys.argv[3], '--work', sys.argv[4]]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    if error.code not in (None, 0):
        raise
print(json.dumps({'guarded_status_check_completed': True}))
'''
    result = run_python('-c', code, entry, base, root, work)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    status = next(row for row in rows if row.get('event') == 'PAPER3_R1_R11_EXPORT_STATUS')
    assert status['startup'] == {}
    assert set(status['roles']) == {'r1-paper', 'r11-paper', 'r1-full', 'r11-full'}
    assert rows[-1]['guarded_status_check_completed'] is True
    assert not work.exists()
    assert scientific_stop.read_bytes() == b'keep science stopped\n'
    assert not list(package.rglob('__pycache__'))


@pytest.mark.parametrize('literal', [
    'ghp_' + 'A' * 30,
    'github_pat_' + 'B' * 35,
    'AKIA' + 'C' * 16,
])
def test_build_rejects_embedded_credential_literals(tmp_path, literal):
    source = copied_sources(tmp_path)
    with (source / 'delivery.py').open('a') as handle:
        handle.write('\n# Test-only synthetic credential marker: ' + literal + '\n')
    result = run_python(source / 'build.py')
    assert result.returncode != 0
    assert 'PUBLIC_SOURCE_CONTAINS_CREDENTIAL_LITERAL' in result.stderr
    assert not (source / 'START-PAPER3-R1-R11-RESULTS.py').exists()
