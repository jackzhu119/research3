"""Offline installer regression tests; no scheduler, GPU, or network operations."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load(ROOT / 'build_delivery.py', 'm12_installer_test_builder')
template = load(ROOT / 'installer_template.py', 'm12_installer_test_template')


@pytest.fixture
def artifact(tmp_path):
    source = tmp_path / 'source'
    package = source / 'paper3m12'
    package.mkdir(parents=True)
    for name in ('build_delivery.py', 'installer_template.py'):
        shutil.copy2(ROOT / name, source / name)
    (source / 'EXPERIMENT_CONFIG.json').write_text('{"execution_enabled": false}\n')
    (source / 'README.md').write_text('Scientific execution requires explicit prepare and submit.\n')
    (package / '__init__.py').write_text('')
    # A real standalone dispatch file lets tests exercise isolated interpreter
    # execution and command boundaries without importing any scientific engine.
    (package / 'entry.py').write_text('''import json,sys\nfrom pathlib import Path\na=sys.argv[1:]\nroot=Path(a[a.index('--root')+1]) if '--root' in a else Path(a[a.index('--plan')+1]).parents[2]\np=root/'dispatch.jsonl'\nwith p.open('a') as h:h.write(json.dumps(a)+'\\n')\nif a[0]=='prepare':\n out=Path(a[a.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)\n (out/'plan.json').write_text('{"status":"PREPARED"}')\nprint(json.dumps({'event':'STUB_ONLY','command':a[0],'scientific_work':False}))\n''')
    (source / 'tests').mkdir()
    (source / 'tests' / 'test_fixture.py').write_text('assert True\n')
    (package / 'vendor').mkdir()
    (package / 'vendor' / 'shim.py').write_text('VALUE=3\n')
    record = builder.build(source)
    module = load(source / 'delivery' / record['entry'], 'm12_built_installer_' + str(tmp_path.name))
    home = tmp_path / 'home'
    project = tmp_path / 'project'
    home.mkdir(); project.mkdir()
    return dict(source=source, home=home, project=project, record=record, module=module,
                entry=source / 'delivery' / record['entry'])


def invoke(artifact, *args):
    return subprocess.run([sys.executable, '-I', '-S', '-B', str(artifact['entry']),
        '--home', str(artifact['home']), '--root', str(artifact['project']),
        '--python', sys.executable, *args], capture_output=True, text=True, timeout=20)


def test_build_deterministic_with_vendor_included_and_no_private_data(artifact):
    x = artifact
    first = (x['source'] / 'delivery' / x['record']['entry']).read_bytes()
    private = x['source'] / 'paper3m12/private'; private.mkdir()
    (private / 'github.json').write_text('{"token":"not a real credential"}')
    (x['source'] / 'paper3m12/checkpoint.npz').write_bytes(b'not code')
    second = builder.build(x['source'])
    assert second == x['record']
    assert first == (x['source'] / 'delivery' / second['entry']).read_bytes()
    archive = zipfile.ZipFile(x['source'] / 'delivery/paper3-mechanism12.zip')
    assert 'paper3m12/vendor/shim.py' in archive.namelist()
    assert not any('private' in n or n.endswith('.npz') for n in archive.namelist())


def test_default_real_entry_install_twice_has_no_science_or_dispatch(artifact):
    x = artifact; project_before = list(x['project'].rglob('*'))
    first = invoke(x)
    assert first.returncode == 0, first.stderr
    assert 'M12_CODE_READY_NO_ACTION' in first.stdout
    assert not (x['project'] / 'dispatch.jsonl').exists()
    assert project_before == list(x['project'].rglob('*'))
    package = x['module'].package_path(x['home'])
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in package.rglob('*') if p.is_file()}
    second = invoke(x)
    assert second.returncode == 0, second.stderr
    assert snapshots == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in package.rglob('*') if p.is_file()}
    assert json.loads((package / 'MANIFEST.json').read_text())['scope'].startswith('Frozen code')


def test_installer_import_does_not_load_scientific_dependencies(artifact):
    x = artifact
    harness = '''import runpy,sys\nm=runpy.run_path(sys.argv[1],run_name='offline_installer_import')\nassert not any(k.split('.')[0] in {'torch','numpy','scipy','sklearn'} for k in sys.modules)\nprint(m['SCHEMA'])\n'''
    checked = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', harness, str(x['entry'])],
                             capture_output=True, text=True, timeout=10)
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.strip() == 'paper3-mechanism12-package-v1'


def test_explicit_prepare_has_only_metadata_dispatch_and_submit_is_separate(artifact):
    x = artifact
    result = invoke(x, '--prepare')
    assert result.returncode == 0, result.stderr
    rows = [json.loads(row) for row in (x['project'] / 'dispatch.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0][0] == 'prepare'
    assert '--submit' not in rows[0]
    assert Path(rows[0][rows[0].index('--output') + 1]) == x['project'] / 'runs/r12_controlled_mechanism'
    assert Path(rows[0][rows[0].index('--package') + 1]) == x['module'].package_path(x['home'])


def test_prepare_submit_preserves_exact_arguments_with_space_paths(artifact, tmp_path):
    x = artifact
    project = tmp_path / 'project with spaces'; project.mkdir()
    x = dict(x, project=project)
    result = invoke(x, '--prepare', '--submit')
    assert result.returncode == 0, result.stderr
    rows = [json.loads(row) for row in (project / 'dispatch.jsonl').read_text().splitlines()]
    assert [row[0] for row in rows] == ['prepare', 'start']
    assert rows[1] == ['start', '--plan', str(project / 'runs/r12_controlled_mechanism/plan.json'), '--submit']


def test_prepare_failure_never_starts_controller(artifact, monkeypatch):
    x = artifact; module = x['module']; calls = []
    def run(command, **kwargs):
        calls.append(command)
        return type('Reply', (), {'returncode': 7})()
    monkeypatch.setattr(subprocess, 'run', run)
    result = module.main(['--home', str(x['home']), '--root', str(x['project']),
        '--python', sys.executable, '--prepare', '--submit'])
    assert result == 7 and len(calls) == 1 and 'prepare' in calls[0]


def test_resume_explicitly_starts_only_existing_plan_route(artifact):
    x = artifact
    assert invoke(x).returncode == 0
    result = invoke(x, '--resume', '--submit')
    assert result.returncode == 0, result.stderr
    rows = [json.loads(row) for row in (x['project'] / 'dispatch.jsonl').read_text().splitlines()]
    assert [row[0] for row in rows] == ['start'] and '--submit' in rows[0]


def test_status_leaves_package_and_project_files_unmodified(artifact):
    x = artifact
    assert invoke(x).returncode == 0
    module = x['module']; calls = []
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in x['home'].rglob('*') if p.is_file()}
    from unittest.mock import patch
    def run(command, **kwargs):
        calls.append(command)
        return type('Reply', (), {'returncode': 0})()
    with patch.object(subprocess, 'run', run):
        result = module.main(['--home', str(x['home']), '--root', str(x['project']),
            '--python', sys.executable, '--status'])
    assert result == 0 and calls[0][6:] == ['status', '--plan', str(x['project'] / 'runs/r12_controlled_mechanism/plan.json')]
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in x['home'].rglob('*') if p.is_file()}
    assert not (x['project'] / 'dispatch.jsonl').exists()


@pytest.mark.parametrize('args', [('--submit',), ('--resume',), ('--prepare', '--resume', '--submit'),
    ('--status', '--prepare'), ('--status', '--submit'), ('--status', '--install')])
def test_unapproved_mutating_combinations_fail_before_install(artifact, args):
    x = artifact; result = invoke(x, *args)
    assert result.returncode != 0
    assert not (x['home'] / 'paper3-mechanism12').exists()
    assert not (x['project'] / 'dispatch.jsonl').exists()


@pytest.mark.parametrize('output', ['..', '../outside', '.', '/tmp/not-the-project'])
def test_output_path_cannot_escape_or_replace_project_root(artifact, output):
    x = artifact; result = invoke(x, '--prepare', '--output', output)
    assert result.returncode != 0
    assert not (x['home'] / 'paper3-mechanism12').exists()


def test_existing_package_corruption_never_overwrites_or_dispatches(artifact):
    x = artifact; module = x['module']; package = module.install(x['home'])
    target = package / 'paper3m12/vendor/shim.py'; target.write_text('changed\n')
    with pytest.raises(ValueError, match='EXISTING_IMMUTABLE_PACKAGE_DIFFERS'):
        module.install(x['home'])
    assert target.read_text() == 'changed\n'
    assert not (x['project'] / 'dispatch.jsonl').exists()


def test_existing_nested_symlink_is_rejected(artifact, tmp_path):
    x = artifact; module = x['module']; package = module.install(x['home'])
    vendor = package / 'paper3m12/vendor'; shutil.rmtree(vendor)
    outside = tmp_path / 'outside'; outside.mkdir(); (outside / 'shim.py').write_text('VALUE=3\n')
    vendor.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='EXISTING_IMMUTABLE_PACKAGE_DIFFERS'):
        module.install(x['home'])
    assert (outside / 'shim.py').read_text() == 'VALUE=3\n'


def forged(module, members, manifest_files=None):
    values = dict(members)
    if manifest_files is None:
        manifest_files = {name: {'bytes': len(data), 'sha256': module.sha(data)} for name, data in values.items()}
    manifest = module.encoded({'schema_version': module.SCHEMA, 'files': manifest_files})
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, 'w') as z:
        z.writestr('MANIFEST.json', manifest)
        for name, data in values.items():
            info = zipfile.ZipInfo(name); info.create_system = 3; info.external_attr = (stat.S_IFREG | 0o600) << 16
            z.writestr(info, data)
    payload = memory.getvalue()
    return {'PAYLOAD_B85': base64.b85encode(payload).decode(), 'PAYLOAD_SHA256': module.sha(payload),
            'MANIFEST_SHA256': module.sha(manifest)}


@pytest.mark.parametrize('name', ['../escaped.py', '/absolute.py', 'paper3m12/../escape.py',
    'paper3m12/private/github.json', 'paper3m12\\escape.py'])
def test_archive_traversal_and_credential_members_are_refused(artifact, monkeypatch, name):
    module = artifact['module']; payload = forged(module, {name: b'bad', 'paper3m12/entry.py': b'pass\n'})
    for key, value in payload.items(): monkeypatch.setattr(module, key, value)
    with pytest.raises(ValueError, match='UNSAFE_PACKAGE_MEMBER_PATH'):
        module.install(artifact['home'])
    assert not (artifact['home'] / 'paper3-mechanism12').exists()


def test_payload_hash_mismatch_is_refused_before_install(artifact, monkeypatch):
    module = artifact['module']; monkeypatch.setattr(module, 'PAYLOAD_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='PACKAGE_SHA256_DIFFERS'):
        module.install(artifact['home'])
    assert not (artifact['home'] / 'paper3-mechanism12').exists()


def test_member_hash_mismatch_cannot_publish_partial_install(artifact, monkeypatch):
    module = artifact['module']
    data = b'pass\n'
    payload = forged(module, {'paper3m12/entry.py': data},
        {'paper3m12/entry.py': {'bytes': len(data), 'sha256': '0' * 64}})
    for key, value in payload.items(): monkeypatch.setattr(module, key, value)
    with pytest.raises(ValueError, match='PACKAGE_MEMBER_SHA256_DIFFERS'):
        module.install(artifact['home'])
    base = artifact['home'] / 'paper3-mechanism12'
    assert not module.package_path(artifact['home']).exists()
    assert not list(base.glob('.install-*'))


def test_current_round_data_and_other_rounds_are_unchanged_by_install_and_prepare(artifact):
    x = artifact
    kept = {}
    for relative in ['p3work/spatial_geometry.py', 'p3work/adapt.py', 'contracts/input_contract.json']:
        path = x['project'] / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('existing immutable ' + relative).encode()); kept[path] = path.read_bytes()
    checked = invoke(x, '--prepare')
    assert checked.returncode == 0, checked.stderr
    assert all(path.read_bytes() == raw for path, raw in kept.items())


def test_package_status_requires_existing_exact_install_without_creating_directories(artifact):
    x = artifact
    checked = invoke(x, '--status')
    assert checked.returncode != 0
    assert 'EXACT_INSTALLED_PACKAGE_REQUIRED_FOR_STATUS' in checked.stderr
    assert not (x['home'] / 'paper3-mechanism12').exists()
    assert not (x['project'] / 'dispatch.jsonl').exists()


def test_live_other_round_status_can_advance_during_prepare_without_a_guard_failure(artifact, monkeypatch):
    x = artifact; module = x['module']; calls = []
    runtime = x['project'] / 'runs/r11_campaign/runtime/status.json'
    runtime.parent.mkdir(parents=True); runtime.write_text('{"complete_tasks": 700}')
    def run(command, **kwargs):
        calls.append(command)
        # A legitimate R11 update must not become an M12 compatibility gate.
        runtime.write_text('{"complete_tasks": 701}')
        return type('Reply', (), {'returncode': 0})()
    monkeypatch.setattr(subprocess, 'run', run)
    result = module.main(['--home', str(x['home']), '--root', str(x['project']),
        '--python', sys.executable, '--prepare', '--submit'])
    assert result == 0 and [row[6] for row in calls] == ['prepare', 'start']
    assert json.loads(runtime.read_text())['complete_tasks'] == 701


def test_compact_public_evidence_and_offline_audit_keep_original_crlf_bytes(artifact):
    x = artifact; evidence = x['source'] / 'evidence'; evidence.mkdir()
    original = b'participant,effect\r\nS1,0.04\r\n'
    (evidence / 'original.csv').write_bytes(original)
    record = {'files': {'original.csv': {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}}}
    (evidence / 'MANIFEST.json').write_text(json.dumps(record))
    (x['source'] / 'audit_design.py').write_text('print("offline design audit fixture")\n')
    built = builder.build(x['source'])
    archive = zipfile.ZipFile(x['source'] / 'delivery/paper3-mechanism12.zip')
    assert archive.read('evidence/original.csv') == original
    assert 'evidence/MANIFEST.json' in archive.namelist() and 'audit_design.py' in archive.namelist()
    module = load(x['source'] / 'delivery' / built['entry'], 'm12_evidence_installer')
    folder = module.install(x['home'])
    assert (folder / 'evidence/original.csv').read_bytes() == original
    assert hashlib.sha256((folder / 'evidence/original.csv').read_bytes()).hexdigest() == record['files']['original.csv']['sha256']


def test_evidence_hash_mismatch_is_a_release_build_error_not_a_scientific_runtime_gate(artifact):
    x = artifact; evidence = x['source'] / 'evidence'; evidence.mkdir()
    (evidence / 'original.csv').write_bytes(b'changed')
    (evidence / 'MANIFEST.json').write_text(json.dumps({'files': {'original.csv': {'bytes': 7, 'sha256': '0' * 64}}}))
    with pytest.raises(ValueError, match='DESIGN_EVIDENCE_SHA256_DIFFERS'):
        builder.build(x['source'])
    assert not (x['project'] / 'dispatch.jsonl').exists()
