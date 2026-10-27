from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import hashlib

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def delivery(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    for p in ROOT.glob('*.py'):
        shutil.copy2(p, source/p.name)
    spec = importlib.util.spec_from_file_location('test_io_build', source/'build_delivery.py')
    build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    record = build.build(source)
    return source/'delivery'/record['entry'], record


def test_real_offline_default_install_no_training_or_network(delivery, tmp_path):
    entry, record = delivery; home = tmp_path/'home'
    code = '''import sys,runpy
def audit(event,args):
 if event in ('subprocess.Popen','socket.__new__','socket.connect'):raise AssertionError(event)
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','requests'):raise AssertionError(args[0])
sys.addaudithook(audit)
sys.argv=[sys.argv[1],'--home',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(entry), str(home)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'R11_IO_REPAIR_INSTALLED' in result.stdout
    assert 'R11_IO_REPAIR_CODE_READY_NO_ACTION' in result.stdout
    installed = home/'paper3-r11-io-repair'/record['payload_sha256'][:20]
    manifest = json.loads((installed/'MANIFEST.json').read_text())
    for name, row in manifest['files'].items():
        data = (installed/name).read_bytes()
        assert len(data) == row['bytes']
        assert hashlib.sha256(data).hexdigest() == row['sha256']
    before = {p: p.stat().st_mtime_ns for p in installed.rglob('*') if p.is_file()}
    repeated = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code, str(entry), str(home)],
                              capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 0, repeated.stderr
    assert before == {p: p.stat().st_mtime_ns for p in installed.rglob('*') if p.is_file()}


def test_submit_cannot_happen_without_repair(delivery, tmp_path):
    entry, _ = delivery
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(entry), '--submit', '--home', str(tmp_path/'home')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'SUBMIT_REQUIRES_EXPLICIT_REPAIR' in result.stderr
    assert not (tmp_path/'home').exists()


def test_explicit_resume_dispatch_not_implicit_training(delivery, tmp_path, monkeypatch):
    entry, _ = delivery
    spec = importlib.util.spec_from_file_location('test_io_installer', entry)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw: calls.append(command) or type('R', (), {'returncode': 0})())
    assert module.main(['--repair', '--submit', '--home', str(tmp_path/'home'), '--python', sys.executable]) == 0
    assert len(calls) == 1
    command = calls[0]
    assert Path(command[5]).name == 'supervisor.py'
    assert command[6] == 'start'
    assert command.count('--submit') == 1
    assert 'worker' not in command and 'scancel' not in command and 'sbatch' not in command


def test_corrupt_installed_code_cannot_be_silently_reused(delivery, tmp_path):
    entry, record = delivery
    spec = importlib.util.spec_from_file_location('test_io_installer_corruption', entry)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    installed = module.install(tmp_path/'home')
    (installed/'supervisor.py').write_text('damaged')
    with pytest.raises(ValueError, match='EXISTING_IMMUTABLE_REPAIR_PACKAGE_DIFFERS'):
        module.install(tmp_path/'home')
