"""Real local bytecode preserves the original source path and frozen checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hpc import project_bytecode as cache


def env():
    return {'SLURM_JOB_ID': '123', 'SLURMD_NODENAME': 'cpu-fixture',
            'SLURM_CPUS_PER_TASK': '16', 'SLURM_MEM_PER_NODE': '65536'}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'; (root / 'hpc').mkdir(parents=True)
    source = root / 'fixture_module.py'; source.write_text('VALUE = 23\n')
    raw = source.read_bytes()
    (root / 'hpc/scientific_code.json').write_text(json.dumps({
        'schema_version': 'paper3-hpc-software-overlay-v1', 'deployment_overrides': {},
        'deployment_support_files': [], 'baseline_files': [{'path': source.name,
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}]}))
    local = tmp_path / 'local'; local.mkdir(mode=0o700)
    return root, local, source


def test_existing_interpreter_reads_local_bytecode_with_original_filename_and_B(project):
    root, local, source = project
    original = source.read_bytes()
    result = cache.compile_project(root, local, environ=env(), hostname='cpu-fixture', progress=None)
    source.chmod(0)
    try:
        child = subprocess.run([sys.executable, '-B', '-c',
            'import fixture_module as m; print(m.VALUE); print(m.__file__)'], cwd=root,
            env=dict(os.environ, PYTHONPYCACHEPREFIX=result['cache_prefix'], PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, timeout=10)
        assert child.returncode == 0, child.stderr
        assert child.stdout.splitlines() == ['23', str(source)]
    finally:
        source.chmod(0o600)
    assert source.read_bytes() == original
    assert not (root / '__pycache__').exists()
    assert result['scientific_fit_performed'] is result['source_modified'] is False


def test_changed_frozen_source_is_rejected_before_any_compilation(project):
    root, local, source = project; source.write_text('VALUE = 99\n')
    with pytest.raises(ValueError, match='SHA256'):
        cache.compile_project(root, local, environ=env(), hostname='cpu-fixture', progress=None)
    assert not list(local.iterdir())


def test_login_or_missing_allocation_cannot_prepare_bytecode(project):
    root, local, _ = project
    with pytest.raises(ValueError):
        cache.compile_project(root, local, environ={}, hostname='login1', progress=None)
    assert not list(local.iterdir())


def test_shared_or_other_user_writable_cache_is_rejected(project):
    root, local, _ = project; local.chmod(0o755)
    with pytest.raises(cache.BytecodeError, match='private_node_cache'):
        cache.compile_project(root, local, environ=env(), hostname='cpu-fixture', progress=None)
