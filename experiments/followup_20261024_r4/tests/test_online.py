"""Standalone delivery regressions: local files and fake Slurm only."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
PYTHON = Path('/tmp/paper3-r3-cu121/bin/python')
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location('paper3_round4_build_online', PACKAGE / 'build_online.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_copy(tmp_path):
    source = tmp_path / 'source'
    shutil.copytree(PACKAGE, source, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', 'references', 'online', 'private', '.git'))
    # Check proof-file delivery without borrowing an earlier round's evidence.
    proof = source / 'compatibility/environment.json'
    if not proof.is_file():
        proof.parent.mkdir(exist_ok=True)
        proof.write_text(json.dumps({'scope': 'SYNTHETIC_INSTALLATION_TEST_ONLY'}))
    return source


@pytest.fixture
def delivery(builder, source_copy, tmp_path):
    wrapper = tmp_path / 'download' / 'START-PAPER3-R4-ONLINE.py'
    manifest = tmp_path / 'download' / 'PACKAGE.json'
    wrapper.parent.mkdir()
    info = builder.build(package=source_copy, output=wrapper, manifest=manifest)
    info['_test_algorithm_sha256'] = hashlib.sha256((source_copy / 'paper3r4/algorithm.py').read_bytes()).hexdigest()
    return wrapper, info, manifest


def invoke(wrapper, *args, cwd=None, env=None, timeout=30):
    return subprocess.run([str(PYTHON), '-I', '-S', '-B', '-u', str(wrapper), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)


def installed(base, info):
    return Path(base) / info['payload_sha256'][:20]


def test_isolated_downloaded_file_installs_and_reuses_exact_sources(delivery, tmp_path):
    wrapper, info, manifest = delivery
    base = tmp_path / 'installed'
    isolated = tmp_path / 'empty-working-directory'
    isolated.mkdir()
    first = invoke(wrapper, '--install-base', base, '--install-only', cwd=isolated)
    assert first.returncode == 0, first.stdout + first.stderr
    destination = installed(base, info)
    assert (destination / 'START.py').is_file()
    assert not (destination / 'paper3r3').exists()
    assert hashlib.sha256((destination / 'paper3r4/algorithm.py').read_bytes()).hexdigest() == info['_test_algorithm_sha256']
    before = (destination / 'paper3r4/algorithm.py').stat().st_mtime_ns
    second = invoke(wrapper, '--install-base', base, '--install-only', cwd=isolated)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (destination / 'paper3r4/algorithm.py').stat().st_mtime_ns == before
    assert json.loads(manifest.read_text())['wrapper_sha256'] == info['wrapper_sha256']


def test_build_is_reproducible_for_the_same_source_snapshot(builder, source_copy, tmp_path):
    first = builder.build(package=source_copy, output=tmp_path / 'first.py', manifest=tmp_path / 'first.json')
    second = builder.build(package=source_copy, output=tmp_path / 'second.py', manifest=tmp_path / 'second.json')
    assert first['payload_sha256'] == second['payload_sha256']
    assert first['wrapper_sha256'] == second['wrapper_sha256']


def test_installed_numeric_source_tamper_is_rejected(delivery, tmp_path):
    wrapper, info, _ = delivery
    base = tmp_path / 'installed'
    assert invoke(wrapper, '--install-base', base, '--install-only').returncode == 0
    numeric = installed(base, info) / 'paper3r4/algorithm.py'
    numeric.write_text(numeric.read_text() + '\n# changed after installation\n')
    result = invoke(wrapper, '--install-base', base, '--install-only')
    assert result.returncode != 0
    assert 'changed after installation' in numeric.read_text()


def test_concurrent_installers_reuse_one_complete_directory(delivery, tmp_path):
    wrapper, info, _ = delivery
    base = tmp_path / 'installed'
    command = [str(PYTHON), '-I', '-S', '-B', '-u', str(wrapper),
               '--install-base', str(base), '--install-only']
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
        destination = installed(base, info)
        assert (destination / 'INSTALL_MANIFEST.json').is_file()
        assert not list(base.glob('.' + info['payload_sha256'][:20] + '.tmp-*'))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_payload_omits_private_data_pdf_and_cache(builder, source_copy, tmp_path):
    forbidden = {
        'private/github.json': '{"token":"test-private-token-never-publish"}',
        '.env': 'TOKEN=test-private-env-never-publish',
        'data/participant.npy': 'raw-person-level-data-never-publish',
        'references/article.pdf': '%PDF-1.4 forbidden reference payload',
        'paper3r4/__pycache__/algorithm.cpython-312.pyc': 'compiled-cache-never-publish',
        'unrelated.bin': 'binary-never-publish',
        'eye_bci_zenodo_metadata.json': '{"unneeded":"raw-provider-response"}',
        '27306_datacite_metadata.json': '{"unneeded":"raw-provider-response"}',
        'weibo_nemar_v103_manifest.json': '{"unneeded":"raw-download-inventory"}',
    }
    for relative, contents in forbidden.items():
        path = source_copy / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    wrapper, manifest = tmp_path / 'standalone.py', tmp_path / 'manifest.json'
    info = builder.build(package=source_copy, output=wrapper, manifest=manifest)
    base = tmp_path / 'installed'
    result = invoke(wrapper, '--install-base', base, '--install-only')
    assert result.returncode == 0, result.stdout + result.stderr
    destination = installed(base, info)
    assert all(not (destination / name).exists() for name in forbidden)
    assert (destination / 'compatibility/environment.json').is_file()
    assert (destination / 'tests/test_runtime.py').is_file()


def test_selected_source_symlink_cannot_publish_outside_content(builder, source_copy, tmp_path):
    outside = tmp_path / 'outside-secret.py'
    outside.write_text('TOKEN = "outside-content-must-not-be-published"\n')
    (source_copy / 'paper3r4/symlinked.py').symlink_to(outside)
    wrapper, manifest = tmp_path / 'standalone.py', tmp_path / 'manifest.json'
    try:
        info = builder.build(package=source_copy, output=wrapper, manifest=manifest)
    except (ValueError, RuntimeError):
        return
    base = tmp_path / 'installed'
    result = invoke(wrapper, '--install-base', base, '--install-only')
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (installed(base, info) / 'paper3r4/symlinked.py').exists()


def test_prepare_only_and_default_do_not_launch_processes(delivery, tmp_path):
    wrapper, info, _ = delivery
    base = tmp_path / 'installed'
    for label, extra in [('explicit', ['--prepare-only']), ('default', [])]:
        root = tmp_path / label
        root.mkdir()
        result = invoke(wrapper, '--install-base', base, '--root', root, '--python', PYTHON, *extra)
        assert result.returncode == 0, result.stdout + result.stderr
        output = root / 'runs/followup_20261024_r4'
        assert (output / 'plan.json').is_file()
        assert not (output / 'runtime/startup.json').exists()
        assert not (output / 'runtime/receipts').exists()


def test_installer_submit_smoke_uses_only_mocked_slurm(delivery, tmp_path):
    wrapper, info, _ = delivery
    fake_bin = tmp_path / 'fake-slurm'
    fake_bin.mkdir()
    receipts = tmp_path / 'fake-sbatch-arguments.txt'
    for name in ('squeue', 'sacct'):
        path = fake_bin / name
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    sbatch = fake_bin / 'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PAPER3_TEST_SBATCH_LOG"\n'
                      'count=$(wc -l < "$PAPER3_TEST_SBATCH_LOG")\nprintf "%s\\n" "$((700000 + count))"\n')
    sbatch.chmod(0o755)
    root = tmp_path / 'project'
    root.mkdir()
    credentials = tmp_path / 'no-token.json'
    credentials.write_text('{}')  # Missing token fails locally before any HTTP request.
    env = {**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ.get('PATH', ''),
           'PAPER3_TEST_SBATCH_LOG': str(receipts)}
    output = root / 'runs/followup_20261024_r4'
    pids = []
    try:
        result = invoke(wrapper, '--install-base', tmp_path / 'installed', '--root', root,
                        '--python', PYTHON, '--credentials', credentials,
                        '--gpus', '3', '--cpus', '16', '--poll-seconds', '1', '--submit', env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        startup = json.loads((output / 'runtime/startup.json').read_text())
        pids = [startup['monitor_pid'], startup['controller_pid']]
        prepared = next(json.loads(line) for line in result.stdout.splitlines()
                        if line.startswith('{') and json.loads(line).get('status') == 'PREPARED')
        assert prepared['slurm_jobs'] == 1 and prepared['gpu_slots'] == 3
        assert prepared['cpus_per_gpu'] == 16 and prepared['cpus_per_job'] == 48
        assert prepared['host_memory_per_job'] == '192G'
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if receipts.is_file() and len(receipts.read_text().splitlines()) == 1:
                break
            time.sleep(.05)
        arguments = receipts.read_text().splitlines()
        assert len(arguments) == 1
        for line in arguments:
            assert '--array' not in line
            assert '--nodes=1' in line
            assert '--ntasks=1' in line
            assert '--partition=hpc_gpu1' in line
            assert '--cpus-per-task=48' in line
            assert '--mem=192G' in line
            assert '--gres=gpu:3' in line
            assert '--nodelist' not in line
            assert '/paper3r3/' not in line
        assert len(list((output / 'runtime/receipts').glob('*.json'))) == 1
    finally:
        if (output / 'runtime').exists():
            (output / 'runtime/STOP').touch()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
