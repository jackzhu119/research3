"""Actual old action-entry import order, without changing immutable releases."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
OLD_ENTRY = Path('/workspace/paper3-r1-single-gpu-20261028/delivery/START-PAPER3-R1-1GPU.py')
FIXED = Path('/workspace/paper3-r1-single-gpu-20261028/finalization_cache.py')
OLD_CACHE_SHA = '865c661075c5c7769c567b431cb70f7997b852f8e641ba3cccd803d2b57fffca'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hotfix():
    spec = importlib.util.spec_from_file_location('_live_hook_test_hotfix', HERE/'hotfix.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def package_hashes(package):
    return {p.name: sha(p.read_bytes()) for p in package.iterdir() if p.is_file()}


@pytest.fixture
def layout(tmp_path):
    assignments = {}
    for node in ast.parse(OLD_ENTRY.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    raw = base64.b85decode(assignments['PAYLOAD'])
    assert sha(raw) == assignments['PAYLOAD_SHA256']
    assert assignments['PAYLOAD_SHA256'].startswith('82bfccb2c75447584639')
    home = tmp_path/'home'
    package = home/'paper3-r1-single-gpu'/assignments['PAYLOAD_SHA256'][:20]
    package.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            assert '/' not in name and name not in ('.', '..')
            (package/name).write_bytes(archive.read(name))
    assert sha((package/'finalization_cache.py').read_bytes()) == OLD_CACHE_SHA
    root = home/'projects/paper3'; root.mkdir(parents=True)
    native = home/'native'; native.mkdir()
    startup = native/'action_startup.py'
    startup.write_text('from contextlib import contextmanager\n@contextmanager\ndef active(root, action, **kwargs):\n yield\n')
    fixed = home/'paper3-r1-finalization-hotfix'/sha(FIXED.read_bytes())[:20]/'finalization_cache.py'
    fixed.parent.mkdir(parents=True)
    fixed.write_bytes(FIXED.read_bytes())
    work = home/'hotfix-work'; work.mkdir()
    (native/'worker.py').write_text('''
import inspect,json,os,sys,time
from pathlib import Path
def main(argv=None):
 import action_startup, finalization_cache
 argv=list(argv or [])
 waiting=Path(argv[argv.index('--wait-until')+1]) if '--wait-until' in argv else None
 if waiting is not None:
  print('CHILD_LOADED',flush=True)
  until=time.monotonic()+10
  while not waiting.exists() and time.monotonic()<until:time.sleep(.01)
  if not waiting.exists():return 7
 value=dict(pid=os.getpid(),install_source=inspect.getsourcefile(finalization_cache.install),
            version=getattr(finalization_cache,'VERSION',None),
            startup_wrapped=hasattr(action_startup,'_paper3_r1_finalization_verification_reuse_v1'))
 print('CHILD_RESULT '+json.dumps(value),flush=True)
 return 0
''')
    # Retain the exact old worker_runner: only the fixture's retained helper
    # loader is injected, avoiding writes to the real user's environment.
    driver = tmp_path/'driver.py'
    driver.write_text('''
import importlib.util,sys,types
from pathlib import Path
package,native,root=sys.argv[1:4]
sys.path[:0]=[package,native]
spec=importlib.util.spec_from_file_location('_real_old_runner',Path(package)/'worker_runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
original_load=runner.load
def retained(name,path):
 if name=='_r1_single_gpu_retained_verified_cache':return types.SimpleNamespace()
 return original_load(name,path)
runner.load=retained
raise SystemExit(runner.main(['--child','--package',native,'--root',root,*sys.argv[4:]]))
''')
    return dict(home=home, root=root, package=package, native=native, startup=startup,
                fixed=fixed, work=work, driver=driver, before=package_hashes(package),
                original_startup=startup.read_bytes())


def command(layout, *extra):
    return [sys.executable, '-I', '-S', '-B', '-u', str(layout['driver']),
            str(layout['package']), str(layout['native']), str(layout['root']), *map(str, extra)]


def result_row(process):
    rows = [line[len('CHILD_RESULT '):] for line in process.stdout.splitlines()
            if line.startswith('CHILD_RESULT ')]
    return json.loads(rows[-1]) if rows else None


def install_hook(layout, *, old_module=None, old_sha=OLD_CACHE_SHA, new_sha=None):
    module = hotfix()
    hook = module.make_hook(old_module or layout['package']/'finalization_cache.py', old_sha,
                            layout['fixed'], new_sha or sha(layout['fixed'].read_bytes()))
    module.patch_startup(layout['startup'], hook, layout['work'])
    return module


def test_fresh_actual_old_worker_runner_uses_fixed_install_and_old_release_unchanged(layout):
    module = install_hook(layout)
    process = subprocess.run(command(layout), capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stdout + process.stderr
    row = result_row(process)
    assert row and Path(row['install_source']) == layout['fixed']
    assert row['startup_wrapped'] is True
    assert package_hashes(layout['package']) == layout['before']
    assert layout['startup'].read_bytes().startswith(layout['original_startup'])
    first = layout['startup'].read_bytes()
    install_hook(layout)
    assert layout['startup'].read_bytes() == first


def test_already_loaded_child_keeps_old_code_and_new_child_gets_patch(layout):
    release = layout['work']/'release-child'
    child = subprocess.Popen(command(layout, '--wait-until', release), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'CHILD_LOADED'
        install_hook(layout)
        assert child.poll() is None
        release.touch()
        stdout, stderr = child.communicate(timeout=10)
        old = result_row(subprocess.CompletedProcess([], child.returncode, stdout, stderr))
        assert child.returncode == 0 and Path(old['install_source']) == layout['package']/'finalization_cache.py'
        new = subprocess.run(command(layout), capture_output=True, text=True, timeout=10)
        assert new.returncode == 0 and Path(result_row(new)['install_source']) == layout['fixed']
        assert package_hashes(layout['package']) == layout['before']
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=3)


def test_old_immutable_bootstrap_check_still_passes_after_operational_hook(layout):
    install_hook(layout)
    process = subprocess.run([sys.executable, '-I', '-S', '-B', '-u', str(OLD_ENTRY),
                              '--check', '--home', str(layout['home']), '--root', str(layout['root'])],
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'EXISTING_SINGLE_GPU_RELEASE_DIFFERS' not in process.stdout + process.stderr
    assert package_hashes(layout['package']) == layout['before']


@pytest.mark.parametrize('wrong', ['old_path', 'old_sha', 'new_sha'])
def test_wrong_path_or_hash_never_claims_fixed_install(layout, wrong):
    kwargs = {'old_module': layout['package']/'wrong.py'} if wrong == 'old_path' else {
        'old_sha' if wrong == 'old_sha' else 'new_sha': '0'*64}
    install_hook(layout, **kwargs)
    process = subprocess.run(command(layout), capture_output=True, text=True, timeout=10)
    row = result_row(process)
    assert process.returncode != 0 or (row and Path(row['install_source']) != layout['fixed'])
    assert package_hashes(layout['package']) == layout['before']
