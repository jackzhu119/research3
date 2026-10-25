"""Regression checks for the CuBLAS split-worker startup failure.

These exercise each real entry path up to its first scientific/Torch import,
without substituting a fake successful CUDA training run.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE = Path('/workspace/paper3-round5-hpcgpu/paper3r5')
VALID = (':4096:8', ':16:8')


@pytest.mark.parametrize('incoming', [None, '', ':4096:8', ':16:8', 'invalid'])
@pytest.mark.parametrize('path', ['entry.py', 'worker.py'])
def test_real_startup_sets_workspace_before_scientific_import(incoming, path):
    source = PACKAGE / path
    expected = incoming if incoming in VALID else ':4096:8'
    setup = (
        "import os,sys,runpy,types,builtins,json; "
        f"incoming={incoming!r}; expected={expected!r}; "
        "os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None); "
        "os.environ.update({} if incoming is None else {'CUBLAS_WORKSPACE_CONFIG':incoming}); "
    )
    if path == 'entry.py':
        script = setup + (
            f"namespace=runpy.run_path({str(source)!r}); "
            "assert 'torch' not in sys.modules; "
            "def_import=namespace['importlib'].import_module; "
            "exec(\"def verify_import(name):\\n"
            " assert name == 'paper3r5.worker'\\n"
            " assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == expected\\n"
            " assert 'torch' not in sys.modules\\n"
            " return types.SimpleNamespace(main=lambda:17)\"); "
            "namespace['importlib'].import_module=verify_import; "
            "assert namespace['main'](['worker']) == 17; "
            "print(json.dumps({'verified':'entry-before-science-import','workspace':expected}))"
        )
    else:
        script = setup + (
            "sys.modules['numpy']=types.ModuleType('numpy'); original=builtins.__import__; "
            "exec(\"def verify_import(name,*args,**kwargs):\\n"
            " if name == 'torch':\\n"
            "  assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == expected\\n"
            "  print(json.dumps({'verified':'worker-before-torch-import','workspace':expected}))\\n"
            "  raise SystemExit(0)\\n"
            " return original(name,*args,**kwargs)\"); "
            "builtins.__import__=verify_import; "
            f"runpy.run_path({str(source)!r}); "
            "raise AssertionError('Torch import boundary was not reached')"
        )
    completed = subprocess.run([sys.executable, '-I', '-S', '-c', script],
                               text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result['workspace'] == expected


def test_modified_startup_sources_compile():
    for name in ('entry.py', 'worker.py'):
        source = PACKAGE / name
        compile(source.read_bytes(), str(source), 'exec')


@pytest.mark.parametrize('round_number', [5, 6])
@pytest.mark.parametrize('incoming', [None, 'invalid', ':4096:8', ':16:8'])
def test_operational_hot_patch_is_scoped_to_round5(tmp_path, round_number, incoming):
    wrapper = Path(__file__).with_name('backend_entry_r5_patch.py')
    package = tmp_path / f'paper3r{round_number}'
    package.mkdir()
    entry = package / 'entry.py'
    entry.write_text("import os,json; print(json.dumps({'effective':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))")
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'root': str(tmp_path / 'root')}))
    import os
    environment = dict(os.environ)
    environment.pop('CUBLAS_WORKSPACE_CONFIG', None)
    if incoming is not None:
        environment['CUBLAS_WORKSPACE_CONFIG'] = incoming
    completed = subprocess.run([sys.executable, '-I', '-S', str(wrapper),
        '--original-entry', str(entry), 'worker', '--plan', str(plan)],
        env=environment, text=True, capture_output=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    effective = json.loads(completed.stdout.splitlines()[-1])['effective']
    expected = incoming
    if round_number == 5 and incoming not in VALID:
        expected = ':4096:8'
    assert effective == expected
