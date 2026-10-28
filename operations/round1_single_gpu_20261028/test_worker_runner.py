import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location('_worker_runner_test',HERE/'worker_runner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_parent_rewrites_only_native_child_script_preserves_flags_environment(tmp_path,monkeypatch):
    wrapper=load();package=tmp_path/'native';package.mkdir();calls=[]
    bridge=types.SimpleNamespace()
    projected=[]
    monkeypatch.setitem(sys.modules,'receipt_projection',types.SimpleNamespace(install=lambda b:projected.append(b)))
    retained_child=lambda command,**kw:calls.append((command,kw)) or 7
    worker=types.SimpleNamespace(_bridge=lambda p:bridge,run_child=retained_child,PROGRESS_KEYS=('status','outer_updates_completed'))
    wrapper.patch_parent(worker,package)
    command=['/existing/python','-B','-u',str(package/'worker.py'),'--child','--package',str(package),
             '--root','/original','--runtime-dir','/original/runtime']
    env={'CUDA_VISIBLE_DEVICES':'7','CUSTOM_ORIGINAL_ENV':'kept'}
    assert worker.run_child(command,env=env,cwd='/original')==7
    passed,kwargs=calls[0]
    assert passed[:3]==command[:3] and passed[3]==str(HERE/'worker_runner.py')
    assert passed[4:]==command[4:] and '-I' not in passed and '-S' not in passed
    assert kwargs['env'] is env and kwargs['cwd']=='/original'
    assert worker._bridge(package) is bridge
    assert worker._bridge(package) is bridge and projected==[bridge]


@pytest.mark.parametrize('command',[
    ['/python','/other/worker.py','--child'],
    ['/python','/native/worker.py'],
    ['/python','/native/worker.py','/native/worker.py','--child']])
def test_parent_refuses_unbound_child_command(tmp_path,monkeypatch,command):
    wrapper=load();monkeypatch.setitem(sys.modules,'receipt_projection',types.SimpleNamespace(install=lambda b:None))
    worker=types.SimpleNamespace(_bridge=lambda p:types.SimpleNamespace(),run_child=lambda *_a,**_k:pytest.fail('unsafe child launched'),
                                  PROGRESS_KEYS=())
    wrapper.patch_parent(worker,Path('/native'))
    with pytest.raises(ValueError,match='EXACT_NATIVE_ACTION_CHILD_COMMAND_REQUIRED'):
        worker.run_child(command)


def test_parent_telemetry_never_loads_large_scientific_receipt(tmp_path,monkeypatch):
    wrapper=load();monkeypatch.setitem(sys.modules,'receipt_projection',types.SimpleNamespace(install=lambda b:None))
    worker=types.SimpleNamespace(_bridge=lambda p:None,run_child=lambda *a,**k:None,PROGRESS_KEYS=('status','outer_updates_completed'))
    wrapper.patch_parent(worker,tmp_path)
    folder=tmp_path/'out';folder.mkdir()
    (folder/'receipt.json').write_bytes(b'x'*(3*1024*1024))
    (folder/'progress.json').write_text(json.dumps(dict(status='RUNNING',outer_updates_completed=22,labels='private')))
    result=worker.progress_for_action(tmp_path,dict(expected_receipt='out/receipt.json'))
    assert result==[dict(file='out/progress.json',values=dict(status='RUNNING',outer_updates_completed=22))]


def test_actual_scientific_child_does_not_import_management_projection(tmp_path):
    # A real isolated wrapper process loads a synthetic native module. It asserts
    # which patches reached science rather than duplicating the wrapper logic.
    package=tmp_path/'native';package.mkdir();home=tmp_path/'home';home.mkdir();root=tmp_path/'root';root.mkdir()
    saved=home/'paper3-r1-full-repair/2bfd664251ee5d1bcaf5';saved.mkdir(parents=True)
    (saved/'verified_context_cache.py').write_text('VERSION="retained"\n')
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'worker_runner.py').write_text((HERE/'worker_runner.py').read_text())
    (bundle/'finalization_cache.py').write_text('''
def install(startup,retained,root):
 assert retained.VERSION=='retained'
 startup.finalization_installed=True
''')
    (bundle/'receipt_projection.py').write_text('raise AssertionError("management projection imported in science")\n')
    (package/'action_startup.py').write_text('finalization_installed=False\n')
    (package/'worker.py').write_text('''
import json,sys,os
import action_startup
def main(argv):
 assert '--child' in argv
 assert 'receipt_projection' not in sys.modules
 assert action_startup.finalization_installed
 print(json.dumps({'scope':'science','isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'env':os.environ['ORIGINAL_SCIENCE_ENV']}))
 return 0
''')
    import os
    result=subprocess.run([sys.executable,'-B','-u',str(bundle/'worker_runner.py'),'--child',
        '--root',str(root),'--package',str(package)],capture_output=True,text=True,timeout=10,
        env=dict(os.environ,HOME=str(home),ORIGINAL_SCIENCE_ENV='kept'))
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==dict(scope='science',isolated=0,no_site=0,env='kept')
