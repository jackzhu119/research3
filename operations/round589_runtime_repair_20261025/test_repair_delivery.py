"""Exercise deployment through the same isolated backend used by queued jobs."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

HERE=Path(__file__).resolve().parent


def load_template():
    spec=importlib.util.spec_from_file_location('repair_delivery_template',HERE/'repair_template.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=(HERE/'r58_runtime_hook.py').read_bytes()
    module.HOOK_B85=base64.b85encode(zlib.compress(raw)).decode()
    module.HOOK_SHA256=hashlib.sha256(raw).hexdigest()
    return module


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def fake_backend(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('''import json,os,runpy,sys
from pathlib import Path
def main():
    entry=Path(sys.argv[1]);plan_path=Path(sys.argv[2])
    plan=json.loads(plan_path.read_text())
    sys.argv=[str(entry),str(plan_path)]
    runpy.run_path(str(entry),run_name='__main__')
if __name__=='__main__':main()
''')


def fixture_tree(tmp_path):
    home=tmp_path/'home';root=home/'projects/paper3'
    wrappers={}
    package=home/'paper3-parallel-migrations/fixture/round56'
    for number,folder in load_template().OUTPUTS.items():
        operational=package if number in (5,6) else home/('paper3-round%d'%number)/'fixture/runtime'
        operational.parent.mkdir(parents=True,exist_ok=True)
        if number!=4:
            wrapper=operational/'backend_entry.py';fake_backend(wrapper);wrappers[number]=wrapper
        output=root/'runs'/folder
        write_json(output/'plan.json',{'output':str(output),'recipe_sha256':'recipe-'+str(number),'tasks':[{'key':'first'}]})
        write_json(output/'runtime/three_jobs/launch.json',{'worker_command':['python',str(operational/'three_jobs.py'),'worker'],'package':str(operational.parent)})
        write_json(output/'first/result.json',{'status':'COMPLETE','identity':'stable'})
        (output/'first/checkpoint.pt').write_bytes(b'original checkpoint\x00')
        (operational.parent/'native-source.py').write_bytes(b'scientific source remains unchanged\n')
    native=home/'paper3-round4-hpcgpu/fixture/paper3r4/entry.py'
    native.parent.mkdir(parents=True,exist_ok=True)
    native.write_bytes(Path('/workspace/paper3-round4-hpcgpu/paper3r4/entry.py').read_bytes())
    (native.parent/'worker.py').write_bytes(b'original hashed R4 worker\n')
    return home,root,wrappers


def frozen_files(home,root):
    selected=[]
    for path in root.glob('runs/*/plan.json'):selected.append(path)
    for path in root.glob('runs/*/first/*'):selected.append(path)
    for path in home.glob('paper3-*/fixture/native-source.py'):selected.append(path)
    for path in home.glob('paper3-round4-hpcgpu/*/paper3r4/worker.py'):selected.append(path)
    return {str(path):path.read_bytes() for path in selected}


def test_launch_discovery_covers_actual_shared_and_each_native_backend(tmp_path):
    home,root,backends=fixture_tree(tmp_path)
    found=load_template().wrappers(home,root)
    assert set(found)==set(backends.values())
    assert all(number in found[path] for number,path in backends.items())


def test_known_shallow_fallback_discovers_workers_before_launch_receipt(tmp_path):
    home,root,backends=fixture_tree(tmp_path)
    for path in root.glob('runs/*/runtime/three_jobs/launch.json'):path.unlink()
    found=load_template().wrappers(home,root)
    assert set(found)==set(backends.values())


def test_apply_is_idempotent_preserves_all_science_and_never_calls_slurm(tmp_path,monkeypatch):
    home,root,backends=fixture_tree(tmp_path)
    original=frozen_files(home,root)
    module=load_template()
    def forbidden(*args,**kwargs):raise AssertionError('No shell/Slurm operations authorized by deployment')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    monkeypatch.setattr(module.subprocess,'Popen',forbidden)
    assert module.main(['--home',str(home),'--root',str(root)])==0
    first={str(path):path.read_bytes() for path in backends.values()}
    assert module.main(['--home',str(home),'--root',str(root)])==0
    assert first=={str(path):path.read_bytes() for path in backends.values()}
    assert original==frozen_files(home,root)
    record=json.loads((home/'paper3-runtime-fixes/r589-dtype-cublas-v1/APPLIED.json').read_text())
    assert record['scientific_source_files_changed']==record['checkpoint_files_changed']==record['plan_files_changed']==0
    assert record['jobs_cancelled']==record['jobs_submitted']==0
    assert len(record['operational_files'])==len(set(backends.values()))+1


@pytest.mark.parametrize('number',[5,6,8,9])
def test_queued_backend_fresh_process_gets_hook_before_native_torch(tmp_path,number):
    home,root,backends=fixture_tree(tmp_path)
    module=load_template();module.main(['--home',str(home),'--root',str(root)])
    wrapper=backends[5 if number==6 else number]
    native=tmp_path/('paper3r%d'%number)/'entry.py';native.parent.mkdir()
    native.write_text('''import json,os,sys
assert 'torch' not in sys.modules
number=int(__file__.split('paper3r')[-1].split('/')[0])
finders=[f for f in sys.meta_path if hasattr(f,'paper3_runtime_correction')]
if number==6:
    assert not finders
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
else:
    assert len(finders)==1
    assert 'paper3r%d.worker'%number in finders[0].targets
if number==5:assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
print('FRESH_QUEUED_CHILD_OK')
''')
    env=dict(os.environ);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    plan=root/'runs'/module.OUTPUTS[5 if number==6 else number]/'plan.json'
    answer=subprocess.run([sys.executable,'-I','-S','-B','-u',str(wrapper),str(native),str(plan)],env=env,text=True,capture_output=True,timeout=15)
    assert answer.returncode==0,answer.stdout+answer.stderr
    assert 'FRESH_QUEUED_CHILD_OK' in answer.stdout


def test_native_r4_entry_discovery_and_idempotent_patch(tmp_path):
    home,root,_=fixture_tree(tmp_path);module=load_template()
    entry=home/'paper3-round4-hpcgpu/fixture/paper3r4/entry.py'
    assert entry in module.native_entries(home,root)
    original=entry.read_bytes();plan=root/'runs'/module.OUTPUTS[4]/'plan.json'
    before_plan=plan.read_bytes();before_worker=(entry.parent/'worker.py').read_bytes()
    assert module.main(['--home',str(home),'--root',str(root)])==0
    patched=entry.read_bytes()
    assert patched!=original
    assert module.patched_entry(patched,home/'paper3-runtime-fixes/r589-dtype-cublas-v1/runtime_hook.py')==patched
    assert before_plan==plan.read_bytes()
    assert before_worker==(entry.parent/'worker.py').read_bytes()


@pytest.mark.parametrize('mode',['worker','allocation'])
def test_actual_r4_native_entry_hook_is_pre_scientific_import(tmp_path,mode):
    home,root,_=fixture_tree(tmp_path);module=load_template()
    module.main(['--home',str(home),'--root',str(root)])
    entry=home/'paper3-round4-hpcgpu/fixture/paper3r4/entry.py'
    plan=root/'runs'/module.OUTPUTS[4]/'plan.json'
    runner=r'''
import importlib,runpy,sys,types
entry,mode,plan=sys.argv[1:]
real_import=importlib.import_module
def intercepted(name,*args,**kwargs):
    if name=='paper3r4.'+mode:
        assert 'torch' not in sys.modules
        finders=[f for f in sys.meta_path if hasattr(f,'paper3_runtime_correction')]
        if mode=='worker':
            assert len(finders)==1
            assert 'paper3r4.worker' in finders[0].targets
        else:assert not finders
        return types.SimpleNamespace(main=lambda: print('R4_NATIVE_ENTRY_PASS'))
    return real_import(name,*args,**kwargs)
importlib.import_module=intercepted
sys.argv=[entry,mode,'--plan',plan]
runpy.run_path(entry,run_name='__main__')
'''
    answer=subprocess.run([sys.executable,'-I','-S','-B','-u','-c',runner,str(entry),mode,str(plan)],text=True,capture_output=True,timeout=15)
    assert answer.returncode==0,answer.stdout+answer.stderr
    assert 'R4_NATIVE_ENTRY_PASS' in answer.stdout


def test_isolated_installer_uses_no_torch_and_no_extra_download(tmp_path):
    home,root,_=fixture_tree(tmp_path)
    module=load_template()
    raw=(HERE/'repair_template.py').read_text().replace('__HOOK_B85__',module.HOOK_B85).replace('__HOOK_SHA256__',module.HOOK_SHA256)
    launcher=tmp_path/'FIX.py';launcher.write_text(raw)
    answer=subprocess.run([sys.executable,'-I','-S','-B','-u',str(launcher),'--home',str(home),'--root',str(root)],text=True,capture_output=True,timeout=15)
    assert answer.returncode==0,answer.stdout+answer.stderr
    assert 'R589_REPAIR_APPLIED' in answer.stdout
    assert 'site-packages' not in answer.stderr


def test_atomic_write_preserves_mode_and_existing_read_handle(tmp_path):
    module=load_template();path=tmp_path/'worker.py';path.write_bytes(b'old');path.chmod(0o640)
    with path.open('rb') as old:
        module.atomic(path,b'new')
        assert old.read()==b'old'
    assert path.read_bytes()==b'new'
    assert path.stat().st_mode&0o777==0o640


def test_unrecognized_backend_rejected_without_source_write(tmp_path):
    module=load_template()
    with pytest.raises(ValueError,match='Unrecognized'):
        module.patched_backend(b'print("unrelated")\n',tmp_path/'hook.py')


def test_status_is_readonly(tmp_path):
    home,root,_=fixture_tree(tmp_path);before=frozen_files(home,root)
    assert load_template().main(['--home',str(home),'--root',str(root),'--status'])==0
    assert before==frozen_files(home,root)
    assert not (home/'paper3-runtime-fixes').exists()
