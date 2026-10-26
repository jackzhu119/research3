"""Exercise the same operational file used by current and queued R9 workers."""
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
NATIVE=Path('/workspace/paper3-round9')


def load_template():
    spec=importlib.util.spec_from_file_location('r9_cumsum_delivery',HERE/'delivery_template.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    hook=(HERE/'r9_runtime_hook.py').read_bytes()
    module.HOOK_B85=base64.b85encode(zlib.compress(hook,9)).decode()
    module.HOOK_SHA256=hashlib.sha256(hook).hexdigest()
    return module


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(tmp_path):
    home=tmp_path/'home';root=home/'projects/paper3'
    package=home/'paper3-round9-thin/172fe6bcacc3c2df8c40'
    backend=package/'runtime/backend_entry.py';backend.parent.mkdir(parents=True)
    backend.write_bytes((NATIVE/'runtime/backend_entry.py').read_bytes());backend.chmod(0o640)
    science=package/'paper3r9';science.mkdir()
    for name in ('entry.py','worker.py','gp.py','plan.py'):
        (science/name).write_bytes((NATIVE/'paper3r9'/name).read_bytes())
    output=root/'runs/followup_20261025_r9'
    plan={'root':str(root),'output':str(output),'recipe_sha256':'original-r9-recipe',
          'tasks':[{'key':'feature-reference-raw-seed20261018'}]}
    save(output/'plan.json',plan)
    save(output/'runtime/three_jobs/launch.json',{
        'package':str(package),'plan':str(output/'plan.json'),'partition':'hpc_gpu',
        'cpus':16,'memories':['64G']*3,
        'worker_command':[sys.executable,str(package/'runtime/three_jobs.py'),'worker']})
    task=output/plan['tasks'][0]['key'];task.mkdir(parents=True)
    save(task/'result.json',{'status':'COMPLETE','identity':'already-complete'})
    (task/'checkpoint.pt').write_bytes(b'original checkpoint state\x00')
    other=home/'paper3-parallel-migrations/original/round56/backend_entry.py'
    other.parent.mkdir(parents=True);other.write_bytes(b'other rounds must not change\n')
    shared=home/'paper3-runtime-fixes/r589-dtype-cublas-v1/runtime_hook.py'
    shared.parent.mkdir(parents=True);shared.write_bytes(b'unchanged v1 shared hook\n')
    return home,root,package,backend,shared,other


def protected_bytes(home,root,package,shared,other):
    paths=list((package/'paper3r9').glob('*.py'))
    output=root/'runs/followup_20261025_r9'
    paths.extend([output/'plan.json',output/'runtime/three_jobs/launch.json',shared,other])
    paths.extend((output/'feature-reference-raw-seed20261018').iterdir())
    return {str(p):p.read_bytes() for p in paths}


def old_v1_backend(original,shared):
    spec=importlib.util.spec_from_file_location('v1_delivery',Path('/workspace/paper3-r5-r8-repair/repair_template.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.patched_backend(original,shared)


def test_actual_launch_discovers_thin_package_without_global_search(tmp_path):
    home,root,package,backend,_,_=fixture(tmp_path)
    module=load_template()
    assert module.wrappers(home,root)==[backend]
    (root/'runs/followup_20261025_r9/runtime/three_jobs/launch.json').unlink()
    assert module.wrappers(home,root)==[backend]


def test_regular_round9_package_and_launch_only_custom_location_are_discovered(tmp_path):
    home,root,_,backend,_,_=fixture(tmp_path)
    regular=home/'paper3-round9/another/runtime/backend_entry.py'
    regular.parent.mkdir(parents=True);regular.write_bytes(backend.read_bytes())
    custom=tmp_path/'saved-launch-only/runtime/backend_entry.py'
    custom.parent.mkdir(parents=True);custom.write_bytes(backend.read_bytes())
    save(root/'runs/followup_20261025_r9/runtime/launch.json',{'package':str(custom.parent.parent)})
    assert load_template().wrappers(home,root)==sorted([backend,regular,custom])


@pytest.mark.parametrize('with_v1',[False,True])
def test_apply_idempotent_preserves_science_plan_checkpoint_and_shared_hook(tmp_path,with_v1):
    home,root,package,backend,shared,other=fixture(tmp_path)
    if with_v1:backend.write_bytes(old_v1_backend(backend.read_bytes(),shared))
    original=protected_bytes(home,root,package,shared,other)
    module=load_template()
    assert module.main(['--home',str(home),'--root',str(root)])==0
    first=backend.read_bytes()
    assert module.NEW_MARKER.encode() in first
    assert module.OLD_MARKER.encode() not in first
    assert module.main(['--home',str(home),'--root',str(root)])==0
    assert backend.read_bytes()==first
    assert all(Path(path).read_bytes()==data for path,data in original.items())
    assert backend.stat().st_mode & 0o777==0o640
    receipt=json.loads((home/'paper3-runtime-fixes/r9-cumsum-v2/APPLIED.json').read_text())
    assert receipt['native_scientific_sources']
    assert receipt['plan_sha256']==hashlib.sha256((root/'runs/followup_20261025_r9/plan.json').read_bytes()).hexdigest()
    assert receipt['scientific_files_changed']==receipt['checkpoint_files_changed']==receipt['plan_files_changed']==0
    assert receipt['jobs_submitted']==receipt['jobs_cancelled']==receipt['other_rounds_changed']==0
    assert receipt['old_shared_hook_changed'] is False


def test_prevalidates_every_backend_before_any_switch(tmp_path):
    home,root,package,backend,shared,other=fixture(tmp_path)
    original=backend.read_bytes()
    bad=home/'paper3-round9/unknown/runtime/backend_entry.py'
    bad.parent.mkdir(parents=True);bad.write_bytes(original+b'\n# unreviewed native backend\n')
    with pytest.raises(ValueError,match='Unrecognized Round9'):
        load_template().main(['--home',str(home),'--root',str(root)])
    assert backend.read_bytes()==original
    assert not (home/'paper3-runtime-fixes/r9-cumsum-v2/r9_runtime_hook.py').exists()
    assert not (root/'runs/followup_20261025_r9/runtime/r9-cumsum-corrections.json').exists()


@pytest.mark.parametrize('kind',['old-incomplete','new-incomplete','duplicate','wrong-round'])
def test_unknown_or_broken_backend_is_rejected_without_science_fallback(tmp_path,kind):
    module=load_template();native=(NATIVE/'runtime/backend_entry.py').read_bytes()
    if kind=='old-incomplete':raw=native.replace(b'    cache=verified_cache_paths',b'    # PAPER3_R589_RUNTIME_FIX_V1\n    cache=verified_cache_paths')
    elif kind=='new-incomplete':raw=native.replace(b'    cache=verified_cache_paths',b'    # PAPER3_R9_CUMSUM_RUNTIME_FIX_V2\n    cache=verified_cache_paths')
    elif kind=='duplicate':
        raw=old_v1_backend(native,tmp_path/'old.py')
        raw=raw.replace(b'    # END_PAPER3_R589_RUNTIME_FIX_V1\n',b'    # END_PAPER3_R589_RUNTIME_FIX_V1\n    # END_PAPER3_R589_RUNTIME_FIX_V1\n')
    else:raw=native.replace(b'paper3r9',b'paper3r8')
    with pytest.raises(ValueError):module.patched_backend(raw,tmp_path/'new.py')


def test_current_or_queued_fresh_backend_installs_new_hook_before_torch(tmp_path):
    home,root,package,backend,shared,other=fixture(tmp_path)
    backend.write_bytes(old_v1_backend(backend.read_bytes(),shared))
    # The real backend launches this minimal entry to prove its startup ordering;
    # mathematical hook behavior is covered separately by real GP/worker tests.
    entry=package/'paper3r9/entry.py'
    entry.write_text('''import json,os,sys
assert "torch" not in sys.modules
finders=[f for f in sys.meta_path if getattr(f,"paper3_runtime_correction",None)=="paper3-r9-cumsum-fix-v2"]
assert len(finders)==1
assert set(finders[0].targets)=={"paper3r9.worker","paper3r9.gp"}
assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8",":16:8")
print("NEW_R9_HOOK_ACTIVE_BEFORE_TORCH")
''')
    load_template().main(['--home',str(home),'--root',str(root)])
    result=subprocess.run([sys.executable,'-I','-S','-B','-u',str(backend),
        '--original-entry',str(entry),'worker','--plan',str(root/'runs/followup_20261025_r9/plan.json')],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'R9_DETERMINISTIC_CUMSUM_READY' in result.stdout
    assert 'NEW_R9_HOOK_ACTIVE_BEFORE_TORCH' in result.stdout
    assert shared.read_bytes()==b'unchanged v1 shared hook\n'


def test_atomic_replace_preserves_old_open_reader_and_file_mode(tmp_path):
    module=load_template();path=tmp_path/'file';path.write_bytes(b'old bytes');path.chmod(0o640)
    with path.open('rb') as stream:
        module.atomic(path,b'new bytes')
        assert stream.read()==b'old bytes'
    assert path.read_bytes()==b'new bytes'
    assert path.stat().st_mode & 0o777==0o640


def test_payload_hash_failure_does_not_switch_any_entry(tmp_path):
    home,root,_,backend,_,_=fixture(tmp_path);original=backend.read_bytes()
    module=load_template();module.HOOK_SHA256='0'*64
    with pytest.raises(ValueError,match='SHA differs'):
        module.main(['--home',str(home),'--root',str(root)])
    assert backend.read_bytes()==original


def test_frozen_delivery_runs_isolated_without_torch_network_or_slurm(tmp_path):
    home,root,_,backend,_,_=fixture(tmp_path)
    template=(HERE/'delivery_template.py').read_text()
    module=load_template()
    frozen=template.replace('__HOOK_B85__',module.HOOK_B85).replace('__HOOK_SHA256__',module.HOOK_SHA256)
    target=tmp_path/'FIX-R9.py';target.write_text(frozen)
    code='''import runpy,sys
class Forbidden:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","numpy","subprocess","requests","socket") or fullname in ("urllib.request","http.client"):
   raise AssertionError("Repair must not import compute/network/job command modules: "+fullname)
sys.meta_path.insert(0,Forbidden())
sys.argv=[sys.argv[1],"--home",sys.argv[2],"--root",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',code,str(target),str(home),str(root)],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'R9_CUMSUM_REPAIR_APPLIED' in result.stdout
    assert load_template().NEW_MARKER.encode() in backend.read_bytes()


def test_status_is_readonly_and_does_not_require_embedded_payload(tmp_path):
    home,root,package,backend,shared,other=fixture(tmp_path)
    original=protected_bytes(home,root,package,shared,other);before=backend.read_bytes()
    spec=importlib.util.spec_from_file_location('unbuilt_template',HERE/'delivery_template.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.main(['--home',str(home),'--root',str(root),'--status'])==0
    assert backend.read_bytes()==before
    assert all(Path(path).read_bytes()==data for path,data in original.items())
