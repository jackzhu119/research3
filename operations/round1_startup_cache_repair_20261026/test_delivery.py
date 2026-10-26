"""Deployment preserves R1's runtime/spawn corrections and native validators."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

HERE=Path(__file__).resolve().parent
OLD=Path('/workspace/paper3-r1-hardware-fix')
BOOT=Path('/workspace/paper3-future-audit-20261025/r1/r1_spawn_bootstrap.py')


def module():
    spec=importlib.util.spec_from_file_location('r1_cache_delivery',HERE/'delivery_template.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    helper=(HERE/'r1_digest_cache.py').read_bytes()
    result.HOOK_B85=base64.b85encode(zlib.compress(helper,9)).decode()
    result.HOOK_SHA256=hashlib.sha256(helper).hexdigest()
    return result


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def fixture(tmp_path):
    home=tmp_path/'home';root=home/'projects/paper3'
    package=home/'paper3-r1-repairs/original/active-round1/c36635429230c732b802'
    package.mkdir(parents=True)
    for name in ('action_startup.py','kernel_bridge.py','audited_runtime.py','hardware_comparison.py'):
        (package/name).write_bytes((OLD/name).read_bytes())
    for name in ('worker.py','scheduler.py','START.py'):
        (package/name).write_text('# original '+name+'\n')
    (package/'action_startup.py').chmod(0o640)
    (package/'r1_spawn_bootstrap.py').write_bytes(BOOT.read_bytes())
    direct=home/'paper3-direct-train/656cb8e704b87c4e'
    direct.mkdir(parents=True);(direct/'r1_spawn_bootstrap.py').write_bytes(BOOT.read_bytes())
    (direct/'campaign_recovery.py').write_bytes(b'original current spawn/F0 recovery unchanged\n')
    runtime=root/'runs/training_campaign_parallel_v1'
    save(runtime/'launch.json',{'package':str(package),'direct_package':str(direct),
        'worker_command':[sys.executable,str(package/'worker.py'),'--package',str(package)]})
    save(runtime/'scheduler-state.json',{'slots':{'0':{'attempts':[{'command':['python','worker.py','--package',str(package)]}]}}})
    checkpoint=root/'runs/meta/completed/checkpoint.pt';checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'original neural checkpoint\x00')
    other=home/'paper3-runtime-fixes/r9-cumsum-v2/r9_runtime_hook.py'
    other.parent.mkdir(parents=True);other.write_bytes(b'other rounds unchanged\n')
    return home,root,runtime,package,direct,checkpoint,other


def frozen(root,runtime,package,direct,checkpoint,other):
    paths=[p for p in package.glob('*.py') if p.name not in ('action_startup.py','r1_spawn_bootstrap.py')]
    paths.extend([direct/'campaign_recovery.py',runtime/'launch.json',runtime/'scheduler-state.json',checkpoint,other])
    return {str(p):p.read_bytes() for p in paths}


def test_active_standalone_launch_and_prior_scheduler_packages_are_discovered(tmp_path):
    home,root,runtime,package,_,_,_=fixture(tmp_path)
    older=home/'paper3-parallel-migrations/fa00/round1';older.mkdir(parents=True)
    for name in ('kernel_bridge.py','worker.py','action_startup.py'):
        (older/name).write_bytes((package/name).read_bytes())
    save(runtime/'scheduler-state.json',{'slots':{'0':{'attempts':[{'command':['python','worker.py','--package',str(older)]}]}}})
    assert module().packages(runtime,home)==sorted([package,older])


def test_shallow_fallback_covers_standalone_when_launch_absent(tmp_path):
    home,root,runtime,package,_,_,_=fixture(tmp_path)
    (runtime/'launch.json').unlink();(runtime/'scheduler-state.json').unlink()
    assert module().packages(runtime,home)==[package]


def test_apply_idempotent_preserves_plan_checkpoint_runtime_and_both_bootstrap_bodies(tmp_path):
    home,root,runtime,package,direct,checkpoint,other=fixture(tmp_path)
    before=frozen(root,runtime,package,direct,checkpoint,other)
    startup=(package/'action_startup.py').read_bytes();bootstrap=(package/'r1_spawn_bootstrap.py').read_bytes()
    delivery=module()
    assert delivery.main(['--home',str(home),'--root',str(root)])==0
    after=[(package/'action_startup.py').read_bytes(),(package/'r1_spawn_bootstrap.py').read_bytes(),
           (direct/'r1_spawn_bootstrap.py').read_bytes()]
    assert after[0].startswith(startup);assert after[1].startswith(bootstrap)
    assert after[2].startswith(bootstrap)
    assert delivery.main(['--home',str(home),'--root',str(root)])==0
    assert after==[(package/'action_startup.py').read_bytes(),(package/'r1_spawn_bootstrap.py').read_bytes(),
                   (direct/'r1_spawn_bootstrap.py').read_bytes()]
    assert all(Path(path).read_bytes()==data for path,data in before.items())
    assert (package/'action_startup.py').stat().st_mode & 0o777==0o640
    receipt=json.loads((home/'paper3-runtime-fixes/r1-digest-cache-v1/APPLIED.json').read_text())
    assert receipt['owned_stage_bootstrap_files']==2
    assert receipt['scientific_files_changed']==receipt['checkpoint_files_changed']==receipt['plan_files_changed']==0
    assert receipt['signals_sent']==receipt['jobs_submitted']==receipt['jobs_cancelled']==0


def test_all_entries_prevalidated_before_any_module_switch(tmp_path):
    home,root,runtime,package,direct,checkpoint,other=fixture(tmp_path)
    original=(package/'action_startup.py').read_bytes()
    bad=direct/'r1_spawn_bootstrap.py';bad.write_bytes(bad.read_bytes()+b'\n# unknown lineage\n')
    with pytest.raises(ValueError,match='Unrecognized R1 operational spawn'):
        module().main(['--home',str(home),'--root',str(root)])
    assert (package/'action_startup.py').read_bytes()==original
    assert not (home/'paper3-runtime-fixes/r1-digest-cache-v1/r1_digest_cache.py').exists()


def make_fake_native_hash_modules(root):
    for package in ('p3work','tools'):
        folder=root/package;folder.mkdir(exist_ok=True);(folder/'__init__.py').write_text('')
    (root/'p3work/contracts.py').write_text('''import hashlib
calls=0
def file_digest(path):
 global calls
 calls+=1
 return hashlib.sha256(open(path,"rb").read()).hexdigest()
''')
    (root/'tools/r2_backup.py').write_text('''import hashlib,pathlib
calls=0
def sha256_file(path):
 global calls
 calls+=1
 value=pathlib.Path(path).read_bytes()
 return hashlib.sha256(value).hexdigest(),len(value)
''')


def test_real_patched_startup_keeps_gpu_probe_validators_and_reuses_actual_hashes(tmp_path):
    home,root,runtime,package,direct,checkpoint,other=fixture(tmp_path)
    make_fake_native_hash_modules(root)
    (root/'frozen-a.bin').write_bytes(b'frozen a');(root/'frozen-b.bin').write_bytes(b'frozen b')
    module().main(['--home',str(home),'--root',str(root)])
    code='''import hashlib,importlib.util,pathlib,sys,types
root,operation=sys.argv[1:];root=pathlib.Path(root)
sys.path[:0]=[str(root),operation]
spec=importlib.util.spec_from_file_location("active_startup",pathlib.Path(operation)/"action_startup.py")
startup=importlib.util.module_from_spec(spec);spec.loader.exec_module(startup)
from p3work import contracts
from tools import r2_backup
probes=[];validators=[]
properties=types.SimpleNamespace(name="NVIDIA A100-SXM4-40GB",uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",total_memory=42298834944,major=8,minor=0)
cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda n:probes.append(n),get_device_properties=lambda n:properties)
torch=types.SimpleNamespace(cuda=cuda,__version__="2.5.1+cu121",version=types.SimpleNamespace(cuda="12.1"))
def inputs(project,*,job,source_receipt,selection_receipt=None):
 validators.append("inputs")
 return contracts.file_digest(root/"frozen-a.bin")
def engineering(project,descriptor,*,job,inputs,runtime_profile=None):
 validators.append("engineering")
 return r2_backup.sha256_file(root/"frozen-b.bin")
meta=types.SimpleNamespace(verify_meta_inputs=inputs);eng=types.SimpleNamespace(verify_meta_engineering=engineering)
for index in range(2):
 with startup.active(root,{"kind":"meta_screening","key":"action-"+str(index)},modules=(meta,eng),torch_module=torch,traceback_interval=0):
  first=meta.verify_meta_inputs(root,job={"method":"P"},source_receipt={})
  assert meta.verify_meta_inputs(root,job={"method":"P"},source_receipt={})==first
  eng.verify_meta_engineering(root,{"descriptor":"same"},job={},inputs={})
assert probes==[0,0]
assert validators==["inputs","engineering","inputs","engineering"]
assert contracts.calls==r2_backup.calls==1
assert meta.verify_meta_inputs is inputs and eng.verify_meta_engineering is engineering
print("NATIVE_VALIDATORS_GPU_PROBES_AND_PERSISTENT_HASH_REUSE_PASS")
'''
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',code,str(root),str(package)],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'NATIVE_VALIDATORS_GPU_PROBES_AND_PERSISTENT_HASH_REUSE_PASS' in result.stdout
    assert 'R1_FILE_DIGEST_CACHE_SUMMARY' in result.stdout


@pytest.mark.parametrize('mode',['return','raise','not-applicable','cache-error'])
def test_owned_bootstrap_preserves_original_result_errors_and_f0_body(tmp_path,mode):
    home,root,runtime,package,direct,checkpoint,other=fixture(tmp_path)
    make_fake_native_hash_modules(root)
    module().main(['--home',str(home),'--root',str(root)])
    code='''import importlib.util,pathlib,sys
root,bootstrap,mode=sys.argv[1:];root=pathlib.Path(root);sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location("native_bootstrap",bootstrap)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
calls=[];value={"status":"R1_OWNED_CHILD_RUNTIME_BOOTSTRAPPED","source_f0_stage_lineage_installed":True}
def original(root):
 calls.append(str(root))
 if mode=="raise":raise RuntimeError("original F0/runtime failure")
 if mode=="not-applicable":return {"status":"NOT_APPLICABLE_NO_PARALLEL_LAUNCH"}
 return value
module._paper3_original_bootstrap_owned_stage=original
if mode=="cache-error":
 import sys
 sys.modules["p3work.contracts"]=None
if mode=="raise":
 try:module.bootstrap_owned_stage(root)
 except RuntimeError as error:assert str(error)=="original F0/runtime failure"
 else:raise AssertionError("Original exception must survive")
elif mode=="not-applicable":assert module.bootstrap_owned_stage(root)["status"]=="NOT_APPLICABLE_NO_PARALLEL_LAUNCH"
else:assert module.bootstrap_owned_stage(root) is value
assert calls==[str(root)]
print("ORIGINAL_BOOTSTRAP_RESULT_AND_LINEAGE_PRESERVED")
'''
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',code,str(root),
        str(direct/'r1_spawn_bootstrap.py'),mode],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ORIGINAL_BOOTSTRAP_RESULT_AND_LINEAGE_PRESERVED' in result.stdout


def test_frozen_installer_does_not_import_compute_network_or_job_modules(tmp_path):
    home,root,runtime,package,direct,checkpoint,other=fixture(tmp_path)
    before=frozen(root,runtime,package,direct,checkpoint,other)
    delivery=module();source=(HERE/'delivery_template.py').read_text()
    source=source.replace('__HOOK_B85__',delivery.HOOK_B85).replace('__HOOK_SHA256__',delivery.HOOK_SHA256)
    script=tmp_path/'FIX-R1-CACHE.py';script.write_text(source)
    code='''import runpy,sys
class Forbidden:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","numpy","subprocess","requests","socket") or fullname in ("urllib.request","http.client"):
   raise AssertionError("No compute, network or job execution import: "+fullname)
sys.meta_path.insert(0,Forbidden())
sys.argv=[sys.argv[1],"--home",sys.argv[2],"--root",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    result=subprocess.run([sys.executable,'-I','-S','-u','-c',code,str(script),str(home),str(root)],
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'R1_STARTUP_CACHE_REPAIR_APPLIED' in result.stdout
    assert all(Path(path).read_bytes()==data for path,data in before.items())
