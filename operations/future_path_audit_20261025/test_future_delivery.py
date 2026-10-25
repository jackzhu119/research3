"""Exercise the future repair through the same isolated school entry contract."""
import ast
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

HERE = Path(__file__).resolve().parent
ORIGINAL_RECOVERY = Path('/workspace/paper3-direct-train/campaign_recovery.py')


def payload():
    files = {
        'r1_spawn_bootstrap.py': (HERE/'r1/r1_spawn_bootstrap.py').read_bytes(),
        'resume_existing.py': (HERE/'runtime/resume_existing.py').read_bytes(),
    }
    raw = json.dumps({k: base64.b64encode(v).decode() for k,v in files.items()},
                     sort_keys=True,separators=(',',':')).encode()
    return base64.b85encode(zlib.compress(raw,9)).decode(), hashlib.sha256(raw).hexdigest()


def module():
    spec = importlib.util.spec_from_file_location('future_fix_test', HERE/'future_fix_template.py')
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    result.PAYLOAD,result.PAYLOAD_SHA256 = payload()
    return result


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


def fixture(tmp_path):
    home=tmp_path/'home';root=home/'projects/paper3'
    active=home/'paper3-r1-repairs/current/active-round1/current'
    direct=home/'paper3-direct-train/current'
    launch=root/'runs/training_campaign_parallel_v1/launch.json'
    value={'package':str(active),'direct_package':str(direct),
           'payload_manifest':str(home/'paper3-runtime-repairs/current/payload_manifest.json'),
           'partition':'hpc_gpu','cpus':12,'memories':['43G','43G','42G'],
           'allocation_seconds':442800}
    save(launch,(json.dumps(value,indent=3)+'\n').encode())
    save(direct/'campaign_recovery.py',ORIGINAL_RECOVERY.read_bytes())
    save(active/'kernel_bridge.py',b'# existing operational bridge\n')
    save(root/'p3work/source_extension_downstream.py',b'# frozen numerical implementation\n')
    save(root/'runs/meta/existing/checkpoint.pt',b'existing optimizer RNG checkpoint\x00')
    save(root/'runs/meta/existing/receipt.json',b'{"status":"META_JOB_COMPLETE"}\n')
    save(root/'runs/followup_20261025_r8/plan.json',b'{"recipe_sha256":"frozen","tasks":[]}\n')
    save(root/'runs/followup_20261024_r5/first/result.json',b'{"status":"COMPLETE"}\n')
    save(home/'paper3-runtime-fixes/r589-dtype-cublas-v1/runtime_hook.py',b'# installed hook42b16\n')
    save(home/'paper3-direct-train/inactive/campaign_recovery.py',ORIGINAL_RECOVERY.read_bytes())
    frozen={p:p.read_bytes() for p in home.rglob('*') if p.is_file() and p!=direct/'campaign_recovery.py'}
    return home,root,active,direct,launch,frozen


def assert_frozen(frozen):
    assert all(path.read_bytes()==raw for path,raw in frozen.items())


def test_only_recorded_direct_package_is_patched_and_all_evidence_preserved(tmp_path):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    original=(direct/'campaign_recovery.py').read_bytes()
    folder=module().apply(root,home)
    assert_frozen(frozen)
    fixed=(direct/'campaign_recovery.py').read_text()
    assert fixed.count('    bootstrap_owned_stage(root)\n')==2
    assert (active/'r1_spawn_bootstrap.py').read_bytes()==(HERE/'r1/r1_spawn_bootstrap.py').read_bytes()
    assert (direct/'r1_spawn_bootstrap.py').read_bytes()==(HERE/'r1/r1_spawn_bootstrap.py').read_bytes()
    assert list((folder/'backups').glob('campaign_recovery.*.py'))[0].read_bytes()==original
    record=json.loads((folder/'APPLIED.json').read_text())
    assert record['r1']['changed_entrypoints']==['stage_worker','extension_stage_worker']
    assert record['r1']['launch_preserved'] is True
    assert record['existing_r45689_runtime_corrections_replaced'] is False
    assert record['plans_changed']==record['checkpoints_changed']==0
    assert record['slurm_jobs_cancelled']==record['slurm_jobs_submitted']==0


def test_repeat_apply_is_idempotent_and_keeps_original_spawn_signatures(tmp_path):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    mod=module();original=ast.parse((direct/'campaign_recovery.py').read_text())
    mod.apply(root,home);first=(direct/'campaign_recovery.py').read_bytes()
    mod.apply(root,home);assert (direct/'campaign_recovery.py').read_bytes()==first
    after=ast.parse(first)
    wanted={'stage_worker','extension_stage_worker'}
    before_args={n.name:ast.dump(n.args) for n in original.body if isinstance(n,ast.FunctionDef) and n.name in wanted}
    after_args={n.name:ast.dump(n.args) for n in after.body if isinstance(n,ast.FunctionDef) and n.name in wanted}
    assert before_args==after_args
    assert_frozen(frozen)
    record=json.loads((home/'paper3-runtime-fixes/future-path-v2/APPLIED.json').read_text())
    assert record['r1']['changed_entrypoints']==[]


def test_default_install_never_starts_any_process_or_slurm_command(tmp_path,monkeypatch):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    def forbidden(*a,**k):raise AssertionError('Default install must not start or stop processes')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden)
    monkeypatch.setattr(os,'system',forbidden)
    assert module().main(['--root',str(root),'--home',str(home)])==0
    assert_frozen(frozen)


def test_no_r1_install_is_optional_and_recovery_helper_still_installed(tmp_path):
    home=tmp_path/'home';root=tmp_path/'project';root.mkdir()
    folder=module().apply(root,home)
    assert (folder/'resume_existing.py').is_file()
    record=json.loads((folder/'APPLIED.json').read_text())
    assert record['r1']['status']=='NO_EXISTING_R1_PARALLEL_LAUNCH'
    assert not (root/'runs').exists()


def test_unknown_recovery_source_is_preserved_instead_of_guessed(tmp_path):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    target=direct/'campaign_recovery.py';target.write_text('# unrelated operational implementation\n')
    before=target.read_bytes()
    with pytest.raises(ValueError,match='Unexpected original R1 owned-stage'):
        module().apply(root,home)
    assert target.read_bytes()==before
    assert_frozen(frozen)


def test_bad_embedded_payload_fails_before_touching_deployment(tmp_path):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    mod=module();mod.PAYLOAD_SHA256='0'*64
    before=(direct/'campaign_recovery.py').read_bytes()
    with pytest.raises(ValueError,match='Embedded future repair'):
        mod.apply(root,home)
    assert (direct/'campaign_recovery.py').read_bytes()==before
    assert_frozen(frozen)


def frozen_entry(tmp_path):
    encoded,digest=payload()
    text=(HERE/'future_fix_template.py').read_text().replace('__PAYLOAD_SHA256__',digest).replace('__PAYLOAD__',encoded)
    path=tmp_path/'FIX-PAPER3-FUTURE.py';path.write_text(text)
    return path


def test_real_isolated_stdlib_install_and_importable_spawn_targets(tmp_path):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    entry=frozen_entry(tmp_path)
    script='''import runpy,sys,pickle
path,home,root,direct=sys.argv[1:]
m=runpy.run_path(path,run_name="_future_fix")
assert m["main"](["--root",root,"--home",home])==0
assert "torch" not in sys.modules and "numpy" not in sys.modules
sys.path.insert(0,direct)
import campaign_recovery as recovery
for name in ("stage_worker","extension_stage_worker"):
 fn=getattr(recovery,name)
 assert pickle.loads(pickle.dumps(fn)) is fn
assert "torch" not in sys.modules
print("ISOLATED_NO_GPU_IMPORT_SPAWN_TARGETS_PICKLABLE")
'''
    result=subprocess.run([sys.executable,'-I','-S','-B','-u','-c',script,str(entry),str(home),str(root),str(direct)],
        capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ISOLATED_NO_GPU_IMPORT_SPAWN_TARGETS_PICKLABLE' in result.stdout
    assert_frozen(frozen)


def test_optional_resume_is_explicit_and_deduplicates_rounds(tmp_path,monkeypatch):
    home,root,active,direct,launch,frozen=fixture(tmp_path)
    mod=module();native=mod.runpy.run_path;calls=[]
    def run(path,**kwargs):
        if Path(path).name=='resume_existing.py':return {'main':lambda args:calls.append(args)}
        return native(path,**kwargs)
    monkeypatch.setattr(mod.runpy,'run_path',run)
    assert mod.main(['--root',str(root),'--home',str(home),'--resume-round','5','--resume-round','5','--resume-round','9'])==0
    assert calls==[['--root',str(root),'--round','5','--execute'],['--root',str(root),'--round','9','--execute']]
    assert_frozen(frozen)
