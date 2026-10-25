"""R1 spawn/bootstrap regressions: no fit, no CUDA initialization, no scheduler."""
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import r1_spawn_bootstrap as boot
import pytest


HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/paper3-direct-train/campaign_recovery.py')
NATIVE=Path('/workspace/paper3_hpc')
FIX=Path('/workspace/paper3-r1-hardware-fix')
OVERLAY=Path('/workspace/paper3-relaunch-repair')
HOST_SITE='/opt/codex/runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages'


def test_source_transform_exact_two_spawn_entrypoints_and_idempotent():
    original=SOURCE.read_text()
    patched,changed=boot.corrected_recovery_source(original)
    assert changed==['stage_worker','extension_stage_worker']
    assert patched.count('    bootstrap_owned_stage(root)\n')==2
    assert boot.corrected_recovery_source(patched)==(patched,[])
    assert SOURCE.read_text()==original


def native_fixture(tmp_path):
    root=tmp_path/'project';root.mkdir()
    shutil.copytree(NATIVE/'p3work',root/'p3work',ignore=shutil.ignore_patterns('__pycache__'))
    active=tmp_path/'active';active.mkdir()
    for name in ('kernel_bridge.py','audited_runtime.py','hardware_comparison.py'):
        shutil.copy2(FIX/name,active/name)
    shutil.copy2(HERE/'r1_spawn_bootstrap.py',active/'r1_spawn_bootstrap.py')
    direct=tmp_path/'direct';direct.mkdir()
    patched,_=boot.corrected_recovery_source(SOURCE.read_text())
    (direct/'campaign_recovery.py').write_text(patched)
    payload=tmp_path/'overlay';payload.mkdir()
    for name in ('episode_json_compat.py','runtime_policy.py','runtime_transfer.py','sitecustomize.py'):
        shutil.copy2(OVERLAY/name,payload/name)
    files=[]
    for p in payload.glob('*.py'):
        data=p.read_bytes();files.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    manifest=payload/'payload_manifest.json';manifest.write_text(json.dumps({'files':files}))
    launch=root/'runs/training_campaign_parallel_v1/launch.json';launch.parent.mkdir(parents=True)
    launch.write_text(json.dumps({'root':str(root),'package':str(active),'payload_manifest':str(manifest),
        'direct_package':str(direct),'allocation_seconds':529200}))
    return root,active,direct,payload


def test_real_active_bridge_bootstraps_all_native_hooks_in_fresh_process(tmp_path):
    root,active,direct,payload=native_fixture(tmp_path)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'p3work').rglob('*.py')}
    script=r'''
import json,sys,pathlib
sys.path.extend([str(pathlib.Path(sys.executable).absolute().parent.parent/'lib/python3.12/site-packages'),sys.argv[2],sys.argv[3]])
from r1_spawn_bootstrap import bootstrap_owned_stage
receipt=bootstrap_owned_stage(sys.argv[1])
import torch,runtime_transfer
from p3work import development_evaluation as dev,internal_validation as internal,source_extension_downstream as downstream
assert receipt['authorized_a100_transfer_installed']
assert receipt['development_comparison_installed']
assert downstream.__paper3_runtime_transfer__==sys.argv[1]
assert '_paper3_parallel_equivalent_runtime' in internal.execute_internal.__code__.co_names
assert not torch.cuda.is_initialized()
print('FULL_NATIVE_BOOTSTRAP_PASSED')
'''
    result=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(root),str(active),HOST_SITE],
        capture_output=True,text=True,timeout=35,env={**os.environ,'SLURM_JOB_ID':'999999999'})
    assert result.returncode==0,result.stdout+result.stderr
    assert 'FULL_NATIVE_BOOTSTRAP_PASSED' in result.stdout
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'p3work').rglob('*.py')}


@pytest.mark.parametrize('mode',['extension','campaign'])
def test_actual_spawn_entry_uses_full_native_bootstrap_before_guard(tmp_path,mode):
    root,active,direct,payload=native_fixture(tmp_path)
    script=tmp_path/'spawn_real.py'
    script.write_text(r'''
import multiprocessing,os,sys,time,pathlib
sys.path[:0]=[sys.argv[2],sys.argv[3]]
sys.path.append(str(pathlib.Path(sys.executable).absolute().parent.parent/'lib/python3.12/site-packages'))
sys.path.append(sys.argv[4])
import campaign_recovery
def main():
    root=sys.argv[1];ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe()
    extension=sys.argv[5]=='extension'
    if extension:
        worker=ctx.Process(target=campaign_recovery.extension_stage_worker,args=(root,{'arm':'BASE4_PLUS_YANG'},
            'f0',(20261018,),{'execute':True,'user_started_gpu':False},child,os.getpid(),time.monotonic()+30))
    else:
        worker=ctx.Process(target=campaign_recovery.stage_worker,args=(root,60,{'kind':'INVALID'},
            child,os.getpid(),time.monotonic()+30))
    worker.start();child.close()
    try:
        assert parent.poll(20),'owned handshake absent'
        ready=parent.recv();assert ready['type']==('owned_group_ready' if extension else 'group_ready')
        parent.send('dispatch_owned_stage');assert parent.poll(10)
        value=parent.recv();assert value['ok'] is False
        assert value['reason']==('execute_and_later_user_started_GPU_both_required' if extension else 'unregistered_campaign_action'),value
        parent.send('result_received' if extension else 'stage_result_received');worker.join(10);assert worker.exitcode==0
        print('ACTUAL_SPAWN_BOOTSTRAP_PASSED')
    finally:
        parent.close()
        if worker.is_alive():worker.kill();worker.join(5)
if __name__=='__main__':main()
''')
    result=subprocess.run([sys.executable,'-I','-S','-B',str(script),str(root),str(active),str(direct),HOST_SITE,mode],
        capture_output=True,text=True,timeout=45,env={**os.environ,'SLURM_JOB_ID':'999999999'})
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ACTUAL_SPAWN_BOOTSTRAP_PASSED' in result.stdout
    rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    row=next(v for v in rows if v.get('event')=='R1_OWNED_CHILD_RUNTIME_BOOTSTRAPPED')
    assert row['authorized_a100_transfer_installed'] and row['development_comparison_installed']
