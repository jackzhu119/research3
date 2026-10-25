"""Exercise the existing extension spawn target with a stdlib-only kernel probe.

The probe performs only the same administrative runtime transfer comparison;
it does not import Torch, fit a model or submit a school job.
"""
import copy
import importlib
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

NATIVE_RECOVERY = Path('/workspace/paper3-direct-train/campaign_recovery.py')
FIX = Path('/workspace/paper3-r1-hardware-fix')
OVERLAY = Path('/workspace/paper3-relaunch-repair')


def profile(size, uuid):
    return {'python':'3.12.3','torch':'2.5.1+cu121','cuda':'12.1',
        'device_name':'NVIDIA A100-SXM4-%dGB'%size,
        'device_uuid':uuid,'device_compute_capability':[8,0],
        'device_total_memory_bytes':size*1024**3,
        'driver_versions':['535.104.12'],'code':{'frozen.py':'a'*64}}


def setup_probe(tmp_path, monkeypatch, *, corrected=False):
    project=tmp_path/'project';project.mkdir()
    package=tmp_path/'probe';(package/'p3work').mkdir(parents=True)
    (package/'p3work/__init__.py').write_text('')
    (package/'p3work/source_extension_downstream.py').write_text('''
import runtime_transfer
def _stage_functions():return {}
def _owned_worker(root,catalog,stage,args,kwargs,connection,owner_pid,deadline):
    try:
        runtime_transfer.validate_transfer(root, args[0], args[1], "owned_extension_resume", record=False)
        value={"ok":True}
    except ValueError as error:value={"ok":False,"reason":str(error)}
    value["audited_policy_installed"]=bool(getattr(runtime_transfer,"_paper3_authorized_a100_transfer",False))
    connection.send(value);connection.close()
''')
    source=NATIVE_RECOVERY.read_text()
    if corrected:
        anchor='    from p3work import source_extension_downstream as downstream\n'
        assert source.count(anchor)==1
        source=source.replace(anchor,
            '    from audited_runtime import install as install_a100_policy\n'
            '    install_a100_policy(root)\n'+anchor)
    name='spawn_probe_corrected' if corrected else 'spawn_probe_original'
    (package/(name+'.py')).write_text(source)
    monkeypatch.syspath_prepend(str(OVERLAY));monkeypatch.syspath_prepend(str(FIX));monkeypatch.syspath_prepend(str(package))
    # The real bridge installs this policy in its parent before calling auto_development.
    import runtime_transfer
    from audited_runtime import install
    runtime_transfer._paper3_authorized_a100_transfer=False
    install(project,transfer_module=runtime_transfer)
    old=profile(80,'b4f600a0-a5a4-b950-504b-b8c069e3e812')
    new=profile(40,'9d54e32d-18fb-5bec-62d4-1d7409204fc4')
    assert runtime_transfer.validate_transfer(project,old,new,'parent',record=False)
    return project,importlib.import_module(name),old,new


@pytest.mark.parametrize('corrected',[False,True])
def test_existing_extension_spawn_target_drops_parent_only_a100_policy(tmp_path,monkeypatch,corrected):
    project,module,old,new=setup_probe(tmp_path,monkeypatch,corrected=corrected)
    copies=copy.deepcopy((old,new))
    ctx=multiprocessing.get_context('spawn');parent,child=ctx.Pipe(False)
    worker=ctx.Process(target=module.extension_stage_worker,args=(str(project),{'arm':'BASE4_PLUS_YANG'},
        'probe',(old,new),{},child,os.getpid(),1e20))
    worker.start();child.close()
    try:
        assert parent.poll(15),'Probe child produced no transfer result'
        value=parent.recv();worker.join(10)
        assert worker.exitcode==0
        assert value['ok'] is corrected,value
        assert value['audited_policy_installed'] is corrected
        if not corrected:
            assert value['reason']=='runtime_transfer_GPU_memory_architecture_software_or_policy_differs'
        assert (old,new)==copies
    finally:
        parent.close()
        if worker.is_alive():worker.kill();worker.join(5)
