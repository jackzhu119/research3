"""Operational R4 migration preserves the already-started numerical recipe."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from paper3r4 import plan as scientific, runtime, submit, monitor


ORIGINAL = Path('/workspace/paper3-round4')
PACKAGE = Path(__file__).resolve().parents[1]


def original_plan_module():
    spec = importlib.util.spec_from_file_location('original_r4_plan_reference', ORIGINAL/'paper3r4/plan.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_original(tmp_path):
    root = tmp_path/'project'
    output = root/'runs/followup_20261024_r4'
    (output/'runtime').mkdir(parents=True)
    old = original_plan_module().make_plan(root, output, cpus=16)
    old['prepared_at'] = 1792837924.4
    path = output/'plan.json'
    path.write_text(json.dumps(old, indent=7)+'\n')
    return root,output,old,path


def arguments(root):
    return ['--root',str(root),'--python',sys.executable,
            '--partition','hpc_gpu','--gpus','3','--cpus','12','--memory-total','128G']


def test_numerical_sources_recipe_and_every_task_are_unchanged(tmp_path):
    original = original_plan_module()
    assert original._code() == scientific._code()
    assert (ORIGINAL/'paper3r4/plan.py').read_bytes() == (PACKAGE/'paper3r4/plan.py').read_bytes()
    old = original.make_plan(tmp_path,tmp_path/'out',cpus=16)
    new = scientific.make_plan(tmp_path,tmp_path/'out',cpus=12)
    assert old['recipe_sha256'] == new['recipe_sha256']
    assert old['tasks'] == new['tasks'] and len(new['tasks']) == 204
    assert old['cpus'] == 16 and new['cpus'] == 12


def test_migration_keeps_original_plan_result_and_checkpoint_bytes(tmp_path):
    root,output,old,path = prepared_original(tmp_path)
    task = output/old['tasks'][0]['key']
    task.mkdir()
    (task/'result.json').write_text('{"status":"COMPLETE","original":true}\n')
    (task/'state-0.pt').write_bytes(b'opaque-original-checkpoint-preserved')
    before = {p:p.read_bytes() for p in (path,task/'result.json',task/'state-0.pt')}
    assert submit.main(arguments(root)) == 0
    assert all(p.read_bytes()==content for p,content in before.items())
    launch = json.loads((output/'runtime/launch.json').read_text())
    assert launch['partition']=='hpc_gpu' and launch['cpus']==12
    assert launch['memory_total']=='128G' and 'memory' not in launch
    command = submit.build_sbatch_command(old,sys.executable,launch)
    assert '--cpus-per-task=36' in command and '--mem=128G' in command
    assert runtime.job_marker(old)==runtime.job_marker(scientific.make_plan(root,output,cpus=12))


def test_same_resources_live_controller_reuses_previous_package_configuration(tmp_path,monkeypatch,capsys):
    root,output,old,path = prepared_original(tmp_path)
    monkeypatch.setattr(submit,'discover_credentials',lambda *_:None)
    assert submit.main(arguments(root))==0
    launch_path=output/'runtime/launch.json'
    launch=json.loads(launch_path.read_text())
    launch['package']='/old-same-recipe-installation'
    launch_path.write_text(json.dumps(launch,indent=3)+'\n')
    before=launch_path.read_bytes()
    with (output/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert submit.main(arguments(root))==0
    assert launch_path.read_bytes()==before
    receipt=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt['existing_controller_reused'] is True


def test_different_resources_live_controller_requires_checkpointed_migrator(tmp_path,monkeypatch):
    root,output,old,path=prepared_original(tmp_path)
    monkeypatch.setattr(submit,'discover_credentials',lambda *_:None)
    old_launch={**runtime.DEFAULT_LAUNCH,'python':sys.executable,'package':'/old-r4',
                'partition':'hpc_gpu1','cpus':16,'memory':'64G'}
    old_launch.pop('memory_total')
    launch_path=output/'runtime/launch.json'
    launch_path.write_text(json.dumps(old_launch)+'\n')
    plan_before,launch_before=path.read_bytes(),launch_path.read_bytes()
    with (output/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit,match='MIGRATE.py'):
            submit.main(arguments(root))
    assert path.read_bytes()==plan_before and launch_path.read_bytes()==launch_before


def test_total_memory_flag_cannot_be_misread_as_old_per_slot_flag(tmp_path):
    with pytest.raises(SystemExit) as error:
        submit.main(['--root',str(tmp_path),'--python',sys.executable,'--memory','64G'])
    assert error.value.code==2
    assert submit.allocation_memory('128G',3)=='128G'
    assert runtime.DEFAULT_LAUNCH['max_resumes']==0


def test_monitor_reports_new_shared_total_and_retains_old_per_slot_history(tmp_path):
    root,output,old,path=prepared_original(tmp_path)
    launch_path=output/'runtime/launch.json'
    launch_path.write_text(json.dumps({'partition':'hpc_gpu','slots':3,'cpus':12,'memory_total':'128G'}))
    resources=monitor.collect(old)['requested_resources']
    assert resources['cpus_per_job']==36 and resources['memory_per_job']=='128G'
    assert resources['memory_per_gpu_budget_gib']==pytest.approx(128/3)
    launch_path.write_text(json.dumps({'partition':'hpc_gpu1','slots':3,'cpus':16,'memory':'64G'}))
    resources=monitor.collect(old)['requested_resources']
    assert resources['cpus_per_job']==48 and resources['memory_per_job']=='192G'
