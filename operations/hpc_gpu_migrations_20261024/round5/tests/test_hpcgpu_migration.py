"""Resource migration keeps R5 numerical inputs and persisted work intact.

All execution here uses synthetic EEG on CPU; no school job is submitted.
"""
from pathlib import Path
import copy
import fcntl
import json
import os
import subprocess
import sys
import threading

import pytest

from paper3r5 import plan as plans, submit, worker
from paper3r5.algorithm import Config
from paper3r5.data import Catalog, load_source
from paper3r5.io import put, read, sha
from paper3r5.monitor import collect
from test_worker import plan_fixture


PACKAGE = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/workspace/paper3-round5')


def isolated_entry(package, mode, plan_path, *extra):
    return [sys.executable, '-I', '-S', '-B', '-u',
            str(package/'paper3r5/entry.py'), mode, '--plan', str(plan_path), *extra]


def environment():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.pop('PAPER3R5_STOP_REQUESTED', None)
    return env


def run(command, **kwargs):
    result = subprocess.run(command, env=environment(), capture_output=True, text=True,
                            timeout=120, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_resource_defaults_request_one_hpc_gpu_job_with_exact_total_memory(tmp_path, capsys):
    assert submit.main(['--root', str(tmp_path), '--python', sys.executable]) == 0
    prepared = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert prepared['partition'] == 'hpc_gpu'
    assert prepared['gpu_slots'] == 3 and prepared['slurm_jobs'] == 1
    assert prepared['cpus_per_job'] == 36 and prepared['host_memory_per_job'] == '128G'
    assert prepared['host_memory_per_gpu_budget_gib'] == pytest.approx(128/3)
    output = tmp_path/'runs/followup_20261024_r5'
    launch = read(output/'runtime/launch.json')
    command = submit.build_sbatch_command(read(output/'plan.json'), sys.executable, launch)
    resources = collect(read(output/'plan.json'))['requested_resources']
    assert resources['partition']=='hpc_gpu'
    assert resources['memory_per_job']=='128G' and resources['cpus_per_job']==36
    for flag in ('--partition=hpc_gpu','--nodes=1','--ntasks=1','--cpus-per-task=36',
                 '--mem=128G','--gres=gpu:3'):
        assert flag in command
    assert not any(flag.startswith(('--array=', '--nodelist=', '--exclude=')) for flag in command)


def test_resource_reprepare_does_not_rewrite_plan_or_scientific_artifacts(tmp_path):
    assert submit.main(['--root',str(tmp_path),'--python',sys.executable]) == 0
    output = tmp_path/'runs/followup_20261024_r5'
    plan_path = output/'plan.json'
    old_launch = read(output/'runtime/launch.json')
    old_launch.update(partition='hpc_gpu1',memory_total='164G')
    put(output/'runtime/launch.json',old_launch)
    folder = output/read(plan_path)['tasks'][0]['key']
    put(folder/'result.json',{'status':'COMPLETE','identity':'synthetic-original-result'})
    (folder/'original-checkpoint.pt').write_bytes(b'synthetic checkpoint bytes retained')
    watched = [plan_path,folder/'result.json',folder/'original-checkpoint.pt']
    original = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in watched}
    assert submit.main(['--root',str(tmp_path),'--python',sys.executable,
                        '--partition','hpc_gpu','--cpus','12','--memory-total','128G']) == 0
    assert original == {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in watched}
    launch = read(output/'runtime/launch.json')
    assert launch['partition']=='hpc_gpu' and launch['memory_total']=='128G'


def test_repackaged_live_controller_is_reused_without_rewriting_its_launch(tmp_path,capsys):
    assert submit.main(['--root',str(tmp_path),'--python',sys.executable])==0
    output=tmp_path/'runs/followup_20261024_r5'
    path=output/'runtime/launch.json'
    old_launch=read(path);old_launch['package']='/synthetic/previous-install'
    put(path,old_launch)
    original=(path.read_bytes(),path.stat().st_mtime_ns)
    with (output/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert submit.main(['--root',str(tmp_path),'--python',sys.executable])==0
    prepared=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert prepared['existing_controller_reused']
    assert original==(path.read_bytes(),path.stat().st_mtime_ns)


def test_different_resource_live_controller_is_not_silently_reconfigured(tmp_path):
    assert submit.main(['--root',str(tmp_path),'--python',sys.executable])==0
    output=tmp_path/'runs/followup_20261024_r5'
    path=output/'runtime/launch.json'
    old_launch=read(path);old_launch.update(partition='hpc_gpu1',memory_total='164G')
    put(path,old_launch);original=path.read_bytes()
    with (output/'runtime/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(SystemExit,match='MIGRATE.py'):
            submit.main(['--root',str(tmp_path),'--python',sys.executable])
    assert path.read_bytes()==original


def test_original_and_migrated_scientific_recipe_task_matrix_are_identical(tmp_path):
    if not ORIGINAL.is_dir():
        pytest.skip('Original local source is required only for migration provenance comparison')
    fingerprints = plans._code()
    assert fingerprints == {name:sha(ORIGINAL/name) for name in fingerprints}
    command = [sys.executable,'-I','-S','-B','-c',
        'import json,sys;sys.path.insert(0,sys.argv[1]);'
        'from paper3r5.plan import make_plan;'
        'print(json.dumps(make_plan(sys.argv[2],sys.argv[3]),sort_keys=True))',
        str(ORIGINAL),str(tmp_path),str(tmp_path/'same-output')]
    previous = json.loads(run(command).stdout)
    current = plans.make_plan(tmp_path,tmp_path/'same-output')
    assert previous == current
    assert previous['recipe_sha256'] == current['recipe_sha256']


def test_original_checkpoint_and_session_rows_resume_under_migrated_allocation(tmp_path):
    if not ORIGINAL.is_dir():
        pytest.skip('Original local source is required only for migration provenance comparison')
    plan,args = plan_fixture(tmp_path)
    basis,evaluation = plan['tasks']
    evaluation['scenarios']=['clean']
    paused = copy.deepcopy(basis)
    paused['key']='basis-original-paused'
    plan['tasks']=[basis,paused,evaluation]
    plan['seeds']=[20261018]
    put(args.plan,plan)
    plan_bytes = Path(args.plan).read_bytes()
    output = Path(plan['output'])
    basis_folder = output/basis['key']
    basis_folder.mkdir(parents=True)
    paused_folder = output/paused['key']
    paused_folder.mkdir(parents=True)
    # Produce the checkpoint using the original package and its own schema.
    # Its synthetic cursor0 is real and the migrated worker performs update1.
    script = (
        'import json,sys,torch;sys.path.insert(0,sys.argv[1]);'
        'from pathlib import Path;'
        'from paper3r5.worker import save_checkpoint,_identity;'
        'from paper3r5.data import Catalog,load_source;'
        'from p3work.spatial_geometry import initialize_dictionary;'
        'p=json.loads(Path(sys.argv[2]).read_text());t=p["tasks"][1];'
        's,g,i=load_source(p["root"],t["seed"],"cpu");'
        'd=Catalog(p["root"],"meta_train").identity;'
        'b=initialize_dictionary(t["config"]["rank"],t["seed"],device="cpu");'
        'save_checkpoint(Path(p["output"])/t["key"],'
        '{"identity":_identity(p,t,i,d),"update":0,"validated_update":0,'
        '"history":[],"b":b,"elapsed_seconds":0.})'
    )
    # Import torch through the same selected environment; the original isolated
    # entry uses this environment too. This process touches only the fixture.
    result = subprocess.run([sys.executable,'-B','-c',script,str(ORIGINAL),args.plan],
                            env=environment(),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    old_identity = worker.read_checkpoint(paused_folder)[0]['identity']
    # Complete a different task using the old worker, so the migration proves
    # both reuse of finished work and actual continuation of the paused task.
    prior = run(isolated_entry(ORIGINAL,'worker',args.plan,'--task','0','--device','cpu'))
    assert 'TASK_COMPLETE' in prior.stdout
    retained = {p:(p.read_bytes(),p.stat().st_mtime_ns)
                for p in basis_folder.rglob('*') if p.is_file() and p.name!='.lock'}
    # A partial original evaluation contains a valid committed B20 session;
    # its result marker is absent, so the new allocation must finish B4.
    source,geometry,source_identity = load_source(tmp_path,20261018,'cpu')
    catalog = Catalog(tmp_path,'internal_holdout')
    b,_ = worker.load_basis(plan,evaluation,'cpu',source_identity=source_identity,
                            data_identity=catalog.identity)
    config = Config(**evaluation['config'])
    row_folder = output/evaluation['key']/'evaluations/holdout/B20/clean'
    worker.evaluate(source,geometry,b,catalog,config,row_folder,threading.Event(),{},
                    budget=20,source_identity=source_identity)
    row_path = row_folder/'S3-1.json'
    original_row = (row_path.read_bytes(),row_path.stat().st_mtime_ns)
    command = isolated_entry(PACKAGE,'allocation',args.plan,'--python',sys.executable,
        '--device','cpu','--slots','3','--cpus-per-slot','1','--poll-seconds','.02')
    completed = run(command)
    assert 'ALLOCATION_TASK_STARTED' in completed.stdout
    progress = read(output/'allocation_progress.json')
    assert progress['allocation_status']=='COMPLETE' and progress['complete_tasks']==3
    assert Path(args.plan).read_bytes()==plan_bytes
    assert retained == {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in retained}
    assert original_row == (row_path.read_bytes(),row_path.stat().st_mtime_ns)
    assert read(output/evaluation['key']/'result.json')['status']=='COMPLETE'
    assert read(paused_folder/'result.json')['identity']==old_identity
    assert read(paused_folder/'result.json')['outer_updates']==1
    assert 'CHECKPOINT_RESUMED' in ''.join(p.read_text() for p in (output/'logs').glob('task-1-attempt*.out'))
    assert (output/evaluation['key']/'evaluations/holdout/B4/clean/S3-1.json').is_file()
    assert not list((output/'logs').glob('task-0-attempt*.out'))
