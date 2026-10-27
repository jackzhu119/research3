"""Production operational modules with fake Slurm; no Torch, network or GPU."""
import copy
import errno
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('/workspace/paper3-round11')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PACKAGE))
import resume_io
from paper3r11.platform import common, runtime, scheduler


@pytest.fixture(autouse=True)
def isolated_patches(monkeypatch):
    """Every fixture gets a fresh original operational module alias set."""
    names = ('atomic', 'atomic_bytes', 'read', 'load_plan')
    before = [(module, key, getattr(module, key))
              for name, module in list(sys.modules.items())
              if name.startswith('paper3r11.') and module is not None
              for key in names if hasattr(module, key)]
    old_sched = scheduler._atomic
    yield
    for module, key, value in before:
        setattr(module, key, value)
    scheduler._atomic = old_sched


def fixture(tmp_path, keys=('independent',), blocked=()):
    root = tmp_path / 'project'; root.mkdir()
    output = root / 'runs' / 'r11-io-fixture'; run = output / 'runtime'; run.mkdir(parents=True)
    tasks = [dict(key=key, kind='fixture', role='meta_validation', requires=[]) for key in keys]
    plan = dict(schema_version=common.PLAN_SCHEMA, round=11, stage='development',
        synthetic_fixture=True, root=str(root), output=str(output), package=str(PACKAGE),
        python=sys.executable, tasks=tasks, secondary_ablation_family=[], reference_strategy='fixture-fixed',
        resources=dict(jobs=4, gpus_per_job=1, cpus_per_job=16, memory_per_job='64G'),
        code_manifest={'paper3r11/platform/common.py':common.sha(PACKAGE / 'paper3r11/platform/common.py')})
    plan = common.sign_plan(plan); path = root / 'plan.json'; common.atomic(path, plan)
    launch = runtime.context(path)['launch']; common.atomic(run / 'launch.json', launch)
    common.atomic(run / 'SUBMIT_AUTHORIZATION.json', dict(explicit_submit=True,
        plan_sha256=common.sha(path), recipe_sha256=plan['recipe_sha256']))
    common.atomic(run / 'tasks.json', dict(recipe_sha256=plan['recipe_sha256'], tasks={}))
    marker = resume_io.legacy.scheduler_identity(run, plan['recipe_sha256'])
    slots = {}
    for slot in range(4):
        attempt = dict(slot=slot, sequence=1, marker=marker, name=f'{marker}-s{slot}-a1',
            status='ACKNOWLEDGED', job_id=str(100+slot), intent_at=1.)
        slots[str(slot)] = dict(attempts=[attempt], next_submit_at=0., rejections=0, worker_failures=0)
    common.atomic(run / 'scheduler-state.json', dict(version=1, marker=marker, created_at=1., slots=slots))
    env = resume_io.load_env(PACKAGE, path, expected_recipe=None)
    return path, plan, env


def monitor_error(env, key):
    target = env['runtime'] / 'numerical-observations' / (key+'.json')
    source = target.with_name('.'+target.name+'.vanished')
    return dict(error_type='FileNotFoundError', reason=f"[Errno 2] No such file or directory: '{source}' -> '{target}'")


def blocked_row(env, plan, key, *, unrelated=False):
    error = monitor_error(env, key) if not unrelated else dict(error_type='FileNotFoundError', reason='Pinned source data missing')
    state = runtime._state(plan)
    state['tasks'][key] = dict(phase='BLOCKED_CONFIGURATION', key=key, attempt=1,
        attempt_token='a'*64, current_error=error, failure_history=[dict(error, attempt=1)], consecutive_failures=1)
    common.atomic(env['runtime'] / 'tasks.json', state)


def unsealed_result(plan, key):
    task = next(t for t in plan['tasks'] if t['key']==key)
    folder = common.task_folder(plan, task); folder.mkdir(parents=True, exist_ok=True)
    payload = folder / 'operator.npz'; payload.write_bytes(b'original immutable synthetic operator bytes')
    descriptor = dict(path=payload.name, bytes=payload.stat().st_size, sha256=common.sha(payload))
    common.atomic(folder / 'result.json', dict(status='COMPLETE', scientific_scope='SOFTWARE_TEST_ONLY',
        recipe_sha256=plan['recipe_sha256'], task_key=key, task_identity=common.task_identity(plan, task),
        artifacts=[descriptor]))
    return folder


def test_monitor_write_failure_is_deferred_but_result_write_remains_strict(tmp_path, capfd):
    def fail(*a, **k):
        raise PermissionError(errno.EACCES, 'synthetic injected access failure')
    c = SimpleNamespace(atomic=fail)
    run = tmp_path / 'runtime'; state = resume_io.install_telemetry_patch(c, run)
    c.atomic(run / 'status.json', {})
    c.atomic(run / 'numerical-observations' / 'task.json', {})
    with pytest.raises(PermissionError):
        c.atomic(tmp_path / 'task' / 'result.json', {})
    with pytest.raises(PermissionError):
        c.atomic(run / 'tasks.json', {})
    with pytest.raises(PermissionError):
        c.atomic(run / 'scheduler-state.json', {})
    assert state['deferred_writes']==2
    assert 'R11_TELEMETRY_WRITE_DEFERRED' in capfd.readouterr().err


def test_actual_original_worker_status_survives_observation_and_status_write_fault(tmp_path, monkeypatch):
    path, plan, env = fixture(tmp_path)
    task, lease, row = runtime.claim_task(plan, 0, '100'); lease.close()
    common.atomic(common.task_folder(plan, task) / 'progress.json',
                  dict(attempt_token=row['attempt_token'], outer_update=4, status='TRAINING'))
    previous = runtime.atomic
    def fail_monitor(path, value):
        if resume_io.is_telemetry_path(path, env['runtime']):
            raise FileNotFoundError(errno.ENOENT, 'synthetic rename disappearance')
        return previous(path, value)
    monkeypatch.setattr(common, 'atomic', fail_monitor)
    monkeypatch.setattr(runtime, 'atomic', fail_monitor)
    # Install after injection so the production status() function is exercised.
    telemetry = resume_io.install_telemetry_patch(common, env['runtime'])
    result = runtime.status(path, plan=plan)
    assert result['tasks'][0]['phase']=='RUNNING'
    assert result['tasks'][0]['numerical_progress']['outer_update']==4
    assert telemetry['deferred_writes']==2
    assert runtime._state(plan)['tasks'][task['key']]['phase']=='RUNNING'


def test_recover_completed_science_after_monitor_failure_without_retraining(tmp_path):
    path, plan, env = fixture(tmp_path, keys=('complete-after-error', 'real-source-failure'))
    blocked_row(env, plan, 'complete-after-error')
    blocked_row(env, plan, 'real-source-failure', unrelated=True)
    folder = unsealed_result(plan, 'complete-after-error')
    before = {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
    recovery = resume_io.recover_monitor_failures(env)
    row = runtime._state(plan)['tasks']['complete-after-error']
    assert recovery['recovered_complete_tasks']==['complete-after-error']
    assert row['phase']=='COMPLETE' and row['failure_history'][0]['error_type']=='FileNotFoundError'
    assert row['io_recoveries'][0]['scientific_computation_executed'] is False
    assert runtime._state(plan)['tasks']['real-source-failure']['phase']=='BLOCKED_CONFIGURATION'
    assert common.verify_result(plan, plan['tasks'][0])
    for name, content in before.items():
        assert (folder / name).read_bytes()==content
    seal = (folder / 'completion.json').read_bytes()
    assert resume_io.recover_monitor_failures(env)['recovered_complete_tasks']==[]
    assert (folder / 'completion.json').read_bytes()==seal


def test_live_scientific_child_lock_prevents_recovery_or_duplicate_claim(tmp_path):
    path, plan, env = fixture(tmp_path)
    blocked_row(env, plan, 'independent'); folder = unsealed_result(plan, 'independent')
    lock = (folder / '.scientific.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX)
    before = (env['runtime'] / 'tasks.json').read_bytes()
    try:
        recovery = resume_io.recover_monitor_failures(env)
        assert recovery['live_tasks_left_untouched']==['independent']
        assert (env['runtime'] / 'tasks.json').read_bytes()==before
        assert not (folder / 'completion.json').exists()
    finally:
        lock.close()


def test_incomplete_monitor_failure_retries_preserve_history_and_payload(tmp_path):
    path, plan, env = fixture(tmp_path)
    blocked_row(env, plan, 'independent'); folder = common.task_folder(plan, plan['tasks'][0]); folder.mkdir()
    checkpoint = folder / 'checkpoint.npz'; checkpoint.write_bytes(b'keep partial original checkpoint')
    result = resume_io.recover_monitor_failures(env)
    row = runtime._state(plan)['tasks']['independent']
    assert result['monitor_failures_requeued']==['independent'] and row['phase']=='PAUSED'
    assert row['failure_history'] and checkpoint.read_bytes()==b'keep partial original checkpoint'


def test_corrupt_complete_payload_is_not_falsely_sealed(tmp_path):
    path, plan, env = fixture(tmp_path)
    blocked_row(env, plan, 'independent'); folder = unsealed_result(plan, 'independent')
    (folder / 'operator.npz').write_bytes(b'changed')
    recovery = resume_io.recover_monitor_failures(env)
    assert not recovery['recovered_complete_tasks']
    assert not (folder / 'completion.json').exists()
    assert runtime._state(plan)['tasks']['independent']['phase']=='PAUSED'


def test_controller_first_observation_preserves_two_live_and_only_replaces_two_failed(tmp_path):
    path, plan, env = fixture(tmp_path)
    launch_path = env['runtime'] / 'launch.json'; launch_before = launch_path.read_bytes()
    sources = {p:common.sha(PACKAGE / p) for p in plan['code_manifest']}
    marker = resume_io.legacy.scheduler_identity(env['runtime'], plan['recipe_sha256'])
    commands=[]; clock=[1000.]; stop=threading.Event(); waits=[0]
    def runner(command, **kwargs):
        commands.append(command)
        if command[0]=='squeue':
            out='\n'.join(f'{100+s}|{marker}|RUNNING|{marker}-s{s}-a1|gpu{s}|gpu{s}' for s in (2,3))
        elif command[0]=='sacct':
            out='\n'.join(f'{100+s}|FAILED|2:0|{marker}|{marker}-s{s}-a1' for s in (0,1))
        elif command[0]=='sbatch':
            out=str(900+sum(c[0]=='sbatch' for c in commands))
        else:
            raise AssertionError('No cancellation or other command permitted: '+str(command))
        return SimpleNamespace(returncode=0, stdout=out, stderr='')
    work=tmp_path / 'work'
    first=resume_io.controller_loop(path, work, package=PACKAGE, submit=True,
        runner=runner, once=True, expected_recipe=None, clock=lambda:clock[0], launch_token='test-first')
    assert not any(c[0]=='sbatch' for c in commands)
    ready=common.read(work / 'controller-ready.json')
    assert ready['launch_token']=='test-first' and ready['atomic_patch_loaded']
    def advance(_):
        waits[0]+=1;clock[0]+=31
        if waits[0]>=2:stop.set()
    resume_io.controller_loop(path, work, package=PACKAGE, submit=True, runner=runner,
        stop=stop, sleep=advance, expected_recipe=None, clock=lambda:clock[0])
    submits=[c for c in commands if c[0]=='sbatch']
    assert len(submits)==2
    assert {'--mem=64G','--gres=gpu:1','--cpus-per-task=16'}<=set(submits[0])
    assert all(str(ROOT / 'resume_io.py') in c for c in submits)
    assert all(c[-3:] != ['--slot','2','--device'] for c in submits)
    state=common.read(env['runtime'] / 'scheduler-state.json')
    assert len(state['slots']['2']['attempts'])==len(state['slots']['3']['attempts'])==1
    assert state['slots']['2']['attempts'][0]['job_id']=='102'
    assert state['slots']['3']['attempts'][0]['job_id']=='103'
    assert len(state['slots']['0']['attempts'])==len(state['slots']['1']['attempts'])==2
    assert launch_path.read_bytes()==launch_before
    assert {p:common.sha(PACKAGE / p) for p in plan['code_manifest']}==sources


def test_science_process_entry_rewrite_preserves_all_scientific_flags(tmp_path):
    path, plan, env = fixture(tmp_path); commands=[]
    spawn=resume_io.science_popen(env, lambda command, **kwargs:commands.append((command,kwargs)))
    original=[sys.executable,'-I','-S','-B','-u',str(PACKAGE / 'paper3r11/platform/entry.py'),
              'science-daemon','--plan',str(path),'--device','cuda:0','--response-dir','/tmp/responses']
    spawn(original, stdin='sentinel')
    changed=commands[0][0]
    assert changed[:5]==original[:5] and changed[6:]==original[6:]+['--package',str(PACKAGE)]
    assert changed[5]==str(ROOT / 'resume_io.py') and commands[0][1]['stdin']=='sentinel'
    assert original[5]==str(PACKAGE / 'paper3r11/platform/entry.py')


def test_science_cli_preserves_task_key_and_original_argument_order(tmp_path, monkeypatch):
    path, plan, env = fixture(tmp_path); calls=[]
    monkeypatch.setattr(resume_io, 'load_env', lambda *a, **k:env)
    entry=importlib.import_module('paper3r11.platform.entry')
    monkeypatch.setattr(entry, 'main', lambda args:calls.append(args) or 0)
    original=['science','--task-key','independent','--plan',str(path),'--device','cpu']
    assert resume_io.main(original+['--package',str(PACKAGE)])==0
    assert calls==[original]


def test_corrupted_durable_task_state_is_not_reset(tmp_path):
    path, plan, env = fixture(tmp_path)
    target=env['runtime'] / 'tasks.json';target.write_bytes(b'{corrupt original')
    with pytest.raises(ValueError,match='unreadable'):
        resume_io.compact_task_status(env)
    assert target.read_bytes()==b'{corrupt original'


def test_actual_persistent_worker_finishes_when_monitor_writes_fail(tmp_path, monkeypatch):
    path, plan, env = fixture(tmp_path)
    strict = common.atomic; closed=[]
    def injected(path, value):
        if resume_io.is_telemetry_path(path, env['runtime']):
            raise FileNotFoundError(errno.ENOENT, 'injected monitor rename disappearance')
        return strict(path, value)
    monkeypatch.setattr(common, 'atomic', injected)
    monkeypatch.setattr(runtime, 'atomic', injected)
    telemetry=resume_io.install_telemetry_patch(common, env['runtime'])
    class Science:
        bootstrap_failed=False
        def __init__(self, *a, **k):pass
        def run(self, task, row, attempt, stop, heartbeat):
            heartbeat(999,False)
            unsealed_result(plan,task['key'])
            return 0
        def close(self):closed.append(True)
    monkeypatch.setattr(runtime, 'PersistentScience', Science)
    assert runtime.persistent_worker_loop(path, 0, device='cpu', once=True)==0
    assert runtime._state(plan)['tasks']['independent']['phase']=='COMPLETE'
    assert common.verify_result(plan,plan['tasks'][0])
    assert telemetry['deferred_writes']>=3 and closed==[True]


def test_actual_persistent_worker_never_claims_completion_when_seal_write_fails(tmp_path, monkeypatch):
    path, plan, env = fixture(tmp_path); strict=common.atomic
    def injected(path, value):
        if Path(path).name=='completion.json':
            raise OSError(errno.ENOSPC, 'injected full result filesystem')
        return strict(path,value)
    monkeypatch.setattr(common,'atomic',injected)
    monkeypatch.setattr(runtime,'atomic',injected)
    resume_io.install_telemetry_patch(common,env['runtime'])
    class Science:
        bootstrap_failed=False
        def __init__(self,*a,**k):pass
        def run(self,task,row,attempt,stop,heartbeat):
            unsealed_result(plan,task['key']);return 0
        def close(self):pass
    monkeypatch.setattr(runtime,'PersistentScience',Science)
    runtime.persistent_worker_loop(path,0,device='cpu',once=True)
    row=runtime._state(plan)['tasks']['independent']
    assert row['phase']!='COMPLETE'
    assert not common.verify_result(plan,plan['tasks'][0])


def test_controller_ready_waits_for_actual_successful_queue_observation(tmp_path, monkeypatch):
    path, plan, env=fixture(tmp_path);work=tmp_path / 'work';calls=[];stop=threading.Event()
    states=['EXISTING_SCHEDULER_REUSED','OBSERVATION_RETRY','RUNNING']
    class Scheduler:
        def __init__(self,*a,**k):pass
        def step(self,keep_workers):
            assert not (work / 'controller-ready.json').exists()
            calls.append(keep_workers)
            status=states[len(calls)-1]
            return dict(controller_status=status,inflight_jobs=4,jobs=[],
                        observation_error='SyntheticSqueueRetry' if status=='OBSERVATION_RETRY' else None)
    monkeypatch.setattr(runtime,'FourJobScheduler',Scheduler)
    def wait(seconds):
        if seconds>2:
            stop.set()
    resume_io.controller_loop(path,work,package=PACKAGE,submit=True,stop=stop,
        sleep=wait,expected_recipe=None,launch_token='fresh-observation')
    assert calls==[False,False,False]
    ready=common.read(work / 'controller-ready.json')
    assert ready['first_slurm_observation_completed'] and ready['queue_observation_successful']
    assert ready['launch_token']=='fresh-observation'
    assert common.read(Path(plan['output'])/'runtime/controller-ready.json') == ready
