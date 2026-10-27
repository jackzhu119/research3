from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import report_fix as r
from atomic_io import atomic_bytes


def encoded(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path, default=None, **kwargs):
    try:
        return json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return default


def atomic(path, value):
    atomic_bytes(path, encoded(value))


@pytest.fixture
def env(tmp_path):
    runtime = tmp_path/'output/runtime'; runtime.mkdir(parents=True)
    plan = dict(output=str(runtime.parent), recipe_sha256='b'*64, stage='campaign',
                python=sys.executable, tasks=[dict(key='done'), dict(key='future')])
    path = tmp_path/'plan.json'; atomic(path, plan)
    rows = {'done':dict(completion_sha256='c'*64)}
    value = dict(runtime=runtime, path=path, package=tmp_path/'package',
                 header=dict(recipe_sha256=plan['recipe_sha256'], python=sys.executable))
    value['common'] = SimpleNamespace(atomic=atomic, read=read, digest=digest, encoded=encoded,
                                     sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                                     load_plan=lambda *a, **k:(path, plan))
    value['runtime_module'] = SimpleNamespace(completed_task_rows=lambda plan:rows)
    value['uploads_module'] = SimpleNamespace(report_completion=lambda *a:False,
                                              publish_stage_index=lambda *a, **k:None)
    value['test_plan'] = plan
    return value


def successful_runner(env, *, stop=None, delay=0):
    def run(command, **kwargs):
        assert command[1:5] == ['-I','-S','-B','-u']
        assert command[5].endswith('report_child.py')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['OMP_NUM_THREADS'] == '2'
        assert 'PYTHONPATH' not in kwargs['env']
        if delay:
            time.sleep(delay)
        report_path=Path(env['test_plan']['output'])/'reports/report.json'
        atomic(report_path, dict(recipe_sha256=env['test_plan']['recipe_sha256'],
             coverage=dict(sealed_complete_tasks=1, complete_task_keys=['done'],
                           complete_task_keys_sha256=digest(['done']))))
        atomic(Path(env['test_plan']['output'])/'reports/report-status.json',
               dict(recipe_sha256=env['test_plan']['recipe_sha256'], sealed_complete_tasks=1,
                    complete_task_keys_sha256=digest(['done']), report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    total_tasks=2, status='PARTIAL'))
        if stop:
            stop.set()
        return SimpleNamespace(returncode=0)
    return run


def test_partial_report_records_actual_coverage_and_preserves_science(env, tmp_path):
    science = Path(env['test_plan']['output'])/'immutable-science.pt'; science.write_bytes(b'checkpoint')
    work = tmp_path/'management'
    result = r.run_report(env, work, once=True, launch_token='nonce',
                          runner=successful_runner(env))
    assert result['status'] == 'PARTIAL_REPORT_GENERATED'
    assert result['observed_complete_tasks'] == result['report_complete_tasks'] == 1
    assert result['total_tasks'] == 2 and result['training_controlled'] is False
    assert read(work/'report-ready.json')['lock_acquired'] is True
    assert read(work/'report-ready.json')['launch_token'] == 'nonce'
    assert science.read_bytes() == b'checkpoint'


def test_whole_cycle_state_read_error_is_retried_then_recovers(env, tmp_path):
    calls = []
    def rows(plan):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('unreadable operational state')
        return {'done':dict(completion_sha256='c'*64)}
    env['runtime_module'].completed_task_rows = rows
    stop = threading.Event(); work=tmp_path/'management'
    result = r.run_report(env, work, stop=stop, interval=.01,
                          runner=successful_runner(env, stop=stop))
    assert len(calls) == 2
    assert read(work/'report-last-error.json')['current_error']['error_type'] == 'ValueError'
    assert read(work/'report-last-success.json')['report_complete_tasks'] == 1
    assert read(env['runtime']/'reporter-status.json')['current_error'] is None
    assert result['status'] == 'STOPPED'


def test_report_process_failure_does_not_replace_last_success_signature(env, tmp_path):
    work = tmp_path/'management'; old=dict(report_complete_tasks=0, last_success_utc='old')
    atomic(work/'report-last-success.json', old)
    result = r.run_report(env, work, once=True,
                          runner=lambda *a, **k:SimpleNamespace(returncode=2))
    assert result['status'] == 'RETRYING'
    assert read(work/'report-last-success.json') == old
    assert not (env['runtime']/'reporter/last-generated.json').exists()


def test_unchanged_completion_frontier_never_reruns_full_report(env, tmp_path):
    rows = env['runtime_module'].completed_task_rows(env['test_plan'])
    atomic(env['runtime']/'reporter/last-generated.json',
           dict(recipe_sha256=env['header']['recipe_sha256'], signature=digest(
                {key:row['completion_sha256'] for key,row in rows.items()}), complete_tasks=1))
    result = r.run_report(env, tmp_path/'management', once=True,
                          runner=lambda *a, **k:pytest.fail('unchanged report recomputed'))
    assert result['status'] == 'WAITING_FOR_NEW_COMPLETION'
    assert result['report_complete_tasks'] == 1


def test_telemetry_memoryerror_cannot_kill_report_fit_or_save_false_completion(env, tmp_path):
    original = env['common'].atomic
    def write(path, value):
        if Path(path) in (env['runtime']/'reporter-status.json',tmp_path/'management/report-status.json',tmp_path/'management/report-ready.json'):
            raise MemoryError('monitor failure')
        return original(path, value)
    env['common'].atomic = write
    result = r.run_report(env, tmp_path/'management', once=True,
                          runner=successful_runner(env))
    assert result['status'] == 'PARTIAL_REPORT_GENERATED'
    assert read(tmp_path/'management/report-last-success.json')['report_complete_tasks'] == 1


def test_successful_exit_without_matching_report_receipt_is_retry_not_success(env, tmp_path):
    result = r.run_report(env, tmp_path/'management', once=True,
                          runner=lambda *a, **k:SimpleNamespace(returncode=0))
    assert result['status'] == 'RETRYING'
    assert not (env['runtime']/'reporter/last-generated.json').exists()


def test_same_count_different_completed_keys_is_not_accepted_as_current_report(env, tmp_path):
    def run(*args, **kwargs):
        successful_runner(env)(*args, **kwargs)
        folder=Path(env['test_plan']['output'])/'reports'
        report=read(folder/'report.json')
        report['coverage'].update(complete_task_keys=['future'],complete_task_keys_sha256=digest(['future']))
        atomic(folder/'report.json',report)
        receipt=read(folder/'report-status.json')
        receipt.update(complete_task_keys_sha256=digest(['future']),
                       report_sha256=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest())
        atomic(folder/'report-status.json',receipt)
        return SimpleNamespace(returncode=0)
    result=r.run_report(env,tmp_path/'management',once=True,runner=run)
    assert result['status']=='RETRYING'
    assert not (env['runtime']/'reporter/last-generated.json').exists()


def test_report_bytes_changed_after_seal_never_advance_signature(env, tmp_path):
    def run(*args, **kwargs):
        result=successful_runner(env)(*args, **kwargs)
        (Path(env['test_plan']['output'])/'reports/report.json').write_text('{}')
        return result
    result=r.run_report(env,tmp_path/'management',once=True,runner=run)
    assert result['status']=='RETRYING'
    assert not (env['runtime']/'reporter/last-generated.json').exists()


def test_live_heartbeat_while_long_report_does_not_claim_fresh_report(env, tmp_path):
    work=tmp_path/'management'; work.mkdir()
    atomic(work/'report-last-success.json', dict(report_complete_tasks=0, last_success_utc='old'))
    seen=[]
    original=successful_runner(env)
    def runner(command, **kwargs):
        time.sleep(.08)
        seen.append(read(env['runtime']/'reporter-status.json'))
        return original(command, **kwargs)
    result=r.run_report(env, work, once=True, runner=runner, heartbeat_interval=.01)
    assert seen[0]['status'] == 'RUNNING_REPORT'
    assert seen[0]['report_complete_tasks'] == 0
    assert seen[0]['observed_complete_tasks'] == 1
    assert seen[0]['heartbeat_utc']
    assert result['report_complete_tasks'] == 1


def test_original_report_lock_is_respected_without_starting_child(env, tmp_path):
    import fcntl
    with (env['runtime']/'report-upload.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=r.run_report(env, tmp_path/'management', once=True,
                           runner=lambda *a, **k:pytest.fail('lock bypassed'))
    assert result['status'] == 'EXISTING_REPORTER_REUSED'


def snapshot(env, *, now=1000):
    return dict(updated_utc='1970-01-01T00:16:40Z', sampled_unix=now,
                complete_tasks=1, total_tasks=2, status='CONTROLLER_ERROR',
                scheduler=dict(updated_at=100,controller_error={'error_type':'ValueError'}),
                uploads={'full':dict(status='RETRYING', updated_utc='1970-01-01T00:01:40Z',
                                     binary_representations=[{'secret':'not duplicated'}])})


def test_monitor_preserves_stale_queue_error_and_adds_actual_observer_evidence(env):
    env['runtime_module'].status=lambda *a, **k:snapshot(env)
    atomic(env['runtime']/'upload-full/observer-status.json',
           dict(recipe_sha256=env['header']['recipe_sha256'],status='OBSERVING',
                transport_activity='LEGACY_UPLOADER_ACTIVE', legacy_verified_event_utc='1970-01-01T00:16:39Z',
                observer_heartbeat_utc='1970-01-01T00:16:40Z',verified_groups=5,
                full_completion_claimed=False))
    value=r._monitor_snapshot(env, env['test_plan'], clock=lambda:1000)
    assert value['status'] == 'CONTROLLER_ERROR'
    assert value['scheduler_observation']['age_seconds'] == 900
    assert value['scheduler_observation']['fresh'] is False
    full=value['uploads']['full']
    assert full['status'] == 'RETRYING'
    assert full['observer']['transport_activity'] == 'LEGACY_UPLOADER_ACTIVE'
    assert full['observer']['full_completion_claimed'] is False
    assert full['observer_heartbeat_not_a_new_remote_verification'] is True
    assert 'binary_representations' not in full


def test_monitor_retries_network_errors_and_preserves_error_history_without_secrets(env, tmp_path):
    env['runtime_module'].status=lambda *a, **k:snapshot(env)
    stop=threading.Event(); calls=[]; snapshots=[]
    class Client:
        def git_publish(self, files, message):
            calls.append(1)
            if len(calls)==1:
                raise RuntimeError('secret-token-in-http-url')
            snapshots.append(json.loads(next(iter(files.values()))));stop.set();return 'commit'
    result=r.run_monitor(env,tmp_path/'management',stop=stop,interval=.01,
                          client_factory=Client,clock=lambda:1000)
    assert len(calls)==2 and result['status']=='STOPPED'
    error=read(tmp_path/'management/monitor-last-error.json')
    assert error['current_error']['error_type']=='RuntimeError'
    assert 'secret-token' not in json.dumps(error)
    assert snapshots[0]['scheduler_observation']['fresh'] is False
    assert read(tmp_path/'management/monitor-last-success.json')['commit']=='commit'


def test_fresh_verified_legacy_upload_progress_supersedes_old_cycle_error(env):
    env['runtime_module'].status=lambda *a, **k:snapshot(env)
    atomic(env['runtime']/'upload-full/observer-status.json',
           dict(recipe_sha256=env['header']['recipe_sha256'],status='OBSERVING_EXISTING_UPLOADER',
                current_transport_activity='LEGACY_UPLOADER_ACTIVE',
                last_verified_part_utc='1970-01-01T00:16:30Z',
                observer_heartbeat_utc='1970-01-01T00:16:39Z', observed_verified_groups=7,
                last_cycle_error=dict(error_type='URLError',updated_utc='1970-01-01T00:01:40Z')))
    value=r._monitor_snapshot(env,env['test_plan'],clock=lambda:1000)['uploads']['full']
    assert value['status']=='LEGACY_UPLOADER_ACTIVE_VERIFIED_PROGRESS'
    assert value['legacy_channel_receipt']['status']=='RETRYING'
    assert value['last_cycle_error_superseded_by_newer_remote_proof'] is True
    assert value['current_error'] is None
    assert value['updated_utc']=='1970-01-01T00:16:39Z'
    assert value['observer']['observed_verified_groups']==7
    assert value['full_completion_claimed_by_observer'] is False


@pytest.mark.parametrize('proof,error,heartbeat',[(None,'1970-01-01T00:01:40Z','1970-01-01T00:16:39Z'),
    ('1970-01-01T00:16:30Z','1970-01-01T00:16:35Z','1970-01-01T00:16:39Z'),
    ('1970-01-01T00:16:30Z','1970-01-01T00:01:40Z','1970-01-01T00:01:40Z')])
def test_heartbeat_or_older_proof_never_erases_current_upload_error(env,proof,error,heartbeat):
    state=dict(status='RETRYING',updated_utc=error,error_type='URLError')
    observer=dict(status='OBSERVING_EXISTING_UPLOADER',observer_heartbeat_utc=heartbeat,
                  last_verified_part_utc=proof,last_cycle_error=dict(updated_utc=error))
    r._effective_observer_status(state,observer,1000)
    assert state['status']=='UPLOAD_PROGRESS_UNCONFIRMED'
    assert state['last_cycle_error_superseded_by_newer_remote_proof'] is False
    assert state['legacy_channel_receipt']['status']=='RETRYING'


def test_actual_cpu_child_inherits_no_cuda_and_writes_owned_report(env, tmp_path, monkeypatch):
    stub=tmp_path/'stub';stub.mkdir()
    (stub/'report_child.py').write_text('''import json,os,sys
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['OMP_NUM_THREADS']=='2'
plan=json.loads(Path(sys.argv[sys.argv.index('--plan')+1]).read_text())
p=Path(plan['output'])/'reports/report-status.json';p.parent.mkdir(parents=True,exist_ok=True)
import hashlib
dg=lambda v:hashlib.sha256((json.dumps(v,sort_keys=True)+'\\n').encode()).hexdigest()
report=p.with_name('report.json');report.write_text(json.dumps(dict(recipe_sha256=plan['recipe_sha256'],coverage=dict(sealed_complete_tasks=1,complete_task_keys=['done'],complete_task_keys_sha256=dg(['done'])))))
p.write_text(json.dumps(dict(recipe_sha256=plan['recipe_sha256'],sealed_complete_tasks=1,total_tasks=2,status='PARTIAL',complete_task_keys_sha256=dg(['done']),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())))
''')
    monkeypatch.setattr(r,'HERE',stub)
    result=r.run_report(env,tmp_path/'management',once=True)
    assert result['status']=='PARTIAL_REPORT_GENERATED'
    assert result['report_complete_tasks']==1


def test_actual_native_cpu_report_bootstrap_with_signed_fixture(tmp_path):
    """Real original report code, real repaired child; no scientific fit."""
    import subprocess
    script = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
from paper3r11.platform import common as c,runtime
import environment,report_fix
package=Path(sys.argv[2]);root=Path(sys.argv[3]);root.mkdir()
output=root/'runs/r11-fixture';rt=output/'runtime';rt.mkdir(parents=True)
task=dict(key='reference',family='feature_reference',kind='fixture',seed=20261018,role='meta_train',requires=[])
plan=dict(schema_version=c.PLAN_SCHEMA,round=11,stage='development',synthetic_fixture=True,
 root=str(root),output=str(output),package=str(package),python=sys.executable,
 resources=dict(jobs=4,gpus_per_job=1,cpus_per_job=16,memory_per_job='64G'),tasks=[task],
 secondary_ablation_family=[],reference_strategy='fixture-fixed',
 code_manifest={'paper3r11/platform/common.py':c.sha(package/'paper3r11/platform/common.py')})
plan=c.sign_plan(plan);path=root/'plan.json';c.atomic(path,plan)
c.atomic(rt/'tasks.json',dict(recipe_sha256=plan['recipe_sha256'],tasks={}))
folder=output/task['key'];folder.mkdir();artifact=folder/'reference.json';artifact.write_text('{}')
c.atomic(folder/'result.json',dict(status='COMPLETE',scientific_scope='SOFTWARE_TEST_ONLY',
 recipe_sha256=plan['recipe_sha256'],task_key=task['key'],task_identity=c.task_identity(plan,task),
 artifacts=[dict(path=artifact.name,bytes=artifact.stat().st_size,sha256=c.sha(artifact))]))
runtime.finish_task(plan,task,0)
env=environment.load_env(package,path,expected_recipe=plan['recipe_sha256'])
result=report_fix.run_report(env,root/'management',once=True)
assert result['status']=='COMPLETE',result
assert result['report_complete_tasks']==1
report=c.read(output/'reports/report.json')
assert report['scientific_scope']=='SOFTWARE_TEST_ONLY' and report['coverage']['sealed_complete_tasks']==1
print(json.dumps(result))
'''
    package=Path('/workspace/paper3-round11')
    if not (package/'paper3r11/platform/common.py').is_file():
        pytest.skip('Original R11 source fixture is unavailable')
    run=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(ROOT),str(package),str(tmp_path/'actual')],
                       capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
    assert json.loads(run.stdout.splitlines()[-1])['training_controlled'] is False
