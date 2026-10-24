"""CPU-only operational regressions; subprocess Slurm calls are fakes."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace
import venv

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from paper3r3 import runtime, submit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def make_plan(tmp_path):
    def make(count=5, indices=None):
        output = tmp_path / 'runs' / 'followup_20261024_r3'
        output.mkdir(parents=True, exist_ok=True)
        indices = list(range(count)) if indices is None else list(indices)
        plan = {
            'version': 'test-round3', 'root': str(tmp_path), 'output': str(output),
            'output_root': str(output), 'recipe_sha256': 'cpu-test-recipe',
            'tasks': [{'id': str(i), 'task_id': str(i), 'index': i,
                       'key': f'task-{i}'} for i in indices],
        }
        plan_path = output / 'plan.json'
        write_json(plan_path, plan)
        return plan, plan_path
    return make


class Slurm:
    """Simple deterministic transport; never invokes school commands."""
    def __init__(self):
        self.commands = []
        self.queue = ''
        self.accounting = ''
        self.queue_error = False
        self.accounting_error = False
        self.submission_results = []
        self.next_job = 100
        self.submitted = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        name = Path(command[0]).name
        if name == 'squeue':
            return SimpleNamespace(returncode=int(self.queue_error), stdout=self.queue,
                                   stderr='queue unavailable' if self.queue_error else '')
        if name == 'sacct':
            return SimpleNamespace(returncode=int(self.accounting_error), stdout=self.accounting,
                                   stderr='accounting unavailable' if self.accounting_error else '')
        assert name == 'sbatch', command
        index = int(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--array=')))
        name_arg = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--job-name='))
        result = (self.submission_results.pop(0) if self.submission_results else
                  SimpleNamespace(returncode=0, stdout=f'{self.next_job}\n', stderr=''))
        self.next_job += 1
        if isinstance(result, Exception):
            raise result
        job = result.stdout.strip().split(';')[0]
        self.submitted.append({'job': job, 'index': index, 'name': name_arg, 'command': list(command)})
        return result

    def queue_rows(self, marker, submitted=None, state='RUNNING'):
        rows = self.submitted if submitted is None else submitted
        return ''.join(f"{r['job']}_{r['index']}|{marker}|{state}|{r['name']}\n" for r in rows)

    def outcome(self, marker, row, state='FAILED', exit_code='1:0'):
        self.accounting = f"{row['job']}_{row['index']}|{state}|{exit_code}|{marker}|{row['name']}\n"
        self.queue = ''


def controller(plan_path, slurm, **launch):
    return runtime.RollingController(plan_path, runner=slurm,
        launch={'python': sys.executable, 'package': str(PACKAGE), **launch})


def test_sbatch_uses_one_preserved_index_and_school_resources(make_plan):
    plan, _ = make_plan(indices=[7, 12])
    command = submit.build_sbatch_command(plan, sys.executable, 12)
    assert '--array=12' in command
    assert '--partition=hpc_gpu' in command
    assert '--cpus-per-task=12' in command
    assert '--mem=64G' in command
    assert '--gres=gpu:1' in command
    assert not any(',' in c or '%' in c for c in command if c.startswith('--array='))
    assert '--comment=' + runtime.job_marker(plan) in command
    typed = submit.build_sbatch_command(plan, sys.executable, 7,
        launch={'gres': 'gpu:NVIDIAA100-PCIE-40GB:1'})
    assert '--gres=gpu:NVIDIAA100-PCIE-40GB:1' in typed


def test_queue_and_accounting_filter_exact_owned_marker():
    marker = 'ours'
    queue = runtime.parse_queue('10_7|ours|RUNNING|paper3r3-7-a1\n'
                                '11_3|other|RUNNING|paper3r3-3-a1\n', marker)
    assert len(queue) == 1 and queue[0]['index'] == 7
    rows = runtime.parse_accounting('10_7|FAILED|75:0|ours|paper3r3-7-a1\n'
        '10_7.batch|FAILED|75:0|ours|batch\n'
        '11_3|FAILED|1:0|other|paper3r3-3-a1\n', marker)
    assert len(rows) == 1 and rows[0]['index'] == 7
    assert str(rows[0]['exit_code']).split(':')[0] == '75'


def test_rolls_at_most_three_independent_jobs_and_fills_one_slot(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    assert [r['index'] for r in fake.submitted] == [0, 1, 2]
    assert len({r['job'] for r in fake.submitted}) == 3
    marker = runtime.job_marker(plan)
    fake.queue = fake.queue_rows(marker, fake.submitted[1:])
    write_json(Path(plan['output']) / 'task-0' / 'result.json', {'status': 'COMPLETE'})
    ctl.step()
    assert [r['index'] for r in fake.submitted] == [0, 1, 2, 3]
    assert ctl.state['tasks']['0']['phase'] == 'COMPLETE'
    assert (Path(plan['output']) / 'runtime' / 'state.json').is_file()
    assert list((Path(plan['output']) / 'runtime' / 'receipts').glob('*.json'))


def test_restart_reconciles_owned_active_jobs_without_duplicate_submit(make_plan):
    plan, path = make_plan(count=3)
    fake = Slurm()
    controller(path, fake).step()
    fake.queue = fake.queue_rows(runtime.job_marker(plan))
    restarted = controller(path, fake)
    restarted.step()
    assert len(fake.submitted) == 3
    assert all(row['phase'] == 'ACTIVE' for row in restarted.state['tasks'].values())


@pytest.mark.parametrize('field', ['queue_error', 'accounting_error'])
def test_scheduler_observation_failure_blocks_new_submissions(make_plan, field):
    _, path = make_plan()
    fake = Slurm()
    setattr(fake, field, True)
    controller(path, fake).step()
    assert fake.submitted == []


def test_lost_ack_is_reconciled_from_accounting_without_duplicate(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=0, stdout='', stderr='')]
    ctl = controller(path, fake)
    ctl.step()
    assert ctl.state['tasks']['0']['phase'] == 'UNCERTAIN'
    name = fake.submitted[0]['name']
    marker = runtime.job_marker(plan)
    fake.accounting = f'999_0|RUNNING|0:0|{marker}|{name}\n'
    controller(path, fake).step()
    assert len(fake.submitted) == 1


def test_uncertain_submission_never_retries_on_absence_alone(make_plan):
    _, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=0, stdout='', stderr='')]
    ctl = controller(path, fake)
    ctl.step()
    for _ in range(3):
        ctl.step()
    assert len(fake.submitted) == 1
    assert ctl.state['tasks']['0']['phase'] == 'UNCERTAIN'


def test_socket_error_is_uncertain_even_with_generic_submission_failed_prefix(make_plan):
    _, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=1, stdout='',
        stderr='sbatch: error: Batch job submission failed: Socket timed out on send/recv operation')]
    ctl = controller(path, fake)
    ctl.step()
    ctl.step()
    assert len(fake.submitted) == 1
    assert ctl.state['tasks']['0']['phase'] == 'UNCERTAIN'
    assert ctl.summary()['inflight_jobs'] == 1


def test_receipt_owns_accounting_without_optional_stored_comment(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=0, stdout='', stderr='')]
    ctl = controller(path, fake)
    ctl.step()
    name = fake.submitted[0]['name']
    fake.accounting = f'999_0|RUNNING|0:0||{name}\n888_0|FAILED|1:0||unrelated-name\n'
    recovered = controller(path, fake)
    recovered.step()
    assert len(fake.submitted) == 1
    assert recovered.state['tasks']['0']['attempts'][0]['job_id'] == '999_0'


def test_stop_recorded_during_submit_prevents_later_submissions(make_plan):
    plan, path = make_plan(count=3)
    fake = Slurm()
    def runner(command, **kwargs):
        result = fake(command, **kwargs)
        if command[0] == 'sbatch':
            (Path(plan['output']) / 'runtime' / 'STOP').touch()
        return result
    ctl = runtime.RollingController(path, runner=runner)
    ctl.step()
    assert len(fake.submitted) == 1
    assert ctl.summary()['controller_status'] == 'STOPPED'


def test_dependencies_submit_only_when_complete_and_ready_siblings_continue(make_plan):
    plan, path = make_plan(count=3)
    plan['tasks'][1]['requires'] = ['task-0']
    write_json(path, plan)
    fake = Slurm()
    ctl = controller(path, fake, max_inflight=2)
    ctl.step()
    assert [row['index'] for row in fake.submitted] == [0, 2]
    assert ctl.state['tasks']['1']['phase'] == 'WAITING_INPUT'
    write_json(Path(plan['output']) / 'task-0' / 'result.json', {'status': 'COMPLETE'})
    fake.queue = fake.queue_rows(runtime.job_marker(plan), fake.submitted[1:])
    ctl.step()
    assert [row['index'] for row in fake.submitted] == [0, 2, 1]


def test_failed_input_ends_dependency_wait_without_cancelling_other_tasks(make_plan):
    plan, path = make_plan(count=3)
    plan['tasks'][1]['requires'] = ['task-0']
    write_json(path, plan)
    fake = Slurm()
    ctl = controller(path, fake, max_inflight=2)
    ctl.step()
    write_json(Path(plan['output']) / 'task-0' / 'progress.json', {'status': 'TRAINING'})
    fake.outcome(runtime.job_marker(plan), fake.submitted[0])
    fake.queue = fake.queue_rows(runtime.job_marker(plan), fake.submitted[1:])
    ctl.step()
    assert ctl.state['tasks']['0']['phase'] == 'FAILED'
    assert ctl.state['tasks']['1']['phase'] == 'WAITING_INPUT'
    assert ctl.state['tasks']['2']['phase'] == 'ACTIVE'
    write_json(Path(plan['output']) / 'task-2' / 'result.json', {'status': 'COMPLETE'})
    fake.queue = ''
    ctl.step()
    assert ctl.summary()['controller_status'] == 'NOT_COMPLETE'
    assert len(fake.submitted) == 2


def test_default_quota_policy_keeps_waiting_with_capped_backoff(make_plan):
    _, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=1, stdout='',
        stderr='sbatch: error: Batch job submission failed: QOSMaxSubmitJobPerUserLimit') for _ in range(12)]
    now = [1000000.0]
    ctl = runtime.RollingController(path, runner=fake, clock=lambda: now[0])
    for _ in range(12):
        ctl.step()
        assert ctl.summary()['controller_status'] == 'WAITING_FOR_CAPACITY'
        assert 0 < ctl.state['next_submit_at'] - now[0] <= 300
        now[0] += 1000
    assert len(fake.submitted) == 12
    assert not ctl.state.get('submission_blocked')


def test_background_monitor_starts_before_controller_without_capacity_check(make_plan, monkeypatch):
    plan, path = make_plan(count=1)
    calls = []
    def spawn(command, log):
        calls.append((command, log))
        return 1000 + len(calls)
    monkeypatch.setattr(submit, '_spawn', spawn)
    receipt = submit.start_background(path, {'python': sys.executable}, credentials='/private/credentials.json')
    assert 'monitor' in calls[0][0]
    assert 'controller' in calls[1][0]
    assert '--credentials' in calls[0][0]
    assert receipt['monitor_pid'] == 1001
    assert receipt['controller_pid'] == 1002


def test_cancelled_task_is_not_revived_and_other_task_continues(make_plan):
    plan, path = make_plan(count=2)
    fake = Slurm()
    ctl = controller(path, fake, max_inflight=1)
    ctl.step()
    fake.outcome(runtime.job_marker(plan), fake.submitted[0], state='CANCELLED by 123', exit_code='0:15')
    ctl.step()
    assert ctl.state['tasks']['0']['phase'] == 'CANCELLED'
    assert [r['index'] for r in fake.submitted] == [0, 1]
    assert not any(Path(c[0]).name == 'scancel' for c in fake.commands)


def test_stop_file_blocks_new_work(make_plan):
    plan, path = make_plan()
    (Path(plan['output']) / 'STOP').write_text('operator requested stop\n')
    fake = Slurm()
    controller(path, fake).step()
    assert fake.submitted == []


def test_hard_failure_does_not_cancel_or_block_sibling(make_plan):
    plan, path = make_plan(count=2)
    fake = Slurm()
    ctl = controller(path, fake, max_inflight=1)
    ctl.step()
    write_json(Path(plan['output']) / 'task-0' / 'progress.json', {'status': 'TRAINING', 'outer_update': 4})
    fake.outcome(runtime.job_marker(plan), fake.submitted[0])
    ctl.step()
    assert ctl.state['tasks']['0']['phase'] == 'FAILED'
    assert [r['index'] for r in fake.submitted] == [0, 1]
    assert not any(Path(c[0]).name == 'scancel' for c in fake.commands)


def test_startup_failures_have_a_finite_retry_budget(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    ctl = controller(path, fake, max_startup_retries=2)
    ctl.step()
    for _ in range(4):
        fake.outcome(runtime.job_marker(plan), fake.submitted[-1], state='BOOT_FAIL')
        ctl.step()
    assert len(fake.submitted) == 3
    assert ctl.state['tasks']['0']['phase'] == 'FAILED'


def test_checkpoint_75_resumes_but_is_bounded(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    ctl = controller(path, fake, max_resumes=1)
    ctl.step()
    write_json(Path(plan['output']) / 'task-0' / 'progress.json', {'status': 'CHECKPOINTED'})
    fake.outcome(runtime.job_marker(plan), fake.submitted[0], exit_code='75:0')
    ctl.step()
    assert len(fake.submitted) == 2
    fake.outcome(runtime.job_marker(plan), fake.submitted[-1], exit_code='75:0')
    ctl.step()
    assert len(fake.submitted) == 2
    assert ctl.state['tasks']['0']['phase'] == 'FAILED'


def test_completed_slurm_job_without_result_is_not_science_complete(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    ctl = controller(path, fake, max_startup_retries=0)
    ctl.step()
    fake.outcome(runtime.job_marker(plan), fake.submitted[0], state='COMPLETED', exit_code='0:0')
    ctl.step()
    assert ctl.state['tasks']['0']['phase'] != 'COMPLETE'


def test_entry_exposes_venv_packages_without_running_site_hooks(tmp_path):
    virtual = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(virtual)
    site = virtual / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    site.mkdir(parents=True, exist_ok=True)
    marker = tmp_path / 'unwanted-hook'
    (site / 'sitecustomize.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\n')
    (site / 'unwanted.pth').write_text('import pathlib; pathlib.Path('+repr(str(marker))+').touch()\n')
    (site / 'round3_probe_dependency.py').write_text('VALUE="existing-venv-dependency"\n')
    package = tmp_path / 'isolated-package'
    (package / 'paper3r3').mkdir(parents=True)
    shutil.copyfile(PACKAGE / 'paper3r3' / 'entry.py', package / 'paper3r3' / 'entry.py')
    (package / 'paper3r3' / '__init__.py').write_text('')
    (package / 'paper3r3' / 'worker.py').write_text('import round3_probe_dependency\ndef main():\n    print(round3_probe_dependency.VALUE)\n    return 0\n')
    result = subprocess.run([str(virtual / 'bin/python'), '-I', '-S', '-B',
        str(package / 'paper3r3' / 'entry.py'), 'worker'],
        capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == 'existing-venv-dependency'
    assert not marker.exists()


@pytest.mark.parametrize('signal_name,stop_reason,stopped', [
    ('SIGTERM', 'terminated', True), ('SIGUSR1', 'time_limit', False),
])
def test_bash42_forwards_checkpoint_signals_and_records_intent(tmp_path, signal_name, stop_reason, stopped):
    bash = Path('/workspace/paper3-bash42-portability/bash-4.2/bash')
    if not bash.is_file():
        pytest.skip('Bash 4.2 fixture not installed')
    package = tmp_path / 'package'
    (package / 'paper3r3').mkdir(parents=True)
    (package / 'paper3r3' / 'entry.py').write_text(
        'import signal,time,sys\n'
        'signal.signal(signal.SIGUSR1,lambda *_:sys.exit(75))\n'
        'print("READY",flush=True)\n'
        'while True: time.sleep(.05)\n')
    output = tmp_path / 'output'
    output.mkdir()
    env = {**os.environ, 'SLURM_ARRAY_TASK_ID': '7',
           'SLURM_JOB_ID': '123', 'SLURM_ARRAY_JOB_ID': '123'}
    process = subprocess.Popen([str(bash), str(PACKAGE / 'train.sbatch'),
        sys.executable, str(package), 'unused-plan', str(output)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        assert process.stdout.readline().startswith('TRAIN_PROCESS_STARTED')
        assert process.stdout.readline().strip() == 'READY'
        process.send_signal(getattr(signal, signal_name))
        process.communicate(timeout=10)
        assert process.returncode == 75
        outcome = json.loads((output / 'runtime' / 'outcomes' / '123_7.json').read_text())
        assert outcome == {'exit_code': 75, 'stop_reason': stop_reason}
        assert (output / 'runtime' / 'task-7.stop').exists() is stopped
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_receipts_recover_when_controller_state_write_was_lost(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    controller(path, fake).step()
    (Path(plan['output']) / 'runtime' / 'state.json').unlink()
    fake.queue = fake.queue_rows(runtime.job_marker(plan))
    recovered = controller(path, fake)
    recovered.step()
    assert len(fake.submitted) == 1
    assert recovered.state['tasks']['0']['phase'] == 'ACTIVE'


def test_explicit_qos_rejection_is_bounded_and_records_actionable_status(make_plan):
    _, path = make_plan(count=2)
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=1, stdout='',
        stderr='sbatch: error: Batch job submission failed: QOSMaxSubmitJobPerUserLimit') for _ in range(6)]
    seconds = [0]
    ctl = runtime.RollingController(path, runner=fake, clock=lambda: seconds[0],
        launch={'python': sys.executable, 'package': str(PACKAGE), 'submit_reject_limit': 2})
    for _ in range(5):
        ctl.step()
        seconds[0] += 1000
    assert len([c for c in fake.commands if c[0] == 'sbatch']) == 2
    assert ctl.state['controller_status'] == 'SUBMISSION_BLOCKED'
    assert ctl.state['submission_blocked']


def test_timeout_after_submit_reserves_slot_without_blind_retry(make_plan):
    _, path = make_plan(count=1)
    fake = Slurm()
    fake.submission_results = [subprocess.TimeoutExpired('sbatch', 45)]
    ctl = controller(path, fake)
    ctl.step()
    ctl.step()
    assert len([c for c in fake.commands if c[0] == 'sbatch']) == 1
    assert ctl.state['tasks']['0']['phase'] == 'UNCERTAIN'


def test_terminated_checkpoint_marker_prevents_automatic_resume(make_plan):
    plan, path = make_plan(count=1)
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    (Path(plan['output']) / 'runtime' / 'task-0.stop').touch()
    fake.outcome(runtime.job_marker(plan), fake.submitted[0], exit_code='75:0')
    ctl.step()
    assert len(fake.submitted) == 1
    assert ctl.state['tasks']['0']['phase'] == 'CANCELLED'


@pytest.fixture
def launcher_plan_stub(monkeypatch):
    calls = []
    def build(mode, root, output, *extra, **resources):
        calls.append((mode, root, output, extra, resources))
        return {'root': str(root), 'output': str(output), 'recipe_sha256': 'stub-' + mode,
                'tasks': [{'key': 'task-0'}]}
    module = SimpleNamespace(
        make_plan=lambda root, output, **kwargs: build('core', root, output, **kwargs),
        make_external_plan=lambda root, output, manifest, basis, **kwargs: build(
            'external', root, output, manifest, basis, **kwargs))
    monkeypatch.setitem(sys.modules, 'paper3r3.plan', module)
    monkeypatch.setattr(submit, 'discover_credentials', lambda explicit=None: None)
    return calls


def test_prepare_core_does_not_require_optional_external_files(tmp_path, launcher_plan_stub):
    assert submit.main(['--root', str(tmp_path), '--python', sys.executable]) == 0
    assert launcher_plan_stub[0][0] == 'core'
    assert (tmp_path / 'runs/followup_20261024_r3/plan.json').is_file()
    assert not (tmp_path / 'runs/followup_20261024_r3_weibo').exists()


def test_prepare_external_uses_manifest_basis_and_its_separate_default(tmp_path, launcher_plan_stub):
    manifest = tmp_path / 'weibo.json'
    manifest.write_text('{}')
    basis = tmp_path / 'runs/followup_20261024_r3'
    assert submit.main(['--root', str(tmp_path), '--python', sys.executable,
                        '--external-manifest', str(manifest), '--basis-run', str(basis)]) == 0
    mode, root, output, extra, resources = launcher_plan_stub[0]
    assert mode == 'external'
    assert extra == (manifest.resolve(), basis.resolve())
    assert resources == {'gpus': 3, 'cpus': 12}
    assert output == tmp_path / 'runs/followup_20261024_r3_weibo'
    assert (output / 'plan.json').is_file()
    assert not basis.exists()


def test_prepare_external_missing_manifest_fails_only_that_command(tmp_path, launcher_plan_stub):
    with pytest.raises(SystemExit) as exc:
        submit.main(['--root', str(tmp_path), '--python', sys.executable,
                     '--external-manifest', str(tmp_path / 'missing.json'),
                     '--basis-run', str(tmp_path / 'core')])
    assert exc.value.code == 2
    assert launcher_plan_stub == []
    assert not (tmp_path / 'runs/followup_20261024_r3_weibo').exists()
    assert submit.main(['--root', str(tmp_path), '--python', sys.executable]) == 0
    assert launcher_plan_stub[0][0] == 'core'


def test_prepare_accepts_custom_new_output_name(tmp_path, launcher_plan_stub):
    output = tmp_path / 'runs/new_round3_recipe'
    assert submit.main(['--root', str(tmp_path), '--python', sys.executable,
                        '--output', str(output)]) == 0
    assert (output / 'plan.json').is_file()


@pytest.mark.parametrize('name', ['source_training_v2', 'meta', 'training_campaign_v1', 'followup_20261023_v2'])
def test_prepare_protects_listed_earlier_outputs(tmp_path, launcher_plan_stub, name):
    output = tmp_path / 'runs' / name
    with pytest.raises(SystemExit) as exc:
        submit.main(['--root', str(tmp_path), '--python', sys.executable,
                     '--output', str(output)])
    assert exc.value.code == 2
    assert not output.exists()
    assert launcher_plan_stub == []


@pytest.mark.parametrize('flag', ['--external-manifest', '--basis-run'])
def test_external_launcher_flags_require_each_other(tmp_path, launcher_plan_stub, flag):
    with pytest.raises(SystemExit) as exc:
        submit.main(['--root', str(tmp_path), '--python', sys.executable, flag, str(tmp_path / 'input')])
    assert exc.value.code == 2
    assert launcher_plan_stub == []
