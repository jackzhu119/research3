"""Single-allocation orchestration checks; all Slurm traffic is simulated."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from paper3r7 import runtime, submit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def make_plan(tmp_path):
    def build(count=5):
        output = tmp_path / 'runs/followup_20261024_r7'
        output.mkdir(parents=True, exist_ok=True)
        plan = {'version': 'test-round7-single-allocation', 'root': str(tmp_path),
                'output': str(output), 'recipe_sha256': 'synthetic-r7-runtime-recipe',
                'tasks': [{'key': f'task-{index}'} for index in range(count)]}
        path = output / 'plan.json'
        write_json(path, plan)
        return plan, path
    return build


class Slurm:
    """No real scheduler or compute node is called."""
    def __init__(self):
        self.commands = []
        self.queue = ''
        self.accounting = ''
        self.queue_error = self.accounting_error = False
        self.submission_results = []
        self.next_job = 500
        self.accepted = []

    @property
    def sbatch_calls(self):
        return [command for command in self.commands if Path(command[0]).name == 'sbatch']

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        name = Path(command[0]).name
        if name == 'squeue':
            return SimpleNamespace(returncode=int(self.queue_error), stdout=self.queue,
                                   stderr='synthetic queue failure' if self.queue_error else '')
        if name == 'sacct':
            return SimpleNamespace(returncode=int(self.accounting_error), stdout=self.accounting,
                                   stderr='synthetic accounting failure' if self.accounting_error else '')
        assert name == 'sbatch', command
        assert not any(value.startswith('--array') for value in command)
        result = self.submission_results.pop(0) if self.submission_results else SimpleNamespace(
            returncode=0, stdout=str(self.next_job) + '\n', stderr='')
        self.next_job += 1
        if isinstance(result, Exception):
            raise result
        if result.returncode == 0 and result.stdout.strip():
            self.accepted.append({'job_id': result.stdout.strip().split(';')[0],
                'name': next(value.split('=', 1)[1] for value in command if value.startswith('--job-name='))})
        return result

    def running(self, marker, state='RUNNING', row=None):
        row = row or self.accepted[-1]
        self.queue = f"{row['job_id']}|{marker}|{state}|{row['name']}|Resources\n"

    def finish(self, marker, state='FAILED', code='75:0', row=None):
        row = row or self.accepted[-1]
        self.queue = ''
        self.accounting = f"{row['job_id']}|{state}|{code}|{marker}|{row['name']}\n"


def controller(path, fake, **launch):
    return runtime.SingleAllocationController(path, runner=fake,
        launch={'python': sys.executable, 'package': str(PACKAGE), **launch})


def test_one_node_three_gpu_request_has_36_cpu_and_128g_total(make_plan):
    plan, _ = make_plan()
    command = submit.build_sbatch_command(plan, sys.executable)
    assert runtime.job_marker(plan).startswith('paper3r7-')
    for value in ('--partition=hpc_gpu', '--nodes=1', '--ntasks=1', '--cpus-per-task=36',
                  '--mem=128G', '--gres=gpu:3', '--no-requeue', '--signal=B:USR1@300'):
        assert value in command
    assert not any(value.startswith(('--array', '--nodelist')) for value in command)
    assert '--comment=' + runtime.job_marker(plan) in command
    assert any(value.endswith('/logs/train-%j.out') for value in command)
    assert any(value.endswith('/logs/train-%j.err') for value in command)
    assert str(PACKAGE / 'train.sbatch') in command
    assert '--mem=126G' not in command and '--mem=129G' not in command
    assert runtime.DEFAULT_LAUNCH['cpus'] == 12
    assert runtime.DEFAULT_LAUNCH['memory_total'] == '128G'
    assert 'memory' not in runtime.DEFAULT_LAUNCH


def test_total_memory_is_not_multiplied_by_gpu_count(make_plan):
    plan, _ = make_plan()
    assert submit.allocation_memory('128G') == '128G'
    assert submit.allocation_memory('128G', 3) == '128G'
    assert submit.allocation_memory('120g', 3) == '120G'
    command = submit.build_sbatch_command(plan, sys.executable, {'memory_total':'120G'})
    assert '--mem=120G' in command
    assert not any(value in command for value in ('--mem=360G','--mem=192G'))


@pytest.mark.parametrize('invalid', ['0G','-1G','164','164.5G','abc','64G;echo unsafe'])
def test_invalid_total_memory_is_rejected_before_submission(make_plan,invalid):
    plan, _ = make_plan()
    with pytest.raises(ValueError):
        submit.build_sbatch_command(plan,sys.executable,{'memory_total':invalid})


def test_cli_custom_total_memory_prepares_exact_total(tmp_path,capsys):
    assert submit.main(['--root',str(tmp_path),'--python',sys.executable,
                        '--cpus','12','--memory-total','120G']) == 0
    output=tmp_path/'runs/followup_20261024_r7'
    launch=json.loads((output/'runtime/launch.json').read_text())
    assert launch['memory_total']=='120G' and 'memory' not in launch
    prepared=json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert prepared['cpus_per_job']==36 and prepared['host_memory_per_job']=='120G'
    assert prepared['host_memory_per_gpu_budget_gib'] == pytest.approx(40.)
    assert 'shared' in prepared['host_memory_scope'].lower()
    assert 'host_memory_per_gpu' not in prepared
    assert submit.allocation_memory(launch['memory_total'],launch['slots'])=='120G'


def test_old_per_slot_memory_flag_is_not_accepted_as_total_memory_abbreviation(tmp_path):
    with pytest.raises(SystemExit) as error:
        submit.main(['--root',str(tmp_path),'--python',sys.executable,'--memory','64G'])
    assert error.value.code == 2
    assert not (tmp_path/'runs').exists()


def test_single_pending_allocation_owns_three_slots_without_duplicate_submit(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    assert len(fake.sbatch_calls) == 1
    fake.running(runtime.job_marker(plan), state='PENDING')
    for _ in range(3):
        ctl.step()
        controller(path, fake).step()
    assert len(fake.sbatch_calls) == 1
    assert ctl.state['controller_status'] == 'QUEUED'


def test_running_allocation_is_adopted_after_controller_restart(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    controller(path, fake).step()
    fake.running(runtime.job_marker(plan))
    fake.queue += '999|paper3r3-other-run|RUNNING|paper3r3-other-run-a1|Resources\n'
    recovered = controller(path, fake)
    recovered.step()
    assert len(fake.sbatch_calls) == 1
    assert recovered.state['controller_status'] == 'ALLOCATION_RUNNING'
    assert recovered.state['attempts'][-1]['job_id'] == fake.accepted[-1]['job_id']


def test_lost_ack_is_adopted_by_exact_owned_accounting_name(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=0, stdout='', stderr='')]
    ctl = controller(path, fake)
    ctl.step()
    name = next(value.split('=', 1)[1] for value in fake.sbatch_calls[0] if value.startswith('--job-name='))
    fake.accounting = f'900|RUNNING|0:0||{name}\n'
    recovered = controller(path, fake)
    recovered.step()
    assert len(fake.sbatch_calls) == 1
    assert recovered.state['attempts'][-1]['job_id'] == '900'
    assert recovered.state['controller_status'] == 'ALLOCATION_RUNNING'


def test_unknown_submission_absence_never_authorizes_duplicate(make_plan):
    _, path = make_plan()
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=0, stdout='', stderr='')]
    ctl = controller(path, fake)
    for _ in range(4):
        ctl.step()
    assert len(fake.sbatch_calls) == 1
    assert ctl.state['controller_status'] == 'SUBMISSION_UNCERTAIN'


def test_sbatch_socket_timeout_reserves_single_allocation(make_plan):
    _, path = make_plan()
    fake = Slurm()
    fake.submission_results = [subprocess.TimeoutExpired('sbatch', 30)]
    ctl = controller(path, fake)
    ctl.step()
    ctl.step()
    assert len(fake.sbatch_calls) == 1
    assert ctl.state['controller_status'] == 'SUBMISSION_UNCERTAIN'


def test_qos_capacity_rejection_waits_and_retries_only_one_job(make_plan):
    _, path = make_plan()
    fake = Slurm()
    fake.submission_results = [SimpleNamespace(returncode=1, stdout='',
        stderr='sbatch: error: QOSMaxSubmitJobPerUserLimit') for _ in range(3)]
    clock = [0]
    ctl = runtime.SingleAllocationController(path, runner=fake, clock=lambda: clock[0],
        launch={'python': sys.executable, 'package': str(PACKAGE)})
    for _ in range(3):
        ctl.step()
        clock[0] += 1000
    assert len(fake.sbatch_calls) == 3
    assert ctl.state['controller_status'] == 'WAITING_FOR_CAPACITY'
    assert ctl.state['submit_rejections'] == 3


def test_checkpoint_exit75_resubmits_one_allocation_to_continue(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    fake.finish(runtime.job_marker(plan), code='75:0')
    ctl.step()
    assert len(fake.sbatch_calls) == 2
    assert len(ctl.state['attempts']) == 2


def test_explicit_cancel_does_not_revive_allocation(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    fake.finish(runtime.job_marker(plan), state='CANCELLED', code='0:15')
    ctl.step()
    controller(path, fake).step()
    assert len(fake.sbatch_calls) == 1
    assert ctl.state['controller_status'] == 'CANCELLED'


def test_all_task_results_complete_finishes_without_new_allocation(make_plan):
    plan, path = make_plan()
    for task in plan['tasks']:
        write_json(Path(plan['output']) / task['key'] / 'result.json', {'status': 'COMPLETE'})
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    assert fake.sbatch_calls == []
    assert ctl.state['controller_status'] == 'FINISHED'


def test_slurm_completed_without_scientific_results_is_not_complete(make_plan):
    plan, path = make_plan()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    fake.finish(runtime.job_marker(plan), state='COMPLETED', code='0:0')
    ctl.step()
    assert len(fake.sbatch_calls) == 1
    assert ctl.state['controller_status'] == 'NOT_COMPLETE'


@pytest.mark.parametrize('field', ['queue_error', 'accounting_error'])
def test_observation_failure_does_not_submit_or_duplicate(make_plan, field):
    _, path = make_plan()
    fake = Slurm()
    setattr(fake, field, True)
    controller(path, fake).step()
    assert fake.sbatch_calls == []


def test_stop_file_is_respected_before_first_submission(make_plan):
    plan, path = make_plan()
    (Path(plan['output']) / 'STOP').touch()
    fake = Slurm()
    ctl = controller(path, fake)
    ctl.step()
    assert fake.sbatch_calls == []
    assert ctl.state['controller_status'] == 'STOPPED'


@pytest.mark.parametrize('base_name', ['paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5', 'paper3-round6', 'paper3-round7'])
def test_discover_existing_credential_readonly(tmp_path, monkeypatch, base_name):
    monkeypatch.delenv('PAPER3_GITHUB_CREDENTIALS', raising=False)
    monkeypatch.setattr(submit.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(submit, 'PACKAGE', tmp_path / 'new-package')
    path = tmp_path / base_name / 'install/private/github.json'
    write_json(path, {'token': 'synthetic-test-only'})
    before = path.read_bytes()
    assert submit.discover_credentials() == path.resolve()
    assert path.read_bytes() == before


@pytest.mark.parametrize('name', ['followup_20261024_r3', 'followup_20261024_r3_weibo',
    'followup_20261024_r4', 'followup_20261024_r4_weibo',
    'followup_20261024_r5', 'followup_20261024_r5_weibo',
    'followup_20261024_r6', 'followup_20261024_r6_external', 'followup_20261024_r6_weibo'])
def test_prior_round_results_cannot_be_used_as_round7_output(tmp_path, name):
    with pytest.raises(SystemExit) as error:
        submit.main(['--root', str(tmp_path), '--python', sys.executable,
                     '--output', str(tmp_path / 'runs' / name)])
    assert error.value.code == 2
    assert not (tmp_path / 'runs' / name).exists()


@pytest.mark.parametrize('delivered_signal,reason,stopped', [
    (signal.SIGUSR1, 'time_limit', False), (signal.SIGTERM, 'terminated', True)])
def test_actual_bash42_forwards_whole_allocation_checkpoint_signal(tmp_path, delivered_signal, reason, stopped):
    bash = Path('/workspace/paper3-bash42-portability/bash-4.2/bash')
    if not bash.is_file():
        pytest.skip('The separately compiled Bash 4.2 validation binary is unavailable')
    package = tmp_path / 'package'
    (package / 'paper3r7').mkdir(parents=True)
    output = tmp_path / 'output'
    output.mkdir()
    ready, saved = output / 'ready', output / 'saved'
    (package / 'paper3r7/entry.py').write_text(
        'import pathlib,signal,time,sys\n'
        f'ready=pathlib.Path({str(ready)!r});saved=pathlib.Path({str(saved)!r})\n'
        'def stop(*args):\n    saved.write_text("checkpoint");sys.exit(75)\n'
        'signal.signal(signal.SIGUSR1,stop)\nready.write_text("allocation ready")\n'
        'while True: time.sleep(.02)\n')
    env = {**os.environ, 'SLURM_JOB_ID': '880001'}
    command = [str(bash), str(PACKAGE / 'train.sbatch'), sys.executable,
               str(package), str(tmp_path / 'plan.json'), str(output), '3', '12']
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), 'Allocation stub did not start'
        process.send_signal(delivered_signal)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 75, stdout + stderr
        assert saved.is_file()
        assert (output / 'runtime/STOP').exists() is stopped
        assert json.loads((output / 'runtime/outcomes/880001.json').read_text()) == {
            'exit_code': 75, 'stop_reason': reason}
        assert 'slots=3' in stdout and 'CHECKPOINT_SAVED' in stdout
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)


def test_round7_monitor_namespace_and_external_routes_are_independent():
    from paper3r7 import monitor
    assert monitor.BRANCH == 'paper3-r7-progress'
    assert monitor.status_path_for_plan({}) == 'monitoring/paper3-round7/status.json'
    assert monitor.status_path_for_plan({'external_manifest':'new_dataset.json'}) == 'monitoring/paper3-round7/external-status.json'
    assert monitor.status_path_for_plan({'external_manifest':'weibo.json','external_dataset':'weibo'}) == 'monitoring/paper3-round7/weibo-status.json'



def test_monitor_uses_plan_evidence_scope_without_claiming_external_chronology(tmp_path, monkeypatch):
    from paper3r7 import monitor, report
    plan = {'output':str(tmp_path), 'recipe_sha256':'synthetic-evidence-scope',
            'tasks':[], 'exploratory_internal_replication':False, 'fresh_confirmatory':False,
            'interpretation':'new prespecified external evaluation; published order only'}
    monkeypatch.setattr(report, 'build_report', lambda plan: {'status':'PARTIAL','tables':{}})
    monitor.cycle(plan, None)
    value = json.loads((tmp_path/'status.json').read_text())
    assert value['exploratory_internal_replication'] is False
    assert value['fresh_confirmatory'] is False
    assert value['interpretation'] == plan['interpretation']


def _same_recipe_plan_factory(monkeypatch, tmp_path):
    from paper3r7 import plan as scientific
    def make(root, output, **_):
        return {'root':str(Path(root).resolve()),'output':str(Path(output).resolve()),
                'recipe_sha256':'same-content-and-numerical-recipe',
                'tasks':[{'key':'task-one','family':'transfer','config':{'radius':.15},
                          'external_manifest':str(submit.PACKAGE/'data/pan2023/manifest.json')}]}
    monkeypatch.setattr(scientific,'make_plan',make)
    monkeypatch.setattr(submit,'discover_credentials',lambda *_:None)
    first=tmp_path/'package-a';second=tmp_path/'package-b'
    first.mkdir();second.mkdir()
    monkeypatch.setattr(submit,'PACKAGE',first)
    return first,second,make


def test_same_recipe_repackage_preserves_prepared_tasks_and_completed_locators(tmp_path,monkeypatch):
    first,second,_=_same_recipe_plan_factory(monkeypatch,tmp_path)
    args=['--root',str(tmp_path),'--python',sys.executable]
    assert submit.main(args)==0
    output=tmp_path/'runs/followup_20261024_r7'
    original=(output/'plan.json').read_bytes()
    monkeypatch.setattr(submit,'PACKAGE',second)
    assert submit.main(args)==0
    assert (output/'plan.json').read_bytes()==original
    task=json.loads(original)['tasks'][0]
    assert task['external_manifest']==str(first/'data/pan2023/manifest.json')
    assert json.loads((output/'runtime/launch.json').read_text())['package']==str(second)


def test_same_recipe_does_not_ignore_real_task_parameter_changes(tmp_path,monkeypatch):
    from paper3r7 import plan as scientific
    _,_,make=_same_recipe_plan_factory(monkeypatch,tmp_path)
    args=['--root',str(tmp_path),'--python',sys.executable]
    assert submit.main(args)==0
    output=tmp_path/'runs/followup_20261024_r7'
    old=(output/'plan.json').read_bytes()
    def changed(*a,**kw):
        result=make(*a,**kw);result['tasks'][0]['config']['radius']=.35;return result
    monkeypatch.setattr(scientific,'make_plan',changed)
    with pytest.raises(SystemExit,match='numerical tasks changed'):
        submit.main(args)
    assert (output/'plan.json').read_bytes()==old


def test_live_controller_keeps_old_package_for_identical_repackage(tmp_path,monkeypatch,capsys):
    import fcntl
    first,second,_=_same_recipe_plan_factory(monkeypatch,tmp_path)
    args=['--root',str(tmp_path),'--python',sys.executable]
    assert submit.main(args)==0
    output=tmp_path/'runs/followup_20261024_r7'
    old=(output/'runtime/launch.json').read_bytes()
    starts=[]
    monkeypatch.setattr(submit,'start_background',lambda p,l,c=None: starts.append(l) or {'status':'SYNTHETIC_START_ACK'})
    monkeypatch.setattr(submit,'PACKAGE',second)
    with (output/'runtime/controller.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert submit.main([*args,'--submit'])==0
        assert (output/'runtime/launch.json').read_bytes()==old
        assert starts[-1]['package']==str(first)
        with pytest.raises(SystemExit,match='different launch configuration'):
            submit.main([*args,'--cpus','16','--submit'])
    prepared=[json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert any(row.get('existing_controller_reused') is True for row in prepared)
    assert len(starts)==1
