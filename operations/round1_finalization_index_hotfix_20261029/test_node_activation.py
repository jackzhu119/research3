"""An optional CPU-verifier unwind is bound to its exact live action claim."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location('_node_activation_hotfix', HERE/'hotfix.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def layout(tmp_path, monkeypatch):
    module = load()
    home = tmp_path/'home'
    root = home/'projects/paper3'
    runtime = root/module.RUNTIME
    work = home/'fix/work'
    runner = home/'paper3-r1-single-gpu/82bfccb2c75447584639/worker_runner.py'
    native = home/'native'
    for path in (runtime, work, native, runner.parent):
        path.mkdir(parents=True, exist_ok=True)
    runner.write_text('# retained immutable entry\n')
    key, kind, token = 'strongest-control', 'strongest_control', 'a'*32
    claim = dict(slot=0, job_id='779398', pid=101, token=token,
                 action_kind=kind, serial=True, claimed_utc=999.)
    worker = dict(slot=0, job_id='779398', pid=101, child_pid=202,
                  action_key=key, action_kind=kind, claim_token=token,
                  phase='ACTION_RUNNING', updated_at=1000.)
    folder = runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token
    folder.mkdir(parents=True)
    action_file = folder/'action.json'
    action = dict(key=key, kind=kind, serial=True,
                  expected_receipt='runs/training_campaign_v1/strongest_control.json')
    module.atomic(action_file, dict(action=action, claim=claim))
    module.atomic(runtime/'claims.json', {key:claim})
    module.atomic(runtime/'worker-status/slot-0.json', worker)
    log = folder/'kernel.log'
    log.write_text(json.dumps(dict(event='R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE',
        version='paper3-r1-finalization-verification-reuse-v1'))+'\nKERNEL_READY '+token+'\n')
    argv = [sys.executable, '-B', '-u', str(runner), '--child', '--action-file', str(action_file),
            '--root', str(root), '--runtime-dir', str(runtime), '--slot', '0', '--package', str(native)]
    current = dict(pid=202, ppid=101, uid=os.getuid(), start_ticks='4242', argv=argv)
    config = dict(root=str(root), home=str(home), runtime=str(runtime), work=str(work),
                  runner=str(runner), native=str(native), python=sys.executable)
    monkeypatch.setenv('SLURM_JOB_ID', '779398')
    return dict(module=module, home=home, root=root, runtime=runtime, work=work, runner=runner,
                native=native, action=action, claim=claim, worker=worker, folder=folder,
                action_file=action_file, log=log, current=current, config=config, token=token)


def invoke(layout, *, process=None, send=None):
    sent = []
    result = layout['module'].node_activate(layout['config'],
        process=process or (lambda pid: copy.deepcopy(layout['current']) if pid == 202 else None),
        send=send or (lambda pid, number:sent.append((pid, number))), clock=lambda:1000.)
    return result, sent


def test_exact_ready_old_cpu_verifier_receives_one_sigint_and_a_durable_record(layout):
    result, sent = invoke(layout)
    assert result['status'] == 'CPU_FINALIZATION_REFRESH_REQUESTED'
    assert sent == [(202, signal.SIGINT)]
    assert result['optimizer_processes_signalled'] == 0
    record = layout['module'].read(layout['work']/'refreshed-claims'/(layout['token']+'.json'))
    assert record['signal_sent'] is True and record['child_pid'] == 202
    again, again_sent = invoke(layout)
    assert again_sent == [] and again['signal_sent'] is False


def test_wrong_slurm_allocation_is_never_signalled(layout, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '888888')
    result, sent = invoke(layout)
    assert result['status'] == 'ALLOCATION_BINDING_DIFFERS' and not sent


@pytest.mark.parametrize('change', ['ppid', 'uid', 'script', 'root', 'runtime', 'slot', 'package', 'action_file', 'child_flag'])
def test_mismatched_process_roles_are_never_signalled(layout, change):
    current = layout['current']
    if change == 'ppid':current['ppid'] += 1
    elif change == 'uid':current['uid'] += 1
    elif change == 'script':current['argv'][3] = str(layout['home']/'other.py')
    elif change == 'child_flag':current['argv'].remove('--child')
    else:
        flag = {'root':'--root', 'runtime':'--runtime-dir', 'slot':'--slot',
                'package':'--package', 'action_file':'--action-file'}[change]
        current['argv'][current['argv'].index(flag)+1] = '1' if change == 'slot' else '/other'
    _, sent = invoke(layout)
    assert not sent


def test_pid_reuse_between_identity_checks_is_never_signalled(layout):
    calls = []
    def process(pid):
        calls.append(pid)
        row = copy.deepcopy(layout['current'])
        if len(calls) >= 2:row['start_ticks'] = 'REUSED'
        return row
    _, sent = invoke(layout, process=process)
    assert not sent and len(calls) >= 2


def test_changed_claim_between_snapshots_is_never_signalled(layout):
    calls = []
    def process(pid):
        calls.append(pid)
        if len(calls) == 1:
            row = dict(layout['claim'], token='b'*32)
            layout['module'].atomic(layout['runtime']/'claims.json', {'strongest-control':row})
        return copy.deepcopy(layout['current'])
    _, sent = invoke(layout, process=process)
    assert not sent


@pytest.mark.parametrize('change', ['kind', 'key_kind', 'serial_action', 'serial_claim', 'freshness'])
def test_internal_or_inconsistent_scientific_action_is_preserved(layout, change):
    module = layout['module']
    if change == 'kind':
        record = dict(action=dict(layout['action'], kind='internal'), claim=layout['claim'])
        module.atomic(layout['action_file'], record)
    elif change == 'key_kind':
        layout['worker']['action_kind'] = 'freeze';layout['claim']['action_kind'] = 'freeze'
        module.atomic(layout['runtime']/'claims.json', {'strongest-control':layout['claim']})
        module.atomic(layout['runtime']/'worker-status/slot-0.json', layout['worker'])
        module.atomic(layout['action_file'], dict(action=dict(layout['action'], kind='freeze'), claim=layout['claim']))
    elif change == 'serial_action':
        module.atomic(layout['action_file'], dict(action=dict(layout['action'], serial=False), claim=layout['claim']))
    elif change == 'serial_claim':
        layout['claim']['serial'] = False
        module.atomic(layout['runtime']/'claims.json', {'strongest-control':layout['claim']})
        module.atomic(layout['action_file'], dict(action=layout['action'], claim=layout['claim']))
    else:
        module.atomic(layout['runtime']/'worker-status/slot-0.json', dict(layout['worker'], updated_at=1.))
    _, sent = invoke(layout)
    assert not sent


@pytest.mark.parametrize('change', ['fixed_imported', 'fixed_scope', 'returned', 'not_ready', 'no_old_scope'])
def test_fixed_returned_or_not_ready_child_is_never_signalled(layout, change):
    if change == 'fixed_imported':
        with layout['log'].open('a') as handle:
            handle.write(json.dumps(dict(event='R1_FINALIZATION_INDEX_FIX_IMPORTED',version=layout['module'].VERSION))+'\n')
    elif change == 'fixed_scope':
        with layout['log'].open('a') as handle:
            handle.write(json.dumps(dict(event='R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE',version=layout['module'].VERSION))+'\n')
    elif change == 'returned':
        layout['module'].atomic(layout['folder']/'returned.json', dict(status='ERROR'))
    elif change == 'not_ready':
        layout['log'].write_text(json.dumps(dict(event='R1_FINALIZATION_VERIFICATION_REUSE_ACTIVE',
             version='paper3-r1-finalization-verification-reuse-v1'))+'\n')
    else:
        layout['log'].write_text('KERNEL_READY '+layout['token']+'\n')
    _, sent = invoke(layout)
    assert not sent


def test_concurrent_duplicate_node_requests_signal_only_once(layout):
    sent = []
    barrier = threading.Barrier(2)
    def send(pid, number):
        sent.append((pid, number));time.sleep(.05)
    def run(_):
        barrier.wait(timeout=2)
        return invoke(layout, send=send)[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert sent == [(202, signal.SIGINT)]
    assert sum(row.get('signal_sent') is True for row in results) == 1


def test_failed_signal_has_durable_uncertain_record_and_is_not_blindly_repeated(layout):
    sent = []
    def failed(pid, number):
        sent.append((pid, number));raise PermissionError('test only')
    result, _ = invoke(layout, send=failed)
    assert result.get('signal_sent') is not True and len(sent) == 1
    record = layout['module'].read(layout['work']/'refreshed-claims'/(layout['token']+'.json'))
    assert record and record.get('signal_sent') is not True
    _, again = invoke(layout)
    assert not again


def test_activation_step_only_uses_exact_running_allocation_metadata_step(layout, monkeypatch):
    module = layout['module']
    original = module.binding
    monkeypatch.setattr(module, 'binding', lambda config:original(config, clock=lambda:1000.))
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        if command[0] == 'squeue':
            return subprocess.CompletedProcess(command, 0, '779398|paper3r1s-test-s0-a1|RUNNING|gpu2\n', '')
        assert command[0] == 'srun'
        assert '--jobid=779398' in command and '--gres=none' in command and '--cpus-per-task=1' in command
        result = dict(status='CPU_FINALIZATION_REFRESH_REQUESTED', signal_sent=True)
        return subprocess.CompletedProcess(command, 0, json.dumps(dict(event='R1_FINALIZATION_NODE_ACTIVATION', result=result))+'\n', '')
    result = module.activation_step(layout['config'], runner=runner)
    assert result['status'] == 'CPU_FINALIZATION_REFRESH_REQUESTED'
    assert [command[0] for command in commands] == ['squeue', 'srun']
    assert not any(command[0] in ('sbatch', 'scancel') for command in commands)


def test_nonrunning_or_unrelated_allocation_never_launches_node_step(layout, monkeypatch):
    module = layout['module']
    original = module.binding
    monkeypatch.setattr(module, 'binding', lambda config:original(config, clock=lambda:1000.))
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '779398|p1-comp-other|RUNNING|gpu2\n', '')
    result = module.activation_step(layout['config'], runner=runner)
    assert result['status'] == 'ALLOCATION_NOT_OBSERVED_RUNNING'
    assert len(commands) == 1 and commands[0][0] == 'squeue'
