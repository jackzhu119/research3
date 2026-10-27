"""Real detached CPU sidecar tests: no network, EEG, Slurm, or GPU activity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper3m12 import background as bg


HELPER = '''import json,sys,time,os\nfrom pathlib import Path\na=sys.argv[1:];p=Path(a[a.index('--plan')+1]);v=json.loads(p.read_text());r=Path(v['runtime'])\nname='reporter' if a[0]=='report-loop' else a[a.index('--channel')+1]+'-upload'\n(r/(name+'-ready.json')).write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))\nwhile not (r/'fixture-stop').exists():time.sleep(.01)\n'''


def wait(predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            value = predicate()
        except (OSError, ValueError):
            value = None
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('CPU fixture did not become ready')


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'project with space'; root.mkdir()
    runtime = root/'runtime'; runtime.mkdir()
    entry = root/'sidecar_fixture.py'; entry.write_text(HELPER)
    plan_path = root/'plan.json'
    plan = dict(root=str(root), runtime=str(runtime), python=sys.executable, worker_entry=str(entry))
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(bg, 'load_plan', lambda value: (Path(value).resolve(), dict(plan)))
    processes = []
    def launch(command, **kwargs):
        child = subprocess.Popen(command, **kwargs)
        processes.append(child)
        return child
    yield dict(root=root, runtime=runtime, entry=entry, plan=plan, plan_path=plan_path,
               processes=processes, popen=launch)
    (runtime/'fixture-stop').touch()
    for child in processes:
        try:
            child.wait(timeout=4)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=4)


def ready(x):
    return all((x['runtime']/(name+'-ready.json')).is_file()
               for name in ['reporter', 'paper-upload', 'full-upload'])


def test_real_detached_helpers_are_exactly_reused_with_no_extra_launch(prepared):
    x = prepared
    first = bg.ensure_sidecars(x['plan_path'], popen=x['popen'])
    assert set(first) == {'reporter', 'paper-upload', 'full-upload'}
    assert all(row['status'] in {'HELPER_STARTED', 'HELPER_STARTUP_PENDING'} for row in first.values())
    wait(lambda: ready(x))
    second = bg.ensure_sidecars(x['plan_path'], popen=lambda *a, **k: pytest.fail('duplicate helper'))
    assert len(x['processes']) == 3
    assert all(row['status'] == 'EXISTING_HELPER_REUSED' for row in second.values())
    assert {name: row['pid'] for name, row in first.items()} == {name: row['pid'] for name, row in second.items()}
    for child in x['processes']:
        assert child.poll() is None
        assert os.getsid(child.pid) == child.pid
    assert all(row['training_controlled'] is False for row in second.values())


def test_stale_pid_for_another_process_is_never_adopted_or_signalled(prepared):
    x = prepared
    unrelated_script = x['root']/'unrelated_science_fixture.py'
    unrelated_script.write_text("import time\nfrom pathlib import Path\np=Path(__file__).parent/'runtime'\nwhile not (p/'fixture-stop').exists():time.sleep(.01)\n")
    unrelated = x['popen']([sys.executable, '-I', '-S', '-B', str(unrelated_script)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    bg.best_effort_json(x['runtime']/'reporter-process.json',
                        dict(pid=unrelated.pid, start_ticks='stale'))
    value = bg.ensure_sidecars(x['plan_path'], popen=x['popen'])
    wait(lambda: ready(x))
    assert unrelated.poll() is None
    assert value['reporter']['pid'] != unrelated.pid
    assert len(x['processes']) == 4


def test_wrong_start_ticks_recovers_only_exact_owned_helper_without_duplicate(prepared):
    x = prepared
    first = bg.ensure_sidecars(x['plan_path'], popen=x['popen']); wait(lambda: ready(x))
    path = x['runtime']/'reporter-process.json'
    value = json.loads(path.read_text()); value['start_ticks'] = 'pid-reuse-stale'
    bg.best_effort_json(path, value)
    second = bg.ensure_sidecars(x['plan_path'], popen=lambda *a, **k: pytest.fail('exact process already exists'))
    assert second['reporter']['status'] == 'EXISTING_HELPER_DISCOVERED'
    assert second['reporter']['pid'] == first['reporter']['pid']
    assert len(x['processes']) == 3


def test_first_helper_failure_does_not_abort_other_two_helpers(prepared):
    x = prepared; calls = []
    def launch(command, **kwargs):
        calls.append(command)
        if 'report-loop' in command:
            raise OSError('synthetic no new process slots for reporter')
        return x['popen'](command, **kwargs)
    value = bg.ensure_sidecars(x['plan_path'], popen=launch)
    assert len(calls) == 3 and len(x['processes']) == 2
    assert value['reporter']['status'] == 'HELPER_START_RETRY_REQUIRED'
    wait(lambda: (x['runtime']/'paper-upload-ready.json').exists()
         and (x['runtime']/'full-upload-ready.json').exists())
    assert all(row['training_controlled'] is False for row in value.values())


def test_unexpected_launch_exception_remains_scoped_to_its_role(prepared):
    x = prepared
    def launch(command, **kwargs):
        if 'report-loop' in command:
            raise RuntimeError('synthetic launcher wrapper failed')
        return x['popen'](command, **kwargs)
    value = bg.ensure_sidecars(x['plan_path'], popen=launch)
    assert value['reporter']['status'] == 'HELPER_START_RETRY_REQUIRED'
    assert value['reporter']['error_type'] == 'RuntimeError'
    assert len(x['processes']) == 2


def test_missing_receipts_are_recovered_by_exact_argv_scan(prepared):
    x = prepared
    first = bg.ensure_sidecars(x['plan_path'], popen=x['popen']); wait(lambda: ready(x))
    for name in first:
        (x['runtime']/(name+'-process.json')).unlink()
    second = bg.ensure_sidecars(x['plan_path'], popen=lambda *a, **k: pytest.fail('lost receipt cannot duplicate helper'))
    assert all(row['status'] == 'EXISTING_HELPER_DISCOVERED' for row in second.values())
    assert len(x['processes']) == 3


def test_delayed_identity_is_pending_then_adopted_instead_of_restarted(prepared, monkeypatch):
    x = prepared; actual = bg._identity
    monkeypatch.setattr(bg, '_identity', lambda *a, **k: None)
    first = bg.ensure_sidecars(x['plan_path'], popen=x['popen'])
    assert all(row['status'] == 'HELPER_STARTUP_PENDING' for row in first.values())
    wait(lambda: ready(x))
    monkeypatch.setattr(bg, '_identity', actual)
    second = bg.ensure_sidecars(x['plan_path'], popen=lambda *a, **k: pytest.fail('pending identity cannot duplicate helper'))
    assert all(row['status'] == 'EXISTING_HELPER_REUSED' for row in second.values())
    assert len(x['processes']) == 3


def test_receipt_write_failure_never_becomes_scientific_stop_or_duplicate(prepared, monkeypatch):
    x = prepared
    monkeypatch.setattr(bg, 'best_effort_json', lambda *a, **k: False)
    first = bg.ensure_sidecars(x['plan_path'], popen=x['popen']); wait(lambda: ready(x))
    second = bg.ensure_sidecars(x['plan_path'], popen=lambda *a, **k: pytest.fail('lost telemetry cannot duplicate helper'))
    assert len(x['processes']) == 3
    assert all(row['status'] == 'EXISTING_HELPER_DISCOVERED' for row in second.values())
    assert all(row['training_controlled'] is False for row in first.values())


def test_identity_does_not_adopt_a_different_uid(prepared, monkeypatch):
    x = prepared
    values = bg.ensure_sidecars(x['plan_path'], popen=x['popen']); wait(lambda: ready(x))
    record = values['reporter']
    command = json.loads((x['runtime']/'reporter-process.json').read_text())['command']
    actual_uid = os.getuid()
    monkeypatch.setattr(bg.os, 'getuid', lambda: actual_uid + 1)
    assert bg._identity(record['pid'], command) is None
