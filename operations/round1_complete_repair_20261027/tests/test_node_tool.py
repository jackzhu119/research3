from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import newnode_tool as node


@pytest.fixture
def owned(tmp_path):
    root = tmp_path/'paper3'; runtime = root/node.RUNTIME_RELATIVE
    status = runtime/'worker-status/slot-0.json'; status.parent.mkdir(parents=True)
    package = tmp_path/'original-package'; package.mkdir()
    worker = package/'worker.py'; worker.write_text('# owned operational worker\n')
    key = 'development-original-identity'; token = 'a'*32; parent, child = 1200, 1201
    action_file = runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'action.json'
    action_file.parent.mkdir(parents=True)
    document = dict(action=dict(key=key, kind='development', expected_receipt='runs/development/fixed/receipt.json'),
        claim=dict(token=token, job_id='772906', slot=0, pid=parent, claimed_utc=123.0))
    action_file.write_text(json.dumps(document))
    row = dict(job_id='772906', slot=0, pid=parent, child_pid=child, action_key=key,
               claim_token=token, phase='ACTION_RUNNING')
    status.write_text(json.dumps(row))
    args = [str(Path(sys.executable).resolve()), '-B', '-u', str(worker), '--root', str(root),
        '--runtime-dir', str(runtime), '--slot', '0', '--package', str(package)]
    common = dict(uid=os.getuid(), state='S', slurm_job_id='772906', visible_devices='1',
                  cublas_workspace_config=':4096:8')
    procs = {parent: dict(common, ppid=900, start_ticks=100, argv=args),
             child: dict(common, ppid=parent, start_ticks=200,
                         argv=args+['--child', '--action-file', str(action_file)])}
    return types.SimpleNamespace(root=root, runtime=runtime, status=status, row=row, document=document,
        action_file=action_file, procs=procs, parent=parent, child=child, key=key, token=token,
        lookup=lambda pid, _: copy.deepcopy(procs[int(pid)]))


def refresh(f, **kwargs):
    return node.refresh(f.runtime, '772906', 0, root=f.root, allocation_job='772906',
                        process_fn=f.lookup, **kwargs)


def test_exact_active_child_only_receives_original_checkpoint_signal(owned):
    # Active numerical work is intentionally allowed to checkpoint gracefully.
    progress = owned.root/'runs/development/fixed/progress.json'
    progress.parent.mkdir(parents=True); progress.write_text('{"stage":"fitting"}')
    signals = []
    result = refresh(owned, kill=lambda pid, sig: signals.append((pid, sig)))
    assert signals == [(owned.child, signal.SIGINT)]
    assert result['status'] == 'CHECKPOINT_REFRESH_REQUESTED'
    assert result['parent_signalled'] is False and result['allocation_cancelled'] is False
    assert result['unfinished_fit_may_need_resume'] is True
    assert json.loads(owned.action_file.read_text()) == owned.document


@pytest.mark.parametrize('change', ['uid', 'ppid', 'job', 'slot', 'claim', 'script', 'python', 'duplicate_flag', 'token'])
def test_foreign_or_changed_child_never_signalled(owned, change):
    if change == 'uid': owned.procs[owned.child]['uid'] = os.getuid()+1
    elif change == 'ppid': owned.procs[owned.child]['ppid'] += 1
    elif change == 'job': owned.procs[owned.child]['slurm_job_id'] = '999'
    elif change == 'slot': owned.row['slot'] = 1; owned.status.write_text(json.dumps(owned.row))
    elif change == 'claim':
        owned.document['claim']['pid'] += 1; owned.action_file.write_text(json.dumps(owned.document))
    elif change == 'script': owned.procs[owned.child]['argv'][3] = '/tmp/another_worker.py'
    elif change == 'python': owned.procs[owned.child]['argv'][0] = '/usr/bin/another-python'
    elif change == 'duplicate_flag': owned.procs[owned.child]['argv'] += ['--slot', '0']
    elif change == 'token': owned.row['claim_token'] = 'b'*32; owned.status.write_text(json.dumps(owned.row))
    result = refresh(owned, kill=lambda *_: pytest.fail('foreign child signalled'))
    assert result['status'] == 'WAIT_NEXT_CHILD'


def test_refresh_for_wrong_allocation_is_read_only(owned):
    result = node.refresh(owned.runtime, '772906', 0, root=owned.root, allocation_job='999',
        process_fn=owned.lookup, kill=lambda *_: pytest.fail('wrong allocation signalled'))
    assert result['status'] == 'WAIT_NEXT_CHILD'
    assert result['reason'] == 'allocation_job_mismatch'


@pytest.mark.parametrize('which', ['parent', 'child', 'claim'])
def test_pid_and_claim_identity_rechecked_immediately_before_signal(owned, which):
    calls = []
    def lookup(pid, root):
        calls.append(pid)
        value = owned.lookup(pid, root)
        if len(calls) > 2 and ((which == 'parent' and pid == owned.parent)
                              or (which == 'child' and pid == owned.child)):
            value['start_ticks'] += 1
        if len(calls) == 2 and which == 'claim':
            row = dict(owned.row, claim_token='b'*32); owned.status.write_text(json.dumps(row))
        return value
    result = node.refresh(owned.runtime, '772906', 0, root=owned.root, allocation_job='772906',
        process_fn=lookup, kill=lambda *_: pytest.fail('reused PID or changed claim signalled'))
    assert result['status'] == 'WAIT_NEXT_CHILD'


def repair_marker(f, **changes):
    value = dict(repair_id=node.REPAIR_ID, action_key=f.key, claim_token=f.token, pid=f.child,
        helpers_sha256={'verified_context_cache.py': '1'*64, 'newdiagnostics.py': '2'*64})
    value.update(changes)
    f.action_file.with_name('repair-active.json').write_text(json.dumps(value))


def test_new_repaired_child_is_not_refreshed_again(owned):
    repair_marker(owned)
    result = refresh(owned, kill=lambda *_: pytest.fail('already repaired child signalled'))
    assert result['status'] == 'CURRENT_REPAIRED_CHILD_KEPT'


def test_stale_other_claim_marker_does_not_prevent_real_refresh(owned):
    repair_marker(owned, claim_token='b'*32)
    sent = []; result = refresh(owned, kill=lambda *x: sent.append(x))
    assert sent == [(owned.child, signal.SIGINT)]
    assert result['status'] == 'CHECKPOINT_REFRESH_REQUESTED'


def test_actual_installer_activation_marker_keeps_repaired_child(owned, monkeypatch):
    import install_hooks
    original = (ROOT/'vendor/action_startup.py').read_bytes()
    helpers = {name: (ROOT/name, hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
               for name in ('verified_context_cache.py', 'newdiagnostics.py')}
    source = install_hooks.patch_source(original, helpers)
    module = types.ModuleType('owned_original_startup_marker_test')
    module.__file__ = str(owned.action_file.parent/'startup.py')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = module.__dict__
    exec(compile(source, '<patched-owned-startup>', 'exec'), namespace)
    monkeypatch.setattr(os, 'getpid', lambda: owned.child)
    namespace['_paper3_r1_complete_note'](owned.root, owned.document['action'], str(owned.action_file))
    marker = json.loads(owned.action_file.with_name('repair-active.json').read_text())
    assert marker['pid'] == owned.child and marker['helpers_sha256'] == {k: v[1] for k, v in helpers.items()}
    result = refresh(owned, kill=lambda *_: pytest.fail('installer-marked repaired child signalled'))
    assert result['status'] == 'CURRENT_REPAIRED_CHILD_KEPT'


def test_marker_appearing_during_identity_read_keeps_new_child(owned):
    calls = []
    def lookup(pid, root):
        calls.append(pid)
        if len(calls) == 4:
            repair_marker(owned)
        return owned.lookup(pid, root)
    result = node.refresh(owned.runtime, '772906', 0, root=owned.root, allocation_job='772906',
        process_fn=lookup, kill=lambda *_: pytest.fail('newly repaired child signalled'))
    assert result['status'] == 'CURRENT_REPAIRED_CHILD_KEPT'


def test_idle_worker_parent_kept_and_can_be_probed(owned):
    owned.row.update(child_pid=None); owned.status.write_text(json.dumps(owned.row))
    result = refresh(owned, kill=lambda *_: pytest.fail('idle parent signalled'))
    assert result['status'] == 'IDLE_PARENT_REUSED_NO_SIGNAL'
    observed = []
    def observer(root, parent):
        observed.append((root, parent['visible_devices']))
        return dict(device_uuid='actual-fixture-uuid', device_name='actual-fixture-device')
    result = node.probe(owned.root, owned.runtime, '772906', 0, allocation_job='772906',
        process_fn=owned.lookup, runtime_observer=observer)
    assert result['status'] == 'ACTUAL_RUNTIME_OBSERVED_NO_FIT'
    assert observed == [(owned.root, '1')]
    assert result['scientific_fit_executed'] is False


def test_probe_rechecks_current_worker_after_runtime_observation(owned):
    def observer(*_):
        owned.procs[owned.parent]['start_ticks'] += 1
        return {'device_uuid': 'fixture'}
    result = node.probe(owned.root, owned.runtime, '772906', 0, allocation_job='772906',
        process_fn=owned.lookup, runtime_observer=observer)
    assert result['status'] == 'PROBE_UNAVAILABLE'


def test_probe_accepts_child_action_turnover_with_same_parent_allocation(owned):
    def observer(*_):
        del owned.procs[owned.child]
        owned.row.update(child_pid=9876, claim_token='b'*32, action_key='next-completely-different-action')
        owned.status.write_text(json.dumps(owned.row))
        return {'device_uuid': 'same-allocation-physical-device'}
    result = node.probe(owned.root, owned.runtime, '772906', 0, allocation_job='772906',
        process_fn=owned.lookup, runtime_observer=observer)
    assert result['status'] == 'ACTUAL_RUNTIME_OBSERVED_NO_FIT'
    assert result['runtime_profile']['device_uuid'] == 'same-allocation-physical-device'


@pytest.mark.parametrize('changed', ['job_id', 'slot', 'pid', 'cuda'])
def test_probe_rejects_changed_parent_allocation_or_gpu_mask(owned, changed):
    def observer(*_):
        if changed == 'cuda':
            owned.procs[owned.parent]['visible_devices'] = 'another-device'
        else:
            owned.row[changed] = {'job_id': 'another-job', 'slot': 1, 'pid': 1210}[changed]
            owned.status.write_text(json.dumps(owned.row))
        return {'device_uuid': 'fixture'}
    result = node.probe(owned.root, owned.runtime, '772906', 0, allocation_job='772906',
        process_fn=owned.lookup, runtime_observer=observer)
    assert result['status'] == 'PROBE_UNAVAILABLE'


def test_native_probe_matches_final_science_threads_and_calls_no_fit(owned, monkeypatch):
    calls = []
    torch = types.SimpleNamespace(set_num_threads=lambda n: calls.append(('threads', n)))
    def runtime():
        assert all(os.environ[k] == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                  'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        calls.append(('runtime', True)); return {'device_uuid': 'native-runtime-fixture'}
    module = types.SimpleNamespace(__file__=str(owned.root/'p3work/development_evaluation.py'),
                                   torch=torch, _production_runtime=runtime)
    monkeypatch.setattr(node.importlib, 'import_module', lambda name: module)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '12')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'different-step-mask')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    before = list(sys.path)
    try:
        result = node.probe(owned.root, owned.runtime, '772906', 0, allocation_job='772906',
                           process_fn=owned.lookup)
    finally:
        sys.path[:] = before
    assert result['status'] == 'ACTUAL_RUNTIME_OBSERVED_NO_FIT'
    assert calls == [('threads', 1), ('runtime', True)]


def test_actual_proc_reader_retains_only_allowlisted_environment_fields(tmp_path):
    pid = 1200; folder = tmp_path/str(pid); folder.mkdir()
    (folder/'cmdline').write_bytes(b'/absolute/python\0worker.py\0')
    fields = ['S', '900']+['0']*17+['12345']+['0']*5
    (folder/'stat').write_text(str(pid)+' (process name) '+' '.join(fields))
    (folder/'environ').write_bytes(b'SLURM_JOB_ID=772906\0CUDA_VISIBLE_DEVICES=1\0SECRET_TOKEN=never-return\0')
    result = node.process(pid, tmp_path)
    assert result['start_ticks'] == 12345 and result['slurm_job_id'] == '772906'
    assert result['visible_devices'] == '1'
    assert 'never-return' not in json.dumps(result)
