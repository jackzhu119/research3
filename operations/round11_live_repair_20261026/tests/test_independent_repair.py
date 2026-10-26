"""Independent live-repair checks. CPU/proc fixtures only; no real Slurm/GPU/network.

All writes target pytest directories. Original scientific code/plan payloads are
read-only; fake scheduler observations never invoke an executable.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPAIR = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/workspace/paper3-round11')
sys.path.insert(0, str(REPAIR))
import controller_fix
import repair


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_actual_low_address_space_old_small_json_fails_repaired_reader_succeeds(tmp_path):
    path = tmp_path / 'tiny.json'; path.write_text('{"actual_bytes":14}')
    code = r'''
import json,resource,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import controller_fix
from paper3r11.platform import common
value=Path(sys.argv[3])
pages=int(Path('/proc/self/statm').read_text().split()[0])
ceiling=pages*resource.getpagesize()+24*1024*1024
resource.setrlimit(resource.RLIMIT_AS,(ceiling,ceiling))
failed=False
try: common.read(value,None,max_bytes=64*1024*1024)
except MemoryError: failed=True
assert failed,'The original oversized-read allocation was not reproduced'
assert controller_fix.bounded_read(value,None,max_bytes=64*1024*1024)=={'actual_bytes':14}
controller_fix.install_read_patch(common)
assert common.read(value,None,max_bytes=256*1024*1024)=={'actual_bytes':14}
print(json.dumps({'old_memory_error':failed,'bounded_read_succeeded':True}))
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code,
        str(REPAIR), str(ORIGINAL), str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['old_memory_error']
    assert path.read_text() == '{"actual_bytes":14}'


def test_bounded_reader_preserves_limits_and_never_turns_bad_state_into_valid_state(tmp_path):
    value = tmp_path / 'record.json'; value.write_text('{"x":[1,2,3]}')
    sentinel = {'not_a_valid_runtime_state': True}
    assert controller_fix.bounded_read(value, sentinel, max_bytes=5) is sentinel
    value.write_text('{"truncated":')
    assert controller_fix.bounded_read(value, sentinel) is sentinel
    assert controller_fix.bounded_read(tmp_path / 'absent', sentinel) is sentinel


def process_fixture(tmp_path):
    plan = tmp_path / 'plan.json'; plan.write_text('{"software_fixture":true}')
    package = tmp_path / 'package'; entry = package / 'paper3r11/platform/entry.py'
    entry.parent.mkdir(parents=True); entry.write_text('# never executed\n')
    directory = tmp_path / 'runtime'; directory.mkdir()
    (directory / 'launch.json').write_text(json.dumps({'python': sys.executable,
        'plan': str(plan.resolve()), 'plan_sha256': checksum(plan), 'campaign_identity': 'a' * 64}))
    env = {'path': plan.resolve(), 'package': package.resolve(),
        'runtime': directory, 'common': SimpleNamespace(sha=checksum),
        'header': {'plan_sha256': checksum(plan), 'recipe_sha256': 'a' * 64}}
    return plan.resolve(), entry.resolve(), env


def fake_proc(root, pid, args, start_ticks='101'):
    path = root / str(pid); path.mkdir(parents=True, exist_ok=True)
    (path / 'cmdline').write_bytes(b'\0'.join(str(a).encode() for a in args) + b'\0')
    # Linux stat fields3..22; field22 is process start time, not a guessed PID.
    (path / 'stat').write_text(f'{pid} (python fixture) S ' + ' '.join(['0'] * 18 + [str(start_ticks)]))


@pytest.mark.parametrize('kind', ['unrelated_script', 'science', 'full', 'controller_sidecar', 'duplicate_plan'])
def test_owned_paper_scope_rejects_tokens_embedded_in_other_invocations(tmp_path, kind):
    plan, entry, env = process_fixture(tmp_path); proc = tmp_path / 'proc'; pid = 12345
    suffix = ['--plan', str(plan), '--channel', 'paper']
    if kind == 'unrelated_script':
        args = [sys.executable, str(tmp_path / 'unrelated.py'), str(entry), 'upload', *suffix]
    elif kind == 'science':
        args = [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'science', 'upload', *suffix]
    elif kind == 'full':
        args = [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'upload', '--channel', 'full',
                '--plan', str(plan), 'paper']
    elif kind == 'duplicate_plan':
        other = tmp_path / 'other-plan.json'; other.write_text('{"other_campaign":true}')
        args = [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'upload',
                *suffix, '--plan', str(other)]
    else:
        args = [sys.executable, '-I', '-S', '-B', '-u', str(REPAIR / 'repair.py'), 'controller',
                '--plan', str(plan), '--credentials', 'paper']
    fake_proc(proc, pid, args)
    assert repair.owned_paper_identity(pid, env, proc_root=proc) is None


def test_owned_paper_scope_accepts_exact_original_and_preserves_uid_and_start_time(tmp_path, monkeypatch):
    plan, entry, env = process_fixture(tmp_path); proc = tmp_path / 'proc'; pid = 12345
    args = [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'upload',
            '--plan', str(plan), '--channel', 'paper']
    fake_proc(proc, pid, args, '777')
    identity = repair.owned_paper_identity(pid, env, proc_root=proc)
    assert identity and identity['uid'] == os.getuid() and identity['start_ticks'] == '777'
    monkeypatch.setattr(repair.os, 'getuid', lambda: identity['uid'] + 1)
    assert repair.owned_paper_identity(pid, env, proc_root=proc) is None


def test_original_preparation_plan_copy_is_owned_only_when_declared_in_launch(tmp_path):
    plan, entry, env = process_fixture(tmp_path); proc = tmp_path / 'proc'; pid = 12345
    prepared = tmp_path / 'round11_preparation/plan-campaign.json'
    prepared.parent.mkdir(); prepared.write_bytes(plan.read_bytes())
    foreign = tmp_path / 'arbitrary-copy.json'; foreign.write_bytes(plan.read_bytes())
    launch = json.loads((env['runtime'] / 'launch.json').read_text())
    launch['plan'] = str(prepared)
    (env['runtime'] / 'launch.json').write_text(json.dumps(launch))
    def arguments(path):
        return [sys.executable, '-I', '-S', '-B', '-u', str(entry), 'upload',
                '--plan', str(path), '--channel', 'paper']
    for accepted in (plan, prepared):
        fake_proc(proc, pid, arguments(accepted))
        assert repair.owned_paper_identity(pid, env, proc_root=proc) is not None
    fake_proc(proc, pid, arguments(foreign))
    assert repair.owned_paper_identity(pid, env, proc_root=proc) is None


def test_pid_reuse_prevents_signal_and_exact_owned_paper_gets_only_term():
    values = [{'pid': 777, 'uid': os.getuid(), 'start_ticks': '123', 'args': ['paper']},
              {'pid': 777, 'uid': os.getuid(), 'start_ticks': '124', 'args': ['paper']}]
    signals = []
    result = repair.stop_owned_paper(777, {}, identity_fn=lambda *_: values.pop(0),
                                    signal_fn=lambda *a: signals.append(a))
    assert not result['signal_sent'] and not signals
    identity = {'pid': 777, 'uid': os.getuid(), 'start_ticks': '123', 'args': ['paper']}
    result = repair.stop_owned_paper(777, {}, identity_fn=lambda *_: identity,
                                    signal_fn=lambda *a: signals.append(a))
    assert result['signal_sent'] and signals == [(777, signal.SIGTERM)]


def test_repeated_start_reuses_waiting_sidecars_without_false_ready_or_duplicate_spawn(tmp_path, monkeypatch):
    sys.path.insert(0, str(ORIGINAL))
    from paper3r11.platform import common
    plan, _, env = process_fixture(tmp_path); env['common'] = common
    work = tmp_path / 'repair-work'; work.mkdir()
    for kind, pid in (('controller', 12345), ('paper', 12346)):
        common.atomic(work / (kind + '-process.json'),
            {'pid': pid, 'status': 'STARTUP_PENDING', 'launch_token': kind + '-new'})
    def owned(pid, _env):
        kind = 'controller' if pid == 12345 else 'paper'
        return {'pid': pid, 'uid': os.getuid(), 'start_ticks': str(pid),
            'args': [sys.executable, '-I', '-S', '-B', '-u', str(REPAIR / 'repair.py'), kind,
                     '--plan', str(plan), '--launch-token', kind + '-new']}
    monkeypatch.setattr(repair, 'load_original', lambda *_a, **_k: env)
    monkeypatch.setattr(repair, 'validate_operational_history', lambda *_: {'verified_fixture': True})
    monkeypatch.setattr(repair, 'lock_is_live', lambda _: True)
    monkeypatch.setattr(repair, 'owned_controller_identity', owned)
    monkeypatch.setattr(repair, 'owned_paper_identity', owned)
    def no_spawn(*_a, **_k):
        raise AssertionError('Repeated start must reuse the known waiting sidecars')
    args = SimpleNamespace(package=env['package'], plan=plan, work=work,
        python=sys.executable, credentials=None, submit=True)
    result = repair.start_repair(args, popen=no_spawn)
    assert result['status'] != 'OPERATIONAL_REPAIR_READY'
    assert result['status'] in ('STARTUP_PENDING', 'PARTIAL_REPAIR')
    assert all('PENDING' in row['status'] for row in result['processes'])
    assert result['scientific_workers_signalled'] == 0 and result['slurm_jobs_cancelled'] == 0


def test_actual_original_scheduler_observes_four_existing_jobs_without_submit(tmp_path):
    """Exercise the unmodified scheduler, not a fake scheduling implementation."""
    sys.path.insert(0, str(ORIGINAL))
    from paper3r11.platform import common, runtime, scheduler
    package_hashes = {p: checksum(ORIGINAL / p) for p in (
        'START.py', 'paper3r11/platform/common.py', 'paper3r11/platform/runtime.py',
        'paper3r11/platform/scheduler.py')}
    root = tmp_path / 'school'; root.mkdir(); output = root / 'runs/r11-existing'
    plan = common.sign_plan({'schema_version': common.PLAN_SCHEMA, 'round': 11, 'stage': 'campaign',
        'root': str(root), 'output': str(output), 'package': str(ORIGINAL),
        'resources': {'jobs': 4, 'gpus_per_job': 1, 'cpus_per_job': 16, 'memory_per_job': '64G'},
        'code_manifest': {'START.py': package_hashes['START.py']},
        'tasks': [{'key': 'pending-fixture', 'family': 'fixture', 'role': 'meta_validation', 'requires': []}]})
    output.mkdir(parents=True); path = output / 'plan.json'; common.atomic(path, plan)
    ctx = runtime.context(path); directory = ctx['runtime']; launch = ctx['launch']
    common.atomic(directory / 'launch.json', launch)
    common.atomic(directory / 'SUBMIT_AUTHORIZATION.json', {'explicit_submit': True,
        'recipe_sha256': plan['recipe_sha256'], 'plan_sha256': common.sha(path)})
    actual = scheduler.FourJobScheduler(directory, launch)
    slots = {}; lines = []
    for i in range(4):
        job = str(775286 + i); name = actual.marker + '-s' + str(i) + '-a1'
        attempt = {'slot': i, 'sequence': 1, 'marker': actual.marker, 'name': name,
                   'job_id': job, 'status': 'SUBMITTED', 'intent_at': 1.}
        slots[str(i)] = {'attempts': [attempt], 'next_submit_at': 0, 'worker_failures': 0, 'rejections': 0}
        lines.append('|'.join((job, actual.marker, 'RUNNING', name, 'None', 'fixture-node')))
    common.atomic(directory / 'scheduler-state.json', {'version': 1, 'created_at': 1.,
        'marker': actual.marker, 'slots': slots})
    requests = []
    def runner(argv, **_kwargs):
        requests.append(argv)
        if argv[0] not in ('squeue', 'sacct'):
            raise AssertionError('Existing allocation observation cannot submit or cancel a job')
        return SimpleNamespace(returncode=0,
            stdout='\n'.join(lines) + '\n' if argv[0] == 'squeue' else '', stderr='')
    plan_hash = checksum(path)
    result = controller_fix.controller_loop(path, tmp_path / 'repair-work', package=ORIGINAL,
        submit=True, runner=runner, once=True, expected_recipe=None)
    assert result['inflight_jobs'] == 4 and len(requests) == 2
    assert [r[0] for r in requests] == ['squeue', 'sacct']
    assert checksum(path) == plan_hash
    assert all(checksum(ORIGINAL / p) == h for p, h in package_hashes.items())
    first = controller_fix.bounded_read(tmp_path / 'repair-work/first-observation.json')
    assert first['first_cycle_submit_allowed'] is False


def test_binary_transport_roundtrip_has_original_sha_and_does_not_touch_science_or_full(tmp_path, monkeypatch):
    """Use actual NumPy NPZ bytes and the unmodified UTF8 fragment verifier."""
    import base64
    import io
    import numpy as np
    import paper_fix
    sys.path.insert(0, str(ORIGINAL))
    from paper3r11.platform import common, uploads
    from paper3r11.platform.vendor.lossless_publish import _part_bytes
    # Restore all patched APIs automatically, so another test can reproduce the
    # original UTF8 failure in this same pytest process.
    for name in ('_capture_paper_task', 'paper_collection', 'paper_cycle', 'channel_loop'):
        monkeypatch.setattr(uploads, name, getattr(uploads, name))
    monkeypatch.setattr(uploads, '_paper_binary_sidecar_installation', None, raising=False)
    monkeypatch.setattr(uploads, '_paper_sidecar_ready_callback', None, raising=False)
    before_functions = {name: getattr(uploads, name) for name in
        ('full_cycle', 'monitor_cycle', 'report_loop', 'stage_file', 'publish_snapshot')}
    task = {'key': 'operator-fixture', 'family': 'operator_train', 'role': 'meta_train',
            'requires': [], 'config': {}}
    plan = common.sign_plan({'schema_version': common.PLAN_SCHEMA, 'round': 11, 'stage': 'campaign',
        'synthetic_fixture': True, 'root': str(tmp_path), 'package': str(ORIGINAL),
        'output': str(tmp_path / 'r11-science'), 'tasks': [task],
        'code_manifest': {'START.py': checksum(ORIGINAL / 'START.py')}})
    path = tmp_path / 'plan.json'; common.atomic(path, plan)
    folder = Path(plan['output']) / task['key']; folder.mkdir(parents=True)
    # >3MiB exercises encoder and strict decoder block boundaries and padding.
    values = np.arange(600003, dtype=np.float64)
    np.savez(folder / 'operator.npz', values=values)
    text = 'participant,metric\r\n受试者甲,0.5\r\n'.encode('utf-8')
    (folder / 'participants.csv').write_bytes(text)
    artifacts = [{'path': name, 'bytes': (folder / name).stat().st_size, 'sha256': checksum(folder / name)}
                 for name in ('operator.npz', 'participants.csv')]
    common.atomic(folder / 'result.json', {'status': 'COMPLETE', 'scientific_scope': 'SOFTWARE_TEST_ONLY',
        'recipe_sha256': plan['recipe_sha256'], 'task_key': task['key'],
        'task_identity': common.task_identity(plan, task), 'artifacts': artifacts,
        'paper_artifacts': artifacts})
    common.seal_result(plan, task)
    row = {'task_identity': common.task_identity(plan, task), 'result_sha256': checksum(folder / 'result.json'),
           'completion_sha256': checksum(folder / 'completion.json')}
    untouched_full = Path(plan['output']) / 'runtime/upload-full/sentinel.json'
    untouched_full.parent.mkdir(parents=True); untouched_full.write_text('{"independent_B":true}')
    protected = [path, untouched_full, *folder.iterdir(), ORIGINAL / 'paper3r11/worker.py',
                 ORIGINAL / 'paper3r11/platform/uploads.py']
    before = {p: (checksum(p), p.stat().st_mtime_ns) for p in protected if p.is_file()}
    installed = paper_fix.install_patch(uploads)
    assert installed['full_archive_channel_unchanged']
    assert all(getattr(uploads, name) is value for name, value in before_functions.items())
    capture = uploads._capture_paper_task(plan, task['key'], row, tmp_path / 'capture', tmp_path / 'paper-work')
    files = {r['source_path']: r for r in capture['files']}
    encoded_record = files[task['key'] + '/binary-base64/operator.npz.b64.txt']
    encoded = b''.join(_part_bytes(part) for part in encoded_record['parts'])
    decoded = base64.b64decode(encoded, validate=True)
    assert decoded == (folder / 'operator.npz').read_bytes()
    with np.load(io.BytesIO(decoded), allow_pickle=False) as restored:
        np.testing.assert_array_equal(restored['values'], values)
    mapping = capture['binary_representations'][0]
    assert mapping['original_sha256'] == checksum(folder / 'operator.npz')
    assert mapping['original_bytes'] == len(decoded)
    assert mapping['original_result_sha256'] == row['result_sha256']
    assert mapping['original_completion_sha256'] == row['completion_sha256']
    assert task['key'] + '/operator.npz' not in files
    assert b''.join(_part_bytes(p) for p in files[task['key'] + '/participants.csv']['parts']) == text
    metadata = paper_fix._scope_metadata({'declared_paper_artifacts_exact_unsanitized_bytes': True}, [mapping])
    assert metadata['declared_paper_artifacts_exact_unsanitized_bytes'] is False
    assert metadata['exact_original_bytes_scope'] == paper_fix.SCOPE
    assert before == {p: (checksum(p), p.stat().st_mtime_ns) for p in before}
