from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import supervisor as s


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def env(tmp_path):
    root = tmp_path/'school'; root.mkdir()
    runtime = root/'runs/r11_campaign/runtime'; runtime.mkdir(parents=True)
    plan = root/'runs/r11_campaign/plan.json'; plan.write_text('{"fixture":"locked"}')
    package = root/'original-package'; (package/'paper3r11/platform').mkdir(parents=True)
    (package/'paper3r11/platform/entry.py').write_text('# original immutable entry\n')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    common = SimpleNamespace(sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                             atomic=write_json, utc=lambda: '2026-10-27T01:00:00Z')
    value = dict(path=plan, package=package, runtime=runtime, common=common,
        header=dict(plan_sha256=digest, recipe_sha256='f'*64, python=sys.executable,
                    total_tasks=10270))
    write_json(runtime/'launch.json', dict(plan=str(plan), plan_sha256=digest,
        campaign_identity='f'*64))
    return value


def make_proc(root, pid, args, *, ticks='12345', state='S'):
    folder = Path(root)/str(pid); folder.mkdir(parents=True, exist_ok=True)
    (folder/'cmdline').write_bytes(b'\0'.join(os.fsencode(x) for x in args)+b'\0')
    fields = [state]+['0']*18+[str(ticks)]+['0']*8
    (folder/'stat').write_text(str(pid)+' (test controller) '+' '.join(fields))
    return folder


def argv(env, script, mode='controller', *extra):
    return [sys.executable, '-I', '-S', '-B', '-u', str(script), mode,
            '--plan', str(env['path']), *extra]


@pytest.mark.parametrize('own', [False, True])
def test_matches_actual_controller_script_position_only(tmp_path, env, own):
    script = s.HERE/'resume_io.py' if own else env['package']/'paper3r11/platform/entry.py'
    args = argv(env, script, 'controller', '--launch-token', 'nonce')
    make_proc(tmp_path/'proc', 90001, args)
    result = s.controller_identity(90001, env, proc_root=tmp_path/'proc')
    assert result['pid'] == 90001
    assert result['start_ticks'] == '12345'
    assert result['current_patch'] is own


def test_previous_deployed_repair_sha_and_exact_home_path_accepted(tmp_path, env, monkeypatch):
    home = tmp_path/'home'
    monkeypatch.setattr(s.Path, 'home', classmethod(lambda cls: home))
    old = home/'paper3-r11-operational-repair/7002725dc6537ff7a4a2/repair.py'
    old.parent.mkdir(parents=True)
    old.write_bytes(Path('/workspace/paper3-r11-live-repair/repair.py').read_bytes())
    assert hashlib.sha256(old.read_bytes()).hexdigest() == s.OLD_SUPERVISOR_SHA
    make_proc(tmp_path/'proc', 90001, argv(env, old))
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is not None
    old.write_bytes(old.read_bytes()+b'\n# changed\n')
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None


@pytest.mark.parametrize('mode', ['worker', 'science', 'science-daemon', 'paper', 'upload', 'report-loop'])
def test_scientific_and_upload_processes_never_owned_controllers(tmp_path, env, mode):
    make_proc(tmp_path/'proc', 90001, argv(env, s.HERE/'resume_io.py', mode))
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None


def test_foreign_python_cannot_embed_controller_tokens(tmp_path, env):
    foreign = tmp_path/'foreign.py'; foreign.write_text('# unrelated')
    args = argv(env, foreign, 'controller', str(s.HERE/'resume_io.py'))
    make_proc(tmp_path/'proc', 90001, args)
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None


def test_duplicate_plan_and_modified_plan_rejected(tmp_path, env):
    original = argv(env, s.HERE/'resume_io.py')
    make_proc(tmp_path/'proc', 90001, original+['--plan', str(env['path'])])
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None
    make_proc(tmp_path/'proc', 90001, original)
    env['path'].write_text('{"fixture":"changed"}')
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None


def test_original_preparation_plan_identical_copy_is_recognized(tmp_path, env):
    copy = tmp_path/'round11_preparation/plan-campaign.json'
    copy.parent.mkdir(); copy.write_bytes(env['path'].read_bytes())
    write_json(env['runtime']/'launch.json', dict(plan=str(copy),
        plan_sha256=env['header']['plan_sha256'], campaign_identity=env['header']['recipe_sha256']))
    args = argv(env, env['package']/'paper3r11/platform/entry.py')
    args[args.index('--plan')+1] = str(copy)
    make_proc(tmp_path/'proc', 90001, args)
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is not None


def test_foreign_uid_and_zombie_rejected(tmp_path, env, monkeypatch):
    make_proc(tmp_path/'proc', 90001, argv(env, s.HERE/'resume_io.py'), state='Z')
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None
    make_proc(tmp_path/'proc', 90001, argv(env, s.HERE/'resume_io.py'))
    actual = os.getuid(); monkeypatch.setattr(s.os, 'getuid', lambda: actual+1)
    assert s.controller_identity(90001, env, proc_root=tmp_path/'proc') is None


def process(pid=90001, **extra):
    return dict(pid=pid, uid=os.getuid(), start_ticks='12345', args=['exact', 'argv'], **extra)


def test_pid_reuse_prevents_any_signal(env):
    calls = []
    identities = iter([process(), dict(process(), start_ticks='different')])
    result = s.stop_controller(90001, env, identify=lambda *_: next(identities),
                               send=lambda *args: calls.append(args))
    assert result['status'] == 'PID_IDENTITY_CHANGED'
    assert calls == []


def test_graceful_controller_stop_never_signals_group_or_workers(env):
    alive = [True]; calls = []
    def send(pid, sig):
        calls.append((pid, sig)); alive[0] = False
    result = s.stop_controller(90001, env, identify=lambda *_: process() if alive[0] else None,
                               send=send, sleep=lambda _: pytest.fail('should already be ended'))
    assert calls == [(90001, signal.SIGTERM)]
    assert result['scientific_workers_signalled'] == 0


def test_hung_controller_escalation_rechecks_exact_identity(env):
    calls = []; clock = [0.]
    def sleep(delay): clock[0] += delay
    result = s.stop_controller(90001, env, identify=lambda *_: process(),
        send=lambda *args: calls.append(args), sleep=sleep, clock=lambda: clock[0], grace=.2)
    assert calls == [(90001, signal.SIGTERM), (90001, signal.SIGKILL)]
    assert result['scientific_workers_signalled'] == 0


@pytest.fixture
def start_fixture(tmp_path, env, monkeypatch):
    work = tmp_path/'work'
    args = Namespace(submit=True, package=None, plan=env['path'], work=work, python=sys.executable)
    monkeypatch.setattr(s, 'load_env', lambda _: env)
    monkeypatch.setattr(s, 'validate_operational_history', lambda _: {'known_job_ids': ['775288', '775289']})
    return args, env, work


def no_popen(*_, **__):
    pytest.fail('must not start another controller')


def test_submit_requires_explicit_flag_before_loading_or_mutating(start_fixture):
    args, env, work = start_fixture; args.submit = False
    with pytest.raises(ValueError, match='explicit --submit'):
        s.start(args, popen=no_popen)
    assert not work.exists()


def test_unknown_live_controller_left_untouched(start_fixture, monkeypatch):
    args, env, work = start_fixture
    write_json(env['runtime']/'controller-process.json', {'pid': 90001})
    monkeypatch.setattr(s, 'lock_live', lambda _: True)
    result = s.start(args, popen=no_popen, identify=lambda *_: None,
                     stop=lambda *_: pytest.fail('unowned controller signalled'))
    assert result['status'] == 'CONTROLLER_OWNERSHIP_UNCONFIRMED_NO_ACTION'
    assert result['scientific_workers_signalled'] == result['slurm_jobs_cancelled'] == 0


def test_start_detached_and_accepts_only_matching_ready_nonce(start_fixture, monkeypatch):
    args, env, work = start_fixture; monkeypatch.setattr(s, 'lock_live', lambda _: False)
    seen = []
    def popen(command, **kwargs):
        seen.append((command, kwargs))
        nonce = command[command.index('--launch-token')+1]
        write_json(work/'controller-ready.json', dict(pid=90002, launch_token=nonce,
            recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
            atomic_patch_loaded=True, memory_patch_loaded=True, queue_observation_successful=True))
        return SimpleNamespace(pid=90002, poll=lambda: None)
    result = s.start(args, popen=popen)
    assert result['status'] == 'IO_REPAIR_CONTROLLER_READY'
    assert len(seen) == 1
    command, kwargs = seen[0]
    assert command[5:7] == [str(s.HERE/'resume_io.py'), 'controller']
    assert '--submit' in command and kwargs['start_new_session'] is True and kwargs['close_fds'] is True
    assert result['slurm_jobs_cancelled'] == result['scientific_workers_signalled'] == 0
    assert result['actual_gpu_allocation_claimed'] is False
    assert result['upload_processes_modified'] == 0


def test_owned_old_controller_only_stopped_before_new_spawn(start_fixture, monkeypatch):
    args, env, work = start_fixture; live = [True]; stopped = []
    write_json(env['runtime']/'controller-process.json', {'pid': 90001})
    monkeypatch.setattr(s, 'lock_live', lambda _: live[0])
    def stop(pid, _):
        stopped.append(pid); live[0] = False
        return {'status': 'OWNED_CONTROLLER_STOP_REQUESTED', 'pid': pid}
    clock = [0.]
    def sleep(delay): clock[0] += delay
    launched = []
    def popen(command, **kwargs):
        launched.append(command)
        return SimpleNamespace(pid=90002, poll=lambda: None)
    result = s.start(args, popen=popen, identify=lambda *_: process(current_patch=False),
                     stop=stop, sleep=sleep, clock=lambda: clock[0])
    assert stopped == [90001] and len(launched) == 1
    assert result['status'] == 'STARTUP_PENDING'
    assert result['terminal_can_close'] is True


def test_current_controller_matching_nonce_reused(start_fixture, monkeypatch):
    args, env, work = start_fixture
    write_json(env['runtime']/'controller-process.json', {'pid': 90001})
    owned = process(current_patch=True); owned['args'] += ['--launch-token', 'nonce']
    write_json(work/'controller-ready.json', dict(pid=90001, launch_token='nonce',
        recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
        memory_patch_loaded=True, atomic_patch_loaded=True, queue_observation_successful=True))
    monkeypatch.setattr(s, 'lock_live', lambda _: True)
    result = s.start(args, identify=lambda *_: owned, popen=no_popen,
                     stop=lambda *_: pytest.fail('reused controller stopped'))
    assert result['status'] == 'EXISTING_IO_CONTROLLER_REUSED'


def test_current_controller_pending_not_restarted(start_fixture, monkeypatch):
    args, env, work = start_fixture
    write_json(env['runtime']/'controller-process.json', {'pid': 90001})
    owned = process(current_patch=True); owned['args'] += ['--launch-token', 'new-nonce']
    write_json(work/'controller-process.json', dict(pid=90001, launch_token='new-nonce',
        recipe_sha256=env['header']['recipe_sha256']))
    # A different launch's ready file is not acknowledgement for this process.
    write_json(work/'controller-ready.json', dict(pid=90001, launch_token='old-nonce',
        recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
        atomic_patch_loaded=True, memory_patch_loaded=True, queue_observation_successful=True))
    monkeypatch.setattr(s, 'lock_live', lambda _: True)
    result = s.start(args, identify=lambda *_: owned, popen=no_popen,
                     stop=lambda *_: pytest.fail('owned startup controller restarted'))
    assert result['status'] == 'EXISTING_IO_CONTROLLER_STARTUP_PENDING'


def test_current_controller_preload_without_lock_not_spawned_twice(start_fixture, monkeypatch):
    args, env, work = start_fixture
    owned = process(current_patch=True); owned['args'] += ['--launch-token', 'preload-nonce']
    write_json(work/'controller-process.json', dict(pid=90001, launch_token='preload-nonce',
        recipe_sha256=env['header']['recipe_sha256']))
    monkeypatch.setattr(s, 'lock_live', lambda _: False)
    result = s.start(args, identify=lambda *_: owned, popen=no_popen,
                     stop=lambda *_: pytest.fail('owned preload controller restarted'))
    assert result['status'] == 'EXISTING_IO_CONTROLLER_STARTUP_PENDING'


def test_new_process_exit_cannot_claim_ready(start_fixture, monkeypatch):
    args, env, work = start_fixture; monkeypatch.setattr(s, 'lock_live', lambda _: False)
    def popen(command, **kwargs):
        nonce = command[command.index('--launch-token')+1]
        write_json(work/'controller-ready.json', dict(pid=90002, launch_token=nonce,
            recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
            atomic_patch_loaded=True, memory_patch_loaded=True, queue_observation_successful=True))
        return SimpleNamespace(pid=90002, poll=lambda: 2)
    assert s.start(args, popen=popen)['status'] == 'STARTUP_FAILED'


def test_partial_ready_without_atomic_patch_ack_stays_pending(start_fixture, monkeypatch):
    args, env, work = start_fixture; monkeypatch.setattr(s, 'lock_live', lambda _: False)
    clock = [0.]
    def sleep(delay): clock[0] += delay
    def popen(command, **kwargs):
        nonce = command[command.index('--launch-token')+1]
        write_json(work/'controller-ready.json', dict(pid=90002, launch_token=nonce,
            recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
            queue_observation_successful=True))
        return SimpleNamespace(pid=90002, poll=lambda: None)
    result = s.start(args, popen=popen, sleep=sleep, clock=lambda: clock[0])
    assert result['status'] == 'STARTUP_PENDING'


def test_unsuccessful_queue_observation_not_reported_ready(start_fixture, monkeypatch):
    args, env, work = start_fixture; monkeypatch.setattr(s, 'lock_live', lambda _: False)
    clock = [0.]
    def sleep(delay): clock[0] += delay
    def popen(command, **kwargs):
        nonce = command[command.index('--launch-token')+1]
        write_json(work/'controller-ready.json', dict(pid=90002, launch_token=nonce,
            recipe_sha256=env['header']['recipe_sha256'], first_slurm_observation_completed=True,
            atomic_patch_loaded=True, memory_patch_loaded=True, queue_observation_successful=False))
        return SimpleNamespace(pid=90002, poll=lambda: None)
    result = s.start(args, popen=popen, sleep=sleep, clock=lambda: clock[0])
    assert result['status'] == 'STARTUP_PENDING'
