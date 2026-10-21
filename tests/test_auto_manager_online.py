"""Tiny fixtures only: no network, credentials, GPU or real tmux jobs."""
import fcntl
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper3_auto_manager as manager


def make_proc(root, pid, cwd, args, parent=1):
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / 'cmdline').write_bytes(b'\0'.join(os.fsencode(a) for a in args) + b'\0')
    (entry / 'cwd').symlink_to(cwd, target_is_directory=True)
    (entry / 'status').write_text('Name:\tpython3\nPPid:\t' + str(parent) + '\n')
    return entry


def test_active_jobs_exact_script_cwd_and_scope_no_environment_read(tmp_path, monkeypatch):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    bundle = workspace / manager.TOP; bundle.mkdir()
    outside = tmp_path / 'elsewhere'; outside.mkdir()
    proc = tmp_path / 'proc'; proc.mkdir()
    make_proc(proc, 201, bundle, ['python3', '-B', 'paper3_one_click.py'])
    make_proc(proc, 202, outside, ['python3', str(bundle / 'paper3_one_click.py')])
    make_proc(proc, 203, outside, ['python3', 'paper3_one_click.py'])
    make_proc(proc, 204, bundle, ['python3', 'other_paper3_one_click.py'])
    for entry in proc.iterdir():
        (entry / 'environ').write_text('fake-secret-must-never-be-read')
    original = Path.read_bytes
    def guarded(path):
        assert path.name != 'environ'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    jobs = manager.active_jobs(workspace, proc)
    assert {job['pid'] for job in jobs} == {201, 202}
    assert all(job['bundle'] == bundle for job in jobs)


def test_active_jobs_recognizes_frozen_core_and_inside_worker(tmp_path):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    project = workspace / 'release' / 'paper3_data_plan_603d299351fa6605'; project.mkdir(parents=True)
    helpers = workspace / '.paper3-auto-tools'; helpers.mkdir()
    proc = tmp_path / 'proc'; proc.mkdir()
    make_proc(proc, 201, project, ['python3', 'tools/bootstrap_data_plan.py'])
    make_proc(proc, 202, project, ['python3', 'tools/run_training_campaign.py'])
    make_proc(proc, 203, helpers, ['python3', 'PAPER3_AUTO_RUN.py', '--inside'])
    make_proc(proc, 204, helpers, ['python3', 'PAPER3_AUTO_RUN.py'])
    assert {job['pid'] for job in manager.active_jobs(workspace, proc)} == {201, 202, 203}


def test_active_jobs_normalizes_dotdot_before_workspace_scope(tmp_path):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    outside_bundle = tmp_path / 'outside' / manager.TOP; outside_bundle.mkdir(parents=True)
    proc = tmp_path / 'proc'; proc.mkdir()
    escaped = workspace / '..' / 'outside' / manager.TOP / 'paper3_one_click.py'
    make_proc(proc, 201, workspace, ['python3', str(escaped)])
    assert manager.active_jobs(workspace, proc) == []


def test_pane_mapping_uses_ancestor_chain_not_session_name(tmp_path):
    proc = tmp_path / 'proc'; proc.mkdir()
    cwd = tmp_path / 'cwd'; cwd.mkdir()
    make_proc(proc, 100, cwd, ['bash'], parent=1)
    make_proc(proc, 200, cwd, ['python3'], parent=100)
    make_proc(proc, 300, cwd, ['python3'], parent=200)
    wrong = {'pid': 999, 'session': '$1', 'pane': '%1'}
    right = {'pid': 100, 'session': '$2', 'pane': '%2'}
    assert manager.pane_for_jobs([{'pid': 300}], [wrong, right], proc) is right
    assert manager.pane_for_jobs([{'pid': 300}], [wrong], proc) is None


@pytest.mark.parametrize('relative', ['launcher.owner.lock',
    'release/paper3_data_plan_603d299351fa6605/.paper3-deployment/owner.lock',
    'release/paper3_data_plan_603d299351fa6605/runs/.training_campaign_v1.lock'])
def test_any_original_held_lock_blocks_new_job(tmp_path, relative):
    bundle = tmp_path / manager.TOP
    path = bundle / manager.WORK / relative
    path.parent.mkdir(parents=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert manager.original_locks_busy(bundle)
    assert not manager.original_locks_busy(bundle)


def test_original_symlink_lock_is_refused(tmp_path):
    bundle = tmp_path / manager.TOP
    root = bundle / manager.WORK; root.mkdir(parents=True)
    target = tmp_path / 'foreign'; target.write_text('keep')
    (root / 'launcher.owner.lock').symlink_to(target)
    with pytest.raises(manager.AutoError, match='^SYMLINK_PATH_REFUSED$'):
        manager.original_locks_busy(bundle)


def tiny_zip(workspace, monkeypatch, members):
    source = workspace / manager.ZIP_NAME
    with zipfile.ZipFile(source, 'w') as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    monkeypatch.setattr(manager, 'ZIP_SHA', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(manager, 'ZIP_BYTES', source.stat().st_size)
    return source


def test_exact_zip_merge_preserves_work_and_identical_source(tmp_path, monkeypatch):
    tiny_zip(tmp_path, monkeypatch, [(manager.TOP + '/START.sh', b'fixed\n')])
    root = manager.ensure_bundle(tmp_path)
    checkpoint = root / manager.WORK / 'checkpoint.pt'
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b'actual-checkpoint')
    inode = (root / 'START.sh').stat().st_ino
    assert manager.ensure_bundle(tmp_path) == root
    assert (root / 'START.sh').stat().st_ino == inode
    assert checkpoint.read_bytes() == b'actual-checkpoint'


def test_missing_zip_uses_embedded_fixed_release_fetch_once(tmp_path, monkeypatch):
    source = tiny_zip(tmp_path, monkeypatch, [(manager.TOP + '/START.sh', b'fixed\n')])
    payload = source.read_bytes()
    source.unlink()
    marker = tmp_path / 'fixture_fetch_called'
    stub = '''from pathlib import Path
def fetch_release_asset(repo, tag, name, sha256, byte_count, destination):
    assert repo == {repo!r} and tag == {tag!r}
    assert name == {name!r} and sha256 == {sha!r} and byte_count == {count!r}
    assert not destination.exists()
    Path({marker!r}).write_text('called once')
    destination.write_bytes({payload!r})
    return {{'status': 'FIXTURE_PINNED_DOWNLOAD'}}
'''.format(repo=manager.ONLINE_REPO, tag=manager.ONLINE_TAG, name=manager.ZIP_NAME,
           sha=manager.ZIP_SHA, count=manager.ZIP_BYTES, marker=str(marker), payload=payload)
    monkeypatch.setattr(manager, 'FETCH_SOURCE', stub)
    root = manager.ensure_bundle(tmp_path)
    assert marker.read_text() == 'called once'
    assert (root / 'START.sh').read_bytes() == b'fixed\n'
    # Existing verified bytes must completely bypass the embedded fetcher.
    monkeypatch.setattr(manager, 'FETCH_SOURCE', "raise RuntimeError('must not fetch twice')")
    assert manager.ensure_bundle(tmp_path) == root


def test_existing_zip_never_calls_download_even_when_fetch_is_unavailable(tmp_path, monkeypatch):
    tiny_zip(tmp_path, monkeypatch, [(manager.TOP + '/START.sh', b'fixed\n')])
    monkeypatch.setattr(manager, 'FETCH_SOURCE', None)
    monkeypatch.setattr(manager, 'download_bundle', lambda workspace: pytest.fail('existing ZIP must not fetch'))
    assert manager.ensure_bundle(tmp_path) == tmp_path / manager.TOP


def test_zip_checksum_failure_happens_before_extract(tmp_path, monkeypatch):
    tiny_zip(tmp_path, monkeypatch, [(manager.TOP + '/START.sh', b'fixed\n')])
    monkeypatch.setattr(manager, 'ZIP_SHA', '0' * 64)
    with pytest.raises(manager.AutoError, match='^ORIGINAL_ZIP_SHA_MISMATCH$'):
        manager.ensure_bundle(tmp_path)
    assert not (tmp_path / manager.TOP).exists()


def test_changed_existing_file_never_overwritten(tmp_path, monkeypatch):
    tiny_zip(tmp_path, monkeypatch, [(manager.TOP + '/START.sh', b'fixed\n')])
    root = tmp_path / manager.TOP; root.mkdir()
    target = root / 'START.sh'; target.write_bytes(b'my changed file')
    with pytest.raises(manager.AutoError, match='^EXISTING_LAUNCHER_FILE_DIFFERS_PRESERVED$'):
        manager.ensure_bundle(tmp_path)
    assert target.read_bytes() == b'my changed file'


@pytest.mark.parametrize('name', ['Paper3_OneClick/../outside', '/outside', 'Other/start',
                                 'Paper3_OneClick/bad\\name'])
def test_unsafe_zip_member_refused(tmp_path, monkeypatch, name):
    tiny_zip(tmp_path, monkeypatch, [(name, b'fixed')])
    with pytest.raises(manager.AutoError, match='^ZIP_MEMBER_REFUSED$'):
        manager.ensure_bundle(tmp_path)


def mock_main(tmp_path, monkeypatch):
    pane = {'session': '$1', 'window': '@2', 'pane': '%3', 'pid': 100}
    helpers = tmp_path / 'helpers'; helpers.mkdir()
    monkeypatch.setattr(manager.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(manager.shutil, 'which', lambda value: '/fixture/' + value)
    monkeypatch.setattr(manager, 'install_helpers', lambda workspace: helpers)
    monkeypatch.setattr(manager, 'list_panes', lambda: [pane])
    monkeypatch.setattr(manager, 'pane_for_jobs', lambda jobs, panes: pane)
    monkeypatch.setattr(manager, 'reserved_pane', lambda bundle: None)
    calls = []
    monkeypatch.setattr(manager, 'add_status_pane', lambda *args: calls.append('status'))
    def attach(target):
        assert target == pane
        # The long-lived attach must not inherit the short manager lock.
        with manager.short_manager_lock(tmp_path):
            pass
        calls.append('attach')
        return 0
    monkeypatch.setattr(manager, 'attach', attach)
    return pane, calls


def test_repeated_invocation_only_attaches_existing_job(tmp_path, monkeypatch):
    pane, calls = mock_main(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, 'active_jobs', lambda workspace: [{'pid': 201, 'bundle': workspace / manager.TOP}])
    monkeypatch.setattr(manager, 'ensure_bundle', lambda *a: pytest.fail('must not reextract'))
    monkeypatch.setattr(manager, 'create_job', lambda *a: pytest.fail('must not restart'))
    for _ in range(2):
        assert manager.main(['--workspace', str(tmp_path)]) == 0
    assert calls == ['status', 'attach', 'status', 'attach']


def test_existing_job_outside_tmux_is_only_monitored_and_lock_released(tmp_path, monkeypatch):
    pane, calls = mock_main(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, 'active_jobs', lambda workspace: [{'pid': 201, 'bundle': workspace / manager.TOP}])
    monkeypatch.setattr(manager, 'pane_for_jobs', lambda *args: None)
    monkeypatch.setattr(manager, 'ensure_bundle', lambda *a: pytest.fail('must not reextract'))
    monkeypatch.setattr(manager, 'create_job', lambda *a: pytest.fail('must not restart'))
    def monitor(command):
        with manager.short_manager_lock(tmp_path):
            pass
        calls.append('monitor')
        assert command[-2:] == ['--bundle', str(tmp_path / manager.TOP)]
        return 0
    monkeypatch.setattr(manager.subprocess, 'call', monitor)
    assert manager.main(['--workspace', str(tmp_path)]) == 0
    assert calls == ['monitor']


def test_held_core_lock_blocks_without_any_new_job(tmp_path, monkeypatch):
    pane, calls = mock_main(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, 'active_jobs', lambda workspace: [])
    monkeypatch.setattr(manager, 'original_locks_busy', lambda bundle: True)
    monkeypatch.setattr(manager, 'ensure_bundle', lambda *a: pytest.fail('must not extract'))
    monkeypatch.setattr(manager, 'create_job', lambda *a: pytest.fail('must not restart'))
    assert manager.main(['--workspace', str(tmp_path)]) == 2
    assert calls == []


def test_idle_entry_creates_one_job_then_attaches(tmp_path, monkeypatch):
    pane, calls = mock_main(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, 'active_jobs', lambda workspace: [])
    monkeypatch.setattr(manager, 'original_locks_busy', lambda bundle: False)
    def ensure(workspace):
        calls.append('extract'); return workspace / manager.TOP
    def create(*args):
        calls.append('create'); return pane
    monkeypatch.setattr(manager, 'ensure_bundle', ensure)
    monkeypatch.setattr(manager, 'create_job', create)
    assert manager.main(['--workspace', str(tmp_path)]) == 0
    assert calls == ['extract', 'create', 'status', 'attach']


def test_reserved_startup_pane_never_starts_another_job(tmp_path, monkeypatch):
    pane, calls = mock_main(tmp_path, monkeypatch)
    monkeypatch.setattr(manager, 'active_jobs', lambda workspace: [])
    monkeypatch.setattr(manager, 'reserved_pane', lambda bundle: pane)
    monkeypatch.setattr(manager, 'ensure_bundle', lambda *a: pytest.fail('must not extract'))
    monkeypatch.setattr(manager, 'create_job', lambda *a: pytest.fail('must not restart'))
    assert manager.main(['--workspace', str(tmp_path)]) == 0
    assert calls == ['status', 'attach']


def test_status_pane_does_not_steal_focus(tmp_path, monkeypatch):
    pane = {'session': '$1', 'window': '@2', 'pane': '%3', 'pid': 100}
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == 'show-options':return subprocess.CompletedProcess(command, 0, '', '')
        if command[1] == 'split-window':
            assert '-d' in command and command[command.index('-t') + 1] == '%3'
            return subprocess.CompletedProcess(command, 0, '%4\n', '')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(manager.subprocess, 'run', fake_run)
    monkeypatch.setattr(manager, 'list_panes', lambda: [pane])
    manager.add_status_pane(pane, tmp_path, tmp_path / manager.TOP)
    assert calls[-1][1:3] == ['set-option', '-w']
    assert all('send-keys' not in command for command in calls)


def upgrade_fixture(tmp_path, monkeypatch, argv, *, marker='%4', version='older-version'):
    main = {'session': '$1', 'window': '@2', 'pane': '%3', 'pid': 100}
    monitor = {'session': '$1', 'window': '@2', 'pane': '%4', 'pid': 200}
    helpers = tmp_path / '.paper3-auto-tools' / 'new-version'
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == 'show-options':
            output = marker if command[-1] == '@paper3_status_pane' else version
            return subprocess.CompletedProcess(command, 0, output, '')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(manager.subprocess, 'run', fake_run)
    monkeypatch.setattr(manager, 'list_panes', lambda: [main, monitor])
    original = Path.read_bytes
    def fake_cmdline(path):
        if path.as_posix() in {'/proc/100/cmdline', '/proc/200/cmdline'}:
            return b'\0'.join(os.fsencode(arg) for arg in argv) + b'\0'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', fake_cmdline)
    manager.add_status_pane(main, helpers, tmp_path / manager.TOP)
    return calls


def test_only_identified_old_readonly_monitor_is_respawned(tmp_path, monkeypatch):
    argv = ['python3', '-B', str(tmp_path / '.paper3-auto-tools' / 'older-version' / 'paper3_status.py')]
    calls = upgrade_fixture(tmp_path, monkeypatch, argv)
    respawn = [command for command in calls if command[1] == 'respawn-pane']
    assert len(respawn) == 1 and respawn[0][2:5] == ['-k', '-t', '%4']
    assert '%3' not in respawn[0]
    assert all(command[1] != 'split-window' for command in calls)
    assert calls[-1][-2:] == ['@paper3_status_version', 'new-version']


def test_marker_pointing_to_training_pid_never_respawns_it(tmp_path, monkeypatch):
    argv = ['python3', '-B', str(tmp_path / manager.TOP / 'paper3_one_click.py')]
    calls = upgrade_fixture(tmp_path, monkeypatch, argv, marker='%3')
    assert all(command[1] not in {'respawn-pane', 'split-window', 'kill-pane', 'send-keys'}
               for command in calls)


def test_current_readonly_monitor_is_not_needlessly_respawned(tmp_path, monkeypatch):
    argv = ['python3', '-B', str(tmp_path / '.paper3-auto-tools' / 'new-version' / 'paper3_status.py')]
    calls = upgrade_fixture(tmp_path, monkeypatch, argv, version='new-version')
    assert all(command[1] not in {'respawn-pane', 'split-window'} for command in calls)


def test_unowned_similar_name_monitor_is_not_respawned(tmp_path, monkeypatch):
    argv = ['python3', '-B', str(tmp_path / 'unrelated' / 'paper3_status.py')]
    calls = upgrade_fixture(tmp_path, monkeypatch, argv)
    assert all(command[1] not in {'respawn-pane', 'split-window'} for command in calls)


def test_attach_selects_real_launcher_pane_and_existing_tmux_client(monkeypatch):
    pane = {'session': '$1', 'window': '@2', 'pane': '%3', 'pid': 100}
    calls = []
    monkeypatch.setenv('TMUX', 'fixture')
    monkeypatch.setattr(manager, 'run', lambda command: calls.append(command))
    monkeypatch.setattr(manager.os, 'execvp', lambda *a: pytest.fail('must not nest'))
    assert manager.attach(pane) == 0
    assert calls == [['tmux', 'select-window', '-t', '@2'],
                     ['tmux', 'select-pane', '-t', '%3'],
                     ['tmux', 'switch-client', '-t', '$1']]
