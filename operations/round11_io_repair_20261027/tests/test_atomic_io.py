from __future__ import annotations

import errno
import json
import multiprocessing
import os
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import atomic_io as io


def test_disappearing_temporary_retried_without_partial_target(tmp_path, monkeypatch):
    target = tmp_path / 'status.json'
    target.write_bytes(b'{"old":true}\n')
    original = io.os.replace
    sources, delays, observed = [], [], []

    def disappearing(source, dest):
        source = Path(source)
        sources.append(source)
        assert source.parent.parent == tmp_path
        assert source.parent.stat().st_mode & 0o777 == 0o700
        assert source.name == 'payload.bin'
        observed.append(target.read_bytes())
        if len(sources) < 3:
            source.unlink()
            raise FileNotFoundError(errno.ENOENT, 'simulated shared filesystem disappearance')
        return original(source, dest)

    monkeypatch.setattr(io.os, 'replace', disappearing)
    io.atomic_bytes(target, b'{"new":true}\n', sleep=delays.append)
    assert target.read_bytes() == b'{"new":true}\n'
    assert observed == [b'{"old":true}\n'] * 3
    assert len({p.parent for p in sources}) == 3
    assert delays == [0.05, 0.15]
    assert list(tmp_path.iterdir()) == [target]


def test_ambiguous_rename_success_verified_byte_for_byte(tmp_path, monkeypatch):
    target = tmp_path / 'completion.json'
    original = io.os.replace
    calls = []
    data = b'\x00\xffscientific bytes' * 1000

    def commit_then_lose_ack(source, dest):
        calls.append(source)
        original(source, dest)
        raise OSError(errno.ENOENT, 'NFS rename reply lost')

    monkeypatch.setattr(io.os, 'replace', commit_then_lose_ack)
    io.atomic_bytes(target, data, sleep=lambda _: pytest.fail('must not retry committed bytes'))
    assert target.read_bytes() == data
    assert len(calls) == 1


def test_false_ambiguous_success_cannot_reuse_old_different_payload(tmp_path, monkeypatch):
    target = tmp_path / 'completion.json'
    target.write_bytes(b'{"old":true}')
    calls = []

    def never_commit(source, dest):
        calls.append(source)
        raise OSError(errno.ESTALE, 'stale metadata')

    monkeypatch.setattr(io.os, 'replace', never_commit)
    with pytest.raises(OSError) as failure:
        io.atomic_bytes(target, b'{"new":true}', attempts=3, sleep=lambda _: None)
    assert failure.value.errno == errno.ESTALE
    assert len(calls) == 3
    assert target.read_bytes() == b'{"old":true}'
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('code', [errno.EACCES, errno.ENOSPC, errno.EXDEV, errno.EROFS])
def test_permanent_errors_raise_and_preserve_old_file(tmp_path, monkeypatch, code):
    target = tmp_path / 'result.json'
    target.write_bytes(b'original scientific result')
    calls = []

    def reject(source, dest):
        calls.append(source)
        raise OSError(code, 'permanent filesystem failure')

    monkeypatch.setattr(io.os, 'replace', reject)
    with pytest.raises(OSError) as failure:
        io.atomic_bytes(target, b'replacement', sleep=lambda _: pytest.fail('permanent error retried'))
    assert failure.value.errno == code
    assert len(calls) == 1
    assert target.read_bytes() == b'original scientific result'
    assert list(tmp_path.iterdir()) == [target]


def test_fsync_interruption_retried(tmp_path, monkeypatch):
    target = tmp_path / 'tasks.json'
    original = io.os.fsync
    calls = []

    def interrupt_once(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise InterruptedError(errno.EINTR, 'interrupted sync')
        return original(fd)

    monkeypatch.setattr(io.os, 'fsync', interrupt_once)
    io.atomic_bytes(target, b'new state', sleep=lambda _: None)
    assert target.read_bytes() == b'new state'
    assert len(calls) == 2


def test_cleanup_never_sweeps_another_writer_or_stale_file(tmp_path):
    foreign = tmp_path / '.r11-atomic-foreign'
    foreign.mkdir()
    payload = foreign / 'payload.bin'
    payload.write_bytes(b'foreign in-flight payload')
    old = tmp_path / '.status.json.oldtmp'
    old.write_bytes(b'old file owned by someone else')
    target = tmp_path / 'status.json'
    io.atomic_bytes(target, b'committed')
    assert payload.read_bytes() == b'foreign in-flight payload'
    assert old.read_bytes() == b'old file owned by someone else'


def _writer(path, writer, count):
    for n in range(count):
        record = dict(writer=writer, n=n, payload=str(writer) * 5000)
        io.atomic_bytes(path, (json.dumps(record) + '\n').encode())


def test_concurrent_processes_publish_only_whole_json(tmp_path):
    target = tmp_path / 'status.json'
    target.write_text(json.dumps(dict(writer=-1, n=0, payload='initial')))
    ctx = multiprocessing.get_context('fork')
    workers = [ctx.Process(target=_writer, args=(target, n, 16)) for n in range(4)]
    for worker in workers:
        worker.start()
    observations = 0
    while any(worker.is_alive() for worker in workers):
        value = json.loads(target.read_bytes())
        assert set(value) == {'writer', 'n', 'payload'}
        observations += 1
    for worker in workers:
        worker.join(timeout=5)
        assert worker.exitcode == 0
    assert observations > 0
    assert list(tmp_path.iterdir()) == [target]


def test_patch_exact_imported_aliases_only_and_preserve_encoding(tmp_path, monkeypatch):
    def old_bytes(path, data):
        raise AssertionError('old atomic path called')
    def old_atomic(path, data):
        raise AssertionError('old atomic alias called')
    common = types.SimpleNamespace(atomic_bytes=old_bytes, atomic=old_atomic,
        encoded=lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    alias = types.ModuleType('paper3r11.example')
    alias.atomic_bytes = old_bytes
    alias.atomic = old_atomic
    untouched = types.ModuleType('unrelated.example')
    untouched.atomic = old_atomic
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    monkeypatch.setitem(sys.modules, untouched.__name__, untouched)
    result = io.install_atomic_patch(common)
    assert result['aliases'] == 2
    assert result['source_files_changed'] is False
    assert untouched.atomic is old_atomic
    expected = common.encoded({'z': 1, 'a': '\u8111\u7535'})
    target = tmp_path / 'result.json'
    alias.atomic(target, {'z': 1, 'a': '\u8111\u7535'})
    assert target.read_bytes() == expected
    assert io.install_atomic_patch(common)['status'] == 'ALREADY_PATCHED'


@pytest.mark.parametrize('attempts', [0, -1, True, 1.2])
def test_invalid_retry_budget_rejected_before_write(tmp_path, attempts):
    with pytest.raises(ValueError):
        io.atomic_bytes(tmp_path / 'not-created', b'x', attempts=attempts)
    assert list(tmp_path.iterdir()) == []


def test_scheduler_atomic_has_exact_original_json_encoding_and_retry(tmp_path, monkeypatch):
    common = types.SimpleNamespace(atomic_bytes=lambda *_: None, atomic=lambda *_: None,
        encoded=lambda value: (json.dumps(value, sort_keys=True, indent=2) + '\n').encode())
    scheduler = types.ModuleType('paper3r11.platform.scheduler')
    scheduler._atomic = lambda *_: pytest.fail('old scheduler write path called')
    monkeypatch.setitem(sys.modules, scheduler.__name__, scheduler)
    receipt = io.install_patch(common)
    assert receipt['scheduler_patched'] is True
    original = io.os.replace
    seen = []
    def fail_once(source, dest):
        seen.append(source)
        if len(seen) == 1:
            raise FileNotFoundError(errno.ENOENT, 'missing temporary')
        return original(source, dest)
    monkeypatch.setattr(io.os, 'replace', fail_once)
    value = {'node': '脑电', 'z': [1, 2.2], 'a': True}
    target = tmp_path / 'scheduler-state.json'
    scheduler._atomic(target, value)
    assert target.read_bytes() == (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    assert len(seen) == 2
    assert io.install_patch(common)['scheduler_patched'] is False


def test_actual_original_common_fails_on_disappearance_and_repair_preserves_bytes(tmp_path, monkeypatch):
    import importlib.util
    original_path = Path(os.environ.get('PAPER3_R11_ORIGINAL_PACKAGE', '/workspace/paper3-round11')) / 'paper3r11/platform/common.py'
    if not original_path.is_file():
        pytest.skip('original immutable package needed for production regression fixture')
    spec = importlib.util.spec_from_file_location('r11_original_atomic_regression_fixture', original_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    original_sha = __import__('hashlib').sha256(original_path.read_bytes()).hexdigest()
    target = tmp_path / 'status.json'
    target.write_bytes(b'original target')
    original_replace = io.os.replace
    calls = []
    fail_next = [True]
    def disappear_once(source, dest):
        calls.append(Path(source))
        if fail_next[0]:
            fail_next[0] = False
            Path(source).unlink()
            raise FileNotFoundError(errno.ENOENT, 'same failure shape as school traceback')
        return original_replace(source, dest)
    monkeypatch.setattr(io.os, 'replace', disappear_once)
    value = {'status': 'COMPLETE', 'scientific_scope': 'fixture', 'result': '\u8111\u7535'}
    encoded_before = common.encoded(value)
    with pytest.raises(FileNotFoundError):
        common.atomic(target, value)
    assert len(calls) == 1
    assert target.read_bytes() == b'original target'
    assert calls[0].parent == tmp_path
    calls.clear()
    fail_next[0] = True
    io.install_patch(common)
    common.atomic(target, value)
    assert len(calls) == 2
    assert target.read_bytes() == encoded_before
    assert all(source.parent.parent == tmp_path for source in calls)
    assert __import__('hashlib').sha256(original_path.read_bytes()).hexdigest() == original_sha
