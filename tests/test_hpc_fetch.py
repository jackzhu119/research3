"""Tiny local objects and fake streams; no real R2, school or model fit."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from hpc import fetch
from hpc import stage_data


class TrackedBody(io.BytesIO):
    eof = False
    def read(self, amount=-1):
        result = super().read(amount)
        if result == b'':
            self.eof = True
        return result


class ReadTimeoutError(Exception):
    pass


class FailedRead(TrackedBody):
    def read(self, amount=-1):
        if self.tell():
            raise ReadTimeoutError('DO_NOT_PRINT_SECRET_ENDPOINT')
        return super().read(3)


class FakeR2:
    def __init__(self, responses):
        self.responses, self.calls, self.streams = list(responses), [], []
    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        selected = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(selected, Exception):
            raise selected
        if isinstance(selected, bytes):
            body = TrackedBody(selected)
            self.streams.append(body)
            return {'Body': body, 'ContentLength': len(selected),
                    'ResponseMetadata': {'HTTPStatusCode': 200}}
        body = selected.get('Body')
        if body is not None:
            self.streams.append(body)
        return selected


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    content = b'fixture-complete-stream\x00' * 100
    row = {'path': 'archives/fixed.tar.gz', 'key': 'paper3/fixed/archives/fixed.tar.gz',
           'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    monkeypatch.setattr(stage_data, 'load_release', lambda: {'objects': {'main_archive': row}})
    monkeypatch.setattr(fetch, 'SLEEP', lambda value: None)
    return tmp_path, row, content


def headers(body, content):
    return {'Body': body, 'ContentLength': len(content), 'ResponseMetadata': {'HTTPStatusCode': 200}}


def test_whole_GET_closed_SHA_verified_and_final_reread(fixed):
    root, row, content = fixed
    client, events = FakeR2([content]), []
    result = fetch.fetch_object(client, 'not-recorded-bucket', row, root, events.append)
    assert result['status'] == 'R2_FULL_GET_SHA_VERIFIED'
    assert result['complete_body_read'] is True and result['bytes_read'] == len(content)
    assert result['get_sha256'] == row['sha256'] and result['local_sha256'] == row['sha256']
    assert result['durability']['file_fsync'] == 'completed'
    assert result['atomic_commit'] == 'link_no_overwrite' and result['attempts'] == 1
    assert (root / row['path']).read_bytes() == content
    assert client.streams[0].eof is True and client.streams[0].closed
    assert events[0]['phase'] == 'get_headers' and events[-1]['phase'] == 'verified'
    assert not list(root.rglob('*.partial'))
    assert 'not-recorded-bucket' not in json.dumps(events) and row['key'] not in json.dumps(events)


def test_existing_identical_is_local_cache_not_fresh_GET(fixed):
    root, row, content = fixed
    target = root / row['path']
    target.parent.mkdir()
    target.write_bytes(content)
    client = FakeR2([content])
    result = fetch.fetch_object(client, 'bucket', row, root)
    assert result['status'] == 'LOCAL_VERIFIED_CACHE_HIT' and result['get_attempted'] is False
    assert result['complete_body_read'] is None and result['attempts'] == 0
    assert client.calls == []


def test_existing_different_file_is_never_overwritten(fixed):
    root, row, content = fixed
    target = root / row['path']
    target.parent.mkdir()
    target.write_bytes(b'existing-original')
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert observed.value.detail['phase'] == 'existing_local_hash'
    assert target.read_bytes() == b'existing-original' and client.calls == []


def test_transport_retries_same_object_at_most_three_whole_GETs(fixed):
    root, row, content = fixed
    client = FakeR2([ReadTimeoutError('SECRET_URL')])
    events = []
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'SECRET_BUCKET', row, root, events.append)
    assert observed.value.code == 'RESPONSE_TIMEOUT'
    assert observed.value.detail['attempt'] == 3 and len(client.calls) == 3
    assert len({call['Key'] for call in client.calls}) == 1
    assert sum(event['phase'] == 'bounded_retry' for event in events) == 2
    assert not (root / row['path']).exists()
    assert 'SECRET' not in json.dumps(observed.value.detail) + json.dumps(events)


def test_body_read_timeout_retries_from_zero_then_succeeds(fixed):
    root, row, content = fixed
    failed = FailedRead(content)
    client = FakeR2([headers(failed, content), content])
    result = fetch.fetch_object(client, 'bucket', row, root)
    assert result['attempts'] == 2 and result['bytes_read'] == len(content)
    assert len(client.calls) == 2 and failed.closed
    assert (root / row['path']).read_bytes() == content
    assert not list(root.rglob('*.partial'))


@pytest.mark.parametrize('bad', ['short', 'wrong_SHA', 'long'])
def test_body_integrity_failure_never_publishes_unverified_bytes(fixed, bad):
    root, row, content = fixed
    payload = content[:-1] if bad == 'short' else b'x' + content[1:] if bad == 'wrong_SHA' else content + b'x'
    client = FakeR2([headers(TrackedBody(payload), content) for _ in range(3)])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code in {'R2_GET_BODY_SHA_OR_LENGTH_MISMATCH', 'R2_GET_BODY_LENGTH_MISMATCH'}
    assert len(client.calls) == 1 and not (root / row['path']).exists()
    assert observed.value.retryable is False
    assert not list(root.rglob('*.partial'))


@pytest.mark.parametrize('status,count', [(403, 1), (404, 1), (429, 3), (503, 3)])
def test_HTTP_errors_are_explicit_and_only_transient_retried(fixed, status, count):
    root, row, _ = fixed
    client = FakeR2([{'ResponseMetadata': {'HTTPStatusCode': status}}])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'R2_GET_HTTP_NOT_200'
    assert observed.value.detail['http_status'] == status and len(client.calls) == count


def test_service_exception_is_redacted_and_fixed_code_preserved(fixed):
    root, row, _ = fixed
    class ClientError(Exception):
        response = {'ResponseMetadata': {'HTTPStatusCode': 403},
                    'Error': {'Code': 'AccessDenied', 'Message': 'SECRET_ENDPOINT_KEY'}}
    client = FakeR2([ClientError('SECRET_BUCKET_TOKEN')])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.detail['service_code'] == 'AccessDenied'
    assert observed.value.detail['http_status'] == 403
    assert 'SECRET' not in str(observed.value) + json.dumps(observed.value.detail)
    assert len(client.calls) == 1


@pytest.mark.parametrize('number', [errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EXDEV, errno.EOPNOTSUPP, errno.EINVAL])
def test_local_temp_errors_keep_errno_and_are_not_network_retried(fixed, monkeypatch, number):
    root, row, content = fixed
    def broken(parent):
        raise OSError(number, 'SECRET_FILESYSTEM_PATH')
    monkeypatch.setattr(fetch, '_new_temp', broken)
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code.startswith('LOCAL_IO_')
    assert observed.value.detail['phase'] == 'temp_create' and observed.value.detail['errno'] == number
    assert len(client.calls) == 1 and observed.value.retryable is False
    assert 'SECRET' not in json.dumps(observed.value.detail)


def test_fchmod_fd_is_used_without_path_chmod_follow_symlinks(fixed, monkeypatch):
    root, row, content = fixed
    monkeypatch.setattr(fetch.os, 'chmod', lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError()))
    assert fetch.fetch_object(FakeR2([content]), 'bucket', row, root)['status'] == 'R2_FULL_GET_SHA_VERIFIED'
    assert stat.S_IMODE((root / row['path']).stat().st_mode) == 0o600


@pytest.mark.parametrize('number', [errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EINVAL])
def test_file_fsync_errors_always_block(fixed, monkeypatch, number):
    root, row, content = fixed
    monkeypatch.setattr(fetch.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError(number, 'SECRET_FS')))
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.detail['phase'] == 'file_fsync' and observed.value.detail['errno'] == number
    assert len(client.calls) == 1 and not (root / row['path']).exists()


@pytest.mark.parametrize('number', [errno.EINVAL, errno.EOPNOTSUPP])
def test_only_input_cache_directory_fsync_unsupported_is_explicit(fixed, monkeypatch, number):
    root, row, content = fixed
    actual = fetch.os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(number, 'unsupported directory fsync')
        return actual(fd)
    monkeypatch.setattr(fetch.os, 'fsync', fsync)
    result = fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
    assert result['status'] == 'R2_FULL_GET_SHA_VERIFIED'
    assert result['durability']['directory_fsync'] == 'unsupported'
    assert result['durability']['file_fsync'] == 'completed'
    assert result['crash_durability_claimed'] is False


@pytest.mark.parametrize('number', [errno.EROFS, errno.EIO])
def test_directory_IO_error_after_commit_keeps_file_but_never_reports_ready(fixed, monkeypatch, number):
    root, row, content = fixed
    actual = fetch.os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(number, 'SECRET_DIRECTORY')
        return actual(fd)
    monkeypatch.setattr(fetch.os, 'fsync', fsync)
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.detail['phase'] == 'dir_fsync' and observed.value.detail['errno'] == number
    assert (root / row['path']).read_bytes() == content
    assert len(client.calls) == 1


def test_hardlink_unsupported_without_true_noreplace_primitive_blocks(fixed, monkeypatch):
    root, row, content = fixed
    monkeypatch.setattr(fetch.os, 'link', lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, 'unsupported')))
    monkeypatch.setattr(fetch.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace())
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
    assert observed.value.code == 'LOCAL_ATOMIC_NOREPLACE_UNAVAILABLE'
    assert not (root / row['path']).exists()


@pytest.mark.parametrize('identical', [True, False])
def test_commit_race_verifies_existing_file_without_overwrite(fixed, monkeypatch, identical):
    root, row, content = fixed
    preserved = content if identical else b'foreign-original'
    def occupied(source, target, **kwargs):
        Path(target).write_bytes(preserved)
        raise FileExistsError(errno.EEXIST, 'destination exists')
    monkeypatch.setattr(fetch.os, 'link', occupied)
    if identical:
        result = fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
        assert result['atomic_commit'] == 'existing_identical_SHA_verified_not_overwritten'
    else:
        with pytest.raises(fetch.FetchError) as observed:
            fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
        assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert (root / row['path']).read_bytes() == preserved


def test_final_local_SHA_reread_is_required(fixed, monkeypatch):
    root, row, content = fixed
    actual = fetch._sync_directory
    def changed(parent):
        result = actual(parent)
        (root / row['path']).write_bytes(b'changed-after-commit')
        return result
    monkeypatch.setattr(fetch, '_sync_directory', changed)
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
    assert observed.value.detail['phase'] == 'final_local_hash'
    assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'


def test_progress_reports_bytes_each_second_not_training_completion(fixed, monkeypatch):
    root, row, _ = fixed
    content = b'x' * (fetch.CHUNK * 5)
    row.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    clock = iter(number * .6 for number in range(100))
    monkeypatch.setattr(fetch, 'CLOCK', lambda: next(clock))
    events = []
    fetch.fetch_object(FakeR2([content]), 'bucket', row, root, events.append)
    streamed = [event for event in events if event['phase'] == 'body_read']
    assert len(streamed) >= 2
    assert all(event['whole_body_verified'] is False for event in streamed)
    assert streamed[-1]['bytes_read'] > streamed[0]['bytes_read']


def test_parallel_fetch_lock_prevents_duplicate_GET(fixed):
    root, row, content = fixed
    parent = root / 'archives'
    parent.mkdir()
    lock = parent / ('.paper3-get-' + hashlib.sha256(row['path'].encode()).hexdigest()[:20] + '.lock')
    client = FakeR2([content])
    with lock.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(fetch.FetchError) as observed:
            fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'FETCH_ALREADY_ACTIVE_NO_DUPLICATE_GET' and client.calls == []


def test_only_five_original_descriptors_are_allowed(fixed):
    root, row, content = fixed
    invalid = dict(row, path='data/sealed/query.bin', key='paper3/sealed/query.bin')
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', invalid, root)
    assert observed.value.code == 'FIXED_RELEASE_OBJECT_REQUIRED' and client.calls == []
    assert not (root / 'data').exists()


def test_symlink_target_does_not_touch_its_destination(fixed, tmp_path):
    root, row, content = fixed
    target = root / row['path']
    target.parent.mkdir()
    outside = root / 'outside'
    outside.write_bytes(b'original')
    target.symlink_to(outside)
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError):
        fetch.fetch_object(client, 'bucket', row, root)
    assert outside.read_bytes() == b'original' and client.calls == []


def cache_temp(root, content=b'local immutable cache'):
    row = {'path': 'cache/source_v2/BNCI2014_001/1/X.npy',
           'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    target = root / row['path']
    target.parent.mkdir(parents=True)
    temporary = target.parent / '.owned.partial'
    temporary.write_bytes(content)
    return row, target, temporary, content


def test_local_materialization_helper_fchmod_fsync_SHA_no_network(fixed, monkeypatch):
    root, _, _ = fixed
    row, target, temporary, content = cache_temp(root)
    monkeypatch.setattr(fetch.os, 'chmod', lambda *a, **k: (_ for _ in ()).throw(NotImplementedError()))
    result = fetch.publish_verified_cache(root, row, temporary)
    assert target.read_bytes() == content and not temporary.exists()
    assert result['cache_payload_verified'] is True and result['network_GET_proof'] is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


@pytest.mark.parametrize('path', ['runs/checkpoint.pt', 'p3work/source.py', 'cache/sealed/X.npy',
                                  'data/raw.edf', 'cache/source_v2/legacy.npz'])
def test_local_helper_cannot_publish_training_outputs_or_unapproved_data(fixed, path):
    root, _, _ = fixed
    row, target, temporary, content = cache_temp(root)
    row['path'] = path
    with pytest.raises(fetch.FetchError) as observed:
        fetch.publish_verified_cache(root, row, temporary)
    assert observed.value.code == 'IMMUTABLE_INPUT_CACHE_SCOPE_REQUIRED'
    assert not target.exists() and temporary.read_bytes() == content


def test_local_helper_requires_same_directory_owned_temporary(fixed):
    root, _, _ = fixed
    row, target, temporary, content = cache_temp(root)
    outside = root / 'other.partial'
    outside.write_bytes(content)
    with pytest.raises(fetch.FetchError) as observed:
        fetch.publish_verified_cache(root, row, outside)
    assert observed.value.code == 'OWNED_SAME_DIRECTORY_TEMPFILE_REQUIRED'
    assert not target.exists()


def test_transient_failure_history_is_retained_after_verified_retry(fixed):
    root, row, content = fixed
    result = fetch.fetch_object(FakeR2([ReadTimeoutError('SECRET_URL'), content]), 'bucket', row, root)
    assert result['attempts'] == 2
    assert len(result['attempt_failures']) == 1
    assert result['attempt_failures'][0]['code'] == 'RESPONSE_TIMEOUT'
    assert result['attempt_failures'][0]['detail']['phase'] == 'get_headers'
    assert 'SECRET' not in json.dumps(result['attempt_failures'])


def test_final_failure_lists_all_attempts_without_a_fourth_retry(fixed):
    root, row, _ = fixed
    client = FakeR2([ReadTimeoutError('SECRET')])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert len(observed.value.detail['attempt_failures']) == 3
    assert observed.value.detail['retries_remaining'] == 0 and len(client.calls) == 3
    assert 'SECRET' not in json.dumps(observed.value.detail)


def test_local_write_ENOSPC_preserves_phase_and_is_not_retried(fixed, monkeypatch):
    root, row, content = fixed
    actual = fetch._new_temp
    class FullFile:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            raise OSError(errno.ENOSPC, 'SECRET_MOUNT_PATH')
        def close(self):
            self.stream.close()
    def full(parent):
        stream, path = actual(parent)
        return FullFile(stream), path
    monkeypatch.setattr(fetch, '_new_temp', full)
    client = FakeR2([content])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'LOCAL_IO_ENOSPC'
    assert observed.value.detail['phase'] == 'local_write' and len(client.calls) == 1
    assert not (root / row['path']).exists() and not list(root.rglob('*.partial'))


def test_progress_callback_error_does_not_duplicate_GET(fixed):
    root, row, content = fixed
    def broken(event):
        raise RuntimeError('display-only failure')
    client = FakeR2([content])
    result = fetch.fetch_object(client, 'bucket', row, root, broken)
    assert result['status'] == 'R2_FULL_GET_SHA_VERIFIED' and len(client.calls) == 1


def test_real_noreplace_fallback_with_mock_libc_preserves_nooverwrite(fixed, monkeypatch):
    root, row, content = fixed
    def no_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, 'hardlinks unavailable')
    calls = []
    class RenameFunction:
        def __call__(self, source_fd, source, target_fd, target, flags):
            calls.append((source_fd, target_fd, flags))
            # Tiny stand-in for atomic RENAME_NOREPLACE; create destination via
            # exclusive fixture write, never use this implementation in prod.
            with open(os.fsdecode(target), 'xb') as stream:
                stream.write(Path(os.fsdecode(source)).read_bytes())
            Path(os.fsdecode(source)).unlink()
            return 0
    monkeypatch.setattr(fetch.os, 'link', no_link)
    monkeypatch.setattr(fetch.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(renameat2=RenameFunction()))
    result = fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
    assert result['atomic_commit'] == 'renameat2_noreplace'
    assert calls == [(-100, -100, 1)] and (root / row['path']).read_bytes() == content


@pytest.mark.parametrize('identical', [True, False])
def test_rename_noreplace_EEXIST_never_replaces_existing_target(fixed, monkeypatch, identical):
    root, row, content = fixed
    preserved = content if identical else b'original-target'
    class RenameFunction:
        def __call__(self, source_fd, source, target_fd, target, flags):
            Path(os.fsdecode(target)).write_bytes(preserved)
            fetch.ctypes.set_errno(errno.EEXIST)
            return -1
    monkeypatch.setattr(fetch.os, 'link', lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EOPNOTSUPP, 'no hardlink')))
    monkeypatch.setattr(fetch.ctypes, 'CDLL', lambda *a, **k: SimpleNamespace(renameat2=RenameFunction()))
    if identical:
        assert fetch.fetch_object(FakeR2([content]), 'bucket', row, root)['atomic_commit'].startswith('existing_identical')
    else:
        with pytest.raises(fetch.FetchError) as observed:
            fetch.fetch_object(FakeR2([content]), 'bucket', row, root)
        assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert (root / row['path']).read_bytes() == preserved


def test_nonbinary_stream_cannot_be_called_a_complete_GET(fixed):
    root, row, content = fixed
    class TextBody:
        def read(self, size):
            return 'text-is-not-binary'
        def close(self):
            pass
    client = FakeR2([headers(TextBody(), content)])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'R2_GET_NON_BINARY_BODY'
    assert len(client.calls) == 1 and not (root / row['path']).exists()


def test_incorrect_HTTP_declared_length_never_reads_body(fixed):
    root, row, content = fixed
    streams = [TrackedBody(content) for _ in range(3)]
    client = FakeR2([dict(headers(stream, content), ContentLength=len(content) + 1) for stream in streams])
    with pytest.raises(fetch.FetchError) as observed:
        fetch.fetch_object(client, 'bucket', row, root)
    assert observed.value.code == 'R2_GET_DECLARED_LENGTH_MISMATCH' and len(client.calls) == 3
    assert all(stream.eof is False and stream.closed for stream in streams)


def test_cache_helper_mismatched_temp_SHA_is_never_committed(fixed):
    root, _, _ = fixed
    row, target, temporary, content = cache_temp(root)
    temporary.write_bytes(b'bad-unverified-cache')
    with pytest.raises(fetch.FetchError) as observed:
        fetch.publish_verified_cache(root, row, temporary)
    assert observed.value.detail['phase'] == 'precommit_local_hash'
    assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'
    assert not target.exists()


def test_cache_helper_final_SHA_is_required(fixed, monkeypatch):
    root, _, _ = fixed
    row, target, temporary, content = cache_temp(root)
    actual = fetch._sync_directory
    def mutate(parent):
        result = actual(parent)
        target.write_bytes(b'changed-after-publish')
        return result
    monkeypatch.setattr(fetch, '_sync_directory', mutate)
    with pytest.raises(fetch.FetchError) as observed:
        fetch.publish_verified_cache(root, row, temporary)
    assert observed.value.detail['phase'] == 'final_local_hash'
    assert observed.value.code == 'LOCAL_SHA_OR_BYTES_MISMATCH'


def test_standard_POSIX_link_uses_no_optional_flags(fixed, monkeypatch):
    root, row, content = fixed
    actual = fetch.os.link
    calls = []
    def portable(source, target, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs == {} and not args
        assert stat.S_ISREG(Path(source).lstat().st_mode) and not Path(source).is_symlink()
        return actual(source, target)
    monkeypatch.setattr(fetch.os, 'link', portable)
    assert fetch.fetch_object(FakeR2([content]), 'bucket', row, root)['status'] == 'R2_FULL_GET_SHA_VERIFIED'
    assert calls == [{}]


def test_commit_rejects_symlink_temp_without_touching_destination(fixed):
    root, row, content = fixed
    target = root / row['path']
    target.parent.mkdir()
    outside = target.parent / 'outside'
    outside.write_bytes(content)
    temporary = target.parent / '.owned.partial'
    temporary.symlink_to(outside)
    with pytest.raises((fetch.FetchError, fetch.original.TransportError)):
        fetch._commit(temporary, target, row)
    assert not target.exists() and outside.read_bytes() == content
