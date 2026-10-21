"""Interrupted code transport fixtures; no network, Slurm, data download or fit."""
from __future__ import annotations

import hashlib
import ssl
import sys
from types import ModuleType
import urllib.error
from pathlib import Path

import pytest

from hpc import code_download as transfer

URL = 'https://raw.githubusercontent.com/example/frozen/bundle'


class Response:
    def __init__(self, pieces, *, status=200, headers=None):
        self.status = self.code = status
        self.headers = headers or {}
        self.pieces = list(pieces)
        self.closed = False
        self.read1_sizes = []

    def getcode(self):
        return self.status

    def read1(self, size=-1):
        self.read1_sizes.append(size)
        if not self.pieces:
            return b''
        value = self.pieces.pop(0)
        if isinstance(value, BaseException):
            raise value
        if size >= 0 and len(value) > size:
            self.pieces.insert(0, value[size:])
            value = value[:size]
        return value

    def read(self, size=-1):
        raise AssertionError('whole-body read must not hide and discard download progress')

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, request, *, context, timeout):
        self.calls.append((request, context, timeout))
        if not self.responses:
            raise AssertionError('unbounded or unexpected retry')
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def fixed(tmp_path, monkeypatch):
    # Retry waiting is observational software timing, never a scientific budget.
    monkeypatch.setattr(transfer.time, 'sleep', lambda seconds: None)
    data = b'0123456789-verified-fixed-code\x00'
    target = tmp_path / 'code.bundle'
    context = ssl.create_default_context()
    return target, data, context


def get(fixed, responses, *, attempts=4, events=None):
    target, data, context = fixed
    opener = Opener(responses)
    events = [] if events is None else events
    value = transfer.download(URL, target, len(data), hashlib.sha256(data).hexdigest(),
                              context=context, opener=opener, attempts=attempts,
                              timeout=1, progress=events.append)
    return value, opener, events


def range_start(call):
    return call[0].get_header('Range')


def whole(data):
    return Response([data], headers={'Content-Length': str(len(data))})


def tail(data, start, **headers):
    return Response([data[start:]], status=206, headers={
        'Content-Length': str(len(data) - start),
        'Content-Range': f'bytes {start}-{len(data)-1}/{len(data)}', **headers})


def test_timeout_preserves_received_prefix_and_resumes_exact_range(fixed):
    target, data, context = fixed
    first = Response([data[:7], TimeoutError('DO_NOT_PRINT_SECRET')],
                     headers={'Content-Length': str(len(data))})
    second = tail(data, 7)
    _, client, events = get(fixed, [first, second])
    assert target.read_bytes() == data
    assert len(client.calls) == 2
    assert range_start(client.calls[0]) in (None, 'bytes=0-')
    assert range_start(client.calls[1]) == 'bytes=7-'
    assert all(call[1] is context for call in client.calls)
    assert first.closed and second.closed
    assert 'DO_NOT_PRINT_SECRET' not in repr(events)
    assert all(0 < size <= 64 * 1024 for size in first.read1_sizes + second.read1_sizes)


def test_server_ignores_range_restarts_from_full_body_without_duplicating_prefix(fixed):
    target, data, _ = fixed
    first = Response([data[:5], TimeoutError()], headers={'Content-Length': str(len(data))})
    second = whole(data)
    _, client, _ = get(fixed, [first, second])
    assert range_start(client.calls[1]) == 'bytes=5-'
    assert target.read_bytes() == data
    assert len(client.calls) == 2


def test_clean_short_body_can_resume_and_is_never_committed_as_complete(fixed):
    target, data, _ = fixed
    first = Response([data[:3]], headers={'Content-Length': str(len(data))})
    _, client, _ = get(fixed, [first, tail(data, 3)])
    assert range_start(client.calls[1]) == 'bytes=3-'
    assert target.read_bytes() == data


def test_interruption_survives_separate_download_invocations(fixed):
    target, data, context = fixed
    client = Opener([Response([data[:9], TimeoutError()],
                             headers={'Content-Length': str(len(data))})])
    with pytest.raises(transfer.DownloadError, match='code_download_retries_exhausted_partial_retained'):
        transfer.download(URL, target, len(data), hashlib.sha256(data).hexdigest(),
                          context=context, opener=client, attempts=1, timeout=1,
                          progress=lambda value: None)
    assert not target.exists()
    # The public API, rather than private partial-file naming, establishes resume.
    _, resumed, _ = get(fixed, [tail(data, 9)])
    assert range_start(resumed.calls[0]) == 'bytes=9-'
    assert target.read_bytes() == data


@pytest.mark.parametrize('content_range', [
    'bytes 0-28/29', 'bytes 8-28/29', 'bytes 7-999/1000',
    'bytes */29', 'DO_NOT_PRINT_SECRET',
])
def test_wrong_content_range_never_commits_or_claims_success(fixed, content_range):
    target, data, context = fixed
    first = Response([data[:7], TimeoutError()], headers={'Content-Length': str(len(data))})
    second = tail(data, 7, **{'Content-Range': content_range})
    events = []
    with pytest.raises(transfer.DownloadError, match='code_download_(invalid_Content_Range|Content_Range_mismatch)'):
        get(fixed, [first, second], events=events)
    assert not target.exists()
    assert 'DO_NOT_PRINT_SECRET' not in repr(events)


def test_wrong_full_sha_never_commits_even_when_byte_count_matches(fixed):
    target, data, _ = fixed
    bad = bytes([data[0] ^ 1]) + data[1:]
    with pytest.raises(transfer.DownloadError, match='code_download_SHA256_mismatch'):
        get(fixed, [whole(bad)])
    assert not target.exists()


def test_transient_open_failures_are_bounded_and_do_not_publish(fixed):
    target, data, _ = fixed
    events = []
    with pytest.raises(transfer.DownloadError, match='code_download_retries_exhausted_partial_retained'):
        get(fixed, [urllib.error.URLError(TimeoutError('DO_NOT_PRINT_SECRET'))] * 3,
            attempts=3, events=events)
    assert not target.exists()
    assert 'DO_NOT_PRINT_SECRET' not in repr(events)


def test_verified_cached_bundle_needs_no_request(fixed):
    target, data, _ = fixed
    target.write_bytes(data)
    _, client, _ = get(fixed, [])
    assert not client.calls and target.read_bytes() == data


def test_corrupted_cached_bundle_is_preserved_and_refused(fixed):
    target, data, _ = fixed
    corrupt = bytes([data[0] ^ 1]) + data[1:]
    target.write_bytes(corrupt)
    with pytest.raises(transfer.DownloadError, match='code_download_existing_SHA256_mismatch'):
        get(fixed, [])
    assert target.read_bytes() == corrupt


def test_supplied_ca_context_keeps_certificate_and_hostname_verification(fixed):
    _, data, context = fixed
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    _, client, _ = get(fixed, [whole(data)])
    assert client.calls[0][1] is context
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_download_larger_than_frozen_size_is_never_published(fixed):
    target, data, _ = fixed
    with pytest.raises(transfer.DownloadError, match='code_download_body_too_large'):
        get(fixed, [Response([data + b'extra'])])
    assert not target.exists()


def test_symlink_output_cannot_replace_unrelated_file(fixed, tmp_path):
    target, data, context = fixed
    unrelated = tmp_path / 'unrelated'
    unrelated.write_bytes(b'preserve')
    target.symlink_to(unrelated)
    with pytest.raises(transfer.DownloadError, match='code_download_symlink_forbidden'):
        get(fixed, [whole(data)])
    assert target.is_symlink() and unrelated.read_bytes() == b'preserve'


def test_CA_context_combines_default_school_trust_and_existing_R2_sdk_CA(monkeypatch):
    class Context:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED
        def __init__(self):
            self.loaded = []
        def load_verify_locations(self, *, cafile):
            self.loaded.append(cafile)
    context = Context()
    monkeypatch.setattr(transfer.ssl, 'create_default_context', lambda: context)
    package = ModuleType('botocore')
    module = ModuleType('botocore.httpsession')
    called = []
    def cert_path(value):
        called.append(value)
        return '/existing/r2-sdk/CA-bundle.pem'
    module.get_cert_path = cert_path
    monkeypatch.setitem(sys.modules, 'botocore', package)
    monkeypatch.setitem(sys.modules, 'botocore.httpsession', module)
    assert transfer.tls_context() is context
    assert called == [True]
    assert context.loaded == ['/existing/r2-sdk/CA-bundle.pem']
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED


def test_unverified_TLS_context_is_rejected_before_network_or_file_write(fixed):
    target, data, _ = fixed
    context = ssl._create_unverified_context()
    client = Opener([])
    with pytest.raises(transfer.DownloadError, match='code_download_verified_TLS_required'):
        transfer.download(URL, target, len(data), hashlib.sha256(data).hexdigest(),
                          context=context, opener=client, progress=lambda value: None)
    assert not client.calls and not target.exists()


def test_certificate_failure_does_not_retry_or_expose_exception_text(fixed):
    target, data, context = fixed
    client = Opener([urllib.error.URLError(ssl.SSLCertVerificationError('DO_NOT_PRINT_SECRET'))])
    events = []
    with pytest.raises(transfer.DownloadError, match='code_download_CA_certificate_required') as caught:
        transfer.download(URL, target, len(data), hashlib.sha256(data).hexdigest(),
                          context=context, opener=client, progress=events.append)
    assert len(client.calls) == 1 and not target.exists()
    assert 'DO_NOT_PRINT_SECRET' not in str(caught.value) + repr(events)
