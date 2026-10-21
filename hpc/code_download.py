#!/usr/bin/env python3
"""Resumable, SHA-pinned deployment download; never installs or submits jobs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 64 * 1024


class DownloadError(ValueError):
    """Only fixed machine-readable failures are exposed to callers."""


def _need(value, code):
    if not value:
        raise DownloadError(code)


def _safe(path):
    path = Path(os.path.abspath(path))
    _need(not any(p.is_symlink() for p in (path, *path.parents)),
          'code_download_symlink_forbidden')
    if path.exists():
        _need(stat.S_ISREG(path.stat().st_mode), 'code_download_regular_file_required')
    return path


def _hash(path):
    digest = hashlib.sha256()
    with _safe(path).open('rb') as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def tls_context():
    """Retain school/default trust and add the already installed R2 SDK CA."""
    context = ssl.create_default_context()
    try:
        from botocore.httpsession import get_cert_path
        context.load_verify_locations(cafile=get_cert_path(True))
    except ImportError:
        try:
            import certifi
            context.load_verify_locations(cafile=certifi.where())
        except ImportError:
            pass
    _need(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
          'code_download_verified_TLS_required')
    return context


def _retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        return _retryable(exc.reason)
    if isinstance(exc, ssl.SSLError):
        return False
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionError,
                            http.client.IncompleteRead, http.client.RemoteDisconnected))


def download(url, destination, expected_bytes, expected_sha256, *, context,
             opener=None, attempts=4, timeout=30, progress=print):
    """Keep a durable prefix on transient failures and verify the complete body.

    The caller pins both size and SHA; Range replies must match the pinned total.
    A 200 response to a resumed request explicitly restarts this owned partial.
    Validation failures never publish an output or pretend a download completed.
    """
    parsed = urllib.parse.urlsplit(url)
    _need(parsed.scheme == 'https' and parsed.hostname and not parsed.username
          and not parsed.password and not parsed.fragment, 'code_download_HTTPS_required')
    _need(type(expected_bytes) is int and expected_bytes > 0
          and isinstance(expected_sha256, str)
          and re.fullmatch('[a-f0-9]{64}', expected_sha256), 'code_download_identity_required')
    _need(type(attempts) is int and 1 <= attempts <= 12 and 0 < timeout <= 120,
          'code_download_bounded_retry_required')
    _need(context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED,
          'code_download_verified_TLS_required')
    destination = _safe(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _safe(destination)
    partial = _safe(str(destination) + '.part')
    lock = _safe(str(destination) + '.lock')
    opener = opener or urllib.request.urlopen

    def emit(message):
        if progress is not None:
            progress(message)

    def publish():
        _need(partial.stat().st_size == expected_bytes, 'code_download_size_mismatch')
        _need(_hash(partial) == expected_sha256, 'code_download_SHA256_mismatch')
        _need(not destination.exists(), 'code_download_output_appeared')
        os.replace(partial, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        emit('VERIFIED: complete code package GET/SHA256 passed')
        return destination

    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError('code_download_already_running') from None
        if destination.exists():
            _need(destination.stat().st_size == expected_bytes
                  and _hash(destination) == expected_sha256, 'code_download_existing_SHA256_mismatch')
            emit('VERIFIED: existing code package SHA256 passed')
            return destination
        if partial.exists():
            _need(partial.stat().st_size <= expected_bytes, 'code_download_partial_too_large')
            if partial.stat().st_size == expected_bytes:
                return publish()
        for attempt in range(1, attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Accept-Encoding': 'identity', 'User-Agent': 'Paper3-HPC-code-update/1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            emit(f'DOWNLOADING: attempt {attempt}/{attempts}; {offset}/{expected_bytes} bytes')
            try:
                with opener(request, timeout=timeout, context=context) as response:
                    final_url = getattr(response, 'url', url)
                    _need(urllib.parse.urlsplit(final_url).scheme == 'https',
                          'code_download_HTTPS_redirect_required')
                    status = response.status
                    _need(status in {200, 206}, 'code_download_unexpected_HTTP_status')
                    _need(response.headers.get('Content-Encoding', 'identity').lower() == 'identity',
                          'code_download_encoded_body_forbidden')
                    if status == 206:
                        value = response.headers.get('Content-Range', '')
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', value)
                        _need(match is not None, 'code_download_invalid_Content_Range')
                        first, last, total = map(int, match.groups())
                        _need(first == offset and last == expected_bytes - 1
                              and total == expected_bytes, 'code_download_Content_Range_mismatch')
                    else:
                        if offset:
                            emit('RESTARTING: server ignored Range; restarting this partial code package')
                        offset = 0
                    remaining = expected_bytes - offset
                    length = response.headers.get('Content-Length')
                    _need(length is None or (length.isdigit() and int(length) == remaining),
                          'code_download_Content_Length_mismatch')
                    mode = 'ab' if offset else 'wb'
                    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
                                         | (os.O_APPEND if offset else os.O_TRUNC), 0o600)
                    current = offset
                    last_progress = time.monotonic()
                    with os.fdopen(descriptor, mode) as output:
                        try:
                            reader = getattr(response, 'read1', response.read)
                            while block := reader(CHUNK):
                                _need(isinstance(block, bytes), 'code_download_invalid_body')
                                _need(current + len(block) <= expected_bytes,
                                      'code_download_body_too_large')
                                output.write(block)
                                output.flush()
                                current += len(block)
                                if time.monotonic() - last_progress >= 2:
                                    emit(f'DOWNLOADING: {current}/{expected_bytes} bytes ({100*current/expected_bytes:.1f}%)')
                                    last_progress = time.monotonic()
                        finally:
                            output.flush()
                            os.fsync(output.fileno())
                    if current != expected_bytes:
                        raise http.client.IncompleteRead(b'', expected_bytes - current)
                return publish()
            except DownloadError:
                raise
            except Exception as exc:
                if not _retryable(exc):
                    code = ('code_download_CA_certificate_required'
                            if isinstance(exc, ssl.SSLCertVerificationError)
                            or isinstance(getattr(exc, 'reason', None), ssl.SSLCertVerificationError)
                            else 'code_download_connection_failed')
                    raise DownloadError(code) from None
                if attempt == attempts:
                    raise DownloadError('code_download_retries_exhausted_partial_retained') from None
                emit('RETRYING: temporary connection failure; saved prefix retained')
                time.sleep(1)
    raise DownloadError('code_download_unreachable')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bytes', type=int, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    try:
        download(args.url, args.output, args.bytes, args.sha256, context=tls_context(),
                 progress=lambda message: print(message, flush=True))
    except DownloadError as exc:
        print('BLOCKED: ' + str(exc), flush=True)
        return 2
    except OSError:
        print('BLOCKED: code_download_local_file_IO_failed', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
