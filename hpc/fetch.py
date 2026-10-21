#!/usr/bin/env python3
"""Verified GET of the five immutable HPC release objects only.

Independent of the retained RunPod transport. Every attempt is a whole GET;
only a complete body/length/SHA proof and final local reread can publish success.
No raw SDK message, endpoint, bucket or object key is included in diagnostics.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import time

import paper3_transport as original

CHUNK = 256 * 1024
ATTEMPTS = 3
CLOCK = time.monotonic
SLEEP = time.sleep
AWS_CODES = frozenset({'AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch', 'NoSuchBucket',
    'NoSuchKey', 'NotFound', '404', '403', 'RequestTimeout', 'SlowDown', 'ExpiredToken', 'InvalidToken',
    'AuthorizationHeaderMalformed', 'InvalidRequest', 'InternalError', 'ServiceUnavailable', 'PermanentRedirect'})
NETWORK_CODES = {
    'EndpointConnectionError': 'NETWORK_DNS_OR_CONNECTION_FAILED',
    'ConnectTimeoutError': 'CONNECTION_TIMEOUT', 'ReadTimeoutError': 'RESPONSE_TIMEOUT',
    'SSLError': 'TLS_CERTIFICATE_FAILED', 'ProxyConnectionError': 'PROXY_CONNECTION_FAILED',
    'ConnectionClosedError': 'CONNECTION_CLOSED', 'HTTPClientError': 'HTTP_TRANSPORT_FAILED',
}
TRANSIENT_HTTP = frozenset({408, 429, 500, 502, 503, 504})
UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP})


class FetchError(ValueError):
    """Only a fixed code and bounded, nonsensitive structural detail escape."""
    def __init__(self, code, *, phase, retryable=False, **detail):
        self.code = code
        self.retryable = retryable
        self._detail = {'phase': phase, 'retryable': retryable, **detail}
        super().__init__(code)

    @property
    def detail(self):
        return dict(self._detail)


def _http(value):
    return value if type(value) is int and 100 <= value <= 599 else None


def _error(exc, phase):
    if isinstance(exc, FetchError):
        return exc
    detail = {}
    number = getattr(exc, 'errno', None)
    if type(number) is int:
        detail.update(errno=number, errno_name=errno.errorcode.get(number, 'UNKNOWN_ERRNO'))
    if phase in {'get_headers', 'body_read', 'body_close'}:
        response = getattr(exc, 'response', None)
        if isinstance(response, dict):
            metadata, service = response.get('ResponseMetadata'), response.get('Error')
            status = _http(metadata.get('HTTPStatusCode')) if isinstance(metadata, dict) else None
            code = service.get('Code') if isinstance(service, dict) else None
            detail.update(http_status=status if type(status) is int and 100 <= status <= 599 else None,
                          service_code=code if isinstance(code, str) and code in AWS_CODES else 'OTHER_S3_ERROR')
            return FetchError('R2_HTTP_REQUEST_FAILED', phase=phase,
                              retryable=status in TRANSIENT_HTTP, **detail)
        category = NETWORK_CODES.get(type(exc).__name__)
        return FetchError(category or ('NETWORK_BODY_READ_FAILED' if phase != 'get_headers' else 'NETWORK_GET_REQUEST_FAILED'),
                          phase=phase, retryable=category != 'TLS_CERTIFICATE_FAILED', **detail)
    if isinstance(exc, original.TransportError):
        return FetchError(exc.code if exc.code.startswith('LOCAL_') else 'LOCAL_' + exc.code, phase=phase, **detail)
    return FetchError('LOCAL_IO_' + detail['errno_name'] if 'errno_name' in detail else 'LOCAL_OPERATION_FAILED',
                      phase=phase, **detail)


def _validate_fixed_object(row):
    # Lazy import avoids a cycle when stage_data calls this transport. The
    # original pinned five descriptors grant no cache/raw/external-query scope.
    from hpc.stage_data import load_release
    try:
        original._descriptor(row, exact=False)
        identity = {name: row[name] for name in ('path', 'key', 'sha256', 'bytes')}
        if identity not in load_release()['objects'].values():
            raise ValueError('fixed_identity_required')
    except Exception:
        raise FetchError('FIXED_RELEASE_OBJECT_REQUIRED', phase='validate_descriptor') from None
    return identity


def _progress(callback, event):
    if callback is not None:
        try:
            callback(dict(event))
        except Exception:
            # A display failure cannot turn a partially read body into success
            # or duplicate an otherwise valid network transfer.
            pass


def _new_temp(parent):
    fd, filename = tempfile.mkstemp(prefix='.paper3-get-', suffix='.partial', dir=parent)
    try:
        return os.fdopen(fd, 'wb'), Path(filename)
    except Exception:
        os.close(fd)
        Path(filename).unlink(missing_ok=True)
        raise


def _rename_noreplace(source, target, original_errno):
    """Optional libc primitive; never guess architecture-specific syscall IDs."""
    function = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if function is None:
        raise FetchError('LOCAL_ATOMIC_NOREPLACE_UNAVAILABLE', phase='commit',
                         errno=original_errno, errno_name=errno.errorcode.get(original_errno, 'UNKNOWN_ERRNO'),
                         atomic_fallback='renameat2_symbol_unavailable')
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, 'atomic_destination_exists')
        if number in UNSUPPORTED or number == errno.EXDEV:
            raise FetchError('LOCAL_ATOMIC_NOREPLACE_UNAVAILABLE', phase='commit',
                             errno=number, errno_name=errno.errorcode.get(number, 'UNKNOWN_ERRNO'),
                             atomic_fallback='renameat2_not_supported')
        raise OSError(number, 'atomic_commit_failed')
    return 'renameat2_noreplace'


def _commit(temporary, target, row):
    # POSIX link is atomic and fails if any destination already exists. fchmod
    # was performed on the open owned fd; no chmod(follow_symlinks=False) call.
    temporary, target = Path(temporary), Path(target)
    if temporary.parent != target.parent or temporary == target:
        raise FetchError('OWNED_SAME_DIRECTORY_TEMPFILE_REQUIRED', phase='commit')
    original._file(temporary.parent, temporary.name)  # lstat: no symlink/nonregular source
    if not stat.S_ISREG(temporary.lstat().st_mode):
        raise FetchError('LOCAL_TEMPFILE_NOT_REGULAR', phase='commit')
    try:
        try:
            os.link(temporary, target)
            operation = 'link_no_overwrite'
        except (OSError, NotImplementedError) as exc:
            if isinstance(exc, FileExistsError):
                raise
            number = getattr(exc, 'errno', errno.ENOSYS)
            if number not in UNSUPPORTED and number != errno.EXDEV:
                raise
            operation = _rename_noreplace(temporary, target, number)
    except FileExistsError:
        original._matches(target, row)
        operation = 'existing_identical_SHA_verified_not_overwritten'
    temporary.unlink(missing_ok=True)
    return operation


def _sync_directory(parent):
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.fsync(fd)
            return {'directory_fsync': 'completed'}
        except OSError as exc:
            # Only immutable, re-fetchable release caches use this narrow
            # allowance; checkpoint/ledger/handoff durability is untouched.
            if exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise
            return {'directory_fsync': 'unsupported', 'errno': exc.errno,
                    'errno_name': errno.errorcode.get(exc.errno, 'UNKNOWN_ERRNO')}
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def publish_verified_cache(root, row, temporary, mode=0o644):
    """Publish only a frozen immutable input cache/metadata tempfile.

    This is a local SHA/storage proof, never a GET proof. It is deliberately
    unavailable to checkpoints, ledgers, result handoffs or arbitrary code.
    """
    phase, fd = 'validate_cache_descriptor', None
    try:
        from hpc.stage_data import _allowed
        original._descriptor(row, exact=False)
        if not _allowed(row['path']) or mode not in {0o600, 0o644}:
            raise FetchError('IMMUTABLE_INPUT_CACHE_SCOPE_REQUIRED', phase=phase)
        root = original._root(root)
        target = original._file(root, row['path'], create_parents=True)
        temporary = Path(os.path.abspath(temporary))
        if temporary.parent != target.parent or temporary == target:
            raise FetchError('OWNED_SAME_DIRECTORY_TEMPFILE_REQUIRED', phase=phase)
        original._file(root, temporary.relative_to(root).as_posix())
        phase = 'precommit_local_hash'
        original._matches(temporary, row)
        phase = 'fchmod'
        fd = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FetchError('LOCAL_TEMPFILE_NOT_REGULAR', phase=phase)
        os.fchmod(fd, mode)
        phase = 'file_fsync'
        os.fsync(fd)
        os.close(fd)
        fd = None
        phase = 'precommit_local_hash'
        original._matches(temporary, row)
        phase = 'commit'
        operation = _commit(temporary, target, row)
        phase = 'dir_fsync'
        durability = _sync_directory(target.parent)
        phase = 'final_local_hash'
        original._matches(target, row)
        return {'atomic_commit': operation, 'local_sha256': row['sha256'], 'bytes_verified': row['bytes'],
                'cache_payload_verified': True, 'network_GET_proof': False,
                'durability': dict(file_fsync='completed', **durability), 'crash_durability_claimed': False}
    except Exception as exc:
        raise _error(exc, phase) from None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _attempt(client, bucket, row, target, *, attempt, progress):
    body = output = temporary = None
    phase, read_bytes, body_verified = 'get_headers', 0, False
    last_progress = CLOCK()
    def emit(current_phase, force=False, **extra):
        nonlocal last_progress
        now = CLOCK()
        if force or now - last_progress >= 1:
            _progress(progress, {'phase': current_phase, 'attempt': attempt, 'attempt_limit': ATTEMPTS,
                'bytes_read': read_bytes, 'expected_bytes': row['bytes'],
                'whole_body_verified': body_verified, **extra})
            last_progress = now
    try:
        emit(phase, True)
        response = client.get_object(Bucket=bucket, Key=row['key'])
        metadata = response.get('ResponseMetadata') if isinstance(response, dict) else None
        http = _http(metadata.get('HTTPStatusCode')) if isinstance(metadata, dict) else None
        body = response.get('Body') if isinstance(response, dict) else None
        if http != 200:
            raise FetchError('R2_GET_HTTP_NOT_200', phase=phase,
                             http_status=http if type(http) is int and 100 <= http <= 599 else None,
                             retryable=http in TRANSIENT_HTTP)
        declared = response.get('ContentLength')
        if type(declared) is not int or declared != row['bytes']:
            raise FetchError('R2_GET_DECLARED_LENGTH_MISMATCH', phase=phase, retryable=True,
                             http_status=200, declared_bytes=declared if type(declared) is int else None)
        if body is None or not callable(getattr(body, 'read', None)):
            raise FetchError('R2_GET_BODY_MISSING', phase=phase)
        phase = 'temp_create'
        output, temporary = _new_temp(target.parent)
        digest = hashlib.sha256()
        while True:
            phase = 'body_read'
            block = body.read(CHUNK)
            if not isinstance(block, bytes):
                raise FetchError('R2_GET_NON_BINARY_BODY', phase=phase)
            if not block:
                break
            read_bytes += len(block)
            if read_bytes > row['bytes']:
                raise FetchError('R2_GET_BODY_LENGTH_MISMATCH', phase=phase)
            digest.update(block)
            phase = 'local_write'
            output.write(block)
            emit('body_read')
        if (digest.hexdigest(), read_bytes) != (row['sha256'], row['bytes']):
            raise FetchError('R2_GET_BODY_SHA_OR_LENGTH_MISMATCH', phase='body_verify')
        body_verified = True
        phase = 'body_close'
        body.close()
        body = None
        phase = 'fchmod'
        os.fchmod(output.fileno(), 0o600)
        phase = 'file_flush'
        output.flush()
        phase = 'file_fsync'
        os.fsync(output.fileno())
        phase = 'file_close'
        output.close()
        output = None
        phase = 'precommit_local_hash'
        original._matches(temporary, row)
        phase = 'commit'
        emit(phase, True)
        atomic_commit = _commit(temporary, target, row)
        temporary = None
        phase = 'dir_fsync'
        durability = _sync_directory(target.parent)
        phase = 'final_local_hash'
        original._matches(target, row)
        emit('verified', True)
        return dict(row, status='R2_FULL_GET_SHA_VERIFIED', get_attempted=True, http_status=200,
                    complete_body_read=True, bytes_read=read_bytes, declared_content_length=declared,
                    get_sha256=digest.hexdigest(), local_sha256=row['sha256'], attempts=attempt,
                    atomic_commit=atomic_commit, durability=dict(file_fsync='completed', **durability),
                    crash_durability_claimed=False)
    except Exception as exc:
        error = _error(exc, phase)
        error._detail.update(attempt=attempt, attempt_limit=ATTEMPTS, bytes_read=read_bytes,
                             expected_bytes=row['bytes'], whole_body_verified=body_verified,
                             retries_remaining=ATTEMPTS-attempt if error.retryable else 0)
        emit('failed_attempt', True, error_code=error.code, detail=error.detail)
        raise error from None
    finally:
        if body is not None:
            try:
                body.close()
            except Exception:
                pass
        if output is not None:
            try:
                output.close()
            except Exception:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def fetch_object(client, bucket, row, root, progress=None):
    """Fetch exactly one approved fixed object, serialized and at most 3 GETs.

    ``progress`` receives only fixed phase/count/error metadata; a body-read
    event is not proof of completion. No thread, range or parallel request is
    created. A cache hit is explicitly local SHA evidence, never a fresh GET.
    """
    identity = _validate_fixed_object(row)
    phase, lock_fd = 'local_prepare', None
    try:
        root = original._root(root, create=True)
        target = original._file(root, identity['path'], create_parents=True)
        lock = target.parent / ('.paper3-get-' + hashlib.sha256(identity['path'].encode()).hexdigest()[:20] + '.lock')
        phase = 'object_lock'
        lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise FetchError('LOCAL_LOCK_NOT_REGULAR', phase=phase)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FetchError('FETCH_ALREADY_ACTIVE_NO_DUPLICATE_GET', phase=phase) from None
        phase = 'existing_local_hash'
        if target.exists():
            original._matches(target, identity)
            _progress(progress, {'phase': 'local_cache_verified', 'bytes_read': 0,
                'expected_bytes': identity['bytes'], 'whole_body_verified': False})
            return dict(identity, status='LOCAL_VERIFIED_CACHE_HIT', get_attempted=False,
                        http_status=None, complete_body_read=None, bytes_read=0,
                        get_sha256=None, local_sha256=identity['sha256'], atomic_commit=None, attempts=0)
        failures = []
        for attempt in range(1, ATTEMPTS + 1):
            try:
                result = _attempt(client, bucket, identity, target, attempt=attempt, progress=progress)
                result['attempt_failures'] = failures
                return result
            except FetchError as exc:
                failures.append({'code': exc.code, 'detail': exc.detail})
                if not exc.retryable or attempt == ATTEMPTS:
                    exc._detail['attempt_failures'] = failures
                    raise
                _progress(progress, {'phase': 'bounded_retry', 'attempt': attempt,
                    'next_attempt': attempt + 1, 'attempt_limit': ATTEMPTS, 'error_code': exc.code,
                    'detail': exc.detail, 'whole_body_verified': False})
                SLEEP(attempt)
    except Exception as exc:
        error = _error(exc, phase)
        raise error from None
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
