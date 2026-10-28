"""Bounded, optional GitHub delivery for one small R1 diagnostic JSON.

No training modules, original progress paths, or scientific outputs are changed.
Private credentials and HTTP responses never become diagnostic log messages.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = 'jackzhu119/research3'
BRANCH = 'paper3-r1-progress'
BASE_BRANCH = 'paper3-r1-progress'
REPORT_PATH = 'monitoring/paper3-round1/status.json'
API_PREFIX = 'https://api.github.com/repos/' + REPOSITORY + '/'
MAX_REPORT_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class DeliveryError(Exception):
    def __init__(self, code, http_status=None):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


def _emit(callback, step):
    event = dict(event='R1_DIAGNOSTIC_UPLOAD_STEP', step=step,
                 training_controlled=False)
    try:
        if callback is None:
            print(json.dumps(event, sort_keys=True), flush=True)
        else:
            callback(event)
    except Exception:
        pass


def _remaining(deadline, clock):
    left = deadline - clock()
    if left <= 0:
        raise DeliveryError('UPLOAD_DEADLINE')
    return min(8., left)


def _isolated(payload, timeout):
    """Stop only the new helper object on timeout; never discover other PIDs."""
    command = [sys.executable, '-I', '-S', '-B', '-u',
               str(Path(__file__).absolute()), '--private-worker']
    child = subprocess.Popen(command, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             close_fds=True)
    try:
        output, _ = child.communicate(json.dumps(payload).encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        try:
            child.communicate(timeout=1.)
        except subprocess.TimeoutExpired:
            pass
        raise DeliveryError('UPLOAD_DEADLINE') from None
    if child.returncode or len(output) > 3 * MAX_RESPONSE_BYTES:
        raise DeliveryError('PRIVATE_HELPER_FAILED')
    try:
        result = json.loads(output)
    except (ValueError, UnicodeError):
        raise DeliveryError('PRIVATE_HELPER_FAILED') from None
    if not isinstance(result, dict):
        raise DeliveryError('PRIVATE_HELPER_FAILED')
    return result


def _token_provider(home, explicit, timeout):
    result = _isolated(dict(operation='credentials', home=str(home),
                            explicit=str(explicit) if explicit else None), timeout)
    token = result.get('token')
    if not isinstance(token, str) or not token.strip():
        raise DeliveryError('CREDENTIAL_UNAVAILABLE')
    return token.strip()


def _requester(endpoint, token, body, method, timeout):
    result = _isolated(dict(operation='request', endpoint=endpoint, token=token,
                            body=body, method=method, timeout=timeout), timeout)
    if not result.get('ok'):
        code = result.get('http_status')
        raise DeliveryError('HTTP_ERROR' if isinstance(code, int) else 'NETWORK_ERROR', code)
    return result.get('value')


def _discover_token(home, explicit):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        value = os.environ.get(name, '').strip()
        if value:
            return value
    home = Path(home).expanduser()
    paths = [Path(x).expanduser() for x in
             (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')) if x]
    paths.append(home/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4',
                 'paper3-round5', 'paper3-round6', 'paper3-round7',
                 'paper3-round4-hpcgpu', 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        base = home/name
        paths.append(base/'private/github.json')
        paths.extend(sorted(base.glob('*/private/github.json'))[:100])
    for path in paths:
        try:
            with path.open('rb') as stream:
                data = stream.read(65537)
            if len(data) > 65536:
                continue
            value = json.loads(data)
            token = value.get('token') if isinstance(value, dict) else None
            if isinstance(token, str) and token.strip():
                return token.strip()
        except (OSError, ValueError, UnicodeError):
            continue
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DeliveryError('REDIRECT_REFUSED')


def _ssl_context():
    context = ssl.create_default_context()
    prefixes = {Path(sys.executable).absolute().parent.parent,
                Path(sys.executable).resolve().parent.parent}
    for prefix in prefixes:
        for directory in prefix.glob('lib/python*/site-packages'):
            for suffix in ('botocore/cacert.pem', 'certifi/cacert.pem'):
                path = directory/suffix
                if path.is_file():
                    context.load_verify_locations(cafile=str(path))
    return context


def _wire_request(value):
    endpoint = value['endpoint']
    if (not isinstance(endpoint, str) or endpoint.startswith('/')
            or '..' in endpoint or ':' in endpoint or '\\' in endpoint):
        raise DeliveryError('ENDPOINT_REFUSED')
    url = API_PREFIX + endpoint
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname != 'api.github.com'
            or parsed.username or parsed.password or parsed.fragment
            or not parsed.path.startswith('/repos/' + REPOSITORY + '/')):
        raise DeliveryError('ENDPOINT_REFUSED')
    body = value.get('body')
    data = json.dumps(body, allow_nan=False).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=value['method'],
        headers={'Authorization':'Bearer ' + value['token'],
                 'Accept':'application/vnd.github+json', 'Content-Type':'application/json',
                 'User-Agent':'paper3-r1-readonly-diagnostic-return',
                 'X-GitHub-Api-Version':'2022-11-28'})
    opener = urllib.request.build_opener(_NoRedirect(),
        urllib.request.HTTPSHandler(context=_ssl_context()))
    with opener.open(request, timeout=min(8., float(value['timeout']))) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise DeliveryError('RESPONSE_TOO_LARGE')
    return json.loads(raw) if raw else {}


def publish(report_bytes, home, *, credentials=None, deadline_seconds=45,
            emit=None, _request=None, _token=None, _clock=time.monotonic):
    """Return a sanitized receipt; delivery failure never raises into training.

    _request/_token are test seams. Production isolates credential reads and
    network I/O in newly owned helpers so slow reads cannot defeat the deadline.
    """
    receipt = dict(status='UNAVAILABLE', repository=REPOSITORY, branch=BRANCH,
                   path=REPORT_PATH, diagnostic_only=False, progress_only=True, training_controlled=False)
    started = _clock()
    try:
        if not isinstance(report_bytes, bytes) or len(report_bytes) > MAX_REPORT_BYTES:
            raise DeliveryError('DIAGNOSTIC_JSON_INVALID')
        try:
            def invalid_constant(_):
                raise ValueError('Nonfinite JSON constant')
            report = json.loads(report_bytes, parse_constant=invalid_constant)
        except (ValueError, UnicodeError):
            raise DeliveryError('DIAGNOSTIC_JSON_INVALID') from None
        if not isinstance(report, dict):
            raise DeliveryError('DIAGNOSTIC_JSON_INVALID')
        if (not isinstance(deadline_seconds, (int, float))
                or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
            raise DeliveryError('DEADLINE_INVALID')
        deadline = started + min(float(deadline_seconds), 45.)
        receipt.update(bytes=len(report_bytes), sha256=hashlib.sha256(report_bytes).hexdigest())
        _emit(emit, 'CREDENTIAL_DISCOVERY')
        token = (_token or _token_provider)(home, credentials, _remaining(deadline, _clock))
        if not isinstance(token, str) or not token.strip():
            raise DeliveryError('CREDENTIAL_UNAVAILABLE')
        if token.encode() in report_bytes:
            raise DeliveryError('CREDENTIAL_IN_REPORT_REFUSED')
        requester = _request or _requester

        def api(endpoint, body=None, method='GET', allow404=False):
            try:
                return requester(endpoint, token, body, method, _remaining(deadline, _clock))
            except DeliveryError as error:
                if allow404 and error.http_status == 404:
                    return None
                raise

        _emit(emit, 'BRANCH_OBSERVATION')
        branch = api('git/ref/heads/' + BRANCH, allow404=True)
        if branch is None:
            base = api('git/ref/heads/' + BASE_BRANCH)
            head = base['object']['sha']
            if not re.fullmatch('[a-fA-F0-9]{40}', head):
                raise DeliveryError('BASE_BRANCH_INVALID')
            _emit(emit, 'DIAGNOSTIC_BRANCH_CREATE')
            try:
                api('git/refs', {'ref':'refs/heads/' + BRANCH, 'sha':head}, 'POST')
            except DeliveryError as error:
                if error.http_status not in (409, 422):
                    raise
                api('git/ref/heads/' + BRANCH)

        encoded = base64.b64encode(report_bytes).decode('ascii')
        response = None
        for attempt in range(3):
            _emit(emit, 'DIAGNOSTIC_PUBLISH_ATTEMPT_' + str(attempt + 1))
            old = api('contents/' + REPORT_PATH + '?ref=' + BRANCH, allow404=True)
            body = dict(message='Update Paper3 R1 single-GPU progress',
                        branch=BRANCH, content=encoded)
            if old is not None:
                body['sha'] = old['sha']
            try:
                response = api('contents/' + REPORT_PATH, body, 'PUT')
                break
            except DeliveryError as error:
                if error.http_status not in (409, 422) or attempt == 2:
                    raise
        commit = response['commit']['sha']
        if not isinstance(commit, str) or not re.fullmatch('[a-fA-F0-9]{40}', commit):
            raise DeliveryError('PUBLISHED_COMMIT_INVALID')
        _emit(emit, 'PINNED_FULL_GET_VERIFY')
        remote = api('contents/' + REPORT_PATH + '?ref=' + commit)
        if remote.get('encoding') != 'base64' or not isinstance(remote.get('content'), str):
            raise DeliveryError('REMOTE_CONTENT_INVALID')
        content = base64.b64decode(''.join(remote['content'].split()), validate=True)
        if content != report_bytes or hashlib.sha256(content).hexdigest() != receipt['sha256']:
            raise DeliveryError('REMOTE_SHA256_MISMATCH')
        receipt.update(status='PROGRESS_UPLOADED_VERIFIED', commit_sha=commit,
                       complete_GET_SHA256_verified=True,
                       report_url='https://github.com/' + REPOSITORY + '/blob/' + commit + '/' + REPORT_PATH)
    except DeliveryError as error:
        unavailable = {'CREDENTIAL_UNAVAILABLE', 'CREDENTIAL_IN_REPORT_REFUSED',
                       'DIAGNOSTIC_JSON_INVALID', 'DEADLINE_INVALID'}
        receipt.update(status='UNAVAILABLE' if error.code in unavailable else 'RETRYING',
                       reason_code=error.code, http_status=error.http_status)
    except Exception as error:
        receipt.update(status='UNAVAILABLE', reason_code='DIAGNOSTIC_DELIVERY_ERROR',
                       error_type=type(error).__name__)
    receipt['elapsed_seconds'] = round(max(0., _clock() - started), 3)
    return receipt


def _worker():
    try:
        data = sys.stdin.buffer.read(2 * MAX_RESPONSE_BYTES + 1)
        if len(data) > 2 * MAX_RESPONSE_BYTES:
            raise DeliveryError('PRIVATE_REQUEST_TOO_LARGE')
        value = json.loads(data)
        if value.get('operation') == 'credentials':
            result = dict(token=_discover_token(value['home'], value.get('explicit')))
        elif value.get('operation') == 'request':
            result = dict(ok=True, value=_wire_request(value))
        else:
            raise DeliveryError('PRIVATE_OPERATION_INVALID')
    except urllib.error.HTTPError as error:
        result = dict(ok=False, http_status=error.code)
    except Exception as error:
        result = dict(ok=False, error_type=type(error).__name__)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == '__main__':
    if sys.argv[1:] == ['--private-worker']:
        _worker()
