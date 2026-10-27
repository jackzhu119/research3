"""Bounded GitHub transport for operational sidecars; never controls science.

The legacy uploader's verified Git/release protocols and cache identities are
retained. This module changes retry liveness and reporting, not archive bytes.
"""
from __future__ import annotations

from collections import OrderedDict
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class TransportRetryLater(RuntimeError):
    pass


class TransportStopped(RuntimeError):
    pass


def error_code(error):
    """Return only fixed diagnostic codes, never request URLs/body/credentials."""
    reason = getattr(error, 'reason', error)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return 'TLS_CERTIFICATE_VERIFICATION_FAILED'
    if isinstance(reason, ssl.SSLError):
        return 'TLS_CONNECTION_FAILED'
    if isinstance(reason, socket.gaierror):
        return 'DNS_LOOKUP_FAILED'
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return 'NETWORK_TIMEOUT'
    if isinstance(error, urllib.error.HTTPError):
        return 'GITHUB_HTTP_ERROR'
    if isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED:
        return 'CONNECTION_REFUSED'
    if isinstance(error, TransportStopped):
        return 'MANAGEMENT_STOP_REQUESTED'
    if isinstance(error, TransportRetryLater):
        return 'BOUNDED_TRANSPORT_RETRY_EXHAUSTED'
    return 'NETWORK_OR_REMOTE_OPERATION_UNAVAILABLE'


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Only GitHub HTTPS redirects; CDN GETs never receive Authorization."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        host = (parsed.hostname or '').lower()
        api = host in {'api.github.com', 'uploads.github.com'}
        cdn = host.endswith('.githubusercontent.com')
        if parsed.scheme != 'https' or not (api or cdn):
            raise TransportRetryLater('UNTRUSTED_REDIRECT_REFUSED')
        if not api and req.get_method() not in {'GET', 'HEAD'}:
            raise TransportRetryLater('CROSS_HOST_WRITE_REDIRECT_REFUSED')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and not api:
            for mapping in (redirected.headers, redirected.unredirected_hdrs):
                for name in list(mapping):
                    if name.lower() == 'authorization':
                        del mapping[name]
        return redirected


class _AckCache(OrderedDict):
    """Bounded imported ACKs; full GET verification remains a separate step."""
    def __init__(self, maximum=8192):
        super().__init__(); self.maximum = maximum; self.dirty = OrderedDict()

    def __setitem__(self, key, value):
        if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{40}', key) or value != key:
            return
        super().__setitem__(key, value); self.move_to_end(key); self.dirty[key] = value
        while len(self) > self.maximum:
            old, _ = self.popitem(last=False); self.dirty.pop(old, None)


def _stream_old_cache(path, target):
    """Import an old JSON object incrementally instead of read_text/full loads."""
    decoder = json.JSONDecoder(); buffer = ''; ended = False; started = False
    try:
        with Path(path).open() as handle:
            while not ended:
                block = handle.read(64 * 1024)
                if block:
                    buffer += block
                elif not buffer.strip():
                    break
                cursor = 0
                try:
                    while True:
                        while cursor < len(buffer) and buffer[cursor].isspace(): cursor += 1
                        if not started:
                            if cursor == len(buffer): break
                            if buffer[cursor] != '{': return False
                            cursor += 1; started = True
                        while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ','): cursor += 1
                        if cursor == len(buffer): break
                        if buffer[cursor] == '}': ended = True; cursor += 1; break
                        begin = cursor
                        key, after = decoder.raw_decode(buffer, cursor)
                        cursor = after
                        while cursor < len(buffer) and buffer[cursor].isspace(): cursor += 1
                        if cursor >= len(buffer): cursor = begin; break
                        if buffer[cursor] != ':': return False
                        cursor += 1
                        while cursor < len(buffer) and buffer[cursor].isspace(): cursor += 1
                        try: value, after = decoder.raw_decode(buffer, cursor)
                        except ValueError: cursor = begin; break
                        target[key] = value; cursor = after
                except ValueError:
                    pass
                buffer = buffer[cursor:]
                if len(buffer) > 1024 * 1024 or not block:
                    return bool(ended)
        return ended
    except (OSError, UnicodeError, ValueError):
        return False


def make_bounded_client_class(base):
    """Subclass the exact installed VerifiedGitHub (required by its publisher)."""
    class BoundedGitHub(base):
        def __init__(self, token, emit, *, stop=None, progress=None, cache_path=None,
                     timeout=120, max_retries=3, **kwargs):
            # The old constructor read the entire growing JSON cache. Keep its
            # old file untouched, import ACKs by streaming, append new ACKs.
            super().__init__(token, emit, cache_path=None, timeout=timeout, **kwargs)
            self.stop = stop or threading.Event(); self.progress = progress or (lambda **_: None)
            self.max_retries = max_retries; self._cas_failures = 0
            self.opener = urllib.request.build_opener(SafeRedirect(), urllib.request.HTTPSHandler(context=self.context))
            self.cache_path = Path(cache_path) if cache_path else None
            self.blob_cache = _AckCache()
            self.journal = self.cache_path.with_name(self.cache_path.name+'.recovery-v2.jsonl') if self.cache_path else None
            if self.cache_path:
                _stream_old_cache(self.cache_path, self.blob_cache)
            if self.journal:
                try:
                    with self.journal.open() as handle:
                        while True:
                            line = handle.readline(4097)
                            if not line: break
                            if len(line) > 4096:
                                while line and not line.endswith('\n'):line=handle.readline(4097)
                                continue
                            try: record = json.loads(line)
                            except ValueError: continue
                            key = record.get('git_blob_sha') if isinstance(record, dict) else None
                            self.blob_cache[key] = key
                except OSError: pass
            self.blob_cache.dirty.clear()

        def _notice(self, **values):
            try: self.progress(**values)
            except Exception: pass

        def _check_stop(self):
            if self.stop.is_set(): raise TransportStopped('MANAGEMENT_STOP_REQUESTED')

        def _emit(self, event, **fields):
            allowed = {name: value for name, value in fields.items()
                       if name in {'attempt','delay_seconds','bytes','reused','branch','commit','name','error_type'}}
            self._notice(transport_event=event, transport_fields=allowed)
            try: super()._emit(event, **fields)
            except Exception: pass

        def _delay(self, attempt, headers=None):
            self._check_stop()
            if attempt > self.max_retries:
                raise TransportRetryLater('BOUNDED_TRANSPORT_RETRY_EXHAUSTED')
            delay = min(30., 2. ** min(attempt, 5))
            try: delay = min(30., max(delay, float((headers or {}).get('Retry-After', 0))))
            except (ValueError, TypeError): pass
            self._notice(transport_event='BOUNDED_RETRY_WAIT', retry_attempt=attempt, retry_delay_seconds=delay)
            if self.stop.wait(delay): raise TransportStopped('MANAGEMENT_STOP_REQUESTED')

        def _request(self, url, data=None, method=None, **kwargs):
            self._check_stop()
            self._notice(network_request_active=True, network_operation=method or ('POST' if data is not None else 'GET'))
            try:
                parsed=urllib.parse.urlsplit(url)
                if parsed.scheme!='https' or parsed.hostname not in {'api.github.com','uploads.github.com'}:
                    raise TransportRetryLater('UNSUPPORTED_GITHUB_ENDPOINT')
                content_type=kwargs.get('content_type','application/json')
                request=urllib.request.Request(url,data=data,method=method,headers={
                    'Authorization':'Bearer '+self._token,
                    'User-Agent':'paper3-r11-management-recovery',
                    'Accept':kwargs.get('accept','application/vnd.github+json'),
                    'Content-Type':content_type,'X-GitHub-Api-Version':'2022-11-28'})
                # Sending a healthy32MiB release asset on the school uplink
                # can take several minutes. Do not turn its socket sendall
                # deadline into a short whole-part limit. Verification GETs
                # still stream1MiB blocks with a120s idle timeout.
                timeout=max(900,self.timeout) if method=='POST' and content_type=='application/octet-stream' else self.timeout
                result = self.opener.open(request,timeout=timeout)
                self._notice(network_request_active=False, last_network_success=True)
                return result
            except Exception as error:
                self._notice(network_request_active=False, network_error_code=error_code(error))
                raise

        def api(self, path, body=None, method=None, allow404=False):
            try:
                result = super().api(path, body, method, allow404=allow404)
            except Exception as error:
                # lossless_publish.publish_entries has a CAS while-loop with
                # no delay hook. Bound only repeated ref conflicts here.
                if method == 'PATCH' and '/git/refs/heads/' in path and getattr(error, 'status', None) in (409,422):
                    self._cas_failures += 1
                    if self._cas_failures > self.max_retries:
                        raise TransportRetryLater('BOUNDED_CAS_RETRY_EXHAUSTED') from None
                raise
            if method == 'PATCH' and '/git/refs/heads/' in path:
                self._cas_failures = 0
            return result

        def _asset_hash(self, asset):
            # Never replace a whole-byte proof with GitHub asset metadata.
            self._check_stop(); checksum = hashlib.sha256(); count = 0
            with self._request(self.prefix_url+'/releases/assets/%s'%asset['id'],
                               accept='application/octet-stream') as response:
                if getattr(response, 'status', 200) != 200:
                    raise TransportRetryLater('ARCHIVE_FULL_HTTP200_REQUIRED')
                while True:
                    self._check_stop(); block = response.read(1024*1024)
                    if not block: break
                    checksum.update(block); count += len(block)
                    self._notice(transport_event='ARCHIVE_GET_VERIFY_PROGRESS', verified_part_bytes=count)
            if count != asset['size']:
                raise TransportRetryLater('ARCHIVE_GET_INCOMPLETE')
            return checksum.hexdigest()

        def _save_cache(self):
            if not self.journal or not self.blob_cache.dirty: return
            records = list(self.blob_cache.dirty)
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open('ab') as handle:
                for key in records:
                    handle.write((json.dumps({'git_blob_sha':key},separators=(',',':'))+'\n').encode())
                handle.flush(); os.fsync(handle.fileno())
            for key in records: self.blob_cache.dirty.pop(key, None)

        def _blob(self, content):
            value = super()._blob(content)
            proof = getattr(self, '_full_get_verified', None)
            if proof is not None:
                while len(proof) > 8192: proof.pop()
            return value

    return BoundedGitHub
