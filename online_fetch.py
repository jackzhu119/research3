#!/usr/bin/env python3
"""Fetch one pinned GitHub release asset with whole-body SHA256 verification.

No service exception or signed redirect URL is printed. Authorization is sent
only to api.github.com; redirects to GitHub asset hosts never carry that header.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import warnings
from urllib import error, parse, request

CHUNK = 1024 * 1024
MAX_REDIRECTS = 6
ALLOWED_HOSTS = frozenset({'api.github.com', 'github.com', 'raw.githubusercontent.com',
    'release-assets.githubusercontent.com', 'objects.githubusercontent.com',
    'github-releases.githubusercontent.com'})


class FetchError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _need(value, code):
    if not value:
        raise FetchError(code)


def _safe_url(url):
    try:
        value = parse.urlsplit(url)
        safe = (value.scheme == 'https' and value.hostname in ALLOWED_HOSTS
                and value.username is None and value.password is None
                and value.port in {None, 443} and not value.fragment)
    except (TypeError, ValueError):
        safe = False
    _need(safe, 'GITHUB_RELEASE_REDIRECT_DESTINATION_REFUSED')
    return value


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_once(req):
    # Default ProxyHandler/HTTPS trust are preserved.
    return request.build_opener(_NoRedirect()).open(req, timeout=60)


def _open_get(url, accept, token):
    for _ in range(MAX_REDIRECTS + 1):
        parsed = _safe_url(url)
        headers = {'Accept': accept, 'User-Agent': 'Paper3-fixed-online-bootstrap',
                   'X-GitHub-Api-Version': '2022-11-28'}
        if token is not None and parsed.hostname == 'api.github.com':
            headers['Authorization'] = 'Bearer ' + token
        req = request.Request(url, headers=headers, method='GET')
        response = None
        try:
            response = _open_once(req)
            status = response.getcode()
        except error.HTTPError as exc:
            response, status = exc, exc.code
        except Exception:
            raise FetchError('GITHUB_RELEASE_CONNECTION_FAILED') from None
        if status in {301, 302, 303, 307, 308}:
            target = response.headers.get('Location')
            response.close()
            _need(isinstance(target, str) and target, 'GITHUB_RELEASE_REDIRECT_LOCATION_MISSING')
            url = parse.urljoin(url, target)
            _safe_url(url)
            continue
        if status == 200:
            return response
        response.close()
        if status == 401:
            raise FetchError('GITHUB_RELEASE_AUTH_REQUIRED')
        if status == 404:
            raise FetchError('GITHUB_RELEASE_NOT_FOUND_OR_AUTH_REQUIRED')
        if status == 403:
            raise FetchError('GITHUB_RELEASE_ACCESS_DENIED_OR_RATE_LIMIT')
        if status == 429:
            raise FetchError('GITHUB_RELEASE_RATE_LIMIT')
        raise FetchError('GITHUB_RELEASE_HTTP_NOT_200')
    raise FetchError('GITHUB_RELEASE_TOO_MANY_REDIRECTS')


def _path(path):
    path = Path(os.path.abspath(path))
    _need(not any(p.is_symlink() for p in (path, *path.parents)), 'GITHUB_DOWNLOAD_SYMLINK_REFUSED')
    if path.exists():
        _need(stat.S_ISREG(path.lstat().st_mode), 'GITHUB_DOWNLOAD_REGULAR_FILE_REQUIRED')
    return path


def _hash(path):
    h, n = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(CHUNK), b''):
            h.update(data)
            n += len(data)
    return h.hexdigest(), n


def fetch_release_asset(repo, tag, name, expected_sha, expected_bytes, destination, token=None):
    """Fetch to a final FILE path. Six positional arguments suffice for public GitHub.

    Existing identical content is reported as a local cache hit, never as a new
    GET. Wrong existing content is preserved and rejected. `token`, if used, is
    memory-only and is attached exclusively to api.github.com requests.
    """
    _need(isinstance(repo, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}', repo),
          'GITHUB_RELEASE_REPOSITORY_INVALID')
    _need(isinstance(tag, str) and tag and len(tag) <= 200
          and not any(ord(c) < 33 or ord(c) == 127 for c in tag), 'GITHUB_RELEASE_TAG_INVALID')
    _need(isinstance(name, str) and name and name not in {'.', '..'} and '/' not in name
          and '\\' not in name and not any(ord(c) < 32 or ord(c) == 127 for c in name),
          'GITHUB_RELEASE_ASSET_NAME_INVALID')
    _need(isinstance(expected_sha, str) and re.fullmatch(r'[a-f0-9]{64}', expected_sha)
          and type(expected_bytes) is int and expected_bytes > 0, 'GITHUB_RELEASE_IDENTITY_INVALID')
    _need(token is None or (isinstance(token, str) and 1 <= len(token) <= 4096
          and all(33 <= ord(c) <= 126 for c in token)), 'GITHUB_RELEASE_TOKEN_INVALID')
    target = _path(destination)
    identity = {'repository': repo, 'tag': tag, 'asset_name': name,
                'sha256': expected_sha, 'bytes': expected_bytes, 'credentials_recorded': False}
    if target.exists():
        _need(_hash(target) == (expected_sha, expected_bytes), 'EXISTING_GITHUB_ASSET_DIFFERS_PRESERVED')
        return dict(identity, status='LOCAL_VERIFIED_CACHE_HIT', get_attempted=False,
                    network_bytes_read=0, http_status=None, complete_body_read=None)
    metadata_url = 'https://api.github.com/repos/' + repo + '/releases/tags/' + parse.quote(tag, safe='')
    response = _open_get(metadata_url, 'application/vnd.github+json', token)
    try:
        raw = response.read(2 * CHUNK + 1)
        _need(len(raw) <= 2 * CHUNK, 'GITHUB_RELEASE_METADATA_TOO_LARGE')
        value = json.loads(raw)
    except FetchError:
        raise
    except Exception:
        raise FetchError('GITHUB_RELEASE_METADATA_INVALID') from None
    finally:
        response.close()
    _need(isinstance(value, dict) and value.get('tag_name') == tag
          and value.get('draft') is False and isinstance(value.get('assets'), list),
          'GITHUB_RELEASE_METADATA_IDENTITY_MISMATCH')
    assets = [row for row in value['assets'] if isinstance(row, dict) and row.get('name') == name]
    _need(len(assets) == 1, 'GITHUB_RELEASE_EXACT_ASSET_MISSING')
    asset = assets[0]
    _need(type(asset.get('id')) is int and asset['id'] > 0
          and type(asset.get('size')) is int and asset['size'] == expected_bytes
          and asset.get('state') == 'uploaded', 'GITHUB_RELEASE_ASSET_METADATA_MISMATCH')
    # For public releases the browser endpoint avoids API asset JSON negotiation.
    # A private asset must use the API endpoint with the in-memory token.
    if token is None:
        asset_url = 'https://github.com/' + repo + '/releases/download/' + parse.quote(tag, safe='') + '/' + parse.quote(name, safe='')
    else:
        asset_url = 'https://api.github.com/repos/' + repo + '/releases/assets/' + str(asset['id'])
    response = _open_get(asset_url, 'application/octet-stream', token)
    identity['asset_id'] = asset['id']
    return _save_response(response, target, identity, expected_sha, expected_bytes,
                          'GITHUB_RELEASE_COMPLETE_GET_SHA256_VERIFIED')


def fetch_pinned_file(repo, commit, path, expected_sha, expected_bytes, destination):
    """Public software file at an immutable commit; no API/token/upload host."""
    _need(isinstance(repo, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}', repo),
          'GITHUB_RELEASE_REPOSITORY_INVALID')
    _need(isinstance(commit, str) and re.fullmatch(r'[a-f0-9]{40}', commit), 'IMMUTABLE_GITHUB_COMMIT_REQUIRED')
    _need(isinstance(path, str) and path and '\\' not in path
          and all(part not in {'', '.', '..'} for part in path.split('/'))
          and not any(ord(c) < 32 or ord(c) == 127 for c in path), 'PINNED_GITHUB_PATH_INVALID')
    _need(isinstance(expected_sha, str) and re.fullmatch(r'[a-f0-9]{64}', expected_sha)
          and type(expected_bytes) is int and expected_bytes > 0, 'GITHUB_RELEASE_IDENTITY_INVALID')
    target = _path(destination)
    identity = {'repository': repo, 'commit': commit, 'path': path, 'sha256': expected_sha,
                'bytes': expected_bytes, 'credentials_recorded': False}
    if target.exists():
        _need(_hash(target) == (expected_sha, expected_bytes), 'EXISTING_GITHUB_ASSET_DIFFERS_PRESERVED')
        return dict(identity, status='LOCAL_VERIFIED_CACHE_HIT', get_attempted=False,
                    network_bytes_read=0, http_status=None, complete_body_read=None)
    url = 'https://raw.githubusercontent.com/' + repo + '/' + commit + '/' + parse.quote(path, safe='/')
    response = _open_get(url, 'application/octet-stream', None)
    return _save_response(response, target, identity, expected_sha, expected_bytes,
                          'PINNED_GITHUB_FILE_COMPLETE_GET_SHA256_VERIFIED')


def _save_response(response, target, identity, expected_sha, expected_bytes, status):
    temporary = None
    try:
        declared = response.headers.get('Content-Length')
        if declared is not None:
            _need(str(declared).isdigit() and int(declared) == expected_bytes,
                  'GITHUB_RELEASE_DECLARED_LENGTH_MISMATCH')
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _path(target)
        descriptor, temporary = tempfile.mkstemp(prefix='.paper3-download-', suffix='.partial', dir=target.parent)
        h, n = hashlib.sha256(), 0
        with os.fdopen(descriptor, 'wb') as output:
            while data := response.read(CHUNK):
                _need(isinstance(data, bytes), 'GITHUB_RELEASE_NON_BINARY_BODY')
                n += len(data)
                _need(n <= expected_bytes, 'GITHUB_RELEASE_BODY_LENGTH_MISMATCH')
                h.update(data)
                output.write(data)
            _need((h.hexdigest(), n) == (expected_sha, expected_bytes), 'GITHUB_RELEASE_BODY_SHA_OR_LENGTH_MISMATCH')
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FetchError('GITHUB_DOWNLOAD_TARGET_CREATED_CONCURRENTLY') from None
        return dict(identity, status=status,
                    get_attempted=True, network_bytes_read=n, http_status=200,
                    complete_body_read=True, atomic_commit='link_no_overwrite')
    except FetchError:
        raise
    except Exception:
        raise FetchError('GITHUB_RELEASE_DOWNLOAD_FAILED') from None
    finally:
        response.close()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='论文3固定 GitHub 发布包下载与完整 SHA256 核验。')
    parser.add_argument('--repo', default='jackzhu119/research3')
    parser.add_argument('--tag', help='仅显式指定时读取 GitHub Release 附件')
    parser.add_argument('--commit', default='4e4fa1504e01bf2d04875c9ce04d543f3248f826')
    parser.add_argument('--path', default='launcher_assets/Paper3_OneClick_20261021.zip')
    parser.add_argument('--name', default='Paper3_OneClick_20261021.zip')
    parser.add_argument('--sha256', default='5bb056d39254854707e91c4eb2ce2a4cc6e8e922a9337a52f3daea08b0fb5489')
    parser.add_argument('--bytes', type=int, default=17717251)
    parser.add_argument('--destination', type=Path, default=Path('/workspace/Paper3_OneClick_20261021.zip'))
    args = parser.parse_args(argv)
    token = None
    try:
        if args.tag is None:
            result = fetch_pinned_file(args.repo, args.commit, args.path, args.sha256, args.bytes, args.destination)
            print(json.dumps(result, ensure_ascii=False))
            return 0
        try:
            result = fetch_release_asset(args.repo, args.tag, args.name, args.sha256, args.bytes, args.destination)
        except FetchError as exc:
            if exc.code not in {'GITHUB_RELEASE_AUTH_REQUIRED', 'GITHUB_RELEASE_NOT_FOUND_OR_AUTH_REQUIRED'} or not sys.stdin.isatty():
                raise
            with warnings.catch_warnings():
                warnings.simplefilter('error', getpass.GetPassWarning)
                try:
                    token = getpass.getpass('若该发布需要授权，请输入 GitHub Token（不回显）：').strip()
                except getpass.GetPassWarning:
                    raise FetchError('INTERACTIVE_HIDDEN_INPUT_REQUIRED') from None
            result = fetch_release_asset(args.repo, args.tag, args.name, args.sha256, args.bytes, args.destination, token)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        code = exc.code if isinstance(exc, FetchError) else type(exc).__name__
        print('固定发布包下载未完成：' + code)
        return 2
    finally:
        token = None


if __name__ == '__main__':
    raise SystemExit(main())
