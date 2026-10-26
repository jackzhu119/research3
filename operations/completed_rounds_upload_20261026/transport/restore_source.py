"""Restore a published lossless source/manifest into an explicitly NEW file.

Both paper3-lossless-source-v1 and paper3-lossless-json-locator-v1 use numbered
parts, byte offsets, part SHA256 and the exact original whole-file SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transport.lossless_publish import ValidationError, _need, _real_directory
from transport.paper3_result_transport import GitHub


def _repo_path(value):
    _need(isinstance(value, str) and value and '\\' not in value
          and not any(ord(c) < 32 for c in value), 'restore_repository_path_invalid')
    pure = PurePosixPath(value)
    _need(not pure.is_absolute() and pure.as_posix() == value
          and not any(p in {'', '.', '..'} for p in value.split('/')), 'restore_repository_path_invalid')
    return value


def restore(manifest, output, read_part):
    """read_part(part) returns bytes. Existing outputs are NEVER replaced."""
    _need(isinstance(manifest, dict) and manifest.get('schema_version') in
          {'paper3-lossless-source-v1', 'paper3-lossless-json-locator-v1'}, 'restore_manifest_schema_invalid')
    _need(type(manifest.get('source_bytes')) is int and manifest['source_bytes'] >= 0
          and re.fullmatch(r'[a-f0-9]{64}', manifest.get('source_sha256', '')) is not None
          and isinstance(manifest.get('parts'), list) and manifest['parts'], 'restore_manifest_identity_invalid')
    output = Path(os.path.abspath(Path(output).expanduser()))
    parent = _real_directory(output.parent, create=True)
    _need(not output.exists() and not output.is_symlink(), 'restore_output_already_exists')
    fd, temporary_name = tempfile.mkstemp(prefix='.' + output.name + '.restore-', dir=parent)
    temporary = Path(temporary_name)
    digest, offset = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as stream:
            for number, part in enumerate(manifest['parts'], 1):
                _need(isinstance(part, dict) and part.get('number') == number
                      and type(part.get('offset_bytes')) is int and part['offset_bytes'] == offset
                      and type(part.get('bytes')) is int and part['bytes'] >= 0
                      and re.fullmatch(r'[a-f0-9]{64}', part.get('sha256', '')) is not None,
                      'restore_part_identity_or_offsets_invalid')
                if 'path' in part:
                    _repo_path(part['path'])
                block = read_part(part)
                _need(isinstance(block, bytes) and len(block) == part['bytes']
                      and hashlib.sha256(block).hexdigest() == part['sha256'], 'restore_part_SHA256_mismatch')
                stream.write(block); digest.update(block); offset += len(block)
            _need(offset == manifest['source_bytes'] and digest.hexdigest() == manifest['source_sha256'],
                  'restore_whole_SHA256_mismatch')
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise ValidationError('restore_output_already_exists') from None
        return dict(status='RESTORED_EXACT_BYTES_SHA256_VERIFIED', output=str(output), bytes=offset,
                    sha256=digest.hexdigest(), original_output_overwritten=False,
                    statistics_recomputed=False)
    finally:
        temporary.unlink(missing_ok=True)


class ReaderGitHub(GitHub):
    """Optional existing credentials; anonymous reads contain no empty token."""
    def _request(self, url, data=None, method=None, *, accept='application/vnd.github+json',
                 content_type='application/json'):
        if self._token:
            return super()._request(url, data, method, accept=accept, content_type=content_type)
        parsed = urllib.parse.urlsplit(url)
        _need(parsed.scheme == 'https' and parsed.hostname == 'api.github.com',
              'restore_GitHub_API_endpoint_required')
        request = urllib.request.Request(url, data=data, method=method,
            headers={'User-Agent': 'paper3-lossless-source-reader', 'Accept': accept,
                     'Content-Type': content_type, 'X-GitHub-Api-Version': '2022-11-28'})
        return self.opener.open(request, timeout=self.timeout)


def remote_blob(gh, blob_sha, expected_bytes, expected_sha256=None):
    _need(re.fullmatch(r'[a-f0-9]{40}', blob_sha or '') is not None
          and type(expected_bytes) is int and expected_bytes >= 0, 'restore_remote_blob_identity_invalid')
    url = 'https://api.github.com/' + gh.prefix + '/git/blobs/' + blob_sha
    blocks, count = [], 0
    with gh._request(url, accept='application/vnd.github.raw+json') as response:
        _need(getattr(response, 'status', 200) == 200, 'restore_complete_HTTP_200_required')
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            _need(count <= expected_bytes, 'restore_remote_blob_bytes_mismatch')
            blocks.append(block)
    raw = b''.join(blocks)
    _need(count == expected_bytes and hashlib.sha1(
        b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob_sha,
        'restore_remote_git_blob_mismatch')
    if expected_sha256 is not None:
        _need(hashlib.sha256(raw).hexdigest() == expected_sha256, 'restore_remote_blob_SHA256_mismatch')
    return raw


def repository_file(gh, path, ref, expected_bytes=None, expected_sha256=None):
    path = _repo_path(path)
    record = gh.api(gh.prefix + '/contents/' + urllib.parse.quote(path, safe='/') +
                    '?ref=' + urllib.parse.quote(ref, safe=''))
    _need(record.get('type') == 'file' and type(record.get('size')) is int,
          'restore_remote_file_metadata_invalid')
    if expected_bytes is not None:
        _need(record['size'] == expected_bytes, 'restore_remote_file_size_mismatch')
    return remote_blob(gh, record['sha'], record['size'], expected_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='Local manifest JSON or repository-relative manifest path')
    parser.add_argument('--output', required=True, help='New output file; any existing target is refused')
    parser.add_argument('--ref', help='Receipt commit SHA; required for all remote retrieval')
    parser.add_argument('--repo', default='jackzhu119/research3')
    parser.add_argument('--credentials', help='Optional existing private/github.json; never copied')
    parser.add_argument('--manifest-sha256', help='Optional manifest SHA256 from the containing file index')
    args = parser.parse_args(argv)
    try:
        _need(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo) is not None,
              'restore_repository_name_invalid')
        token = None
        try:
            from paper_credentials import upload_discover_token
            token = upload_discover_token(args.credentials)
        except RuntimeError:
            _need(args.credentials is None, 'restore_explicit_credentials_unavailable')
        gh = ReaderGitHub(token, lambda *a, **k: None, repo=args.repo)
        local = Path(args.manifest).expanduser()
        if local.is_file():
            _need(not local.is_symlink(), 'restore_manifest_symlink_refused')
            raw = local.read_bytes()
        else:
            _need(re.fullmatch(r'[a-f0-9]{40}', args.ref or '') is not None,
                  'restore_immutable_commit_ref_required')
            raw = repository_file(gh, args.manifest, args.ref)
        if args.manifest_sha256:
            _need(hashlib.sha256(raw).hexdigest() == args.manifest_sha256, 'restore_manifest_SHA256_mismatch')
        manifest = json.loads(raw)

        def read(part):
            _need(re.fullmatch(r'[a-f0-9]{40}', args.ref or '') is not None,
                  'restore_immutable_commit_ref_required')
            if part.get('git_blob_sha'):
                return remote_blob(gh, part['git_blob_sha'], part['bytes'], part['sha256'])
            return repository_file(gh, part['path'], args.ref, part['bytes'], part['sha256'])

        result = restore(manifest, args.output, read)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        result = {'status': 'RESTORE_BLOCKED_OR_FAILED', 'error_type': type(error).__name__}
        if isinstance(error, ValidationError):
            result['error_code'] = str(error)
        print(json.dumps(result))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
