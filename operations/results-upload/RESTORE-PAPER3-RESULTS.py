#!/usr/bin/env python3
"""Download, verify and restore completed Paper 3 round output archives.

Stdlib only; usable with the school's Python -I -S -B -u. The tool does not
import training code, contact Slurm or modify original experiment directories.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request


class RestoreError(Exception):
    pass


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, ensure_ascii=False), flush=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    if temporary.is_symlink():
        raise RestoreError('Restore metadata temporary path is a symbolic link')
    with temporary.open('wb') as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def digest_file(path):
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
    return count, digest.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value:
        raise RestoreError('Unsafe archive or repository path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in path.parts[0]:
        raise RestoreError('Unsafe archive or repository path')
    if str(path) != value.rstrip('/'):
        raise RestoreError('Noncanonical archive or repository path')
    return path


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_no_symlinks(base, relative):
    current = base
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RestoreError('A restore path is a symbolic link: ' + str(relative))
    return current


def certificate_context(extra=None):
    context = ssl.create_default_context()
    candidates = [Path(extra)] if extra else []
    for prefix in {Path(sys.executable).absolute().parent.parent,
                   Path(sys.executable).resolve().parent.parent}:
        for directory in prefix.glob('lib/python*/site-packages'):
            candidates.extend((directory/'botocore/cacert.pem', directory/'certifi/cacert.pem'))
    for path in candidates:
        if path.is_file():
            context.load_verify_locations(cafile=str(path))
    return context


class NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urllib.parse.urlsplit(newurl)
        if destination.scheme != 'https':
            raise RestoreError('A download redirected to a non-HTTPS URL')
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and destination.hostname != urllib.parse.urlsplit(req.full_url).hostname:
            for mapping in (result.headers, result.unredirected_hdrs):
                for key in list(mapping):
                    if key.lower() == 'authorization':
                        del mapping[key]
        return result


class Downloader:
    def __init__(self, repo, branch, *, attempts=0, ca_file=None, timeout=60):
        self.repo, self.branch = repo, branch
        self.attempts, self.timeout = attempts, timeout
        self.token = (os.environ.get('PAPER3_GITHUB_TOKEN') or
                      os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN'))
        self.opener = urllib.request.build_opener(
            NoCredentialRedirect(), urllib.request.HTTPSHandler(context=certificate_context(ca_file)))

    def open(self, url, *, binary=False):
        target = urllib.parse.urlsplit(url)
        if target.scheme != 'https' or not target.hostname or target.username or target.password:
            raise RestoreError('Invalid HTTPS download URL')
        headers = {'User-Agent': 'paper3-complete-results-restore',
                   'Accept': 'application/octet-stream' if binary else 'application/vnd.github.raw+json'}
        # An environment token is sent only to the GitHub API origin. Redirects
        # remove it when changing hosts, including signed release-asset hosts.
        if self.token and target.hostname == 'api.github.com':
            headers['Authorization'] = 'Bearer ' + self.token
        return self.opener.open(urllib.request.Request(url, headers=headers), timeout=self.timeout)

    def urls(self, path):
        path = str(relative_path(path))
        ref = urllib.parse.quote(self.branch, safe='')
        quoted = urllib.parse.quote(path, safe='/')
        raw = 'https://raw.githubusercontent.com/%s/%s/%s' % (self.repo, ref, quoted)
        api = 'https://api.github.com/repos/%s/contents/%s?ref=%s' % (self.repo, quoted, ref)
        return [api, raw] if self.token else [raw, api]

    def retry(self, attempt, context):
        if self.attempts and attempt >= self.attempts:
            raise RestoreError(context + ': retry limit reached; verified cached parts are retained')
        delay = min(30, 2 ** min(attempt, 5))
        emit('DOWNLOAD_RETRY', attempt=attempt, delay_seconds=delay, context=context)
        time.sleep(delay)

    def metadata(self, path, maximum=128*1024*1024):
        attempt = 0
        while True:
            attempt += 1
            failures = []
            for url in self.urls(path):
                try:
                    with self.open(url) as response:
                        content = response.read(maximum + 1)
                    if len(content) > maximum:
                        raise RestoreError('Metadata file exceeds the supported size')
                    value = json.loads(content)
                    if isinstance(value, dict) and value.get('encoding') == 'base64' and value.get('type') == 'file':
                        content = base64.b64decode(value['content'])
                        value = json.loads(content)
                    if not isinstance(value, dict):
                        raise RestoreError('Expected a JSON object for metadata')
                    return content, value
                except urllib.error.HTTPError as error:
                    failures.append(error.code)
                except (OSError, ValueError, urllib.error.URLError):
                    failures.append(None)
            if failures and all(code in (401, 403, 404) for code in failures):
                raise RestoreError('Metadata unavailable (not uploaded yet or permission denied): ' + path)
            self.retry(attempt, path)

    def part(self, record, destination):
        expected_size, expected_sha = record['bytes'], record['sha256']
        if destination.is_symlink():
            raise RestoreError('Cached part is a symbolic link')
        if destination.is_file() and digest_file(destination) == (expected_size, expected_sha):
            emit('PART_CACHE_VERIFIED', name=record['name'], bytes=expected_size)
            return destination
        temporary = destination.with_name(destination.name + '.partial')
        if temporary.is_symlink():
            raise RestoreError('Partial download is a symbolic link')
        urls = []
        api = None
        if isinstance(record.get('asset_id'), int):
            api = 'https://api.github.com/repos/%s/releases/assets/%s' % (self.repo, record['asset_id'])
        if self.token and api:
            urls.append(api)
        if record.get('download_url'):
            urls.append(record['download_url'])
        if api and api not in urls:
            urls.append(api)
        if not urls:
            raise RestoreError('Archive part has no download URL or asset ID')
        attempt = 0
        while True:
            attempt += 1
            failures = []
            for url in urls:
                try:
                    digest, count = hashlib.sha256(), 0
                    with self.open(url, binary=True) as response, temporary.open('wb') as output:
                        while True:
                            block = response.read(min(1024*1024, expected_size-count+1))
                            if not block:
                                break
                            count += len(block)
                            if count > expected_size:
                                raise RestoreError('Downloaded part is larger than its manifest size')
                            digest.update(block)
                            output.write(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if count != expected_size or digest.hexdigest() != expected_sha:
                        failures.append(None)
                        emit('PART_CHECKSUM_RETRY', name=record['name'], attempt=attempt)
                        continue
                    os.replace(temporary, destination)
                    emit('PART_DOWNLOADED_VERIFIED', name=record['name'], bytes=count)
                    return destination
                except urllib.error.HTTPError as error:
                    failures.append(error.code)
                except (OSError, urllib.error.URLError):
                    failures.append(None)
            if failures and all(code in (401, 404) for code in failures):
                raise RestoreError('Archive part is unavailable or permission denied: ' + record['name'])
            self.retry(attempt, record['name'])


class ConcatenatedParts(io.RawIOBase):
    """Read part files as one stream without creating another archive-sized file."""
    def __init__(self, parts):
        self.paths = iter(parts)
        self.handle = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            return b''.join(iter(lambda: self.read(1024*1024), b''))
        result = bytearray()
        while len(result) < size:
            if self.handle is None:
                try:
                    self.handle = next(self.paths).open('rb')
                except StopIteration:
                    break
            block = self.handle.read(size-len(result))
            if block:
                result.extend(block)
            else:
                self.handle.close()
                self.handle = None
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        super().close()


def validate_metadata(archive, manifest, round_number, snapshot):
    if archive.get('round') != round_number or manifest.get('round') != round_number:
        raise RestoreError('Requested round does not match archive metadata')
    if archive.get('snapshot_id') != snapshot or manifest.get('snapshot_id') != snapshot:
        raise RestoreError('Snapshot identities do not match')
    parts = archive.get('parts')
    if not isinstance(parts, list) or not parts:
        raise RestoreError('Archive part list is missing')
    names = set()
    for part in parts:
        name = str(relative_path(part.get('name')))
        if '/' in name or name in names or not valid_sha(part.get('sha256')):
            raise RestoreError('Invalid or duplicate archive part')
        if not isinstance(part.get('bytes'), int) or isinstance(part['bytes'], bool) or part['bytes'] < 0:
            raise RestoreError('Invalid archive part size')
        names.add(name)
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise RestoreError('Manifest file list is missing')
    expected = {}
    for record in files:
        path = str(relative_path(record.get('path')))
        if path in expected or not valid_sha(record.get('sha256')):
            raise RestoreError('Invalid or duplicate manifest file')
        if not isinstance(record.get('bytes'), int) or isinstance(record['bytes'], bool) or record['bytes'] < 0:
            raise RestoreError('Invalid manifest file size')
        if any(component.startswith('.paper3-restore-') for component in PurePosixPath(path).parts):
            raise RestoreError('Manifest uses a restore-reserved path')
        expected[path] = record
    declared_parts = manifest.get('parts')
    if isinstance(declared_parts, list):
        for left, right in zip(parts, declared_parts):
            if any(left.get(k) != right.get(k) for k in ('name', 'bytes', 'sha256')):
                raise RestoreError('Archive and manifest part lists disagree')
        if len(parts) != len(declared_parts):
            raise RestoreError('Archive and manifest part counts disagree')
    if manifest.get('archive_bytes') is not None and sum(p['bytes'] for p in parts) != manifest['archive_bytes']:
        raise RestoreError('Archive total byte count does not match')
    if manifest.get('schema_version') == 'paper3-scientific-output-snapshot-v1':
        identity = {k: manifest[k] for k in ('schema_version', 'round', 'source_output', 'plan_sha256',
                                            'archive_layout', 'files', 'excluded')}
        if hashlib.sha256(json_bytes(identity)).hexdigest() != snapshot:
            raise RestoreError('Manifest content does not match its snapshot identity')
    return expected


def restore_parts(parts, target, expected):
    seen, restored, reused = set(), 0, 0
    with ConcatenatedParts(parts) as stream, tarfile.open(mode='r|gz', fileobj=stream) as archive:
        for member in archive:
            relative = relative_path(member.name)
            name = str(relative)
            path = ensure_no_symlinks(target, relative)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issparse():
                raise RestoreError('Nonregular archive member refused: ' + name)
            if name not in expected or name in seen:
                raise RestoreError('Unexpected or duplicate archive file: ' + name)
            seen.add(name)
            record = expected[name]
            if member.size != record['bytes']:
                raise RestoreError('Archive file size does not match manifest: ' + name)
            if path.is_file() and digest_file(path) == (record['bytes'], record['sha256']):
                reused += 1
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name('.paper3-restore-partial-' + path.name)
                if temporary.is_symlink():
                    raise RestoreError('Restore temporary path is a symbolic link')
                digest, size = hashlib.sha256(), 0
                source = archive.extractfile(member)
                if source is None:
                    raise RestoreError('Cannot read regular archive file: ' + name)
                with source, temporary.open('wb') as output:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        size += len(block)
                        digest.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if size != record['bytes'] or digest.hexdigest() != record['sha256']:
                    temporary.unlink(missing_ok=True)
                    raise RestoreError('Restored file checksum mismatch: ' + name)
                os.replace(temporary, path)
                restored += 1
            if len(seen) % 250 == 0:
                emit('RESTORE_PROGRESS', verified_files=len(seen), total_files=len(expected),
                     restored_files=restored, cached_files=reused)
    missing = set(expected)-seen
    if missing:
        raise RestoreError('Archive lacks %d manifest files' % len(missing))
    return {'verified_files': len(seen), 'restored_files': restored, 'cached_files': reused}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True, type=int, choices=(2, 3, 7), dest='round_number')
    parser.add_argument('--destination', required=True, type=Path,
                        help='Parent directory for a new round-NN-SNAPSHOT restore folder')
    parser.add_argument('--snapshot', help='Exact 64-character snapshot ID; default is LATEST.json')
    parser.add_argument('--branch', default='paper3-results')
    parser.add_argument('--repo', default='jackzhu119/research3')
    parser.add_argument('--attempts', type=int, default=0, help='Download attempts; 0 retries without a total deadline')
    parser.add_argument('--ca-file')
    parser.add_argument('--timeout', type=int, default=60, help='Per network request timeout, seconds')
    args = parser.parse_args(argv)
    if args.attempts < 0 or args.timeout < 1 or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('Invalid repository, attempts or timeout')
    if args.snapshot and not valid_sha(args.snapshot):
        parser.error('--snapshot must be a lowercase SHA256 snapshot ID')
    client = Downloader(args.repo, args.branch, attempts=args.attempts,
                        ca_file=args.ca_file, timeout=args.timeout)
    prefix = 'results/paper3/round-%02d' % args.round_number
    if args.snapshot:
        snapshot = args.snapshot
        archive_path = prefix+'/final/'+snapshot+'/archive.json'
        manifest_path = prefix+'/final/'+snapshot+'/manifest.json'
    else:
        _, latest = client.metadata(prefix+'/LATEST.json', maximum=2*1024*1024)
        snapshot = latest.get('snapshot_id')
        if not valid_sha(snapshot) or latest.get('round') != args.round_number:
            raise RestoreError('LATEST index does not identify the requested round')
        archive_path = latest.get('archive')
        manifest_path = latest.get('manifest')
    expected_prefix = prefix+'/final/'+snapshot+'/'
    if archive_path != expected_prefix+'archive.json' or manifest_path != expected_prefix+'manifest.json':
        raise RestoreError('Latest index references an unexpected snapshot directory')
    archive_bytes, archive = client.metadata(archive_path)
    manifest_bytes, manifest = client.metadata(manifest_path)
    if hashlib.sha256(manifest_bytes).hexdigest() != archive.get('manifest_sha256'):
        raise RestoreError('Manifest checksum does not match the archive index')
    expected = validate_metadata(archive, manifest, args.round_number, snapshot)
    base = args.destination.expanduser().resolve()
    target = base/('round-%02d-%s' % (args.round_number, snapshot))
    source = Path(manifest.get('source_output', '/public/home/zhuziyuan/projects/paper3/runs')).resolve()
    if inside(target, source) or inside(source, target):
        raise RestoreError('Restore destination overlaps the original training output')
    if source.parent.name == 'runs' and inside(target, source.parent):
        raise RestoreError('Choose a destination outside the original training runs directory')
    marker = target/'.paper3-restore-state.json'
    identity = {'repo': args.repo, 'round': args.round_number, 'snapshot_id': snapshot}
    if target.is_symlink():
        raise RestoreError('Restore folder is a symbolic link')
    if target.exists():
        if marker.is_symlink() or not marker.is_file():
            raise RestoreError('Destination folder already exists without a matching restore marker')
        previous = json.loads(marker.read_text())
        if any(previous.get(k) != v for k,v in identity.items()):
            raise RestoreError('Destination folder belongs to a different restore')
    else:
        target.mkdir(parents=True)
    atomic_json(marker, {**identity, 'status': 'RESTORING'})
    cache = base/'.paper3-restore-cache'/('round-%02d-%s' % (args.round_number, snapshot))
    if (base/'.paper3-restore-cache').is_symlink() or cache.is_symlink():
        raise RestoreError('Restore cache is a symbolic link')
    cache.mkdir(parents=True, exist_ok=True)
    atomic_json(cache/'archive.json', archive)
    atomic_json(cache/'manifest.json', manifest)
    emit('RESTORE_STARTED', round=args.round_number, snapshot_id=snapshot,
         destination=str(target), parts=len(archive['parts']), files=len(expected))
    parts = []
    for index, record in enumerate(archive['parts'], 1):
        emit('PART_DOWNLOADING', part=index, total_parts=len(archive['parts']), name=record['name'])
        parts.append(client.part(record, cache/record['name']))
    summary = restore_parts(parts, target, expected)
    atomic_json(marker, {**identity, 'status': 'RESTORED_AND_ALL_FILES_SHA256_VERIFIED', **summary})
    emit('RESTORE_COMPLETE', round=args.round_number, snapshot_id=snapshot,
         destination=str(target), **summary)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        emit('RESTORE_INTERRUPTED', verified_part_cache_preserved=True)
        raise SystemExit(130)
    except Exception as error:
        emit('RESTORE_FAILED', error_type=type(error).__name__, reason=str(error),
             verified_part_cache_preserved=True)
        raise SystemExit(2)
