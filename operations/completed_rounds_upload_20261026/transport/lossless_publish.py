"""Stream exact existing report bytes to GitHub without touching experiments.

Only explicitly supplied files are read. No recursive discovery, statistical
recomputation, checkpoint archive, dependency installation or scheduler action.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import urllib.error
import urllib.parse

from .paper3_result_transport import ApiError, GitHub

MIB = 1024 * 1024


class ValidationError(ValueError):
    """Only a fixed local error code is exposed, never file or HTTP contents."""


def _need(value, code):
    if not value:
        raise ValidationError(code)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _relative(value):
    _need(isinstance(value, str) and value and '\\' not in value
          and not any(ord(c) < 32 for c in value), 'invalid_source_relative_path')
    pure = PurePosixPath(value)
    _need(not pure.is_absolute() and pure.as_posix() == value
          and all(p not in {'', '.', '..'} for p in value.split('/')), 'invalid_source_relative_path')
    _need(not any(p.lower() in {'private', '.git', '.ssh', '.aws'} for p in pure.parts)
          and not any(p.lower().startswith('.env') for p in pure.parts)
          and pure.name.lower() not in {'github.json', 'credentials.json'}, 'private_source_path_refused')
    return pure


def _real_directory(path, create=False):
    path = Path(os.path.abspath(os.fspath(path)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if create:
            current.mkdir(mode=0o700, exist_ok=True)
        _need(stat.S_ISDIR(current.lstat().st_mode) and not current.is_symlink(),
              'directory_symlink_or_non_directory')
    return path


def _atomic(path, data):
    path = Path(path)
    parent = _real_directory(path.parent, create=True)
    _need(not path.is_symlink(), 'staging_symlink_refused')
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name + '.tmp-', dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _signature(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def _open_source(base, relative):
    base = _real_directory(base)
    parts = _relative(relative).parts
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    parent = os.open(base, flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent); parent = child
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
    finally:
        os.close(parent)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValidationError('source_not_regular')
    return descriptor


def _current_version(base, relative):
    descriptor = _open_source(base, relative)
    try:
        return _signature(os.fstat(descriptor))
    finally:
        os.close(descriptor)


def _chunks(stream, chunk_bytes):
    while True:
        block = stream.read(chunk_bytes)
        if not block:
            return
        while True:
            try:
                block.decode('utf-8')
                break
            except UnicodeDecodeError as error:
                _need(error.reason == 'unexpected end of data'
                      and len(block) - error.start <= 3, 'source_utf8_invalid')
                following = stream.read(1)
                _need(bool(following), 'source_utf8_incomplete')
                block += following
        yield block


def _part_bytes(part):
    path = Path(part['local_path'])
    _need(type(part['bytes']) is int and 0 <= part['bytes'] <= 4 * MIB + 3
          and re.fullmatch(r'[a-f0-9]{64}', part['sha256']) is not None, 'staged_part_identity_invalid')
    before = path.lstat()
    _need(not path.is_symlink() and stat.S_ISREG(before.st_mode), 'staged_part_not_regular')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(descriptor)
        _need(_signature(opened) == _signature(before)
              and opened.st_size == part['bytes'], 'staged_part_size_changed')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            data = stream.read(part['bytes'] + 1)
        _need(_signature(os.fstat(descriptor)) == _signature(before) == _signature(path.lstat()),
              'staged_part_changed_during_read')
        _need(len(data) == part['bytes'] and _sha(data) == part['sha256'], 'staged_part_SHA256_changed')
        return data
    finally:
        os.close(descriptor)


def _hydrate(folder, record):
    _need(record.get('schema_version') == 'paper3-lossless-staged-file-v1', 'staged_manifest_invalid')
    parts = []
    for part in record['parts']:
        _need(re.fullmatch(r'part-[0-9]{6}\.txt', part['local_name']) is not None,
              'staged_part_name_invalid')
        parts.append(dict(part, local_path=str(folder / part['local_name'])))
    return dict(record, parts=parts, cache_manifest=str(folder / 'manifest.json'))


def stage_file(source_base, relative, cache_dir, *, chunk_bytes=MIB, notify=lambda *a, **k: None):
    """Stage a stable explicit UTF-8 file with exact offsets and whole SHA256.

    No source-size ceiling or JSON/CSV parse/rewrite is imposed. UTF-8 boundary
    fragments reconstruct the exact bytes by concatenation without separators.
    Cached parts are rehashed; an unchanged source is stat-checked, not reread.
    """
    _relative(relative)
    _need(type(chunk_bytes) is int and 4 <= chunk_bytes <= 4 * MIB, 'invalid_chunk_bytes')
    descriptor = _open_source(source_base, relative)
    try:
        version = _signature(os.fstat(descriptor))
        key = _sha(encoded(dict(path=relative, version=version, chunk_bytes=chunk_bytes)))
        folder = _real_directory(Path(cache_dir) / key, create=True)
        _need(folder.stat().st_uid == os.getuid() and folder.stat().st_mode & 0o077 == 0,
              'private_owned_staging_required')
        cached = folder / 'manifest.json'
        if cached.is_file() and not cached.is_symlink():
            try:
                record = _hydrate(folder, json.loads(cached.read_text()))
                _need(record['source_path'] == relative and record['source_version'] == version,
                      'staged_manifest_source_changed')
                digest, offset = hashlib.sha256(), 0
                for number, part in enumerate(record['parts'], 1):
                    _need(part['number'] == number and part['offset_bytes'] == offset,
                          'staged_part_offsets_invalid')
                    data = _part_bytes(part); digest.update(data); offset += len(data)
                _need(offset == version[2] == record['source_bytes']
                      and digest.hexdigest() == record['source_sha256'], 'staged_manifest_SHA256_changed')
                _need(_signature(os.fstat(descriptor)) == version
                      and _current_version(source_base, relative) == version, 'source_changed_during_copy')
                notify('SOURCE_STAGE_CACHE_REUSED', source_path=relative, source_bytes=offset,
                       chunks=len(record['parts']))
                return record
            except (OSError, ValueError, KeyError, TypeError):
                # The original source remains authoritative; repair only this
                # content-addressed owned staging cache, never source files.
                pass
        digest, offset, parts = hashlib.sha256(), 0, []
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            for number, block in enumerate(_chunks(stream, chunk_bytes), 1):
                name = 'part-%06d.txt' % number
                _atomic(folder / name, block)
                parts.append(dict(number=number, local_name=name, bytes=len(block),
                                  offset_bytes=offset, sha256=_sha(block)))
                digest.update(block); offset += len(block)
                notify('SOURCE_STAGE_PROGRESS', source_path=relative, copied_bytes=offset,
                       source_bytes=version[2], chunks=number)
            if not parts:
                _atomic(folder / 'part-000001.txt', b'')
                parts.append(dict(number=1, local_name='part-000001.txt', bytes=0,
                                  offset_bytes=0, sha256=_sha(b'')))
            final = _signature(os.fstat(descriptor))
        _need(final == version and offset == version[2]
              and _current_version(source_base, relative) == version, 'source_changed_during_copy')
        record = dict(schema_version='paper3-lossless-staged-file-v1', source_path=relative,
                      source_version=version, source_bytes=offset, source_sha256=digest.hexdigest(),
                      parts=parts, statistics_recomputed=False, exact_original_bytes=True,
                      reconstruction='Concatenate numbered parts without adding separators; verify source_sha256.')
        _atomic(cached, encoded(record))
        return _hydrate(folder, record)
    finally:
        os.close(descriptor)


class VerifiedGitHub(GitHub):
    """Git blob ACKs and local cache hits are followed by full remote GET/SHA."""
    def _blob(self, content):
        expected = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        verified = getattr(self, '_full_get_verified', None)
        if verified is None:
            verified = self._full_get_verified = set()
        proof = (expected, len(content), _sha(content))
        if proof in verified:
            return expected
        for attempt in range(2):
            actual = super()._blob(content)
            try:
                self._verify_blob(actual, content)
            except ApiError as error:
                if error.status != 404 or attempt:
                    raise
                self.blob_cache.pop(expected, None); self._save_cache()
                continue
            verified.add(proof)
            return actual
        raise ValidationError('remote_blob_unavailable')

    def _verify_blob(self, blob_sha, content):
        url = 'https://api.github.com/' + self.prefix + '/git/blobs/' + blob_sha
        attempt = 0
        while True:
            try:
                digest, count = hashlib.sha256(), 0
                with self._request(url, accept='application/vnd.github.raw+json') as response:
                    status = getattr(response, 'status', 200)
                    _need(status == 200, 'remote_blob_full_HTTP_200_required')
                    while True:
                        block = response.read(MIB)
                        if not block:
                            break
                        count += len(block)
                        _need(count <= len(content), 'remote_blob_bytes_mismatch')
                        digest.update(block)
                _need(count == len(content) and digest.hexdigest() == _sha(content),
                      'remote_blob_SHA256_mismatch')
                self._emit('GIT_BLOB_FULL_GET_SHA256_VERIFIED', bytes=count, sha256=digest.hexdigest())
                return
            except urllib.error.HTTPError as error:
                body = error.read(65536)
                if error.code == 404 or not self._retryable(error, body):
                    raise ApiError(error.code) from None
                attempt += 1; self._delay(attempt, error.headers)
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                attempt += 1
                self._emit('GIT_BLOB_GET_RETRY', error_type=type(error).__name__)
                self._delay(attempt)


def publish_entries(gh, entries, message):
    """Non-force CAS publication preserves both result channels and other rounds."""
    gh.ensure_branch()
    endpoint = gh.prefix + '/git/ref/heads/' + urllib.parse.quote(gh.branch, safe='/')
    while True:
        head = gh.api(endpoint)['object']['sha']
        base = gh.api(gh.prefix + '/git/commits/' + head)['tree']['sha']
        tree = gh.api(gh.prefix + '/git/trees', dict(base_tree=base, tree=entries), 'POST')['sha']
        if tree == base:
            return head
        commit = gh.api(gh.prefix + '/git/commits',
                        dict(message=message, tree=tree, parents=[head]), 'POST')['sha']
        check = gh.api(gh.prefix + '/git/commits/' + commit)
        _need(check.get('tree', {}).get('sha') == tree, 'remote_commit_tree_mismatch')
        try:
            result = gh.api(gh.prefix + '/git/refs/heads/' + urllib.parse.quote(gh.branch, safe='/'),
                            dict(sha=commit, force=False), 'PATCH')
            _need(result.get('object', {}).get('sha') == commit, 'remote_ref_ack_mismatch')
            return commit
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            gh._emit('RESULT_CAS_REBASE', branch=gh.branch)


def publish_snapshot(gh, round_number, files, work_dir, metadata=None):
    """Publish exact staged report bytes; final latest pointer follows all proofs.

    ``files`` are stage_file records. The driver owns credential discovery,
    detached process locks, completed-round selection and missing-file policy.
    This adapter never touches the independent complete-output LATEST pointer.
    """
    _need(type(round_number) is int and 1 <= round_number <= 99, 'invalid_round_number')
    _need(isinstance(gh, VerifiedGitHub), 'complete_remote_GET_verifier_required')
    _need(isinstance(files, list) and bool(files), 'staged_files_required')
    metadata = {} if metadata is None else metadata
    _need(isinstance(metadata, dict), 'snapshot_metadata_object_required')
    names, summaries = set(), []
    for source in files:
        path = str(_relative(source['source_path']))
        _need(path not in names, 'duplicate_source_path')
        names.add(path)
        summaries.append(dict(source_path=path, bytes=source['source_bytes'], sha256=source['source_sha256'],
                              source_version=source['source_version']))
    snapshot = _sha(encoded(dict(round=round_number, files=sorted(summaries, key=lambda r:r['source_path']),
                                metadata=metadata)))
    prefix = 'results/paper3/round-%02d/paper-data/%s' % (round_number, snapshot)
    pending, sources, verified_bytes = [], [], 0

    def add(path, content):
        nonlocal pending
        pending.append(dict(path=path, mode='100644', type='blob', sha=gh._blob(content)))
        if len(pending) == 32:
            publish_entries(gh, pending, 'Preserve verified Paper3 round %d report fragments' % round_number)
            pending = []

    for source in files:
        digest, offset, parts = hashlib.sha256(), 0, []
        for number, part in enumerate(source['parts'], 1):
            _need(part['number'] == number and part['offset_bytes'] == offset, 'staged_part_offsets_invalid')
            data = _part_bytes(part)
            path = (prefix + '/files/' + source['source_path'] if len(source['parts']) == 1 else
                    prefix + '/fragments/' + source['source_path'] + '/part-%06d.txt' % number)
            blob = gh._blob(data)
            pending.append(dict(path=path, mode='100644', type='blob', sha=blob))
            if len(pending) == 32:
                publish_entries(gh, pending, 'Preserve verified Paper3 round %d report fragments' % round_number)
                pending = []
            parts.append(dict(number=number, path=path, offset_bytes=offset, bytes=len(data),
                              sha256=_sha(data), git_blob_sha=blob))
            digest.update(data); offset += len(data); verified_bytes += len(data)
            _atomic(Path(work_dir) / 'transport-status.json', encoded(dict(status='UPLOADING_VERIFIED_FRAGMENTS',
                    round=round_number, source_path=source['source_path'], verified_source_bytes=verified_bytes,
                    scientific_completion_claimed=False)))
        _need(offset == source['source_bytes'] and digest.hexdigest() == source['source_sha256'],
              'staged_source_SHA256_changed')
        manifest_path = prefix + '/source-manifests/' + source['source_path'] + '.json'
        manifest = dict(schema_version='paper3-lossless-source-v1', source_path=source['source_path'],
                        source_bytes=offset, source_sha256=source['source_sha256'],
                        source_version=source['source_version'], parts=parts, exact_original_bytes=True,
                        encoding='UTF-8 original byte fragments; concatenate without separators',
                        statistics_recomputed=False)
        content = encoded(manifest); add(manifest_path, content)
        sources.append(dict(source_path=source['source_path'], source_bytes=offset,
                            source_sha256=source['source_sha256'], manifest=manifest_path,
                            manifest_sha256=_sha(content), manifest_git_blob_sha=hashlib.sha1(
                                b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest(),
                            chunks=len(parts)))
    if pending:
        publish_entries(gh, pending, 'Preserve verified Paper3 round %d report sources' % round_number)
    complete = (metadata.get('collection_status') == 'COMPLETE' and not metadata.get('required_missing')
                and not metadata.get('required_warnings'))
    manifest = dict(schema_version='paper3-completed-paper-data-manifest-v2', round=round_number,
                    snapshot_id=snapshot, files=sources, metadata=metadata,
                    exact_original_bytes=True, remote_complete_blob_GET_SHA256_verified=True,
                    statistics_recomputed=False, full_raw_results_uploaded=False)
    manifest_path = prefix + '/manifest.json'
    pointer = dict(schema_version='paper3-paper-data-pointer-v2', round=round_number,
                   status='PAPER_DATA_FILES_VERIFIED' if complete else 'PAPER_DATA_PARTIAL_FILES_VERIFIED',
                   snapshot_id=snapshot, manifest=manifest_path, index=prefix + '/README_ZH.md',
                   files=len(sources), bytes=verified_bytes, collection_status=metadata.get('collection_status', 'UNKNOWN'),
                   exact_original_bytes=True, remote_complete_blob_GET_SHA256_verified=True,
                   full_raw_results_uploaded=False, report_recomputed=False,
                   scientific_completion_claimed=False)
    lines = ['# 第%d轮 · 已有论文分析数据' % round_number, '',
             '原样保存已有报告、CSV、计划及任务回执；没有重新训练或重新计算统计。', '',
             '[完整文件与校验清单](manifest.json)', '', '| 原文件 | 字节数 | 精确分片清单 |', '|---|---:|---|']
    for source in sources:
        relative = source['manifest'][len(prefix) + 1:]
        lines.append('| %s | %d | [原始字节与 SHA256](%s) |' %
                     (source['source_path'].replace('|', '\\|'), source['source_bytes'], relative))
    lines += ['', '按各源清单的分片顺序直接拼接，不添加换行。所有分片均经完整远端 GET/SHA256 核验。',
              '缺失项、报告时效和任务完整性见清单 metadata；文件已回传不代表科研结论已验证。',
              '本快速通道与完整逐试次输出归档的 LATEST.json 分别保留。', '']
    latest = 'results/paper3/round-%02d/PAPER_DATA_LATEST.json' % round_number
    finals = {manifest_path:encoded(manifest), prefix + '/README_ZH.md':('\n'.join(lines)).encode(),
              latest:encoded(pointer), 'results/paper3/round-%02d/PAPER_DATA_ZH.md' % round_number:
              ('# 第%d轮论文分析数据\n\n[最新精确文件与分片](paper-data/%s/README_ZH.md) · [机器索引](PAPER_DATA_LATEST.json)\n' %
               (round_number, snapshot)).encode()}
    entries = [dict(path=path, mode='100644', type='blob', sha=gh._blob(data)) for path,data in sorted(finals.items())]
    commit = publish_entries(gh, entries, 'Publish verified existing Paper3 round %d analysis data' % round_number)
    for path, data in finals.items():
        expected = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        observed = gh.api(gh.prefix + '/contents/' + urllib.parse.quote(path, safe='/') +
                          '?ref=' + urllib.parse.quote(commit, safe=''))
        _need(observed.get('sha') == expected, 'published_index_path_mismatch')
    receipt = dict(pointer, commit=commit, published_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   published_index_paths_verified=True)
    _atomic(Path(work_dir) / ('round-%02d-PAPER-UPLOADED.json' % round_number), encoded(receipt))
    return receipt
