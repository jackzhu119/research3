"""Lossless one-file supplement for Paper3 R7; independent of all training jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper3_result_transport import GitHub, ApiError
from paper_credentials import upload_discover_token

SOURCE_REL = 'runs/followup_20261024_r7/reports/report.json'
PREFIX = 'results/paper3/round-07'
CHUNK_BYTES = 1024 * 1024


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields), ensure_ascii=False), flush=True)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-%d' % os.getpid())
    temporary.write_bytes(encoded(value))
    os.replace(temporary, path)


def version(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utf8_chunks(handle, chunk_bytes=CHUNK_BYTES):
    """At most chunk_bytes+3 source bytes per chunk; never split a UTF-8 codepoint."""
    if chunk_bytes < 4:
        raise ValueError('Chunk size must be at least four bytes')
    while True:
        data = handle.read(chunk_bytes)
        if not data:
            return
        while True:
            try:
                data.decode('utf-8')
                break
            except UnicodeDecodeError as error:
                if error.reason != 'unexpected end of data' or len(data) - error.start > 3:
                    raise ValueError('Report is not UTF-8; original source remains untouched') from None
                following = handle.read(1)
                if not following:
                    raise ValueError('Report ends in an incomplete UTF-8 codepoint') from None
                data += following
        yield data


class TopLevelIndex:
    """Bounded structural index, not a JSON validity claim or statistics recomputation."""
    def __init__(self):
        self.depth = 0
        self.in_string = False
        self.escaped = False
        self.expect_key = False
        self.key_buffer = None
        self.current = None
        self.fields = []
        self.offset = 0

    def feed(self, data):
        for byte in data:
            here = self.offset
            self.offset += 1
            if self.in_string:
                if self.key_buffer is not None and len(self.key_buffer) < 4096:
                    self.key_buffer.append(byte)
                if self.escaped:
                    self.escaped = False
                elif byte == 92:
                    self.escaped = True
                elif byte == 34:
                    self.in_string = False
                    if self.key_buffer is not None:
                        try:
                            name = json.loads(bytes(self.key_buffer).decode())
                        except (ValueError, UnicodeError):
                            name = '[key longer than structural index limit]'
                        self.current = dict(key=name, key_start_byte=self.key_start)
                        self.fields.append(self.current)
                        self.key_buffer = None
                        self.expect_key = False
                continue
            if byte == 34:
                self.in_string = True
                if self.depth == 1 and self.expect_key:
                    self.key_start = here
                    self.key_buffer = bytearray(b'"')
            elif byte in (123, 91):
                self.depth += 1
                if self.depth == 1 and byte == 123:
                    self.expect_key = True
            elif byte in (125, 93):
                if self.depth == 1 and self.current is not None:
                    self.current['end_byte_exclusive'] = here
                self.depth -= 1
            elif byte == 44 and self.depth == 1:
                if self.current is not None:
                    self.current['end_byte_exclusive'] = here
                self.expect_key = True
            elif byte == 58 and self.depth == 1 and self.current is not None:
                self.current.setdefault('value_start_byte', here + 1)

    def result(self):
        return dict(method='streamed top-level lexical structure; JSON not reparsed',
                    balanced_container_depth=self.depth == 0, unterminated_string=self.in_string,
                    fields=self.fields)


def prepare(root, work, *, chunk_bytes=CHUNK_BYTES, notify=emit):
    """Copy exactly one stable report to durable chunks; no walk, glob or task reads."""
    source = Path(root) / SOURCE_REL
    initial = version(source.stat())
    identity = sha(encoded(dict(path=SOURCE_REL, version=initial, chunk_bytes=chunk_bytes)))[:24]
    snapshot_dir = Path(work) / 'source-cache' / identity
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cached_manifest = snapshot_dir / 'manifest.json'
    if cached_manifest.is_file():
        cached = json.loads(cached_manifest.read_text())
        if cached['source_version'] == initial:
            good = all((snapshot_dir / p['local_name']).is_file() and
                       sha((snapshot_dir / p['local_name']).read_bytes()) == p['sha256']
                       for p in cached['parts'])
            if good and version(source.stat()) == initial:
                notify('R7_REPORT_CACHE_REUSED', source_bytes=cached['source_bytes'], chunks=len(cached['parts']))
                return snapshot_dir, cached
    digest, offset, parts, index = hashlib.sha256(), 0, [], TopLevelIndex()
    notify('R7_REPORT_COPY_STARTED', source=SOURCE_REL, source_bytes=initial['bytes'], task_tree_scan=False)
    with source.open('rb') as handle:
        if version(os.fstat(handle.fileno())) != initial:
            raise RuntimeError('Source changed before the report copy')
        for number, block in enumerate(utf8_chunks(handle, chunk_bytes), 1):
            name = 'part-%06d.json-fragment.txt' % number
            temporary = snapshot_dir / (name + '.tmp')
            temporary.write_bytes(block)
            os.replace(temporary, snapshot_dir / name)
            digest.update(block)
            index.feed(block)
            parts.append(dict(number=number, local_name=name, bytes=len(block), offset_bytes=offset,
                              sha256=sha(block)))
            offset += len(block)
            notify('R7_REPORT_COPY_PROGRESS', copied_bytes=offset, source_bytes=initial['bytes'], chunks=number)
        final_fd = version(os.fstat(handle.fileno()))
    if final_fd != initial or version(source.stat()) != initial or offset != initial['bytes']:
        raise RuntimeError('Source report changed while copying; no complete pointer published')
    if not parts:
        raise ValueError('Source report is empty; no complete pointer published')
    structure = index.result()
    for field in structure['fields']:
        start = field['key_start_byte']
        end = field.get('end_byte_exclusive', offset)
        field['parts'] = [p['number'] for p in parts if p['offset_bytes'] < end and
                          p['offset_bytes'] + p['bytes'] > start]
    manifest = dict(schema_version='paper3-report-json-lossless-v1', round=7, source_path=SOURCE_REL,
                    source_version=initial, source_bytes=offset, source_sha256=digest.hexdigest(),
                    parts=parts, structural_index=structure,
                    content_encoding='UTF-8; exact original bytes in ordered fragments',
                    reconstruction='Concatenate parts in number order without separators; verify source_sha256',
                    statistics_recomputed=False, report_validity_claimed=False,
                    training_jobs_changed=0, full_archive_changed=False,
                    copied_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic(cached_manifest, manifest)
    return snapshot_dir, manifest


def publish_entries(gh, entries, message):
    """Commit references to previously uploaded blobs, without loading all chunk bytes."""
    gh.ensure_branch()
    quoted = urllib.parse.quote(gh.branch, safe='/')
    endpoint = gh.prefix + '/git/ref/heads/' + quoted
    while True:
        head = gh.api(endpoint)['object']['sha']
        base = gh.api(gh.prefix + '/git/commits/' + head)['tree']['sha']
        tree = gh.api(gh.prefix + '/git/trees', dict(base_tree=base, tree=entries), 'POST')['sha']
        if tree == base:
            return head
        commit = gh.api(gh.prefix + '/git/commits', dict(message=message, tree=tree, parents=[head]), 'POST')['sha']
        try:
            gh.api(gh.prefix + '/git/refs/heads/' + quoted, dict(sha=commit, force=False), 'PATCH')
            return commit
        except ApiError as error:
            if error.status not in (409, 422):
                raise
            gh.emit('R7_REPORT_REF_REBASE', branch=gh.branch)


def publish(gh, cache, source_manifest, work, notify=emit):
    snapshot = source_manifest['source_sha256']
    prefix = PREFIX + '/report-json/' + snapshot
    manifest = dict(source_manifest)
    manifest['parts'] = [dict(p, path=prefix + '/parts/' + p['local_name']) for p in manifest['parts']]
    entries, uploaded = [], 0
    for part in manifest['parts']:
        block = (cache / part['local_name']).read_bytes()
        if len(block) != part['bytes'] or sha(block) != part['sha256']:
            raise ValueError('Cached report part hash differs; no complete pointer published')
        git_sha = gh._blob(block)
        entries.append(dict(path=part['path'], mode='100644', type='blob', sha=git_sha))
        uploaded += part['bytes']
        atomic(Path(work) / 'status.json', dict(status='UPLOADING', uploaded_parts=part['number'],
               total_parts=len(manifest['parts']), uploaded_bytes=uploaded, total_bytes=manifest['source_bytes']))
        notify('R7_REPORT_CHUNK_UPLOADED', part=part['number'], total_parts=len(manifest['parts']),
               uploaded_bytes=uploaded, source_bytes=manifest['source_bytes'])
    pointer = dict(schema_version='paper3-report-json-pointer-v1', round=7,
                   status='REPORT_JSON_LOSSLESS_UPLOADED', source_sha256=snapshot,
                   source_bytes=manifest['source_bytes'], chunks=len(manifest['parts']),
                   manifest=prefix + '/manifest.json', index=prefix + '/README_ZH.md',
                   all_original_report_bytes_uploaded=True, statistics_recomputed=False,
                   full_raw_results_uploaded=False, training_jobs_changed=0)
    lines = ['# 第七轮完整总报告 JSON', '',
             '此补传只读取 reports/report.json 一份文件。原报告大于旧版快速通道的 64 MiB 限额，现按 UTF-8 边界分片，所有原始字节均保留。', '',
             '[完整校验清单](manifest.json) · [原有 CSV 与统计报告](../../PAPER_DATA_ZH.md)', '',
             '原始大小：%d 字节；SHA256：`%s`。' % (manifest['source_bytes'], snapshot), '',
             '按分片编号顺序直接拼接（不能添加换行）即可恢复原文件。每片和整文件都有 SHA256；分片本身可能不是独立 JSON。', '',
             '这是已有报告的原样补传，不重新计算统计，不代表完整逐试次输出归档已结束。', '',
             '| 分片 | 字节偏移 | 字节数 |', '|---|---:|---:|']
    for p in manifest['parts']:
        lines.append('| [part %d](parts/%s) | %d | %d |' % (p['number'], p['local_name'], p['offset_bytes'], p['bytes']))
    lines += ['', '## 顶层字段定位', '', '下表来自流式结构扫描，便于定位分片；不作为 JSON 语法或科研结论的验证。', '', '| 字段 | 分片编号 |', '|---|---|']
    for field in manifest['structural_index']['fields']:
        lines.append('| %s | %s |' % (str(field['key']).replace('|', '\\|'), ', '.join(map(str, field['parts']))))
    files = {prefix + '/manifest.json': encoded(manifest), prefix + '/README_ZH.md': ('\n'.join(lines) + '\n').encode(),
             PREFIX + '/REPORT_JSON_LATEST.json': encoded(pointer),
             PREFIX + '/REPORT_JSON_ZH.md': ('# 第七轮总报告补传\n\n[完整报告分片与校验](report-json/%s/README_ZH.md) · [原有分析 CSV](PAPER_DATA_ZH.md)\n' % snapshot).encode()}
    for path, block in sorted(files.items()):
        entries.append(dict(path=path, mode='100644', type='blob', sha=gh._blob(block)))
    commit = publish_entries(gh, entries, 'Supplement complete Paper3 round 7 aggregate report JSON')
    receipt = dict(pointer, commit=commit, published_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    atomic(Path(work) / 'R7-REPORT-JSON-UPLOADED.json', receipt)
    atomic(Path(work) / 'status.json', dict(receipt, status='COMPLETE'))
    notify('R7_REPORT_JSON_UPLOADED', commit=commit, source_bytes=manifest['source_bytes'], chunks=len(manifest['parts']),
           url='https://github.com/jackzhu119/research3/blob/paper3-results/' + PREFIX + '/REPORT_JSON_ZH.md')
    return receipt


def worker(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('R7_REPORT_UPLOADER_ALREADY_RUNNING', log=str(work / 'upload.log'))
            return 0
        atomic(work / 'worker.json', dict(pid=os.getpid(), source_path=SOURCE_REL, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        while True:
            try:
                cache, manifest = prepare(args.root, work)
                gh = GitHub(upload_discover_token(args.credentials), emit, cache_path=work / 'git-blobs.json')
                publish(gh, cache, manifest, work)
                return 0
            except Exception as error:
                # Log type only; network exceptions may contain sensitive request URLs.
                atomic(work / 'status.json', dict(status='RETRYING', error_type=type(error).__name__, training_jobs_changed=0))
                emit('R7_REPORT_RETRY', error_type=type(error).__name__, retry_seconds=30,
                     existing_upload_cache_preserved=True, training_jobs_changed=0)
                time.sleep(30)


def start(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        with (work / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                emit('R7_REPORT_UPLOADER_ALREADY_RUNNING', log=str(work / 'upload.log'))
                return 0
            fcntl.flock(lock, fcntl.LOCK_UN)
        upload_discover_token(args.credentials)
        command = [sys.executable, '-I', '-S', '-B', '-u', str(Path(__file__).resolve()), '--worker',
                   '--root', args.root, '--work-dir', str(work)]
        if args.credentials:
            command += ['--credentials', args.credentials]
        with (work / 'upload.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True)
        emit('R7_REPORT_UPLOADER_STARTED', pid=child.pid, log=str(work / 'upload.log'),
             training_jobs_changed=0, previous_csv_upload_unchanged=True, full_archive_unchanged=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home() / 'paper3-r7-report-json-upload'))
    parser.add_argument('--credentials')
    args = parser.parse_args()
    if args.start:
        return start(args)
    if args.worker:
        return worker(args)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
