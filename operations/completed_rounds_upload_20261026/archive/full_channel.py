"""Full completed-round archive channel. No experiment imports or Slurm calls."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
for directory in (HERE, PACKAGE, PACKAGE / 'transport'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from paper3_result_pack import prepare_round

UPLOAD_REPOSITORY = 'jackzhu119/research3'
UPLOAD_BRANCH = 'paper3-results'
ALLOWED_ROUNDS = (4, 5, 6, 8)


def upload_emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, ensure_ascii=False, sort_keys=True), flush=True)


def upload_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def upload_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as handle:
        handle.write(upload_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def upload_read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default



def store_manifest(files, prefix, data, *, inline_limit=64 * 1024 * 1024,
                   part_bytes=1024 * 1024):
    """Use the generic lossless-source dialect for an oversized exact manifest."""
    digest = hashlib.sha256(data).hexdigest()
    if len(data) <= inline_limit:
        path = prefix + '/manifest.json'
        files[path] = data
        return {'path': path, 'sha256': digest, 'storage': 'original'}
    parts = []
    for number, offset in enumerate(range(0, len(data), part_bytes), 1):
        chunk = data[offset:offset + part_bytes]
        path = prefix + '/manifest-parts/part-%06d.json-fragment.txt' % number
        files[path] = chunk
        parts.append({'number': number, 'offset_bytes': offset, 'path': path,
                      'bytes': len(chunk), 'sha256': hashlib.sha256(chunk).hexdigest(),
                      'git_blob_sha': hashlib.sha1(('blob %d\0' % len(chunk)).encode() + chunk).hexdigest()})
    path = prefix + '/manifest-source.json'
    index = {'schema_version': 'paper3-lossless-source-v1', 'source_path': 'manifest.json',
             'source_bytes': len(data), 'source_sha256': digest, 'source_version': None,
             'parts': parts, 'exact_original_bytes': True, 'statistics_recomputed': False,
             'encoding': 'UTF-8 original byte fragments; concatenate without separators'}
    files[path] = upload_json_bytes(index)
    return {'path': path, 'sha256': digest, 'storage': 'exact-byte-fragments',
            'index_sha256': hashlib.sha256(files[path]).hexdigest(),
            'source_bytes': len(data), 'parts': len(parts)}


def restore_manifest_bytes(index, read_bytes):
    """Restore an oversized manifest from local or remote byte-reader callback."""
    if index.get('schema_version') != 'paper3-lossless-source-v1':
        raise ValueError('Unsupported exact manifest source dialect')
    chunks, offset, digest = [], 0, hashlib.sha256()
    for number, part in enumerate(index['parts'], 1):
        data = read_bytes(part['path'])
        if (part['number'] != number or part['offset_bytes'] != offset
                or len(data) != part['bytes']
                or hashlib.sha256(data).hexdigest() != part['sha256']):
            raise ValueError('Manifest fragment is missing, out of order or corrupt')
        digest.update(data)
        offset += len(data)
        chunks.append(data)
    if offset != index['source_bytes'] or digest.hexdigest() != index['source_sha256']:
        raise ValueError('Restored manifest SHA256 differs')
    return b''.join(chunks)


def upload_remote_round(gh, packed, work, emit=upload_emit):
    number = packed['round']
    manifest = packed['manifest']
    snapshot = packed['snapshot_id']
    prefix = 'results/paper3/round-%02d/final/%s' % (number, snapshot)
    tag = 'paper3-results-r%02d-%s' % (number, snapshot)
    title = 'Paper3 第%d轮完整输出 %s' % (number, snapshot)
    description = ('第%d轮 %d/%d 任务完成。全部计划任务的预测、验证轨迹、选择文件和checkpoint见分片归档。\n'
                   '按 part 编号拼接后为一个 tar.gz；每片及每个解压文件的SHA256见清单。\n'
                   '原始EEG、共享源EEGNet、安装环境及运行控制文件不属于本轮输出归档。\n'
                   '可浏览索引：https://github.com/%s/tree/%s/%s') % (
                       number, packed['complete_tasks'], packed['total_tasks'],
                       UPLOAD_REPOSITORY, UPLOAD_BRANCH, prefix)
    release = gh.ensure_release(tag, title, description)
    assets = []
    for index, part in enumerate(packed['parts'], 1):
        emit('PART_UPLOADING', round=number, part=index, total_parts=len(packed['parts']),
                    bytes=part['bytes'], name=part['name'])
        record = gh.upload_part(release, Path(part['path']), part['name'], part['sha256'])
        assets.append({'name': part['name'], 'bytes': part['bytes'], 'sha256': part['sha256'],
                       'asset_id': record['id'], 'download_url': record['url']})
        emit('PART_VERIFIED', round=number, part=index, total_parts=len(packed['parts']))
    # An identical immutable manifest is also present among the browsable Git files.
    files = {}
    omitted_browsable = []
    for record in packed.get('summary_files', []):
        relative = record.get('relative_path', record.get('relative', record.get('path_in_archive', record.get('path'))))
        local = record.get('local_path', record.get('path'))
        if not relative or not local:
            raise ValueError('Packager summary_files must provide relative_path and local_path')
        data = Path(local).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Frozen summary checksum changed')
        if len(data) > 16*1024*1024:
            omitted_browsable.append(relative)
            continue
        files[prefix+'/files/'+relative] = data
    manifest_storage = store_manifest(files, prefix, upload_json_bytes(manifest))
    files[prefix+'/TASKS.json'] = upload_json_bytes(manifest.get('tasks', []))
    archive = {
        'schema_version': 'paper3-complete-output-archive-v1', 'round': number,
        'snapshot_id': snapshot, 'status': 'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED',
        'complete_tasks': packed['complete_tasks'], 'total_tasks': packed['total_tasks'],
        'release_tag': tag, 'release_url': release['html_url'],
        'manifest_sha256': manifest_storage['sha256'],
        'manifest_path': manifest_storage['path'],
        'manifest_storage': manifest_storage['storage'],
        'manifest_index_sha256': manifest_storage.get('index_sha256'), 'parts': assets,
        'part_order': 'Concatenate parts in this array order to reconstruct one tar.gz stream.',
        'existing_reports_recomputed': False,
        'large_browsable_files_available_in_archive': omitted_browsable,
        'scope': 'Complete declared-round scientific output; not raw EEG, shared source model or environment backup.',
    }
    files[prefix+'/archive.json'] = upload_json_bytes(archive)
    readme = ('# 第%d轮完整结果\n\n本次逐任务回执：**%d/%d 完成**。\n\n'
              '- [原计划与已有报告、CSV](files/)\n- [任务完成索引](TASKS.json)\n'
              '- [全部科学文件SHA256清单](manifest.json)\n- [完整分片下载索引](archive.json)\n'
              '- [GitHub Release：完整预测、验证轨迹与checkpoint](%s)\n'
              '- [本轮论文分析文件入口](../../PAPER_DATA_ZH.md)\n\n'
              '分片按 archive.json 中的顺序拼接为 tar.gz；下载校验/还原工具见仓库 '
              '`operations/results-upload/RESTORE-PAPER3-RESULTS.py`。\n\n'
              '报告和CSV复制自服务器既有文件，不在上传时重新统计。'
              '若已有报告仍标PARTIAL，完整任务原始输出与本次完成索引均已包含，不能把旧报告计数当成缺文件。\n\n'
              '原始EEG、共享源模型、环境和运行控制文件不在本轮归档范围。\n') % (
                  number, packed['complete_tasks'], packed['total_tasks'], release['html_url'])
    readme = readme.replace('](manifest.json)', '](' + Path(manifest_storage['path']).name + ')')
    if manifest_storage['storage'] == 'exact-byte-fragments':
        readme += ('\n本轮逐文件清单较大，以`paper3-lossless-source-v1`原字节分片保存，内容没有删减。'
                   '恢复前先按`manifest-source.json`的编号、偏移和SHA256重建`manifest.json`，'
                   '再执行逐文件校验；本发布包`archive.full_channel.restore_manifest_bytes`提供该辅助函数。'
                   '旧版还原工具不能直接把分片索引当作原清单。\n')
    files[prefix+'/README_ZH.md'] = readme.encode()
    latest = {'round': number, 'snapshot_id': snapshot,
              'status': 'COMPLETE_OUTPUT_UPLOADED', 'complete_tasks': packed['complete_tasks'],
              'total_tasks': packed['total_tasks'], 'archive': prefix+'/archive.json',
              'manifest': manifest_storage['path'],
              'manifest_storage': manifest_storage['storage'],
              'manifest_sha256': manifest_storage['sha256'],
              'results': prefix+'/README_ZH.md',
              'release_url': release['html_url'],
              'verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    files['results/paper3/round-%02d/LATEST.json'%number] = upload_json_bytes(latest)
    files['results/paper3/round-%02d/README_ZH.md'%number] = (
        '# 第%d轮结果\n\n完整输出已上传：**%d/%d**。\n\n'
        '- [最新完整结果](final/%s/README_ZH.md)\n- [机器可读最新索引](LATEST.json)\n'
        '- [论文分析表格与报告](PAPER_DATA_ZH.md)\n- [以前自动回传的快照](previous/)\n\n'
        '完整输出包括逐试次预测和checkpoint；以前监控回传只是异步汇总，两类分别保留。\n'
        ) % (number, packed['complete_tasks'], packed['total_tasks'], snapshot)
    round_readme = 'results/paper3/round-%02d/README_ZH.md'%number
    files[round_readme] = files[round_readme].encode()
    emit('GIT_INDEX_UPLOADING', round=number, files=len(files))
    commit = gh.git_publish(files, 'Archive completed Paper3 round %d output %s'%(number, snapshot))
    latest['commit'] = commit
    upload_atomic(Path(work)/('round-%02d-UPLOADED.json'%number), latest)
    emit('ROUND_UPLOAD_COMPLETE', round=number, complete_tasks=packed['complete_tasks'],
                total_tasks=packed['total_tasks'], commit=commit, release_url=release['html_url'])
    return latest


def output_directory(root, number):
    if number not in ALLOWED_ROUNDS:
        raise ValueError('Full export only supports rounds 4, 5, 6 and 8')
    name = 'followup_20261025_r8' if number == 8 else 'followup_20261024_r%d' % number
    return Path(root) / 'runs' / name


def remote_json(gh, path):
    endpoint = ('repos/' + UPLOAD_REPOSITORY + '/contents/' +
                urllib.parse.quote(path, safe='/') + '?ref=' + UPLOAD_BRANCH)
    record = gh.api(endpoint, allow404=True)
    if record is None:
        return None
    if not isinstance(record, dict) or record.get('encoding') != 'base64':
        raise ValueError('GitHub result JSON has an unsupported representation')
    return json.loads(base64.b64decode(record['content']))


def remote_complete(gh, root, number):
    """Read only the small plan/pointers; never walk completed source outputs."""
    local_plan = (output_directory(root, number) / 'plan.json').read_bytes()
    plan = json.loads(local_plan)
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Source task plan is empty or invalid')
    latest = remote_json(gh, 'results/paper3/round-%02d/LATEST.json' % number)
    if not isinstance(latest, dict) or latest.get('status') != 'COMPLETE_OUTPUT_UPLOADED':
        return None
    if (latest.get('round') != number or latest.get('complete_tasks') != len(tasks)
            or latest.get('total_tasks') != len(tasks)):
        return None
    snapshot = latest.get('snapshot_id')
    if (not isinstance(snapshot, str) or len(snapshot) != 64
            or any(c not in '0123456789abcdef' for c in snapshot)):
        return None
    prefix = 'results/paper3/round-%02d/final/%s' % (number, snapshot)
    if latest.get('archive') != prefix + '/archive.json':
        return None
    archive = remote_json(gh, latest['archive'])
    if (not isinstance(archive, dict)
            or archive.get('status') != 'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED'
            or archive.get('round') != number or archive.get('snapshot_id') != snapshot
            or archive.get('complete_tasks') != len(tasks)
            or archive.get('total_tasks') != len(tasks) or not archive.get('parts')):
        return None
    # Comparing the Git blob identity avoids reading a possibly huge archive manifest.
    metadata = gh.api('repos/' + UPLOAD_REPOSITORY + '/contents/' +
        urllib.parse.quote(prefix + '/files/plan.json', safe='/') +
        '?ref=' + UPLOAD_BRANCH, allow404=True)
    blob = hashlib.sha1(('blob %d\0' % len(local_plan)).encode() + local_plan).hexdigest()
    if not isinstance(metadata, dict) or metadata.get('sha') != blob:
        return None
    return latest


@contextmanager
def exclusive_file(path, create=True):
    """Use flock only; never delete or replace a different uploader's lock."""
    path = Path(path)
    if not create and not path.exists():
        yield True
        return
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open('a+b' if create else 'rb')
    except FileNotFoundError:
        yield True
        return
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        handle.close()


class FullArchiveWorker:
    def __init__(self, root, work, rounds, gh, emit=upload_emit, *, lock_home=None,
                 prepare=None, publish=None):
        self.root = Path(root).expanduser().resolve()
        self.work = Path(work).expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        if any(n not in ALLOWED_ROUNDS for n in rounds):
            raise ValueError('Rounds 2, 3 and 7 already have archives and are excluded')
        # A busy legacy R6 uploader cannot hold up new R8 output.
        self.todo = [n for n in (4, 5, 8, 6) if n in set(rounds)]
        self.gh, self.emit = gh, emit
        self.prepare = prepare or prepare_round
        self.publish = publish or (lambda gh, packed, work: upload_remote_round(
            gh, packed, work, emit=self.event))
        self.completed, self.states = {}, {}
        home = Path(lock_home) if lock_home is not None else Path.home()
        root_key = hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        self.locks = home / '.cache/paper3-completion-export-locks' / root_key

    def event(self, event, **fields):
        self.emit(event, **fields)
        upload_atomic(self.work / 'progress.json', {
            'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            **fields})

    def status(self):
        value = {'status': 'COMPLETE' if not self.todo else 'UPLOADING_OR_WAITING',
                 'completed_rounds': sorted(self.completed), 'remaining_rounds': list(self.todo),
                 'round_states': self.states, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'training_jobs_changed': 0, 'source_outputs_modified': False}
        upload_atomic(self.work / 'status.json', value)
        return value

    def complete(self, number, receipt, reused=False):
        self.completed[number] = receipt
        self.todo.remove(number)
        self.states[str(number)] = 'REMOTE_ALREADY_COMPLETE' if reused else 'COMPLETE'
        upload_atomic(self.work / ('round-%02d-UPLOADED.json' % number), receipt)
        self.event('ROUND_REMOTE_ALREADY_COMPLETE' if reused else 'ROUND_CHANNEL_COMPLETE',
                   round=number, snapshot_id=receipt.get('snapshot_id'),
                   skipped_scientific_scan=reused, remaining_rounds=list(self.todo))

    def one(self, number):
        existing = remote_complete(self.gh, self.root, number)
        if existing is not None:
            self.complete(number, existing, reused=True)
            return
        with exclusive_file(self.locks / ('round-%02d-completion-export.lock' % number)) as locked:
            if not locked:
                self.states[str(number)] = 'WAITING_OTHER_FULL_EXPORT'
                self.event('ROUND_FULL_EXPORT_LOCK_BUSY', round=number)
                return
            # Recheck after acquiring the lock in case the prior owner just published.
            existing = remote_complete(self.gh, self.root, number)
            if existing is not None:
                self.complete(number, existing, reused=True)
                return
            legacy = (self.root / 'runs/followup_20261024_r6/runtime/three_jobs/'
                      'result-channels/full/worker.lock') if number == 6 else None
            with exclusive_file(legacy, create=False) if legacy else exclusive_file(
                    self.work / ('round-%02d-local.lock' % number)) as legacy_free:
                if not legacy_free:
                    self.states[str(number)] = 'WAITING_LEGACY_R6_FULL_UPLOAD'
                    self.event('LEGACY_R6_FULL_UPLOAD_ACTIVE', round=number,
                               legacy_lock=str(legacy), other_rounds_continue=True)
                    return
                self.states[str(number)] = 'PACKING'
                self.event('ROUND_SCAN_STARTED', round=number)
                packed = self.prepare(self.root, number, self.work, self.event)
                if packed.get('status') != 'READY':
                    self.states[str(number)] = 'WAITING_TASK_COMPLETION'
                    self.event('ROUND_NOT_COMPLETE_YET', round=number,
                               complete_tasks=packed.get('complete_tasks'), total_tasks=packed.get('total_tasks'))
                    return
                self.states[str(number)] = 'UPLOADING'
                self.event('ROUND_ARCHIVE_UPLOAD_STARTED', round=number,
                           snapshot_id=packed.get('snapshot_id'), parts=len(packed.get('parts', [])))
                latest = self.publish(self.gh, packed, self.work)
                self.complete(number, latest)

    def cycle(self):
        for number in list(self.todo):
            try:
                self.one(number)
            except Exception as error:
                if getattr(error, 'status', None) in (401, 403):
                    self.states[str(number)] = 'WAITING_CREDENTIAL_REFRESH'
                    self.event('FULL_CHANNEL_AUTH_REFRESH_REQUIRED', round=number,
                               error_type=type(error).__name__, http_status=error.status)
                    self.status()
                    raise
                self.states[str(number)] = 'RETRYING:' + type(error).__name__
                self.event('ROUND_UPLOAD_RETRY', round=number, error_type=type(error).__name__,
                           local_parts_preserved=True, other_training_unchanged=True)
            self.status()
        return self.status()

    def run(self, poll_seconds=60):
        self.status()
        while self.todo:
            self.cycle()
            if self.todo:
                self.event('FULL_CHANNEL_RETRY_WAIT', seconds=poll_seconds,
                           remaining_rounds=list(self.todo))
                time.sleep(poll_seconds)
        self.event('ALL_REQUESTED_FULL_ROUNDS_UPLOADED', rounds=sorted(self.completed),
                   training_jobs_changed=0)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='Full completed-round output upload, separate from fast paper tables.')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home() / 'paper3-completed-results-upload/full'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=ALLOWED_ROUNDS, default=list(ALLOWED_ROUNDS))
    parser.add_argument('--credentials')
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.print_help()
        return 0
    from paper3_result_transport import GitHub
    from paper_credentials import upload_discover_token
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with exclusive_file(work / 'worker.lock') as locked:
        if not locked:
            upload_emit('FULL_CHANNEL_ALREADY_RUNNING', work=str(work))
            return 0
        upload_atomic(work / 'worker.json', {'pid': os.getpid(), 'started_utc': time.time(),
                                           'rounds': args.rounds, 'channel': 'full'})
        try:
            os.nice(10)
        except OSError:
            pass
        upload_emit('FULL_CHANNEL_WORKER_STARTED', rounds=args.rounds, root=args.root,
                    work=str(work), legacy_r6_preserved=True, training_jobs_changed=0)
        while True:
            try:
                gh = GitHub(upload_discover_token(args.credentials), upload_emit,
                            repo=UPLOAD_REPOSITORY, branch=UPLOAD_BRANCH,
                            cache_path=work / 'git-blob-cache.json')
                gh.ensure_branch()
                return FullArchiveWorker(args.root, work, args.rounds, gh,
                                         emit=upload_emit).run(max(1, args.poll_seconds))
            except Exception as error:
                # Credentials may be supplied or repaired after this detached
                # process starts. Keep the worker lock/PID and rediscover them;
                # never log the error message, HTTP body, token or headers.
                fields = {'error_type': type(error).__name__, 'retry_seconds': 30,
                          'local_parts_preserved': True, 'training_jobs_changed': 0}
                if getattr(error, 'status', None) is not None:
                    fields['http_status'] = error.status
                upload_emit('FULL_CHANNEL_INITIALIZATION_OR_AUTH_RETRY', **fields)
                upload_atomic(work / 'status.json', {
                    'status': 'WAITING_CREDENTIAL_OR_GITHUB_RETRY',
                    'rounds': args.rounds,
                    'error_type': type(error).__name__,
                    'http_status': getattr(error, 'status', None),
                    'retry_seconds': 30, 'training_jobs_changed': 0})
                time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())


