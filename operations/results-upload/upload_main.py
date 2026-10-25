"""Upload completed Paper3 rounds; never import an experiment or control Slurm."""
import argparse
import fcntl
import getpass
import glob
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

UPLOAD_REPOSITORY = 'jackzhu119/research3'
UPLOAD_BRANCH = 'paper3-results'


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


def upload_discover_token(explicit=None):
    for name in ('PAPER3_GITHUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN'):
        token = os.environ.get(name, '').strip()
        if token:
            return token
    candidates = []
    for candidate in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if candidate:
            candidates.append(Path(candidate).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round4-hpcgpu',
                 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        base = Path.home()/name
        candidates.append(base/'private/github.json')
        candidates.extend(sorted(base.glob('*/private/github.json')))
    for path in candidates:
        value = upload_read(path, {})
        token = value.get('token') if isinstance(value, dict) else None
        if isinstance(token, str) and token.strip():
            return token.strip()
    raise RuntimeError('未找到现有GitHub凭证；可用 --credentials 指定原有 private/github.json。凭证不进入归档。')


def upload_remote_round(gh, packed, work):
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
        upload_emit('PART_UPLOADING', round=number, part=index, total_parts=len(packed['parts']),
                    bytes=part['bytes'], name=part['name'])
        record = gh.upload_part(release, Path(part['path']), part['name'], part['sha256'])
        assets.append({'name': part['name'], 'bytes': part['bytes'], 'sha256': part['sha256'],
                       'asset_id': record['id'], 'download_url': record['url']})
        upload_emit('PART_VERIFIED', round=number, part=index, total_parts=len(packed['parts']))
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
    files[prefix+'/manifest.json'] = upload_json_bytes(manifest)
    files[prefix+'/TASKS.json'] = upload_json_bytes(manifest.get('tasks', []))
    archive = {
        'schema_version': 'paper3-complete-output-archive-v1', 'round': number,
        'snapshot_id': snapshot, 'status': 'ALL_PARTS_UPLOADED_AND_SHA256_VERIFIED',
        'complete_tasks': packed['complete_tasks'], 'total_tasks': packed['total_tasks'],
        'release_tag': tag, 'release_url': release['html_url'],
        'manifest_sha256': hashlib.sha256(files[prefix+'/manifest.json']).hexdigest(),
        'manifest_path': prefix+'/manifest.json', 'parts': assets,
        'part_order': 'Concatenate parts in this array order to reconstruct one tar.gz stream.',
        'existing_reports_recomputed': False,
        'large_browsable_files_available_in_archive': omitted_browsable,
        'scope': 'Complete declared-round scientific output; not raw EEG, shared source model or environment backup.',
    }
    files[prefix+'/archive.json'] = upload_json_bytes(archive)
    readme = ('# 第%d轮完整结果\n\n本次逐任务回执：**%d/%d 完成**。\n\n'
              '- [原计划与已有报告、CSV](files/)\n- [任务完成索引](TASKS.json)\n'
              '- [全部科学文件SHA256清单](manifest.json)\n- [完整分片下载索引](archive.json)\n'
              '- [GitHub Release：完整预测、验证轨迹与checkpoint](%s)\n\n'
              '分片按 archive.json 中的顺序拼接为 tar.gz；下载校验/还原工具见仓库 '
              '`operations/results-upload/RESTORE-PAPER3-RESULTS.py`。\n\n'
              '报告和CSV复制自服务器既有文件，不在上传时重新统计。'
              '若已有报告仍标PARTIAL，完整任务原始输出与本次完成索引均已包含，不能把旧报告计数当成缺文件。\n\n'
              '原始EEG、共享源模型、环境和运行控制文件不在本轮归档范围。\n') % (
                  number, packed['complete_tasks'], packed['total_tasks'], release['html_url'])
    files[prefix+'/README_ZH.md'] = readme.encode()
    latest = {'round': number, 'snapshot_id': snapshot,
              'status': 'COMPLETE_OUTPUT_UPLOADED', 'complete_tasks': packed['complete_tasks'],
              'total_tasks': packed['total_tasks'], 'archive': prefix+'/archive.json',
              'manifest': prefix+'/manifest.json', 'results': prefix+'/README_ZH.md',
              'release_url': release['html_url'],
              'verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    files['results/paper3/round-%02d/LATEST.json'%number] = upload_json_bytes(latest)
    files['results/paper3/round-%02d/README_ZH.md'%number] = (
        '# 第%d轮结果\n\n完整输出已上传：**%d/%d**。\n\n'
        '- [最新完整结果](final/%s/README_ZH.md)\n- [机器可读最新索引](LATEST.json)\n'
        '- [以前自动回传的快照](previous/)\n\n'
        '完整输出包括逐试次预测和checkpoint；以前监控回传只是异步汇总，两类分别保留。\n'
        ) % (number, packed['complete_tasks'], packed['total_tasks'], snapshot)
    round_readme = 'results/paper3/round-%02d/README_ZH.md'%number
    files[round_readme] = files[round_readme].encode()
    upload_emit('GIT_INDEX_UPLOADING', round=number, files=len(files))
    commit = gh.git_publish(files, 'Archive completed Paper3 round %d output %s'%(number, snapshot))
    latest['commit'] = commit
    upload_atomic(Path(work)/('round-%02d-UPLOADED.json'%number), latest)
    upload_emit('ROUND_UPLOAD_COMPLETE', round=number, complete_tasks=packed['complete_tasks'],
                total_tasks=packed['total_tasks'], commit=commit, release_url=release['html_url'])
    return latest


def upload_worker(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        upload_emit('UPLOAD_ALREADY_RUNNING', log=str(work/'upload.log'))
        return 0
    upload_atomic(work/'worker.json', {'pid': os.getpid(), 'started_utc': time.time(), 'rounds': args.rounds})
    try:
        os.nice(10)
    except OSError:
        pass
    upload_emit('UPLOAD_WORKER_STARTED', rounds=args.rounds, root=args.root, repository=UPLOAD_REPOSITORY,
                branch=UPLOAD_BRANCH, training_modified=False, log=str(work/'upload.log'))
    gh = GitHub(upload_discover_token(args.credentials), upload_emit,
                repo=UPLOAD_REPOSITORY, branch=UPLOAD_BRANCH, cache_path=work/'git-blob-cache.json')
    gh.ensure_branch()
    todo, completed = list(args.rounds), []
    while todo:
        for number in list(todo):
            try:
                upload_emit('ROUND_SCAN_STARTED', round=number)
                packed = prepare_round(Path(args.root), number, work, upload_emit)
                if packed['status'] != 'READY':
                    upload_emit('ROUND_NOT_COMPLETE_YET', round=number,
                                complete_tasks=packed.get('complete_tasks'), total_tasks=packed.get('total_tasks'))
                    continue
                previous = upload_read(work/('round-%02d-UPLOADED.json'%number), {})
                if previous.get('snapshot_id') == packed['snapshot_id']:
                    upload_emit('ROUND_PREVIOUS_UPLOAD_RECHECK', round=number)
                latest = upload_remote_round(gh, packed, work)
                completed.append(latest)
                todo.remove(number)
                upload_atomic(work/'status.json', {'status': 'UPLOADING' if todo else 'COMPLETE',
                    'completed_rounds': [r['round'] for r in completed], 'remaining_rounds': todo,
                    'completed': completed})
            except Exception as error:
                # Only error types are logged: exception messages may contain an HTTP body or credential.
                upload_emit('ROUND_UPLOAD_RETRY', round=number, error_type=type(error).__name__,
                            local_parts_preserved=True, other_training_unchanged=True)
                upload_atomic(work/'status.json', {'status': 'RETRYING', 'round': number,
                    'error_type': type(error).__name__, 'remaining_rounds': todo,
                    'completed_rounds': [r['round'] for r in completed]})
        if todo:
            upload_emit('RETRY_WAIT', seconds=60, remaining_rounds=todo)
            time.sleep(60)
    upload_emit('ALL_REQUESTED_ROUNDS_UPLOADED', rounds=args.rounds,
                index_url='https://github.com/'+UPLOAD_REPOSITORY+'/tree/'+UPLOAD_BRANCH+'/results/paper3')
    return 0


def upload_start(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work/'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        with (work/'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                upload_emit('UPLOAD_ALREADY_RUNNING', log=str(work/'upload.log'),
                            view_command='tail -n 40 '+str(work/'upload.log'))
                return 0
            fcntl.flock(lock, fcntl.LOCK_UN)
        # Confirm the existing credential is discoverable before detaching; never write its contents.
        upload_discover_token(args.credentials)
        command = [sys.executable, '-S', '-B', '-u', str(Path(__file__).resolve()), '--worker',
                   '--root', args.root, '--work-dir', str(work), '--rounds'] + list(map(str,args.rounds))
        if args.credentials:
            command += ['--credentials', args.credentials]
        with (work/'upload.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True)
        upload_emit('UPLOAD_BACKGROUND_STARTED', pid=child.pid, log=str(work/'upload.log'),
                    status=str(work/'status.json'), rounds=args.rounds,
                    view_command='tail -n 40 '+str(work/'upload.log'),
                    training_jobs_started_or_stopped=0)
    return 0


def upload_main():
    parser = argparse.ArgumentParser(description='Archive completed Paper3 rounds 2/3/7 without running experiments.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true', help='Detach uploader and print its log path.')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home()/'paper3-results-upload'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=(2,3,7), default=[2,3,7])
    parser.add_argument('--credentials')
    args = parser.parse_args()
    if not args.start and not args.worker:
        parser.print_help()
        return 0
    return upload_start(args) if args.start else upload_worker(args)


if __name__ == '__main__':
    raise SystemExit(upload_main())
