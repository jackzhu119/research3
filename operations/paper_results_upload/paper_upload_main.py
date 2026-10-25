"""Upload existing paper reports independently of full-output archives."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_credentials import upload_discover_token
from paper_results_collect import collect_round
from paper_report_views import render_files
from paper3_result_transport import GitHub


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields), ensure_ascii=False), flush=True)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_bytes(encoded(value))
    os.replace(temp, path)


def publish_round(gh, collected, work):
    number, snapshot = collected['round'], collected['snapshot_id']
    prefix = 'results/paper3/round-%02d/paper-data/%s' % (number, snapshot)
    upload_files, representations = render_files(collected['files'])
    files = {prefix + '/' + name: data for name, data in upload_files.items()}
    if not files:
        emit('PAPER_NO_FILES_YET', round=number)
        return None
    manifest = dict(collected['manifest'])
    manifest['github_representations'] = representations
    manifest['large_table_policy'] = 'Small files keep exact bytes; large CSV/JSON tables are split without recomputing statistics. Per-source SHA and semantic reconstruction metadata are retained. Full original bytes remain in the school archive.'
    files[prefix + '/manifest.json'] = encoded(manifest)
    reports = [r for r in collected['files'] if r['path'] != 'plan.json' and not r['path'].endswith('/result.json')]
    pointer = dict(schema_version='paper3-paper-data-pointer-v1', round=number,
        status='PAPER_REPORT_FILES_UPLOADED' if reports else 'PAPER_FILES_PARTIAL',
        snapshot_id=snapshot, manifest=prefix + '/manifest.json', index=prefix + '/README_ZH.md',
        files=len(collected['files']), bytes=sum(r['bytes'] for r in collected['files']),
        report_files=len(reports), collection_status=collected['status'],
        full_raw_results_uploaded=False, report_recomputed=False,
        raw_trial_predictions_included=False, model_checkpoints_included=False,
        task_receipts_included=manifest.get('task_counts', {}).get('task_receipts_included', False),
        scope='Exact existing report/CSV files, not a full model or trial-output backup; missing/stale reports remain explicit in manifest.')
    links = ['# 第%d轮 · 论文报告数据' % number, '',
        '已有报告与CSV按原始字节上传，没有重新训练或重新计算统计。', '',
        '[文件与校验清单](manifest.json)', '', '| 原文件 | 字节数 |', '|---|---:|']
    for source, view in zip(collected['files'], representations):
        linked = view.get('paths', [])
        if not linked:
            linked = [part['path'] for part in view.get('parts', [])]
        if not linked and view.get('path'):
            linked = [view['path']]
        targets = ' '.join('[part %d](%s)' % (i + 1, path) for i, path in enumerate(linked))
        links.append('| %s %s | %d |' % (source['path'], targets, source['bytes']))
    links += ['', '缺失、读取变化、超限和报告状态见manifest。报告可能早于任务回执；文件已上传不表示所有任务或所有统计已完成。',
        '', '本入口不含逐试次预测、模型断点或完整验证轨迹。CSV可以用于已有统计的检查与写作；重新设计参与者级配对统计可能仍需另导出相应会话指标。', '']
    files[prefix + '/README_ZH.md'] = '\n'.join(links).encode()
    files['results/paper3/round-%02d/PAPER_DATA_LATEST.json' % number] = encoded(pointer)
    files['results/paper3/round-%02d/PAPER_DATA_ZH.md' % number] = (
        '# 第%d轮 · 论文可读报告\n\n[最新已上传报告](paper-data/%s/README_ZH.md) · [机器索引](PAPER_DATA_LATEST.json)\n\n'
        '这里先传已有报告与完整CSV，不等待完整输出归档。范围与缺失项见快照manifest。\n' % (number, snapshot)).encode()
    commit = gh.git_publish(files, 'Publish paper report files for Paper3 round %d' % number)
    pointer['commit'] = commit
    atomic(Path(work) / ('round-%02d-PAPER-UPLOADED.json' % number), pointer)
    emit('PAPER_REPORTS_UPLOADED', round=number, files=len(collected['files']), bytes=pointer['bytes'],
        collection_status=collected['status'], commit=commit,
        url='https://github.com/jackzhu119/research3/blob/paper3-results/results/paper3/round-%02d/PAPER_DATA_ZH.md' % number)
    return pointer


def worker(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('PAPER_UPLOADER_ALREADY_RUNNING', log=str(work / 'upload.log'))
            return 0
        atomic(work / 'worker.json', dict(pid=os.getpid(), rounds=args.rounds, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        gh = GitHub(upload_discover_token(args.credentials), emit, cache_path=work / 'git-blobs.json')
        todo, done = list(args.rounds), []
        while todo:
            for number in list(todo):
                try:
                    emit('PAPER_REPORT_SCAN_STARTED', round=number, task_tree_scan=False, full_archive_untouched=True)
                    result = collect_round(Path(args.root), number, work / 'snapshots', emit, include_tasks=args.include_task_receipts,
                        max_report_bytes=64 * 1024**2, max_file_bytes=64 * 1024**2)
                    receipt = publish_round(gh, result, work)
                    if receipt:
                        done.append(number)
                        todo.remove(number)
                    atomic(work / 'status.json', dict(status='COMPLETE' if not todo else 'UPLOADING', completed_rounds=done, remaining_rounds=todo))
                except Exception as error:
                    emit('PAPER_UPLOAD_RETRY', round=number, error_type=type(error).__name__, other_rounds_continue=True)
                    atomic(work / 'status.json', dict(status='RETRYING', round=number, error_type=type(error).__name__, completed_rounds=done, remaining_rounds=todo))
            if todo:
                emit('PAPER_RETRY_WAIT', seconds=30, remaining_rounds=todo)
                time.sleep(30)
        emit('PAPER_ALL_REQUESTED_ROUNDS_PUBLISHED', rounds=done, full_archive_untouched=True, training_jobs_changed=0)
    return 0


def start(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / 'launch.lock').open('a') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        with (work / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                emit('PAPER_UPLOADER_ALREADY_RUNNING', log=str(work / 'upload.log'))
                return 0
            fcntl.flock(lock, fcntl.LOCK_UN)
        upload_discover_token(args.credentials)
        command = [sys.executable, '-I', '-S', '-B', '-u', str(Path(__file__).resolve()), '--worker', '--root', args.root, '--work-dir', str(work), '--rounds'] + [str(n) for n in args.rounds]
        if args.credentials:
            command += ['--credentials', args.credentials]
        if args.include_task_receipts:
            command += ['--include-task-receipts']
        with (work / 'upload.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        emit('PAPER_UPLOADER_STARTED', pid=child.pid, log=str(work / 'upload.log'), rounds=args.rounds, training_jobs_changed=0, full_archive_untouched=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home() / 'paper3-paper-data-upload'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=(2, 3, 7), default=[2, 3, 7])
    parser.add_argument('--credentials')
    parser.add_argument('--include-task-receipts', action='store_true', help='Optional extra result.json collection; reports-only is the default.')
    args = parser.parse_args()
    if args.start:
        return start(args)
    if args.worker:
        return worker(args)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
