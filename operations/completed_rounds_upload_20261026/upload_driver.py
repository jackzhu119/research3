"""Detached independent paper-evidence/full-archive delivery; no training control."""
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'transport')]
ROUNDS = (4, 5, 6, 8)


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      **fields}, ensure_ascii=False), flush=True)


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    os.replace(temporary, path)


def remote_paper_pointer(gh, number):
    path = 'results/paper3/round-%02d/PAPER_DATA_LATEST.json'%number
    value = gh.api(gh.prefix+'/contents/'+urllib.parse.quote(path, safe='/')+
                   '?ref='+urllib.parse.quote(gh.branch, safe='/'), allow404=True)
    if not value or not value.get('content'):
        return {}
    try:
        return json.loads(base64.b64decode(value['content']))
    except (ValueError, TypeError):
        return {}


def source_plan_unchanged(bundle, root, number):
    from collector import OUTPUT_NAMES, report_candidates
    import hashlib
    try:
        current = hashlib.sha256((Path(root)/'runs'/OUTPUT_NAMES[number]/'plan.json').read_bytes()).hexdigest()
        if current != bundle['metadata']['plan_sha256']:
            return False
        output = Path(root)/'runs'/OUTPUT_NAMES[number]
        reports = bundle['metadata'].get('reports', [])
        for report in reports:
            signature = report.get('source_signature')
            if not signature:
                return False
            st = (Path(root)/report['source_path']).stat()
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] != signature:
                return False
        inventory = sorted(p.relative_to(Path(root)).as_posix() for p in report_candidates(output))
        return inventory == bundle['metadata'].get('report_inventory')
    except (OSError, KeyError):
        return False


def paper_worker(args, work):
    from collector import collect_round
    from paper_credentials import upload_discover_token
    from transport.lossless_publish import VerifiedGitHub, publish_snapshot, stage_file
    try:
        os.nice(10)
    except OSError:
        pass
    done = set()
    todo = list(args.rounds)
    gh = None
    while todo:
        for number in list(todo):
            try:
                if gh is None:
                    gh = VerifiedGitHub(upload_discover_token(args.credentials), emit,
                        branch='paper3-results', cache_path=work/'git-blobs.json')
                round_work = work/('round-%02d'%number)
                round_work.mkdir(parents=True, exist_ok=True)
                saved = read(round_work/'bundle.json')
                bundle = None
                reused_bundle = False
                if (saved.get('paths') and saved.get('source_base')
                        and saved.get('metadata', {}).get('collection_status')=='COMPLETE'
                        and source_plan_unchanged(saved, args.root, number)
                        and all((Path(saved['source_base'])/path).is_file() for path in saved.get('paths', []))):
                    bundle = saved
                    reused_bundle = True
                    emit('PAPER_FROZEN_COLLECTION_REUSED', round=number)
                if bundle is None:
                    emit('PAPER_COLLECTION_STARTED', round=number,
                         recursive_trial_scan=False, aggregate_statistics_recomputed=False)
                    bundle = collect_round(args.root, number, round_work/'collections', emit=emit)
                    bundle['source_base'] = str(bundle['source_base'])
                    atomic(round_work/'bundle.json', bundle)
                previous = read(round_work/('round-%02d-PAPER-UPLOADED.json'%number))
                if (reused_bundle and previous.get('status')=='PAPER_DATA_FILES_VERIFIED'
                        and previous.get('collection_status')=='COMPLETE'
                        and not bundle['metadata'].get('required_missing')
                        and not bundle['metadata'].get('required_warnings')):
                    remote = remote_paper_pointer(gh, number)
                    if (remote.get('snapshot_id')==previous.get('snapshot_id')
                            and remote.get('status')=='PAPER_DATA_FILES_VERIFIED'):
                        emit('PAPER_PREVIOUS_VERIFIED_UPLOAD_REUSED', round=number)
                        todo.remove(number)
                        done.add(number)
                        continue
                files = []
                for index, path in enumerate(bundle['paths'], 1):
                    files.append(stage_file(bundle['source_base'], path, round_work/'source-cache', notify=emit))
                    atomic(work/'status.json', {'status': 'STAGING_ANALYSIS_EVIDENCE', 'round': number,
                        'staged_files': index, 'total_files': len(bundle['paths']),
                        'completed_rounds': sorted(done), 'remaining_rounds': todo,
                        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
                receipt = publish_snapshot(gh, number, files, round_work, metadata=bundle['metadata'])
                if (receipt.get('status')=='PAPER_DATA_FILES_VERIFIED'
                        and receipt.get('collection_status')=='COMPLETE'
                        and not bundle['metadata'].get('required_missing')
                        and not bundle['metadata'].get('required_warnings')):
                    todo.remove(number)
                    done.add(number)
                    emit('PAPER_ROUND_COMPLETE', round=number, commit=receipt.get('commit'),
                         tasks=bundle['metadata'].get('complete_tasks'),
                         condition_summaries=bundle['metadata'].get('captured_condition_summaries'))
                else:
                    emit('PAPER_PARTIAL_EVIDENCE_UPLOADED_RETRY_MISSING', round=number,
                         missing=bundle['metadata'].get('required_missing', []))
                atomic(work/'status.json', {'status': 'COMPLETE' if not todo else 'UPLOADING',
                    'completed_rounds': sorted(done), 'remaining_rounds': todo,
                    'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            except Exception as error:
                if getattr(error, 'status', None) in (401, 403):
                    gh = None
                emit('PAPER_CHANNEL_RETRY', round=number, error_type=type(error).__name__,
                     http_status=getattr(error, 'status', None), source_files_unchanged=True)
                atomic(work/'status.json', {'status': 'RETRYING', 'round': number,
                    'error_type': type(error).__name__, 'http_status': getattr(error, 'status', None),
                    'completed_rounds': sorted(done), 'remaining_rounds': todo})
        if todo:
            time.sleep(60)
    atomic(work/'status.json', {'status': 'COMPLETE', 'completed_rounds': sorted(done),
        'remaining_rounds': [], 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    emit('PAPER_ALL_COMPLETED_ROUNDS_UPLOADED', rounds=sorted(done))


def worker(args):
    work = Path(args.work_dir).expanduser().resolve()/args.channel
    work.mkdir(parents=True, exist_ok=True)
    with (work/'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            emit('UPLOAD_CHANNEL_ALREADY_RUNNING', channel=args.channel)
            return 0
        atomic(work/'worker.json', {'pid': os.getpid(), 'channel': args.channel,
            'rounds': args.rounds, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        if args.channel=='paper':
            paper_worker(args, work)
            return 0
        from archive.full_channel import main as full_main
        command = ['--worker', '--root', args.root, '--work-dir', str(work),
                   '--rounds', *map(str, args.rounds)]
        if args.credentials:
            command += ['--credentials', args.credentials]
        return full_main(command)


def start(args):
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    with (work/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX)
        for channel in ('paper', 'full'):
            folder = work/channel
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'driver.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    rows.append({'channel': channel, 'reused': True,
                                 'worker': read(folder/'worker.json'), 'log': str(folder/'upload.log')})
                    continue
                fcntl.flock(lock, fcntl.LOCK_UN)
                command = [sys.executable, '-I', '-S', '-B', '-u', str(HERE/'upload_driver.py'),
                    '--worker', '--channel', channel, '--root', args.root,
                    '--work-dir', str(work), '--rounds', *map(str, args.rounds)]
                if args.credentials:
                    command += ['--credentials', args.credentials]
                with (folder/'upload.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True, close_fds=True)
                # Wait only for the child's local startup acknowledgement,
                # never for credentials, GitHub, source scanning or transfer.
                acknowledged = False
                for _ in range(30):
                    info = read(folder/'worker.json')
                    if info.get('pid')==child.pid:
                        acknowledged = True
                        break
                    if child.poll() is not None:
                        break
                    time.sleep(.1)
                rows.append({'channel': channel, 'pid': child.pid, 'acknowledged': acknowledged,
                    'reused': False, 'log': str(folder/'upload.log'), 'status': str(folder/'status.json')})
    result = {'status': 'DETACHED_UPLOAD_WORKERS_STARTED', 'rounds': args.rounds,
        'processes': rows, 'work_directory': str(work), 'training_jobs_modified': 0,
        'terminal_can_close': all(r.get('reused') or r.get('acknowledged') for r in rows),
        'scope': 'Startup acknowledgement only; GitHub pointers acknowledge verified remote completion.'}
    atomic(work/'STARTED.json', result)
    emit('COMPLETED_ROUNDS_UPLOAD_BACKGROUND_STARTED', **result)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--start', action='store_true')
    modes.add_argument('--worker', action='store_true')
    parser.add_argument('--channel', choices=('paper', 'full'), default='paper')
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--work-dir', default=str(Path.home()/'paper3-completed-results-upload'))
    parser.add_argument('--rounds', nargs='+', type=int, choices=ROUNDS, default=list(ROUNDS))
    parser.add_argument('--credentials')
    args = parser.parse_args(argv)
    return start(args) if args.start else worker(args)


if __name__=='__main__':
    raise SystemExit(main())
