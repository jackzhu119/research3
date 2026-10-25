"""Independent lightweight Round 8 three-job monitoring; never controls training.

Only declared progress/receipt JSON and the live Slurm queue are observed. This
module imports neither an experiment package nor torch, and never aggregates
trial predictions or rebuilds a result report.
"""
import argparse
import base64
import collections
import fcntl
import getpass
import importlib.util
import json
import os
from pathlib import Path
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

REPOSITORY = 'jackzhu119/research3'
DEFAULT_OPERATIONAL_NAME = 'runtime/three_jobs'


def destination(round_number):
    if round_number not in (8,):
        raise ValueError('Only round 8 is supported')
    return ('paper3-r%d-progress' % round_number,
            'monitoring/paper3-round%d/three-jobs-status.json' % round_number,
            'monitoring/paper3-round%d/THREE_JOBS_ZH.md' % round_number)


MAX_JSON_BYTES = 8 * 1024 * 1024
COUNTER_KEYS = ('status', 'stage', 'phase', 'outer_update', 'total_updates',
    'outer_updates_completed', 'inner_updates_actual', 'confirmed_outer_updates_observed',
    'outer_attempts_dispatched', 'replayed_observed_outer_updates', 'validation_checkpoints',
    'session', 'sessions_total', 'budget', 'last_observed_trajectory_update',
    'observed_optimizer_calls_total', 'completed_updates', 'completed_epochs', 'epoch',
    'actual_neural_fits_complete', 'actual_classical_fits_complete', 'updated_utc')
TASK_KEYS = ('index', 'key', 'family', 'kind', 'method', 'seed', 'status', 'slot', 'job_id', 'attempt', 'error_type')
JOB_KEYS = ('slot', 'job_id', 'status', 'state', 'slurm_state', 'queue_reason', 'cpus', 'memory',
    'partition', 'updated_utc', 'attempt')
COMPLETE = {'COMPLETE', 'COMPLETED', 'REUSED_COMPLETE', 'ALREADY_COMPLETE', 'FINISHED', 'REUSED'}
JOB_CONFIG_KEYS = ('method', 'dimension', 'rank', 'radius', 'seed', 'phase', 'updates')


def emit(event, **fields):
    print(json.dumps({'event': event, 'utc': utc(), **fields}, ensure_ascii=False,
                     allow_nan=False), flush=True)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read_json(path, limit=MAX_JSON_BYTES):
    try:
        with Path(path).open('rb') as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            return {}, 'JSON_TOO_LARGE'
        value = json.loads(data)
        return (value, None) if isinstance(value, dict) else ({}, 'JSON_NOT_OBJECT')
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as error:
        return {}, type(error).__name__


def age(path, now=None):
    try:
        return round(max(0, (time.time() if now is None else now) - Path(path).stat().st_mtime), 1)
    except OSError:
        return None


def scalars(value, keys):
    return {key: value[key] for key in keys if key in value and
            (value[key] is None or isinstance(value[key], (str, int, float, bool)))}


def safe_root_path(root, candidate):
    """Declared progress paths must stay inside the project, never private configs."""
    try:
        candidate = Path(candidate)
        if '..' in candidate.parts:
            return None
        path = candidate if candidate.is_absolute() else root / candidate
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return None
        if any(part in {'private', '.git', 'credentials'} for part in resolved.parts):
            return None
        return path
    except (OSError, TypeError, ValueError):
        return None


def progress_for_task(root, task, now=None):
    candidates = []
    if task.get('progress_path'):
        candidates.append(task['progress_path'])
    expected = task.get('expected_receipt')
    if expected:
        base = Path(expected).parent
        candidates.extend(str(base / name) for name in ('progress.json', 'execution_progress.json'))
    observed = []
    for candidate in dict.fromkeys(candidates):
        path = safe_root_path(root, candidate)
        if path is None:
            continue
        value, error = read_json(path)
        if value:
            observed.append({'path': str(path.relative_to(root)), 'age_seconds': age(path, now),
                             'values': scalars(value, COUNTER_KEYS)})
        elif error:
            observed.append({'path': str(path.relative_to(root)), 'read_error': error})
    return observed


def query_queue(job_ids=(), runner=subprocess.run, round_number=8):
    try:
        result = runner(['squeue', '--array', '--noheader', '--user', getpass.getuser(),
                         '--format=%i|%k|%T|%j|%P|%M|%R|%N'],
                        capture_output=True, text=True, timeout=15)
        if result.returncode:
            return {'status': 'UNAVAILABLE', 'error_type': 'SQUEUE_RETURNED_NONZERO', 'jobs': []}
        rows, ids = [], set(map(str, job_ids))
        for line in result.stdout.splitlines():
            columns = [value.strip() for value in line.split('|', 7)]
            if len(columns) != 8:
                continue
            job, marker, state, name, partition, elapsed, reason, node = columns
            prefix = 'paper3r%dp-' % round_number
            if job not in ids and not marker.startswith(prefix) and not name.startswith(prefix):
                continue
            rows.append(dict(job_id=job, state=state, name=name, partition=partition,
                             elapsed=elapsed, reason_or_node=reason, node=node, marker=marker))
        return {'status': 'OBSERVED', 'observed_utc': utc(), 'jobs': rows}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'UNAVAILABLE', 'error_type': type(error).__name__, 'jobs': []}


def resource_target(operational, progress):
    """Only the new three-job launch/progress can override the destination."""
    launch, _ = read_json(Path(operational)/'launch.json')
    candidates = [launch]
    candidates.extend(launch.get(key, {}) for key in ('requested_resources', 'resources', 'launch'))
    candidates.extend(progress.get(key, {}) for key in ('requested_resources', 'resources', 'launch'))
    candidates.append(progress)
    partition, source = 'hpc_gpu', 'new_default'
    for index, value in enumerate(candidates):
        if isinstance(value, dict) and isinstance(value.get('partition'), str) and value['partition']:
            partition = value['partition']
            source = 'new_launch' if index < 4 else 'new_progress'
            break
    return {'partition': partition, 'gres': 'gpu:1', 'slurm_jobs': 3,
        'gpus_per_job': 1, 'gpus_total': 3, 'nodes_per_job': 1,
        'cpus_per_job': 16, 'cpus_total': 48, 'memory_per_job': '64G', 'memory_total': '192G',
        'gpu_model_requested_by_group': {'hpc_gpu': 'A100-PCIe-40GB',
            'hpc_gpu1': 'A100-SXM4-80GB'}.get(partition),
        'configuration_source': source, 'observed_gpu_hardware_claimed': False}


def collect(root, round_number, operational=None, *, runner=subprocess.run, now=None, include_queue=True):
    branch, status_path, _ = destination(round_number)
    root = Path(root).absolute()
    output = root / ('runs/followup_20261025_r%d' % round_number)
    operational = Path(operational) if operational else output / DEFAULT_OPERATIONAL_NAME
    if not operational.is_absolute():
        operational = root / operational
    now = time.time() if now is None else now
    path = operational / 'progress.json'
    progress, error = read_json(path)
    tasks_raw = progress.get('tasks', [])
    if isinstance(tasks_raw, dict):
        tasks_raw = list(tasks_raw.values())
    tasks, active = [], []
    for task in tasks_raw if isinstance(tasks_raw, list) else []:
        if not isinstance(task, dict):
            continue
        row = scalars(task, TASK_KEYS)
        tasks.append(row)
        if str(task.get('status', '')).upper() not in COMPLETE:
            active.append((task, row))
    active.sort(key=lambda pair: (str(pair[0].get('status', '')).upper() not in
        {'RUNNING', 'TRAINING', 'EVALUATING', 'STARTING', 'ACTIVE'}, str(pair[0].get('key', ''))))
    for task, row in active[:12]:
        row['observations'] = progress_for_task(root, task, now)
    jobs_raw = progress.get('jobs', [])
    if isinstance(jobs_raw, dict):
        jobs_raw = list(jobs_raw.values())
    jobs = [scalars(job, JOB_KEYS) for job in jobs_raw if isinstance(job, dict)] if isinstance(jobs_raw, list) else []
    slots = {}
    for slot, raw in progress.get('slots', {}).items():
        if not isinstance(raw, dict):
            continue
        slots[str(slot)] = scalars(raw, ('status', 'task_key', 'job_id', 'pid', 'progress_age_seconds'))
        current = raw.get('worker_progress')
        if isinstance(current, dict):
            slots[str(slot)]['worker_progress'] = scalars(current, COUNTER_KEYS)
    done = sum(str(task.get('status', '')).upper() in COMPLETE for task in tasks)
    family_counts = {}
    for task in tasks:
        counts = family_counts.setdefault(task.get('family', task.get('kind', 'other')), {'complete': 0, 'total': 0})
        counts['total'] += 1
        counts['complete'] += str(task.get('status', '')).upper() in COMPLETE
    snapshot = {
        'schema_version': 'paper3-round8-three-jobs-monitor-v1',
        'experiment': 'paper3-round%d' % round_number, 'round': round_number,
        'updated_utc': utc(), 'sampled_unix': now,
        'status': progress.get('status', 'WAITING_FOR_THREE_JOB_CONTROLLER'),
        'phase': progress.get('phase'), 'recipe_sha256': progress.get('recipe_sha256'),
        'coordinator': {'progress_path': str(path.relative_to(root)),
                        'progress_age_seconds': age(path, now),
                        **scalars(progress, ('status', 'phase', 'updated_utc'))},
        'complete_tasks': progress.get('complete_tasks', done),
        'total_tasks': progress.get('total_tasks', len(tasks)),
        'task_count_scope': 'Three-job controller local COMPLETE receipt observations; no aggregate report rebuilt.',
        'tasks': tasks, 'jobs': jobs, 'gpu_slots': slots, 'families': family_counts,
        'allocation_mode': 'THREE_INDEPENDENT_SINGLE_GPU_JOBS',
        'requested_resources': resource_target(operational, progress),
        'resource_scope': 'Requested configuration; current Slurm queue observations are separate, not GPU utilization.',
        'scientific_completion_claimed': progress.get('scientific_completion_claimed') is True,
        'report_recomputed': False, 'trial_predictions_read': False,
        'progress_branch': branch, 'progress_path': status_path,
        'round_output': str(output),
        'aggregate_results_destination': 'paper3-results:results/paper3/round-%02d' % round_number,
        'interpretation': 'Round 8 independent three-job stream; optimizer progress and scheduler state are reported separately.'}
    if error:
        snapshot['coordinator']['read_error'] = error
    if include_queue:
        snapshot['live_queue'] = query_queue([j['job_id'] for j in jobs if j.get('job_id') is not None],
                                              runner=runner, round_number=round_number)
        current = {row['job_id']: row for row in snapshot['live_queue']['jobs']}
        for job in jobs:
            if str(job.get('job_id')) in current:
                live = current[str(job['job_id'])]
                job['live_state'] = live['state']
                job['live_reason_or_node'] = live['reason_or_node']
            elif snapshot['live_queue']['status'] == 'OBSERVED':
                job['live_state'] = 'NOT_IN_CURRENT_QUEUE'
    else:
        snapshot['live_queue'] = {'status': 'NOT_QUERIED', 'jobs': []}
    return snapshot


def discover_credentials(explicit=None):
    candidates = []
    for value in (explicit, os.environ.get('PAPER3_GITHUB_CREDENTIALS')):
        if value:
            candidates.append(Path(value).expanduser())
    candidates.append(Path.home()/'.config/paper3/github.json')
    for name in ('paper3-followup', 'paper3-round3', 'paper3-round4', 'paper3-round5',
                 'paper3-round6', 'paper3-round7', 'paper3-round4-hpcgpu',
                 'paper3-round5-hpcgpu', 'paper3-round6-hpcgpu'):
        base = Path.home()/name
        candidates += [base/'private/github.json', *sorted(base.glob('*/private/github.json'))]
    return next((path for path in candidates if path.is_file()), None)


def ssl_context():
    context = ssl.create_default_context()
    for prefix in {Path(sys.executable).absolute().parent.parent, Path(sys.executable).resolve().parent.parent}:
        for directory in prefix.glob('lib/python*/site-packages'):
            for suffix in ('botocore/cacert.pem', 'certifi/cacert.pem'):
                path = directory/suffix
                if path.is_file():
                    context.load_verify_locations(cafile=str(path))
    return context


class _ProgressRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Non-HTTPS API redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).hostname != 'api.github.com':
            for group in (redirected.headers, redirected.unredirected_hdrs):
                for key in list(group):
                    if key.lower() == 'authorization':
                        del group[key]
        return redirected


class ProgressGitHub:
    """Fixed per-round three-job destination, bounded request retries, never a training dependency."""
    def __init__(self, token, round_number, *, opener=None):
        if not isinstance(token, str) or not token:
            raise ValueError('Missing GitHub token')
        self.token = token
        self.round_number = round_number
        self.branch, self.status_path, self.readme_path = destination(round_number)
        self.opener = opener or urllib.request.build_opener(
            _ProgressRedirect(), urllib.request.HTTPSHandler(context=ssl_context()))
        self.branch_ready = False
        self.readme_ready = False

    def api(self, path, value=None, method=None, allow404=False):
        url = 'https://api.github.com/repos/'+REPOSITORY + ('/'+path if path else '')
        request = urllib.request.Request(url,
            data=json.dumps(value, allow_nan=False).encode() if value is not None else None,
            headers={'Authorization': 'Bearer '+self.token, 'User-Agent': 'paper3-round8-three-job-monitor',
                     'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json'}, method=method)
        try:
            with self.opener.open(request, timeout=20) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            if allow404 and error.code == 404:
                return None
            raise

    def ensure_branch(self):
        if self.branch_ready:
            return
        endpoint = 'git/ref/heads/'+self.branch
        if self.api(endpoint, allow404=True) is None:
            default = self.api('')['default_branch']
            head = self.api('git/ref/heads/'+urllib.parse.quote(default, safe='/'))['object']['sha']
            try:
                self.api('git/refs', {'ref': 'refs/heads/'+self.branch, 'sha': head}, 'POST')
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422):
                    raise
                self.api(endpoint)
        self.branch_ready = True

    def put_file(self, path, data, message):
        encoded = base64.b64encode(data).decode()
        for attempt in range(3):
            old = self.api('contents/'+path+'?ref='+self.branch, allow404=True)
            body = {'message': message, 'branch': self.branch, 'content': encoded}
            if old is not None:
                body['sha'] = old['sha']
            try:
                return self.api('contents/'+path, body, 'PUT')
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422) or attempt == 2:
                    raise

    def publish(self, snapshot):
        if snapshot.get('round') != self.round_number:
            raise ValueError('Snapshot round differs from monitor destination')
        self.ensure_branch()
        data = (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2,
                           allow_nan=False)+'\n').encode()
        self.put_file(self.status_path, data, 'Update Paper3 round %d three-job progress' % self.round_number)
        if not self.readme_ready:
            readme = ('# 第%d轮：三个独立单GPU作业实时进度\n\n'
                '[三作业轻量状态](three-jobs-status.json) · '
                '[按轮分类结果](https://github.com/%s/tree/paper3-results/results/paper3/round-%02d)\n\n'
                '当前方案为 %s、三个独立单GPU作业，每作业16CPU/64G；总3GPU/48CPU/192G。'
                '目标资源组显卡为%s，资源申请与实际队列状态分列。\n\n'
                '首次启动立即回传，此后约180秒；只读控制器完成计数和实际worker进度，'
                '不重算统计、不读取逐试次预测。网络失败只影响同步，不控制训练。\n\n'
                '第八轮使用独立目录、作业标记与状态分支，已有轮次继续各自运行。\n') % (
                    self.round_number, REPOSITORY, self.round_number,
                    snapshot['requested_resources']['partition'],
                    snapshot['requested_resources']['gpu_model_requested_by_group'])
            self.put_file(self.readme_path, readme.encode(), 'Document Paper3 round %d three-job progress stream' % self.round_number)
            self.readme_ready = True


def atomic_snapshot(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def monitor_cycle(root, round_number, operational, credentials=None, *, client=None,
                  runner=subprocess.run, include_queue=True):
    snapshot = collect(root, round_number, operational, runner=runner, include_queue=include_queue)
    atomic_snapshot(Path(operational)/'monitor_snapshot.json', snapshot)
    if client is None:
        credentials = discover_credentials(credentials)
        value, error = read_json(credentials) if credentials else ({}, None)
        token = value.get('token')
        if not token:
            emit('THREE_JOB_MONITOR_CREDENTIALS_UNAVAILABLE', round=round_number, training_controlled=False)
            return snapshot, None, False
        client = ProgressGitHub(token, round_number)
    try:
        client.publish(snapshot)
        emit('THREE_JOB_GITHUB_SYNCED', round=round_number, updated_utc=snapshot['updated_utc'],
             complete_tasks=snapshot['complete_tasks'], branch=destination(round_number)[0], training_controlled=False)
        return snapshot, client, True
    except Exception as error:
        emit('THREE_JOB_GITHUB_SYNC_RETRY', round=round_number, error_type=type(error).__name__, training_controlled=False)
        if isinstance(error, urllib.error.HTTPError) and error.code == 401:
            client = None
        return snapshot, client, False


def make_summary_dispatcher(root, round_number, operational, credentials):
    path = Path(__file__).with_name('auto_summary.py')
    spec = importlib.util.spec_from_file_location('paper3_automatic_result_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SummaryDispatcher(root, round_number, operational, credentials, emit=emit)


def run_monitor(root, round_number, operational=None, *, credentials=None, interval=180, once=False,
                local_only=False, runner=subprocess.run, stop=None, summary_dispatcher=None, report_dispatcher=None):
    root = Path(root).absolute()
    operational = Path(operational) if operational else root / ('runs/followup_20261025_r%d' % round_number) / DEFAULT_OPERATIONAL_NAME
    if not operational.is_absolute():
        operational = root/operational
    operational.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    if summary_dispatcher is None and not local_only:
        try:
            summary_dispatcher = make_summary_dispatcher(root, round_number, operational, credentials)
        except Exception as error:
            emit('AUTO_SUMMARY_INIT_RETRY', error_type=type(error).__name__, training_controlled=False)
            summary_dispatcher = False
    if report_dispatcher is None and not local_only:
        try:
            path=Path(__file__).with_name('report_dispatch.py')
            spec=importlib.util.spec_from_file_location('paper3_r8_report_dispatch',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            report_dispatcher=module.ReportDispatcher(root,operational,emit=emit)
        except Exception as error:
            emit('ROUND_8_REPORT_INIT_RETRY',error_type=type(error).__name__,training_controlled=False)
            report_dispatcher=False
    # Report and result synchronization are independent of science processes.
    with (operational/'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('THREE_JOB_MONITOR_ALREADY_RUNNING', round=round_number, training_controlled=False)
            return 0
        client = None
        emit('THREE_JOB_MONITOR_STARTED', round=round_number, operational=str(operational),
             branch=destination(round_number)[0], interval=interval, training_controlled=False)
        while not stop.is_set():
            started = time.monotonic()
            try:
                if local_only:
                    snapshot = collect(root, round_number, operational, runner=runner)
                    atomic_snapshot(operational/'monitor_snapshot.json', snapshot)
                    emit('THREE_JOB_MONITOR_LOCAL_SNAPSHOT', round=round_number, training_controlled=False)
                else:
                    snapshot, client, synced = monitor_cycle(root, round_number, operational, credentials, client=client, runner=runner)
                    if report_dispatcher:
                        report_dispatcher.kick(snapshot)
                    if synced and summary_dispatcher:
                        try:
                            summary_dispatcher.kick(snapshot)
                        except Exception as error:
                            emit('AUTO_SUMMARY_START_RETRY', error_type=type(error).__name__, training_controlled=False)
            except Exception as error:
                emit('THREE_JOB_MONITOR_READ_RETRY', round=round_number, error_type=type(error).__name__, training_controlled=False)
            if once:
                return 0
            stop.wait(max(1, float(interval)-(time.monotonic()-started)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=int, choices=(8,), required=True, dest='round_number')
    parser.add_argument('--root', default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--operational')
    parser.add_argument('--credentials')
    parser.add_argument('--interval', type=float, default=180)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args(argv)
    stop = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_args: stop.set())
    return run_monitor(args.root, args.round_number, args.operational, credentials=args.credentials,
                       interval=args.interval, once=args.once, local_only=args.local_only, stop=stop)


if __name__ == '__main__':
    raise SystemExit(main())
