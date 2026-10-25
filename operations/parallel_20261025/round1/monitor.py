"""Independent, lightweight Round 1 monitoring; never controls training.

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
BRANCH = 'paper3-r1-progress'
STATUS_PATH = 'monitoring/paper3-round1/status.json'
README_PATH = 'monitoring/paper3-round1/README_ZH.md'
DEFAULT_OPERATIONAL = 'runs/training_campaign_parallel_v1'
MAX_JSON_BYTES = 8 * 1024 * 1024
COUNTER_KEYS = ('status', 'stage', 'phase', 'outer_update', 'total_updates',
    'outer_updates_completed', 'inner_updates_actual', 'confirmed_outer_updates_observed',
    'outer_attempts_dispatched', 'replayed_observed_outer_updates', 'validation_checkpoints',
    'session', 'sessions_total', 'budget', 'last_observed_trajectory_update',
    'observed_optimizer_calls_total', 'completed_updates', 'completed_epochs', 'epoch',
    'actual_neural_fits_complete', 'actual_classical_fits_complete', 'updated_utc')
TASK_KEYS = ('key', 'kind', 'method', 'arm', 'track', 'status', 'slot', 'job_id', 'error_type')
JOB_KEYS = ('slot', 'job_id', 'state', 'slurm_state', 'queue_reason', 'cpus', 'memory',
    'partition', 'updated_utc', 'attempt')
COMPLETE = {'COMPLETE', 'REUSED_COMPLETE', 'ALREADY_COMPLETE', 'FINISHED', 'REUSED'}
JOB_CONFIG_KEYS = ('method', 'arm', 'dimension', 'rank', 'radius', 'seed', 'phase', 'updates')
EXTENSION_ARMS = ('BASE4_PLUS_YANG', 'BASE4_PLUS_DREYER')


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


def query_queue(job_ids=(), runner=subprocess.run):
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
            if (job not in ids and not marker.startswith(('paper3r1p-', 'paper3r1-', 'paper3-r1-'))
                    and not name.startswith(('paper3r1p-', 'paper3r1-', 'paper3-r1-'))):
                continue
            rows.append(dict(job_id=job, state=state, name=name, partition=partition,
                             elapsed=elapsed, reason_or_node=reason, node=node, marker=marker))
        return {'status': 'OBSERVED', 'observed_utc': utc(), 'jobs': rows}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'UNAVAILABLE', 'error_type': type(error).__name__, 'jobs': []}


def track_groups(root, tasks, coordinator, now=None):
    """Separate the primary line from the two pre-existing extension pipelines.

    One extension task can encompass many native fits. Its wrapper-action count
    is deliberately not presented as a fit count or a wall-time percentage.
    """
    main_tasks = []
    arms = {arm: [] for arm in EXTENSION_ARMS}
    unassigned_extensions = []
    for task in tasks:
        arm = task.get('arm')
        if arm in arms:
            arms[arm].append(task)
        elif str(task.get('kind', '')).startswith('extension_'):
            unassigned_extensions.append(task)
        else:
            main_tasks.append(task)

    def summary(rows):
        return {'observed_actions': len(rows),
                'completed_observed_actions': sum(str(row.get('status', '')).upper() in COMPLETE for row in rows),
                'active_actions': [row for row in rows if str(row.get('status', '')).upper() not in COMPLETE]}

    mainline = summary(main_tasks)
    mainline['coordinator_declares_complete'] = coordinator.get('mainline_complete') is True
    mainline['count_scope'] = 'Observed primary actions, not all original work or equal-duration fits.'
    extensions = {}
    catalog_sections = ('source_f0', 'references', 'source_banks', 'meta_engineering', 'meta',
                        'profiles', 'validation', 'meta_selections', 'configurations')
    for arm, rows in arms.items():
        base = root/'runs/source_extension_downstream'/arm
        catalog_path = base/'catalog.json'
        catalog, error = read_json(catalog_path)
        if not catalog:
            catalog_path = base/'launch.json'
            catalog, launch_error = read_json(catalog_path)
            error = error or launch_error
        receipt_path = base/'pre_internal_development_freeze.json'
        receipt, receipt_error = read_json(receipt_path)
        if receipt.get('status') == 'SOURCE_EXTENSION_PRE_INTERNAL_DEVELOPMENT_FROZEN':
            arm_status = 'COMPLETE_RECEIPT_OBSERVED'
        elif any(str(row.get('status', '')).upper() in {'RUNNING', 'TRAINING', 'EVALUATING', 'ACTIVE'} for row in rows):
            arm_status = 'RUNNING'
        elif any(str(row.get('status', '')).upper() not in COMPLETE for row in rows):
            arm_status = 'PENDING_OR_PAUSED_ACTIONS'
        else:
            arm_status = 'INITIALIZED_OR_WAITING' if catalog else 'NOT_STARTED'
        entry = {'arm': arm, **summary(rows),
                 'status': arm_status,
                 'native_completed_artifact_counts': {section: len(catalog.get(section, {}))
                     for section in catalog_sections if isinstance(catalog.get(section), dict)},
                 'catalog_age_seconds': age(catalog_path, now),
                 'completion_receipt_status': receipt.get('status'),
                 'count_scope': 'Native catalog receipt counts; extension wrapper actions are not individual fits.'}
        if error or receipt_error:
            entry['read_error'] = error or receipt_error
        extensions[arm] = entry
    return mainline, extensions, unassigned_extensions


def resource_target(operational, progress):
    """Read only the new launch/progress contract, never the old handoff job."""
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
    return {'partition': partition, 'jobs': 3, 'gpus_per_job': 1,
            'cpus_per_job': 12, 'cpus_total': 36,
            'memory_per_job': ['43G', '43G', '42G'], 'memory_total': '128G',
            'gpu_model_requested_by_group': {'hpc_gpu': 'A100-PCIe-40GB',
                'hpc_gpu1': 'A100-SXM4-80GB'}.get(partition),
            'configuration_source': source, 'observed_gpu_hardware_claimed': False}


def collect(root, operational=None, *, runner=subprocess.run, now=None, include_queue=True):
    root = Path(root).absolute()
    operational = Path(operational) if operational else root / DEFAULT_OPERATIONAL
    if not operational.is_absolute():
        operational = root / operational
    now = time.time() if now is None else now
    path = operational / 'progress.json'
    progress, progress_error = read_json(path)
    # Startup observation can precede the first coordinator progress write.
    if not progress:
        progress, state_error = read_json(operational / 'state.json')
        progress_error = progress_error or state_error
        path = operational / 'state.json'
    original_path = root / 'runs/training_campaign_v1/campaign_progress.json'
    original, original_error = read_json(original_path)
    historical = collections.Counter(row.get('kind', 'unknown') for row in original.get('actions', [])
                                     if isinstance(row, dict))
    tasks_raw = progress.get('tasks', [])
    if isinstance(tasks_raw, dict):
        tasks_raw = list(tasks_raw.values())
    tasks = []
    active = []
    for task in tasks_raw if isinstance(tasks_raw, list) else []:
        if not isinstance(task, dict):
            continue
        row = scalars(task, TASK_KEYS)
        if isinstance(task.get('job'), dict):
            row['configuration'] = scalars(task['job'], JOB_CONFIG_KEYS)
        if isinstance(task.get('configuration'), dict):
            row['configuration'] = scalars(task['configuration'], JOB_CONFIG_KEYS)
        tasks.append(row)
        if str(task.get('status', '')).upper() not in COMPLETE:
            active.append((task, row))
    # Three live workers are typical. Bound incidental reads even if a plan is large.
    active.sort(key=lambda pair: (str(pair[0].get('status', '')).upper() not in
        {'RUNNING', 'TRAINING', 'EVALUATING', 'STARTING', 'ACTIVE'}, str(pair[0].get('key', ''))))
    for task, row in active[:12]:
        row['observations'] = progress_for_task(root, task, now)
    mainline, extension_arms, unassigned_extensions = track_groups(root, tasks, progress, now)
    jobs_raw = progress.get('jobs', [])
    if isinstance(jobs_raw, dict):
        jobs_raw = list(jobs_raw.values())
    jobs = [scalars(job, JOB_KEYS) for job in jobs_raw if isinstance(job, dict)] if isinstance(jobs_raw, list) else []
    job_ids = [job['job_id'] for job in jobs if job.get('job_id') is not None]
    snapshot = {'schema_version': 'paper3-round1-parallel-monitor-v1',
        'experiment': 'paper3-original-campaign-round1', 'round': 1,
        'updated_utc': utc(), 'sampled_unix': now,
        'coordinator': {'progress_path': str(path.relative_to(root)),
            'progress_age_seconds': age(path, now),
            **scalars(progress, ('status', 'phase', 'updated_utc', 'partition', 'max_parallel_jobs'))},
        'parallel_plan_total_tasks': len(tasks),
        'parallel_plan_complete_tasks': sum(str(t.get('status', '')).upper() in COMPLETE for t in tasks),
        'completed_by_kind': progress.get('completed_by_kind', {}),
        'planned_counts': progress.get('planned_counts', {}),
        'tasks': tasks, 'jobs': jobs,
        'mainline': mainline, 'extension_arms': extension_arms,
        'unassigned_extension_actions': unassigned_extensions,
        'mainline_complete': progress.get('mainline_complete') is True,
        'all_original_complete': (progress.get('all_original_complete') is True
                                  or progress.get('all_complete') is True),
        'historical_campaign': {'status': original.get('status'),
            'progress_age_seconds': age(original_path, now),
            'completed_stage_counts': dict(historical),
            'active_is_historical_after_handoff': True},
        'scientific_completion_claimed': (progress.get('scientific_completion_claimed') is True
            and (progress.get('all_original_complete') is True or progress.get('all_complete') is True)),
        'interpretation': 'Live operational observations; cached TRAINING is not proof of a live Slurm job.',
        'report_recomputed': False, 'trial_predictions_read': False,
        'aggregate_results_destination': 'paper3-results:results/paper3/round-01',
        'resources_target': resource_target(operational, progress)}
    if progress_error:
        snapshot['coordinator']['read_error'] = progress_error
    if original_error:
        snapshot['historical_campaign']['read_error'] = original_error
    for name in ('completed_by_kind', 'planned_counts'):
        if not isinstance(snapshot[name], dict):
            snapshot[name] = {}
        else:
            snapshot[name] = {str(k): v for k, v in snapshot[name].items() if isinstance(v, (int, float))}
    source_path = root / 'runs/source_training_v2/source_training_progress.json'
    source, source_error = read_json(source_path)
    classical_path = root / 'runs/source_classical_v2/receipt.json'
    classical, classical_error = read_json(classical_path)
    snapshot['source_neural'] = {**scalars(source, ('status', 'actual_neural_fits_complete')),
                                'age_seconds': age(source_path, now)}
    snapshot['source_classical'] = {**scalars(classical, ('status',)),
                                    'age_seconds': age(classical_path, now)}
    if include_queue:
        snapshot['live_queue'] = query_queue(job_ids, runner=runner)
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
    """Fixed Round 1 destination, bounded request retries, never a training dependency."""
    def __init__(self, token, *, opener=None):
        if not isinstance(token, str) or not token:
            raise ValueError('Missing GitHub token')
        self.token = token
        self.opener = opener or urllib.request.build_opener(
            _ProgressRedirect(), urllib.request.HTTPSHandler(context=ssl_context()))
        self.branch_ready = False
        self.readme_ready = False

    def api(self, path, value=None, method=None, allow404=False):
        url = 'https://api.github.com/repos/'+REPOSITORY + ('/'+path if path else '')
        request = urllib.request.Request(url,
            data=json.dumps(value, allow_nan=False).encode() if value is not None else None,
            headers={'Authorization': 'Bearer '+self.token, 'User-Agent': 'paper3-round1-monitor',
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
        endpoint = 'git/ref/heads/'+BRANCH
        if self.api(endpoint, allow404=True) is None:
            default = self.api('')['default_branch']
            head = self.api('git/ref/heads/'+urllib.parse.quote(default, safe='/'))['object']['sha']
            try:
                self.api('git/refs', {'ref': 'refs/heads/'+BRANCH, 'sha': head}, 'POST')
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422):
                    raise
                self.api(endpoint)
        self.branch_ready = True

    def put_file(self, path, data, message):
        encoded = base64.b64encode(data).decode()
        for attempt in range(3):
            old = self.api('contents/'+path+'?ref='+BRANCH, allow404=True)
            body = {'message': message, 'branch': BRANCH, 'content': encoded}
            if old is not None:
                body['sha'] = old['sha']
            try:
                return self.api('contents/'+path, body, 'PUT')
            except urllib.error.HTTPError as error:
                if error.code not in (409, 422) or attempt == 2:
                    raise

    def publish(self, snapshot):
        self.ensure_branch()
        data = (json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2,
                           allow_nan=False)+'\n').encode()
        self.put_file(STATUS_PATH, data, 'Update Paper3 round-one parallel progress')
        if not self.readme_ready:
            readme = ('# 第一轮实时进度\n\n'
                '[最新轻量状态](status.json) · [按轮分类结果](https://github.com/'+REPOSITORY+
                '/tree/paper3-results/results/paper3/round-01)\n\n'
                '同一原始科学计划，三个独立单GPU作业：'+snapshot['resources_target']['partition']+'、每作业12CPU，'
                '内存43G/43G/42G，共36CPU/128G。\n\n'
                '首次启动即回传，之后约180秒一次；不重新统计逐试次结果。'
                '主线mainline与两条extension_arms分栏显示；主线完成不等于全部原任务完成。'
                '两条既有扩展为BASE4_PLUS_YANG与BASE4_PLUS_DREYER，扩展包装动作可包含很多原生fit。'
                'coordinator记录与live_queue分别显示，运行/排队以live_queue本次观察为准。'
                '网络失败只影响回传，不控制训练。\n').encode()
            self.put_file(README_PATH, readme, 'Describe independent Paper3 round-one progress stream')
            self.readme_ready = True


def atomic_snapshot(path, snapshot):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def monitor_cycle(root, operational, credentials=None, *, client=None, runner=subprocess.run,
                  include_queue=True):
    snapshot = collect(root, operational, runner=runner, include_queue=include_queue)
    atomic_snapshot(Path(operational)/'monitor_snapshot.json', snapshot)
    if client is None:
        credentials = discover_credentials(credentials)
        value, error = read_json(credentials) if credentials else ({}, None)
        token = value.get('token')
        if not token:
            emit('R1_MONITOR_CREDENTIALS_UNAVAILABLE', training_controlled=False)
            return snapshot, None, False
        client = ProgressGitHub(token)
    try:
        client.publish(snapshot)
        emit('R1_GITHUB_SYNCED', updated_utc=snapshot['updated_utc'],
             parallel_plan_complete_tasks=snapshot['parallel_plan_complete_tasks'],
             branch=BRANCH, training_controlled=False)
        return snapshot, client, True
    except Exception as error:
        emit('R1_GITHUB_SYNC_RETRY', error_type=type(error).__name__, training_controlled=False)
        if isinstance(error, urllib.error.HTTPError) and error.code == 401:
            client = None  # Re-read a rotated private credential on the next observation cycle.
        return snapshot, client, False


def make_summary_dispatcher(root, round_number, operational, credentials):
    path = Path(__file__).with_name('auto_summary.py')
    spec = importlib.util.spec_from_file_location('paper3_automatic_result_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SummaryDispatcher(root, round_number, operational, credentials, emit=emit)


def run_monitor(root, operational=None, *, credentials=None, interval=180, once=False,
                local_only=False, runner=subprocess.run, stop=None, summary_dispatcher=None):
    root = Path(root).absolute()
    operational = Path(operational) if operational else root/DEFAULT_OPERATIONAL
    if not operational.is_absolute():
        operational = root/operational
    operational.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()
    if summary_dispatcher is None and not local_only:
        try:
            summary_dispatcher = make_summary_dispatcher(root, 1, operational, credentials)
        except Exception as error:
            emit('AUTO_SUMMARY_INIT_RETRY', error_type=type(error).__name__, training_controlled=False)
            summary_dispatcher = False
    with (operational/'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('R1_MONITOR_ALREADY_RUNNING', training_controlled=False)
            return 0
        client = None
        emit('R1_MONITOR_STARTED', root=str(root), operational=str(operational),
             branch=BRANCH, interval=interval, training_controlled=False)
        while not stop.is_set():
            started = time.monotonic()
            try:
                if local_only:
                    snapshot = collect(root, operational, runner=runner)
                    atomic_snapshot(operational/'monitor_snapshot.json', snapshot)
                    emit('R1_MONITOR_LOCAL_SNAPSHOT', training_controlled=False)
                else:
                    snapshot, client, synced = monitor_cycle(root, operational, credentials, client=client, runner=runner)
                    if synced and summary_dispatcher:
                        try:
                            summary_dispatcher.kick(snapshot)
                        except Exception as error:
                            emit('AUTO_SUMMARY_START_RETRY', error_type=type(error).__name__, training_controlled=False)
            except Exception as error:
                emit('R1_MONITOR_READ_RETRY', error_type=type(error).__name__, training_controlled=False)
            if once:
                return 0
            stop.wait(max(1, float(interval)-(time.monotonic()-started)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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
    return run_monitor(args.root, args.operational, credentials=args.credentials,
                       interval=args.interval, once=args.once, local_only=args.local_only, stop=stop)


if __name__ == '__main__':
    raise SystemExit(main())
