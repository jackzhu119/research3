#!/usr/bin/env python3
"""Read-only status panel: own-account Slurm queries, no data transfer or fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

STAGES = ('test', 'prepare', 'pilot', 'train', 'backup')
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_DIRECTORY_ENTRIES = 4096
MAX_QUEUE_BYTES = 256 * 1024
JOB_ID = re.compile(r'[0-9]+(?:_[0-9]+)?\Z')
QUEUE_FIELDS = '%i|%j|%T|%M|%C|%m|%b|%R|%E'
RUNNING_STATES = {'RUNNING', 'COMPLETING'}
DIAGNOSTIC_OBJECTS = {'main_archive', 'main_inputs', 'control_archive', 'control_inputs', 'ready'}
AWS_CODES = {'AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch', 'NoSuchBucket',
             'NoSuchKey', 'NotFound', '404', '403', 'RequestTimeout', 'SlowDown', 'ExpiredToken',
             'InvalidToken', 'AuthorizationHeaderMalformed', 'InvalidRequest', 'InternalError',
             'ServiceUnavailable', 'PermanentRedirect', 'OTHER_S3_ERROR'}
ERROR_CATEGORIES = {'NETWORK_DNS_OR_CONNECTION_FAILED', 'CONNECTION_TIMEOUT', 'RESPONSE_TIMEOUT',
    'TLS_CERTIFICATE_FAILED', 'PROXY_CONNECTION_FAILED', 'CONNECTION_CLOSED', 'HTTP_TRANSPORT_FAILED',
    'R2_CREDENTIAL_CONFIGURATION_MISSING', 'R2_CREDENTIAL_CONFIGURATION_INCOMPLETE',
    'S3_REQUEST_REJECTED', 'INVALID_HEAD_RESPONSE', 'UNCLASSIFIED_ERROR',
    'INVALID_LOCAL_DIAGNOSTIC_CONFIGURATION', 'existing_r2_configuration_required_on_network_allowed_node',
    'existing_environment_missing_minimal_r2_sdk', 'r2_https_account_endpoint_required',
    'interactive_hidden_input_required', 'invalid_hidden_credential'}


def safe_text(value, maximum=150):
    """Only bounded printable status characters; never render arbitrary JSON."""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return 'N/A'
    result = re.sub(r'[^A-Za-z0-9 _.,:;/=()+\-\[\]%@]', '?', str(value))[:maximum]
    return result or 'N/A'


def _safe(path):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('symlink_refused')
    return path


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def record(root, relative, *, limit=MAX_JSON_BYTES):
    """Read one bounded regular file, never follow links or trust JSON paths."""
    result = {'path': relative, 'value': None, 'mtime': None, 'error': None}
    try:
        path = _safe(root / relative)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('nonregular_or_oversized')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if len(raw) > limit or identity(before) != identity(after) or identity(after) != identity(path.lstat()):
            raise ValueError('changing_or_oversized')
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('object_required')
        result.update(value=value, mtime=before.st_mtime)
    except FileNotFoundError:
        result['error'] = 'missing'
    except (OSError, ValueError, UnicodeError):
        result['error'] = 'unavailable_or_unsafe'
    return result


def entries(root, relative):
    """List only the named control directory, without recursive discovery."""
    try:
        parent = _safe(root / relative)
        values = []
        for number, item in enumerate(parent.iterdir()):
            if number >= MAX_DIRECTORY_ENTRIES:
                break
            if item.is_symlink():
                continue
            values.append(item)
        return values
    except (OSError, ValueError):
        return []


def newest(paths, count=32):
    def modified(path):
        try:
            return path.stat(follow_symlinks=False).st_mtime
        except OSError:
            return -1
    return sorted(paths, key=modified, reverse=True)[:count]


def latest_data(root):
    candidates = []
    for path in newest(entries(root, '.paper3-hpc/data')):
        if re.fullmatch(r'attempt-[a-f0-9]+', path.name) and path.is_dir():
            current = record(root, '.paper3-hpc/data/' + path.name + '/receipt.json')
            if current['value'] is not None:
                candidates.append(current)
    return max(candidates, key=lambda item: item['mtime']) if candidates else None


def latest_diagnostic(root, now):
    candidates = []
    for path in newest(entries(root, '.paper3-hpc/diagnostics')):
        if not re.fullmatch(r'r2-head-[a-f0-9]+\.json', path.name) or not path.is_file():
            continue
        current = record(root, '.paper3-hpc/diagnostics/' + path.name, limit=128 * 1024)
        if (current['value'] or {}).get('schema_version') == 'paper3-hpc-r2-head-diagnostic-v1':
            candidates.append(current)
    if not candidates:
        return None
    chosen = max(candidates, key=lambda item: item['mtime'])
    value = chosen['value']
    def error_fields(error):
        error = error if isinstance(error, dict) else {}
        category = error.get('category')
        service = error.get('service_code')
        http = error.get('http_status')
        return {'category': category if category in ERROR_CATEGORIES else 'N/A',
                'service_code': service if service in AWS_CODES else 'N/A',
                'http_status': http if type(http) is int and 100 <= http <= 599 else None}
    rows = []
    values = value.get('objects') if isinstance(value.get('objects'), list) else []
    for row in values[:5]:
        if not isinstance(row, dict) or row.get('object') not in DIAGNOSTIC_OBJECTS:
            continue
        clean = {'object': row['object'], 'status': safe_text(row.get('status'), 70)}
        clean.update(error_fields(row.get('error')))
        if type(row.get('http_status')) is int and 100 <= row['http_status'] <= 599:
            clean['http_status'] = row['http_status']
        rows.append(clean)
    return {'status': safe_text(value.get('status'), 100), 'age_seconds': max(0, now - chosen['mtime']),
            'objects': rows, 'error': error_fields(value.get('error'))}


def queue(runner=subprocess.run):
    """squeue performs the only external command: own-account queue inspection."""
    command = ['squeue', '--user=' + str(os.getuid()), '--noheader', '--format=' + QUEUE_FIELDS]
    try:
        result = runner(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                        check=False, timeout=5)
        if result.returncode != 0 or len(result.stdout.encode('utf-8')) > MAX_QUEUE_BYTES:
            return {'status': 'unavailable', 'jobs': []}
        jobs = []
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split('|')]
            if len(fields) != 9 or not JOB_ID.fullmatch(fields[0]):
                continue
            stage = fields[1][len('paper3-'):] if fields[1].startswith('paper3-') else None
            if stage not in STAGES:
                continue
            jobs.append({'job_id': fields[0], 'stage': stage, 'state': safe_text(fields[2], 35),
                         'elapsed': safe_text(fields[3], 35), 'allocated_cpus': safe_text(fields[4], 12),
                         'requested_memory': safe_text(fields[5], 20), 'requested_gres': safe_text(fields[6], 50),
                         'reason_or_nodes': safe_text(fields[7]), 'dependency': safe_text(fields[8])})
        return {'status': 'observed', 'jobs': jobs}
    except (OSError, subprocess.TimeoutExpired, ValueError, UnicodeError):
        return {'status': 'unavailable', 'jobs': []}


def recent_submissions(root):
    found = {}
    for path in newest(entries(root, '.paper3-hpc/submissions'), 64):
        if not re.fullmatch(r'[0-9]+-[a-f0-9]+\.json', path.name) or not path.is_file():
            continue
        item = record(root, '.paper3-hpc/submissions/' + path.name, limit=128 * 1024)
        value = item['value'] or {}
        job_id = value.get('job_id')
        if (value.get('status') == 'SLURM_JOB_SUBMITTED_NOT_TRAINING_COMPLETE'
                and isinstance(job_id, str) and JOB_ID.fullmatch(job_id)
                and value.get('stage') in STAGES and path.name.startswith(job_id + '-')):
            found[job_id] = {'job_id': job_id, 'stage': value['stage'], 'mtime': item['mtime']}
    # Observed final/live job files are also actual allocation evidence, even
    # if a user did not copy the original submission receipt to this checkout.
    for path in newest(entries(root, '.paper3-hpc/jobs')):
        if not JOB_ID.fullmatch(path.name) or not path.is_dir() or path.name in found:
            continue
        for name in ('resources.json', 'resources.live.json'):
            item = record(root, '.paper3-hpc/jobs/' + path.name + '/' + name, limit=256 * 1024)
            value = item['value'] or {}
            allocation = value.get('allocation')
            if isinstance(allocation, dict) and allocation.get('job_id') == path.name and value.get('stage') in STAGES:
                found[path.name] = {'job_id': path.name, 'stage': value['stage'], 'mtime': item['mtime']}
                break
    return found


def resources(root, job_id, queue_state, *, now, stale_seconds):
    live = record(root, '.paper3-hpc/jobs/' + job_id + '/resources.live.json', limit=256 * 1024)
    final = record(root, '.paper3-hpc/jobs/' + job_id + '/resources.json', limit=256 * 1024)
    candidates = [item for item in (live, final) if isinstance(item['value'], dict)
                  and isinstance(item['value'].get('allocation'), dict)
                  and item['value']['allocation'].get('job_id') == job_id]
    if not candidates:
        return {'source': 'N/A', 'age_seconds': None, 'current': False,
                'cpu_cores': None, 'cpu_allocation_percent': None, 'rss_bytes': None,
                'peak_rss_bytes': None, 'owned_gpu_memory_mib': None, 'owned_gpu_peak_mib': None,
                'cpu_sample_status': 'N/A',
                'devices': [], 'actual_exit_code': None}
    chosen = max(candidates, key=lambda item: item['mtime'])
    value = chosen['value']
    final_chosen = chosen is final
    measured = value.get('measurements', {}) if final_chosen else value
    measured = measured if isinstance(measured, dict) else {}
    age = max(0, now - chosen['mtime'])
    current = not final_chosen and queue_state in RUNNING_STATES and age <= stale_seconds
    source = 'FINAL_HISTORY' if final_chosen else ('LIVE' if current else 'LAST_SAMPLE_NOT_CURRENT')
    gpu_available = measured.get('gpu_observation_available') is True
    allocation = value.get('allocation', {})
    cpu_cores = _number(measured.get('used_cpu_cores'))
    cpu_percent = _number(measured.get('allocated_CPU_util_percent'))
    if final_chosen:
        cpu_seconds, wall = _number(measured.get('cpu_seconds')), _number(measured.get('wall_seconds'))
        cpus = _number(allocation.get('cpus'))
        if cpu_seconds is not None and wall:
            cpu_cores = cpu_seconds / wall
            cpu_percent = cpu_cores / cpus * 100 if cpus else None
    devices = []
    rows = measured.get('owned_gpu_peaks' if final_chosen else 'owned_gpus', [])
    if isinstance(rows, list):
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            devices.append({'device_utilization_percent': _number(row.get('device_utilization_percent')),
                            'total_memory_mib': _number(row.get('total_memory_mib')),
                            'owned_process_memory_mib': _number(row.get('owned_process_memory_mib'))})
    peak_gpu = _number(measured.get('owned_process_gpu_peak_memory_mib'))
    if not gpu_available and not (peak_gpu is not None and peak_gpu > 0):
        peak_gpu = None
    return {'source': source, 'age_seconds': age, 'current': current,
            'cpu_cores': cpu_cores, 'cpu_allocation_percent': cpu_percent,
            'cpu_sample_status': safe_text(measured.get('cpu_sample_status'), 90),
            'rss_bytes': None if final_chosen else _number(measured.get('sampled_tree_rss_bytes')),
            'peak_rss_bytes': _number(measured.get('peak_rss_bytes', measured.get('sampled_tree_peak_rss_bytes'))),
            'owned_gpu_memory_mib': _number(measured.get('owned_process_gpu_memory_mib')) if gpu_available else None,
            'owned_gpu_peak_mib': peak_gpu, 'devices': devices,
            'actual_exit_code': value.get('actual_exit_code') if type(value.get('actual_exit_code')) is int else None}


def collect(root, *, runner=subprocess.run, now=None, interval=10):
    root = _safe(root)
    if not root.is_dir():
        raise ValueError('project_directory_missing')
    now = time.time() if now is None else now
    observed = queue(runner)
    submitted = recent_submissions(root)
    queued = {row['job_id']: row for row in observed['jobs']}
    all_jobs = dict(submitted)
    all_jobs.update(queued)
    jobs = []
    order = lambda item: tuple(map(int, item.replace('_', '.').split('.')))
    for job_id in sorted(all_jobs, key=order, reverse=True)[:3]:
        item = dict(all_jobs[job_id])
        item['in_current_queue'] = job_id in queued
        item['state'] = queued[job_id]['state'] if job_id in queued else (
            'NOT_IN_CURRENT_QUEUE' if observed['status'] == 'observed' else 'QUEUE_UNKNOWN')
        item['resources'] = resources(root, job_id, item['state'], now=now,
                                      stale_seconds=max(30, interval * 3))
        stage_receipt = record(root, '.paper3-hpc/jobs/' + job_id + '/stage-' + item['stage'] + '.json')
        value = stage_receipt['value'] or {}
        item['stage_returncode'] = value.get('returncode') if value.get('job_id') == job_id and type(value.get('returncode')) is int else None
        jobs.append(item)
    data = latest_data(root)
    data_summary = None
    if data:
        value = data['value']
        getters, groups = value.get('release_object_GETs', []), value.get('group_restore_receipts', [])
        getters = getters if isinstance(getters, list) else []
        groups = groups if isinstance(groups, list) else []
        data_summary = {'status': safe_text(value.get('status'), 70),
            'operation': safe_text(value.get('operation'), 30), 'error_code': safe_text(value.get('error_code'), 120),
            'age_seconds': max(0, now - data['mtime']),
            'full_GET_objects': sum(isinstance(row, dict) and row.get('status') == 'R2_FULL_GET_SHA_VERIFIED'
                                   and row.get('complete_body_read') is True for row in getters),
            'local_verified_objects': sum(isinstance(row, dict) and row.get('status') == 'LOCAL_VERIFIED_CACHE_HIT' for row in getters),
            'restored_groups': sum(isinstance(row, dict) and row.get('status') == 'restored_verified' for row in groups),
            'group_receipts': len(groups),
            'missing_cache_files': value.get('missing_cache_files') if type(value.get('missing_cache_files')) is int else None,
            'locally_SHA_verified': value.get('locally_SHA_verified') if type(value.get('locally_SHA_verified')) is int else None}
        transfer = value.get('active_transfer')
        if isinstance(transfer, dict):
            data_summary['transfer'] = {key: safe_text(transfer.get(key), 100)
                for key in ('phase', 'object', 'io_profile', 'updated_at_utc')}
            data_summary['transfer'].update({key: transfer.get(key) if type(transfer.get(key)) is int
                and transfer[key] >= 0 else None for key in
                ('attempt', 'attempt_limit', 'bytes_read', 'expected_bytes', 'group', 'groups_total')})
        failure = value.get('failure_detail')
        if isinstance(failure, dict):
            data_summary['failure'] = {key: safe_text(failure.get(key), 100)
                for key in ('phase', 'errno_name', 'service_code')}
            data_summary['failure'].update({key: failure.get(key) if type(failure.get(key)) is int else None
                for key in ('errno', 'http_status')})
    progress = record(root, 'runs/training_campaign_v1/campaign_progress.json')
    campaign = None
    if progress['value'] is not None:
        value = progress['value']
        active = value.get('active') if isinstance(value.get('active'), dict) else {}
        campaign = {'status': safe_text(value.get('status'), 100),
                    'kind': safe_text(active.get('kind'), 60), 'key': safe_text(active.get('key'), 130),
                    'age_seconds': max(0, now - progress['mtime'])}
    software = record(root, '.paper3-hpc/software_test_passed.json', limit=256 * 1024)
    software_observed = software['value'] is not None and software['value'].get('returncode') == 0
    return {'timestamp': now, 'interval': interval, 'queue_status': observed['status'],
            'queue_count': len(queued), 'queue_running_count': sum(row['state'] in RUNNING_STATES for row in queued.values()),
            'jobs': jobs, 'data': data_summary, 'campaign': campaign,
            'r2_diagnostic': latest_diagnostic(root, now), 'software_receipt_observed': software_observed}


def _amount(value, divisor=1, suffix='', digits=1):
    value = _number(value)
    return 'N/A' if value is None else f'{value / divisor:.{digits}f}{suffix}'


def next_step(snapshot):
    data, jobs = snapshot['data'], snapshot['jobs']
    active = [item for item in jobs if item['in_current_queue']]
    running = [item for item in active if item['state'] in RUNNING_STATES]
    if running:
        current = running[0]
        if current['stage'] == 'test':
            return '条件预测：test 成功后 prepare；数据与软件门通过后 train（包含原 pilot）。'
        if current['stage'] == 'prepare':
            return '条件预测：prepare 成功且软件/数据门仍通过后 train（包含原 pilot）。'
        if current['stage'] in {'pilot', 'train'}:
            return '控制器按原冻结门禁推进；安全暂停后由用户以原 checkpoint 恢复；监控不代表科研完成。'
        return '备份状态以原完整 GET/SHA 回执为准；监控不宣称上传成功。'
    if data and data['status'] in {'failed', 'blocked'}:
        return '数据门最近失败/阻塞：先按错误修复并重新核验；当前回执不能放行训练。'
    if active:
        current = min(active, key=lambda item: int(item['job_id'].split('_')[0]))
        return '等待 Slurm 调度/依赖：' + safe_text(current.get('dependency')) + '；失败的 afterok 前置不会放行后续作业。'
    if data and data['status'] == 'running':
        return '先确认数据进程及回执是否继续更新；最近 running 回执本身不能证明进程仍活跃。'
    if snapshot['queue_status'] != 'observed':
        return '先恢复 squeue 可读性以确认当前作业；不应据此重复提交。'
    if jobs and jobs[0]['stage_returncode'] not in (None, 0):
        return '最近 ' + jobs[0]['stage'] + ' 作业停止/失败；先查该作业阶段回执和门禁，再由用户恢复或重提。'
    if any(item['stage'] == 'prepare' and item['stage_returncode'] == 0 for item in jobs):
        return '条件预测：若当前代码、环境和数据门仍通过，可由用户提交 train；未自动启动。'
    if snapshot['software_receipt_observed']:
        return '条件预测：核当前代码/环境仍有效后提交 prepare，再按 afterok 进入 train。'
    return '条件预测：数据备齐后提交 test → prepare → train；当前未观察到排队或运行中的 paper3 作业。'


def render(snapshot):
    stamp = datetime.fromtimestamp(snapshot['timestamp'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [f'论文3只读监控 | {stamp} | 刷新 {snapshot["interval"]:g}s']
    data = snapshot['data']
    if snapshot['queue_running_count']:
        current = 'Slurm 已观察到 RUNNING/COMPLETING 作业；完成状态另看真实回执。'
    elif snapshot['queue_count']:
        current = 'Slurm 有待调度/依赖作业；尚不能视为训练已开始。'
    elif data and data['status'] in {'failed', 'blocked'}:
        current = 'STOPPED：数据步骤失败/阻塞；未观察到当前 paper3 Slurm 作业。' if snapshot['queue_status'] == 'observed' else 'STOPPED：数据步骤失败/阻塞；Slurm 当前状态未知。'
    elif data and data['status'] == 'running':
        current = 'UNCONFIRMED：仅有数据 running 回执，不能证明进程仍活跃。'
    elif snapshot['queue_status'] == 'unavailable':
        current = 'UNKNOWN：squeue 不可用，不能据历史回执判断作业在运行。'
    else:
        current = '未观察到当前 paper3 Slurm 作业。'
    lines.append('当前：' + current)
    if data:
        lines.append(f'数据最近回执：{data["operation"]} / {data["status"]} | 错误 {data["error_code"]} | {_amount(data["age_seconds"], suffix="s")} 前')
        lines.append(f'  固定对象已核验 {data["full_GET_objects"] + data["local_verified_objects"]}/5：完整 GET {data["full_GET_objects"]}；本地 SHA 命中 {data["local_verified_objects"]}；成功恢复组 {data["restored_groups"]}/{data["group_receipts"]}')
        lines.append(f'  本地 SHA 文件 {_amount(data["locally_SHA_verified"], digits=0)}；缺缓存 {_amount(data["missing_cache_files"], digits=0)}（回执计数，非实时下载速度）')
        transfer = data.get('transfer')
        if transfer:
            lines.append(f'  最近传输阶段 {transfer["phase"]} | 对象 {transfer["object"]} | 尝试 {_amount(transfer["attempt"],digits=0)} | 正文 {_amount(transfer["bytes_read"],1024**2,"MiB")} / {_amount(transfer["expected_bytes"],1024**2,"MiB")}；阶段进度不等于校验成功。')
            if transfer['group'] is not None:
                lines.append(f'  原权限恢复组 {transfer["group"]}/{transfer["groups_total"]} | {transfer["io_profile"]}')
        failure = data.get('failure')
        if failure:
            lines.append(f'  精确失败阶段 {failure["phase"]} | errno={_amount(failure["errno"],digits=0)} {failure["errno_name"]} | HTTP {_amount(failure["http_status"],digits=0)} {failure["service_code"]}')
    else:
        lines.append('数据：N/A（没有可读的最近数据回执）。')
    diagnostic = snapshot.get('r2_diagnostic')
    if diagnostic:
        lines.append(f'R2最近 HEAD 诊断：{diagnostic["status"]} | {_amount(diagnostic["age_seconds"],suffix="s")} 前；HEAD 不等于完整 GET/SHA。')
        errors = diagnostic['objects'] or [dict(object='configuration', status='BLOCKED', **diagnostic['error'])]
        for row in errors:
            lines.append(f'  {row["object"]}: {row["status"]} | HTTP {_amount(row["http_status"],digits=0)} | {row["category"]} / {row["service_code"]}')
    elif data and data['error_code'] == 'R2_GET_FAILED':
        lines.append('R2_GET_FAILED 尚不足判定网络、权限或密钥原因；未见 HEAD 诊断。')
    if data and data['error_code'] == 'R2_GET_FAILED':
        lines.append('可在获准联网节点手动运行 python hpc/r2_diagnose.py --network-host（--root 指向项目）；本监控不发起探针。')
    lines.append(f'Slurm 本账号 paper3 作业：{snapshot["queue_count"] if snapshot["queue_status"] == "observed" else "N/A"} | 显示最多最近 3 个实际作业；历史回执不等于运行。')
    for job in snapshot['jobs']:
        lines.append(f'作业 {job["job_id"]} {job["stage"]} | {job["state"]} | 已用 {job.get("elapsed", "N/A")}')
        if job['in_current_queue']:
            lines.append(f'  原因/节点 {job.get("reason_or_nodes", "N/A")} | 依赖 {job.get("dependency", "N/A")}')
        resource = job['resources']
        lines.append(f'  资源 {resource["source"]} | {_amount(resource["age_seconds"], suffix="s")} 前；未采样项为 N/A')
        cpu_label = ('平均占核' if resource['source'] == 'FINAL_HISTORY' else
                     '历史采样占核' if resource['source'] == 'LAST_SAMPLE_NOT_CURRENT' else '采样占核')
        lines.append(f'  CPU {cpu_label} {_amount(resource["cpu_cores"])} | 申请 CPU 利用率 {_amount(resource["cpu_allocation_percent"], suffix="%")} | RSS {_amount(resource["rss_bytes"], 1024**3, "GiB")} / 峰值 {_amount(resource["peak_rss_bytes"], 1024**3, "GiB")}')
        if resource['cpu_sample_status'] != 'N/A':
            lines.append('  CPU采样状态 ' + resource['cpu_sample_status'])
        gpu_label = ('本作业 GPU 历史采样显存' if resource['source'] == 'LAST_SAMPLE_NOT_CURRENT'
                     else '本作业 GPU 历史显存' if resource['source'] == 'FINAL_HISTORY' else '本作业 GPU 采样显存')
        lines.append(f'  {gpu_label} {_amount(resource["owned_gpu_memory_mib"], 1024, "GiB")} / 峰值 {_amount(resource["owned_gpu_peak_mib"], 1024, "GiB")}')
        device_label = ('设备历史峰值利用率' if resource['source'] == 'FINAL_HISTORY' else
                        '设备历史采样利用率' if resource['source'] == 'LAST_SAMPLE_NOT_CURRENT' else '设备采样利用率')
        for index, device in enumerate(resource['devices']):
            lines.append(f'  GPU观测{index}: {device_label} {_amount(device["device_utilization_percent"], suffix="%")}；本作业显存 {_amount(device["owned_process_memory_mib"],1024,"GiB")} / 设备容量 {_amount(device["total_memory_mib"],1024,"GiB")}（设备整体利用率可能含邻居作业）')
        if resource['actual_exit_code'] is not None:
            lines.append('  历史进程退出码 ' + str(resource['actual_exit_code']) + '，不代表科研或备份完成。')
    if snapshot['campaign']:
        campaign = snapshot['campaign']
        lines.append(f'主线最近记录：{campaign["status"]} | kind={campaign["kind"]} key={campaign["key"]} | {_amount(campaign["age_seconds"],suffix="s")} 前；不证明 fit 仍活跃。')
    lines.append('下一步：' + next_step(snapshot))
    lines.append('只读：不启动作业、下载、训练或备份；Ctrl+C 退出监控。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('--interval must be at least one second')
    try:
        while True:
            snapshot = collect(args.root, interval=args.interval)
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(render(snapshot), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError):
        print('监控不可用：项目路径或控制回执不可安全读取；没有启动任何操作。')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
