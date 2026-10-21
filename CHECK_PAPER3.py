#!/usr/bin/env python3
"""Read-only Paper3 activity check; no credentials, logs, fit or Pod actions."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import time

RELEASE = '603d299351fa6605aa0bc258bbbe2c421b2048e5becbeb33521e119a62f20fd7'
TOP = 'paper3_data_plan_' + RELEASE[:16]
CONTROLLERS = {'paper3_one_click.py', 'bootstrap_data_plan.py', 'run_training_campaign.py',
               'run_source_extension_downstream.py'}


def safe(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('SYMLINK_PATH_REFUSED')
    return path


def count_files(root):
    root = safe(root)
    pending, count = [root] if root.is_dir() else [], 0
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and not entry.name.endswith('.partial'):
                        count += 1
        except FileNotFoundError:
            continue
    return count


def controllers(workspace, proc=Path('/proc')):
    found = []
    for row in proc.iterdir():
        if not row.name.isdigit():
            continue
        try:
            args = [os.fsdecode(v) for v in (row / 'cmdline').read_bytes().split(b'\0') if v]
            cwd = (row / 'cwd').resolve(strict=True)
            paths = [Path(os.path.abspath(v if Path(v).is_absolute() else cwd / v)) for v in args[1:]]
            if not any(p.name in CONTROLLERS and p.is_relative_to(workspace) for p in paths):
                continue
            fields = (row / 'stat').read_text().rsplit(')', 1)[1].split()
            found.append({'pid': int(row.name), 'state': fields[0],
                          'cpu_ticks': int(fields[11]) + int(fields[12])})
        except (OSError, ValueError, IndexError):
            continue
    return sorted(found, key=lambda row: row['pid'])


def gpu_query(runner=subprocess.run):
    result = {}
    for name, query, width in [('devices', 'utilization.gpu,memory.used,memory.total', 3),
                                ('compute_processes', 'pid,used_gpu_memory', 2)]:
        try:
            output = runner(['nvidia-smi', '--query-' + ('gpu' if name == 'devices' else 'compute-apps')
                             + '=' + query, '--format=csv,noheader,nounits'],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
            rows = [line.split(',') for line in output.stdout.splitlines() if line.strip()]
            if output.returncode or any(len(row) != width for row in rows):
                result[name] = None
            else:
                result[name] = [[int(value.strip()) for value in row] for row in rows]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            result[name] = None
    return result


def collect(workspace, proc=Path('/proc'), runner=subprocess.run):
    workspace = safe(workspace)
    work = safe(workspace / 'Paper3_OneClick' / ('work_' + RELEASE[:16]))
    stage = '未见有效的本次固定进度记录'
    complete = cached = 0
    try:
        path = safe(work / 'RUN_PROGRESS.json')
        with path.open('rb') as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        state = json.loads(raw) if len(raw) <= 4 * 1024 * 1024 else {}
        if not isinstance(state, dict) or state.get('closure_sha256') != RELEASE:
            state = {}
        objects = state.get('r2_objects', {})
        for name in ('ready', 'main_inputs', 'control_inputs', 'main_archive', 'control_archive'):
            row = objects.get(name, {}) if isinstance(objects, dict) else {}
            if not isinstance(row, dict):
                continue
            complete += (row.get('status') == 'R2_FULL_GET_SHA_VERIFIED'
                         and row.get('complete_body_read') is True and row.get('get_attempted') is True)
            cached += row.get('status') == 'LOCAL_VERIFIED_CACHE_HIT' and row.get('get_attempted') is False
        if state.get('entry_issued_at_utc'):
            stage = '原部署/训练入口已有派发记录；实际阶段需结合下方活动判断'
        elif state.get('fixed_release_verified') is True:
            stage = '固定部署包已校验，等待原训练入口'
        elif complete + cached == 5:
            stage = '固定对象已取得，正在逐文件核验/解包'
        elif isinstance(state.get('github'), dict):
            stage = '已有 GitHub 推送记录，正在恢复固定对象'
        elif isinstance(state.get('github_target'), dict):
            stage = 'SDK 已有准备记录；可能正在重验、等待凭证或连接'
        elif state:
            stage = '离线 SDK 展开/校验'
    except (OSError, ValueError, TypeError):
        pass
    return {'observed_at': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
            'stage': stage, 'controllers': controllers(workspace, proc),
            'sdk_files': count_files(work / 'sdk_site'),
            'project_files': count_files(work / 'release' / TOP),
            'full_GET_objects': complete, 'cached_objects': cached, 'gpu': gpu_query(runner)}


def render(value, previous=None):
    rows = [f"[北京时间 {value['observed_at']}] {value['stage']}",
            f"SDK 文件 {value['sdk_files']}/2208；项目已落盘文件 {value['project_files']}",
            f"已有回执：固定对象完整 GET {value['full_GET_objects']}，本地 SHA 命中 {value['cached_objects']}"]
    jobs = value['controllers']
    rows.append('论文3控制进程：' + ('；'.join(f"PID {r['pid']} 状态 {r['state']} CPU累计ticks {r['cpu_ticks']}" for r in jobs)
                                      if jobs else '未发现；请保留原窗口查看返回信息'))
    if previous:
        old = {r['pid']: r['cpu_ticks'] for r in previous['controllers']}
        changes = value['project_files'] != previous['project_files'] or value['sdk_files'] != previous['sdk_files']
        changes |= any(old.get(r['pid']) != r['cpu_ticks'] for r in jobs)
        rows.append('本次观察：检测到文件数量或控制进程 CPU 活动。' if changes else
                    '本次观察：计数暂未变化；可能在 I/O、完整 SHA 校验或等待输入，不能据此认定失败。')
    for index, row in enumerate(value['gpu'].get('devices') or []):
        rows.append(f'GPU {index}：利用率 {row[0]}%，显存 {row[1]}/{row[2]} MiB')
    processes = value['gpu'].get('compute_processes')
    rows.append('GPU计算进程：' + ('；'.join(f'PID {r[0]}，{r[1]} MiB' for r in processes)
                                     if processes else '未查询到可用记录'))
    rows.append('此检查只读、不启动或中断任务；GPU占用与入口返回均不等于科学训练完成。')
    return '\n'.join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='论文3只读活动检查，不读取凭证或原始日志。')
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args(argv)
    if not 2 <= args.interval <= 60:
        parser.error('刷新间隔须为 2–60 秒')
    previous = None
    try:
        while True:
            value = collect(args.workspace)
            print(render(value, previous), flush=True)
            if args.once:
                return 0
            previous = value
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError):
        print('只读检查暂不可用：请确认目录权限及固定路径；原任务未改变。', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
