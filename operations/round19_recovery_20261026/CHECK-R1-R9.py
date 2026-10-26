"""Bounded, read-only school-side R1/R9 recovery diagnostics; standard library."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HPC_HOME = Path(os.environ.get('PAPER3_CHECK_HOME', '/public/home/zhuziyuan'))
ROOT = HPC_HOME/'projects/paper3'
KEYS = ('status', 'phase', 'stage', 'outer_update', 'outer_updates_completed',
        'confirmed_outer_updates_observed', 'total_updates', 'session', 'sessions_total',
        'budget', 'updated_utc', 'checks_completed', 'cache_hits', 'elapsed_seconds')


def out(label, value):
    print(label, json.dumps(value, ensure_ascii=False, separators=(',', ':')), flush=True)


def read(path):
    try:
        with Path(path).open('rb') as f:
            data = f.read(4*1024*1024+1)
        if len(data)>4*1024*1024:
            return {'_read_error': '文件超过4MB，仅跳过本项读取'}
        value = json.loads(data)
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        return {'_read_error': type(e).__name__}


def small(value):
    return {k: value[k] for k in KEYS+('_read_error',) if k in value}


def age(path):
    try:
        return round(max(0, time.time()-Path(path).stat().st_mtime), 1)
    except OSError:
        return None


def epoch(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError, AttributeError):
        return 0


def events(path, since=0):
    try:
        with Path(path).open('rb') as f:
            f.seek(0, 2)
            length = f.tell()
            f.seek(max(0, length-128*1024))
            text = f.read().decode('utf-8', 'replace')
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (isinstance(row, dict) and epoch(row.get('utc'))>=since
                and row.get('event') in ('R1_FILE_DIGEST_CACHE_ACTIVE',
                    'R1_FILE_DIGEST_CACHE_PROGRESS', 'R1_FILE_DIGEST_CACHE_SUMMARY',
                    'R1_INPUT_VERIFY_FINISHED', 'R1_INPUT_VERIFY_FAILED',
                    'R1_ACTION_RETURNED', 'R1_ACTION_FINISHED')):
            rows.append({k: row[k] for k in ('event', 'utc', 'phase', 'status',
                'hits', 'misses', 'bytes_reused', 'loaded_entries', 'check_elapsed_seconds',
                'error_type', 'returncode') if k in row})
    return rows[-6:]


print('只读检查；不下载、不导入Torch、不停止或提交作业。', flush=True)
out('检查时间北京时间', dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds'))
try:
    q = subprocess.run(['squeue', '--noheader', '--user', HPC_HOME.name,
        '--format=%i|%T|%j|%P|%M|%R'], capture_output=True, text=True, timeout=12)
    out('当前第一/九轮队列', [s for s in q.stdout.splitlines()
        if 'paper3r1p-' in s or 'paper3r9p-' in s] if q.returncode==0
        else {'不可读取': q.stderr[-200:]})
except (OSError, subprocess.TimeoutExpired) as e:
    out('队列不可读取', type(e).__name__)

print('\n第一轮', flush=True)
r1 = ROOT/'runs/training_campaign_parallel_v1'
a1 = read(HPC_HOME/'paper3-runtime-fixes/r1-digest-cache-v1/APPLIED.json')
out('安装回执', {k: a1.get(k) for k in ('applied_utc', 'helper_sha256', 'owned_stage_bootstrap_files', 'activation')})
for slot in range(3):
    row = read(r1/'worker-status'/('slot-%d.json'%slot))
    key, token = row.get('action_key'), row.get('claim_token')
    if not key or not token:
        out('槽%d'%slot, {k: row[k] for k in ('status', 'job_id', 'action_key') if k in row})
        continue
    attempt = r1/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/str(token)
    action = read(attempt/'action.json')
    claimed = epoch(action.get('claim', {}).get('claimed_utc'))
    receipt = row.get('expected_receipt') or action.get('action', {}).get('expected_receipt')
    out('槽%d 当前行动'%slot, {'job': row.get('job_id'), 'action': key,
        'kind': action.get('action', {}).get('kind'), 'child_pid': row.get('child_pid')})
    startup = read(attempt/'startup-progress.json')
    out('启动阶段', dict(small(startup), age_seconds=age(attempt/'startup-progress.json')))
    if receipt:
        base = (ROOT/receipt).parent
        for name in ('progress.json', 'execution_progress.json'):
            path = base/name
            value = read(path)
            if value:
                try:
                    older = bool(claimed and path.stat().st_mtime<claimed)
                except OSError:
                    older = False
                out('数值进度', dict(small(value), age_seconds=age(path),
                    older_than_current_action=older))
    recent = events(attempt/'kernel.log', max(claimed, epoch(a1.get('applied_utc'))))
    out('本行动修复后日志', recent or '当前行动尚无缓存采用证据；安装前启动的进程会继续旧流程')
index = r1/'digest-cache-v1/index.json'
out('缓存索引', {'exists': index.is_file(), 'age_seconds': age(index)})

print('\n第九轮', flush=True)
r9 = ROOT/'runs/followup_20261025_r9'
a9 = read(HPC_HOME/'paper3-runtime-fixes/r9-cumsum-v2/APPLIED.json')
out('安装回执', {k: a9.get(k) for k in ('applied_utc', 'runtime_hook_sha256', 'activation')})
progress = read(r9/'runtime/three_jobs/progress.json')
out('总进度', {k: progress.get(k) for k in ('status', 'complete_tasks', 'total_tasks', 'updated_utc')})
rows = progress.get('tasks', [])
for family in ('feature_reference', 'select'):
    family_rows = [r for r in rows if r.get('family')==family]
    out('调度记录 '+family+' 完成数', {'complete': sum(r.get('status')=='COMPLETE' for r in family_rows),
        'total': len(family_rows)})
new_reference = 0
for seed in (20261018, 20261019, 20261020):
    for mode in ('raw', 'ea'):
        key = 'feature-reference-%s-seed%d'%(mode, seed)
        folder = r9/key
        result, runtime, error = [read(folder/name) for name in ('result.json', 'runtime.json', 'error.json')]
        correction = (result if result.get('status')=='COMPLETE' else runtime).get('runtime_corrections', {})
        gp = next((d for d in correction.get('dependency_corrections', []) if d.get('module')=='paper3r9.gp'), {})
        adopted = (correction.get('version')=='paper3-r9-cumsum-fix-v2'
            and gp.get('executed_source_sha256')=='bb3dbd077632251d8fd34a43f83645c8f411f2954d8110c8d907da9e840e17ee')
        complete = result.get('status')=='COMPLETE'
        new_reference += int(adopted and complete)
        info = {'task': key, 'complete': complete, 'new_gp_fix_adopted': adopted,
            'progress': small(read(folder/'progress.json'))}
        if error and not complete:
            version = error.get('runtime_corrections', {}).get('version')
            info['error_record'] = {'scope': '新版进程错误记录，需结合当前进度' if version=='paper3-r9-cumsum-fix-v2'
                else '旧版进程错误记录，不据此判定本次修复失败',
                'version': version, 'age_seconds': age(folder/'error.json'),
                'reason': str(error.get('reason', error.get('_read_error', '')))[:650]}
        out('参考任务', info)
out('已用新版完成参考数', new_reference)
for slot in range(3):
    row = read(r9/'runtime/three_jobs'/('slot-%d.json'%slot))
    out('槽%d'%slot, {'job': row.get('job_id'), 'status': row.get('status'), 'task': row.get('task_key'),
        'progress': small(row.get('worker_progress') or {})})
print('CHECK_DONE；请贴回完整输出。旧错误、排队和正在校验均不能单独判定新进程失败。', flush=True)
