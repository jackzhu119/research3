"""Visible, bounded R1 diagnostics and future-action source-cache installation."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DEFAULT_ROOT = '/public/home/zhuziyuan/projects/paper3'


def emit(stage, **values):
    print(json.dumps(dict(event='PAPER3_VISIBLE_CHECK', stage=stage,
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values),
        ensure_ascii=False, default=str), flush=True)


def bounded_command(command, *, seconds=5, cap=2*1024*1024):
    # Only these newly created read/query children can be terminated on timeout.
    # No PID from a scientific/worker receipt is used for process control.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=out, stderr=err, close_fds=True)
            try:
                code = child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                child.kill()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                code = 124
            out.seek(0); data = out.read(cap+1)
            err.seek(0); message = err.read(1000).decode('utf-8','replace')
            return dict(code=code, data=data[:cap], oversized=len(data)>cap,
                        detail=message)
        except OSError as error:
            return dict(code=-1, data=b'', oversized=False, detail=type(error).__name__)


def read_file(path, roots, *, tail=False, cap=2*1024*1024):
    emit('READ_START', path=str(path))
    reader = """import os,sys
p=os.path.realpath(sys.argv[1]); roots=[os.path.realpath(x) for x in sys.argv[4:]]
if not any(os.path.commonpath([p,r])==r for r in roots):sys.exit(3)
n=int(sys.argv[3])
with open(p,'rb') as f:
 if sys.argv[2]=='tail':f.seek(max(0,os.fstat(f.fileno()).st_size-n))
 sys.stdout.buffer.write(f.read(n if sys.argv[2]=='tail' else n+1))
"""
    result = bounded_command([sys.executable,'-I','-S','-B','-c',reader,
        str(path),'tail' if tail else 'head',str(cap),*map(str,roots)],cap=cap)
    if result['code'] or result['oversized']:
        emit('READ_UNAVAILABLE', path=str(path), code=result['code'],
             oversized=result['oversized'])
        return dict(status='UNAVAILABLE', code=result['code'],
                    reason=result['detail'], oversized=result['oversized'])
    try:
        value = json.loads(result['data']) if not tail else result['data'].decode('utf-8','replace')
    except (ValueError,UnicodeError):
        value = None
    state = 'OK' if value is not None else 'INVALID_JSON'
    emit('READ_DONE', path=str(path), status=state, bytes=len(result['data']))
    return dict(status=state, value=value)


def value(row):
    result = row.get('value') if isinstance(row,dict) else None
    return result if isinstance(result,dict) else {}


def summarize_progress(row):
    v = value(row); tasks = v.get('tasks')
    tasks = list(tasks.values()) if isinstance(tasks,dict) else tasks
    valid = isinstance(tasks,list) and all(isinstance(t,dict)
        and isinstance(t.get('key'),str) and isinstance(t.get('status'),str) for t in tasks)
    return dict(status=v.get('status'), phase=v.get('phase'), updated_utc=v.get('updated_utc'),
        task_observation_valid=valid,
        counts=dict(collections.Counter(t['status'] for t in tasks)) if valid else None,
        completed_by_kind=v.get('completed_by_kind'), jobs=v.get('jobs'),
        active_or_retained_error_tasks=[t for t in tasks if t.get('status')=='RUNNING'
            or (t.get('error_type') and t.get('status')!='COMPLETE')] if valid else None)


def collect(root, home):
    root=Path(root).absolute(); home=Path(home).absolute()
    r1=root/'runs/training_campaign_parallel_v1'; r11=root/'runs/r11_campaign/runtime'
    work=home/'paper3-r1-full-repair/work'; roots=(root,work)
    report=dict(schema_version='paper3-r1-r11-visible-diagnostic-v1',
        sampled_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        root=str(root), scientific_jobs_signalled=0, jobs_submitted=0,
        plan_or_checkpoint_files_changed=0, report_only_memory_repair_applied=False)
    report['r1_progress']=summarize_progress(read_file(r1/'progress.json',roots))
    emit('R1_SUMMARY', **report['r1_progress'])
    report['workers']=[read_file(r1/'worker-status'/('slot-%d.json'%slot),roots) for slot in range(3)]
    report['r1_controller']=read_file(work/'controller-status.json',roots)
    report['r1_refresh_policy']=read_file(work/'refresh-hotfix-applied.json',roots)
    report['r1_claims']=read_file(r1/'claims.json',roots)
    backoff=read_file(r1/'action-backoff.json',roots)
    report['b5_backoff']=value(backoff).get('profile-B5')
    b5=root/'runs/training_campaign_v1/profiles/B5'
    for label,name in [('b5_progress','progress.json'),('b5_wall_budget','wall_budget.json')]:
        report[label]=read_file(b5/name,roots)
    folder=r1/'actions'/hashlib.sha256(b'profile-B5').hexdigest()
    report['b5_latest']=read_file(folder/'latest.json',roots)
    last=value(report['b5_latest']); path=last.get('log')
    if isinstance(path,str) and path:
        report['b5_last_kernel']=read_file(path,roots,tail=True,cap=18000)
    for slot,record in enumerate(report['workers']):
        w=value(record); key=w.get('action_key'); token=w.get('claim_token')
        if isinstance(key,str) and isinstance(token,str) and re.fullmatch('[a-fA-F0-9]{16,128}',token):
            attempt=r1/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token
            report['slot_%d_kernel'%slot]=read_file(attempt/'kernel.log',roots,tail=True,cap=14000)
    report['r11_scheduler']=read_file(r11/'scheduler-state.json',roots)
    report['r11_controller']=read_file(r11/'controller-status.json',roots)
    report['r11_errors']=read_file(r11/'controller-error.json',roots)
    ids=set()
    for state in [value(report['r11_scheduler']),value(report['r1_controller']).get('scheduler',{})]:
        if not isinstance(state,dict):continue
        slots=state.get('slots',{})
        for row in slots.values() if isinstance(slots,dict) else []:
            attempts=row.get('attempts',[]) if isinstance(row,dict) else []
            if attempts and isinstance(attempts[-1],dict):
                job=str(attempts[-1].get('job_id',''))
                if job.isdigit():ids.add(job)
        for row in state.get('jobs',[]) or []:
            if isinstance(row,dict) and str(row.get('job_id','')).isdigit():ids.add(str(row['job_id']))
    for row in report['workers']:
        job=str(value(row).get('job_id',''))
        if job.isdigit():ids.add(job)
    if ids:
        emit('QUEUE_READ_START',job_ids=sorted(ids))
        q=bounded_command(['squeue','-h','-j',','.join(sorted(ids)),'-o','%i|%T|%M|%N|%R'],seconds=8,cap=20000)
        report['live_queue']=dict(code=q['code'],output=q['data'].decode('utf-8','replace'),detail=q['detail'])
        emit('QUEUE_READ_DONE',**report['live_queue'])
    p=value(report['b5_progress'])
    emit('B5_SUMMARY', fit_attempts_dispatched=p.get('fit_attempts_dispatched'),
        fit_attempt_limit=p.get('fit_attempt_limit'), completed_fits=len(p.get('completed',{}))
            if isinstance(p.get('completed'),dict) else None,
        active=p.get('active'), last_finished=last, retained_backoff=report.get('b5_backoff'),
        budgets_or_history_reset=False)
    return report


def main(argv=None):
    emit('START',message='诊断已启动；逐项输出；不等待45秒；正常作业保持运行。')
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',type=Path,default=Path(DEFAULT_ROOT))
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--no-upload',action='store_true')
    args=parser.parse_args(argv)
    report=collect(args.root,args.home)
    folder=Path(tempfile.mkdtemp(prefix='paper3-r1-diagnostic-'))
    path=folder/'report.json'
    def save():
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
    save(); emit('LOCAL_REPORT_SAVED',path=str(path))
    if args.apply:
        emit('FUTURE_SELECTION_HOOK_START',current_children_untouched=True)
        try:
            from install_selection import apply
            report['selection_hook']=apply(args.root,args.home,package=HERE)
        except Exception as error:
            report['selection_hook']=dict(status='APPLY_NOT_CONFIRMED',error_type=type(error).__name__,reason=str(error)[:500])
        emit('FUTURE_SELECTION_HOOK_RESULT',**report['selection_hook']); save()
    if not args.no_upload:
        emit('DIAGNOSTIC_UPLOAD_START',training_controlled=False)
        try:
            from diagnostic_upload import publish
            report['diagnostic_upload']=publish(path.read_bytes(),args.home)
        except Exception as error:
            report['diagnostic_upload']=dict(status='UNAVAILABLE',error_type=type(error).__name__)
        emit('DIAGNOSTIC_UPLOAD_RESULT',**report['diagnostic_upload']); save()
    emit('FINISHED',report=str(path),scientific_jobs_signalled=0,jobs_submitted=0,
        note='B5是否需要额外恢复，以实际进度/退出回执为准；R11既有排队替代作业不重复提交。')
    return 2 if args.apply and report.get('selection_hook',{}).get('status')!='R1_FUTURE_STARTUP_CONTINUATION_INSTALLED' else 0


if __name__=='__main__':
    raise SystemExit(main())
