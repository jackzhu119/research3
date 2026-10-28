#!/usr/bin/env python3
"""Bounded, read-only R1 snapshots; run with the installed Python standard library."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

JSON_LIMIT, LOG_LIMIT = 2*1024*1024, 12*1024
NUMERIC = frozenset(('outer_update', 'outer_updates_completed', 'inner_updates_actual',
    'episodes_completed', 'confirmed_outer_updates_observed', 'last_observed_trajectory_update',
    'observed_optimizer_calls_total'))
VERIFY_KINDS = frozenset(('strongest_control', 'freeze', 'development_selection', 'meta_selection'))
READ_CODE = '''import json,os,sys
p,mode,cap=sys.argv[1:];cap=int(cap)
try:
 with open(p,'rb') as f:
  s=os.fstat(f.fileno())
  if mode=='tail':f.seek(max(0,s.st_size-cap))
  data=f.read(cap+1)
 print(json.dumps(dict(size=s.st_size,mtime=s.st_mtime,uid=s.st_uid,truncated=len(data)>cap)))
 sys.stdout.flush();sys.stdout.buffer.write(data[:cap])
except OSError as e:print(json.dumps(dict(error=type(e).__name__)))
'''


def say(message):
    print(message, flush=True)


def read_file(path, tail=False, limit=JSON_LIMIT, json_expected=True):
    say('读取：'+str(path))
    try:
        out=subprocess.run([sys.executable,'-I','-S','-B','-c',READ_CODE,str(path),
            'tail' if tail else 'head',str(limit)],capture_output=True,timeout=4,
            stdin=subprocess.DEVNULL,check=False)
        header,sep,raw=out.stdout.partition(b'\n');record=json.loads(header)
        if out.returncode or not sep:record['error']='ReadProcessFailed'
        record['text']=raw.decode('utf-8',errors='replace')
        if json_expected and not tail and not record.get('error') and not record.get('truncated'):
            try:record['value']=json.loads(raw)
            except (ValueError,UnicodeError):record['error']='InvalidJSON'
        if record.get('truncated') and not tail:record['error']='JSONTooLarge'
        if record.get('error'):say('  无法读取：'+record['error'])
        return record
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        say('  无法读取：'+type(error).__name__)
        return dict(error=type(error).__name__)


def data(record):
    value=record.get('value')
    return value if isinstance(value,dict) and not record.get('error') else {}


def valid_job(value):
    return re.fullmatch('[1-9][0-9]*',str(value)) is not None


def age(value, now):
    try:
        stamp=float(value) if isinstance(value,(float,int)) else dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
        return max(0,round(now-stamp,1))
    except (ValueError,TypeError,AttributeError):return None


def last_event(text, name):
    for line in reversed(text.splitlines()):
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict) and value.get('event')==name:return value
    return {}


def numerical(worker):
    result={}
    for row in worker.get('action_progress',[]):
        if not isinstance(row,dict):continue
        values=row.get('values',{})
        if not isinstance(values,dict):continue
        for key in NUMERIC:
            value=values.get(key)
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                result[(str(row.get('file',''))+'|'+key)]=value
    return result


def query_queue(job_ids):
    if not job_ids:return dict(error='NoObservedJobIDs',jobs=[])
    say('读取 Slurm 队列（最多6秒）')
    try:
        out=subprocess.run(['squeue','-h','-j',','.join(sorted(job_ids)),
            '-o','%i|%T|%M|%N|%R'],capture_output=True,text=True,timeout=6,
            stdin=subprocess.DEVNULL,check=False)
        if out.returncode:return dict(error='squeue退出码'+str(out.returncode),jobs=[])
        jobs=[]
        for line in out.stdout.splitlines():
            parts=line.split('|',4)
            if len(parts)==5:jobs.append(dict(zip(('job_id','state','elapsed','node','reason'),parts)))
        return dict(jobs=jobs)
    except (OSError,subprocess.TimeoutExpired) as error:return dict(error=type(error).__name__,jobs=[])


def process_matches(record, reader=read_file):
    pid=record.get('pid');argv=record.get('argv')
    if not isinstance(pid,int) or pid<=1 or not isinstance(argv,list):return None
    cmd=reader(Path('/proc')/str(pid)/'cmdline',limit=65536,json_expected=False)
    stat=reader(Path('/proc')/str(pid)/'stat',limit=4096,json_expected=False)
    if cmd.get('error') not in (None,'InvalidJSON') or stat.get('error') not in (None,'InvalidJSON'):return None
    actual=cmd.get('text','').rstrip('\0').split('\0')
    fields=stat.get('text','').rsplit(')',1)[-1].split()
    if len(fields)<20:return None
    return (cmd.get('uid')==os.getuid() and actual==argv and fields[0] not in ('Z','X','x') and record.get('start_ticks') is not None
        and fields[19]==str(record['start_ticks']))


def snapshot(root, home, reader=read_file, queue=query_queue, now=None):
    now=time.time() if now is None else now;runtime=Path(root)/'runs/training_campaign_parallel_v1'
    work=Path(home)/'paper3-r1-full-repair/work'
    guard=data(reader(work/'memory-recovery-status.json'))
    progress=data(reader(runtime/'progress.json'));frontier=data(reader(runtime/'queue.json'))
    claims=data(reader(runtime/'claims.json'));record=data(reader(work/'memory-recovery-process.json'))
    workers=[];ids={str(j.get('job_id')) for j in progress.get('jobs',[])
        if isinstance(j,dict) and valid_job(j.get('job_id'))}
    for slot in range(3):
        worker=data(reader(runtime/'worker-status'/('slot-%d.json'%slot)))
        worker=dict(worker,slot=slot);job=str(worker.get('job_id',''))
        if valid_job(job):ids.add(job)
        worker['age_seconds']=age(worker.get('updated_at',worker.get('updated_utc')),now)
        worker['numerical']=numerical(worker);key=worker.get('action_key');token=worker.get('claim_token')
        claim=claims.get(key,{}) if isinstance(key,str) else {}
        # Read an attempt log only when the current durable claim binds its
        # slot, job, action key and opaque token. Never follow a supplied path.
        bound=(isinstance(claim,dict) and claim.get('slot')==slot and str(claim.get('job_id'))==job
            and claim.get('token')==token and isinstance(token,str) and re.fullmatch('[0-9a-f]{32}',token))
        worker['claim_bound']=bool(bound)
        if bound:
            path=runtime/'actions'/hashlib.sha256(key.encode()).hexdigest()/'attempts'/token/'kernel.log'
            log=reader(path,tail=True,limit=LOG_LIMIT)
            worker['log']=dict(text=log.get('text',''),mtime=log.get('mtime'),size=log.get('size'),error=log.get('error'))
            worker['cache_event']=last_event(log.get('text',''),'R1_FILE_DIGEST_CACHE_PROGRESS')
        else:worker['log']={};worker['cache_event']={}
        if valid_job(job):
            log=reader(runtime/'logs'/('slot-%d-%s.out'%(slot,job)),tail=True,limit=LOG_LIMIT)
            worker['slurm_tail']='\n'.join(log.get('text','').splitlines()[-3:])
        workers.append(worker)
    tasks=progress.get('tasks');tasks=list(tasks.values()) if isinstance(tasks,dict) else tasks
    tasks=tasks if isinstance(tasks,list) else None
    return dict(at=now,guard=guard,guard_age=age(guard.get('updated_utc'),now),
        dispatch_age=age(frontier.get('updated_utc',progress.get('updated_utc')),now),
        guardian_process_matches=process_matches(record,reader) if record else None,
        phase=frontier.get('phase',progress.get('phase')),progress=progress,queue=frontier,
        completed=sum(t.get('status')=='COMPLETE' for t in tasks if isinstance(t,dict)) if tasks is not None else None,
        total=len(tasks) if tasks is not None else None,claims=claims,workers=workers,live_queue=queue(ids))


def classify(before, after):
    live=after['live_queue'];workers=after['workers'];active=[w for w in workers if w.get('child_pid')]
    known_jobs={str(w.get('job_id')) for w in workers if valid_job(w.get('job_id'))}
    allocation_known=(len(workers)==3 and {w.get('slot') for w in workers}=={0,1,2}
        and len(known_jobs)==3 and not live.get('error'))
    if after['progress'].get('all_original_complete') is True and after.get('queue',{}).get('status')=='COMPLETE' and allocation_known and not live.get('jobs'):
        return 'ALL_COMPLETE','原计划已标记全部完成，且本次队列查询确认这些作业均已退出。'
    if after['progress'].get('all_original_complete') is True and not allocation_known:
        return 'SCIENCE_COMPLETE_ALLOCATION_UNKNOWN','状态标记原计划已完成，但作业编号或实时队列证据不足，尚不能确认资源均已释放。'
    completed=(before.get('completed') is not None and after.get('completed') is not None
        and after['completed']>before['completed'])
    numeric=any(w.get('action_key')==p.get('action_key') and w.get('claim_token')==p.get('claim_token')
        and any(v>p.get('numerical',{}).get(k,v) for k,v in w.get('numerical',{}).items())
        for w,p in zip(workers,before['workers']))
    if completed or numeric:return 'SCIENCE_PROGRESS_OBSERVED','采样期间已完成任务数或实际数值更新计数增加。'
    if after['guard'].get('status')=='RETRYING':
        return 'MANAGEMENT_RETRYING','最近管理状态记录为重试（需结合状态年龄与进程核对）：'+str(after['guard'].get('reason',after['guard'].get('error_type','未知错误')))
    all_wait=all(not w.get('child_pid') and str(w.get('phase','')).startswith('WAITING') for w in workers)
    if after.get('dispatch_age') is not None and after['dispatch_age']>300 and all_wait:
        return 'SUSPECT_DISPATCH_STALL','三个槽位均无训练子进程，调度队列超过5分钟未更新，疑似再次停在派发。'
    running={j['job_id'] for j in live.get('jobs',[]) if j.get('state')=='RUNNING'}
    stale=[w for w in active if str(w.get('job_id')) in running and w.get('log',{}).get('mtime') is not None
        and not w.get('log',{}).get('error') and after['at']-w['log']['mtime']>600
        and w.get('action_key')==before['workers'][w['slot']].get('action_key')
        and w.get('claim_token')==before['workers'][w['slot']].get('claim_token')
        and w.get('log',{}).get('text')==before['workers'][w['slot']].get('log',{}).get('text')]
    if stale:return 'SUSPECT_ACTION_STALL','仍分配着作业，但当前子任务日志超过10分钟未变且未观察到数值推进，需继续定位该子任务。'
    fresh=after.get('guard_age') is not None and after['guard_age']<300
    changed=any(w.get('child_pid') and not w.get('log',{}).get('error') and not p.get('log',{}).get('error')
        and w.get('log',{}).get('text') and w.get('action_key')==p.get('action_key') and w.get('claim_token')==p.get('claim_token')
        and w.get('log',{}).get('text')!=p.get('log',{}).get('text') for w,p in zip(workers,before['workers']))
    if fresh and changed:
        if any(w.get('action_kind') in VERIFY_KINDS for w in active):
            return 'ACTIVE_RESULT_CHECKS','当前结果选择或冻结任务的日志在更新；缓存命中和读文件字节不是训练更新次数。'
        return 'ACTION_ACTIVE','当前子任务日志在更新；仅凭日志活动不能证明GPU优化器正在更新。'
    if len(active)==1 and active[0].get('claim_bound') and active[0].get('age_seconds') is not None and active[0]['age_seconds']<180 and any(w.get('phase')=='WAITING_SERIAL_ACTION' for w in workers):
        return 'EXPECTED_SERIAL_WAIT','新鲜状态和当前claim记录显示一个串行动作，其余槽位按协议等待；当前短窗口未证明GPU数值更新。'
    return 'INSUFFICIENT_EVIDENCE','本次短间隔未观察到明确推进；短时间日志不变不足以判定卡死。'


def display(s, label):
    stamp=dt.datetime.fromtimestamp(s['at'],dt.timezone(dt.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    say('\n'+label+'时间：'+stamp+' 北京时间')
    say('\n'+label+'：完成 %s / %s（已观察到的动态任务数），阶段 %s'%(s['completed'],s['total'],s['phase']))
    say('管理器状态=%s，状态年龄=%s秒，调度队列年龄=%s秒，管理进程身份核对=%s'%(
        s['guard'].get('status','未知'),s['guard_age'],s['dispatch_age'],s['guardian_process_matches']))
    for w in s['workers']:
        say('槽位%d 作业%s：%s；子进程=%s；任务=%s'%(w['slot'],w.get('job_id','未知'),w.get('phase','未知'),w.get('child_pid'),w.get('action_key','无')))
        if w.get('numerical'):say('  实际数值计数：'+json.dumps(w['numerical'],ensure_ascii=False))
        if w.get('cache_event'):say('  文件校验缓存（不等于数值更新）：'+json.dumps(w['cache_event'],ensure_ascii=False))
        lines=w.get('log',{}).get('text','').splitlines()[-3:]
        if lines:say('  当前子任务末尾：\n'+'\n'.join(lines))
        elif w.get('slurm_tail'):say('  作业日志末尾：\n'+w['slurm_tail'])
    say('本次队列：'+json.dumps(s['live_queue'],ensure_ascii=False))


def main(argv=None):
    say('R1只读检查已启动：逐项显示；两次采样；不停止或提交作业。')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/public/home/zhuziyuan/projects/paper3'))
    p.add_argument('--home',type=Path,default=Path.home());p.add_argument('--interval',type=float,default=35);p.add_argument('--json',action='store_true')
    a=p.parse_args(argv)
    if not 0<=a.interval<=300:p.error('--interval must be between0 and300seconds')
    first=snapshot(a.root,a.home);display(first,'第一次采样')
    remaining=a.interval
    while remaining>0:
        step=min(5,remaining);say('等待下一次采样：还剩约%d秒（检查仍在进行）'%remaining);time.sleep(step);remaining-=step
    second=snapshot(a.root,a.home);display(second,'第二次采样');code,message=classify(first,second)
    say('\n检查结论 ['+code+']：'+message)
    if a.json:say(json.dumps(dict(classification=code,message=message,first=first,second=second),ensure_ascii=False))
    say('R1只读检查结束。');return 0


if __name__=='__main__':raise SystemExit(main())
