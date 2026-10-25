#!/usr/bin/env python3
"""Prepare/start R5 or R6's three-job checkpoint continuation and GitHub monitor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))
from coordinator import prepare_context
from three_jobs import atomic,read,emit


def _process_receipt(pid,script,context):
    result={'pid':pid,'script':str(script),'round':context['round'],'root':str(context['root']),
            'runtime':str(context['runtime']),'started_at':time.time()}
    try:
        result['start_ticks']=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
        result['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (OSError,ValueError,IndexError):pass
    return result


def _alive(receipt,script,context):
    try:
        pid=int(receipt['pid']);folder=Path('/proc')/str(pid)
        if folder.stat().st_uid!=os.getuid():return False
        fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X','x'):return False
        if receipt.get('start_ticks')!=int(fields[19]):return False
        if receipt.get('boot_id')!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return False
        tokens=[x.decode(errors='replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]
        expected=['--round',str(context['round']),'--root',str(context['root'])]
        if str(script) not in tokens:return False
        return all(expected[i] in tokens and tokens[tokens.index(expected[i])+1]==expected[i+1]
                   for i in (0,2))
    except (OSError,ValueError,KeyError,IndexError,TypeError):return False


def _lock_held(path):
    with Path(path).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def start_process(context,kind,credentials=None,popen=None):
    runtime=context['runtime'];launch=context['launch'];popen=popen or subprocess.Popen
    script=PACKAGE/('coordinator.py' if kind=='controller' else 'monitor.py')
    receipt_path=runtime/(kind+'-process.json')
    prior=read(receipt_path,{}) or {}
    if _alive(prior,script,context) or _lock_held(runtime/(kind+'.lock')):
        emit('EXISTING_THREE_JOBS_'+kind.upper()+'_REUSED',round=context['round'],pid=prior.get('pid'))
        return {'kind':kind,'reused':True,'pid':prior.get('pid'),'log':str(runtime/(kind+'.log'))}
    command=[launch['python'],'-I','-S','-B','-u',str(script),'--round',str(context['round']),
             '--root',str(context['root'])]
    if kind=='controller':
        command+=['--runtime',str(runtime),'--old-job',launch['old_job'],'--python',launch['python'],
                  '--package',launch['package'],'--account',launch['account']]
    else:
        command+=['--operational',str(runtime)]
        if credentials:command+=['--credentials',str(Path(credentials).expanduser())]
    log=runtime/(kind+'.log')
    with log.open('ab',buffering=0) as handle:
        child=popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=handle,
                    start_new_session=True,close_fds=True)
    receipt=_process_receipt(child.pid,script,context)
    atomic(receipt_path,receipt)
    return {'kind':kind,'reused':False,'pid':child.pid,'log':str(log)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--round',type=int,choices=(5,6),required=True,dest='round_number')
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--runtime')
    parser.add_argument('--old-job')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--scientific-package')
    parser.add_argument('--account')
    parser.add_argument('--credentials')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    context=prepare_context(args.root,args.round_number,args.runtime,old_job=args.old_job,
        python=args.python,package=args.scientific_package,account=args.account)
    runtime=context['runtime'];launch=context['launch']
    atomic(runtime/'launch.json',launch)
    emit('ROUND_56_THREE_JOBS_PREPARED',round=context['round'],complete_tasks=context['complete_tasks'],
         total_tasks=context['total_tasks'],plan=str(context['plan_path']),
         scientific_package=launch['package'],partition='hpc_gpu1',jobs=3,gpus_total=3,
         cpus_per_job=12,cpus_total=36,memory_per_job='64G',memory_total='192G',
         old_job=launch['old_job'],original_plan_and_checkpoints_unchanged=True,
         handoff_policy='three replacement held jobs observed before old allocation stop',
         github_branch='paper3-r%d-progress'%context['round'],
         github_status='monitoring/paper3-round%d/three-jobs-status.json'%context['round'],
         runtime=str(runtime))
    if not args.execute:return 0
    with (runtime/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Explicit execute resumes only this new controller's operational STOP;
        # the original controller's STOP tombstone is deliberately preserved.
        own_stop=runtime/'STOP'
        if own_stop.exists():
            history=runtime/'startup-history';history.mkdir(exist_ok=True)
            os.replace(own_stop,history/('STOP-'+str(time.time_ns())))
        processes=[start_process(context,'controller'),start_process(context,'monitor',args.credentials)]
    emit('ROUND_56_BACKGROUND_PROCESSES_STARTED',round=context['round'],processes=processes,
         submission_receipts=str(runtime/'scheduler-submissions'),progress=str(runtime/'progress.json'),
         handoff=str(runtime/'migration.json'),
         note='Process startup only; actual held job IDs and queue proofs appear in submission receipts and migration log.')
    return 0


if __name__=='__main__':raise SystemExit(main())
