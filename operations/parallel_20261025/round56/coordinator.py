"""Round 5/6 controller: queue three held replacements before old handoff."""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))
from three_jobs import controller_step,plan_context,result_complete,read,atomic,emit
from migrate import HeldQueueMigration,MigrationTransient,MigrationOwnershipError


def scientific_package(root,round_number,output,explicit=None):
    """Locate the existing numerical entry; never copy or import it on login."""
    old_launch=read(output/'runtime/launch.json',{}) or {}
    plan=read(output/'plan.json',{}) or {}
    candidates=[]
    if explicit:candidates.append(Path(explicit).expanduser())
    for source in (old_launch,plan):
        for key in ('package','scientific_package','source_package'):
            if source.get(key):candidates.append(Path(source[key]).expanduser())
    candidates.append(Path(root)/('experiments/followup_20261024_r%d'%round_number))
    for name in ('paper3-round%d-hpcgpu'%round_number,'paper3-round%d'%round_number):
        base=Path.home()/name
        candidates.append(base)
        candidates.extend(path.parent.parent for path in sorted(base.glob('*/paper3r%d/entry.py'%round_number)))
    for candidate in candidates:
        candidate=candidate.absolute()
        if (candidate/('paper3r%d/entry.py'%round_number)).is_file():
            return candidate.resolve()
    raise RuntimeError('Existing round %d scientific entry.py not found; original outputs unchanged'%round_number)


def prepare_context(root,round_number,runtime=None,*,old_job=None,python=None,package=None,account=None):
    root=Path(root).expanduser().resolve()
    if round_number not in (5,6) or not root.is_dir():
        raise ValueError('Existing root and round 5/6 are required')
    output=root/('runs/followup_20261024_r%d'%round_number)
    plan_path,plan,output,runtime,plan_sha=plan_context(output/'plan.json',runtime)
    if Path(plan.get('root',root)).resolve()!=root:
        raise ValueError('Original plan root differs from requested root')
    package=scientific_package(root,round_number,output,package)
    python=Path(python or sys.executable).expanduser().absolute()
    if not python.is_file():raise ValueError('Existing Python interpreter not found')
    old_launch=read(output/'runtime/launch.json',{}) or {}
    worker=[str(python),'-I','-S','-B','-u',str(PACKAGE/'three_jobs.py'),'worker',
            '--plan',str(plan_path),'--runtime',str(runtime),'--round',str(round_number),
            '--slot','{slot}','--package',str(package),'--python',str(python)]
    launch={'round':round_number,'worker_command':worker,'partition':'hpc_gpu1','old_partition':'hpc_gpu','cpus':12,
            'memories':['64G','64G','64G'],'gres':'gpu:1','submit_hold':True,
            'account':account or old_launch.get('account') or getpass.getuser(),
            'campaign_identity':plan.get('recipe_sha256'),'python':str(python),
            'package':str(package),'time_limit':old_launch.get('time_limit','5-03:00:00'),
            'old_job':str(old_job or {5:772012,6:772013}[round_number]),
            'root':str(root),'plan':str(plan_path),'runtime':str(runtime),
            'original_plan_sha256':plan_sha,'poll_seconds':30}
    completed=sum(result_complete(output,t,plan.get('recipe_sha256')) for t in plan['tasks'])
    return {'root':root,'round':round_number,'output':output,'runtime':runtime,
            'plan_path':plan_path,'plan':plan,'launch':launch,
            'complete_tasks':completed,'total_tasks':len(plan['tasks'])}


def stop_owned_new_workers(runtime,marker,account,*,runner=None,emit_callback=emit):
    """Drain only this exact new three-job marker after completion/explicit stop."""
    runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True)
    stop=runtime/'STOP'
    if not stop.exists():stop.write_text('New three-job controller completed or explicitly stopped.\n')
    if not marker:return True
    runner=runner or subprocess.run
    result=runner(['squeue','--array','--noheader','--user',account,
                   '--format=%i|%k|%T|%j'],capture_output=True,text=True,timeout=25)
    if result.returncode:raise RuntimeError('New worker completion queue observation failed')
    owned=[]
    for raw in result.stdout.splitlines():
        f=[x.strip() for x in raw.split('|')]
        if len(f)<4 or f[1]!=marker or not re.fullmatch(r'\d+',f[0]):continue
        if not re.fullmatch(re.escape(marker)+r'-s[012]-a[1-9]\d*',f[3]):
            raise MigrationOwnershipError('New worker job name differs from exact owned marker')
        state=f[2].split()[0].rstrip('+')
        if state=='PENDING':command=['scancel','--state=PENDING',f[0]]
        elif state in ('RUNNING','CONFIGURING','SUSPENDED','RESIZING','COMPLETING'):
            command=['scancel','--batch','--signal=USR1',f[0]]
        else:continue
        owned.append(f[0])
        answer=runner(command,capture_output=True,text=True,timeout=25)
        emit_callback('NEW_OWNED_WORKER_DRAIN_REQUESTED',job_id=f[0],returncode=answer.returncode)
    return not owned


def controller_iteration(context,*,runner=None,clock=None,migration=None,step_function=None,emit_callback=emit):
    launch,runtime,plan_path=context['launch'],context['runtime'],context['plan_path']
    # The scheduler first records/submits/adopts the held jobs. The migrator
    # independently verifies all three with squeue+scontrol before touching old.
    function=step_function or controller_step
    summary=function(plan_path,launch,runtime,runner=runner,clock=clock,emit_callback=emit_callback)
    progress=summary.get('progress',{})
    complete=(progress.get('total_tasks',0)>0 and
              progress.get('complete_tasks')==progress.get('total_tasks'))
    if complete:
        idle=stop_owned_new_workers(runtime,summary.get('marker'),launch['account'],runner=runner,
                                    emit_callback=emit_callback)
        summary['handoff']={'status':'ALL_TASKS_ALREADY_COMPLETE_NO_OLD_CANCELLATION_NEEDED'}
        summary['status']='FINISHED' if idle else 'DRAINING_COMPLETED_WORKERS'
    else:
        migration=migration or HeldQueueMigration(plan_path,runtime,context['round'],launch,
                runner=runner,clock=clock,emit=emit_callback,old_job=launch['old_job'])
        summary['handoff']=migration.step(summary)
        summary['status']=summary['handoff'].get('status','WAITING_HANDOFF')
    atomic(Path(runtime)/'controller-status.json',summary)
    emit_callback('THREE_JOBS_CONTROLLER_PROGRESS',round=context['round'],status=summary['status'],
                  complete_tasks=progress.get('complete_tasks'),total_tasks=progress.get('total_tasks'),
                  jobs=summary.get('jobs',[]),handoff=summary.get('handoff',{}).get('status'))
    return summary


def controller_loop(context,*,runner=None,clock=None,sleeper=None,stop=None,once=False,emit_callback=emit):
    runtime=context['runtime'];runtime.mkdir(parents=True,exist_ok=True)
    stop=stop or threading.Event();sleeper=sleeper or stop.wait
    with (runtime/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            emit_callback('EXISTING_THREE_JOBS_CONTROLLER_REUSED',round=context['round'])
            return 0
        launch=context['launch']
        atomic(runtime/'launch.json',launch)
        migration=HeldQueueMigration(context['plan_path'],runtime,context['round'],launch,
            runner=runner,clock=clock,emit=emit_callback,old_job=launch['old_job'])
        emit_callback('THREE_JOBS_CONTROLLER_STARTED',round=context['round'],runtime=str(runtime),
                      original_output=str(context['output']),old_job=launch['old_job'])
        while not stop.is_set():
            try:
                summary=controller_iteration(context,runner=runner,clock=clock,migration=migration,
                                             emit_callback=emit_callback)
                if summary['status']=='FINISHED':
                    emit_callback('THREE_JOBS_ROUND_COMPLETE',round=context['round'])
                    return 0
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                # Temporary scheduler/filesystem failures preserve old ownership
                # and submission receipts. The independent monitor is unaffected.
                emit_callback('THREE_JOBS_CONTROLLER_RETRY',round=context['round'],
                              error_type=type(error).__name__)
            if once:return 0
            sleeper(float(launch.get('poll_seconds',30)))
        state=read(runtime/'scheduler-state.json',{}) or {}
        try:stop_owned_new_workers(runtime,state.get('marker'),launch['account'],runner=runner,
                                   emit_callback=emit_callback)
        except Exception as error:
            emit_callback('THREE_JOBS_STOP_OBSERVATION_RETRY',error_type=type(error).__name__)
        return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',type=int,choices=(5,6),required=True,dest='round_number')
    parser.add_argument('--root',default='/public/home/zhuziyuan/projects/paper3')
    parser.add_argument('--runtime')
    parser.add_argument('--old-job')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--package')
    parser.add_argument('--account')
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    context=prepare_context(args.root,args.round_number,args.runtime,old_job=args.old_job,
        python=args.python,package=args.package,account=args.account)
    stop=threading.Event()
    for number in (signal.SIGTERM,signal.SIGINT):signal.signal(number,lambda *_:stop.set())
    return controller_loop(context,stop=stop,once=args.once)


if __name__=='__main__':raise SystemExit(main())
