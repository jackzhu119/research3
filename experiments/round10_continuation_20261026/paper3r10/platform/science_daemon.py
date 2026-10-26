"""Allocation-local scientific dispatcher; imports numerical worker exactly once.

The stdin protocol contains only task keys/tokens. Acknowledgements are atomic
files, so scientific stdout/progress output cannot corrupt control messages.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout,redirect_stderr
import os
from pathlib import Path
import re
import signal
import sys
import threading
import traceback

from .common import atomic,read,utc


def main(argv,*,plan,worker,stop):
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--response-dir',required=True)
    args=parser.parse_args(argv);responses=Path(args.response_dir).resolve()
    runtime=(Path(plan['output'])/'runtime').resolve()
    if not responses.is_relative_to(runtime):raise ValueError('Scientific responses must stay in this R10 runtime')
    responses.mkdir(parents=True,exist_ok=True);tasks={t['key']:t for t in plan['tasks']}
    worker_stop=getattr(worker,'_STOP_REQUESTED',stop)
    def request_stop(*_):
        stop.set();worker_stop.set();os.environ['PAPER3_R10_STOP_REQUESTED']='1'
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    atomic(responses/'READY.json',dict(status='SCIENTIFIC_DISPATCHER_READY',pid=os.getpid(),
        recipe_sha256=plan['recipe_sha256'],worker_module=worker.__name__,updated_utc=utc(),
        training_progress_claimed=False))
    for line in sys.stdin:
        if stop.is_set():return 75
        request=__import__('json').loads(line)
        if request.get('shutdown'):return 0
        key=request.get('task_key');token=request.get('attempt_token');sequence=request.get('sequence')
        if key not in tasks or not re.fullmatch('[0-9a-f]{64}',str(token)) or type(sequence) is not int or sequence<1:
            raise ValueError('Invalid scientific dispatch request')
        attempt=Path(request['attempt_dir']).resolve()
        if not attempt.is_relative_to(runtime):raise ValueError('Attempt logs must stay in this R10 runtime')
        attempt.mkdir(parents=True,exist_ok=True);task=tasks[key];folder=Path(plan['output'])/key
        os.environ['PAPER3_R10_ATTEMPT_TOKEN']=token
        finished=threading.Event()
        def heartbeat(folder=folder,token=token,finished=finished,worker_stop=worker_stop):
            while not finished.wait(20):
                lock=getattr(worker,'_PROGRESS_LOCK',None)
                if lock:lock.acquire()
                try:
                    current=read(folder/'progress.json',{})
                    if current.get('attempt_token')==token:
                        current.update(updated_utc=utc(),heartbeat_only=True,checkpoint_stop_requested=worker_stop.is_set())
                        atomic(folder/'progress.json',current)
                finally:
                    if lock:lock.release()
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start();code=0;error_type=None;restart_required=False
        with (attempt/'stdout.log').open('a',buffering=1,encoding='utf-8') as out,(attempt/'stderr.log').open('a',buffering=1,encoding='utf-8') as err:
            with redirect_stdout(out),redirect_stderr(err):
                try:
                    worker.run_task(plan,task,args.device)
                except Exception as error:
                    error_type=type(error).__name__
                    pause=getattr(worker,'PauseRequested',())
                    if pause and isinstance(error,pause):
                        code=75
                        atomic(folder/'progress.json',dict(status='PAUSED_CHECKPOINT_SAVED',stage='paused',
                            attempt_token=token,updated_utc=utc()))
                    else:
                        code=1
                        text=str(error).lower()
                        restart_required=any(value in text for value in ('device-side assert','illegal memory access','context is destroyed'))
                        folder.mkdir(parents=True,exist_ok=True)
                        atomic(folder/'scientific_error.json',dict(status='FAILED',utc=utc(),task_key=key,
                            attempt_token=token,error_type=error_type,reason=str(error),traceback=traceback.format_exc()))
                        traceback.print_exc()
                finally:
                    finished.set();thread.join(timeout=1)
        atomic(responses/('%06d.json'%sequence),dict(task_key=key,attempt_token=token,sequence=sequence,
            exit_code=code,error_type=error_type,restart_required=restart_required,finished_utc=utc(),scientific_pid=os.getpid()))
        if restart_required:return 2
        if code==75 or stop.is_set():return 75
    return 0
