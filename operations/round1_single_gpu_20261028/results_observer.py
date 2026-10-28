"""Asynchronous existing R1 result summaries, isolated from live progress.

The original collector/publisher is retained verbatim. Only its scheduling and
process lifetime change: no reporting recomputation, raw model/prediction scan,
or participation in scientific control occurs here.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import observer


def signal_key(snapshot):
    if type(snapshot.get('parallel_plan_complete_tasks')) is not int:
        return None
    arms=snapshot.get('extension_arms',{})
    value=dict(completed=snapshot['parallel_plan_complete_tasks'],
        mainline=snapshot.get('mainline_complete') is True,
        complete=snapshot.get('all_original_complete') is True,
        arms={arm:dict(status=row.get('status'),counts=row.get('native_completed_artifact_counts',{}))
              for arm,row in arms.items() if isinstance(row,dict)})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def command(root,home,work,credentials=None):
    argv=[sys.executable,'-I','-S','-B','-u',str(HERE/'results_observer.py'),'--cycle',
          '--root',str(root),'--home',str(home),'--work',str(work)]
    if credentials:argv+=['--credentials',str(credentials)]
    return argv


class Dispatcher:
    def __init__(self,root,home,work,credentials=None,*,clock=time.time,popen=subprocess.Popen,
                 process=observer.process_record,send=os.kill,min_interval=600,deadline=1200):
        self.root,self.home,self.work=Path(root),Path(home),Path(work)
        self.credentials,self.clock,self.popen=credentials,clock,popen
        self.process,self.send=process,send
        self.min_interval,self.deadline=min_interval,deadline
        self.work.mkdir(parents=True,exist_ok=True)
        self.schedule=self.work/'results-observer-schedule.json'
        self.pid_path=self.work/'results-observer-process.json'
        self.child=None

    def current(self,record):
        found=self.process(record.get('pid',0))
        keys=('pid','uid','argv','start_ticks')
        expected=command(self.root,self.home,self.work,self.credentials)
        return (found is not None and record.get('argv')==expected
                and all(found.get(k)==record.get(k) for k in keys))

    def tick(self,snapshot):
        """Bounded local metadata only; no join, collection or network requests."""
        now=self.clock();state,_=observer.read(self.schedule,65536)
        record,_=observer.read(self.pid_path,65536)
        if self.child is not None and self.child.poll() is not None:
            self.child=None
        if record and self.current(record):
            if now-record.get('started_unix',now)>self.deadline:
                # Only this exact recorded result helper is eligible. A newly
                # reused PID or unrelated training process never matches.
                if self.current(record):
                    try:self.send(record['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
                state.update(next_attempt_unix=now+self.min_interval,
                    last_status='OWNED_RESULTS_HELPER_DEADLINE',last_signal=None)
                observer.atomic(self.schedule,state)
                observer.emit('R1_RESULTS_HELPER_TIMEOUT',scientific_jobs_signalled=0)
            return False
        result,_=observer.read(self.work/'results-observer-status.json',65536)
        if record and state.get('inflight_signal'):
            if result.get('attempt_signal')==state['inflight_signal'] and result.get('status') in (
                    'SUMMARY_UPLOADED','UNCHANGED_EXISTING_SUMMARY'):
                state.update(last_signal=state['inflight_signal'],failures=0,last_status=result['status'])
            else:
                failures=int(state.get('failures',0))+1
                state.update(failures=failures,last_status=result.get('status','RESULT_HELPER_EXITED'),
                    next_attempt_unix=max(state.get('next_attempt_unix',0),
                        now+min(3600,self.min_interval*2**min(failures-1,3))))
            state.pop('inflight_signal',None);observer.atomic(self.schedule,state)
        desired=signal_key(snapshot)
        if desired is None or desired==state.get('last_signal') or now<state.get('next_attempt_unix',0):
            return False
        argv=command(self.root,self.home,self.work,self.credentials)
        observer.atomic(self.work/'results-observer-request.json',dict(attempt_signal=desired))
        with (self.work/'results-observer.log').open('ab',buffering=0) as log:
            child=self.popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                             start_new_session=True,close_fds=True)
        record=self.process(child.pid) or dict(pid=child.pid,uid=os.getuid(),argv=argv,start_ticks=None)
        record.update(argv=argv,started_unix=now)
        observer.atomic(self.pid_path,record);self.child=child
        state.update(inflight_signal=desired,last_attempt_unix=now,next_attempt_unix=now+self.min_interval)
        observer.atomic(self.schedule,state)
        observer.emit('R1_RESULTS_HELPER_STARTED',pid=child.pid,minimum_interval=self.min_interval,
                      report_recomputed=False,scientific_jobs_signalled=0)
        return True


def cycle(root,home,work,credentials=None,*,collector=None,publisher=None,client_factory=None,
          token_provider=None):
    root,home,work=Path(root).resolve(),Path(home).resolve(),Path(work).resolve()
    runtime=root/observer.OPERATIONAL;work.mkdir(parents=True,exist_ok=True)
    request,_=observer.read(work/'results-observer-request.json',65536)
    attempt_signal=request.get('attempt_signal')
    try:
        try:os.nice(10)
        except OSError:pass
        import auto_summary
        import upload_transport
        import summary_transport
        with (runtime/'auto_summary.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                result=dict(status='EXISTING_RESULTS_WORKER_ACTIVE',attempt_signal=attempt_signal,
                            updated_utc=observer.utc(),training_controlled=False)
                observer.atomic(work/'results-observer-status.json',result);return result
            # The retained collector captures existing completed receipts/CSVs;
            # it does not invoke any scientific report builder or model fit.
            bundle=(collector or auto_summary.collect_summary)(root,1,runtime)
            prior,_=observer.read(runtime/'auto_summary_state.json',65536)
            if prior.get('content_hash')==bundle['content_hash']:
                result=dict(status='UNCHANGED_EXISTING_SUMMARY',content_hash=bundle['content_hash'],
                    commit=prior.get('commit'),attempt_signal=attempt_signal,updated_utc=observer.utc(),
                    report_recomputed=False,training_controlled=False)
            else:
                token=(token_provider or upload_transport._token_provider)(home,credentials,8)
                factory=client_factory or summary_transport.GitHub
                client=factory(token,observer.emit,repo='jackzhu119/research3',branch='paper3-results',
                    cache_path=runtime/'auto_summary_blob_cache.json')
                published=(publisher or auto_summary.publish_summary)(bundle,1,client,emit=observer.emit)
                observer.atomic(runtime/'auto_summary_state.json',dict(content_hash=bundle['content_hash'],
                    commit=published['commit'],round=1,summary_only=True))
                result=dict(status='SUMMARY_UPLOADED',content_hash=bundle['content_hash'],commit=published['commit'],
                    branch='paper3-results',pointer='results/paper3/round-01/AUTO_SUMMARY_LATEST.json',
                    attempt_signal=attempt_signal,updated_utc=observer.utc(),report_recomputed=False,
                    full_raw_results_uploaded=False,raw_predictions_or_checkpoints_included=False,
                    training_controlled=False)
        observer.atomic(work/'results-observer-status.json',result);return result
    except Exception as error:
        result=dict(status='RETRYING',error_type=type(error).__name__,attempt_signal=attempt_signal,
                    updated_utc=observer.utc(),training_controlled=False,report_recomputed=False)
        observer.atomic(work/'results-observer-status.json',result)
        observer.emit('R1_RESULTS_SUMMARY_RETRY',error_type=type(error).__name__)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle',action='store_true',required=True)
    for key in ('root','home','work'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--credentials',type=Path)
    args=parser.parse_args(argv)
    result=cycle(args.root,args.home,args.work,args.credentials)
    observer.emit('R1_RESULTS_HELPER_FINISHED',**result)
    return 0


if __name__=='__main__':raise SystemExit(main())
