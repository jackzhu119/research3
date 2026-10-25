"""Trigger report refresh in a separate process when completed task count changes."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent

def _read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


class ReportDispatcher:
    def __init__(self,root,operational,*,interval=600,clock=time.time,popen=subprocess.Popen,emit=None):
        self.root=Path(root);self.operational=Path(operational)
        self.interval=interval;self.clock=clock;self.popen=popen
        self.emit=emit or (lambda *a,**k:None)
        self.thread=None;self.next_at=0;self.last_attempt_count=None;self._mutex=threading.Lock()
    def kick(self,snapshot):
        # This foreground path does no filesystem/network work and never joins.
        complete=snapshot.get('complete_tasks',0)
        if not isinstance(complete,int) or complete<=0:return False
        final=complete==snapshot.get('total_tasks')
        with self._mutex:
            if self.thread is not None and self.thread.is_alive():return False
            if self.clock()<self.next_at and (not final or complete==self.last_attempt_count):return False
            self.next_at=self.clock()+self.interval;self.last_attempt_count=complete
            self.thread=threading.Thread(target=self._run,args=(snapshot,),daemon=True,
                                         name='paper3-r9-report-dispatch')
            self.thread.start();return True
    def _run(self,snapshot):
        try:
            launch=_read(self.operational/'launch.json')
            plan=Path(launch.get('plan',str(self.root/'runs/followup_20261025_r9/plan.json')))
            package=Path(launch.get('package',str(HERE.parent)))
            previous=_read(self.operational/'report-state.json')
            if (previous.get('status')=='BUILT' and
                previous.get('recipe_sha256')==snapshot.get('recipe_sha256') and
                previous.get('complete_tasks')==snapshot.get('complete_tasks')):
                return
            python=launch.get('python',sys.executable)
            command=[python,'-I','-S','-B','-u',str(HERE/'report_entry.py'),
                     '--plan',str(plan),'--operational',str(self.operational),'--package',str(package)]
            self.operational.mkdir(parents=True,exist_ok=True)
            with (self.operational/'report.log').open('ab',buffering=0) as log:
                child=self.popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                                 start_new_session=True,close_fds=True)
                self.emit('ROUND_9_REPORT_PROCESS_STARTED',pid=child.pid,training_controlled=False)
                code=child.wait()
            self.emit('ROUND_9_REPORT_PROCESS_RETURNED',returncode=code,training_controlled=False,
                      log=str(self.operational/'report.log'))
        except Exception as error:
            self.emit('ROUND_9_REPORT_RETRY',error_type=type(error).__name__,training_controlled=False)
