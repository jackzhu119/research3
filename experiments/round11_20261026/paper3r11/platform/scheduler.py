"""Four durable, independent single-GPU Slurm workers for the original campaign.

This module controls only its exact operational marker, never the old campaign
allocation. Numerical task selection and checkpoint handling belong to worker.py.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


MAX_WORKER_FAILURES = 3

ACTIVE_STATES = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
                 'REQUEUED', 'REQUEUE_FED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'}
DEFAULT_LAUNCH = {'partition': 'hpc_gpu', 'account': 'zhuziyuan', 'cpus': 16,
                  'gres': 'gpu:1', 'memories': ['64G'] * 4,
                  'time_limit': '5-03:00:00', 'poll_seconds': 30,
                  'exclude': ''}


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _state(value):
    return value.strip().split()[0].rstrip('+').upper() if value.strip() else 'UNKNOWN'


def write_worker_outcome(runtime_dir, slot, job_id, exit_code, **fields):
    """Optional worker receipt; a completed receipt can bridge sacct propagation lag."""
    value = {'slot': int(slot), 'job_id': str(job_id), 'exit_code': int(exit_code),
             'name': os.environ.get('PAPER3_MULTI_ATTEMPT_NAME'),
             'marker': os.environ.get('PAPER3_MULTI_MARKER'), 'finished_at': time.time(), **fields}
    _atomic(Path(runtime_dir)/'scheduler-outcomes'/(str(job_id)+'.json'), value)


class FourJobScheduler:
    """Call step() repeatedly; workers keep claiming independent numerical tasks.

    ``launch['worker_command']`` is a list of arguments. Literal ``{slot}`` in an
    argument is replaced with 0, 1, 2 or 3. No shell interpolation or eval is used.
    All attempts are durable before calling sbatch; ambiguous acknowledgements
    occupy their slot until exact queue/accounting/local outcome evidence arrives.
    """
    def __init__(self, runtime_dir, launch, runner=None, clock=None, emit=None):
        self.runtime = Path(runtime_dir).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.launch = {**DEFAULT_LAUNCH, **launch}
        if self.launch.get('round') not in (11,):
            raise ValueError('Round must be 11')
        command = self.launch.get('worker_command')
        if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
            raise ValueError('worker_command must be a nonempty list of arguments')
        if int(self.launch['cpus']) != 16 or self.launch['memories'] != ['64G'] * 4:
            raise ValueError('Four workers require 16 CPU and 64G memory each')
        if self.launch['gres'] != 'gpu:1' or self.launch['partition'] != 'hpc_gpu':
            raise ValueError('Round11 parallel workers use hpc_gpu, gpu:1')
        identity = [str(self.runtime), self.launch.get('campaign_identity', 'round11-recipe')]
        self.marker = ('paper3r%dp-' % int(self.launch['round'])) + hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()[:16]
        self.name_pattern = re.compile(re.escape(self.marker) + r'-s([0123])-a([1-9]\d*)$')
        self.runner, self.clock = runner or subprocess.run, clock or time.time
        self.emit = emit or (lambda *args, **kwargs: None)
        self.state_path = self.runtime/'scheduler-state.json'
        self.receipts = self.runtime/'scheduler-submissions'
        self.receipts.mkdir(exist_ok=True)
        (self.runtime/'scheduler-outcomes').mkdir(exist_ok=True)
        (self.runtime/'logs').mkdir(exist_ok=True)
        self.batch_path = self.runtime/'single-gpu-worker.sbatch'
        self._write_batch()
        self.state = None
        self._lock_depth = 0

    def _write_batch(self):
        # Slurm sends USR1 five minutes before its per-job time limit. The worker
        # receives the signal and finishes the current checkpoint transaction.
        content = '''#!/bin/bash
set -u
printf 'ROUND_11_PARALLEL_PROCESS_STARTED job=%s slot=%s node=%s time=%s\\n' "${SLURM_JOB_ID:-}" "${PAPER3_MULTI_SLOT:-}" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1
"$@" &
worker_pid=$!
interrupted=0
forward_usr1() { interrupted=1; kill -USR1 "$worker_pid" 2>/dev/null || true; }
forward_term() { interrupted=1; kill -TERM "$worker_pid" 2>/dev/null || true; }
trap forward_usr1 USR1
trap forward_term TERM INT
while :; do
  interrupted=0
  wait "$worker_pid"
  code=$?
  if [ "$interrupted" -eq 0 ]; then break; fi
done
exit "$code"
'''
        if not self.batch_path.exists() or self.batch_path.read_text() != content:
            temporary = self.batch_path.with_suffix('.tmp-'+str(os.getpid()))
            temporary.write_text(content)
            os.chmod(temporary, 0o700)
            os.replace(temporary, self.batch_path)

    @contextmanager
    def acquire_lock(self):
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        with (self.runtime/'scheduler.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0

    def _load(self):
        state = _read(self.state_path, None) if self.state_path.exists() else {}
        if not isinstance(state,dict):
            raise ValueError('Existing scheduler state is unreadable; retained jobs and retry limits were not reset')
        if state and state.get('marker') != self.marker:
            raise ValueError('Scheduler state belongs to a different campaign')
        self.state = state or {'version': 1, 'marker': self.marker, 'created_at': self.clock(),
                               'slots': {str(i): {'attempts': [], 'next_submit_at': 0,
                                                  'rejections': 0, 'worker_failures': 0} for i in range(4)}}
        for row in self.state['slots'].values():
            row.setdefault('worker_failures', 0)
        for receipt_path in sorted(self.receipts.glob('*.json')):
            receipt = _read(receipt_path, {})
            if receipt.get('marker') != self.marker or receipt.get('slot') not in (0,1,2,3):
                continue
            attempts = self.state['slots'][str(receipt['slot'])]['attempts']
            previous = next((a for a in attempts if a['sequence'] == receipt['sequence']), None)
            if previous is None:
                attempts.append(receipt)
            else:
                previous.update(receipt)
            attempts.sort(key=lambda row: row['sequence'])

    def _save(self):
        self.state['updated_at'] = self.clock()
        _atomic(self.state_path, self.state)

    def _receipt(self, attempt):
        _atomic(self.receipts/('slot-%s-attempt-%05d.json'%(attempt['slot'],attempt['sequence'])), attempt)

    def _new_attempt(self, slot):
        row = self.state['slots'][str(slot)]
        sequence = max([a['sequence'] for a in row['attempts']] or [0]) + 1
        attempt = {'slot': slot, 'sequence': sequence, 'marker': self.marker,
                   'name': '%s-s%d-a%d'%(self.marker,slot,sequence), 'intent_at': self.clock(),
                   'status': 'INTENT', 'job_id': None}
        row['attempts'].append(attempt)
        self._receipt(attempt)
        self._save()
        return attempt

    def build_sbatch_command(self, slot, attempt):
        command = ['sbatch', '--parsable', '--job-name='+attempt['name'],
                   '--comment='+self.marker, '--partition='+self.launch['partition'],
                   '--nodes=1', '--ntasks=1', '--cpus-per-task=16',
                   '--mem='+self.launch['memories'][slot], '--gres=gpu:1',
                   '--time='+self.launch['time_limit'], '--signal=B:USR1@300',
                   '--no-requeue', '--chdir=/tmp', '--open-mode=append',
                   '--output='+str(self.runtime/'logs'/('slot-%d-%%j.out'%slot)),
                   '--error='+str(self.runtime/'logs'/('slot-%d-%%j.err'%slot)),
                   '--export=ALL,BASH_ENV=,ENV=,PYTHONHOME=,PYTHONPATH=,PYTHONUNBUFFERED=1,'
                   'PAPER3_MULTI_SLOT=%d,PAPER3_MULTI_MARKER=%s,PAPER3_MULTI_ATTEMPT_NAME=%s'%(slot,self.marker,attempt['name'])]
        if self.launch.get('account'):
            command.append('--account='+str(self.launch['account']))
        if self.launch.get('exclude'):
            command.append('--exclude='+str(self.launch['exclude']))
        if self.launch.get('submit_hold', False):
            command.append('--hold')
        return command + [str(self.batch_path)] + [s.replace('{slot}',str(slot)) for s in self.launch['worker_command']]

    def _run(self, command, timeout=30):
        return self.runner(command, capture_output=True, text=True, timeout=timeout)

    def _owned(self, comment, name, job_id):
        match = self.name_pattern.fullmatch(name)
        if not match:
            return None
        if comment == self.marker:
            return int(match.group(1)), int(match.group(2))
        if comment in ('', '(null)', 'Unknown', 'None'):
            # Some school accounting configurations omit Comment. An exact,
            # durable unique attempt name is sufficient independent evidence.
            slot, sequence = int(match.group(1)), int(match.group(2))
            attempts = self.state['slots'][str(slot)]['attempts']
            if any(a['name']==name and (a.get('job_id') in (None,job_id)) for a in attempts):
                return slot, sequence
        return None

    def _query(self):
        user = self.launch.get('user') or os.environ.get('USER','zhuziyuan')
        result = self._run(['squeue','--noheader','--user',user,'--format=%i|%k|%T|%j|%R|%N'])
        if result.returncode:
            raise RuntimeError('SQUEUE_RETRY')
        queue = []
        for line in result.stdout.splitlines():
            fields = [v.strip() for v in line.split('|')]
            if len(fields)<4 or not fields[0].isdigit():
                continue
            owned = self._owned(fields[1],fields[3],fields[0])
            if owned is not None:
                queue.append({'slot':owned[0],'sequence':owned[1],'job_id':fields[0],
                    'state':_state(fields[2]),'name':fields[3],
                    'queue_reason':fields[4] if len(fields)>4 else None,
                    'node':fields[5] if len(fields)>5 else None})
        start = datetime.fromtimestamp(self.state['created_at']-86400, timezone.utc).strftime('%Y-%m-%d')
        result = self._run(['sacct','--noheader','--parsable2','--allocations','--user',user,
                            '--starttime',start,'--format=JobID%80,State%40,ExitCode,Comment%100,JobName%150'])
        if result.returncode:
            raise RuntimeError('SACCT_RETRY')
        accounting = []
        for line in result.stdout.splitlines():
            fields = [v.strip() for v in line.split('|')]
            if len(fields)<5 or not fields[0].isdigit():
                continue
            owned = self._owned(fields[3],fields[4],fields[0])
            if owned is not None:
                codes = fields[2].split(':')
                accounting.append({'slot':owned[0],'sequence':owned[1],'job_id':fields[0],
                    'state':_state(fields[1]),'name':fields[4],
                    'exit_code':int(codes[0]) if codes[0].isdigit() else None,
                    'exit_signal':int(codes[1]) if len(codes)>1 and codes[1].isdigit() else None})
        return queue, accounting

    def _adopt(self, rows, is_queue=False):
        for row in rows:
            attempts = self.state['slots'][str(row['slot'])]['attempts']
            attempt = next((a for a in attempts if a['sequence']==row['sequence']),None)
            if attempt is None:
                attempt = {'slot':row['slot'],'sequence':row['sequence'],'marker':self.marker,
                           'name':row['name'],'intent_at':self.clock(),'status':'INTENT',
                           'job_id':None,'origin':'RECOVERED_FROM_SLURM'}
                attempts.append(attempt)
                attempts.sort(key=lambda a:a['sequence'])
            if attempt.get('job_id') not in (None,row['job_id']):
                raise ValueError('One unique attempt name maps to multiple Slurm jobs')
            attempt['job_id'] = row['job_id']
            if is_queue or attempt['status'] not in ('FINISHED','REJECTED'):
                attempt['status'] = 'ACKNOWLEDGED'
            self._receipt(attempt)

    def _local_terminal(self, attempt):
        if not attempt.get('job_id'):
            return None
        value = _read(self.runtime/'scheduler-outcomes'/(attempt['job_id']+'.json'),{})
        if (value.get('marker')==self.marker and value.get('name')==attempt['name']
                and str(value.get('job_id'))==attempt['job_id'] and value.get('slot')==attempt['slot']
                and isinstance(value.get('exit_code'),int)):
            return {'state':'COMPLETED' if value['exit_code']==0 else 'FAILED',
                    'exit_code':value['exit_code'],'exit_signal':0,'local_outcome':True}
        return None

    def _finish(self, attempt, observed):
        attempt.update(status='FINISHED',slurm_state=observed['state'],
                       exit_code=observed.get('exit_code'),exit_signal=observed.get('exit_signal'),
                       finished_at=self.clock())
        clean = observed.get('exit_signal') in (None, 0)
        interrupted=observed.get('state') in ('TIMEOUT','PREEMPTED')
        attempt['outcome'] = 'CHECKPOINT' if interrupted or (clean and observed.get('exit_code')==75) else (
            'COMPLETE' if clean and observed.get('exit_code')==0 else 'RETRY_WORKER')
        slot = self.state['slots'][str(attempt['slot'])]
        if attempt['outcome']=='RETRY_WORKER':
            slot['worker_failures'] += 1
            if slot['worker_failures']>=MAX_WORKER_FAILURES:
                slot['quarantine']={'reason':'WORKER_ALLOCATION_FAILURE_LIMIT',
                                    'failures':slot['worker_failures'],'at':self.clock()}
                attempt['outcome']='QUARANTINED_WORKER'
        elif attempt['outcome']=='COMPLETE':
            slot['worker_failures']=0
        slot['next_submit_at'] = self.clock() + (0 if attempt['outcome']=='COMPLETE' else 30)
        self._receipt(attempt)

    def _submit(self, slot):
        attempt = self._new_attempt(slot)
        command = self.build_sbatch_command(slot,attempt)
        attempt['command'] = command
        self._receipt(attempt)
        self.emit('ROUND_11_WORKER_SUBMITTING', slot=slot, attempt=attempt['sequence'], memory=self.launch['memories'][slot])
        try:
            result = self._run(command,timeout=45)
        except FileNotFoundError:
            attempt.update(status='REJECTED',error_type='SbatchExecutableMissing')
            self.state['submission_block']={'reason':'SbatchExecutableMissing','at':self.clock()}
        except (OSError,subprocess.SubprocessError) as error:
            attempt.update(status='UNCERTAIN',error_type=type(error).__name__)
        else:
            match = re.fullmatch(r'(\d+)(?:;[^\s;]+)?',result.stdout.strip()) if result.returncode==0 else None
            if match:
                attempt.update(status='ACKNOWLEDGED',job_id=match.group(1),acknowledged_at=self.clock())
                self.state['slots'][str(slot)]['rejections'] = 0
                self.emit('ROUND_11_WORKER_SUBMITTED', slot=slot, job_id=attempt['job_id'])
            elif result.returncode:
                structural = re.search(r'Invalid (?:generic resource|partition|account|qos)|'
                    r'Requested (?:node configuration is not available|time limit is invalid)|'
                    r'Batch job submission failed: Invalid',result.stderr or '',re.I)
                quota = re.search(r'QOSMaxSubmitJobPerUserLimit|Job violates accounting/QOS policy|'
                    r'AssociationJobLimit|QOSMaxJobsPerUserLimit',result.stderr or '',re.I)
                if structural:
                    self.state['submission_block']={'reason':'InvalidSchedulerConfiguration','at':self.clock()}
                    attempt.update(status='REJECTED',error_type='InvalidSchedulerConfiguration',returncode=result.returncode)
                    self.emit('ROUND_11_WORKER_SUBMISSION_BLOCKED',slot=slot,error_type='InvalidSchedulerConfiguration')
                elif quota:
                    row=self.state['slots'][str(slot)];row['rejections']+=1
                    delay=min(300,30*2**min(row['rejections']-1,4))
                    row['next_submit_at']=self.clock()+delay
                    attempt.update(status='REJECTED',error_type='SchedulerQuotaWait',returncode=result.returncode)
                    self.emit('ROUND_11_WORKER_SUBMISSION_WAIT',slot=slot,delay_seconds=delay)
                else:
                    attempt.update(status='UNCERTAIN',error_type='UncertainSbatchFailure',returncode=result.returncode)
            else:
                attempt.update(status='UNCERTAIN',error_type='UnclearAcknowledgement')
        self._receipt(attempt)
        self._save()
        return attempt['status']

    def step(self, keep_workers=True):
        """Observe once; True keeps 4 workers, False none, or pass desired slots.

        Reducing desired slots only stops new submission; active jobs are never
        cancelled. Workers exit themselves after their assigned work is drained.
        """
        if isinstance(keep_workers, bool):
            desired_slots = {0,1,2,3} if keep_workers else set()
        else:
            desired_slots = set(keep_workers)
            if any(slot not in (0,1,2,3) for slot in desired_slots):
                raise ValueError('Desired worker slots must be 0, 1, 2 or 3')
        try:
            with self.acquire_lock():
                self._load()
                try:
                    queue,accounting=self._query()
                    self._adopt(accounting)
                    self._adopt(queue, is_queue=True)
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                    self.state.update(controller_status='OBSERVATION_RETRY',observation_error=type(error).__name__)
                    self._save()
                    self.emit('ROUND_11_SCHEDULER_OBSERVATION_RETRY',error_type=type(error).__name__)
                    return self.summary()
                active={row['job_id']:row for row in queue}
                accounts={row['job_id']:row for row in accounting}
                for slot in range(4):
                    for attempt in self.state['slots'][str(slot)]['attempts']:
                        if attempt['status'] in ('FINISHED','REJECTED'):
                            continue
                        job_id=attempt.get('job_id')
                        if job_id in active:
                            observed=active[job_id]
                            attempt.update(status='ACKNOWLEDGED',slurm_state=observed['state'],
                                queue_reason=observed.get('queue_reason'),node=observed.get('node'),observed_at=self.clock())
                        elif job_id in accounts and accounts[job_id]['state'] not in ACTIVE_STATES:
                            self._finish(attempt,accounts[job_id])
                        else:
                            local=self._local_terminal(attempt)
                            if local is not None:
                                self._finish(attempt,local)
                            else:
                                attempt.update(status='UNCERTAIN',error_type='AwaitingQueueOrAccountingEvidence')
                        self._receipt(attempt)
                stopped=(self.runtime/'STOP').exists()
                self.state['controller_status']='STOPPED' if stopped else 'RUNNING'
                if desired_slots and not stopped and not self.state.get('submission_block'):
                    for slot in range(4):
                        if slot not in desired_slots:
                            continue
                        reserved = sum(a['status'] not in ('FINISHED','REJECTED')
                                       for s in self.state['slots'].values() for a in s['attempts'])
                        if reserved >= 4:
                            break
                        row=self.state['slots'][str(slot)]
                        unresolved=[a for a in row['attempts'] if a['status'] not in ('FINISHED','REJECTED')]
                        if row.get('quarantine') or unresolved or self.clock()<row['next_submit_at']:
                            continue
                        outcome=self._submit(slot)
                        # An explicit policy rejection is a shared account limit;
                        # do not issue a storm of four identical rejected RPCs.
                        if outcome=='REJECTED':
                            break
                unresolved=[a for s in self.state['slots'].values() for a in s['attempts']
                            if a['status'] not in ('FINISHED','REJECTED')]
                self.state['inflight_jobs']=len(unresolved)
                if not desired_slots and not unresolved:
                    self.state['controller_status']='FINISHED'
                elif not unresolved and self.state.get('submission_block'):
                    self.state['controller_status']='BLOCKED_SUBMISSION_CONFIGURATION'
                elif not unresolved and desired_slots and all(self.state['slots'][str(s)].get('quarantine') for s in desired_slots):
                    self.state['controller_status']='BLOCKED_WORKER_FAILURE'
                elif any(a['status']=='UNCERTAIN' for a in unresolved):
                    self.state['controller_status']='AWAITING_SUBMISSION_EVIDENCE'
                self._save()
                return self.summary()
        except BlockingIOError:
            return {'controller_status':'EXISTING_SCHEDULER_REUSED','marker':self.marker,'jobs':[]}

    def retry_blocked(self):
        """Explicit operator retry after fixing an allocation/Slurm configuration."""
        with self.acquire_lock():
            self._load()
            self.state.setdefault('explicit_retries', []).append({'at':self.clock(),
                'submission_block':self.state.pop('submission_block',None),
                'slot_quarantines':{k:v.get('quarantine') for k,v in self.state['slots'].items() if v.get('quarantine')}})
            for row in self.state['slots'].values():
                row.pop('quarantine',None);row['worker_failures']=0;row['next_submit_at']=0
            self._save()

    def summary(self):
        state=self.state or _read(self.state_path,{})
        jobs=[]
        for slot in range(4):
            attempts=state.get('slots',{}).get(str(slot),{}).get('attempts',[])
            attempt=attempts[-1] if attempts else {}
            jobs.append({'slot':slot,'job_id':attempt.get('job_id'),'status':attempt.get('status','NOT_SUBMITTED'),
                'name':attempt.get('name'),
                'slurm_state':attempt.get('slurm_state'),'queue_reason':attempt.get('queue_reason'),
                'attempt':attempt.get('sequence',0),'memory':self.launch['memories'][slot],
                'cpus':16,'gpus':1,'partition':self.launch['partition'],
                'excluded_nodes':self.launch.get('exclude'),'node':attempt.get('node'),
                'worker_failures':state.get('slots',{}).get(str(slot),{}).get('worker_failures',0),
                'quarantine':state.get('slots',{}).get(str(slot),{}).get('quarantine')})
        return {'controller_status':state.get('controller_status','NOT_STARTED'),'marker':self.marker,
                'inflight_jobs':state.get('inflight_jobs',0),'jobs':jobs,
                'submission_block':state.get('submission_block'),
                'resources':{'jobs':4,'gpus_total':4,'cpus_total':64,'memory_total':'256G'}}


# Retained import spelling for source interoperability; the implementation always uses four slots.
ThreeJobScheduler = FourJobScheduler
