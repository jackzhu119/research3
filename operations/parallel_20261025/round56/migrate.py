"""Replace R5/R6's old allocation only after three new held jobs are visible.

No scientific result, checkpoint, plan or old scheduler state is rewritten.
The old controller gets a STOP tombstone only after the replacement queue proof.
The replacement workers also require HANDOFF_READY.json before touching tasks.
"""
from contextlib import ExitStack
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


TERMINAL = {'COMPLETED','CANCELLED','FAILED','TIMEOUT','NODE_FAIL','BOOT_FAIL',
            'OUT_OF_MEMORY','PREEMPTED','REVOKED','DEADLINE'}
ACTIVE = {'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED','RESIZING'}


class MigrationTransient(RuntimeError):
    pass


class MigrationOwnershipError(RuntimeError):
    pass


def _migration_read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as error:
        raise MigrationTransient('metadata_' + type(error).__name__) from None


def _migration_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def _migration_state(value):
    return value.strip().split()[0].rstrip('+').upper() if value.strip() else 'UNKNOWN'


def _migration_memory_mib(value):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:[cn])?', str(value).strip(), re.I)
    if not match:
        return None
    multiplier = {'':1,'K':1/1024,'M':1,'G':1024,'T':1024*1024}[match[2].upper()]
    return round(float(match[1])*multiplier)


class HeldQueueMigration:
    def __init__(self, plan_path, runtime, round_number, launch, *, runner=None, clock=None,
                 emit=None, old_job=None, account=None):
        self.path = Path(plan_path).resolve()
        self.plan = _migration_read(self.path)
        if round_number not in (5,6) or not isinstance(self.plan,dict):
            raise ValueError('R5_or_R6_existing_plan_required')
        self.round = round_number
        self.output = Path(self.plan['output']).resolve()
        if self.path.parent != self.output:
            raise MigrationOwnershipError('original_plan_output_mismatch')
        self.old_runtime = self.output/'runtime'
        self.runtime = Path(runtime).resolve()
        if self.runtime == self.old_runtime:
            raise ValueError('new_runtime_must_be_separate_from_original')
        self.runtime.mkdir(parents=True,exist_ok=True)
        self.launch, self.run = dict(launch), runner or subprocess.run
        self.clock, self.emit = clock or time.time, emit
        self.account = account or launch.get('account') or getpass.getuser()
        self.partition = launch.get('partition','hpc_gpu1')
        self.old_partition = launch.get('old_partition','hpc_gpu')
        if self.partition != 'hpc_gpu1' or self.old_partition != 'hpc_gpu':
            raise ValueError('new_R5_R6_partition_hpc_gpu1_and_original_partition_hpc_gpu_required')
        self.cpus = int(launch.get('cpus',12))
        self.memories = list(launch.get('memories',['64G']*3))
        if len(self.memories)!=3:
            raise ValueError('three_per_job_memory_requests_required')
        self.old_job = str(old_job or {5:'772012',6:'772013'}[round_number])
        payload = json.dumps([self.plan.get('recipe_sha256'),str(self.output)],separators=(',',':'))
        self.old_marker = 'paper3r%d-'%round_number+hashlib.sha256(payload.encode()).hexdigest()[:16]
        old = _migration_read(self.old_runtime/'state.json',{}) or {}
        if old.get('marker') not in (None,self.old_marker):
            raise MigrationOwnershipError('original_runtime_marker_does_not_match_original_plan')
        self.plan_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.state_path = self.runtime/'migration.json'
        self.gate = self.runtime/'HANDOFF_READY.json'
        self.state = _migration_read(self.state_path,{}) or {}
        if self.state and (self.state.get('round')!=self.round or self.state.get('original_plan_sha256')!=self.plan_sha):
            raise MigrationOwnershipError('migration_record_belongs_to_another_round_or_plan')
        self.state.setdefault('round',self.round)
        self.state.setdefault('original_plan_sha256',self.plan_sha)
        self.state.setdefault('old_marker',self.old_marker)
        self.state.setdefault('known_old_jobs',[self.old_job])
        self.state.setdefault('signal_times',{})

    def _emit(self,event,**fields):
        if self.emit:
            self.emit(event,round=self.round,**fields)

    def _save(self,status,**fields):
        self.state.update(status=status,updated_at=self.clock(),**fields)
        _migration_atomic(self.state_path,self.state)
        return dict(self.state)

    def _call(self,command):
        try:
            return self.run(command,capture_output=True,text=True,timeout=25)
        except (OSError,subprocess.TimeoutExpired) as error:
            raise MigrationTransient('scheduler_'+type(error).__name__) from None

    def _queue(self):
        result = self._call(['squeue','--array','--noheader','--user',self.account,
                            '--format=%i|%u|%k|%T|%j|%P|%R'])
        if result.returncode:
            raise MigrationTransient('squeue_failed')
        rows=[]
        for raw in (result.stdout or '').splitlines():
            f=[s.strip() for s in raw.split('|')]
            if len(f)<7:
                continue
            if not re.fullmatch(r'\d+',f[0]):
                # Other papers may use arrays; only regular IDs can belong to
                # either this round's old allocation or our three new jobs.
                continue
            rows.append(dict(job_id=f[0],user=f[1],marker=f[2],state=_migration_state(f[3]),
                             name=f[4],partition=f[5],reason=f[6]))
        return rows

    def _job_info(self,job):
        result=self._call(['scontrol','--oneliner','show','job',str(job)])
        if result.returncode:
            raise MigrationTransient('replacement_job_details_not_available')
        return {key.lower():value for key,value in re.findall(r'([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',result.stdout)}

    def _resource_error(self,code,job,slot,details):
        observed={key:details.get(key) for key in (
            'partition','numcpus','numnodes','reqtres','tres','minmemorynode',
            'minmemorycpu','trespernode','gres')}
        expected={'partition':self.partition,'cpus':self.cpus,'nodes':1,
                  'memory':self.memories[slot],'gpus':1}
        self._emit('REPLACEMENT_VERIFICATION_MISMATCH',reason=code,job_id=str(job),
                   observed=observed,expected=expected)
        raise MigrationOwnershipError(code+'; observed='+json.dumps(observed,sort_keys=True)+
                                      '; expected='+json.dumps(expected,sort_keys=True))

    def _new_jobs(self,summary,queue,*,require_three=True):
        marker=summary.get('marker')
        if not isinstance(marker,str) or not marker.startswith('paper3r%dp-'%self.round) or marker==self.old_marker:
            raise MigrationOwnershipError('replacement_marker_is_not_this_round_threejob_marker')
        if self.state.get('new_marker') not in (None,marker):
            raise MigrationOwnershipError('replacement_marker_changed_during_handoff')
        by_slot={}
        for row in summary.get('jobs',[]):
            if row.get('job_id') is None:
                continue
            slot=int(row['slot'])
            if slot not in range(3) or slot in by_slot or not re.fullmatch(r'\d+',str(row['job_id'])):
                raise MigrationOwnershipError('replacement_jobs_need_three_unique_slots_and_regular_ids')
            by_slot[slot]=row
        if require_three and set(by_slot)!={0,1,2}:
            return None
        queued={r['job_id']:r for r in queue}
        confirmed=[]
        for slot,row in sorted(by_slot.items()):
            job=str(row['job_id'])
            current=queued.get(job)
            if current is None:
                if require_three:return None
                continue
            if (current['user']!=self.account or current['marker']!=marker or
                    current['partition']!=self.partition or
                    not re.fullmatch(re.escape(marker)+r'-s'+str(slot)+r'-a\d+',current['name'])):
                raise MigrationOwnershipError('replacement_queue_identity_mismatch')
            if not require_three:
                confirmed.append({**current,'slot':slot})
                continue
            if current['state']!='PENDING':
                return None
            details=self._job_info(job)
            name=details.get('jobname')
            user=details.get('userid','').split('(',1)[0]
            if (details.get('jobid')!=job or name!=current['name'] or user!=self.account or
                    details.get('comment')!=marker or details.get('partition')!=self.partition):
                raise MigrationOwnershipError('replacement_scontrol_identity_mismatch')
            if details.get('jobstate')!='PENDING' or details.get('priority')!='0':
                return None
            if (details.get('reason') not in ('JobHeldUser','JobHeldAdmin') and
                    'JobHeld' not in current['reason'] and self.launch.get('submit_hold') is not True):
                return None
            tres={}
            for value in details.get('reqtres',details.get('tres','')).split(','):
                if '=' in value:
                    key,item=value.split('=',1);tres[key]=item
            cpus=details.get('numcpus',tres.get('cpu'))
            nodes=details.get('numnodes',tres.get('node'))
            # Slurm 20.11 renders a held fixed-one-node request as NumNodes=1-1.
            # Accept only that exact range or the usual scalar form.
            if cpus!=str(self.cpus) or nodes not in ('1','1-1'):
                self._resource_error('replacement_cpu_or_node_request_mismatch',job,slot,details)
            expected_memory=_migration_memory_mib(self.memories[slot])
            actual_memory=_migration_memory_mib(tres.get('mem',details.get('minmemorynode','')))
            if actual_memory is None and details.get('minmemorycpu'):
                per_cpu=_migration_memory_mib(details['minmemorycpu'])
                actual_memory=None if per_cpu is None else per_cpu*self.cpus
            if actual_memory!=expected_memory:
                self._resource_error('replacement_memory_request_mismatch',job,slot,details)
            gpu=tres.get('gres/gpu')
            if gpu is None:
                text=details.get('trespernode',details.get('gres',''))
                gpu='1' if re.search(r'(?:^|[:,])gpu(?::[^,]*)?:1(?:,|$)',text) else None
            if gpu!='1':
                self._resource_error('replacement_must_request_one_gpu',job,slot,details)
            confirmed.append({**current,'slot':slot,'cpus':self.cpus,'memory':self.memories[slot],
                              'gpus':1,'nodes':1,'held':True})
        if require_three and (len(confirmed)!=3 or len({r['job_id'] for r in confirmed})!=3):
            return None
        self.state['new_marker']=marker
        return confirmed

    def _old_attempts(self):
        old=_migration_read(self.old_runtime/'state.json',{}) or {}
        if old.get('marker') not in (None,self.old_marker):
            raise MigrationOwnershipError('old_controller_marker_changed')
        attempts={}
        for row in old.get('attempts',[]):
            if isinstance(row,dict):attempts[str(row.get('sequence',len(attempts)))]=dict(row)
        for path in sorted((self.old_runtime/'receipts').glob('*.json')):
            row=_migration_read(path,{}) or {}
            if row.get('marker')==self.old_marker:
                attempts.setdefault(str(row.get('sequence',len(attempts))),{}).update(row)
        return list(attempts.values())

    def _old_accounting(self,known,attempts):
        start=time.strftime('%Y-%m-%d',time.gmtime(float(self.plan.get('prepared_at',self.clock()))-86400))
        known={str(job) for job in known}
        by_id={str(a['job_id']):a for a in attempts if a.get('job_id')}
        uncertain_names={a['name'] for a in attempts if a.get('name') and not a.get('job_id')
                         and a.get('status') in ('INTENT','UNCERTAIN')}
        # Read exact durable IDs normally. Only an unacknowledged submission
        # needs a date-bounded name lookup; unrelated user history cannot block
        # this handoff merely because it came from another earlier partition.
        ids=sorted(known|set(by_id))
        base=['sacct','--allocations','--noheader','--parsable2','--user',self.account,
              '--starttime='+start,
              '--format=JobIDRaw,User%64,State%32,ExitCode,JobName%128,Partition%64,Comment%128']
        outputs=[]
        if ids:
            result=self._call(base+['--jobs='+','.join(ids)])
            if result.returncode:raise MigrationTransient('old_accounting_not_available')
            outputs.append(result.stdout or '')
        if uncertain_names:
            result=self._call(base)
            if result.returncode:raise MigrationTransient('unacknowledged_old_accounting_not_available')
            outputs.append(result.stdout or '')
        rows={}
        for raw in '\n'.join(outputs).splitlines():
            f=[s.strip() for s in raw.split('|')]
            if len(f)<7 or not re.fullmatch(r'\d+',f[0]):continue
            if f[0] not in known and f[0] not in by_id and f[4] not in uncertain_names:continue
            observed=dict(job_id=f[0],user=f[1],state=_migration_state(f[2]),
                          exit_code=f[3],name=f[4],partition=f[5],marker=f[6])
            def mismatch(reason,transient=False):
                self._emit('OLD_ACCOUNTING_OBSERVATION_MISMATCH',reason=reason,
                           observed=observed,expected_user=self.account,
                           expected_marker=self.old_marker,live_partition=self.old_partition)
                error=MigrationTransient if transient else MigrationOwnershipError
                raise error(reason+'; observed='+json.dumps(observed,sort_keys=True))
            if f[1] not in ('',self.account):
                mismatch('old_accounting_owner_mismatch')
            if f[6] not in ('','(null)',self.old_marker):
                mismatch('old_accounting_marker_mismatch')
            attempt=by_id.get(f[0])
            expected_name=attempt.get('name') if attempt else None
            name_bound=(expected_name==f[4] or f[4] in uncertain_names)
            if expected_name and expected_name!=f[4]:
                mismatch('old_accounting_durable_job_name_mismatch')
            if f[6] in ('','(null)') and not name_bound:
                mismatch('commentless_old_job_has_no_durable_matching_name')
            if observed['state'] not in TERMINAL:
                if f[5]!=self.old_partition:
                    # Accounting is observation only. It never authorizes a
                    # signal, and an active historical allocation must drain.
                    mismatch('active_old_accounting_unexpected_partition_wait',transient=True)
            else:
                # A terminal record is history, not an allocation to control.
                # The same scientific marker survives resource migrations; old
                # rows may name hpc_gpu1, an earlier partition, or no partition.
                # Ownership was verified above. Preserve the recorded fields
                # without inventing an allocation/compute-completion claim.
                pass
            previous=rows.get(f[0])
            if previous and previous!=observed:
                mismatch('old_accounting_conflicting_rows_wait',transient=True)
            rows[f[0]]=observed
            if observed['state'] in TERMINAL and f[5]!=self.old_partition:
                historical=self.state.setdefault('historical_terminal_jobs',{})
                if historical.get(f[0])!=observed:
                    historical[f[0]]=observed
                    self._emit('OWNED_TERMINAL_HISTORY_ACCEPTED',observed=observed,
                               reason='historical_terminal_partition_preserved')
        return rows

    def _stop_old(self,queue):
        # Persist this proof before creating any old STOP file or signal. A
        # process crash can replay the request but cannot reverse the ordering.
        if not self.state.get('held_replacement_queue_proof'):
            raise MigrationOwnershipError('three_held_replacements_not_proven')
        stop=self.old_runtime/'STOP'
        if not stop.exists():
            stop.parent.mkdir(parents=True,exist_ok=True)
            with stop.open('x') as stream:
                stream.write('Replaced by held three-job migration; see '+str(self.state_path)+'\n')
                stream.flush();os.fsync(stream.fileno())
        old_rows=[r for r in queue if r['marker']==self.old_marker]
        known=set(self.state.get('known_old_jobs',[]))
        for row in old_rows:
            if row['user']!=self.account or row['partition']!=self.old_partition:
                raise MigrationOwnershipError('old_queue_owner_or_partition_mismatch')
            known.add(row['job_id'])
            key=row['job_id']+':'+('pending' if row['state']=='PENDING' else 'checkpoint')
            if self.clock()-self.state['signal_times'].get(key,0)<30:continue
            if row['state']=='PENDING':
                command=['scancel','--state=PENDING',row['job_id']]
                event='OLD_PENDING_ALLOCATION_CANCEL_REQUESTED'
            elif row['state'] in {'RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING'}:
                command=['scancel','--batch','--signal=USR1',row['job_id']]
                event='OLD_RUNNING_ALLOCATION_CHECKPOINT_REQUESTED'
            else:continue
            result=self._call(command)
            self.state['signal_times'][key]=self.clock()
            self._emit(event,job_id=row['job_id'],returncode=result.returncode)
        self.state['known_old_jobs']=sorted(known)
        return old_rows

    def _archive_and_open_gate(self,accounting):
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self.plan_sha:
            raise MigrationOwnershipError('original_plan_changed_during_handoff')
        with ExitStack() as stack:
            for name in ('controller.lock','allocation.lock'):
                path=self.old_runtime/name
                try:handle=stack.enter_context(path.open('rb'))
                except FileNotFoundError:continue
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:return False
            if any(r['marker']==self.old_marker for r in self._queue()):
                return False
            history=self.runtime/'old-allocation-history'
            history.mkdir(parents=True,exist_ok=True)
            for name in ('state.json','launch.json','allocation_state.json','controller.pid.json','STOP'):
                source=self.old_runtime/name
                if source.is_file() and not (history/name).exists():shutil.copy2(source,history/name)
            source=self.output/'allocation_progress.json'
            if source.is_file() and not (history/'allocation_progress.json').exists():
                shutil.copy2(source,history/'allocation_progress.json')
            # The old STOP remains as a tombstone. New parents/workers read only
            # new runtime/STOP and HANDOFF_READY, so it cannot pause replacements.
            gate={'status':'READY','ready':True,'round':self.round,'original_plan_sha256':self.plan_sha,
                  'old_marker':self.old_marker,'new_marker':self.state['new_marker'],
                  'old_terminal_jobs':accounting,'held_replacement_queue_proof':self.state['held_replacement_queue_proof'],
                  'history':str(history),'completed_results_and_checkpoints_rewritten':False,
                  'ready_at':self.clock()}
            _migration_atomic(self.gate,gate)
            self._emit('THREE_JOB_HANDOFF_READY',old_jobs=list(accounting),
                       new_jobs=[r['job_id'] for r in self.state['held_replacement_queue_proof']])
            return True

    def _release(self,summary,queue):
        rows=self._new_jobs(summary,queue,require_three=False)
        released=[]
        for row in rows:
            held = 'JobHeld' in row['reason']
            if row['state']=='PENDING' and not held:
                details = self._job_info(row['job_id'])
                held = details.get('priority')=='0'
            if row['state']=='PENDING' and held:
                result=self._call(['scontrol','release',row['job_id']])
                self._emit('NEW_HELD_JOB_RELEASE_REQUESTED',job_id=row['job_id'],returncode=result.returncode)
                if not result.returncode:released.append(row['job_id'])
            elif row['state']!='PENDING' or not held:
                released.append(row['job_id'])
        return released

    def step(self,summary):
        queue=self._queue()
        ready=_migration_read(self.gate,{}) or {}
        if ready.get('status')=='READY':
            if (ready.get('round')!=self.round or ready.get('original_plan_sha256')!=self.plan_sha or
                    ready.get('new_marker')!=summary.get('marker')):
                raise MigrationOwnershipError('ready_gate_identity_mismatch')
            self.state['new_marker']=summary['marker']
            released=self._release(summary,queue)
            return self._save('READY',released_jobs=released)
        replacement=self._new_jobs(summary,queue)
        if replacement is None:
            return self._save('WAITING_THREE_HELD_REPLACEMENTS')
        self.state['held_replacement_queue_proof']=replacement
        self.state.setdefault('held_replacements_confirmed_at',self.clock())
        self._save('THREE_HELD_REPLACEMENTS_CONFIRMED')
        old_rows=self._stop_old(queue)
        attempts=self._old_attempts()
        known=set(self.state['known_old_jobs'])
        for a in attempts:
            if a.get('job_id') and a.get('status') not in ('FINISHED','REJECTED'):known.add(str(a['job_id']))
        observed=self._old_accounting(known,attempts)
        for row in observed.values():
            if row['state'] not in TERMINAL:known.add(row['job_id'])
        unresolved=[a for a in attempts if not a.get('job_id') and a.get('status') in ('INTENT','UNCERTAIN')
                    and not any(r['name']==a.get('name') for r in observed.values())]
        self.state['known_old_jobs']=sorted(known)
        if old_rows or unresolved or any(job not in observed or observed[job]['state'] not in TERMINAL for job in known):
            return self._save('WAITING_OLD_ALLOCATION_TERMINAL',unresolved_old_names=[a.get('name') for a in unresolved])
        if not self._archive_and_open_gate({job:row for job,row in sorted(observed.items())
                                           if row['state'] in TERMINAL}):
            return self._save('WAITING_OLD_CONTROLLER_AND_ALLOCATION_LOCKS')
        released=self._release(summary,self._queue())
        return self._save('READY',released_jobs=released)
