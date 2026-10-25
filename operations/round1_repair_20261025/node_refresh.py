"""Interrupt only a verified pre-training child inside its existing allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    return json.loads(Path(path).read_text())


def option(argv, flag):
    try:return argv[argv.index(flag)+1]
    except (ValueError, IndexError):return None


def process(pid, proc_root=Path('/proc')):
    folder=Path(proc_root)/str(pid)
    fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
    return {'uid':folder.stat().st_uid,'ppid':int(fields[1]),'start_ticks':int(fields[19]),
            'state':fields[0],'argv':[x.decode(errors='replace') for x in (folder/'cmdline').read_bytes().split(b'\0') if x]}


def refresh(runtime, job_id, slot, *, proc_root=Path('/proc'), allocation_job=None, kill=os.kill):
    runtime=Path(runtime).resolve();job_id=str(job_id)
    if str(allocation_job if allocation_job is not None else os.environ.get('SLURM_JOB_ID',''))!=job_id:
        return {'status':'WAIT_NEXT_CHILD','reason':'allocation_job_mismatch','job_id':job_id}
    path=runtime/'worker-status'/('slot-%d.json'%slot)
    try:
        row=read(path)
        if row.get('job_id')!=job_id or int(row.get('slot',-1))!=slot:
            raise ValueError('worker_slot_or_job_changed')
        child, parent=int(row['child_pid']),int(row['pid'])
        if min(child,parent)<=1 or child==parent:raise ValueError('invalid_child_pid')
        action_key,token=row['action_key'],row['claim_token']
        if not isinstance(action_key,str) or not isinstance(token,str) or not token.isalnum():raise ValueError('invalid_claim')
        expected=runtime/'actions'/hashlib.sha256(action_key.encode()).hexdigest()/'attempts'/token/'action.json'
        action=read(expected)
        claim=action['claim']
        if (action['action']['key']!=action_key or claim.get('token')!=token or
                str(claim.get('job_id'))!=job_id or int(claim.get('slot',-1))!=slot or int(claim.get('pid',-1))!=parent):
            raise ValueError('action_claim_changed')
        claimed=float(claim['claimed_utc'])
        observed=process(child,proc_root)
        argv=observed['argv']
        if (observed['uid']!=os.getuid() or observed['ppid']!=parent or observed['state'] in ('Z','X','x') or
                '--child' not in argv or option(argv,'--action-file')!=str(expected) or
                option(argv,'--runtime-dir')!=str(runtime) or option(argv,'--slot')!=str(slot)):
            raise ValueError('child_process_identity_mismatch')
        root=runtime.parents[1]
        receipt=Path(action['action']['expected_receipt'])
        if receipt.is_absolute() or '..' in receipt.parts:raise ValueError('invalid_receipt_path')
        if row.get('expected_receipt') and row['expected_receipt']!=str(receipt):raise ValueError('receipt_changed')
        progress=root/receipt.parent/'progress.json'
        if progress.is_file():
            native=read(progress)
            # Any progress written by this claim is conservatively treated as
            # entered numerical work, including OUTER_ATTEMPT_DISPATCHED.
            if progress.stat().st_mtime>=claimed:
                return {'status':'CURRENT_CHILD_KEPT','reason':'native_progress_written_after_claim',
                        'job_id':job_id,'slot':slot,'child_pid':child,'native_status':native.get('status'),
                        'outer_updates_completed':native.get('outer_updates_completed')}
        receipt_path=root/receipt
        if receipt_path.is_file() and receipt_path.stat().st_mtime>=claimed:
            native_receipt=read(receipt_path)
            if (str(native_receipt.get('status','')).endswith('COMPLETE') or native_receipt.get('training_started') is True):
                return {'status':'CURRENT_CHILD_KEPT','reason':'native_training_or_completion_observed','job_id':job_id,'slot':slot}
        again=read(path);last=process(child,proc_root)
        keys=('job_id','slot','pid','child_pid','action_key','claim_token')
        identity=('uid','ppid','start_ticks','argv')
        if any(again.get(k)!=row.get(k) for k in keys) or any(last[k]!=observed[k] for k in identity) or last['state'] in ('Z','X','x'):raise ValueError('process_or_claim_changed_before_signal')
        # Recheck numerical progress immediately before the signal. A phase
        # transition during the process-identity reads must keep the child.
        if progress.is_file() and progress.stat().st_mtime>=claimed:
            current=read(progress)
            return {'status':'CURRENT_CHILD_KEPT','reason':'native_progress_became_current_before_signal',
                    'job_id':job_id,'slot':slot,'child_pid':child,'native_status':current.get('status')}
        if receipt_path.is_file() and receipt_path.stat().st_mtime>=claimed:
            current=read(receipt_path)
            if str(current.get('status','')).endswith('COMPLETE') or current.get('training_started') is True:
                return {'status':'CURRENT_CHILD_KEPT','reason':'native_training_became_current_before_signal','job_id':job_id,'slot':slot}
        kill(child,signal.SIGINT)
        return {'status':'PRETRAIN_CHILD_REFRESH_REQUESTED','signal':'SIGINT','job_id':job_id,
                'slot':slot,'child_pid':child,'supervisor_pid':parent,'action_key':action_key,
                'allocation_cancelled':False,'checkpoint_or_scientific_files_changed':False}
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        return {'status':'WAIT_NEXT_CHILD','reason':str(error)[:160] if isinstance(error,ValueError) else type(error).__name__,
                'job_id':job_id,'slot':slot,'allocation_cancelled':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True);parser.add_argument('--job-id',required=True)
    parser.add_argument('--slot',type=int,choices=(0,1,2),required=True)
    args=parser.parse_args();print(json.dumps(refresh(args.runtime,args.job_id,args.slot),ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
