import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import fcntl
from unittest.mock import patch

import handoff


class Runner:
    def __init__(self):
        self.active = True
        self.state = 'RUNNING'
        self.bad_owner = False
        self.fail_queue = False
        self.calls = []
        self.signal_returncode = 0
        self.on_plain_cancel = None
    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if command[0] == 'squeue':
            if self.fail_queue:
                return SimpleNamespace(returncode=1, stdout='', stderr='controller unavailable')
            text = ('770043|u|%s|%s|paper3-train-direct|hpc_gpu1|gpu4\n' %
                    ('another-project' if self.bad_owner else handoff.OLD_COMMENT,self.state)) if self.active else ''
            return SimpleNamespace(returncode=0, stdout=text, stderr='')
        if command[0] == 'sacct':
            return SimpleNamespace(returncode=0, stdout='770043|u|FAILED|3:0|paper3-train-direct|hpc_gpu1|%s|gpu4\n'%handoff.OLD_COMMENT, stderr='')
        if command[0] == 'scancel':
            if command == ['scancel','770043'] and self.on_plain_cancel:
                self.on_plain_cancel()
            code=self.signal_returncode if '--signal=USR1' in command else 0
            return SimpleNamespace(returncode=code, stdout='', stderr='')
        raise AssertionError(command)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'project'
        self.root.mkdir()
        self.state = Path(self.temp.name)/'controller'
        self.folder = self.root/'runs/meta/abc'
        self.folder.mkdir(parents=True)
        self.checkpoint = self.folder/'outer-0049.pt'
        self.checkpoint.write_bytes(b'FIXTURE_CHECKPOINT_NOT_A_REAL_MODEL')
        self.put('runs/meta/abc/latest_checkpoint.json', {'path':'outer-0049.pt',
            'relative_to_job_output':True,'sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),'outer_cursor':49})
        self.put('runs/meta/abc/progress.json', {'status':'INTERRUPTED_NOT_COMPLETE','outer_updates_completed':49})
        self.put('runs/meta/abc/wall_budget.json', {'interval':None})
        (self.folder/'.job.lock').touch()
        self.put('runs/training_campaign_v1/wall_budget.json', {'interval':None,'durable_wall_seconds':1234})
        self.campaign = {'status':'PAUSED_SCHEDULER_SIGNAL_SAME_JOB','actions':[{'kind':'pilot'}],
            'active':{'kind':'meta_screening','key':'meta-abc','expected_receipt':'runs/meta/abc/receipt.json'},
            'active_process':{'cleanup_verified':True}}
        self.put('runs/training_campaign_v1/campaign_progress.json', self.campaign)
        self.put('.paper3-hpc/jobs/770043/direct_result.json', {'job_id':'770043','status':'PAUSED_FOR_CHECKPOINT_CONTINUATION','returncode':3})
        self.runner = Runner()
        self.replacements={'ready':True,'marker':'paper3r1p-fixture',
                           'jobs':[{'job_id':str(900001+i),'slot':i} for i in range(3)]}
    def tearDown(self):
        self.temp.cleanup()
    def put(self, name, value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def step(self,**kwargs):
        return handoff.checkpoint_handoff_step(self.root,self.state,run=self.runner,account='u',**kwargs)
    def timed_step(self,now,proof=True):
        with patch.object(handoff.time,'time',return_value=now):
            return self.step(replacement_queue_proof=self.replacements if proof else None)
    def plain_cancelled(self):
        return ['scancel','770043'] in self.runner.calls
    def test_usr1_only_then_real_quiescence_keeps_artifacts(self):
        before={p.relative_to(self.root).as_posix():(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        first=self.step();self.assertEqual(first['status'],'WAITING_OLD_JOB')
        self.assertIn(['scancel','--batch','--signal=USR1','770043'],self.runner.calls)
        self.step();self.assertEqual(sum(c[0]=='scancel' for c in self.runner.calls),1)
        self.runner.active=False
        value=self.step();self.assertEqual(value['status'],'READY')
        self.assertEqual(value['quiescence']['checkpoint']['outer_cursor'],49)
        after={p.relative_to(self.root).as_posix():(p.read_bytes(),p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_mismatched_marker_never_signaled(self):
        self.runner.bad_owner=True
        with self.assertRaises(handoff.HandoffOwnershipError):self.step()
        self.assertFalse(any(c[0]=='scancel' for c in self.runner.calls))
    def test_scheduler_failure_is_not_empty_queue(self):
        self.runner.fail_queue=True
        with self.assertRaises(handoff.HandoffTransient):self.step()
        self.assertFalse(any(c[0]=='scancel' for c in self.runner.calls))
    def test_open_wall_interval_waits(self):
        self.runner.active=False
        self.put('runs/meta/abc/wall_budget.json',{'interval':{'owner':'still-active'}})
        self.assertEqual(self.step()['status'],'WAITING_QUIESCENT')
    def test_cleanup_ack_required(self):
        self.runner.active=False
        self.campaign['active_process']['cleanup_verified']=False
        self.put('runs/training_campaign_v1/campaign_progress.json',self.campaign)
        self.assertEqual(self.step()['status'],'WAITING_QUIESCENT')
    def test_held_original_lock_waits(self):
        self.runner.active=False
        lock=self.root/'runs/.training_campaign_v1.lock'
        with lock.open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertEqual(self.step()['status'],'WAITING_QUIESCENT')
    def test_bad_checkpoint_digest_preserved(self):
        self.runner.active=False
        self.checkpoint.write_bytes(b'OTHER_BYTES')
        with self.assertRaises(handoff.HandoffTransient):self.step()
        self.assertEqual(self.checkpoint.read_bytes(),b'OTHER_BYTES')
    def test_unstarted_meta_does_not_require_fake_checkpoint(self):
        self.runner.active=False
        (self.folder/'latest_checkpoint.json').unlink();self.checkpoint.unlink()
        self.put('runs/meta/abc/progress.json',{'outer_updates_completed':0,'outer_attempts_dispatched':0})
        value=self.step();self.assertEqual(value['status'],'READY');self.assertIsNone(value['quiescence']['checkpoint'])
    def test_pending_cancellation_is_filtered_and_running_race_gets_usr1(self):
        self.runner.state='PENDING';self.step()
        self.assertIn(['scancel','--state=PENDING','770043'],self.runner.calls)
        self.assertFalse(any('--signal=USR1' in c for c in self.runner.calls))
        self.runner.state='RUNNING';self.step()
        self.assertIn(['scancel','--batch','--signal=USR1','770043'],self.runner.calls)
    def test_saved_native_checkpoint_drains_only_legacy_backup_then_waits_all_locks(self):
        self.timed_step(1000)
        direct=self.root/'.paper3-hpc/direct-train.lock'
        campaign=self.root/'runs/.training_campaign_v1.lock'
        def before_cancel():
            record=json.loads((self.state/'handoff.json').read_text())
            self.assertEqual(record['status'],'CHECKPOINT_SAVED_LEGACY_BACKUP_ONLY')
            proof=record['legacy_bookkeeping_drain']['native_checkpoint_proof']
            self.assertTrue(proof['active_process_cleanup_verified'])
            self.assertTrue(proof['campaign_wall_interval_closed'])
            self.assertTrue(proof['active_kernel_wall_interval_closed'])
            self.assertEqual(proof['checkpoint']['outer_cursor'],49)
        self.runner.on_plain_cancel=before_cancel
        with direct.open('a') as d,campaign.open('a') as c:
            fcntl.flock(d,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
            value=self.timed_step(1016)
            self.assertEqual(value['status'],'WAITING_OLD_BOOKKEEPING_TERMINAL')
            self.assertTrue(self.plain_cancelled())
            self.runner.active=False
            self.assertEqual(self.timed_step(1017)['status'],'WAITING_QUIESCENT')
        value=self.timed_step(1018)
        self.assertEqual(value['status'],'READY')
        self.assertEqual(value['quiescence']['checkpoint']['sha256'],hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
    def test_bookkeeping_cancel_waits_at_least_fifteen_seconds(self):
        self.timed_step(1000);self.timed_step(1014)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_successful_usr1(self):
        self.runner.signal_returncode=1
        self.timed_step(1000);self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_three_queued_replacements(self):
        self.timed_step(1000,proof=False);self.timed_step(1020,proof=False)
        self.assertFalse(self.plain_cancelled())
        self.replacements['jobs'][2]['job_id']=self.replacements['jobs'][1]['job_id']
        self.timed_step(1021)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_existing_latest_checkpoint(self):
        self.timed_step(1000);(self.folder/'latest_checkpoint.json').unlink()
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_native_group_cleanup(self):
        self.timed_step(1000)
        self.campaign['active_process']['cleanup_verified']=False
        self.put('runs/training_campaign_v1/campaign_progress.json',self.campaign)
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_closed_campaign_ledger(self):
        self.timed_step(1000)
        self.put('runs/training_campaign_v1/wall_budget.json',{'interval':{'still':'open'}})
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_closed_native_ledger(self):
        self.timed_step(1000)
        self.put('runs/meta/abc/wall_budget.json',{'interval':{'still':'open'}})
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_never_skips_native_job_lock(self):
        self.timed_step(1000)
        with (self.folder/'.job.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_saved_cursor_match(self):
        self.timed_step(1000)
        self.put('runs/meta/abc/progress.json',{'outer_updates_completed':50})
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())
    def test_bookkeeping_cancel_requires_actual_active_meta_task(self):
        self.timed_step(1000)
        self.campaign['active']=None
        self.put('runs/training_campaign_v1/campaign_progress.json',self.campaign)
        self.timed_step(1020)
        self.assertFalse(self.plain_cancelled())


if __name__=='__main__': unittest.main()
