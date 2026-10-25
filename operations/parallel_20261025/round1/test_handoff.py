import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import fcntl

import handoff


class Runner:
    def __init__(self):
        self.active = True
        self.state = 'RUNNING'
        self.bad_owner = False
        self.fail_queue = False
        self.calls = []
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
            return SimpleNamespace(returncode=0, stdout='', stderr='')
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
        self.put('runs/training_campaign_v1/wall_budget.json', {'interval':None,'durable_wall_seconds':1234})
        self.campaign = {'status':'PAUSED_SCHEDULER_SIGNAL_SAME_JOB','actions':[{'kind':'pilot'}],
            'active':{'kind':'meta_screening','key':'meta-abc','expected_receipt':'runs/meta/abc/receipt.json'},
            'active_process':{'cleanup_verified':True}}
        self.put('runs/training_campaign_v1/campaign_progress.json', self.campaign)
        self.put('.paper3-hpc/jobs/770043/direct_result.json', {'job_id':'770043','status':'PAUSED_FOR_CHECKPOINT_CONTINUATION','returncode':3})
        self.runner = Runner()
    def tearDown(self):
        self.temp.cleanup()
    def put(self, name, value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    def step(self):
        return handoff.checkpoint_handoff_step(self.root,self.state,run=self.runner,account='u')
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


if __name__=='__main__': unittest.main()
