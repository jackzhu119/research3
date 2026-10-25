import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import retire_legacy as legacy


class RetireLegacyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'project'
        self.package = Path(self.temp.name)/'original-direct'
        self.runtime = self.root/'runs/training_campaign_parallel_v1'
        self.base = self.root/'.paper3-hpc/direct-train'
        self.base.mkdir(parents=True)
        self.state = {'root':str(self.root), 'current_job_id':'770043',
                      'history':[{'job_id':'770043','submission':{'comment':'paper3-direct-original'}}]}
        self.write_state()
        self.before = (self.base/'state.json').read_bytes()
        self.addCleanup(self.release_locks)

    def release_locks(self):
        for key, handles in list(legacy._LEGACY_HELD_LOCKS.items()):
            if key[0] == str(self.root):
                for handle in handles.values():
                    handle.close()
                del legacy._LEGACY_HELD_LOCKS[key]

    def write_state(self):
        (self.base/'state.json').write_text(json.dumps(self.state))

    def process(self, pid=121, ticks=1234, **changes):
        row={'pid':pid,'uid':os.getuid(),'state':'S','start_time_ticks':ticks,
             'argv':['/venv/bin/python','-I','-S','-B','-u',str(self.package/'launch.py'),
                     '--execute','--monitor','--root',str(self.root)]}
        row.update(changes)
        return row

    def call(self):
        return legacy.retire_legacy_monitor(self.root,self.package,self.runtime)

    def test_no_monitor_ready_and_old_state_unchanged(self):
        with patch.object(legacy,'_legacy_processes',return_value=[]), patch.object(legacy.os,'kill') as kill:
            row=self.call()
            self.assertEqual(row['status'],'READY')
            self.assertEqual(row['legacy_job_ids'],['770043'])
            self.assertEqual((self.base/'state.json').read_bytes(),self.before)
            self.assertIsNone(legacy._legacy_try_lock(self.base/'launch.lock'))
            self.assertIsNone(legacy._legacy_try_lock(self.base/'monitor.lock'))
            kill.assert_not_called()

    def test_only_exact_monitor_term_with_rechecked_start_ticks(self):
        old=self.process()
        with patch.object(legacy,'_legacy_processes',side_effect=[[old],[]]), \
             patch.object(legacy,'_legacy_read_process',return_value=old), \
             patch.object(legacy.os,'kill') as kill:
            row=self.call()
            kill.assert_called_once_with(old['pid'],signal.SIGTERM)
            self.assertEqual(row['status'],'READY')
            self.assertFalse(row['scientific_processes_signalled'])
            self.assertEqual(row['retired_monitors'][0]['start_time_ticks'],1234)

    def test_pid_reuse_never_signalled(self):
        with patch.object(legacy,'_legacy_processes',side_effect=[[self.process()],[]]), \
             patch.object(legacy,'_legacy_read_process',return_value=self.process(ticks=9999)), \
             patch.object(legacy.os,'kill') as kill:
            self.assertEqual(self.call()['status'],'READY')
            kill.assert_not_called()

    def test_wrong_uid_root_script_and_nonmonitor_are_not_matches(self):
        good=self.process()
        self.assertTrue(legacy._legacy_matches(good,self.root,self.package/'launch.py'))
        self.assertFalse(legacy._legacy_matches(self.process(uid=os.getuid()+1),self.root,self.package/'launch.py'))
        self.assertFalse(legacy._legacy_matches(good,self.root/'other',self.package/'launch.py'))
        self.assertFalse(legacy._legacy_matches(good,self.root,self.package/'worker.py'))
        argv=list(good['argv']);argv.remove('--monitor')
        self.assertFalse(legacy._legacy_matches(self.process(argv=argv),self.root,self.package/'launch.py'))
        self.assertFalse(legacy._legacy_matches(self.process(state='Z'),self.root,self.package/'launch.py'))

    def test_sbatch_step_lock_waits_then_exposes_new_job(self):
        holder=(self.base/'launch.lock').open('a')
        fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with patch.object(legacy,'_legacy_processes',return_value=[]), patch.object(legacy.os,'kill') as kill:
            self.assertEqual(self.call()['reason'],'legacy_launch_step_finishing')
            kill.assert_not_called()
            self.state['current_job_id']='772999'
            self.state['history'].append({'job_id':'772999','submission':{'comment':'paper3-direct-race'}})
            self.write_state()
            holder.close()
            row=self.call()
            self.assertEqual(row['status'],'WAITING')
            self.assertEqual(row['reason'],'legacy_current_job_changed_requires_handoff')
            self.assertEqual(row['existing_current_job_id'],'772999')
            self.assertIn('paper3-direct-race',row['legacy_comments'])
            self.assertTrue(row['retirement_complete'])

    def test_unknown_submission_intent_preserved_for_reconciliation(self):
        self.state['intent']={'comment':'paper3-direct-lost-ack','started_utc':'2026-10-25T00:00:00Z'}
        self.write_state()
        with patch.object(legacy,'_legacy_processes',return_value=[]):
            row=self.call()
        self.assertEqual(row['status'],'WAITING')
        self.assertTrue(row['auto_resubmission_disabled'])
        self.assertEqual(row['unresolved_submission_intents'],[self.state['intent']])
        self.assertEqual(json.loads((self.base/'state.json').read_text()),self.state)

    def test_live_monitor_not_yet_exited_waits_without_kill_escalation(self):
        process=self.process()
        with patch.object(legacy,'_legacy_processes',return_value=[process]), \
             patch.object(legacy,'_legacy_read_process',return_value=process), \
             patch.object(legacy.os,'kill') as kill:
            row=self.call()
            self.assertEqual(row['status'],'WAITING')
            self.assertEqual(row['reason'],'waiting_for_exact_legacy_monitor_exit')
            self.assertEqual(kill.call_args.args,(121,signal.SIGTERM))


if __name__ == '__main__':
    unittest.main()
