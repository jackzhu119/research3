"""Pure local fixtures for coordinator locks, exact selection and stage gates."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from p3work.contracts import CHANNELS, validate_signal_contract
from p3work.splits import freeze_subject_split
from tools import continue_local_preparation as coordinator


class CoordinatorFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = b"fixture-only-native-payload"
        self.raw_sha = hashlib.sha256(self.raw).hexdigest()
        md5 = hashlib.md5(self.raw).hexdigest()
        files = []
        # Exactly 598 official-shaped fixture sessions for 62 people.
        file_id = 1
        for person in range(1, 63):
            for session in range(1, 11 if person <= 40 else 10):
                files.append({"id": file_id, "name": f"S{person}_Session_{session}.mat",
                              "size": len(self.raw), "provider_subject": f"S{person}",
                              "session_ordinal": session,
                              "download_url": f"https://ndownloader.figshare.com/files/{file_id}",
                              "computed_md5": md5, "supplied_md5": md5})
                file_id += 1
        self.inventory = self.root / "inventory.json"
        self.write(self.inventory, {"article_id": 13123148, "version": 1,
                                    "license": {"name": "CC BY 4.0"}, "files": files})
        self.qual = coordinator.base.build_plan(self.inventory)
        self.qbatch = self.root / "qualification"
        self.qbatch.mkdir()
        (self.qbatch / ".batch.lock").touch()
        self.write(self.qbatch / "batch_progress.json", {"status": "RUNNING", "completed_file_index": 61,
                                                        "plan_sha256": self.qual["plan_sha256"]})
        # A deliberately stale and invalid completion file must never be read.
        (self.qbatch / "completion.json").write_text("stale PARTIAL, not authoritative JSON")
        self.progress_path = self.root / "status/local_preparation.json"
        self.progress = {}
        roster = [{"participant_id": f"S{i}", "identity_verified": True,
                   "clinic11_present_verified": True, "pure_lr_present_verified": True,
                   "intervention_group": None} for i in range(1, 63)]
        self.split = freeze_subject_split(roster)
        self.split_path = self.root / "split.json"
        self.write(self.split_path, self.split)
        self.selection = coordinator.base.build_plan(self.inventory, stage="meta-duration", split_path=self.split_path)
        self.amendment = coordinator.build_execution_amendment(
            self.selection, qualification_sha256=coordinator.QUALIFICATION_HASH)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_object(self, entry, plan, batch, *, status="COMPLETE", duration=3.1):
        directory = batch / "objects" / str(entry["id"])
        directory.mkdir(parents=True, exist_ok=True)
        rows = [{"participant_id": entry["provider_subject"], "session_id": str(entry["session_ordinal"]),
                 "source_record_id": entry["name"], "source_file_sha256": self.raw_sha,
                 "tasknumber": 1, "label": label, "trial_uid": f"{entry['id']}-{label}",
                 "feedback_end_verified": True, "available_feedback_seconds": duration}
                for label in (0, 1)]
        self.write(directory / "trial_metadata.json", rows)
        self.write(directory / "adapter_audit.json", {"participant_id": entry["provider_subject"],
                   "session_id": str(entry["session_ordinal"]), "source_file_sha256": self.raw_sha,
                   "pure_lr_trials": 2, "mbsr_subject_raw": 0})
        receipt_path = directory / "backup_receipt.json"
        self.write(receipt_path, {"status": "verified", "execute": True, "objects": [{
            "role": "source_development_raw", "sha256": self.raw_sha, "bytes": len(self.raw),
            "status": "verified", "get": {"status": "verified", "complete_body_read": True,
                                           "sha256": self.raw_sha, "bytes_read": len(self.raw)}}]})
        retained = [directory / "trial_metadata.json", directory / "adapter_audit.json", receipt_path]
        state = {"file_id": entry["id"], "name": entry["name"], "plan_sha256": plan["plan_sha256"],
                 "status": status, "raw_integrity": {"bytes": len(self.raw), "sha256": self.raw_sha,
                                                        "md5": entry["computed_md5"]},
                 "backup_receipt": receipt_path.relative_to(self.root).as_posix(),
                 "retained_metadata_sha256": {path.relative_to(self.root).as_posix():
                                               coordinator.base.sha256_file(path)[0] for path in retained}}
        self.write(directory / "state.json", state)
        return state

    def make_qualification(self, unknown=False):
        for index, entry in enumerate(self.qual["files"]):
            self.make_object(entry, self.qual, self.qbatch,
                             status="SCHEMA_BLOCKED" if unknown and index == 0 else "COMPLETE")

    def pending_contract(self):
        value = {"status": "pending_stieger_roster_and_meta_train_duration_audit",
                 "channels": list(CHANNELS), "sfreq_hz": 128, "selected_profile": None,
                 "context_s": None, "crop_s": None, "n_times": None,
                 "protocol_freeze_sha256": None, "window_audit_receipt": None}
        self.write(self.root / "contracts/input_contract.json", value)
        self.write(self.root / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"},
                   "W2": {"status": "NOT_STARTED_DEPENDS_ON_W1"},
                   "W7": {"status": "NOT_FROZEN"}, "W8": {"status": "LOCKED"}})
        return value

    def test_dry_run_never_waits_starts_workers_or_writes_status(self):
        self.write(self.root / "plans/stieger_qualification.json", self.qual)
        with patch.object(coordinator, "QUALIFICATION_HASH", self.qual["plan_sha256"]), \
                patch.object(coordinator.time, "sleep", side_effect=AssertionError("must not sleep")), \
                patch.object(coordinator.subprocess, "Popen", side_effect=AssertionError("must not launch")), \
                patch.object(coordinator.base, "process_file", side_effect=AssertionError("must not process")):
            result = coordinator.coordinate(root=self.root, inventory_path="inventory.json")
        self.assertEqual(result["status"], "DRY_RUN_NO_NETWORK_NO_WAIT")
        self.assertFalse(self.progress_path.exists())
        self.assertFalse((self.root / "audit/local_preparation/.coordinator.lock").exists())

    def test_waits_for_existing_lock_then_verifies_all62_ignoring_stale_completion(self):
        self.make_qualification()
        with (self.qbatch / ".batch.lock").open("a+") as old_worker:
            fcntl.flock(old_worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            waits = []
            def release_existing_worker(seconds):
                waits.append(seconds)
                self.assertFalse((self.qbatch / "participant_split.json").exists())
                fcntl.flock(old_worker, fcntl.LOCK_UN)
            split_path = coordinator.wait_for_qualification(
                self.qual, root=self.root, batch_dir=self.qbatch, inventory=self.inventory,
                progress_path=self.progress_path, progress=self.progress,
                poll_seconds=0.1, sleep=release_existing_worker)
        self.assertEqual(waits, [0.1])
        self.assertTrue(split_path.exists())
        self.assertEqual(json.loads(split_path.read_text())["counts"],
                         {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11})
        self.assertEqual(self.progress["qualification"]["verified_complete_files"], 62)

    def test_unknown_qualification_schema_prevents_split(self):
        self.make_qualification(unknown=True)
        with self.assertRaisesRegex(coordinator.PreparationError, "qualification_unknown_schema_no_split"):
            coordinator.wait_for_qualification(self.qual, root=self.root, batch_dir=self.qbatch,
                inventory=self.inventory, progress_path=self.progress_path, progress=self.progress)
        self.assertFalse((self.qbatch / "participant_split.json").exists())

    def test_incomplete_qualification_is_not_restarted(self):
        with patch.object(coordinator.base, "process_file") as process:
            with self.assertRaisesRegex(coordinator.PreparationError, "qualification_incomplete_no_worker_restart"):
                coordinator.wait_for_qualification(self.qual, root=self.root, batch_dir=self.qbatch,
                    inventory=self.inventory, progress_path=self.progress_path, progress=self.progress)
            process.assert_not_called()

    def test_operational_amendment_preserves_exact_scientific_selection(self):
        original = copy.deepcopy(self.selection)
        self.assertEqual(self.amendment["max_wall_seconds"], 64800)
        self.assertEqual(self.selection["max_wall_seconds"], 14400)
        self.assertEqual(self.selection, original)
        self.assertEqual(self.amendment["files"], self.selection["files"])
        self.assertEqual(self.amendment["max_jobs"], self.selection["max_jobs"])
        self.assertEqual(self.amendment["max_download_bytes"], self.selection["max_download_bytes"])
        self.assertEqual(self.amendment["max_gpu_hours"], 0)
        changed = copy.deepcopy(self.amendment)
        changed["files"].pop()
        with self.assertRaisesRegex(coordinator.PreparationError, "operational_amendment_not_exact"):
            coordinator.validate_execution_amendment(self.selection, changed,
                inventory=self.inventory, split_path=self.split_path,
                qualification_sha256=coordinator.QUALIFICATION_HASH)

    def test_wall_exhaustion_blocks_processing_and_window_freeze(self):
        ticks = iter([0, 64800, 64800])
        with patch.object(coordinator.base, "process_file") as process, \
                patch.object(coordinator.base, "finish_durations") as finish:
            with self.assertRaisesRegex(coordinator.PreparationError, "duration_operational_wall_budget_exhausted"):
                coordinator.execute_duration_selection(self.selection, self.amendment, root=self.root,
                    batch_dir=self.root / "duration", inventory=self.inventory, split_path=self.split_path,
                    progress_path=self.progress_path, progress=self.progress, monotonic=lambda: next(ticks))
            process.assert_not_called()
            finish.assert_not_called()
        self.assertEqual(self.progress["status"], "WALL_BUDGET_EXHAUSTED_NO_STAGE_ADVANCE")
        self.assertEqual(self.progress["duration"]["verified_complete_files"], 0)

    def test_real_fixture_window_evidence_freezes_signal_contract_only(self):
        self.pending_contract()
        batch = self.root / "duration"
        for entry in self.selection["files"]:
            self.make_object(entry, self.selection, batch)
        result = coordinator.base.finish_durations(self.selection, self.root, batch, self.split_path)
        self.assertEqual(result["status"], "WINDOW_FROZEN")
        sha = coordinator.freeze_input_contract(root=self.root, selection=self.selection,
            amendment=self.amendment, batch_dir=batch, split_path=self.split_path)
        contract = json.loads((self.root / "contracts/input_contract.json").read_text())
        self.assertEqual(validate_signal_contract(contract), sha)
        self.assertEqual(contract["selected_profile"], "long")
        self.assertIsNone(contract["protocol_freeze_sha256"])
        tasks = json.loads((self.root / "status/tasks.json").read_text())
        self.assertEqual(tasks["W1"]["status"], "COMPLETE")
        self.assertEqual(tasks["W2"]["status"], "NOT_STARTED_DEPENDS_ON_W1")
        self.assertIsNone(tasks["W7"]["protocol_freeze_sha256"])
        self.assertEqual(tasks["W7"]["status"], "NOT_FROZEN")
        self.assertEqual(tasks["W8"]["status"], "LOCKED")

    def test_both_failed_windows_never_freeze_input_contract(self):
        pending = self.pending_contract()
        batch = self.root / "duration"
        for entry in self.selection["files"]:
            self.make_object(entry, self.selection, batch, duration=1.0)
        result = coordinator.base.finish_durations(self.selection, self.root, batch, self.split_path)
        self.assertEqual(result["status"], "BLOCKED_SHORT_WINDOW_COVERAGE")
        with self.assertRaisesRegex(coordinator.PreparationError, "actual_window_evidence_not_frozen"):
            coordinator.freeze_input_contract(root=self.root, selection=self.selection,
                amendment=self.amendment, batch_dir=batch, split_path=self.split_path)
        self.assertEqual(json.loads((self.root / "contracts/input_contract.json").read_text()), pending)
        self.assertEqual(json.loads((self.root / "status/tasks.json").read_text())["W1"]["status"], "IN_PROGRESS")

    def test_missing_downstream_script_stops_with_minimal_gap(self):
        with patch.object(coordinator.subprocess, "Popen") as process:
            with self.assertRaisesRegex(coordinator.PreparationError, "source_preparation_entrypoint_missing"):
                coordinator.invoke_cpu_script(self.root / "tools/prepare_all_sources.py", root=self.root,
                    output=self.root / "audit/source/progress.json", progress_path=self.progress_path,
                    progress=self.progress, stage="SOURCE_PREPARATION")
            process.assert_not_called()

    def test_coordinator_lock_refuses_duplicate_and_releases_cleanly(self):
        lock = self.root / "audit/local_preparation/.coordinator.lock"
        with coordinator.exclusive_lock(lock):
            with self.assertRaisesRegex(coordinator.PreparationError, "coordinator_already_active"):
                with coordinator.exclusive_lock(lock):
                    self.fail("duplicate coordinator obtained the lock")
        with coordinator.exclusive_lock(lock):
            pass


if __name__ == "__main__":
    unittest.main()
