"""Failure-closed staging and replay without real network or credentials."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import stieger_metadata_batch as batch
from p3work.splits import freeze_subject_split


class FakeSession:
    rows = [{"trial_uid": "fixture-only"}]

    def __init__(self, digest):
        self.digest = digest

    def schema_audit(self):
        return {"participant_id": "S1", "session_id": "1", "pure_lr_trials": 2,
                "endpoint_errors": 0, "source_file_sha256": self.digest, "mbsr_subject_raw": 1}


def fake_backup(*, manifest_path, root, prefix, receipt_path, execute):
    files = json.loads(manifest_path.read_text())["files"]
    objects = [{**f, "status": "verified", "get": {"status": "verified", "complete_body_read": True,
                                                        "sha256": f["sha256"], "bytes_read": f["bytes"]}} for f in files]
    result = {"status": "verified", "execute": execute, "objects": objects, "prefix": prefix}
    batch.atomic_json(receipt_path, result)
    return result


class MetadataBatchTest(unittest.TestCase):
    @staticmethod
    def own_temporary(root, batch_dir):
        batch.atomic_json(batch_dir / "objects/123/state.json",
                          {"file_id": 123, "name": "S1_Session_1.mat", "plan_sha256": "fixture",
                           "status": "RAW_VERIFIED", "temporary_raw_owned": True,
                           "raw_path": "data/stieger2021/w1-batch-temp/123/S1_Session_1.mat"})

    @staticmethod
    def entry(data):
        return {"id": 123, "name": "S1_Session_1.mat", "provider_subject": "S1", "session_ordinal": 1,
                "size": len(data), "computed_md5": hashlib.md5(data).hexdigest(),
                "download_url": "https://ndownloader.figshare.com/files/123"}

    def test_success_deletes_only_owned_raw_and_resumes_without_redownload(self):
        data = b"synthetic-native-fixture"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "data/stieger2021/w1-batch-temp/123/S1_Session_1.mat"
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            self.own_temporary(root, root/"audit/fixture")
            entry = self.entry(data)
            plan = {"plan_sha256": "fixture", "stage": "qualification"}
            with patch.object(batch, "check_space"):
                state = batch.process_file(entry, root=root, batch_dir=root/"audit/fixture", plan=plan,
                                           backup_fn=fake_backup, load_fn=lambda *a, **kw: FakeSession(kw["expected_sha256"]))
                self.assertEqual(state["status"], "COMPLETE")
                self.assertFalse(target.exists())
                with patch.object(batch, "download_object", side_effect=AssertionError("must not download")):
                    replay = batch.process_file(entry, root=root, batch_dir=root/"audit/fixture", plan=plan,
                                                backup_fn=lambda **kw: self.fail("must not upload again"))
                self.assertEqual(replay, state)

    def test_existing_schema_sample_is_preserved_and_failed_backup_keeps_temp(self):
        data = b"synthetic-native-fixture"
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                rel = "raw" if existing else "w1-batch-temp/123"
                target = root / "data/stieger2021" / rel / "S1_Session_1.mat"
                target.parent.mkdir(parents=True)
                target.write_bytes(data)
                if not existing:
                    self.own_temporary(root, root/"audit/fixture")
                def backup(**kw):
                    return fake_backup(**kw) if existing else {"status": "failed", "execute": True}
                with patch.object(batch, "check_space"):
                    if existing:
                        state = batch.process_file(self.entry(data), root=root, batch_dir=root/"audit/fixture", plan={"plan_sha256": "fixture", "stage": "qualification"},
                                                   backup_fn=backup, load_fn=lambda *a, **kw: FakeSession(kw["expected_sha256"]))
                        self.assertTrue(state["existing_raw_preserved"])
                    else:
                        with self.assertRaisesRegex(batch.BatchError, "r2_put"):
                            batch.process_file(self.entry(data), root=root, batch_dir=root/"audit/fixture", plan={"plan_sha256": "fixture", "stage": "qualification"},
                                               backup_fn=backup, load_fn=lambda *a, **kw: FakeSession(kw["expected_sha256"]))
                self.assertEqual(target.read_bytes(), data)

    def test_schema_failure_is_archived_as_unknown_and_never_qualified(self):
        data = b"bad-schema-synthetic-fixture"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "data/stieger2021/w1-batch-temp/123/S1_Session_1.mat"
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            self.own_temporary(root, root/"audit/fixture")
            with patch.object(batch, "check_space"):
                state = batch.process_file(self.entry(data), root=root, batch_dir=root/"audit/fixture", plan={"plan_sha256": "fixture", "stage": "qualification"},
                                           backup_fn=fake_backup, load_fn=lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad schema")))
            self.assertEqual(state["status"], "SCHEMA_BLOCKED")
            self.assertFalse(target.exists())
            audit = json.loads((root/"audit/fixture/objects/123/adapter_audit.json").read_text())
            self.assertIsNone(audit["identity_channel_task_qualification"])
            result = batch.finish_qualification({"files": [self.entry(data)]}, root, root/"audit/fixture", root/"not-used.json")
            self.assertEqual(result["status"], "ALL_CANDIDATES_AUDITED_SCHEMA_UNKNOWN_NO_SPLIT")
            self.assertFalse((root/"audit/fixture/participant_split.json").exists())

    def test_resume_requires_exact_http_range_before_appending(self):
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
            partial.write_bytes(b"ab")
            response = io.BytesIO(b"abcdef")
            response.status = 200
            response.headers = {}
            with patch.object(batch.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(batch.BatchError, "exact_206"):
                    batch.download_object(self.entry(b"abcdef"), partial, final)
            self.assertEqual(partial.read_bytes(), b"ab")
            self.assertFalse(final.exists())

    def test_verified_backup_deletion_interruption_resumes_and_reuses_across_stage(self):
        data = b"synthetic-native-fixture"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            qualification_dir = root/"audit/stieger_w1/batches/fixture"
            target = root/"data/stieger2021/w1-batch-temp/123/S1_Session_1.mat"
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            self.own_temporary(root, qualification_dir)
            entry = self.entry(data)
            with patch.object(batch, "check_space"):
                state = batch.process_file(entry, root=root, batch_dir=qualification_dir, plan={"plan_sha256": "fixture", "stage": "qualification"},
                                           backup_fn=fake_backup, load_fn=lambda *a, **kw: FakeSession(kw["expected_sha256"]))
            state["status"] = "BACKUP_VERIFIED"
            batch.atomic_json(qualification_dir/"objects/123/state.json", state)
            with patch.object(batch, "download_object", side_effect=AssertionError("must not redownload")):
                resumed = batch.process_file(entry, root=root, batch_dir=qualification_dir, plan={"plan_sha256": "fixture", "stage": "qualification"})
            self.assertEqual(resumed["status"], "COMPLETE")
            next_dir = root/"audit/stieger_w1/batches/meta-fixture"
            with patch.object(batch, "build_plan", return_value={"plan_sha256": "fixture"}):
                self.assertTrue(batch.reuse_qualification_object(entry, root=root, batch_dir=next_dir,
                                                                plan={"stage": "meta-duration", "plan_sha256": "meta-fixture"}, inventory_path=root/"unused"))
            reused = json.loads((next_dir/"objects/123/state.json").read_text())
            self.assertTrue(batch.verify_completed(reused, root))
            self.assertTrue(reused["reused_qualification_object"])

    def test_unmarked_preexisting_temporary_original_is_preserved(self):
        data = b"synthetic-native-fixture"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root/"data/stieger2021/w1-batch-temp/123/S1_Session_1.mat"
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            with patch.object(batch, "check_space"):
                state = batch.process_file(self.entry(data), root=root, batch_dir=root/"audit/fixture", plan={"plan_sha256": "fixture", "stage": "qualification"},
                                           backup_fn=fake_backup, load_fn=lambda *a, **kw: FakeSession(kw["expected_sha256"]))
            self.assertTrue(state["existing_raw_preserved"])
            self.assertEqual(target.read_bytes(), data)

    def test_actual_official_inventory_dry_plan_and_meta_partition_boundary(self):
        inventory = batch.REPO/"data_catalog/stieger2021/official_inventory.json"
        if not inventory.exists():
            self.skipTest("Local pinned Stieger catalog is excluded from public git; no network fallback")
        plan = batch.build_plan(inventory)
        self.assertEqual(plan["file_count"], 62)
        self.assertEqual(plan["total_official_bytes"], 38395648006)
        actual = batch.read_inventory(inventory)
        supplied_only = [f for f in actual["files"] if f["name"].endswith(".mat") and not f["computed_md5"]]
        self.assertEqual(len(supplied_only), 2)
        for entry in supplied_only:
            self.assertEqual(batch.official_md5(entry)[1], "supplied_md5_only_computed_missing")
        # This split is an ephemeral synthetic qualification fixture; no actual
        # study roster or split is written by this test.
        roster = [{"participant_id": f"S{i}", "identity_verified": True, "clinic11_present_verified": True,
                   "pure_lr_present_verified": True, "intervention_group": "fixture"} for i in range(1, 63)]
        split = freeze_subject_split(roster)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"synthetic-split-fixture.json"
            batch.atomic_json(p, split)
            meta = batch.build_plan(inventory, stage="meta-duration", split_path=p)
        self.assertEqual({f["provider_subject"] for f in meta["files"]}, set(split["partitions"]["meta_train"]))
        self.assertFalse({f["provider_subject"] for f in meta["files"]} & set(split["partitions"]["internal_holdout"]))

    def test_exhausted_wall_budget_stops_before_next_file_and_preserves_raw(self):
        inventory = batch.REPO/"data_catalog/stieger2021/official_inventory.json"
        if not inventory.exists():
            self.skipTest("No local pinned official catalog; no network fallback")
        plan = batch.build_plan(inventory)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch_dir = root/"audit/stieger_w1/batches"/plan["plan_sha256"]
            batch.atomic_json(batch_dir/"batch_progress.json", {"active_wall_elapsed_s": 14400})
            raw = root/"data/unverified-original.partial"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"unverified fixture must remain")
            with patch.object(batch, "process_file", side_effect=AssertionError("wall budget must stop before next download")):
                result = batch.execute_plan(plan, root=root, batch_dir=batch_dir, inventory_path=inventory)
            self.assertEqual(result["status"], "WALL_BUDGET_EXHAUSTED_NO_STAGE_ADVANCE")
            self.assertTrue(raw.exists())


if __name__ == "__main__":
    unittest.main()
