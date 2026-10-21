"""Failure-closed staging and replay without real network or credentials."""
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
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

    @staticmethod
    def response(data, status=200, content_range=None):
        response = io.BytesIO(data)
        response.status = status
        response.headers = {} if content_range is None else {"Content-Range": content_range}
        return response

    def test_transient_5xx_retries_original_URL_then_verifies_official_bytes(self):
        data = b"abcdef"
        error = urllib.error.HTTPError("DO_NOT_LOG_URL", 503, "DO_NOT_LOG_BODY", {}, io.BytesIO(b"private"))
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
            entry = self.entry(data)
            with patch.object(batch.urllib.request, "urlopen", side_effect=[error, self.response(data)]) as opening, \
                    patch.object(batch.time, "sleep") as sleeping:
                result = batch.download_object(entry, partial, final)
            self.assertEqual(final.read_bytes(), data)
            self.assertEqual(result["md5"], entry["computed_md5"])
            self.assertEqual(result["download_attempts"], 2)
            self.assertEqual(result["download_retry_events"][0]["http_status"], 503)
            self.assertEqual(result["download_retry_events"][0]["phase"], "open")
            self.assertEqual(sleeping.call_count, 1)
            self.assertTrue(all(call.args[0].full_url == entry["download_url"] for call in opening.call_args_list))
            self.assertNotIn("DO_NOT_LOG", json.dumps(result))

    def test_incomplete_read_keeps_exception_partial_and_recomputes_exact_range(self):
        data = b"abcdef"
        first = self.response(b"")
        first.read = unittest.mock.Mock(side_effect=[b"ab", http.client.IncompleteRead(b"c", 3)])
        second = self.response(b"def", 206, "bytes 3-5/6")
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
            with patch.object(batch.urllib.request, "urlopen", side_effect=[first, second]) as opening, \
                    patch.object(batch.time, "sleep"):
                result = batch.download_object(self.entry(data), partial, final)
            self.assertEqual(final.read_bytes(), data)
            self.assertIsNone(opening.call_args_list[0].args[0].get_header("Range"))
            self.assertEqual(opening.call_args_list[1].args[0].get_header("Range"), "bytes=3-")
            self.assertEqual(result["download_retry_events"][0]["retained_partial_bytes"], 3)
            self.assertEqual(result["download_retry_events"][0]["phase"], "read")

    def test_nonretry_403_reports_only_numeric_HTTP_and_fixed_phase(self):
        sensitive = "DO_NOT_LOG_URL_HEADER_OR_BODY"
        error = urllib.error.HTTPError(sensitive, 403, sensitive, {"Authorization": sensitive}, io.BytesIO(sensitive.encode()))
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
            partial.write_bytes(b"ab")
            with patch.object(batch.urllib.request, "urlopen", side_effect=error) as opening, \
                    patch.object(batch.time, "sleep") as sleeping:
                with self.assertRaises(batch.BatchError) as caught:
                    batch.download_object(self.entry(b"abcdef"), partial, final)
            self.assertEqual(str(caught.exception), "download_http_status_403_phase_open_attempt_1")
            self.assertNotIn(sensitive, str(caught.exception))
            self.assertEqual(opening.call_count, 1)
            sleeping.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"ab")
            self.assertFalse(final.exists())

    def test_midread_retry_never_appends_range_200_and_preserves_partial(self):
        first = self.response(b"")
        first.read = unittest.mock.Mock(side_effect=[b"ab", TimeoutError("DO_NOT_LOG")])
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
            with patch.object(batch.urllib.request, "urlopen", side_effect=[first, self.response(b"abcdef")]) as opening, \
                    patch.object(batch.time, "sleep") as sleeping:
                with self.assertRaisesRegex(batch.BatchError, "exact_206.*http_200_phase_headers_attempt_2"):
                    batch.download_object(self.entry(b"abcdef"), partial, final)
            self.assertEqual(partial.read_bytes(), b"ab")
            self.assertFalse(final.exists())
            self.assertEqual(opening.call_count, 2)
            self.assertEqual(sleeping.call_count, 1)

    def test_midread_network_timeout_or_connection_interrupt_retries_only_remaining_range(self):
        failures = [TimeoutError("PRIVATE_REASON"), ConnectionResetError("PRIVATE_REASON"),
                    ConnectionAbortedError("PRIVATE_REASON"), BrokenPipeError("PRIVATE_REASON"),
                    http.client.RemoteDisconnected("PRIVATE_REASON"), urllib.error.URLError("PRIVATE_REASON")]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as d:
                partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
                first = self.response(b"")
                first.read = unittest.mock.Mock(side_effect=[b"ab", failure])
                second = self.response(b"cdef", 206, "bytes 2-5/6")
                with patch.object(batch.urllib.request, "urlopen", side_effect=[first, second]) as opening, \
                        patch.object(batch.time, "sleep"):
                    result = batch.download_object(self.entry(b"abcdef"), partial, final)
                self.assertEqual(final.read_bytes(), b"abcdef")
                self.assertEqual(result["download_attempts"], 2)
                self.assertEqual(result["download_retry_events"][0]["phase"], "read")
                self.assertEqual(result["download_retry_events"][0]["retained_partial_bytes"], 2)
                self.assertEqual(opening.call_args_list[1].args[0].get_header("Range"), "bytes=2-")
                self.assertNotIn("PRIVATE_REASON", json.dumps(result))

    def test_local_flush_timeout_or_connection_error_never_retries_network(self):
        for failure in (TimeoutError("PRIVATE_DISK_REASON"), ConnectionResetError("PRIVATE_DISK_REASON")):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as d:
                partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
                with patch.object(batch.urllib.request, "urlopen", return_value=self.response(b"abcdef")) as opening, \
                        patch.object(batch.os, "fsync", side_effect=failure), patch.object(batch.time, "sleep") as sleeping:
                    with self.assertRaisesRegex(batch.BatchError, "download_local_.*phase_flush_attempt_1") as caught:
                        batch.download_object(self.entry(b"abcdef"), partial, final)
                self.assertEqual(opening.call_count, 1)
                sleeping.assert_not_called()
                self.assertEqual(partial.read_bytes(), b"abcdef")
                self.assertFalse(final.exists())
                self.assertNotIn("PRIVATE_DISK_REASON", str(caught.exception))

    def test_transient_retries_exhaust_at_four_attempts_with_bounded_waits(self):
        for kind in ("http", "network", "timeout", "short_eof"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
                def opening(*args, **kwargs):
                    if kind == "http":
                        raise urllib.error.HTTPError("PRIVATE_URL", 429, "PRIVATE_REASON", {}, io.BytesIO())
                    if kind == "network":
                        raise urllib.error.URLError("PRIVATE_REASON")
                    if kind == "timeout":
                        raise TimeoutError("PRIVATE_REASON")
                    return self.response(b"")
                with patch.object(batch.urllib.request, "urlopen", side_effect=opening) as requests, \
                        patch.object(batch.time, "sleep") as sleeping:
                    with self.assertRaisesRegex(batch.BatchError, "download_retry_exhausted.*attempt_4") as caught:
                        batch.download_object(self.entry(b"abcdef"), partial, final)
                self.assertEqual(requests.call_count, 4)
                self.assertEqual([call.args[0] for call in sleeping.call_args_list], [1, 2, 4])
                self.assertNotIn("PRIVATE", str(caught.exception))
                self.assertFalse(final.exists())

    def test_MD5_or_size_error_and_wrong_range_are_not_retried(self):
        for kind in ("md5", "overrun", "wrong_range"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                partial, final = Path(d)/"sample.partial", Path(d)/"sample.mat"
                if kind == "wrong_range":
                    partial.write_bytes(b"ab")
                    response = self.response(b"cdef", 206, "bytes 1-5/6")
                else:
                    response = self.response(b"xxxxxx" if kind == "md5" else b"abcdefg")
                with patch.object(batch.urllib.request, "urlopen", return_value=response) as opening, \
                        patch.object(batch.time, "sleep") as sleeping:
                    with self.assertRaises(batch.BatchError):
                        batch.download_object(self.entry(b"abcdef"), partial, final)
                self.assertEqual(opening.call_count, 1)
                sleeping.assert_not_called()
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
