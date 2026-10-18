"""Synthetic/mock only: no real network, raw source data, cache, or credentials."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import source_preparation_batch as batch
from tools.prepare_source_cache import begin_owned_cache_restore, recover_owned_cache_restore


DATA = b"synthetic-fixture-official-body"


def response(body=b"", *, status=200, size=None, content_range=None, etag='"opaque-not-sha256"'):
    obj = io.BytesIO(body)
    obj.status = status
    obj.headers = {"Content-Length": str(len(body) if size is None else size), "ETag": etag,
                   "Last-Modified": "Wed, 12 Oct 2022 07:29:37 GMT"}
    if content_range:
        obj.headers["Content-Range"] = content_range
    return obj


def fake_backup(*, manifest_path, root, prefix, receipt_path, execute):
    files = json.loads(manifest_path.read_text())["files"]
    result = {"schema_version": 1, "operation": "paper3_explicit_manifest_backup", "status": "verified",
              "execute": execute, "prefix": prefix, "verification": "full_get_body_sha256", "objects": []}
    for item in files:
        result["objects"].append({**item, "key": prefix + item["path"], "status": "verified",
            "put": {"status": "created", "http_status": 200, "request_body_bytes": item["bytes"], "request_body_sha256": item["sha256"]},
            "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                    "sha256": item["sha256"], "bytes_read": item["bytes"], "declared_content_length": item["bytes"]}})
    batch.atomic_json(receipt_path, result)
    return result


def fake_native(dataset, path, *, provider_subject, session_id, source_version, expected_sha256):
    base = {"dataset": dataset, "role": "source", "provider_subject": str(provider_subject),
            "session_id": str(session_id), "source_version": source_version, "source_file_sha256": expected_sha256,
            "source_file_bytes": path.stat().st_size}
    rows = [{**base, "canonical_label": "left_hand", "trial_uid": hashlib.sha256(expected_sha256.encode()).hexdigest(),
             "task_end_verified": True, "task_end_sample": 100, "analysis_anchor_sample": 0,
             "original_metadata_extra": "preserve-verbatim"}]
    return {"summary": {**base, "n_native_lr_trials": 1}, "trials": rows}


def synthetic_w1(root):
    batch.atomic_json(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}})
    batch.atomic_json(root / "contracts/input_contract.json", {"synthetic": "gate mocked separately"})


class SourceBatchTest(unittest.TestCase):
    def setUp(self):
        self.plan = batch.build_plan()
        self.entry = self.plan["files"][0]
        self.contract_sha = "1" * 64

    def test_exact_candidate_math_and_no_inferred_evidence_or_network(self):
        with patch.object(batch.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            plan = batch.build_plan()
            batch.validate_plan(plan)
        self.assertEqual(len(plan["files"]), 505)
        self.assertEqual({k: sum(f["dataset"] == k for f in plan["files"]) for k in batch.VERSIONS},
                         {"BNCI2014_001": 18, "Cho2017": 52, "Lee2019_MI": 108, "PhysionetMI": 327})
        self.assertEqual(len({f["object_id"] for f in plan["files"]}), 505)
        self.assertTrue(all(f["bytes"] is None and f["sha256"] is None and not f["existence_verified"] for f in plan["files"]))
        bad = json.loads(json.dumps(plan))
        bad["files"][0]["url"] = "https://unapproved.invalid/file"
        with self.assertRaisesRegex(batch.BatchError, "exact_registered"):
            batch.validate_plan(bad)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(batch.BatchError):
                batch.build_plan(max_download_bytes=value)

    def test_w1_and_signal_gate_before_raw_network_or_progress(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch.atomic_json(root / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"}})
            with patch.object(batch.urllib.request, "urlopen", side_effect=AssertionError("must gate first")):
                with self.assertRaisesRegex(batch.BatchError, "w1_complete"):
                    batch.execute_plan(self.plan, root=root)
            self.assertFalse((root / "audit").exists())
            synthetic_w1(root)
            with self.assertRaisesRegex(batch.BatchError, "valid_signal"):
                batch.execute_plan(self.plan, root=root)
            self.assertFalse((root / "audit").exists())

    def test_adapter_code_sha_binds_plan_and_rejects_stale_or_tampered_identity(self):
        with patch.object(batch, "file_digest", return_value="a"*64) as code_hash:
            first = batch.build_plan()
            code_hash.assert_called_once_with(batch.SOURCE_ADAPTER_PATH)
            batch.validate_plan(first)
            tampered = dict(first, adapter_code_sha256="b"*64)
            with self.assertRaisesRegex(batch.BatchError, "exact_registered"):
                batch.validate_plan(tampered)
        with patch.object(batch, "file_digest", return_value="b"*64):
            second = batch.build_plan()
            batch.validate_plan(second)
            with self.assertRaisesRegex(batch.BatchError, "exact_registered"):
                batch.validate_plan(first)
        self.assertEqual(first["adapter_code_sha256"], "a"*64)
        self.assertEqual(second["adapter_code_sha256"], "b"*64)
        self.assertNotEqual(first["plan_sha256"], second["plan_sha256"])

    def test_actual_head_status_length_and_both_validators_required(self):
        for status, size, etag in ((404, 3, '"v"'), (200, 0, '"v"'), (200, 3, None)):
            with self.subTest(status=status, size=size, etag=etag):
                with self.assertRaises(batch.BatchError):
                    batch.head_object(self.entry, opener=lambda *a, **kw: response(size=size, status=status, etag=etag))
        requests = []
        def opener(req, **kw):
            requests.append(req)
            return response(size=len(DATA))
        observed = batch.head_object(self.entry, opener=opener)
        self.assertEqual(requests[0].get_method(), "HEAD")
        self.assertEqual(observed["bytes"], len(DATA))
        self.assertFalse(observed["etag_is_sha256"])

    def test_range_requires_exact_206_and_headers_before_appending(self):
        observed = batch.head_object(self.entry, opener=lambda *a, **kw: response(size=6))
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d) / "x.partial", Path(d) / "x.mat"
            for status, crange in ((200, None), (206, "bytes 1-5/6"), (206, "bytes 2-4/6")):
                partial.write_bytes(b"ab")
                with self.subTest(status=status, crange=crange), self.assertRaisesRegex(batch.BatchError, "exact_206"):
                    batch.download_object(self.entry, observed, partial, final,
                        opener=lambda *a, **kw: response(b"cdef", status=status, content_range=crange))
                self.assertEqual(partial.read_bytes(), b"ab")
                self.assertFalse(final.exists())
            seen = []
            def opener(req, **kw):
                seen.append(req)
                return response(b"cdef", status=206, content_range="bytes 2-5/6")
            info = batch.download_object(self.entry, observed, partial, final, opener=opener)
            self.assertEqual(final.read_bytes(), b"abcdef")
            self.assertEqual(seen[0].get_header("Range"), "bytes=2-")
            self.assertEqual(info["sha256"], hashlib.sha256(b"abcdef").hexdigest())
            self.assertEqual(info["download_bytes_this_invocation"], 4)
            self.assertNotEqual(info["sha256"], observed["response_headers"]["ETag"])

    def test_initial_empty_partial_and_incomplete_get_preserve_bytes(self):
        observed = batch.head_object(self.entry, opener=lambda *a, **kw: response(size=6))
        with tempfile.TemporaryDirectory() as d:
            partial, final = Path(d)/"x.partial", Path(d)/"x.mat"
            partial.write_bytes(b"")
            with self.assertRaisesRegex(batch.BatchError, "incomplete_body"):
                batch.download_object(self.entry, observed, partial, final, opener=lambda *a, **kw: response(b"abc", size=6))
            self.assertEqual(partial.read_bytes(), b"abc")
            self.assertFalse(final.exists())

    def fixture_object(self, root, *, owned=True, sample=False):
        batch_dir = root / "audit/fixture"
        raw = batch.raw_staging_path(root, self.plan, self.entry)
        if sample:
            raw = root / "data/source_schema" / self.entry["dataset"] / self.entry["file"]
            batch.atomic_json(root / "audit/source_schema/raw_inventory.json", {"status": "downloaded_schema_specimens_only", "files": [{
                **self.entry, "project_relative_path": raw.relative_to(root).as_posix(), "http_status": 200,
                "bytes": len(DATA), "expected_bytes": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest()}]})
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(DATA)
        if owned and not sample:
            batch.atomic_json(batch_dir / "objects" / self.entry["object_id"] / "state.json", {
                "identity": list(batch.identity(self.entry)), "object_id": self.entry["object_id"],
                "plan_sha256": self.plan["plan_sha256"], "input_contract_sha256": self.contract_sha,
                "status": "RAW_VERIFIED", "raw_path": raw.relative_to(root).as_posix(), "temporary_raw_owned": True})
        return batch_dir, raw

    def process(self, root, batch_dir, **kw):
        with patch.object(batch, "check_space"):
            return batch.process_file(self.entry, root=root, batch_dir=batch_dir, plan=self.plan,
                contract_sha256=self.contract_sha, backup_fn=kw.pop("backup_fn", fake_backup),
                audit_fn=kw.pop("audit_fn", fake_native), opener=kw.pop("opener", lambda *a, **kw: response(size=len(DATA))), **kw)

    def restore_item(self, state):
        return {**self.entry, "object_uid": self.entry["object_id"], "path": state["raw_path"],
                "sha256": state["raw_integrity"]["sha256"], "bytes": state["raw_integrity"]["bytes"],
                "backup_receipt": state["backup_receipt"], "temporary_raw_owned": True,
                "raw_storage": "verified_r2_only"}

    def test_verified_archive_removes_owned_only_and_replays_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch_dir, raw = self.fixture_object(root)
            state = self.process(root, batch_dir)
            self.assertEqual(state["status"], "COMPLETE")
            self.assertFalse(raw.exists())
            self.assertTrue(state["raw_removed_after_verified_backup"])
            self.assertTrue(batch.verify_completed(state, root, entry=self.entry, plan=self.plan))
            rows = [json.loads(line) for line in (root / state["native_trials"]["path"]).read_text().splitlines()]
            self.assertEqual(rows[0]["task_end_sample"], 100)
            self.assertEqual(rows[0]["original_metadata_extra"], "preserve-verbatim")
            replay = self.process(root, batch_dir, backup_fn=lambda **kw: self.fail("backup repeated"),
                                  opener=lambda *a, **kw: self.fail("network repeated"))
            self.assertEqual(replay, state)
            # Missing 504 candidates can never become a complete inventory.
            batch.atomic_json(root / "audit/source_schema/evidence_manifest.json", {"files": []})
            partial = batch.aggregate_inventory(self.plan, root=root, batch_dir=batch_dir)
            self.assertEqual(partial["status"], "SOURCE_RAW_INVENTORY_PARTIAL")
            self.assertFalse(partial["provider_roster_verified"])
            self.assertEqual(partial["actual_verified_objects"], 1)
            self.assertEqual(partial["files"][0]["raw_storage"], "verified_r2_only")
            prior_sha = batch.file_digest(batch_dir / "source_native_inventory.json")
            batch.aggregate_inventory(self.plan, root=root, batch_dir=batch_dir)
            self.assertEqual(batch.file_digest(batch_dir / "source_native_inventory.json"), prior_sha)

    def test_existing_specimen_and_unowned_staging_are_always_preserved(self):
        for sample in (False, True):
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                batch_dir, raw = self.fixture_object(root, owned=False, sample=sample)
                state = self.process(root, batch_dir)
                self.assertEqual(raw.read_bytes(), DATA)
                self.assertFalse(state["temporary_raw_owned"])
                self.assertFalse(state["raw_removed_after_verified_backup"])

    def test_backup_failure_or_incomplete_get_keeps_owned_raw_and_suppresses_sdk_message(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                batch_dir, raw = self.fixture_object(root)
                def backup(**kw):
                    if not corrupt:
                        raise RuntimeError("secret-looking exception must never be emitted")
                    result = fake_backup(**kw)
                    result["objects"][0]["get"]["complete_body_read"] = False
                    batch.atomic_json(kw["receipt_path"], result)
                    return result
                with self.assertRaises(batch.BatchError):
                    self.process(root, batch_dir, backup_fn=backup)
                state_path = batch_dir / "objects" / self.entry["object_id"] / "state.json"
                self.assertEqual(raw.read_bytes(), DATA)
                self.assertNotIn("secret-looking", state_path.read_text())
                self.assertEqual(json.loads(state_path.read_text())["status"], "FAILED")

    def test_native_failure_archived_unknown_prevents_complete_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch_dir, raw = self.fixture_object(root)
            state = self.process(root, batch_dir, audit_fn=lambda *a, **kw: (_ for _ in ()).throw(ValueError("sensitive native details")))
            self.assertEqual(state["status"], "SCHEMA_BLOCKED")
            self.assertFalse(state["native_schema_audited"])
            self.assertFalse(raw.exists())
            self.assertTrue(batch.verify_completed(state, root))
            audit = (root / state["adapter_audit"]["path"]).read_text()
            self.assertNotIn("sensitive native details", audit)
            self.assertIsNone(json.loads(audit)["identity_channel_task_qualification"])

    def test_changed_head_or_unowned_partial_is_preserved_and_not_appended(self):
        for prior_head in (False, True):
            with self.subTest(prior_head=prior_head), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                raw = batch.raw_staging_path(root, self.plan, self.entry)
                raw.parent.mkdir(parents=True)
                partial = raw.with_name(raw.name + ".partial")
                partial.write_bytes(b"preserve")
                batch_dir = root / "audit/fixture"
                if prior_head:
                    observed = batch.head_object(self.entry, opener=lambda *a, **kw: response(size=len(DATA), etag='"old"'))
                    batch.atomic_json(batch_dir / "objects" / self.entry["object_id"] / "state.json", {
                        "identity": list(batch.identity(self.entry)), "plan_sha256": self.plan["plan_sha256"],
                        "input_contract_sha256": self.contract_sha, "status": "FAILED", "head_observation": observed})
                with self.assertRaisesRegex(batch.BatchError, "changed|prior_head"):
                    self.process(root, batch_dir)
                self.assertEqual(partial.read_bytes(), b"preserve")
                self.assertFalse(raw.exists())

    def test_budget_accumulates_actual_failed_read_bytes_and_wall_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"
            plan = batch.build_plan(max_download_bytes=10, max_wall_seconds=100)
            budget = batch.ResourceBudget(plan, path, clock=lambda: 5, utc_clock=lambda: 1005, boot_id="fixture")
            token = budget.reserve_transfer(7, object_id="SOFTWARE_TEST_DOUBLE")
            budget.consume(7, transfer_id=token)
            budget.finish_transfer(token, complete_body_read=True)
            with self.assertRaisesRegex(batch.BatchError, "byte_budget"):
                budget.reserve_transfer(4, object_id="must_not_open_HTTP")
            budget.close()
            replay = batch.ResourceBudget(plan, path, clock=lambda: 7, utc_clock=lambda: 1007, boot_id="fixture")
            self.assertEqual(replay.data["download_bytes_read"], 7)
            self.assertEqual(replay.data["download_bytes_charged"], 7)
            replay.close()
            saved = json.loads(path.read_text()); saved["active_wall_elapsed_s"] = 100
            batch.atomic_json(path, saved)
            exhausted = batch.ResourceBudget(plan, path, clock=lambda: 9, utc_clock=lambda: 1009, boot_id="fixture")
            with self.assertRaisesRegex(batch.BatchError, "wall_budget"):
                exhausted.check()
            exhausted.close()

    def test_complete_GET_body_is_reserved_before_opener_and_success_releases_only_probe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = batch.build_plan(max_download_bytes=50)
            budget = batch.ResourceBudget(plan, root / "budget.json")
            observation = batch.head_object(self.entry, opener=lambda *a, **k: response(size=len(DATA)))
            def opener(*args, **kwargs):
                saved = json.loads((root / "budget.json").read_text())
                ledger = saved["byte_scopes"][batch.ResourceBudget.OFFICIAL]
                self.assertEqual(ledger["charged_bytes"], len(DATA) + 1)
                self.assertEqual(ledger["observed_bytes"], 0)
                return response(DATA)
            batch.download_object(self.entry, observation, root / "raw.partial", root / "raw.mat", budget=budget, opener=opener)
            self.assertEqual(budget.data["download_bytes_read"], len(DATA))
            self.assertEqual(budget.data["download_bytes_charged"], len(DATA))
            budget.close()

    def test_actual_SIGKILL_between_body_read_and_consume_preserves_full_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"
            script = '''
import io, os, signal, sys
from pathlib import Path
from tools.source_preparation_batch import ResourceBudget, build_plan
b = ResourceBudget(build_plan(max_download_bytes=20, max_wall_seconds=100), Path(sys.argv[1]),
    clock=lambda:10., utc_clock=lambda:1010., boot_id='fixture')
b.reserve_transfer(10, object_id='SOFTWARE_TEST_DOUBLE', probe_bytes=1)
io.BytesIO(b'0123456789').read(4)
os.kill(os.getpid(), signal.SIGKILL)
'''
            child = subprocess.run([sys.executable, "-c", script, str(path)], cwd=batch.REPO, capture_output=True, timeout=20)
            self.assertEqual(child.returncode, -signal.SIGKILL)
            plan = batch.build_plan(max_download_bytes=20, max_wall_seconds=100)
            replay = batch.ResourceBudget(plan, path, clock=lambda:20., utc_clock=lambda:1020., boot_id="fixture")
            self.assertEqual(replay.data["download_bytes_read"], 0)
            self.assertEqual(replay.data["download_bytes_charged"], 11)
            self.assertEqual(replay.data["active_wall_elapsed_s"], 10)
            self.assertEqual(replay.data["byte_scopes"][batch.ResourceBudget.OFFICIAL]["attempts"][0]["status"],
                             "UNCLEAN_FULL_RESERVATION_REMAINS_CHARGED")
            with self.assertRaisesRegex(batch.BatchError, "byte_budget"):
                replay.reserve_transfer(10, object_id="second_attempt")
            replay.close()

    def test_failed_body_remains_charged_and_R2_full_GET_has_independent_cap(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"; plan = batch.build_plan(max_download_bytes=10)
            budget = batch.ResourceBudget(plan, path, max_r2_readback_bytes=3)
            token = budget.reserve_transfer(8, object_id="official_partial")
            budget.consume(2, transfer_id=token)
            budget.finish_transfer(token, complete_body_read=False)
            self.assertEqual((budget.data["download_bytes_read"], budget.data["download_bytes_charged"]), (2, 8))
            r2 = budget.reserve_transfer(3, object_id="R2_FULL_GET", scope=batch.ResourceBudget.R2_BACKUP)
            budget.consume(3, transfer_id=r2, scope=batch.ResourceBudget.R2_BACKUP)
            budget.finish_transfer(r2, complete_body_read=True, scope=batch.ResourceBudget.R2_BACKUP)
            self.assertEqual(budget.data["download_bytes_charged"], 8)
            with self.assertRaisesRegex(batch.BatchError, "byte_budget"):
                budget.reserve_transfer(1, object_id="R2_over_cap", scope=batch.ResourceBudget.R2_BACKUP)
            budget.close()

    def test_explicit_operational_amendments_preserve_identity_counters_and_resume_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "budget.json"; plan = batch.build_plan(max_download_bytes=10, max_wall_seconds=10)
            budget = batch.ResourceBudget(plan, path); token = budget.reserve_transfer(8, object_id="fixture")
            budget.finish_transfer(token); original_identity = budget.data["resource_identity_sha256"]; budget.close()
            with self.assertRaisesRegex(batch.BatchError, "amendment"):
                batch.ResourceBudget(plan, path, max_download_bytes=20)
            amended = batch.ResourceBudget(plan, path, max_download_bytes=20, max_wall_seconds=20,
                                           amendment_note="Explicit operational fixture extension")
            self.assertEqual(amended.data["resource_identity_sha256"], original_identity)
            self.assertEqual(amended.data["download_bytes_charged"], 8)
            amended.close()
            resumed = batch.ResourceBudget(plan, path)
            self.assertEqual(resumed.data["byte_scopes"][batch.ResourceBudget.OFFICIAL]["limit_bytes"], 20)
            self.assertEqual(resumed.wall.max_wall_seconds, 20)
            resumed.close()

    def test_R2_proxy_rejects_wrong_length_before_read_and_bounds_overlong_EOF_probe(self):
        class Body(io.BytesIO):
            def __init__(self, data):
                super().__init__(data); self.requests = []
            def read(self, count=-1):
                self.requests.append(count)
                return super().read(count)
        class Client:
            def __init__(self, body, length): self.body, self.length = body, length
            def get_object(self, **kwargs):
                return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ContentLength": self.length, "Body": self.body}
        with tempfile.TemporaryDirectory() as d:
            budget = batch.ResourceBudget(batch.build_plan(max_download_bytes=10), Path(d) / "resource.json")
            scope = batch.ResourceBudget.R2_BACKUP
            token = budget.reserve_transfer(5, object_id="bad_declared_length", scope=scope, probe_bytes=1)
            wrong = Body(b"0123456789")
            proxy = batch.BudgetedR2Client(Client(wrong, 10), budget, token, {"fixture-key": 5}, scope=scope)
            with self.assertRaisesRegex(batch.BatchError, "ContentLength_before_read"):
                proxy.get_object(Bucket="SOFTWARE_TEST_DOUBLE", Key="fixture-key")
            self.assertEqual(wrong.requests, [])
            budget.finish_transfer(token, scope=scope)
            token = budget.reserve_transfer(5, object_id="overlong_body", scope=scope, probe_bytes=1)
            too_long = Body(b"0123456789")
            proxy = batch.BudgetedR2Client(Client(too_long, 5), budget, token, {"fixture-key": 5}, scope=scope)
            body = proxy.get_object(Bucket="SOFTWARE_TEST_DOUBLE", Key="fixture-key")["Body"]
            self.assertEqual(body.read(100), b"01234")
            with self.assertRaisesRegex(batch.BatchError, "exceeds_exact_reserved"):
                body.read(100)
            self.assertEqual(too_long.requests, [5, 1])
            self.assertEqual(budget.data["byte_scopes"][scope]["observed_bytes"], 6)
            self.assertEqual(budget.data["byte_scopes"][scope]["charged_bytes"], 12)
            budget.finish_transfer(token, scope=scope); budget.close()

    def test_native_backup_helper_proxy_counts_GET_chunks_once_and_checks_upload_wall(self):
        class Client:
            def __init__(self): self.data = {}
            def put_object(self, **kwargs):
                self.data[kwargs["Key"]] = kwargs["Body"].read()
                return {"ResponseMetadata": {"HTTPStatusCode": 200}}
            def get_object(self, **kwargs):
                body = self.data[kwargs["Key"]]
                return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ContentLength": len(body), "Body": io.BytesIO(body)}
        native_backup = batch.run_backup
        def fixture_bucket_backup(**kwargs):
            return native_backup(**kwargs, bucket="SOFTWARE_TEST_DOUBLE_BUCKET")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); batch_dir, raw = self.fixture_object(root)
            budget = batch.ResourceBudget(self.plan, root / "resource.json")
            with patch.object(batch, "create_client", return_value=Client()), patch.object(batch, "_NATIVE_BACKUP_IMPLEMENTATION", fixture_bucket_backup):
                state = self.process(root, batch_dir, budget=budget, backup_fn=fixture_bucket_backup)
            self.assertEqual(state["status"], "COMPLETE")
            ledger = budget.data["byte_scopes"][batch.ResourceBudget.R2_BACKUP]
            self.assertEqual(ledger["observed_bytes"], state["r2_full_get_bytes_verified"])
            self.assertEqual(ledger["charged_bytes"], ledger["observed_bytes"])
            self.assertEqual(ledger["attempts"][0]["reserved_bytes"], ledger["observed_bytes"] + 3)
            self.assertEqual(budget.data["download_bytes_read"], 0)
            budget.close()

    def test_independent_object_failure_continues_and_bounded_stop_never_advances(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic_w1(root)
            calls = []
            def process(entry, **kw):
                calls.append(entry["object_id"])
                if len(calls) == 1:
                    raise batch.BatchError("fixture_failure")
                return {"status": "COMPLETE"}
            with patch.object(batch, "validate_signal_contract", return_value=self.contract_sha), patch.object(batch, "aggregate_inventory", return_value={
                    "status": "SOURCE_RAW_INVENTORY_PARTIAL", "actual_verified_objects": 0}):
                result = batch.execute_plan(self.plan, root=root, max_new_files=2, process_fn=process)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["archived_schema_verified"], 1)
            self.assertEqual(result["status"], "PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE")
            self.assertFalse(result["w2_complete"])
            self.assertEqual(json.loads((root / "status/tasks.json").read_text())["W2"]["status"], "PENDING")

    def test_corrupt_retained_metadata_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch_dir, raw = self.fixture_object(root)
            state = self.process(root, batch_dir)
            (root / state["native_trials"]["path"]).write_text("modified")
            with self.assertRaisesRegex(batch.BatchError, "metadata_integrity"):
                batch.verify_completed(state, root)

    def test_aggregate_recovers_active_interrupted_cache_restore_without_changing_inventory_sha(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch_dir, raw = self.fixture_object(root)
            state = self.process(root, batch_dir)
            batch.atomic_json(root / "audit/source_schema/evidence_manifest.json", {"files": []})
            batch.aggregate_inventory(self.plan, root=root, batch_dir=batch_dir)
            inventory_sha = batch.file_digest(batch_dir / "source_native_inventory.json")
            marker = begin_owned_cache_restore(root, self.restore_item(state))
            # Simulate a complete mock GET committed just before SIGKILL, so the
            # context manager's normal cleanup did not run.
            raw.write_bytes(DATA)
            result = batch.aggregate_inventory(self.plan, root=root, batch_dir=batch_dir)
            self.assertEqual(result["actual_verified_objects"], 1)
            self.assertFalse(raw.exists())
            self.assertEqual(json.loads(marker.read_text())["status"], "RAW_REMOVED_AFTER_VERIFIED_BACKUP")
            saved = json.loads((batch_dir / "objects" / self.entry["object_id"] / "state.json").read_text())
            self.assertTrue(saved["recovered_owned_cache_restore"])
            self.assertEqual(batch.file_digest(batch_dir / "source_native_inventory.json"), inventory_sha)

    def test_reappeared_raw_without_active_marker_or_with_changed_bytes_is_preserved(self):
        for condition in ("missing_marker", "changed_body", "already_removed_marker"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                batch_dir, raw = self.fixture_object(root)
                state = self.process(root, batch_dir)
                item = self.restore_item(state)
                if condition != "missing_marker":
                    begin_owned_cache_restore(root, item)
                raw.write_bytes(DATA)
                if condition == "changed_body":
                    raw.write_bytes(b"modified mock body must be preserved")
                elif condition == "already_removed_marker":
                    self.assertTrue(recover_owned_cache_restore(root, item))
                    raw.write_bytes(DATA)
                expected = raw.read_bytes()
                with self.assertRaisesRegex(batch.BatchError, "ownership_or_integrity_unverified_raw_preserved"):
                    self.process(root, batch_dir, opener=lambda *a, **kw: self.fail("must not download"))
                self.assertEqual(raw.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
