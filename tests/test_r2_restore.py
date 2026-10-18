"""Restores must require audited evidence and publish only verified whole files."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("r2_restore", TOOLS / "r2_restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)
sys.path.pop(0)


class TrackedBody(io.BytesIO):
    eof = False

    def read(self, amount=-1):
        value = super().read(amount)
        if value == b"":
            self.eof = True
        return value


class MockS3:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.body = None

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        self.body = TrackedBody(self.content)
        return {"Body": self.body, "ContentLength": len(self.content),
                "ResponseMetadata": {"HTTPStatusCode": 200}}


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "restore"
        self.root.mkdir()
        self.source = self.base / "backup.json"
        self.report = self.base / "restore-report.json"
        self.content = b"real-stream-test-data\x00" * 100000
        digest = hashlib.sha256(self.content).hexdigest()
        self.row = {
            "path": "data/stieger.bin", "sha256": digest, "bytes": len(self.content),
            "role": "source_development_raw", "dataset": "Stieger2021",
            "key": "paper3/backup/data/stieger.bin", "status": "verified",
            "put": {"status": "created", "http_status": 200,
                    "request_body_bytes": len(self.content), "request_body_sha256": digest},
            "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                    "bytes_read": len(self.content), "sha256": digest,
                    "declared_content_length": len(self.content)},
        }
        self.receipt = {
            "schema_version": 1, "operation": "paper3_explicit_manifest_backup",
            "execute": True, "status": "verified", "verification": "full_get_body_sha256",
            "prefix": "paper3/backup/", "objects": [self.row],
        }
        self.client = MockS3(self.content)
        self.write_receipt()

    def write_receipt(self):
        self.source.write_text(json.dumps(self.receipt))

    def run_restore(self, **kwargs):
        arguments = dict(backup_receipt_path=self.source, root=self.root,
                         report_path=self.report, paths=[self.row["path"]],
                         execute=True, client=self.client, bucket="test-bucket")
        arguments.update(kwargs)
        return restore.run_restore(**arguments)

    def test_full_get_is_verified_before_atomic_commit(self):
        before = self.source.read_bytes()
        result = self.run_restore()
        self.assertEqual(result["status"], "restored_verified")
        self.assertEqual((self.root / self.row["path"]).read_bytes(), self.content)
        self.assertTrue(self.client.body.eof)
        self.assertTrue(self.client.body.closed)
        self.assertEqual(result["objects"][0]["bytes_read"], len(self.content))
        self.assertEqual(result["objects"][0]["get_sha256"], self.row["sha256"])
        self.assertEqual(list(self.root.rglob("*.partial")), [])
        self.assertEqual(self.source.read_bytes(), before)

    def test_wrong_get_hash_retains_partial_without_complete_destination(self):
        self.client.content = b"x" + self.content[1:]
        result = self.run_restore()
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.root / self.row["path"]).exists())
        partials = list(self.root.rglob("*.partial"))
        self.assertEqual(len(partials), 1)
        self.assertEqual(partials[0].read_bytes(), self.client.content)
        self.assertEqual(result["objects"][0]["error_code"], "remote_integrity_mismatch")

    def test_existing_different_file_blocks_network_and_is_not_overwritten(self):
        target = self.root / self.row["path"]
        target.parent.mkdir()
        target.write_bytes(b"existing-different")
        result = self.run_restore()
        self.assertEqual(result["error_code"], "different_local_file_not_overwritten")
        self.assertEqual(target.read_bytes(), b"existing-different")
        self.assertEqual(self.client.calls, [])

    def test_identical_existing_file_is_verified_locally(self):
        target = self.root / self.row["path"]
        target.parent.mkdir()
        target.write_bytes(self.content)
        result = self.run_restore()
        self.assertEqual(result["status"], "restored_verified")
        self.assertEqual(result["objects"][0]["status"], "already_present_verified")
        self.assertEqual(self.client.calls, [])

    def test_dry_run_performs_no_get_and_creates_no_payload(self):
        result = self.run_restore(execute=False, paths=None, all_approved=True)
        self.assertEqual(result["status"], "dry_run_validated_no_network")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_full_get_evidence_or_unexecuted_receipt_is_rejected(self):
        original = copy.deepcopy(self.receipt)
        variants = []
        self.receipt["execute"] = False
        variants.append(copy.deepcopy(self.receipt))
        self.receipt = copy.deepcopy(original)
        self.receipt["objects"][0]["get"]["complete_body_read"] = False
        variants.append(copy.deepcopy(self.receipt))
        self.receipt = copy.deepcopy(original)
        self.receipt["objects"][0]["get"]["sha256"] = "0" * 64
        variants.append(copy.deepcopy(self.receipt))
        for variant in variants:
            with self.subTest(variant=variant):
                self.receipt = variant
                self.write_receipt()
                self.assertEqual(self.run_restore()["status"], "failed")
        self.assertEqual(self.client.calls, [])

    def test_raw_allowlist_and_safe_paths_are_enforced(self):
        original = copy.deepcopy(self.row)
        for changes in ({"dataset": "HSStroke2026_v6"}, {"dataset": "unknown"},
                        {"path": "../escape"}, {"path": ".env"},
                        {"key": "paper1/private.bin"}):
            with self.subTest(changes=changes):
                self.receipt["objects"] = [{**original, **changes}]
                self.write_receipt()
                self.assertEqual(self.run_restore(paths=None, all_approved=True)["status"], "failed")
        self.assertEqual(self.client.calls, [])

    def test_symlink_destination_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        (self.root / "data").symlink_to(outside, target_is_directory=True)
        result = self.run_restore()
        self.assertEqual(result["error_code"], "symlink_forbidden")
        self.assertEqual(self.client.calls, [])
        self.assertEqual(list(outside.iterdir()), [])

    def test_catalog_metadata_may_name_sealed_dataset(self):
        self.row.update(path="catalog/HSStroke2026_v6_external_metadata.json",
                        key="paper3/backup/catalog/HSStroke2026_v6_external_metadata.json",
                        role="catalog", dataset="HSStroke2026_v6")
        self.write_receipt()
        result = self.run_restore(execute=False)
        self.assertEqual(result["status"], "dry_run_validated_no_network")
        self.assertEqual(self.client.calls, [])

    def test_selection_outside_verified_receipt_is_rejected(self):
        result = self.run_restore(paths=["data/unlisted.bin"])
        self.assertEqual(result["error_code"], "restore_selection_not_in_verified_receipt")
        self.assertEqual(self.client.calls, [])

    def test_backup_receipt_cannot_be_used_as_restore_report(self):
        before = self.source.read_bytes()
        result = self.run_restore(report_path=self.source)
        self.assertEqual(result["error_code"], "backup_receipt_cannot_be_modified")
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.client.calls, [])

    def test_read_failure_leaves_only_partial_and_sanitized_error(self):
        class FailingBody(TrackedBody):
            def read(self, amount=-1):
                if self.tell():
                    raise OSError("sensitive endpoint and keys must never appear")
                return super().read(amount)
        def get_object(**kwargs):
            self.client.calls.append(kwargs)
            self.client.body = FailingBody(self.content)
            return {"Body": self.client.body, "ContentLength": len(self.content),
                    "ResponseMetadata": {"HTTPStatusCode": 200}}
        self.client.get_object = get_object
        result = self.run_restore()
        self.assertEqual(result["status"], "failed")
        self.assertFalse((self.root / self.row["path"]).exists())
        self.assertEqual(len(list(self.root.rglob("*.partial"))), 1)
        self.assertFalse(result["objects"][0]["complete_body_read"])
        self.assertNotIn("get_sha256", result["objects"][0])
        self.assertNotIn("sensitive endpoint", json.dumps(result))
        self.assertTrue(self.client.body.closed)


if __name__ == "__main__":
    unittest.main()
