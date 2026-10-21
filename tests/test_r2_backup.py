"""Exercise object bytes, conditional writes, failures and resumed full GETs."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "r2_backup", Path(__file__).resolve().parents[1] / "tools" / "r2_backup.py"
)
r2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r2)


class ReadTrackedBody(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.read_calls = []
        self.eof_observed = False

    def read(self, amount=-1):
        self.read_calls.append(amount)
        chunk = super().read(amount)
        if chunk == b"":
            self.eof_observed = True
        return chunk


class ConditionalConflict(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 412}}


class MockS3:
    def __init__(self):
        self.objects = {}
        self.put_calls = []
        self.get_calls = []
        self.bodies = []
        self.corrupt_get = False

    def put_object(self, **kwargs):
        self.put_calls.append({key: value for key, value in kwargs.items() if key != "Body"})
        assert kwargs["IfNoneMatch"] == "*"
        key = kwargs["Key"]
        if key in self.objects:
            raise ConditionalConflict("sensitive SDK details must not appear")
        value = kwargs["Body"].read()
        assert len(value) == kwargs["ContentLength"]
        self.objects[key] = value
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        value = self.objects[kwargs["Key"]]
        if self.corrupt_get:
            value = bytes([value[0] ^ 1]) + value[1:]
        body = ReadTrackedBody(value)
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(value),
                "ResponseMetadata": {"HTTPStatusCode": 200}}


class R2BackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # More than one chunk prevents a single short read masquerading as a GET.
        self.content = b"paper3-source-development\x00" * 100000
        (self.root / "raw.bin").write_bytes(self.content)
        self.manifest = self.root / "manifest.json"
        self.receipt = self.root / "receipt.json"
        self.entry = {"path": "raw.bin", "sha256": hashlib.sha256(self.content).hexdigest(),
                      "bytes": len(self.content), "role": "source_development_raw",
                      "dataset": "Stieger2021"}
        self.write_manifest()
        self.client = MockS3()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"schema_version": 1, "files": [self.entry]}))

    def run_backup(self, execute=True, prefix="paper3/test-run/"):
        return r2.run_backup(manifest_path=self.manifest, root=self.root,
                             prefix=prefix, receipt_path=self.receipt,
                             execute=execute, client=self.client, bucket="test-bucket")

    def test_put_then_entire_get_and_resume_repeats_entire_get(self):
        first = self.run_backup()
        self.assertEqual(first["status"], "verified")
        self.assertEqual(len(self.client.put_calls), 1)
        self.assertTrue(self.client.bodies[0].eof_observed)
        self.assertTrue(self.client.bodies[0].closed)
        self.assertGreaterEqual(len(self.client.bodies[0].read_calls), 3)
        verified = first["objects"][0]["get"]
        self.assertEqual(verified["sha256"], self.entry["sha256"])
        self.assertEqual(verified["bytes_read"], len(self.content))
        second = self.run_backup()
        self.assertEqual(second["status"], "verified")
        self.assertEqual(len(self.client.put_calls), 1)
        self.assertEqual(len(self.client.get_calls), 2)
        self.assertTrue(self.client.bodies[1].eof_observed)
        self.assertTrue(second["objects"][0]["put_reused_from_prior_receipt"])

    def test_remote_hash_mismatch_fails_even_with_200_and_expected_length(self):
        self.client.corrupt_get = True
        result = self.run_backup()
        self.assertEqual(result["status"], "failed")
        get = result["objects"][0]["get"]
        self.assertEqual(get["http_status"], 200)
        self.assertEqual(get["bytes_read"], len(self.content))
        self.assertEqual(get["error_code"], "remote_integrity_mismatch")
        self.assertTrue(self.client.bodies[0].eof_observed)

    def test_existing_object_is_never_overwritten_and_is_fully_verified(self):
        key = "paper3/test-run/raw.bin"
        self.client.objects[key] = self.content
        result = self.run_backup()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["objects"][0]["put"]["http_status"], 412)
        self.assertEqual(result["objects"][0]["put"]["status"], "already_exists_not_overwritten")
        self.assertTrue(self.client.bodies[0].eof_observed)
        self.assertNotIn("sensitive SDK", json.dumps(result))

    def test_mismatched_local_bytes_block_all_network(self):
        (self.root / "raw.bin").write_bytes(b"changed")
        result = self.run_backup()
        self.assertEqual(result["error_code"], "local_integrity_mismatch")
        self.assertEqual(self.client.put_calls, [])
        self.assertEqual(self.client.get_calls, [])

    def test_dry_run_verifies_local_files_without_network(self):
        result = self.run_backup(execute=False)
        self.assertEqual(result["status"], "dry_run_verified_local_only")
        self.assertEqual(self.client.put_calls, [])
        self.assertEqual(self.client.get_calls, [])

    def test_forbidden_dataset_and_role_are_rejected(self):
        for dataset in ("HS-BCI2025", "HSBCI2025", "HSStroke2026_v6", "Wang2024",
                        "clinical", "external", "internal_test", "internaltest",
                        "unknown_source_dataset"):
            with self.subTest(dataset=dataset):
                self.entry["dataset"] = dataset
                self.write_manifest()
                self.assertEqual(self.run_backup()["error_code"], "dataset_or_role_forbidden")
        self.entry["dataset"] = "Stieger2021"
        self.entry["role"] = "external_query"
        self.write_manifest()
        self.assertEqual(self.run_backup()["error_code"], "dataset_or_role_forbidden")
        self.assertEqual(self.client.put_calls, [])

    def test_protocol_metadata_can_describe_sealed_datasets(self):
        for role in ("catalog", "protocol"):
            with self.subTest(role=role):
                metadata = self.root / "HSStroke2026_v6_external_metadata.json"
                metadata.write_bytes(self.content)
                self.entry.update(path=metadata.name, role=role, dataset="HSStroke2026_v6")
                self.write_manifest()
                result = self.run_backup(execute=False)
                self.assertEqual(result["status"], "dry_run_verified_local_only")
        self.assertEqual(self.client.put_calls, [])

    def test_prefix_escape_and_symlink_are_rejected(self):
        for prefix in ("paper1/a/", "paper3/", "paper3/../paper1/", "paper3//a/"):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.run_backup(prefix=prefix)["error_code"],
                                 "invalid_isolated_paper3_prefix")
        (self.root / "link.bin").symlink_to(self.root / "raw.bin")
        self.entry["path"] = "link.bin"
        self.write_manifest()
        self.assertEqual(self.run_backup()["error_code"], "symlink_forbidden")

    def test_partial_read_failure_cannot_be_reported_as_verified(self):
        class FailingBody(ReadTrackedBody):
            def read(self, amount=-1):
                if self.tell():
                    raise OSError("credentials and endpoints are never output")
                return super().read(amount)
        body = FailingBody(self.content)
        with patch.object(self.client, "get_object", return_value={
            "Body": body, "ContentLength": len(self.content),
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }):
            result = self.run_backup()
        self.assertEqual(result["status"], "failed")
        get = result["objects"][0]["get"]
        self.assertFalse(get["complete_body_read"])
        self.assertEqual(get["bytes_read"], r2.CHUNK_SIZE)
        self.assertNotIn("sha256", get)
        self.assertTrue(body.closed)
        self.assertNotIn("credentials and endpoints", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
