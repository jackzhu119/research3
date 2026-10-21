"""Real byte-stream IO semantics with a local fake object store; no network IO."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from p3work import source_extension_io as extension


class TrackedBody(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.saw_eof = False
        self.read_count = 0

    def read(self, amount=-1):
        self.read_count += 1
        value = super().read(amount)
        if value == b"":
            self.saw_eof = True
        return value


class Conflict(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 412}}


class BrokenBody(TrackedBody):
    def read(self, amount=-1):
        if self.read_count:
            raise RuntimeError("DO_NOT_PRINT_TOKEN_123")
        return super().read(17)


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.put_count = 0
        self.get_count = 0
        self.bodies = []
        self.corrupt = False
        self.broken_body = False
        self.put_error = False

    def put_object(self, **kwargs):
        self.put_count += 1
        if self.put_error:
            raise RuntimeError("DO_NOT_PRINT_TOKEN_123")
        assert kwargs["IfNoneMatch"] == "*"
        if kwargs["Key"] in self.objects:
            raise Conflict("DO_NOT_PRINT_TOKEN_123")
        value = kwargs["Body"].read()
        assert kwargs["ContentLength"] == len(value)
        assert kwargs["Metadata"]["sha256"] == hashlib.sha256(value).hexdigest()
        self.objects[kwargs["Key"]] = value
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **kwargs):
        self.get_count += 1
        value = self.objects[kwargs["Key"]]
        if self.corrupt:
            value = b"!" + value[1:]
        body = (BrokenBody if self.broken_body else TrackedBody)(value)
        self.bodies.append(body)
        return {"Body": body, "ContentLength": len(value),
                "ResponseMetadata": {"HTTPStatusCode": 200}}


class SourceExtensionIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "input"
        self.root.mkdir()
        self.restore_root = self.base / "fresh"
        self.manifest = self.base / "manifest.json"
        self.receipt = self.base / "backup_receipt.json"
        self.report = self.base / "restore_report.json"
        self.client = FakeR2()
        self.content = b"paper3-real-source-bytes\x00" * 100000
        self.entry = self.make_entry("Yang2025")
        self.write_input()

    def make_entry(self, dataset, *, role="source_development_raw", path=None):
        return {"path": path or f"{extension.RAW_ROOT}/{dataset}/participant_001.mat",
                "sha256": hashlib.sha256(self.content).hexdigest(), "bytes": len(self.content),
                "dataset": dataset, "role": role}

    def write_input(self, *, create=True):
        if create:
            target = self.root / self.entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.content)
        self.manifest.write_text(json.dumps({"schema_version": 1, "files": [self.entry]}))

    def backup(self, **changes):
        kwargs = dict(manifest_path=self.manifest, root=self.root,
                      prefix="paper3/test-source-extension/", receipt_path=self.receipt,
                      execute=True, client=self.client, bucket="fake-test-bucket")
        kwargs.update(changes)
        return extension.run_backup(**kwargs)

    def restore(self, **changes):
        kwargs = dict(backup_receipt_path=self.receipt, root=self.restore_root,
                      report_path=self.report, all_approved=True,
                      execute=True, require_fresh=True, client=self.client, bucket="fake-test-bucket")
        kwargs.update(changes)
        return extension.run_restore(**kwargs)

    def mutate_receipt(self, callback):
        value = json.loads(self.receipt.read_bytes())
        callback(value)
        self.receipt.write_text(json.dumps(value))

    def assert_no_network(self):
        self.assertEqual((self.client.put_count, self.client.get_count), (0, 0))

    def test_exact_profile_and_registered_unique_roles(self):
        value = extension.profile()
        self.assertEqual(value["document"]["sha256"],
            "74383bb55cd1287ad806e5d83812e95bdb4bf0bccf7a59292b91204f230a073a")
        self.assertEqual(value["registered_dataset_roles"], {
            "Yang2025": "source", "Dreyer2023": "source", "Ma2022": "independent_development"})
        self.assertIs(value["sealed_external_payload_access_allowed"], False)
        value["registered_dataset_roles"]["Wang2025"] = "source"
        self.assertNotIn("Wang2025", extension.profile()["registered_dataset_roles"])

    def test_original_policy_remains_unchanged(self):
        specification = importlib.util.spec_from_file_location(
            "original_untouched_backup_for_test", extension.REPOSITORY / "tools/r2_backup.py")
        original = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(original)
        self.assertNotIn("Yang2025", original.RAW_DATASETS)
        extension.validate_entry(self.entry)
        with self.assertRaises(original.BackupError):
            original.validate_entry(self.entry)
        self.assertEqual(original.RAW_DATASETS, frozenset({
            "BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI", "Stieger2021"}))

    def test_three_explicit_raw_ids_pass_local_dry_run_with_correct_labels(self):
        for dataset in ("Yang2025", "Dreyer2023", "Ma2022"):
            with self.subTest(dataset=dataset):
                self.entry = self.make_entry(dataset)
                self.write_input()
                result = self.backup(execute=False)
                self.assertEqual(result["status"], "dry_run_verified_local_only")
                self.assertEqual(result["objects"][0]["dataset"], dataset)
                self.assertEqual(result["objects"][0]["role"], "source_development_raw")
        self.assert_no_network()

    def test_full_get_eof_after_conditional_put_and_fresh_restore(self):
        result = self.backup()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["source_extension_profile"], extension.profile())
        first = self.client.bodies[0]
        self.assertTrue(first.saw_eof)
        self.assertTrue(first.closed)
        self.assertGreater(first.read_count, 2)
        restored = self.restore()
        self.assertEqual(restored["status"], "restored_verified")
        self.assertTrue(restored["fresh_full_get_verified"])
        row = restored["objects"][0]
        self.assertEqual(row["status"], "restored_verified")
        self.assertTrue(row["complete_body_read"])
        self.assertEqual(row["get_sha256"], self.entry["sha256"])
        self.assertEqual((self.restore_root / self.entry["path"]).read_bytes(), self.content)
        self.assertEqual(self.client.get_count, 2)

    def test_repeated_backup_reuses_put_but_repeats_full_get(self):
        self.assertEqual(self.backup()["status"], "verified")
        result = self.backup()
        self.assertEqual(result["status"], "verified")
        self.assertEqual((self.client.put_count, self.client.get_count), (1, 2))
        self.assertTrue(result["objects"][0]["put_reused_from_prior_receipt"])
        self.assertTrue(all(body.saw_eof for body in self.client.bodies))

    def test_existing_remote_object_412_does_not_overwrite_and_is_read(self):
        key = "paper3/test-source-extension/" + self.entry["path"]
        self.client.objects[key] = self.content
        result = self.backup()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["objects"][0]["put"]["http_status"], 412)
        self.assertEqual(self.client.objects[key], self.content)
        self.assertTrue(self.client.bodies[0].saw_eof)

    def test_corrupt_remote_existing_bytes_are_never_overwritten(self):
        key = "paper3/test-source-extension/" + self.entry["path"]
        self.client.objects[key] = b"wrong"
        result = self.backup()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.client.objects[key], b"wrong")
        self.assertEqual(result["objects"][0]["get"]["error_code"], "remote_integrity_mismatch")

    def test_changed_local_hash_blocks_put_and_get(self):
        (self.root / self.entry["path"]).write_bytes(b"tampered")
        result = self.backup()
        self.assertEqual(result["error_code"], "local_integrity_mismatch")
        self.assert_no_network()

    def test_unknown_or_external_dataset_cannot_be_laundered_under_metadata_role(self):
        for dataset in ("Wang2025", "HSStroke2026_v6", "Leeuwis2021", "MILimbEEG", "unknown",
                        "Stieger2021", "BNCI2014_001"):
            for role in ("source_development_raw", "archive", "catalog", "audit", "result"):
                with self.subTest(dataset=dataset, role=role):
                    entry = self.make_entry(dataset, role=role,
                        path=f"catalog/source_extensions_20261019/{dataset}/rows.json")
                    with self.assertRaises(Exception):
                        extension.validate_entry(entry)
        self.assert_no_network()

    def test_approved_raw_path_cannot_use_archive_metadata_or_result_labels(self):
        for role in ("archive", "audit", "catalog", "manifest", "execution_package", "result"):
            with self.subTest(role=role):
                self.entry = self.make_entry("Yang2025", role=role)
                self.write_input(create=False)
                result = self.backup()
                self.assertEqual(result["status"], "failed")
        self.assert_no_network()

    def test_dataset_must_match_exact_path_namespace(self):
        for path in (f"{extension.RAW_ROOT}/Dreyer2023/subject.mat",
                     "data/Yang2025/subject.mat", "data/source_extensions_20261019/Yang2025extra/a.mat",
                     "cache/source_extensions_20261019/Yang2025/subject.mat"):
            with self.subTest(path=path):
                self.entry = self.make_entry("Yang2025", path=path)
                self.write_input(create=False)
                result = self.backup()
                self.assertEqual(result["status"], "failed")
        self.assert_no_network()

    def test_result_arrays_keep_exact_source_label_and_namespace(self):
        entry = self.make_entry("Dreyer2023", role="result",
            path="cache/source_extensions_20261019/Dreyer2023/subject_001/X.npy")
        extension.validate_entry(entry)
        for wrong in ("source", "unknown", "Ma2022", "Wang2025"):
            with self.subTest(dataset=wrong):
                with self.assertRaises(Exception):
                    extension.validate_entry({**entry, "dataset": wrong})

    def test_legacy_npz_and_payload_disguised_as_metadata_rejected(self):
        for role, path in (
            ("result", "cache/source_extensions_20261019/Yang2025/old.npz"),
            ("catalog", "catalog/source_extensions_20261019/Yang2025/raw.mat"),
            ("audit", "audit/source_extensions_20261019/Yang2025/raw.zip"),
            ("source_development_raw", f"{extension.RAW_ROOT}/Yang2025/old.npz")):
            with self.subTest(role=role, path=path):
                with self.assertRaises(Exception):
                    extension.validate_entry(self.make_entry("Yang2025", role=role, path=path))

    def test_credential_named_files_and_paths_forbidden(self):
        for name in (".env", ".env.r2", "credentials.json", "aws_secret.txt", "token.json",
                     "password.txt", "x.key", "x.pem", ".aws/config.json", "secrets/data.json",
                     "AWS_ACCESS_KEY_ID.json", "r2_endpoint.json", "private_key.json"):
            with self.subTest(name=name):
                entry = self.make_entry("Ma2022", path=f"{extension.RAW_ROOT}/Ma2022/{name}")
                with self.assertRaises(Exception):
                    extension.validate_entry(entry)

    def test_external_named_result_cannot_be_relabelled_as_new_source(self):
        for dataset in extension.REGISTERED_ROLES:
            with self.subTest(dataset=dataset):
                entry = self.make_entry(dataset, role="result",
                    path=f"{extension.RESULT_ROOT}/{dataset}/external_query.npy")
                with self.assertRaises(extension.SourceExtensionIOError):
                    extension.validate_entry(entry)

    def test_unsafe_relative_paths_forbidden(self):
        for path in ("/tmp/data.mat", "../data.mat", "data/../file.mat", "data//file.mat",
                     "data/./file.mat", "data\\file.mat", "data/file.mat\n"):
            with self.subTest(path=path):
                with self.assertRaises(Exception):
                    extension.validate_entry(self.make_entry("Yang2025", path=path))

    def test_symlink_payload_and_root_are_rejected_before_network(self):
        target = self.root / self.entry["path"]
        real = self.base / "real.mat"
        target.rename(real)
        target.symlink_to(real)
        self.assertEqual(self.backup()["error_code"], "symlink_forbidden")
        target.unlink()
        real.rename(target)
        linked_root = self.base / "rootlink"
        linked_root.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(self.backup(root=linked_root)["error_code"], "symlink_forbidden")
        self.assert_no_network()

    def test_output_receipt_cannot_overwrite_manifest_or_payload(self):
        for target in (self.manifest, self.root / self.entry["path"]):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                result = self.backup(receipt_path=target)
                self.assertEqual(result["error_code"], "receipt_cannot_overwrite_input_or_payload")
                self.assertEqual(target.read_bytes(), before)
        self.assert_no_network()

    def test_hardlinked_receipt_and_symlink_receipt_are_rejected(self):
        target = self.root / self.entry["path"]
        self.receipt.hardlink_to(target)
        self.assertEqual(self.backup()["error_code"], "receipt_cannot_overwrite_input_or_payload")
        self.receipt.unlink()
        self.receipt.symlink_to(target)
        self.assertEqual(self.backup()["error_code"], "symlink_forbidden")
        self.assert_no_network()

    def test_unrelated_existing_receipt_is_not_overwritten(self):
        value = b'{"important":"unrelated work"}'
        self.receipt.write_bytes(value)
        result = self.backup()
        self.assertEqual(result["error_code"], "unrelated_receipt_file_not_overwritten")
        self.assertEqual(self.receipt.read_bytes(), value)
        self.assert_no_network()

    def test_restore_dry_run_reads_no_remote_bytes(self):
        self.assertEqual(self.backup()["status"], "verified")
        before = self.client.get_count
        result = self.restore(execute=False)
        self.assertEqual(result["status"], "dry_run_validated_no_network")
        self.assertFalse(result["fresh_full_get_verified"])
        self.assertEqual(self.client.get_count, before)
        self.assertFalse((self.restore_root / self.entry["path"]).exists())

    def test_missing_profile_in_old_receipt_cannot_restore_new_raw(self):
        self.assertEqual(self.backup()["status"], "verified")
        self.mutate_receipt(lambda value: value.pop("source_extension_profile"))
        before = self.client.get_count
        result = self.restore()
        self.assertEqual(result["error_code"], "source_extension_receipt_profile_mismatch")
        self.assertEqual(self.client.get_count, before)

    def test_receipt_doc_sha_or_registered_role_tamper_prevents_restore(self):
        self.assertEqual(self.backup()["status"], "verified")
        original = self.receipt.read_bytes()
        for mutation in (
            lambda value: value["source_extension_profile"]["document"].update(sha256="0" * 64),
            lambda value: value["source_extension_profile"]["registered_dataset_roles"].update(Ma2022="source"),
            lambda value: value["source_extension_profile"].update(sealed_external_payload_access_allowed=True)):
            self.receipt.write_bytes(original)
            self.mutate_receipt(mutation)
            before = self.client.get_count
            result = self.restore()
            self.assertEqual(result["error_code"], "source_extension_receipt_profile_mismatch")
            self.assertEqual(self.client.get_count, before)

    def test_receipt_row_namespace_or_integrity_tamper_prevents_get(self):
        self.assertEqual(self.backup()["status"], "verified")
        original = self.receipt.read_bytes()
        for mutation in (
            lambda value: value["objects"][0].update(dataset="Ma2022"),
            lambda value: value["objects"][0].update(role="archive"),
            lambda value: value["objects"][0]["get"].update(complete_body_read=False),
            lambda value: value["objects"][0]["put"].update(http_status=201),
            lambda value: value["objects"][0].update(key="paper3/wrong-key")):
            self.receipt.write_bytes(original)
            self.mutate_receipt(mutation)
            before = self.client.get_count
            self.assertEqual(self.restore()["status"], "failed")
            self.assertEqual(self.client.get_count, before)

    def test_fresh_restore_rejects_existing_identical_file_no_get(self):
        self.assertEqual(self.backup()["status"], "verified")
        self.assertEqual(self.restore()["status"], "restored_verified")
        before = self.client.get_count
        result = self.restore()
        self.assertEqual(result["error_code"], "fresh_restore_requires_absent_selected_payloads")
        self.assertEqual(self.client.get_count, before)
        resumed = self.restore(require_fresh=False)
        self.assertEqual(resumed["status"], "restored_verified")
        self.assertEqual(resumed["objects"][0]["status"], "already_present_verified")
        self.assertFalse(resumed["fresh_full_get_verified"])
        self.assertEqual(self.client.get_count, before)

    def test_different_existing_local_file_is_never_overwritten(self):
        self.assertEqual(self.backup()["status"], "verified")
        target = self.restore_root / self.entry["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"keep different original")
        before = self.client.get_count
        result = self.restore(require_fresh=False)
        self.assertEqual(result["error_code"], "different_local_file_not_overwritten")
        self.assertEqual(target.read_bytes(), b"keep different original")
        self.assertEqual(self.client.get_count, before)

    def test_restore_corrupt_get_never_publishes_completed_file(self):
        self.assertEqual(self.backup()["status"], "verified")
        self.client.corrupt = True
        result = self.restore()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["objects"][0]["error_code"], "remote_integrity_mismatch")
        self.assertFalse((self.restore_root / self.entry["path"]).exists())
        self.assertTrue(list(self.restore_root.rglob("*.partial")))

    def test_interrupted_get_retains_partial_but_never_reports_completed(self):
        self.assertEqual(self.backup()["status"], "verified")
        self.client.broken_body = True
        result = self.restore()
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["objects"][0]["complete_body_read"])
        self.assertFalse((self.restore_root / self.entry["path"]).exists())
        self.assertNotIn("DO_NOT_PRINT_TOKEN_123", json.dumps(result))

    def test_restore_report_cannot_overwrite_original_receipt(self):
        self.assertEqual(self.backup()["status"], "verified")
        original = self.receipt.read_bytes()
        before = self.client.get_count
        result = self.restore(report_path=self.receipt)
        self.assertEqual(result["error_code"], "receipt_cannot_overwrite_input_or_payload")
        self.assertEqual(self.receipt.read_bytes(), original)
        self.assertEqual(self.client.get_count, before)

    def test_restore_symlink_destination_blocks_all_gets(self):
        self.assertEqual(self.backup()["status"], "verified")
        outside = self.base / "outside"
        outside.mkdir()
        self.restore_root.symlink_to(outside, target_is_directory=True)
        before = self.client.get_count
        result = self.restore()
        self.assertEqual(result["error_code"], "symlink_forbidden")
        self.assertEqual(self.client.get_count, before)

    def test_explicit_selection_and_no_duplicate_paths_required(self):
        self.assertEqual(self.backup()["status"], "verified")
        before = self.client.get_count
        for kwargs in ({"all_approved": False}, {"all_approved": True, "paths": [self.entry["path"]]},
                       {"all_approved": False, "paths": ["unknown.mat"]},
                       {"all_approved": False, "paths": [self.entry["path"], self.entry["path"]]}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.restore(**kwargs)["status"], "failed")
        self.assertEqual(self.client.get_count, before)

    def test_sdk_failure_details_never_escape_to_receipt_or_return(self):
        self.client.put_error = True
        result = self.backup()
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("DO_NOT_PRINT_TOKEN_123", json.dumps(result))
        self.assertNotIn("DO_NOT_PRINT_TOKEN_123", self.receipt.read_text())


if __name__ == "__main__":
    unittest.main()
