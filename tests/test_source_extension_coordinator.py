"""CPU coordinator failure, integrity, ownership and restart semantics; no network."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from p3work import source_extension_coordinator as work
from p3work import source_extension_io as storage


class FakeR2:
    def __init__(self):
        self.values = {}
        self.gets = 0

    def put_object(self, **kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        if kwargs["Key"] in self.values:
            error = RuntimeError("SDK_PRIVATE_DETAILS")
            error.response = {"ResponseMetadata": {"HTTPStatusCode": 412}}
            raise error
        self.values[kwargs["Key"]] = kwargs["Body"].read()
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **kwargs):
        self.gets += 1
        value = self.values[kwargs["Key"]]
        return {"Body": io.BytesIO(value), "ContentLength": len(value),
                "ResponseMetadata": {"HTTPStatusCode": 200}}


class Response:
    def __init__(self, value, status=200, headers=None):
        self.value, self.status_code, self.closed = value, status, False
        self.headers = {"Content-Length": str(len(value)), **(headers or {})}

    def iter_content(self, size):
        for offset in range(0, len(self.value), size):
            yield self.value[offset:offset+size]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, value, *, ignore_range=False):
        self.value, self.ignore_range, self.calls = value, ignore_range, []

    def get(self, url, *, headers, stream, timeout):
        self.calls.append(headers)
        if "Range" in headers and not self.ignore_range:
            first, last = map(int, headers["Range"][6:].split("-"))
            return Response(self.value[first:last+1], status=206,
                headers={"Content-Range": f"bytes {first}-{last}/{len(self.value)}"})
        return Response(self.value)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.content = b"native-source-test\x00" * 400
        self.dataset = "Dreyer2023"
        self.raw_path = f"{storage.RAW_ROOT}/{self.dataset}/sub-01/eeg/sub-01_task-R1acquisition_eeg.edf"
        self.raw = self.root/self.raw_path
        self.item = {"path": self.raw_path, "bytes": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
            "download": {"kind": "direct", "url": "https://data.nemar.org/test.edf"}}
        self.record = {"record_id": "sub-01_run-1", "subject_id": "sub-01",
                       "session_id": "provider_single_day", "raw_files": [self.item]}
        adapter = self.root/"p3work/source_extensions/dreyer2023.py"
        adapter.parent.mkdir(parents=True)
        adapter.write_text("# frozen test-only adapter bytes\n")
        profile = self.root/"audit/frozen_profile.json"
        profile.parent.mkdir(parents=True)
        profile.write_text("{}")
        self.plan = {"dataset": self.dataset, "plan_sha256": "a" * 64,
            "records": [self.record], "record_count": 1, "task_subset": "LR",
            "native_profile": work.descriptor(self.root, profile), "inventory": work.descriptor(self.root, profile),
            "budgets": {"provider_body_bytes": 10**7, "r2_get_body_bytes": 10**7,
                        "wall_seconds": 3600, "cache_bytes": 10**7, "disk_reserve_bytes": 1},
            "code": {"p3work/source_extensions/dreyer2023.py": work.descriptor(self.root, adapter)}}
        self.ledger_path = self.root/"audit/test_ledger.json"
        self.ledger = work.Ledger(self.ledger_path, self.plan)
        self.r2 = FakeR2()
        self.session = Session(self.content)
        self.x = np.arange(2*11*128, dtype=np.float32).reshape(2, 11, 128)
        self.y = np.asarray([0, 1], dtype=np.int64)
        trials = [{"trial_uid": "native-"+str(i), "dataset": self.dataset, "label": i,
            "canonical_label": "left_hand" if i == 0 else "right_hand", "raw_label_code": str(769+i),
            "original_trial_index": i+1, "original_annotation_index": 6*i+4,
            "analysis_anchor_sample": 1000+2000*i, "task_end_sample": 3000+2000*i,
            "original_context_samples": [1000+2000*i, 1768+2000*i], "session_id": "provider_single_day",
            "physical_run_id": "R1acquisition", "source_file_sha256": self.item["sha256"]} for i in range(2)]
        self.native = {"trials": trials, "source_file_sha256": self.item["sha256"],
            "EEG_array_samples_read": 0, "waveform_QC_applied": False,
            "native_signal_order": work.CHANNELS, "native_signal_units": dict.fromkeys(work.CHANNELS, "µV")}
        self.events = [dict(row, state="valid", cache_row=i,
            preprocessing={"stored_unit": "V", "normalization_applied": False}) for i, row in enumerate(trials)]

    def backup(self, **kwargs):
        return storage.run_backup(**kwargs, client=self.r2, bucket="unit-test")

    def restore(self, **kwargs):
        return storage.run_restore(**kwargs, client=self.r2, bucket="unit-test")

    def materialize(self, *args, **kwargs):
        return self.x.copy(), self.y.copy(), copy.deepcopy(self.events), {
            "independent_native_metadata_semantic_sha256": work.digest(self.native),
            "SI_conversion": "native µV to V once by reader", "exclusions": []}

    def audit_native(self, *args, **kwargs):
        return copy.deepcopy(self.native)

    def process(self, **changes):
        kwargs = dict(native_profile={}, ledger=self.ledger, session=self.session,
            materialize=self.materialize, audit_native=self.audit_native,
            backup_fn=self.backup, restore_fn=self.restore)
        kwargs.update(changes)
        return work.process_record(self.root, self.plan, self.record, **kwargs)

    def test_end_to_end_one_owned_record_actual_fake_store_bytes_and_cleanup(self):
        state = self.process()
        self.assertEqual(state["status"], "COMPLETE")
        self.assertFalse(self.raw.exists())
        self.assertTrue(work._state_verified(self.root, state, self.plan, self.record))
        shard = state["shard"]
        self.assertEqual(np.load(work.checked(self.root, shard["X"]), allow_pickle=False).shape, (2, 11, 128))
        self.assertEqual(shard["session_id"], "provider_single_day")
        self.assertEqual(self.ledger.data["charged"]["provider_body_bytes"], len(self.content))
        self.assertEqual(len(self.session.calls), 1)

    def test_resume_complete_record_rechecks_hashes_without_download_or_r2_get(self):
        self.process()
        gets, calls = self.r2.gets, len(self.session.calls)
        state = self.process()
        self.assertEqual(state["status"], "COMPLETE")
        self.assertEqual((self.r2.gets, len(self.session.calls)), (gets, calls))

    def test_existing_pilot_is_rehashed_and_never_adopted_or_deleted(self):
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(self.content)
        state = self.process()
        self.assertEqual(state["owned_raw"], [])
        self.assertEqual(self.raw.read_bytes(), self.content)
        self.assertEqual(self.session.calls, [])
        self.assertFalse(state["reused_existing_raw"][0]["new_provider_GET_performed"])
        self.assertFalse(state["reused_existing_raw"][0]["ownership_adopted"])

    def test_existing_wrong_raw_is_preserved_and_blocks_all_io(self):
        self.raw.parent.mkdir(parents=True)
        self.raw.write_bytes(b"do not overwrite")
        with self.assertRaisesRegex(work.PreparationError, "existing_raw_different_preserved"):
            self.process()
        self.assertEqual(self.raw.read_bytes(), b"do not overwrite")
        self.assertEqual(self.r2.gets, 0)

    def test_unowned_partial_is_not_deleted_or_adopted(self):
        partial = self.raw.with_name(self.raw.name+".owned.partial")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"someone else's partial")
        with self.assertRaisesRegex(work.PreparationError, "unowned_existing_partial_preserved"):
            self.process()
        self.assertEqual(partial.read_bytes(), b"someone else's partial")
        self.assertEqual(self.session.calls, [])

    def test_raw_backup_failure_preserves_original_and_charges_reserved_get(self):
        def fail(**kwargs):
            return {"status": "failed", "execute": True}
        with self.assertRaisesRegex(work.PreparationError, "r2_put_full_get_sha_not_verified"):
            self.process(backup_fn=fail)
        self.assertEqual(self.raw.read_bytes(), self.content)
        self.assertEqual(self.ledger.data["charged"]["r2_get_body_bytes"], len(self.content))

    def test_cache_backup_failure_preserves_raw_after_raw_backup(self):
        calls = 0
        def backup(**kwargs):
            nonlocal calls
            calls += 1
            return self.backup(**kwargs) if calls == 1 else {"status": "failed", "execute": True}
        with self.assertRaisesRegex(work.PreparationError, "r2_put_full_get_sha_not_verified"):
            self.process(backup_fn=backup)
        self.assertTrue(self.raw.exists())
        state_path = self.root/f"audit/source_extensions_20261019/{self.dataset}/{'a'*64}/{self.record['record_id']}/state.json"
        state = json.loads(state_path.read_bytes())
        self.assertIn("raw_backup_receipt", state)
        self.assertNotIn("cache_backup_receipt", state)

    def test_restart_missing_raw_uses_verified_r2_instead_of_provider(self):
        calls = 0
        def backup(**kwargs):
            nonlocal calls
            calls += 1
            return self.backup(**kwargs) if calls == 1 else {"status": "failed", "execute": True}
        with self.assertRaises(work.PreparationError):
            self.process(backup_fn=backup)
        self.raw.unlink()
        provider_calls = len(self.session.calls)
        state = self.process()
        self.assertEqual(state["status"], "COMPLETE")
        self.assertEqual(len(self.session.calls), provider_calls)
        self.assertTrue(any(row["identity"].endswith(":raw_restore") for row in self.ledger.data["history"]))

    def test_tampered_cached_array_prevents_complete_resume(self):
        state = self.process()
        x = work.checked(self.root, state["shard"]["X"])
        x.write_bytes(b"tampered")
        with self.assertRaisesRegex(work.PreparationError, "artifact_sha_or_bytes_mismatch"):
            self.process()

    def test_native_label_mismatch_blocks_before_numerical_backup_and_preserves_raw(self):
        self.events[0]["label"] = 1
        with self.assertRaisesRegex(work.PreparationError, "cache_event_label_clock_or_raw_identity_mismatch"):
            self.process()
        self.assertTrue(self.raw.exists())

    def test_duplicate_uid_or_missing_original_event_is_rejected(self):
        for events in ([self.events[0], self.events[0]], [self.events[0]]):
            with self.subTest(count=len(events)):
                with self.assertRaises(work.PreparationError):
                    work.validate_native_cache(self.dataset, self.record, self.native,
                        self.x, self.y, events, {"exclusions": [],
                            "independent_native_metadata_semantic_sha256": work.digest(self.native)})

    def test_processed_label_and_cache_row_coverage_are_verified(self):
        events = copy.deepcopy(self.events)
        events[1]["cache_row"] = 0
        with self.assertRaisesRegex(work.PreparationError, "cache_index_or_binary_label_mismatch"):
            work.validate_native_cache(self.dataset, self.record, self.native, self.x, self.y, events,
                {"exclusions": [], "independent_native_metadata_semantic_sha256": work.digest(self.native)})

    def test_actual_physical_units_must_match_selected_native_header(self):
        self.native["native_signal_units"][work.CHANNELS[0]] = "unknown"
        with self.assertRaisesRegex(work.PreparationError, "dreyer_actual_selected_native_units_unverified"):
            self.process()
        self.assertTrue(self.raw.exists())

    def test_qc_excluded_original_trial_still_has_native_identity(self):
        events = [copy.deepcopy(self.events[0])]
        excluded = dict(self.events[1], state="excluded", cache_row=None, exclusion_reason="zero_energy")
        rows, exclusions, unit = work.validate_native_cache(self.dataset, self.record, self.native,
            self.x[:1], self.y[:1], events, {"exclusions": [excluded],
                "independent_native_metadata_semantic_sha256": work.digest(self.native)})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(exclusions), 1)
        self.assertTrue(unit["si_units_verified"])

    def test_yang_unknown_si_requires_actual_common_positive_gain_and_retains_unknowns(self):
        native = {"derived_before_waveform_preprocessing": True, "independent_second_parser_claim": False,
            "common_gain_evidence": {"common_positive_gain_verified": True, "SI_unit_verified": False},
            "events": [{"event_uid": "lr", "label_code": "1", "label": 0, "original_event_index": 0,
                        "context_start_sample": 2250, "context_stop_sample_exclusive": 3750},
                       {"event_uid": "unknown", "label_code": "255", "original_event_index": 1}]}
        events = [dict(native["events"][0], valid=True, cache_row=0, label=0),
                  dict(native["events"][1], valid=False, cache_row=None)]
        record = dict(self.record, raw_files=[self.item, dict(self.item, sha256="b"*64)])
        rows, exclusions, unit = work.validate_native_cache("Yang2025", record, native,
            self.x[:1], self.y[:1], events, {})
        self.assertEqual(unit["stored_unit"], "calibrated_numeric_unknown_SI")
        self.assertFalse(unit["si_units_verified"])
        self.assertFalse(unit["physical_MI_onset_verified"])
        self.assertFalse(rows[0]["task_end_verified"])
        self.assertEqual(rows[0]["raw_event_sha256"], "b"*64)
        self.assertEqual(len(exclusions), 1)
        native["common_gain_evidence"]["common_positive_gain_verified"] = False
        with self.assertRaisesRegex(work.PreparationError, "yang_actual_common_gain_native_evidence_required"):
            work.validate_native_cache("Yang2025", record, native, self.x[:1], self.y[:1], events, {})

    def test_failed_transfer_retains_whole_reservation_across_ledger_reload(self):
        self.ledger.reserve("provider_body_bytes", 100, "interrupted")
        renewed = work.Ledger(self.ledger_path, self.plan)
        self.assertEqual(renewed.data["charged"]["provider_body_bytes"], 100)
        self.assertEqual(renewed.data["history"][-1]["status"], "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED")
        renewed.reserve("provider_body_bytes", 50, "failure")
        renewed.finish()
        self.assertEqual(renewed.data["charged"]["provider_body_bytes"], 150)

    def test_reservation_is_durable_before_request_and_exhaustion_prevents_io(self):
        self.ledger.reserve("provider_body_bytes", 100, "request")
        disk = json.loads(self.ledger_path.read_bytes())
        self.assertEqual(disk["active_transfer"]["reserved_bytes"], 100)
        self.ledger.finish(verified_bytes=80)
        self.assertEqual(self.ledger.data["charged"]["provider_body_bytes"], 80)
        self.plan["budgets"]["provider_body_bytes"] = 81
        with self.assertRaisesRegex(work.PreparationError, "cumulative_body_budget_exhausted"):
            self.ledger.reserve("provider_body_bytes", 2, "no request")

    def test_clock_changes_fail_closed_instead_of_resetting_elapsed(self):
        a = {"utc": "2026-10-19T10:00:00+00:00", "monotonic_ns": 10**9, "boot_id": "A"}
        b = {"utc": "2026-10-19T10:00:10+00:00", "monotonic_ns": 11*10**9, "boot_id": "A"}
        self.assertEqual(work.elapsed(a, b), 10)
        for changed in (dict(b, boot_id="B"), dict(b, monotonic_ns=0),
                        dict(b, utc="2026-10-19T10:02:10+00:00")):
            with self.assertRaisesRegex(work.PreparationError, "elapsed_unknown_or_clock_changed_budget_closed"):
                work.elapsed(a, changed)

    def test_strict_range_rejects_full_archive_200_without_reading_body(self):
        session = Session(b"large archive body", ignore_range=True)
        with self.assertRaisesRegex(work.PreparationError, "provider_did_not_honor_exact_range"):
            work._range("https://ndownloader.figshare.com/test", 0, 3, 18, session=session, ledger=self.ledger)
        self.assertEqual(self.ledger.data["charged"]["provider_body_bytes"], 3)

    def test_zip_member_stream_checks_crc_sha_and_multiple_exact_ranges(self):
        raw = b"some native BDF bytes\x00" * 5000
        compressor = zlib.compressobj(wbits=-15)
        compressed = compressor.compress(raw)+compressor.flush()
        name = b"official/sub-001/data.bdf"
        crc = zlib.crc32(raw)&0xffffffff
        header = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, crc,
                             len(compressed), len(raw), len(name), 0)
        archive = header+name+compressed+b"central directory metadata"
        target = self.root/"member.bdf"
        item = {"path": "member.bdf", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "crc32": f"{crc:08x}", "download": {"kind": "zip_range", "url": "https://ndownloader.figshare.com/test",
                "archive_bytes": len(archive), "member": name.decode(), "header_offset": 0,
                "compressed_bytes": len(compressed), "compression": 8}}
        with patch.object(work, "RANGE_CHUNK", 19):
            result = work.download_component(item, target, session=Session(archive), ledger=self.ledger)
        self.assertEqual(target.read_bytes(), raw)
        self.assertEqual(result["sha256"], item["sha256"])
        self.assertGreater(len(self.ledger.data["history"]), 3)

    def test_direct_get_hash_failure_does_not_publish_target(self):
        target = self.root/"wrong.edf"
        item = dict(self.item, sha256="0"*64)
        with self.assertRaisesRegex(work.PreparationError, "direct_native_sha_or_size_mismatch"):
            work.download_component(item, target, session=self.session, ledger=self.ledger)
        self.assertFalse(target.exists())
        self.assertEqual(self.ledger.data["charged"]["provider_body_bytes"], len(self.content))

    def test_all_records_required_for_complete_manifest(self):
        with self.assertRaisesRegex(work.PreparationError, "all_frozen_records_required_before_complete"):
            work.assemble_manifest(self.root, self.plan, [])

    def test_final_complete_manifest_and_role_inventory_are_full_get_backed(self):
        with patch.object(work, "validate_plan", return_value={}):
            result = work.prepare(self.root, self.plan, execute=True, session=self.session,
                materialize=self.materialize, audit_native=self.audit_native, backup_fn=self.backup)
        self.assertEqual(result["status"], "SOURCE_EXTENSION_CACHE_COMPLETE")
        manifest = json.loads(work.checked(self.root, result["manifest"]).read_bytes())
        self.assertEqual(manifest["schema_version"], "paper3-source-extension-cache-v1")
        self.assertEqual(manifest["valid_trials"], 2)
        self.assertEqual(manifest["provider_subject_ids"], 1)
        for field in ("role_assignment", "native_inventory", "unit_evidence"):
            self.assertTrue(work.checked(self.root, manifest[field]).is_file())
        self.assertFalse(manifest["training_started"])
        self.assertFalse(manifest["biological_independent_people_verified"])
        self.assertEqual(len(work.receipt_rows(self.root, result["backup_receipt"])), 5)

    def test_singleton_lock_rejects_second_worker_without_io(self):
        path = self.root/"audit/source_extensions_20261019/.source-extension-singleton.lock"
        path.parent.mkdir(parents=True)
        with path.open("a+") as lock, patch.object(work, "validate_plan", return_value={}):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(work.PreparationError, "source_extension_coordinator_already_running"):
                work.prepare(self.root, self.plan, execute=True)
        self.assertEqual(self.session.calls, [])

    def test_dry_run_has_no_download_import_fit_or_filesystem_outputs(self):
        before = set(self.root.rglob("*"))
        with patch.object(work, "validate_plan", return_value={}):
            result = work.prepare(self.root, self.plan, execute=False)
        self.assertEqual(result["status"], "DRY_RUN_FROZEN_SOURCE_PLAN_NO_NETWORK")
        self.assertEqual(before, set(self.root.rglob("*")))
        self.assertEqual(self.session.calls, [])

    def test_external_ma_and_credential_urls_are_not_accepted(self):
        for dataset in ("Ma2022", "HSStroke2026_v6", "Wang2025", "Leeuwis2021", "unknown"):
            with self.assertRaises(work.PreparationError):
                work.normalize_inventory(dataset, {})
        for url in ("http://data.nemar.org/x", "https://user:password@data.nemar.org/x",
                    "https://data.nemar.org/x?token=hidden", "https://unknown.org/data.edf"):
            with self.assertRaises(work.PreparationError):
                work.safe_url(url)

    def test_exact_official_dreyer_membership_and_sourcedata_urls_required(self):
        inventory = {"schema_version": "paper3-dreyer-source-object-plan-v1", "objects": []}
        for subject in range(1, 88):
            for run in range(1, 7):
                if subject == 59 and run in {5, 6}:
                    continue
                identity = f"sub-{subject:02d}"
                relative = f"{identity}/eeg/{identity}_task-R{run}{'acquisition' if run <= 2 else 'online'}_eeg.edf"
                inventory["objects"].append({"dataset_id": "Dreyer2023", "provider_subject_id": str(subject),
                    "run_index": run, "native_relative_path": relative, "manifest_path": "sourcedata/"+relative,
                    "url": "https://data.nemar.org/nm000250/v1.0.5/sourcedata/"+relative,
                    "bytes": 10, "sha256": "b"*64})
        normalized = work.normalize_inventory("Dreyer2023", inventory)
        self.assertEqual(len(normalized), 520)
        self.assertTrue(all(r["session_id"] == "provider_single_day" for r in normalized))
        inventory["objects"][0]["url"] = "https://data.nemar.org/nm000250/v1.0.5/sub-01/derived.bdf"
        with self.assertRaisesRegex(work.PreparationError, "official_identity_or_run_invalid"):
            work.normalize_inventory("Dreyer2023", inventory)

    def test_full_yang_2c_pair_selection_and_frozen_archive_identity_required(self):
        inventory = {"schema_version": "paper3-yang-source-download-selection-v1",
            "current_all312_inventory_fields_verified": True,
            "archive_url": "https://ndownloader.figshare.com/files/51001884", "archive_bytes": 65630556299,
            "current_central_directory_sha256": "cecd7ea0895da6ef85f01405960f4844c42d788ad33186b1f3fca91744d6874a",
            "members": []}
        for subject in range(1, 52):
            for session in range(1, 4):
                for name in ("data.bdf", "evt.bdf"):
                    inventory["members"].append({"path": "WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset/"
                        f"sub-{subject:03d}/ses-{session:02d}/eeg/{name}", "bytes": 10,
                        "expected_native_sha256": "b"*64, "crc32": "1234abcd", "header_offset": 1000,
                        "compressed_bytes": 5, "compression": 8})
        normalized = work.normalize_inventory("Yang2025", inventory)
        self.assertEqual(len(normalized), 153)
        self.assertEqual(len(normalized[0]["raw_files"]), 2)
        inventory["members"][0]["path"] = inventory["members"][0]["path"].replace("2C dataset", "3C dataset")
        with self.assertRaisesRegex(work.PreparationError, "yang_selection_contains_non2C_member"):
            work.normalize_inventory("Yang2025", inventory)

    def test_paths_symlinks_and_immutable_overwrite_rejected(self):
        for relative in ("../data", "/tmp/data", "a//b", "a/./b", "a\\b", "a/token.json"):
            with self.assertRaises(Exception):
                work.local(self.root, relative)
        link = self.root/"link"
        link.symlink_to(self.root/"audit", target_is_directory=True)
        with self.assertRaisesRegex(work.PreparationError, "symlink_forbidden"):
            work.local(self.root, "link/output.json")
        path = self.root/"immutable.json"
        work.atomic_json(path, {"x": 1}, immutable=True)
        with self.assertRaisesRegex(work.PreparationError, "existing_immutable_artifact_differs"):
            work.atomic_json(path, {"x": 2}, immutable=True)


if __name__ == "__main__":
    unittest.main()
