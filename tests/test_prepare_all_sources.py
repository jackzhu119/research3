"""Orchestration mocks with ephemeral fixtures; no actual data or network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_all_sources as wrapper
from tools import source_preparation_batch as batch
from tests.test_source_preparation_batch import DATA, fake_backup, fake_native, response, synthetic_w1
from tools.prepare_source_cache import begin_owned_cache_restore


class PrepareAllSourcesTest(unittest.TestCase):
    def setUp(self):
        # Component orchestration uses an explicit software proof. The real
        # guard is tested unmocked in test_window_acceptance.py.
        proof = {"artifact_sha256": "e" * 64, "input_contract_sha256": "1" * 64,
                 "SOFTWARE_TEST_DOUBLE": True}
        for module in (wrapper, batch):
            mock = patch.object(module, "verify_frozen_window", return_value=proof)
            mock.start()
            self.addCleanup(mock.stop)

    def test_default_dry_run_is_candidate_only_and_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch.object(batch, "execution_gate", side_effect=AssertionError("dry run gate not needed")):
                result = wrapper.prepare_all(root, raw_fn=lambda *a, **kw: self.fail("network path"))
            self.assertEqual(result["status"], "CANDIDATE_PLAN_ONLY")
            self.assertEqual(result["candidate_objects"], 505)
            self.assertFalse(result["provider_roster_verified"])
            self.assertEqual(list(root.iterdir()), [])

    def test_w1_gate_blocks_before_progress_or_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch.atomic_json(root / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"}})
            with self.assertRaisesRegex(batch.BatchError, "w1_complete"):
                wrapper.prepare_all(root, execute=True, raw_fn=lambda *a, **kw: self.fail("not gated"))
            self.assertFalse((root / "audit").exists())

    def test_paths_must_stay_in_independent_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for path in ("../progress.json", "/tmp/paper3-outside-root-progress.json"):
                with self.subTest(path=path), self.assertRaises((wrapper.PreparationError, batch.BatchError)):
                    wrapper.prepare_all(root, progress=path)

    def test_partial_raw_does_not_create_cache_or_advance_w2(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic_w1(root)
            with patch.object(batch, "validate_signal_contract", return_value="1"*64):
                with self.assertRaisesRegex(wrapper.PreparationError, "raw_preparation_incomplete"):
                    wrapper.prepare_all(root, execute=True,
                        raw_fn=lambda *a, **kw: {"inventory_status": "SOURCE_RAW_INVENTORY_PARTIAL", "actual_verified_objects": 4},
                        cache_fn=lambda *a, **kw: self.fail("cache ran before full raw"))
            progress = json.loads((root / "audit/source_full/progress.json").read_text())
            self.assertEqual(progress["verified_raw_objects"], 4)
            self.assertEqual(progress["status"], "BLOCKED_NO_W2_ADVANCE")
            self.assertEqual(json.loads((root / "status/tasks.json").read_text())["W2"]["status"], "PENDING")

    def fixture(self, root):
        synthetic_w1(root)
        batch.atomic_json(root / "status/tasks.json", {"W0": {"status": "COMPLETE", "keep": "W0"},
            "W1": {"status": "COMPLETE"}, "W2": {"status": "PENDING"}, "W3": {"status": "BLOCKED", "keep": "W3"}})
        plan = batch.build_plan()
        inventory = root / "audit/source_preparation/batches" / plan["plan_sha256"] / "source_native_inventory.json"
        evidence = root / "audit/source_schema/evidence_manifest.json"
        batch.atomic_json(evidence, {"files": []})
        batch.atomic_json(inventory, {"status": "SOURCE_RAW_INVENTORY_COMPLETE", "provider_roster_verified": True,
            "files": [], "provider_terms_evidence": batch.descriptor(root, evidence)})
        catalog = root / "audit/source_full/raw_r2_receipts_catalog.json"
        batch.atomic_json(catalog, {"synthetic_test_fixture_only": True})
        cache_dir = root / "cache/source_v2"
        shards = []
        for dataset in sorted(batch.VERSIONS):
            shard = {"dataset": dataset}
            for field, name in (("X", "X.npy"), ("y", "y.npy"), ("events", "events.json"), ("exclusions", "exclusions.json")):
                path = cache_dir / dataset / "synthetic-fixture" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic-file-fixture-not-real-cache")
                shard[field] = batch.descriptor(root, path)
            shards.append(shard)
        cache_path = cache_dir / "source_cache.json"
        batch.atomic_json(cache_path, {"schema_version": "paper3-source-cache-v2", "status": "SOURCE_CACHE_COMPLETE",
            "synthetic": False, "old_npz_reused": False, "full_source_inventory_verified": True,
            "input_contract_sha256": "1"*64,
            "training_started": False, "raw_inventory": batch.descriptor(root, inventory), "shards": shards,
            "participants_without_usable_trials": []})
        cached = {"status": "SOURCE_CACHE_COMPLETE", "formal_w2_complete": True, "cache_manifest": batch.descriptor(root, cache_path)}
        return catalog, cache_path, cached

    def run_success_path(self, root, *, backup_fn=fake_backup, cache_override=None):
        catalog, cache_path, cached = self.fixture(root)
        if cache_override:
            cached = cache_override
        calls = []
        def cache_fn(*args, **kwargs):
            calls.append((args, kwargs))
            return cached
        with patch.object(batch, "validate_signal_contract", return_value="1"*64), patch.object(wrapper, "raw_receipt_catalog", return_value=(catalog, [])):
            result = wrapper.prepare_all(root, execute=True,
                progress=root/"audit/source_full/custom_progress.json",
                raw_fn=lambda *a, **kw: {"inventory_status": "SOURCE_RAW_INVENTORY_COMPLETE", "actual_verified_objects": 505},
                cache_fn=cache_fn, backup_fn=backup_fn)
        return result, calls, cache_path

    def test_w2_only_after_real_cache_manifest_and_strict_complete_get_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result, calls, cache_path = self.run_success_path(root)
            self.assertEqual(result["status"], "W2_COMPLETE_CACHE_AND_R2_VERIFIED")
            self.assertTrue(calls[0][1]["execute"])
            self.assertTrue(calls[0][1]["restore_backed_raw"])
            tasks = json.loads((root / "status/tasks.json").read_text())
            self.assertEqual(tasks["W0"]["keep"], "W0")
            self.assertEqual(tasks["W3"]["keep"], "W3")
            self.assertEqual(tasks["W2"]["status"], "COMPLETE")
            self.assertEqual(tasks["W2"]["cache_manifest"]["sha256"], batch.file_digest(cache_path))
            receipt_path = root / "audit/source_full/cache_backup_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            self.assertTrue(all(row["get"]["complete_body_read"] for row in receipt["objects"]))
            self.assertFalse(any(row["role"] == "source_development_raw" for row in receipt["objects"]))
            self.assertFalse(any(Path(row["path"]).suffix.lower() in {".mat", ".edf"} for row in receipt["objects"]))

    def test_incomplete_or_corrupt_cache_receipt_cannot_advance_w2(self):
        for corruption in ("failed_status", "incomplete_get", "wrong_sha", "missing_object"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                def backup(**kw):
                    result = fake_backup(**kw)
                    if corruption == "failed_status":
                        result["status"] = "failed"
                    elif corruption == "incomplete_get":
                        result["objects"][0]["get"]["complete_body_read"] = False
                    elif corruption == "wrong_sha":
                        result["objects"][0]["get"]["sha256"] = "0"*64
                    else:
                        result["objects"].pop()
                    batch.atomic_json(kw["receipt_path"], result)
                    return result
                with self.assertRaises(wrapper.PreparationError):
                    self.run_success_path(root, backup_fn=backup)
                tasks = json.loads((root / "status/tasks.json").read_text())
                self.assertEqual(tasks["W2"]["status"], "PENDING")

    def test_partial_cache_blocks_before_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(wrapper.PreparationError, "cache_incomplete"):
                self.run_success_path(root, cache_override={"status": "SOURCE_CACHE_INCOMPLETE", "formal_w2_complete": False},
                    backup_fn=lambda **kw: self.fail("must not backup incomplete cache"))
            self.assertEqual(json.loads((root / "status/tasks.json").read_text())["W2"]["status"], "PENDING")

    def test_worker_exception_details_suppressed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            synthetic_w1(root)
            with patch.object(batch, "validate_signal_contract", return_value="1"*64):
                with self.assertRaisesRegex(wrapper.PreparationError, "details_suppressed"):
                    wrapper.prepare_all(root, execute=True,
                        raw_fn=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("secret exception should be suppressed")))
            self.assertNotIn("secret exception", (root / "audit/source_full/progress.json").read_text())

    def test_cli_success_exit_code_requires_actual_final_completion(self):
        for status, code in (("CANDIDATE_PLAN_ONLY", 2), ("SOURCE_CACHE_INCOMPLETE", 2),
                             ("W2_COMPLETE_CACHE_AND_R2_VERIFIED", 0)):
            with self.subTest(status=status), patch.object(wrapper, "prepare_all", return_value={"status": status}), patch("builtins.print"):
                self.assertEqual(wrapper.main([]), code)

    def test_wrapper_hard_interruption_after_restore_restarts_worker_and_continues_cache(self):
        class SimulatedHardInterruption(BaseException):
            """Bypass normal Exception handlers and cache finally cleanup."""

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog, cache_path, cached = self.fixture(root)
            plan = batch.build_plan()
            entry = plan["files"][0]
            batch_dir = root / "audit/source_preparation/batches" / plan["plan_sha256"]
            raw = batch.raw_staging_path(root, plan, entry)
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(DATA)
            state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
            batch.atomic_json(state_path, {"identity": list(batch.identity(entry)), "object_id": entry["object_id"],
                "plan_sha256": plan["plan_sha256"], "input_contract_sha256": "1"*64, "status": "RAW_VERIFIED",
                "raw_path": raw.relative_to(root).as_posix(), "temporary_raw_owned": True})
            with patch.object(batch, "check_space"):
                state = batch.process_file(entry, root=root, batch_dir=batch_dir, plan=plan,
                    contract_sha256="1"*64, backup_fn=fake_backup, audit_fn=fake_native,
                    opener=lambda *a, **kw: response(size=len(DATA)))
            self.assertFalse(raw.exists())
            item = {**entry, "object_uid": entry["object_id"], "path": state["raw_path"],
                    "sha256": state["raw_integrity"]["sha256"], "bytes": state["raw_integrity"]["bytes"],
                    "backup_receipt": state["backup_receipt"], "temporary_raw_owned": True,
                    "raw_storage": "verified_r2_only"}

            def raw_stage(received_plan, *, root):
                self.assertEqual(received_plan, plan)
                # Actual completed-object replay must resolve the durable cache
                # restore marker before the wrapper can enter cache preparation.
                batch.process_file(entry, root=root, batch_dir=batch_dir, plan=plan, contract_sha256="1"*64,
                    opener=lambda *a, **kw: self.fail("replay must not download"),
                    backup_fn=lambda **kw: self.fail("replay must not re-upload"))
                return {"inventory_status": "SOURCE_RAW_INVENTORY_COMPLETE", "actual_verified_objects": 505}

            def interrupted_cache(*args, **kwargs):
                begin_owned_cache_restore(root, item)
                raw.write_bytes(DATA)  # Complete mock R2 body, never real data.
                raise SimulatedHardInterruption()

            with patch.object(batch, "validate_signal_contract", return_value="1"*64), patch.object(wrapper, "raw_receipt_catalog", return_value=(catalog, [])):
                with self.assertRaises(SimulatedHardInterruption):
                    wrapper.prepare_all(root, execute=True, raw_fn=raw_stage, cache_fn=interrupted_cache,
                                        backup_fn=lambda **kw: self.fail("interrupted cache cannot archive"))
                self.assertTrue(raw.exists())
                self.assertEqual(json.loads((root / "status/tasks.json").read_text())["W2"]["status"], "PENDING")

                def resumed_cache(*args, **kwargs):
                    self.assertFalse(raw.exists(), "worker must reconcile interrupted ownership before cache restart")
                    return cached

                result = wrapper.prepare_all(root, execute=True, raw_fn=raw_stage, cache_fn=resumed_cache,
                                             backup_fn=fake_backup)
            self.assertEqual(result["status"], "W2_COMPLETE_CACHE_AND_R2_VERIFIED")
            self.assertTrue(json.loads(state_path.read_text())["recovered_owned_cache_restore"])
            self.assertFalse(raw.exists())
            self.assertEqual(json.loads((root / "status/tasks.json").read_text())["W2"]["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
