"""Protocol protections; synthetic fixtures are never research observations."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from p3work.adapters.stieger_native import CLINIC11, file_sha256, load_session
from p3work.splits import build_prefix_episode, freeze_subject_split, _hash
from p3work.window import choose_window


class NativeReaderTest(unittest.TestCase):
    @staticmethod
    def write_numbered_run_fixture(path, runs):
        """Small native-structure software fixture; never real EEG evidence."""
        data, times, trials = [], [], []
        for run, numbers, tasks in runs:
            for number, task in zip(numbers, tasks):
                n = 5002
                data.append(np.ones((11, n), dtype=np.float32))
                times.append(np.arange(n) - 2000)
                trials.append({"tasknumber": task, "targetnumber": 1 + number % 2,
                    "runnumber": run, "trialnumber": number, "triallength": .001,
                    "resultind": 4002, "artifact": 0})
        cells, clocks = np.empty(len(data), object), np.empty(len(times), object)
        cells[:], clocks[:] = data, times
        savemat(path, {"BCI": {"data": cells, "time": clocks, "SRATE": 1000,
            "TrialData": np.array(trials, dtype=object), "metadata": {"MBSRsubject": 0},
            "chaninfo": {"label": np.array(CLINIC11, dtype=object), "noisechan": np.empty(0)}}},
            do_compression=True)

    def test_complete_physical_runs_accept_session_global_numbers_without_renumbering(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "S1_Session_1.mat"
            first, second = list(range(26, 51)), list(range(300, 325))
            self.write_numbered_run_fixture(path, [(2, first, [1] * 25), (13, second, [1] * 25)])
            session = load_session(path, expected_sha256=file_sha256(path))
            self.assertTrue(all(row["physical_run_complete_verified"] for row in session.rows))
            self.assertEqual([row["provider_trialnumber"] for row in session.rows], first + second)
            self.assertEqual([row["original_trial_index"] for row in session.rows], list(range(50)))
            self.assertEqual([row["physical_run_id"] for row in session.rows], [2] * 25 + [13] * 25)

    def test_incomplete_physical_run_keeps_24_actual_records_and_native_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "S1_Session_1.mat"
            numbers = list(range(276, 300))
            self.write_numbered_run_fixture(path, [(12, numbers, [1] * 24)])
            session = load_session(path, expected_sha256=file_sha256(path))
            self.assertEqual(len(session.rows), 24)
            self.assertEqual([row["provider_trialnumber"] for row in session.rows], numbers)
            self.assertTrue(all(row["physical_run_complete_verified"] is False for row in session.rows))

    def test_count_alone_cannot_verify_gap_nonpositive_or_mixed_task_run(self):
        cases = [
            (list(range(26, 38)) + list(range(39, 52)), [1] * 25),
            (list(range(25)), [1] * 25),
            (list(range(26, 51)), [1] * 24 + [3]),
        ]
        for numbers, tasks in cases:
            with self.subTest(numbers=numbers, tasks=tasks), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "S1_Session_1.mat"
                self.write_numbered_run_fixture(path, [(2, numbers, tasks)])
                session = load_session(path, expected_sha256=file_sha256(path))
                self.assertEqual(len(session.rows), 25)
                self.assertEqual([row["provider_trialnumber"] for row in session.rows], numbers)
                self.assertTrue(all(row["physical_run_complete_verified"] is False for row in session.rows))

    def test_provider_run_number_disorder_or_duplicate_is_rejected_before_completion(self):
        disordered = list(range(26, 51))
        disordered[1], disordered[2] = disordered[2], disordered[1]
        duplicated = list(range(26, 51)); duplicated[2] = duplicated[1]
        for numbers in (disordered, duplicated):
            with self.subTest(numbers=numbers), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "S1_Session_1.mat"
                self.write_numbered_run_fixture(path, [(2, numbers, [1] * 25)])
                with self.assertRaisesRegex(ValueError, "chronology inconsistent"):
                    load_session(path, expected_sha256=file_sha256(path))

    def test_native_task_filter_units_endpoint_and_sha(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "S1_Session_1.mat"
            data, times = np.empty(3, object), np.empty(3, object)
            td = []
            for i, (task, target, length) in enumerate([(1, 2, 3.), (3, 1, 3.), (1, 1, 1.)]):
                n = 4000 + int(length * 1000) + 1 + 1000
                times[i] = np.arange(n) - 2000
                data[i] = np.ones((11, n), dtype=np.float32) * 7
                td.append({"tasknumber": task, "targetnumber": target, "runnumber": i+1,
                           "trialnumber": 1, "triallength": length,
                           "resultind": 4000 + int(length*1000) + 1, "artifact": 0})
            savemat(p, {"BCI": {"data": data, "time": times, "SRATE": 1000,
                                "TrialData": np.array(td, dtype=object), "metadata": {"MBSRsubject": 0},
                                "chaninfo": {"label": np.array(CLINIC11, dtype=object), "noisechan": np.empty(0)}}})
            s = load_session(p, expected_sha256=file_sha256(p))
            self.assertEqual([r["label"] for r in s.pure_lr_metadata()], [0, 1])
            self.assertEqual(s.rows[1]["exclusion_reason"], "not_pure_horizontal_lr")
            np.testing.assert_allclose(s.context_volts(0, 3), 7e-6)
            self.assertEqual(s.context_volts(0, 3).shape, (11, 3000))
            with self.assertRaisesRegex(ValueError, "post-result padding"):
                s.context_volts(2, 2)
            with self.assertRaisesRegex(ValueError, "tasknumber"):
                s.context_volts(1, 2)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_session(p, expected_sha256="0"*64)


class PersonSplitTest(unittest.TestCase):
    @staticmethod
    def roster(n=62):
        return [{"participant_id": f"fixture-{i}", "identity_verified": True,
                 "clinic11_present_verified": True, "pure_lr_present_verified": True,
                 "intervention_group": str(i % 2)} for i in range(n)]

    def test_stable_person_disjoint_counts_and_unknown_evidence_rejected(self):
        roster = self.roster()
        a = freeze_subject_split(roster)
        self.assertEqual(a, freeze_subject_split(roster))
        self.assertEqual(a["counts"], {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11})
        self.assertEqual(sum(len(v) for v in a["partitions"].values()), 62)
        self.assertEqual(len(set(sum(a["partitions"].values(), []))), 62)
        roster[0]["clinic11_present_verified"] = None
        with self.assertRaisesRegex(ValueError, "clinic11"):
            freeze_subject_split(roster)
        with self.assertRaisesRegex(ValueError, "all 62"):
            freeze_subject_split(self.roster(3))

    def test_invalid_calibration_counts_and_boundary_run_tail_is_excluded(self):
        trials = [{"participant_id": "fixture", "session_id": "1", "trial_uid": str(i),
                   "original_trial_index": i, "tasknumber": 1, "label": i % 2,
                   "physical_run_id": 1 + i // 25, "physical_run_complete_verified": True, "valid": i != 0} for i in range(75)]
        e = build_prefix_episode(trials, acquisition_order_verified=True)
        self.assertEqual(e["acquired_calibration_trials"], 20)
        self.assertEqual(e["invalid_prefix_trials"], 1)
        self.assertEqual(e["support_uids"], [str(i) for i in range(1, 20)])
        self.assertEqual(e["unused_boundary_run_uids"], [str(i) for i in range(20, 25)])
        self.assertEqual(e["query_uids"], [str(i) for i in range(25, 75)])
        self.assertTrue(e["query_eligible"])
        missing_run_proof = copy.deepcopy(trials)
        for trial in missing_run_proof[25:]:
            trial["physical_run_complete_verified"] = None
        rejected = build_prefix_episode(missing_run_proof, acquisition_order_verified=True)
        self.assertFalse(rejected["query_eligible"])
        self.assertEqual(len(rejected["incomplete_or_unverified_future_run_uids"]), 50)
        with self.assertRaisesRegex(ValueError, "unverified"):
            build_prefix_episode(trials, acquisition_order_verified=False)
        for t in trials[:20]:
            t["label"] = 0
        self.assertTrue(build_prefix_episode(trials, acquisition_order_verified=True)["fallback_frozen"])


class WindowTest(unittest.TestCase):
    @staticmethod
    def session_manifest(rows, split):
        groups = {(r["participant_id"], r["session_id"]) for r in rows}
        manifest = {"split_sha256": split["split_sha256"], "inventory_sha256": "fixture-inventory",
                    "selection": "all_official_sessions_for_frozen_meta_train", "sessions": []}
        for person, session in sorted(groups):
            rr = [r for r in rows if (r["participant_id"], r["session_id"]) == (person, session)]
            manifest["sessions"].append({"participant_id": person, "session_id": session,
                                         "source_record_id": rr[0]["source_record_id"], "source_file_sha256": rr[0]["source_file_sha256"],
                                         "original_lr_trials": len(rr), "adapter_audit_sha256": "fixture-audit",
                                         "trial_metadata_sha256": "fixture-metadata"})
        manifest["manifest_sha256"] = _hash(manifest)
        return manifest

    def test_equal_person_weight_and_forbid_holdout_duration(self):
        split = {"status": "FROZEN_METADATA_ONLY", "split_sha256": "fixture",
                 "partitions": {"meta_train": ["a", "b"]}}
        rows = []
        # Person a: 100% long, b: 50% long => 75%, despite trial-weighted
        # retention near 100%. All have >=2s, so short must be selected.
        for p, n in [("a", 100), ("b", 4)]:
            for i in range(n):
                rows.append({"participant_id": p, "session_id": "1", "trial_uid": f"{p}-{i}",
                             "source_record_id": f"{p}-1.mat", "source_file_sha256": "a"*64,
                             "tasknumber": 1, "label": i%2, "feedback_end_verified": True,
                             "available_feedback_seconds": 3. if p == "a" or i < 2 else 2.})
        manifest = self.session_manifest(rows, split)
        result = choose_window(rows, split, expected_session_manifest=manifest)
        self.assertEqual(result["selected"], "short")
        self.assertEqual(result["candidates"]["3.0"]["participant_mean_retention"], .75)
        contaminated = copy.deepcopy(rows)
        contaminated[0]["participant_id"] = "internal-holdout"
        with self.assertRaisesRegex(ValueError, "forbidden"):
            choose_window(contaminated, split, expected_session_manifest=manifest)
        second_sessions = [dict(r, session_id="2", trial_uid=r["trial_uid"]+"-session2", source_record_id=r["participant_id"]+"-2.mat") for r in rows]
        full_manifest = self.session_manifest(rows+second_sessions, split)
        with self.assertRaisesRegex(ValueError, "partial-session"):
            choose_window(rows, split, expected_session_manifest=full_manifest)
        for row in rows:
            row["available_feedback_seconds"] = 1.
        self.assertEqual(choose_window(rows, split, expected_session_manifest=manifest)["status"], "BLOCKED_SHORT_WINDOW_COVERAGE")


if __name__ == "__main__":
    unittest.main()
