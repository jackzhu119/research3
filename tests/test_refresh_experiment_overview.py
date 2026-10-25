"""Status semantics matter: a running coordinator is not a running GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/refresh_experiment_overview.py"
SPEC = importlib.util.spec_from_file_location("overview", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FixtureClient:
    repository = "jackzhu119/research3"
    def __init__(self, snapshots=None, receipts=None):
        self.snapshots, self.receipts = snapshots or {}, receipts or {}
    def file(self, branch, path):
        if branch == "paper3-results":
            return self.receipts.get(path), "results-head"
        return self.snapshots.get(branch), "source-head"


class StatusTests(unittest.TestCase):
    def test_old_counter_cannot_erase_confirmed_completion(self):
        value = m.normalize(7, {"complete_tasks": 20, "total_tasks": 273, "report_status": "COMPLETE"},
                            {"completed": True, "complete_tasks": 273, "total_tasks": 273, "evidence": "school diagnostic"})
        self.assertEqual(value["complete_tasks"], 273)
        self.assertEqual(value["snapshot_complete_tasks"], 20)
        self.assertTrue(value["counts_conflict"])

    def test_running_controller_with_pending_queue_is_queued(self):
        value = m.normalize(8, {"status": "RUNNING", "complete_tasks": 0, "total_tasks": 2835,
            "live_queue": {"status": "OBSERVED", "jobs": [{"job_id": "1", "state": "PENDING", "reason_or_node": "(AssocGrpGRES)"}]}})
        self.assertEqual(value["status"], "作业排队")
        self.assertEqual(value["queue"]["running"], 0)

    def test_cached_job_state_is_not_live_queue_evidence(self):
        value = m.normalize(8, {"status": "RUNNING", "jobs": [{"slurm_state": "RUNNING"}], "complete_tasks": 0, "total_tasks": 2835})
        self.assertFalse(value["queue"]["observed"])
        self.assertEqual(value["status"], "最近回传状态；未确认当前调度")

    def test_first_round_stage_count_is_not_whole_round_percentage(self):
        value = m.normalize(1, {"parallel_plan_complete_tasks": 47, "parallel_plan_total_tasks": 61,
                              "completed_by_kind": {"meta_screening": 33}})
        self.assertIsNone(value["complete_tasks"])
        self.assertEqual(value["completed_by_kind"]["meta_screening"], 33)

    def test_first_round_future_phase_is_not_permanently_screening(self):
        value = m.normalize(1, {"coordinator": {"phase": "META_CONFIRMATION"}})
        self.assertEqual(value["status"], "阶段记录：额外种子元学习确认")

    def test_first_round_all_complete_and_mainline_complete_are_distinct(self):
        self.assertEqual(m.normalize(1, {"all_original_complete": True})["status"], "全部原计划完成")
        self.assertEqual(m.normalize(1, {"mainline_complete": True})["status"], "主线完成；既有扩展进度另计")

    def test_first_round_old_interruption_only_when_current_action_contains_it(self):
        snapshot = {"coordinator": {"phase": "META_SCREENING"}, "updated_utc": "2026-10-25T05:00:00Z",
            "mainline": {"active_actions": [{"key": "meta-one", "status": "RUNNING", "observations": [
                {"age_seconds": 3000, "values": {"status": "INTERRUPTED_NOT_COMPLETE", "outer_updates_completed": 350}}]}]}}
        value = m.normalize(1, snapshot)
        self.assertIn("旧中断断点", value["status"])
        self.assertEqual(value["old_interruption_observations"][0]["snapshot_observed_utc"], snapshot["updated_utc"])
        snapshot["mainline"]["active_actions"][0]["observations"][0]["values"]["status"] = "RESUMED"
        self.assertNotIn("旧中断断点", m.normalize(1, snapshot)["status"])

    def test_missing_full_receipt_and_published_summary_stay_distinct(self):
        client = FixtureClient(receipts={"results/paper3/round-02/AUTO_SUMMARY_LATEST.json": {
            "round": 2, "status": "AUTOMATIC_EXISTING_RESULT_SUMMARY_UPLOADED", "full_raw_results_uploaded": False}})
        value = m.collect(client, {}, {}, "2026-10-25T06:00:00Z")
        row = value["rounds"][1]
        self.assertFalse(row["full_archive"]["published"])
        self.assertTrue(row["automatic_summary"]["published"])

    def test_full_receipt_requires_status_and_matching_round(self):
        client = FixtureClient(receipts={"results/paper3/round-02/LATEST.json": {
            "round": 2, "status": "COMPLETE_OUTPUT_UPLOADED"},
            "results/paper3/round-03/LATEST.json": {"round": 7, "status": "COMPLETE_OUTPUT_UPLOADED"}})
        value = m.collect(client, {}, {}, "2026-10-25T06:00:00Z")
        self.assertTrue(value["rounds"][1]["full_archive"]["published"])
        self.assertFalse(value["rounds"][2]["full_archive"]["published"])

    def test_fetch_failure_retains_last_good_observation_with_error(self):
        old = {"rounds": [{"round": 2, "status": "任务完成", "complete_tasks": 66, "total_tasks": 66,
                           "updated_utc": "2026-10-25T02:16:55Z"}]}
        value = m.collect(FixtureClient(), old, {}, "2026-10-25T06:00:00Z")
        self.assertEqual(value["rounds"][1]["complete_tasks"], 66)
        self.assertIn("source_error", value["rounds"][1])

    def test_first_fetch_failure_still_retains_documented_completion(self):
        known = {"rounds": {"7": {"completed": True, "complete_tasks": 273, "total_tasks": 273}}}
        value = m.collect(FixtureClient(), {}, known, "2026-10-25T06:00:00Z")
        self.assertEqual(value["rounds"][6]["complete_tasks"], 273)
        self.assertEqual(value["rounds"][6]["status"], "任务完成")

    def test_timestamp_only_refresh_does_not_rewrite_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("before\n" + m.START + "\n" + m.END + "\nafter\n")
            value = m.collect(FixtureClient(), {}, {}, "2026-10-25T06:00:00Z")
            self.assertTrue(m.update(root, value))
            paths = [root / "README.md", root / "docs/LIVE_STATUS.json", root / "docs/LIVE_STATUS_ZH.md"]
            before = [p.stat().st_mtime_ns for p in paths]
            value["refreshed_utc"] = "2026-10-25T06:30:00Z"
            self.assertFalse(m.update(root, value))
            self.assertEqual(before, [p.stat().st_mtime_ns for p in paths])


if __name__ == "__main__":
    unittest.main()
