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

    def test_paper_reports_complete_and_partial_are_separate_from_full_archive(self):
        client = FixtureClient(receipts={
            "results/paper3/round-02/PAPER_DATA_LATEST.json": {"round": 2, "status": "PAPER_REPORT_FILES_UPLOADED"},
            "results/paper3/round-03/PAPER_DATA_LATEST.json": {"round": 3, "status": "PAPER_FILES_PARTIAL"}})
        value = m.collect(client, {}, {}, "2026-10-25T06:00:00Z")
        complete, partial = value["rounds"][1:3]
        self.assertTrue(complete["paper_reports"]["published"])
        self.assertTrue(complete["paper_reports"]["complete"])
        self.assertTrue(partial["paper_reports"]["published"])
        self.assertFalse(partial["paper_reports"]["complete"])
        self.assertFalse(complete["full_archive"]["published"])
        self.assertFalse(partial["automatic_summary"]["published"])
        table = m.compact_table(value)
        self.assertIn("[论文报告已上传]", table)
        self.assertIn("[部分论文文件已上传]", table)
        self.assertIn("round-02/PAPER_DATA_ZH.md", table)

    def test_paper_report_read_error_and_unknown_status_preserve_last_good_receipt(self):
        old_paper = {"published": True, "complete": True,
                     "receipt": {"round": 2, "status": "PAPER_REPORT_FILES_UPLOADED"}}
        previous = {"rounds": [{"round": 2, "paper_reports": old_paper}]}
        class DeniedClient(FixtureClient):
            def file(self, branch, path):
                if path == "results/paper3/round-02/PAPER_DATA_LATEST.json":
                    raise PermissionError("GitHubHTTP403")
                return super().file(branch, path)
        for client in (DeniedClient(), FixtureClient(receipts={
                "results/paper3/round-02/PAPER_DATA_LATEST.json": {"round": 2, "status": "UNKNOWN"}})):
            with self.subTest(client=type(client).__name__):
                value = m.collect(client, previous, {}, "2026-10-25T06:00:00Z")
                paper = value["rounds"][1]["paper_reports"]
                self.assertTrue(paper["published"])
                self.assertTrue(paper["complete"])
                self.assertEqual(paper["receipt"], old_paper["receipt"])
                self.assertIn("read_error", paper)
                self.assertIn("读取失败，保留上次记录", m.compact_table(value))

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


class RoundNineAndSupplementTests(unittest.TestCase):
    def navigation(self):
        return {"rounds": [{"round": 9, "status": "DESIGNED_NOT_SUBMITTED", "total_tasks": 1836,
            "source_url": "https://github.com/jackzhu119/research3/tree/paper3-round9-20261025"}]}

    def test_existing_round_monitor_paths_are_unchanged(self):
        self.assertEqual(m.SOURCES[1], ("paper3-r1-progress", "monitoring/paper3-round1/status.json"))
        self.assertEqual(m.SOURCES[6], ("paper3-r6-progress", "monitoring/paper3-round6/three-jobs-status.json"))
        self.assertEqual(m.SOURCES[7], ("paper3-r7-progress", "monitoring/paper3-round7/status.json"))
        self.assertEqual(m.SOURCES[9], ("paper3-r9-progress", "monitoring/paper3-round9/three-jobs-status.json"))

    def test_declared_unsubmitted_round_has_no_fake_failure_or_queue(self):
        data=m.collect(FixtureClient(), {}, {}, "2026-10-25T08:30:00Z", self.navigation())
        row=data['rounds'][8]
        self.assertEqual(row['declared_status'], 'DESIGNED_NOT_SUBMITTED')
        self.assertFalse(row['queue']['observed'])
        self.assertFalse(row['snapshot_stale'])
        self.assertNotIn('source_error',row)
        self.assertIsNone(row['complete_tasks'])
        self.assertIn('尚未提交',m.compact_table(data))
        self.assertIn('计划1836项',m.compact_table(data))

    def test_observed_monitor_overrides_design_declaration(self):
        c=FixtureClient(snapshots={'paper3-r9-progress':{'complete_tasks':0,'total_tasks':1836,
            'updated_utc':'2026-10-25T08:29:00Z','live_queue':{'status':'OBSERVED','jobs':[{'job_id':'900','state':'PENDING'}]}}})
        row=m.collect(c,{}, {}, '2026-10-25T08:30:00Z',self.navigation())['rounds'][8]
        self.assertEqual(row['status'],'作业排队')
        self.assertNotIn('declared_status',row)

    def test_r7_aggregate_receipt_is_distinct_from_csv_and_full_archive(self):
        path='results/paper3/round-07/REPORT_JSON_LATEST.json'
        r={'round':7,'status':'REPORT_JSON_LOSSLESS_UPLOADED','all_original_report_bytes_uploaded':True,'full_raw_results_uploaded':False}
        data=m.collect(FixtureClient(receipts={path:r}),{}, {},'2026-10-25T08:30:00Z')
        row=data['rounds'][6]
        self.assertTrue(row['aggregate_report']['published'])
        self.assertFalse(row['paper_reports']['published'])
        self.assertFalse(row['full_archive']['published'])
        self.assertIn('聚合JSON已补传',m.compact_table(data))
        self.assertIn('REPORT_JSON_ZH.md',m.compact_table(data))
        bad=m.collect(FixtureClient(receipts={path:{**r,'all_original_report_bytes_uploaded':False}}),{}, {},'2026-10-25T08:30:00Z')['rounds'][6]
        self.assertFalse(bad['aggregate_report']['published'])
        self.assertIn('read_error',bad['aggregate_report'])

    def test_new_full_archive_receipt_requires_completion_and_scope_flags(self):
        path='results/paper3/round-09/LATEST.json'
        r={'round':9,'status':'FULL_DECLARED_SCIENTIFIC_OUTPUT_UPLOADED','scientific_completion_claimed':True,'full_raw_results_uploaded':True}
        good=m.collect(FixtureClient(receipts={path:r}),{}, {},'2026-10-25T08:30:00Z')['rounds'][8]
        self.assertTrue(good['full_archive']['published'])
        bad=m.collect(FixtureClient(receipts={path:{**r,'scientific_completion_claimed':False}}),{}, {},'2026-10-25T08:30:00Z')['rounds'][8]
        self.assertFalse(bad['full_archive']['published'])


if __name__ == "__main__":
    unittest.main()
