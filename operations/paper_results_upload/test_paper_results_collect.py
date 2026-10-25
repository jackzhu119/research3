import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import paper_results_collect as collect


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "project"
        self.output = self.root / "runs/followup_20261023_v2"
        self.output.mkdir(parents=True)
        self.work = self.base / "paper-export"
        self.write("plan.json", {"tasks": [{"key": "a"}, {"key": "b"}]})
        self.write("reports/report.json", {"status": "COMPLETE", "complete_tasks": 2, "total_tasks": 2})
        self.write("reports/paired.csv", "participant,delta\np1,.1\n")
        self.write("a/result.json", {"status": "COMPLETE"})
        self.write("b/result.json", {"status": "COMPLETE"})

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        return path

    def run_collect(self, **kwargs):
        return collect.collect_round(self.root, 2, self.work, **kwargs)

    def test_complete_receipts_and_reports_are_independent(self):
        value = self.run_collect()
        self.assertTrue(value["manifest"]["task_counts"]["all_plan_tasks_verified_complete"])
        self.assertEqual(value["manifest"]["reports_observed"][0]["path"], "reports/report.json")
        for row in value["files"]:
            source = self.output / row["path"]
            self.assertEqual(Path(row["local_path"]).read_bytes(), source.read_bytes())
        self.assertFalse(value["manifest"]["raw_trial_predictions_included"])

    def test_fast_export_does_not_read_task_files_or_descend(self):
        self.write("a/checkpoints/secret.pt", "do not read")
        self.write("reports/nested/unwanted.csv", "do not read")
        original = collect._read_file
        paths = []
        def read(base, relative, maximum):
            paths.append(relative)
            return original(base, relative, maximum)
        with patch.object(collect, "_read_file", side_effect=read), patch("os.walk", side_effect=AssertionError("recursive walk")), patch.object(Path, "rglob", side_effect=AssertionError("recursive glob")):
            value = self.run_collect(include_tasks=False)
        self.assertEqual(set(paths), {"plan.json", "report.json", "reports/report.json", "reports/paired.csv"})
        self.assertIsNone(value["manifest"]["task_counts"]["verified_complete_tasks"])
        self.assertFalse(value["manifest"]["task_counts"]["all_plan_tasks_verified_complete"])

    def test_incomplete_receipt_does_not_erase_complete_report(self):
        self.write("b/result.json", {"status": "RUNNING"})
        value = self.run_collect()
        self.assertEqual(value["manifest"]["task_counts"]["verified_complete_tasks"], 1)
        self.assertEqual(value["manifest"]["reports_observed"][0]["status"], "COMPLETE")
        self.assertFalse(value["manifest"]["task_counts"]["all_plan_tasks_verified_complete"])

    def test_bad_json_is_skipped_but_csv_continues(self):
        self.write("reports/report.json", "{broken")
        value = self.run_collect(include_tasks=False)
        self.assertIn("reports/paired.csv", [r["path"] for r in value["files"]])
        self.assertIn("invalid_json", [r["reason"] for r in value["manifest"]["warnings"]])

    def test_nonfinite_json_is_skipped_without_breaking_manifest(self):
        self.write("reports/report.json", '{"status":"COMPLETE","complete_tasks":NaN}')
        value = self.run_collect(include_tasks=False)
        self.assertIn("invalid_json", [r["reason"] for r in value["manifest"]["warnings"]])
        self.assertTrue(Path(value["manifest_path"]).is_file())

    def test_duplicate_and_escaping_keys_never_read_outside(self):
        self.write("plan.json", {"tasks": [{"key": "a"}, {"key": "a"}, {"key": "../outside"}, {"key": "/tmp/outside"}]})
        value = self.run_collect()
        self.assertEqual(value["manifest"]["task_counts"]["task_receipts_read"], 1)
        self.assertFalse(value["manifest"]["task_counts"]["all_plan_tasks_verified_complete"])

    def test_symlink_files_and_task_directories_are_not_followed(self):
        outside = self.base / "outside.json"
        outside.write_text('{"status":"COMPLETE"}')
        (self.output / "a/result.json").unlink()
        (self.output / "a/result.json").symlink_to(outside)
        (self.output / "b/result.json").unlink()
        (self.output / "b").rmdir()
        (self.output / "b").symlink_to(outside.parent, target_is_directory=True)
        value = self.run_collect()
        self.assertEqual(value["manifest"]["task_counts"]["verified_complete_tasks"], 0)

    def test_reports_symlink_is_not_enumerated(self):
        import shutil
        shutil.rmtree(self.output / "reports")
        (self.output / "reports").symlink_to(self.base, target_is_directory=True)
        value = self.run_collect(include_tasks=False)
        self.assertIn("symlink_directory", [r["reason"] for r in value["manifest"]["warnings"]])

    def test_report_oversize_is_not_fatal(self):
        value = self.run_collect(include_tasks=False, max_report_bytes=5)
        self.assertIn("file_size_limit", [r["reason"] for r in value["manifest"]["warnings"]])
        self.assertIn("reports/paired.csv", [r["path"] for r in value["files"]])

    def test_explicit_64mib_limit_keeps_report_above_10mib(self):
        report = {"status": "COMPLETE", "large_table": "x" * (10 * collect.MIB + 10)}
        path = self.write("reports/report.json", report)
        self.assertGreater(path.stat().st_size, 10 * collect.MIB)
        value = self.run_collect(include_tasks=False, max_report_bytes=64 * collect.MIB, max_file_bytes=64 * collect.MIB)
        self.assertIn("reports/report.json", [r["path"] for r in value["files"]])

    def test_mutation_of_open_file_rejected(self):
        path = self.write("changing.csv", "a,b\n1,2\n")
        original = collect.os.fstat
        calls = 0
        def fstat(fd):
            nonlocal calls
            value = original(fd)
            calls += 1
            if calls == 2:
                return types.SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino,
                    st_size=value.st_size, st_mtime_ns=value.st_mtime_ns+1,
                    st_ctime_ns=value.st_ctime_ns)
            return value
        with patch.object(collect.os, "fstat", side_effect=fstat):
            with self.assertRaisesRegex(ValueError, "source_changed_during_read"):
                collect._read_file(self.output, path.name, 100)

    def test_source_files_unchanged_and_no_operational_private_files(self):
        self.write("private/github.json", {"token": "must not be collected"})
        before = {str(p.relative_to(self.output)): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()}
        value = self.run_collect(include_tasks=False)
        after = {str(p.relative_to(self.output)): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn("private/github.json", [r["path"] for r in value["files"]])

    def test_output_work_must_not_be_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside_source"):
            collect.collect_round(self.root, 2, self.output / "upload")


if __name__ == "__main__":
    unittest.main()
