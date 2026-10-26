"""Collect completed metrics from actual plan declarations without trial scans."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

import collector


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def real_plan(round_number, root, output):
    folder = Path("/workspace") / ("paper3-round%d%s" % (round_number, "-hpcgpu" if round_number in (4, 5, 6) else ""))
    file = folder / ("paper3r%d/plan.py" % round_number)
    spec = importlib.util.spec_from_file_location("actual_plan_r%d" % round_number, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if round_number == 8:
        plan = module.build(root, output)
    else:
        plan = module.make_plan(root, output)
    chosen = []
    for family in (("class_reference", "train", "core", "stress", "label_null") if round_number == 8 else
                   (("basis", "core", "transfer", "reference", "sensitivity") if round_number == 6 else
                    ("basis", "core", "reference", "sensitivity"))):
        chosen.append(next(task for task in plan["tasks"] if task.get("family") == family))
    plan["tasks"] = chosen
    return plan


def make_round(tmp_path, number):
    root = tmp_path / "school-project"
    output = root / "runs" / collector.OUTPUT_NAMES[number]
    plan = real_plan(number, root, output)
    put(output / "plan.json", plan)
    receipts, summaries = {}, {}
    for index, task in enumerate(plan["tasks"]):
        receipt = {"status": "COMPLETE", "task": task, "identity": "frozen-identity-" + str(index),
                   "source_identity": {"source_receipt_sha256": "source-v1"},
                   "data_identity": {"manifest_sha256": "cache-v1"},
                   "runtime_corrections": {"runtime_hook_sha256": "actual-hook-version"},
                   "selection": {"chosen_on": "meta_validation", "unabridged": [1, 2, {"中文": "保留"}]} }
        if number == 8:
            receipt["recipe_sha256"] = plan["recipe_sha256"]
        path = output / task["key"] / "result.json"
        put(path, receipt)
        receipts[task["key"]] = (path.read_bytes(), receipt)
        for partition, budget, scenario in collector._coordinates(plan, task):
            summary = {"participants": [{"participant_id": "S1", "ba": .625, "bce": .681},
                                         {"participant_id": "S2", "ba": .75, "bce": .52}],
                       "participant_mean_ba": .6875, "participant_mean_bce": .6005,
                       "sessions": 3, "partition": partition, "budget": budget, "scenario": scenario,
                       "expected_sessions": [{"participant_id": "S1", "session_id": "1", "query_signature": "q1"},
                                             {"participant_id": "S2", "session_id": "2", "query_signature": "q2"}],
                       "parameter_id": "fixed-parameters", "data_identity": {"cache": "source-cache"}}
            path = output / task["key"] / "evaluations" / partition / ("B%d" % budget) / scenario / "summary.json"
            put(path, summary)
            summaries[str(path.relative_to(root))] = (path.read_bytes(), summary)
    put(output / "reports/report.json", {"status": "PARTIAL", "complete_tasks": 1, "total_tasks": 999})
    (output / "reports/paired.csv").write_text("metric,value\nba,0.6875\n", encoding="utf-8")
    (output / "reports/README.md").write_text("Existing scientific report\n")
    (output / "participant_effects.csv").write_text("participant_id,ba,bce\nS1,0.625,0.681\n")
    # Such files must never be traversed by compact capture.
    put(output / plan["tasks"][0]["key"] / "evaluations" / "unregistered-session.json",
        {"secret_trial_tree_marker": "NOT_A_DECLARED_SUMMARY"})
    return root, output, plan, receipts, summaries


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@pytest.mark.parametrize("number", [4, 5, 6, 8])
def test_real_plan_coordinates_receipts_and_exact_metrics_are_preserved(tmp_path, monkeypatch, number):
    root, output, plan, receipts, summaries = make_round(tmp_path, number)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    def no_recursive_scan(*args, **kwargs):
        raise AssertionError("Compact upload must not scan trial/session trees")
    monkeypatch.setattr(Path, "rglob", no_recursive_scan)
    events = []
    result = collector.collect_round(root, number, tmp_path / "upload-work", lambda event, **kw: events.append((event, kw)))
    stage, metadata = result["source_base"], result["metadata"]
    assert metadata["all_complete"] is True
    assert metadata["paper_metric_evidence_complete"] is True
    assert metadata["complete_tasks"] == len(plan["tasks"])
    assert metadata["expected_condition_summaries"] == metadata["captured_condition_summaries"] == len(summaries)
    assert metadata["per_session_or_trial_files_scanned"] is False
    rows = lines(stage / "analysis/task-results.jsonl")
    assert len(rows) == len(receipts)
    for row in rows:
        raw, expected = receipts[row["task_key"]]
        assert row["receipt"] == expected
        assert row["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["receipt"]["runtime_corrections"]["runtime_hook_sha256"] == "actual-hook-version"
        if number != 8:
            assert row["recipe_verification"].startswith("legacy_receipt_no_explicit_recipe_hash")
    for row in lines(stage / "analysis/condition-summaries.jsonl"):
        raw, expected = summaries[row["source_path"]]
        assert row["summary"] == expected
        assert row["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["summary"]["expected_sessions"][0]["query_signature"] == "q1"
    assert (stage / "plan.json").read_bytes() == (output / "plan.json").read_bytes()
    assert (stage / "reports/paired.csv").read_bytes() == (output / "reports/paired.csv").read_bytes()
    report = next(row for row in metadata["reports"] if row["path"] == "reports/report.json")
    assert report["reported_metadata"]["status"] == "PARTIAL"
    assert metadata["all_complete"] is True  # stale aggregate report is not a new gate
    assert metadata["collection_status"] == "COMPLETE"
    assert metadata["reports_stale_or_unknown"] is True
    assert "participant_effects.csv" in result["paths"]
    assert not any("unregistered" in path for path in result["paths"])
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert events[-1][0] == "COLLECT_ROUND_READY"


def test_single_budget_diagnostics_follow_worker_skip_rule():
    for number in (4, 5, 6, 8):
        task = {"key": "a", "family": "core", "partition": "external_published_order",
                "budgets": [20, 4, 8], "scenarios": ["clean", "gain25", "label_noise10"]}
        actual = collector._coordinates({}, task)
        assert len(actual) == 5
        assert ("external_published_order", 4, "gain25") not in actual
        assert ("external_published_order", 20, "gain25") in actual


def test_missing_or_mismatching_receipt_does_not_discard_other_evidence(tmp_path):
    root, output, plan, receipts, summaries = make_round(tmp_path, 8)
    path = output / plan["tasks"][0]["key"] / "result.json"
    path.unlink()
    path = output / plan["tasks"][1]["key"] / "result.json"
    bad = json.loads(path.read_text()); bad["recipe_sha256"] = "different"; put(path, bad)
    result = collector.collect_round(root, 8, tmp_path / "upload-work")
    assert result["metadata"]["all_complete"] is False
    assert result["metadata"]["collection_status"] == "PARTIAL"
    assert result["metadata"]["required_missing"]
    assert result["metadata"]["complete_tasks"] == len(plan["tasks"]) - 2
    assert len(lines(result["source_base"] / "analysis/task-results.jsonl")) == len(plan["tasks"]) - 1
    assert result["metadata"]["captured_condition_summaries"] == len(summaries)
    assert {x["code"] for x in result["metadata"]["warnings"]} >= {"receipt_unreadable", "receipt_recipe_mismatch"}


def test_missing_summary_and_missing_participant_nll_are_explicit(tmp_path):
    root, output, plan, receipts, summaries = make_round(tmp_path, 5)
    source_paths = [root / name for name in summaries]
    source_paths[0].unlink()
    bad = json.loads(source_paths[1].read_text()); bad["participants"][0].pop("bce"); put(source_paths[1], bad)
    result = collector.collect_round(root, 5, tmp_path / "upload-work")
    meta = result["metadata"]
    assert meta["all_complete"] is True
    assert meta["paper_metric_evidence_complete"] is False
    assert meta["captured_condition_summaries"] == len(summaries) - 1
    assert meta["condition_summaries_with_participant_BA_NLL"] == len(summaries) - 2
    assert {x["code"] for x in meta["warnings"]} >= {"condition_summary_unreadable", "summary_participant_BA_or_NLL_missing"}


def test_report_above_64_mib_is_streamed_exactly_and_not_parsed(tmp_path, monkeypatch):
    root, output, plan, receipts, summaries = make_round(tmp_path, 4)
    large = output / "reports/full-report.json"
    with large.open("wb") as stream:
        stream.write(b'{"huge":"')
        for _ in range(65):
            stream.write(b"x" * 1024 * 1024)
        stream.write(b'"}')
    # The large file must never pass through a whole read_bytes JSON parser.
    native_read = Path.read_bytes
    def guarded_read(path):
        if path.name == "full-report.json":
            raise AssertionError("Large report must be streamed")
        return native_read(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    result = collector.collect_round(root, 4, tmp_path / "upload-work")
    item = next(row for row in result["metadata"]["reports"] if row["path"] == "reports/full-report.json")
    assert item["bytes"] > 64 * 1024 * 1024
    assert item["reported_metadata"]["parse_status"] == "LARGE_REPORT_CAPTURED_NOT_PARSED"
    staged = result["source_base"] / item["path"]
    assert staged.stat().st_size == large.stat().st_size
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
        return h.hexdigest()
    assert digest(staged) == digest(large) == item["sha256"]
    assert item["path"] in result["paths"]
    assert result["metadata"]["large_reports_skipped"] is False


def test_repeat_capture_and_reject_staging_inside_outputs(tmp_path):
    root, output, plan, receipts, summaries = make_round(tmp_path, 6)
    first = collector.collect_round(root, 6, tmp_path / "upload-work")
    second = collector.collect_round(root, 6, tmp_path / "upload-work")
    assert first["paths"] == second["paths"]
    assert first["metadata"]["complete_tasks"] == second["metadata"]["complete_tasks"]
    with pytest.raises(ValueError, match="outside scientific"):
        collector.collect_round(root, 6, output / "not-allowed")


def test_collector_has_no_numeric_network_or_scheduler_imports():
    import ast
    tree = ast.parse(Path(collector.__file__).read_text())
    modules = {node.module.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    modules |= {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert not (modules & {"numpy", "torch", "requests", "urllib", "subprocess"})


def test_report_atomically_replaced_during_capture_retries_to_consistent_version(tmp_path, monkeypatch):
    source = tmp_path / "report.csv"
    source.write_bytes(b"old-header\n" + b"a" * 2048)
    replacement = tmp_path / "replacement.csv"
    new = b"new-header\n" + b"b" * 4096
    replacement.write_bytes(new)
    actual_open = Path.open
    replaced = False
    class ReplacingReader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, size):
            nonlocal replaced
            block = self.stream.read(size)
            if not replaced:
                replacement.replace(source)
                replaced = True
            return block
    def open_snapshot(path, *args, **kwargs):
        stream = actual_open(path, *args, **kwargs)
        if path == source and args and args[0] == "rb": return ReplacingReader(stream)
        return stream
    monkeypatch.setattr(Path, "open", open_snapshot)
    destination = tmp_path / "stage/report.csv"
    result = collector._capture_file(source, destination)
    assert destination.read_bytes() == new
    assert result["bytes"] == len(new)
    assert result["sha256"] == hashlib.sha256(new).hexdigest()
    assert result["source_signature"] == list(collector._signature(source.stat()))


def test_report_inventory_and_signatures_expose_new_and_replaced_reports(tmp_path):
    root, output, plan, receipts, summaries = make_round(tmp_path, 5)
    first = collector.collect_round(root, 5, tmp_path / "upload-work")
    metadata = first["metadata"]
    assert metadata["report_inventory"] == sorted(str(path.relative_to(root))
                                                  for path in collector.report_candidates(output))
    csv_path = output / "participant_effects.csv"
    row = next(item for item in metadata["reports"] if item["source_path"] == str(csv_path.relative_to(root)))
    assert row["source_signature"] == list(collector._signature(csv_path.stat()))
    with csv_path.open("a") as stream:
        stream.write("S3,0.875,0.4\n")
    assert row["source_signature"] != list(collector._signature(csv_path.stat()))
    put(output / "runtime/runtime-corrections.json", {"version": "new-runtime-evidence"})
    new_csv = output / "reports/new_analysis.csv"
    new_csv.write_text("metric,value\nba,0.625\n")
    new_inventory = sorted(str(path.relative_to(root)) for path in collector.report_candidates(output))
    assert new_inventory != metadata["report_inventory"]
    assert str(new_csv.relative_to(root)) in new_inventory
    assert str((output / "runtime/runtime-corrections.json").relative_to(root)) in new_inventory
