"""Synthetic reporting-contract tests; no EEG performance is asserted."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from paper3r7.io import put, digest
from paper3r8 import report


def plan(tmp_path, people=3, sessions=2, seeds=(11, 12, 13), partitions=("pan", "weibo")):
    return {"output": str(tmp_path), "recipe_sha256": "synthetic-r8", "seeds": list(seeds),
            "budgets": [20], "primary_metrics": ["ba", "nll_balanced"],
            "primary_partitions": list(partitions), "comparisons": [["new", "base"]],
            "expected_cohorts": {p: {"participants": people, "sessions": people*sessions} for p in partitions},
            "tasks": [{"key": f"{p}-{group}-{seed}", "seed": seed, "group": group,
                       "family": "transfer", "partition": p, "budgets": [20], "scenarios": ["clean"]}
                      for p in partitions for group in ("base", "new") for seed in seeds]}


def row(person="P1", session="D1", *, improve=False):
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 1, 1, 1, 0]
    if improve:
        predictions = y
    probability = [[.8, .2] if c == 0 else [.2, .8] for c in predictions]
    ids = [f"{person}-{session}-{i}" for i in range(len(y))]
    return {"participant_id": person, "session_id": session, "budget": 20, "scenario": "clean",
            "labels": y, "event_uids": ids, "query_signature": digest({"uids": ids, "labels": y}),
            "predictions": predictions, "probabilities": probability,
            "log_probabilities": np.log(probability).tolist(),
            "f0_predictions": predictions, "f0_probabilities": probability,
            "f0_log_probabilities": np.log(probability).tolist(), "parameter_id": "parameter",
            "diagnostics": {"alpha": .5}, "fallback": False}


def save(p, task, *, people=3, sessions=2, force_new=False):
    folder = Path(p["output"])/task["key"]
    evaldir = folder/"evaluations"/task["partition"]/"B20"/"clean"
    rows = []
    for person in range(people):
        for session in range(sessions):
            r = row(f"P{person}", f"D{session}", improve=task["group"] == "new" or force_new)
            r["partition"] = task["partition"]
            put(evaldir/f'{r["participant_id"]}-{r["session_id"]}.json', r)
            rows.append(r)
    put(evaldir/"summary.json", {"sessions": len(rows), "parameter_id": "parameter", "expected_sessions": [
        {k: r[k] for k in ("participant_id", "session_id", "query_signature")} for r in rows]})
    put(folder/"result.json", {"status": "COMPLETE", "recipe_sha256": p["recipe_sha256"], "task": task,
                              "identity": "synthetic-identity"})


def complete(p, **kwargs):
    for task in p["tasks"]:
        save(p, task, **kwargs)


def test_paired_participant_family_uses_all_datasets_metrics_and_matching_seeds(tmp_path):
    p = plan(tmp_path)
    complete(p)
    r = report.build_report(p)
    assert r["status"] == "COMPLETE" and r["primary_family_complete"]
    assert r["complete_tasks"] == r["total_tasks"] == 12
    assert r["planned_primary_hypothesis_count"] == 4
    primary = r["primary_multimetric_comparisons"]
    assert len(primary) == 4
    assert all(v["n_participants"] == 3 and v["n_paired_session_seed_rows"] == 18 for v in primary)
    assert all(v["all_required_seeds_ready"] and v["matched_session_grid"] for v in primary)
    assert [v["p_holm"] for v in primary] == pytest.approx(report.holm([v["p"] for v in primary]))
    assert all(v["delta"] > 0 for v in primary)
    ba = next(v for v in primary if v["metric"] == "ba")
    nll = next(v for v in primary if v["metric"] == "nll_balanced")
    assert ba["delta"] == pytest.approx(25.) and ba["unit"] == "percentage_points"
    assert nll["direction"] == "baseline_minus_candidate" and nll["candidate_mean"] < nll["baseline_mean"]
    assert r["fresh_confirmatory"] is False
    assert r["input_sha256"] == r["provenance"]["input_merkle_sha256"]
    assert len(r["provenance"]["input_files"]) == 12 * (1+1+6)


def test_missing_seed_withholds_whole_holm_family_but_reports_partial_observations(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"][:-1]:
        save(p, task)
    r = report.build_report(p)
    assert r["status"] == "PARTIAL" and r["complete_tasks"] == 11
    assert not r["primary_family_complete"]
    assert len(r["primary_multimetric_comparisons"]) == 4
    assert all(v["p"] is None and v["p_holm"] is None for v in r["primary_multimetric_comparisons"])
    assert r["evaluation_summaries"]


def test_missing_session_is_error_not_silent_complete_case_selection(tmp_path):
    p = plan(tmp_path)
    complete(p)
    task = p["tasks"][-1]
    next((Path(p["output"])/task["key"]/"evaluations"/task["partition"]/"B20"/"clean").glob("P0-*.json")).unlink()
    r = report.build_report(p)
    assert r["status"] == "PARTIAL" and r["complete_tasks"] == 11
    assert any("Missing or extra" in i["error"] for i in r["issues"])
    assert not r["primary_family_complete"]


def test_all_receipts_complete_but_wrong_cohort_size_precludes_inference(tmp_path):
    p = plan(tmp_path, people=14)
    complete(p, people=3)
    r = report.build_report(p)
    assert r["status"] == "COMPLETE"
    assert not r["primary_family_complete"]
    assert all(not x["expected_cohort_ready"] and x["p"] is None for x in r["primary_multimetric_comparisons"])


def test_query_identity_diff_across_methods_precludes_inference(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    task = p["tasks"][-1]
    d = Path(p["output"])/task["key"]/"evaluations"/"pan"/"B20"/"clean"
    f = d/"P0-D0.json"
    r = json.loads(f.read_text())
    r["event_uids"][0] += "-wrong"
    r["query_signature"] = digest({"uids": r["event_uids"], "labels": r["labels"]})
    put(f, r)
    s = json.loads((d/"summary.json").read_text())
    s["expected_sessions"][0]["query_signature"] = r["query_signature"]
    put(d/"summary.json", s)
    result = report.build_report(p)
    assert result["status"] == "COMPLETE" and not result["primary_family_complete"]
    assert all(not x["matched_session_grid"] for x in result["primary_multimetric_comparisons"])


def test_cross_seed_grid_must_match_not_only_each_candidate_baseline_pair(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    for task in p["tasks"]:
        if task["seed"] != 13:
            continue
        d = Path(p["output"])/task["key"]/"evaluations"/"pan"/"B20"/"clean"
        f = d/"P0-D0.json"
        r = json.loads(f.read_text()); r["session_id"] = "other"
        put(f, r)
        s = json.loads((d/"summary.json").read_text());s["expected_sessions"][0]["session_id"] = "other"
        put(d/"summary.json", s)
    result = report.build_report(p)
    assert not result["primary_family_complete"]
    assert all(not x["matched_session_grid"] for x in result["primary_multimetric_comparisons"])


def test_corrupted_probability_or_claimed_metric_is_explicit_error(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    task = p["tasks"][0]
    f = Path(p["output"])/task["key"]/"evaluations"/"pan"/"B20"/"clean"/"P0-D0.json"
    r = json.loads(f.read_text());r["ba"] = 0.99
    put(f, r)
    result = report.build_report(p)
    assert result["status"] == "PARTIAL" and result["issues"]
    assert "disagrees" in result["issues"][0]["error"]


def test_fallback_rows_are_included_not_dropped(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    for task in p["tasks"]:
        d = Path(p["output"])/task["key"]/"evaluations"/"pan"/"B20"/"clean"
        for f in d.glob("P*.json"):
            r = json.loads(f.read_text());r.update(fallback=True, fallback_reason="single_support_class")
            put(f, r)
    result = report.build_report(p)
    assert result["primary_family_complete"]
    assert all(x["metrics"]["fallback_fraction"]["mean"] == 1 for x in result["evaluation_summaries"])


def test_true_log_probabilities_preserve_nll_beyond_probability_underflow():
    r = row()
    r["labels"] = [0, 1, 0, 1]
    r["event_uids"] = ["a", "b", "c", "d"]
    lp = np.array([[0., -1000.], [-1000., 0.], [-1000., 0.], [0., -1000.]])
    r["probabilities"] = np.exp(lp).tolist();r["log_probabilities"] = lp.tolist()
    r["predictions"] = [0, 1, 1, 0]
    for k in ["f0_probabilities", "f0_log_probabilities", "f0_predictions", "query_signature"]:
        del r[k]
    assert report._compact_row(r)["metrics"]["nll_balanced"] == pytest.approx(500.)


def test_statistical_unit_is_person_then_equal_seed_not_pooled_sessions():
    def r(person, value):
        return {"participant_id": person, "metrics": {"ba": value}}
    values = {1: [r("A", 1), r("A", 1), r("A", 1), r("B", 0)],
              2: [r("A", 0), r("B", 0)]}
    people, count, missing = report._by_person(values, "ba")
    assert people == {"A": .5, "B": 0.} and np.mean(list(people.values())) == .25
    assert count == 6 and missing == 0


def test_exact_sign_flip_matches_brute_force_and_includes_twenty_participants():
    d = (.1, -.2, .3, .4)
    brute = sum(abs(np.dot(s, d)) >= abs(sum(d))-1e-12 for s in itertools.product((-1,1), repeat=4))/16
    assert report.sign_flip(d)["p"] == brute
    assert report.sign_flip((.1,)*20)["p"] == pytest.approx(2/(1<<20))
    assert report.sign_flip((.1,)*20)["p_method"] == "exact_paired_sign_flip"
    assert report.sign_flip((0.,)*20)["p"] == 1.


def test_monte_carlo_pvalue_is_deterministic_positive_with_fixed_seed():
    d = tuple(np.linspace(-.1, .3, 21))
    a = report.sign_flip(d)
    report.sign_flip.cache_clear()
    assert a == report.sign_flip(d)
    assert a["p_method"] == "monte_carlo_paired_sign_flip" and a["p"] > 0


def test_completed_task_cache_avoids_reopening_predictions_and_refresh_rechecks(tmp_path, monkeypatch):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    first = report.build_report(p)
    original = report._condition_rows
    monkeypatch.setattr(report, "_condition_rows", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reread")))
    second = report.build_report(p)
    assert second["status"] == "COMPLETE" and second["provenance"]["cached_tasks_reused"] == 6
    assert first["input_sha256"] == second["input_sha256"]
    monkeypatch.setattr(report, "_condition_rows", original)
    task = p["tasks"][0]
    f = Path(p["output"])/task["key"]/"evaluations"/"pan"/"B20"/"clean"/"P0-D0.json"
    r = json.loads(f.read_text());r["ba"] = -1
    put(f, r)
    assert report.build_report(p, refresh=True)["status"] == "PARTIAL"


def test_changed_receipt_invalidates_only_changed_task_cache(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    report.build_report(p)
    f = Path(p["output"])/p["tasks"][0]["key"]/"result.json"
    v = json.loads(f.read_text());v["identity"] = "new"
    put(f,v)
    r = report.build_report(p)
    assert r["provenance"]["cached_tasks_reused"] == 5 and r["provenance"]["validated_tasks_this_pass"] == 1


def test_wrong_recipe_and_duplicate_primary_declarations_are_explicit(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    f = Path(p["output"])/p["tasks"][0]["key"]/"result.json"
    v = json.loads(f.read_text());v["recipe_sha256"] = "other"
    put(f,v)
    assert report.build_report(p)["status"] == "PARTIAL"
    p["comparisons"].append(["new","base"])
    r = report.build_report(p)
    assert not r["primary_family_complete"]
    assert any(i.get("scope") == "primary_declarations" for i in r["issues"])


def test_failed_task_stays_partial_and_is_not_counted_as_complete(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    f = Path(p["output"])/p["tasks"][0]["key"]/"result.json"
    v = json.loads(f.read_text());v["status"] = "FAILED"
    put(f,v)
    r = report.build_report(p)
    assert r["status"] == "PARTIAL" and r["complete_tasks"] == 5
    assert any(x["status"] == "FAILED" for x in r["task_status"])


def test_non_evaluation_preparation_tasks_are_counted_without_fake_eeg_rows(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    task = {"key":"class-reference", "seed":11, "family":"class_reference"}
    p["tasks"].append(task)
    put(tmp_path/task["key"]/"result.json", {"status":"COMPLETE", "recipe_sha256":p["recipe_sha256"],"task":task})
    r = report.build_report(p)
    assert r["complete_tasks"] == 7 and len(r["evaluation_summaries"]) == 2


def test_csv_chunks_contain_all_rows_and_remove_only_stale_own_chunks(tmp_path):
    destination = tmp_path/"reports"
    destination.mkdir()
    rows = [{"group": str(i), "value": i, "participant_differences": {"private":1}} for i in range(2001)]
    files = report._csv_tables(destination, "diagnostics", rows)
    assert len(files) == 3
    assert all("private" not in (destination/name).read_text() for name in files)
    unrelated = destination/"unrelated.csv"; unrelated.write_text("retained")
    files = report._csv_tables(destination, "diagnostics", rows[:2])
    assert files == ["diagnostics-0000.csv"]
    assert len(list(destination.glob("diagnostics-*.csv"))) == 1 and unrelated.exists()


def test_prediction_source_identity_must_agree_with_receipt(tmp_path):
    p = plan(tmp_path, partitions=("pan",))
    complete(p)
    f = tmp_path/p["tasks"][0]["key"]/"result.json"
    value = json.loads(f.read_text());value["source_identity"] = {"checkpoint_sha256":"wrong"}
    put(f, value)
    r = report.build_report(p)
    assert r["status"] == "PARTIAL"
    assert "source_identity" in r["issues"][0]["error"]

