"""Synthetic-output tests; no EEG source caches or scientific results are used."""
import copy
from pathlib import Path

import numpy as np
import pytest

from paper3r4.io import digest, put, read
from paper3r4.report import (build_report, compare_rows, holm, paired_effect,
                            participant_summary, row_metrics, cv_selection_summary)


def plan(root, groups=("base", "candidate"), seeds=(20261018, 20261019, 20261020)):
    return {"root": str(root), "output": str(root), "recipe_sha256": "synthetic-recipe",
            "seeds": list(seeds), "budgets": [4, 20], "comparisons": [("candidate", "base")],
            "tasks": [{"key": f"{group}-{seed}", "family": "core", "seed": seed,
                       "group": group, "variant": group, "config": {"variant": group},
                       "budgets": [4, 20], "scenarios": ["clean"]}
                      for group in groups for seed in seeds]}


def row(person, session="1", *, budget=20, improved=False):
    labels = [0] * 5 + [1] * 5
    baseline = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    pred = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0] if improved else baseline
    probs = lambda values: [[.8, .2] if value == 0 else [.2, .8] for value in values]
    uids = [f"{person}-{session}-{i}" for i in range(len(labels))]
    return {"participant_id": person, "session_id": session, "budget": budget,
            "event_uids": uids, "labels": labels, "query_signature": digest({"uids": uids, "labels": labels}),
            "ba": .8 if improved else .6, "bce": .5, "f0_ba": .6,
            "negative_flip_rate": 0., "corrected_error_rate": .2 if improved else 0.,
            "predictions": pred, "f0_predictions": baseline,
            "probabilities": probs(pred), "f0_probabilities": probs(baseline),
            "calibration_seconds": .1 if improved else .05,
            "diagnostics": {"jacobian": {"effective_rank": 2., "eigenvalues": [1., .5, .1]}}}


def save_evaluation(folder, rows):
    for value in rows:
        put(folder / f"{value['participant_id']}-{value['session_id']}.json", value)
    put(folder / "summary.json", {"budget": rows[0]["budget"], "expected_sessions": [
        {key: value[key] for key in ("participant_id", "session_id", "query_signature")} for value in rows]})


def save_task(p, task, *, people=("S1", "S2")):
    folder = Path(p["output"]) / task["key"]
    put(folder / "result.json", {"status": "COMPLETE", "recipe_sha256": p["recipe_sha256"]})
    if task["family"] == "basis":
        return
    for budget in task.get("budgets", p["budgets"]):
        for scenario in task.get("scenarios", ["clean"]):
            save_evaluation(folder / "evaluations" / "holdout" / f"B{budget}" / scenario,
                            [row(person, budget=budget, improved=task["variant"] == "candidate") for person in people])


def test_balanced_probability_and_flip_metrics_are_recomputed():
    value = row("S1", improved=True)
    metrics = row_metrics(value)
    assert metrics["ba"] == pytest.approx(.8)
    assert metrics["ba"] - metrics["f0_ba"] == pytest.approx(
        metrics["corrected_error_rate"] - metrics["negative_flip_rate"])
    assert metrics["brier_balanced"] == pytest.approx(.8 * .2 ** 2 + .2 * .8 ** 2)
    assert metrics["ece_balanced_10bins"] == pytest.approx(0)
    assert metrics["f0_ece_balanced_10bins"] == pytest.approx(.2)
    invalid = copy.deepcopy(value)
    invalid["probabilities"][0] = [.9, .9]
    with pytest.raises(ValueError, match="probability"):
        row_metrics(invalid)
    invalid = copy.deepcopy(value)
    invalid["negative_flip_rate"] = .1
    with pytest.raises(ValueError, match="negative_flip"):
        row_metrics(invalid)


def test_equal_seed_then_equal_participant_weight_with_unequal_sessions():
    # S1 has three +.2 sessions in seed one, one zero session in seed two.
    # S2 has one zero session per seed. Correct mean is (.1 + 0) / 2.
    candidate, baseline = {}, {}
    for seed, sessions in ((1, ("1", "2", "3")), (2, ("1",))):
        candidate[seed], baseline[seed] = {}, {}
        for session in sessions:
            candidate[seed][("S1", session)] = row("S1", session, improved=seed == 1)
            baseline[seed][("S1", session)] = row("S1", session)
        candidate[seed][("S2", "1")] = row("S2")
        baseline[seed][("S2", "1")] = row("S2")
    result = compare_rows(candidate, baseline)
    assert result["n_participants"] == 2
    assert result["delta_ba_pp"] == pytest.approx(5.)
    assert result["participant_deltas_ba"]["S1"] == pytest.approx(.1)
    assert participant_summary(candidate)["ba"] == pytest.approx(.65)


def test_cv_choice_frequency_is_participant_weighted_and_discloses_missing_metadata():
    # S1: three A sessions in seed 1, one B in seed 2 => A fraction .5.
    # S2: one B in each seed => 0. Equal-person A frequency is .25, not .5.
    values = {1: {}, 2: {}}
    for seed, sessions in ((1, ("1", "2", "3")), (2, ("1",))):
        for session in sessions:
            value = row("S1", session)
            value["diagnostics"] = {"cv": {"selected_candidate": "A" if seed == 1 else "B",
                "selected_config": {"head_l2": .1 if seed == 1 else 1.},
                "selection_rule": "one_standard_error", "scores": [3., 5., 6.]}}
            values[seed][("S1", session)] = value
        value = row("S2")
        value["diagnostics"] = {"cv": {"selected_candidate": "B"}}
        values[seed][("S2", "1")] = value
    values[1][("S3", "1")] = row("S3")  # Fixed adapter/older row: no CV metadata.
    result = cv_selection_summary(values)
    choices = {r["selection_value"]: r for r in result if r["selection_field"] == "cv.selected_candidate"}
    assert choices["A"]["participant_mean_fraction"] == pytest.approx(.25)
    assert choices["B"]["participant_mean_fraction"] == pytest.approx(.75)
    assert choices["A"]["n_participants"] == 2
    assert choices["A"]["observed_choice_session_seed_rows"] == 3
    assert choices["A"]["n_session_seed_rows"] == 6
    assert choices["A"]["excluded_session_seed_rows"] == 1
    assert choices["A"]["n_excluded_participants"] == 1
    assert any(r["selection_field"] == "cv.selected_config.head_l2" for r in result)
    assert not any("scores" in r["selection_field"] for r in result)
    assert cv_selection_summary({1: {("S1", "1"): row("S1")}}) == []


def test_report_adds_cv_diagnostics_without_requiring_metadata_for_fixed_baseline(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        save_task(p, task)
        if task["variant"] == "candidate":
            folder = Path(p["output"]) / task["key"] / "evaluations/holdout/B20/clean"
            for path in folder.glob("S*.json"):
                value = read(path)
                value["diagnostics"]["cv"] = {"selected_candidate": "ridge1-radius0.05",
                    "selected_config": {"head_l2": 1., "radius": .05}, "valid_folds": 20,
                    "selection_rule": "one_standard_error", "status": "SELECTED"}
                put(path, value)
    report = build_report(p)
    assert report["status"] == "COMPLETE" and report["primary_family_complete"]
    assert report["cv_selection_frequency"]
    assert all(r["group"] == "candidate" for r in report["cv_selection_frequency"])
    selected = [r for r in report["cv_selection_frequency"] if r["selection_field"] == "cv.selected_candidate"]
    assert len(selected) == 1 and selected[0]["participant_mean_fraction"] == 1.
    assert selected[0]["n_participants"] == 2
    assert any(r["diagnostic"] == "cv.valid_folds" and r["mean"] == 20. for r in report["diagnostics"])
    assert Path(report["paths"]["csv"]["cv_selection_frequency"]).is_file()


def test_fixed_holm_family_withholds_all_p_until_all_primary_seeds(tmp_path):
    p = plan(tmp_path, groups=("base", "candidate", "other"))
    p["comparisons"] = [("candidate", "base"), ("candidate", "other")]
    for task in p["tasks"]:
        if task["group"] != "other":
            save_task(p, task)
    partial = build_report(p)
    assert partial["status"] == "PARTIAL"
    assert not partial["primary_family_complete"]
    assert partial["planned_comparison_count"] == 2
    assert len(partial["comparisons"]) == 2
    assert partial["comparisons"][0]["ci95_pp"] is not None
    assert all(value["p"] is None and value["p_holm"] is None for value in partial["comparisons"])
    for task in p["tasks"]:
        if task["group"] == "other":
            save_task(p, task)
    complete = build_report(p)
    assert complete["status"] == "COMPLETE"
    assert complete["primary_family_complete"]
    assert all(value["n_participants"] == 2 for value in complete["comparisons"])
    assert all(value["holm_family_size"] == 2 for value in complete["comparisons"])
    assert all(value["p_holm"] is not None for value in complete["comparisons"])
    assert complete["exploratory_internal_replication"] and not complete["fresh_confirmatory"]
    assert Path(complete["paths"]["json"]).is_file()
    assert (tmp_path / "reports" / "budget_curves.csv").is_file()
    assert read(tmp_path / "reports" / "report.json")["comparisons"] == complete["comparisons"]


def test_one_missing_seed_never_gets_partial_primary_p(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"][:-1]:
        save_task(p, task)
    result = build_report(p)
    comparison = result["comparisons"][0]
    assert comparison["completed_seed_count"] == 2
    assert comparison["pending_seeds"] == [20261020]
    assert comparison["n_participants"] == 2  # Not four seed/person observations.
    assert comparison["p"] is None and comparison["ci95_pp"] is not None


def test_unrelated_pending_tasks_do_not_expand_or_block_primary_family(tmp_path):
    p = plan(tmp_path)
    p["tasks"].extend([
        {"key": "basis", "family": "basis", "seed": 20261018},
        {"key": "sensitivity-pending", "family": "sensitivity", "seed": 20261018,
         "group": "candidate-r2", "variant": "candidate", "baseline_group": "base"}])
    for task in p["tasks"]:
        if task["family"] != "sensitivity":
            save_task(p, task)
    result = build_report(p)
    assert result["status"] == "PARTIAL" and result["primary_family_complete"]
    assert result["planned_comparison_count"] == 1
    assert result["comparisons"][0]["p_holm"] is not None
    assert "basis" in result["completed_tasks"]


def test_secondary_missing_values_use_own_explicit_complete_case_n():
    a = {1: {("S1", "1"): row("S1", improved=True), ("S2", "1"): row("S2", improved=True)}}
    b = {1: {("S1", "1"): row("S1"), ("S2", "1"): row("S2")}}
    a[1][("S2", "1")]["calibration_seconds"] = None
    result = compare_rows(a, b)
    timing = result["secondary_metrics"]["calibration_seconds"]
    assert result["n_participants"] == 2
    assert timing["n_participants"] == 1 and timing["n_excluded_participants"] == 1
    assert timing["delta"] == pytest.approx(.05) and timing["ci95"] is None
    assert timing["excluded_metric_session_seed_rows"] == 1


def test_exact_signflip_11_people_and_zero_case():
    effect = paired_effect([.1] * 11)
    assert effect["p_method"] == "exact_paired_sign_flip"
    assert effect["p"] == pytest.approx(2 / 2048)
    assert paired_effect([0.] * 11)["p"] == 1.
    assert holm([.04, .001, .02]) == pytest.approx([.04, .003, .04])


def test_invalid_completed_task_blocks_primary_without_touching_worker(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        save_task(p, task)
    folder = tmp_path / p["tasks"][0]["key"] / "evaluations" / "holdout" / "B20" / "clean"
    (folder / "S2-1.json").unlink()
    before = read(tmp_path / p["tasks"][0]["key"] / "result.json")
    result = build_report(p)
    assert result["status"] == "ERROR" and result["invalid_tasks"] == [p["tasks"][0]["key"]]
    assert not result["primary_family_complete"]
    assert result["comparisons"][0]["p"] is None
    assert "Missing or extra" in result["issues"][0]["message"]
    assert read(tmp_path / p["tasks"][0]["key"] / "result.json") == before


def test_query_mismatch_never_produces_significance(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        save_task(p, task)
    folder = tmp_path / "candidate-20261018" / "evaluations" / "holdout" / "B20" / "clean"
    values = [row("S1", improved=True), row("S2", improved=True)]
    values[0]["event_uids"][0] = "different-trial"
    values[0]["query_signature"] = digest({"uids": values[0]["event_uids"], "labels": values[0]["labels"]})
    save_evaluation(folder, values)
    result = build_report(p)
    assert result["status"] == "ERROR" and not result["primary_family_complete"]
    assert "Query trial" in result["issues"][0]["message"]


def test_stress_and_sensitivity_pairs_remain_descriptive_with_diagnostics(tmp_path):
    p = plan(tmp_path)
    for seed in p["seeds"]:
        p["tasks"].append({"key": f"sensitivity-{seed}", "family": "sensitivity", "seed": seed,
                           "group": "candidate-r2", "variant": "candidate", "baseline_group": "base",
                           "budgets": [20], "scenarios": ["clean"]})
        p["tasks"].append({"key": f"stress-{seed}", "family": "stress", "seed": seed,
                           "group": "candidate-stress", "variant": "candidate", "clean_group": "candidate",
                           "budgets": [20], "scenarios": ["gain25"]})
    for task in p["tasks"]:
        save_task(p, task)
    result = build_report(p)
    assert result["primary_family_complete"]
    assert len(result["sensitivity_comparisons"]) == len(result["stress_comparisons"]) == 1
    assert result["sensitivity_comparisons"][0]["p"] is None
    assert result["stress_comparisons"][0]["p"] is None
    assert result["stress_comparisons"][0]["baseline_scenario"] == "clean"
    assert result["diagnostics"] and result["timings"]
    assert any(value["diagnostic"] == "jacobian.eigenvalues.min" for value in result["diagnostics"])


def test_same_group_stress_compares_against_own_clean_coordinate(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        task["scenarios"] = ["clean", "gain25"]
        save_task(p, task)
        if task["group"] == "candidate":
            save_evaluation(tmp_path / task["key"] / "evaluations" / "holdout" / "B20" / "gain25",
                            [row("S1"), row("S2")])
    result = build_report(p)
    own = [value for value in result["stress_comparisons"]
           if value["candidate"] == value["baseline"] == "candidate" and value["budget"] == 20]
    assert len(own) == 1
    assert own[0]["baseline_scenario"] == "clean"
    assert own[0]["delta_ba_pp"] == pytest.approx(-20.)
    assert own[0]["p"] is None


def test_diagnostics_budget_requires_stress_only_at_b20_and_keeps_all_clean_budgets(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        task["scenarios"] = ["clean", "gain25"]
        task["diagnostics_budget"] = 20
        save_task(p, task)
        folder = tmp_path / task["key"] / "evaluations" / "holdout" / "B4" / "gain25"
        for path in folder.glob("*.json"):
            path.unlink()
        folder.rmdir()
    result = build_report(p)
    assert result["status"] == "COMPLETE" and result["primary_family_complete"]
    assert {value["budget"] for value in result["budget_curves"]} == {4, 20}
    assert {value["budget"] for value in result["stress"]} == {20}
    assert all(value["p"] is None for value in result["stress_comparisons"])
    # The explicitly promised B20 scenario is still mandatory.
    missing = tmp_path / p["tasks"][0]["key"] / "evaluations" / "holdout" / "B20" / "gain25"
    for path in missing.glob("*.json"):
        path.unlink()
    missing.rmdir()
    invalid = build_report(p)
    assert invalid["status"] == "ERROR" and not invalid["primary_family_complete"]
    assert "B20/gain25" in invalid["issues"][0]["message"]


def test_query_mismatch_across_budgets_is_invalid_even_before_other_seeds(tmp_path):
    p = plan(tmp_path)
    task = p["tasks"][0]
    save_task(p, task)
    values = [row("S1", budget=4), row("S2", budget=4)]
    values[0]["event_uids"][0] = "different-trial"
    values[0]["query_signature"] = digest({"uids": values[0]["event_uids"], "labels": values[0]["labels"]})
    save_evaluation(tmp_path / task["key"] / "evaluations" / "holdout" / "B4" / "clean", values)
    result = build_report(p)
    assert result["status"] == "ERROR"
    assert result["invalid_tasks"] == [task["key"]]
    assert "across seeds or budgets" in result["issues"][0]["message"]


def test_report_failure_is_returned_and_never_raised(tmp_path):
    result = build_report({"output": str(tmp_path), "tasks": [], "seeds": []})
    assert result["status"] == "ERROR"
    assert result["issues"] and result["comparisons"] == []
    assert (tmp_path / "reports" / "report.json").is_file()


def test_duplicate_session_cannot_silently_overwrite(tmp_path):
    p = plan(tmp_path)
    for task in p["tasks"]:
        save_task(p, task)
    folder = tmp_path / "base-20261018" / "evaluations" / "holdout" / "B20" / "clean"
    put(folder / "duplicate.json", row("S1"))
    result = build_report(p)
    assert result["status"] == "ERROR" and not result["primary_family_complete"]
    assert "Duplicate participant/session" in result["issues"][0]["message"]
