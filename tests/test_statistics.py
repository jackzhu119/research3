"""Synthetic mathematical fixtures only; no real data analysis, fit or network."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from p3work import statistics as stats
from p3work.contracts import file_digest


def prediction_fixture(specifications, seeds=stats.EXTERNAL_PIPELINE_SEEDS):
    """specifications: participant -> session -> (class size, P-correct, B-correct)."""
    records, manifest = [], {}
    for person, sessions in specifications.items():
        manifest[person] = {}
        for session, (n, candidate_correct, baseline_correct) in sessions.items():
            uids = [f"software-fixture:{person}:{session}:{c}:{trial}" for c in (0, 1) for trial in range(n)]
            manifest[person][session] = uids
            for seed in seeds:
                for method, correct in (("P", candidate_correct), ("B2", baseline_correct)):
                    for c in (0, 1):
                        for index in range(n):
                            records.append({"dataset_id": "software_fixture", "participant_id": person,
                                "session_id": session, "pipeline_seed": seed, "method_id": method,
                                "event_uid": uids[c * n + index], "label": c,
                                "prediction": c if index < correct else 1 - c,
                                "fallback": candidate_correct == baseline_correct})
    return records, manifest


def summary(records, manifest, seeds=stats.EXTERNAL_PIPELINE_SEEDS):
    return stats.hierarchical_paired_summary(records, manifest, dataset_id="software_fixture",
        candidate="P", comparator="B2", pipeline_seeds=seeds)


def test_BA_equal_class_recall_and_G_H_common_denominators_not_accuracy_or_conditionals():
    y = np.array([0] * 90 + [1] * 10)
    old = y.copy()
    old[:45] = 1
    old[95:] = 0
    new = old.copy()
    new[:10] = 0
    new[45:50] = 1
    new[95:99] = 1
    new[90:92] = 0
    result = stats.error_transitions(y, old, new)
    assert result["classes"]["0"]["N_c"] == 90
    assert result["classes"]["0"]["G_c"] == pytest.approx(10 / 90)
    assert result["classes"]["0"]["H_c"] == pytest.approx(5 / 90)
    assert result["classes"]["1"]["G_c"] == .4
    assert result["classes"]["1"]["H_c"] == .2
    expected = ((10 - 5) / 90 + (4 - 2) / 10) / 2
    assert result["delta_balanced_accuracy"] == pytest.approx(expected)
    assert result["delta_balanced_accuracy"] == pytest.approx(stats.balanced_accuracy(y, new) - stats.balanced_accuracy(y, old))
    assert result["ordinary_accuracy_difference"] == pytest.approx(.07)
    assert result["delta_balanced_accuracy"] != pytest.approx(.07)
    assert result["classes"]["1"]["conditional_correction_given_old_error"] == .8


def test_zero_conditional_denominator_is_none_and_single_class_BA_is_rejected():
    result = stats.error_transitions([0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1])
    assert result["classes"]["0"]["conditional_correction_given_old_error"] is None
    with pytest.raises(stats.StatisticsGateError, match="both_query_classes"):
        stats.balanced_accuracy([0, 0], [0, 1])


def test_windows_average_probabilities_before_argmax_and_never_become_extra_trials():
    result = stats.average_window_probabilities(
        [[.9, .1], [0., 1.], [.25, .75], [.75, .25]], ["trial-a", "trial-a", "trial-b", "trial-b"],
        ["a0", "a1", "b0", "b1"], aggregation_rule="arithmetic_mean_probability_then_argmax_tie0")
    assert len(result) == 2
    assert result[0]["probabilities"] == [.45, .55]
    assert result[0]["prediction"] == 1
    assert result[1]["prediction"] == 0
    assert all(row["window_count"] == 2 for row in result)
    with pytest.raises(stats.StatisticsGateError, match="aggregation_rule"):
        stats.average_window_probabilities([[.5, .5]], ["a"], ["w"], aggregation_rule=None)
    with pytest.raises(stats.StatisticsGateError, match="unique_window"):
        stats.average_window_probabilities([[.5, .5], [.5, .5]], ["a", "a"], ["w", "w"],
            aggregation_rule="arithmetic_mean_probability_then_argmax_tie0")


def test_session_then_seed_then_participant_weights_and_all_negative_zero_cases_retained():
    records, manifest = prediction_fixture({
        "person-a": {"1": (10, 10, 5)},
        "person-b": {"1": (20, 0, 10), "2": (20, 0, 10), "3": (20, 0, 10)},
        "fallback-person": {"1": (10, 5, 5)},
    })
    result = summary(records, manifest)
    assert result["participant_count"] == 3
    assert result["pipeline_seeds"] == list(stats.EXTERNAL_PIPELINE_SEEDS)
    assert result["participant_differences"] == {"fallback-person": 0., "person-a": .5, "person-b": -.5}
    assert result["paired_mean_difference"] == 0.
    assert result["declined_participant_proportion"] == pytest.approx(1 / 3)
    assert result["declined_more_than_5pp_proportion"] == pytest.approx(1 / 3)
    assert result["G0"] - result["H0"] + result["G1"] - result["H1"] == pytest.approx(0.)
    assert result["participants"]["person-b"]["eligible_sessions"] == 3
    assert result["participants"]["person-b"]["methods"]["P"]["participant_mean_ba"] == 0.
    # Pooling 5 sessions, or their unequal trial counts, would change this 0.
    assert result["paired_mean_difference"] != pytest.approx((.5 - 3 * .5) / 5)


def test_decline_exactly_five_percentage_points_is_not_greater_than_five():
    records, manifest = prediction_fixture({"exact-5pp": {"1": (20, 9, 10)}, "over-5pp": {"1": (20, 8, 10)}})
    result = summary(records, manifest)
    assert result["declined_participant_proportion"] == 1.
    assert result["declined_more_than_5pp_proportion"] == .5


@pytest.mark.parametrize("mutation", ["missing_trial", "duplicate_trial", "changed_label", "foreign_uid", "missing_seed"])
def test_complete_paired_UID_and_seed_grid_is_required_no_nan_or_case_deletion(mutation):
    records, manifest = prediction_fixture({"person": {"1": (10, 8, 6)}})
    if mutation == "missing_trial":
        records.pop()
    elif mutation == "duplicate_trial":
        records.append(dict(records[0]))
    elif mutation == "changed_label":
        records[-1]["label"] = 1 - records[-1]["label"]
    elif mutation == "foreign_uid":
        records[-1]["event_uid"] = "foreign-query"
    else:
        records = [row for row in records if row["pipeline_seed"] != stats.EXTERNAL_PIPELINE_SEEDS[-1]]
    with pytest.raises(stats.StatisticsGateError):
        summary(records, manifest)


def test_bootstrap_uses_20000_participant_clusters_and_explicit_reproducible_numerics():
    np.random.seed(111)
    before = copy.deepcopy(np.random.get_state())
    first = stats.participant_percentile_bootstrap({"person-a": -.2, "person-b": .4},
        percentile_method="linear", rng_algorithm="PCG64")
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    second = stats.participant_percentile_bootstrap({"person-b": .4, "person-a": -.2},
        percentile_method="linear", rng_algorithm="PCG64")
    assert first == second
    assert first["replicates"] == 20000 and first["seed"] == 20261018 and first["confidence"] == .95
    assert first["participant_count"] == 2
    assert first["paired_mean_difference"] == pytest.approx(.1)
    assert first["percentile_interval"] == [-.2, .4]
    assert first["resampling_unit"] == "participant_entire_session_seed_cluster"
    with pytest.raises(stats.StatisticsGateError, match="explicit_supported"):
        stats.participant_percentile_bootstrap({"a": .1}, percentile_method=None, rng_algorithm="PCG64")
    with pytest.raises(stats.StatisticsGateError, match="no_nan_deletion"):
        stats.participant_percentile_bootstrap({"a": .1, "b": float("nan")}, percentile_method="linear", rng_algorithm="PCG64")


def test_Holm_complete_predeclared_family_monotone_adjustments_and_no_default_alpha():
    result = stats.holm_adjust({"h1": .01, "h2": .04, "h3": .03}, frozen_family=["h1", "h2", "h3"])
    assert result["adjusted_p_values"] == pytest.approx({"h1": .03, "h2": .06, "h3": .06})
    assert "reject" not in result and result["alpha"] is None
    assert stats.holm_adjust({"a": .02, "b": .02}, frozen_family=["a", "b"], alpha=.05)["reject"] == {"a": True, "b": True}
    with pytest.raises(stats.StatisticsGateError, match="complete_predeclared"):
        stats.holm_adjust({"h1": .01}, frozen_family=["h1", "h2"])


def test_optional_sign_flip_requires_all_frozen_parameters_and_exact_enumeration_budget():
    parameters = dict(statistic="mean_difference", max_patterns=8, tail_rule="inclusive", null_assumption=stats.SIGN_FLIP_NULL)
    differences = {"a": .1, "b": .2, "c": .3}
    assert stats.exact_sign_flip(differences, alternative="two_sided", **parameters)["p_value"] == .25
    assert stats.exact_sign_flip(differences, alternative="greater", **parameters)["p_value"] == .125
    assert stats.exact_sign_flip({"a": 0., "b": 0., "c": 0.}, alternative="two_sided", **parameters)["p_value"] == 1.
    with pytest.raises(stats.StatisticsGateError, match="no_MC_substitution"):
        stats.exact_sign_flip(differences, alternative="two_sided", **{**parameters, "max_patterns": 7})
    with pytest.raises(stats.StatisticsGateError, match="explicit_frozen"):
        stats.exact_sign_flip(differences, alternative=None, **parameters)
    with pytest.raises(stats.StatisticsGateError, match="explicit_frozen"):
        stats.exact_sign_flip(differences, alternative="two_sided", **{**parameters, "null_assumption": "randomized_trial"})


def test_single_HS_primary_hypothesis_is_never_switched_to_Wang_or_B0():
    config = dict(candidate="P", comparator="B2", acquired_budget=20, frozen_b_star="B2")
    assert stats.comparison_role(dataset_id="HSStroke2026_v6", **config) == "single_primary_H1"
    assert stats.comparison_role(dataset_id="Wang2026", **config) == "independent_replication_not_primary"
    assert stats.comparison_role(dataset_id="HSStroke2026_v6", **{**config, "comparator": "B0"}).startswith("secondary")
    with pytest.raises(stats.StatisticsGateError, match="B_star_id_not_frozen"):
        stats.comparison_role(dataset_id="HSStroke2026_v6", **{**config, "frozen_b_star": None})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_real_W9_default_gate_never_reads_query_payloads_or_claims_internal_is_clinical(tmp_path, monkeypatch):
    write(tmp_path / "status/tasks.json", {"W7": {"status": "NOT_FROZEN"}, "W8": {"status": "LOCKED"}})
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("real query payload must not be opened"))
    result = stats.w9_preflight(tmp_path)
    assert result["status"] == "BLOCKED_W9_NOT_EXECUTED"
    assert "W7_FINAL_PROTOCOL_FREEZE_REQUIRED" in result["gaps"]
    assert result["query_labels_read"] is False and result["analysis_performed"] is False
    assert result["W9_completed"] is False and result["development_statistics_are_clinical_W9"] is False


def test_frozen_metadata_still_blocks_without_external_transaction_permission_and_UID_verifier(tmp_path, monkeypatch):
    code = tmp_path / "p3work/statistics.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(Path(stats.__file__).read_bytes())
    freeze = tmp_path / "contracts/freeze.json"
    write(freeze, {"statistics": {"B_star": "B2", "pipeline_seeds": list(stats.EXTERNAL_PIPELINE_SEEDS),
        "statistics_code": {"path": "p3work/statistics.py", "sha256": file_digest(code)},
        "bootstrap": {"rng_algorithm": "PCG64", "percentile_method": "linear", "replicates": 20000,
                      "seed": 20261018, "confidence": .95, "unit": "participant"},
        "formal_secondary_p_values": False, "sign_flip": False}})
    write(tmp_path / "status/tasks.json", {"W7": {"status": "FROZEN", "protocol_freeze_sha256": file_digest(freeze),
        "freeze_manifest": {"path": "contracts/freeze.json", "sha256": file_digest(freeze)}}, "W8": {"status": "LOCKED"}})
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("must not read query"))
    result = stats.w9_preflight(tmp_path)
    assert result["W7_freeze_metadata_read"] is True
    assert result["external_permissions_verified"] is False and result["paired_UIDs_verified"] is False
    assert "EXTERNAL_PREDICTION_TRANSACTION_SHA_AND_PAIRED_UID_VERIFIER_NOT_IMPLEMENTED" in result["gaps"]
    assert "ARCHIVED_EXTERNAL_PREDICTION_AND_QUERY_LABEL_PERMISSION_REQUIRED" in result["gaps"]


def test_read_only_analysis_CLI_is_blocked_and_does_not_mutate_stage(tmp_path, capsys):
    from tools.analyze_statistics import main
    path = tmp_path / "status/tasks.json"
    write(path, {"W7": {"status": "NOT_FROZEN"}, "W9": {"status": "NOT_STARTED"}})
    before = path.read_bytes()
    assert main(["--root", str(tmp_path)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["analysis_performed"] is False
    assert path.read_bytes() == before
