"""Protocol-v2 statistical mathematics and a closed W9 archival entry gate.

Pure functions consume caller-provided values; they never load EEG, fit models,
change stages, download, or publish results. Their outputs are mathematics, not
evidence of a clinical/W9 run. Real W9 remains blocked until W7, evaluator
permissions, paired original-event manifests and an external prediction
transaction verifier exist. The current prediction store supports Stieger only.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .contracts import file_digest


EXTERNAL_PIPELINE_SEEDS = tuple(range(20261018, 20261023))
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_CONFIDENCE = .95
PERCENTILE_METHODS = frozenset({"linear", "lower", "higher", "midpoint", "nearest"})
SIGN_FLIP_NULL = "paired_difference_sign_symmetry_or_method_exchangeability"


class StatisticsGateError(ValueError):
    """A missing specification, denominator or immutable archival gate."""


def _binary(values, name):
    result = np.asarray(values)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or not np.isin(result, [0, 1]).all()):
        raise StatisticsGateError(name + "_must_be_nonempty_integer_binary_original_trials")
    return result


def _vectors(labels, *predictions):
    y = _binary(labels, "labels")
    output = [y] + [_binary(value, "predictions") for value in predictions]
    if any(value.shape != y.shape for value in output):
        raise StatisticsGateError("paired_original_trial_lengths_disagree")
    if any(not np.any(y == c) for c in (0, 1)):
        raise StatisticsGateError("both_query_classes_required_no_undefined_recall")
    return output


def _ba_fraction(y, prediction):
    return sum((Fraction(int(np.sum((y == c) & (prediction == c))), 2 * int(np.sum(y == c)))
                for c in (0, 1)), Fraction())


def balanced_accuracy(labels, predictions) -> float:
    """Equal left/right recall, with one prediction per original query trial."""
    y, predicted = _vectors(labels, predictions)
    return float(_ba_fraction(y, predicted))


def error_transitions(labels, frozen_predictions, candidate_predictions) -> dict:
    """Gc/Hc use ALL common class-c trials, never old-correct/error subsets."""
    y, old, new = _vectors(labels, frozen_predictions, candidate_predictions)
    classes, delta = {}, Fraction()
    for c in (0, 1):
        mask = y == c
        n = int(mask.sum())
        corrected = int(np.sum(mask & (old != y) & (new == y)))
        harmed = int(np.sum(mask & (old == y) & (new != y)))
        old_errors = int(np.sum(mask & (old != y)))
        old_correct = n - old_errors
        g, h = Fraction(corrected, n), Fraction(harmed, n)
        delta += (g - h) / 2
        classes[str(c)] = {"N_c": n, "corrected": corrected, "new_errors": harmed,
            "G_c": float(g), "H_c": float(h),
            "conditional_correction_given_old_error": corrected / old_errors if old_errors else None,
            "conditional_new_error_given_old_correct": harmed / old_correct if old_correct else None}
    if delta != _ba_fraction(y, new) - _ba_fraction(y, old):
        raise AssertionError("class_denominator_delta_BA_identity_failed")
    return {"classes": classes, "delta_balanced_accuracy": float(delta),
            "baseline_balanced_accuracy": float(_ba_fraction(y, old)),
            "candidate_balanced_accuracy": float(_ba_fraction(y, new)),
            "ordinary_accuracy_difference": float(np.mean(new == y) - np.mean(old == y)),
            "common_original_trials": len(y)}


def average_window_probabilities(probabilities, event_uids, window_uids, *, aggregation_rule) -> list[dict]:
    """Explicit arithmetic mean probabilities -> ONE argmax prediction per trial.

    Window identities must be unique. Their input order is preserved, and exact
    probability ties choose class 0. No window is an independent participant.
    """
    if aggregation_rule != "arithmetic_mean_probability_then_argmax_tie0":
        raise StatisticsGateError("window_probability_aggregation_rule_must_be_explicitly_frozen")
    x = np.asarray(probabilities, dtype=np.float64)
    if (x.shape != (len(event_uids), 2) or not len(x) or len(window_uids) != len(x)
            or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)
            or not np.allclose(x.sum(1), 1, atol=1e-6, rtol=0)
            or any(not isinstance(uid, str) or not uid for uid in (*event_uids, *window_uids))
            or len(set(window_uids)) != len(window_uids)):
        raise StatisticsGateError("complete_unique_window_probability_records_required")
    groups = {}
    for index, uid in enumerate(event_uids):
        groups.setdefault(uid, []).append(index)
    result = []
    for uid, indices in groups.items():
        mean = x[indices].mean(0)
        result.append({"event_uid": uid, "probabilities": mean.tolist(), "prediction": int(mean.argmax()),
                       "window_uids": [window_uids[index] for index in indices], "window_count": len(indices)})
    return result


def _query_structure(query_manifest):
    if not isinstance(query_manifest, Mapping) or not query_manifest:
        raise StatisticsGateError("explicit_qualified_participant_session_query_manifest_required")
    expected, seen = {}, set()
    for person, sessions in query_manifest.items():
        if not isinstance(person, str) or not person or not isinstance(sessions, Mapping) or not sessions:
            raise StatisticsGateError("qualified_participant_and_session_identity_required")
        expected[person] = {}
        for session, uids in sessions.items():
            if (not isinstance(session, str) or not session or not isinstance(uids, (list, tuple)) or len(uids) < 20
                    or any(not isinstance(uid, str) or not uid for uid in uids)
                    or len(set(uids)) != len(uids) or seen.intersection(uids)):
                raise StatisticsGateError("unique_complete_B20_future_query_original_uids_required")
            expected[person][session] = tuple(uids)
            seen.update(uids)
    return expected


def hierarchical_paired_summary(records, query_manifest, *, dataset_id, candidate, comparator,
                                pipeline_seeds) -> dict:
    """Session BA -> seed mean -> participant mean -> paired participant effect.

    Every frozen method×person×session×seed×query-UID cell is mandatory. Missing
    or failed predictions cannot cause silent deletion; a predeclared fallback
    must supply its actual saved predictions. All seeds/methods share labels and
    the exact same query manifest. No seed/session/trial becomes a new person.
    """
    if (not isinstance(dataset_id, str) or not dataset_id or not isinstance(candidate, str)
            or not isinstance(comparator, str) or not candidate or not comparator or candidate == comparator):
        raise StatisticsGateError("explicit_dataset_and_distinct_paired_methods_required")
    seeds = tuple(pipeline_seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise StatisticsGateError("complete_explicit_pipeline_seed_list_required")
    expected = _query_structure(query_manifest)
    methods, groups, labels = (candidate, comparator), defaultdict(dict), {}
    required = {"dataset_id", "participant_id", "session_id", "pipeline_seed", "event_uid", "method_id", "label", "prediction"}
    for row in records:
        if not isinstance(row, Mapping) or not required <= set(row):
            raise StatisticsGateError("complete_saved_original_trial_prediction_required")
        person, session, seed, uid, method = (row[key] for key in
            ("participant_id", "session_id", "pipeline_seed", "event_uid", "method_id"))
        if (row["dataset_id"] != dataset_id or person not in expected or session not in expected[person]
                or type(seed) is not int or seed not in seeds or method not in methods
                or uid not in expected[person][session] or type(row["label"]) is not int
                or row["label"] not in (0, 1) or type(row["prediction"]) is not int or row["prediction"] not in (0, 1)):
            raise StatisticsGateError("foreign_missing_or_unpaired_prediction_identity")
        group = groups[person, session, seed, method]
        if uid in group:
            raise StatisticsGateError("duplicate_original_trial_prediction_no_window_pseudoreplication")
        if uid in labels and labels[uid] != row["label"]:
            raise StatisticsGateError("same_original_query_labels_disagree_across_method_or_seed")
        labels[uid] = row["label"]
        group[uid] = row["prediction"]
    participants, differences, transition_means = {}, {}, []
    exact_differences = []
    for person in sorted(expected):
        per_method = {method: {"session_ba_by_seed": {}, "seed_mean_ba": {}} for method in methods}
        exact_seed_means = {method: [] for method in methods}
        seed_transitions = []
        for seed in seeds:
            session_scores = {method: [] for method in methods}
            session_transitions = []
            for session, uids in expected[person].items():
                predictions = {}
                for method in methods:
                    group = groups.get((person, session, seed, method), {})
                    if set(group) != set(uids):
                        raise StatisticsGateError("incomplete_frozen_method_session_seed_query_grid_no_deletion")
                    predictions[method] = np.array([group[uid] for uid in uids], dtype=np.int64)
                y = np.array([labels[uid] for uid in uids], dtype=np.int64)
                if min(int(np.sum(y == c)) for c in (0, 1)) < 5:
                    raise StatisticsGateError("predeclared_query_needs_at_least_five_original_trials_per_class")
                for method in methods:
                    score = _ba_fraction(y, predictions[method])
                    session_scores[method].append(score)
                    per_method[method]["session_ba_by_seed"].setdefault(str(seed), {})[session] = float(score)
                # Keep exact rational denominators through the same aggregation
                # as BA, including unequal class counts and session lengths.
                rates = []
                old, new = predictions[comparator], predictions[candidate]
                for c in (0, 1):
                    n = int(np.sum(y == c))
                    rates.extend((Fraction(int(np.sum((y == c) & (old != y) & (new == y))), n),
                                  Fraction(int(np.sum((y == c) & (old == y) & (new != y))), n)))
                session_transitions.append(rates)
            for method in methods:
                score = sum(session_scores[method], Fraction()) / len(session_scores[method])
                exact_seed_means[method].append(score)
                per_method[method]["seed_mean_ba"][str(seed)] = float(score)
            seed_transitions.append([sum((rates[i] for rates in session_transitions), Fraction()) / len(session_transitions)
                                     for i in range(4)])
        exact = {method: sum(exact_seed_means[method], Fraction()) / len(seeds) for method in methods}
        for method in methods:
            per_method[method]["participant_mean_ba"] = float(exact[method])
        difference = exact[candidate] - exact[comparator]
        rates = [sum((row[i] for row in seed_transitions), Fraction()) / len(seeds) for i in range(4)]
        if (rates[0] - rates[1] + rates[2] - rates[3]) / 2 != difference:
            raise AssertionError("hierarchical_G_H_delta_BA_identity_failed")
        participants[person] = {"methods": per_method, "difference": float(difference), "eligible_sessions": len(expected[person]),
                                "G0": float(rates[0]), "H0": float(rates[1]), "G1": float(rates[2]), "H1": float(rates[3])}
        differences[person] = float(difference)
        exact_differences.append(difference)
        transition_means.append(rates)
    n = len(participants)
    cohort_rates = [sum((row[i] for row in transition_means), Fraction()) / n for i in range(4)]
    effect = sum(exact_differences, Fraction()) / n
    if (cohort_rates[0] - cohort_rates[1] + cohort_rates[2] - cohort_rates[3]) / 2 != effect:
        raise AssertionError("cohort_G_H_delta_BA_identity_failed")
    return {"scope": "pure_mathematics_not_W9_evidence", "dataset_id": dataset_id, "candidate": candidate,
        "comparator": comparator, "participant_count": n, "pipeline_seeds": list(seeds),
        "aggregation_order": "class_recalls_then_session_then_pipeline_seed_then_participant",
        "participants": participants, "participant_differences": differences, "paired_mean_difference": float(effect),
        "effect_unit": "balanced_accuracy_fraction", "declined_participant_proportion": sum(d < 0 for d in exact_differences) / n,
        "declined_more_than_5pp_proportion": sum(d < -Fraction(1, 20) for d in exact_differences) / n,
        "transition_baseline_method": comparator, "G0": float(cohort_rates[0]), "H0": float(cohort_rates[1]),
        "G1": float(cohort_rates[2]), "H1": float(cohort_rates[3])}


def _participant_values(differences):
    if not isinstance(differences, Mapping) or not differences:
        raise StatisticsGateError("all_paired_participant_differences_required")
    people = sorted(differences)
    if any(not isinstance(person, str) or not person for person in people):
        raise StatisticsGateError("unique_participant_identifiers_required")
    values = np.array([differences[person] for person in people], dtype=np.float64)
    if (any(isinstance(differences[person], (bool, str)) for person in people)
            or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
        raise StatisticsGateError("finite_BA_fraction_differences_required_no_nan_deletion")
    return people, values


def participant_percentile_bootstrap(differences, *, percentile_method, rng_algorithm) -> dict:
    """Protocol 20,000 participant clusters, 95% percentile CI, seed 20261018.

    RNG algorithm and percentile interpolation were not specified in the prose;
    callers must name them explicitly and freeze them in W7. Collapsed correctly
    aggregated participant differences preserve their complete session/seed
    clusters and are mathematically equivalent to resampling those clusters.
    """
    if percentile_method not in PERCENTILE_METHODS or rng_algorithm != "PCG64":
        raise StatisticsGateError("explicit_supported_bootstrap_rng_and_percentile_method_required")
    people, values = _participant_values(differences)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 512):
        count = min(512, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(1)
    interval = np.quantile(means, [.025, .975], method=percentile_method)
    return {"scope": "pure_mathematics_not_W9_evidence", "participant_count": len(people),
        "participant_differences": {person: float(value) for person, value in zip(people, values)},
        "paired_mean_difference": float(values.mean()), "confidence": BOOTSTRAP_CONFIDENCE,
        "percentile_interval": interval.tolist(), "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED, "resampling_unit": "participant_entire_session_seed_cluster",
        "rng_algorithm": rng_algorithm, "percentile_method": percentile_method,
        "effect_unit": "balanced_accuracy_fraction"}


def holm_adjust(p_values, *, frozen_family, alpha=None) -> dict:
    """Holm monotone adjusted p-values for the COMPLETE predeclared family."""
    family = tuple(frozen_family)
    if (not family or len(family) != len(set(family)) or any(not isinstance(key, str) or not key for key in family)
            or not isinstance(p_values, Mapping) or set(p_values) != set(family)):
        raise StatisticsGateError("complete_predeclared_hypothesis_family_required")
    if any(isinstance(value, (bool, str)) or not math.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise StatisticsGateError("finite_valid_p_values_required")
    if alpha is not None and (isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise StatisticsGateError("explicit_valid_alpha_required_for_rejections")
    ordered = sorted(family, key=lambda key: p_values[key])
    adjusted, running = {}, 0.
    for index, key in enumerate(ordered):
        running = max(running, (len(family) - index) * p_values[key])
        adjusted[key] = min(1., running)
    result = {"family": list(family), "adjusted_p_values": {key: adjusted[key] for key in family},
              "method": "Holm", "alpha": alpha}
    if alpha is not None:
        result["reject"] = {key: adjusted[key] <= alpha for key in family}
    return result


def exact_sign_flip(differences, *, alternative, statistic, max_patterns, tail_rule, null_assumption) -> dict:
    """Optional exact paired sign-flip test; no unspecified MC or sidedness."""
    people, values = _participant_values(differences)
    if (alternative not in {"two_sided", "greater", "less"} or statistic != "mean_difference"
            or tail_rule != "inclusive" or null_assumption != SIGN_FLIP_NULL
            or type(max_patterns) is not int or max_patterns < 1):
        raise StatisticsGateError("sign_flip_requires_explicit_frozen_sidedness_statistic_null_tail_and_budget")
    patterns = 2 ** len(values)
    if patterns > max_patterns:
        raise StatisticsGateError("exact_sign_flip_exceeds_explicit_enumeration_budget_no_MC_substitution")
    exact = [Fraction.from_float(float(value)) for value in values]
    observed_sum = sum(exact, Fraction())
    extreme = 0
    for mask in range(patterns):
        flipped = sum((value if mask & (1 << index) else -value for index, value in enumerate(exact)), Fraction())
        extreme += (abs(flipped) >= abs(observed_sum) if alternative == "two_sided"
                    else flipped >= observed_sum if alternative == "greater" else flipped <= observed_sum)
    return {"p_value": extreme / patterns, "patterns": patterns, "extreme_patterns": extreme,
        "participant_count": len(people), "alternative": alternative, "statistic": statistic,
        "tail_rule": tail_rule, "null_assumption": null_assumption,
        "interpretation": "paired_sign_symmetry_or_method_exchangeability_not_randomized_clinical_trial"}


def comparison_role(*, dataset_id, candidate, comparator, acquired_budget, frozen_b_star) -> str:
    if frozen_b_star not in {f"B{i}" for i in range(1, 12)}:
        raise StatisticsGateError("actual_pre_external_B_star_id_not_frozen")
    if dataset_id == "HSStroke2026_v6" and candidate == "P" and comparator == frozen_b_star and acquired_budget == 20:
        return "single_primary_H1"
    if dataset_id == "Wang2026" and candidate == "P" and comparator == frozen_b_star and acquired_budget == 20:
        return "independent_replication_not_primary"
    return "secondary_requires_frozen_family_for_formal_p_values"


def _local_receipt(root, descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise StatisticsGateError("explicit_archival_receipt_path_sha256_required")
    path, sha = descriptor["path"], descriptor["sha256"]
    if (not isinstance(path, str) or not path or path.startswith("/") or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(part.lower().startswith(".env") or part.lower() in {".aws", ".ssh", "credentials", "secrets"}
                   for part in path.split("/"))
            or not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None):
        raise StatisticsGateError("invalid_archival_receipt_reference")
    target = root
    for part in path.split("/"):
        target = target / part
        if target.is_symlink():
            raise StatisticsGateError("archival_receipt_symlink_forbidden")
    if not target.resolve().is_relative_to(root) or file_digest(target) != sha:
        raise StatisticsGateError("archival_receipt_full_sha256_mismatch")
    return target


def w9_preflight(root) -> dict:
    """Read only stage/freeze metadata; NEVER read query labels or predictions.

    A future W8 external prediction adapter must validate every payload SHA,
    paired UID, frozen cohort/session/seed grid and label permission before an
    actual analysis can call the pure mathematics. The Stieger development store
    cannot be silently reused as that adapter. No function here marks W9 done.
    """
    root = Path(root).resolve(strict=True)
    tasks_path = root / "status/tasks.json"
    if (root / "status").is_symlink() or tasks_path.is_symlink():
        raise StatisticsGateError("stage_status_symlink_forbidden")
    tasks = json.loads(tasks_path.read_text())
    w7 = tasks.get("W7", {})
    gaps = []
    freeze = None
    if w7.get("status") != "FROZEN":
        gaps.append("W7_FINAL_PROTOCOL_FREEZE_REQUIRED")
    else:
        descriptor = w7.get("freeze_manifest")
        if (not isinstance(descriptor, dict)
                or Path(descriptor.get("path", "")).name not in {"freeze.json", "protocol_freeze.json"}):
            raise StatisticsGateError("explicit_W7_freeze_metadata_descriptor_required")
        frozen_path = _local_receipt(root, descriptor)
        if w7.get("protocol_freeze_sha256") != file_digest(frozen_path):
            raise StatisticsGateError("W7_protocol_freeze_file_identity_mismatch")
        freeze = json.loads(frozen_path.read_text())
        spec = freeze.get("statistics", {})
        if spec.get("B_star") not in {f"B{i}" for i in range(1, 12)}:
            gaps.append("ACTUAL_B_STAR_ID_AND_CONFIGURATION_NOT_FROZEN")
        if not isinstance(spec.get("B_star_configuration"), dict):
            gaps.append("ACTUAL_B_STAR_CONFIGURATION_SHA_NOT_FROZEN")
        else:
            _local_receipt(root, spec["B_star_configuration"])
        bootstrap = spec.get("bootstrap", {})
        if bootstrap.get("rng_algorithm") != "PCG64":
            gaps.append("BOOTSTRAP_RNG_ALGORITHM_NOT_FROZEN")
        if bootstrap.get("percentile_method") not in PERCENTILE_METHODS:
            gaps.append("BOOTSTRAP_PERCENTILE_INTERPOLATION_NOT_FROZEN")
        if (bootstrap.get("replicates") != BOOTSTRAP_REPLICATES or bootstrap.get("seed") != BOOTSTRAP_SEED
                or bootstrap.get("confidence") != BOOTSTRAP_CONFIDENCE or bootstrap.get("unit") != "participant"):
            gaps.append("PROTOCOL_BOOTSTRAP_CONSTANTS_NOT_FROZEN")
        if spec.get("pipeline_seeds") != list(EXTERNAL_PIPELINE_SEEDS):
            gaps.append("FIVE_EXTERNAL_PIPELINE_SEEDS_NOT_FROZEN")
        code = spec.get("statistics_code")
        if not isinstance(code, dict) or code.get("path") != "p3work/statistics.py":
            gaps.append("STATISTICS_CODE_SHA_NOT_FROZEN")
        else:
            _local_receipt(root, code)
        if spec.get("formal_secondary_p_values") is True and (
                not spec.get("secondary_family") or spec.get("multiplicity") != "Holm"
                or type(spec.get("alpha")) not in {float, int} or not 0 < spec["alpha"] < 1):
            gaps.append("FORMAL_SECONDARY_FAMILY_TEST_SPECIFICATION_ALPHA_NOT_FROZEN")
        if type(spec.get("formal_secondary_p_values")) is not bool or "sign_flip" not in spec:
            gaps.append("OPTIONAL_P_VALUE_ANALYSES_NOT_EXPLICITLY_FROZEN")
    if not tasks.get("W8", {}).get("archived_statistics_permission_receipt"):
        gaps.append("ARCHIVED_EXTERNAL_PREDICTION_AND_QUERY_LABEL_PERMISSION_REQUIRED")
    gaps.append("EXTERNAL_PREDICTION_TRANSACTION_SHA_AND_PAIRED_UID_VERIFIER_NOT_IMPLEMENTED")
    return {"status": "BLOCKED_W9_NOT_EXECUTED", "gaps": gaps, "W7_freeze_metadata_read": freeze is not None,
        "prediction_payloads_read": False, "query_labels_read": False, "analysis_performed": False,
        "prediction_payload_SHA_verified": False, "paired_UIDs_verified": False, "external_permissions_verified": False,
        "development_statistics_are_clinical_W9": False, "training_started": False,
        "downloads_started": False, "W9_completed": False}
