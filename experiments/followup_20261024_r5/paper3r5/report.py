"""Independent Round 4 reporting from immutable worker outputs only.

The participant is the sampling unit. Source caches and Round 2 modules are
never imported or read. ``build_report`` isolates report failures from workers.
"""
from collections import defaultdict
from functools import lru_cache
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .io import atomic, digest, put, read

BOOTSTRAP_REPLICATES = 20000
STATISTICS_SEED = 20261018
METRICS = ("ba", "bce", "f0_ba", "negative_flip_rate", "corrected_error_rate",
           "brier_balanced", "nll_balanced", "ece_balanced_10bins",
           "f0_brier_balanced", "f0_nll_balanced", "f0_ece_balanced_10bins",
           "calibration_seconds")
RATE_METRICS = {"ba", "f0_ba", "negative_flip_rate", "corrected_error_rate"}
TABLE_NAMES = ("primary_comparisons", "evaluation_summaries", "budget_curves",
               "budget_comparisons", "sensitivities", "sensitivity_comparisons", "stress",
               "stress_comparisons", "transfer", "transfer_comparisons", "timings", "diagnostics",
               "cv_selection_frequency")
INTERPRETATION = ("Exploratory internal replication on reused internal participants; "
                  "not fresh confirmatory, newly blinded, external, or clinical validation.")


def _key(row):
    return str(row["participant_id"]), str(row["session_id"])


def _query_signature(row):
    ids, labels = row.get("event_uids"), row.get("labels")
    if (not isinstance(ids, list) or not ids or not isinstance(labels, list)
            or len(ids) != len(labels) or len(set(ids)) != len(ids)):
        raise ValueError("Empty, duplicated, or mismatched query identities")
    if set(labels) != {0, 1}:
        raise ValueError("Both binary query classes are required")
    signature = digest({"uids": ids, "labels": labels})
    if row.get("query_signature", signature) != signature:
        raise ValueError("Stored query signature disagrees with trial identities/labels")
    return signature


def _probability_metrics(values, labels, predictions=None):
    y, p = np.asarray(labels, dtype=int), np.asarray(values, dtype=float)
    if (p.shape != (len(y), 2) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(1), 1., atol=2e-6, rtol=0)):
        raise ValueError("Invalid two-class probability output")
    predicted = p.argmax(1) if predictions is None else np.asarray(predictions)
    if predicted.shape != y.shape or not set(predicted.tolist()) <= {0, 1}:
        raise ValueError("Invalid prediction rows")
    if np.any(p.max(1) - p[np.arange(len(y)), predicted.astype(int)] > 1e-7):
        raise ValueError("Probabilities disagree with stored predictions")
    weights = 1. / (2 * np.bincount(y, minlength=2)[y])
    confidence, correct = p.max(1), predicted == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(abs(float(np.sum(weights[bins == i] *
                  (correct[bins == i] - confidence[bins == i])))) for i in range(10))
    return {"brier_balanced": float(np.sum(weights * (p[:, 1] - y) ** 2)),
            "nll_balanced": float(np.sum(weights * -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))),
            "ece_balanced_10bins": float(ece)}


def row_metrics(row):
    """Validate persisted predictions and derive explicitly balanced metrics."""
    _query_signature(row)
    y = np.asarray(row["labels"], dtype=int)
    metrics = {name: row.get(name) for name in METRICS}
    predictions = row.get("predictions")
    if predictions is None and row.get("probabilities") is not None:
        predictions = np.asarray(row["probabilities"]).argmax(1).tolist()
    frozen = row.get("f0_predictions")
    if frozen is None and row.get("f0_probabilities") is not None:
        frozen = np.asarray(row["f0_probabilities"]).argmax(1).tolist()
    balanced = lambda values: .5 * sum(float(np.mean(values[y == c])) for c in (0, 1))
    computed = {}
    if predictions is not None:
        pred = np.asarray(predictions)
        if pred.shape != y.shape or not set(pred.tolist()) <= {0, 1}:
            raise ValueError("Invalid prediction rows")
        computed["ba"] = balanced(pred == y)
        if frozen is not None:
            base = np.asarray(frozen)
            if base.shape != y.shape or not set(base.tolist()) <= {0, 1}:
                raise ValueError("Invalid frozen-source prediction rows")
            computed.update(f0_ba=balanced(base == y),
                            negative_flip_rate=balanced((base == y) & (pred != y)),
                            corrected_error_rate=balanced((base != y) & (pred == y)))
    for name, value in computed.items():
        if row.get(name) is not None and not np.isclose(row[name], value, atol=2e-7, rtol=0):
            raise ValueError("Stored " + name + " disagrees with predictions")
        metrics[name] = value
    if row.get("probabilities") is not None:
        metrics.update(_probability_metrics(row["probabilities"], y, predictions))
    if row.get("f0_probabilities") is not None:
        metrics.update({"f0_" + k: v for k, v in
                        _probability_metrics(row["f0_probabilities"], y, frozen).items()})
    if all(metrics.get(k) is not None for k in
           ("ba", "f0_ba", "negative_flip_rate", "corrected_error_rate")):
        if not np.isclose(metrics["ba"] - metrics["f0_ba"],
                          metrics["corrected_error_rate"] - metrics["negative_flip_rate"], atol=2e-7):
            raise ValueError("Inconsistent correction and negative-flip accounting")
    for name, value in metrics.items():
        if value is not None and (not isinstance(value, (int, float, np.number))
                                  or not np.isfinite(value)):
            raise ValueError("Non-finite or non-numeric metric: " + name)
    return metrics


def evaluation_rows(folder):
    """Read only a worker evaluation and verify its declared session manifest."""
    folder = Path(folder)
    summary = read(folder / "summary.json")
    expected = summary.get("expected_sessions")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Evaluation summary lacks expected_sessions: " + str(folder))
    expected_map = {_key(row): row["query_signature"] for row in expected}
    if len(expected_map) != len(expected):
        raise ValueError("Duplicate expected participant/session")
    rows = []
    for path in sorted(folder.glob("*.json")):
        if path.name == "summary.json":
            continue
        row = read(path)
        if isinstance(row, dict) and "participant_id" in row and "session_id" in row:
            rows.append(row)
    actual = {_key(row): row for row in rows}
    if len(actual) != len(rows):
        raise ValueError("Duplicate participant/session output")
    if set(actual) != set(expected_map):
        raise ValueError("Missing or extra participant/session outputs: " + str(folder))
    for key, row in actual.items():
        if _query_signature(row) != expected_map[key]:
            raise ValueError("Query identity/label mismatch against summary")
        if row.get("budget") is not None and summary.get("budget") is not None:
            if row["budget"] != summary["budget"]:
                raise ValueError("Evaluation budget mismatch")
        row["_report_metrics"] = row_metrics(row)
    return actual


@lru_cache(maxsize=4096)
def _bootstrap(values, scale=1.):
    d = np.asarray(values, dtype=float)
    if len(d) < 2:
        return None
    rng = np.random.default_rng(STATISTICS_SEED)
    boot = d[rng.integers(0, len(d), (BOOTSTRAP_REPLICATES, len(d)))].mean(1)
    return (scale * np.quantile(boot, [.025, .975])).tolist()


def paired_description(differences, *, scale=100):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("Paired differences must be a finite vector")
    return {"n_participants": len(d),
            "delta": float(scale * d.mean()) if len(d) else None,
            "ci95": _bootstrap(tuple(d), scale),
            "inference": "descriptive_individual_unadjusted_interval" if len(d) >= 2
                         else "insufficient_paired_participants"}


def paired_effect(differences):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite paired participants are required")
    observed = abs(d.mean())
    if len(d) <= 18:
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
        p = float(np.mean(np.abs((signs * d).mean(1)) >= observed - 1e-14))
        method = "exact_paired_sign_flip"
    else:
        rng = np.random.default_rng(STATISTICS_SEED)
        signs = rng.choice((-1, 1), (BOOTSTRAP_REPLICATES, len(d)))
        p = float((1 + np.sum(np.abs((signs * d).mean(1)) >= observed - 1e-14)) /
                  (BOOTSTRAP_REPLICATES + 1))
        method = "monte_carlo_paired_sign_flip"
    return {"p": p, "p_method": method,
            "test_assumption": "participant paired differences sign-symmetric under null"}


def holm(values):
    """Adjust exactly the supplied, predeclared primary family."""
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid primary p-value")
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted, previous = [0.] * len(values), 0.
    for position, index in enumerate(order):
        previous = max(previous, min(1., values[index] * (len(values) - position)))
        adjusted[index] = previous
    return adjusted


def _metrics(row):
    return row.get("_report_metrics") or row_metrics(row)


def _metric_people(by_seed, metric):
    """Complete session cases -> person/seed means -> equal seed means/person."""
    people, session_count, possible = defaultdict(list), 0, set()
    excluded_sessions = 0
    for seed, rows in sorted(by_seed.items()):
        sessions = defaultdict(list)
        for key, row in rows.items():
            possible.add(key[0])
            value = _metrics(row).get(metric)
            if value is None:
                excluded_sessions += 1
            else:
                sessions[key[0]].append(value)
                session_count += 1
        for person, values in sessions.items():
            people[person].append((seed, float(np.mean(values))))
    return {person: float(np.mean([v for _, v in seeds])) for person, seeds in people.items()}, {
        "n_session_seed_rows": session_count, "excluded_session_seed_rows": excluded_sessions,
        "n_excluded_participants": len(possible - set(people)),
        "participant_seed_counts": {p: len(v) for p, v in sorted(people.items())}}


def participant_summary(by_seed):
    result = {"completed_seeds": sorted(by_seed), "completed_seed_count": len(by_seed), "metrics": {}}
    for metric in METRICS:
        people, coverage = _metric_people(by_seed, metric)
        values = tuple(people[p] for p in sorted(people))
        value = float(np.mean(values)) if values else None
        result["metrics"][metric] = {"mean": value, "n_participants": len(values), **coverage}
        if metric == "ba":
            result["metrics"][metric]["ci95"] = _bootstrap(values)
        result[metric], result[metric + "_n_participants"] = value, len(values)
    result["n_participants"] = result["ba_n_participants"]
    return result


def compare_rows(candidate, baseline):
    """Pair by identity, average sessions within seed, then seeds within person."""
    seeds = sorted(set(candidate) & set(baseline))
    metric_people, baselines, candidates = {}, {}, {}
    coverage, per_seed = {}, []
    unmatched, eligible_pairs = 0, 0
    possible_people = set()
    for metric in METRICS:
        deltas, base_values, candidate_values = defaultdict(list), defaultdict(list), defaultdict(list)
        valid_sessions, missing_metric = 0, 0
        for seed in seeds:
            a, b = candidate[seed], baseline[seed]
            common = set(a) & set(b)
            if metric == "ba":
                unmatched += len(set(a) ^ set(b))
                possible_people.update(k[0] for k in set(a) | set(b))
            within = defaultdict(list)
            for key in sorted(common):
                if _query_signature(a[key]) != _query_signature(b[key]):
                    raise ValueError("Query trial or label mismatch across paired methods")
                ma, mb = _metrics(a[key]).get(metric), _metrics(b[key]).get(metric)
                if ma is None or mb is None:
                    missing_metric += 1
                    continue
                within[key[0]].append((ma, mb))
                valid_sessions += 1
            for person, values in within.items():
                av, bv = np.mean(values, axis=0)
                deltas[person].append(float(av - bv))
                base_values[person].append(float(bv))
                candidate_values[person].append(float(av))
            if metric == "ba" and within:
                per_seed.append({"seed": seed, "n_participants": len(within),
                                 "n_paired_sessions": sum(map(len, within.values())),
                                 "delta_ba_pp": 100 * float(np.mean([
                                     np.mean([av - bv for av, bv in values]) for values in within.values()]))})
        people = {p: float(np.mean(v)) for p, v in deltas.items()}
        metric_people[metric] = people
        baselines[metric] = {p: float(np.mean(v)) for p, v in base_values.items()}
        candidates[metric] = {p: float(np.mean(v)) for p, v in candidate_values.items()}
        coverage[metric] = {"n_paired_session_seed_rows": valid_sessions,
                            "excluded_metric_session_seed_rows": missing_metric,
                            "participant_seed_counts": {p: len(v) for p, v in sorted(deltas.items())}}
        if metric == "ba":
            eligible_pairs = valid_sessions
    people = metric_people["ba"]
    described = paired_description([people[p] for p in sorted(people)])
    secondary = {}
    for metric in METRICS:
        if metric == "ba":
            continue
        values = metric_people[metric]
        secondary[metric] = {**paired_description([values[p] for p in sorted(values)],
                            scale=100 if metric in RATE_METRICS else 1), **coverage[metric],
                            "unit": "percentage_points" if metric in RATE_METRICS else
                                    "seconds" if metric == "calibration_seconds" else "metric_units",
                            "n_excluded_participants": len(possible_people - set(values))}
    base = float(np.mean(list(baselines["ba"].values()))) if people else None
    cand = float(np.mean(list(candidates["ba"].values()))) if people else None
    return {"n_participants": len(people), "delta_ba_pp": described["delta"],
            "ci95_pp": described["ci95"], "baseline_ba": base, "candidate_ba": cand,
            "p": None, "p_holm": None, "p_method": None, "positive_evidence": None,
            "completed_seeds": seeds, "completed_seed_count": len(seeds),
            "n_paired_session_seed_rows": eligible_pairs, "unmatched_session_seed_rows": unmatched,
            "n_excluded_participants": len(possible_people - set(people)),
            "participant_seed_counts": coverage["ba"]["participant_seed_counts"],
            "participant_deltas_ba": people, "per_seed": per_seed, "secondary_metrics": secondary,
            "participant_negative_transfer_fraction": float(np.mean(np.array(list(people.values())) < 0)) if people else None,
            "relative_error_reduction_pct": 100 * (cand - base) / (1 - base) if base is not None and base < 1 else None,
            "inference": "descriptive_no_significance_claim"}


def _diagnostic_values(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(_diagnostic_values(child, prefix + ("." if prefix else "") + str(key)))
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        if np.isfinite(value):
            result[prefix] = float(value)
    elif isinstance(value, list) and value and ("eigen" in prefix.lower() or "singular" in prefix.lower()):
        vector = np.asarray(value, dtype=float)
        if vector.ndim == 1 and np.isfinite(vector).all():
            result.update({prefix + ".min": float(vector.min()), prefix + ".max": float(vector.max()),
                           prefix + ".sum": float(vector.sum())})
    return result


def diagnostic_summary(by_seed):
    names = sorted({k for rows in by_seed.values() for row in rows.values()
                    for k in _diagnostic_values(row.get("diagnostics", {}))})
    result = []
    for name in names:
        synthetic = {}
        for seed, rows in by_seed.items():
            synthetic[seed] = {key: {"_report_metrics": {name: _diagnostic_values(row.get("diagnostics", {})).get(name)}}
                               for key, row in rows.items()}
        people, coverage = _metric_people(synthetic, name)
        values = list(people.values())
        result.append({"diagnostic": name, "mean": float(np.mean(values)) if values else None,
                       "min_participant_mean": float(min(values)) if values else None,
                       "max_participant_mean": float(max(values)) if values else None,
                       "n_participants": len(values), **coverage})
    return result


def _cv_choices(row):
    """Finite scalar choices only; fold labels, scores and predictions stay local.

    Fixed-rule adapters need not have CV metadata.  A categorical choice is a
    descriptive output, never a new statistical sampling unit or primary test.
    """
    diagnostics = row.get("diagnostics", {})
    cv = diagnostics.get("cv", {}) if isinstance(diagnostics, dict) else {}
    if not isinstance(cv, dict):
        return {}
    result = {}

    def visit(value, prefix="cv", selected=False):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = prefix + "." + str(key)
            is_choice = selected or str(key).startswith(("selected_", "chosen_"))
            is_choice |= str(key) in ("selected", "chosen", "status", "cv_status", "selection_rule", "fallback_reason")
            if isinstance(child, dict):
                visit(child, path, is_choice)
            elif is_choice and isinstance(child, (str, bool, int, float, np.number)):
                if isinstance(child, (float, np.floating)) and not np.isfinite(child):
                    continue
                # JSON scalars give deterministic, type-preserving CSV values.
                scalar = child.item() if isinstance(child, np.generic) else child
                result[path] = json.dumps(scalar, sort_keys=True, allow_nan=False)
    visit(cv)
    return result


def cv_selection_summary(by_seed):
    """Participant-weighted frequencies over observed support-only choices."""
    choices = {seed: {key: _cv_choices(row) for key, row in rows.items()}
               for seed, rows in by_seed.items()}
    fields = sorted({field for rows in choices.values() for row in rows.values() for field in row})
    result = []
    for field in fields:
        values = sorted({row[field] for rows in choices.values() for row in rows.values() if field in row})
        for value in values:
            synthetic = {seed: {key: {"_report_metrics": {"selected": float(row[field] == value)
                                                     if field in row else None}}
                                for key, row in rows.items()} for seed, rows in choices.items()}
            people, coverage = _metric_people(synthetic, "selected")
            result.append({"selection_field": field, "selection_value": json.loads(value),
                "participant_mean_fraction": float(np.mean(list(people.values()))) if people else None,
                "n_participants": len(people), **coverage,
                "observed_choice_session_seed_rows": sum(field in row and row[field] == value
                    for rows in choices.values() for row in rows.values()),
                "interpretation": "descriptive support-only choice frequency; participant-weighted, no significance test"})
    return result


def _coordinate(path):
    partition, budget, scenario, _ = path.parts[-4:]
    if not budget.startswith("B"):
        raise ValueError("Invalid evaluation budget directory")
    return partition, int(budget[1:]), scenario


def _primary_coordinate(plan):
    return (str(plan.get("primary_partition", "holdout")),
            int(plan.get("primary_budget", max(plan.get("budgets", [20])))),
            str(plan.get("primary_scenario", "clean")))


def _comparisons(plan):
    pairs = []
    for pair in plan.get("comparisons", []):
        candidate, baseline = ((pair["candidate"], pair["baseline"]) if isinstance(pair, dict) else pair)
        if candidate == baseline:
            raise ValueError("A primary comparison cannot compare a group with itself")
        pairs.append((str(candidate), str(baseline)))
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate primary comparisons")
    return pairs


def _load(plan):
    root, groups, metadata = Path(plan["output"]), defaultdict(dict), {}
    query_references = {}
    completed, pending, invalid, issues = [], [], [], []
    tasks = plan["tasks"]
    if len({t["key"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task keys")
    planned_seeds = {int(seed) for seed in plan["seeds"]}
    primary = _primary_coordinate(plan)
    primary_groups = {group for pair in _comparisons(plan) for group in pair}
    for task in tasks:
        key = task["key"]
        if Path(key).name != key or key in (".", ".."):
            raise ValueError("Task key must name one output directory")
        folder = root / key
        result_path = folder / "result.json"
        if not result_path.is_file():
            pending.append(key)
            continue
        try:
            result = read(result_path)
            if result.get("status") != "COMPLETE":
                pending.append(key)
                continue
            if result.get("recipe_sha256") is not None and result["recipe_sha256"] != plan.get("recipe_sha256"):
                raise ValueError("Worker recipe hash differs from plan")
            if task.get("family") == "basis":
                completed.append(key)
                continue
            if int(task["seed"]) not in planned_seeds:
                raise ValueError("Task seed is outside the planned pipeline seed set")
            paths = sorted((folder / "evaluations").glob("*/B*/*/summary.json"))
            if not paths:
                raise ValueError("Completed non-basis task has no evaluation summaries")
            task_evaluations = {}
            for path in paths:
                coordinate = _coordinate(path)
                if coordinate in task_evaluations:
                    raise ValueError("Duplicate evaluation coordinate")
                task_evaluations[coordinate] = evaluation_rows(path.parent)
            group, seed = str(task.get("group", task.get("variant", key))), int(task["seed"])
            required_scenarios = task.get("scenarios", [task.get("scenario", "clean")])
            if isinstance(required_scenarios, str):
                required_scenarios = [required_scenarios]
            for budget in task.get("budgets", []):
                for scenario in required_scenarios:
                    if scenario != "clean" and "diagnostics_budget" in task and budget != task["diagnostics_budget"]:
                        continue
                    if not any(coordinate[1:] == (int(budget), str(scenario)) for coordinate in task_evaluations):
                        raise ValueError("Completed task lacks planned budget/scenario: B%s/%s" % (budget, scenario))
            if group in primary_groups and task.get("family", "core") == "core" and primary not in task_evaluations:
                raise ValueError("Completed primary task lacks its primary evaluation coordinate")
            staged_references = {}
            for coordinate, rows in task_evaluations.items():
                index = (group, *coordinate)
                if seed in groups[index]:
                    raise ValueError("Duplicate group/seed/evaluation coordinate")
                reference_key = (group, coordinate[0], coordinate[2])
                reference = staged_references.setdefault(reference_key, dict(query_references.get(reference_key, {})))
                for identity in set(reference) & set(rows):
                    if reference[identity] != _query_signature(rows[identity]):
                        raise ValueError("Query trial or label mismatch across seeds or budgets")
                reference.update({key: _query_signature(row) for key, row in rows.items()})
            for coordinate, rows in task_evaluations.items():
                index = (group, *coordinate)
                groups[index][seed] = rows
                metadata[index] = {k: task[k] for k in ("family", "variant", "basis_key", "config", "setting",
                                                       "baseline_group", "clean_group") if k in task}
            query_references.update(staged_references)
            completed.append(key)
        except Exception as exc:
            invalid.append(key)
            issues.append({"task": key, "error": type(exc).__name__, "message": str(exc)})
    return groups, metadata, completed, pending, invalid, issues


def _row_context(index, metadata):
    group, partition, budget, scenario = index
    return {"group": group, "partition": partition, "budget": budget, "scenario": scenario,
            **metadata.get(index, {})}


def _safe_compare(candidate_rows, baseline_rows, issues, **context):
    try:
        return compare_rows(candidate_rows, baseline_rows)
    except Exception as exc:
        issues.append({**context, "error": type(exc).__name__, "message": str(exc)})
        return {**compare_rows({}, {}), "error": str(exc)}


def _build(plan):
    groups, metadata, completed, pending, invalid, issues = _load(plan)
    pairs, primary = _comparisons(plan), _primary_coordinate(plan)
    seeds = sorted(int(seed) for seed in plan["seeds"])
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Planned seeds must be distinct and nonempty")
    tables = {name: [] for name in TABLE_NAMES}
    for index, by_seed in sorted(groups.items()):
        context = _row_context(index, metadata)
        summary = {**context, **participant_summary(by_seed), "all_seeds_complete": set(by_seed) == set(seeds)}
        tables["evaluation_summaries"].append(summary)
        family = context.get("family", "core")
        if family == "core" and index[1] == primary[0] and index[3] == primary[2]:
            tables["budget_curves"].append(summary)
        if family == "sensitivity":
            tables["sensitivities"].append(summary)
        if family == "stress" or index[3] != primary[2]:
            tables["stress"].append(summary)
        if family == "transfer":
            tables["transfer"].append(summary)
        timing = summary["metrics"]["calibration_seconds"]
        tables["timings"].append({**context, **timing, "unit": "seconds_per_session"})
        tables["diagnostics"].extend({**context, **row} for row in diagnostic_summary(by_seed))
        tables["cv_selection_frequency"].extend({**context, **row} for row in cv_selection_summary(by_seed))
    family_complete = bool(pairs)
    for candidate, baseline in pairs:
        a, b = groups.get((candidate, *primary), {}), groups.get((baseline, *primary), {})
        row = _safe_compare(a, b, issues, candidate=candidate, baseline=baseline,
                            partition=primary[0], budget=primary[1], scenario=primary[2])
        row.update(candidate=candidate, baseline=baseline, partition=primary[0], budget=primary[1], scenario=primary[2],
                   pending_seeds=sorted(set(seeds) - (set(a) & set(b))), primary=True)
        required_ready = set(a) == set(seeds) and set(b) == set(seeds)
        # Exact session grids are required for primary inference; descriptive
        # complete-case comparisons disclose any unmatched session rows.
        matched_grid = required_ready and all(set(a[s]) == set(b[s]) for s in seeds)
        row["all_required_seeds_ready"] = required_ready
        row["matched_session_grid"] = bool(matched_grid)
        family_complete &= required_ready and matched_grid and row["n_participants"] >= 2
        tables["primary_comparisons"].append(row)
    if family_complete:
        for row in tables["primary_comparisons"]:
            row.update(paired_effect(list(row["participant_deltas_ba"].values())))
        adjusted = holm([row["p"] for row in tables["primary_comparisons"]])
        for row, p in zip(tables["primary_comparisons"], adjusted):
            row.update(p_holm=p, positive_evidence=p < .05 and row["delta_ba_pp"] > 0,
                       inference="exploratory_fixed_primary_family", holm_family_size=len(pairs))
    # All other coordinates and metadata-directed matches remain descriptive.
    seen = set()
    for index, by_seed in sorted(groups.items()):
        group, partition, budget, scenario = index
        context = _row_context(index, metadata)
        family = context.get("family", "core")
        matches = [(a, b, scenario) for a, b in pairs if a == group]
        if context.get("baseline_group"):
            matches.append((group, str(context["baseline_group"]), scenario))
        if context.get("clean_group"):
            matches.append((group, str(context["clean_group"]), primary[2]))
        elif scenario != primary[2] and (group, partition, budget, primary[2]) in groups:
            matches.append((group, group, primary[2]))
        for candidate, baseline, baseline_scenario in matches:
            identity = (candidate, baseline, partition, budget, scenario, baseline_scenario)
            if identity in seen:
                continue
            seen.add(identity)
            reference = groups.get((baseline, partition, budget, baseline_scenario))
            if reference is None and scenario != primary[2]:
                reference = groups.get((baseline, partition, budget, primary[2]))
                baseline_scenario = primary[2]
            if reference is None:
                continue
            row = {"candidate": candidate, "baseline": baseline, "partition": partition,
                   "budget": budget, "scenario": scenario, "baseline_scenario": baseline_scenario,
                   **_safe_compare(by_seed, reference, issues, candidate=candidate, baseline=baseline,
                                   partition=partition, budget=budget, scenario=scenario)}
            name = ("sensitivity_comparisons" if family == "sensitivity" else
                    "transfer_comparisons" if family == "transfer" else
                    "stress_comparisons" if family == "stress" or scenario != primary[2] else "budget_comparisons")
            tables[name].append(row)
    return {"status": "ERROR" if issues else "PARTIAL" if pending else "COMPLETE",
            "schema_version": 1, "recipe_sha256": plan.get("recipe_sha256"),
            "complete_tasks": len(completed), "total_tasks": len(plan["tasks"]),
            "completed_tasks": completed, "pending_tasks": pending, "invalid_tasks": invalid, "issues": issues,
            "primary_family_complete": bool(family_complete), "core_complete": bool(family_complete),
            "planned_comparison_count": len(pairs), "planned_seeds": seeds,
            "primary_coordinate": {"partition": primary[0], "budget": primary[1], "scenario": primary[2]},
            "exploratory_internal_replication": True, "fresh_confirmatory": False,
            "interpretation": INTERPRETATION,
            "statistical_unit": "participant; paired sessions averaged within seed, then available seed means averaged within participant",
            "missingness": "metric-specific complete paired session cases, no imputation; explicit participant and session counts",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "statistics_seed": STATISTICS_SEED,
            "confidence_interval_scope": "individual unadjusted percentile participant bootstrap; conditional on fixed split, pipeline, and available seeds; not simultaneous intervals",
            "inference_rule": "No primary p-values until every predeclared primary group/seed and comparison is ready; Holm uses the entire fixed family. All other comparisons are descriptive.",
            "tables": tables, "comparisons": tables["primary_comparisons"], **tables}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def writer(temporary):
        with open(temporary, "w", newline="") as handle:
            if not fields:
                return
            output = csv.DictWriter(handle, fieldnames=fields)
            output.writeheader()
            output.writerows({key: json.dumps(value, sort_keys=True, allow_nan=False)
                              if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
                             for row in rows)
    atomic(path, writer)


def build_report(plan):
    """Return a status dictionary; a reporting error never propagates to jobs.

    Writes ``<output>/reports/report.json`` and one CSV per aggregate table.
    Only task-local result/evaluation JSON files are read.
    """
    try:
        report = _build(plan)
    except Exception as exc:
        report = {"status": "ERROR", "primary_family_complete": False,
                  "exploratory_internal_replication": True, "fresh_confirmatory": False,
                  "interpretation": INTERPRETATION, "tables": {name: [] for name in TABLE_NAMES}, "comparisons": [],
                  "issues": [{"error": type(exc).__name__, "message": str(exc)}]}
    try:
        folder = Path(plan["output"]) / "reports"
        report["paths"] = {"json": str(folder / "report.json"),
                           "csv": {name: str(folder / (name + ".csv")) for name in report["tables"]}}
        for name, rows in report["tables"].items():
            write_csv(folder / (name + ".csv"), rows)
        put(folder / "report.json", report)
    except Exception as exc:
        report["status"] = "ERROR"
        report.setdefault("issues", []).append({"error": type(exc).__name__, "message": str(exc),
                                                "stage": "write_reports"})
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build local Round 4 aggregate reports from saved results")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    report = build_report(read(args.plan))
    print(json.dumps({key: report.get(key) for key in
                      ("status", "complete_tasks", "total_tasks", "primary_family_complete", "paths")},
                     ensure_ascii=False), flush=True)
    return 1 if report["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
