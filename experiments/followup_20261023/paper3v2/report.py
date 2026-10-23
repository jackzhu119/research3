"""Participant-paired reporting. Training never waits for this module.

Primary tests use the fixed core family only. Sensitivities, stress tests and
budget curves remain descriptive; incomplete batches never claim significance.
"""
from collections import defaultdict
import csv
import itertools
from pathlib import Path
import numpy as np
from .io import read, put, atomic, digest

DEFAULT_COMPARISONS = [("dual", "P"), ("temporal", "P"), ("softPCT", "PCT"),
    ("scalar", "P"), ("global", "P"), ("dual", "scalar"), ("dual", "global"),
    ("dual_temporal", "dual")]
METRICS = ("ba", "bce", "f0_ba", "negative_flip_rate", "corrected_error_rate",
    "brier_balanced", "nll_balanced", "ece_balanced_10bins", "f0_brier_balanced",
    "f0_nll_balanced", "f0_ece_balanced_10bins", "run_ba_p10", "run_ba_min",
    "calibration_seconds", "fallback_fraction", "temporal_eligible_fraction")
RATE_METRICS = ("negative_flip_rate", "corrected_error_rate", "run_ba_p10", "run_ba_min")
STRESS_NAMES = ("gain10", "gain25", "rotation10", "rotation25", "label_noise10", "label_noise20")


def session_rows(folder):
    return [value for p in sorted(Path(folder).glob("*.json"))
            if "participant_id" in (value := read(p)) and "session_id" in value]


def _key(row):
    return str(row["participant_id"]), str(row["session_id"])


def _query_signature(row):
    ids, labels = row["event_uids"], row["labels"]
    if not ids or len(ids) != len(labels) or len(set(ids)) != len(ids):
        raise ValueError("Empty, duplicated or mismatched query trials")
    if set(labels) != {0, 1}:
        raise ValueError("Both query classes are required for balanced metrics")
    actual = digest({"uids": ids, "labels": labels})
    if row.get("query_signature", actual) != actual:
        raise ValueError("Stored query signature disagrees with actual trial/label rows")
    return actual


def evaluation_rows(folder, reference=None):
    """Validate outputs only: no source cache scan and no effect on training jobs."""
    folder = Path(folder)
    summary_path = folder / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Evaluation summary missing: " + str(folder))
    summary = read(summary_path)
    expected = summary.get("expected_sessions")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Evaluation summary lacks expected sessions: " + str(folder))
    expected_map = {_key(row): row["query_signature"] for row in expected}
    if len(expected_map) != len(expected):
        raise ValueError("Duplicate expected participant/session")
    rows = session_rows(folder)
    actual = {_key(row): row for row in rows}
    if len(actual) != len(rows):
        raise ValueError("Duplicate participant/session output")
    if set(actual) != set(expected_map):
        raise ValueError("Missing or extra participant/session outputs: " + str(folder))
    for key, row in actual.items():
        row.pop("_report_metrics", None)
        if _query_signature(row) != expected_map[key]:
            raise ValueError("Query trial or label mismatch against evaluation summary")
        if row.get("budget") is not None and summary.get("budget") is not None and row["budget"] != summary["budget"]:
            raise ValueError("Evaluation budget mismatch")
        row["_report_metrics"] = row_metrics(row)
    if reference is not None:
        if set(expected_map) != set(reference):
            raise ValueError("Participant/session grid differs across methods, seeds or budgets")
        if expected_map != reference:
            raise ValueError("Query trial or label mismatch across methods, seeds or budgets")
    return actual, expected_map


def _probability_metrics(values, labels, predictions=None):
    probability = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    if (probability.shape != (len(y), 2) or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)
            or not np.allclose(probability.sum(1), 1., atol=2e-6, rtol=0)):
        raise ValueError("Invalid two-class probability output")
    weights = 1. / (2 * np.bincount(y, minlength=2)[y])
    brier = float(np.sum(weights * (probability[:, 1] - y)**2))
    nll = float(np.sum(weights * -np.log(np.maximum(probability[np.arange(len(y)), y], 1e-12))))
    predicted = probability.argmax(1) if predictions is None else np.asarray(predictions)
    if predicted.shape != y.shape or not set(predicted.tolist()) <= {0, 1}:
        raise ValueError("Invalid prediction rows")
    confidence = probability.max(1)
    correct = predicted == y
    ece = 0.
    # Fixed ten equal-width bins. Class-balanced weighting is explicit, not the
    # ordinary prevalence-weighted ECE; reporting never fits calibration.
    bin_ids = np.minimum((confidence * 10).astype(int), 9)
    for index in range(10):
        mask = bin_ids == index
        if mask.any():
            ece += abs(float(np.sum(weights[mask] * (correct[mask] - confidence[mask]))))
    return {"brier_balanced": brier, "nll_balanced": nll, "ece_balanced_10bins": ece}


def row_metrics(row):
    if "_report_metrics" in row:
        return row["_report_metrics"]
    result = {k: row.get(k) for k in METRICS}
    result["fallback_fraction"] = float(bool(row.get("fallback")))
    result["temporal_eligible_fraction"] = float(row.get("future_runs_with_both_classes", 0) >= 2)
    if row.get("probabilities") is not None:
        result.update(_probability_metrics(row["probabilities"], row["labels"], row.get("predictions")))
    if row.get("f0_probabilities") is not None:
        result.update({"f0_" + k: v for k, v in _probability_metrics(row["f0_probabilities"], row["labels"], row.get("f0_predictions")).items()})
    if row.get("predictions") is not None:
        y, pred = np.asarray(row["labels"]), np.asarray(row["predictions"])
        if pred.shape != y.shape or not set(pred.tolist()) <= {0, 1}:
            raise ValueError("Invalid prediction rows")
        if row.get("probabilities") is not None:
            probabilities = np.asarray(row["probabilities"])
            if np.any(probabilities.max(1) - probabilities[np.arange(len(y)), pred.astype(int)] > 1e-7):
                raise ValueError("Probabilities disagree with stored predictions")
        balanced = lambda values: .5 * sum(float(np.mean(values[y == c])) for c in (0, 1))
        measured = balanced(pred == y)
        if row.get("ba") is not None and not np.isclose(measured, row["ba"], atol=1e-8, rtol=0):
            raise ValueError("Stored BA disagrees with actual predictions")
        result["ba"] = measured
        if row.get("run_ids") is not None:
            run_ids = np.asarray(row["run_ids"], dtype=str)
            if run_ids.shape != y.shape:
                raise ValueError("Run identities differ from query rows")
            run_ba = []
            for rid in sorted(set(run_ids)):
                mask = run_ids == rid
                if all(np.any(mask & (y == c)) for c in (0, 1)):
                    run_ba.append(.5 * sum(float(np.mean(pred[mask & (y == c)] == c)) for c in (0, 1)))
            result["run_ba_p10"] = float(np.quantile(run_ba, .1)) if run_ba else None
            result["run_ba_min"] = float(min(run_ba)) if run_ba else None
            result["temporal_eligible_fraction"] = float(len(run_ba) >= 2)
        if row.get("f0_predictions") is not None:
            base = np.asarray(row["f0_predictions"])
            if base.shape != y.shape or not set(base.tolist()) <= {0, 1}:
                raise ValueError("Invalid frozen-source prediction rows")
            if row.get("f0_probabilities") is not None:
                probabilities = np.asarray(row["f0_probabilities"])
                if np.any(probabilities.max(1) - probabilities[np.arange(len(y)), base.astype(int)] > 1e-7):
                    raise ValueError("Frozen-source probabilities disagree with predictions")
            result.update(f0_ba=balanced(base == y), negative_flip_rate=balanced((base == y) & (pred != y)),
                          corrected_error_rate=balanced((base != y) & (pred == y)))
            # Net correction and BA gain are the same class-balanced quantity.
            if not np.isclose(result["ba"] - result["f0_ba"],
                result["corrected_error_rate"] - result["negative_flip_rate"], atol=1e-12):
                raise ValueError("Inconsistent correction/negative-flip accounting")
    for k, value in result.items():
        if value is not None and (not isinstance(value, (int, float, np.number)) or not np.isfinite(value)):
            raise ValueError("Non-finite or non-numeric report metric: " + k)
    return result


def _participant_summary(rows_by_seed):
    people = defaultdict(list)
    for rows in rows_by_seed.values():
        for row in rows.values():
            people[str(row["participant_id"])].append(row_metrics(row))
    result = {"n_participants": len(people), "completed_seeds": sorted(rows_by_seed),
              "completed_seed_count": len(rows_by_seed), "sessions_per_seed": len(next(iter(rows_by_seed.values()))) if rows_by_seed else 0}
    for metric in METRICS:
        values = []
        eligible_sessions = 0
        for entries in people.values():
            available = [r[metric] for r in entries if r[metric] is not None]
            eligible_sessions += len(available)
            # Missing probability arrays are not silently removed from denominators.
            # Run-tail metrics are conditional on having a physical-run score.
            if available and (len(available) == len(entries) or metric.startswith("run_ba_")):
                values.append(float(np.mean(available)))
        complete = len(values) == len(people)
        result[metric] = float(np.mean(values)) if values and (complete or metric.startswith("run_ba_")) else None
        if metric.startswith("run_ba_"):
            result[metric + "_n_participants"] = len(values)
            result[metric + "_n_session_seed_rows"] = eligible_sessions
    # Existing names retained, now explicitly participant-equal.
    result["temporal_eligible_session_fraction"] = result["temporal_eligible_fraction"]
    result["fallback_session_fraction"] = result["fallback_fraction"]
    return result


def write_csv(path, rows):
    if not rows:
        atomic(path, lambda p: Path(p).write_text(""))
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    def writer(p):
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    atomic(path, writer)


def paired_effect(differences, *, seed=20261018, replicates=20000):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two paired participants are required")
    rng = np.random.default_rng(seed)
    bootstrap = d[rng.integers(0, len(d), size=(replicates, len(d)))].mean(1)
    observed = abs(d.mean())
    if len(d) <= 18:
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(d))))
        p = float(np.mean(np.abs((signs * d).mean(1)) >= observed - 1e-14))
        p_kind = "exact_paired_sign_flip"
    else:
        signs = rng.choice((-1, 1), size=(replicates, len(d)))
        p = float((1 + np.sum(np.abs((signs * d).mean(1)) >= observed - 1e-14)) / (replicates + 1))
        p_kind = "monte_carlo_paired_sign_flip"
    return {"n_participants": len(d), "delta_ba_pp": float(100 * d.mean()),
        "ci95_pp": (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
        "p": p, "p_method": p_kind,
        "test_assumption": "paired participant differences sign-symmetric under null; conditional on this split and fixed pipeline seeds",
        "participant_negative_transfer_fraction": float(np.mean(d < 0))}


def paired_description(differences, *, scale=100, seed=20261018, replicates=20000):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        return {"n_participants": len(d), "delta": float(scale * d.mean()) if len(d) else None,
                "ci95": None, "inference": "insufficient_paired_participants"}
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(replicates, len(d)))].mean(1)
    return {"n_participants": len(d), "delta": float(scale * d.mean()),
            "ci95": (scale * np.quantile(boot, [.025, .975])).tolist(), "inference": "descriptive_unadjusted_interval"}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.] * len(values), 0.
    for position, index in enumerate(order):
        previous = max(previous, min(1., values[index] * (len(values) - position)))
        result[index] = previous
    return result


def compare_rows(candidate, baseline, *, inferential=False):
    seeds = sorted(set(candidate) & set(baseline))
    if not seeds:
        return None
    differences, bases = defaultdict(list), defaultdict(list)
    secondary = {k: defaultdict(list) for k in RATE_METRICS}
    per_seed = []
    reference = None
    for seed in seeds:
        a, b = candidate[seed], baseline[seed]
        if not a or set(a) != set(b) or (reference is not None and set(a) != reference):
            raise ValueError("Incomplete participant/session pair")
        reference = set(a)
        seed_differences = defaultdict(list)
        for key in a:
            if _query_signature(a[key]) != _query_signature(b[key]):
                raise ValueError("Query trial or label mismatch")
            ma, mb = row_metrics(a[key]), row_metrics(b[key])
            person = key[0]
            delta = ma["ba"] - mb["ba"]
            differences[person].append(delta); bases[person].append(mb["ba"])
            seed_differences[person].append(delta)
            for metric in RATE_METRICS:
                if ma[metric] is not None and mb[metric] is not None:
                    secondary[metric][person].append(ma[metric] - mb[metric])
        per_seed.append({"seed": seed, "delta_ba_pp": float(100 * np.mean([np.mean(v) for v in seed_differences.values()])),
                         "n_participants": len(seed_differences)})
    values = [np.mean(v) for _, v in sorted(differences.items())]
    if inferential:
        result = paired_effect(values)
    else:
        described = paired_description(values)
        result = {"n_participants": described["n_participants"], "delta_ba_pp": described["delta"],
                  "ci95_pp": described["ci95"], "p": None, "p_method": None,
                  "participant_negative_transfer_fraction": float(np.mean(np.asarray(values) < 0))}
    base = float(np.mean([np.mean(v) for v in bases.values()]))
    result.update(baseline_ba=base, candidate_ba=base + result["delta_ba_pp"] / 100,
                  relative_error_reduction_pct=result["delta_ba_pp"] / (1-base) if base < 1 else None,
                  completed_seeds=seeds, completed_seed_count=len(seeds), per_seed=per_seed,
                  inference="fixed_core_family" if inferential else "descriptive_no_significance_claim",
                  secondary_tradeoffs={k: {**paired_description([np.mean(v) for _, v in sorted(groups.items())]),
                                          "unit": "percentage_points"} for k, groups in secondary.items()})
    return result


def frozen_rows(rows):
    result = {}
    for key, row in rows.items():
        value = dict(row)
        value.pop("_report_metrics", None)
        value.update(ba=row.get("f0_ba"), bce=None, predictions=row.get("f0_predictions"),
                     probabilities=row.get("f0_probabilities"), calibration_seconds=0.,
                     negative_flip_rate=0., corrected_error_rate=0., fallback=None,
                     run_ba_p10=None, run_ba_min=None)
        if value["ba"] is None:
            return None
        result[key] = value
    return result


def curve_summary(plan, evaluations=None):
    if evaluations is None:
        evaluations = {}
        reference = None
        for task in plan["tasks"]:
            if task.get("arm", "core") not in ("core", "reference"):
                continue
            for budget in task.get("budgets", plan["budgets"]):
                folder = Path(plan["output"]) / task["key"] / f"holdout/B{budget}"
                if (folder / "summary.json").is_file():
                    rows, signature = evaluation_rows(folder, reference)
                    reference = signature if reference is None else reference
                    evaluations[task["key"], budget] = rows
    result = []
    for variant in list(plan["variants"]) + list(plan.get("reference_variants", [])):
        for budget in plan["budgets"]:
            by_seed = {task["seed"]: evaluations[task["key"], budget] for task in plan["tasks"]
                       if task.get("arm", "core") in ("core", "reference") and task["config"]["variant"] == variant
                       and (task["key"], budget) in evaluations}
            if by_seed:
                result.append({"variant": variant, "arm": "reference" if variant in plan.get("reference_variants", []) else "core",
                               "budget": budget, **_participant_summary(by_seed),
                               "all_seeds_complete": set(by_seed) == set(plan["seeds"]),
                               "interpretation": ("per-session support adaptation; no outer meta-training; same future query"
                                   if variant in plan.get("reference_variants", []) else
                                   "B20-trained checkpoint evaluated with shorter support; same future query")})
    # The frozen-source reference costs no extra fit and shares exact query trials.
    for budget in plan["budgets"]:
        sources = {}
        for task in plan["tasks"]:
            if task.get("arm", "core") not in ("core", "reference") or task["seed"] in sources:
                continue
            if (task["key"], budget) in evaluations:
                base_rows = frozen_rows(evaluations[task["key"], budget])
                if base_rows is not None:
                    sources[task["seed"]] = base_rows
        if sources:
            result.append({"variant": "f0", "arm": "reference", "budget": budget,
                           **_participant_summary(sources), "all_seeds_complete": set(sources) == set(plan["seeds"]),
                           "interpretation": "frozen source; no target calibration; identical future query"})
    return result


def aggregate(plan):
    root = Path(plan["output"])
    complete, missing, evaluations, reference = [], [], {}, None
    for task in plan["tasks"]:
        p = root / task["key"] / "result.json"
        ready = p.exists() and read(p).get("status") == "COMPLETE"
        (complete if ready else missing).append(task)
        if ready:
            for budget in task.get("budgets", plan["budgets"]):
                rows, signature = evaluation_rows(root / task["key"] / f"holdout/B{budget}", reference)
                reference = signature if reference is None else reference
                evaluations[task["key"], budget] = rows
    core = [task for task in plan["tasks"] if task.get("arm", "core") == "core"]
    completed_core = [task for task in complete if task.get("arm", "core") == "core"]
    core_complete = bool(core) and len(core) == len(completed_core)
    pairs = [tuple(pair) for pair in plan.get("comparisons", DEFAULT_COMPARISONS)]
    methods = defaultdict(dict)
    for task in completed_core:
        methods[task["config"]["variant"]][task["seed"]] = evaluations[task["key"], 20]
    status = {"status": "COMPLETE" if not missing else "PARTIAL", "complete_tasks": len(complete),
              "total_tasks": len(plan["tasks"]), "pending_tasks": [t["key"] for t in missing],
              "core_complete": core_complete, "planned_comparison_count": len(pairs),
              "comparisons": [], "interpretation": "internal heldout replication; not new external/clinical proof",
              "confidence_interval_scope": "participant resampling conditional on the fixed split and completed seeds; individual, not simultaneous intervals"}
    for candidate, baseline in pairs:
        available = set(methods[candidate]) & set(methods[baseline])
        inferential = core_complete and available == set(plan["seeds"])
        row = compare_rows(methods[candidate], methods[baseline], inferential=inferential)
        if row is not None:
            row.update(candidate=candidate, baseline=baseline, p_holm=None, positive_evidence=None,
                       pending_seeds=sorted(set(plan["seeds"]) - available))
            status["comparisons"].append(row)
    if core_complete:
        if len(status["comparisons"]) != len(pairs) or any(c["p"] is None for c in status["comparisons"]):
            raise ValueError("Complete core results do not cover the fixed comparison family")
        adjusted = holm([c["p"] for c in status["comparisons"]])
        for row, p in zip(status["comparisons"], adjusted):
            row["p_holm"] = p
            row["p_holm_" + str(len(pairs)) + "_comparisons"] = p
            row["positive_evidence"] = p < .05 and row["delta_ba_pp"] > 0
    status["budget_curves"] = curve_summary(plan, evaluations)
    status["budget_comparisons"], status["reference_comparisons"] = [], []
    for budget in plan["budgets"]:
        group = defaultdict(dict)
        for task in complete:
            if task.get("arm", "core") not in ("core", "reference") or (task["key"], budget) not in evaluations:
                continue
            group[task["config"]["variant"]][task["seed"]] = evaluations[task["key"], budget]
            if task["seed"] not in group["f0"]:
                base = frozen_rows(evaluations[task["key"], budget])
                if base is not None:
                    group["f0"][task["seed"]] = base
        for candidate, baseline in pairs:
            row = compare_rows(group[candidate], group[baseline])
            if row:
                status["budget_comparisons"].append({"budget": budget, "candidate": candidate, "baseline": baseline, **row})
        for candidate, baseline in (("P", "f0"), ("P", "head_only"), ("P", "support_EA"), ("dual", "support_EA")):
            row = compare_rows(group[candidate], group[baseline])
            if row:
                status["reference_comparisons"].append({"budget": budget, "candidate": candidate, "baseline": baseline, **row})
    # Matched P/dual sensitivity pairs are planned explorations, not another winner search.
    settings = defaultdict(lambda: defaultdict(dict))
    for task in complete:
        if task.get("arm", "core") == "sensitivity":
            key = task.get("setting") or digest({k:v for k,v in task["config"].items() if k != "variant"})
            settings[key][task["config"]["variant"]][task["seed"]] = evaluations[task["key"], 20]
    status["sensitivities"] = []
    for setting, values in sorted(settings.items()):
        row = compare_rows(values["dual"], values["P"])
        if row:
            status["sensitivities"].append({"setting": setting, "candidate": "dual", "baseline": "P", **row})
    status["diagnostics"], status["diagnostic_comparisons"] = [], []
    diagnostic_groups = defaultdict(lambda: defaultdict(dict))
    for task in completed_core:
        for scenario in set(STRESS_NAMES) | set(task.get("diagnostics", [])):
            folder = root / task["key"] / "diagnostics" / scenario
            if scenario in task.get("diagnostics", []) and not (folder / "summary.json").exists():
                raise ValueError("Completed task lacks planned stress result: " + str(folder))
            if (folder / "summary.json").exists():
                rows, _ = evaluation_rows(folder, reference)
                diagnostic_groups[scenario][task["config"]["variant"]][task["seed"]] = rows
    for scenario, variants in sorted(diagnostic_groups.items()):
        for variant, by_seed in sorted(variants.items()):
            clean = methods[variant]
            change = compare_rows(by_seed, clean)
            status["diagnostics"].append({"scenario": scenario, "variant": variant,
                **_participant_summary(by_seed), "change_from_own_clean": change,
                "interpretation": "synthetic stress; exploratory; not real external validation"})
        for candidate, baseline in (("dual", "P"), ("softPCT", "PCT"), ("softPCT", "softPCT_mean")):
            row = compare_rows(variants[candidate], variants[baseline])
            if row:
                status["diagnostic_comparisons"].append({"scenario": scenario, "candidate": candidate,
                    "baseline": baseline, **row, "synthetic_stress": True})
    status["secondary_metrics_note"] = ("Descriptive paired intervals, no non-inferiority or extra significance claims. "
        "Negative flips/corrections and probability metrics use class-balanced weights. "
        "Run P10 is a within-session empirical interpolation; report eligible run coverage alongside it.")
    for name in ("comparisons", "budget_curves", "budget_comparisons", "reference_comparisons", "sensitivities", "diagnostics", "diagnostic_comparisons"):
        write_csv(root / (name + ".csv"), status[name])
    put(root / "report.json", status)
    return status
