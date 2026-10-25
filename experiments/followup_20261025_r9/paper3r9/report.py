"""Participant-level R9 reporting; reporting errors never alter training outputs.

Only stdlib and NumPy are imported. Finished task receipts act as immutable
commit markers: trial predictions are checked once and compact sufficient
statistics are cached under their receipt SHA. ``refresh=True`` rechecks them.
"""
from collections import defaultdict
from functools import lru_cache
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from paper3r7.io import atomic, digest, put
from paper3r7.report import row_metrics, _query_signature, _diagnostic_values

VERSION = "paper3-r9-report-v1"
STATISTICS_SEED = 20261025
BOOTSTRAP_REPLICATES = 10000
MONTE_CARLO_REPLICATES = 100000
METRICS = ("ba", "nll_balanced", "brier_balanced", "f0_ba", "f0_nll_balanced",
           "negative_flip_rate", "corrected_error_rate", "calibration_seconds",
           "prediction_seconds", "prediction_seconds_per_trial", "fallback_fraction")
INTERPRETATION = ("Exploratory repeated internal/external evaluation; prior round results informed "
                  "this design. Not fresh blinded, prospective or clinical confirmation.")


def _read(path, inputs):
    path = Path(path)
    raw = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


@lru_cache(maxsize=8192)
def bootstrap_interval(values, scale=1.):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(STATISTICS_SEED)
    samples = values[rng.integers(len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(1)
    return (scale * np.quantile(samples, [.025, .975])).tolist()


@lru_cache(maxsize=8192)
def sign_flip(values):
    """Two-sided paired mean statistic; exact enumeration through 20 people."""
    d = np.asarray(values, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("At least two finite participant differences required")
    if np.all(d == 0):
        return {"p": 1., "p_method": "exact_paired_sign_flip" if len(d) <= 20
                else "monte_carlo_paired_sign_flip"}
    observed = abs(float(d.sum()))
    tolerance = 1e-12 * max(1., float(np.abs(d).sum()))
    count = 0
    if len(d) <= 20:
        total = 1 << len(d)
        bits = np.arange(len(d), dtype=np.uint64)
        for start in range(0, total, 8192):
            numbers = np.arange(start, min(start + 8192, total), dtype=np.uint64)
            signs = (((numbers[:, None] >> bits) & 1).astype(np.float64) * 2 - 1)
            count += int(np.count_nonzero(np.abs(np.sum(signs * d, axis=1)) >= observed - tolerance))
        return {"p": count / total, "p_method": "exact_paired_sign_flip"}
    rng = np.random.default_rng(STATISTICS_SEED)
    for start in range(0, MONTE_CARLO_REPLICATES, 8192):
        size = min(8192, MONTE_CARLO_REPLICATES-start)
        signs = rng.integers(0, 2, (size, len(d))) * 2 - 1
        count += int(np.count_nonzero(np.abs(np.sum(signs * d, axis=1)) >= observed - tolerance))
    return {"p": (1 + count) / (1 + MONTE_CARLO_REPLICATES),
            "p_method": "monte_carlo_paired_sign_flip"}


def holm(values):
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Invalid primary p-value")
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.] * len(values), 0.
    for position, index in enumerate(order):
        previous = max(previous, min(1., values[index]*(len(values)-position)))
        result[index] = previous
    return result


def _compact_row(row):
    metrics = row_metrics(row)
    for name in ("nll_balanced", "brier_balanced", "f0_nll_balanced", "f0_brier_balanced"):
        if row.get(name) is not None and metrics.get(name) is not None:
            if not np.isclose(row[name], metrics[name], atol=2e-6, rtol=2e-6):
                raise ValueError("Stored metric disagrees with predictions: " + name)
    metrics["fallback_fraction"] = float(bool(row.get("fallback", row.get("fallback_reason"))))
    required = ("ba", "nll_balanced")
    if any(metrics.get(name) is None for name in required):
        raise ValueError("Missing primary probability metric")
    return {"participant_id": str(row["participant_id"]), "session_id": str(row["session_id"]),
            "query_signature": _query_signature(row),
            "source_identity": digest(row["source_identity"]) if row.get("source_identity") is not None else None,
            "data_identity": digest(row["data_identity"]) if row.get("data_identity") is not None else None,
            "metrics": {k: float(v) for k, v in metrics.items() if v is not None},
            "diagnostics": _diagnostic_values(row.get("diagnostics", {})),
            "fallback_reason": row.get("fallback_reason"), "query_bins":row.get("query_mechanism_bins",[])}


def _condition_rows(folder, partition, budget, scenario, inputs, receipt=None):
    summary = _read(folder / "summary.json", inputs)
    expected = summary.get("expected_sessions")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Evaluation summary missing nonempty expected_sessions")
    expected_map = {(str(r["participant_id"]), str(r["session_id"])): r["query_signature"]
                    for r in expected}
    if len(expected_map) != len(expected):
        raise ValueError("Duplicate expected session identity")
    if summary.get("sessions") is not None and summary["sessions"] != len(expected):
        raise ValueError("Summary session count disagrees with expected_sessions")
    rows = []
    seen = set()
    for path in sorted(folder.glob("*.json")):
        if path.name == "summary.json":
            continue
        row = _read(path, inputs)
        compact = _compact_row(row)
        key = compact["participant_id"], compact["session_id"]
        if key in seen:
            raise ValueError("Duplicate output session identity")
        seen.add(key)
        if key not in expected_map or compact["query_signature"] != expected_map[key]:
            raise ValueError("Query identity/labels disagree with summary manifest")
        for name, value in (("partition", partition), ("budget", budget), ("scenario", scenario)):
            if row.get(name) is not None and row[name] != value:
                raise ValueError("Evaluation coordinate mismatch: " + name)
        if summary.get("parameter_id") is not None and row.get("parameter_id") != summary["parameter_id"]:
            raise ValueError("Per-session fitted parameter identity differs")
        if summary.get("data_identity") is not None and row.get("data_identity") != summary["data_identity"]:
            raise ValueError("Per-session dataset identity differs")
        for identity in ("source_identity", "data_identity"):
            if receipt and receipt.get(identity) is not None and row.get(identity) != receipt[identity]:
                raise ValueError("Per-session identity disagrees with completion receipt: " + identity)
        rows.append(compact)
    if seen != set(expected_map):
        raise ValueError("Missing or extra output sessions")
    return rows


def _expected_conditions(plan, task):
    if task.get("family") in ("train", "basis", "class_reference", "feature_reference", "select"):
        return []
    partitions = task.get("partitions", [task.get("partition", "holdout")])
    return [(str(p), int(b), str(s)) for p in partitions
            for b in task.get("budgets", plan.get("budgets", [20]))
            for s in task.get("scenarios", ["clean"])]


def _task_data(plan, task, root, receipt, receipt_sha):
    inputs, conditions = {}, []
    for partition, budget, scenario in _expected_conditions(plan, task):
        folder = root / task["key"] / "evaluations" / partition / ("B"+str(budget)) / scenario
        rows = _condition_rows(folder, partition, budget, scenario, inputs, receipt)
        conditions.append({"partition": partition, "budget": budget, "scenario": scenario, "rows": rows})
    return {"receipt_sha256": receipt_sha, "task_sha256": digest(task),
            "conditions": conditions, "inputs": inputs,
            "task_identity": receipt.get("identity"),
            "source_identity": receipt.get("source_identity"),
            "data_identity": receipt.get("data_identity")}


def _by_person(by_seed, metric, *, diagnostics=False):
    people = defaultdict(list)
    rows_count, missing = 0, 0
    for seed in sorted(by_seed):
        sessions = defaultdict(list)
        for row in by_seed[seed]:
            value = row["diagnostics" if diagnostics else "metrics"].get(metric)
            if value is None:
                missing += 1
                continue
            sessions[row["participant_id"]].append(value)
            rows_count += 1
        for person, values in sessions.items():
            people[person].append(float(np.mean(values)))
    return {p: float(np.mean(v)) for p, v in sorted(people.items())}, rows_count, missing


def _description(values, scale=1., *, interval=False):
    values = tuple(float(v) for v in values)
    return {"mean": float(np.mean(values))*scale if values else None,
            "ci95": bootstrap_interval(values, scale) if interval else None, "n_participants": len(values),
            "interval_scope": "individual descriptive participant bootstrap; not simultaneous" if interval else "not calculated for exploratory summary grid"}


def _csv_tables(destination, name, rows):
    """Bounded CSV chunks keep full exploratory grids eligible for GitHub sync."""
    rows = [{k: v for k, v in row.items() if k != "participant_differences"} for row in rows]
    paths = []
    for start in range(0, len(rows), 1000):
        chunk = rows[start:start+1000]
        columns = sorted({k for row in chunk for k in row})
        path = destination / f"{name}-{start//1000:04d}.csv"
        def write(filename):
            with open(filename, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in chunk:
                    writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False)
                                     if isinstance(v, (list, dict)) else v for k, v in row.items()})
        atomic(path, write)
        paths.append(path.name)
    for stale in destination.glob(name+"-*.csv"):
        if stale.name not in paths:
            stale.unlink()
    return paths


def _primary_hypotheses(plan):
    result = []
    default_partitions = plan.get("primary_partitions", [plan.get("primary_partition", "holdout")])
    default_metrics = plan.get("primary_metrics", ["ba", "nll_balanced"])
    for comparison in plan.get("comparisons", []):
        if isinstance(comparison, dict):
            a, b = comparison["candidate"], comparison["baseline"]
            partitions = comparison.get("partitions", [comparison["partition"]] if "partition" in comparison else default_partitions)
            metrics = comparison.get("metrics", [comparison["metric"]] if "metric" in comparison else default_metrics)
            budget = int(comparison.get("budget", plan.get("primary_budget", 20)))
            scenario = comparison.get("scenario", plan.get("primary_scenario", "clean"))
        else:
            a, b = comparison
            partitions, metrics = default_partitions, default_metrics
            budget, scenario = int(plan.get("primary_budget", 20)), plan.get("primary_scenario", "clean")
        for partition in partitions:
            for metric in metrics:
                if metric not in ("ba", "nll_balanced"):
                    raise ValueError("Primary metric must be ba or nll_balanced")
                result.append({"candidate": a, "baseline": b, "partition": partition,
                               "budget": budget, "scenario": scenario, "metric": metric})
    identities = [digest(v) for v in result]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate primary hypothesis declaration")
    return result


def _pair(plan, declaration, groups):
    name_a, name_b = declaration["candidate"], declaration["baseline"]
    coordinate = declaration["partition"], declaration["budget"], declaration["scenario"]
    candidate = groups.get((name_a,)+coordinate, {})
    baseline = groups.get((name_b,)+coordinate, {})
    seeds = list(plan.get("seeds", []))
    shared = sorted(set(candidate) & set(baseline))
    all_seeds = bool(seeds) and set(shared) == set(seeds) == set(candidate) == set(baseline)
    grid, query_map, matched = None, None, True
    deltas, cmeans, bmeans = defaultdict(list), defaultdict(list), defaultdict(list)
    pairs, missing = 0, 0
    metric = declaration["metric"]
    direction = 1 if metric == "ba" else -1
    for seed in shared:
        a = {(r["participant_id"], r["session_id"]): r for r in candidate[seed]}
        b = {(r["participant_id"], r["session_id"]): r for r in baseline[seed]}
        same = set(a) == set(b)
        matched &= same
        keys = set(a) & set(b)
        current_query = {k: a[k]["query_signature"] for k in keys}
        matched &= all(a[k]["query_signature"] == b[k]["query_signature"] for k in keys)
        matched &= all(a[k].get("source_identity") == b[k].get("source_identity")
                       and a[k].get("data_identity") == b[k].get("data_identity") for k in keys)
        if grid is None:
            grid, query_map = keys, current_query
        else:
            matched &= keys == grid and current_query == query_map
        within = defaultdict(list)
        for k in sorted(keys):
            av, bv = a[k]["metrics"].get(metric), b[k]["metrics"].get(metric)
            if av is None or bv is None:
                missing += 1
                continue
            within[k[0]].append((av, bv))
            pairs += 1
        for person, values in within.items():
            av, bv = np.mean(values, axis=0)
            cmeans[person].append(float(av)); bmeans[person].append(float(bv))
            deltas[person].append(direction*float(av-bv))
    people = {p: float(np.mean(v)) for p, v in sorted(deltas.items())}
    cohort = plan.get("expected_cohorts", {}).get(coordinate[0], {})
    expected_people = cohort.get("participants", cohort.get("n_participants", plan.get("primary_expected_participants")))
    expected_sessions = cohort.get("sessions", cohort.get("n_sessions", plan.get("primary_expected_sessions")))
    expected_ready = (expected_people is not None and expected_sessions is not None
                      and len(people) == expected_people and len(grid or []) == expected_sessions)
    metric_ready = bool(people) and missing == 0 and all(len(v)==len(seeds) for v in deltas.values())
    ready = all_seeds and matched and expected_ready and metric_ready and len(people) >= 2
    scale = 100 if metric == "ba" else 1
    description = _description(people.values(), scale, interval=True)
    result = dict(declaration, n_participants=len(people), delta=description["mean"], ci95=description["ci95"],
        unit="percentage_points" if metric=="ba" else "balanced_NLL_units",
        direction="candidate_minus_baseline" if metric=="ba" else "baseline_minus_candidate",
        positive_delta_means_improvement=True,
        candidate_mean=float(np.mean([np.mean(v) for v in cmeans.values()])) if people else None,
        baseline_mean=float(np.mean([np.mean(v) for v in bmeans.values()])) if people else None,
        completed_seeds=shared, required_seeds=seeds, all_required_seeds_ready=all_seeds,
        matched_session_grid=bool(matched), expected_cohort_ready=bool(expected_ready), metric_ready=metric_ready,
        comparison_ready=bool(ready), n_paired_session_seed_rows=pairs,
        expected_participants=expected_people, expected_sessions=expected_sessions,
        participant_differences=people, p=None, p_holm=None, positive_evidence=None, p_method=None,
        inference="descriptive_until_entire_prespecified_family_ready",
        ci_scope="individual unadjusted participant bootstrap; not simultaneous confidence interval")
    return result


def build_report(plan, output=None, *, refresh=False):
    """Write compact outputs; invalid inputs produce PARTIAL plus explicit issues.

    Completion is assessed independently of primary-family inference. All p-values
    remain absent until every declared pair/metric/cohort/seed grid is complete.
    """
    if isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text())
    root = Path(output or plan["output"])
    destination = root / "reports"
    cache_path = destination / "report-cache.json"
    cache = {}
    if not refresh:
        try:
            saved = json.loads(cache_path.read_text())
            if saved.get("version") == VERSION and saved.get("recipe_sha256") == plan["recipe_sha256"]:
                cache = saved.get("tasks", {})
        except (OSError, ValueError, TypeError):
            pass
    issues, task_status, input_hashes = [], [], {}
    groups, group_families, summaries = {}, {}, []
    valid_cache, seen_keys, completed = {}, set(), 0
    reused, validated = 0, 0
    for task in plan["tasks"]:
        key = task["key"]
        if Path(key).name != key or key in (".", "..") or key in seen_keys:
            issues.append({"task": key, "error": "Unsafe or duplicate task key"})
            continue
        seen_keys.add(key)
        result_path = root / key / "result.json"
        try:
            receipt = _read(result_path, input_hashes)
        except FileNotFoundError:
            task_status.append({"task": key, "status": "PENDING_OR_RUNNING"})
            continue
        except (OSError, ValueError) as error:
            issues.append({"task": key, "error": str(error)})
            task_status.append({"task": key, "status": "UNREADABLE_RESULT"})
            continue
        if receipt.get("status") != "COMPLETE":
            task_status.append({"task": key, "status": receipt.get("status", "INCOMPLETE")})
            continue
        try:
            if receipt.get("recipe_sha256") != plan["recipe_sha256"] or receipt.get("task") != task:
                raise ValueError("Task completion receipt differs from declared plan")
            receipt_sha = input_hashes[str(result_path)]
            item = cache.get(key)
            if (item and item.get("receipt_sha256") == receipt_sha
                    and item.get("task_sha256") == digest(task)):
                reused += 1
            else:
                item = _task_data(plan, task, root, receipt, receipt_sha)
                validated += 1
            duplicate = False
            for condition in item["conditions"]:
                group_key = (task.get("group", task.get("variant", key)), condition["partition"],
                             condition["budget"], condition["scenario"])
                if task["seed"] in groups.get(group_key, {}):
                    duplicate = True
            if duplicate:
                raise ValueError("Duplicate group/partition/budget/scenario/seed output")
            valid_cache[key] = item
            input_hashes.update(item["inputs"])
            for condition in item["conditions"]:
                group_key = (task.get("group", task.get("variant", key)), condition["partition"],
                             condition["budget"], condition["scenario"])
                groups.setdefault(group_key, {})[task["seed"]] = condition["rows"]
                group_families[group_key] = task.get("family", "evaluation")
            completed += 1
            task_status.append({"task": key, "status": "COMPLETE", "receipt_sha256": receipt_sha})
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"task": key, "error_type": type(error).__name__, "error": str(error)})
            task_status.append({"task": key, "status": "INVALID_COMPLETION_OUTPUTS"})
    diagnostics = []
    for coordinate, by_seed in sorted(groups.items()):
        group, partition, budget, scenario = coordinate
        metadata = dict(group=group, partition=partition, budget=budget, scenario=scenario,
                        family=group_families[coordinate], completed_seeds=sorted(by_seed))
        metrics = {}
        for metric in METRICS:
            people, rows_count, missing = _by_person(by_seed, metric)
            if people:
                metrics[metric] = dict(_description(people.values()), n_session_seed_rows=rows_count,
                                       missing_metric_session_rows=missing)
        summaries.append(dict(metadata, metrics=metrics, inference="descriptive_no_best_method_claim"))
        names = sorted({name for rows in by_seed.values() for row in rows for name in row["diagnostics"]})
        for name in names:
            people, rows_count, missing = _by_person(by_seed, name, diagnostics=True)
            diagnostics.append(dict(metadata, diagnostic=name, mean=float(np.mean(list(people.values()))),
                n_participants=len(people), n_session_seed_rows=rows_count, missing_metric_session_rows=missing,
                inference="mechanism_description_no_significance_claim"))
    try:
        declarations = _primary_hypotheses(plan)
        primary = [_pair(plan, d, groups) for d in declarations]
    except (ValueError, KeyError, TypeError) as error:
        issues.append({"scope": "primary_declarations", "error": str(error)})
        primary, declarations = [], []
    # Any invalid output is explicit; invalid primary groups cannot silently pass
    # through a second, duplicate task that happened to be readable.
    family_ready = bool(primary) and not issues and all(row["comparison_ready"] for row in primary)
    if family_ready:
        for row in primary:
            row.update(sign_flip(tuple(row["participant_differences"].values())))
            row["inference"] = "prespecified_exploratory_repeated_cohort_family"
            row["test_assumption"] = "Paired participant differences sign-symmetric under the null"
        for row, adjusted in zip(primary, holm([row["p"] for row in primary])):
            row.update(p_holm=adjusted, positive_evidence=adjusted < .05 and row["delta"] > 0)
    for row in primary:
        row["holm_family_size"] = len(declarations)
    ba_primary = [dict(row, delta_ba_pp=row["delta"], ci95_pp=row["ci95"],
                       baseline_ba=row["baseline_mean"], candidate_ba=row["candidate_mean"])
                  for row in primary if row["metric"] == "ba"]
    canonical_inputs = {str(Path(p).relative_to(root)): sha for p, sha in sorted(input_hashes.items())}
    provenance = {"plan_sha256": digest(plan), "recipe_sha256": plan["recipe_sha256"],
                  "report_version": VERSION, "input_files": canonical_inputs,
                  "input_merkle_sha256": digest(canonical_inputs),
                  "validated_tasks_this_pass": validated, "cached_tasks_reused": reused,
                  "cache_assumption": "Completed task receipt commits immutable evaluation outputs; refresh=True revalidates",
                  "source_identities": {k: v["source_identity"] for k, v in valid_cache.items()},
                  "data_identities": {k: v["data_identity"] for k, v in valid_cache.items()}}
    report = {"schema_version": VERSION, "recipe_sha256": plan["recipe_sha256"], "status": "COMPLETE" if completed == len(plan["tasks"]) and not issues else "PARTIAL",
              "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "complete_tasks": completed, "total_tasks": len(plan["tasks"]), "task_status": task_status,
              "primary_family_complete": family_ready, "planned_primary_hypothesis_count": len(declarations),
              "primary_comparisons": ba_primary, "primary_multimetric_comparisons": primary,
              "evaluation_summaries": summaries, "diagnostics": diagnostics,
              "issues": issues, "interpretation": INTERPRETATION, "fresh_confirmatory": False,
              "exploratory_internal_replication": True, "exploratory_external_replication": True,
              "provenance": provenance, "input_sha256": provenance["input_merkle_sha256"],
              "statistics": {"unit": "participant", "seeds_are_independent_participants": False,
                  "bootstrap_replicates": BOOTSTRAP_REPLICATES, "statistical_seed": STATISTICS_SEED,
                  "exact_sign_flip_max_participants": 20, "monte_carlo_replicates": MONTE_CARLO_REPLICATES}}
    session_metrics, participant_metrics, query_bins = [], [], []
    for (group, partition, budget, scenario), by_seed in sorted(groups.items()):
        for seed, rows in sorted(by_seed.items()):
            for row in rows:
                for item in row.get('query_bins',[]):
                    query_bins.append(dict(group=group,partition=partition,budget=budget,scenario=scenario,
                        seed=seed,participant_id=row['participant_id'],session_id=row['session_id'],
                        query_signature=row['query_signature'],**item))
                session_metrics.append(dict(group=group, partition=partition, budget=budget,
                    scenario=scenario, seed=seed, participant_id=row['participant_id'],
                    session_id=row['session_id'], query_signature=row['query_signature'],
                    source_identity=row.get('source_identity'), data_identity=row.get('data_identity'),
                    **row['metrics'], **{'diagnostic_'+k:v for k,v in row['diagnostics'].items()}))
        for metric in METRICS:
            people, _, _ = _by_person(by_seed, metric)
            for person, value in people.items():
                participant_metrics.append(dict(group=group, partition=partition, budget=budget,
                    scenario=scenario, participant_id=person, metric=metric, value=value,
                    averaging='sessions_within_seed_then_seeds_equal_weight'))
    report['compact_metrics'] = {'session_metrics': _csv_tables(destination, 'session_metrics', session_metrics),
        'participant_metrics': _csv_tables(destination, 'participant_metrics', participant_metrics),
        'query_mechanism_bins':_csv_tables(destination,'query_mechanism_bins',query_bins)}
    put(cache_path, {"version": VERSION, "recipe_sha256": plan["recipe_sha256"], "tasks": valid_cache})
    # The full local report retains every input hash and all diagnostics. Small
    # table chunks contain the entire aggregate grid for asynchronous sync.
    report["csv_tables"] = {name: _csv_tables(destination, name, rows) for name, rows in (
        ("primary_comparisons", primary), ("evaluation_summaries", summaries), ("diagnostics", diagnostics))}
    put(destination/"full-report.json", report)
    compact = dict(report)
    compact["task_status"] = [{k:v for k,v in item.items() if k!="receipt_sha256"} for item in task_status]
    compact["primary_comparisons"] = [{k:v for k,v in item.items() if k!="participant_differences"} for item in ba_primary]
    compact["primary_multimetric_comparisons"] = [{k:v for k,v in item.items() if k!="participant_differences"} for item in primary]
    compact["diagnostics"] = diagnostics[:400]
    compact["report_table_rows"] = {"primary_multimetric_comparisons": {"total": len(primary), "published": len(primary)},
        "evaluation_summaries": {"total": len(summaries), "published": len(summaries)},
        "diagnostics": {"total": len(diagnostics), "published": min(400,len(diagnostics)), "full_csv_rows": len(diagnostics)}}
    compact["provenance"] = {k:v for k,v in provenance.items() if k not in ("input_files", "source_identities", "data_identities")}
    compact["provenance"].update(input_file_count=len(canonical_inputs), full_local_report="reports/full-report.json",
                                  identity_manifest_sha256=digest({k:provenance[k] for k in ("source_identities","data_identities")}))
    put(destination/"report.json", compact)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = build_report(args.plan, refresh=args.refresh)
    print(json.dumps({k: result[k] for k in ("status", "complete_tasks", "total_tasks", "primary_family_complete", "issues")}, ensure_ascii=False))


if __name__ == '__main__':main()
