"""Participant-paired statistics; seeds, sessions and trials are NOT independent n."""
from collections import defaultdict
import itertools
import csv
from pathlib import Path
import numpy as np
from .io import read, put, atomic

def session_rows(folder):
    return [value for p in sorted(Path(folder).glob("*.json"))
            if "participant_id" in (value := read(p))]

def curve_summary(plan):
    result = []
    metrics = ("ba", "bce", "f0_ba", "negative_flip_rate", "corrected_error_rate", "run_ba_p10", "calibration_seconds")
    for variant in plan["variants"]:
        for budget in plan["budgets"]:
            people, coverage, fallback, grids = {}, [], [], []
            for seed in plan["seeds"]:
                rows = session_rows(Path(plan["output"]) / f"{variant}-seed{seed}" / f"holdout/B{budget}")
                grids.append({(r["participant_id"], r["session_id"]) for r in rows})
                for r in rows:
                    values = people.setdefault(r["participant_id"], defaultdict(list))
                    for k in metrics:
                        if r.get(k) is not None:
                            values[k].append(r[k])
                    coverage.append(r.get("future_runs_with_both_classes", 0) >= 2)
                    fallback.append(bool(r.get("fallback")))
            if not people or any(grid != grids[0] for grid in grids):
                raise ValueError("Incomplete budget/seed evaluation grid")
            row = {"variant": variant, "budget": budget, "n_participants": len(people)}
            for k in metrics:
                v = [np.mean(p[k]) for p in people.values() if p[k]]
                row[k] = float(np.mean(v)) if v else None
            row.update(temporal_eligible_session_fraction=float(np.mean(coverage)),
                       fallback_session_fraction=float(np.mean(fallback)))
            result.append(row)
    return result

def write_csv(path, rows):
    if not rows:
        return
    def writer(p):
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
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
        "test_assumption": "paired participant differences sign-symmetric under null; observational comparison",
        "participant_negative_transfer_fraction": float(np.mean(d < 0))}

def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [0.] * len(values), 0.
    for position, index in enumerate(order):
        previous = max(previous, min(1., values[index] * (len(values) - position)))
        result[index] = previous
    return result

def aggregate(plan):
    root = Path(plan["output"])
    complete, missing = [], []
    for task in plan["tasks"]:
        p = root / task["key"] / "result.json"
        (complete if p.exists() and read(p).get("status") == "COMPLETE" else missing).append(task)
    status = {"status": "COMPLETE" if not missing else "PARTIAL", "complete_tasks": len(complete),
              "total_tasks": len(plan["tasks"]), "pending_tasks": [t["key"] for t in missing],
              "comparisons": [], "interpretation": "internal heldout replication; not new external/clinical proof"}
    if missing:
        put(root / "report.json", status); return status
    # Pair at original event level before averaging to participant level.
    sessions = {}
    for task in complete:
        rows = session_rows(root / task["key"] / "holdout/B20")
        sessions[task["config"]["variant"], task["seed"]] = {(r["participant_id"], r["session_id"]): r for r in rows}
    comparisons = [("dual", "P"), ("temporal", "P"), ("softPCT", "PCT"),
                   ("scalar", "P"), ("global", "P"), ("dual", "scalar"),
                   ("dual", "global"), ("dual_temporal", "dual")]
    for candidate, baseline in comparisons:
        differences, base_values = defaultdict(list), defaultdict(list)
        for seed in plan["seeds"]:
            a, b = sessions[candidate, seed], sessions[baseline, seed]
            if not a or set(a) != set(b):
                raise ValueError("Incomplete participant/session pair: " + candidate)
            for key in a:
                if a[key]["event_uids"] != b[key]["event_uids"] or a[key]["labels"] != b[key]["labels"]:
                    raise ValueError("Query trial or label mismatch")
                differences[key[0]].append(a[key]["ba"] - b[key]["ba"])
                base_values[key[0]].append(b[key]["ba"])
        # Equal sessions and equal seeds; same session grid for every method/seed required below.
        grids = [set(sessions[candidate, s]) for s in plan["seeds"]]
        if any(g != grids[0] for g in grids):
            raise ValueError("Session grid differs across pipeline seeds")
        values = [np.mean(v) for _, v in sorted(differences.items())]
        result = paired_effect(values)
        baseline_ba = float(np.mean([np.mean(v) for v in base_values.values()]))
        delta = result["delta_ba_pp"] / 100
        result.update(candidate=candidate, baseline=baseline, baseline_ba=baseline_ba,
            candidate_ba=baseline_ba + delta,
            relative_error_reduction_pct=100 * delta / (1 - baseline_ba) if baseline_ba < 1 else None)
        status["comparisons"].append(result)
    adjusted = holm([c["p"] for c in status["comparisons"]])
    for row, p in zip(status["comparisons"], adjusted):
        row["p_holm_8_comparisons"] = p
        row["positive_evidence"] = p < .05 and row["ci95_pp"][0] > 0
    status["budget_curves"] = curve_summary(plan)
    status["secondary_metrics_note"] = "Descriptive, no post-hoc significance claims. P10 uses eligible physical runs; no invented windows."
    write_csv(root / "comparisons.csv", status["comparisons"])
    write_csv(root / "budget_curves.csv", status["budget_curves"])
    put(root / "report.json", status)
    return status
