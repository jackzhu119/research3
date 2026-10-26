"""Metadata-only person splits and chronological B20 episodes for protocol v2.

This module does not load EEG, train models, or infer acquisition order.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_subject_split(roster: list[dict], seed: int = 20261018, *,
                         reduced_roster_amendment: dict | None = None,
                         all_candidate_qualification_records: list[dict] | None = None) -> dict:
    """Split already qualified people; reject unknown identity/channel/task evidence.

    Window length, trial QC, and model outcomes are deliberately absent from the
    eligibility predicate. Unknown intervention groups remain an explicit stratum.
    """
    if len({r["participant_id"] for r in roster}) != len(roster):
        raise ValueError("Duplicate participant identity")
    for r in roster:
        for field in ("identity_verified", "clinic11_present_verified", "pure_lr_present_verified"):
            if r.get(field) is not True:
                raise ValueError(f"Unverified {field}: {r['participant_id']}")
    n = len(roster)
    if n != 62:
        records = all_candidate_qualification_records or []
        expected = {f"S{i}" for i in range(1, 63)}
        if (not reduced_roster_amendment or not reduced_roster_amendment.get("reason")
                or not reduced_roster_amendment.get("evidence_sha256")
                or len(records) != 62 or {r.get("participant_id") for r in records} != expected
                or any(r.get("status") not in {"QUALIFIED", "INELIGIBLE"} or not r.get("evidence_sha256") for r in records)
                or {r["participant_id"] for r in records if r["status"] == "QUALIFIED"} != {r["participant_id"] for r in roster}):
            raise ValueError("Reduced roster requires explicit amendment and all 62 official candidate qualification audits")
    if n < 3:
        raise ValueError("At least three verified people are needed for three person-disjoint partitions")
    targets = [40, 11, 11] if n == 62 else _proportional_counts(n)
    names = ["meta_train", "meta_validation", "internal_holdout"]
    strata = defaultdict(list)
    for r in roster:
        strata[str(r.get("intervention_group") or "unknown")].append(r["participant_id"])
    # Allocate each stratum in stable-hash order, favoring its largest quota
    # deficit while respecting exact overall counts. No signal outcome is used.
    remaining = dict(zip(names, targets))
    assigned = {name: [] for name in names}
    stratum_counts = {}
    for group in sorted(strata, key=lambda g: (len(strata[g]), g)):
        people = sorted(strata[group], key=lambda p: _hash([seed, "Stieger2021", group, p]))
        counts = Counter()
        for p in people:
            available = [i for i, name in enumerate(names) if remaining[name] > 0]
            i = max(available, key=lambda i: (len(people) * targets[i] / n - counts[names[i]], remaining[names[i]], -i))
            name = names[i]
            assigned[name].append(p)
            remaining[name] -= 1
            counts[name] += 1
        stratum_counts[group] = dict(counts)
    result = {
        "schema_version": "1", "dataset": "Stieger2021", "seed": seed,
        "status": "FROZEN_METADATA_ONLY", "roster_sha256": _hash(roster),
        "algorithm": "stable-sha256-with-stratum-deficit-exact-quota-v1",
        "basis": "identity, real clinic11 channels and pure-LR presence; before candidate window/QC",
        "counts": dict(zip(names, targets)), "stratum_counts": stratum_counts,
        "partitions": {name: sorted(assigned[name]) for name in names},
        "no_post_window_reassignment": True,
        "reduced_roster_amendment": reduced_roster_amendment,
        "all_candidate_qualification_records_sha256": _hash(all_candidate_qualification_records) if all_candidate_qualification_records else None,
    }
    result["split_sha256"] = _hash(result)
    return result


def _proportional_counts(n: int) -> list[int]:
    raw = [n * .65, n * .175, n * .175]
    counts = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[:n-sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            j = max(range(3), key=lambda j: counts[j])
            counts[j] -= 1
            counts[i] += 1
    return counts


def build_prefix_episode(trials: list[dict], *, acquisition_order_verified: bool, budget: int = 20) -> dict:
    """Use acquired original LR trials, with no free replacement of invalid data.

    Input must be one participant/session in original acquisition order. Presence
    of real run IDs is all-or-none. Metadata is sufficient; query signals/labels
    need not be passed to any learner. `label` here is for eligibility auditing.
    """
    if not acquisition_order_verified:
        raise ValueError("Acquisition order is unverified; random split is prohibited")
    if not trials or budget < 1:
        raise ValueError("Nonempty original trial sequence and positive budget required")
    if len({(t["participant_id"], t["session_id"]) for t in trials}) != 1:
        raise ValueError("An episode must contain exactly one participant/session")
    if any(t.get("tasknumber") != 1 or t.get("label") not in (0, 1) for t in trials):
        raise ValueError("Episode sequence must contain only verified pure-LR original trials")
    uids = [t["trial_uid"] for t in trials]
    orders = [t["original_trial_index"] for t in trials]
    if len(set(uids)) != len(uids) or orders != sorted(set(orders)):
        raise ValueError("Duplicate UID or non-increasing original acquisition index")
    prefix = trials[:budget]
    runs = [t.get("physical_run_id") for t in trials]
    if any(r is None for r in runs) and not all(r is None for r in runs):
        raise ValueError("Partial physical run identity; schema audit required")
    discarded = []
    incomplete = []
    query = trials[budget:]
    if prefix and prefix[-1].get("physical_run_id") is not None:
        # Real runs must be contiguous; a repeated run ID cannot define a future
        # complete run. The tail of the budget-boundary run is never query.
        seen = set()
        previous = object()
        for r in runs:
            if r != previous:
                if r in seen:
                    raise ValueError("Physical run appears in discontiguous trial blocks")
                seen.add(r)
                previous = r
        boundary_run = prefix[-1]["physical_run_id"]
        discarded = [t for t in query if t["physical_run_id"] == boundary_run]
        query = [t for t in query if t["physical_run_id"] != boundary_run]
        complete_runs = {r for r in {t["physical_run_id"] for t in query}
                         if all(t.get("physical_run_complete_verified") is True for t in query if t["physical_run_id"] == r)}
        incomplete = [t for t in query if t["physical_run_id"] not in complete_runs]
        query = [t for t in query if t["physical_run_id"] in complete_runs]
    valid_support = [t for t in prefix if t.get("valid") is True]
    valid_query = [t for t in query if t.get("valid") is True]
    classes = Counter(t["label"] for t in valid_query)
    query_ok = len(valid_query) >= 20 and min(classes.get(0, 0), classes.get(1, 0)) >= 5
    support_ok = len(prefix) == budget and len({t["label"] for t in valid_support}) == 2 and len(valid_support) >= 2
    return {
        "budget": budget, "acquired_calibration_trials": len(prefix),
        "support_uids": [t["trial_uid"] for t in valid_support],
        "acquired_prefix_uids": [t["trial_uid"] for t in prefix],
        "invalid_prefix_trials": len(prefix) - len(valid_support),
        "unused_boundary_run_uids": [t["trial_uid"] for t in discarded],
        "incomplete_or_unverified_future_run_uids": [t["trial_uid"] for t in incomplete],
        "query_uids": [t["trial_uid"] for t in valid_query],
        "query_excluded_uids": [t["trial_uid"] for t in query if t.get("valid") is not True],
        "query_class_counts": dict(classes), "query_eligible": query_ok,
        "fallback_frozen": not support_ok,
        "fallback_reason": None if support_ok else "insufficient_acquired_prefix_or_valid_two_class_support",
        "query_rule": "future_complete_physical_runs" if any(r is not None for r in runs) else "verified_continuous_original_trial_block",
    }
