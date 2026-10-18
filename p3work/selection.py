"""Predeclared development selection; no fit or final-query access."""
from __future__ import annotations

import math
from collections import defaultdict


def checkpoint_key(row):
    """Highest participant BA, then lower balanced BCE, then earlier update."""
    ba, bce, update = row["participant_mean_ba"], row["participant_mean_bce"], row["outer_update"]
    if (type(ba) not in (int, float) or type(bce) not in (int, float)
            or not math.isfinite(ba) or not math.isfinite(bce) or not 0 <= ba <= 1 or bce < 0
            or type(update) is not int or update < 1 or update > 1000 or update % 50):
        raise ValueError("Measured validation score at a registered checkpoint required")
    return (-ba, bce, update)


def choose_checkpoint(rows):
    rows = tuple(rows)
    if not rows or len({r["outer_update"] for r in rows}) != len(rows):
        raise ValueError("Unique completed validation checkpoints required")
    return min(rows, key=checkpoint_key)


def participant_equal_metrics(session_rows, *, expected_participants):
    """Average eligible sessions within each participant, then participants.

    No run weighting is introduced: each session score already pools its full
    eligible future-query trials and applies the protocol's class-balanced loss.
    Missing people fail instead of changing the pre-frozen denominator.
    """
    groups, sessions = defaultdict(list), set()
    for row in session_rows:
        key = (row["participant_id"], row["session_id"])
        if key in sessions:
            raise ValueError("Repeated validation session")
        sessions.add(key)
        for field in ("ba", "balanced_bce"):
            value = row[field]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or field == "ba" and value > 1:
                raise ValueError("Invalid validation session metric")
        groups[key[0]].append(row)
    expected = set(expected_participants)
    if set(groups) != expected or not expected:
        raise ValueError("Every frozen validation participant must remain accounted for")
    people = [{"participant_id": p, "sessions": len(groups[p]),
               "ba": sum(r["ba"] for r in groups[p]) / len(groups[p]),
               "balanced_bce": sum(r["balanced_bce"] for r in groups[p]) / len(groups[p])}
              for p in sorted(expected)]
    return {"participant_mean_ba": sum(p["ba"] for p in people) / len(people),
            "participant_mean_bce": sum(p["balanced_bce"] for p in people) / len(people),
            "participants": people, "aggregation": "participant_equal/session_equal/class_equal/trial_equal_within_class"}


def choose_strongest_control(rows):
    """B1..B11, three confirmation pipelines, then .005 BA/timing/ID tie rule.

    Each input is already participant-equal, with measured calibration timing
    from the same device. This function never accesses or chooses on external
    outcomes. Missing methods/replicates/timing cannot be silently dropped.
    """
    expected_seeds = {20261018, 20261019, 20261020}
    groups = defaultdict(list)
    device = set()
    for row in rows:
        if row.get("role") != "meta_validation" or row.get("measured") is not True:
            raise PermissionError("Measured development-validation evidence required")
        method = row.get("method")
        if method not in {f"B{i}" for i in range(1, 12)}:
            raise ValueError("B* candidates are B1 through B11")
        for name in ("participant_mean_ba", "calibration_median_seconds"):
            if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError("Measured BA and timing required")
        if row["participant_mean_ba"] > 1 or row["calibration_median_seconds"] <= 0 or not row.get("device_identity_sha256"):
            raise ValueError("Invalid BA/timing/device evidence")
        groups[method].append(row)
        device.add(row["device_identity_sha256"])
    if set(groups) != {f"B{i}" for i in range(1, 12)} or len(device) != 1:
        raise ValueError("All eleven controls and one common measured device required")
    candidates = []
    import statistics
    for method, values in groups.items():
        if len(values) != 3 or {r.get("seed") for r in values} != expected_seeds:
            raise ValueError("Three predeclared confirmation pipelines per control required")
        candidates.append({"method": method,
                           "participant_mean_ba": sum(r["participant_mean_ba"] for r in values) / 3,
                           "calibration_median_seconds": statistics.median(r["calibration_median_seconds"] for r in values)})
    best = max(r["participant_mean_ba"] for r in candidates)
    tied = [r for r in candidates if best - r["participant_mean_ba"] <= .005 + 1e-15]
    return min(tied, key=lambda r: (r["calibration_median_seconds"], int(r["method"][1:])))
