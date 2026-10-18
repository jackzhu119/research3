"""Predeclared window choice from frozen Stieger meta-train duration metadata."""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from statistics import mean
from .splits import _hash


def choose_window(rows: list[dict], split: dict, *, expected_session_manifest: dict) -> dict:
    """Choose long/short with equal person weights; never use validation/query.

    Rows are all original pure-LR trials, before waveform QC. Available feedback
    duration must end at resultind, excluding the published post-result padding.
    The complete frozen meta-train roster must be represented with both classes.
    """
    if split.get("status") != "FROZEN_METADATA_ONLY":
        raise ValueError("Metadata-only participant split must be frozen first")
    allowed = set(split["partitions"]["meta_train"])
    if not rows or {r["participant_id"] for r in rows} != allowed:
        raise ValueError("Complete meta-train roster required; validation/holdout forbidden")
    manifest = dict(expected_session_manifest)
    manifest_sha = manifest.pop("manifest_sha256", None)
    if (manifest_sha != _hash(manifest) or manifest.get("split_sha256") != split["split_sha256"]
            or manifest.get("selection") != "all_official_sessions_for_frozen_meta_train"
            or not manifest.get("inventory_sha256")):
        raise ValueError("Integrity-checked complete official meta-train session manifest required")
    expected = {}
    for entry in manifest.get("sessions", []):
        key = (entry["participant_id"], entry["session_id"])
        if key in expected or entry["participant_id"] not in allowed:
            raise ValueError("Duplicate or unauthorized session manifest identity")
        if not all(entry.get(k) for k in ("source_record_id", "source_file_sha256", "trial_metadata_sha256", "adapter_audit_sha256", "original_lr_trials")):
            raise ValueError("Session manifest needs raw/metadata/audit SHA and exact original-LR trial count")
        expected[key] = entry
    actual = defaultdict(list)
    for row in rows:
        actual[(row["participant_id"], row["session_id"])].append(row)
    if set(actual) != set(expected):
        raise ValueError("All registered meta-train sessions required; partial-session coverage forbidden")
    for key, rr in actual.items():
        entry = expected[key]
        if len(rr) != entry["original_lr_trials"] or any(r.get("source_file_sha256") != entry["source_file_sha256"] or r.get("source_record_id") != entry["source_record_id"] for r in rr):
            raise ValueError("Session original-trial counts/raw SHA disagree with extracted audit")
    seen = set()
    for r in rows:
        if r.get("tasknumber") != 1 or r.get("label") not in (0, 1):
            raise ValueError("Only verified tasknumber==1 LR duration metadata is permitted")
        if r.get("feedback_end_verified") is not True or r.get("available_feedback_seconds") is None:
            raise ValueError("Feedback endpoint and actual duration must be verified")
        if not isfinite(r["available_feedback_seconds"]) or r["available_feedback_seconds"] < 0:
            raise ValueError("Nonfinite or negative feedback duration")
        if r["trial_uid"] in seen:
            raise ValueError("Duplicate original trial UID")
        seen.add(r["trial_uid"])
    candidates = {}
    for duration in (3.0, 2.0):
        people = {}
        groups = defaultdict(list)
        for r in rows:
            groups[(r["participant_id"], r["session_id"])].append(r)
        for person in sorted(allowed):
            pr = [r for r in rows if r["participant_id"] == person]
            by_class = {c: [r for r in pr if r["label"] == c] for c in (0, 1)}
            if any(not cr for cr in by_class.values()):
                raise ValueError(f"Cannot compute predeclared two-class exclusion gap for {person}")
            retained = lambda rr: sum(r["available_feedback_seconds"] + 1e-9 >= duration for r in rr)
            class_rates = {str(c): 1 - retained(cr) / len(cr) for c, cr in by_class.items()}
            people[person] = {
                "original_lr_trials": len(pr), "retained": retained(pr),
                "retention": retained(pr) / len(pr), "class_exclusion_rates": class_rates,
                "absolute_class_exclusion_gap": abs(class_rates["0"] - class_rates["1"]),
                "sessions": {session: {"n": len(rr), "retained": retained(rr), "retention": retained(rr) / len(rr)}
                             for (p, session), rr in sorted(groups.items()) if p == person},
            }
        retention = mean(p["retention"] for p in people.values())
        gap = mean(p["absolute_class_exclusion_gap"] for p in people.values())
        candidates[str(duration)] = {
            "context_seconds": [0, duration], "participants": people,
            "participant_mean_retention": retention,
            "participant_mean_absolute_class_exclusion_gap": gap,
            "passes": retention >= .80 and gap <= .10,
        }
    selected = "long" if candidates["3.0"]["passes"] else "short" if candidates["2.0"]["passes"] else None
    return {
        "status": "WINDOW_FROZEN" if selected else "BLOCKED_SHORT_WINDOW_COVERAGE",
        "selected": selected, "split_sha256": split["split_sha256"],
        "raw_metadata_manifest_sha256": manifest_sha,
        "official_inventory_sha256": manifest["inventory_sha256"],
        "denominator": "all original pure-LR trials before waveform QC; participant-equal aggregation",
        "thresholds": {"participant_mean_retention_min": .80, "participant_mean_absolute_class_exclusion_gap_max": .10},
        "candidates": candidates,
        "context_seconds": [0, 3 if selected == "long" else 2] if selected else None,
        "crop_seconds": [.5, 2.5] if selected == "long" else [.5, 1.5] if selected else None,
        "analysis_samples_128hz": 256 if selected == "long" else 128 if selected else None,
    }
