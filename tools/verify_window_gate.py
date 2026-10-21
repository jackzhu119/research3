#!/usr/bin/env python3
"""Read-only exact W1 acceptance, or a complete-duration diagnostic audit.

No downloads, raw EEG reads, training, writes or window changes. Diagnostic mode
never authorizes caches and requires the entire pinned meta-train batch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.window_acceptance import WindowAcceptanceError, audit_duration_batch, verify_frozen_window
from p3work.splits import _hash


def summarize_proof(proof):
    """Keep the verified proof identity while omitting per-person CLI payloads.

    v2.1 has one separately approved exact candidate. It must never be made to
    look like the original v2 long/short floating decision.
    """
    result = deepcopy(proof)
    schema = result.get("schema_version")
    if schema == "paper3-approved-window-acceptance-v2.1":
        if any(key in result for key in ("floating_coverage", "floating_coverage_sha256", "floating_selected", "candidates", "exact_selected")):
            raise ValueError("Mixed original and amended window proof schemas")
        candidate = result["candidate"]
        people = candidate.pop("participants")
        if not isinstance(people, dict) or not people or len(people) != result["participants"]:
            raise ValueError("Amendment participant summary mismatch")
        original, retained = 0, 0
        for person in people.values():
            n, kept = person["original_lr_trials"], person["retained"]
            if type(n) is not int or type(kept) is not int or not 0 <= kept <= n:
                raise ValueError("Amendment original-trial counts required")
            original, retained = original + n, retained + kept
        if original != result["original_lr_trials"]:
            raise ValueError("Amendment original-trial summary mismatch")
        result["retained_original_lr_trials"] = retained
        result["summary_window_version"] = "2.1-approved-single-candidate"
    elif schema == "paper3-exact-window-acceptance-v1":
        result["floating_coverage_sha256"] = _hash(result.pop("floating_coverage"))
        for candidate in result["candidates"].values():
            candidate.pop("participants")
    else:
        raise ValueError("Unknown window proof schema")
    result["full_artifact_sha256"] = result.pop("artifact_sha256")
    result["artifact_sha256"] = _hash(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--audit-duration-batch", action="store_true")
    parser.add_argument("--selection-plan", default="plans/stieger_meta_duration_selection.json")
    parser.add_argument("--split")
    parser.add_argument("--inventory", default="data_catalog/stieger2021/official_inventory.json")
    args = parser.parse_args(argv)
    try:
        if args.audit_duration_batch:
            proof = audit_duration_batch(args.root, selection_plan=args.selection_plan,
                split_path=args.split, inventory_path=args.inventory)
        else:
            if args.split or args.selection_plan != "plans/stieger_meta_duration_selection.json" or args.inventory != "data_catalog/stieger2021/official_inventory.json":
                raise WindowAcceptanceError("formal_gate_uses_canonical_frozen_evidence_only")
            proof = verify_frozen_window(args.root)
        proof = summarize_proof(proof)
        print(json.dumps(proof, allow_nan=False), flush=True)
        return 0 if proof.get("status") == "EXACT_FROZEN_WINDOW_VERIFIED" and proof.get("cache_authorized") is True else 2
    except WindowAcceptanceError as error:
        print(json.dumps({"status": "BLOCKED_EXACT_WINDOW_ACCEPTANCE", "reason": error.code,
            "cache_authorized": False, "details": error.details}), flush=True)
        return 2
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        print(json.dumps({"status": "BLOCKED_EXACT_WINDOW_ACCEPTANCE", "reason": "window_evidence_invalid",
            "cache_authorized": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
