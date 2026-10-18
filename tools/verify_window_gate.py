#!/usr/bin/env python3
"""Read-only exact W1 acceptance, or a complete-duration diagnostic audit.

No downloads, raw EEG reads, training, writes or window changes. Diagnostic mode
never authorizes caches and requires the entire pinned meta-train batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.window_acceptance import WindowAcceptanceError, audit_duration_batch, verify_frozen_window
from p3work.splits import _hash


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
        # Per-person floating coverage is reproducible but unnecessary in CLI
        # output; preserve its identity and the exact rational summaries.
        proof = dict(proof)
        proof["floating_coverage_sha256"] = _hash(proof.pop("floating_coverage"))
        for candidate in proof["candidates"].values():
            candidate.pop("participants")
        proof["full_artifact_sha256"] = proof.pop("artifact_sha256")
        proof["artifact_sha256"] = _hash(proof)
        print(json.dumps(proof, allow_nan=False), flush=True)
        return 0 if proof.get("status") == "EXACT_FROZEN_WINDOW_VERIFIED" else 2
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
