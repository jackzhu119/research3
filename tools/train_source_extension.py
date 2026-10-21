#!/usr/bin/env python3
"""Independent source-enhanced fit; no remote boot, download, R2 or meta access.

Without --execute this reports missing prerequisites or verifies a source-only
plan. Real fitting requires later manual GPU boot, the actual original pilot,
a real baseline final-epoch proof, exact native caches and an explicit budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.source_extension_bridge import ARMS, SEEDS, ExtensionGateError
from p3work.source_extension_training import build_plan, execute_extension, pending_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=20261018)
    parser.add_argument("--extension-manifest")
    parser.add_argument("--extension-sha256")
    parser.add_argument("--baseline-receipt")
    parser.add_argument("--pilot-receipt", default="runs/controlled_source_pilot_seed20261018/engineering_receipt.json")
    parser.add_argument("--budget")
    parser.add_argument("--output")
    parser.add_argument("--max-new-updates", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    args = parser.parse_args(argv)
    entered = False
    try:
        if args.execute and (not args.user_started_gpu or not args.budget):
            raise ExtensionGateError("manual_GPU_flag_and_explicit_budget_required")
        supplied = (args.extension_manifest, args.extension_sha256, args.baseline_receipt)
        if not any(supplied) and not args.execute:
            result = pending_plan(args.arm, args.seed)
        elif not all(supplied):
            raise ExtensionGateError("manifest_SHA_and_real_baseline_receipt_required_together")
        elif not args.execute:
            _, plan = build_plan(args.root, arm=args.arm, seed=args.seed,
                extension_manifest=args.extension_manifest, extension_sha256=args.extension_sha256,
                baseline_receipt=args.baseline_receipt, output=args.output)
            result = {"status": "SOURCE_EXTENSION_INPUT_PLAN_VERIFIED_NOT_EXECUTED", "plan": plan,
                "actual_fits": 0, "training_started": False, "GPU_checked": False,
                "downstream_meta_ready": False}
        else:
            entered = True
            result = execute_extension(args.root, arm=args.arm, seed=args.seed,
                extension_manifest=args.extension_manifest, extension_sha256=args.extension_sha256,
                baseline_receipt=args.baseline_receipt, budget_path=args.budget,
                pilot_receipt=args.pilot_receipt, output=args.output,
                max_new_updates=args.max_new_updates, user_started_gpu=args.user_started_gpu)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 3 if result["status"] == "PAUSED_AT_COMMITTED_UPDATE" else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED_OR_INTERRUPTED_NOT_COMPLETE",
            "error_code": str(error) if isinstance(error, ExtensionGateError) else "extension_gate_or_runtime_failed_details_suppressed",
            "execution_attempted": entered,
            "training_started_this_invocation": None if entered else False,
            "primary_H1_changed": False, "downstream_meta_ready": False,
            "server_started": False, "network_operations": 0, "R2_operations": 0}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
