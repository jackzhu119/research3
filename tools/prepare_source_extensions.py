#!/usr/bin/env python3
"""Freeze, validate or execute one serial source-extension CPU preparation plan."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.source_extension_coordinator import PreparationError, atomic_json, build_plan, local, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", choices=["Yang2025", "Dreyer2023"])
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--max-provider-body-bytes", type=int)
    parser.add_argument("--max-r2-get-body-bytes", type=int)
    parser.add_argument("--max-wall-seconds", type=int, default=172800)
    parser.add_argument("--max-cache-bytes", type=int, default=4*1024**3)
    parser.add_argument("--disk-reserve-bytes", type=int, default=2*1024**3)
    parser.add_argument("--max-new-records", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root.absolute()
        plan_path = args.plan if args.plan.is_absolute() else root/args.plan
        local(root, plan_path.relative_to(root).as_posix())
        if plan_path.exists():
            plan = json.loads(plan_path.read_bytes())
        else:
            if not all((args.dataset, args.inventory, args.profile)):
                raise PreparationError("new_plan_requires_dataset_inventory_and_frozen_profile")
            plan = build_plan(root, dataset=args.dataset,
                inventory_path=args.inventory if args.inventory.is_absolute() else root/args.inventory,
                profile_path=args.profile if args.profile.is_absolute() else root/args.profile,
                max_provider_body_bytes=args.max_provider_body_bytes, max_r2_get_body_bytes=args.max_r2_get_body_bytes,
                max_wall_seconds=args.max_wall_seconds, max_cache_bytes=args.max_cache_bytes,
                disk_reserve_bytes=args.disk_reserve_bytes)
            atomic_json(plan_path, plan, immutable=True)
        result = prepare(root, plan, execute=args.execute, max_new_records=args.max_new_records)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
            "error_code": str(exc) if isinstance(exc, PreparationError) else "operation_failed_details_suppressed",
            "training_started": False, "GPU_actions": 0, "external_payload_accessed": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
