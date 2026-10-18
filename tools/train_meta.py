#!/usr/bin/env python3
"""Plan registered meta jobs; execution is explicit, stage-gated and local-only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.meta_training import MetaJob, execute_meta_job, search_plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--method", choices=("P", "P+PCT", "B8", "B10", "B11"), default="P")
    p.add_argument("--dimension", type=int, default=2)
    p.add_argument("--radius", type=float, default=.15)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--phase", choices=("engineering", "screening", "confirmation"), default="screening")
    p.add_argument("--source-receipt")
    p.add_argument("--reference-receipt")
    p.add_argument("--split")
    p.add_argument("--budget")
    p.add_argument("--selection-receipt")
    p.add_argument("--resume")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--user-started-gpu", action="store_true")
    a = p.parse_args(argv)
    try:
        job = MetaJob(a.method, a.dimension, a.radius, a.seed, a.phase)
        if not a.execute:
            result = {**search_plan(a.method), "requested_job": asdict(job),
                      "status": "PLANNED_NOT_EXECUTED", "stages_verified": False,
                      "user_later_gpu_boot_required": True, "server_actions": False}
        else:
            if not a.user_started_gpu or not all((a.source_receipt, a.reference_receipt, a.split, a.budget)):
                raise ValueError("Later user GPU boot plus source/reference/split/budget artifacts required")
            result = execute_meta_job(a.root, job=job, source_receipt=a.source_receipt,
                reference_receipt=a.reference_receipt, split_path=a.split, budget_path=a.budget,
                user_started_gpu=a.user_started_gpu, resume=a.resume, selection_receipt=a.selection_receipt)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] in {"PLANNED_NOT_EXECUTED", "META_ENGINEERING_COMPLETE", "META_JOB_COMPLETE"} else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
                          "minimum_gap": "Frozen real-data/source/reference, measured budget, or explicit GPU gate failed",
                          "server_actions": False, "external_access": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
