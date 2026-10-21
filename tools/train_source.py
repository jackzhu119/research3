#!/usr/bin/env python3
"""Formal local source EEGNet fits; read-only dry run unless both GPU flags exist.

Never starts a server, downloads data, reads external queries, or trains on CPU.
Classical source fits are separate pending jobs; no overall W4 completion claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from tools.run_controlled_pilot import verify_inputs
from p3work.source_training import (SourceTrainingError, build_training_plan, execute_plan, local, verify_engineering_pilot)


def cuda_runtime():
    if not torch.cuda.is_available():
        raise SourceTrainingError("User-started CUDA device is unavailable; no server will be booted")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") not in {":4096:8", ":16:8"}:
        raise SourceTrainingError("Deterministic CUBLAS configuration required")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()
    except (FileNotFoundError, subprocess.SubprocessError):
        driver = None
    return {"device": "cuda:0", "device_name": props.name, "device_uuid": str(getattr(props, "uuid", "not_exposed")),
        "device_total_memory": props.total_memory, "compute_capability": [props.major, props.minor],
        "driver_versions": driver, "torch": str(torch.__version__), "numpy": np.__version__, "python": sys.version,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "AMP": False, "TF32": False, "deterministic_algorithms": True,
        "cuda_rng_device_count": torch.cuda.device_count()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--profile", choices=("development", "confirmation"), default="development")
    parser.add_argument("--extend-confirmation-plan", action="store_true",
                        help="Reuse a verified completed development plan while adding only the two registered seeds")
    parser.add_argument("--output", default="runs/source_training_v2")
    parser.add_argument("--job", default=None)
    parser.add_argument("--max-new-updates", type=int)
    parser.add_argument("--pilot-receipt", default="runs/controlled_source_pilot_seed20261018/engineering_receipt.json")
    parser.add_argument("--max-wall-seconds", type=int, default=86400)
    parser.add_argument("--amend-operational-budget-seconds", type=int)
    parser.add_argument("--operational-amendment-note")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    args = parser.parse_args(argv)
    execution_entered = False
    try:
        if args.execute and not args.user_started_gpu:
            raise SourceTrainingError("Both --execute and --user-started-gpu are required")
        cache = verify_inputs(args.root)
        plan = build_training_plan(cache, profile=args.profile)
        if not args.execute:
            result = {"status": "SOURCE_PLAN_DRY_RUN", "plan_sha256": plan["plan_sha256"],
                "profile": args.profile, "planned_neural_fits": plan["planned_neural_fits"],
                "planned_core_classical_fits": 10, "planned_core_total_fits": plan["planned_core_total_fits"],
                "max_neural_optimizer_updates": plan["max_neural_optimizer_updates"],
                "actual_fits_started_this_invocation": 0, "gpu_checked": False,
                "training_started_this_invocation": False, "source_f0_receipt_created": False}
        else:
            verify_engineering_pilot(cache, args.pilot_receipt)
            runtime = cuda_runtime()
            evidence = verify_engineering_pilot(cache, args.pilot_receipt, runtime=runtime)
            output = local(cache.root, args.output, exists=False)
            execution_entered = True
            result = execute_plan(cache, plan, output, runtime=runtime, pilot_evidence=evidence,
                max_new_updates=args.max_new_updates, only_job=args.job, max_wall_seconds=args.max_wall_seconds,
                amend_budget_seconds=args.amend_operational_budget_seconds,
                operational_amendment_note=args.operational_amendment_note,
                extend_confirmation=args.extend_confirmation_plan)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in {"SOURCE_PLAN_DRY_RUN", "NEURAL_SOURCE_COMPLETE_CLASSICAL_PENDING"} else 3
    except (ValueError, FileNotFoundError, KeyError, TypeError, PermissionError, RuntimeError, FloatingPointError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "training_started_this_invocation": None if execution_entered else False,
                          "execution_attempted": execution_entered,
                          "server_started": False, "external_data_accessed": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
