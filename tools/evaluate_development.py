#!/usr/bin/env python3
"""Plan W6 validation or explicitly run one gated local GPU evaluation job."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.development_evaluation import (DevelopmentJob, METHODS, build_development_plan,
    build_calibration_profile_plan, execute_calibration_profile, execute_validation_job,
    verify_calibration_profile_inputs, verify_development_inputs)
from p3work.contracts import file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--method", choices=METHODS, default="B0")
    parser.add_argument("--configuration-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--phase", choices=("screening", "confirmation"), default="screening")
    parser.add_argument("--profile", action="store_true", help="Calibration-only engineering profile; no query labels or scientific selection")
    parser.add_argument("--profile-method", choices=METHODS, action="append",
        help="Whole registered grids in registry order; omitted means all 118 settings")
    parser.add_argument("--artifact-catalog", help="Root-relative JSON artifact catalog with SHA-bound dependency descriptors")
    parser.add_argument("--budget", help="Root-relative explicit local budget JSON")
    parser.add_argument("--configuration-receipt", help="Root-relative frozen selection JSON")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    execution_attempted = False
    try:
        job = DevelopmentJob(args.method, args.configuration_index, args.seed, args.phase)
        if args.profile and (args.seed != 20261018 or args.phase != "screening"):
            raise ValueError("Calibration engineering profiles use only the fixed screening pipeline seed")
        if not args.execute and not args.verify_inputs and args.profile:
            result = {**build_calibration_profile_plan(args.profile_method), "status": "PLANNED_NOT_EXECUTED",
                "training_started": False, "stages_verified": False, "external_access": False, "server_actions": False}
        elif not args.execute and not args.verify_inputs:
            result = {**build_development_plan(args.root), "requested_job_sha256": job.identity,
                "requested_configuration": job.configuration, "stages_verified": False}
        else:
            # Validate paths with the same external/credential guard before read.
            from p3work.prediction_store import _local
            root = args.root.resolve(strict=True)
            def descriptor(relative):
                path = _local(root, relative)
                return {"path": relative, "sha256": file_digest(path)}
            catalog = descriptor(args.artifact_catalog)
            selection = descriptor(args.configuration_receipt) if args.configuration_receipt else None
            budget = descriptor(args.budget) if args.execute else None
            if args.execute and args.profile:
                execution_attempted = True
                result = execute_calibration_profile(root, artifact_catalog=catalog, budget=budget,
                    methods=args.profile_method, user_started_gpu=args.user_started_gpu, output=args.output)
            elif args.execute:
                execution_attempted = True
                result = execute_validation_job(root, job=job, artifact_catalog=catalog,
                    budget=budget, configuration_receipt=selection,
                    user_started_gpu=args.user_started_gpu, output=args.output)
            elif args.profile:
                result = verify_calibration_profile_inputs(root, artifact_catalog=catalog, methods=args.profile_method)
            else:
                _, _, inputs = verify_development_inputs(root, job=job, artifact_catalog=catalog,
                    configuration_receipt=selection)
                result = {"status": "INPUTS_VERIFIED_NOT_EXECUTED", "inputs": inputs,
                    "training_started": False, "CUDA_checked": False, "server_actions": False}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] in {"PLANNED_NOT_EXECUTED", "INPUTS_VERIFIED_NOT_EXECUTED",
            "PROFILE_INPUTS_VERIFIED_NOT_EXECUTED", "DEVELOPMENT_EVALUATION_COMPLETE",
            "DEVELOPMENT_CALIBRATION_PROFILE_COMPLETE"} else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
            "minimum_gap": "W1/W2/W3, verified same-seed artifacts, measured timing budget or explicit user-started CUDA gate",
            "training_started": None if execution_attempted else False,
            "execution_attempted": execution_attempted, "external_access": False, "server_actions": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
