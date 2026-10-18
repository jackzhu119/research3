#!/usr/bin/env python3
"""Plan W6 validation or explicitly run one gated local GPU evaluation job."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.development_evaluation import (DevelopmentJob, METHODS, build_development_plan,
    execute_validation_job, verify_development_inputs)
from p3work.contracts import file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--method", choices=METHODS, default="B0")
    parser.add_argument("--configuration-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--phase", choices=("screening", "confirmation"), default="screening")
    parser.add_argument("--artifact-catalog", help="Root-relative JSON artifact catalog with SHA-bound dependency descriptors")
    parser.add_argument("--budget", help="Root-relative explicit local budget JSON")
    parser.add_argument("--configuration-receipt", help="Root-relative frozen selection JSON")
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        job = DevelopmentJob(args.method, args.configuration_index, args.seed, args.phase)
        if not args.execute and not args.verify_inputs:
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
            if args.execute:
                result = execute_validation_job(root, job=job, artifact_catalog=catalog,
                    budget=descriptor(args.budget), configuration_receipt=selection,
                    user_started_gpu=args.user_started_gpu, output=args.output)
            else:
                _, _, inputs = verify_development_inputs(root, job=job, artifact_catalog=catalog,
                    configuration_receipt=selection)
                result = {"status": "INPUTS_VERIFIED_NOT_EXECUTED", "inputs": inputs,
                    "training_started": False, "CUDA_checked": False, "server_actions": False}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "DEVELOPMENT_EVALUATION_COMPLETE" else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__,
            "minimum_gap": "W1/W2/W3, verified same-seed artifacts, measured timing budget or explicit user-started CUDA gate",
            "training_started": False, "external_access": False, "server_actions": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
