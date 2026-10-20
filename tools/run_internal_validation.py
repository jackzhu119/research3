#!/usr/bin/env python3
"""Plan/freeze W6 selection or execute its once-only user-started GPU check."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.internal_validation import (InternalValidationError, execute_internal, freeze_development,
    plan_internal, verify_internal_completion)
from p3work.prediction_store import _local
from p3work.contracts import file_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--freeze-spec", help="Metadata JSON: all configs, all 42 selected dev receipts, Bstar, artifact catalog")
    parser.add_argument("--write-freeze", help="New immutable runs/*.json output, selection metadata only")
    parser.add_argument("--freeze", help="Previously created immutable pre-internal freeze JSON")
    parser.add_argument("--budget", help="Explicit root-relative one-check budget JSON")
    parser.add_argument("--verify-completion", action="store_true", help="Read-only replay of an already sealed check; no new authority, fit or CUDA")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        def ref(relative):
            path = _local(root, relative)
            return {"path": relative, "sha256": file_digest(path)}
        if args.verify_completion:
            if args.execute or args.freeze_spec or args.write_freeze or args.budget or args.user_started_gpu:
                raise ValueError("Completed-check replay cannot start or alter any execution")
            result = verify_internal_completion(root, freeze=ref(args.freeze))
        elif args.freeze_spec:
            if args.freeze or args.budget or args.user_started_gpu:
                raise ValueError("Selection freeze and internal check are separate immutable operations")
            spec = json.loads(_local(root, args.freeze_spec).read_text())
            result = freeze_development(root, spec, output=args.write_freeze, execute=args.execute)
        elif args.execute:
            result = execute_internal(root, freeze=ref(args.freeze), budget=ref(args.budget),
                user_started_gpu=args.user_started_gpu)
        else:
            result = plan_internal(root, ref(args.freeze) if args.freeze else None)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_NO_INTERNAL_COMPLETION", "error_type": type(error).__name__,
            "error_code": str(error) if isinstance(error, InternalValidationError) else "required_artifact_or_execution_gate_missing",
            "minimum_gap": "complete real W3, all selected three-seed development artifacts/Bstar whole-package freeze, exact persistent claim, measured budget and later user-started GPU",
            "execution_requested": args.execute, "server_actions": False, "external_payloads_accessed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
