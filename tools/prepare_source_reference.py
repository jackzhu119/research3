#!/usr/bin/env python3
"""CPU-only Rref preparation; default dry run, blocked without a real source f0."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.source_reference_io import prepare_source_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-receipt", required=True)
    parser.add_argument("--output", default="runs/source_reference_v2")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract = json.loads((args.root / "contracts/input_contract.json").read_text())
        result = prepare_source_reference(args.root, args.source_receipt, contract,
                                           execute=args.execute, output=args.output)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "training_started": False, "gpu_execution": False,
                          "external_query_accessed": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
