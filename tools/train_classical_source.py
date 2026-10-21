#!/usr/bin/env python3
"""Explicit CPU source-classical fitting; defaults to a read-only gated plan.

Does not start servers, touch CUDA, download, upload, or access sealed datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.classical_training import prepare_classical_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", default="runs/source_classical_v2")
    parser.add_argument("--include-source-bank", action="store_true", help="Six additional B4/B5 all-source fits; never hidden in core ten")
    parser.add_argument("--budget", help="Explicit root-relative CPU job/attempt/wall/input-byte budget JSON")
    parser.add_argument("--max-new-fits", type=int, help="Pause at a fit checkpoint without altering the fixed plan")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_classical_sources(args.root, execute=args.execute, output=args.output,
            include_source_bank=args.include_source_bank, budget_path=args.budget, max_new_fits=args.max_new_fits)
    except (ValueError, FileNotFoundError, PermissionError, KeyError, TypeError, OSError) as error:
        started, attempts = False, 0
        output = Path(args.output)
        if not output.is_absolute() and ".." not in output.parts:
            progress = args.root / output / "progress.json"
            if progress.is_file() and not progress.is_symlink():
                try:
                    ledger = json.loads(progress.read_text())
                    started = ledger.get("training_started") is True
                    attempts = ledger.get("actual_fit_attempts_started", 0)
                except (ValueError, OSError):
                    started, attempts = None, None
        print(json.dumps({"status": "BLOCKED_OR_INTERRUPTED", "reason": str(error),
            "training_started": started, "actual_fit_attempts_started": attempts,
            "scientific_completion_claimed": False, "gpu_execution": False}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 3 if result["status"].startswith("PAUSED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
