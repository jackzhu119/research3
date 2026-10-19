#!/usr/bin/env python3
"""Validate or derive CPU-only C3/Cz/C4 source sensitivity cache. No network/fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.car3_cache import Car3PreparationError, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true", help="Create new preparation-only cache; never trains or uploads")
    args = parser.parse_args()
    try:
        result = prepare(args.root, execute=args.execute)
    except (Car3PreparationError, OSError, ValueError, KeyError, TypeError) as error:
        code = str(error) if isinstance(error, Car3PreparationError) else "local_preparation_failed_details_suppressed"
        print(json.dumps({"status": "BLOCKED", "error_code": code, "training_started": False,
                          "GPU_actions": 0, "network_operations": 0}))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
