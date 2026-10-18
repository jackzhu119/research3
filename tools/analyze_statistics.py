#!/usr/bin/env python3
"""Read-only W9 gate inspection; no real analysis, labels, training or downloads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.statistics import w9_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        result = w9_preflight(args.root)
    except Exception as exc:
        result = {"status": "BLOCKED_W9_NOT_EXECUTED", "error_type": type(exc).__name__,
                  "gap": "stage_or_freeze_metadata_validation_failed_details_suppressed",
                  "analysis_performed": False, "query_labels_read": False, "W9_completed": False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
