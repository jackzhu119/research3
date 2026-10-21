#!/usr/bin/env python3
"""Profile-bound source-extension R2 backup/restore; default is no-network dry run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.source_extension_io import run_backup, run_restore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="operation", required=True)
    backup = modes.add_parser("backup", help="Explicit manifest; conditional PUT followed by full GET/SHA")
    backup.add_argument("--manifest", dest="manifest_path", type=Path, required=True)
    backup.add_argument("--prefix", required=True)
    backup.add_argument("--receipt", dest="receipt_path", type=Path, required=True)
    restore = modes.add_parser("restore", help="Exact profiled verified receipt; complete verified GETs")
    restore.add_argument("--receipt", dest="backup_receipt_path", type=Path, required=True)
    restore.add_argument("--report", dest="report_path", type=Path, required=True)
    selection = restore.add_mutually_exclusive_group(required=True)
    selection.add_argument("--paths", nargs="+")
    selection.add_argument("--all-approved", action="store_true")
    restore.add_argument("--require-fresh", action="store_true",
                         help="Reject existing selected files; require actual GET for each object")
    for mode in (backup, restore):
        mode.add_argument("--root", type=Path, required=True)
        mode.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    operation = args.pop("operation")
    try:
        result = (run_backup if operation == "backup" else run_restore)(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] in {"verified", "dry_run_verified_local_only",
            "restored_verified", "dry_run_validated_no_network"} else 1
    except Exception as error:
        # Even failures to load local dependencies must not print SDK messages.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
