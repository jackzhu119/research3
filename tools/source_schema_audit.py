#!/usr/bin/env python3
"""Audit complete downloaded four-source native specimens, without preprocessing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3work.adapters.source_native import audit_native_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=Path("audit/source_schema/raw_inventory.json"))
    parser.add_argument("--output", type=Path, default=Path("audit/source_schema"))
    parser.add_argument("--raw-root", type=Path,
                        help="Relocated raw specimen root; resolves dataset/file rather than old absolute path")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    results = []
    for item in inventory["files"]:
        native_path = args.raw_root / item["dataset"] / item["file"] if args.raw_root else Path(item["path"])
        result = audit_native_source(item["dataset"], native_path,
                                     provider_subject=item["provider_subject"],
                                     session_id=item["session_id"], source_version=item["version"],
                                     expected_sha256=item["sha256"])
        rows_path = args.output / (item["dataset"] + "_native_trials.jsonl")
        rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result["trials"]))
        summary = result["summary"]
        summary["native_trial_metadata_file"] = rows_path.name
        summary["native_trial_metadata_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
        results.append(summary)
        print(item["dataset"], summary["n_native_lr_trials"], summary["label_counts"], flush=True)
    output = {"schema_version": "paper3-source-native-audit-1", "role": "source_schema_only",
              "all_source_cohorts_complete": False, "window_frozen": False,
              "numerical_cache_created": False, "training_started": False, "datasets": results}
    (args.output / "native_schema_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
