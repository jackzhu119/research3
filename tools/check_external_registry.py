#!/usr/bin/env python3
"""Validate a metadata-only external candidate registry offline; no raw reads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.external_registry import read_metadata, safe_relative, validate_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--registry", default="data_catalog/external_expansion_registry_20261019_v1.json")
    parser.add_argument("--output", help="explicit root-relative JSON report path; never an input/evidence file")
    parser.add_argument("--require-evaluation-ready", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        relative = safe_relative(args.registry)
        registry = json.loads(read_metadata(root, relative))
        report = validate_registry(registry, root)
        if not isinstance(registry, dict):
            registry = {}
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError):
        registry = {}
        report = {"registry_valid": False, "evaluation_ready": False, "network_actions": False,
                  "training_started": False, "errors": [{"field": "registry", "reason": "registry JSON is missing, unsafe, oversized, or malformed"}]}
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        try:
            target_relative = safe_relative(args.output)
            if not target_relative.startswith("audit/") or Path(target_relative).suffix != ".json":
                raise ValueError("report must be a JSON file in audit/")
            if target_relative == args.registry or any(
                item.get("path") == target_relative for item in registry.get("all_evidence", []) if isinstance(item, dict)
            ):
                raise ValueError("report may not overwrite registry or evidence")
            target = root / target_relative
            parent = target.parent
            while parent != root:
                if parent.is_symlink():
                    raise ValueError("report parent may not be a symlink")
                parent = parent.parent
            if target.exists() or target.is_symlink():
                raise ValueError("report path already exists; choose a new explicit output")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        except (ValueError, OSError):
            print(json.dumps({"registry_valid": False, "evaluation_ready": False,
                              "errors": [{"field": "output", "reason": "output must be a new safe audit JSON file, distinct from all inputs"}]}))
            return 2
    print(rendered, end="")
    return 0 if report["registry_valid"] and not args.require_evaluation_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
