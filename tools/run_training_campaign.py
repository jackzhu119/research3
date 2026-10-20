#!/usr/bin/env python3
"""Compile or resume the local base4+Stieger W4--W6 campaign; never boots a server."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.training_campaign import compile_campaign, execute_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--user-started-gpu", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=43200,
                        help="Persistent cumulative mainline cap, default12h; increasing needs an audit note")
    parser.add_argument("--operational-amendment-note")
    parser.add_argument("--max-actions", type=int, help="Pause after this many stage dispatches; same jobs resume")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.execute:
            result = execute_campaign(args.root, user_started_gpu=args.user_started_gpu,
                max_wall_seconds=args.max_wall_seconds, amendment_note=args.operational_amendment_note,
                max_actions=args.max_actions)
        else:
            result = compile_campaign(args.root)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.require_ready and result.get("minimum_gaps"):
            return 1
        if args.execute and result["status"] != "MAINLINE_THROUGH_ONE_INTERNAL_CHECK_COMPLETE":
            return 3
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "BLOCKED_PRESERVE_EXISTING_JOBS", "error_type": type(error).__name__,
            "reason": str(error), "server_actions": False, "external_query_accessed": False,
            "completion_claimed": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
