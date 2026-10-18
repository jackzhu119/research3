#!/usr/bin/env python3
"""Read-only pilot gate. Never downloads, fits, starts a server, or calls RunPod."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.contracts import validate_signal_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    blockers = []
    task_path = root / "status/tasks.json"
    tasks = json.loads(task_path.read_text()) if task_path.is_file() else {}
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        blockers.append("W1 native metadata, participant split and common window gate incomplete")
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    try:
        validate_signal_contract(contract)
    except ValueError:
        blockers.append("Common signal contract is not frozen and validated")
    if tasks.get("W2", {}).get("status") != "COMPLETE":
        blockers.append("New four-source input-bound cache manifests are not complete")
    plan = json.loads((root / "plans/controlled_pilot.json").read_text())
    if not plan.get("execution_entrypoint"):
        blockers.append("Complete stage-gated source pilot runner is not implemented")
    if not blockers:
        from tools.run_controlled_pilot import verify_inputs
        try:
            verify_inputs(root)
        except (ValueError, FileNotFoundError, KeyError, TypeError):
            blockers.append("Actual source cache/code/event hashes or original-trial provenance failed read-only verification")
    print(json.dumps({"ready_for_gpu_training": not blockers,
                      "ready_for_controlled_source_pilot": not blockers,
                      "scientific_training_ready": False, "blockers": blockers,
                      "scope": "controlled source engineering pilot only; scientific fits are not authorized by this check",
                      "user_later_gpu_boot_required": True, "remote_actions": False,
                      "training_started": False}, indent=2))
    return 1 if args.require_ready and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
