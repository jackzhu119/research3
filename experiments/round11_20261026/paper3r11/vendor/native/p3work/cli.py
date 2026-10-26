"""Read-only preparation status. No implicit data fetch, training or remote actions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from .contracts import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "status", "pilot-plan"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "doctor":
        versions = {}
        for package in ("torch", "numpy", "scipy", "mne", "h5py", "boto3"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        result = {"dependency_versions": versions,
                  "protocol_sha256": file_digest(root / "protocol/EXPERIMENT_PROTOCOL.md"),
                  "remote_operations": False, "training_started": False}
    else:
        path = root / ("status/tasks.json" if args.command == "status" else "plans/controlled_pilot.json")
        result = json.loads(path.read_text())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
