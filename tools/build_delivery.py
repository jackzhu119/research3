#!/usr/bin/env python3
"""Build an immutable preparation bundle and back it up with full GET/SHA256.

Default is local planning only. --execute requires actual W1/W2 and source-cache
gates; --checkpoint labels an intermediate snapshot explicitly NOT pilot ready.
No GPU work, server operations, credentials, raw EEG or full wheelhouse in tar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.r2_backup import run_backup, sha256_file, write_receipt
from tools.r2_restore import load_verified_receipt, no_symlink_components

PRIVATE_TREES = ("audit", "data_catalog", "manifests", "contracts", "status", "plans", "provenance")
BOOTSTRAP_NAMES = ("boto3-", "botocore-", "jmespath-", "s3transfer-", "python_dateutil-", "six-", "urllib3-")


def safe_member(relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return False
    if any(part.lower() in {".git", "__pycache__", ".aws", ".ssh", "secrets", "credentials"}
           or part.lower().startswith(".env") or Path(part).stem.lower() in {"secrets", "credentials", "credential", "private_key"}
           for part in p.parts):
        return False
    if p.suffix.lower() in {".mat", ".edf", ".bdf", ".fif", ".npz", ".h5", ".hdf5", ".dat",
                            ".p12", ".pfx", ".der", ".crt", ".csr", ".keystore", ".pt", ".pth"}:
        return False
    if p.suffix.lower() in {".zip", ".gz", ".tar", ".whl"}:
        return p.as_posix() == "archives/Paper3_Work_Handoff_20261018.zip"
    if p.suffix.lower() == ".npy" and p.parts[:2] != ("manifests", "geometry_constants"):
        return False
    if p.suffix.lower() == ".lock" and p.name in {"requirements-cpu.lock", "requirements-gpu.lock"}:
        return True
    return p.suffix.lower() not in {".pem", ".key", ".pyc", ".partial", ".tmp", ".lock"}


def selected_files(root):
    """Only code-tracked files and specifically authorized evidence trees."""
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    selected = {name for name in names if name and safe_member(name)}
    # The protocol must be present even in an uncommitted checkpoint. Recheck
    # the actual imported bytes against the original 52-entry package manifest.
    protocol_manifest = root / "protocol/PACKAGE_CONTENTS.sha256"
    lines = protocol_manifest.read_text().splitlines()
    protocol_entries = []
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*")
        name = "protocol/" + relative
        if not safe_member(name) or len(digest) != 64:
            raise ValueError("Unsafe imported protocol entry")
        path = root / name
        no_symlink_components(path)
        if sha256_file(path)[0] != digest:
            raise ValueError("Imported protocol bytes changed")
        protocol_entries.append(name)
    if len(protocol_entries) != 52 or len(set(protocol_entries)) != 52:
        raise ValueError("All 52 original protocol entries required")
    selected.update(protocol_entries)
    selected.add("protocol/PACKAGE_CONTENTS.sha256")
    for tree in PRIVATE_TREES:
        for path in (root / tree).rglob("*"):
            if (path.is_file() and safe_member(path.relative_to(root)) and
                    path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".log", ".sha256", ".pdf", ".html", ".tsv", ".npy"}):
                selected.add(path.relative_to(root).as_posix())
    # Tools may still be uncommitted when building a clearly marked checkpoint.
    for tree in ("tools", "p3work", "tests"):
        for path in (root / tree).rglob("*.py"):
            if safe_member(path.relative_to(root)):
                selected.add(path.relative_to(root).as_posix())
    for name in ("requirements-cpu.lock", "requirements-gpu.lock", "README.md", "pyproject.toml"):
        if (root / name).is_file():
            selected.add(name)
    entries = {}
    for name in sorted(selected):
        path = root / name
        no_symlink_components(path)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Unsafe delivery artifact")
        # Raw EEG and cache/wheel payloads are restored from explicit R2 receipts.
        if Path(name).parts[0] in {"data", "cache", "runs", "checkpoints"}:
            raise ValueError("Large data/weights must use explicit R2 manifests")
        entries[name] = path.read_bytes()
    dependency_rows, _ = load_verified_receipt(root / "audit/r2_dependencies_receipt.json")
    backed = {row["path"]: row for row in dependency_rows}
    bootstrap = sorted((root / "data/dependency_wheels/cpu").glob("*.whl"))
    for prefix in BOOTSTRAP_NAMES:
        matches = [p for p in bootstrap if p.name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError("Complete pinned offline R2 bootstrap wheels required")
        path = matches[0]
        no_symlink_components(path)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Bootstrap wheel outside root")
        digest, size = sha256_file(path)
        row = backed.get(path.relative_to(root).as_posix())
        if row is None or row["sha256"] != digest or row["bytes"] != size:
            raise ValueError("Bootstrap wheel differs from the actual verified R2 backup")
        entries["bootstrap_wheels/" + path.name] = path.read_bytes()
    return entries


def create_archive(destination, entries):
    manifest = "".join(hashlib.sha256(data).hexdigest() + "  " + name + "\n"
                       for name, data in sorted(entries.items()))
    contents = {**entries, "EXECUTION_MANIFEST.sha256": manifest.encode()}
    with destination.open("wb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(contents.items()):
                    info = tarfile.TarInfo("paper3/" + name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    archive.addfile(info, io.BytesIO(data))
    with tarfile.open(destination, "r:gz") as archive:
        for name, data in entries.items():
            if hashlib.sha256(archive.extractfile("paper3/" + name).read()).digest() != hashlib.sha256(data).digest():
                raise ValueError("Delivery tar member integrity failed")
    return len(contents)


def validate_cache_backup(root, cache, receipt_path):
    """Bind the full-GET receipt to every artifact the restored pilot needs."""
    rows, _ = load_verified_receipt(receipt_path)
    backed = {row["path"]: row for row in rows}
    manifest = json.loads((root / cache.manifest_path).read_text())
    required = [{"path": cache.manifest_path, "sha256": cache.manifest_sha256},
                manifest["raw_inventory"]]
    for shard in manifest["shards"]:
        required.extend(shard[key] for key in ("X", "y", "events", "exclusions"))
    for person in manifest.get("participants_without_usable_trials", []):
        required.append(person["exclusions"])
    required_paths = {item["path"] for item in required}
    for row in rows:
        relative = row["path"]
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or row["role"] == "source_development_raw"
                or (relative not in required_paths and
                    (not safe_member(relative) or row["role"] not in {"audit", "manifest", "catalog", "protocol"}
                     or Path(relative).suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".html", ".pdf", ".sha256", ".tsv"}))):
            raise ValueError("Pilot receipt may restore only required caches and explicit metadata")
        path = root / relative
        no_symlink_components(path)
        digest, size = sha256_file(path)
        if row["sha256"] != digest or row["bytes"] != size:
            raise ValueError("Restore metadata differs from the packaged current evidence")
    for item in required:
        relative = item["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Cache backup path must be root relative")
        path = root / relative
        no_symlink_components(path)
        digest, size = sha256_file(path)
        row = backed.get(relative)
        if digest != item["sha256"] or row is None or row["sha256"] != digest or row["bytes"] != size:
            raise ValueError("Full-GET receipt does not cover the actual complete source cache")


def validate_dependency_backups(root):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name, parse_wheel_filename
    wheel_versions = set()
    for rel in ("audit/r2_dependencies_receipt.json", "audit/r2_dependency_extras_receipt.json"):
        rows, _ = load_verified_receipt(root / rel)
        for row in rows:
            if Path(row["path"]).suffix == ".whl":
                name, version, _, _ = parse_wheel_filename(Path(row["path"]).name)
                wheel_versions.add((canonicalize_name(name), str(version)))
    for filename in ("requirements-cpu.lock", "requirements-gpu.lock"):
        for line in (root / filename).read_text().splitlines():
            if not line.strip() or line.startswith(("#", "--")):
                continue
            requirement = Requirement(line)
            pins = list(requirement.specifier)
            if len(pins) != 1 or pins[0].operator != "==" or (canonicalize_name(requirement.name), pins[0].version) not in wheel_versions:
                raise ValueError("Every exact CPU/GPU dependency must have an actual full-GET wheel receipt")


def startup_text(*, checkpoint, cache_receipt):
    state = "CHECKPOINT: W1/W2 pending; no GPU pilot is permitted." if checkpoint else "CPU preparation complete; GPU pilot waits for the user's later manual boot."
    cache_command = (f"python tools/r2_restore.py --receipt {cache_receipt} --root . --report audit/restored_source_cache.json --all-approved --execute"
                     if cache_receipt else "# Source cache unavailable: checkpoint bundle is not pilot ready.")
    return f"""# Paper3 fixed execution bundle\n\n{state}\n\nThis bundle contains no credentials and never starts or restarts RunPod. Use CPython 3.12, Linux x86_64, and a driver compatible with the pinned CUDA12.8 profile after user boot. GPU compatibility and timings remain unmeasured until that execution.\n\nFrom the extracted paper3 directory, in a fresh virtual environment:\n\n```bash\nsha256sum -c EXECUTION_MANIFEST.sha256\npython -m pip install --no-index --find-links bootstrap_wheels boto3==1.43.114\npython tools/r2_restore.py --receipt audit/r2_dependencies_receipt.json --root . --report audit/restored_dependencies.json --all-approved --execute\npython tools/r2_restore.py --receipt audit/r2_dependency_extras_receipt.json --root . --report audit/restored_dependency_extras.json --all-approved --execute\npython -m pip install --no-index --find-links data/dependency_wheels/cpu --find-links data/dependency_wheels/gpu -r requirements-gpu.lock\n{cache_command}\npython -m p3work doctor\npython tools/check_pilot_readiness.py --require-ready\n```\n\nR2 tools use only existing runtime variables. Do not copy .env files, keys or Paper1/2 upload settings. No public URLs are generated.\n\nThe source-cache receipt is `{cache_receipt or 'NOT_YET_AVAILABLE'}`. Restore only that cache receipt for a source timing pilot. The bundle includes the separate original/raw/dependency backup receipts and manifests; restoring raw EEG is unnecessary for this pilot.\n\nOnly after W1/W2, all hashes, cache restore and the user boot are verified, the manual command is:\n\n```bash\npython tools/run_controlled_pilot.py --root . --execute --user-started-gpu\n```\n\nDo not run that command for a checkpoint bundle. It is blocked by the actual stage/data gates. The pilot is source engineering only: 100 continuous versus 50+checkpoint+50, total200 updates; it is not a scientific f0 fit or a meta/clinical result. No automatic GPU command is executed by this package.\n"""


def build(root, *, execute=False, checkpoint=False, progress_path=None):
    root = root.resolve(strict=True)
    result = {"operation": "paper3_delivery_build", "status": "PLANNED", "execute": execute,
              "checkpoint": checkpoint, "training_started": False, "server_actions": False,
              "credentials_included": False}
    if not execute:
        return result
    if not checkpoint:
        from tools.run_controlled_pilot import verify_inputs
        cache = verify_inputs(root)
        result["verified_source_cache_sha256"] = cache.manifest_sha256
    tasks = json.loads((root / "status/tasks.json").read_text())
    cache_receipt = tasks.get("W2", {}).get("cache_r2_receipt")
    for rel in ("audit/r2_original_archive_receipt.json", "audit/r2_schema_raw_receipt.json", "audit/r2_dependencies_receipt.json"):
        load_verified_receipt(root / rel)
        receipt = json.loads((root / rel).read_text())
        if receipt.get("status") != "verified" or receipt.get("execute") is not True:
            raise ValueError("Existing prerequisite backups must have actual verified receipts")
    validate_dependency_backups(root)
    if not checkpoint:
        validate_cache_backup(root, cache, root / cache_receipt)
        plan_path = root / "plans/controlled_pilot.json"
        plan = json.loads(plan_path.read_text())
        plan.update(status="CPU_PREPARED_WAITING_USER_GPU_BOOT", auto_start=False,
                    training_started=False, gpu_pilot_executed=False,
                    input_contract_sha256=cache.input_contract_sha256,
                    cache_manifest_sha256=cache.manifest_sha256,
                    current_allowed_gpu_training_hours=0)
        write_receipt(plan_path, plan)
    result.update(status="PACKAGING", recorded_at_utc=datetime.now(timezone.utc).isoformat())
    if progress_path:
        write_receipt(progress_path, result)
    entries = selected_files(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    build_info = {**result, "git_commit": commit, "file_manifest_is_authoritative_for_actual_worktree": True,
                  "ready_for_user_boot_pilot": not checkpoint, "scientific_fit_completed": False,
                  "external_query_opened": False, "stage_status": tasks}
    entries["BUILD.json"] = (json.dumps(build_info, indent=2) + "\n").encode()
    entries["STARTUP.md"] = startup_text(checkpoint=checkpoint, cache_receipt=cache_receipt).encode()
    output_dir = root / "dist"
    output_dir.mkdir(exist_ok=True)
    label = "checkpoint" if checkpoint else "controlled-pilot-ready"
    temporary = output_dir / (label + ".tar.gz.partial")
    count = create_archive(temporary, entries)
    digest, size = sha256_file(temporary)
    archive = output_dir / f"paper3-{label}-{digest[:16]}.tar.gz"
    temporary.replace(archive)
    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar.write_text(digest + "  " + archive.name + "\n")
    manifest_path = output_dir / (archive.stem + ".backup-manifest.json")
    rows = []
    for path in (archive, sidecar):
        sha, length = sha256_file(path)
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": length,
                     "role": "execution_package", "dataset": "Paper3Delivery"})
    manifest_path.write_text(json.dumps({"schema_version": 1, "files": rows}, indent=2) + "\n")
    receipt_path = root / "audit" / f"r2_delivery_{digest[:16]}_receipt.json"
    receipt = run_backup(manifest_path=manifest_path, root=root,
                         prefix=f"paper3/deliveries/{digest}/", receipt_path=receipt_path, execute=True)
    if receipt.get("status") != "verified":
        raise ValueError("Delivery package R2 full GET/SHA256 not verified")
    # Receipt is a separate, finite second layer; the tar does not pretend to
    # contain evidence of its own future upload.
    receipt_sha, receipt_size = sha256_file(receipt_path)
    receipt_manifest = output_dir / (archive.stem + ".receipt-manifest.json")
    receipt_manifest.write_text(json.dumps({"schema_version": 1, "files": [{
        "path": receipt_path.relative_to(root).as_posix(), "sha256": receipt_sha,
        "bytes": receipt_size, "role": "audit", "dataset": "Paper3Delivery"}]}, indent=2) + "\n")
    receipt_backup_path = root / "audit" / f"r2_delivery_{digest[:16]}_receipt_backup.json"
    receipt_backup = run_backup(manifest_path=receipt_manifest, root=root,
                                prefix=f"paper3/deliveries/{digest}/receipts/",
                                receipt_path=receipt_backup_path, execute=True)
    if receipt_backup.get("status") != "verified":
        raise ValueError("Delivery receipt backup not GET verified")
    shared = Path("/workspace/shared")
    shared.mkdir(exist_ok=True)
    import shutil
    shutil.copyfile(archive, shared / archive.name)
    shutil.copyfile(sidecar, shared / sidecar.name)
    result.update(status="CHECKPOINT_BACKED_UP" if checkpoint else "READY_WAITING_USER_GPU_BOOT",
                  path=str(shared / archive.name), bytes=size, sha256=digest,
                  members_verified=count, backup_receipt=str(receipt_path),
                  receipt_backup=str(receipt_backup_path), ready_for_user_boot_pilot=not checkpoint)
    result.update(package_key=f"paper3/deliveries/{digest}/{archive.relative_to(root).as_posix()}",
                  receipt_key=f"paper3/deliveries/{digest}/receipts/{receipt_path.relative_to(root).as_posix()}",
                  receipt_sha256=receipt_sha)
    write_receipt(root / "status/delivery.json", result)
    if progress_path:
        write_receipt(progress_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.root, execute=args.execute, checkpoint=args.checkpoint, progress_path=args.progress)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        result = {"status": "BLOCKED", "error_type": type(exc).__name__,
                  "minimum_gap": "Actual stage/cache/backup gate or local artifact validation failed; no GPU action performed"}
        if args.progress:
            write_receipt(args.progress, result)
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
