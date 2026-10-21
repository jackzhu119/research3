#!/usr/bin/env python3
"""Restore exact historical test metadata from already staged fixed archives.

This entry is offline. It never changes scientific source, creates fit evidence,
opens numeric caches, downloads external query, or weakens an existing test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "paper3-hpc-historical-audit-inputs-v1"
PREFIXES = {"audit", "contracts", "protocol", "catalog", "data_catalog", "plans", "manifests", "provenance", "cache"}
CHUNK = 1024 * 1024


class RestoreError(ValueError):
    """Only a fixed diagnostic code is emitted, never a file's contents."""


def need(condition, code):
    if not condition:
        raise RestoreError(code)


def safe_root(value):
    path = Path(os.path.abspath(value))
    need(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_path_forbidden")
    return path


def checked_path(root, relative):
    need(isinstance(relative, str) and relative and "\\" not in relative
         and not any(ord(c) < 32 or ord(c) == 127 for c in relative), "unsafe_relative_path")
    parts = relative.split("/")
    path = PurePosixPath(relative)
    need(not path.is_absolute() and path.as_posix() == relative
         and all(part not in {"", ".", ".."} for part in parts), "unsafe_relative_path")
    target = safe_root(root)
    for part in parts:
        target /= part
        need(not target.is_symlink(), "symlink_path_forbidden")
    return target


def file_hash(path):
    path = safe_root(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode), "regular_file_required")
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
            count += len(block)
        after = os.fstat(stream.fileno())
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    need(identity(before) == identity(after) == identity(path.stat()) and count == after.st_size,
         "file_changed_while_hashing")
    return digest.hexdigest(), count


def load_manifest(filename):
    filename = safe_root(filename)
    with filename.open("rb") as stream:
        raw = stream.read(8 * CHUNK + 1)
    need(len(raw) <= 8 * CHUNK, "audit_manifest_too_large")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        raise RestoreError("audit_manifest_invalid_json") from None
    need(isinstance(value, dict) and value.get("schema_version") == SCHEMA,
         "historical_audit_manifest_required")
    top = value.get("top_level_directory")
    need(isinstance(top, str) and re.fullmatch(r"[A-Za-z0-9_-]+", top), "single_archive_top_directory_required")
    sources = value.get("sources")
    rows = value.get("files")
    need(isinstance(sources, dict) and isinstance(rows, list) and rows, "audit_manifest_empty")
    paths = set()
    for key, source in sources.items():
        need(isinstance(key, str) and isinstance(source, dict)
             and isinstance(source.get("filename"), str) and "/" not in source["filename"]
             and re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256")))
             and type(source.get("bytes")) is int and source["bytes"] > 0,
             "invalid_archive_identity")
    for row in rows:
        need(isinstance(row, dict) and set(row) == {"path", "sha256", "bytes", "archive"},
             "exact_audit_descriptor_required")
        relative = row["path"]
        checked_path(Path("/"), relative)
        parts = PurePosixPath(relative).parts
        need(parts[0] in PREFIXES and not any(p.lower().startswith(".env")
             or p.lower() in {".git", ".ssh", ".aws", "credentials", "secrets"} for p in parts),
             "audit_metadata_scope_required")
        need(Path(relative).suffix.lower() not in {".npy", ".npz", ".pt", ".mat", ".edf", ".bdf", ".h5", ".hdf5", ".whl", ".zip", ".gz"},
             "numeric_cache_or_raw_payload_forbidden")
        need(parts[0] != "cache" or Path(relative).suffix == ".json",
             "only_original_cache_metadata_allowed")
        need(relative not in paths and row["archive"] in sources
             and re.fullmatch(r"[a-f0-9]{64}", str(row["sha256"]))
             and type(row["bytes"]) is int and row["bytes"] >= 0,
             "invalid_or_duplicate_audit_descriptor")
        paths.add(relative)
    need(value.get("file_count") == len(rows) and value.get("bytes") == sum(r["bytes"] for r in rows),
         "audit_manifest_counters_mismatch")
    return value


def inspect(root, manifest):
    missing, exact = [], []
    for row in manifest["files"]:
        path = checked_path(root, row["path"])
        if not path.exists():
            missing.append(row)
        else:
            need(file_hash(path) == (row["sha256"], row["bytes"]), "existing_audit_SHA_mismatch_preserve_file")
            exact.append(row)
    return missing, exact


def _restore_archive(root, archive_path, top, rows):
    wanted = {top + "/" + row["path"]: row for row in rows}
    found, restored = set(), 0
    # Stream in original tar order; do not read or materialize any unselected
    # raw/cache payload. Gzip seeking per selected member would be quadratic.
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            need(member.name not in found, "duplicate_selected_archive_member")
            if member.name not in wanted:
                continue
            found.add(member.name)
            row = wanted[member.name]
            need(member.isfile() and member.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
                 and member.size == row["bytes"], "exact_regular_audit_member_required")
            target = checked_path(root, row["path"])
            if target.exists():
                need(file_hash(target) == (row["sha256"], row["bytes"]),
                     "concurrent_existing_audit_SHA_mismatch")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            checked_path(root, row["path"])
            descriptor, name = tempfile.mkstemp(prefix=".hpc-audit-", suffix=".partial", dir=target.parent)
            try:
                digest, size = hashlib.sha256(), 0
                with os.fdopen(descriptor, "wb") as output:
                    source = archive.extractfile(member)
                    need(source is not None, "archive_member_body_missing")
                    with source:
                        while block := source.read(CHUNK):
                            size += len(block)
                            need(size <= row["bytes"], "audit_member_length_mismatch")
                            digest.update(block)
                            output.write(block)
                    need((digest.hexdigest(), size) == (row["sha256"], row["bytes"]),
                         "audit_member_SHA_or_length_mismatch")
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(name, 0o644)
                os.link(name, target)
                restored += 1
            finally:
                Path(name).unlink(missing_ok=True)
    need(found == set(wanted), "selected_original_audit_member_missing")
    return restored


def restore(root, manifest_path, archive_dir, *, execute=False):
    root = safe_root(root)
    manifest = load_manifest(manifest_path)
    missing, exact = inspect(root, manifest)
    report = {"schema_version": "paper3-hpc-audit-restore-receipt-v1",
              "manifest_sha256": file_hash(manifest_path)[0], "expected_files": len(manifest["files"]),
              "preexisting_exact_files": len(exact), "missing_files": len(missing),
              "scientific_fits": 0, "raw_or_numeric_cache_restored": False,
              "external_query_accessed": False, "network_operations": 0,
              "restored_files": 0}
    if not missing:
        return dict(report, status="ORIGINAL_HISTORICAL_AUDIT_ALL_SHA256_VERIFIED",
                    verified_files=len(exact))
    report["missing_paths"] = [r["path"] for r in missing]
    if not execute:
        return dict(report, status="MISSING_ORIGINAL_HISTORICAL_AUDIT_STAGE_FIXED_ARCHIVE")
    archive_dir = safe_root(archive_dir)
    # Verify every archive needed by this exact selection before writing.
    sources = {row["archive"] for row in missing}
    for key in sources:
        source = manifest["sources"][key]
        path = checked_path(archive_dir, source["filename"])
        need(path.is_file(), "required_original_archive_missing")
        need(file_hash(path) == (source["sha256"], source["bytes"]), "original_archive_SHA_or_length_mismatch")
    for key in sorted(sources):
        source = manifest["sources"][key]
        report["restored_files"] += _restore_archive(root, archive_dir / source["filename"],
                manifest["top_level_directory"], [r for r in missing if r["archive"] == key])
    remaining, verified = inspect(root, manifest)
    need(not remaining and len(verified) == len(manifest["files"]), "restored_historical_audit_incomplete")
    report.pop("missing_paths", None)
    return dict(report, status="ORIGINAL_HISTORICAL_AUDIT_RESTORED_ALL_SHA256_VERIFIED",
                missing_files=0, verified_files=len(verified))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        archives = args.archive_dir or args.root / ".hpc/packages"
        if args.archive_dir is None and (archives / "archives").is_dir():
            archives /= "archives"
        report = restore(args.root, args.manifest or args.root / "hpc/audit_inputs.json",
                         archives, execute=args.execute)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["missing_files"] == 0 else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_PRESERVE_EXISTING_AUDIT_AND_CHECKPOINTS",
            "error": str(exc) if isinstance(exc, RestoreError) else type(exc).__name__,
            "scientific_fits": 0, "network_operations": 0}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
