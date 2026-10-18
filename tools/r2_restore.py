#!/usr/bin/env python3
"""Restore explicitly selected approved objects from a verified Paper3 receipt.

Default: validate the backup receipt and local destinations without network I/O.
--execute streams complete GET bodies to unique *.partial files. A file becomes
complete only after byte-count/SHA256 checks and an atomic no-overwrite commit.
An existing identical local file is verified locally and does not need a GET.
Failed partial files remain for inspection; partial bytes are never treated as
complete files. No remote objects or original backup receipts are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

try:
    import r2_backup as backup
except ModuleNotFoundError:
    from tools import r2_backup as backup


ENTRY_FIELDS = ("path", "sha256", "bytes", "role", "dataset")


def load_verified_receipt(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if (not isinstance(receipt, dict) or receipt.get("schema_version") != 1
            or receipt.get("operation") != "paper3_explicit_manifest_backup"
            or receipt.get("status") != "verified" or receipt.get("execute") is not True
            or receipt.get("verification") != "full_get_body_sha256"
            or not isinstance(receipt.get("objects"), list) or not receipt["objects"]):
        raise backup.BackupError("backup_receipt_not_verified")
    prefix = receipt.get("prefix")
    if not isinstance(prefix, str):
        raise backup.BackupError("invalid_isolated_paper3_prefix")
    backup.validate_prefix(prefix)
    rows = []
    seen_paths = set()
    for source in receipt["objects"]:
        if not isinstance(source, dict) or not all(field in source for field in ENTRY_FIELDS):
            raise backup.BackupError("invalid_backup_object")
        entry = {field: source[field] for field in ENTRY_FIELDS}
        backup.validate_entry(entry)
        if entry["path"] in seen_paths:
            raise backup.BackupError("duplicate_backup_path")
        seen_paths.add(entry["path"])
        key = source.get("key")
        put = source.get("put", {})
        get = source.get("get", {})
        if (key != prefix + entry["path"] or source.get("status") != "verified"
                or not isinstance(put, dict) or not isinstance(get, dict)
                or put.get("status") not in {"created", "already_exists_not_overwritten"}
                or put.get("http_status") != {
                    "created": 200, "already_exists_not_overwritten": 412,
                }.get(put.get("status"))
                or type(put.get("request_body_bytes")) is not int
                or put.get("request_body_bytes") != entry["bytes"]
                or put.get("request_body_sha256") != entry["sha256"]
                or get.get("status") != "verified" or get.get("http_status") != 200
                or get.get("complete_body_read") is not True
                or type(get.get("bytes_read")) is not int
                or get["bytes_read"] != entry["bytes"]
                or get.get("sha256") != entry["sha256"]
                or (get.get("declared_content_length") is not None
                    and get["declared_content_length"] != entry["bytes"])):
            raise backup.BackupError("backup_object_not_fully_verified")
        rows.append({**entry, "key": key})
    return rows, hashlib.sha256(raw).hexdigest()


def no_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise backup.BackupError("symlink_forbidden")


def destination(root: Path, entry: dict) -> Path:
    parts = backup.validate_entry({field: entry[field] for field in ENTRY_FIELDS})
    root = Path(os.path.abspath(root))
    no_symlink_components(root)
    target = root.joinpath(*parts)
    no_symlink_components(target)
    if not target.resolve().is_relative_to(root.resolve()):
        raise backup.BackupError("restore_path_outside_root")
    if target.exists() and not target.is_file():
        raise backup.BackupError("restore_destination_not_file")
    return target


def existing_file_state(target: Path, entry: dict) -> bool:
    if not target.exists():
        return False
    digest, size = backup.sha256_file(target)
    if digest != entry["sha256"] or size != entry["bytes"]:
        raise backup.BackupError("different_local_file_not_overwritten")
    return True


def restore_one(client: Any, bucket: str, entry: dict, root: Path) -> dict:
    result = {**entry, "status": "failed", "started_at_utc": backup.utc_now(),
              "bytes_read": 0, "complete_body_read": False}
    body = None
    partial = None
    digest = hashlib.sha256()
    try:
        target = destination(root, entry)
        if existing_file_state(target, entry):
            result.update(status="already_present_verified", get_attempted=False,
                          local_bytes=entry["bytes"], local_sha256=entry["sha256"])
            return result
        target.parent.mkdir(parents=True, exist_ok=True)
        destination(root, entry)  # Recheck newly created directories before writing.
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=target.name + ".", suffix=".partial",
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            result["partial_path"] = partial.relative_to(root).as_posix()
            result["get_attempted"] = True
            response = client.get_object(Bucket=bucket, Key=entry["key"])
            result["http_status"] = backup.http_status(response)
            body = response["Body"]
            while True:
                chunk = body.read(backup.CHUNK_SIZE)
                if not isinstance(chunk, bytes):
                    raise backup.BackupError("non_binary_get_body")
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                result["bytes_read"] += len(chunk)
            result["complete_body_read"] = True
            result["get_sha256"] = digest.hexdigest()
            length = response.get("ContentLength")
            result["declared_content_length"] = length if type(length) is int else None
            stream.flush()
            os.fsync(stream.fileno())
        # Close the network body before committing: a close/read failure must not
        # produce a completed destination file.
        body.close()
        body = None
        if result["http_status"] != 200:
            raise backup.BackupError("get_http_status_not_200")
        if (result["bytes_read"] != entry["bytes"]
                or result["get_sha256"] != entry["sha256"]
                or (length is not None and length != result["bytes_read"])):
            raise backup.BackupError("remote_integrity_mismatch")
        destination(root, entry)
        try:
            # link() atomically publishes a verified inode and never overwrites a
            # target that appeared during the download. Both paths share a dir.
            os.link(partial, target)
        except FileExistsError:
            destination(root, entry)
            existing_file_state(target, entry)
        partial.unlink()
        result.update(status="restored_verified", atomic_commit="link_no_overwrite")
        result.pop("partial_path", None)
    except Exception as exc:
        result.update(backup.safe_error(exc))
        if not result["complete_body_read"]:
            result["partial_sha256"] = digest.hexdigest()
    finally:
        if body is not None:
            try:
                body.close()
            except Exception as exc:
                result.update(status="failed", **backup.safe_error(exc))
        result["finished_at_utc"] = backup.utc_now()
    return result


def run_restore(*, backup_receipt_path: Path, root: Path, report_path: Path,
                paths: list[str] | None = None, all_approved: bool = False,
                execute: bool = False, client: Any = None,
                bucket: str | None = None) -> dict:
    report: dict[str, Any] = {
        "schema_version": 1, "operation": "paper3_verified_receipt_restore",
        "started_at_utc": backup.utc_now(), "execute": execute, "status": "failed",
        "credentials_recorded": False, "objects": [],
    }
    may_write_report = True
    try:
        if (report_path.resolve() == backup_receipt_path.resolve()
                or (report_path.exists() and backup_receipt_path.exists()
                    and report_path.samefile(backup_receipt_path))):
            may_write_report = False
            raise backup.BackupError("backup_receipt_cannot_be_modified")
        if bool(paths) == bool(all_approved):
            raise backup.BackupError("explicit_restore_selection_required")
        entries, source_sha = load_verified_receipt(backup_receipt_path)
        by_path = {entry["path"]: entry for entry in entries}
        if paths and (len(paths) != len(set(paths)) or any(path not in by_path for path in paths)):
            raise backup.BackupError("restore_selection_not_in_verified_receipt")
        selected = entries if all_approved else [by_path[path] for path in paths]
        root = Path(os.path.abspath(root))
        for entry in selected:
            target = destination(root, entry)
            if report_path.resolve() == target.resolve():
                may_write_report = False
                raise backup.BackupError("report_cannot_overwrite_restore_target")
            existing_file_state(target, entry)
        report.update(backup_receipt_sha256=source_sha, selected_objects=len(selected))
        if not execute:
            report["objects"] = [
                {**entry, "status": "planned_no_network"} for entry in selected
            ]
            report["status"] = "dry_run_validated_no_network"
        else:
            if client is None:
                if not all(os.environ.get(name) for name in backup.REQUIRED_VARIABLES):
                    raise backup.BackupError("missing_existing_r2_environment")
                client = backup.create_client()
            bucket = bucket if bucket is not None else os.environ.get("R2_BUCKET")
            if not bucket:
                raise backup.BackupError("missing_existing_r2_bucket")
            for entry in selected:
                row = restore_one(client, bucket, entry, root)
                report["objects"].append(row)
                backup.write_receipt(report_path, report)
                if row["status"] not in {"restored_verified", "already_present_verified"}:
                    raise backup.BackupError("restore_object_not_verified")
            report["status"] = "restored_verified"
    except Exception as exc:
        report.update(backup.safe_error(exc))
    report["finished_at_utc"] = backup.utc_now()
    if may_write_report:
        backup.write_receipt(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", dest="backup_receipt_path", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", dest="report_path", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--paths", nargs="+", help="Exact relative object paths from the receipt")
    selection.add_argument("--all-approved", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = run_restore(**vars(args))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"restored_verified", "dry_run_validated_no_network"} else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", **backup.safe_error(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
