#!/usr/bin/env python3
"""Back up an explicit Paper3 manifest and verify every object with a full GET.

No directory discovery is performed. The manifest has ``schema_version: 1`` and
``files`` containing path (relative to --root), sha256, bytes, role and dataset.
Roles are allowlisted; raw data must use ``source_development_raw`` and an exact
approved source/development dataset label. Catalog/protocol metadata may name
sealed datasets. This tool does not grant access to any dataset.
The default is local verification only; --execute enables network PUT and GET.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any


CHUNK_SIZE = 1024 * 1024
ALLOWED_ROLES = frozenset({
    "archive", "manifest", "source_development_raw", "code", "protocol",
    "audit", "catalog", "execution_package", "result",
})
RAW_DATASETS = frozenset({
    "BNCI2014_001", "Cho2017", "Lee2019_MI", "PhysionetMI", "Stieger2021",
})
METADATA_ROLES = frozenset({
    "archive", "manifest", "code", "protocol", "audit", "catalog", "execution_package",
})
REQUIRED_VARIABLES = (
    "R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
)


class BackupError(Exception):
    """An error with a fixed, non-sensitive machine-readable code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def blocked_label(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    tokens = normalized.split("_")
    return (
        "internal_test" in normalized or "internaltest" in normalized
        or any(token == "hs"
               or token.startswith(("wang", "clinical", "external", "sealed", "holdout", "hsbci", "hsstroke"))
               or re.fullmatch(r"hs\d+", token)
               for token in tokens)
    )


def validate_prefix(prefix: str) -> str:
    if (not prefix.startswith("paper3/") or not prefix.endswith("/")
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", prefix)
            or any(part in {"", ".", ".."} for part in prefix[:-1].split("/"))
            or prefix == "paper3/"):
        raise BackupError("invalid_isolated_paper3_prefix")
    return prefix


def validate_entry(entry: dict) -> tuple[str, ...]:
    """Validate explicit file metadata without reading a local or remote file."""
    if (not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "bytes", "role", "dataset"}):
        raise BackupError("invalid_manifest_entry")
    relative, expected_hash, size, role, dataset = (
        entry["path"], entry["sha256"], entry["bytes"],
        entry["role"], entry["dataset"],
    )
    if not all(isinstance(value, str) for value in (relative, expected_hash, role, dataset)):
        raise BackupError("invalid_manifest_entry_types")
    parts = PurePosixPath(relative).parts
    if (not relative or not parts or relative.startswith("/") or "\\" in relative
            or any(part in {".", ".."} for part in relative.split("/"))
            or "//" in relative or relative.endswith("/")
            or any(ord(char) < 32 for char in relative)):
        raise BackupError("unsafe_relative_path")
    if (any(part.lower().startswith(".env") for part in parts)
            or any(part.lower() in {".aws", ".ssh", "credentials", "secrets"}
                   for part in parts)
            or Path(relative).suffix.lower() in {".pem", ".key"}):
        raise BackupError("credential_path_forbidden")
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or type(size) is not int or size < 0 or not dataset.strip()):
        raise BackupError("invalid_manifest_integrity_fields")
    # Metadata may describe sealed datasets; raw payloads require exact approval.
    if (role not in ALLOWED_ROLES or blocked_label(role)
            or (role == "source_development_raw"
                and (dataset not in RAW_DATASETS or blocked_label(relative)))
            or (role not in METADATA_ROLES and blocked_label(dataset))):
        raise BackupError("dataset_or_role_forbidden")
    return parts


def load_manifest(manifest_path: Path, root: Path) -> tuple[list[dict], str]:
    raw = manifest_path.read_bytes()
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get("schema_version") != 1
            or not isinstance(data.get("files"), list) or not data["files"]):
        raise BackupError("invalid_manifest_schema")
    root = root.resolve(strict=True)
    files = []
    seen = set()
    for entry in data["files"]:
        parts = validate_entry(entry)
        relative = entry["path"]
        if relative in seen:
            raise BackupError("duplicate_manifest_path")
        seen.add(relative)
        candidate = root.joinpath(*parts)
        # Reject all symlink components, including links that happen to stay in root.
        cursor = root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise BackupError("symlink_forbidden")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise BackupError("manifest_file_missing_or_outside_root")
        actual_hash, actual_size = sha256_file(candidate)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            raise BackupError("local_integrity_mismatch")
        files.append({**entry, "local_path": candidate})
    return files, hashlib.sha256(raw).hexdigest()


def http_status(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    value = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return value if type(value) is int else None


def safe_error(exc: Exception) -> dict:
    # Do not stringify SDK exceptions: those can contain endpoints, headers,
    # request parameters or credentials. Only allow primitive status metadata.
    result = {"error_type": type(exc).__name__}
    if isinstance(exc, BackupError):
        result["error_code"] = exc.code
    status = http_status(getattr(exc, "response", None))
    if status is not None:
        result["http_status"] = status
    return result


def full_get(client: Any, bucket: str, key: str, entry: dict) -> dict:
    result: dict[str, Any] = {
        "status": "failed", "started_at_utc": utc_now(),
        "bytes_read": 0, "complete_body_read": False,
    }
    body = None
    digest = hashlib.sha256()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        result["http_status"] = http_status(response)
        body = response["Body"]
        while chunk := body.read(CHUNK_SIZE):
            if not isinstance(chunk, bytes):
                raise BackupError("non_binary_get_body")
            result["bytes_read"] += len(chunk)
            digest.update(chunk)
        result["complete_body_read"] = True
        result["sha256"] = digest.hexdigest()
        length = response.get("ContentLength")
        result["declared_content_length"] = length if type(length) is int else None
        if result["http_status"] != 200:
            raise BackupError("get_http_status_not_200")
        if (result["bytes_read"] != entry["bytes"]
                or result["sha256"] != entry["sha256"]
                or (length is not None and length != result["bytes_read"])):
            raise BackupError("remote_integrity_mismatch")
        result["status"] = "verified"
    except Exception as exc:
        result.update(safe_error(exc))
        if not result["complete_body_read"]:
            result["partial_sha256"] = digest.hexdigest()
    finally:
        if body is not None:
            try:
                body.close()
            except Exception as exc:
                result.update(status="failed", **safe_error(exc))
        result["finished_at_utc"] = utc_now()
    return result


def put_if_absent(client: Any, bucket: str, key: str, entry: dict) -> dict:
    result: dict[str, Any] = {
        "status": "failed", "started_at_utc": utc_now(),
        "request_body_bytes": entry["bytes"],
        "request_body_sha256": entry["sha256"],
    }
    try:
        # Recheck immediately before submitting: a prior dry run is not evidence
        # that the local bytes are still identical at execution time.
        actual_hash, actual_size = sha256_file(entry["local_path"])
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            raise BackupError("local_integrity_mismatch_before_put")
        with entry["local_path"].open("rb") as stream:
            response = client.put_object(
                Bucket=bucket, Key=key, Body=stream,
                ContentLength=entry["bytes"], IfNoneMatch="*",
                Metadata={"sha256": entry["sha256"]},
            )
        result["http_status"] = http_status(response)
        if result["http_status"] != 200:
            raise BackupError("put_http_status_not_200")
        result["status"] = "created"
    except Exception as exc:
        result.update(safe_error(exc))
        if result.get("http_status") == 412:
            result["status"] = "already_exists_not_overwritten"
    result["finished_at_utc"] = utc_now()
    return result


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".r2-receipt-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def create_client() -> Any:
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"], region_name="auto",
        config=Config(connect_timeout=15, read_timeout=120,
                      retries={"max_attempts": 2, "mode": "standard"}),
    )


def run_backup(*, manifest_path: Path, root: Path, prefix: str,
               receipt_path: Path, execute: bool = False,
               client: Any = None, bucket: str | None = None) -> dict:
    receipt: dict[str, Any] = {
        "schema_version": 1, "operation": "paper3_explicit_manifest_backup",
        "started_at_utc": utc_now(), "execute": execute,
        "status": "failed", "credentials_recorded": False,
        "verification": "full_get_body_sha256", "objects": [],
    }
    try:
        validate_prefix(prefix)
        files, manifest_sha = load_manifest(manifest_path, root)
        receipt.update(prefix=prefix, manifest_sha256=manifest_sha,
                       local_files_verified=len(files))
        if not execute:
            receipt["status"] = "dry_run_verified_local_only"
            receipt["objects"] = [
                {key: value for key, value in entry.items() if key != "local_path"}
                for entry in files
            ]
        else:
            # Only our own prior receipt is used, with exact manifest and prefix
            # matching. Full GET verification is repeated even for completed rows.
            prior_rows = {}
            if receipt_path.exists():
                prior = json.loads(receipt_path.read_text())
                if (prior.get("operation") == receipt["operation"]
                        and prior.get("prefix") == prefix
                        and prior.get("manifest_sha256") == manifest_sha
                        and prior.get("execute") is True):
                    prior_rows = {row.get("path"): row for row in prior.get("objects", [])}
            if client is None:
                if not all(os.environ.get(name) for name in REQUIRED_VARIABLES):
                    raise BackupError("missing_existing_r2_environment")
                client = create_client()
            if bucket is None:
                bucket = os.environ.get("R2_BUCKET")
            if not bucket:
                raise BackupError("missing_existing_r2_bucket")
            for entry in files:
                key = prefix + entry["path"]
                row = {field: value for field, value in entry.items() if field != "local_path"}
                row["key"] = key
                previous_put = prior_rows.get(entry["path"], {}).get("put", {})
                if previous_put.get("status") in {"created", "already_exists_not_overwritten"}:
                    # Copy only fields this program creates; arbitrary old receipt
                    # fields must not leak into a newly written receipt.
                    row["put"] = {field: previous_put[field] for field in (
                        "status", "started_at_utc", "finished_at_utc", "http_status",
                        "request_body_bytes", "request_body_sha256", "error_type",
                    ) if field in previous_put}
                    row["put_reused_from_prior_receipt"] = True
                else:
                    row["put"] = put_if_absent(client, bucket, key, entry)
                if row["put"]["status"] in {"created", "already_exists_not_overwritten"}:
                    row["get"] = full_get(client, bucket, key, entry)
                    row["status"] = "verified" if row["get"]["status"] == "verified" else "failed"
                else:
                    row["status"] = "failed"
                receipt["objects"].append(row)
                write_receipt(receipt_path, receipt)
                if row["status"] != "verified":
                    raise BackupError("object_backup_not_verified")
            receipt["status"] = "verified"
    except Exception as exc:
        receipt.update(safe_error(exc))
    receipt["finished_at_utc"] = utc_now()
    write_receipt(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Isolated paper3/<backup-id>/ prefix")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        receipt = run_backup(manifest_path=args.manifest, root=args.root,
                             prefix=args.prefix, receipt_path=args.receipt,
                             execute=args.execute)
        # The receipt excludes SDK messages, endpoints, bucket and credentials.
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt["status"] in {"verified", "dry_run_verified_local_only"} else 1
    except Exception as exc:
        # E.g. an unwritable receipt directory: still never emit a traceback.
        print(json.dumps({"status": "failed", **safe_error(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
