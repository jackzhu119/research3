"""Profile-bound R2 IO for three newly registered source/development datasets.

The original backup and restore tools are loaded as private module instances.
Their integrity, conditional PUT and streaming GET implementations are reused;
their public policies and module globals are never changed. This profile grants
no access to any external cohort, and performs no dataset download or training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
PROFILE_ID = "paper3-source-extension-data-plan-20261019-v1"
DOCUMENT_SHA256 = "74383bb55cd1287ad806e5d83812e95bdb4bf0bccf7a59292b91204f230a073a"
REGISTERED_ROLES = {
    "Yang2025": "source", "Dreyer2023": "source",
    "Ma2022": "independent_development",
}
RAW_ROOT = "data/source_extensions_20261019"
RESULT_ROOT = "cache/source_extensions_20261019"
METADATA_ROOTS = ("catalog/source_extensions_20261019", "audit/source_extensions_20261019")
METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".xml"})
RESULT_SUFFIXES = frozenset({".npy", ".json", ".jsonl", ".csv", ".tsv", ".txt"})
RAW_SUFFIXES = frozenset({".mat", ".edf", ".bdf", ".gdf", ".set", ".fdt", ".cnt",
                          ".dat", ".eeg", ".vhdr", ".vmrk", ".csv", ".tsv",
                          ".txt", ".json", ".zip", ".7z", ".tar", ".gz"})
ENTRY_FIELDS = ("path", "sha256", "bytes", "role", "dataset")


class SourceExtensionIOError(ValueError):
    """Fixed public error code; never interpolate SDK or credential data."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def profile() -> dict:
    """Return a new, exact descriptor so callers cannot mutate global policy."""
    return {
        "profile_id": PROFILE_ID,
        "document": {"filename": "Paper3_Research_Data_Plan.docx", "sha256": DOCUMENT_SHA256},
        "registered_dataset_roles": dict(REGISTERED_ROLES),
        "raw_root": RAW_ROOT,
        "result_root": RESULT_ROOT,
        "metadata_roots": list(METADATA_ROOTS),
        "sealed_external_payload_access_allowed": False,
    }


def _load_tool(name: str):
    specification = importlib.util.spec_from_file_location(
        "_paper3_source_extension_" + name, REPOSITORY / "tools" / (name + ".py"))
    if specification is None or specification.loader is None:
        raise SourceExtensionIOError("original_io_implementation_unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _credential_components(path: Path | PurePosixPath) -> None:
    # In addition to the original IO guard, prohibit common credential stores
    # and filenames even when they happen to use a text or JSON suffix.
    for component in path.parts:
        value = component.lower()
        tokens = set(re.split(r"[^a-z0-9]+", value))
        normalized = re.sub(r"[^a-z0-9]+", "_", value)
        if (value.startswith(".env") or value in {".aws", ".ssh", ".boto", "credentials", "secrets"}
                or tokens.intersection({"credential", "credentials", "secret", "secrets", "password",
                                        "token", "apikey", "privatekey"})
                or any(marker in normalized for marker in
                       ("aws_access_key", "aws_secret_access_key", "r2_endpoint", "r2_bucket",
                        "api_key", "private_key"))
                or value in {"rclone.conf", "boto.cfg"}
                or Path(value).suffix in {".env", ".pem", ".key", ".p12", ".pfx"}):
            raise SourceExtensionIOError("credential_path_forbidden")


def _entry_validator(original):
    def validate(entry: dict):
        parts = original(entry)
        relative, dataset, role = entry["path"], entry["dataset"], entry["role"]
        _credential_components(PurePosixPath(relative))
        if dataset not in REGISTERED_ROLES:
            raise SourceExtensionIOError("dataset_not_in_source_extension_profile")
        suffix = PurePosixPath(relative).suffix.lower()
        if role == "source_development_raw":
            prefix = RAW_ROOT + "/" + dataset + "/"
            suffixes = RAW_SUFFIXES
        elif role == "result":
            prefix = RESULT_ROOT + "/" + dataset + "/"
            suffixes = RESULT_SUFFIXES
        elif role in {"catalog", "audit", "manifest"}:
            prefixes = tuple(root + "/" + dataset + "/" for root in METADATA_ROOTS)
            if not relative.startswith(prefixes):
                raise SourceExtensionIOError("metadata_outside_registered_dataset_namespace")
            if suffix not in METADATA_SUFFIXES:
                raise SourceExtensionIOError("raw_or_unsupported_file_cannot_use_metadata_role")
            return parts
        else:
            raise SourceExtensionIOError("source_extension_role_forbidden")
        if not relative.startswith(prefix):
            raise SourceExtensionIOError("payload_outside_registered_dataset_namespace")
        if suffix not in suffixes:
            raise SourceExtensionIOError("unsupported_source_extension_payload_type")
        return parts
    return validate


def _modules():
    backup = _load_tool("r2_backup")
    original_safe_error = backup.safe_error

    def safe_error(exc):
        if isinstance(exc, SourceExtensionIOError):
            return {"error_type": type(exc).__name__, "error_code": exc.code}
        return original_safe_error(exc)

    backup.safe_error = safe_error
    backup.RAW_DATASETS = backup.RAW_DATASETS | frozenset(REGISTERED_ROLES)
    validate = _entry_validator(backup.validate_entry)

    def validate_entry(entry):
        parts = validate(entry)
        if backup.blocked_label(entry["path"]):
            raise SourceExtensionIOError("external_or_sealed_path_forbidden")
        return parts

    backup.validate_entry = validate_entry
    restore = _load_tool("r2_restore")
    restore.backup = backup
    original_loader = restore.load_verified_receipt

    def load_profiled_receipt(path: Path):
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("source_extension_profile") != profile():
            raise SourceExtensionIOError("source_extension_receipt_profile_mismatch")
        entries, digest = original_loader(path)
        if digest != hashlib.sha256(raw).hexdigest():
            raise SourceExtensionIOError("backup_receipt_changed_during_validation")
        return entries, digest

    restore.load_verified_receipt = load_profiled_receipt
    return backup, restore


def validate_entry(entry: dict) -> tuple[str, ...]:
    """Validate a manifest row against the exact new profile, without IO."""
    backup, _ = _modules()
    return backup.validate_entry(entry)


def _safe_path(path: Path, restore) -> Path:
    absolute = Path(os.path.abspath(path))
    _credential_components(absolute)
    restore.no_symlink_components(absolute)
    return absolute


def _same_file(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and left.samefile(right))


def _writer(backup, restore, *, output: Path, protected: list[Path], operation: str):
    original = backup.write_receipt

    def write(path: Path, receipt: dict):
        _safe_path(path, restore)
        if path != output or any(_same_file(path, item) for item in protected):
            raise SourceExtensionIOError("receipt_cannot_overwrite_input_or_payload")
        if path.exists():
            previous = json.loads(path.read_bytes())
            if (not isinstance(previous, dict) or previous.get("operation") != operation
                    or previous.get("source_extension_profile") != profile()):
                raise SourceExtensionIOError("unrelated_receipt_file_not_overwritten")
        receipt["source_extension_profile"] = profile()
        original(path, receipt)

    backup.write_receipt = write
    return write


def _failure(backup, exc: Exception, *, execute: bool, operation: str) -> dict:
    error = ({"error_type": type(exc).__name__, "error_code": exc.code}
             if isinstance(exc, SourceExtensionIOError) else backup.safe_error(exc))
    return {"schema_version": 1, "operation": operation, "status": "failed", "execute": execute,
            "credentials_recorded": False, "source_extension_profile": profile(), **error}


def run_backup(*, manifest_path: Path, root: Path, prefix: str,
               receipt_path: Path, execute: bool = False,
               client: Any = None, bucket: str | None = None) -> dict:
    """Back up explicit approved rows; fully GET/SHA each object after PUT.

    The default verifies local inputs only. A single-object manifest supports
    bounded local raw staging. Conditional writes never overwrite remote data.
    Errors during unsafe output-path preflight are returned without writing.
    """
    backup, restore = _modules()
    operation = "paper3_explicit_manifest_backup"
    try:
        root = _safe_path(Path(root), restore)
        manifest_path = _safe_path(Path(manifest_path), restore)
        receipt_path = _safe_path(Path(receipt_path), restore)
        files, _ = backup.load_manifest(manifest_path, root)
        writer = _writer(backup, restore, output=receipt_path,
                         protected=[manifest_path] + [entry["local_path"] for entry in files],
                         operation=operation)
        # Preflight output identity before even an existing object can be read.
        if any(_same_file(receipt_path, path) for path in [manifest_path] +
               [entry["local_path"] for entry in files]):
            raise SourceExtensionIOError("receipt_cannot_overwrite_input_or_payload")
        if receipt_path.exists():
            value = json.loads(receipt_path.read_bytes())
            if (not isinstance(value, dict) or value.get("operation") != operation
                    or value.get("source_extension_profile") != profile()):
                raise SourceExtensionIOError("unrelated_receipt_file_not_overwritten")
        result = backup.run_backup(manifest_path=manifest_path, root=root, prefix=prefix,
                                   receipt_path=receipt_path, execute=execute,
                                   client=client, bucket=bucket)
        # Original tool catches local guard errors; retain their fixed codes.
        writer(receipt_path, result)
        return result
    except Exception as exc:
        return _failure(backup, exc, execute=execute, operation=operation)


def run_restore(*, backup_receipt_path: Path, root: Path, report_path: Path,
                paths: list[str] | None = None, all_approved: bool = False,
                execute: bool = False, require_fresh: bool = False,
                client: Any = None, bucket: str | None = None) -> dict:
    """Restore only an exact profiled verified receipt; optionally require fresh GETs.

    ``require_fresh=True`` rejects any pre-existing selected payload before any
    network operation. Successful fresh mode therefore proves a complete GET
    and SHA256 check for every selected object into an empty destination set.
    """
    backup, restore = _modules()
    operation = "paper3_verified_receipt_restore"
    try:
        root = _safe_path(Path(root), restore)
        backup_receipt_path = _safe_path(Path(backup_receipt_path), restore)
        report_path = _safe_path(Path(report_path), restore)
        if bool(paths) == bool(all_approved):
            raise SourceExtensionIOError("explicit_restore_selection_required")
        entries, _ = restore.load_verified_receipt(backup_receipt_path)
        by_path = {entry["path"]: entry for entry in entries}
        if paths and (len(paths) != len(set(paths)) or any(path not in by_path for path in paths)):
            raise SourceExtensionIOError("restore_selection_not_in_verified_receipt")
        selected = entries if all_approved else [by_path[path] for path in paths]
        targets = [restore.destination(root, entry) for entry in selected]
        if require_fresh and any(target.exists() for target in targets):
            raise SourceExtensionIOError("fresh_restore_requires_absent_selected_payloads")
        protected = [backup_receipt_path] + targets
        if any(_same_file(report_path, path) for path in protected):
            raise SourceExtensionIOError("receipt_cannot_overwrite_input_or_payload")
        if report_path.exists():
            previous = json.loads(report_path.read_bytes())
            if (not isinstance(previous, dict) or previous.get("operation") != operation
                    or previous.get("source_extension_profile") != profile()):
                raise SourceExtensionIOError("unrelated_receipt_file_not_overwritten")
        _writer(backup, restore, output=report_path, protected=protected, operation=operation)
        result = restore.run_restore(backup_receipt_path=backup_receipt_path, root=root,
                                     report_path=report_path, paths=paths, all_approved=all_approved,
                                     execute=execute, client=client, bucket=bucket)
        result["require_fresh"] = require_fresh
        result["fresh_full_get_verified"] = bool(execute and require_fresh and
            result["status"] == "restored_verified" and all(
                row["status"] == "restored_verified" and row.get("get_attempted") is True
                and row.get("complete_body_read") is True for row in result["objects"]))
        backup.write_receipt(report_path, result)
        return result
    except Exception as exc:
        return _failure(backup, exc, execute=execute, operation=operation)
