#!/usr/bin/env python3
"""Resumable, budgeted Stieger W1 metadata preparation, default dry run.

Only pinned official Stieger v1 objects are accepted. One file is downloaded and
parsed at a time (at most one raw object in flight). Newly downloaded temporary
raw files are removed only after an explicit-manifest R2 PUT and complete GET
SHA256 verification. Existing raw files, including the S1 schema sample, remain.
This tool never trains or accesses any external confirmation data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.adapters.stieger_native import load_session
from p3work.splits import freeze_subject_split, _hash
from p3work.window import choose_window
from tools.r2_backup import run_backup, utc_now, sha256_file

CHUNK = 4 * 1024 * 1024
PREFIX = "paper3/stieger2021-v1-w1/"


class BatchError(Exception):
    pass


def official_md5(entry: dict) -> tuple[str, str]:
    """Prefer Figshare computed MD5; use explicitly supplied MD5 if absent.

    Original v1 has two files with an empty computed_md5 and a valid
    supplied_md5. Preserve that provenance distinction, never invent a digest.
    """
    computed, supplied = entry.get("computed_md5"), entry.get("supplied_md5")
    if computed:
        if not re.fullmatch(r"[0-9a-f]{32}", computed):
            raise BatchError("invalid_computed_md5")
        if supplied and supplied != computed:
            raise BatchError("conflicting_official_md5")
        return computed, "computed_md5"
    if supplied and re.fullmatch(r"[0-9a-f]{32}", supplied):
        return supplied, "supplied_md5_only_computed_missing"
    raise BatchError("official_md5_missing")


def atomic_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".stieger-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_inventory(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("article_id") != 13123148 or data.get("version") != 1 or data.get("license", {}).get("name") != "CC BY 4.0":
        raise BatchError("official_stieger_v1_license_inventory_required")
    fs = [f for f in data["files"] if f["name"].endswith(".mat")]
    if len(fs) != 598:
        raise BatchError("official_stieger_v1_598_mat_inventory_required")
    seen = set()
    for f in fs:
        m = re.fullmatch(r"(S[1-9][0-9]*)_Session_([1-9][0-9]*)\.mat", f["name"])
        if not m or f["id"] in seen or f["provider_subject"] != m[1] or f["session_ordinal"] != int(m[2]):
            raise BatchError("invalid_official_identity_or_session")
        if (f["download_url"] != f"https://ndownloader.figshare.com/files/{f['id']}"
                or not isinstance(f["size"], int) or f["size"] <= 0):
            raise BatchError("invalid_official_download_object")
        official_md5(f)
        seen.add(f["id"])
    if {f["provider_subject"] for f in fs} != {f"S{i}" for i in range(1, 63)}:
        raise BatchError("official_62_provider_identities_required")
    return data


def build_plan(inventory_path: Path, *, stage: str = "qualification", split_path: Path | None = None) -> dict:
    inv = read_inventory(inventory_path)
    fs = [f for f in inv["files"] if f["name"].endswith(".mat")]
    split_sha = None
    if stage == "qualification":
        selected = [min([f for f in fs if f["provider_subject"] == f"S{i}"], key=lambda f: f["session_ordinal"]) for i in range(1, 63)]
    elif stage == "meta-duration":
        if split_path is None:
            raise BatchError("frozen_split_required")
        split = json.loads(split_path.read_text())
        split_sha = split.pop("split_sha256", None)
        if split.get("status") != "FROZEN_METADATA_ONLY" or _hash(split) != split_sha:
            raise BatchError("frozen_split_integrity_failed")
        flat = sum(split["partitions"].values(), [])
        if len(flat) != len(set(flat)) or set(flat) != {f"S{i}" for i in range(1, 63)}:
            raise BatchError("split_identity_isolation_failed")
        allowed = set(split["partitions"]["meta_train"])
        selected = [f for f in fs if f["provider_subject"] in allowed]
        selected.sort(key=lambda f: (int(f["provider_subject"][1:]), f["session_ordinal"]))
    else:
        raise BatchError("invalid_stage")
    plan = {
        "schema_version": 1, "dataset": "Stieger2021", "source_version": 1,
        "stage": stage, "inventory_sha256": sha256_file(inventory_path)[0],
        "split_sha256": split_sha, "files": selected,
        "file_count": len(selected), "total_official_bytes": sum(f["size"] for f in selected),
        "max_single_raw_bytes": max(f["size"] for f in selected),
        "max_download_bytes": sum(f["size"] for f in selected), "max_jobs": len(selected),
        "max_gpu_hours": 0, "max_wall_seconds": 14400,
        "inflight_downloads": 1, "parallel_mat_parsers": 1,
        "raw_removal": "only newly downloaded tool-owned temporary raw, after complete R2 GET/SHA256 success",
        "r2_prefix": PREFIX,
        "next_stage_policy": "qualification completion freezes person split and generates meta-duration plan only; no automatic next-stage execution",
        "gpu_training": False, "external_confirmation_access": False,
    }
    plan["plan_sha256"] = _hash(plan)
    return plan


def validate_plan(plan: dict, inventory_path: Path, split_path: Path | None) -> None:
    expected = build_plan(inventory_path, stage=plan.get("stage"), split_path=split_path)
    if plan != expected:
        raise BatchError("plan_not_exact_official_stage_selection")


def check_space(path: Path, size: int) -> None:
    # R2 verifier streams full GET without a local duplicate. Reserve twice the
    # raw size plus 512 MiB for partial/final state and compact metadata.
    if shutil.disk_usage(path).free < 2 * size + 512 * 1024**2:
        raise BatchError("insufficient_disk_for_single_file_stage")
    if Path("/proc/meminfo").exists():
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
        if available < max(3 * 1024**3, 6 * size):
            raise BatchError("insufficient_available_ram_for_serial_mat_parse")


def download_object(entry: dict, partial: Path, final: Path) -> dict:
    """Resume only with an exact Content-Range; never append a 200 response."""
    n = partial.stat().st_size if partial.exists() else 0
    resumed = n
    if n > entry["size"]:
        raise BatchError("partial_larger_than_expected")
    started = time.monotonic()
    if n < entry["size"]:
        headers = {"Range": f"bytes={n}-"} if n else {}
        with urllib.request.urlopen(urllib.request.Request(entry["download_url"], headers=headers), timeout=90) as response:
            if n:
                expected = f"bytes {n}-{entry['size']-1}/{entry['size']}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise BatchError("resume_requires_exact_206_content_range")
            elif response.status != 200:
                raise BatchError("initial_download_requires_200")
            with partial.open("ab" if n else "wb") as out:
                while block := response.read(CHUNK):
                    n += len(block)
                    if n > entry["size"]:
                        raise BatchError("download_larger_than_official_size")
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
    sha, actual = sha256_file(partial)
    md5 = hashlib.md5()
    with partial.open("rb") as f:
        while block := f.read(CHUNK):
            md5.update(block)
    expected_md5, md5_evidence = official_md5(entry)
    if actual != entry["size"] or md5.hexdigest() != expected_md5:
        raise BatchError("official_size_or_md5_failed")
    os.replace(partial, final)
    return {"bytes": actual, "sha256": sha, "md5": md5.hexdigest(), "download_duration_s": time.monotonic()-started,
            "download_completed_utc": utc_now(), "resumed_from_bytes": resumed,
            "official_md5_evidence": md5_evidence}


def verify_local(entry: dict, path: Path) -> dict:
    sha, actual = sha256_file(path)
    md5 = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            md5.update(block)
    expected_md5, md5_evidence = official_md5(entry)
    if actual != entry["size"] or md5.hexdigest() != expected_md5:
        raise BatchError("existing_original_size_or_md5_failed")
    return {"bytes": actual, "sha256": sha, "md5": md5.hexdigest(), "official_md5_evidence": md5_evidence}


def verify_completed(state: dict, root: Path) -> bool:
    if state.get("status") not in {"COMPLETE", "SCHEMA_BLOCKED"}:
        return False
    if not state.get("retained_metadata_sha256"):
        raise BatchError("completed_metadata_manifest_missing")
    for rel, sha in state["retained_metadata_sha256"].items():
        if sha256_file(root / rel)[0] != sha:
            raise BatchError("completed_metadata_integrity_failed")
    receipt = json.loads((root / state["backup_receipt"]).read_text())
    if receipt.get("status") != "verified" or not receipt.get("execute"):
        raise BatchError("completed_backup_receipt_not_verified")
    raw = [row for row in receipt.get("objects", []) if row.get("role") == "source_development_raw"]
    if (len(raw) != 1 or raw[0].get("sha256") != state["raw_integrity"]["sha256"]
            or raw[0].get("bytes") != state["raw_integrity"]["bytes"]
            or raw[0].get("status") != "verified" or raw[0].get("get", {}).get("status") != "verified"
            or raw[0]["get"].get("complete_body_read") is not True
            or raw[0]["get"].get("sha256") != raw[0]["sha256"]
            or raw[0]["get"].get("bytes_read") != raw[0]["bytes"]):
        raise BatchError("completed_raw_full_get_evidence_failed")
    return True


def process_file(entry: dict, *, root: Path, batch_dir: Path, plan: dict, backup_fn=run_backup, load_fn=load_session) -> dict:
    name = entry["name"]
    object_dir = batch_dir / "objects" / str(entry["id"])
    object_dir.mkdir(parents=True, exist_ok=True)
    state_path = object_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "file_id": entry["id"], "name": name, "plan_sha256": plan["plan_sha256"], "status": "PENDING", "started_utc": utc_now()}
    if state["plan_sha256"] != plan["plan_sha256"]:
        raise BatchError("object_state_plan_identity_mismatch")
    if verify_completed(state, root):
        return state
    if state.get("status") == "BACKUP_VERIFIED" or (state.get("status") == "FAILED" and state.get("retained_metadata_sha256") and state.get("backup_receipt")):
        candidate = dict(state, status="COMPLETE")
        verify_completed(candidate, root)
        old_raw = root / state["raw_path"]
        expected_temporary = root / "data/stieger2021/w1-batch-temp" / str(entry["id"]) / name
        if state.get("temporary_raw_owned"):
            if old_raw.resolve() != expected_temporary.resolve():
                raise BatchError("resumed_temporary_ownership_failed")
            if old_raw.exists():
                if verify_local(entry, old_raw)["sha256"] != state["raw_integrity"]["sha256"]:
                    raise BatchError("resumed_temporary_integrity_failed")
                old_raw.unlink()
            state["temporary_raw_removed_after_verified_backup"] = True
        else:
            state["existing_raw_preserved"] = True
        state.update(status="SCHEMA_BLOCKED" if state.get("schema_blocked") else "COMPLETE", completed_utc=utc_now())
        atomic_json(state_path, state)
        return state
    if state.get("status") == "FAILED":
        state["resumed_after_failure_utc"] = utc_now()
    existing = root / "data" / "stieger2021" / "raw" / name
    temporary_dir = root / "data" / "stieger2021" / "w1-batch-temp" / str(entry["id"])
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_raw = temporary_dir / name
    # Existing files are never adopted as disposable tool-owned input.
    if existing.exists():
        raw, owned = existing, False
    else:
        raw = temporary_raw
        owned = (not raw.exists() or
                 (state.get("temporary_raw_owned") is True and state.get("raw_path") == raw.relative_to(root).as_posix()))
    try:
        check_space(root, entry["size"])
        state.update(status="DOWNLOADING_OR_VERIFYING", raw_path=raw.relative_to(root).as_posix(), temporary_raw_owned=owned)
        atomic_json(state_path, state)
        info = verify_local(entry, raw) if raw.exists() else download_object(entry, raw.with_suffix(".mat.partial"), raw)
        state.update(status="RAW_VERIFIED", raw_integrity=info)
        atomic_json(state_path, state)
        t0 = time.monotonic()
        session = None
        try:
            session = load_fn(raw, expected_sha256=info["sha256"])
            audit = session.schema_audit()
            if audit["participant_id"] != entry["provider_subject"] or audit["session_id"] != str(entry["session_ordinal"]):
                raise BatchError("actual_provider_identity_mismatch")
            if audit["pure_lr_trials"] <= 0 or audit["endpoint_errors"]:
                raise BatchError("pure_lr_presence_or_feedback_schema_failed")
            metadata = session.rows
            state["schema_blocked"] = False
        except MemoryError:
            raise BatchError("parser_memory_exhausted") from None
        except Exception as exc:
            # A failed native identity is retained as unknown, never inferred
            # eligible. Archive the original and failure evidence, then continue
            # other independent identities. Any unknown prevents split freeze.
            state["schema_blocked"] = True
            audit = {"status": "SCHEMA_BLOCKED", "dataset": "Stieger2021", "source_version": 1,
                     "official_provider_subject": entry["provider_subject"], "official_session_ordinal": entry["session_ordinal"],
                     "source_file_sha256": info["sha256"], "source_file_bytes": info["bytes"],
                     "error_type": type(exc).__name__, "error_code": str(exc) if isinstance(exc, BatchError) else "native_schema_unverified_details_suppressed",
                     "identity_channel_task_qualification": None}
            metadata = []
        state["parse_duration_s"] = time.monotonic()-t0
        audit_path, metadata_path = object_dir / "adapter_audit.json", object_dir / "trial_metadata.json"
        atomic_json(audit_path, audit)
        atomic_json(metadata_path, metadata)
        del session
        state["status"] = "METADATA_EXTRACTED"
        atomic_json(state_path, state)
        manifest_path, receipt_path = object_dir / "backup_manifest.json", object_dir / "backup_receipt.json"
        paths = [(raw, "source_development_raw"), (audit_path, "audit"), (metadata_path, "audit")]
        manifest = {"schema_version": 1, "files": []}
        for path, role in paths:
            sha, size = sha256_file(path)
            manifest["files"].append({"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": size, "role": role, "dataset": "Stieger2021"})
        atomic_json(manifest_path, manifest)
        prefix = f"{PREFIX}{plan['stage']}/{plan['plan_sha256']}/{entry['id']}/"
        backup = backup_fn(manifest_path=manifest_path, root=root, prefix=prefix, receipt_path=receipt_path, execute=True)
        if backup.get("status") != "verified" or not backup.get("execute"):
            raise BatchError("r2_put_full_get_sha256_not_verified")
        expected = {(row["path"], row["sha256"], row["bytes"]) for row in manifest["files"]}
        verified = {(row["path"], row["sha256"], row["bytes"]) for row in backup.get("objects", [])
                    if row.get("status") == "verified" and row.get("get", {}).get("status") == "verified"
                    and row["get"].get("complete_body_read") is True and row["get"].get("sha256") == row["sha256"]
                    and row["get"].get("bytes_read") == row["bytes"]}
        if expected != verified:
            raise BatchError("r2_receipt_complete_object_set_mismatch")
        retained = [audit_path, metadata_path, manifest_path, receipt_path]
        state.update(status="BACKUP_VERIFIED", backup_receipt=receipt_path.relative_to(root).as_posix(), r2_prefix=prefix,
                     retained_metadata_sha256={p.relative_to(root).as_posix(): sha256_file(p)[0] for p in retained})
        atomic_json(state_path, state)
        # Recheck local bytes and ownership immediately before removing only the
        # temporary raw created inside this object's isolated staging directory.
        if owned:
            if raw.resolve() != temporary_raw.resolve() or verify_local(entry, raw)["sha256"] != info["sha256"]:
                raise BatchError("temporary_raw_ownership_or_integrity_failed")
            raw.unlink()
            state["temporary_raw_removed_after_verified_backup"] = True
        else:
            state["existing_raw_preserved"] = True
        state.update(status="SCHEMA_BLOCKED" if state.get("schema_blocked") else "COMPLETE", completed_utc=utc_now())
        atomic_json(state_path, state)
        return state
    except Exception as exc:
        # Do not stringify SDK/network exceptions: URLs and credential-bearing
        # headers must never enter logs. BatchError only contains fixed codes.
        state.update(status="FAILED", failed_utc=utc_now(), error_type=type(exc).__name__,
                     error_code=str(exc) if isinstance(exc, BatchError) else "operation_failed_details_suppressed")
        atomic_json(state_path, state)
        raise BatchError(state["error_code"]) from None


def finish_qualification(plan: dict, root: Path, batch_dir: Path, inventory_path: Path) -> dict:
    roster = []
    blocked = []
    for entry in plan["files"]:
        object_dir = batch_dir / "objects" / str(entry["id"])
        state = json.loads((object_dir / "state.json").read_text())
        if not verify_completed(state, root):
            raise BatchError("all_62_verified_backups_required_before_split")
        if state["status"] == "SCHEMA_BLOCKED":
            blocked.append({"participant_id": entry["provider_subject"], "file_id": entry["id"],
                            "adapter_audit": (object_dir / "adapter_audit.json").relative_to(root).as_posix()})
            continue
        audit = json.loads((object_dir / "adapter_audit.json").read_text())
        group = {0: "control", 1: "MBSR"}.get(audit.get("mbsr_subject_raw"))
        roster.append({"participant_id": entry["provider_subject"], "identity_verified": True,
                       "clinic11_present_verified": True, "pure_lr_present_verified": audit["pure_lr_trials"] > 0,
                       "intervention_group": group, "qualification_audit_sha256": sha256_file(object_dir / "adapter_audit.json")[0]})
    if blocked:
        result = {"status": "ALL_CANDIDATES_AUDITED_SCHEMA_UNKNOWN_NO_SPLIT", "qualified_people": len(roster),
                  "schema_unknown_people": blocked, "split_frozen": False,
                  "gap": "Resolve unknown native schemas, or explicitly document ineligibility and reduced-roster amendment after all62 audits; never silently discard unknowns"}
        atomic_json(batch_dir / "qualification_blockers.json", result)
        atomic_json(batch_dir / "qualified_roster_partial_not_frozen.json", roster)
        return result
    roster_path, split_path = batch_dir / "qualified_roster.json", batch_dir / "participant_split.json"
    atomic_json(roster_path, roster)
    split = freeze_subject_split(roster)
    atomic_json(split_path, split)
    split_path.chmod(0o444)
    duration = build_plan(inventory_path, stage="meta-duration", split_path=split_path)
    atomic_json(batch_dir / "meta_duration_plan.json", duration)
    return {"status": "QUALIFICATION_COMPLETE_SPLIT_FROZEN_META_DURATION_PLANNED_NOT_STARTED",
            "split_sha256": split["split_sha256"], "counts": split["counts"],
            "meta_duration_files": duration["file_count"], "meta_duration_bytes": duration["total_official_bytes"]}


def finish_durations(plan: dict, root: Path, batch_dir: Path, split_path: Path) -> dict:
    rows = []
    manifest = {"dataset": "Stieger2021", "split_sha256": plan["split_sha256"],
                "inventory_sha256": plan["inventory_sha256"], "plan_sha256": plan["plan_sha256"],
                "selection": "all_official_sessions_for_frozen_meta_train", "sessions": []}
    for entry in plan["files"]:
        object_dir = batch_dir / "objects" / str(entry["id"])
        state = json.loads((object_dir / "state.json").read_text())
        if not verify_completed(state, root) or state["status"] != "COMPLETE":
            raise BatchError("all_meta_train_session_backups_required_before_window")
        rr = [row for row in json.loads((object_dir / "trial_metadata.json").read_text()) if row["tasknumber"] == 1 and row["label"] is not None]
        audit = json.loads((object_dir / "adapter_audit.json").read_text())
        if audit["pure_lr_trials"] != len(rr):
            raise BatchError("extracted_original_lr_count_mismatch")
        manifest["sessions"].append({"participant_id": entry["provider_subject"], "session_id": str(entry["session_ordinal"]),
                                     "source_record_id": entry["name"], "source_file_sha256": audit["source_file_sha256"],
                                     "original_lr_trials": audit["pure_lr_trials"],
                                     "adapter_audit_sha256": sha256_file(object_dir / "adapter_audit.json")[0],
                                     "trial_metadata_sha256": sha256_file(object_dir / "trial_metadata.json")[0]})
        rows.extend(rr)
    manifest["manifest_sha256"] = _hash(manifest)
    atomic_json(batch_dir / "window_input_session_manifest.json", manifest)
    split = json.loads(split_path.read_text())
    coverage = choose_window(rows, split, expected_session_manifest=manifest)
    atomic_json(batch_dir / "window_coverage.json", coverage)
    return {"status": coverage["status"], "selected_window": coverage["selected"], "meta_train_original_lr_trials": len(rows)}


def reuse_qualification_object(entry: dict, *, root: Path, batch_dir: Path, plan: dict, inventory_path: Path) -> bool:
    """Reuse previously backed-up first-session metadata without redownloading."""
    if plan["stage"] != "meta-duration":
        return False
    target = batch_dir / "objects" / str(entry["id"])
    if (target / "state.json").exists():
        return False
    qualification = build_plan(inventory_path)
    prior_dir = root / "audit/stieger_w1/batches" / qualification["plan_sha256"] / "objects" / str(entry["id"])
    prior_path = prior_dir / "state.json"
    if not prior_path.exists():
        return False
    prior = json.loads(prior_path.read_text())
    expected_md5, _ = official_md5(entry)
    if (not verify_completed(prior, root) or prior.get("name") != entry["name"]
            or prior.get("file_id") != entry["id"] or prior.get("status") != "COMPLETE"
            or prior.get("raw_integrity", {}).get("md5") != expected_md5
            or prior.get("raw_integrity", {}).get("bytes") != entry["size"]):
        raise BatchError("prior_qualification_integrity_failed")
    prior_audit = json.loads((prior_dir / "adapter_audit.json").read_text())
    if prior_audit.get("source_file_sha256") != prior["raw_integrity"]["sha256"]:
        raise BatchError("prior_native_audit_source_hash_mismatch")
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json"):
        dest = target / name
        shutil.copyfile(prior_dir / name, dest)
        copied.append(dest)
    state = dict(prior, plan_sha256=plan["plan_sha256"], reused_qualification_object=True,
                 reused_qualification_state=prior_path.relative_to(root).as_posix(),
                 backup_receipt=(target / "backup_receipt.json").relative_to(root).as_posix(),
                 retained_metadata_sha256={p.relative_to(root).as_posix(): sha256_file(p)[0] for p in copied})
    atomic_json(target / "state.json", state)
    verify_completed(state, root)
    return True


def execute_plan(plan: dict, *, root: Path, batch_dir: Path, inventory_path: Path, split_path: Path | None = None,
                 max_new_files: int | None = None) -> dict:
    validate_plan(plan, inventory_path, split_path)
    batch_dir.mkdir(parents=True, exist_ok=True)
    with (batch_dir / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BatchError("batch_already_active") from None
        progress_path = batch_dir / "batch_progress.json"
        prior = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        prior_elapsed = prior.get("active_wall_elapsed_s", 0.0)
        t0 = time.monotonic()
        progress = {"plan_sha256": plan["plan_sha256"], "status": "RUNNING",
                    "invocation_started_utc": utc_now(), "active_wall_elapsed_s": prior_elapsed,
                    "max_wall_seconds": plan["max_wall_seconds"], "max_download_bytes": plan["max_download_bytes"],
                    "max_jobs": plan["max_jobs"], "max_gpu_hours": 0}
        atomic_json(progress_path, progress)
        new_files = 0
        for index, entry in enumerate(plan["files"]):
            progress["active_wall_elapsed_s"] = prior_elapsed + time.monotonic()-t0
            if progress["active_wall_elapsed_s"] >= plan["max_wall_seconds"]:
                progress.update(status="WALL_BUDGET_EXHAUSTED_NO_STAGE_ADVANCE", next_file=entry["name"],
                                split_or_window_frozen=False, unverified_raw_removed=False)
                atomic_json(progress_path, progress)
                return progress
            progress.update(next_file=entry["name"], file_index=index)
            atomic_json(progress_path, progress)
            reuse_qualification_object(entry, root=root, batch_dir=batch_dir, plan=plan, inventory_path=inventory_path)
            state_path = batch_dir / "objects" / str(entry["id"]) / "state.json"
            completed_before = state_path.exists() and verify_completed(json.loads(state_path.read_text()), root)
            if not completed_before and max_new_files is not None and new_files >= max_new_files:
                progress.update(status="PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE", new_files_completed=new_files,
                                split_or_window_frozen=False, active_wall_elapsed_s=prior_elapsed + time.monotonic()-t0)
                atomic_json(progress_path, progress)
                return progress
            try:
                state = process_file(entry, root=root, batch_dir=batch_dir, plan=plan)
            except BaseException:
                progress.update(status="INTERRUPTED_OR_FAILED_CLOSED", active_wall_elapsed_s=prior_elapsed + time.monotonic()-t0)
                atomic_json(progress_path, progress)
                raise
            new_files += not completed_before
            progress.update(active_wall_elapsed_s=prior_elapsed + time.monotonic()-t0, completed_file_index=index)
            atomic_json(progress_path, progress)
            print(json.dumps({"file": entry["name"], "status": state["status"], "completed": index+1, "total": len(plan["files"])}), flush=True)
        if prior_elapsed + time.monotonic()-t0 >= plan["max_wall_seconds"]:
            progress.update(status="WALL_BUDGET_EXHAUSTED_NO_STAGE_ADVANCE", active_wall_elapsed_s=prior_elapsed + time.monotonic()-t0,
                            split_or_window_frozen=False, unverified_raw_removed=False)
            atomic_json(progress_path, progress)
            return progress
        result = finish_qualification(plan, root, batch_dir, inventory_path) if plan["stage"] == "qualification" else finish_durations(plan, root, batch_dir, split_path)
        progress.update(status=result["status"], active_wall_elapsed_s=prior_elapsed + time.monotonic()-t0)
        atomic_json(progress_path, progress)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=REPO / "data_catalog/stieger2021/official_inventory.json")
    parser.add_argument("--stage", choices=["qualification", "meta-duration"], default="qualification")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-new-files", type=int, help="Bounded live timing batch; no split/window stage advance while any object remains")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text()) if args.plan else build_plan(args.inventory, stage=args.stage, split_path=args.split)
        validate_plan(plan, args.inventory, args.split)
        if args.write_plan:
            atomic_json(args.write_plan, plan)
        summary = {key: plan[key] for key in ("plan_sha256", "stage", "file_count", "total_official_bytes", "max_single_raw_bytes", "inflight_downloads", "parallel_mat_parsers", "max_download_bytes", "max_jobs", "max_gpu_hours", "max_wall_seconds")}
        summary["execute"] = args.execute
        print(json.dumps(summary), flush=True)
        if args.execute:
            batch_dir = REPO / "audit/stieger_w1/batches" / plan["plan_sha256"]
            if args.max_new_files is not None and args.max_new_files < 1:
                raise BatchError("max_new_files_must_be_positive")
            result = execute_plan(plan, root=REPO, batch_dir=batch_dir, inventory_path=args.inventory, split_path=args.split,
                                  max_new_files=args.max_new_files)
            atomic_json(batch_dir / "completion.json", result)
            print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                          "error_code": str(exc) if isinstance(exc, BatchError) else "operation_failed_details_suppressed"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
