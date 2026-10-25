"""Read a small, explicit whitelist of paper-result files; never visit trial trees."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time

ROUND_OUTPUTS = {2: "followup_20261023_v2", 3: "followup_20261024_r3", 7: "followup_20261024_r7"}
MIB = 1024 * 1024


def _json(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _signature(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _reject_constant(value):
    raise ValueError("non_finite_json_constant")


def _read_file(base, relative, maximum):
    """Open each path component relative to a directory descriptor, without symlinks."""
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in {".", ".."} for p in parts):
        raise ValueError("invalid_relative_path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    parent = os.open(base, flags | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], flags, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not_regular_file")
        if before.st_size > maximum:
            raise ValueError("file_size_limit")
        blocks, size = [], 0
        while True:
            block = os.read(descriptor, min(MIB, maximum + 1 - size))
            if not block:
                break
            blocks.append(block)
            size += len(block)
            if size > maximum:
                raise ValueError("file_size_limit")
        after = os.fstat(descriptor)
        if _signature(before) != _signature(after) or size != after.st_size:
            raise ValueError("source_changed_during_read")
        return b"".join(blocks), _signature(after)
    finally:
        os.close(descriptor)


def _task_key(value):
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if Path(value).is_absolute() or any(p in {"", ".", ".."} for p in value.split("/")):
        return False
    return not any(ord(c) < 32 for c in value)


def collect_round(root, round_number, work, emit=None, *, include_tasks=True,
                  max_report_bytes=10 * MIB, max_file_bytes=16 * MIB):
    """Collect reports first, optionally task result receipts; return a frozen local snapshot.

    source files are never written, locked or recomputed. No task directory is
    enumerated. include_tasks=False is the preferred fast paper-writing export.
    """
    if round_number not in ROUND_OUTPUTS:
        raise ValueError("supported_rounds_are_2_3_7")
    if min(max_report_bytes, max_file_bytes) < 1:
        raise ValueError("positive_file_limits_required")
    root, work = Path(root).expanduser().resolve(), Path(work).expanduser().resolve()
    output = root / "runs" / ROUND_OUTPUTS[round_number]
    if work == output or output in work.parents:
        raise ValueError("work_directory_must_be_outside_source_output")
    round_work = work / ("round-%02d" % round_number)
    round_work.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".collect-", dir=round_work))
    rows, warnings, documents = [], [], {}
    last = time.monotonic()

    def event(name, **fields):
        if emit is not None:
            try:
                emit(name, round=round_number, **fields)
            except Exception:
                pass

    def warn(path, reason):
        warnings.append({"path": path, "reason": reason})
        event("PAPER_FILE_SKIPPED", path=path, reason=reason)

    def capture(relative, optional=False):
        nonlocal last
        relative = str(relative)
        limit = max_report_bytes if Path(relative).name == "report.json" else max_file_bytes
        try:
            data, signature = _read_file(output, relative, limit)
        except FileNotFoundError:
            if not optional:
                warn(relative, "missing")
            return None
        except (OSError, ValueError) as error:
            warn(relative, str(error) if isinstance(error, ValueError) else type(error).__name__)
            return None
        parsed = None
        if Path(relative).suffix == ".json":
            try:
                parsed = json.loads(data, parse_constant=_reject_constant)
            except (ValueError, UnicodeDecodeError):
                warn(relative, "invalid_json")
                return None
            if relative == "plan.json" or Path(relative).name in {"report.json", "result.json"}:
                if not isinstance(parsed, dict):
                    warn(relative, "expected_json_object")
                    return None
        destination = stage / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "source_signature": signature})
        documents[relative] = parsed
        if time.monotonic() - last >= 10:
            event("PAPER_COLLECT_PROGRESS", files=len(rows), bytes=sum(r["bytes"] for r in rows))
            last = time.monotonic()
        return parsed

    event("PAPER_REPORT_COLLECTION_STARTED", source_output=str(output), include_tasks=include_tasks)
    plan = capture("plan.json")
    capture("report.json", optional=True)
    # Each listing is exactly one directory, never a recursive walk or glob.
    try:
        root_names = sorted(p.name for p in output.iterdir() if p.suffix == ".csv")
        for name in root_names:
            capture(name)
    except OSError as error:
        warn(".", type(error).__name__)
    reports = output / "reports"
    if not reports.is_symlink():
        try:
            names = sorted(p.name for p in reports.iterdir() if p.suffix in {".csv", ".json", ".md"})
            for name in names:
                capture("reports/" + name)
        except FileNotFoundError:
            pass
        except OSError as error:
            warn("reports", type(error).__name__)
    else:
        warn("reports", "symlink_directory")

    reports_observed = []
    for name in ("report.json", "reports/report.json"):
        value = documents.get(name)
        if isinstance(value, dict):
            reports_observed.append({"path": name, **{k: value[k] for k in
                ("status", "complete_tasks", "total_tasks", "core_complete", "primary_family_complete", "updated_utc") if k in value}})
            if not value.get("status"):
                warn(name, "report_status_not_provided")
    if not reports_observed:
        warn("report.json|reports/report.json", "no_readable_aggregate_report")

    task_rows, unique, plan_valid = [], set(), isinstance(plan, dict) and isinstance(plan.get("tasks"), list)
    tasks = plan.get("tasks", []) if plan_valid else []
    if not plan_valid:
        warn("plan.json", "missing_or_invalid_task_plan")
    for task in tasks:
        key = task.get("key") if isinstance(task, dict) else None
        if not _task_key(key) or key in unique:
            plan_valid = False
            warn("plan.json", "invalid_or_duplicate_task_key")
            continue
        unique.add(key)
        if include_tasks:
            value = capture(key + "/result.json")
            task_rows.append({"key": key, "status": value.get("status", "UNREADABLE") if value else "MISSING_OR_UNREADABLE"})
    complete = sum(row["status"] == "COMPLETE" for row in task_rows) if include_tasks else None
    all_tasks = bool(plan_valid and include_tasks and tasks and complete == len(tasks))
    counts = {"plan_task_count": len(tasks) if isinstance(plan, dict) else None,
              "task_receipts_included": include_tasks, "task_receipts_read": len(task_rows),
              "verified_complete_tasks": complete, "all_plan_tasks_verified_complete": all_tasks}
    identity = {"round": round_number, "include_tasks": include_tasks,
                "files": [{k: r[k] for k in ("path", "bytes", "sha256")} for r in rows]}
    snapshot_id = hashlib.sha256(_json(identity)).hexdigest()
    manifest = {"schema_version": "paper3-paper-results-snapshot-v1", "round": round_number,
                "snapshot_id": snapshot_id, "source_output": str(output),
                "collected_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "files": rows, "warnings": warnings, "task_counts": counts,
                "reports_observed": reports_observed, "task_statuses": task_rows,
                "scope": "Existing aggregate reports, tables and plan, optionally task result receipts; no report recomputation.",
                "asynchronous_reports": True,
                "report_completeness_note": "Report status and counts describe the captured report, independently of task receipt observations.",
                "raw_trial_predictions_included": False, "checkpoints_or_models_included": False,
                "participant_vectors_guaranteed": False,
                "analysis_limit": "Only columns present in collected tables/reports can be analysed. New participant/session/trial-level statistics may require a later dedicated export; this is not a full reanalysis or reproducibility archive.",
                "source_outputs_modified": False, "recursive_source_scan_performed": False}
    (stage / "manifest.json").write_bytes(_json(manifest))
    destination = round_work / snapshot_id
    if destination.exists():
        destination = round_work / (snapshot_id + "-" + str(os.getpid()) + "-" + str(time.time_ns()))
    os.rename(stage, destination)
    files = [{**r, "local_path": str(destination / "files" / r["path"])} for r in rows]
    event("PAPER_RESULTS_COLLECTED", files=len(files), bytes=sum(r["bytes"] for r in rows),
          snapshot_id=snapshot_id, warnings=len(warnings), **counts)
    return {"status": "READY" if rows else "UNAVAILABLE", "round": round_number,
            "snapshot_id": snapshot_id, "directory": str(destination), "files": files,
            "manifest": manifest, "manifest_path": str(destination / "manifest.json")}
