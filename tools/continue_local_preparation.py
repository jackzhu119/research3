#!/usr/bin/env python3
"""Coordinate the existing W1 worker, serial CPU preparation, and delivery.

Default dry-run only reads local metadata: no waits, downloads, subprocesses or
status mutations. --execute waits for the existing qualification lock; it never
starts or restarts that worker. All 62 real qualification objects must verify
before split/window decisions. A separate operational execution plan extends
only the meta-duration wall budget to 18 h; the exact scientific selection and
the original 14,400 s selection plan remain intact. No GPU/RunPod job is run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.contracts import validate_signal_contract
from p3work.splits import _hash
from tools import stieger_metadata_batch as base


QUALIFICATION_HASH = "c825be2636a5f423838816f76ededbae8cb08e92a3ddbc57032fb911db8763fa"
EXECUTION_WALL_SECONDS = 64800
OBSERVED_SECONDS_PER_DECIMAL_GB = 147.5
POLL_SECONDS = 15
QUALIFIED = "QUALIFICATION_COMPLETE_SPLIT_FROZEN_META_DURATION_PLANNED_NOT_STARTED"


class PreparationError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def safe_failure(exc: BaseException) -> dict:
    result = {"error_type": type(exc).__name__}
    if isinstance(exc, PreparationError):
        result["error_code"] = exc.code
    elif isinstance(exc, base.BatchError) and re.fullmatch(r"[a-z0-9_]+", str(exc)):
        result["error_code"] = str(exc)
    else:
        result["error_code"] = "operation_failed_details_suppressed"
    return result


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise PreparationError("json_object_required")
    return value


def local(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or relative.startswith("/")
            or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise PreparationError("unsafe_local_artifact_path")
    path = root / relative
    cursor = root
    for part in relative.split("/"):
        cursor = cursor / part
        if cursor.is_symlink():
            raise PreparationError("symlink_artifact_forbidden")
    if not path.resolve().is_relative_to(root):
        raise PreparationError("artifact_outside_root")
    return path


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise PreparationError("existing_frozen_artifact_differs")
        return
    base.atomic_json(path, value)


def validate_qualification_plan(plan: dict, inventory: Path) -> None:
    base.validate_plan(plan, inventory, None)
    if (plan.get("stage") != "qualification" or plan.get("plan_sha256") != QUALIFICATION_HASH
            or plan.get("file_count") != 62
            or {row["provider_subject"] for row in plan["files"]} != {f"S{i}" for i in range(1, 63)}):
        raise PreparationError("pinned_62_identity_qualification_plan_required")


def object_state(entry: dict, batch_dir: Path) -> dict | None:
    path = batch_dir / "objects" / str(entry["id"]) / "state.json"
    return read_json(path) if path.exists() else None


def verify_object(entry: dict, state: dict, *, root: Path, plan: dict,
                  allow_schema_unknown: bool = False) -> bool:
    if (state.get("file_id") != entry["id"] or state.get("name") != entry["name"]
            or state.get("plan_sha256") != plan["plan_sha256"]):
        raise PreparationError("completed_object_selection_identity_mismatch")
    if not base.verify_completed(state, root):
        return False
    md5, _ = base.official_md5(entry)
    if (state.get("raw_integrity", {}).get("bytes") != entry["size"]
            or state.get("raw_integrity", {}).get("md5") != md5):
        raise PreparationError("completed_object_official_integrity_mismatch")
    if state.get("status") == "SCHEMA_BLOCKED":
        if allow_schema_unknown:
            return True
        raise PreparationError("unknown_native_schema_blocks_stage")
    if state.get("status") != "COMPLETE":
        return False
    return True


def count_verified(plan: dict, root: Path, batch_dir: Path) -> dict:
    complete = unknown = missing = 0
    verified_bytes = 0
    for entry in plan["files"]:
        state = object_state(entry, batch_dir)
        if state is None or state.get("status") not in {"COMPLETE", "SCHEMA_BLOCKED"}:
            missing += 1
            continue
        if not verify_object(entry, state, root=root, plan=plan, allow_schema_unknown=True):
            missing += 1
            continue
        verified_bytes += entry["size"]
        if state["status"] == "SCHEMA_BLOCKED":
            unknown += 1
        else:
            complete += 1
    return {"verified_complete_files": complete, "verified_schema_unknown_files": unknown,
            "incomplete_files": missing, "verified_official_bytes": verified_bytes,
            "selected_files": len(plan["files"])}


def worker_snapshot(batch_dir: Path) -> dict:
    path = batch_dir / "batch_progress.json"
    if not path.exists():
        return {"status": "batch_progress_missing"}
    source = read_json(path)
    # completion.json can be an old bounded-run result. It is intentionally not
    # read anywhere in this coordinator.
    return {key: source[key] for key in (
        "plan_sha256", "status", "active_wall_elapsed_s", "file_index", "completed_file_index",
        "max_wall_seconds", "max_jobs", "max_download_bytes",
    ) if key in source}


def progress_write(path: Path, progress: dict, **updates) -> None:
    progress.update(updates, updated_at_utc=base.utc_now(), gpu_jobs_started=0,
                    remote_jobs_started=0, external_payload_accessed=False)
    base.atomic_json(path, progress)


@contextmanager
def exclusive_lock(path: Path, code="coordinator_already_active"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreparationError(code) from None
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_for_qualification(plan: dict, *, root: Path, batch_dir: Path,
                           inventory: Path, progress_path: Path, progress: dict,
                           poll_seconds: float = POLL_SECONDS, sleep=time.sleep) -> Path:
    if not 0 < poll_seconds <= 30:
        raise PreparationError("poll_interval_must_be_at_most_30_seconds")
    if not batch_dir.is_dir() or not (batch_dir / ".batch.lock").exists():
        raise PreparationError("existing_qualification_worker_lock_missing")
    with (batch_dir / ".batch.lock").open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                counts = count_verified(plan, root, batch_dir)
                progress_write(progress_path, progress, status="WAITING_EXISTING_QUALIFICATION_WORKER",
                               qualification=counts, authoritative_worker_progress=worker_snapshot(batch_dir))
                sleep(poll_seconds)
        try:
            counts = count_verified(plan, root, batch_dir)
            progress_write(progress_path, progress, status="VERIFYING_QUALIFICATION_AFTER_WORKER_EXIT",
                           qualification=counts, authoritative_worker_progress=worker_snapshot(batch_dir))
            if counts["incomplete_files"] or counts["verified_complete_files"] + counts["verified_schema_unknown_files"] != 62:
                raise PreparationError("qualification_incomplete_no_worker_restart")
            # Reuse the real finisher, including its explicit unknown-schema gate.
            result = base.finish_qualification(plan, root, batch_dir, inventory)
            base.atomic_json(batch_dir / "coordinator_qualification_verification.json", result)
            if counts["verified_schema_unknown_files"] or result.get("status") != QUALIFIED:
                raise PreparationError("qualification_unknown_schema_no_split")
            split_path = batch_dir / "participant_split.json"
            split = read_json(split_path)
            body = dict(split)
            sha = body.pop("split_sha256", None)
            if (_hash(body) != sha or split.get("status") != "FROZEN_METADATA_ONLY"
                    or split.get("counts") != {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11}):
                raise PreparationError("full_roster_frozen_split_integrity_failed")
            return split_path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_execution_amendment(selection: dict, *, qualification_sha256: str) -> dict:
    """Keep scientific selection byte-for-byte; revise only operational limits."""
    if (selection.get("stage") != "meta-duration" or selection.get("max_wall_seconds") != 14400
            or selection.get("max_jobs") != len(selection.get("files", []))
            or selection.get("max_download_bytes") != sum(row["size"] for row in selection["files"])
            or selection.get("max_gpu_hours") != 0 or selection.get("gpu_training") is not False
            or selection.get("external_confirmation_access") is not False):
        raise PreparationError("exact_unmodified_cpu_selection_plan_required")
    result = copy.deepcopy(selection)
    result.pop("plan_sha256")
    result.update(
        base_selection_plan_sha256=selection["plan_sha256"],
        max_wall_seconds=EXECUTION_WALL_SECONDS,
        operational_execution_amendment={
            "kind": "wall_budget_only_no_scientific_protocol_change",
            "original_max_wall_seconds": 14400,
            "observed_qualification_plan_sha256": qualification_sha256,
            "observed_serial_seconds_per_decimal_GB": OBSERVED_SECONDS_PER_DECIMAL_GB,
            "estimated_full_selection_seconds": round(
                selection["total_official_bytes"] / 1e9 * OBSERVED_SECONDS_PER_DECIMAL_GB, 3),
            "basis": "Actual serial download/native parsing/PUT/full-GET timing; all official frozen meta-train sessions require hours",
            "file_selection_sha256": _hash(selection["files"]),
            "max_jobs_and_official_bytes_unchanged": True,
            "parallel_downloads_and_parsers": 1,
            "gpu_hours": 0,
        },
    )
    result["plan_sha256"] = _hash(result)
    return result


def validate_execution_amendment(selection: dict, amendment: dict, *, inventory: Path,
                                 split_path: Path, qualification_sha256: str) -> None:
    # Never pass an amended selection through a weaker schema check. Validate
    # the original official selection with the unchanged base implementation.
    base.validate_plan(selection, inventory, split_path)
    expected = build_execution_amendment(selection, qualification_sha256=qualification_sha256)
    if amendment != expected:
        raise PreparationError("operational_amendment_not_exact_approved_selection")


def execute_duration_selection(selection: dict, amendment: dict, *, root: Path,
                               batch_dir: Path, inventory: Path, split_path: Path,
                               progress_path: Path, progress: dict,
                               monotonic=time.monotonic) -> dict:
    validate_execution_amendment(selection, amendment, inventory=inventory,
                                 split_path=split_path, qualification_sha256=QUALIFICATION_HASH)
    with exclusive_lock(batch_dir / ".batch.lock", "duration_batch_already_active"):
        worker_progress = batch_dir / "batch_progress.json"
        prior = read_json(worker_progress) if worker_progress.exists() else {}
        elapsed_before = prior.get("active_wall_elapsed_s", 0.0)
        if (type(elapsed_before) not in {int, float} or not math.isfinite(elapsed_before)
                or elapsed_before < 0
                or (prior and prior.get("plan_sha256") != selection["plan_sha256"])):
            raise PreparationError("duration_resume_budget_identity_failed")
        started = monotonic()
        current = {
            "plan_sha256": selection["plan_sha256"],
            "operational_execution_plan_sha256": amendment["plan_sha256"],
            "max_wall_seconds": amendment["max_wall_seconds"],
            "max_jobs": selection["max_jobs"], "max_download_bytes": selection["max_download_bytes"],
            "max_gpu_hours": 0, "status": "RUNNING_SERIAL_CPU",
        }

        def checkpoint(status="RUNNING_SERIAL_CPU", **extra):
            elapsed = elapsed_before + monotonic() - started
            counts = count_verified(selection, root, batch_dir)
            current.update(status=status, active_wall_elapsed_s=elapsed, **counts, **extra)
            base.atomic_json(worker_progress, current)
            progress_write(progress_path, progress, status=status, duration=dict(current))
            if counts["verified_schema_unknown_files"]:
                raise PreparationError("duration_unknown_schema_blocks_window")
            if (counts["verified_complete_files"] > selection["max_jobs"]
                    or counts["verified_official_bytes"] > selection["max_download_bytes"]):
                raise PreparationError("exact_duration_job_or_byte_budget_exceeded")
            if elapsed >= amendment["max_wall_seconds"]:
                current["status"] = "WALL_BUDGET_EXHAUSTED_NO_STAGE_ADVANCE"
                base.atomic_json(worker_progress, current)
                progress_write(progress_path, progress, status=current["status"], duration=dict(current))
                raise PreparationError("duration_operational_wall_budget_exhausted")
            return counts

        try:
            for index, entry in enumerate(selection["files"]):
                checkpoint(next_file=entry["name"], file_index=index)
                base.reuse_qualification_object(entry, root=root, batch_dir=batch_dir,
                                                plan=selection, inventory_path=inventory)
                state = base.process_file(entry, root=root, batch_dir=batch_dir, plan=selection)
                if not verify_object(entry, state, root=root, plan=selection):
                    raise PreparationError("duration_object_not_complete_after_processing")
                checkpoint(completed_file_index=index)
            counts = checkpoint()
            if counts["verified_complete_files"] != len(selection["files"]) or counts["incomplete_files"]:
                raise PreparationError("all_meta_train_sessions_required_before_window")
            base.validate_plan(selection, inventory, split_path)
            result = base.finish_durations(selection, root, batch_dir, split_path)
            checkpoint(status=result["status"])
            base.atomic_json(batch_dir / "coordinator_completion.json", result)
            return result
        except BaseException as exc:
            current.update(status="INTERRUPTED_OR_FAILED_CLOSED", **safe_failure(exc),
                           active_wall_elapsed_s=elapsed_before + monotonic() - started)
            base.atomic_json(worker_progress, current)
            raise


def freeze_input_contract(*, root: Path, selection: dict, amendment: dict,
                          batch_dir: Path, split_path: Path) -> str:
    coverage_path = batch_dir / "window_coverage.json"
    manifest_path = batch_dir / "window_input_session_manifest.json"
    coverage, manifest, split = map(read_json, (coverage_path, manifest_path, split_path))
    manifest_body = dict(manifest)
    manifest_sha = manifest_body.pop("manifest_sha256", None)
    split_body = dict(split)
    split_sha = split_body.pop("split_sha256", None)
    selected = coverage.get("selected")
    if (coverage.get("status") != "WINDOW_FROZEN" or selected not in {"long", "short"}
            or split.get("status") != "FROZEN_METADATA_ONLY" or split_sha != _hash(split_body)
            or manifest_sha != _hash(manifest_body)
            or manifest.get("selection") != "all_official_sessions_for_frozen_meta_train"
            or manifest.get("plan_sha256") != selection["plan_sha256"]
            or manifest.get("split_sha256") != split.get("split_sha256")
            or coverage.get("split_sha256") != split.get("split_sha256")
            or coverage.get("raw_metadata_manifest_sha256") != manifest_sha
            or manifest.get("inventory_sha256") != selection["inventory_sha256"]):
        raise PreparationError("actual_window_evidence_not_frozen")
    expected_sessions = {(row["provider_subject"], str(row["session_ordinal"]), row["name"])
                         for row in selection["files"]}
    actual_sessions = [(row["participant_id"], row["session_id"], row["source_record_id"])
                       for row in manifest["sessions"]]
    if len(actual_sessions) != len(set(actual_sessions)) or set(actual_sessions) != expected_sessions:
        raise PreparationError("window_manifest_not_exact_meta_train_session_selection")
    path = root / "contracts/input_contract.json"
    existing = read_json(path)
    contract = copy.deepcopy(existing)
    receipt = {
        "decision_pool_role": "meta_train", "split_sha256": split["split_sha256"],
        "raw_metadata_manifest_sha256": manifest_sha,
        "window_coverage": {"path": coverage_path.relative_to(root).as_posix(),
                            "sha256": base.sha256_file(coverage_path)[0]},
        "window_input_session_manifest": {"path": manifest_path.relative_to(root).as_posix(),
                                          "sha256": base.sha256_file(manifest_path)[0]},
        "base_selection_plan_sha256": selection["plan_sha256"],
        "operational_execution_plan_sha256": amendment["plan_sha256"],
    }
    for name, key in (("long", "3.0"), ("short", "2.0")):
        row = coverage["candidates"][key]
        receipt[name] = {
            "participant_mean_coverage": row["participant_mean_retention"],
            "participant_mean_abs_class_exclusion_gap": row["participant_mean_absolute_class_exclusion_gap"],
        }
    contract.update(status="window_frozen", selected_profile=selected,
                    context_s=coverage["context_seconds"], crop_s=coverage["crop_seconds"],
                    n_times=coverage["analysis_samples_128hz"], window_audit_receipt=receipt,
                    protocol_freeze_sha256=None)
    contract_sha = validate_signal_contract(contract)
    if existing.get("status") == "window_frozen" and validate_signal_contract(existing) != contract_sha:
        raise PreparationError("existing_frozen_input_contract_differs")
    tasks_path = root / "status/tasks.json"
    tasks = read_json(tasks_path)
    if tasks.get("W7", {}).get("protocol_freeze_sha256") is not None:
        raise PreparationError("unexpected_final_protocol_freeze_requires_review")
    base.atomic_json(path, contract)
    path.chmod(0o444)
    evidence = list(tasks.get("W1", {}).get("evidence", []))
    for artifact in (split_path, coverage_path, manifest_path, path):
        relative = artifact.relative_to(root).as_posix()
        if relative not in evidence:
            evidence.append(relative)
    tasks.setdefault("W1", {}).update(
        status="COMPLETE", qualified_stieger_roster_frozen=True,
        qualification_complete_files=62, qualified_stieger_people=62,
        window_selected=selected, window_decision_pool="meta_train",
        complete_meta_train_sessions=len(actual_sessions), input_contract_sha256=contract_sha,
        evidence=evidence, completed_at_utc=base.utc_now(),
    )
    tasks.setdefault("W2", {}).setdefault("status", "NOT_STARTED_DEPENDS_ON_W1")
    tasks.setdefault("W7", {}).update(protocol_freeze_sha256=None)
    tasks["recorded_at_utc"] = base.utc_now()
    base.atomic_json(tasks_path, tasks)
    return contract_sha


def invoke_cpu_script(script: Path, *, root: Path, output: Path, progress_path: Path,
                      progress: dict, stage: str, contract: str | None = None,
                      poll_seconds: float = POLL_SECONDS, sleep=time.sleep) -> None:
    if not script.is_file():
        raise PreparationError(stage.lower() + "_entrypoint_missing")
    command = [sys.executable, str(script), "--root", str(root), "--progress", str(output), "--execute"]
    if contract is not None:
        command.extend(["--contract", contract])
    progress_write(progress_path, progress, status=stage + "_RUNNING_CPU", downstream_progress=output.relative_to(root).as_posix())
    with subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as child:
        while child.poll() is None:
            progress_write(progress_path, progress, status=stage + "_RUNNING_CPU")
            sleep(poll_seconds)
        if child.returncode != 0:
            progress_write(progress_path, progress, downstream_exit_code=child.returncode)
            raise PreparationError(stage.lower() + "_failed_see_progress")


def coordinate(*, root: Path, execute: bool = False,
               qualification_plan: str = "plans/stieger_qualification.json",
               inventory_path: str = "data_catalog/stieger2021/official_inventory.json",
               progress_path: str = "status/local_preparation.json",
               poll_seconds: float = POLL_SECONDS) -> dict:
    root = root.resolve(strict=True)
    if not 0 < poll_seconds <= 30:
        raise PreparationError("poll_interval_must_be_at_most_30_seconds")
    plan_path, inventory, output = (local(root, rel) for rel in
                                    (qualification_plan, inventory_path, progress_path))
    qualification = read_json(plan_path)
    validate_qualification_plan(qualification, inventory)
    qbatch = root / "audit/stieger_w1/batches" / qualification["plan_sha256"]
    progress = {
        "schema_version": "paper3-local-preparation-v1", "execute": execute,
        "status": "DRY_RUN_NO_NETWORK_NO_WAIT", "started_at_utc": base.utc_now(),
        "qualification_plan_sha256": qualification["plan_sha256"],
        "qualification": count_verified(qualification, root, qbatch),
        "authoritative_worker_progress": worker_snapshot(qbatch),
        "qualification_worker_restart_permitted": False,
        "planned_duration_wall_seconds": EXECUTION_WALL_SECONDS,
        "gpu_jobs_started": 0, "remote_jobs_started": 0, "external_payload_accessed": False,
        "next_steps": ["wait_existing_qualification_lock", "verify_all62_and_freeze_split",
                       "all_official_meta_train_session_duration_metadata_serial_CPU",
                       "measured_window_gate_and_input_contract", "complete_four_source_CPU_cache",
                       "fixed_delivery_package_and_R2_full_GET_receipt"],
    }
    if not execute:
        return progress
    with exclusive_lock(root / "audit/local_preparation/.coordinator.lock"):
        try:
            progress_write(output, progress, status="STARTED_CPU_COORDINATOR")
            split_path = wait_for_qualification(qualification, root=root, batch_dir=qbatch,
                                                inventory=inventory, progress_path=output,
                                                progress=progress, poll_seconds=poll_seconds)
            selection = read_json(qbatch / "meta_duration_plan.json")
            base.validate_plan(selection, inventory, split_path)
            selection_path = root / "plans/stieger_meta_duration_selection.json"
            write_once(selection_path, selection)
            amendment = build_execution_amendment(selection, qualification_sha256=qualification["plan_sha256"])
            validate_execution_amendment(selection, amendment, inventory=inventory,
                                         split_path=split_path, qualification_sha256=qualification["plan_sha256"])
            amendment_path = root / "plans/stieger_meta_duration_execution_amendment.json"
            write_once(amendment_path, amendment)
            batch_dir = root / "audit/stieger_w1/batches" / selection["plan_sha256"]
            result = execute_duration_selection(selection, amendment, root=root, batch_dir=batch_dir,
                                                inventory=inventory, split_path=split_path,
                                                progress_path=output, progress=progress)
            if result.get("status") != "WINDOW_FROZEN":
                raise PreparationError("both_candidate_windows_failed_no_input_contract")
            contract_sha = freeze_input_contract(root=root, selection=selection, amendment=amendment,
                                                 batch_dir=batch_dir, split_path=split_path)
            progress_write(output, progress, status="W1_COMPLETE_WINDOW_FROZEN",
                           input_contract_sha256=contract_sha, final_protocol_freeze_sha256=None)
            invoke_cpu_script(root / "tools/prepare_all_sources.py", root=root,
                              output=root / "audit/source_preparation/progress.json", progress_path=output,
                              progress=progress, stage="SOURCE_PREPARATION", contract="contracts/input_contract.json",
                              poll_seconds=poll_seconds)
            if read_json(root / "status/tasks.json").get("W2", {}).get("status") != "COMPLETE":
                raise PreparationError("source_script_returned_without_complete_W2_evidence")
            validate_signal_contract(read_json(root / "contracts/input_contract.json"))
            invoke_cpu_script(root / "tools/build_delivery.py", root=root,
                              output=root / "audit/delivery/progress.json", progress_path=output,
                              progress=progress, stage="DELIVERY", poll_seconds=poll_seconds)
            progress_write(output, progress, status="CPU_PREPARATION_AND_VERIFIED_DELIVERY_COMPLETE_GPU_NOT_STARTED",
                           completed_at_utc=base.utc_now(), final_protocol_freeze_sha256=None)
        except BaseException as exc:
            progress_write(output, progress, status="STOPPED_CLOSED_MINIMUM_GAP", **safe_failure(exc))
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--qualification-plan", default="plans/stieger_qualification.json")
    parser.add_argument("--inventory-path", default="data_catalog/stieger2021/official_inventory.json")
    parser.add_argument("--progress-path", default="status/local_preparation.json")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--execute", action="store_true")
    try:
        result = coordinate(**vars(parser.parse_args()))
    except Exception as exc:
        result = {"status": "STOPPED_CLOSED_MINIMUM_GAP", **safe_failure(exc), "gpu_jobs_started": 0,
                  "remote_jobs_started": 0, "external_payload_accessed": False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in {"DRY_RUN_NO_NETWORK_NO_WAIT", "CPU_PREPARATION_AND_VERIFIED_DELIVERY_COMPLETE_GPU_NOT_STARTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
