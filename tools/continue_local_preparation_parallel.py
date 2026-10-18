#!/usr/bin/env python3
"""Opt-in two-file CPU/R2 continuation; default is a read-only proposal.

Two complete file lifecycles share one native parser. This module never stops
the serial coordinator, changes scientific selection, starts a GPU or opens
external data. Actual cutover requires the old coordinator and batch locks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work import window_acceptance
from p3work.recovery_budget import PersistentWallBudget
from tools import continue_local_preparation as serial
from tools import r2_backup as r2

base = serial.base
PreparationError = serial.PreparationError
PINNED_SELECTION_SHA = "0c53f43ca0f4eb424796d3d7c83c736936bc886f92b0555c35811b4daa306c37"
PINNED_PRIOR_AMENDMENT_SHA = "dd3233cbaa432f853b8ab7e68db0b1dd1a75bf374355bb2747db306cb5e9c64c"
SELECTED_FILES = 396
MAX_FILE_WORKERS = 2
MAX_NATIVE_PARSERS = 1
WALL_SECONDS = 64800
MEMORY_CAP_BYTES = 16 * 1024**3
HEARTBEAT_SECONDS = 2
OPERATIONAL_PATH = "plans/stieger_meta_duration_parallel_operational_amendment.json"
POLICY_FILES = ("protocol/WORK_TASKS.md", "protocol/EXPERIMENT_PROTOCOL.md")
CODE_FILES = ("tools/continue_local_preparation.py", "tools/stieger_metadata_batch.py", "tools/r2_backup.py",
              "p3work/adapters/stieger_native.py", "p3work/window.py", "p3work/window_acceptance.py",
              "p3work/splits.py", "p3work/recovery_budget.py")


def _descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": base.sha256_file(path)[0]}


def _checked_descriptor(root, value):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise PreparationError("explicit_local_path_sha256_required")
    path = serial.local(root, value["path"])
    if base.sha256_file(path)[0] != value["sha256"]:
        raise PreparationError("operational_reference_sha256_changed")
    return path


def build_operational_amendment(root, selection, prior, *, split_path):
    """A separate concurrency amendment; the original two plans stay intact."""
    value = {"schema_version": "paper3-two-file-cpu-operational-v1", "kind": "concurrency_operational_only",
        "base_selection_plan_sha256": selection["plan_sha256"], "file_selection_sha256": serial._hash(selection["files"]),
        "prior_operational_execution_plan_sha256": prior["plan_sha256"],
        "prior_operational_execution_plan": _descriptor(root, root / "plans/stieger_meta_duration_execution_amendment.json"),
        "selection_plan": _descriptor(root, root / "plans/stieger_meta_duration_selection.json"),
        "participant_split": _descriptor(root, split_path), "split_sha256": selection["split_sha256"],
        "max_file_workers": MAX_FILE_WORKERS, "max_native_parsers": MAX_NATIVE_PARSERS,
        "max_inflight_complete_file_lifecycles": MAX_FILE_WORKERS, "unbounded_prefetch_permitted": False,
        "max_jobs": selection["max_jobs"], "max_download_bytes": selection["max_download_bytes"],
        "byte_budget_scope": "unchanged distinct official object-selection size; retries and R2 verification are not this field",
        "max_wall_seconds": WALL_SECONDS, "max_gpu_hours": 0,
        "wall_budget": "inherit_original_batch_progress_high_water_then_durable_dual_clock_no_reset",
        "legacy_unclosed_wall_interval": "blocked_until_authoritative_graceful_handover",
        "resource_cap_bytes": MEMORY_CAP_BYTES, "raw_staging_bound_files": 2,
        "qualification_reuse": "unchanged_strict_three_object_full_GET_provenance",
        "policy": {p: base.sha256_file(root / p)[0] for p in POLICY_FILES},
        "code_dependencies": {p: base.sha256_file(root / p)[0] for p in CODE_FILES},
        "source_continuation": _descriptor(root, root / "tools/source_preparation_parallel.py"),
        "scheduler_code_sha256": base.sha256_file(Path(__file__))[0],
        "scientific_selection_changed": False, "external_payload_accessed": False,
        "gpu_training": False, "remote_server_actions": False, "final_protocol_freeze_sha256": None}
    value["plan_sha256"] = serial._hash(value)
    return value


def _inputs(root):
    inventory = serial.local(root, "data_catalog/stieger2021/official_inventory.json")
    qualification = serial.read_json(serial.local(root, "plans/stieger_qualification.json"))
    serial.validate_qualification_plan(qualification, inventory)
    qbatch = root / "audit/stieger_w1/batches" / qualification["plan_sha256"]
    split_path = qbatch / "participant_split.json"
    selection = serial.read_json(serial.local(root, "plans/stieger_meta_duration_selection.json"))
    prior = serial.read_json(serial.local(root, "plans/stieger_meta_duration_execution_amendment.json"))
    serial.validate_execution_amendment(selection, prior, inventory=inventory,
        split_path=split_path, qualification_sha256=qualification["plan_sha256"])
    if (selection["plan_sha256"] != PINNED_SELECTION_SHA or prior["plan_sha256"] != PINNED_PRIOR_AMENDMENT_SHA
            or len(selection["files"]) != SELECTED_FILES or selection["max_jobs"] != SELECTED_FILES
            or prior["max_wall_seconds"] != WALL_SECONDS
            or len({entry["id"] for entry in selection["files"]}) != SELECTED_FILES
            or len({entry["name"] for entry in selection["files"]}) != SELECTED_FILES):
        raise PreparationError("exact_396_selection_and_original_18h_amendment_required")
    return selection, prior, qualification, inventory, split_path, qbatch


def _plain_metadata(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if type(value) in {list, tuple}:
        for item in value:
            _plain_metadata(item)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for item in value.values():
            _plain_metadata(item)
        return
    raise PreparationError("native_proxy_must_contain_only_plain_finite_metadata")


@dataclass(frozen=True, slots=True)
class TinyMetadataSession:
    rows: list
    audit: dict

    def schema_audit(self):
        return self.audit


def serialized_native_loader(parser_lock, load_session=None):
    native_load = base.load_session if load_session is None else load_session

    def load(path, *, expected_sha256):
        native = proxy = failure = None
        with parser_lock:
            try:
                native = native_load(path, expected_sha256=expected_sha256)
                audit, rows = native.schema_audit(), native.rows
                _plain_metadata(audit); _plain_metadata(rows)
                proxy = TinyMetadataSession(rows, audit)
            except BaseException as error:
                # Keep the native exception class for base's MemoryError/schema
                # handling, but sever loadmat frames/large args before unlocking.
                failure = error
                safe_args = (str(error),) if isinstance(error, base.BatchError) and serial.re.fullmatch(r"[a-z0-9_]+", str(error)) else ()
                error.args = safe_args
                error.__traceback__ = error.__context__ = error.__cause__ = None
                error.__dict__.clear()
            finally:
                native = None
                if failure is not None:
                    audit = rows = proxy = None
                gc.collect()
        if failure is not None:
            raise failure from None
        return proxy
    return load


def _resource_observation(root):
    folder = Path("/sys/fs/cgroup")
    limit_text = (folder / "memory.max").read_text().strip()
    if not limit_text.isdecimal():
        raise PreparationError("finite_cgroup_v2_memory_limit_required")
    stats = {key: int(value) for key, value in (line.split() for line in (folder / "memory.stat").read_text().splitlines())}
    host = {key.rstrip(":"): int(value.split()[0]) * 1024 for key, value in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    return {"cgroup_memory_limit_bytes": int(limit_text),
        "cgroup_memory_current_bytes": int((folder / "memory.current").read_text()),
        "cgroup_memory_stat": stats, "host_available_bytes": host["MemAvailable"],
        "disk_free_bytes": shutil.disk_usage(root).free}


def resource_preflight(root: Path, max_raw_bytes: int) -> dict:
    """Two staged raw files, one parser; cgroup is authoritative, not host RAM."""
    if type(max_raw_bytes) is not int or max_raw_bytes <= 0:
        raise PreparationError("positive_official_max_raw_size_required")
    observed = _resource_observation(root)
    maximum = min(observed["cgroup_memory_limit_bytes"], MEMORY_CAP_BYTES)
    current, stat = observed["cgroup_memory_current_bytes"], observed["cgroup_memory_stat"]
    needed = {"file", "shmem", "file_dirty", "file_writeback", "unevictable", "active_file", "inactive_file"}
    if (maximum <= 0 or current < 0 or not needed <= set(stat)
            or any(type(stat[key]) is not int or stat[key] < 0 for key in needed)):
        raise PreparationError("complete_finite_cgroup_memory_accounting_required")
    reclaim = max(0, min(stat["file"] - stat["shmem"] - stat["file_dirty"] - stat["file_writeback"] - stat["unevictable"],
                         stat["active_file"] + stat["inactive_file"]))
    available = min(maximum, observed["host_available_bytes"], max(0, maximum - current + reclaim))
    memory_required = max(3 * 1024**3, 6 * max_raw_bytes) + 1024**3 + 2 * (r2.CHUNK_SIZE + base.CHUNK)
    disk_required = 4 * max_raw_bytes + 1024**3
    result = {**observed, "effective_memory_cap_bytes": maximum, "clean_file_cache_reclaimable_bytes": reclaim,
        "effective_available_memory_bytes": available, "required_available_memory_bytes": memory_required,
        "required_disk_free_bytes": disk_required, "maximum_staged_raw_files": 2, "maximum_native_parsers": 1,
        "drop_caches_or_service_changes_permitted": False}
    if available < memory_required:
        raise PreparationError("cgroup_headroom_insufficient_for_two_files_one_parser")
    if observed["disk_free_bytes"] < disk_required:
        raise PreparationError("disk_headroom_insufficient_for_two_file_staging")
    return result


def verify_object(entry, state, *, root, plan, qualification):
    if not state or state.get("status") not in {"COMPLETE", "SCHEMA_BLOCKED"}:
        return False
    if not serial.verify_object(entry, state, root=root, plan=plan):
        return False
    obj = root / "audit/stieger_w1/batches" / plan["plan_sha256"] / "objects" / str(entry["id"])
    window_acceptance._verify_state_evidence(base, root, entry, obj, state, plan["plan_sha256"])
    source, _ = window_acceptance._qualification_source(base, root, entry, obj, state, qualification)
    window_acceptance._verify_three_object_backup(root, entry, obj, state, source)
    return True


def _shared_backup():
    if threading.current_thread() is not threading.main_thread():
        raise PreparationError("R2_client_must_be_initialized_by_main_thread")
    client = r2.create_client()
    bucket = os.environ.get("R2_BUCKET")
    if not bucket:
        raise PreparationError("existing_R2_bucket_required")
    def backup(**kwargs):
        return r2.run_backup(**kwargs, client=client, bucket=bucket)
    return backup, client


def _staged_objects(root, selection, batch_dir):
    """Stat only; unknown/unowned leftovers never become free staging slots."""
    folder = serial.local(root, "data/stieger2021/w1-batch-temp")
    if not folder.exists():
        return set()
    entries = {str(entry["id"]): entry for entry in selection["files"]}
    occupied = set()
    for directory in folder.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            raise PreparationError("unknown_or_symlink_temporary_staging_preserved")
        files = list(directory.iterdir())
        if not files:
            continue
        entry = entries.get(directory.name)
        if entry is None:
            raise PreparationError("temporary_object_outside_exact_selection_preserved")
        allowed = {entry["name"], entry["name"] + ".partial"}
        if any(path.is_symlink() or not path.is_file() or path.name not in allowed or path.stat().st_size > entry["size"] for path in files):
            raise PreparationError("temporary_object_path_or_official_size_invalid_preserved")
        state = serial.object_state(entry, batch_dir)
        if (not state or state.get("plan_sha256") != selection["plan_sha256"]
                or state.get("file_id") != entry["id"] or state.get("name") != entry["name"]
                or state.get("temporary_raw_owned") is not True
                or state.get("raw_path") != f"data/stieger2021/w1-batch-temp/{entry['id']}/{entry['name']}"):
            raise PreparationError("temporary_raw_ownership_not_authorized_preserved")
        occupied.add(entry["id"])
    if len(occupied) > MAX_FILE_WORKERS:
        raise PreparationError("existing_temporary_raw_exceeds_two_file_bound_preserved")
    return occupied


def _run_locked(*, root, selection, amendment, qualification, inventory, batch_dir,
                progress_path, progress, backup_factory=None, process_fn=None,
                preflight=None, monotonic=time.monotonic):
    """Both original locks are owned by coordinate; only this thread journals."""
    worker_progress = batch_dir / "batch_progress.json"
    if not worker_progress.is_file():
        raise PreparationError("authoritative_legacy_batch_progress_required_no_zero_restart")
    prior = serial.read_json(worker_progress)
    floor = prior.get("active_wall_elapsed_s")
    if (prior.get("plan_sha256") != selection["plan_sha256"] or type(floor) not in {int, float}
            or not math.isfinite(floor) or floor < 0):
        raise PreparationError("legacy_wall_high_water_or_selection_invalid")
    journal_path = batch_dir / "parallel_wall_budget.json"
    was_parallel = "parallel_operational_plan_sha256" in prior
    if was_parallel and (prior["parallel_operational_plan_sha256"] != amendment["plan_sha256"] or not journal_path.is_file()):
        raise PreparationError("parallel_budget_evidence_missing_or_changed_no_reset")
    if journal_path.exists() and not was_parallel:
        raise PreparationError("parallel_journal_without_matching_progress_no_reset")
    if not was_parallel and prior.get("status") in {"RUNNING_SERIAL_CPU", "RUNNING", "DOWNLOADING_OR_VERIFYING"}:
        raise PreparationError("legacy_unclosed_wall_interval_requires_graceful_handover")
    snapshot_path = batch_dir / "parallel_cutover_progress.json"
    if not was_parallel:
        serial.write_once(snapshot_path, prior)
    cutover = _descriptor(root, snapshot_path)
    cutover_body = serial.read_json(_checked_descriptor(root, cutover))
    if cutover_body.get("plan_sha256") != selection["plan_sha256"] or cutover_body.get("active_wall_elapsed_s", -1) > floor:
        raise PreparationError("legacy_cutover_high_water_changed")
    check_resource = resource_preflight if preflight is None else preflight
    counts = {"verified_complete_files": 0, "verified_schema_unknown_files": 0,
              "verified_official_bytes": 0, "selected_files": len(selection["files"])}
    completed = set()
    current = {"plan_sha256": selection["plan_sha256"], "operational_execution_plan_sha256": amendment["plan_sha256"],
        "parallel_operational_plan_sha256": amendment["plan_sha256"], "legacy_cutover_progress": cutover,
        "max_wall_seconds": WALL_SECONDS, "max_jobs": selection["max_jobs"],
        "max_download_bytes": selection["max_download_bytes"], "max_gpu_hours": 0,
        "max_file_workers": 2, "max_native_parsers": 1, "final_protocol_freeze_sha256": None}
    process = base.process_file if process_fn is None else process_fn
    pending = None
    active, scheduled = {}, set(completed)
    failure = None
    parser_lock = threading.Lock()
    load_fn = serialized_native_loader(parser_lock)
    client = None
    with PersistentWallBudget(journal_path, amendment["plan_sha256"], WALL_SECONDS,
                              checkpoint_wall_seconds=floor, monotonic=monotonic) as journal:
        def checkpoint(status="RUNNING_TWO_FILE_CPU"):
            journal.checkpoint()
            current.update(status=status, active_wall_elapsed_s=journal.wall_seconds,
                active_file_ids=sorted(entry["id"] for entry in active.values()), **counts,
                incomplete_files=len(selection["files"]) - len(completed))
            base.atomic_json(worker_progress, current)
            serial.progress_write(progress_path, progress, status=status, duration=dict(current))
        try:
            check_resource(root, max(entry["size"] for entry in selection["files"]))
            for entry in selection["files"]:
                base.reuse_qualification_object(entry, root=root, batch_dir=batch_dir, plan=selection, inventory_path=inventory)
                state = serial.object_state(entry, batch_dir)
                if verify_object(entry, state, root=root, plan=selection, qualification=qualification):
                    completed.add(entry["id"])
                    counts["verified_complete_files"] += 1; counts["verified_official_bytes"] += entry["size"]
            occupied = _staged_objects(root, selection, batch_dir)
            if occupied & completed:
                raise PreparationError("completed_temporary_raw_still_occupies_staging_slot_preserved")
            scheduled.update(completed)
            # Resume retained partials first; the finisher still uses the exact
            # unchanged scientific plan order and participant/session roster.
            pending = [entry for entry in selection["files"] if entry["id"] not in completed]
            pending.sort(key=lambda entry: entry["id"] not in occupied)
            checkpoint()
            if journal.exhausted:
                raise PreparationError("duration_operational_wall_budget_exhausted")
            if counts["verified_complete_files"] == len(selection["files"]):
                return {"status": "ALL_DURATION_OBJECTS_VERIFIED", **counts}
            backup_fn, client = (_shared_backup if backup_factory is None else backup_factory)()
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="p3-stieger-file") as executor:
                exhausted_pending = False
                while active or not exhausted_pending:
                    if failure is not None:
                        # Clock failure must not prevent wait/pop: querying a
                        # permanently blocked journal here would busy-loop even
                        # after the existing workers had safely completed.
                        if not active:
                            break
                        done, _ = wait(active, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
                        for future in done:
                            entry = active.pop(future)
                            try:
                                state = future.result()
                                if verify_object(entry, state, root=root, plan=selection, qualification=qualification):
                                    completed.add(entry["id"])
                                    counts["verified_complete_files"] += 1
                                    counts["verified_official_bytes"] += entry["size"]
                            except BaseException:
                                pass  # The first failure remains authoritative.
                        continue
                    try:
                        if journal.exhausted:
                            failure = failure or PreparationError("duration_operational_wall_budget_exhausted")
                        while failure is None and len(active) < 2 and not exhausted_pending:
                            if not pending:
                                exhausted_pending = True; break
                            entry = pending[0]
                            occupied = _staged_objects(root, selection, batch_dir)
                            reserved = occupied | {row["id"] for row in active.values()}
                            if entry["id"] not in reserved and len(reserved) >= MAX_FILE_WORKERS:
                                if not active:
                                    raise PreparationError("retained_temporary_raw_blocks_new_staging_slot")
                                break
                            if entry["id"] in scheduled:
                                raise PreparationError("duplicate_object_dispatch_forbidden")
                            check_resource(root, max(row["size"] for row in selection["files"]))
                            pending.pop(0)
                            scheduled.add(entry["id"])
                            future = executor.submit(process, entry, root=root, batch_dir=batch_dir, plan=selection,
                                backup_fn=backup_fn, load_fn=load_fn)
                            active[future] = entry
                            checkpoint()
                        if not active:
                            break
                        done, _ = wait(active, timeout=HEARTBEAT_SECONDS, return_when=FIRST_COMPLETED)
                        for future in done:
                            entry = active.pop(future)
                            try:
                                state = future.result()
                                if not verify_object(entry, state, root=root, plan=selection, qualification=qualification):
                                    raise PreparationError("processed_object_missing_strict_three_object_proof")
                                completed.add(entry["id"])
                                counts["verified_complete_files"] += 1; counts["verified_official_bytes"] += entry["size"]
                            except BaseException as error:
                                failure = failure or error
                        if counts["verified_official_bytes"] > selection["max_download_bytes"] or len(completed) > selection["max_jobs"]:
                            failure = failure or PreparationError("unchanged_official_job_or_byte_budget_exceeded")
                        checkpoint("DRAINING_INFLIGHT_AFTER_FAILURE" if failure else "RUNNING_TWO_FILE_CPU")
                    except BaseException as error:
                        failure = failure or error
                        if not active:
                            break
                        # No new submission after any observed failure; the two
                        # existing bounded file lifecycles keep their own state.
                if failure is not None:
                    raise failure
            checkpoint("ALL_DURATION_OBJECTS_VERIFIED")
            return {"status": "ALL_DURATION_OBJECTS_VERIFIED", **counts}
        except BaseException as error:
            clock_report = journal.report()
            current.update(status="INTERRUPTED_OR_FAILED_CLOSED", active_wall_elapsed_s=clock_report["wall_seconds"],
                           operational_wall_budget=clock_report, **serial.safe_failure(error), **counts)
            base.atomic_json(worker_progress, current)
            serial.progress_write(progress_path, progress, status=current["status"], duration=dict(current), **serial.safe_failure(error))
            raise
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()


def coordinate(*, root: Path, execute=False, expected_operational_sha256=None):
    root = root.resolve(strict=True)
    selection, prior, qualification, inventory, split_path, qbatch = _inputs(root)
    amendment = build_operational_amendment(root, selection, prior, split_path=split_path)
    result = {"status": "DRY_RUN_NO_NETWORK_NO_NEW_STATE", "execute": execute,
        "operational_amendment_proposal": amendment, "actual_cutover_performed": False,
        "old_process_stop_or_restart_permitted": False, "gpu_jobs_started": 0, "remote_jobs_started": 0,
        "external_payload_accessed": False, "final_protocol_freeze_sha256": None}
    if not execute:
        return result
    if expected_operational_sha256 != amendment["plan_sha256"]:
        raise PreparationError("explicit_reviewed_parallel_operational_SHA_required")
    batch_dir = root / "audit/stieger_w1/batches" / selection["plan_sha256"]
    with serial.exclusive_lock(root / "audit/local_preparation/.coordinator.lock"), \
            serial.exclusive_lock(batch_dir / ".batch.lock", "duration_batch_already_active"):
        progress = dict(result)
        try:
            return _continue_after_handover(root=root, selection=selection, prior=prior, qualification=qualification,
                inventory=inventory, split_path=split_path, qbatch=qbatch, batch_dir=batch_dir,
                amendment=amendment, progress=progress)
        except BaseException as error:
            serial.progress_write(root / "status/local_preparation.json", progress,
                status="STOPPED_CLOSED_MINIMUM_GAP", **serial.safe_failure(error), final_protocol_freeze_sha256=None)
            raise


def _continue_after_handover(*, root, selection, prior, qualification, inventory, split_path, qbatch,
                             batch_dir, amendment, progress):
        # Repeat local authority checks only after exclusive handover.
        fresh = _inputs(root)
        if fresh[0] != selection or fresh[1] != prior:
            raise PreparationError("selection_changed_while_acquiring_handover_locks")
        counts = serial.count_verified(qualification, root, qbatch)
        if counts["verified_complete_files"] != 62 or counts["verified_schema_unknown_files"] or counts["incomplete_files"]:
            raise PreparationError("all_62_qualification_objects_must_remain_verified")
        serial.write_once(root / OPERATIONAL_PATH, amendment)
        progress_path = root / "status/local_preparation.json"
        progress["actual_cutover_performed"] = True
        _run_locked(root=root, selection=selection, amendment=amendment, qualification=qualification,
            inventory=inventory, batch_dir=batch_dir, progress_path=progress_path, progress=progress)
        # The exact same native metadata/window/source/delivery finishers run
        # only after all original selected objects pass the strict proof chain.
        saved = serial.read_json(batch_dir / "batch_progress.json")
        with PersistentWallBudget(batch_dir / "parallel_wall_budget.json", amendment["plan_sha256"], WALL_SECONDS,
                                  checkpoint_wall_seconds=saved["active_wall_elapsed_s"]) as journal:
            base.validate_plan(selection, inventory, split_path)
            finished = base.finish_durations(selection, root, batch_dir, split_path)
            journal.checkpoint()
            if journal.exhausted:
                raise PreparationError("duration_wall_budget_exhausted_before_window_freeze")
            if finished.get("status") != "WINDOW_FROZEN":
                raise PreparationError("both_candidate_windows_failed_no_input_contract")
            contract_sha = serial.freeze_input_contract(root=root, selection=selection, amendment=amendment,
                batch_dir=batch_dir, split_path=split_path)
            journal.checkpoint()
            saved.update(status="WINDOW_FROZEN", active_wall_elapsed_s=journal.wall_seconds)
            base.atomic_json(batch_dir / "batch_progress.json", saved)
            if journal.exhausted:
                raise PreparationError("duration_wall_budget_exhausted_after_window_freeze")
        serial.progress_write(progress_path, progress, status="W1_COMPLETE_WINDOW_FROZEN",
                              input_contract_sha256=contract_sha, final_protocol_freeze_sha256=None)
        serial.invoke_cpu_script(root / "tools/source_preparation_parallel.py", root=root,
            output=root / "audit/source_preparation/progress.json", progress_path=progress_path,
            progress=progress, stage="SOURCE_PREPARATION", contract="contracts/input_contract.json")
        if serial.read_json(root / "status/tasks.json").get("W2", {}).get("status") != "COMPLETE":
            raise PreparationError("source_script_returned_without_complete_W2_evidence")
        serial.invoke_cpu_script(root / "tools/build_delivery.py", root=root,
            output=root / "audit/delivery/progress.json", progress_path=progress_path, progress=progress, stage="DELIVERY")
        serial.progress_write(progress_path, progress,
            status="CPU_PREPARATION_AND_VERIFIED_DELIVERY_COMPLETE_GPU_NOT_STARTED", final_protocol_freeze_sha256=None)
        return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-operational-sha256")
    try:
        result = coordinate(**vars(parser.parse_args()))
    except BaseException as error:
        result = {"status": "STOPPED_CLOSED_MINIMUM_GAP", **serial.safe_failure(error),
                  "gpu_jobs_started": 0, "remote_jobs_started": 0, "external_payload_accessed": False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in {"DRY_RUN_NO_NETWORK_NO_NEW_STATE", "CPU_PREPARATION_AND_VERIFIED_DELIVERY_COMPLETE_GPU_NOT_STARTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
