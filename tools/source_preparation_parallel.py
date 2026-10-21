#!/usr/bin/env python3
"""Optional bounded source preparation: two file lifecycles, one native audit.

The original 505-candidate plan, object states, R2 prefixes and durable resource
journal remain unchanged. Concurrency is a separate operational amendment.
Default CLI is a read-only candidate plan. No training or server actions exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import source_preparation_batch as base


OPEN = "PARALLEL_OPEN_FULL_RESERVATION_REMAINS_CHARGED"
UNCLEAN = "UNCLEAN_PARALLEL_FULL_RESERVATION_REMAINS_CHARGED"


class ResourcePreflightPause(BaseException):
    """A retriable resource stop, never a native-schema eligibility judgment."""


class LockedResourceBudget:
    """Two durable leases in the original v2 attempts ledger, under one lock.

    The v2 reader charges every non-COMPLETE attempt its full reservation. Open
    parallel leases therefore survive even an unclean stop without a byte
    refund. The original single active slot is not overwritten or multiplexed.
    """
    def __init__(self, budget):
        self._budget, self._lock = budget, threading.RLock()
        self.path, self.scope = budget.path, budget.scope
        self._open, self._closed = {}, False
        with self._lock:
            for ledger in budget.data["byte_scopes"].values():
                if ledger.get("active") is not None:
                    raise base.BatchError("base_legacy_active_must_be_recovered_before_parallel_facade")
                for attempt in ledger["attempts"]:
                    if attempt.get("status") == OPEN:
                        attempt["status"] = UNCLEAN
            budget._persist()

    def _live(self):
        if self._closed:
            raise base.BatchError("parallel_resource_budget_closed")

    @property
    def data(self):
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return deepcopy(self._budget.data)

    def check(self, *, required_bytes=0, scope=None):
        with self._lock:
            self._live()
            self._budget.check(required_bytes=required_bytes, scope=scope)

    def checkpoint(self):
        with self._lock:
            self._live()
            self._budget.checkpoint()

    def timeout(self):
        with self._lock:
            self._live()
            return self._budget.timeout()

    def reserve_transfer(self, expected_bytes, *, object_id, scope=None, probe_bytes=0):
        with self._lock:
            self._live()
            scope = scope or self.scope
            if (type(expected_bytes) is not int or expected_bytes <= 0 or type(probe_bytes) is not int
                    or probe_bytes < 0 or not isinstance(object_id, str) or not object_id):
                raise base.BatchError("complete_known_positive_GET_body_reservation_required")
            self.check(required_bytes=expected_bytes + probe_bytes, scope=scope)
            if len(self._open) >= 2:
                raise base.BatchError("parallel_GET_reservation_limit_two")
            ledger = self._budget.data["byte_scopes"][scope]
            attempt = {"sequence": ledger["next_sequence"], "object_id": object_id,
                "expected_body_bytes": expected_bytes, "reserved_bytes": expected_bytes + probe_bytes,
                "observed_bytes": 0, "scope": scope,
                "intent": "durably_reserved_before_GET_or_body_read", "status": OPEN}
            ledger["next_sequence"] += 1
            ledger["charged_bytes"] += attempt["reserved_bytes"]
            ledger["attempts"].append(attempt)
            self._open[scope, attempt["sequence"]] = attempt
            self._budget._persist()  # Must finish durably before opening a GET.
            return attempt["sequence"]

    def _lease(self, transfer_id, scope):
        if type(transfer_id) is not int or (scope, transfer_id) not in self._open:
            raise base.BatchError("body_or_completion_requires_matching_open_parallel_GET_reservation")
        return self._open[scope, transfer_id]

    def consume(self, count, *, transfer_id=None, scope=None):
        with self._lock:
            self._live()
            scope = scope or self.scope
            attempt = self._lease(transfer_id, scope)
            if type(count) is not int or count < 0:
                raise base.BatchError("body_read_requires_nonnegative_integer_bytes")
            ledger = self._budget.data["byte_scopes"][scope]
            attempt["observed_bytes"] += count
            ledger["observed_bytes"] += count
            excess = max(0, attempt["observed_bytes"] - attempt["reserved_bytes"])
            if excess:
                attempt["reserved_bytes"] += excess
                ledger["charged_bytes"] += excess
            self._budget._persist()  # Preserve an overread before any cap error.
            self.check(scope=scope)

    def finish_transfer(self, transfer_id, *, complete_body_read=False, scope=None):
        with self._lock:
            self._live()
            scope = scope or self.scope
            attempt = self._lease(transfer_id, scope)
            exact = complete_body_read is True and attempt["observed_bytes"] == attempt["expected_body_bytes"]
            if exact:
                self._budget.data["byte_scopes"][scope]["charged_bytes"] -= attempt["reserved_bytes"] - attempt["observed_bytes"]
            attempt["status"] = "COMPLETE_OBSERVED_BODY" if exact else "FAILED_OR_UNKNOWN_FULL_RESERVATION_REMAINS_CHARGED"
            del self._open[scope, transfer_id]
            self._budget._persist()

    def close(self):
        with self._lock:
            if not self._closed:
                try:
                    self._budget.close()
                finally:
                    self._closed = True


def resource_preflight(root, measured_raw_bytes):
    # This is an actual fully verified raw size, never a candidate-size estimate.
    from tools.continue_local_preparation_parallel import resource_preflight as check
    return check(root, measured_raw_bytes)


class SingleNativeAudit:
    """Only one loadmat/EDF native parser; result already contains only metadata."""
    def __init__(self, root):
        self.root, self._lock = root, threading.Lock()

    def __call__(self, dataset, path, **kwargs):
        failure, result = None, None
        with self._lock:
            try:
                try:
                    resource_preflight(self.root, path.stat().st_size)
                except Exception:
                    raise ResourcePreflightPause("source_parallel_native_resource_preflight_pause") from None
                # Unlike StiegerSession, source audit_native_source returns a
                # summary + trial dicts and no retained EEG/time container.
                result = base.audit_native_source(dataset, path, **kwargs)
                _plain_metadata(result)
            except BaseException as error:
                failure = error
                # A failed loadmat frame may own the entire decoded recording.
                # Keep its exception type/args but never release the parser lock
                # while that traceback or its chained traceback owns the EEG.
                seen, current = set(), error
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    following = current.__cause__ or current.__context__
                    current.__traceback__ = None
                    current.__cause__ = current.__context__ = None
                    current.args = (str(current),) if isinstance(current, base.BatchError) and re.fullmatch(r"[A-Za-z0-9_]+", str(current)) else ()
                    current.__dict__.clear()
                    current = following
                result = None
            finally:
                gc.collect()
        if failure is not None:
            raise failure from None
        return result


def _plain_metadata(value):
    if value is None or isinstance(value, str) or type(value) in (int, float, bool):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _plain_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _plain_metadata(item)
        return
    raise TypeError("native_audit_must_return_only_plain_metadata")


def _staged_priority(root, plan, batch_dir):
    staged = []
    staging_root = root / "data/source_preparation/staging"
    known = {entry["object_id"]: entry for entry in plan["files"]}
    if staging_root.exists():
        for folder in staging_root.iterdir():
            if folder.is_symlink() or not folder.is_dir():
                raise base.BatchError("unknown_source_staging_artifact_requires_inspection")
            contents = list(folder.iterdir())
            if not contents:
                continue
            entry = known.get(folder.name)
            if entry is None or any(p.name not in {entry["file"], entry["file"] + ".partial"}
                                    or p.is_symlink() or not p.is_file() for p in contents):
                raise base.BatchError("unknown_source_staging_artifact_requires_inspection")
    for entry in plan["files"]:
        folder = base.raw_staging_path(root, plan, entry).parent
        if not folder.exists() or not any(folder.iterdir()):
            continue
        state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if (state.get("temporary_raw_owned") is not True
                or state.get("raw_path") != base.raw_staging_path(root, plan, entry).relative_to(root).as_posix()
                or state.get("plan_sha256") != plan["plan_sha256"]
                or state.get("identity") != list(base.identity(entry))):
            raise base.BatchError("unowned_existing_source_staging_requires_inspection")
        staged.append(entry)
    if len(staged) > 2:
        raise base.BatchError("more_than_two_existing_source_staging_lifecycles_preserved")
    ids = {entry["object_id"] for entry in staged}
    return staged + [entry for entry in plan["files"] if entry["object_id"] not in ids]


def _require_accounting_pair(root, batch_dir):
    resource = root / "audit/source_preparation/resource_progress.json"
    wall = resource.with_name(resource.stem + ".wall.json")
    for path in (resource, wall):
        base.no_symlink_components(path)
    if resource.exists() != wall.exists():
        raise base.BatchError("source_existing_byte_and_wall_journals_must_both_survive")
    # Empty directories, locks and a proposal are not evidence of a transfer.
    # Existing object state/partial/final raw and real batch progress are.
    work = any((batch_dir / "objects").glob("*/state.json"))
    work = work or (batch_dir / "batch_progress.json").is_file()
    work = work or any((root / "data/source_preparation/staging").glob("*/*"))
    if not resource.exists() and work:
        raise base.BatchError("source_existing_work_requires_original_byte_and_wall_journals")


def operational_amendment(plan, contract_sha256, resource_snapshot):
    record = {"schema_version": "paper3-source-parallel-operational-amendment-v1",
        "base_plan_sha256": plan["plan_sha256"], "scientific_selection_changed": False,
        "input_contract_sha256": contract_sha256,
        "base_inflight_downloads": plan["inflight_downloads"], "base_parallel_native_parsers": plan["parallel_native_parsers"],
        "inflight_file_lifecycles": 2, "max_concurrent_body_transfers": 2, "parallel_native_parsers": 1,
        "max_staged_file_lifecycles_including_partials": 2, "max_jobs": plan["max_jobs"], "max_gpu_hours": 0,
        "limits": {key: plan[key] for key in ("max_download_bytes", "max_single_raw_bytes", "max_wall_seconds",
            "min_free_disk_bytes", "min_available_ram_bytes", "parser_ram_multiplier")},
        "budget_accounting": "same_v2_journal_multi_open_attempts_full_reserved_until_exact_EOF",
        "effective_byte_limits": {name: row["limit_bytes"] for name, row in resource_snapshot["byte_scopes"].items()},
        "effective_wall_limit_seconds": resource_snapshot["wall_budget"]["max_wall_seconds"],
        "source_worker_sha256": base.file_digest(base.__file__),
        "executor_sha256": base.file_digest(__file__), "training_started": False}
    record["amendment_sha256"] = base.digest(record)
    return record


def execute_plan(plan, *, root=REPO, batch_dir=None, max_new_files=None, client=None, bucket=None,
                 operational_limits=None, operational_amendment_note=None):
    """Explicit execution API for prepare_all_sources(raw_fn=execute_plan).

    CLI calls this only with --execute. No replacement backup/process function
    is accepted: every backup keeps the native before-body instrumentation.
    """
    base.validate_plan(plan)
    root = Path(root).resolve(strict=True)
    contract_sha = base.execution_gate(root)  # Before SDK, staging or mutations.
    if client is None and threading.current_thread() is not threading.main_thread():
        raise base.BatchError("source_shared_SDK_client_must_be_created_in_main_thread")
    if max_new_files is not None and (type(max_new_files) is not int or max_new_files < 1):
        raise base.BatchError("max_new_files_must_be_positive")
    batch_dir = Path(batch_dir) if batch_dir is not None else root / "audit/source_preparation/batches" / plan["plan_sha256"]
    if batch_dir != root / "audit/source_preparation/batches" / plan["plan_sha256"]:
        raise base.BatchError("source_parallel_requires_original_batch_directory")
    base.no_symlink_components(root)
    base.no_symlink_components(batch_dir)
    # The disk bound uses the actual configured upper bound, not four samples.
    if shutil.disk_usage(root).free < 4 * plan["max_single_raw_bytes"] + 1024**3:
        raise base.BatchError("source_parallel_insufficient_global_two_file_staging_disk")
    lock_path = root / "audit/source_preparation/.source_worker.lock"
    base.no_symlink_components(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise base.BatchError("source_batch_already_active") from None
        _require_accounting_pair(root, batch_dir)
        ordered_entries = _staged_priority(root, plan, batch_dir)
        operational_limits = operational_limits or {}
        if set(operational_limits) - {"max_download_bytes", "max_wall_seconds", "max_r2_readback_bytes"}:
            raise base.BatchError("unknown_operational_resource_limit")
        identity_sha = base.digest({"schema": "source-raw-resource-identity-v2", "files": plan["files"],
            "adapter_code_sha256": plan["adapter_code_sha256"], "input_contract_sha256": contract_sha})
        budget = LockedResourceBudget(base.ResourceBudget(plan, root / "audit/source_preparation/resource_progress.json",
            identity_sha256=identity_sha, amendment_note=operational_amendment_note, **operational_limits))
        shared_client, owns_client = None, client is None
        try:
            fixed_bucket = bucket if bucket is not None else os.environ.get("R2_BUCKET")
            if not isinstance(fixed_bucket, str) or not fixed_bucket:
                raise base.BatchError("missing_existing_r2_bucket")
            shared_client = client if client is not None else base.create_client()
            amendment = operational_amendment(plan, contract_sha, budget.snapshot())
            batch_dir.mkdir(parents=True, exist_ok=True)
            amendment_path = batch_dir / ("parallel_operational_amendment_" + amendment["amendment_sha256"] + ".json")
            base.atomic_json(amendment_path, amendment)
            audit = SingleNativeAudit(root)
            return _execute_two(plan, root, batch_dir, contract_sha, budget, audit, shared_client, fixed_bucket,
                                max_new_files, base.descriptor(root, amendment_path), ordered_entries)
        finally:
            try:
                budget.close()  # _execute_two always drains both worker lifecycles.
            finally:
                if owns_client and shared_client is not None:
                    shared_client.close()


def _execute_two(plan, root, batch_dir, contract_sha, budget, audit, client, bucket, max_new_files, amendment, ordered_entries):
    counts = {"new_objects_attempted": 0, "archived_schema_verified": 0, "schema_blocked": 0, "failed": 0}
    entries = iter(ordered_entries)
    status, stopped = "ALL_CANDIDATES_ATTEMPTED", False

    def progress(state):
        base.atomic_json(batch_dir / "batch_progress.json", {"status": state, **counts,
            "plan_sha256": plan["plan_sha256"], "operational_amendment": amendment,
            "inflight_file_lifecycles": 2, "parallel_native_parsers": 1, "training_started": False})

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="paper3-source") as executor:
        pending = {}
        while pending or not stopped:
            while not stopped and len(pending) < 2:
                try:
                    budget.check()
                    entry = next(entries)
                except StopIteration:
                    stopped = True
                    break
                except base.BatchError as error:
                    status, stopped = str(error).upper() + "_NO_STAGE_ADVANCE", True
                    break
                state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
                try:
                    prior = json.loads(state_path.read_text()) if state_path.exists() else {}
                    completed = base.verify_completed(prior, root, entry=entry, plan=plan)
                except Exception:
                    counts["failed"] += 1
                    status, stopped = "PRIOR_COMPLETED_EVIDENCE_INVALID_NO_STAGE_ADVANCE", True
                    progress("RUNNING_BOUNDED_PARALLEL_CPU")
                    break
                if not completed:
                    if max_new_files is not None and counts["new_objects_attempted"] >= max_new_files:
                        status, stopped = "PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE", True
                        break
                    counts["new_objects_attempted"] += 1
                future = executor.submit(base.process_file, entry, root=root, batch_dir=batch_dir, plan=plan,
                    contract_sha256=contract_sha, budget=budget, audit_fn=audit, client=client, bucket=bucket)
                pending[future] = entry["object_id"]
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                object_id = pending.pop(future)
                try:
                    state = future.result()
                    counts["archived_schema_verified" if state["status"] == "COMPLETE" else "schema_blocked"] += 1
                except Exception:
                    counts["failed"] += 1
                except ResourcePreflightPause:
                    counts["failed"] += 1
                    status, stopped = "NATIVE_RESOURCE_PREFLIGHT_PAUSE_NO_STAGE_ADVANCE", True
                # Failed transfer/backup bytes are preserved. They still own a
                # staging slot; do not accumulate additional partial raw files.
                folder = root / "data/source_preparation/staging" / object_id
                if folder.exists() and any(folder.iterdir()):
                    status, stopped = "RETAINED_RAW_OR_PARTIAL_REQUIRES_RESUME_NO_STAGE_ADVANCE", True
            budget.checkpoint()
            progress("RUNNING_BOUNDED_PARALLEL_CPU")
    # Every task has returned before aggregate or the caller closes the budget.
    try:
        inventory = base.aggregate_inventory(plan, root=root, batch_dir=batch_dir, budget=budget)
    except (base.BatchError, OSError, ValueError, KeyError, TypeError) as error:
        status = str(error).upper() + "_NO_STAGE_ADVANCE" if isinstance(error, base.BatchError) else "SOURCE_INVENTORY_EVIDENCE_INVALID_NO_STAGE_ADVANCE"
        inventory = {"status": "SOURCE_RAW_INVENTORY_PARTIAL", "actual_verified_objects": None}
    snapshot = budget.snapshot()
    result = {"status": status, "inventory_status": inventory["status"], **counts,
        "actual_verified_objects": inventory["actual_verified_objects"], "required_target_object_count": 505,
        "plan_sha256": plan["plan_sha256"], "operational_amendment": amendment,
        "download_bytes_read": snapshot["download_bytes_read"], "download_bytes_charged": snapshot["download_bytes_charged"],
        "byte_scopes": {name: {key: ledger[key] for key in ("observed_bytes", "charged_bytes", "limit_bytes")}
                        for name, ledger in snapshot["byte_scopes"].items()},
        "resource_progress_path": budget.path.relative_to(root).as_posix(),
        "active_wall_elapsed_s": snapshot["active_wall_elapsed_s"], "inflight_file_lifecycles": 2,
        "parallel_native_parsers": 1, "numerical_cache_created": False, "training_started": False, "w2_complete": False}
    base.atomic_json(batch_dir / "batch_progress.json", result)
    return result


def main(argv=None):
    from tools import prepare_all_sources as pipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--contract", default="contracts/input_contract.json")
    parser.add_argument("--progress", default="audit/source_full/progress.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        # Replace only raw scheduling. Original exact-W1, serial cache, full-GET
        # backup and W2 transaction remain the canonical orchestration path.
        result = pipeline.prepare_all(args.root, contract=args.contract, progress=args.progress,
                                      execute=args.execute, raw_fn=execute_plan)
        print(json.dumps(result))
        return 0 if result.get("status") == "W2_COMPLETE_CACHE_AND_R2_VERIFIED" or not args.execute else 2
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_NO_STAGE_ADVANCE", "error_type": type(error).__name__,
            "gap": str(error) if isinstance(error, (base.BatchError, pipeline.PreparationError)) else "operation_failed_details_suppressed",
            "training_started": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
