#!/usr/bin/env python3
"""Optional source scheduling: serial 178 MAT objects, then <=16 small EDFs.

This is an operational amendment to the same 505-object source plan, locks,
object states, R2 prefixes and durable byte/wall journals. No scientific choice,
resource cap, credential, dataset URL or original implementation is rewritten.
The CLI defaults to a read-only candidate plan; only --execute performs work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import partial
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import source_preparation_parallel as parallel
from tools import r2_backup

base = parallel.base
SMALL_LIMIT = 16 * 1024**2
MAX_WORKERS = 16


def _limits(max_file_workers, max_small_raw_bytes):
    if type(max_file_workers) is not int or not 1 <= max_file_workers <= MAX_WORKERS:
        raise base.BatchError("small_source_workers_must_be_integer_one_through_sixteen")
    if type(max_small_raw_bytes) is not int or not 1 <= max_small_raw_bytes <= SMALL_LIMIT:
        raise base.BatchError("small_source_limit_must_be_positive_integer_at_most_16MiB")


class SmallFileResourceBudget(parallel.LockedResourceBudget):
    """The original v2 full-body ledger, with <=16 total cross-scope leases.

    All inherited methods, wall recovery, unknown-tail charges and per-scope
    limits remain unchanged. A reservation is durable before a caller can GET.
    """
    def __init__(self, budget, max_open=MAX_WORKERS):
        _limits(max_open, SMALL_LIMIT)
        self.max_open = max_open
        super().__init__(budget)

    def reserve_transfer(self, expected_bytes, *, object_id, scope=None, probe_bytes=0):
        with self._lock:
            self._live()
            scope = scope or self.scope
            if (type(expected_bytes) is not int or expected_bytes <= 0 or type(probe_bytes) is not int
                    or probe_bytes < 0 or not isinstance(object_id, str) or not object_id):
                raise base.BatchError("complete_known_positive_GET_body_reservation_required")
            self.check(required_bytes=expected_bytes + probe_bytes, scope=scope)
            if len(self._open) >= self.max_open:
                raise base.BatchError("small_parallel_GET_reservation_limit_exceeded")
            ledger = self._budget.data["byte_scopes"][scope]
            attempt = {"sequence": ledger["next_sequence"], "object_id": object_id,
                "expected_body_bytes": expected_bytes, "reserved_bytes": expected_bytes + probe_bytes,
                "observed_bytes": 0, "scope": scope,
                "intent": "durably_reserved_before_GET_or_body_read", "status": parallel.OPEN}
            ledger["next_sequence"] += 1
            ledger["charged_bytes"] += attempt["reserved_bytes"]
            ledger["attempts"].append(attempt)
            self._open[scope, attempt["sequence"]] = attempt
            self._budget._persist()
            return attempt["sequence"]


def _small_observation(observation, max_small_raw_bytes):
    if (not isinstance(observation, dict) or observation.get("http_status") != 200
            or type(observation.get("bytes")) is not int or observation["bytes"] <= 0
            or observation["bytes"] > max_small_raw_bytes):
        raise base.BatchError("PhysionetMI_fresh_HEAD_must_be_positive_and_at_most_16MiB")
    return observation


class SmallOfficialOpener:
    """Recheck every internal HEAD/GET bound before any official body read.

    process_file still performs its own fresh HEAD and original strict validator
    checks. An earlier admission HEAD cannot authorize a subsequently larger
    object, a different URL or a GET without a fresh HEAD in this lifecycle.
    """
    def __init__(self, entry, *, max_small_raw_bytes=SMALL_LIMIT):
        _limits(1, max_small_raw_bytes)
        base.validate_entry(entry)
        if entry["dataset"] != "PhysionetMI" or not entry["file"].endswith(".edf"):
            raise base.BatchError("small_parallel_pool_is_PhysionetMI_EDF_only")
        self.entry, self.limit, self.head_seen = entry, max_small_raw_bytes, False

    def __call__(self, request, **kwargs):
        method = request.get_method()
        if request.full_url != self.entry["url"] or method not in {"HEAD", "GET"}:
            raise base.BatchError("small_source_request_must_match_exact_official_object")
        if method == "GET" and not self.head_seen:
            raise base.BatchError("small_source_GET_requires_this_lifecycle_fresh_HEAD")
        response = base.urllib.request.urlopen(request, **kwargs)
        try:
            length = response.headers.get("Content-Length")
            if (not isinstance(length, str) or re.fullmatch(r"[1-9][0-9]*", length) is None
                    or int(length) > self.limit):
                raise base.BatchError("small_source_HEAD_and_GET_ContentLength_must_not_exceed_16MiB")
            if method == "HEAD":
                if response.status != 200:
                    raise base.BatchError("small_source_fresh_HEAD_requires_200")
                self.head_seen = True
            return response
        except BaseException:
            response.close()
            raise


def inspect_staging(root, plan, batch_dir, *, contract_sha256,
                    max_file_workers=MAX_WORKERS, max_small_raw_bytes=SMALL_LIMIT):
    """Preserve and validate every original owned partial/complete staging file.

    The union of existing and new file lifecycles is at most 16. At most one
    existing non-Physio recovery is allowed and occupies one of those slots;
    it must finish before the small-file phase can start.
    """
    _limits(max_file_workers, max_small_raw_bytes)
    root, batch_dir = Path(root), Path(batch_dir)
    staging = root / "data/source_preparation/staging"
    base.no_symlink_components(staging)
    known = {entry["object_id"]: entry for entry in plan["files"]}
    large, small = [], []
    if not staging.exists():
        return large, small
    for folder in sorted(staging.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            raise base.BatchError("unknown_source_staging_artifact_requires_inspection")
        contents = list(folder.iterdir())
        if not contents:
            continue
        entry = known.get(folder.name)
        if entry is None or any(p.name not in {entry["file"], entry["file"] + ".partial"}
                                or p.is_symlink() or not p.is_file() for p in contents):
            raise base.BatchError("unknown_source_staging_artifact_requires_inspection")
        base.validate_entry(entry)
        state_path = batch_dir / "objects" / entry["object_id"] / "state.json"
        base.no_symlink_components(state_path)
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        observation = state.get("head_observation", {})
        if (state.get("temporary_raw_owned") is not True
                or state.get("raw_path") != base.raw_staging_path(root, plan, entry).relative_to(root).as_posix()
                or state.get("plan_sha256") != plan["plan_sha256"]
                or state.get("identity") != list(base.identity(entry))
                or state.get("input_contract_sha256") != contract_sha256
                or type(observation.get("bytes")) is not int or observation["bytes"] <= 0
                or observation["bytes"] > plan["max_single_raw_bytes"]
                or any(p.stat().st_size > observation["bytes"] for p in contents)):
            raise base.BatchError("unowned_or_unverified_existing_source_staging_requires_inspection")
        if entry["dataset"] == "PhysionetMI":
            _small_observation(observation, max_small_raw_bytes)
            small.append(entry)
        else:
            large.append(entry)
    if len(large) > 1 or len(large) + len(small) > max_file_workers:
        raise base.BatchError("existing_source_staging_exceeds_global_sixteen_or_serial_one")
    return large, small


def resource_preflight(root, measured_raw_bytes, *, max_file_workers):
    """Cgroup preflight from measured size or the explicit entry upper bound."""
    _limits(max_file_workers, SMALL_LIMIT)
    try:
        observed = parallel.resource_preflight(root, measured_raw_bytes)
    except parallel.ResourcePreflightPause:
        raise
    except Exception:
        raise parallel.ResourcePreflightPause("source_small_cgroup_resource_preflight_pause") from None
    extra = max(0, max_file_workers - 2) * (base.CHUNK + r2_backup.CHUNK_SIZE)
    required = observed["required_available_memory_bytes"] + extra
    if observed["effective_available_memory_bytes"] < required:
        raise parallel.ResourcePreflightPause("source_small_stream_buffer_cgroup_headroom_insufficient")
    return {**observed, "additional_small_stream_buffer_bytes": extra,
        "base_required_available_memory_bytes": observed["required_available_memory_bytes"],
        "required_available_memory_bytes": required, "maximum_staged_raw_files": max_file_workers,
        "small_stream_buffer_capacity": max_file_workers,
        "disk_preflight_scope": "parser_local_original_gate; additional global small staging bound checked at entry"}


class BoundedNativeAudit:
    def __init__(self, root, max_file_workers=MAX_WORKERS):
        _limits(max_file_workers, SMALL_LIMIT)
        self.root, self.workers, self._lock = root, max_file_workers, threading.Lock()
        self._audit = parallel.SingleNativeAudit(root)

    def __call__(self, dataset, path, **kwargs):
        with self._lock:
            try:
                resource_preflight(self.root, path.stat().st_size,
                    max_file_workers=self.workers if dataset == "PhysionetMI" else 1)
            except parallel.ResourcePreflightPause:
                raise
            except Exception:
                raise parallel.ResourcePreflightPause("source_small_native_resource_preflight_pause") from None
            return self._audit(dataset, path, **kwargs)


def operational_amendment(plan, contract_sha256, snapshot, *, max_file_workers, max_small_raw_bytes):
    record = parallel.operational_amendment(plan, contract_sha256, snapshot)
    record.pop("amendment_sha256")
    record.update(schema_version="paper3-source-small-file-operational-amendment-v1",
        inflight_file_lifecycles=max_file_workers, max_concurrent_body_transfers=max_file_workers,
        max_staged_file_lifecycles_including_partials=max_file_workers,
        max_existing_and_new_staging_lifecycles=max_file_workers,
        serial_nonphysio_uses_one_of_global_slots=True,
        parallel_native_parsers=1, max_small_raw_bytes=max_small_raw_bytes,
        small_parallel_dataset="PhysionetMI", small_parallel_extension=".edf",
        nonphysio_phase="serial_all_178_before_PhysionetMI", nonphysio_objects=178, small_physio_objects=327,
        prior_executor_sha256=base.file_digest(parallel.__file__), executor_sha256=base.file_digest(__file__),
        fresh_HEAD_limit_checked_before_parallel_admission=True,
        internal_fresh_HEAD_and_GET_limit_rechecked_before_body=True,
        journal_paths=["audit/source_preparation/resource_progress.json", "audit/source_preparation/resource_progress.wall.json"],
        global_staging_min_free_disk_bytes=4 * plan["max_single_raw_bytes"] + 1024**3 + 2 * max_file_workers * max_small_raw_bytes,
        max_small_stream_buffer_bytes=max_file_workers * (base.CHUNK + r2_backup.CHUNK_SIZE),
        training_started=False)
    record["amendment_sha256"] = base.digest(record)
    return record


def _immutable_json(path, value):
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    base.no_symlink_components(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise base.BatchError("existing_small_operational_amendment_is_immutable")
        return
    base.atomic_json(path, value)


def _state_path(batch_dir, entry):
    path = batch_dir / "objects" / entry["object_id"] / "state.json"
    base.no_symlink_components(path)
    return path


def _completed(entry, root, batch_dir, plan, contract_sha):
    path = _state_path(batch_dir, entry)
    state = json.loads(path.read_text()) if path.exists() else {}
    if state and (state.get("identity") != list(base.identity(entry))
            or state.get("plan_sha256") != plan["plan_sha256"]
            or state.get("input_contract_sha256") != contract_sha):
        raise base.BatchError("source_state_identity_plan_or_contract_mismatch")
    valid = base.verify_completed(state, root, entry=entry, plan=plan)
    return valid and state.get("status") == "COMPLETE" and state.get("native_schema_audited") is True


def _occupied(root, plan, entry):
    folder = base.raw_staging_path(root, plan, entry).parent
    return folder.exists() and any(folder.iterdir())


def execute_plan(plan, *, root=REPO, batch_dir=None, max_new_files=None, client=None, bucket=None,
                 operational_limits=None, operational_amendment_note=None,
                 max_file_workers=MAX_WORKERS, max_small_raw_bytes=SMALL_LIMIT):
    _limits(max_file_workers, max_small_raw_bytes)
    base.validate_plan(plan)
    root = Path(root).resolve(strict=True)
    contract_sha = base.execution_gate(root)  # Before SDK, staging or journal changes.
    if client is None and threading.current_thread() is not threading.main_thread():
        raise base.BatchError("source_shared_SDK_client_must_be_created_in_main_thread")
    if max_new_files is not None and (type(max_new_files) is not int or max_new_files < 1):
        raise base.BatchError("max_new_files_must_be_positive")
    batch_dir = Path(batch_dir) if batch_dir is not None else root / "audit/source_preparation/batches" / plan["plan_sha256"]
    if batch_dir != root / "audit/source_preparation/batches" / plan["plan_sha256"]:
        raise base.BatchError("source_small_requires_original_batch_directory")
    base.no_symlink_components(root)
    base.no_symlink_components(batch_dir)
    required_disk = 4 * plan["max_single_raw_bytes"] + 1024**3 + 2 * max_file_workers * max_small_raw_bytes
    if shutil.disk_usage(root).free < required_disk:
        raise base.BatchError("source_small_insufficient_global_staging_disk")
    try:
        # This is a declared conservative cap, never a measured inventory size.
        # Reject insufficient cgroup stream headroom before even the serial phase.
        resource_preflight(root, max_small_raw_bytes, max_file_workers=max_file_workers)
    except parallel.ResourcePreflightPause:
        raise base.BatchError("source_small_cgroup_admission_preflight_pause") from None
    lock_path = root / "audit/source_preparation/.source_worker.lock"
    base.no_symlink_components(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise base.BatchError("source_batch_already_active") from None
        parallel._require_accounting_pair(root, batch_dir)
        staged_large, staged_small = inspect_staging(root, plan, batch_dir, contract_sha256=contract_sha,
            max_file_workers=max_file_workers, max_small_raw_bytes=max_small_raw_bytes)
        operational_limits = operational_limits or {}
        if set(operational_limits) - {"max_download_bytes", "max_wall_seconds", "max_r2_readback_bytes"}:
            raise base.BatchError("unknown_operational_resource_limit")
        identity = base.digest({"schema": "source-raw-resource-identity-v2", "files": plan["files"],
            "adapter_code_sha256": plan["adapter_code_sha256"], "input_contract_sha256": contract_sha})
        budget = SmallFileResourceBudget(base.ResourceBudget(plan, root / "audit/source_preparation/resource_progress.json",
            identity_sha256=identity, amendment_note=operational_amendment_note, **operational_limits), max_open=max_file_workers)
        shared_client, owns_client = None, client is None
        try:
            fixed_bucket = bucket if bucket is not None else os.environ.get("R2_BUCKET")
            if not isinstance(fixed_bucket, str) or not fixed_bucket:
                raise base.BatchError("missing_existing_r2_bucket")
            shared_client = client if client is not None else base.create_client()
            amendment = operational_amendment(plan, contract_sha, budget.snapshot(),
                max_file_workers=max_file_workers, max_small_raw_bytes=max_small_raw_bytes)
            path = batch_dir / ("small_files_operational_amendment_" + amendment["amendment_sha256"] + ".json")
            _immutable_json(path, amendment)
            return _execute(plan, root, batch_dir, contract_sha, budget,
                BoundedNativeAudit(root, max_file_workers=max_file_workers), shared_client, fixed_bucket,
                max_new_files, base.descriptor(root, path), staged_large, staged_small,
                max_file_workers, max_small_raw_bytes)
        finally:
            try:
                budget.close()  # Every future is drained before control reaches here.
            finally:
                if owns_client and shared_client is not None:
                    shared_client.close()


def _execute(plan, root, batch_dir, contract_sha, budget, audit, client, bucket,
             max_new_files, amendment, staged_large, staged_small, workers, small_limit):
    counts = {"new_objects_attempted": 0, "archived_schema_verified": 0, "schema_blocked": 0,
              "failed": 0, "admission_failed": 0}
    occupied = {e["object_id"] for e in staged_large + staged_small}
    admissions = []
    run_id = base.digest({"started_utc": base.utc_now(), "pid": os.getpid(), "operational_amendment": amendment})
    admission_path = batch_dir / "small_file_admission_attempts" / (run_id + ".json")
    if admission_path.exists():
        raise base.BatchError("source_small_admission_run_identity_collision")
    status, phase = "ALL_CANDIDATES_ATTEMPTED", "SERIAL_NONPHYSIO_178"
    nonphysio = [e for e in plan["files"] if e["dataset"] != "PhysionetMI"]
    physio = [e for e in plan["files"] if e["dataset"] == "PhysionetMI"]
    if len(nonphysio) != 178 or len(physio) != 327:
        raise base.BatchError("unchanged_178_nonphysio_and_327_PhysionetMI_required")

    def progress():
        base.atomic_json(batch_dir / "batch_progress.json", {"status": status, "phase": phase, **counts,
            "plan_sha256": plan["plan_sha256"], "operational_amendment": amendment,
            "inflight_file_lifecycles": workers if phase == "PARALLEL_SMALL_PHYSIONET" else 1,
            "parallel_native_parsers": 1, "occupied_file_lifecycles": len(occupied), "training_started": False})

    def permitted(entry):
        complete = _completed(entry, root, batch_dir, plan, contract_sha)
        if not complete and max_new_files is not None and counts["new_objects_attempted"] >= max_new_files:
            return None
        if not complete:
            counts["new_objects_attempted"] += 1
        return complete

    def process(entry, *, small=False):
        kwargs = {"opener": SmallOfficialOpener(entry, max_small_raw_bytes=small_limit)} if small else {}
        return base.process_file(entry, root=root, batch_dir=batch_dir, plan=plan,
            contract_sha256=contract_sha, budget=budget, audit_fn=audit, client=client, bucket=bucket, **kwargs)

    # Recover any existing large file first so no extra large staging is added.
    large_ids = {e["object_id"] for e in staged_large}
    nonphysio = staged_large + [e for e in nonphysio if e["object_id"] not in large_ids]
    for entry in nonphysio:
        budget.check()
        complete = permitted(entry)
        if complete is None:
            status = "PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE"
            break
        if not complete and entry["object_id"] not in occupied and len(occupied) >= workers:
            status = "EXISTING_STAGING_CAPACITY_BLOCKS_NEW_SERIAL_OBJECT"
            break
        if not complete:
            occupied.add(entry["object_id"])
        try:
            state = process(entry)
            if state.get("status") != "COMPLETE" or not _completed(entry, root, batch_dir, plan, contract_sha):
                counts["schema_blocked"] += 1
                status = "NONPHYSIO_NOT_COMPLETE_NO_SMALL_PHASE"
                break
            counts["archived_schema_verified"] += 1
        except parallel.ResourcePreflightPause:
            counts["failed"] += 1
            status = "NATIVE_RESOURCE_PREFLIGHT_PAUSE_NO_STAGE_ADVANCE"
            break
        except Exception:
            counts["failed"] += 1
            status = "NONPHYSIO_FAILED_NO_SMALL_PHASE"
            break
        finally:
            if not _occupied(root, plan, entry):
                occupied.discard(entry["object_id"])
        if _occupied(root, plan, entry):
            status = "RETAINED_NONPHYSIO_RAW_NO_SMALL_PHASE"
            break
        occupied.discard(entry["object_id"])
        budget.checkpoint()
        progress()
    else:
        # Recheck every non-Physio final receipt/contract before the phase change.
        if not all(_completed(e, root, batch_dir, plan, contract_sha) and not _occupied(root, plan, e)
                   for e in nonphysio):
            raise base.BatchError("all_178_nonphysio_COMPLETE_required_before_small_phase")
        phase, stopped = "PARALLEL_SMALL_PHYSIONET", False
        small_ids = {e["object_id"] for e in staged_small}
        ordered = staged_small + [e for e in physio if e["object_id"] not in small_ids]
        iterator, waiting = iter(ordered), None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paper3-small-edf") as pool:
            pending = {}
            while pending or not stopped:
                while not stopped and len(pending) < workers:
                    budget.check()
                    if waiting is None:
                        waiting = next(iterator, None)
                    if waiting is None:
                        stopped = True
                        break
                    entry = waiting
                    if entry["object_id"] not in occupied and len(occupied) >= workers:
                        break  # Wait for a real lifecycle to finish and free its slot.
                    try:
                        complete = permitted(entry)
                        if complete is None:
                            status, stopped = "PARTIAL_BATCH_LIMIT_NO_STAGE_ADVANCE", True
                            break
                        if not complete:
                            observed = _small_observation(base.head_object(entry, timeout=budget.timeout()), small_limit)
                            resource_preflight(root, observed["bytes"], max_file_workers=workers)
                    except parallel.ResourcePreflightPause:
                        status, stopped = "SMALL_ADMISSION_CGROUP_PAUSE_NO_STAGE_ADVANCE", True
                        break
                    except Exception as error:
                        counts["admission_failed"] += 1
                        admissions.append({"object_id": entry["object_id"], "identity": list(base.identity(entry)),
                            "error_type": type(error).__name__, "error_code": str(error) if isinstance(error, base.BatchError) else "HEAD_admission_details_suppressed"})
                        base.atomic_json(admission_path, {"schema_version": "paper3-small-source-admission-failures-v1",
                            "run_id": run_id, "operational_amendment": amendment, "failures": admissions})
                        waiting = None
                        if entry["object_id"] in occupied or isinstance(error, base.BatchError):
                            status, stopped = "SMALL_HEAD_ADMISSION_FAILED_NO_STAGE_ADVANCE", True
                        continue
                    occupied.add(entry["object_id"])
                    pending[pool.submit(process, entry, small=True)] = entry
                    waiting = None
                if not pending:
                    if not stopped:
                        status, stopped = "RETAINED_SMALL_STAGING_CAPACITY_REQUIRES_INSPECTION", True
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    entry = pending.pop(future)
                    try:
                        state = future.result()
                        counts["archived_schema_verified" if state.get("status") == "COMPLETE" else "schema_blocked"] += 1
                    except parallel.ResourcePreflightPause:
                        counts["failed"] += 1
                        status, stopped = "NATIVE_RESOURCE_PREFLIGHT_PAUSE_NO_STAGE_ADVANCE", True
                    except Exception:
                        counts["failed"] += 1
                    if _occupied(root, plan, entry):
                        status, stopped = "RETAINED_RAW_OR_PARTIAL_REQUIRES_RESUME_NO_STAGE_ADVANCE", True
                    else:
                        occupied.discard(entry["object_id"])
                budget.checkpoint()
                progress()
    budget.check()
    inventory = base.aggregate_inventory(plan, root=root, batch_dir=batch_dir, budget=budget)
    snapshot = budget.snapshot()
    result = {"status": status, "phase": phase, "inventory_status": inventory["status"], **counts,
        "actual_verified_objects": inventory["actual_verified_objects"], "required_target_object_count": 505,
        "plan_sha256": plan["plan_sha256"], "operational_amendment": amendment,
        "download_bytes_read": snapshot["download_bytes_read"], "download_bytes_charged": snapshot["download_bytes_charged"],
        "byte_scopes": {name: {key: ledger[key] for key in ("observed_bytes", "charged_bytes", "limit_bytes")}
                        for name, ledger in snapshot["byte_scopes"].items()},
        "resource_progress_path": budget.path.relative_to(root).as_posix(),
        "admission_failure_log": base.descriptor(root, admission_path) if admission_path.exists() else None,
        "active_wall_elapsed_s": snapshot["active_wall_elapsed_s"], "inflight_file_lifecycles": workers,
        "parallel_native_parsers": 1, "numerical_cache_created": False, "training_started": False, "w2_complete": False}
    base.atomic_json(batch_dir / "batch_progress.json", result)
    return result


def main(argv=None):
    from tools import prepare_all_sources as pipeline
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--contract", default="contracts/input_contract.json")
    parser.add_argument("--progress", default="audit/source_full/progress.json")
    parser.add_argument("--max-file-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-small-raw-bytes", type=int, default=SMALL_LIMIT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        _limits(args.max_file_workers, args.max_small_raw_bytes)
        result = pipeline.prepare_all(args.root, contract=args.contract, progress=args.progress, execute=args.execute,
            raw_fn=partial(execute_plan, max_file_workers=args.max_file_workers, max_small_raw_bytes=args.max_small_raw_bytes))
        print(json.dumps(result))
        return 0 if result.get("status") == "W2_COMPLETE_CACHE_AND_R2_VERIFIED" or not args.execute else 2
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_NO_STAGE_ADVANCE", "error_type": type(error).__name__,
            "gap": str(error) if isinstance(error, (base.BatchError, pipeline.PreparationError)) else "operation_failed_details_suppressed",
            "training_started": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
