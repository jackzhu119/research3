#!/usr/bin/env python3
"""Explicit CPU-only reconciliation after this managed workspace changed boot.

The frozen scientific plans and numerical kernels are never patched. Original
clock/transfer records are copied byte-for-byte before a transaction publishes
a new runtime reconciliation. The entire UTC interval is conservatively billed;
no open transfer reservation is refunded. Default is read-only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work import source_extension_coordinator as source
from tools import prepare_stieger_cache as stieger

ENTRY = "tools/recover_cpu_preparation_20261020.py"
BASE = "audit/data_plan_20261020/runtime_recovery"
STIEGER_PLAN = "audit/data_plan_20261019/stieger/cache_plan.json"
YANG_PLAN = "audit/source_extensions_20261019/Yang2025/plan.json"
STIEGER_SHA = "2f05d3142d6352420cadae30a8e1ff7410d37d9e517177a890b091757b171783"
YANG_SHA = "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
PROGRESS = "cache/stieger_v2/progress.json"
YANG_LEDGER = f"audit/source_extensions_20261019/Yang2025/{YANG_SHA}/budget_ledger.json"
SCHEMA = "paper3-explicit-CPU-runtime-reconciliation-v1"
POLICY = "Root executes the user's renewed instruction to continue CPU preparation after the managed environment was replaced. This is a documented operational reconciliation, not an automatic bypass of the old clock guard. No GPU, fit, remote server, external dataset, scientific split, input window or body budget is authorized by it."


class RecoveryError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(root, path):
    path = Path(path)
    return source.descriptor(root, path if path.is_absolute() else root/path)


def checked(root, value):
    return source.checked(root, value)


def snapshot(root):
    return stieger.clock_snapshot()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def conservative_gap(intent, now):
    """Only an explicit different-boot reconciliation uses UTC, never old mono."""
    try:
        before = datetime.fromisoformat(intent["utc"])
        after = datetime.fromisoformat(now["utc"])
        if (before.tzinfo is None or after.tzinfo is None
                or not intent.get("boot_id") or not now.get("boot_id")
                or intent["boot_id"] == now["boot_id"]):
            raise ValueError
        gap = (after-before).total_seconds()
        if not math.isfinite(gap) or not 0 <= gap <= 7*86400:
            raise ValueError
    except (ValueError, KeyError, TypeError, OverflowError):
        raise RecoveryError("different_boot_and_bounded_nonnegative_UTC_interval_required") from None
    return math.ceil(gap)


def closed_transfer_history(ledger):
    counted = {"provider_body_bytes": 0, "r2_get_body_bytes": 0}
    for row in ledger.get("history", []):
        kind, reserve = row.get("kind"), row.get("reserved_bytes")
        if kind not in counted or type(reserve) is not int or reserve < 0:
            raise RecoveryError("transfer_history_invalid")
        if row.get("status") == "SUCCESS_FULL_BODY_VERIFIED":
            amount = row.get("verified_bytes")
            if type(amount) is not int or not 0 <= amount <= reserve:
                raise RecoveryError("successful_transfer_history_invalid")
        elif row.get("status") in {"FAILURE_FULL_RESERVATION_CONSUMED", "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}:
            amount = reserve
        else:
            raise RecoveryError("unknown_transfer_history_status")
        counted[kind] += amount
    active = ledger.get("active_transfer")
    if active is not None:
        if active.get("kind") not in counted or type(active.get("reserved_bytes")) is not int or active["reserved_bytes"] < 0:
            raise RecoveryError("open_transfer_reservation_invalid")
        counted[active["kind"]] += active["reserved_bytes"]
    if counted != ledger.get("charged"):
        raise RecoveryError("charged_body_bytes_do_not_equal_durable_history_plus_open_reservation")
    return counted


def reconcile_values(progress, ledger, now):
    """Pure transaction calculation; no mutation of input or original clocks."""
    progress, ledger = json.loads(json.dumps(progress)), json.loads(json.dumps(ledger))
    if progress.get("plan_sha256") != STIEGER_SHA or ledger.get("plan_sha256") != YANG_SHA:
        raise RecoveryError("only_original_fixed_Stieger_and_Yang_plans_allowed")
    if not finite(progress.get("active_wall_elapsed_s")) or not finite(ledger.get("elapsed_seconds")):
        raise RecoveryError("prior_wall_high_water_required")
    charges = closed_transfer_history(ledger)
    stieger_gap = conservative_gap(progress.get("wall_intent"), now)
    yang_gap = conservative_gap(ledger.get("wall_intent"), now)
    old_intents = {"Stieger2021": progress.pop("wall_intent"), "Yang2025": ledger.pop("wall_intent")}
    progress["active_wall_elapsed_s"] += stieger_gap
    progress["status"] = "EXPLICIT_RUNTIME_RECONCILED_PENDING_SOLE_CPU_WORKER"
    ledger["elapsed_seconds"] += yang_gap
    transfer = ledger.pop("active_transfer", None)
    if transfer is not None:
        ledger.setdefault("history", []).append({**transfer,
            "status": "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"})
    assert ledger["charged"] == charges
    return progress, ledger, {"original_intents": old_intents,
        "entire_UTC_gap_billed_seconds": {"Stieger2021": stieger_gap, "Yang2025": yang_gap},
        "new_wall_high_water_seconds": {"Stieger2021": progress["active_wall_elapsed_s"], "Yang2025": ledger["elapsed_seconds"]},
        "body_reservation_refunded_bytes": 0, "Yang_preserved_body_charges": charges,
        "Yang_interrupted_transfer": transfer}


def load_plans(root):
    p, y = read(root/STIEGER_PLAN), read(root/YANG_PLAN)
    if p.get("plan_sha256") != STIEGER_SHA or y.get("plan_sha256") != YANG_SHA:
        raise RecoveryError("exact_original_preparation_plan_required")
    stieger.validate_plan(root, p, execute=False)
    source.validate_plan(root, y)
    return p, y


@contextmanager
def locks(root, paths):
    handles = []
    try:
        for relative in paths:
            path = source.local(root, relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RecoveryError("existing_CPU_worker_or_recovery_owns_lock") from None
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def archive_original(root, transaction, relative):
    original = source.local(root, relative)
    if not original.is_file():
        raise RecoveryError("required_original_recovery_record_missing")
    destination = transaction/"originals"/relative
    data = original.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise RecoveryError("immutable_original_snapshot_changed")
    else:
        with destination.open("xb") as output:
            output.write(data); output.flush(); os.fsync(output.fileno())
    return {"original_path": relative, "snapshot": ref(root, destination),
            "original_sha256": hashlib.sha256(data).hexdigest()}


def recovery_marker(root, boot):
    return source.local(root, f"{BASE}/{boot}/transaction.json")


def apply_transaction(root, marker_path):
    marker = read(marker_path)
    bundle_path = checked(root, marker["amendment"])
    bundle = read(bundle_path)
    if bundle.get("schema_version") != SCHEMA or bundle.get("new_runtime", {}).get("boot_id") != snapshot(root)["boot_id"]:
        raise RecoveryError("reconciliation_transaction_runtime_mismatch")
    checked(root, bundle["implementation"])
    if marker.get("status") == "RECONCILIATION_COMMITTED":
        for relative, floor in bundle["proposed_runtime_records"].items():
            current = read(source.local(root, relative))
            if current.get("explicit_runtime_reconciliation") != marker["amendment"]:
                raise RecoveryError("committed_reconciliation_pointer_missing_no_automatic_rewrite")
            key = "active_wall_elapsed_s" if relative == PROGRESS else "elapsed_seconds"
            if not finite(current.get(key)) or current[key] < floor[key]:
                raise RecoveryError("published_reconciled_wall_floor_rolled_back")
            if relative == YANG_LEDGER and any(current.get("charged",{}).get(k,-1) < v for k,v in floor["charged"].items()):
                raise RecoveryError("published_reconciled_body_floor_rolled_back")
        return {"status": "RECONCILIATION_ALREADY_COMMITTED_NO_RUNTIME_REWRITE", "amendment": marker["amendment"],
                "CPU_only": True, "network_operations": 0, "scientific_fits": 0}
    if marker.get("status") != "RECONCILIATION_PREPARED":
        raise RecoveryError("invalid_recovery_transaction_state")
    for relative, value in bundle["proposed_runtime_records"].items():
        current = source.local(root, relative)
        proposed = {**value, "explicit_runtime_reconciliation": marker["amendment"]}
        old = next(r for r in bundle["original_records"] if r["original_path"] == relative)
        data = current.read_bytes()
        if hashlib.sha256(data).hexdigest() != old["original_sha256"] and read(current) != proposed:
            raise RecoveryError("live_record_changed_since_reconciliation_snapshot")
        source.atomic_json(current, proposed)
    marker["status"] = "RECONCILIATION_COMMITTED"
    source.atomic_json(marker_path, marker)
    return {"status": marker["status"], "amendment": marker["amendment"],
            "reconciliation": bundle["conservative_accounting"], "CPU_only": True,
            "network_operations": 0, "scientific_fits": 0}


def reconcile(root, *, execute=False, confirmed_runtime_replacement=False, cpu_wall_cap_seconds=172800):
    root = Path(root).resolve(strict=True)
    p, y = load_plans(root)
    if type(cpu_wall_cap_seconds) is not int or not 172800 <= cpu_wall_cap_seconds <= 259200:
        raise RecoveryError("explicit_CPU_wall_cap_must_be_48_to_72_hours")
    now = snapshot(root)
    marker_path = recovery_marker(root, now["boot_id"])
    if marker_path.exists():
        if not execute:
            return {"status": "EXISTING_RECONCILIATION_TRANSACTION", "transaction": ref(root, marker_path), "network_operations": 0}
        with locks(root, ["cache/stieger_v2/.prepare.lock", "audit/source_extensions_20261019/.source-extension-singleton.lock", f"{BASE}/.reconcile.lock"]):
            return apply_transaction(root, marker_path)
    before_p, before_y = read(root/PROGRESS), read(root/YANG_LEDGER)
    new_p, new_y, accounting = reconcile_values(before_p, before_y, now)
    if new_p["active_wall_elapsed_s"] >= cpu_wall_cap_seconds or new_y["elapsed_seconds"] >= y["budgets"]["wall_seconds"]:
        raise RecoveryError("proposed_reconciliation_exceeds_remaining_authorized_CPU_wall_cap")
    preview = {"status": "PLANNED_EXPLICIT_RUNTIME_RECONCILIATION_NO_MUTATION", "new_runtime": now,
               "conservative_accounting": accounting, "effective_Stieger_CPU_wall_cap_seconds": cpu_wall_cap_seconds,
               "original_scientific_plan_SHA_unchanged": True, "body_caps_unchanged": True,
               "network_operations": 0, "scientific_fits": 0, "GPU_actions": 0}
    if not execute:
        return preview
    if confirmed_runtime_replacement is not True:
        raise RecoveryError("root_operator_runtime_replacement_attestation_required")
    with locks(root, ["cache/stieger_v2/.prepare.lock", "audit/source_extensions_20261019/.source-extension-singleton.lock", f"{BASE}/.reconcile.lock"]):
        # Never bill against a moving worker. The snapshot under lock is the
        # actual transaction start, rather than the earlier preview timestamp.
        now = snapshot(root)
        new_p, new_y, accounting = reconcile_values(read(root/PROGRESS), read(root/YANG_LEDGER), now)
        if new_p["active_wall_elapsed_s"] >= cpu_wall_cap_seconds or new_y["elapsed_seconds"] >= y["budgets"]["wall_seconds"]:
            raise RecoveryError("locked_reconciliation_exceeds_remaining_authorized_CPU_wall_cap")
        guard_failures = {}
        for name, function in (("Stieger2021", stieger.elapsed_guard), ("Yang2025", source.elapsed)):
            try:
                function(accounting["original_intents"][name], now)
            except (stieger.CacheError, source.PreparationError) as exc:
                guard_failures[name] = {"error_code": str(exc), "automatic_resume_allowed": False}
            else:
                raise RecoveryError("old_runtime_guard_did_not_exhibit_expected_boot_block")
        transaction = marker_path.parent
        transaction.mkdir(parents=True, exist_ok=True)
        originals = [archive_original(root, transaction, path) for path in (PROGRESS, YANG_LEDGER, "status/tasks.json")]
        for entry in p["files"]:
            relative = f"{p['output']}/{entry['partition']}/{entry['provider_subject']}/session-{entry['session_ordinal']}/state.json"
            path = root/relative
            if path.is_file() and read(path).get("transfer_intent"):
                originals.append(archive_original(root, transaction, relative))
        scope = root/f"audit/source_extensions_20261019/Yang2025/{YANG_SHA}"
        for path in sorted(scope.glob("*/state.json")):
            if read(path).get("status") != "COMPLETE":
                originals.append(archive_original(root, transaction, path.relative_to(root).as_posix()))
        bundle = {"schema_version": SCHEMA, "operator_attestation": POLICY,
            "historical_clock_gate_preserved": "Old boot-mismatch guards remain unchanged in frozen kernels; original records are immutable snapshots.",
            "legacy_guards_at_reconciliation": guard_failures,
            "new_runtime": now, "original_records": originals, "conservative_accounting": accounting,
            "implementation": ref(root, ENTRY), "Stieger_plan": ref(root, STIEGER_PLAN), "Yang_plan": ref(root, YANG_PLAN),
            "original_Stieger_plan_wall_cap_seconds": p["max_wall_seconds"],
            "effective_Stieger_CPU_wall_cap_seconds": cpu_wall_cap_seconds,
            "Yang_plan_wall_cap_seconds_unchanged": y["budgets"]["wall_seconds"],
            "body_caps_unchanged": True, "scientific_fits": 0, "GPU_actions": 0,
            "remote_server_actions": False, "external_data_access": False,
            "proposed_runtime_records": {PROGRESS: new_p, YANG_LEDGER: new_y}}
        bundle_path = transaction/"runtime_reconciliation_amendment.json"
        source.atomic_json(bundle_path, bundle, immutable=True)
        source.atomic_json(marker_path, {"schema_version": "paper3-runtime-recovery-transaction-v1",
            "status": "RECONCILIATION_PREPARED", "amendment": ref(root, bundle_path)}, immutable=True)
        return apply_transaction(root, marker_path)


def load_stieger_authorization(root):
    now = snapshot(root)
    marker = read(recovery_marker(root, now["boot_id"]))
    if marker.get("status") != "RECONCILIATION_COMMITTED":
        raise RecoveryError("explicit_reconciliation_must_commit_before_CPU_resume")
    path = checked(root, marker["amendment"])
    value = read(path)
    if value.get("schema_version") != SCHEMA or value.get("new_runtime", {}).get("boot_id") != now["boot_id"]:
        raise RecoveryError("same_reconciled_runtime_required")
    checked(root, value["implementation"])
    p, _ = load_plans(root)
    if ref(root, STIEGER_PLAN) != value["Stieger_plan"] or value.get("body_caps_unchanged") is not True:
        raise RecoveryError("fixed_scientific_plan_and_body_caps_required")
    cap = value.get("effective_Stieger_CPU_wall_cap_seconds")
    if type(cap) is not int or not 172800 <= cap <= 259200:
        raise RecoveryError("invalid_reconciled_CPU_wall_cap")
    return p, marker["amendment"], cap


def resume_stieger(root, *, execute=False, max_new_files=None):
    """Same original numerical/IO kernel, with a separately declared CPU cap."""
    root = Path(root).resolve(strict=True)
    plan, amendment, cap = load_stieger_authorization(root)
    contract, _ = stieger.validate_plan(root, plan, execute=execute)
    if not execute:
        return {"status": "RECOVERED_STIEGER_RESUME_PLAN_NO_NETWORK", "plan_sha256": plan["plan_sha256"],
            "effective_CPU_wall_cap_seconds": cap, "amendment": amendment, "GPU_actions": 0, "scientific_fits": 0}
    if max_new_files is not None and (type(max_new_files) is not int or max_new_files < 1):
        raise RecoveryError("positive_new_file_limit_required")
    output = stieger.local(root, plan["output"])
    expected_owner = {"schema_version": stieger.SCHEMA, "created_by": "tools/prepare_stieger_cache.py", "plan_sha256": STIEGER_SHA}
    if read(output/"ownership.json") != expected_owner:
        raise RecoveryError("original_stieger_output_ownership_required")
    with locks(root, [f"{plan['output']}/.prepare.lock"]):
        progress_path = output/"progress.json"
        progress = read(progress_path)
        if progress.get("plan_sha256") != STIEGER_SHA or progress.get("explicit_runtime_reconciliation") != amendment:
            raise RecoveryError("reconciled_progress_identity_required")
        if progress.get("status") == "W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY":
            raise RecoveryError("completed_W3_uses_read_only_final_auditor_no_new_execution")
        progress.update(status="RUNNING_RECOVERED_LOCAL_CPU", effective_CPU_wall_cap_seconds=cap,
                        original_plan_wall_cap_seconds=plan["max_wall_seconds"], operational_builder=ref(root, ENTRY))
        states, official_bytes, restore_bytes, cache_bytes, new_files = [], 0, 0, 0, 0
        try:
            stieger.wall_checkpoint(progress_path, progress, reopen=True)
            catalog = stieger.raw_receipt_catalog(root, plan["files"])
            for entry in plan["files"]:
                sp = output/entry["partition"]/entry["provider_subject"]/f"session-{entry['session_ordinal']}"/"state.json"
                if sp.exists():
                    state = read(sp)
                    if state.get("plan_sha256") != STIEGER_SHA:
                        raise RecoveryError("prior_session_state_different_plan")
                    stieger.recover_open_transfer(state, root=root, object_dir=sp.parent, plan=plan, state_path=sp)
                    official_bytes += state.get("official_provider_body_bytes", 0)
                    restore_bytes += state.get("r2_restore_body_bytes", 0)
            for entry in plan["files"]:
                progress.update(next_file=entry["name"], official_provider_body_bytes=official_bytes, r2_restore_body_bytes=restore_bytes)
                stieger.wall_checkpoint(progress_path, progress, reopen=True)
                if (progress["active_wall_elapsed_s"] >= cap or cache_bytes >= plan["max_cache_bytes"]
                        or official_bytes > plan["max_download_bytes"] or restore_bytes > plan["max_r2_restore_bytes"]):
                    raise RecoveryError("effective_CPU_or_original_body_resource_budget_exhausted")
                state_path = output/entry["partition"]/entry["provider_subject"]/f"session-{entry['session_ordinal']}"/"state.json"
                old = read(state_path) if state_path.exists() else {}
                done = bool(old) and stieger.verify_session_state(old, root, plan)
                if not done and max_new_files is not None and new_files >= max_new_files:
                    progress.update(status="PARTIAL_BATCH_NO_W3_COMPLETE")
                    break
                state = stieger.process_session(entry, root=root, output=output, contract=contract, plan=plan,
                    raw_catalog=catalog, remaining_source_bytes=plan["max_download_bytes"]-official_bytes,
                    remaining_restore_bytes=plan["max_r2_restore_bytes"]-restore_bytes)
                official_bytes += state.get("official_provider_body_bytes", 0)-old.get("official_provider_body_bytes", 0)
                restore_bytes += state.get("r2_restore_body_bytes", 0)-old.get("r2_restore_body_bytes", 0)
                states.append(state); new_files += int(not done); cache_bytes += state["cache_bytes"]
                progress.update(completed_sessions=max(progress.get("completed_sessions", 0),len(states)),
                    cache_bytes=cache_bytes, official_provider_body_bytes=official_bytes, r2_restore_body_bytes=restore_bytes)
                stieger.wall_checkpoint(progress_path, progress, reopen=True)
                if not done:
                    print(json.dumps({"file": entry["name"], "status": "COMPLETE_CACHE_AND_FULL_GET", "sessions": len(states), "total": 598}), flush=True)
            if len(states) == 598:
                if cache_bytes > plan["max_cache_bytes"] or progress["active_wall_elapsed_s"] >= cap:
                    raise RecoveryError("effective_CPU_budget_reached_before_global_commit")
                manifest = stieger.assemble_manifest(root, output, plan, states)
                manifest_path = output/"stieger_cache.json"
                stieger.write_once_json(manifest_path, manifest)
                files = [(manifest_path,"manifest"),(output/"duplicate_audit.json","audit")]
                files.extend((stieger.checked(root,plan[k]),"protocol") for k in ("official_inventory","input_contract","participant_split"))
                files.extend((stieger.checked(root,s["cache_backup_receipt"]),"audit") for s in states)
                receipt = stieger.verified_backup(root, output/"manifest_backup_manifest.json", output/"manifest_backup_receipt.json",
                    prefix=f"paper3/stieger-cache-v2/{STIEGER_SHA}/manifest/", files=files, backup_fn=stieger.r2_backup.run_backup)
                stieger.wall_checkpoint(progress_path, progress, reopen=True)
                if progress["active_wall_elapsed_s"] >= cap:
                    raise RecoveryError("effective_CPU_wall_cap_reached_before_complete")
                completion = {"schema_version": "paper3-recovered-Stieger-CPU-completion-v1", "amendment": amendment,
                    "operational_builder": ref(root, ENTRY), "scientific_plan_sha256": STIEGER_SHA,
                    "cache_manifest": ref(root,manifest_path), "original_global_cache_GET_receipt": receipt,
                    "effective_CPU_wall_cap_seconds": cap, "cumulative_wall_seconds_before_operational_backup": progress["active_wall_elapsed_s"],
                    "original_plan_wall_cap_seconds":plan["max_wall_seconds"], "original_body_caps_unchanged":True,
                    "official_provider_body_bytes":official_bytes, "r2_restore_body_bytes":restore_bytes,
                    "scientific_fits":0,"GPU_actions":0,"remote_server_actions":False,"external_data_access":False}
                completion_path = output/"cpu_recovery_completion.json"
                stieger.write_once_json(completion_path, completion)
                op_receipt = stieger.verified_backup(root, output/"cpu_recovery_backup_manifest.json", output/"cpu_recovery_backup_receipt.json",
                    prefix=f"paper3/stieger-cache-v2/{STIEGER_SHA}/cpu-recovery-20261020/",
                    files=[(completion_path,"audit"),(checked(root,amendment),"audit"),(root/ENTRY,"code")],backup_fn=stieger.r2_backup.run_backup)
                stieger.wall_checkpoint(progress_path, progress, reopen=True)
                if progress["active_wall_elapsed_s"] >= cap:
                    raise RecoveryError("effective_CPU_wall_cap_reached_during_operational_backup")
                stieger.update_w3_task(root,plan,"COMPLETE",cache_manifest=ref(root,manifest_path),cache_r2_receipt=receipt,sessions=598,people=62,
                    cpu_operational_recovery=amendment,cpu_operational_completion=ref(root,completion_path),cpu_operational_R2_receipt=op_receipt,
                    effective_CPU_wall_cap_seconds=cap)
                progress.update(status="W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY",cache_manifest=ref(root,manifest_path),cache_r2_receipt=receipt,
                    cpu_operational_completion=ref(root,completion_path),cpu_operational_R2_receipt=op_receipt)
            else:
                stieger.update_w3_task(root,plan,"PARTIAL_CACHE_NOT_COMPLETE",completed_sessions=len(states),cpu_operational_recovery=amendment)
        except BaseException as exc:
            progress.update(status="FAILED_CLOSED_NO_W3_COMPLETE",error_type=type(exc).__name__,
                error_code=str(exc) if isinstance(exc,(RecoveryError,stieger.CacheError,stieger.base.BatchError)) else "operation_failed_details_suppressed")
            stieger.update_w3_task(root,plan,"FAILED_CLOSED_NOT_COMPLETE",error_code=progress["error_code"],
                completed_sessions=progress.get("completed_sessions",0),cpu_operational_recovery=amendment)
            raise
        finally:
            persisted = [read(path) for path in output.glob("*/*/session-*/state.json")]
            progress["official_provider_body_bytes"] = sum(s.get("official_provider_body_bytes",0) for s in persisted)
            progress["r2_restore_body_bytes"] = sum(s.get("r2_restore_body_bytes",0) for s in persisted)
            stieger.wall_checkpoint(progress_path,progress,reopen=False)
            if progress["active_wall_elapsed_s"] >= cap:
                progress["status"] = "RESOURCE_BUDGET_STOP_NO_W3_COMPLETE"
                source.atomic_json(progress_path,progress)
                stieger.update_w3_task(root,plan,"RESOURCE_BUDGET_STOP_NOT_COMPLETE",
                    completed_sessions=progress.get("completed_sessions",0),cpu_operational_recovery=amendment)
                raise RecoveryError("effective_CPU_wall_cap_reached_at_final_closed_snapshot")
        return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--confirm-runtime-replacement",action="store_true")
    parser.add_argument("--cpu-wall-cap-seconds",type=int,default=172800)
    parser.add_argument("--resume-stieger",action="store_true")
    parser.add_argument("--max-new-files",type=int)
    args = parser.parse_args(argv)
    try:
        result = (resume_stieger(args.root, execute=args.execute, max_new_files=args.max_new_files)
            if args.resume_stieger else reconcile(args.root, execute=args.execute,
                confirmed_runtime_replacement=args.confirm_runtime_replacement,cpu_wall_cap_seconds=args.cpu_wall_cap_seconds))
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED_NO_FALSE_COMPLETION","error_type":type(exc).__name__,
            "error_code":str(exc) if isinstance(exc,(RecoveryError,stieger.CacheError,source.PreparationError)) else "operation_failed_details_suppressed",
            "scientific_fits":0,"GPU_actions":0,"remote_server_actions":False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
