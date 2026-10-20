#!/usr/bin/env python3
"""Settle the specifically audited interrupted Yang V2 call07 without retrying it.

Default is read-only. The root-authorized external receipt explicitly records
NO observed Python exception and NO worker return, rather than claiming a
normal return, an HTTP failure or completion. Original V2 call08 remains the
only possible later preparation call. No client, network or prepare is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import resume_yang_source_transport_v2 as parent
from p3work import source_extension_coordinator as core

SCHEMA = "paper3-Yang-V2-call07-external-hard-interruption-settlement-v1"
ENTRY = "tools/recover_yang_v2_call07_hard_interruption.py"
TESTS = "tests/test_recover_yang_v2_call07_hard_interruption.py"
BASE = "audit/data_plan_20261020/yang_V2_call07_hard_interruption_recovery"
AMENDMENT = "plans/data_plan_20261019/yang_V2_call07_hard_interruption_amendment.json"
PARENT_SHA = "2ae1e75e47a75032155da44c1cb45afa78a1ec82da3ecd11d6f85ec60e1c0f71"
OPEN_STATE_SHA = "bc23bf4fbe41de34c04fb9384c6b53583edf00c1e129a4da3690f5a91cfdc5ee"
OPEN_LEDGER_SHA = "a7678617e73925f17991bb897366e7e724e6fa71f63d517a9f4d0d1aa3949bdf"
INTENT_SHA = "c4b56e9de0214f251be53a8de38ce1542ea94bc68359584c709eca22678526b6"
BEFORE_SHA = "0271d03bacf89ad9dd9997b2d98c9b43fed6fa190a0e7d33387934a9970cc1b8"
EXPECTED_BOOT = "d75c8fa0-1d33-45dd-b8d9-78f3e0dfbb34"
ORIGINAL_PID = 14755
STATUS = "HARD_INTERRUPTION_EXTERNALLY_SETTLED_NOT_COMPLETE"
AUTHORIZATION = {
    "authorized_by": "root_20261020_under_user_Paper3_CPU_preparation_authorization",
    "scope": "external_forensic_accounting_settlement_of_one_already_issued_call07_only",
    "error_type_null_means": "no_Python_exception_object_was_observed_not_a_normal_worker_return",
    "coordinator_result_null_means": "no_original_coordinator_return_value_was_observed",
    "worker_returned_normally": False, "worker_result_observed": False,
    "retryable_transport_error": False, "original_call_limit": 8, "calls_used_preserved": 7,
    "no_network_Session_R2_prepare_fit_GPU_RunPod_or_external_query": True,
    "original_resource_caps_charges_and_complete_history_never_reset": True,
}
read = parent.read
path = parent.path
ref = parent.ref
checked = parent.checked
immutable_bytes = parent.immutable_bytes


class RecoveryError(ValueError):
    pass


def source_ledger(root):
    return parent.previous.ledger_path(root, {"plan_sha256": parent.PLAN_SHA})


def parent_state(root):
    return path(root, parent.HISTORY + "/recovery_ledger.json")


def call_dir(root):
    return path(root, parent.HISTORY + "/call-07")


def code(root):
    filename = path(root, ENTRY)
    if filename.read_bytes() != Path(__file__).read_bytes():
        raise RecoveryError("executed_recovery_differs_from_root")
    return ref(root, filename)


def load(root):
    root, plan, plan_ref = parent.load_plan(root)
    if ref(root, path(root, parent.ENTRY))["sha256"] != PARENT_SHA:
        raise RecoveryError("original_frozen_V2_code_required")
    authority, authority_ref = parent.verify_static_authority(root, plan)
    return root, plan, plan_ref, authority, authority_ref


def proc_record(pid, *, proc_root=Path("/proc")):
    """Primitive process facts only; never read environment or expose argv."""
    folder = proc_root / str(pid)
    try:
        raw = (folder / "stat").read_text()
    except FileNotFoundError:
        return {"pid": pid, "status": "ABSENT"}
    values = raw.rsplit(") ", 1)[1].split()
    if len(values) < 20:
        raise RecoveryError("process_stat_identity_unavailable")
    state, start_ticks = values[0], int(values[19])
    command = (folder / "cmdline").read_bytes()
    result = {"pid": pid, "state": state, "ppid": int(values[1]), "start_ticks": start_ticks,
              "cmdline_empty": command == b""}
    if state != "Z":
        return {**result, "status": "ALIVE_OR_UNPROVEN"}
    status_lines = (folder / "status").read_text().splitlines()
    fdsize = next((int(row.split(":", 1)[1].strip()) for row in status_lines if row.startswith("FDSize:")), None)
    try:
        count = len(list((folder / "fd").iterdir()))
        readable = True
    except PermissionError:
        count, readable = None, False
    second = (folder / "stat").read_text().rsplit(") ", 1)[1].split()
    if second[0] != "Z" or int(second[19]) != start_ticks or command or fdsize != 0 or count not in {None, 0}:
        raise RecoveryError("zombie_has_unproven_execution_or_file_descriptors")
    return {**result, "status": "DEAD_ZOMBIE_NO_COMMAND_NO_FDS", "open_fd_count": count,
            "FDSize": fdsize, "fd_directory_readable": readable}


def active_original_workers(*, proc_root=Path("/proc")):
    result = []
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            args = (folder / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if any(arg == parent.ENTRY.encode() or arg.endswith(b"/" + parent.ENTRY.encode()) for arg in args):
            result.append(int(folder.name))
    return sorted(result)


def dead_worker_evidence():
    evidence = proc_record(ORIGINAL_PID)
    if evidence.get("status") not in {"ABSENT", "DEAD_ZOMBIE_NO_COMMAND_NO_FDS"}:
        raise RecoveryError("original_worker_still_alive_or_unproven")
    active = active_original_workers()
    if active:
        raise RecoveryError("original_V2_entry_has_active_process_no_recovery")
    return {"original_process": evidence, "active_original_entry_pids": active}


def count_history(value):
    counts = {"provider_body_bytes": 0, "r2_get_body_bytes": 0}
    for row in value.get("history", []):
        kind, size = row.get("kind"), row.get("reserved_bytes")
        if kind not in counts or type(size) is not int or size < 0:
            raise RecoveryError("invalid_original_transfer_history")
        if row.get("status") == "SUCCESS_FULL_BODY_VERIFIED":
            used = row.get("verified_bytes")
            if type(used) is not int or not 0 <= used <= size:
                raise RecoveryError("invalid_original_verified_transfer")
        elif row.get("status") in {"FAILURE_FULL_RESERVATION_CONSUMED", "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}:
            used = size
        else:
            raise RecoveryError("unrecognized_original_transfer_status")
        counts[kind] += used
    return counts


def open_accounting(value, plan, now):
    if value.get("schema_version") != "paper3-source-extension-budget-ledger-v1" or value.get("plan_sha256") != plan["plan_sha256"]:
        raise RecoveryError("exact_original_ledger_and_plan_required")
    transfer = value.get("active_transfer")
    if (not isinstance(transfer, dict) or set(transfer) != {"kind", "reserved_bytes", "identity"}
            or transfer["kind"] not in {"provider_body_bytes", "r2_get_body_bytes"}
            or type(transfer["reserved_bytes"]) is not int or transfer["reserved_bytes"] < 0):
        raise RecoveryError("exact_unclosed_transfer_reservation_required")
    if value.get("wall_intent", {}).get("boot_id") != EXPECTED_BOOT or now.get("boot_id") != EXPECTED_BOOT:
        raise RecoveryError("same_original_boot_required_no_clock_rebase")
    gap = core.elapsed(value["wall_intent"], now)
    spent = value.get("elapsed_seconds")
    if type(spent) not in (int, float) or not 0 <= spent + gap <= plan["budgets"]["wall_seconds"]:
        raise RecoveryError("original_cumulative_wall_budget_exhausted_or_invalid")
    counts = count_history(value)
    counts[transfer["kind"]] += transfer["reserved_bytes"]
    if (counts != value.get("charged")
            or any(counts[k] > plan["budgets"][k] for k in counts)):
        raise RecoveryError("open_reservation_and_original_charges_must_match_full_history")
    return {"observed_clock": now, "downtime_seconds_at_observation": gap,
            "minimum_cumulative_wall_seconds": spent + gap,
            "original_charged_bytes_preserved": counts, "unclosed_transfer": transfer}


def verify_six_closed_calls(root, plan, authority, state):
    identity = {"schema_version": parent.SCHEMA, "amendment": ref(root, path(root, parent.AMENDMENT_PATH)),
                "plan_sha256": parent.PLAN_SHA}
    if (state.get("identity") != identity or read(path(root, parent.HISTORY + "/claim.json")) != identity
            or state.get("calls_used") != 7 or state.get("active_call") != 7
            or state.get("terminal") is not None or len(state.get("call_receipts", [])) != 6):
        raise RecoveryError("only_exact_issued_call07_with_six_closed_calls_may_be_settled")
    last, _ = parent.closed_ledger(root, plan, checked(root, authority["post_probe_closed_ledger_snapshot"]))
    for number, item in enumerate(state["call_receipts"], 1):
        receipt = read(checked(root, item)); intent = read(checked(root, receipt["intent"]))
        if (receipt.get("schema_version") != parent.SCHEMA or receipt.get("call_number") != number
                or receipt.get("amendment") != identity["amendment"] or receipt.get("whole_job_prepare_calls") != 1
                or intent.get("schema_version") != parent.SCHEMA or intent.get("call_number") != number
                or intent.get("amendment") != identity["amendment"] or intent.get("plan") != authority["frozen_plan"]
                or intent.get("before_closed_ledger") != receipt.get("before_closed_ledger")
                or intent.get("session_headers") != parent.HEADERS or intent.get("requests_adapter_retries") != 0
                or intent.get("one_prepare_call_only") is not True
                or receipt.get("error_type") is None or receipt.get("retryable_transport_error") is not True):
            raise RecoveryError("all_six_original_closed_receipts_and_intents_required")
        before, _ = parent.closed_ledger(root, plan, checked(root, receipt["before_closed_ledger"]))
        after, _ = parent.closed_ledger(root, plan, checked(root, receipt["after_closed_ledger"]))
        parent._prefix(last, before); parent._prefix(before, after); last = after
    return last


def preflight(root):
    root, plan, plan_ref, authority, authority_ref = load(root)
    state = read(parent_state(root)); intent_path = call_dir(root) / "intent.json"
    intent = read(intent_path)
    if (ref(root, parent_state(root))["sha256"] != OPEN_STATE_SHA
            or ref(root, source_ledger(root))["sha256"] != OPEN_LEDGER_SHA
            or ref(root, intent_path)["sha256"] != INTENT_SHA
            or intent.get("before_closed_ledger", {}).get("sha256") != BEFORE_SHA
            or intent.get("call_number") != 7 or intent.get("schema_version") != parent.SCHEMA
            or intent.get("amendment") != authority_ref or intent.get("plan") != plan_ref
            or intent.get("session_headers") != parent.HEADERS or intent.get("requests_adapter_retries") != 0
            or intent.get("one_prepare_call_only") is not True
            or intent.get("started_clock_for_audit_only", {}).get("boot_id") != EXPECTED_BOOT):
        raise RecoveryError("exact_original_open_call07_bytes_and_intent_required")
    if sorted(p.name for p in path(root, parent.HISTORY).glob("call-*")) != [f"call-{i:02d}" for i in range(1, 8)]:
        raise RecoveryError("original_call_directories_changed_no_counter_reset")
    if (call_dir(root) / "receipt.json").exists() or (call_dir(root) / "after_closed_ledger.json").exists():
        raise RecoveryError("existing_call07_closure_never_overwritten")
    last = verify_six_closed_calls(root, plan, authority, state)
    before, _ = parent.closed_ledger(root, plan, checked(root, intent["before_closed_ledger"]))
    parent._prefix(last, before)
    opened = read(source_ledger(root)); parent._prefix(before, opened)
    accounting = open_accounting(opened, plan, core.clock_snapshot())
    if (opened["wall_intent"]["monotonic_ns"] < intent["started_clock_for_audit_only"]["monotonic_ns"]):
        raise RecoveryError("open_clock_precedes_original_call07_intent")
    death = dead_worker_evidence()
    return root, plan, plan_ref, state, intent, accounting, death


def snapshot(root, filename, relative):
    immutable_bytes(path(root, relative), Path(filename).read_bytes())
    return ref(root, path(root, relative))


def write_amendment(root):
    with parent.owner(Path(root)), parent.previous.coordinator_lock(Path(root)):
        root, plan, plan_ref, state, intent, accounting, death = preflight(root)
        filename = path(root, AMENDMENT)
        if filename.exists():
            verify_authority(root, plan)
            return {"status": "RECOVERY_AMENDMENT_ALREADY_VERIFIED_NO_SETTLEMENT", "amendment": ref(root, filename)}
        inventory = []
        audit = path(root, f"audit/source_extensions_20261019/Yang2025/{parent.PLAN_SHA}")
        for record in plan["records"]:
            folder = path(root, (audit / record["record_id"]).relative_to(root).as_posix())
            for name in ("state.json", "raw_backup_receipt.json", "cache_backup_receipt.json", "raw_restore_report.json"):
                original = folder / name
                if original.is_file():
                    inventory.append({"source_at_snapshot": original.relative_to(root).as_posix(),
                        "opaque_snapshot": snapshot(root, original, f"{BASE}/record_evidence/{record['record_id']}/{name}.txt")})
        value = {"schema_version": SCHEMA, "status": "AUTHORIZED_EXTERNAL_CALL07_HARD_INTERRUPTION_SETTLEMENT",
            "authorization": AUTHORIZATION, "implementation": code(root), "tests": ref(root, path(root, TESTS)),
            "frozen_plan": plan_ref, "original_V2_wrapper": ref(root, path(root, parent.ENTRY)),
            "original_V2_amendment": ref(root, path(root, parent.AMENDMENT_PATH)),
            "open_V2_state_opaque": snapshot(root, parent_state(root), BASE + "/open_V2_state.json.txt"),
            "open_original_ledger_opaque": snapshot(root, source_ledger(root), BASE + "/open_original_ledger.json.txt"),
            "original_call07_intent": ref(root, call_dir(root) / "intent.json"),
            "original_call07_before_closed_ledger": intent["before_closed_ledger"],
            "original_six_closed_call_receipts": state["call_receipts"], "record_evidence_opaque": inventory,
            "process_death_at_authorization": death, "accounting_at_authorization": accounting,
            "scientific_fits": 0, "GPU_actions": 0, "network_requests": 0}
        core.atomic_json(filename, value, immutable=True)
        return {"status": value["status"], "amendment": ref(root, filename), "settlement_performed": False}


def verify_authority(root, plan):
    value = read(path(root, AMENDMENT))
    if (value.get("schema_version") != SCHEMA or value.get("authorization") != AUTHORIZATION
            or value.get("status") != "AUTHORIZED_EXTERNAL_CALL07_HARD_INTERRUPTION_SETTLEMENT"
            or value.get("implementation") != code(root) or value.get("tests") != ref(root, path(root, TESTS))
            or value.get("frozen_plan") != ref(root, path(root, parent.PLAN_PATH))
            or value.get("original_V2_wrapper") != ref(root, path(root, parent.ENTRY))
            or value["original_V2_wrapper"]["sha256"] != PARENT_SHA
            or value.get("original_V2_amendment") != ref(root, path(root, parent.AMENDMENT_PATH))
            or value.get("original_call07_intent") != ref(root, call_dir(root) / "intent.json")
            or value["original_call07_intent"]["sha256"] != INTENT_SHA
            or value.get("original_call07_before_closed_ledger", {}).get("sha256") != BEFORE_SHA
            or value.get("open_V2_state_opaque", {}).get("sha256") != OPEN_STATE_SHA
            or value.get("open_original_ledger_opaque", {}).get("sha256") != OPEN_LEDGER_SHA):
        raise RecoveryError("reviewed_external_settlement_authority_changed")
    old_state = read(checked(root, value["open_V2_state_opaque"]))
    old_ledger = read(checked(root, value["open_original_ledger_opaque"]))
    if old_state.get("call_receipts") != value.get("original_six_closed_call_receipts"):
        raise RecoveryError("original_six_call_history_must_be_preserved")
    authority, _ = parent.verify_static_authority(root, plan)
    verify_six_closed_calls(root, plan, authority, old_state)
    checked(root, value["original_call07_before_closed_ledger"])
    for row in value.get("record_evidence_opaque", []):
        if not row.get("opaque_snapshot", {}).get("path", "").endswith(".txt"):
            raise RecoveryError("interrupted_record_evidence_must_be_opaque")
        checked(root, row["opaque_snapshot"])
    open_accounting(old_ledger, plan, value["accounting_at_authorization"]["observed_clock"])
    return value, ref(root, path(root, AMENDMENT))


def verify_settlement(root):
    """Pure reads/hashes/math: no locks, clients, Ledger or preparation."""
    root, plan, _, _, _ = load(root)
    authority, amendment_ref = verify_authority(root, plan)
    receipt_path = call_dir(root) / "receipt.json"
    receipt = read(receipt_path)
    expected = {"schema_version": parent.SCHEMA, "call_number": 7,
        "amendment": authority["original_V2_amendment"], "intent": authority["original_call07_intent"],
        "before_closed_ledger": authority["original_call07_before_closed_ledger"],
        "error_type": None, "error_code": "external_hard_interruption_no_worker_return_observed",
        "retryable_transport_error": False, "coordinator_result": None, "whole_job_prepare_calls": 1,
        "receipt_producer": ENTRY, "receipt_semantics": SCHEMA,
        "worker_returned_normally": False, "worker_result_observed": False,
        "external_settlement_status": STATUS, "operational_recovery_amendment": amendment_ref,
        "GPU_actions": 0, "training_started": False}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RecoveryError("external_receipt_must_not_claim_normal_return_HTTP_retry_or_COMPLETE")
    old = read(checked(root, authority["open_original_ledger_opaque"]))
    after, _ = parent.closed_ledger(root, plan, checked(root, receipt["after_closed_ledger"]))
    if "wall_intent" in after or "active_transfer" in after:
        raise RecoveryError("settled_ledger_must_have_no_open_intent_keys")
    expected_history = old["history"] + [{**old["active_transfer"], "status": "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}]
    settlement_clock = receipt["settlement_accounting"]["observed_clock"]
    accounting = open_accounting(old, plan, settlement_clock)
    if (after["history"] != expected_history or after["charged"] != old["charged"]
            or count_history(after) != after["charged"]
            or after["elapsed_seconds"] < accounting["minimum_cumulative_wall_seconds"]
            or after.get("explicit_call07_hard_interruption_amendment") != amendment_ref):
        raise RecoveryError("all_downtime_and_full_failed_reservation_must_be_conserved")
    state = read(parent_state(root))
    if (state.get("identity") != read(checked(root, authority["open_V2_state_opaque"]))["identity"]
            or state.get("active_call") is not None or state.get("calls_used") not in {7, 8}
            or state.get("call_receipts", [])[:6] != authority["original_six_closed_call_receipts"]
            or state.get("call_receipts", [])[6:7] != [ref(root, receipt_path)]):
        raise RecoveryError("original_call07_count_and_history_must_be_preserved")
    # Frozen V2 itself verifies its complete history, including this transparent
    # external receipt and a later genuine call08 if one has occurred.
    parent._history(root, plan, authority["original_V2_amendment"])
    current, _ = parent.closed_ledger(root, plan)
    parent._prefix(after, current)
    if current.get("explicit_call07_hard_interruption_amendment") != amendment_ref:
        raise RecoveryError("live_ledger_must_retain_recovery_provenance")
    return {"status": STATUS, "external_receipt": ref(root, receipt_path), "amendment": amendment_ref,
        "worker_returned_normally": False, "worker_result_observed": False, "retryable_transport_error": False,
        "original_V2_calls_used": state["calls_used"], "remaining_V2_calls": 8 - state["calls_used"],
        "scientific_fits": 0, "network_requests": 0, "GPU_actions": 0}


def run(root, *, execute=False):
    if not execute:
        root, _, plan_ref, _, _, accounting, death = preflight(root)
        return {"status": "DRY_RUN_EXTERNAL_CALL07_SETTLEMENT_NO_WRITES", "frozen_plan": plan_ref,
            "process_death": death, "accounting_if_settled_now": accounting, "calls_used_preserved": 7,
            "remaining_calls_after_settlement": 1, "network_requests": 0, "scientific_fits": 0}
    root = Path(root).resolve(strict=True)
    if (path(root, BASE + "/committed.json")).exists():
        return verify_settlement(root)
    with parent.owner(root), parent.previous.coordinator_lock(root):
        if (path(root, BASE + "/execution_claim.json")).exists():
            raise RecoveryError("issued_external_settlement_must_not_automatically_replay")
        root, plan, _, state, _, accounting, death = preflight(root)
        authority, amendment_ref = verify_authority(root, plan)
        claim = {"schema_version": SCHEMA, "amendment": amendment_ref, "settlement_invocations": 1,
                 "process_death_at_execution": death, "accounting_before_execution": accounting}
        core.atomic_json(path(root, BASE + "/execution_claim.json"), claim, immutable=True)
        old = read(source_ledger(root))
        if "explicit_call07_hard_interruption_amendment" in old:
            raise RecoveryError("existing_recovery_provenance_must_not_be_replaced")
        core.atomic_json(source_ledger(root), {**old, "explicit_call07_hard_interruption_amendment": amendment_ref})
        # Original durable clock and hard-interruption accounting, no transfers.
        ledger = core.Ledger(source_ledger(root), plan)
        ledger.checkpoint(close=True)
        after_ref = snapshot(root, source_ledger(root), parent.HISTORY + "/call-07/after_closed_ledger.json")
        receipt = {"schema_version": parent.SCHEMA, "call_number": 7,
            "amendment": authority["original_V2_amendment"], "intent": authority["original_call07_intent"],
            "before_closed_ledger": authority["original_call07_before_closed_ledger"], "after_closed_ledger": after_ref,
            "error_type": None, "error_code": "external_hard_interruption_no_worker_return_observed",
            "retryable_transport_error": False, "coordinator_result": None, "whole_job_prepare_calls": 1,
            "receipt_producer": ENTRY, "receipt_semantics": SCHEMA,
            "worker_returned_normally": False, "worker_result_observed": False,
            "external_settlement_status": STATUS, "operational_recovery_amendment": amendment_ref,
            "execution_claim": ref(root, path(root, BASE + "/execution_claim.json")),
            "settlement_accounting": accounting, "GPU_actions": 0, "training_started": False}
        receipt_path = call_dir(root) / "receipt.json"
        core.atomic_json(receipt_path, receipt, immutable=True)
        state = {**state, "call_receipts": [*state["call_receipts"], ref(root, receipt_path)], "active_call": None}
        core.atomic_json(parent_state(root), state)
        result = verify_settlement(root)
        core.atomic_json(path(root, BASE + "/committed.json"), {"schema_version": SCHEMA,
            "amendment": amendment_ref, "external_receipt": ref(root, receipt_path), "status": STATUS}, immutable=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write-amendment", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = write_amendment(args.root) if args.write_amendment else verify_settlement(args.root) if args.verify else run(args.root, execute=args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "BLOCKED_PRESERVE_ALL_OPEN_AND_RECOVERED_EVIDENCE",
            "error_type": type(error).__name__, "error_code": str(error) if isinstance(error,
            (RecoveryError, core.PreparationError, parent.TransportRecoveryError, parent.previous.TransportRecoveryError))
            else "local_evidence_unavailable_no_recovery", "network_requests": 0, "scientific_fits": 0}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
