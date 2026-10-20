#!/usr/bin/env python3
"""A root-authorized, single Dreyer CPU worker beside the frozen Yang worker.

This operational wrapper never changes the frozen preparation kernel or IO.
Default is read-only. Only the root coordinator starts --execute after review.
The original raw/cache GET proofs, scientific contracts and cumulative ledger
remain mandatory. No GPU, fit, external cohort or remote server is involved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DREYER_PLAN_SHA = "8a3b924f83e545de17f3529edbcd7f840424fcfa39f0ca56416a4f9f86bc5112"
YANG_PLAN_SHA = "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
OLD_DREYER_SHA = "c19448093361767715840f162297173951e064bae9e5e3f52075cf5bfee56b20"
ENTRY = "tools/prepare_dreyer_source_cpu.py"
AUDIT_ROOT = "audit/source_extensions_20261019"
PLAN_PATH = AUDIT_ROOT + "/Dreyer2023/plan_after_eof_review.json"
YANG_PATH = AUDIT_ROOT + "/Yang2025/plan.json"
DECLARATION_SCHEMA = "paper3-dreyer-parallel-root-declaration-v1"
EVIDENCE_SCHEMA = "paper3-dreyer-parallel-cpu-execution-v1"
LIMITS = {"cache_bytes": 512*1024**2, "minimum_free_disk_bytes": 4*1024**3,
          "maximum_anon_plus_kernel_bytes": 12*1024**3, "cpu_threads": 1, "nice": 10,
          "other_inflight_raw_bytes": 935203397+626191826,
          "dreyer_peak_memory_reserve_bytes": 512*1024**2}
AUTHORIZATION = (
    "Root authorized this standalone Dreyer CPU worker beside the unique frozen Yang worker "
    "and the existing Stieger worker. Dreyer uses this entrypoint only; ordinary Dreyer "
    "prepare is forbidden until this worker finishes. Original budgets and prior charges "
    "are retained. GPU, fit, sealed external data and remote server actions are forbidden."
)


class WrapperError(ValueError):
    pass


def kernel():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("p3work.source_extension_coordinator")


def read_json(path):
    return json.loads(Path(path).read_bytes())


def audit_path(root):
    return kernel().local(root, f"{AUDIT_ROOT}/Dreyer2023/{DREYER_PLAN_SHA}")


def implementation(root):
    core = kernel()
    path = core.local(root, ENTRY)
    if path.read_bytes() != Path(__file__).read_bytes():
        raise WrapperError("executed_wrapper_code_differs_from_root")
    return core.descriptor(root, path)


def fixed_plans(root, plan_path=PLAN_PATH):
    core = kernel()
    plan = read_json(core.local(root, plan_path))
    yang = read_json(core.local(root, YANG_PATH))
    if (plan.get("plan_sha256") != DREYER_PLAN_SHA or plan.get("dataset") != "Dreyer2023"
            or plan.get("record_count") != 520 or len(plan.get("records", [])) != 520
            or yang.get("plan_sha256") != YANG_PLAN_SHA or yang.get("dataset") != "Yang2025"):
        raise WrapperError("only_exact_root_authorized_Dreyer_and_Yang_plans_allowed")
    profile = core.validate_plan(root, plan)
    core.validate_plan(root, yang)
    return plan, yang, profile


def closed_history(ledger):
    if (ledger.get("schema_version") != "paper3-source-extension-budget-ledger-v1"
            or "wall_intent" in ledger or "active_transfer" in ledger):
        raise WrapperError("prior_ledger_must_be_closed_not_reinitialized")
    counted = {"provider_body_bytes": 0, "r2_get_body_bytes": 0}
    for row in ledger.get("history", []):
        kind, reserved = row.get("kind"), row.get("reserved_bytes")
        if kind not in counted or type(reserved) is not int or reserved < 0:
            raise WrapperError("prior_transfer_history_invalid")
        if row.get("status") == "SUCCESS_FULL_BODY_VERIFIED":
            value = row.get("verified_bytes")
            if type(value) is not int or not 0 <= value <= reserved:
                raise WrapperError("prior_transfer_history_invalid")
        elif row.get("status") in {"FAILURE_FULL_RESERVATION_CONSUMED", "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}:
            value = reserved
        else:
            raise WrapperError("prior_transfer_history_not_closed")
        counted[kind] += value
    if counted != ledger.get("charged"):
        raise WrapperError("prior_byte_charges_do_not_match_history")
    wall = ledger.get("elapsed_seconds")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise WrapperError("prior_wall_charge_invalid")
    return counted, wall


def prior_budget_evidence(root, plan):
    core = kernel()
    base = core.local(root, f"{AUDIT_ROOT}/Dreyer2023/{OLD_DREYER_SHA}")
    old_file, ledger_file = base/"frozen_plan.json", base/"budget_ledger.json"
    old, ledger = read_json(old_file), read_json(ledger_file)
    if (old.get("plan_sha256") != OLD_DREYER_SHA
            or core.digest({k:v for k,v in old.items() if k != "plan_sha256"}) != OLD_DREYER_SHA
            or ledger.get("plan_sha256") != OLD_DREYER_SHA):
        raise WrapperError("actual_prior_frozen_plan_and_ledger_required")
    charged, wall = closed_history(ledger)
    expected = {k: old["budgets"][k]-charged[k] for k in charged}
    expected["wall_seconds"] = old["budgets"]["wall_seconds"]-math.ceil(wall)
    if any(plan["budgets"][k] != value for k,value in expected.items()):
        raise WrapperError("prior_charges_must_be_deducted_from_remaining_plan_budget")
    return {"prior_plan": core.descriptor(root, old_file),
            "prior_closed_ledger": core.descriptor(root, ledger_file),
            "deducted_body_bytes": charged, "deducted_wall_seconds": math.ceil(wall),
            "carried_over_verified_R2_body_bytes": 0,
            "old_cache_or_receipts_adopted_as_new_records": False}


def process_fingerprint(pid):
    """No environment/credentials are read. PID reuse is rejected by boot/start."""
    base = Path("/proc")/str(pid)
    try:
        stat = (base/"stat").read_text()
        fields = stat[stat.rfind(")")+2:].split()
        command = [v.decode("utf-8") for v in (base/"cmdline").read_bytes().split(b"\0") if v]
        return {"pid": int(pid), "starttime_ticks": int(fields[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "command": command, "cwd": str((base/"cwd").resolve(strict=True))}
    except (OSError, ValueError, UnicodeError, IndexError):
        return None


def yang_identity(root, fingerprint):
    if not isinstance(fingerprint, dict):
        return None
    command = fingerprint["command"]
    entries = [v for v in command if v.endswith("tools/prepare_source_extensions.py")]
    if len(entries) != 1 or "--execute" not in command or command.count("--plan") != 1:
        return None
    try:
        script = Path(entries[0]); script = script if script.is_absolute() else Path(fingerprint["cwd"])/script
        option = command[command.index("--plan")+1]
        plan_file = Path(option); plan_file = plan_file if plan_file.is_absolute() else root/plan_file
        if script.resolve() != root/"tools/prepare_source_extensions.py" or plan_file.resolve() != root/YANG_PATH:
            return None
        if fingerprint["cwd"] != str(root) or read_json(plan_file).get("plan_sha256") != YANG_PLAN_SHA:
            return None
        return {k:fingerprint[k] for k in ("pid", "starttime_ticks", "boot_id")} | {
            "command_sha256": kernel().digest(command), "plan_sha256": YANG_PLAN_SHA}
    except (OSError, ValueError, IndexError):
        return None


def lock_owners(path):
    stat = path.stat()
    identity = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    owners = []
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or "->" in fields or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        try:
            device = fields[5].split(":")
            candidate = (int(device[0],16), int(device[1],16), int(device[2]))
            if candidate == identity:
                owners.append(int(fields[4]))
        except (ValueError, IndexError):
            raise WrapperError("global_lock_owner_cannot_be_verified") from None
    return owners


def current_yang(root):
    core = kernel()
    lock = core.local(root, AUDIT_ROOT+"/.source-extension-singleton.lock")
    owners = lock_owners(lock)
    if len(owners) != 1:
        raise WrapperError("exact_one_live_Yang_global_lock_owner_required")
    value = yang_identity(root, process_fingerprint(owners[0]))
    if value is None:
        raise WrapperError("global_lock_owner_is_not_exact_frozen_Yang_worker")
    return value


def build_declaration(root, plan):
    return {"schema_version": DECLARATION_SCHEMA, "dataset": "Dreyer2023",
            "plan_sha256": DREYER_PLAN_SHA, "concurrent_yang_plan_sha256": YANG_PLAN_SHA,
            "root_authorization": AUTHORIZATION, "only_dreyer_entrypoint": ENTRY,
            "ordinary_dreyer_prepare_forbidden": True,
            "initial_yang_process": current_yang(root), "wrapper_implementation": implementation(root),
            "prior_budget_evidence": prior_budget_evidence(root, plan), "limits": dict(LIMITS),
            "new_record_namespace": f"cache/source_extensions_20261019/Dreyer2023/{DREYER_PLAN_SHA}",
            "GPU_actions": 0, "actual_fits": 0, "sealed_external_access_allowed": False}


def verify_declaration(root, path, plan):
    core = kernel()
    path = core.local(root, path)
    value = read_json(path)
    fixed = {"schema_version": DECLARATION_SCHEMA, "dataset": "Dreyer2023",
        "plan_sha256": DREYER_PLAN_SHA, "concurrent_yang_plan_sha256": YANG_PLAN_SHA,
        "root_authorization": AUTHORIZATION, "only_dreyer_entrypoint": ENTRY,
        "ordinary_dreyer_prepare_forbidden": True, "wrapper_implementation": implementation(root),
        "prior_budget_evidence": prior_budget_evidence(root, plan), "limits": LIMITS,
        "new_record_namespace": f"cache/source_extensions_20261019/Dreyer2023/{DREYER_PLAN_SHA}",
        "GPU_actions": 0, "actual_fits": 0, "sealed_external_access_allowed": False}
    if (set(value) != set(fixed) | {"initial_yang_process"}
            or any(value.get(k) != v for k,v in fixed.items())
            or not isinstance(value["initial_yang_process"], dict)
            or value["initial_yang_process"].get("plan_sha256") != YANG_PLAN_SHA
            or path != audit_path(root)/"parallel_root_declaration.json"):
        raise WrapperError("immutable_root_declaration_or_code_budget_changed")
    return value, core.descriptor(root, path)


class PeerGuard:
    def __init__(self, root):
        self.root, self.core = root, kernel()
        self.path = self.core.local(root, AUDIT_ROOT+"/.source-extension-singleton.lock")
        self.handle = self.path.open("a+")
        self.owned, self.peer = False, None

    def check(self):
        if self.owned:
            return
        owners = lock_owners(self.path)
        if owners:
            if len(owners) != 1:
                raise WrapperError("unknown_global_lock_owner_stop_new_records")
            live = yang_identity(self.root, process_fingerprint(owners[0]))
            if live is None or self.peer is not None and live != self.peer:
                raise WrapperError("Yang_failed_replaced_or_unknown_owner_stop_new_records")
            self.peer = live
            return
        # A vanished/failed peer is never taken as success. Only the real global
        # completion receipt permits takeover. Root may later resume this same
        # wrapper/ledger beside a restarted, same-plan Yang worker.
        status_file = self.core.local(self.root, f"{AUDIT_ROOT}/Yang2025/{YANG_PLAN_SHA}/coordinator_status.json")
        if not status_file.is_file():
            raise WrapperError("Yang_not_complete_stop_new_records_preserve_ledger")
        status = read_json(status_file)
        if (status.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE"
                or status.get("dataset") != "Yang2025" or status.get("record_count") != 153):
            raise WrapperError("Yang_failed_or_paused_stop_new_records_preserve_ledger")
        manifest = read_json(self.core.checked(self.root, status.get("manifest")))
        if (manifest.get("plan_sha256") != YANG_PLAN_SHA or manifest.get("record_count") != 153
                or len(manifest.get("shards", [])) != 153):
            raise WrapperError("Yang_complete_manifest_identity_invalid")
        self.core.receipt_rows(self.root, status["backup_receipt"])
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WrapperError("global_lock_takeover_race_stop_new_records") from None
        self.owned = True

    def close(self):
        self.handle.close()


def resource_snapshot(root):
    stats = {}
    for row in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
        key, value = row.split()
        stats[key] = int(value)
    return {"free_disk_bytes": shutil.disk_usage(root).free,
            "anon_plus_kernel_bytes": stats["anon"]+stats["kernel"]}


def resource_guard(root, output, record_bytes=0):
    snapshot = resource_snapshot(root)
    if snapshot["free_disk_bytes"] < LIMITS["minimum_free_disk_bytes"]+LIMITS["other_inflight_raw_bytes"]+record_bytes:
        raise WrapperError("shared_disk_reserve_stop_before_new_IO")
    if snapshot["anon_plus_kernel_bytes"]+LIMITS["dreyer_peak_memory_reserve_bytes"] > LIMITS["maximum_anon_plus_kernel_bytes"]:
        raise WrapperError("shared_memory_guard_stop_before_new_IO")
    total = 0
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink():
                raise WrapperError("symlink_in_Dreyer_cache_forbidden")
            if path.is_file():
                total += path.stat().st_size
    if total > LIMITS["cache_bytes"]:
        raise WrapperError("tight_Dreyer_cache_budget_exhausted")
    return snapshot


def configure_cpu():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise WrapperError("CPU_affinity_unavailable")
    os.sched_setaffinity(0, {max(allowed)})
    os.setpriority(os.PRIO_PROCESS, 0, max(os.getpriority(os.PRIO_PROCESS, 0), LIMITS["nice"]))


@contextmanager
def dataset_lock(root):
    core = kernel()
    path = core.local(root, AUDIT_ROOT+"/Dreyer2023/.parallel-cpu-dataset.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WrapperError("Dreyer_dataset_worker_already_running") from None
        yield


def _finalize(root, plan, states, ledger, declaration_descriptor):
    """Identical kernel assembly, plus two completely GET-backed audit objects."""
    core = kernel()
    audit, dataset = audit_path(root), "Dreyer2023"
    output = core.local(root, f"cache/source_extensions_20261019/{dataset}/{DREYER_PLAN_SHA}")
    manifest = core.assemble_manifest(root, plan, states)
    native = {"schema_version": "paper3-source-extension-native-inventory-v1", "dataset": dataset,
        "records": [{"record_id":s["record_id"], "subject_id":s["subject_id"], "session_id":s["session_id"],
            "raw_files":s["raw_files"], "native_metadata":s["native_metadata"], "raw_sha256":s["raw_sha256"],
            "raw_bytes":s["raw_files"][0]["bytes"], "raw_backup_receipt":s["raw_backup_receipt"]}
            for s in manifest["shards"]]}
    core.atomic_json(audit/"native_inventory.json", native, immutable=True)
    core.atomic_json(audit/"unit_evidence.json", {"dataset":dataset,
        "records":[s["unit_evidence"] for s in manifest["shards"]]}, immutable=True)
    assignment = {"dataset":dataset, "role":"source_train", "source_only":True,
        "unique_role_within_branch":True, "development":False, "longitudinal_meta_ready":False,
        "external_evaluation":False, "new_unseen_external_claim":False,
        "document_sha256":core.storage.DOCUMENT_SHA256, "plan_sha256":DREYER_PLAN_SHA}
    core.atomic_json(audit/"role_assignment.json", assignment, immutable=True)
    evidence = {"schema_version":EVIDENCE_SCHEMA, "status":"RECORDS_VERIFIED_FINAL_BACKUP_PENDING",
        "dataset":dataset, "plan_sha256":DREYER_PLAN_SHA, "concurrent_yang_plan_sha256":YANG_PLAN_SHA,
        "wrapper_implementation":implementation(root), "root_declaration":declaration_descriptor,
        "record_count":520, "record_ids_sha256":core.digest([r["record_id"] for r in plan["records"]]),
        "scientific_kernel_unchanged":True, "limits":dict(LIMITS),
        "carried_over_verified_R2_body_bytes":0, "GPU_actions":0, "actual_fits":0,
        "external_payload_accessed":False, "server_actions":False}
    core.atomic_json(audit/"parallel_execution_receipt.json", evidence, immutable=True)
    for name in ("native_inventory", "unit_evidence", "role_assignment"):
        manifest[name] = core.descriptor(root, audit/(name+".json"))
    manifest["operational_evidence"] = core.descriptor(root, audit/"parallel_execution_receipt.json")
    manifest_path = output/"source_cache.json"
    core.atomic_json(manifest_path, manifest, immutable=True)
    files = [(manifest_path,"result")] + [(audit/name,"audit") for name in
        ("native_inventory.json", "unit_evidence.json", "frozen_plan.json", "role_assignment.json",
         "parallel_execution_receipt.json", "parallel_root_declaration.json")]
    receipt = core.verified_backup(root, audit/"final_backup", dataset=dataset, files=files,
        prefix=f"paper3/source-extensions-20261019/{dataset}/{DREYER_PLAN_SHA}/final/", ledger=ledger,
        backup_fn=core.storage.run_backup)
    ledger.checkpoint(close=True)
    result = {"status":"SOURCE_EXTENSION_CACHE_COMPLETE", "dataset":dataset,
        "manifest":core.descriptor(root,manifest_path), "backup_receipt":receipt,
        "record_count":520, "valid_trials":manifest["valid_trials"],
        "operational_evidence":manifest["operational_evidence"],
        "training_started":False, "GPU_actions":0, "external_payload_accessed":False}
    core.atomic_json(audit/"coordinator_status.json", result)
    return result


def verified_completion(root, plan, declaration_descriptor):
    """Read-only reuse: no fresh Ledger, provider request, PUT or GET after done."""
    core, audit = kernel(), audit_path(root)
    status_path = audit/"coordinator_status.json"
    if not status_path.exists():
        return None
    status = read_json(status_path)
    if status.get("status") != "SOURCE_EXTENSION_CACHE_COMPLETE":
        return None
    manifest_file = core.checked(root, status["manifest"])
    manifest = read_json(manifest_file)
    states = []
    for record in plan["records"]:
        state = read_json(core.local(root, (audit/record["record_id"]/"state.json").relative_to(root).as_posix()))
        if not core._state_verified(root, state, plan, record):
            raise WrapperError("complete_parallel_record_proof_missing")
        states.append(state)
    base = core.assemble_manifest(root, plan, states)
    if any(manifest.get(k) != v for k,v in base.items()):
        raise WrapperError("complete_parallel_manifest_not_actual_frozen_records")
    evidence_file = core.checked(root, manifest["operational_evidence"])
    evidence = read_json(evidence_file)
    required = {"schema_version":EVIDENCE_SCHEMA, "status":"RECORDS_VERIFIED_FINAL_BACKUP_PENDING",
        "dataset":"Dreyer2023", "plan_sha256":DREYER_PLAN_SHA,
        "concurrent_yang_plan_sha256":YANG_PLAN_SHA, "wrapper_implementation":implementation(root),
        "root_declaration":declaration_descriptor, "record_count":520,
        "record_ids_sha256":core.digest([r["record_id"] for r in plan["records"]]),
        "scientific_kernel_unchanged":True, "limits":LIMITS,
        "carried_over_verified_R2_body_bytes":0, "GPU_actions":0, "actual_fits":0,
        "external_payload_accessed":False, "server_actions":False}
    if evidence != required or status.get("operational_evidence") != manifest["operational_evidence"]:
        raise WrapperError("complete_parallel_operational_proof_changed")
    final = [(status["manifest"],"result")] + [(core.descriptor(root,audit/name),"audit") for name in
        ("native_inventory.json", "unit_evidence.json", "frozen_plan.json", "role_assignment.json",
         "parallel_execution_receipt.json", "parallel_root_declaration.json")]
    rows = core.receipt_rows(root, status["backup_receipt"])
    fields = ("path","sha256","bytes","role","dataset")
    expected = [{**item,"role":role,"dataset":"Dreyer2023"} for item,role in final]
    if (len(rows) != 7 or {tuple(r[k] for k in fields) for r in rows}
            != {tuple(r[k] for k in fields) for r in expected}):
        raise WrapperError("complete_parallel_final_seven_full_GET_proofs_required")
    ledger = read_json(audit/"budget_ledger.json")
    charged, wall = closed_history(ledger)
    minimum_get = sum(item["bytes"] for item,_ in final)
    for shard in manifest["shards"]:
        minimum_get += sum(f["bytes"] for f in shard["raw_files"])
        minimum_get += sum(shard[k]["bytes"] for k in
            ("X","y","events","exclusions","metadata","native_metadata","unit_evidence"))
    if (ledger.get("plan_sha256") != DREYER_PLAN_SHA or wall > plan["budgets"]["wall_seconds"]
            or any(value > plan["budgets"][k] for k,value in charged.items())
            or charged["r2_get_body_bytes"] < minimum_get
            or status.get("record_count") != 520 or status.get("valid_trials") != manifest["valid_trials"]
            or status.get("training_started") is not False or status.get("GPU_actions") != 0
            or status.get("external_payload_accessed") is not False):
        raise WrapperError("complete_parallel_closed_budget_or_status_invalid")
    return {**status,"actual_new_records_this_invocation":0,"new_provider_GETs_this_invocation":0,
            "new_R2_operations_this_invocation":0,"completed_local_proofs_reverified":True}


def execute(root, *, declaration_path, plan_path=PLAN_PATH, max_new_records=None):
    root = Path(root).resolve(strict=True)
    configure_cpu()
    core = kernel()
    plan, _, profile = fixed_plans(root, plan_path)
    _, declaration_descriptor = verify_declaration(root, declaration_path, plan)
    if max_new_records is not None and (type(max_new_records) is not int or max_new_records < 1):
        raise WrapperError("positive_record_limit_required")
    with dataset_lock(root):
        audit = audit_path(root)
        output = core.local(root, f"cache/source_extensions_20261019/Dreyer2023/{DREYER_PLAN_SHA}")
        audit.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
        core.atomic_json(audit/"frozen_plan.json", plan, immutable=True)
        claim_path = audit/"parallel_worker_claim.json"
        identity = {"plan_sha256":DREYER_PLAN_SHA, "root_declaration":declaration_descriptor,
                    "wrapper_implementation":implementation(root)}
        if claim_path.exists():
            claim = read_json(claim_path)
            if claim.get("identity") != identity or type(claim.get("invocations")) is not int:
                raise WrapperError("existing_parallel_claim_identity_changed")
            if claim["invocations"] > 0 and not (audit/"budget_ledger.json").is_file():
                raise WrapperError("dispatched_parallel_ledger_missing_do_not_restart")
        else:
            if (audit/"budget_ledger.json").exists() or list(audit.glob("*/state.json")):
                raise WrapperError("existing_parallel_work_missing_claim_do_not_restart")
            claim = {"identity":identity, "invocations":0, "status":"CLAIMED_NOT_STARTED"}
        core.atomic_json(claim_path, claim)
        complete = verified_completion(root, plan, declaration_descriptor)
        if complete is not None:
            if claim["invocations"] < 1:
                raise WrapperError("complete_parallel_run_requires_existing_dispatched_claim")
            claim.update(status="COMPLETE",active_invocation=False)
            core.atomic_json(claim_path,claim)
            return complete
        peer = PeerGuard(root)
        def guard():
            peer.check()
            resource_guard(root, output)
        class ControlledLedger(core.Ledger):
            def checkpoint(self, *, close=False):
                if not close:
                    guard()
                return super().checkpoint(close=close)
        ledger = None
        try:
            guard()
            ledger = ControlledLedger(audit/"budget_ledger.json", plan)
            claim.update(invocations=claim["invocations"]+1, status="RUNNING",
                         peer_at_invocation_start=peer.peer, active_invocation=True)
            core.atomic_json(claim_path, claim)
            adapter = importlib.import_module(core.ADAPTERS["Dreyer2023"])
            import requests
            with requests.Session() as session:
                states, new = [], 0
                for record in plan["records"]:
                    guard()
                    resource_guard(root, output, sum(f["bytes"] for f in record["raw_files"]))
                    old_path = audit/record["record_id"]/"state.json"
                    previous = read_json(old_path) if old_path.exists() else {}
                    complete = previous.get("status") == "COMPLETE"
                    if not complete and max_new_records is not None and new >= max_new_records:
                        break
                    state = core.process_record(root, plan, record, native_profile=profile,
                        ledger=ledger, session=session, materialize=adapter.materialize,
                        audit_native=adapter.audit_native)
                    states.append(state); new += int(not complete)
                    guard()
                if len(states) == 520:
                    result = _finalize(root, plan, states, ledger, declaration_descriptor)
                    claim["status"] = "COMPLETE"
                else:
                    result = {"status":"PAUSED_EXPLICIT_RECORD_LIMIT", "dataset":"Dreyer2023",
                        "verified_records":len(states), "total_records":520, "plan_sha256":DREYER_PLAN_SHA,
                        "training_started":False, "GPU_actions":0, "external_payload_accessed":False}
                    ledger.checkpoint(close=True)
                    core.atomic_json(audit/"coordinator_status.json", result)
                    claim["status"] = result["status"]
                return result
        except BaseException as exc:
            claim["status"] = "STOPPED_CLOSED_PRESERVE_ALL_RECORDS_AND_BUDGET"
            claim["error_type"] = type(exc).__name__
            claim["error_code"] = str(exc) if isinstance(exc,(WrapperError,core.PreparationError)) else "operation_failed_details_suppressed"
            raise
        finally:
            try:
                if ledger is not None:
                    ledger.checkpoint(close=True)
            finally:
                claim["active_invocation"] = False
                core.atomic_json(claim_path, claim)
                peer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", default=PLAN_PATH)
    parser.add_argument("--root-declaration")
    parser.add_argument("--write-declaration", action="store_true")
    parser.add_argument("--max-new-records", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        if args.write_declaration and args.execute:
            raise WrapperError("freeze_root_declaration_separately_before_execution")
        if args.execute:
            if not args.root_declaration:
                raise WrapperError("immutable_root_declaration_required")
            result = execute(root, declaration_path=args.root_declaration,
                             plan_path=args.plan, max_new_records=args.max_new_records)
        else:
            plan, _, _ = fixed_plans(root, args.plan)
            declaration = build_declaration(root, plan)
            result = {"status":"DRY_RUN_NO_NETWORK_R2_OR_FIT", "root_declaration":declaration,
                      "GPU_actions":0, "actual_fits":0, "training_started":False}
            if args.write_declaration:
                path = audit_path(root)/"parallel_root_declaration.json"
                kernel().atomic_json(path, declaration, immutable=True)
                result["root_declaration_artifact"] = kernel().descriptor(root,path)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        core = kernel()
        print(json.dumps({"status":"STOPPED_CLOSED", "error_type":type(exc).__name__,
            "error_code":str(exc) if isinstance(exc,(WrapperError,core.PreparationError)) else "operation_failed_details_suppressed",
            "GPU_actions":0, "actual_fits":0, "training_started":False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
