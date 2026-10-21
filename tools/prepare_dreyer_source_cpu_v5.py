#!/usr/bin/env python3
"""Version 5 candidate of the root-authorized Dreyer CPU worker, in a new namespace.

This operational wrapper never changes the frozen preparation kernel or IO.
Default is read-only. Only the root coordinator starts --execute after review.
The original raw/cache GET proofs, scientific contracts and cumulative ledger
remain mandatory. No GPU, fit, external cohort or remote server is involved.
This is a static, independently hashed fork of frozen parent wrapper 3bdfd19d;
the parent file and all running preparation/IO files are never patched.
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
DREYER_PLAN_SHA = "afc8dd0d2d873f455e901c6f702fc2d5649fde15f76873c40a24424c10d48e5a"
NATIVE_ADAPTER_SHA = "5c31e969eeabd17634864f6783d8473a0a79f1888faf274d758cd9c5ba4cef77"
QUALIFICATION_COMPLETION_SHA = "585ade4a3d6101c3d922466fbcacbc89ddac907ff6978a9a0f608c834737d35e"
ADMISSION_REVIEW_SHA = "560ec9bdd0ddba53f5b926beaebdd153c61fa0ca8de1e2033273f9d8d022eed9"
YANG_PLAN_SHA = "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
OLD_DREYER_SHA = "c19448093361767715840f162297173951e064bae9e5e3f52075cf5bfee56b20"
V2_DREYER_SHA = "8a3b924f83e545de17f3529edbcd7f840424fcfa39f0ca56416a4f9f86bc5112"
V3_DREYER_SHA = "0fbc5d8d5fc6181649ea1c2ac87bc570294ee9264f45da234116a9f5876463ed"
V4_DREYER_SHA = "971c1a1272fdb25b0f21e7d69c4e7349593d352341bdaa2eaa0725b600dbc1d1"
QUALIFICATION_PLAN_SHA = "f39a24a3d26c6eb3de6e286e2efb8afc032397b91626f84fd647ae2c182b5efd"
AUXILIARY_SHA = "7446e8688214951ed56c7bd24ebb80f23aa7c3271d7abe3a5a758fb99c1024d3"
PARENT_WRAPPER_PATH = "tools/prepare_dreyer_source_cpu_v4.py"
PARENT_WRAPPER_SHA = "3bdfd19d9d1596f92b689121f8231d8b2403449ecbc26dd0b27c0f7ddee59a1a"
ENTRY = "tools/prepare_dreyer_source_cpu_v5.py"
AUDIT_ROOT = "audit/source_extensions_20261019"
HISTORY_SOURCES = (
    ("V1", f"{AUDIT_ROOT}/Dreyer2023/{OLD_DREYER_SHA}/frozen_plan.json",
     f"{AUDIT_ROOT}/Dreyer2023/{OLD_DREYER_SHA}/budget_ledger.json", OLD_DREYER_SHA),
    ("V2", f"{AUDIT_ROOT}/Dreyer2023/{V2_DREYER_SHA}/frozen_plan.json",
     f"{AUDIT_ROOT}/Dreyer2023/{V2_DREYER_SHA}/budget_ledger.json", V2_DREYER_SHA),
    ("auxiliary_schema_review", "audit/data_plan_20261019/dreyer/all_run_schema_review/plan.json",
     "audit/data_plan_20261019/dreyer/all_run_schema_review/budget_ledger.json", AUXILIARY_SHA),
    ("V3", f"{AUDIT_ROOT}/Dreyer2023/{V3_DREYER_SHA}/frozen_plan.json",
     f"{AUDIT_ROOT}/Dreyer2023/{V3_DREYER_SHA}/budget_ledger.json", V3_DREYER_SHA),
    ("V4", f"{AUDIT_ROOT}/Dreyer2023/{V4_DREYER_SHA}/frozen_plan.json",
     f"{AUDIT_ROOT}/Dreyer2023/{V4_DREYER_SHA}/budget_ledger.json", V4_DREYER_SHA),
    ("native_qualification", f"{AUDIT_ROOT}/Dreyer2023/native_structure_qualification/plan.json",
     f"{AUDIT_ROOT}/Dreyer2023/native_structure_qualification/{QUALIFICATION_PLAN_SHA}/budget_ledger.json", QUALIFICATION_PLAN_SHA),
)
PLAN_PATH = AUDIT_ROOT + "/Dreyer2023/plan_after_all520_native_review_v5.json"
YANG_PATH = AUDIT_ROOT + f"/Yang2025/{YANG_PLAN_SHA}/frozen_plan.json"
YANG_ENTRY = "tools/resume_yang_source_transport_v2.py"
YANG_WRAPPER_SHA = "2ae1e75e47a75032155da44c1cb45afa78a1ec82da3ecd11d6f85ec60e1c0f71"
YANG_TRANSPORT_MODULE = "tools.resume_yang_source_transport_v2"
YANG_SCHEMA = "paper3-yang-sameplan-no-cache-transport-recovery-v2"
YANG_MAX_CALLS = 8
YANG_OWNER_PATH = "audit/data_plan_20261020/yang_transport_recovery_v2/no_cache_wrapper_v2/owner.lock"
YANG_HISTORY_PATH = "audit/data_plan_20261020/yang_transport_recovery_v2/no_cache_wrapper_v2/recovery_ledger.json"
YANG_AMENDMENT_PATH = "plans/data_plan_20261019/yang_transport_recovery_v2_amendment.json"
QUALIFICATION_COMPLETION_PATH = "audit/data_plan_20261020/dreyer_structure_case_review/all520_closed_qualification_proof.json"
ADMISSION_REVIEW_PATH = "audit/data_plan_20261020/dreyer_structure_case_review/v5_native_admission.json"
INTERSECT_QUALIFICATION_STAGE_CAP = False
PROVIDER_HEADERS = {"Cache-Control":"no-cache","Pragma":"no-cache"}
DECLARATION_SCHEMA = "paper3-dreyer-parallel-root-declaration-v5"
EVIDENCE_SCHEMA = "paper3-dreyer-parallel-cpu-execution-v5"
LIMITS = {"cache_bytes": 512*1024**2, "minimum_free_disk_bytes": 4*1024**3,
          "maximum_anon_plus_kernel_bytes": 12*1024**3, "cpu_threads": 1, "nice": 10,
          "other_inflight_raw_bytes": 935203397+626191826,
          "dreyer_peak_memory_reserve_bytes": 512*1024**2}
AUTHORIZATION = (
    "Root authorized this version 5 standalone Dreyer CPU worker beside the unique frozen Yang NoCache V2 transport wrapper "
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
    parent = core.descriptor(root, core.local(root, PARENT_WRAPPER_PATH))
    if parent["sha256"] != PARENT_WRAPPER_SHA:
        raise WrapperError("frozen_parent_wrapper_must_remain_unchanged")
    return core.descriptor(root, path)


def fixed_plans(root, plan_path=PLAN_PATH):
    core = kernel()
    if any(v is None for v in (DREYER_PLAN_SHA,NATIVE_ADAPTER_SHA,QUALIFICATION_COMPLETION_SHA,ADMISSION_REVIEW_SHA,YANG_WRAPPER_SHA)):
        raise WrapperError("version5_plan_not_frozen_execution_and_declaration_blocked")
    plan = read_json(core.local(root, plan_path))
    yang = read_json(core.local(root, YANG_PATH))
    if (plan.get("plan_sha256") != DREYER_PLAN_SHA or plan.get("dataset") != "Dreyer2023"
            or plan.get("record_count") != 520 or len(plan.get("records", [])) != 520
            or plan.get("code", {}).get("p3work/source_extensions/dreyer2023.py", {}).get("sha256") != NATIVE_ADAPTER_SHA
            or yang.get("plan_sha256") != YANG_PLAN_SHA or yang.get("dataset") != "Yang2025"):
        raise WrapperError("only_exact_root_authorized_Dreyer_and_Yang_plans_allowed")
    profile = core.validate_plan(root, plan)
    core.validate_plan(root, yang)
    qualification_evidence(root,plan)
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


def remaining_budget_evidence(root):
    """Six actual closed journals, original pool and conservative lineage bounds."""
    core = kernel()
    expected = ["V1","V2","auxiliary_schema_review","V3","V4","native_qualification"]
    if [r[0] for r in HISTORY_SOURCES] != expected:
        raise WrapperError("all_six_original_history_sources_are_required_once")
    rows, details, seen = [], {}, set()
    total = {"provider_body_bytes":0,"r2_get_body_bytes":0}
    walls = []
    for label,plan_path,ledger_path,expected_sha in HISTORY_SOURCES:
        plan_file,ledger_file = core.local(root,plan_path),core.local(root,ledger_path)
        if ledger_file in seen:
            raise WrapperError("duplicate_prior_ledger_would_double_count_usage")
        seen.add(ledger_file)
        old,ledger = read_json(plan_file),read_json(ledger_file)
        if (old.get("plan_sha256") != expected_sha
                or core.digest({k:v for k,v in old.items() if k!="plan_sha256"}) != expected_sha
                or ledger.get("plan_sha256") != expected_sha):
            raise WrapperError("actual_prior_frozen_plan_and_ledger_required")
        charged,wall = closed_history(ledger)
        budgets = old.get("budgets",{})
        if (any(type(budgets.get(k)) is not int or budgets[k]<=0 or charged[k]>budgets[k] for k in charged)
                or type(budgets.get("wall_seconds")) is not int or budgets["wall_seconds"]<=0
                or wall>budgets["wall_seconds"]):
            raise WrapperError("actual_prior_usage_outside_its_original_authorized_budget")
        caps = {k:budgets[k] for k in (*total,"wall_seconds")}
        details[label] = {"authorized_caps":caps,"charged":charged,"wall":wall}
        for key,value in charged.items():total[key]+=value
        walls.append(wall)
        rows.append({"source":label,"plan":core.descriptor(root,plan_file),
            "closed_ledger":core.descriptor(root,ledger_file),"plan_sha256":expected_sha,
            "charged_body_bytes":charged,"elapsed_seconds":wall})
    baseline = details["V1"]["authorized_caps"]
    elapsed = math.fsum(walls); rounded = math.ceil(elapsed)
    initial = {k:baseline[k]-v for k,v in total.items()}
    initial["wall_seconds"] = baseline["wall_seconds"]-rounded
    bounds = []
    # Auxiliary review is a separately scoped journal, not a parent budget pool.
    for label in ("V2","V3","V4"):
        descendants = expected[expected.index(label):]
        charged = {k:sum(details[x]["charged"][k] for x in descendants) for k in total}
        wall = math.fsum(details[x]["wall"] for x in descendants)
        caps = details[label]["authorized_caps"]
        residual = {k:caps[k]-v for k,v in charged.items()}
        residual["wall_seconds"] = math.floor(caps["wall_seconds"]-wall)
        bounds.append({"source":label,"scope":"conserved_mainline_ancestor_caps",
            "authorized_caps":caps,"descendant_actual_sources":descendants,
            "descendant_actual_charged":charged,"descendant_actual_elapsed_seconds":wall,
            "remaining_caps":residual})
    ancestor = {k:min(initial[k],*(r["remaining_caps"][k] for r in bounds)) for k in initial}
    if INTERSECT_QUALIFICATION_STAGE_CAP:
        q = details["native_qualification"];caps=q["authorized_caps"]
        residual = {k:caps[k]-v for k,v in q["charged"].items()}
        residual["wall_seconds"] = math.floor(caps["wall_seconds"]-q["wall"])
        bounds.append({"source":"native_qualification",
            "scope":"metadata_stage_residual_as_conservative_upper_bound_never_new_global_baseline",
            "authorized_caps":caps,"descendant_actual_sources":["native_qualification"],
            "descendant_actual_charged":q["charged"],"descendant_actual_elapsed_seconds":q["wall"],
            "remaining_caps":residual})
    remaining = {k:min(initial[k],*(r["remaining_caps"][k] for r in bounds)) for k in initial}
    if any(v<=0 for v in remaining.values()):
        raise WrapperError("aggregate_original_budget_exhausted_no_new_plan")
    return {"accounting_schema":"paper3-dreyer-six-ledger-budget-v1",
        "original_caps":baseline,"closed_history_sources":rows,"deducted_body_bytes":total,
        "deducted_elapsed_seconds":elapsed,"deducted_wall_seconds":rounded,
        "wall_rounding":"ceil_sum_actual_elapsed_once_then_floor_each_ancestor_residual",
        "original_caps_minus_all_actual_usage":initial,"conservation_bounds":bounds,
        "remaining_cap_policy":"intersection_of_original_pool_and_all_declared_conservation_bounds",
        "remaining_caps":remaining,"uses_initial_V1_caps_as_global_baseline":True,
        "retained_ancestor_wall_rounding_seconds":initial["wall_seconds"]-ancestor["wall_seconds"],
        "qualification_stage_cap_intersected":INTERSECT_QUALIFICATION_STAGE_CAP,
        "carried_over_verified_R2_body_bytes":0,"old_cache_or_receipts_adopted_as_new_records":False}


def prior_budget_evidence(root, plan):
    evidence = remaining_budget_evidence(root)
    if any(plan["budgets"][k] != v for k,v in evidence["remaining_caps"].items()):
        raise WrapperError("all_six_actual_prior_charges_must_be_deducted_from_remaining_plan_budget")
    return evidence


def qualification_evidence(root, plan):
    """Verify frozen historical closure without executing its current-path validator."""
    core = kernel()
    if QUALIFICATION_COMPLETION_SHA is None or ADMISSION_REVIEW_SHA is None:
        raise WrapperError("all520_qualification_and_admission_not_frozen")
    proof_file = core.local(root,QUALIFICATION_COMPLETION_PATH)
    admission_file = core.local(root,ADMISSION_REVIEW_PATH)
    proof_ref,admission_ref = core.descriptor(root,proof_file),core.descriptor(root,admission_file)
    if proof_ref["sha256"] != QUALIFICATION_COMPLETION_SHA or admission_ref["sha256"] != ADMISSION_REVIEW_SHA:
        raise WrapperError("frozen_all520_proof_or_admission_changed")
    proof,admission = read_json(proof_file),read_json(admission_file)
    if (proof.get("schema_version") != "paper3-dreyer-all520-closed-qualification-proof-v1"
            or proof.get("offline_all520_verified_before_canonical_change") is not True
            or proof.get("scientific_admission_granted") is not False):
        raise WrapperError("exact_completed_metadata_only_historical_proof_required")
    qplan = read_json(core.checked(root,proof["qualification_plan"]))
    status = read_json(core.checked(root,proof["qualification_status"]))
    summary = read_json(core.checked(root,proof["qualification_summary"]))
    qledger = read_json(core.checked(root,proof["closed_ledger"]))
    sources={row[0]:row for row in HISTORY_SOURCES}
    qualification_source=sources["native_qualification"]
    parent_source=sources["V4"]
    if (proof["qualification_plan"]!=core.descriptor(root,core.local(root,qualification_source[1]))
            or proof["closed_ledger"]!=core.descriptor(root,core.local(root,qualification_source[2]))
            or qplan.get("parent_closed_ledger")!=core.descriptor(root,core.local(root,parent_source[2]))):
        raise WrapperError("qualification_and_parent_ledgers_must_match_actual_six_history_sources")
    if (qplan.get("plan_sha256") != QUALIFICATION_PLAN_SHA
            or core.digest({k:v for k,v in qplan.items() if k!="plan_sha256"}) != QUALIFICATION_PLAN_SHA
            or qplan.get("record_count") != 520 or qplan.get("records") != plan["records"]
            or qplan.get("scientific_admission_granted") is not False
            or qledger.get("plan_sha256") != QUALIFICATION_PLAN_SHA
            or status.get("status") != "ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY"
            or status.get("scientific_admission_granted") is not False
            or status.get("summary") != proof["qualification_summary"]
            or summary.get("plan") != proof["qualification_plan"]
            or summary.get("records") != 520 or summary.get("scientific_admission_granted") is not False
            or summary.get("status") != status["status"] or summary.get("EEG_array_samples_read") is not False
            or summary.get("fits") != 0 or summary.get("GPU_actions") != 0):
        raise WrapperError("all520_historical_qualification_identity_or_scope_invalid")
    aliases = proof.get("historical_code_aliases",{})
    if set(aliases) != set(qplan["code"]):
        raise WrapperError("all_historical_code_aliases_required")
    for name,original in qplan["code"].items():
        alias = aliases[name];core.checked(root,alias)
        if any(alias.get(k) != original.get(k) for k in ("sha256","bytes")):
            raise WrapperError("historical_code_alias_bytes_differ_from_frozen_qualification")
    additional=proof.get("additional_historical_proofs",[])
    if not isinstance(additional,list):raise WrapperError("historical_transport_proofs_must_be_descriptors")
    for item in additional:core.checked(root,item)
    parent = read_json(core.checked(root,qplan["parent_plan"]))
    if (parent.get("plan_sha256") != V4_DREYER_SHA
            or core.digest({k:v for k,v in parent.items() if k!="plan_sha256"}) != V4_DREYER_SHA
            or parent.get("records") != qplan["records"]):
        raise WrapperError("exact_historical_V4_parent_plan_required")
    refs = qplan.get("reused_complete_parent_states",{})
    if len(refs) != 128 or len(summary.get("record_states",[])) != 520:
        raise WrapperError("all520_states_and_exact128_historical_references_required")
    state_refs,raw_owners = [],[]
    minimum_r2 = minimum_provider = 0
    for record,item in zip(plan["records"],summary["record_states"]):
        state = read_json(core.checked(root,item));rid=record["record_id"]
        if (state.get("status") != "STRUCTURE_QUALIFICATION_RECORDED"
                or state.get("plan_sha256") != QUALIFICATION_PLAN_SHA or state.get("record_id") != rid
                or state.get("scientific_admission_granted") is not False
                or state.get("EEG_array_samples_read") is not False):
            raise WrapperError("all520_exact_metadata_only_states_required")
        native_file = core.checked(root,state["native_structure"]);native=read_json(native_file)
        if rid in refs:
            if state.get("reused_parent_state") != refs[rid]:
                raise WrapperError("immutable_old128_reference_binding_changed")
            old = read_json(core.checked(root,refs[rid]))
            if not core._state_verified(root,old,parent,record) or state["native_structure"] not in old["artifacts"]:
                raise WrapperError("old128_reference_requires_complete_historical_cache_proofs")
            raw=record["raw_files"][0]
            if (native.get("source_file_sha256") != raw["sha256"] or native.get("source_file_bytes") != raw["bytes"]
                    or state.get("existing_V4_gate") != {"status":"EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS"}):
                raise WrapperError("old128_original_native_identity_changed")
        else:
            core.verify_raw_receipt(root,state["raw_backup_receipt"],parent,record)
            raw=record["raw_files"][0]
            if (native.get("source_file") != {k:raw[k] for k in ("path","sha256","bytes")}
                    or native.get("annotations_modified_or_synthesized") is not False
                    or native.get("scientific_admission_granted") is not False
                    or native.get("EEG_array_samples_read") is not False):
                raise WrapperError("qualified_native_raw_identity_and_metadata_only_scope_required")
            expected=dict(state["native_structure"],dataset="Dreyer2023",role="audit")
            got=core.receipt_rows(root,state["metadata_backup_receipt"])
            if len(got)!=1 or any(got[0].get(k)!=expected[k] for k in core.storage.ENTRY_FIELDS):
                raise WrapperError("qualified_native_metadata_fresh_full_GET_proof_required")
            minimum_r2 += sum(r["bytes"] for r in record["raw_files"])+state["native_structure"]["bytes"]
        if state.get("owned_raw"):
            raw=record["raw_files"][0]
            owner={k:raw[k] for k in ("path","sha256","bytes")}
            owner["created_by"]="tools/qualify_dreyer_native_structures.py"
            if state["owned_raw"] != owner:
                raise WrapperError("qualifier_raw_ownership_identity_changed")
            raw_owners.append(owner);minimum_provider+=owner["bytes"]
        state_refs.append(item)
    expected=[dict(proof["qualification_plan"],dataset="Dreyer2023",role="audit"),
        dict(proof["qualification_summary"],dataset="Dreyer2023",role="audit"),
        *[dict(r,dataset="Dreyer2023",role="audit") for r in state_refs]]
    rows=core.receipt_rows(root,status["backup_receipt"]);fields=core.storage.ENTRY_FIELDS
    if (len(rows)!=522 or {tuple(r[k] for k in fields) for r in rows}
            != {tuple(r[k] for k in fields) for r in expected}):
        raise WrapperError("exact522_completed_qualification_final_full_GET_proofs_required")
    charged,wall=closed_history(qledger)
    minimum_r2+=sum(r["bytes"] for r in expected)
    if (any(charged[k]>qplan["budgets"][k] for k in charged) or wall>qplan["budgets"]["wall_seconds"]
            or charged["r2_get_body_bytes"]<minimum_r2 or charged["provider_body_bytes"]<minimum_provider):
        raise WrapperError("closed_qualification_budget_below_real_proofs_or_above_caps")
    expected_admission={"schema_version":"paper3-dreyer-v5-native-admission-v1",
        "status":"ALL520_NATIVE_ADMISSION_FROZEN","qualification_completion_proof":proof_ref,
        "adapter_sha256":NATIVE_ADAPTER_SHA,"record_count":520,
        "record_ids_sha256":core.digest([r["record_id"] for r in plan["records"]]),
        "fits":0,"GPU_actions":0,"external_payload_access_allowed":False}
    if any(admission.get(k)!=v for k,v in expected_admission.items()):
        raise WrapperError("all520_explicit_new_adapter_admission_required")
    # A descriptor-bound all-record decision/grammar inventory is required;
    # its scientific rules are implemented by the newly frozen native parser.
    core.checked(root,admission["all520_review"])
    return {"completion_proof":proof_ref,"admission_review":admission_ref,
        "qualification_plan":proof["qualification_plan"],"qualification_status":proof["qualification_status"],
        "qualification_summary":proof["qualification_summary"],"closed_ledger":proof["closed_ledger"],
        "parent_plan":qplan["parent_plan"],
        "additional_historical_proofs":additional,
        "historical_code_aliases":aliases,"reference_states":refs,"qualifier_owned_raw":raw_owners,
        "record_count":520,"reference_record_count":128,"historical_metadata_only_not_new_cache":True}


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


def yang_authority(root):
    """Use V2's pure static verifier; never read the active original Yang ledger."""
    if YANG_WRAPPER_SHA is None:raise WrapperError("Yang_V2_peer_code_not_frozen")
    core=kernel();wrapper=core.descriptor(root,core.local(root,YANG_ENTRY))
    if wrapper["sha256"]!=YANG_WRAPPER_SHA:raise WrapperError("frozen_Yang_NoCache_V2_wrapper_changed")
    transport=importlib.import_module(YANG_TRANSPORT_MODULE)
    plan_file=core.local(root,YANG_PATH);plan=read_json(plan_file)
    if plan.get("plan_sha256")!=YANG_PLAN_SHA:raise WrapperError("exact_original_Yang_plan_required")
    transport.verify_static_authority(root,plan)
    filename=core.local(root,YANG_AMENDMENT_PATH);amendment=read_json(filename)
    expected={"schema_version":YANG_SCHEMA,"status":"AUTHORIZED_SINGLE_CALL_SAMEPLAN_TRANSPORT_V2",
        "frozen_plan":core.descriptor(root,plan_file),"wrapper_code":wrapper,
        "session_headers":{"Cache-Control":"no-cache","Pragma":"no-cache"},
        "requests_adapter_retries":0,"max_independent_execute_calls":YANG_MAX_CALLS,
        "no_tight_retry":True,"original_kernel_is_only_transfer_and_wall_budget_authority":True}
    if any(core.digest(amendment.get(k))!=core.digest(v)for k,v in expected.items()):
        raise WrapperError("frozen_Yang_NoCache_V2_static_authority_changed")
    return wrapper,core.descriptor(root,filename),amendment


def yang_identity(root, fingerprint):
    """Identify the real persistent NoCache wrapper, never the old CLI disguise."""
    if not isinstance(fingerprint,dict):return None
    core=kernel();command=fingerprint["command"]
    try:
        entries=[v for v in command if v.endswith(YANG_ENTRY)]
        if (len(entries)!=1 or command.count("--execute")!=1 or "--write-amendment" in command
                or command.count("--plan")>1 or command.count("--root")>1
                or command.count("--max-new-records")>1):return None
        prefix=command[1:command.index(entries[0])]
        if any(value not in {"-u","-B"} for value in prefix):return None
        script=Path(entries[0]);script=script if script.is_absolute() else Path(fingerprint["cwd"])/script
        if script.resolve()!=root/YANG_ENTRY or fingerprint["cwd"]!=str(root):return None
        arguments=command[command.index(entries[0])+1:]
        allowed={"--execute","--plan","--root","--max-new-records"};index=0
        while index<len(arguments):
            option=arguments[index]
            if option not in allowed:return None
            index+=1
            if option!="--execute":
                if index>=len(arguments):return None
                if option=="--max-new-records" and (not arguments[index].isdecimal() or int(arguments[index])<1):return None
                index+=1
        requested=command[command.index("--plan")+1] if "--plan" in command else YANG_PATH
        filename=Path(requested);filename=filename if filename.is_absolute() else root/filename
        requested_root=Path(command[command.index("--root")+1]) if "--root" in command else root
        selected_root=(requested_root if requested_root.is_absolute() else root/requested_root).resolve()
        if filename.resolve()!=root/YANG_PATH or selected_root!=root:return None
        wrapper,amendment_ref,_=yang_authority(root)
        if read_json(filename).get("plan_sha256")!=YANG_PLAN_SHA:return None
        state=read_json(core.local(root,YANG_HISTORY_PATH))
        identity={"schema_version":YANG_SCHEMA,
            "amendment":amendment_ref,"plan_sha256":YANG_PLAN_SHA}
        count,active=state.get("calls_used"),state.get("active_call")
        receipts=state.get("call_receipts")
        if (state.get("identity")!=identity or type(count)is not int or not 0<=count<=YANG_MAX_CALLS
                or not isinstance(receipts,list)
                or active is not None and (type(active)is not int or active!=count or not 1<=active<=YANG_MAX_CALLS)
                or len(receipts)!=count-int(active is not None)
                or state.get("terminal") not in (None,"SOURCE_EXTENSION_CACHE_COMPLETE")
                or state.get("terminal")=="SOURCE_EXTENSION_CACHE_COMPLETE" and active is not None):return None
        return {k:fingerprint[k] for k in ("pid","starttime_ticks","boot_id")} | {
            "mode":"concurrent_no_cache_wrapper_v2","command_sha256":core.digest(command),
            "plan_sha256":YANG_PLAN_SHA,"wrapper_implementation":wrapper,"transport_amendment":amendment_ref}
    except (OSError,ValueError,IndexError,KeyError,TypeError):return None


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


def completed_yang(root):
    """Delegate full V1+V2/source closure to the frozen lock-free V2 verifier."""
    core=kernel();plan=read_json(core.local(root,YANG_PATH));core.validate_plan(root,plan)
    if (plan.get("plan_sha256")!=YANG_PLAN_SHA or plan.get("dataset")!="Yang2025"
            or plan.get("record_count")!=153 or len(plan.get("records",[]))!=153):
        raise WrapperError("exact_original_Yang153_plan_required_for_completion")
    wrapper,amendment_ref,_=yang_authority(root)
    transport=importlib.import_module(YANG_TRANSPORT_MODULE)
    result=transport.verify_complete_transport_history(root,plan)
    if result is None:raise WrapperError("Yang_not_complete_stop_new_records_preserve_ledger")
    if (result.get("status")!="SOURCE_EXTENSION_CACHE_COMPLETE"
            or result.get("already_complete_offline_verified")is not True or result.get("verified_records")!=153):
        raise WrapperError("Yang_V2_offline_verifier_did_not_prove_all153_complete")
    audit=core.local(root,f"{AUDIT_ROOT}/Yang2025/{YANG_PLAN_SHA}")
    status_file=audit/"coordinator_status.json";status=read_json(status_file)
    ledger_ref=core.descriptor(root,audit/"budget_ledger.json")
    history=result.get("transport_history",{})
    if not isinstance(history,dict):
        raise WrapperError("actual_completed_Yang_V2_history_required")
    if (status.get("status")!="SOURCE_EXTENSION_CACHE_COMPLETE" or status.get("dataset")!="Yang2025"
            or status.get("record_count")!=153 or status.get("training_started")is not False
            or status.get("GPU_actions")!=0 or status.get("external_payload_accessed")is not False
            or result.get("manifest")!=status.get("manifest") or result.get("backup_receipt")!=status.get("backup_receipt")
            or result.get("original_closed_ledger")!=ledger_ref
            or history.get("amendment")!=amendment_ref
            or history.get("recovery_ledger")!=core.descriptor(root,core.local(root,YANG_HISTORY_PATH))
            or not isinstance(history.get("provenance_closure"),dict) or not history["provenance_closure"]):
        raise WrapperError("Yang_V2_full_completion_proof_identity_changed")
    state=read_json(core.checked(root,history["recovery_ledger"]))
    identity={"schema_version":YANG_SCHEMA,"amendment":amendment_ref,"plan_sha256":YANG_PLAN_SHA}
    count=state.get("calls_used")
    if (state.get("identity")!=identity or type(count)is not int or not 1<=count<=YANG_MAX_CALLS
            or state.get("active_call")is not None or state.get("terminal")!="SOURCE_EXTENSION_CACHE_COMPLETE"
            or not isinstance(state.get("call_receipts"),list) or len(state["call_receipts"])!=count
            or history.get("calls_used")!=count):
        raise WrapperError("actual_Yang_V2_call_terminal_and_identity_required")
    core.checked(root,result["manifest"]);core.checked(root,result["backup_receipt"])
    return {"mode":"already_complete","plan_sha256":YANG_PLAN_SHA,
        "coordinator_status":core.descriptor(root,status_file),"manifest":result["manifest"],
        "backup_receipt":result["backup_receipt"],"closed_ledger":ledger_ref,
        "transport_history":history,"transport_completion_proof":result,
        "transport_amendment":amendment_ref,"wrapper_implementation":wrapper}


def current_yang(root):
    core=kernel();owner=core.local(root,YANG_OWNER_PATH)
    owners=lock_owners(owner) if owner.exists() else []
    global_lock=core.local(root,AUDIT_ROOT+"/.source-extension-singleton.lock")
    global_owners=lock_owners(global_lock) if global_lock.exists() else []
    if len(owners)==1:
        value=yang_identity(root,process_fingerprint(owners[0]))
        if value is None or global_owners not in ([],[owners[0]]):
            raise WrapperError("unknown_or_changed_NoCache_Yang_peer")
        return value
    if owners or global_owners:raise WrapperError("exact_one_live_NoCache_Yang_owner_or_verified_completion_required")
    return completed_yang(root)


def build_declaration(root, plan):
    return {"schema_version": DECLARATION_SCHEMA, "dataset": "Dreyer2023",
            "plan_sha256": DREYER_PLAN_SHA, "concurrent_yang_plan_sha256": YANG_PLAN_SHA,
            "root_authorization": AUTHORIZATION, "only_dreyer_entrypoint": ENTRY,
            "ordinary_dreyer_prepare_forbidden": True,
            "initial_yang_process": current_yang(root), "wrapper_implementation": implementation(root),
            "prior_budget_evidence": prior_budget_evidence(root, plan), "limits": dict(LIMITS),
            "qualification_evidence":qualification_evidence(root,plan),
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
        "qualification_evidence":qualification_evidence(root,plan),
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
    def __init__(self, root, expected_peer=None):
        self.root,self.core=root,kernel()
        self.path=self.core.local(root,AUDIT_ROOT+"/.source-extension-singleton.lock")
        self.handle=self.path.open("a+")
        self.owned,self.peer=False,expected_peer
        self.initial_peer,self.completion=expected_peer,None

    def check(self):
        if self.owned:return
        owner=self.core.local(self.root,YANG_OWNER_PATH)
        owners=lock_owners(owner) if owner.exists() else []
        global_owners=lock_owners(self.path)
        if owners:
            if len(owners)!=1:raise WrapperError("unknown_NoCache_Yang_persistent_owner_stop_new_records")
            live=yang_identity(self.root,process_fingerprint(owners[0]))
            if live is None or self.peer is not None and live!=self.peer:
                raise WrapperError("Yang_failed_replaced_or_unknown_owner_stop_new_records")
            if global_owners not in ([],[owners[0]]):
                raise WrapperError("unknown_global_owner_during_NoCache_Yang_stop_new_records")
            self.peer=live
            return  # Same persistent wrapper safely spans bounded global-lock gaps.
        if global_owners:raise WrapperError("global_owner_without_verified_NoCache_wrapper_stop_new_records")
        complete=completed_yang(self.root)
        if self.peer is not None and self.peer.get("mode")=="already_complete" and self.peer!=complete:
            raise WrapperError("frozen_Yang_completion_identity_changed")
        try:fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise WrapperError("global_lock_takeover_race_stop_new_records") from None
        self.owned,self.completion=True,complete

    def coordination_evidence(self):
        """Last verified observation retained for the receipt's first immutable seal."""
        return {"schema_version":"paper3-dreyer-yang-peer-coordination-v1",
            "initial_yang_process":self.initial_peer,
            "last_verified_live_peer":self.peer if self.peer and self.peer.get("mode")=="concurrent_no_cache_wrapper_v2" else None,
            "global_lock_takeover":self.owned,"completed_yang_proof":self.completion}

    def close(self):self.handle.close()


def verify_peer_coordination(root, declaration_descriptor, evidence):
    """Recheck completed takeover offline, preserving the original live observation."""
    core=kernel();declaration=read_json(core.checked(root,declaration_descriptor))
    initial=declaration.get("initial_yang_process")
    keys={"schema_version","initial_yang_process","last_verified_live_peer",
        "global_lock_takeover","completed_yang_proof"}
    if (not isinstance(evidence,dict) or set(evidence)!=keys
            or evidence.get("schema_version")!="paper3-dreyer-yang-peer-coordination-v1"
            or not isinstance(initial,dict) or evidence.get("initial_yang_process")!=initial
            or type(evidence.get("global_lock_takeover"))is not bool):
        raise WrapperError("frozen_Yang_peer_coordination_identity_changed")
    mode=initial.get("mode");takeover=evidence["global_lock_takeover"]
    if (mode not in ("concurrent_no_cache_wrapper_v2","already_complete")
            or evidence["last_verified_live_peer"]!=(initial if mode=="concurrent_no_cache_wrapper_v2" else None)
            or mode=="already_complete" and not takeover):
        raise WrapperError("frozen_Yang_live_peer_observation_changed")
    completion=evidence["completed_yang_proof"]
    if takeover:
        if (not isinstance(completion,dict) or completion!=completed_yang(root)
                or mode=="already_complete" and completion!=initial):
            raise WrapperError("frozen_Yang_takeover_completion_proof_changed")
    elif completion is not None:
        raise WrapperError("Yang_completion_claim_without_global_lock_takeover")
    return evidence


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


def provider_session():
    """Fresh provider transport only; original URL/integrity/budgets stay authoritative."""
    import requests
    from requests.adapters import HTTPAdapter
    session=requests.Session()
    try:
        session.headers.update(PROVIDER_HEADERS)
        session.mount("https://",HTTPAdapter(max_retries=0))
        session.mount("http://",HTTPAdapter(max_retries=0))
    except BaseException:
        session.close()
        raise
    return session


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


def protect_qualified_raw(root, record, qualification):
    """Qualified raw is readable, never implicitly owned/deleted by this worker."""
    core=kernel();audit=audit_path(root)/record["record_id"]
    owners={r["path"]:r for r in qualification["qualifier_owned_raw"]}
    state_path=audit/"state.json";state=read_json(state_path) if state_path.exists() else {}
    for raw in record["raw_files"]:
        if raw["path"] not in owners:continue
        marker=audit/(Path(raw["path"]).name+".ownership.json")
        if raw["path"] in state.get("owned_raw",[]) or marker.exists():
            raise WrapperError("qualifier_raw_ownership_cannot_be_adopted_by_V5")
        filename=core.local(root,raw["path"])
        if not filename.is_file():
            raise WrapperError("qualifier_owned_raw_missing_no_implicit_ownership_takeover")
        core.verify_raw_file(filename,raw)


def new_record_provenance(root, plan, record, state):
    """Independently bind shard references to actual fresh namespace artifacts."""
    rid=record["record_id"];core=kernel();shard=state.get("shard",{})
    adapter=plan["code"]["p3work/source_extensions/dreyer2023.py"]
    if (state.get("status")!="COMPLETE" or state.get("plan_sha256")!=DREYER_PLAN_SHA
            or state.get("record_id")!=rid or shard.get("record_id")!=rid
            or shard.get("adapter_code")!=adapter or adapter.get("sha256")!=NATIVE_ADAPTER_SHA
            or shard.get("raw_files")!=record["raw_files"]):
        raise WrapperError("new_V5_record_plan_adapter_and_raw_identity_required")
    expected_paths={k:f"cache/source_extensions_20261019/Dreyer2023/{DREYER_PLAN_SHA}/{rid}/{k}."+
        ("npy" if k in {"X","y"} else "json") for k in ("X","y","events","exclusions","metadata")}
    expected_paths.update({k:f"{AUDIT_ROOT}/Dreyer2023/{DREYER_PLAN_SHA}/{rid}/{k}.json"
        for k in ("native_metadata","unit_evidence")})
    artifacts=state.get("artifacts",[])
    if len(artifacts)!=7 or len({a.get("path")for a in artifacts})!=7:
        raise WrapperError("new_V5_exact_seven_record_artifacts_required")
    for key,path in expected_paths.items():
        item=shard.get(key)
        if not isinstance(item,dict) or item.get("path")!=path or item not in artifacts:
            raise WrapperError("new_V5_shard_must_bind_own_namespace_and_verified_artifacts")
        core.checked(root,item)


def numeric_regression(root, plan, record, state, qualification, *, write=False):
    """Compare genuinely new V5 cache to immutable old V4 cache; never fit."""
    core=kernel();rid=record["record_id"]
    reference=qualification["reference_states"].get(rid)
    if reference is None:return None
    new_record_provenance(root,plan,record,state)
    import hashlib
    import numpy as np
    parent=read_json(core.checked(root,qualification["parent_plan"]))
    old=read_json(core.checked(root,reference))
    if (state.get("plan_sha256")!=DREYER_PLAN_SHA or old.get("plan_sha256")!=V4_DREYER_SHA
            or not core._state_verified(root,old,parent,record)
            or not core._state_verified(root,state,plan,record)):
        raise WrapperError("real_new_V5_and_frozen_V4_complete_cache_proofs_required")
    oldshard,newshard=old["shard"],state["shard"]
    if newshard["adapter_code"].get("sha256")!=NATIVE_ADAPTER_SHA:
        raise WrapperError("numeric_regression_requires_actual_new_adapter_cache")
    arrays={}
    for name,dtype in (("X","float32"),("y","int64")):
        before=np.load(core.checked(root,oldshard[name]),allow_pickle=False)
        after=np.load(core.checked(root,newshard[name]),allow_pickle=False)
        left,right=before.tobytes(order="C"),after.tobytes(order="C")
        if (str(before.dtype)!=dtype or before.dtype!=after.dtype or before.shape!=after.shape
                or left!=right):
            raise WrapperError("old128_X_y_bitwise_regression_failed")
        arrays[name]={"shape":list(after.shape),"dtype":str(after.dtype),"bitwise_identical":True,
            "old_array_bytes_sha256":hashlib.sha256(left).hexdigest(),
            "new_array_bytes_sha256":hashlib.sha256(right).hexdigest(),
            "old_file":oldshard[name],"new_file":newshard[name]}
    # Only adapter identity and the plan-namespaced unit-evidence descriptor
    # may differ in per-trial rows. All scientific, clock, membership and QC
    # fields stay equal, including processed-signal hashes and row ordering.
    strip=lambda rows:[{k:v for k,v in row.items() if k not in {"adapter_sha256","unit_evidence"}}
        for row in rows]
    bindings={}
    for name in ("events","exclusions"):
        before=read_json(core.checked(root,oldshard[name]));after=read_json(core.checked(root,newshard[name]))
        if strip(before)!=strip(after):raise WrapperError("old128_event_UID_window_QC_regression_failed")
        bindings[name]={"old_file":oldshard[name],"new_file":newshard[name],"scientific_rows_identical":True,
            "ordered_trial_UIDs_sha256":core.digest([r["trial_uid"] for r in after]),"rows":len(after)}
    before=read_json(core.checked(root,oldshard["native_metadata"]))
    after=read_json(core.checked(root,newshard["native_metadata"]))
    if strip(before["trials"])!=strip(after["trials"]):
        raise WrapperError("old128_native_cue_end_UID_binding_regression_failed")
    oldmeta=read_json(core.checked(root,oldshard["metadata"]))
    newmeta=read_json(core.checked(root,newshard["metadata"]))
    scientific_meta=("original_trials","valid_trials","excluded_trials","original_class_counts",
        "valid_class_counts","native_sfreq_hz","native_n_times","native_record_count","native_signal_order",
        "native_signal_units","selected_channels","MNE_version","SI_conversion","signal_contract_sha256",
        "native_profile_sha256","preprocess_file_sha256","stage","normalization_applied")
    if any(oldmeta.get(k)!=newmeta.get(k) for k in scientific_meta):
        raise WrapperError("old128_scientific_metadata_regression_failed")
    result={"schema_version":"paper3-dreyer-V5-vs-V4-actual-numeric-regression-v1",
        "status":"ACTUAL_NEW_CACHE_X_Y_UID_WINDOW_QC_BITWISE_VERIFIED_NOT_FITTED",
        "record_id":rid,"new_plan_sha256":DREYER_PLAN_SHA,"old_reference_state":reference,
        "raw_files":record["raw_files"],"new_adapter_sha256":NATIVE_ADAPTER_SHA,
        "arrays":arrays,"event_bindings":bindings,
        "old_native_metadata":oldshard["native_metadata"],"new_native_metadata":newshard["native_metadata"],
        "native_trial_bindings_identical":True,"scientific_metadata_identical":True,
        "allowed_row_provenance_differences":["adapter_sha256","unit_evidence"],
        "old_cache_adopted_as_new_computation":False,"actual_fits":0,"GPU_actions":0}
    if write:core.atomic_json(audit_path(root)/rid/"v4_numeric_regression.json",result,immutable=True)
    return result


def all_numeric_regressions(root, plan, states, qualification):
    by_id={s["record_id"]:s for s in states};comparisons=[];core=kernel()
    for record in plan["records"]:
        if record["record_id"] not in qualification["reference_states"]:continue
        current=numeric_regression(root,plan,record,by_id[record["record_id"]],qualification)
        filename=audit_path(root)/record["record_id"]/"v4_numeric_regression.json"
        if not filename.is_file() or read_json(filename)!=current:
            raise WrapperError("all128_actual_numeric_regression_reports_required")
        comparisons.append(current)
    if len(comparisons)!=128:raise WrapperError("exact128_real_numerical_regressions_required")
    return {"reference_records":128,"verified_records":len(comparisons),
        "record_ids_sha256":core.digest([r["record_id"]for r in comparisons]),
        "comparisons":comparisons,"embedded_in_final_GET_backed_operational_evidence":True,
        "actual_fits":0,"GPU_actions":0}


def _finalize(root, plan, states, ledger, declaration_descriptor, peer_coordination):
    """Seal observations once; retries recheck and retain the same historical receipt."""
    core = kernel()
    audit, dataset = audit_path(root), "Dreyer2023"
    sealed_receipt=audit/"parallel_execution_receipt.json"
    # A later Yang completion cannot rewrite an already backed-up observation.
    # Runtime guards still verify every later live transition independently.
    observation=(read_json(sealed_receipt).get("yang_peer_coordination")
        if sealed_receipt.exists() else peer_coordination)
    coordination=verify_peer_coordination(root,declaration_descriptor,observation)
    qualification=qualification_evidence(root,plan)
    for record,state in zip(plan["records"],states):new_record_provenance(root,plan,record,state)
    regressions=all_numeric_regressions(root,plan,states,qualification)
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
        "qualification_evidence":qualification,"old128_numerical_regression":regressions,
        "yang_peer_coordination":coordination,
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
        new_record_provenance(root,plan,record,state)
    qualification=qualification_evidence(root,plan)
    regressions=all_numeric_regressions(root,plan,states,qualification)
    base = core.assemble_manifest(root, plan, states)
    if any(manifest.get(k) != v for k,v in base.items()):
        raise WrapperError("complete_parallel_manifest_not_actual_frozen_records")
    evidence_file = core.checked(root, manifest["operational_evidence"])
    evidence = read_json(evidence_file)
    coordination=verify_peer_coordination(root,declaration_descriptor,evidence.get("yang_peer_coordination"))
    required = {"schema_version":EVIDENCE_SCHEMA, "status":"RECORDS_VERIFIED_FINAL_BACKUP_PENDING",
        "dataset":"Dreyer2023", "plan_sha256":DREYER_PLAN_SHA,
        "concurrent_yang_plan_sha256":YANG_PLAN_SHA, "wrapper_implementation":implementation(root),
        "root_declaration":declaration_descriptor, "record_count":520,
        "record_ids_sha256":core.digest([r["record_id"] for r in plan["records"]]),
        "scientific_kernel_unchanged":True, "limits":LIMITS,
        "qualification_evidence":qualification,"old128_numerical_regression":regressions,
        "yang_peer_coordination":coordination,
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
    core = kernel()
    plan, _, profile = fixed_plans(root, plan_path)
    declaration, declaration_descriptor = verify_declaration(root, declaration_path, plan)
    qualification=qualification_evidence(root,plan)
    if max_new_records is not None and (type(max_new_records) is not int or max_new_records < 1):
        raise WrapperError("positive_record_limit_required")
    configure_cpu()
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
        peer = PeerGuard(root,expected_peer=declaration["initial_yang_process"])
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
            with provider_session() as session:
                states, new = [], 0
                for record in plan["records"]:
                    guard()
                    resource_guard(root, output, sum(f["bytes"] for f in record["raw_files"]))
                    old_path = audit/record["record_id"]/"state.json"
                    previous = read_json(old_path) if old_path.exists() else {}
                    complete = previous.get("status") == "COMPLETE"
                    if not complete and max_new_records is not None and new >= max_new_records:
                        break
                    protect_qualified_raw(root,record,qualification)
                    state = core.process_record(root, plan, record, native_profile=profile,
                        ledger=ledger, session=session, materialize=adapter.materialize,
                        audit_native=adapter.audit_native)
                    protect_qualified_raw(root,record,qualification)
                    new_record_provenance(root,plan,record,state)
                    numeric_regression(root,plan,record,state,qualification,write=True)
                    states.append(state); new += int(not complete)
                    guard()
                if len(states) == 520:
                    result = _finalize(root, plan, states, ledger, declaration_descriptor,peer.coordination_evidence())
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
            if any(v is None for v in (DREYER_PLAN_SHA,NATIVE_ADAPTER_SHA,QUALIFICATION_COMPLETION_SHA,ADMISSION_REVIEW_SHA,YANG_WRAPPER_SHA)):
                if args.write_declaration:
                    raise WrapperError("version5_plan_not_frozen_execution_and_declaration_blocked")
                print(json.dumps({"status":"PENDING_ALL520_REVIEW_AND_ROOT_FROZEN_V5_NO_NETWORK_R2_OR_FIT",
                    "GPU_actions":0,"actual_fits":0,"training_started":False}))
                return 0
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
