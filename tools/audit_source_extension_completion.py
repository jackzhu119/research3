#!/usr/bin/env python3
"""Read-only full native/cache/R2-evidence audit; never downloads, fits or opens queries.

Only a completed frozen dataset can produce an immutable PASS receipt. A running
or incomplete coordinator returns PENDING before its mutable ledger is read.
Counts are calculated from actual cache/native rows, not publication metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from p3work import source_extension_bridge as bridge
from p3work import source_extension_coordinator as coordinator
from p3work.contracts import digest

SPECS = {
    "Yang2025": {
        "plan": "audit/source_extensions_20261019/Yang2025/plan.json",
        "sha": "ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609",
        "records": 153, "role": "source_train", "arm": "BASE4_PLUS_YANG",
        "completion": "SOURCE_EXTENSION_CACHE_COMPLETE",
        "final_full_GET_objects": 5,
    },
    "Dreyer2023": {
        "plan": "audit/source_extensions_20261019/Dreyer2023/plan_after_all520_native_review_v5.json",
        "sha": "afc8dd0d2d873f455e901c6f702fc2d5649fde15f76873c40a24424c10d48e5a",
        "plan_file_sha256": "440c78ca7d1d2a3be685041924bdfe1fbcdd5ad9683592f97c39d50184ad2967",
        "records": 520, "role": "source_train", "arm": "BASE4_PLUS_DREYER",
        "completion": "SOURCE_EXTENSION_CACHE_COMPLETE",
        "native_trials": 20792, "final_full_GET_objects": 7,
    },
    "Ma2022": {
        "plan": "catalog/source_extensions_20261019/Ma2022/development_plan_native_v2.json",
        "sha": "46e6abb9458925d7827bafa5d9feaf12cb075ecb9c1f3b980d804f9daa4b6f8b",
        "records": 125, "role": "independent_development_unassigned",
        "completion": "MA2022_DEVELOPMENT_CACHE_COMPLETE",
    },
}


class CompletionAuditError(ValueError):
    """Non-sensitive evidence error; no remote action is attempted."""


def read(path):
    try:
        result = json.loads(Path(path).read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise CompletionAuditError("invalid_audit_JSON") from exc
    return result


def checked(root, item, closure):
    return bridge.artifact(root, item, closure)


def tracked(root, path, closure):
    item = bridge.descriptor(root, path)
    checked(root, item, closure)
    return item


def numeric_qc(x):
    """Float32 precision-aware CAR and temporal-mean check, per actual trial."""
    value = np.asarray(x, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 128 or value.shape[0] not in (3, 11):
        raise CompletionAuditError("trial_shape_invalid")
    scale = float(np.max(np.abs(value)))
    if not np.isfinite(value).all() or not math.isfinite(scale) or scale <= 0:
        raise CompletionAuditError("finite_nonzero_signal_required")
    if (np.max(np.abs(value.mean(axis=0))) > scale*2e-6
            or np.max(np.abs(value.mean(axis=1))) > scale*2e-6):
        raise CompletionAuditError("actual_CAR_and_temporal_mean_QC_failed")
    return float(np.sqrt(np.mean(value**2)))


def closed_ledger(root, path, plan, required_get, closure):
    """Independent pure reader; never constructs a mutable coordinator Ledger."""
    tracked(root, path, closure)
    ledger = read(path)
    charged = ledger.get("charged")
    history = ledger.get("history")
    wall = ledger.get("elapsed_seconds")
    budgets = plan["budgets"]
    kinds = {"provider_body_bytes", "r2_get_body_bytes"}
    if (ledger.get("schema_version") != "paper3-source-extension-budget-ledger-v1"
            or ledger.get("plan_sha256") != plan["plan_sha256"]
            or "wall_intent" in ledger or "active_transfer" in ledger
            or not isinstance(charged, dict) or set(charged) != kinds
            or not isinstance(history, list) or type(wall) not in (int, float)
            or not math.isfinite(wall) or not 0 <= wall <= budgets["wall_seconds"]
            or any(type(v) is not int or not 0 <= v <= budgets[k] for k,v in charged.items())):
        raise CompletionAuditError("actual_closed_in_budget_ledger_required")
    counted = dict.fromkeys(kinds, 0)
    for entry in history:
        if (not isinstance(entry, dict) or entry.get("kind") not in kinds
                or type(entry.get("reserved_bytes")) is not int or entry["reserved_bytes"] < 0):
            raise CompletionAuditError("invalid_transfer_history")
        amount = entry["reserved_bytes"]
        if entry.get("status") == "SUCCESS_FULL_BODY_VERIFIED":
            amount = entry.get("verified_bytes")
            if type(amount) is not int or not 0 <= amount <= entry["reserved_bytes"]:
                raise CompletionAuditError("invalid_verified_transfer_bytes")
        elif entry.get("status") not in {
            "FAILURE_FULL_RESERVATION_CONSUMED", "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}:
            raise CompletionAuditError("transfer_history_not_closed")
        counted[entry["kind"]] += amount
    if counted != charged or charged["r2_get_body_bytes"] < required_get:
        raise CompletionAuditError("ledger_does_not_cover_exact_history_and_required_full_GET")
    return {"charged": charged, "elapsed_seconds": wall,
        "required_full_GET_bytes_lower_bound": required_get,
        "history_entries": len(history), "ledger": bridge.descriptor(root, path)}


def add_unique_uids(seen, uids):
    for uid in uids:
        if not isinstance(uid, str) or not uid or uid in seen:
            raise CompletionAuditError("missing_or_cross_record_duplicate_native_UID")
        seen.add(uid)


def exact_roster(plan, shards, count):
    records = plan.get("records")
    if (not isinstance(records, list) or len(records) != count or plan.get("record_count") != count
            or len({r["record_id"] for r in records}) != count
            or not isinstance(shards, list) or len(shards) != count
            or {r.get("record_id") for r in shards} != {r["record_id"] for r in records}):
        raise CompletionAuditError("complete_exact_actual_record_roster_required")
    return {r["record_id"]: r for r in records}


def source_summary(root, dataset, spec, plan, manifest, manifest_path, audit, status, closure):
    records = exact_roster(plan, manifest.get("shards"), spec["records"])
    cache = bridge.load_extension_cache(root, arm=spec["arm"],
        manifest_path=manifest_path.relative_to(root).as_posix(),
        manifest_sha256=bridge.descriptor(root, manifest_path)["sha256"],
        output=f"audit/completion_readonly_capability/{dataset}")
    closure.update(cache.artifact_sha256)
    retained = [r for r in cache.rows if r["dataset"] == dataset]
    retained_by_uid = {}
    for row in retained:
        uid = row.get("event_uid")
        if not isinstance(uid, str) or not uid or uid in retained_by_uid:
            raise CompletionAuditError("retained_native_UID_must_be_present_and_unique")
        retained_by_uid[uid] = row
    native_uids, raw_files, subjects, sessions, labels = set(), {}, set(), set(), Counter()
    native_labels = Counter()
    native_non_binary = 0
    excluded = 0
    cache_bytes = 0
    for shard in manifest["shards"]:
        record = records[shard["record_id"]]
        state_path = audit/shard["record_id"]/"state.json"
        tracked(root, state_path, closure)
        state = read(state_path)
        if not coordinator._state_verified(root, state, plan, record) or state.get("shard") != shard:
            raise CompletionAuditError("actual_complete_state_must_match_global_shard")
        for item in state["artifacts"]:
            checked(root, item, closure)
        cache_bytes += sum(item["bytes"] for item in state["artifacts"])
        native = bridge._native_rows(root, shard, dataset, closure)
        add_unique_uids(native_uids, native)
        # Native metadata contains genuine original events before waveform QC.
        # Dreyer V5's terminal no-cue prefix is control metadata, outside trials;
        # it receives no UID, label, cache row or waveform exclusion.
        for original in native.values():
            label = original.get("label")
            if type(label) is int and label in (0, 1):
                native_labels[str(label)] += 1
            elif dataset == "Dreyer2023":
                raise CompletionAuditError("Dreyer_native_MI_binary_labels_required")
            else:
                native_non_binary += 1
        exclusions = read(checked(root, shard["exclusions"], closure))
        if not isinstance(exclusions, list):
            raise CompletionAuditError("explicit_native_exclusion_inventory_required")
        excluded_uids = set()
        for row in exclusions:
            uid = row.get("event_uid") if isinstance(row, dict) else None
            if (uid not in native or uid in excluded_uids or uid in retained_by_uid
                    or not row.get("reason", row.get("exclusion_reason"))):
                raise CompletionAuditError("native_retained_and_excluded_UIDs_must_partition_each_record")
            excluded_uids.add(uid)
        if set(native) != (set(native) & retained_by_uid.keys()) | excluded_uids:
            raise CompletionAuditError("native_retained_and_excluded_UIDs_must_partition_each_record")
        excluded += len(excluded_uids)
        subjects.add(shard["subject_id"])
        sessions.add((shard["subject_id"], shard["session_id"]))
        for item in shard["raw_files"]:
            key = item["path"]
            identity = (item["sha256"], item["bytes"])
            if key in raw_files:
                raise CompletionAuditError("raw_object_reused_across_provider_records")
            raw_files[key] = identity
    for row in retained:
        if row["event_uid"] not in native_uids:
            raise CompletionAuditError("retained_trial_missing_from_actual_native_inventory")
        labels[str(row["label"])] += 1
    if (len(retained) != manifest.get("valid_trials")
            or len(native_uids) != len(retained)+excluded
            or subjects != {r["subject_id"] for r in plan["records"]}):
        raise CompletionAuditError("native_retained_excluded_denominators_disagree")
    if "native_trials" in spec and len(native_uids) != spec["native_trials"]:
        raise CompletionAuditError("exact_current_native_pre_QC_trial_total_required")
    final_rows = bridge._receipt_entries(root, status["backup_receipt"], closure)
    if len(final_rows) != spec["final_full_GET_objects"]:
        raise CompletionAuditError("exact_current_final_full_GET_object_count_required")
    required = sum(n for _,n in raw_files.values()) + cache_bytes + sum(r["bytes"] for r in final_rows)
    ledger = closed_ledger(root, audit/"budget_ledger.json", plan, required, closure)
    return {"provider_participant_ids":len(subjects), "provider_subject_ids_sha256":digest(sorted(subjects)),
        "provider_sessions":len(sessions), "records":len(records),
        "native_trials":len(native_uids), "retained_trials":len(retained), "excluded_trials":excluded,
        "native_binary_labels":dict(sorted(native_labels.items())),
        "native_non_binary_events":native_non_binary,
        "unique_native_trial_UIDs":len(native_uids), "native_UIDs_sorted_sha256":digest(sorted(native_uids)),
        "retained_labels":dict(sorted(labels.items())), "raw_objects":len(raw_files),
        "raw_bytes":sum(n for _,n in raw_files.values()), "cache_artifact_bytes":cache_bytes,
        "final_full_GET_objects":len(final_rows), "ledger":ledger,
        "unit_policy":("unknown_SI_verified_common_positive_gain_no_absolute_amplitude_QC"
            if dataset == "Yang2025" else "native_uV_to_V_exact_once"),
        "physical_MI_onset_and_end_verified":dataset == "Dreyer2023"}


def ma_units_and_bounds(native, unit, events, x):
    header = native["header"]
    calibration = header.get("selected_calibration")
    if (native.get("dataset") != "Ma2022" or native.get("provider_version") != 1
            or native.get("event_clock") != "observed_1_based_native_samples_not_generic_sidecar_milliseconds"
            or native.get("reference_evidence") != "official_v1_sidecar_unipolar_M1"
            or header.get("sfreq_hz") != 250 or header.get("stored_trial_samples") != 1000
            or header.get("n_samples") != len(events)*1000
            or not isinstance(calibration, list) or len(calibration) != 3
            or [r.get("name") for r in calibration] != ["C3", "Cz", "C4"]
            or any(r.get("unit") not in {"uV", "µV"}
                or not np.isfinite([r.get("physical_min"),r.get("physical_max")]).all()
                or r["physical_max"] <= r["physical_min"]
                or type(r.get("digital_min")) is not int or type(r.get("digital_max")) is not int
                or not -32768 <= r["digital_min"] < r["digital_max"] <= 32767 for r in calibration)
            or unit.get("dataset") != "Ma2022" or unit.get("role") != SPECS["Ma2022"]["role"]
            or unit.get("native_unit") != "uV" or unit.get("stored_unit") != "V"
            or unit.get("conversion_to_V") != 1e-6 or unit.get("selected_calibration") != calibration
            or unit.get("reference_evidence") != native["reference_evidence"]
            or unit.get("original_B20_prefix_recoverable") is not False):
        raise CompletionAuditError("actual_native_unit_calibration_clock_evidence_required")
    for i,event in enumerate(events):
        start = i*1000
        if (event.get("retained_trial_index") != i or event.get("analysis_anchor_sample") != start
                or event.get("task_end_sample") != start+1000
                or event.get("original_context_samples") != [start,start+375]
                or event.get("native_sfreq_hz") != 250 or start+375 > event["task_end_sample"]
                or event.get("original_acquisition_ordinal") is not None
                or event.get("physical_run_id") is not None
                or event.get("original_paid_B20_prefix_recoverable") is not False
                or event.get("data_after_context_accessed") is not False
                or event.get("reference") != "CAR_exact_C3_Cz_C4"
                or event.get("normalization_applied") is not False
                or event.get("native_unit") != "uV" or event.get("stored_unit") != "V"):
            raise CompletionAuditError("actual_native_retained_window_bound_or_B20_policy_changed")
        rms = numeric_qc(x[i])
        recorded = event.get("pre_RMS_joint_rms_volts")
        if type(recorded) not in (int,float) or not math.isfinite(recorded) or recorded <= 0 or not math.isclose(rms,recorded,rel_tol=2e-6):
            raise CompletionAuditError("native_cache_joint_energy_evidence_changed")


def ma_summary(root, spec, plan, manifest, manifest_path, audit, status, closure):
    ma = importlib.import_module("tools.prepare_ma2022_development")
    ma.validate_plan(root, plan)
    for item in [plan["catalog"], *plan["code"].values()]:
        checked(root,item,closure)
    records = exact_roster(plan, manifest.get("shards"), spec["records"])
    expected_status = {"dataset":"Ma2022","role":spec["role"],"plan_sha256":plan["plan_sha256"],
        "verified_records":spec["records"],"training_ready":False,"original_B20_prefix_recoverable":False,
        "actual_fits":0,"GPU_actions":0,"external_evaluation":False}
    if any(status.get(k) != v for k,v in expected_status.items()):
        raise CompletionAuditError("Ma_actual_completion_status_and_development_gates_required")
    expected = {"schema_version":"paper3-Ma2022-development-cache-v1",
        "status":"ALL125_DEVELOPMENT_PAIRS_VERIFIED", "dataset":"Ma2022", "role":spec["role"],
        "training_ready":False,"source_training_allowed":False,"source3_training_pool_member":False,
        "primary11_compatible":False,"original_B20_prefix_recoverable":False,
        "scientific_pool_assignment_frozen":False,"actual_fits":0,"GPU_actions":0,
        "normalization_applied":False,"biological_independence_verified":False,
        "stage":ma.ma2022.STAGE,"channels":["C3","Cz","C4"],"plan_sha256":plan["plan_sha256"],
        "catalog":plan["catalog"],"document_sha256":plan["document_sha256"]}
    if any(manifest.get(k) != v for k,v in expected.items()):
        raise CompletionAuditError("Ma_complete_independent_unassigned_development_policy_required")
    inventory = read(checked(root, manifest["native_inventory"], closure))
    actual_inventory, native_uids, raw_paths, subjects, sessions = [], set(), set(), set(), set()
    labels = Counter(); raw_bytes = cache_bytes = headerless = 0
    for shard in manifest["shards"]:
        record = records[shard["record_id"]]
        state_path = audit/shard["record_id"]/"state.json"
        tracked(root, state_path, closure); state = read(state_path)
        if not ma._complete(root,state,plan,record):
            raise CompletionAuditError("all125_actual_complete_states_required")
        expected_shard = {"record_id":state["record_id"], "retained_trials":state["retained_trials"],
            "artifacts":state["artifacts"], "raw_backup_receipt":state["raw_backup_receipt"],
            "cache_backup_receipt":state["cache_backup_receipt"]}
        if shard != expected_shard:
            raise CompletionAuditError("Ma_global_shard_does_not_match_actual_complete_state")
        for item in [*state["artifacts"], state["raw_backup_receipt"],state["cache_backup_receipt"]]:
            checked(root,item,closure)
        native = read(checked(root,state["native_metadata"],closure))
        unit = read(checked(root,state["unit_evidence"],closure))
        if unit.get("native_metadata") != state["native_metadata"]:
            raise CompletionAuditError("Ma_unit_evidence_must_bind_actual_native_metadata")
        output = bridge.local(root,record["output"]+"/events.json")
        events = read(output)
        x = np.load(output.with_name("X.npy"),mmap_mode="r",allow_pickle=False)
        ma_units_and_bounds(native,unit,events,x)
        add_unique_uids(native_uids,[r["event_uid"] for r in events])
        for event in events: labels[str(event["label"])] += 1
        for raw in state["raw_identity"]:
            if raw["path"] in raw_paths or raw.get("provider_MD5_and_length_verified") is not True:
                raise CompletionAuditError("unique_provider_MD5_verified_native_objects_required")
            raw_paths.add(raw["path"]); raw_bytes += raw["bytes"]
        cache_bytes += sum(i["bytes"] for i in state["artifacts"])
        subjects.add(record["subject_id"]);sessions.add((record["subject_id"],record["session_id"]))
        headerless += native.get("native_event_table_layout",{}).get("header_present") is False
        actual_inventory.append({"record_id":state["record_id"],"raw_identity":state["raw_identity"],
            "native_metadata":state["native_metadata"],"unit_evidence":state["unit_evidence"],
            "raw_backup_receipt":state["raw_backup_receipt"],"cache_backup_receipt":state["cache_backup_receipt"]})
    if inventory != {"dataset":"Ma2022","records":actual_inventory}:
        raise CompletionAuditError("Ma_complete_native_inventory_must_equal_actual_states")
    actual_counts = {"provider_participant_ids":len(subjects),"sessions":len(sessions),
        "original_raw_objects":len(raw_paths),"original_raw_bytes":raw_bytes,"retained_trials":len(native_uids)}
    if any(manifest.get(k) != v for k,v in actual_counts.items()) or status.get("retained_trials") != len(native_uids):
        raise CompletionAuditError("Ma_reported_denominators_must_equal_actual_validated_cache")
    files = [(manifest_path,"result"),(checked(root,manifest["native_inventory"],closure),"audit"),(audit/"frozen_plan.json","audit")]
    final = ma._backup_set(root,status["backup_receipt"],files)
    checked(root,status["backup_receipt"],closure)
    if len(final) != 3:
        raise CompletionAuditError("Ma_exact_three_final_full_GET_objects_required")
    required = raw_bytes+cache_bytes+sum(r["bytes"] for r in final)
    ledger = closed_ledger(root,audit/"budget_ledger.json",plan,required,closure)
    return {"provider_participant_ids":len(subjects),"provider_subject_ids_sha256":digest(sorted(subjects)),
        "provider_sessions":len(sessions),"records":len(records),"native_trials":len(native_uids),
        "retained_trials":len(native_uids),"excluded_trials":0,"unique_native_trial_UIDs":len(native_uids),
        "native_UIDs_sorted_sha256":digest(sorted(native_uids)),"retained_labels":dict(sorted(labels.items())),
        "raw_objects":len(raw_paths),"raw_bytes":raw_bytes,"cache_artifact_bytes":cache_bytes,
        "official_headerless_sessions":headerless,"final_full_GET_objects":len(final),"ledger":ledger,
        "unit_policy":"native_uV_to_V_exact_once","physical_MI_onset_and_end_verified":False,
        "original_paid_B20_prefix_recoverable":False,"scientific_pool_assignment_frozen":False}


def audit_completion(root, dataset):
    root = Path(root).resolve(strict=True)
    if dataset not in SPECS:
        raise CompletionAuditError("registered_permitted_dataset_required")
    spec = SPECS[dataset]
    audit = root/f"audit/source_extensions_20261019/{dataset}/{spec['sha']}"
    # Do not inspect a live ledger or partial cache while a CPU worker owns it.
    status_path = bridge.local(root,(audit/"coordinator_status.json").relative_to(root).as_posix(),exists=False)
    if not status_path.is_file():
        return {"status":"PENDING_COMPLETE_DATA_GATE","dataset":dataset,"completion_receipt_written":False}
    status = read(status_path)
    if status.get("status") != spec["completion"]:
        return {"status":"PENDING_COMPLETE_DATA_GATE","dataset":dataset,"completion_receipt_written":False,
            "coordinator_status":status.get("status")}
    plan_path = bridge.local(root,spec["plan"])
    plan = read(plan_path)
    if (plan.get("plan_sha256") != spec["sha"] or digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != spec["sha"]
            or plan.get("dataset") != dataset or plan.get("role") != spec["role"]
            or ("plan_file_sha256" in spec
                and bridge.descriptor(root,plan_path)["sha256"] != spec["plan_file_sha256"])):
        raise CompletionAuditError("exact_current_frozen_dataset_plan_required")
    frozen = read(bridge.local(root,(audit/"frozen_plan.json").relative_to(root).as_posix()))
    if frozen != plan:
        raise CompletionAuditError("coordinator_frozen_plan_must_equal_registered_plan")
    closure = {}
    for path in [plan_path,audit/"frozen_plan.json",status_path]:tracked(root,path,closure)
    manifest_path = checked(root,status["manifest"],closure)
    canonical = (f"cache/source_extensions_20261019/{dataset}/{spec['sha']}/source_cache.json"
        if dataset != "Ma2022" else "cache/source_extensions_20261019/Ma2022/development_cache_native_v2.json")
    if manifest_path != root/canonical:
        raise CompletionAuditError("canonical_global_manifest_required")
    manifest = read(manifest_path)
    if dataset == "Ma2022":
        summary = ma_summary(root,spec,plan,manifest,manifest_path,audit,status,closure)
    else:
        summary = source_summary(root,dataset,spec,plan,manifest,manifest_path,audit,status,closure)
    for relative,sha in closure.items():checked(root,{"path":relative,"sha256":sha},{})
    # Compare the status and ledger snapshots after all expensive hash/numeric checks.
    # A changed completion file/ledger cannot be used to publish a mixed snapshot.
    for path in [status_path,audit/"budget_ledger.json"]:
        if bridge.descriptor(root,path)["sha256"] != closure[path.relative_to(root).as_posix()]:
            raise CompletionAuditError("completion_evidence_changed_during_audit")
    return {"schema_version":"paper3-independent-source-extension-completion-audit-v1",
        "status":"PASS_COMPLETE_NATIVE_CACHE_AND_FULL_GET_EVIDENCE","dataset":dataset,"role":spec["role"],
        "plan_sha256":spec["sha"],"manifest":bridge.descriptor(root,manifest_path),"actual_summary":summary,
        "artifact_closure":{k:closure[k] for k in sorted(closure)},
        "artifact_closure_sha256":digest(closure),"auditor":bridge.descriptor(root,Path(__file__).resolve()),
        "biological_independence_verified":False,"scientific_training_ready":False,
        "scientific_pool_assignment_or_training_gate_granted":False,"raw_payloads_restored":0,
        "actual_scientific_fits":0,"GPU_actions":0,"network_actions":0,"R2_actions":0,
        "external_payload_accessed":False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--dataset",choices=sorted(SPECS),required=True)
    parser.add_argument("--report",help="Write a NEW immutable receipt only after a complete PASS.")
    args = parser.parse_args(argv)
    try:
        result = audit_completion(args.root,args.dataset)
        if result["status"] == "PENDING_COMPLETE_DATA_GATE":
            print(json.dumps(result,ensure_ascii=False))
            return 3
        if args.report:
            root = args.root.resolve(strict=True)
            path = bridge.local(root,args.report,exists=False)
            if not args.report.startswith("audit/") or path.suffix != ".json" or path.exists():
                raise CompletionAuditError("new_audit_JSON_receipt_path_required")
            coordinator.atomic_json(path,result,immutable=True)
        print(json.dumps({k:v for k,v in result.items() if k != "artifact_closure"},ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"FAILED_CLOSED","error_type":type(exc).__name__,
            "error_code":str(exc) if isinstance(exc,(CompletionAuditError,bridge.ExtensionGateError,coordinator.PreparationError))
                else "audit_failed_details_suppressed","completion_receipt_written":False,
            "actual_scientific_fits":0,"GPU_actions":0,"network_actions":0,"R2_actions":0,
            "external_payload_accessed":False},ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
