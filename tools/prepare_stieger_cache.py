#!/usr/bin/env python3
"""W1-gated serial W3 CPU cache preparation; dry run unless --execute.

All 598 official sessions retain their frozen person partition. Cache creation
does not fit models, choose a window, evaluate internal holdout, access external
cohorts, or contact RunPod. Per-session outputs and original/raw backup proofs
must pass complete R2 GET/SHA256 checks before owned raw removal. Existing
originals and unmarked files are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import re
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.adapters.stieger_native import CLINIC11, load_session
from p3work.contracts import file_digest, validate_signal_contract
from p3work.preprocess import prepare_trial
from p3work.splits import _hash
from tools import r2_backup, r2_restore, stieger_metadata_batch as base

SCHEMA = "paper3-stieger-cache-v2"
STAGE = "pre_rms_car11_temporal_demeaned"
PARTITIONS = ("meta_train", "meta_validation", "internal_holdout")
ADAPTER = Path(sys.modules[load_session.__module__].__file__).resolve()
PROCESSOR = Path(sys.modules[prepare_trial.__module__].__file__).resolve()
QUALIFICATION_SHA = "c825be2636a5f423838816f76ededbae8cb08e92a3ddbc57032fb911db8763fa"


class CacheError(Exception):
    """Only fixed, non-sensitive error codes belong here."""


def local(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or "\\" in relative or any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise CacheError("unsafe_paper3_relative_path")
    path = root / relative
    r2_restore.no_symlink_components(path)
    if not path.resolve().is_relative_to(root):
        raise CacheError("path_outside_paper3_root")
    return path


def descriptor(root: Path, path: Path) -> dict:
    path = local(root, path.relative_to(root).as_posix())
    return {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}


def checked(root: Path, item: dict) -> Path:
    path = local(root, item["path"])
    if not path.is_file() or file_digest(path) != item["sha256"]:
        raise CacheError("artifact_sha256_mismatch")
    return path


def write_once_json(path: Path, value) -> None:
    r2_restore.no_symlink_components(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise CacheError("existing_immutable_artifact_differs")
        return
    base.atomic_json(path, value)


def write_array(path: Path, x: np.ndarray) -> None:
    r2_restore.no_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".stieger-array-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, x, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            if file_digest(path) != file_digest(temporary):
                raise CacheError("existing_numerical_cache_differs")
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def clock_snapshot() -> dict:
    boot_path=Path("/proc/sys/kernel/random/boot_id")
    boot=boot_path.read_text().strip() if boot_path.is_file() else None
    if boot is None or re.fullmatch(r"[0-9a-f-]{36}",boot) is None:
        raise CacheError("clock_boot_identity_unavailable")
    return {"utc":datetime.now(timezone.utc).isoformat(),"monotonic_ns":time.monotonic_ns(),"boot_id":boot}


def elapsed_guard(start: dict, now: dict | None = None) -> float:
    """Conservative UTC/monotonic interval, including hard-kill downtime.

    A different boot or >5 s disagreement is not silently treated as zero.
    Such a clock change requires an explicit new runtime/budget reconciliation.
    """
    now=now or clock_snapshot()
    try:
        a,b=datetime.fromisoformat(start["utc"]),datetime.fromisoformat(now["utc"])
        if a.tzinfo is None or b.tzinfo is None or start["boot_id"]!=now["boot_id"]:
            raise ValueError
        utc=(b-a).total_seconds();mono=(now["monotonic_ns"]-start["monotonic_ns"])/1e9
        if not all(math.isfinite(x) and x>=0 for x in (utc,mono)) or abs(utc-mono)>5.:
            raise ValueError
    except (KeyError,TypeError,ValueError,OverflowError):
        raise CacheError("elapsed_time_unknown_or_clock_changed_budget_closed") from None
    return max(utc,mono)


def wall_checkpoint(path: Path, progress: dict, *, reopen: bool) -> None:
    now=clock_snapshot()
    if progress.get("wall_intent"):
        try:
            progress["active_wall_elapsed_s"]=progress.get("active_wall_elapsed_s",0.)+elapsed_guard(progress["wall_intent"],now)
        except CacheError:
            progress["status"]="UNKNOWN_ELAPSED_TIME_BUDGET_CLOSED"
            base.atomic_json(path,progress)
            raise
        progress.pop("wall_intent",None)
    if reopen:progress["wall_intent"]=now
    base.atomic_json(path,progress)


def split_path_default(root: Path) -> Path:
    tasks = json.loads((root / "status/tasks.json").read_text())
    candidates = [p for p in tasks.get("W1", {}).get("evidence", []) if p.endswith("/participant_split.json")]
    if len(candidates) > 1:
        raise CacheError("ambiguous_w1_split_evidence")
    return local(root, candidates[0]) if candidates else root / "audit/stieger_w1/batches" / QUALIFICATION_SHA / "participant_split.json"


def read_frozen_inputs(root: Path, contract_path: Path, split_path: Path, *, execute: bool) -> tuple[dict, dict, str]:
    tasks = json.loads((root / "status/tasks.json").read_text())
    if execute and tasks.get("W1", {}).get("status") != "COMPLETE":
        raise CacheError("w1_complete_required_before_w3_execution")
    contract = json.loads(contract_path.read_text())
    try:
        contract_sha = validate_signal_contract(contract)
    except ValueError:
        raise CacheError("measured_frozen_window_required_before_w3") from None
    split = json.loads(split_path.read_text())
    copy = dict(split)
    split_sha = copy.pop("split_sha256", None)
    if split.get("status") != "FROZEN_METADATA_ONLY" or _hash(copy) != split_sha:
        raise CacheError("frozen_split_integrity_failed")
    if set(split.get("partitions", {})) != set(PARTITIONS):
        raise CacheError("exact_three_frozen_partitions_required")
    people = sum(split["partitions"].values(), [])
    if len(people) != len(set(people)) or set(people) != {f"S{i}" for i in range(1, 63)}:
        raise CacheError("all62_fixed_person_partition_required")
    if contract["window_audit_receipt"]["split_sha256"] != split_sha:
        raise CacheError("window_and_participant_split_disagree")
    if execute and tasks["W1"].get("input_contract_sha256") != contract_sha:
        raise CacheError("w1_receipt_input_contract_mismatch")
    return contract, split, contract_sha


def build_plan(root: Path, *, inventory_path: Path, contract_path: Path, split_path: Path,
               output="cache/stieger_v2", max_wall_seconds=86400, max_cache_bytes=4*1024**3) -> dict:
    root = root.resolve(strict=True)
    inv = base.read_inventory(inventory_path)
    contract, split, contract_sha = read_frozen_inputs(root, contract_path, split_path, execute=False)
    if type(max_wall_seconds) is not int or max_wall_seconds < 1 or type(max_cache_bytes) is not int or max_cache_bytes < 1:
        raise CacheError("positive_explicit_resource_budgets_required")
    local(root, output)
    role = {p: name for name, people in split["partitions"].items() for p in people}
    fs = [dict(f, partition=role[f["provider_subject"]]) for f in inv["files"] if f["name"].endswith(".mat")]
    fs.sort(key=lambda f: (PARTITIONS.index(f["partition"]), int(f["provider_subject"][1:]), f["session_ordinal"]))
    plan = {
        "schema_version": "paper3-stieger-cache-plan-v1", "dataset": "Stieger2021", "source_version": 1,
        "official_inventory": descriptor(root, inventory_path), "input_contract": descriptor(root, contract_path),
        "input_contract_sha256": contract_sha, "participant_split": descriptor(root, split_path),
        "split_sha256": split["split_sha256"], "adapter_code": descriptor(root, ADAPTER),
        "processing_code": descriptor(root, PROCESSOR), "cache_builder_code": descriptor(root, Path(__file__).resolve()),
        "files": fs, "file_count": len(fs), "participant_count": 62, "output": output,
        "partition_scope": list(PARTITIONS), "inflight_downloads": 1, "parallel_mat_parsers": 1,
        "max_jobs": len(fs), "max_download_bytes": sum(f["size"] for f in fs),
        "max_r2_restore_bytes": sum(f["size"] for f in fs), "max_cache_bytes": max_cache_bytes,
        "max_single_raw_bytes": max(f["size"] for f in fs), "max_wall_seconds": max_wall_seconds,
        "max_gpu_hours": 0, "download_budget_scope": "successful actual bodies; failed/unclosed transfers conservatively consume their entire durable reservation; official/R2 restoration separate",
        "stage": STAGE, "normalization_applied": False, "synthetic": False, "old_npz_reused": False,
        "training_started": False, "internal_holdout_evaluated": False,
        "internal_role_policy": "CPU cache only; no episode learner/evaluator may load internal role before its later frozen test gate",
    }
    plan["plan_sha256"] = _hash(plan)
    return plan


def validate_plan(root: Path, plan: dict, *, execute: bool) -> tuple[dict, dict]:
    inv = checked(root, plan["official_inventory"])
    cp, sp = checked(root, plan["input_contract"]), checked(root, plan["participant_split"])
    contract, split, _ = read_frozen_inputs(root, cp, sp, execute=execute)
    expected = build_plan(root, inventory_path=inv, contract_path=cp, split_path=sp, output=plan["output"],
                          max_wall_seconds=plan["max_wall_seconds"], max_cache_bytes=plan["max_cache_bytes"])
    if plan != expected or len(plan["files"]) != 598:
        raise CacheError("plan_not_exact_frozen_all598_official_selection_or_code_changed")
    return contract, split


def raw_receipt_catalog(root: Path, files: list[dict]) -> dict:
    """Only complete W1 object states with strict original R2 receipt evidence."""
    by_id = {f["id"]: f for f in files}
    catalog = {}
    for path in sorted((root / "audit/stieger_w1/batches").glob("*/objects/*/state.json")):
        state = json.loads(path.read_text())
        entry = by_id.get(state.get("file_id"))
        if not entry or state.get("status") != "COMPLETE":
            continue
        if not base.verify_completed(state, root):
            raise CacheError("existing_w1_raw_state_unverified")
        md5, _ = base.official_md5(entry)
        if (state.get("name") != entry["name"] or state["raw_integrity"]["bytes"] != entry["size"]
                or state["raw_integrity"]["md5"] != md5):
            raise CacheError("w1_raw_object_not_exact_official_identity")
        receipt = local(root, state["backup_receipt"])
        rows, _ = r2_restore.load_verified_receipt(receipt)
        raw = [r for r in rows if r["role"] == "source_development_raw" and r["dataset"] == "Stieger2021"]
        if (len(raw) != 1 or Path(raw[0]["path"]).name != entry["name"]
                or raw[0]["sha256"] != state["raw_integrity"]["sha256"] or raw[0]["bytes"] != entry["size"]):
            raise CacheError("w1_raw_receipt_sha_or_size_mismatch")
        value = {"entry": raw[0], "backup_receipt": descriptor(root, receipt), "official_md5": md5}
        if entry["id"] in catalog and catalog[entry["id"]]["entry"]["sha256"] != raw[0]["sha256"]:
            raise CacheError("conflicting_original_raw_sha256")
        catalog[entry["id"]] = value
    return catalog


def verified_backup(root: Path, manifest_path: Path, receipt_path: Path, *, prefix: str, files: list[tuple[Path, str]], backup_fn) -> dict:
    manifest = {"schema_version": 1, "files": []}
    for path, role in files:
        sha, size = r2_backup.sha256_file(path)
        manifest["files"].append({"path": path.relative_to(root).as_posix(), "sha256": sha, "bytes": size,
                                  "role": role, "dataset": "Stieger2021"})
    write_once_json(manifest_path, manifest)
    result = backup_fn(manifest_path=manifest_path, root=root, prefix=prefix, receipt_path=receipt_path, execute=True)
    if result.get("status") != "verified" or not result.get("execute"):
        raise CacheError("r2_backup_full_get_failed_original_preserved")
    rows, _ = r2_restore.load_verified_receipt(receipt_path)
    keys = ("path", "sha256", "bytes", "role", "dataset")
    if {tuple(row[k] for k in keys) for row in rows} != {tuple(row[k] for k in keys) for row in manifest["files"]}:
        raise CacheError("r2_full_get_receipt_object_set_mismatch")
    return descriptor(root, receipt_path)


def array_sha(x: np.ndarray) -> str:
    h = hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode())
    h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def materialize_session(session, entry: dict, contract: dict, plan: dict) -> tuple[np.ndarray, list[dict], dict]:
    """Bounded context only; invalid originals remain in the event sequence."""
    valid_signals, events = [], []
    context_seconds = contract["context_s"][1]
    for native in session.pure_lr_metadata():
        row = dict(native)
        start = row["analysis_anchor_sample"]
        stop = start + round(context_seconds * session.native_sfreq)
        row.update(event_uid=row["trial_uid"], partition=entry["partition"], acquisition_order_verified=True,
                   native_sfreq_hz=session.native_sfreq, context_start_sample=start, context_stop_sample=stop,
                   input_contract_sha256=plan["input_contract_sha256"], adapter_sha256=plan["adapter_code"]["sha256"],
                   processing_sha256=plan["processing_code"]["sha256"], valid=False, cache_index=None,
                   preprocessing_qc=None, signal_sha256=None, raw_context_sha256=None, exclusion_reason=None)
        if (row.get("feedback_end_verified") is not True or row.get("available_feedback_seconds") is None
                or row["available_feedback_seconds"] + 1e-9 < context_seconds
                or stop > row["feedback_stop_exclusive_sample"] or stop > row["native_trial_samples_including_padding"]):
            row["exclusion_reason"] = "insufficient_or_unverified_real_feedback"
        else:
            try:
                signal = session.context_volts(row["original_trial_index"], context_seconds)
                crop, qc = prepare_trial(signal, channel_names=list(CLINIC11), sfreq_hz=session.native_sfreq,
                                         task_sample=0, task_end_sample=signal.shape[1], task_end_verified=True,
                                         contract=contract, native_unit="V")
                qc = dict(qc, filter_context_ends_at_native_sample=stop)
                crop = np.asarray(crop, dtype=np.float32)
                if not np.isfinite(crop).all():
                    raise ValueError("Nonfinite context")
                row.update(valid=True, cache_index=len(valid_signals), preprocessing_qc=qc,
                           raw_context_sha256=array_sha(signal), signal_sha256=array_sha(crop))
                valid_signals.append(crop)
            except ValueError as exc:
                reasons = {"Nonfinite signal": "nonfinite_context", "Nonfinite context": "nonfinite_context",
                           "Zero-energy signal": "zero_energy_context", "Zero/nonfinite trial energy": "zero_or_nonfinite_preprocessed_energy",
                           "Insufficient real feedback; post-result padding is forbidden": "insufficient_real_feedback",
                           "Insufficient original-trial context; padding is forbidden": "insufficient_real_feedback"}
                if str(exc) not in reasons:
                    raise CacheError("unexpected_trial_schema_or_processing_error") from None
                row["exclusion_reason"] = reasons[str(exc)]
        events.append(row)
    x = np.stack(valid_signals) if valid_signals else np.empty((0, 11, contract["n_times"]), dtype=np.float32)
    audit = session.schema_audit()
    meta = {"dataset": "Stieger2021", "source_version": 1, "participant_id": entry["provider_subject"],
            "session_id": str(entry["session_ordinal"]), "partition": entry["partition"],
            "source_record_id": entry["name"], "source_file_sha256": session.source_sha256,
            "source_file_bytes": entry["size"], "native_sfreq_hz": session.native_sfreq,
            "source_n_times": [r["native_trial_samples_including_padding"] for r in session.rows],
            "acquisition_order_verified": True, "original_lr_trials": len(events), "valid_trials": len(x),
            "invalid_trials": len(events)-len(x), "stage": STAGE, "stored_unit": "V", "normalization_applied": False,
            "input_contract_sha256": plan["input_contract_sha256"], "split_sha256": plan["split_sha256"],
            "adapter_code": plan["adapter_code"], "processing_code": plan["processing_code"],
            "native_schema_audit": audit, "synthetic": False, "old_npz_reused": False}
    if len({r["trial_uid"] for r in events}) != len(events) or any(r["tasknumber"] != 1 or r["label"] not in (0, 1) for r in events):
        raise CacheError("original_lr_event_identity_invalid")
    return x, events, meta


def restore_raw(entry: dict, proof: dict, raw: Path, *, root: Path, report: Path, restore_fn=None) -> dict:
    source = proof["entry"]
    selected = dict(source, path=raw.relative_to(root).as_posix())
    if restore_fn is None:
        if not all(os.environ.get(v) for v in r2_backup.REQUIRED_VARIABLES):
            raise CacheError("existing_r2_environment_required")
        result = r2_restore.restore_one(r2_backup.create_client(), os.environ["R2_BUCKET"], selected, root)
    else:
        result = restore_fn(entry=selected, root=root)
    base.atomic_json(report, {"schema_version": 1, "source_backup_receipt": proof["backup_receipt"],
                              "source_key": source["key"], "object": result, "credentials_recorded": False})
    if (result.get("status") not in {"restored_verified", "already_present_verified"}
            or (result.get("status") == "restored_verified" and
                (result.get("complete_body_read") is not True or result.get("get_sha256") != source["sha256"]
                 or result.get("bytes_read") != source["bytes"]))):
        raise CacheError("raw_restore_full_get_sha256_failed")
    info = base.verify_local(entry, raw)
    if info["sha256"] != source["sha256"]:
        raise CacheError("restored_raw_sha_disagrees_with_original_receipt")
    return info


def verify_session_state(state: dict, root: Path, plan: dict) -> bool:
    if state.get("status") != "COMPLETE":
        return False
    if state.get("plan_sha256") != plan["plan_sha256"]:
        raise CacheError("cache_state_plan_identity_changed")
    raw_proof=checked(root,state["raw_backup_receipt"])
    originals,_=r2_restore.load_verified_receipt(raw_proof)
    originals=[r for r in originals if r["role"]=="source_development_raw" and r["dataset"]=="Stieger2021"]
    if (len(originals)!=1 or originals[0]["sha256"]!=state["raw_integrity"]["sha256"]
            or originals[0]["bytes"]!=state["raw_integrity"]["bytes"] or Path(originals[0]["path"]).name!=state["name"]):
        raise CacheError("original_full_get_proof_disagrees_with_cache_state")
    for artifact in state["artifacts"]:
        checked(root, artifact)
    receipt = checked(root, state["cache_backup_receipt"])
    rows, _ = r2_restore.load_verified_receipt(receipt)
    expected = {(d["path"], d["sha256"]) for d in state["artifacts"]}
    if {(r["path"], r["sha256"]) for r in rows} != expected:
        raise CacheError("cache_state_backup_full_get_set_mismatch")
    return True


def cleanup_owned_raw(state: dict, *, root: Path, object_dir: Path, plan: dict) -> None:
    if not state.get("raw_owned"):
        return
    marker = object_dir / "raw_ownership.json"
    r2_restore.no_symlink_components(marker)
    if not marker.exists():
        raise CacheError("raw_ownership_marker_missing_preserve_raw")
    ownership = json.loads(marker.read_text())
    binding = {k: state[k] for k in ("plan_sha256", "file_id", "raw_path")}
    if any(ownership.get(k) != v for k, v in binding.items()) or ownership.get("created_by") != "tools/prepare_stieger_cache.py":
        raise CacheError("raw_ownership_marker_mismatch_preserve_raw")
    raw = local(root, state["raw_path"])
    if raw.exists():
        sha, size = r2_backup.sha256_file(raw)
        if sha != state["raw_integrity"]["sha256"] or size != state["raw_integrity"]["bytes"]:
            raise CacheError("owned_raw_integrity_changed_preserve_raw")
        raw.unlink()
    state["owned_raw_removed_after_all_full_get_verifications"] = True


def owned_partial_snapshot(state: dict, *, root: Path, object_dir: Path, plan: dict) -> dict:
    marker_path=object_dir/"raw_ownership.json"
    r2_restore.no_symlink_components(marker_path)
    if not state.get("raw_owned") or not marker_path.is_file():
        raise CacheError("transfer_raw_ownership_unverified")
    marker=json.loads(marker_path.read_text())
    if (marker.get("created_by")!="tools/prepare_stieger_cache.py"
            or any(marker.get(k)!=state.get(k) for k in ("plan_sha256","file_id","raw_path"))
            or state["plan_sha256"]!=plan["plan_sha256"]):
        raise CacheError("transfer_raw_ownership_marker_mismatch")
    raw=local(root,state["raw_path"])
    expected=root/"data/stieger-cache-raw"/plan["plan_sha256"]/str(state["file_id"])/state["name"]
    if raw!=expected:raise CacheError("transfer_raw_outside_owned_stage")
    snapshot={}
    for path in raw.parent.iterdir():
        local(root,path.relative_to(root).as_posix())
        if path==raw:continue
        if (not path.is_file() or not path.name.startswith(raw.name+".") or not path.name.endswith(".partial")):
            raise CacheError("untracked_file_in_exclusive_raw_stage_preserved")
        sha,size=r2_backup.sha256_file(path)
        snapshot[path.relative_to(root).as_posix()]={"sha256":sha,"bytes":size}
    return snapshot


def recover_open_transfer(state: dict, *, root: Path, object_dir: Path, plan: dict, state_path: Path) -> None:
    intent=state.get("transfer_intent")
    if not intent:return
    snapshot=owned_partial_snapshot(state,root=root,object_dir=object_dir,plan=plan)
    counter={"official_get":"official_provider_body_bytes","r2_restore":"r2_restore_body_bytes"}.get(intent.get("kind"))
    if (counter is None or type(intent.get("reserved_bytes")) is not int or intent["reserved_bytes"]<0
            or intent.get("raw_path")!=state["raw_path"] or intent.get("file_id")!=state["file_id"]
            or state.get(counter,0)<intent["reserved_bytes"]):
        raise CacheError("open_transfer_reservation_unverified_budget_closed")
    # Reservation was already durably charged before network I/O. Hard-kill
    # recovery never turns a partially observed response into free byte budget.
    state.setdefault("transfer_history",[]).append(dict(intent,status="HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED",
                                                       recovered_at_utc=r2_backup.utc_now(),owned_partials_after=snapshot))
    state["owned_partials"]=snapshot
    state.pop("transfer_intent")
    base.atomic_json(state_path,state)


def begin_transfer(state: dict, *, kind: str, reserve: int, available: int | None,
                   root: Path, object_dir: Path, plan: dict, state_path: Path) -> None:
    if state.get("transfer_intent"):raise CacheError("prior_transfer_intent_not_reconciled")
    if reserve<0 or (available is not None and reserve>available):
        raise CacheError("official_provider_download_budget_exhausted" if kind=="official_get" else "r2_restore_download_budget_exhausted")
    snapshot=owned_partial_snapshot(state,root=root,object_dir=object_dir,plan=plan)
    if snapshot!=state.get("owned_partials",{}):
        raise CacheError("unmarked_or_changed_partial_cannot_be_adopted")
    counter={"official_get":"official_provider_body_bytes","r2_restore":"r2_restore_body_bytes"}[kind]
    state[counter]=state.get(counter,0)+reserve
    state["transfer_budget_accounting"]="successful_actual_body_or_conservative_full_reservation_after_failure"
    state["transfer_intent"]={"kind":kind,"file_id":state["file_id"],"raw_path":state["raw_path"],
                              "reserved_bytes":reserve,"started_at_utc":r2_backup.utc_now(),"owned_partials_before":snapshot}
    base.atomic_json(state_path,state)


def finish_transfer(state: dict, *, actual: int | None, root: Path, object_dir: Path, plan: dict, state_path: Path) -> None:
    intent=state.get("transfer_intent")
    if not intent:return
    counter={"official_get":"official_provider_body_bytes","r2_restore":"r2_restore_body_bytes"}[intent["kind"]]
    snapshot=owned_partial_snapshot(state,root=root,object_dir=object_dir,plan=plan)
    if actual is not None:
        if type(actual) is not int or actual<0 or actual>intent["reserved_bytes"]:
            raise CacheError("successful_transfer_exceeded_reserved_byte_budget")
        state[counter]+=actual-intent["reserved_bytes"]
        verified=counter.replace("body_bytes","verified_body_bytes")
        state[verified]=state.get(verified,0)+actual
    state.setdefault("transfer_history",[]).append(dict(intent,status="SUCCESS_ACTUAL_BODY_SETTLED" if actual is not None else "FAILURE_FULL_RESERVATION_CONSUMED",
                                                       actual_verified_bytes=actual,owned_partials_after=snapshot))
    state["owned_partials"]=snapshot
    state.pop("transfer_intent")
    base.atomic_json(state_path,state)


def process_session(entry: dict, *, root: Path, output: Path, contract: dict, plan: dict, raw_catalog: dict,
                    backup_fn=r2_backup.run_backup, load_fn=load_session, restore_fn=None, download_fn=base.download_object,
                    remaining_source_bytes=None, remaining_restore_bytes=None) -> dict:
    object_dir = local(root,(output / entry["partition"] / entry["provider_subject"] / f"session-{entry['session_ordinal']}").relative_to(root).as_posix())
    object_dir.mkdir(parents=True, exist_ok=True)
    state_path = local(root,(object_dir / "state.json").relative_to(root).as_posix())
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "PENDING", "plan_sha256": plan["plan_sha256"], "file_id": entry["id"], "name": entry["name"],
        "partition": entry["partition"], "started_at_utc": r2_backup.utc_now()}
    if state.get("plan_sha256") != plan["plan_sha256"] or state.get("file_id") != entry["id"]:
        raise CacheError("existing_session_state_identity_differs")
    recover_open_transfer(state,root=root,object_dir=object_dir,plan=plan,state_path=state_path)
    if verify_session_state(state, root, plan):
        cleanup_owned_raw(state, root=root, object_dir=object_dir, plan=plan)
        return state
    # A committed full-GET cache proof survives a crash between raw removal and
    # COMPLETE. Resume verifies the proof before touching raw or downloading.
    if state.get("cache_backup_receipt") and state.get("artifacts"):
        candidate = dict(state, status="COMPLETE")
        if verify_session_state(candidate, root, plan):
            cleanup_owned_raw(state, root=root, object_dir=object_dir, plan=plan)
            state.update(status="COMPLETE", completed_at_utc=r2_backup.utc_now())
            base.atomic_json(state_path, state)
            return state
    existing = local(root,(Path("data/stieger2021/raw") / entry["name"]).as_posix())
    temporary = local(root,(Path("data/stieger-cache-raw") / plan["plan_sha256"] / str(entry["id"]) / entry["name"]).as_posix())
    proof = raw_catalog.get(entry["id"])
    partial_before=0
    try:
        base.check_space(root, entry["size"])
        if existing.exists():
            raw, owned = existing, False
        else:
            raw = temporary
            marker_path = object_dir / "raw_ownership.json"
            if raw.exists():
                marker = json.loads(marker_path.read_text()) if marker_path.exists() else {}
                owned = (marker.get("created_by") == "tools/prepare_stieger_cache.py"
                         and marker.get("plan_sha256") == plan["plan_sha256"] and marker.get("file_id") == entry["id"]
                         and marker.get("raw_path") == raw.relative_to(root).as_posix())
            else:
                owned = True
                write_once_json(marker_path, {"created_by": "tools/prepare_stieger_cache.py", "plan_sha256": plan["plan_sha256"],
                                             "file_id": entry["id"], "raw_path": raw.relative_to(root).as_posix()})
        raw.parent.mkdir(parents=True, exist_ok=True)
        state.update(status="RAW_VERIFY_OR_FETCH", raw_path=raw.relative_to(root).as_posix(), raw_owned=owned)
        base.atomic_json(state_path, state)
        if raw.exists():
            info = base.verify_local(entry, raw)
        elif proof:
            begin_transfer(state,kind="r2_restore",reserve=entry["size"],available=remaining_restore_bytes,
                           root=root,object_dir=object_dir,plan=plan,state_path=state_path)
            report=object_dir/"raw_restore_receipt.json"
            info = restore_raw(entry, proof, raw, root=root, report=object_dir/"raw_restore_receipt.json", restore_fn=restore_fn)
            restore_result=json.loads(report.read_text())["object"]
            finish_transfer(state,actual=restore_result.get("bytes_read",0),root=root,object_dir=object_dir,plan=plan,state_path=state_path)
            state["raw_restore_receipt"] = descriptor(root, object_dir/"raw_restore_receipt.json")
        else:
            partial=raw.with_suffix(".mat.partial")
            partial_before=partial.stat().st_size if partial.exists() else 0
            begin_transfer(state,kind="official_get",reserve=entry["size"]-partial_before,available=remaining_source_bytes,
                           root=root,object_dir=object_dir,plan=plan,state_path=state_path)
            info = download_fn(entry, raw.with_suffix(".mat.partial"), raw)
            finish_transfer(state,actual=max(0,info["bytes"]-info.get("resumed_from_bytes",partial_before)),
                            root=root,object_dir=object_dir,plan=plan,state_path=state_path)
        if proof and info["sha256"] != proof["entry"]["sha256"]:
            raise CacheError("local_raw_disagrees_with_prior_full_get_receipt")
        state.update(status="RAW_VERIFIED", raw_integrity=info)
        base.atomic_json(state_path, state)
        prior_raw_receipt=state.get("raw_backup_receipt")
        if prior_raw_receipt:
            originals,_=r2_restore.load_verified_receipt(checked(root,prior_raw_receipt))
            originals=[r for r in originals if r["role"]=="source_development_raw" and r["dataset"]=="Stieger2021"]
            if (len(originals)!=1 or originals[0]["sha256"]!=info["sha256"] or originals[0]["bytes"]!=info["bytes"]
                    or Path(originals[0]["path"]).name!=entry["name"]):
                raise CacheError("prior_original_receipt_changed_on_resume")
        elif not proof:
            native_integrity = object_dir/"raw_integrity.json"
            write_once_json(native_integrity, {"file_id": entry["id"], "name": entry["name"], "dataset": "Stieger2021",
                                              "raw_integrity": {k:info[k] for k in ("sha256", "bytes", "md5")},
                                              "official_md5_evidence": base.official_md5(entry)[1]})
            raw_receipt = verified_backup(root, object_dir/"raw_backup_manifest.json", object_dir/"raw_backup_receipt.json",
                                          prefix=f"paper3/stieger-original-v1/{entry['id']}/{info['sha256']}/",
                                          files=[(raw,"source_development_raw"),(native_integrity,"audit")], backup_fn=backup_fn)
            state["raw_backup_receipt"] = raw_receipt
        else:
            state["raw_backup_receipt"] = proof["backup_receipt"]
        base.atomic_json(state_path, state)
        session = load_fn(raw, expected_sha256=info["sha256"])
        if session.participant_id != entry["provider_subject"] or session.session_id != str(entry["session_ordinal"]):
            raise CacheError("native_identity_changed")
        x, events, meta = materialize_session(session, entry, contract, plan)
        del session
        meta.update(raw_integrity={k:info[k] for k in ("sha256","bytes","md5")},
                    raw_backup_receipt=state["raw_backup_receipt"], raw_restore_receipt=state.get("raw_restore_receipt"))
        x_path, events_path, meta_path = object_dir/"X.npy", object_dir/"events.json", object_dir/"metadata.json"
        write_array(x_path, x)
        write_once_json(events_path, events)
        write_once_json(meta_path, meta)
        artifacts = [descriptor(root,p) for p in (x_path,events_path,meta_path)]
        state.update(status="NUMERICAL_CACHE_WRITTEN", artifacts=artifacts,
                     cache_bytes=sum((root/d["path"]).stat().st_size for d in artifacts))
        base.atomic_json(state_path, state)
        cache_receipt = verified_backup(root, object_dir/"cache_backup_manifest.json", object_dir/"cache_backup_receipt.json",
                                        prefix=f"paper3/stieger-cache-v2/{plan['plan_sha256']}/{entry['id']}/",
                                        files=[(x_path,"result"),(events_path,"result"),(meta_path,"audit")], backup_fn=backup_fn)
        record = {k:meta[k] for k in ("dataset","source_version","participant_id","session_id","partition","source_record_id",
                                      "source_file_sha256","source_file_bytes","native_sfreq_hz","source_n_times","acquisition_order_verified",
                                      "original_lr_trials","valid_trials","invalid_trials","raw_integrity","raw_backup_receipt")}
        record.update(X=descriptor(root,x_path), events=descriptor(root,events_path), metadata=descriptor(root,meta_path),
                      adapter_code=plan["adapter_code"], processing_code=plan["processing_code"], cache_backup_receipt=cache_receipt)
        state.update(status="CACHE_BACKUP_VERIFIED", cache_backup_receipt=cache_receipt, session_record=record)
        base.atomic_json(state_path, state)
        verify_session_state(dict(state,status="COMPLETE"), root, plan)
        cleanup_owned_raw(state, root=root, object_dir=object_dir, plan=plan)
        state.update(status="COMPLETE", completed_at_utc=r2_backup.utc_now())
        base.atomic_json(state_path, state)
        return state
    except BaseException as exc:
        finish_transfer(state,actual=None,root=root,object_dir=object_dir,plan=plan,state_path=state_path)
        state.update(status="FAILED_CLOSED", error_type=type(exc).__name__,
                     error_code=str(exc) if isinstance(exc, (CacheError,base.BatchError)) else "operation_failed_details_suppressed",
                     failed_at_utc=r2_backup.utc_now())
        base.atomic_json(state_path, state)
        raise CacheError(state["error_code"]) from None


def assemble_manifest(root: Path, output: Path, plan: dict, states: list[dict]) -> dict:
    if len(states) != 598 or {s["file_id"] for s in states} != {f["id"] for f in plan["files"]}:
        raise CacheError("all598_session_states_required")
    uids, hashes, duplicates = set(), {}, []
    records = []
    for state in states:
        if not verify_session_state(state, root, plan):
            raise CacheError("unverified_session_cannot_enter_complete_manifest")
        record = state["session_record"]
        events = json.loads(checked(root,record["events"]).read_text())
        for row in events:
            if row["event_uid"] in uids:
                raise CacheError("duplicate_original_event_uid")
            uids.add(row["event_uid"])
            if row["valid"]:
                h = row["signal_sha256"]
                if h in hashes:
                    old = hashes[h]
                    duplicates.append({"signal_sha256":h,"first":old,"second":{"event_uid":row["event_uid"],"partition":record["partition"]}})
                    if old["partition"] != record["partition"]:
                        base.atomic_json(output/"duplicate_audit.json", {"status":"SPLIT_INVALID","duplicates":duplicates})
                        raise CacheError("identical_processed_trial_crosses_frozen_person_split")
                else:
                    hashes[h] = {"event_uid":row["event_uid"],"partition":record["partition"]}
        records.append(record)
    duplicate_path=output/"duplicate_audit.json"
    write_once_json(duplicate_path,{"status":"NO_CROSS_SPLIT_EXACT_SIGNAL_DUPLICATES","within_partition_duplicates":duplicates,
                                    "basis":"float32 pre-RMS processed original trial bytes plus shape/dtype; no near-duplicate biological identity claim"})
    return {"schema_version":SCHEMA,"status":"STIEGER_CACHE_COMPLETE","dataset":"Stieger2021","source_version":1,
            "synthetic":False,"old_npz_reused":False,"stage":STAGE,"normalization_applied":False,"stored_unit":"V",
            "input_contract":plan["input_contract"],"input_contract_sha256":plan["input_contract_sha256"],
            "participant_split":plan["participant_split"],"split_sha256":plan["split_sha256"],
            "official_inventory":plan["official_inventory"],"adapter_code":plan["adapter_code"],
            "processing_code":plan["processing_code"],"cache_builder_code":plan["cache_builder_code"],
            "partition_scope":list(PARTITIONS),"participant_count":62,"session_count":598,"sessions":records,
            "original_lr_trials":sum(r["original_lr_trials"] for r in records),"valid_trials":sum(r["valid_trials"] for r in records),
            "duplicate_audit":descriptor(root,duplicate_path),"plan_sha256":plan["plan_sha256"],
            "training_started":False,"internal_holdout_evaluated":False,"external_payload_accessed":False}


def update_w3_task(root: Path, plan: dict, status: str, **extra) -> None:
    path=root/"status/tasks.json"
    tasks=json.loads(path.read_text())
    tasks["W3"]={"status":status,"plan_sha256":plan["plan_sha256"],"input_contract_sha256":plan["input_contract_sha256"],
                 "split_sha256":plan["split_sha256"],"training_started":False,"internal_holdout_evaluated":False,**extra}
    base.atomic_json(path,tasks)


def prepare(root: Path, plan: dict, *, execute=False, max_new_files=None, backup_fn=r2_backup.run_backup) -> dict:
    root=root.resolve(strict=True)
    contract,split=validate_plan(root,plan,execute=execute)
    if not execute:
        return {"status":"DRY_RUN_EXACT_ALL598_PLAN_NO_NETWORK","plan_sha256":plan["plan_sha256"],
                "files":598,"max_download_bytes":plan["max_download_bytes"],"max_r2_restore_bytes":plan["max_r2_restore_bytes"],
                "max_wall_seconds":plan["max_wall_seconds"],"max_gpu_hours":0,"training_started":False}
    output=local(root,plan["output"])
    marker=output/"ownership.json"
    if output.exists() and not marker.exists():
        raise CacheError("preexisting_cache_directory_not_owned")
    output.mkdir(parents=True,exist_ok=True)
    write_once_json(marker,{"schema_version":SCHEMA,"created_by":"tools/prepare_stieger_cache.py","plan_sha256":plan["plan_sha256"]})
    with (output/".prepare.lock").open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise CacheError("stieger_cache_worker_already_active") from None
        progress_path=output/"progress.json"
        old=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        previous=old.get("active_wall_elapsed_s",0.)
        if old and old.get("plan_sha256")!=plan["plan_sha256"]:raise CacheError("progress_plan_identity_changed")
        if not isinstance(previous,(float,int)) or not math.isfinite(previous) or previous<0:
            raise CacheError("unknown_prior_elapsed_time_budget_closed")
        states=[];new_files=0;cache_bytes=0
        official_bytes=restore_bytes=0
        progress={"status":"RUNNING_LOCAL_CPU","plan_sha256":plan["plan_sha256"],"active_wall_elapsed_s":previous,
                  "gpu_jobs_started":0,"remote_jobs_started":0,"external_payload_accessed":False,"completed_sessions":0,
                  "official_provider_body_bytes":0,"r2_restore_body_bytes":0,
                  "transfer_budget_accounting":"successful actual bodies or conservative full reservations after interruption"}
        if old.get("wall_intent"):progress["wall_intent"]=old["wall_intent"]
        try:
            # Reconcile a hard-killed interval before opening any new transfer.
            # UTC/monotonic disagreement closes the budget instead of dropping
            # the interrupted session's elapsed time.
            wall_checkpoint(progress_path,progress,reopen=True)
            update_w3_task(root,plan,"IN_PROGRESS_LOCAL_CPU",progress_path=progress_path.relative_to(root).as_posix())
            if progress["active_wall_elapsed_s"]>=plan["max_wall_seconds"]:
                progress.update(status="RESOURCE_BUDGET_STOP_NO_W3_COMPLETE",unverified_raw_removed=False)
                update_w3_task(root,plan,"RESOURCE_BUDGET_STOP_NOT_COMPLETE")
                wall_checkpoint(progress_path,progress,reopen=False)
                return progress
            catalog=raw_receipt_catalog(root,plan["files"])
            for entry in plan["files"]:
                sp=output/entry["partition"]/entry["provider_subject"]/f"session-{entry['session_ordinal']}"/"state.json"
                if sp.exists():
                    prior_state=json.loads(sp.read_text())
                    if prior_state.get("plan_sha256")!=plan["plan_sha256"]:raise CacheError("session_state_plan_mismatch")
                    recover_open_transfer(prior_state,root=root,object_dir=sp.parent,plan=plan,state_path=sp)
                    official_bytes+=prior_state.get("official_provider_body_bytes",0)
                    restore_bytes+=prior_state.get("r2_restore_body_bytes",0)
            progress.update(official_provider_body_bytes=official_bytes,r2_restore_body_bytes=restore_bytes)
            for entry in plan["files"]:
                progress.update(next_file=entry["name"])
                wall_checkpoint(progress_path,progress,reopen=True)
                if (progress["active_wall_elapsed_s"]>=plan["max_wall_seconds"] or cache_bytes>=plan["max_cache_bytes"]
                        or official_bytes>plan["max_download_bytes"] or restore_bytes>plan["max_r2_restore_bytes"]):
                    progress.update(status="RESOURCE_BUDGET_STOP_NO_W3_COMPLETE",unverified_raw_removed=False)
                    update_w3_task(root,plan,"RESOURCE_BUDGET_STOP_NOT_COMPLETE")
                    wall_checkpoint(progress_path,progress,reopen=False);return progress
                state_path=output/entry["partition"]/entry["provider_subject"]/f"session-{entry['session_ordinal']}"/"state.json"
                done=state_path.exists() and verify_session_state(json.loads(state_path.read_text()),root,plan)
                if not done and max_new_files is not None and new_files>=max_new_files:
                    progress.update(status="PARTIAL_BATCH_NO_W3_COMPLETE")
                    update_w3_task(root,plan,"PARTIAL_CACHE_NOT_COMPLETE",completed_sessions=len(states))
                    wall_checkpoint(progress_path,progress,reopen=False);return progress
                old_state=json.loads(state_path.read_text()) if state_path.exists() else {}
                state=process_session(entry,root=root,output=output,contract=contract,plan=plan,raw_catalog=catalog,backup_fn=backup_fn,
                                      remaining_source_bytes=plan["max_download_bytes"]-official_bytes,
                                      remaining_restore_bytes=plan["max_r2_restore_bytes"]-restore_bytes)
                official_bytes+=state.get("official_provider_body_bytes",0)-old_state.get("official_provider_body_bytes",0)
                restore_bytes+=state.get("r2_restore_body_bytes",0)-old_state.get("r2_restore_body_bytes",0)
                states.append(state);new_files+=not done;cache_bytes+=state["cache_bytes"]
                progress.update(completed_sessions=len(states),cache_bytes=cache_bytes,
                                official_provider_body_bytes=official_bytes,r2_restore_body_bytes=restore_bytes)
                wall_checkpoint(progress_path,progress,reopen=True)
                print(json.dumps({"file":entry["name"],"status":"COMPLETE_CACHE_AND_FULL_GET","sessions":len(states),"total":598}),flush=True)
            if cache_bytes>plan["max_cache_bytes"] or progress["active_wall_elapsed_s"]>=plan["max_wall_seconds"]:
                raise CacheError("resource_budget_reached_before_global_commit")
            manifest=assemble_manifest(root,output,plan,states)
            manifest_path=output/"stieger_cache.json";write_once_json(manifest_path,manifest)
            files=[(manifest_path,"manifest"),(output/"duplicate_audit.json","audit")]
            for item in (plan["official_inventory"],plan["input_contract"],plan["participant_split"]):files.append((checked(root,item),"protocol"))
            for state in states:files.append((checked(root,state["cache_backup_receipt"]),"audit"))
            receipt=verified_backup(root,output/"manifest_backup_manifest.json",output/"manifest_backup_receipt.json",
                                    prefix=f"paper3/stieger-cache-v2/{plan['plan_sha256']}/manifest/",files=files,backup_fn=backup_fn)
            wall_checkpoint(progress_path,progress,reopen=True)
            if progress["active_wall_elapsed_s"]>=plan["max_wall_seconds"]:
                raise CacheError("wall_budget_reached_before_w3_complete_receipt")
            update_w3_task(root,plan,"COMPLETE",cache_manifest=descriptor(root,manifest_path),cache_r2_receipt=receipt,sessions=598,people=62)
            progress.update(status="W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY",cache_manifest=descriptor(root,manifest_path),cache_r2_receipt=receipt)
        except BaseException as exc:
            progress.update(status="FAILED_CLOSED_NO_W3_COMPLETE",error_type=type(exc).__name__,
                            error_code=str(exc) if isinstance(exc,(CacheError,base.BatchError)) else "operation_failed_details_suppressed")
            update_w3_task(root,plan,"FAILED_CLOSED_NOT_COMPLETE",error_code=progress["error_code"],completed_sessions=len(states))
            # Each object persists byte counters even on interrupted partials.
            # Reflect them in the top-level receipt without discarding failed
            # source/R2 transfer costs during a later resume.
            persisted=[]
            for entry in plan["files"]:
                sp=output/entry["partition"]/entry["provider_subject"]/f"session-{entry['session_ordinal']}"/"state.json"
                if sp.exists():persisted.append(json.loads(sp.read_text()))
            progress["official_provider_body_bytes"]=sum(s.get("official_provider_body_bytes",0) for s in persisted)
            progress["r2_restore_body_bytes"]=sum(s.get("r2_restore_body_bytes",0) for s in persisted)
        wall_checkpoint(progress_path,progress,reopen=False)
        return progress


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=REPO)
    parser.add_argument("--inventory",default="data_catalog/stieger2021/official_inventory.json")
    parser.add_argument("--contract",default="contracts/input_contract.json")
    parser.add_argument("--split")
    parser.add_argument("--output",default="cache/stieger_v2")
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--write-plan",type=Path)
    parser.add_argument("--max-wall-seconds",type=int,default=86400)
    parser.add_argument("--max-cache-bytes",type=int,default=4*1024**3)
    parser.add_argument("--max-new-files",type=int)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    try:
        root=args.root.resolve(strict=True)
        plan=json.loads(args.plan.read_text()) if args.plan else build_plan(root,inventory_path=local(root,args.inventory),
                contract_path=local(root,args.contract),split_path=local(root,args.split) if args.split else split_path_default(root),
                output=args.output,max_wall_seconds=args.max_wall_seconds,max_cache_bytes=args.max_cache_bytes)
        if args.write_plan:write_once_json(args.write_plan,plan)
        if args.max_new_files is not None and args.max_new_files<1:raise CacheError("max_new_files_must_be_positive")
        result=prepare(root,plan,execute=args.execute,max_new_files=args.max_new_files)
        print(json.dumps(result,indent=2),flush=True)
        return 0 if result["status"] in {"DRY_RUN_EXACT_ALL598_PLAN_NO_NETWORK","W3_COMPLETE_CPU_CACHE_AND_FULL_GET_ONLY","PARTIAL_BATCH_NO_W3_COMPLETE","RESOURCE_BUDGET_STOP_NO_W3_COMPLETE"} else 1
    except BaseException as exc:
        print(json.dumps({"status":"EXECUTION_STOPPED_NOT_COMPLETE" if args.execute else "BLOCKED_NO_EXECUTION","execution_requested":args.execute,"error_type":type(exc).__name__,
                          "error_code":str(exc) if isinstance(exc,(CacheError,base.BatchError)) else "operation_failed_details_suppressed",
                          "training_started":False,"gpu_jobs_started":0}),flush=True)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
