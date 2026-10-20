#!/usr/bin/env python3
"""Prepare all 125 Ma2022 v1 sessions, with explicit CPU-only development gates.

Default: freeze and inspect the local plan, without network or R2 operations.
Execution must be started by the root coordinator. Ma is never added to source
training, main clinic11, an original B20 experiment, or sealed external data.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urljoin,urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from p3work import source_extension_io as storage
from p3work.source_extension_coordinator import (
    Ledger, PreparationError, atomic_json, checked, descriptor, digest, local,
    receipt_rows, sha_file,
)
from p3work.source_extensions import ma2022


DATASET = "Ma2022"
CATALOG = "audit/data_plan_20261019/other_datasets/evidence/ma2022_figshare_v1_api.json"
CATALOG_SHA256 = "81b92a8f3470cd951e6805ea6056348368d28b4373f851002aca916a0a43c9f4"
RAW_BYTES = 768836911
CREATOR = "tools/prepare_ma2022_development.py"
ROLE = "independent_development_unassigned"
MAX_NATIVE = 64*1024**2
REDIRECT_ALLOWANCE = 65536
CHUNK = 1024**2
DEFAULT_PLAN = "catalog/source_extensions_20261019/Ma2022/development_plan_native_v2.json"


def _records(catalog):
    if (catalog.get("version") != 1 or catalog.get("license",{}).get("name") != "CC BY 4.0"
            or catalog.get("download_disabled") is not False or catalog.get("is_confidential") is not False
            or catalog.get("is_embargoed") is not False):
        raise PreparationError("public_Ma2022_v1_CC_BY4_catalog_required")
    pairs = {}
    for item in catalog["files"]:
        name = item.get("name","")
        if not name.endswith(("_eeg.edf","_events.tsv")):
            continue
        match = re.fullmatch(r"(sub-\d{3})_(ses-\d{2})_task_motorimagery_(eeg\.edf|events\.tsv)",name)
        if not match:
            raise PreparationError("unexpected_native_component_name")
        subject,session,kind = match.groups()
        md5 = item.get("computed_md5")
        url = item.get("download_url")
        if (not re.fullmatch("[0-9a-f]{32}",str(md5)) or item.get("supplied_md5") != md5
                or type(item.get("size")) is not int or not 0 < item["size"] < MAX_NATIVE
                or type(item.get("id")) is not int or item.get("is_link_only") is not False
                or url != f"https://ndownloader.figshare.com/files/{item['id']}"):
            raise PreparationError("official_native_size_MD5_or_URL_invalid")
        key = (subject,session)
        rows = pairs.setdefault(key,{})
        if kind in rows:
            raise PreparationError("duplicate_native_pair_component")
        rows[kind] = {"path":f"{storage.RAW_ROOT}/{DATASET}/{name}","name":name,
            "bytes":item["size"],"provider_md5":md5,"url":url,"file_id":item["id"]}
    expected = {(f"sub-{s:03d}",f"ses-{d:02d}") for s in range(1,26) for d in range(1,6)}
    if set(pairs) != expected or any(set(v) != {"eeg.edf","events.tsv"} for v in pairs.values()):
        raise PreparationError("all_25_people_times_5_native_pairs_required")
    records = [{"record_id":s+"_"+d,"subject_id":s,"session_id":d,
        "raw_files":[pairs[s,d]["eeg.edf"],pairs[s,d]["events.tsv"]],
        "output":f"{storage.RESULT_ROOT}/{DATASET}/development_car3_native_v2/{s}/{d}"} for s,d in sorted(pairs)]
    if sum(f["bytes"] for row in records for f in row["raw_files"]) != RAW_BYTES:
        raise PreparationError("frozen_native_pool_byte_total_mismatch")
    return records


def build_plan(root, catalog_path, *, max_provider_body_bytes=None,max_r2_get_body_bytes=None,
               max_wall_seconds=7200,max_cache_bytes=512*1024**2,disk_reserve_bytes=1024**3):
    root = Path(root).absolute()
    item = descriptor(root,Path(catalog_path))
    if item["sha256"] != CATALOG_SHA256:
        raise PreparationError("official_frozen_Ma2022_catalog_SHA_mismatch")
    records = _records(json.loads(checked(root,item).read_bytes()))
    budgets = {"provider_body_bytes":max_provider_body_bytes if max_provider_body_bytes is not None else 2*RAW_BYTES+500*REDIRECT_ALLOWANCE,
        "r2_get_body_bytes":max_r2_get_body_bytes if max_r2_get_body_bytes is not None else 2*RAW_BYTES+2*max_cache_bytes,
        "wall_seconds":max_wall_seconds,"cache_bytes":max_cache_bytes,"disk_reserve_bytes":disk_reserve_bytes}
    if any(type(v) is not int or v <= 0 for v in budgets.values()):
        raise PreparationError("positive_integer_resource_budgets_required")
    code_names = [CREATOR,"p3work/source_extension_io.py","p3work/source_extension_coordinator.py",
        "tools/r2_backup.py","tools/r2_restore.py","p3work/source_extensions/ma2022.py"]
    plan = {"schema_version":"paper3-Ma2022-all125-development-plan-native-v2","dataset":DATASET,
        "role":ROLE,"document_sha256":storage.DOCUMENT_SHA256,"catalog":item,
        "code":{p:descriptor(root,local(root,p)) for p in code_names},"records":records,
        "record_count":125,"raw_object_count":250,"raw_bytes":RAW_BYTES,"provider_version":1,
        "native_profile":{"clock":ma2022.CLOCK,"sfreq_hz":250,"retained_trial_samples":1000,
            "native_unit":"uV","conversion_to_V":1e-6,"stage":ma2022.STAGE,
            "event_layout":"declared_columns_or_strict_headerless_six_column_native_v2",
            "original_B20_prefix_recoverable":False,"released_windows_seconds":4},
        "training_ready":False,"scientific_pool_assignment_frozen":False,"source_training_allowed":False,
        "primary11_compatible":False,"source3_training_pool_member":False,"actual_fits":0,
        "GPU_actions":0,"sealed_external_access_allowed":False,"inflight_records":1,"budgets":budgets}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(root,plan):
    original = dict(plan)
    if digest({k:v for k,v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256"):
        raise PreparationError("immutable_development_plan_SHA_mismatch")
    for item in [plan["catalog"],*plan["code"].values()]:checked(root,item)
    b = plan["budgets"]
    rebuilt = build_plan(root,checked(root,plan["catalog"]),max_provider_body_bytes=b["provider_body_bytes"],
        max_r2_get_body_bytes=b["r2_get_body_bytes"],max_wall_seconds=b["wall_seconds"],
        max_cache_bytes=b["cache_bytes"],disk_reserve_bytes=b["disk_reserve_bytes"])
    if rebuilt != original:
        raise PreparationError("frozen_catalog_code_document_selection_or_budget_changed")


def native_identity(root,item,*,expected_sha=None):
    path = local(root,item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or path.stat().st_size >= MAX_NATIVE:
        raise PreparationError("native_object_missing_or_length_mismatch")
    sha,md5 = hashlib.sha256(),hashlib.md5()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):sha.update(block);md5.update(block)
    if md5.hexdigest() != item["provider_md5"] or (expected_sha is not None and sha.hexdigest() != expected_sha):
        raise PreparationError("native_provider_MD5_or_pinned_SHA_mismatch")
    return {"path":item["path"],"bytes":item["bytes"],"sha256":sha.hexdigest(),
        "provider_md5":md5.hexdigest(),"provider_MD5_and_length_verified":True}


def download_object(item,target,*,session,ledger):
    partial = target.with_name(target.name+".owned.partial")
    if target.exists() or partial.is_symlink():
        raise PreparationError("existing_or_symlink_download_destination")
    target.parent.mkdir(parents=True,exist_ok=True)
    partial.unlink(missing_ok=True)  # caller verified this worker's durable ownership marker
    # Include one bounded overflow-read chunk as well as redirect bodies.
    reserved = item["bytes"]+REDIRECT_ALLOWANCE+CHUNK
    ledger.reserve("provider_body_bytes",reserved,item["path"])
    response = None
    try:
        url,redirect_bytes = item["url"],0
        for attempt in range(4):
            response = session.get(url,headers={"Accept-Encoding":"identity"},stream=True,
                                   timeout=(20,90),allow_redirects=False)
            if response.status_code not in {301,302,303,307,308}:break
            if attempt == 3:raise PreparationError("native_redirect_limit_exceeded")
            for block in response.iter_content(4096):
                redirect_bytes+=len(block)
                if redirect_bytes>REDIRECT_ALLOWANCE:raise PreparationError("redirect_body_exceeded_reserved_allowance")
            target_url=urljoin(url,response.headers.get("Location",""))
            parsed=urlsplit(target_url)
            if (parsed.scheme!="https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment
                    or not(parsed.hostname.endswith(".amazonaws.com") or parsed.hostname=="figshare.com"
                           or parsed.hostname.endswith(".figshare.com"))):
                raise PreparationError("unexpected_public_provider_redirect")
            response.close();response=None;url=target_url
        if (response.status_code != 200 or response.headers.get("Content-Encoding","identity") != "identity"
                or (response.headers.get("Content-Length") is not None
                    and int(response.headers["Content-Length"]) != item["bytes"])):
            raise PreparationError("native_provider_HTTP_or_length_mismatch")
        sha,md5,total = hashlib.sha256(),hashlib.md5(),0
        with partial.open("xb") as stream:
            for block in response.iter_content(CHUNK):
                if not isinstance(block,bytes) or total+len(block) > item["bytes"]:
                    raise PreparationError("native_body_overflow_or_nonbinary")
                stream.write(block);sha.update(block);md5.update(block);total+=len(block)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        if total != item["bytes"] or md5.hexdigest() != item["provider_md5"]:
            raise PreparationError("native_GET_incomplete_or_provider_MD5_mismatch")
        partial.replace(target)
        ledger.finish(verified_bytes=total+redirect_bytes)
        return {"sha256":sha.hexdigest(),"bytes":total,"provider_md5":md5.hexdigest(),
            "HTTP_status":200,"full_provider_GET_verified":True,"redirect_body_bytes":redirect_bytes}
    except BaseException:
        ledger.finish()
        raise
    finally:
        if response is not None:response.close()


def _backup_set(root,receipt,files):
    rows = receipt_rows(root,receipt)
    fields = storage.ENTRY_FIELDS
    expected = [{**descriptor(root,p),"dataset":DATASET,"role":role} for p,role in files]
    if {tuple(x[k] for k in fields) for x in rows} != {tuple(x[k] for k in fields) for x in expected}:
        raise PreparationError("R2_receipt_exact_file_role_SHA_set_mismatch")
    return rows


def backup_files(root,base,files,*,prefix,ledger,backup_fn):
    entries = [{**descriptor(root,p),"dataset":DATASET,"role":role} for p,role in files]
    manifest,receipt = base.with_name(base.name+"_manifest.json"),base.with_name(base.name+"_receipt.json")
    atomic_json(manifest,{"schema_version":1,"files":entries},immutable=True)
    if receipt.exists():
        existing = json.loads(receipt.read_bytes())
        if existing.get("status") == "verified":
            proof = descriptor(root,receipt)
            _backup_set(root,proof,files)
            if existing.get("prefix") != prefix:
                raise PreparationError("existing_backup_prefix_changed")
            return proof
    reservation = sum(e["bytes"] for e in entries)
    ledger.reserve("r2_get_body_bytes",reservation,prefix)
    try:
        result = backup_fn(manifest_path=manifest,root=root,prefix=prefix,receipt_path=receipt,execute=True)
        if result.get("status") != "verified" or result.get("execute") is not True:
            raise PreparationError("R2_PUT_full_GET_SHA_not_verified")
        proof = descriptor(root,receipt)
        _backup_set(root,proof,files)
        ledger.finish(verified_bytes=reservation)
        return proof
    except BaseException:
        ledger.finish()
        raise


def _raw_receipt(root,state,record):
    rows = receipt_rows(root,state["raw_backup_receipt"])
    actual = state.get("raw_identity",[])
    if (len(actual) != 2 or {x["path"] for x in actual} != {x["path"] for x in record["raw_files"]}
            or {(x["path"],x["bytes"],x["sha256"]) for x in rows} != {(x["path"],x["bytes"],x["sha256"]) for x in actual}
            or any(x["role"] != "source_development_raw" or x["dataset"] != DATASET for x in rows)):
        raise PreparationError("raw_R2_exact_identity_set_mismatch")
    for expected in record["raw_files"]:
        item = next(x for x in actual if x["path"] == expected["path"])
        if item.get("provider_md5") != expected["provider_md5"] or item["bytes"] != expected["bytes"]:
            raise PreparationError("native_identity_no_longer_matches_official_catalog")
    return actual


def validate_cache(root,record,raw_identity,native,output,plan):
    paths = {n:output/n for n in ["X.npy","y.npy","events.json","manifest.json"]}
    for p in paths.values():local(root,p.relative_to(root).as_posix())
    manifest = json.loads(paths["manifest.json"].read_bytes())
    if (manifest.get("dataset") != DATASET or manifest.get("provider_version") != 1
            or manifest.get("stage") != ma2022.STAGE or manifest.get("training_ready") is not False
            or manifest.get("raw_acquisition_prefix_eligible") is not False
            or manifest.get("source_acquisition_order_recovered") is not False
            or manifest.get("normalization_applied") is not False
            or manifest.get("scientific_pool_assignment_frozen") is not False or manifest.get("actual_fits") != 0):
        raise PreparationError("development_only_manifest_or_B20_gate_changed")
    adapter_sha = plan["code"]["p3work/source_extensions/ma2022.py"]["sha256"]
    if manifest.get("adapter_code",{}).get("sha256") != adapter_sha:
        raise PreparationError("cache_native_adapter_SHA_changed")
    for key,item in zip(["edf","events"],raw_identity):
        if manifest["identity"][key]["sha256"] != item["sha256"] or native["identity"][key]["sha256"] != item["sha256"]:
            raise PreparationError("cache_or_native_audit_raw_identity_mismatch")
    for name in ["X.npy","y.npy","events.json"]:
        old = manifest["artifacts"][name]
        if old["path"] != str(paths[name]) or sha_file(paths[name]) != (old["sha256"],old["bytes"]):
            raise PreparationError("native_adapter_output_artifact_changed")
    x,y = np.load(paths["X.npy"],mmap_mode="r",allow_pickle=False),np.load(paths["y.npy"],mmap_mode="r",allow_pickle=False)
    events = json.loads(paths["events.json"].read_bytes())
    n = native["header"]["retained_trials"]
    if (x.dtype != np.float32 or x.shape != (n,3,128) or not np.isfinite(x).all()
            or y.dtype != np.int64 or y.shape != (n,) or not np.isin(y,[0,1]).all()
            or len(events) != n or len(native["events"]) != n or manifest["retained_trials"] != n):
        raise PreparationError("retained_window_arrays_or_native_denominator_mismatch")
    uids = set()
    for i,(event,original) in enumerate(zip(events,native["events"])):
        identity = {"edf_sha256":raw_identity[0]["sha256"],"events_sha256":raw_identity[1]["sha256"],
            "retained_trial_index":i,"stage":ma2022.STAGE}
        uid = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if (any(event.get(k) != v for k,v in original.items()) or event.get("trial_uid") != uid
                or event.get("event_uid") != uid or uid in uids or event.get("label") != int(y[i])
                or event.get("role") != ROLE or event.get("original_paid_B20_prefix_recoverable") is not False
                or event.get("processed_signal_sha256") != hashlib.sha256(np.asarray(x[i],dtype="<f4").tobytes()).hexdigest()):
            raise PreparationError("retained_native_label_UID_bounds_or_array_binding_changed")
        uids.add(uid)
    return n,paths


def _complete(root,state,plan,record):
    if state.get("status") != "COMPLETE":return False
    if not all(k in state for k in ("raw_backup_receipt","raw_identity","artifacts","artifact_roles",
                                    "cache_backup_receipt","native_metadata","retained_trials")):
        raise PreparationError("completed_state_missing_dual_verification_evidence")
    if state.get("plan_sha256") != plan["plan_sha256"] or state.get("record_id") != record["record_id"]:
        raise PreparationError("record_state_plan_or_identity_mismatch")
    actual = _raw_receipt(root,state,record)
    files = [(checked(root,x),state["artifact_roles"][x["path"]]) for x in state["artifacts"]]
    _backup_set(root,state["cache_backup_receipt"],files)
    native = json.loads(checked(root,state["native_metadata"]).read_bytes())
    n,_ = validate_cache(root,record,actual,native,local(root,record["output"]),plan)
    if n != state["retained_trials"]:
        raise PreparationError("completed_record_retained_count_changed")
    return True


def _ownership(plan,record,item):
    return {"created_by":CREATOR,"plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],
        "path":item["path"],"provider_md5":item["provider_md5"],"bytes":item["bytes"]}


def cleanup(root,state,plan,record,audit):
    if not _complete(root,state,plan,record):
        raise PreparationError("both_complete_raw_and_cache_proofs_required_for_cleanup")
    actual = {x["path"]:x for x in state["raw_identity"]}
    for item in record["raw_files"]:
        if item["path"] not in state.get("owned_raw",[]):continue
        marker = audit/(item["name"]+".ownership.json")
        if json.loads(marker.read_bytes()) != _ownership(plan,record,item):
            raise PreparationError("raw_ownership_invalid_preserve_file")
        path = local(root,item["path"])
        if path.exists():
            native_identity(root,item,expected_sha=actual[item["path"]]["sha256"])
            path.unlink()


def process_record(root,plan,record,*,ledger,session,backup_fn,restore_fn,create_cache):
    audit = local(root,f"audit/source_extensions_20261019/{DATASET}/{plan['plan_sha256']}/{record['record_id']}")
    audit.mkdir(parents=True,exist_ok=True)
    path = audit/"state.json"
    state = json.loads(path.read_bytes()) if path.exists() else {"status":"PENDING",
        "plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],"owned_raw":[]}
    if state.get("plan_sha256") != plan["plan_sha256"] or state.get("record_id") != record["record_id"]:
        raise PreparationError("existing_record_state_binding_mismatch")
    if _complete(root,state,plan,record):
        cleanup(root,state,plan,record,audit)
        return state
    try:
        needed = sum(x["bytes"] for x in record["raw_files"] if not local(root,x["path"]).exists())
        if shutil.disk_usage(root).free < needed+plan["budgets"]["disk_reserve_bytes"]:
            raise PreparationError("insufficient_disk_for_one_native_pair")
        previous = {x["path"]:x for x in state.get("raw_identity",[])}
        if state.get("raw_backup_receipt"):
            previous = {x["path"]:x for x in _raw_receipt(root,state,record)}
            missing = [x for x in record["raw_files"] if not local(root,x["path"]).exists()]
            if missing:
                for item in missing:
                    atomic_json(audit/(item["name"]+".ownership.json"),_ownership(plan,record,item),immutable=True)
                    if item["path"] not in state["owned_raw"]:state["owned_raw"].append(item["path"])
                atomic_json(path,state)
                reserve = sum(x["bytes"] for x in missing)
                ledger.reserve("r2_get_body_bytes",reserve,record["record_id"]+":raw_restore")
                try:
                    restored = restore_fn(backup_receipt_path=checked(root,state["raw_backup_receipt"]),root=root,
                        report_path=audit/"raw_restore_report.json",paths=[x["path"] for x in missing],execute=True,require_fresh=True)
                    if restored.get("status") != "restored_verified" or restored.get("fresh_full_get_verified") is not True:
                        raise PreparationError("raw_restore_fresh_full_GET_failed")
                    ledger.finish(verified_bytes=reserve)
                except BaseException:
                    ledger.finish();raise
        identities = []
        for item in record["raw_files"]:
            raw = local(root,item["path"])
            marker = audit/(item["name"]+".ownership.json")
            if raw.exists():
                if marker.exists():
                    if json.loads(marker.read_bytes()) != _ownership(plan,record,item):
                        raise PreparationError("existing_raw_ownership_marker_mismatch")
                    if item["path"] not in state["owned_raw"]:state["owned_raw"].append(item["path"])
                # No marker: earlier pilot stays unowned, even after reuse.
            else:
                if raw.with_name(raw.name+".owned.partial").exists() and not marker.exists():
                    raise PreparationError("unowned_partial_native_preserved")
                atomic_json(marker,_ownership(plan,record,item),immutable=True)
                if item["path"] not in state["owned_raw"]:state["owned_raw"].append(item["path"])
                atomic_json(path,state)
                download_object(item,raw,session=session,ledger=ledger)
            identities.append(native_identity(root,item,expected_sha=previous.get(item["path"],{}).get("sha256")))
        state["raw_identity"] = identities
        atomic_json(path,state)
        raw_files = [(local(root,x["path"]),"source_development_raw") for x in identities]
        state["raw_backup_receipt"] = backup_files(root,audit/"raw_backup",raw_files,
            prefix=f"paper3/ma2022-development/{plan['plan_sha256']}/{record['record_id']}/raw/",ledger=ledger,backup_fn=backup_fn)
        atomic_json(path,state)
        _raw_receipt(root,state,record)
        edf,events = [local(root,x["path"]) for x in identities]
        native = ma2022.audit_pair(edf,events,edf_sha256=identities[0]["sha256"],events_sha256=identities[1]["sha256"])
        native_path = audit/"native_metadata.json"
        atomic_json(native_path,native,immutable=True)
        unit_path = audit/"unit_evidence.json"
        atomic_json(unit_path,{"dataset":DATASET,"role":ROLE,"native_metadata":descriptor(root,native_path),
            "selected_calibration":native["header"]["selected_calibration"],"native_unit":"uV","stored_unit":"V",
            "conversion_to_V":1e-6,"reference_evidence":native["reference_evidence"],"original_B20_prefix_recoverable":False},immutable=True)
        output = local(root,record["output"])
        cache_marker = audit/"cache_ownership.json"
        binding = {"created_by":CREATOR,"plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],"output":record["output"]}
        if output.exists() and not (output/"manifest.json").exists():
            if not cache_marker.exists() or json.loads(cache_marker.read_bytes()) != binding:
                raise PreparationError("unowned_incomplete_cache_preserved")
            quarantine = output.with_name(output.name+".incomplete-"+digest(sorted(str(p.relative_to(output)) for p in output.rglob('*')))[:12])
            if quarantine.exists():raise PreparationError("prior_incomplete_cache_already_preserved")
            output.rename(quarantine)
        if not output.exists():
            atomic_json(cache_marker,binding,immutable=True)
            create_cache(edf,events,output,edf_sha256=identities[0]["sha256"],events_sha256=identities[1]["sha256"])
        n,paths = validate_cache(root,record,identities,native,output,plan)
        evidence_path = audit/"cache_evidence.json"
        atomic_json(evidence_path,{"dataset":DATASET,"role":ROLE,"training_ready":False,"actual_fits":0,
            "raw_identity":identities,"raw_backup_receipt":state["raw_backup_receipt"],
            "native_metadata":descriptor(root,native_path),"unit_evidence":descriptor(root,unit_path),
            "retained_trials":n,"original_B20_prefix_recoverable":False,
            "source3_training_pool_member":False,"scientific_pool_assignment_frozen":False},immutable=True)
        files = [(p,"result") for p in paths.values()]+[(p,"audit") for p in [native_path,unit_path,evidence_path]]
        artifacts = [descriptor(root,p) for p,_ in files]
        state.update(status="CACHE_LOCAL_VERIFIED",artifacts=artifacts,
            artifact_roles={str(p.relative_to(root)):role for p,role in files},native_metadata=descriptor(root,native_path),
            unit_evidence=descriptor(root,unit_path),retained_trials=n)
        atomic_json(path,state)
        receipt = backup_files(root,audit/"cache_backup",files,
            prefix=f"paper3/ma2022-development/{plan['plan_sha256']}/{record['record_id']}/cache/",ledger=ledger,backup_fn=backup_fn)
        state.update(status="COMPLETE",cache_backup_receipt=receipt)
        atomic_json(path,state)
        cleanup(root,state,plan,record,audit)
        return state
    except BaseException as error:
        ledger.finish()
        state.update(status="FAILED_CLOSED",error_type=type(error).__name__,
            error_code=str(error) if isinstance(error,PreparationError) else "operation_failed_details_suppressed")
        atomic_json(path,state)
        raise PreparationError(state["error_code"]) from None


def prepare(root,plan,*,execute=False,max_new_records=None,session=None,backup_fn=None,restore_fn=None,create_cache=None):
    root = Path(root).absolute()
    validate_plan(root,plan)
    if max_new_records is not None and (type(max_new_records) is not int or max_new_records < 1):
        raise PreparationError("positive_new_record_limit_required")
    if not execute:return {"status":"DRY_RUN_NO_NETWORK_R2_OR_FIT","dataset":DATASET,"role":ROLE,
        "plan_sha256":plan["plan_sha256"],"records":125,"native_objects":250,"native_bytes":RAW_BYTES,
        "training_ready":False,"original_B20_prefix_recoverable":False,"GPU_actions":0,"actual_fits":0}
    shared = local(root,f"audit/source_extensions_20261019/{DATASET}")
    shared.mkdir(parents=True,exist_ok=True)
    lock_path = local(root,str((shared/".development-singleton.lock").relative_to(root)))
    with lock_path.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise PreparationError("Ma2022_development_coordinator_already_running") from None
        audit = shared/plan["plan_sha256"]
        audit.mkdir(parents=True,exist_ok=True)
        atomic_json(audit/"frozen_plan.json",plan,immutable=True)
        ledger = Ledger(audit/"budget_ledger.json",plan)
        if session is None:
            import requests
            session = requests.Session()
        backup_fn,restore_fn = backup_fn or storage.run_backup,restore_fn or storage.run_restore
        create_cache = create_cache or ma2022.create_development_cache
        states,new = [],0
        try:
            for record in plan["records"]:
                ledger.checkpoint()
                p = audit/record["record_id"]/"state.json"
                old = json.loads(p.read_bytes()) if p.exists() else {}
                already = old.get("status") == "COMPLETE"
                if not already and max_new_records is not None and new >= max_new_records:break
                state = process_record(root,plan,record,ledger=ledger,session=session,backup_fn=backup_fn,
                    restore_fn=restore_fn,create_cache=create_cache)
                states.append(state);new+=int(not already)
                if sum(x["bytes"] for s in states for x in s["artifacts"]) > plan["budgets"]["cache_bytes"]:
                    raise PreparationError("cumulative_development_cache_byte_budget_exhausted")
            if len(states) != 125:
                result = {"status":"PAUSED_EXPLICIT_NEW_RECORD_LIMIT","verified_records":len(states),"total_records":125}
            else:
                seen = set()
                for state,record in zip(states,plan["records"]):
                    if not _complete(root,state,plan,record):raise PreparationError("all125_dual_verified_pairs_required")
                    for event in json.loads((local(root,record["output"])/"events.json").read_bytes()):
                        if event["trial_uid"] in seen:raise PreparationError("cross_session_retained_UID_duplicate")
                        seen.add(event["trial_uid"])
                inventory = {"dataset":DATASET,"records":[{"record_id":s["record_id"],"raw_identity":s["raw_identity"],
                    "native_metadata":s["native_metadata"],"unit_evidence":s["unit_evidence"],
                    "raw_backup_receipt":s["raw_backup_receipt"],"cache_backup_receipt":s["cache_backup_receipt"]} for s in states]}
                manifest = {"schema_version":"paper3-Ma2022-development-cache-v1","status":"ALL125_DEVELOPMENT_PAIRS_VERIFIED",
                    "dataset":DATASET,"role":ROLE,"training_ready":False,"source_training_allowed":False,
                    "source3_training_pool_member":False,"primary11_compatible":False,"original_B20_prefix_recoverable":False,
                    "scientific_pool_assignment_frozen":False,"actual_fits":0,"GPU_actions":0,"normalization_applied":False,
                    "provider_participant_ids":25,"sessions":125,"original_raw_objects":250,"original_raw_bytes":RAW_BYTES,
                    "retained_trials":sum(s["retained_trials"] for s in states),"biological_independence_verified":False,
                    "plan_sha256":plan["plan_sha256"],"document_sha256":storage.DOCUMENT_SHA256,"catalog":plan["catalog"],
                    "stage":ma2022.STAGE,"channels":["C3","Cz","C4"],"shards":[{"record_id":s["record_id"],
                        "retained_trials":s["retained_trials"],"artifacts":s["artifacts"],
                        "raw_backup_receipt":s["raw_backup_receipt"],"cache_backup_receipt":s["cache_backup_receipt"]} for s in states]}
                inv_path = audit/"native_inventory.json"
                atomic_json(inv_path,inventory,immutable=True)
                manifest["native_inventory"] = descriptor(root,inv_path)
                manifest_path = local(root,f"{storage.RESULT_ROOT}/{DATASET}/development_cache_native_v2.json")
                atomic_json(manifest_path,manifest,immutable=True)
                receipt = backup_files(root,audit/"final_backup",[(manifest_path,"result"),(inv_path,"audit"),(audit/"frozen_plan.json","audit")],
                    prefix=f"paper3/ma2022-development/{plan['plan_sha256']}/final/",ledger=ledger,backup_fn=backup_fn)
                result = {"status":"MA2022_DEVELOPMENT_CACHE_COMPLETE","verified_records":125,
                    "manifest":descriptor(root,manifest_path),"backup_receipt":receipt,"retained_trials":manifest["retained_trials"]}
            result.update(dataset=DATASET,role=ROLE,training_ready=False,original_B20_prefix_recoverable=False,
                plan_sha256=plan["plan_sha256"],actual_fits=0,GPU_actions=0,external_evaluation=False)
            atomic_json(audit/"coordinator_status.json",result)
            return result
        finally:ledger.checkpoint(close=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--catalog",type=Path,default=Path(CATALOG))
    parser.add_argument("--plan",type=Path,default=Path(DEFAULT_PLAN))
    parser.add_argument("--max-provider-body-bytes",type=int)
    parser.add_argument("--max-r2-get-body-bytes",type=int)
    parser.add_argument("--max-wall-seconds",type=int,default=7200)
    parser.add_argument("--max-cache-bytes",type=int,default=512*1024**2)
    parser.add_argument("--disk-reserve-bytes",type=int,default=1024**3)
    parser.add_argument("--max-new-records",type=int)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    try:
        root=args.root.absolute()
        plan_path=args.plan if args.plan.is_absolute() else root/args.plan
        local(root,plan_path.relative_to(root).as_posix())
        if plan_path.exists():plan=json.loads(plan_path.read_bytes())
        else:
            catalog=args.catalog if args.catalog.is_absolute() else root/args.catalog
            plan=build_plan(root,catalog,max_provider_body_bytes=args.max_provider_body_bytes,
                max_r2_get_body_bytes=args.max_r2_get_body_bytes,max_wall_seconds=args.max_wall_seconds,
                max_cache_bytes=args.max_cache_bytes,disk_reserve_bytes=args.disk_reserve_bytes)
            atomic_json(plan_path,plan,immutable=True)
        result=prepare(root,plan,execute=args.execute,max_new_records=args.max_new_records)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status":"FAILED_CLOSED","error_type":type(error).__name__,
            "error_code":str(error) if isinstance(error,PreparationError) else "operation_failed_details_suppressed",
            "training_ready":False,"actual_fits":0,"GPU_actions":0}))
        return 1


if __name__ == "__main__":raise SystemExit(main())
