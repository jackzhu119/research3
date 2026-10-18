#!/usr/bin/env python3
"""Local CPU W2 source cache preparation; dry-run by default and W1-gated.

No downloads, uploads, GPU calls, or status mutations. --execute materializes
new NPY participant shards only after W1 COMPLETE and a frozen input contract.
--partial-scope can prepare an explicitly partial engineering cache after W1;
it always emits PILOT_CACHE_PARTIAL and cannot satisfy the formal W2/pilot gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3work.adapters.source_native import iter_native_contexts
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest, validate_signal_contract
from p3work.preprocess import prepare_trial
from tools.r2_backup import write_receipt
from tools.r2_restore import load_verified_receipt, no_symlink_components, run_restore


class SourceCacheGateError(ValueError):
    pass


def expected_source_objects():
    """Protocol-required download targets, NOT assertions that objects exist.

    Formal completion requires an actual complete receipt and native schema for
    every target. Their URLs follow the retained fixed provider adapters.
    """
    targets = {}
    for subject in range(1, 10):
        for session in ("T", "E"):
            name = f"A{subject:02d}{session}.mat"
            targets["BNCI2014_001", str(subject), session, name] = (
                "https://bnci-horizon-2020.eu/database/data-sets/001-2014/" + name)
    giga = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/"
    for subject in range(1, 53):
        name = f"s{subject:02d}.mat"
        targets["Cho2017", str(subject), "provider_single_visit", name] = giga + "100295/mat_data/" + name
    for subject in range(1, 55):
        for session in (1, 2):
            name = f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
            targets["Lee2019_MI", str(subject), str(session), name] = (
                giga + f"100542/session{session}/s{subject}/" + name)
    for subject in range(1, 110):
        for run in (4, 8, 12):
            name = f"S{subject:03d}R{run:02d}.edf"
            targets["PhysionetMI", str(subject), "provider_single_visit", name] = (
                f"https://physionet.org/files/eegmmidb/1.0.0/S{subject:03d}/" + name)
    return targets


def _local(root, relative, *, must_exist=True):
    if not isinstance(relative, str) or not relative:
        raise SourceCacheGateError("Explicit root-relative artifact path required")
    relative = Path(relative)
    if (relative.is_absolute() or ".." in relative.parts or "legacy_baseline_v1" in relative.parts or
            relative.suffix.lower() in {".npz", ".pkl", ".pickle"}):
        raise SourceCacheGateError("Old NPZ, pickle, legacy code, and escaping paths are forbidden")
    path = (root / relative).resolve(strict=must_exist)
    if not path.is_relative_to(root):
        raise SourceCacheGateError("Artifact leaves the independent Paper3 root")
    return path


def _artifact(root, value):
    if not isinstance(value, dict) or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None:
        raise SourceCacheGateError("An artifact needs an explicit path and SHA256")
    path = _local(root, value.get("path"))
    if not path.is_file() or file_digest(path) != value["sha256"]:
        raise SourceCacheGateError("Artifact does not match its complete SHA256 receipt")
    return path


def _remote_raw_receipt(root, item):
    if (item.get("raw_storage") != "verified_r2_only" or
            item.get("temporary_raw_owned") is not True):
        raise SourceCacheGateError("Missing raw needs a verified tool-owned R2-only staging receipt")
    path = _local(root, item["path"], must_exist=False)
    staging = root / "data/source_preparation/staging"
    if not path.is_relative_to(staging) or path.name != item["file"]:
        raise SourceCacheGateError("R2 restore is restricted to source worker staging ownership")
    receipt = _artifact(root, item.get("backup_receipt"))
    entries, _ = load_verified_receipt(receipt)
    matches = [row for row in entries if row["path"] == item["path"]]
    if (len(matches) != 1 or matches[0]["sha256"] != item["sha256"] or matches[0]["bytes"] != item["bytes"] or
            matches[0]["dataset"] != item["dataset"] or matches[0]["role"] != "source_development_raw"):
        raise SourceCacheGateError("R2 backup receipt does not prove this exact registered native raw")
    return receipt


def _cache_restore_marker(root, item):
    """Canonical per-object ownership proof; separate from a content checksum."""
    source_identity = [item["dataset"], str(item["provider_subject"]), str(item["session_id"]), item["file"]]
    object_uid = digest(source_identity)
    expected = Path("data/source_preparation/staging") / object_uid / item["file"]
    if (item.get("object_uid", object_uid) != object_uid or item.get("path") != expected.as_posix() or
            item.get("temporary_raw_owned") is not True or item.get("raw_storage") != "verified_r2_only"):
        raise SourceCacheGateError("Cache restore ownership needs exact source-worker staging identity")
    path = root / expected
    marker = root / "audit/source_preparation/cache_restores" / object_uid / "ownership.json"
    no_symlink_components(path)
    no_symlink_components(marker)
    binding = {"source_identity": source_identity, "object_uid": object_uid,
               "raw": {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
                       "dataset": item["dataset"], "role": "source_development_raw"},
               "backup_receipt": {key: item["backup_receipt"][key] for key in ("path", "sha256")}}
    return path, marker, binding


def begin_owned_cache_restore(root, item):
    """Persist ownership BEFORE R2 can atomically commit a new local raw file.

    A kill between restore commit and the context manager's finally block can
    then be recovered without claiming ownership of pre-existing local files.
    """
    _remote_raw_receipt(root, item)
    path, marker, binding = _cache_restore_marker(root, item)
    if path.exists():
        raise SourceCacheGateError("Pre-existing raw cannot be adopted as a newly owned cache restore")
    record = {"schema_version": "paper3-owned-cache-raw-restore-v1", "created_by": "tools/prepare_source_cache.py",
              "created_when_raw_absent": True, "status": "RESTORE_IN_PROGRESS", **binding}
    write_receipt(marker, record)
    return marker


def _verify_active_cache_restore(root, item):
    _remote_raw_receipt(root, item)
    path, marker, binding = _cache_restore_marker(root, item)
    if not marker.is_file():
        raise SourceCacheGateError("Reappeared staging raw has no durable cache restore ownership proof")
    record = json.loads(marker.read_text())
    if (record.get("schema_version") != "paper3-owned-cache-raw-restore-v1" or
            record.get("created_by") != "tools/prepare_source_cache.py" or
            record.get("created_when_raw_absent") is not True or
            record.get("status") not in {"RESTORE_IN_PROGRESS", "RESTORED_VERIFIED"} or
            any(record.get(key) != value for key, value in binding.items())):
        raise SourceCacheGateError("Reappeared raw ownership proof differs from the verified source restore")
    if not path.is_file() or file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
        raise SourceCacheGateError("Owned cache restore raw integrity changed; preserve it for investigation")
    return path, marker, record


def recover_owned_cache_restore(root, item):
    """Remove an interrupted tool restore only after ownership + original R2 proof.

    The raw worker calls this before replaying an already archived object. A
    checksum match alone never authorizes deleting an unrelated existing file.
    """
    path, marker, _ = _cache_restore_marker(root, item)
    if not path.exists():
        return False
    path, marker, record = _verify_active_cache_restore(root, item)
    path.unlink()
    record["status"] = "RAW_REMOVED_AFTER_VERIFIED_BACKUP"
    record["removal_evidence"] = "active durable cache-restore ownership; exact original full-GET backup receipt; complete local SHA256 and bytes"
    write_receipt(marker, record)
    return True


@contextmanager
def _available_raw_context(root, item, path, *, restore_backed_raw=False):
    """Restore just one approved R2 raw object, then remove only our own restore."""
    created = False
    if path.exists() and item.get("raw_storage") == "verified_r2_only":
        _, marker, _ = _cache_restore_marker(root, item)
        if marker.exists():
            _verify_active_cache_restore(root, item)
            created = True
    if not path.exists():
        if not restore_backed_raw:
            raise SourceCacheGateError("R2-only raw requires explicit --restore-backed-raw for numerical preparation")
        receipt = _remote_raw_receipt(root, item)
        marker = begin_owned_cache_restore(root, item)
        report = marker.with_name("restore_receipt.json")
        result = run_restore(backup_receipt_path=receipt, root=root, report_path=report,
                             paths=[item["path"]], execute=True)
        if result.get("status") != "restored_verified":
            raise SourceCacheGateError("Approved native R2 full GET restore did not verify")
        if not path.is_file() or file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise SourceCacheGateError("Restored native raw differs from the verified backup")
        record = json.loads(marker.read_text())
        record["status"] = "RESTORED_VERIFIED"
        if report.is_file():
            record["restore_receipt"] = {"path": str(report.relative_to(root)), "sha256": file_digest(report)}
        write_receipt(marker, record)
        created = True
    try:
        yield path
    finally:
        if created:
            recover_owned_cache_restore(root, item)


@contextmanager
def _available_raw(root, item, path, *, restore_backed_raw=False):
    # Use the same process lock as the raw worker. Recovery must never remove
    # an active cache restore; a kill releases flock before the replay begins.
    lock_path = root / "audit/source_preparation/.source_worker.lock"
    no_symlink_components(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceCacheGateError("Another source staging operation is active; preserve its raw") from None
        with _available_raw_context(root, item, path, restore_backed_raw=restore_backed_raw) as available:
            yield available


def import_verified_specimen_inventory(root, *, output="manifests/source_native_inventory.partial.json"):
    """Metadata-only import of measured specimen receipts; never a complete roster.

    This is permitted before W1. It rechecks complete raw SHA256, retained GET
    byte receipts, and native metadata hashes. Required but unseen URLs are not
    emitted as observed objects or represented as already downloaded.
    """
    root = Path(root).resolve(strict=True)
    raw_receipt = root / "audit/source_schema/raw_inventory.json"
    schema_receipt = root / "audit/source_schema/native_schema_audit.json"
    raw = json.loads(raw_receipt.read_text())
    schema = json.loads(schema_receipt.read_text())
    if raw.get("status") != "downloaded_schema_specimens_only" or schema.get("all_source_cohorts_complete") is not False:
        raise SourceCacheGateError("This importer is restricted to explicitly partial schema specimens")
    summaries = {row["dataset"]: row for row in schema["datasets"]}
    expected, objects = expected_source_objects(), []
    descriptor = lambda path: {"path": str(path.relative_to(root)), "sha256": file_digest(path)}
    for item in raw["files"]:
        identity = (item["dataset"], str(item["provider_subject"]), str(item["session_id"]), item["file"])
        if identity not in expected or item["url"] != expected[identity]:
            raise SourceCacheGateError("Schema specimen URL is not an exact registered source object")
        native_path = _local(root, item["project_relative_path"])
        if (file_digest(native_path) != item["sha256"] or native_path.stat().st_size != item["bytes"] or
                item.get("http_status") != 200 or
                str(item["response_headers"].get("Content-Length")) != str(item["bytes"])):
            raise SourceCacheGateError("Schema specimen lacks complete actual GET/byte/SHA evidence")
        summary = summaries[item["dataset"]]
        if (summary["source_file_sha256"] != item["sha256"] or
                summary["adapter_file_sha256"] != file_digest(root / "p3work/adapters/source_native.py")):
            raise SourceCacheGateError("Native schema is stale relative to the raw file or current adapter")
        metadata = root / "audit/source_schema" / summary["native_trial_metadata_file"]
        if file_digest(metadata) != summary["native_trial_metadata_sha256"]:
            raise SourceCacheGateError("Native trial metadata SHA changed")
        objects.append({"dataset": item["dataset"], "provider_subject": item["provider_subject"],
            "session_id": item["session_id"], "file": item["file"], "version": item["version"],
            **descriptor(native_path), "bytes": item["bytes"], "url": item["url"],
            "complete_download": True, "http_status": item["http_status"],
            "response_content_length": item["response_headers"]["Content-Length"],
            "native_schema_audited": True, "native_trials": descriptor(metadata)})
    evidence = root / "audit/source_schema/evidence_manifest.json"
    for item in json.loads(evidence.read_text())["files"]:
        _artifact(root, item)
    record = {"schema_version": "paper3-source-native-inventory-v2", "status": "SOURCE_RAW_INVENTORY_PARTIAL",
              "role": "source_train", "synthetic": False, "old_npz_reused": False,
              "provider_roster_verified": False, "scope": "four_registered_schema_specimens_only",
              "required_target_object_count": len(expected), "actual_verified_objects": len(objects),
              "required_objects_not_yet_verified": len(expected) - len(objects),
              "unverified_target_urls_emitted_as_objects": False,
              "raw_get_receipt": descriptor(raw_receipt), "native_schema_receipt": descriptor(schema_receipt),
              "provider_terms_evidence": descriptor(evidence), "files": objects,
              "numerical_cache_created": False, "training_started": False}
    target = _local(root, output, must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return {"status": record["status"], "inventory": descriptor(target),
            "actual_verified_objects": len(objects), "required_target_object_count": len(expected),
            "numerical_cache_created": False, "training_started": False}


def verify_preparation(root, inventory_path, *, partial_scope=False):
    """Fail before raw EEG access when W1/window/full-inventory gates are unmet."""
    root = Path(root).resolve(strict=True)
    tasks = json.loads((root / "status/tasks.json").read_text())
    if tasks.get("W1", {}).get("status") != "COMPLETE":
        raise SourceCacheGateError("W1 must be COMPLETE before numerical source cache preparation")
    contract = json.loads((root / "contracts/input_contract.json").read_text())
    contract_sha = validate_signal_contract(contract)
    manifest_path = _local(root, inventory_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != "paper3-source-native-inventory-v2" or
            manifest.get("role") != "source_train" or manifest.get("synthetic") is not False or
            manifest.get("old_npz_reused") is not False):
        raise SourceCacheGateError("A new real native source inventory is required")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise SourceCacheGateError("No actual verified native source objects")
    expected = expected_source_objects()
    identities = [(row.get("dataset"), str(row.get("provider_subject")), str(row.get("session_id")), row.get("file"))
                  for row in files]
    if len(set(identities)) != len(identities) or not set(identities) <= set(expected):
        raise SourceCacheGateError("Unregistered or duplicated source object identity")
    complete = (manifest.get("status") == "SOURCE_RAW_INVENTORY_COMPLETE" and
                manifest.get("provider_roster_verified") is True and set(identities) == set(expected))
    if not partial_scope and not complete:
        raise SourceCacheGateError(f"Full source inventory incomplete: {len(identities)}/{len(expected)} verified target objects; partial specimens cannot complete W2")
    if partial_scope and manifest.get("status") not in {"SOURCE_RAW_INVENTORY_PARTIAL", "SOURCE_RAW_INVENTORY_COMPLETE"}:
        raise SourceCacheGateError("Partial scope still requires explicit measured native inventory status")
    _artifact(root, manifest.get("provider_terms_evidence"))
    prepared_files = []
    for item, identity in zip(files, identities):
        if (item.get("url") != expected[identity] or item.get("complete_download") is not True or
                item.get("http_status") != 200 or item.get("native_schema_audited") is not True):
            raise SourceCacheGateError("Every source object needs official URL, complete GET receipt, and actual native schema audit")
        raw = _local(root, item.get("path"), must_exist=False)
        if raw.exists():
            _artifact(root, item)
            actual_bytes = raw.stat().st_size
        else:
            _remote_raw_receipt(root, item)
            actual_bytes = item.get("bytes")
        if (raw.name != item["file"] or type(item.get("bytes")) is not int or
                item["bytes"] != actual_bytes or
                str(item.get("response_content_length")) != str(item["bytes"])):
            raise SourceCacheGateError("Native source complete byte size disagrees with provider response")
        metadata = _artifact(root, item.get("native_trials"))
        rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
        if not rows or any(row.get("dataset") != item["dataset"] or row.get("role") != "source" or
                           row.get("source_file_sha256") != item["sha256"] or
                           str(row.get("provider_subject")) != str(item["provider_subject"]) or
                           str(row.get("session_id")) != str(item["session_id"]) or
                           row.get("source_file_bytes") != item["bytes"] or
                           row.get("source_version") != item.get("version")
                           for row in rows):
            raise SourceCacheGateError("Native metadata differs from verified source object identity")
        prepared_files.append((item, raw, rows))
    return root, contract, contract_sha, manifest_path, manifest, prepared_files, complete


def _prepare_shard(root, destination, temporary, *, dataset, subject, records,
                   contract, contract_sha, adapter, processing, restore_backed_raw):
    x, labels, events, exclusions = [], [], [], []
    for item, raw, rows in records:
        with _available_raw(root, item, raw, restore_backed_raw=restore_backed_raw) as available:
            for native in iter_native_contexts(dataset, available, rows=rows, contract=contract, expected_sha256=item["sha256"]):
                row = native["row"]
                if native.get("excluded"):
                    exclusions.append({"event_uid": row["trial_uid"], "reason": native["excluded"], "original_metadata": row})
                    continue
                try:
                    signal, qc = prepare_trial(native["signal"], channel_names=native["channel_names"],
                        sfreq_hz=native["sfreq_hz"], task_sample=native["task_sample"],
                        task_end_sample=native["task_end_sample"], task_end_verified=native["task_end_verified"],
                        contract=contract, native_unit=native["native_unit"])
                except ValueError as error:
                    if str(error) not in {"Nonfinite context", "Zero/nonfinite trial energy"}:
                        raise
                    exclusions.append({"event_uid": row["trial_uid"], "reason": str(error), "original_metadata": row})
                    continue
                label = {"left_hand": 0, "right_hand": 1}[row["canonical_label"]]
                stored_signal = signal.astype(np.float32)
                x.append(stored_signal)
                labels.append(label)
                events.append({**row, "event_uid": row["trial_uid"], "dataset": dataset, "subject_id": subject,
                    "original_trial_id": row["original_trial_index"], "label": label, "role": "source_train",
                    "input_contract_sha256": contract_sha, "raw_sha256": item["sha256"],
                    "adapter_sha256": adapter["sha256"], "processing_sha256": processing["sha256"],
                    "effective_numeric_unit": native["native_unit"], "unit_evidence_kind": native["unit_evidence_kind"],
                    "native_context_sha256": hashlib.sha256(np.asarray(native["signal"], dtype="<f8").tobytes(order="C")).hexdigest(),
                    "processed_signal_sha256": hashlib.sha256(np.asarray(stored_signal, dtype="<f4").tobytes(order="C")).hexdigest(),
                    "original_context_samples": native["original_context_samples"], "preprocessing_qc": qc})
    (temporary / "exclusions.json").write_text(json.dumps(exclusions, indent=2, allow_nan=False) + "\n")
    final_directory = destination / dataset / subject
    artifact = lambda name: {"path": str((final_directory / name).relative_to(root)),
                             "sha256": file_digest(temporary / name)}
    if not x:
        return {"dataset": dataset, "subject_id": subject, "excluded_original_trials": len(exclusions),
                "reason": "no_valid_trials_under_frozen_contract", "exclusions": artifact("exclusions.json")}
    np.save(temporary / "X.npy", np.stack(x), allow_pickle=False)
    np.save(temporary / "y.npy", np.asarray(labels, dtype=np.int64), allow_pickle=False)
    (temporary / "events.json").write_text(json.dumps(events, indent=2, allow_nan=False) + "\n")
    return {"dataset": dataset, "subject_id": subject, "role": "source_train",
            "stage": "pre_rms_car11_temporal_demeaned", "input_contract_sha256": contract_sha,
            "adapter_code": adapter, "processing_code": processing,
            "X": artifact("X.npy"), "y": artifact("y.npy"), "events": artifact("events.json"),
            "exclusions": artifact("exclusions.json"), "original_trials_processed": len(events) + len(exclusions),
            "valid_trials": len(events), "excluded_trials": len(exclusions)}


def prepare_cache(root, inventory_path="manifests/source_native_inventory.json", *,
                  output="cache/source_v2", execute=False, partial_scope=False, restore_backed_raw=False):
    root, contract, contract_sha, inventory_file, inventory, files, complete = verify_preparation(
        root, inventory_path, partial_scope=partial_scope)
    if not execute:
        return {"status": "DRY_RUN_GATES_PASSED", "numerical_cache_created": False,
                "training_started": False, "raw_files_verified": len(files),
                "scope": "partial" if partial_scope else "complete_four_source",
                "would_emit_status": "PILOT_CACHE_PARTIAL" if partial_scope else "SOURCE_CACHE_COMPLETE",
                "input_contract_sha256": contract_sha}
    destination = _local(root, output, must_exist=False)
    if not restore_backed_raw and any(not raw.exists() for _, raw, _ in files):
        raise SourceCacheGateError("R2-only source preparation requires explicit --restore-backed-raw")
    adapter_path, processing_path = root / "p3work/adapters/source_native.py", root / "p3work/preprocess.py"
    if (adapter_path.read_bytes() != Path(sys.modules[iter_native_contexts.__module__].__file__).read_bytes() or
            processing_path.read_bytes() != Path(sys.modules[prepare_trial.__module__].__file__).read_bytes()):
        raise SourceCacheGateError("Executed adapter/processing code differs from packaged provenance")
    descriptor = lambda path: {"path": str(path.relative_to(root)), "sha256": file_digest(path)}
    adapter, processing = descriptor(adapter_path), descriptor(processing_path)
    shard_files = defaultdict(list)
    seen = set()
    for item, raw, rows in files:
        for row in rows:
            if not isinstance(row.get("trial_uid"), str) or row["trial_uid"] in seen:
                raise SourceCacheGateError("Repeated or missing original source trial UID")
            seen.add(row["trial_uid"])
        shard_files[item["dataset"], str(item["provider_subject"])].append((item, raw, rows))
    identity = {"input_contract_sha256": contract_sha, "inventory": descriptor(inventory_file),
                "adapter_code": adapter, "processing_code": processing, "partial_scope": partial_scope,
                "cache_builder_sha256": file_digest(__file__)}
    state_path = destination / "build_state.json"
    if destination.exists():
        if not state_path.is_file():
            raise SourceCacheGateError("Existing directory is not an owned resumable source cache build")
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise SourceCacheGateError("Existing source cache code/data/input identity differs; use a new output")
    else:
        destination.mkdir(parents=True)
        state = {"status": "IN_PROGRESS", "identity": identity, "completed_participants": {},
                 "failed_participants": {}, "training_started": False}
        write_receipt(state_path, state)
    manifest = {"schema_version": "paper3-source-cache-v2", "status": "PILOT_CACHE_PARTIAL" if partial_scope else "SOURCE_CACHE_COMPLETE",
                "synthetic": False, "old_npz_reused": False, "role": "source_train",
                "input_contract_sha256": contract_sha, "stage": "pre_rms_car11_temporal_demeaned",
                "full_source_inventory_verified": complete and not partial_scope,
                "raw_inventory": descriptor(inventory_file), "shards": [], "participants_without_usable_trials": [],
                "training_started": False, "normalization_applied": False}
    for (dataset, subject), records in sorted(shard_files.items()):
        key = dataset + "/" + subject
        if key in state["completed_participants"]:
            shard = state["completed_participants"][key]
            for field in (("X", "y", "events", "exclusions") if "X" in shard else ("exclusions",)):
                _artifact(root, shard[field])
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".participant-building-", dir=destination))
            try:
                shard = _prepare_shard(root, destination, temporary, dataset=dataset, subject=subject,
                    records=records, contract=contract, contract_sha=contract_sha, adapter=adapter,
                    processing=processing, restore_backed_raw=restore_backed_raw)
                target = destination / dataset / subject
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise SourceCacheGateError("Uncheckpointed participant output exists; preserve for inspection")
                temporary.rename(target)
                state["completed_participants"][key] = shard
                state["failed_participants"].pop(key, None)
            except Exception as error:
                shutil.rmtree(temporary, ignore_errors=True)
                state["failed_participants"][key] = {"error_type": type(error).__name__,
                    "error_code": "source_participant_preparation_failed_details_suppressed"}
                write_receipt(state_path, state)
                continue
            write_receipt(state_path, state)
        manifest["shards" if "X" in shard else "participants_without_usable_trials"].append(shard)
    if state["failed_participants"] or not manifest["shards"] or (not partial_scope and {s["dataset"] for s in manifest["shards"]} != SOURCE_IDS):
        state["status"] = "SOURCE_CACHE_INCOMPLETE"
        write_receipt(state_path, state)
        return {"status": "SOURCE_CACHE_INCOMPLETE", "numerical_cache_created": bool(state["completed_participants"]),
                "formal_w2_complete": False, "training_started": False,
                "completed_participants": len(state["completed_participants"]),
                "failed_participants": state["failed_participants"], "status_tasks_mutated": False}
    manifest["original_trials_processed"] = len(seen)
    signal_groups = defaultdict(list)
    for shard in manifest["shards"]:
        for event in json.loads((root / shard["events"]["path"]).read_text()):
            signal_groups[event["processed_signal_sha256"]].append(event["event_uid"])
    manifest["duplicate_processed_signal_groups"] = [
        {"processed_signal_sha256": key, "event_uids": sorted(values)}
        for key, values in sorted(signal_groups.items()) if len(values) > 1]
    manifest["duplicate_policy"] = "retain original identities; use signal groups to audit future train/validation splits; no silent deduplication"
    write_receipt(destination / "source_cache.json", manifest)
    state.update(status=manifest["status"], cache_manifest=descriptor(destination / "source_cache.json"))
    write_receipt(state_path, state)
    result = {"status": manifest["status"], "numerical_cache_created": True, "training_started": False,
              "cache_manifest": descriptor(destination / "source_cache.json"),
              "formal_w2_complete": complete and not partial_scope,
              "status_tasks_mutated": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", default="manifests/source_native_inventory.json")
    parser.add_argument("--output", default="cache/source_v2")
    parser.add_argument("--partial-scope", action="store_true")
    parser.add_argument("--restore-backed-raw", action="store_true",
                        help="After W1 only: restore one verified R2 source raw at a time to owned staging")
    parser.add_argument("--import-schema-specimens", action="store_true",
                        help="Only create a measured partial inventory from existing schema receipts; no numerical cache")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.import_schema_specimens:
            if args.execute:
                raise SourceCacheGateError("Metadata import and numeric execution must be separate commands")
            result = import_verified_specimen_inventory(args.root)
        else:
            result = prepare_cache(args.root, args.inventory, output=args.output,
                                   execute=args.execute, partial_scope=args.partial_scope,
                                   restore_backed_raw=args.restore_backed_raw)
    except (ValueError, FileNotFoundError, KeyError, TypeError, PermissionError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error),
                          "numerical_cache_created": False, "training_started": False}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 3 if result.get("status") == "SOURCE_CACHE_INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
