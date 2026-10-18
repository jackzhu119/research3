#!/usr/bin/env python3
"""W1-gated CPU source preparation and verified cache archival; no training.

Default is a candidate-only dry run. --execute serially archives official source
raw/metadata, creates a resumable real numerical cache, and backs up only the
GPU-required cache and metadata. W2 COMPLETE is written only after all measured
cache artifacts and their complete R2 GET/SHA256 receipts pass verification.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.contracts import SOURCE_IDS, file_digest
from tools import source_preparation_batch as batch
from tools.prepare_source_cache import prepare_cache
from tools.r2_backup import run_backup, sha256_file
from tools.r2_restore import load_verified_receipt, no_symlink_components


class PreparationError(Exception):
    pass


def within_root(root: Path, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            value = path.relative_to(root).as_posix()
        except ValueError:
            raise PreparationError("artifact_path_outside_paper3_root") from None
    return batch.local_path(root, str(value))


def checked_artifact(root: Path, item: dict) -> Path:
    if not isinstance(item, dict):
        raise PreparationError("artifact_descriptor_required")
    path = within_root(root, item.get("path", ""))
    if not path.is_file() or file_digest(path) != item.get("sha256"):
        raise PreparationError("artifact_full_local_sha256_mismatch")
    return path


def raw_receipt_catalog(root: Path, inventory_path: Path) -> tuple[Path, list[Path]]:
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("status") != "SOURCE_RAW_INVENTORY_COMPLETE" or inventory.get("provider_roster_verified") is not True
            or len(inventory.get("files", [])) != 505):
        raise PreparationError("all_505_raw_native_audits_and_backups_required")
    receipts, paths, raw_objects = [], [], []
    identities = set()
    candidates = batch.expected_source_objects()
    for item in inventory["files"]:
        ident = batch.identity(item)
        if ident in identities or item.get("url") != candidates.get(ident):
            raise PreparationError("raw_inventory_identity_not_exact_source_roster")
        identities.add(ident)
        receipt_path = checked_artifact(root, item["backup_receipt"])
        rows, _ = load_verified_receipt(receipt_path)
        raw = [r for r in rows if r["role"] == "source_development_raw"]
        if (len(raw) != 1 or any(raw[0][k] != item[k] for k in ("path", "sha256", "bytes", "dataset"))):
            raise PreparationError("raw_inventory_receipt_identity_mismatch")
        paths.append(receipt_path)
        raw_objects.append({"identity": list(ident), "raw": raw[0], "backup_receipt": item["backup_receipt"]})
        receipts.append(item["backup_receipt"])
    if identities != set(candidates):
        raise PreparationError("raw_inventory_505_candidate_set_incomplete")
    catalog_path = root / "audit/source_full/raw_r2_receipts_catalog.json"
    batch.atomic_json(catalog_path, {"schema_version": 1, "status": "ALL_505_RAW_OBJECTS_FULL_GET_SHA256_VERIFIED",
        "raw_inventory": batch.descriptor(root, inventory_path), "receipts": receipts, "objects": raw_objects,
        "raw_payloads_in_cache_backup": False, "training_started": False})
    return catalog_path, paths


def cache_backup_manifest(root, cache_path, inventory_path, catalog_path, receipt_paths, contract_path) -> tuple[Path, dict]:
    manifest = json.loads(cache_path.read_text())
    if (manifest.get("schema_version") != "paper3-source-cache-v2" or manifest.get("status") != "SOURCE_CACHE_COMPLETE"
            or manifest.get("synthetic") is not False or manifest.get("old_npz_reused") is not False
            or manifest.get("full_source_inventory_verified") is not True or manifest.get("training_started") is not False
            or manifest.get("input_contract_sha256") != batch.execution_gate(root)
            or {s.get("dataset") for s in manifest.get("shards", [])} != SOURCE_IDS
            or checked_artifact(root, manifest.get("raw_inventory")) != inventory_path):
        raise PreparationError("complete_real_four_source_cache_required")
    files = {}

    def add(path, role, dataset):
        path = within_root(root, path)
        sha, size = sha256_file(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = {"path": relative, "sha256": sha, "bytes": size, "role": role, "dataset": dataset}

    for shard in manifest["shards"]:
        for field in ("X", "y", "events", "exclusions"):
            path = checked_artifact(root, shard[field])
            if not path.is_relative_to(cache_path.parent):
                raise PreparationError("cache_shard_outside_owned_cache_directory")
            add(path, "result", shard["dataset"])
    for shard in manifest.get("participants_without_usable_trials", []):
        if shard.get("dataset") not in SOURCE_IDS:
            raise PreparationError("excluded_participant_outside_source_roster")
        add(checked_artifact(root, shard["exclusions"]), "result", shard["dataset"])
    for path, role in ((cache_path, "manifest"), (inventory_path, "manifest"), (catalog_path, "manifest"),
                       (contract_path, "protocol")):
        add(path, role, "Paper3")
    for receipt in receipt_paths:
        add(receipt, "audit", "Paper3")
    evidence = checked_artifact(root, json.loads(inventory_path.read_text())["provider_terms_evidence"])
    add(evidence, "manifest", "Paper3")
    for item in json.loads(evidence.read_text())["files"]:
        add(checked_artifact(root, item), "audit", "Paper3")
    target = root / "audit/source_full/cache_backup_manifest.json"
    result = {"schema_version": 1, "files": [files[p] for p in sorted(files)]}
    if any(f["role"] == "source_development_raw" for f in result["files"]):
        raise PreparationError("raw_payload_forbidden_in_gpu_cache_backup")
    batch.atomic_json(target, result)
    return target, result


def verify_cache_backup(root, receipt_path, manifest):
    rows, _ = load_verified_receipt(receipt_path)
    fields = ("path", "sha256", "bytes", "role", "dataset")
    if {tuple(r[k] for k in fields) for r in rows} != {tuple(r[k] for k in fields) for r in manifest["files"]}:
        raise PreparationError("cache_backup_full_get_object_set_mismatch")
    for item in manifest["files"]:
        checked_artifact(root, item)


def prepare_all(root=REPO, *, contract="contracts/input_contract.json", progress="audit/source_full/progress.json",
                execute=False, raw_fn=None, cache_fn=None, backup_fn=None) -> dict:
    root = Path(root).resolve(strict=True)
    no_symlink_components(root)
    contract_path, progress_path = within_root(root, contract), within_root(root, progress)
    plan = batch.build_plan()
    if not execute:
        return {"status": "CANDIDATE_PLAN_ONLY", "candidate_objects": 505, "candidate_sizes_known": False,
                "provider_roster_verified": False, "numerical_cache_created": False,
                "training_started": False, "execute": False, "plan_sha256": plan["plan_sha256"]}
    # Worker/cache builder share this canonical contract; reject an alternate
    # file rather than silently preparing against a different contract.
    if contract_path != root / "contracts/input_contract.json":
        raise PreparationError("canonical_input_contract_path_required")
    contract_sha = batch.execution_gate(root)
    lock_path = root / "audit/source_full/.prepare_all_sources.lock"
    no_symlink_components(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    raw_fn, cache_fn, backup_fn = raw_fn or batch.execute_plan, cache_fn or prepare_cache, backup_fn or run_backup
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreparationError("all_sources_preparation_already_active") from None
        checkpoint = {"status": "RAW_PREPARATION", "execute": True, "training_started": False,
                      "input_contract_sha256": contract_sha, "verified_raw_objects": 0, "candidate_objects": 505}
        batch.atomic_json(progress_path, checkpoint)
        try:
            raw_result = raw_fn(plan, root=root)
            checkpoint.update(raw_result=raw_result, verified_raw_objects=raw_result.get("actual_verified_objects", 0))
            batch.atomic_json(progress_path, checkpoint)
            if raw_result.get("inventory_status") != "SOURCE_RAW_INVENTORY_COMPLETE" or checkpoint["verified_raw_objects"] != 505:
                raise PreparationError("raw_preparation_incomplete_see_per_object_state_and_resource_progress")
            inventory_path = root / "audit/source_preparation/batches" / plan["plan_sha256"] / "source_native_inventory.json"
            catalog_path, receipts = raw_receipt_catalog(root, inventory_path)
            checkpoint.update(status="NUMERICAL_CACHE_PREPARATION", raw_inventory=batch.descriptor(root, inventory_path),
                              raw_r2_receipts_catalog=batch.descriptor(root, catalog_path))
            batch.atomic_json(progress_path, checkpoint)
            cached = cache_fn(root, inventory_path.relative_to(root).as_posix(), execute=True, restore_backed_raw=True)
            checkpoint["cache_result"] = cached
            batch.atomic_json(progress_path, checkpoint)
            if cached.get("status") != "SOURCE_CACHE_COMPLETE" or cached.get("formal_w2_complete") is not True:
                raise PreparationError("numerical_source_cache_incomplete_see_participant_checkpoints")
            cache_path = checked_artifact(root, cached["cache_manifest"])
            manifest_path, manifest = cache_backup_manifest(root, cache_path, inventory_path, catalog_path, receipts, contract_path)
            receipt_path = root / "audit/source_full/cache_backup_receipt.json"
            checkpoint.update(status="CACHE_R2_PUT_FULL_GET_VERIFICATION", cache_manifest=batch.descriptor(root, cache_path),
                              cache_backup_bytes=sum(f["bytes"] for f in manifest["files"]), cache_backup_objects=len(manifest["files"]))
            batch.atomic_json(progress_path, checkpoint)
            result = backup_fn(manifest_path=manifest_path, root=root, prefix=f"paper3/source-cache-v2/{file_digest(cache_path)}/",
                               receipt_path=receipt_path, execute=True)
            if result.get("status") != "verified" or result.get("execute") is not True:
                raise PreparationError("cache_r2_put_full_get_sha256_incomplete")
            verify_cache_backup(root, receipt_path, manifest)
            if batch.execution_gate(root) != contract_sha:
                raise PreparationError("input_contract_changed_during_preparation")
            # Read latest status immediately before merge; preserve every other W.
            tasks_path = root / "status/tasks.json"
            no_symlink_components(tasks_path)
            tasks = json.loads(tasks_path.read_text())
            tasks["W2"] = {"status": "COMPLETE", "cache_manifest": batch.descriptor(root, cache_path),
                "cache_r2_receipt": receipt_path.relative_to(root).as_posix(),
                "cache_r2_receipt_sha256": file_digest(receipt_path),
                "raw_r2_receipts_catalog": catalog_path.relative_to(root).as_posix(),
                "raw_r2_receipts_catalog_sha256": file_digest(catalog_path),
                "input_contract_sha256": contract_sha, "training_started": False}
            batch.atomic_json(tasks_path, tasks)
            checkpoint.update(status="W2_COMPLETE_CACHE_AND_R2_VERIFIED", verified_raw_objects=505,
                              cache_r2_receipt=batch.descriptor(root, receipt_path), w2_complete=True)
            batch.atomic_json(progress_path, checkpoint)
            return checkpoint
        except Exception as exc:
            checkpoint.update(status="BLOCKED_NO_W2_ADVANCE", w2_complete=False,
                error_type=type(exc).__name__, gap=str(exc) if isinstance(exc, (PreparationError, batch.BatchError)) else "operation_failed_details_suppressed")
            batch.atomic_json(progress_path, checkpoint)
            raise PreparationError(checkpoint["gap"]) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--contract", default="contracts/input_contract.json")
    parser.add_argument("--progress", default="audit/source_full/progress.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = prepare_all(args.root, contract=args.contract, progress=args.progress, execute=args.execute)
        print(json.dumps(result), flush=True)
        return 0 if result.get("status") == "W2_COMPLETE_CACHE_AND_R2_VERIFIED" else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_NO_W2_ADVANCE", "error_type": type(exc).__name__,
                          "gap": str(exc) if isinstance(exc, (PreparationError, batch.BatchError)) else "operation_failed_details_suppressed",
                          "training_started": False}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
