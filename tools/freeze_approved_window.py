#!/usr/bin/env python3
"""Freeze only the pinned, approved v2.1 window; default is a read-only dry run.

No downloads, raw EEG, training or W7 freeze. An interrupted commit is replayed
from its owned intent, with the same contract and independently rechecked proof.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from p3work.contracts import digest, file_digest, validate_signal_contract
from p3work import window_acceptance as windows
from p3work.window_amendment import (
    APPROVAL_PATH, AUDIT_PATH, CONTEXT_S, CROP_S, N_TIMES, V21_PROFILE,
    V21_SCHEMA, build_v21_window_receipt,
)

TRANSACTION = "audit/stieger_w1/window_amendment_v2_1_20261019/freeze_transaction.json"
CONTRACT = "contracts/input_contract.json"
TASKS = "status/tasks.json"
LOCKS = ("audit/source_full/.prepare_all_sources.lock",
         "audit/source_preparation/.source_worker.lock",
         "audit/stieger_w1/window_amendment_v2_1_20261019/.freeze.lock")


class FreezeError(ValueError):
    """Fixed local failure codes only; no credentials or exception payloads."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _path(root, relative):
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in {".", ".."} for p in parts):
        raise FreezeError("freeze_path_invalid")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise FreezeError("freeze_symlinks_forbidden")
    return path


def _root(value):
    root = Path(value).absolute()
    if (any(p.is_symlink() for p in (root, *root.parents)) or not root.is_dir()
            or root != root.resolve()):
        raise FreezeError("freeze_canonical_existing_root_required")
    return root


def _read(path):
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError):
        raise FreezeError("freeze_json_missing_or_invalid") from None
    if not isinstance(value, dict):
        raise FreezeError("freeze_json_object_required")
    return value


def _write_json(path, value, *, mode=0o600):
    """Durable file replacement, including its directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".window-freeze-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _published_sha(value):
    return sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def _no_w7(root, tasks, contract):
    w7 = tasks.get("W7", {})
    if (contract.get("protocol_freeze_sha256") is not None
            or w7.get("status") not in {None, "NOT_FROZEN"}
            or any(w7.get(k) is not None for k in ("protocol_freeze_sha256", "freeze_sha256"))):
        raise FreezeError("w7_must_remain_unfrozen")
    work_path = _path(root, "contracts/work_contract.json")
    if work_path.is_file() and _read(work_path).get("freeze", {}).get("protocol_freeze_sha256") is not None:
        raise FreezeError("w7_must_remain_unfrozen")


def _no_downstream(root, tasks):
    if not str(tasks.get("W2", {}).get("status", "NOT_STARTED")).startswith("NOT_STARTED"):
        raise FreezeError("w2_started_contract_change_forbidden")
    if any(tasks.get("W2", {}).get(k) is not None for k in
           ("cache_manifest", "cache_r2_receipt", "raw_r2_receipts_catalog")):
        raise FreezeError("w2_evidence_exists_contract_change_forbidden")
    if _path(root, "cache").exists():
        raise FreezeError("cache_exists_contract_change_forbidden")
    for relative in ("audit/source_full/progress.json", "audit/source_preparation/resource_progress.json",
                     "audit/source_preparation/resource_progress.wall.json", "audit/source_cache/resource_progress.json"):
        if _path(root, relative).exists():
            raise FreezeError("source_preparation_started_contract_change_forbidden")
    batch_root = _path(root, "audit/source_preparation/batches")
    if batch_root.exists() and any(batch_root.glob("*/objects/*/state.json")):
        raise FreezeError("source_preparation_started_contract_change_forbidden")


def _candidate(existing, proof):
    contract = deepcopy(existing)
    contract.update(schema_version=V21_SCHEMA, scientific_protocol_version="2.1",
        base_scientific_protocol_version="2.0",
        status="window_frozen", selected_profile=V21_PROFILE,
        context_s=list(CONTEXT_S), crop_s=list(CROP_S), n_times=N_TIMES,
        window_audit_receipt=build_v21_window_receipt(proof), protocol_freeze_sha256=None)
    profiles = contract.setdefault("profile_candidates", {})
    profiles[V21_PROFILE] = {"context_s": list(CONTEXT_S), "crop_s": list(CROP_S), "n_times": N_TIMES,
        "registration": "user_approved_scientific_amendment_v2.1",
        "approval": deepcopy(proof["approval"])}
    validate_signal_contract(contract)
    return contract


def _proof(root):
    proof = windows.verify_approved_window_audit(root)
    if (not isinstance(proof, dict) or proof.get("cache_authorized") is not False
            or not isinstance(proof.get("artifact_sha256"), str)):
        raise FreezeError("independent_approved_window_audit_required")
    return proof


def _journal_identity(candidate, proof):
    return {"schema_version": "paper3-approved-window-freeze-identity-v1",
        "contract_sha256": validate_signal_contract(candidate),
        "approved_window_acceptance_sha256": proof["artifact_sha256"],
        "approval": proof["approval"], "single_candidate_audit": proof["single_candidate_audit"],
        "freeze_implementation_sha256": file_digest(__file__)}


def _final_verification(root, contract_sha):
    verified = windows.verify_frozen_window(root)
    if (verified.get("input_contract_sha256") != contract_sha
            or verified.get("status") != "EXACT_FROZEN_WINDOW_VERIFIED"
            or verified.get("cache_authorized") is not True):
        raise FreezeError("post_freeze_contract_identity_mismatch")
    return verified


def _owned_transaction(journal_path, contract_path, candidate, identity):
    prior = _read(journal_path) if journal_path.exists() else None
    if prior is None:
        if _read(contract_path).get("status") == "window_frozen":
            raise FreezeError("existing_frozen_contract_cannot_be_replaced")
        return None
    try:
        original_sha = sha256(bytes.fromhex(prior.get("original_contract_bytes_hex", ""))).hexdigest()
    except (ValueError, TypeError):
        raise FreezeError("freeze_transaction_original_snapshot_invalid") from None
    if (prior.get("schema_version") != "paper3-approved-window-freeze-transaction-v1"
            or prior.get("identity") != identity or prior.get("identity_sha256") != digest(identity)
            or prior.get("candidate_contract") != candidate
            or prior.get("published_contract_file_sha256") != _published_sha(candidate)
            or original_sha != prior.get("original_contract_file_sha256")):
        raise FreezeError("freeze_transaction_identity_changed_preserve_artifacts")
    if file_digest(contract_path) not in {original_sha, prior["published_contract_file_sha256"]}:
        raise FreezeError("contract_changed_outside_owned_freeze_transaction")
    return prior


def _already_complete(root, identity, *, execute):
    verified = _final_verification(root, identity["contract_sha256"])
    return {"status": "APPROVED_WINDOW_ALREADY_FROZEN_VERIFIED", "execute": execute,
        "writes_performed": False, "input_contract_sha256": identity["contract_sha256"],
        "window_acceptance_sha256": verified["artifact_sha256"], "W7_frozen": False}


def _complete_tasks(tasks, journal, proof):
    result = deepcopy(tasks)
    w1 = result.setdefault("W1", {})
    evidence = list(w1.get("evidence", []))
    for path in (APPROVAL_PATH, AUDIT_PATH, CONTRACT, TRANSACTION):
        if path not in evidence:
            evidence.append(path)
    w1.update(status="COMPLETE", window_selected=V21_PROFILE, window_decision_pool="meta_train",
        scientific_protocol_version="2.1", complete_meta_train_sessions=proof["complete_objects"],
        meta_train_original_lr_trials=proof["original_lr_trials"],
        input_contract_sha256=journal["identity"]["contract_sha256"],
        approved_window_acceptance_sha256=proof["artifact_sha256"],
        window_amendment_approval=deepcopy(proof["approval"]),
        window_amendment_audit=deepcopy(proof["single_candidate_audit"]),
        freeze_transaction={"path": TRANSACTION, "identity_sha256": journal["identity_sha256"]},
        evidence=evidence, completed_at_utc=journal["created_at_utc"],
        next="New source505/cache/R2 preparation under the approved signal contract; no GPU start.")
    result.setdefault("W2", {}).update(status="NOT_STARTED", old_npz_accepted=False,
        blocking_reason=None)
    result.setdefault("W7", {}).update(status="NOT_FROZEN", protocol_freeze_sha256=None)
    result["recorded_at_utc"] = _utc()
    return result


def freeze_approved_window(root=REPO, *, execute=False):
    root = _root(root)
    contract_path, tasks_path, journal_path = (_path(root, rel) for rel in (CONTRACT, TASKS, TRANSACTION))
    existing, tasks = _read(contract_path), _read(tasks_path)
    initial_contract_file_sha = file_digest(contract_path)
    _no_w7(root, tasks, existing)
    proof = _proof(root)  # Actual pinned approval and complete original metadata; no W1 claim.
    candidate = _candidate(existing, proof)
    identity = _journal_identity(candidate, proof)
    prior = _owned_transaction(journal_path, contract_path, candidate, identity)
    if prior is not None and prior.get("status") == "COMPLETE":
        return _already_complete(root, identity, execute=execute)
    _no_downstream(root, tasks)
    if not execute:
        return {"status": "DRY_RUN_APPROVED_WINDOW_FREEZE_VALIDATED", "execute": False,
            "writes_performed": False, "would_complete_W1": True, "W1_marked_complete": False,
            "input_contract_sha256": identity["contract_sha256"],
            "approved_window_acceptance_sha256": proof["artifact_sha256"], "W7_frozen": False}
    with ExitStack() as stack:
        for relative in LOCKS:
            lock_path = _path(root, relative)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise FreezeError("source_or_freeze_writer_already_active") from None
        tasks = _read(tasks_path)
        _no_w7(root, tasks, _read(contract_path))
        # Another freeze may have completed between our read-only preflight and
        # obtaining these locks. Never replay a stale prior=None over its intent.
        prior = _owned_transaction(journal_path, contract_path, candidate, identity)
        if prior is not None and prior.get("status") == "COMPLETE":
            return _already_complete(root, identity, execute=execute)
        _no_downstream(root, tasks)
        if prior is None and file_digest(contract_path) != initial_contract_file_sha:
            raise FreezeError("unowned_contract_changed_during_freeze_preflight")
        # Recheck after acquiring every writer lock. Prior evidence or a contract
        # edit cannot be adopted merely because it passed an earlier preflight.
        locked_proof = _proof(root)
        if _journal_identity(_candidate(_read(contract_path), locked_proof), locked_proof) != identity:
            raise FreezeError("freeze_inputs_changed_before_locked_commit")
        if prior is None:
            journal = {"schema_version": "paper3-approved-window-freeze-transaction-v1",
                "status": "PREPARED_CLOSED", "created_at_utc": _utc(), "identity": identity,
                "identity_sha256": digest(identity), "candidate_contract": candidate,
                "original_contract_file_sha256": file_digest(contract_path),
                "original_contract_bytes_hex": contract_path.read_bytes().hex(),
                "published_contract_file_sha256": _published_sha(candidate),
                "original_W1": deepcopy(tasks.get("W1", {})), "recovery_attempts": 0}
        else:
            journal = prior
            journal["recovery_attempts"] += 1
        _write_json(journal_path, journal)
        try:
            tasks.setdefault("W1", {}).update(status="WINDOW_FREEZE_IN_PROGRESS_CLOSED",
                freeze_transaction={"path": TRANSACTION, "identity_sha256": journal["identity_sha256"]})
            _write_json(tasks_path, tasks, mode=0o644)
            _write_json(contract_path, candidate, mode=0o444)
            journal.update(status="CONTRACT_PUBLISHED_CLOSED", published_contract_file_sha256=file_digest(contract_path))
            _write_json(journal_path, journal)
            # While W1 remains closed, independently re-read all scientific inputs
            # and compare the actual published canonical contract to its receipt.
            final_proof = _proof(root)
            published = _read(contract_path)
            if (_journal_identity(published, final_proof) != identity
                    or published != candidate
                    or published["window_audit_receipt"] != build_v21_window_receipt(final_proof)):
                raise FreezeError("published_contract_or_original_evidence_changed")
            precommit = windows.verify_approved_frozen_contract(root, published)
            if (precommit.get("status") != "APPROVED_FROZEN_CONTRACT_VERIFIED"
                    or precommit.get("cache_authorized") is not False
                    or precommit.get("input_contract_sha256") != identity["contract_sha256"]):
                raise FreezeError("closed_canonical_contract_verification_required")
            journal.update(status="ALL_SCIENTIFIC_INPUTS_VERIFIED_BEFORE_W1_COMMIT",
                precommit_verified_at_utc=_utc(), precommit_contract_acceptance_sha256=precommit["artifact_sha256"])
            _write_json(journal_path, journal)
            _write_json(tasks_path, _complete_tasks(tasks, journal, final_proof), mode=0o644)
            verified = _final_verification(root, identity["contract_sha256"])
            journal.update(status="COMPLETE", completed_at_utc=_utc(),
                final_window_acceptance_sha256=verified["artifact_sha256"])
            _write_json(journal_path, journal)
            return {"status": "APPROVED_WINDOW_FROZEN_AND_VERIFIED", "execute": True,
                "writes_performed": True, "W1_marked_complete": True, "W7_frozen": False,
                "input_contract_sha256": identity["contract_sha256"],
                "window_acceptance_sha256": verified["artifact_sha256"],
                "freeze_transaction": {"path": TRANSACTION, "sha256": file_digest(journal_path)}}
        except BaseException as error:
            closed = _read(tasks_path)
            closed.setdefault("W1", {}).update(status="WINDOW_FREEZE_FAILED_CLOSED")
            _write_json(tasks_path, closed, mode=0o644)
            journal.update(status="FAILED_CLOSED", failure_type=type(error).__name__,
                failure_code=str(error) if isinstance(error, FreezeError) else "verification_failed_details_suppressed")
            _write_json(journal_path, journal)
            raise FreezeError("approved_window_freeze_failed_closed_resume_same_transaction") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = freeze_approved_window(args.root, execute=args.execute)
    except Exception as error:
        result = {"status": "BLOCKED_NO_FREEZE_SUCCESS", "error_type": type(error).__name__,
            "gap": str(error) if isinstance(error, FreezeError) else "verification_failed_details_suppressed"}
        print(json.dumps(result))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
