"""SOFTWARE_TEST_DOUBLE local transactions; no production freeze, EEG or I/O services."""
from copy import deepcopy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from p3work.contracts import file_digest, validate_signal_contract
from p3work.window_amendment import V21_PROFILE, V21_SCHEMA
from tools import freeze_approved_window as freeze
from test_contracts import software_approved_proof, software_legacy_contract


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def software_root(tmp_path, monkeypatch):
    contract = software_legacy_contract()
    contract.update(schema_version="paper3-signal-v2-preparation", status="pending_audit",
        selected_profile=None, context_s=None, crop_s=None, n_times=None, window_audit_receipt=None,
        profile_candidates={"long": {"context_s": [0, 3]}, "short": {"context_s": [0, 2]}})
    write(tmp_path / freeze.CONTRACT, contract)
    write(tmp_path / freeze.TASKS, {"SOFTWARE_TEST_DOUBLE": True,
        "W1": {"status": "BLOCKED_WINDOW_COVERAGE_PROTOCOL_REVIEW_REQUIRED",
            "participant_split_counts": {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11},
            "evidence": ["software/old_coverage.json"], "original_review": "preserve"},
        "W2": {"status": "NOT_STARTED_DEPENDS_ON_W1"},
        "W3": {"status": "NOT_STARTED_DEPENDS_ON_W1"}, "W7": {"status": "NOT_FROZEN"}})
    write(tmp_path / "software/old_coverage.json", {"SOFTWARE_TEST_DOUBLE": True, "selected": None})
    write(tmp_path / "contracts/work_contract.json", {"freeze": {"protocol_freeze_sha256": None}})
    raw = tmp_path / "data/source_schema/software_specimen.mat"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"SOFTWARE_TEST_DOUBLE-not-EEG")
    proof, calls = software_approved_proof(), []

    def preflight(root):
        assert Path(root) == tmp_path
        calls.append("independent_metadata_gate")
        return deepcopy(proof)

    def precommit(root, candidate):
        assert freeze._read(tmp_path / freeze.TASKS)["W1"]["status"] == "WINDOW_FREEZE_IN_PROGRESS_CLOSED"
        assert candidate == freeze._read(tmp_path / freeze.CONTRACT)
        sha = validate_signal_contract(candidate)
        calls.append("closed_published_contract_gate")
        return {"SOFTWARE_TEST_DOUBLE": True, "status": "APPROVED_FROZEN_CONTRACT_VERIFIED",
            "cache_authorized": False, "input_contract_sha256": sha, "artifact_sha256": "9" * 64}

    def final(root):
        assert freeze._read(tmp_path / freeze.TASKS)["W1"]["status"] == "COMPLETE"
        journal = freeze._read(tmp_path / freeze.TRANSACTION)
        assert journal["status"] in {"ALL_SCIENTIFIC_INPUTS_VERIFIED_BEFORE_W1_COMMIT", "COMPLETE"}
        calls.append("public_frozen_window_gate")
        return {"SOFTWARE_TEST_DOUBLE": True, "status": "EXACT_FROZEN_WINDOW_VERIFIED", "cache_authorized": True,
            "input_contract_sha256": validate_signal_contract(freeze._read(tmp_path / freeze.CONTRACT)),
            "artifact_sha256": "7" * 64}

    monkeypatch.setattr(freeze.windows, "verify_approved_window_audit", preflight)
    monkeypatch.setattr(freeze.windows, "verify_approved_frozen_contract", precommit)
    monkeypatch.setattr(freeze.windows, "verify_frozen_window", final)
    return tmp_path, proof, calls


def test_dry_run_validates_without_writes_or_complete_claim(software_root):
    root, _, calls = software_root
    before = snapshot(root)
    result = freeze.freeze_approved_window(root)
    assert result["status"] == "DRY_RUN_APPROVED_WINDOW_FREEZE_VALIDATED"
    assert result["W1_marked_complete"] is False and result["writes_performed"] is False
    assert snapshot(root) == before
    assert calls == ["independent_metadata_gate"]


def test_execute_commits_only_after_closed_canonical_gate_and_preserves_history(software_root):
    root, _, calls = software_root
    historical = {rel: sha for rel, sha in snapshot(root).items()
        if rel not in {freeze.CONTRACT, freeze.TASKS}}
    result = freeze.freeze_approved_window(root, execute=True)
    assert result["status"] == "APPROVED_WINDOW_FROZEN_AND_VERIFIED"
    contract = freeze._read(root / freeze.CONTRACT)
    tasks = freeze._read(root / freeze.TASKS)
    assert contract["schema_version"] == V21_SCHEMA and contract["selected_profile"] == V21_PROFILE
    assert contract["context_s"] == [0, 1.5] and contract["crop_s"] == [.25, 1.25]
    assert set(contract["profile_candidates"]) == {"long", "short", V21_PROFILE}
    assert contract["base_scientific_protocol_version"] == "2.0"
    assert contract["profile_candidates"][V21_PROFILE]["registration"] == "user_approved_scientific_amendment_v2.1"
    assert contract["profile_candidates"][V21_PROFILE]["approval"] == contract["window_audit_receipt"]["approval"]
    assert tasks["W1"]["status"] == "COMPLETE" and tasks["W1"]["original_review"] == "preserve"
    assert tasks["W1"]["participant_split_counts"] == {"meta_train": 40, "meta_validation": 11, "internal_holdout": 11}
    assert tasks["W2"]["status"] == "NOT_STARTED" and tasks["W7"]["protocol_freeze_sha256"] is None
    assert calls[-2:] == ["closed_published_contract_gate", "public_frozen_window_gate"]
    assert all(file_digest(root / rel) == sha for rel, sha in historical.items())
    assert (root / freeze.CONTRACT).stat().st_mode & 0o777 == 0o444


def test_identical_completed_transaction_is_read_only_even_after_downstream_exists(software_root):
    root, _, _ = software_root
    freeze.freeze_approved_window(root, execute=True)
    (root / "cache").mkdir()
    before = snapshot(root)
    result = freeze.freeze_approved_window(root, execute=True)
    assert result["status"] == "APPROVED_WINDOW_ALREADY_FROZEN_VERIFIED"
    assert result["writes_performed"] is False and snapshot(root) == before


@pytest.mark.parametrize("kind", ["cache", "W2_status", "W2_evidence", "raw_journal", "W7", "work_W7"])
def test_existing_cache_or_execution_or_w7_blocks_before_any_contract_write(software_root, kind):
    root, _, _ = software_root
    tasks = freeze._read(root / freeze.TASKS)
    if kind == "cache": (root / "cache").mkdir()
    elif kind == "W2_status": tasks["W2"]["status"] = "COMPLETE"
    elif kind == "W2_evidence": tasks["W2"]["cache_manifest"] = {"path": "old/cache.json"}
    elif kind == "raw_journal": write(root / "audit/source_preparation/resource_progress.json", {"SOFTWARE_TEST_DOUBLE": True})
    elif kind == "W7": tasks["W7"]["protocol_freeze_sha256"] = "1" * 64
    else: write(root / "contracts/work_contract.json", {"freeze": {"protocol_freeze_sha256": "1" * 64}})
    write(root / freeze.TASKS, tasks)
    before = snapshot(root)
    with pytest.raises(freeze.FreezeError):
        freeze.freeze_approved_window(root, execute=True)
    assert snapshot(root) == before


def test_precommit_gate_failure_never_publishes_w1_complete(software_root, monkeypatch):
    root, _, calls = software_root
    def failure(*args):
        assert freeze._read(root / freeze.TASKS)["W1"]["status"] == "WINDOW_FREEZE_IN_PROGRESS_CLOSED"
        raise ValueError("SOFTWARE_TEST_DOUBLE gate failure")
    monkeypatch.setattr(freeze.windows, "verify_approved_frozen_contract", failure)
    with pytest.raises(freeze.FreezeError, match="failed_closed"):
        freeze.freeze_approved_window(root, execute=True)
    assert freeze._read(root / freeze.TASKS)["W1"]["status"] == "WINDOW_FREEZE_FAILED_CLOSED"
    assert "public_frozen_window_gate" not in calls


def test_failed_public_gate_closes_stage_and_can_resume_same_owned_contract(software_root, monkeypatch):
    root, _, _ = software_root
    real_double = freeze.windows.verify_frozen_window
    monkeypatch.setattr(freeze.windows, "verify_frozen_window", lambda *_: (_ for _ in ()).throw(ValueError("SOFTWARE_TEST_DOUBLE")))
    with pytest.raises(freeze.FreezeError, match="failed_closed"):
        freeze.freeze_approved_window(root, execute=True)
    assert freeze._read(root / freeze.TASKS)["W1"]["status"] == "WINDOW_FREEZE_FAILED_CLOSED"
    assert freeze._read(root / freeze.TRANSACTION)["status"] == "FAILED_CLOSED"
    monkeypatch.setattr(freeze.windows, "verify_frozen_window", real_double)
    assert freeze.freeze_approved_window(root, execute=True)["W1_marked_complete"] is True
    assert freeze._read(root / freeze.TRANSACTION)["recovery_attempts"] == 1


def test_hard_exit_after_contract_replace_restarts_from_durable_owned_intent(software_root):
    root, _, _ = software_root
    original_write = freeze._write_json
    def child():
        def hard_exit(path, value, **kwargs):
            original_write(path, value, **kwargs)
            if path == root / freeze.CONTRACT:
                os._exit(73)  # Deliberately skips every finally handler.
        freeze._write_json = hard_exit
        freeze.freeze_approved_window(root, execute=True)
    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(10)
    assert process.exitcode == 73
    assert freeze._read(root / freeze.TASKS)["W1"]["status"] == "WINDOW_FREEZE_IN_PROGRESS_CLOSED"
    assert freeze._read(root / freeze.TRANSACTION)["status"] == "PREPARED_CLOSED"
    assert freeze.freeze_approved_window(root, execute=True)["status"] == "APPROVED_WINDOW_FROZEN_AND_VERIFIED"


def test_changed_owned_transaction_and_external_contract_edit_are_rejected(software_root, monkeypatch):
    root, proof, _ = software_root
    final = freeze.windows.verify_frozen_window
    monkeypatch.setattr(freeze.windows, "verify_frozen_window", lambda *_: (_ for _ in ()).throw(ValueError()))
    with pytest.raises(freeze.FreezeError): freeze.freeze_approved_window(root, execute=True)
    monkeypatch.setattr(freeze.windows, "verify_frozen_window", final)
    proof["raw_metadata_manifest_sha256"] = "0" * 64
    with pytest.raises(freeze.FreezeError, match="identity_changed"):
        freeze.freeze_approved_window(root, execute=True)


def test_source_lock_contention_does_not_begin_freeze(software_root):
    root, _, _ = software_root
    path = root / "audit/source_preparation/.source_worker.lock"
    path.parent.mkdir(parents=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = (root / freeze.CONTRACT).read_bytes()
        with pytest.raises(freeze.FreezeError, match="already_active"):
            freeze.freeze_approved_window(root, execute=True)
        assert (root / freeze.CONTRACT).read_bytes() == before
        assert not (root / freeze.TRANSACTION).exists()


def test_commit_between_preflight_and_lock_is_revalidated_and_never_overwritten(software_root, monkeypatch):
    root, _, _ = software_root
    original = freeze._no_downstream
    completed_snapshot, entered = {}, False
    def interleaved(root_arg, tasks):
        nonlocal entered
        if not entered:
            entered = True
            assert freeze.freeze_approved_window(root, execute=True)["W1_marked_complete"] is True
            completed_snapshot.update(snapshot(root))
        return original(root_arg, tasks)
    monkeypatch.setattr(freeze, "_no_downstream", interleaved)
    result = freeze.freeze_approved_window(root, execute=True)
    assert result["status"] == "APPROVED_WINDOW_ALREADY_FROZEN_VERIFIED"
    assert result["writes_performed"] is False
    assert snapshot(root) == completed_snapshot


def test_contract_bytes_changed_before_lock_are_not_silently_adopted(software_root, monkeypatch):
    root, _, _ = software_root
    original = freeze._no_downstream
    entered = False
    def edited(root_arg, tasks):
        nonlocal entered
        if not entered:
            entered = True
            changed = freeze._read(root / freeze.CONTRACT)
            changed["status"] = "pending_different_owner_edit"
            write(root / freeze.CONTRACT, changed)
        return original(root_arg, tasks)
    monkeypatch.setattr(freeze, "_no_downstream", edited)
    with pytest.raises(freeze.FreezeError, match="unowned_contract_changed"):
        freeze.freeze_approved_window(root, execute=True)
    assert not (root / freeze.TRANSACTION).exists()
    assert freeze._read(root / freeze.CONTRACT)["status"] == "pending_different_owner_edit"


def test_symlink_or_noncanonical_root_cannot_begin_a_transaction(software_root, tmp_path):
    root, _, _ = software_root
    alias = root.parent / (root.name + "-alias")
    alias.symlink_to(root, target_is_directory=True)
    before = snapshot(root)
    with pytest.raises(freeze.FreezeError, match="canonical"):
        freeze.freeze_approved_window(alias, execute=True)
    with pytest.raises(freeze.FreezeError, match="canonical"):
        freeze.freeze_approved_window(root / ".." / root.name, execute=True)
    assert snapshot(root) == before


def test_cli_defaults_to_read_only_dry_run_and_suppresses_arbitrary_failure(software_root, capsys, monkeypatch):
    root, _, _ = software_root
    before = snapshot(root)
    assert freeze.main(["--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["writes_performed"] is False
    assert snapshot(root) == before
    monkeypatch.setattr(freeze.windows, "verify_approved_window_audit", lambda *_: (_ for _ in ()).throw(ValueError("private-details")))
    assert freeze.main(["--root", str(root), "--execute"]) == 2
    output = capsys.readouterr().out
    assert "private-details" not in output
    assert json.loads(output)["status"] == "BLOCKED_NO_FREEZE_SUCCESS"


def test_actual_full_metadata_software_gates_accept_published_contract(tmp_path, monkeypatch):
    """Use all actual reader/gate code, with explicitly replaced fixture pins."""
    from p3work import window_amendment as amendment
    from test_window_amendment import build_software_amendment
    pins = build_software_amendment(tmp_path)
    for name, value in pins.items():
        monkeypatch.setattr(amendment, name, value)
    contract = freeze._read(tmp_path / freeze.CONTRACT)
    contract.update(schema_version="paper3-signal-v2-preparation", status="pending_audit",
        selected_profile=None, context_s=None, crop_s=None, n_times=None, window_audit_receipt=None,
        profile_candidates={"long": {"context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256},
            "short": {"context_s": [0, 2], "crop_s": [.5, 1.5], "n_times": 128}})
    write(tmp_path / freeze.CONTRACT, contract)
    tasks = freeze._read(tmp_path / freeze.TASKS)
    tasks.update(SOFTWARE_TEST_DOUBLE=True, W1={"status": "BLOCKED_WINDOW_COVERAGE_PROTOCOL_REVIEW_REQUIRED"},
        W2={"status": "NOT_STARTED_DEPENDS_ON_W1"}, W7={"status": "NOT_FROZEN"})
    write(tmp_path / freeze.TASKS, tasks)
    immutable = {rel: sha for rel, sha in snapshot(tmp_path).items()
        if rel not in {freeze.CONTRACT, freeze.TASKS}}
    before = snapshot(tmp_path)
    assert freeze.freeze_approved_window(tmp_path)["W1_marked_complete"] is False
    assert snapshot(tmp_path) == before
    result = freeze.freeze_approved_window(tmp_path, execute=True)
    assert result["W1_marked_complete"] is True
    actual_gate = freeze.windows.verify_frozen_window(tmp_path)
    assert actual_gate["cache_authorized"] is True
    assert actual_gate["input_contract_sha256"] == result["input_contract_sha256"]
    assert actual_gate["original_lr_trials"] == 792  # Software-only population.
    assert all(file_digest(tmp_path / rel) == sha for rel, sha in immutable.items())
