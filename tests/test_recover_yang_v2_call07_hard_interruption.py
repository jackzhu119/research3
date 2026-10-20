"""Synthetic accounting fixtures: no original ledger, network, data or fit."""
import copy
import fcntl
from pathlib import Path

import pytest

from tools import recover_yang_v2_call07_hard_interruption as work
from tests import test_resume_yang_source_transport_v2 as originals


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, plan, filename = originals.fixture_root.__wrapped__(tmp_path, monkeypatch)
    originals.authorize((root, plan, filename))
    originals.single_prepare(monkeypatch, (root, plan, filename),
        error=work.core.PreparationError("provider_did_not_honor_exact_range"))
    for _ in range(6):
        work.parent.run(root, execute=True)
    for rel in (work.ENTRY, work.TESTS):
        target = root / rel; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((work.ROOT / rel).read_bytes())
    folder = work.call_dir(root)
    before = folder / "before_closed_ledger.json"
    work.immutable_bytes(before, filename.read_bytes())
    plan_ref = work.ref(root, work.path(root, work.parent.PLAN_PATH))
    amendment_ref = work.ref(root, work.path(root, work.parent.AMENDMENT_PATH))
    intent = {"schema_version": work.parent.SCHEMA, "call_number": 7, "amendment": amendment_ref,
        "plan": plan_ref, "before_closed_ledger": work.ref(root, before),
        "session_headers": work.parent.HEADERS, "requests_adapter_retries": 0,
        "one_prepare_call_only": True, "max_new_records": None,
        "started_clock_for_audit_only": work.core.clock_snapshot()}
    work.core.atomic_json(folder / "intent.json", intent, immutable=True)
    state = work.read(work.parent_state(root)); state.update(calls_used=7, active_call=7)
    work.core.atomic_json(work.parent_state(root), state)
    ledger = work.core.Ledger(filename, plan)
    ledger.reserve("r2_get_body_bytes", 100, "SYNTHETIC_INTERRUPTED_RAW_GET_NO_NETWORK")
    monkeypatch.setattr(work, "OPEN_STATE_SHA", work.ref(root, work.parent_state(root))["sha256"])
    monkeypatch.setattr(work, "OPEN_LEDGER_SHA", work.ref(root, filename)["sha256"])
    monkeypatch.setattr(work, "INTENT_SHA", work.ref(root, folder / "intent.json")["sha256"])
    monkeypatch.setattr(work, "BEFORE_SHA", work.ref(root, before)["sha256"])
    monkeypatch.setattr(work, "EXPECTED_BOOT", work.core.clock_snapshot()["boot_id"])
    monkeypatch.setattr(work, "proc_record", lambda pid: {"pid": pid, "status": "ABSENT"})
    monkeypatch.setattr(work, "active_original_workers", lambda: [])
    return root, plan, filename


def authorize(fixture):
    root, plan, filename = fixture
    work.write_amendment(root)
    return root, plan, filename


def test_default_reads_only_without_mutable_Ledger_or_new_artifacts(fixture, monkeypatch):
    root, _, _ = fixture
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(work.core, "Ledger", lambda *a, **k: pytest.fail("dryrun Ledger"))
    monkeypatch.setattr(work.parent.previous, "session", lambda: pytest.fail("dryrun Session"))
    value = work.run(root)
    assert value["status"] == "DRY_RUN_EXTERNAL_CALL07_SETTLEMENT_NO_WRITES"
    assert value["calls_used_preserved"] == 7 and value["remaining_calls_after_settlement"] == 1
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_amendment_preserves_open_bytes_without_settling_or_falsely_closing(fixture):
    root, plan, filename = fixture; before = filename.read_bytes()
    old_state = work.parent_state(root).read_bytes()
    result = work.write_amendment(root); authority, descriptor = work.verify_authority(root, plan)
    assert result["amendment"] == descriptor
    assert work.checked(root, authority["open_original_ledger_opaque"]).read_bytes() == before
    assert authority["open_original_ledger_opaque"]["path"].endswith(".txt")
    assert work.checked(root, authority["open_V2_state_opaque"]).read_bytes() == old_state
    assert filename.read_bytes() == before and work.parent_state(root).read_bytes() == old_state
    assert not (work.call_dir(root) / "receipt.json").exists()


@pytest.mark.parametrize("mutation", ["calls", "active", "terminal", "ledger", "intent", "after_exists"])
def test_only_exact_original_open_call07_can_be_authorized(fixture, mutation):
    root, _, filename = fixture; state = work.read(work.parent_state(root))
    if mutation == "calls": state["calls_used"] = 8
    elif mutation == "active": state["active_call"] = None
    elif mutation == "terminal": state["terminal"] = "SOURCE_EXTENSION_CACHE_COMPLETE"
    elif mutation == "ledger":
        value = work.read(filename); value["charged"]["r2_get_body_bytes"] -= 1; work.core.atomic_json(filename, value)
    elif mutation == "intent":
        value = work.read(work.call_dir(root) / "intent.json"); value["one_prepare_call_only"] = False
        work.core.atomic_json(work.call_dir(root) / "intent.json", value)
    else: work.core.atomic_json(work.call_dir(root) / "after_closed_ledger.json", {"forged": True})
    work.core.atomic_json(work.parent_state(root), state)
    with pytest.raises(work.RecoveryError): work.write_amendment(root)
    assert not work.path(root, work.AMENDMENT).exists()


@pytest.mark.parametrize("alive", [True, False])
def test_alive_PID_or_another_entry_blocks_recovery(fixture, monkeypatch, alive):
    root, _, _ = fixture
    if alive: monkeypatch.setattr(work, "proc_record", lambda pid: {"pid": pid, "status": "ALIVE_OR_UNPROVEN"})
    else: monkeypatch.setattr(work, "active_original_workers", lambda: [999999])
    with pytest.raises(work.RecoveryError): work.write_amendment(root)
    assert not work.path(root, work.AMENDMENT).exists()


@pytest.mark.parametrize("global_lock", [False, True])
def test_original_owner_and_global_locks_prevent_parallel_settlement(fixture, global_lock):
    root, _, _ = fixture
    relative = "audit/source_extensions_20261019/.source-extension-singleton.lock" if global_lock else work.parent.HISTORY + "/owner.lock"
    lock_file = work.path(root, relative)
    with lock_file.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((work.parent.TransportRecoveryError, work.parent.previous.TransportRecoveryError)):
            work.write_amendment(root)


def test_external_settlement_charges_full_interruption_and_downtime_without_retry(fixture, monkeypatch):
    root, plan, filename = authorize(fixture)
    old = work.read(filename); old_six = copy.deepcopy(work.read(work.parent_state(root))["call_receipts"])
    monkeypatch.setattr(work.core, "prepare", lambda *a, **k: pytest.fail("recovery prepared"))
    monkeypatch.setattr(work.parent.previous, "session", lambda: pytest.fail("recovery Session"))
    result = work.run(root, execute=True)
    after = work.read(filename); state = work.read(work.parent_state(root)); receipt = work.read(work.call_dir(root) / "receipt.json")
    assert result["status"] == work.STATUS and result["remaining_V2_calls"] == 1
    assert after["charged"] == old["charged"] and after["elapsed_seconds"] >= old["elapsed_seconds"]
    assert after["history"] == old["history"] + [{**old["active_transfer"], "status": "HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"}]
    assert "wall_intent" not in after and "active_transfer" not in after
    assert state["calls_used"] == 7 and state["active_call"] is None and state["terminal"] is None
    assert state["call_receipts"][:6] == old_six and len(state["call_receipts"]) == 7
    assert receipt["worker_returned_normally"] is False and receipt["worker_result_observed"] is False
    assert receipt["error_type"] is None and receipt["coordinator_result"] is None and receipt["retryable_transport_error"] is False
    assert receipt["receipt_producer"] == work.ENTRY and receipt["before_closed_ledger"] == work.read(work.call_dir(root) / "intent.json")["before_closed_ledger"]
    _, amendment_ref = work.parent.verify_static_authority(root, plan)
    assert work.parent._history(root, plan, amendment_ref)[2]["calls_used"] == 7


def test_completed_recovery_is_pure_idempotent_without_Ledger_or_new_clock(fixture, monkeypatch):
    root, _, filename = authorize(fixture); work.run(root, execute=True)
    before = filename.read_bytes()
    monkeypatch.setattr(work.core, "Ledger", lambda *a, **k: pytest.fail("repeated settlement"))
    monkeypatch.setattr(work.parent.previous, "session", lambda: pytest.fail("offline Session"))
    assert work.run(root, execute=True)["status"] == work.STATUS
    assert filename.read_bytes() == before


@pytest.mark.parametrize("mutation", ["normal", "HTTP", "COMPLETE", "return_value", "refund"])
def test_settlement_verifier_rejects_false_normal_retry_completion_or_refund(fixture, mutation):
    root, _, filename = authorize(fixture); work.run(root, execute=True)
    target = work.call_dir(root) / "receipt.json"; receipt = work.read(target)
    if mutation == "normal": receipt["worker_returned_normally"] = True
    elif mutation == "HTTP": receipt["retryable_transport_error"] = True
    elif mutation == "COMPLETE": receipt["coordinator_result"] = {"status": "SOURCE_EXTENSION_CACHE_COMPLETE"}
    elif mutation == "return_value": receipt["worker_result_observed"] = True
    else:
        after = work.checked(root, receipt["after_closed_ledger"]); value = work.read(after)
        value["charged"]["r2_get_body_bytes"] -= 100; work.core.atomic_json(after, value)
    if mutation != "refund": work.core.atomic_json(target, receipt)
    with pytest.raises((ValueError, work.core.PreparationError)): work.verify_settlement(root)


def test_interruption_after_settlement_before_receipt_cannot_automatically_replay(fixture, monkeypatch):
    root, _, filename = authorize(fixture); real_snapshot = work.snapshot
    def interrupted(root, filename, relative):
        if relative.endswith("call-07/after_closed_ledger.json"): raise KeyboardInterrupt()
        return real_snapshot(root, filename, relative)
    monkeypatch.setattr(work, "snapshot", interrupted)
    with pytest.raises(KeyboardInterrupt): work.run(root, execute=True)
    assert "active_transfer" not in work.read(filename)
    assert work.read(work.parent_state(root))["active_call"] == 7
    assert not (work.call_dir(root) / "receipt.json").exists()
    with pytest.raises(work.RecoveryError, match="must_not_automatically_replay"):
        work.run(root, execute=True)


def test_later_original_call08_uses_last_slot_and_preserves_external_proof(fixture, monkeypatch):
    root, plan, filename = authorize(fixture); work.run(root, execute=True)
    originals.single_prepare(monkeypatch, fixture)
    result = work.parent.run(root, execute=True)
    assert result["calls_used"] == 8 and result["status"] == "V2_EXPLICIT_CALL_LIMIT_EXHAUSTED"
    assert work.verify_settlement(root)["remaining_V2_calls"] == 0
    assert work.parent.run(root, execute=True)["coordinator_calls_this_invocation"] == 0


@pytest.mark.parametrize("mutation", ["boot", "wallcap", "bodycap", "refund", "clock_backwards"])
def test_open_accounting_refuses_rebase_caps_or_inconsistent_charges(fixture, mutation):
    _, plan, filename = fixture
    opened = work.read(filename); now = work.core.clock_snapshot(); plan = copy.deepcopy(plan)
    if mutation == "boot": now["boot_id"] = "different"
    elif mutation == "wallcap": plan["budgets"]["wall_seconds"] = 0
    elif mutation == "bodycap": plan["budgets"]["r2_get_body_bytes"] = 1
    elif mutation == "refund": opened["charged"]["r2_get_body_bytes"] -= 1
    else: now["monotonic_ns"] = opened["wall_intent"]["monotonic_ns"] - 1
    with pytest.raises((work.RecoveryError, work.core.PreparationError)): work.open_accounting(opened, plan, now)


def process_tree(tmp_path, *, state="Z", command=b"", fdsize=0):
    folder = tmp_path / "14755"; folder.mkdir()
    # Fields after comm: state,ppid, then field5..22. starttime is index19.
    fields = [state, "1", *(["0"] * 17), "1234"]
    (folder / "stat").write_text("14755 (python) " + " ".join(fields))
    (folder / "cmdline").write_bytes(command)
    (folder / "status").write_text(f"State:\t{state}\nFDSize:\t{fdsize}\n")
    (folder / "fd").mkdir()
    return folder


def test_zombie_kernel_evidence_accepts_empty_command_and_FDSize0(tmp_path):
    process_tree(tmp_path)
    evidence = work.proc_record(14755, proc_root=tmp_path)
    assert evidence["status"] == "DEAD_ZOMBIE_NO_COMMAND_NO_FDS"
    assert evidence["FDSize"] == 0 and evidence["open_fd_count"] == 0


def test_zombie_unreadable_fd_is_reported_honestly_with_kernel_FDSize0(tmp_path, monkeypatch):
    folder = process_tree(tmp_path); original = Path.iterdir
    def listing(self):
        if self == folder / "fd": raise PermissionError()
        return original(self)
    monkeypatch.setattr(Path, "iterdir", listing)
    evidence = work.proc_record(14755, proc_root=tmp_path)
    assert evidence["status"] == "DEAD_ZOMBIE_NO_COMMAND_NO_FDS"
    assert evidence["fd_directory_readable"] is False and evidence["open_fd_count"] is None


@pytest.mark.parametrize("state,command,fdsize", [("R", b"", 0), ("Z", b"python", 0), ("Z", b"", 64)])
def test_live_or_unproven_zombie_is_never_treated_as_absent(tmp_path, state, command, fdsize):
    process_tree(tmp_path, state=state, command=command, fdsize=fdsize)
    if state == "R": assert work.proc_record(14755, proc_root=tmp_path)["status"] == "ALIVE_OR_UNPROVEN"
    else:
        with pytest.raises(work.RecoveryError): work.proc_record(14755, proc_root=tmp_path)


def test_active_worker_scan_matches_entry_without_exposing_arguments(tmp_path):
    folder = process_tree(tmp_path, state="R", command=b"python\0/workspace/research3/" + work.parent.ENTRY.encode() + b"\0--execute\0")
    assert work.active_original_workers(proc_root=tmp_path) == [14755]
