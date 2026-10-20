"""Synthetic software fixtures only: no real HTTP, R2, data, fit or admission."""
import copy
import fcntl
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace
import zlib

import pytest
import requests

from tools import prepare_yang_range_transport_supplement as work


def put(root, rel, value):
    filename = work.path(root, rel)
    work.core.atomic_json(filename, value)
    return filename


def state_file(root, rid):
    return work.audit_dir(root) / rid / "state.json"


def fake_complete(root, record):
    artifact = work.path(root, f"cache/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/{record['record_id']}/fixture.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"fixture")
    proof = {"path": artifact.relative_to(root).as_posix(), "sha256": hashlib.sha256(b"fixture").hexdigest(), "bytes": 7}
    state = {"status": "COMPLETE", "plan_sha256": work.PLAN_SHA, "record_id": record["record_id"],
             "synthetic_software_only": True, "artifacts": [proof],
             "raw_backup_receipt": proof, "cache_backup_receipt": proof}
    work.core.atomic_json(state_file(root, record["record_id"]), state)
    return state


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path
    for rel in (work.ENTRY, work.TESTS, work.parent.ENTRY):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((work.ROOT / rel).read_bytes())
    records = [{"record_id": f"fixture-{i}", "raw_files": [{"path": f"data/source_extensions_20261019/Yang2025/fixture-{i}/data.bdf",
        "sha256": "a" * 64, "bytes": 10, "download": {"kind": "zip_range", "url": work.OFFICIAL_URL,
        "archive_bytes": 1000, "header_offset": i * 50}}]} for i in range(3)]
    plan = {"plan_sha256": work.PLAN_SHA, "dataset": "Yang2025", "records": records, "record_count": 3,
        "budgets": {"provider_body_bytes": 100000, "r2_get_body_bytes": 100000,
                    "wall_seconds": 10000, "cache_bytes": 100000, "disk_reserve_bytes": 1},
        "fixture_scope": "NO_REAL_SCIENTIFIC_AUTHORITY"}
    plan_file = put(root, work.parent.PLAN_PATH, plan)
    amendment_file = put(root, work.parent.AMENDMENT_PATH, {"fixture_scope": "NO_REAL_V2_AUTHORITY"})
    ledger = work.core.Ledger(work.ledger_file(root), plan)
    ledger.checkpoint(close=True)
    after = put(root, work.parent.HISTORY + "/call-06/after_closed_ledger.json", work.read(work.ledger_file(root)))
    receipt = put(root, work.parent.HISTORY + "/call-06/receipt.json", {"call_number": 6,
        "error_type": "PreparationError", "error_code": "provider_did_not_honor_exact_range",
        "retryable_transport_error": True, "after_closed_ledger": work.ref(root, after)})
    parent_state = put(root, work.parent.HISTORY + "/recovery_ledger.json",
        {"calls_used": 6, "active_call": None, "terminal": None, "call_receipts": [work.ref(root, receipt)] * 6})
    monkeypatch.setattr(work, "PARENT_STATE_SHA", work.ref(root, parent_state)["sha256"])
    monkeypatch.setattr(work, "PARENT_RECEIPT_SHA", work.ref(root, receipt)["sha256"])
    monkeypatch.setattr(work, "PARENT_AFTER_SHA", work.ref(root, after)["sha256"])
    monkeypatch.setattr(work, "INITIAL_COMPLETE", 1)
    monkeypatch.setattr(work, "RECORD_COUNT", 3)
    monkeypatch.setattr(work.parent, "load_plan", lambda root: (root, plan, work.ref(root, plan_file)))
    monkeypatch.setattr(work.parent, "verify_static_authority", lambda *a: (work.read(amendment_file), work.ref(root, amendment_file)))
    monkeypatch.setattr(work.parent, "_history", lambda *a: (root, parent_state, work.read(parent_state), work.read(after)))
    def verified(root, state, plan, record):
        assert state["record_id"] == record["record_id"] and state["synthetic_software_only"] is True
        for item in state["artifacts"]:
            work.checked(root, item)
        return state["status"] == "COMPLETE"
    monkeypatch.setattr(work.core, "_state_verified", verified)
    fake_complete(root, records[0])
    put(root, f"audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/fixture-1/state.json",
        {"status": "FAILED_CLOSED", "record_id": "fixture-1", "plan_sha256": work.PLAN_SHA})
    return SimpleNamespace(root=root, plan=plan, records=records, parent_state=parent_state)


def authorize(fixture):
    work.write_amendment(fixture.root)
    return fixture


def actual_ledger(fixture):
    return work.core.Ledger(work.ledger_file(fixture.root), fixture.plan)


def budget(fixture, monkeypatch):
    return work.RetryBudget(fixture.root, {"path": "synthetic_amendment", "sha256": "b" * 64}, sleep=lambda delay: None)


def response(status, body=b"0123456789", start=100, total=1000, **headers):
    closed = []
    fields = {"Content-Length": str(len(body)), "Content-Range": f"bytes {start}-{start + len(body) - 1}/{total}",
              "Content-Encoding": "identity", **headers}
    return SimpleNamespace(status_code=status, headers=fields, close=lambda: closed.append(True),
        iter_content=lambda size: iter([body]), closed=closed)


def session(monkeypatch, outcomes):
    value = work.parent.previous.session()
    calls = []
    def get(url, **kwargs):
        calls.append((url, copy.deepcopy(kwargs)))
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(value, "get", get)
    return value, calls


def test_default_is_readonly_without_client_ledger_or_new_claim(fixture, monkeypatch):
    before = {str(p.relative_to(fixture.root)): p.read_bytes() for p in fixture.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(work.core, "Ledger", lambda *a: pytest.fail("mutable Ledger in dryrun"))
    monkeypatch.setattr(work.parent.previous, "session", lambda: pytest.fail("Session in dryrun"))
    assert work.run(fixture.root)["status"] == "DRY_RUN_ONE_SUPPLEMENT_NO_NETWORK"
    assert before == {str(p.relative_to(fixture.root)): p.read_bytes() for p in fixture.root.rglob("*") if p.is_file()}


def test_amendment_exact_ancestry_opaque_states_and_no_ledger_write(fixture):
    before = work.ledger_file(fixture.root).read_bytes()
    value = work.write_amendment(fixture.root)
    authority, item = work.verify_amendment(fixture.root, fixture.plan)
    assert value["amendment"] == item and authority["authorization"]["original_V2_call_limit_unchanged"] == 8
    assert authority["V2_state_opaque"]["path"].endswith(".txt")
    assert work.ledger_file(fixture.root).read_bytes() == before
    assert not (fixture.root / work.BASE / "execution_claim.json").exists()


@pytest.mark.parametrize("mutation", ["counter", "active", "terminal", "ledger", "complete_count"])
def test_changed_start_ancestry_or_complete_count_cannot_authorize(fixture, mutation):
    state = work.read(fixture.parent_state)
    if mutation == "counter": state["calls_used"] = 7
    elif mutation == "active": state["active_call"] = 6
    elif mutation == "terminal": state["terminal"] = "SOURCE_EXTENSION_CACHE_COMPLETE"
    elif mutation == "ledger":
        value = work.read(work.ledger_file(fixture.root)); value["elapsed_seconds"] += 1
        work.core.atomic_json(work.ledger_file(fixture.root), value)
    else: fake_complete(fixture.root, fixture.records[1])
    work.core.atomic_json(fixture.parent_state, state)
    with pytest.raises(work.SupplementError): work.write_amendment(fixture.root)
    assert not (fixture.root / work.AMENDMENT).exists()


@pytest.mark.parametrize("global_lock", [False, True])
def test_original_owner_or_global_lock_prevents_supplement(fixture, global_lock):
    name = ("audit/source_extensions_20261019/.source-extension-singleton.lock" if global_lock else work.parent.HISTORY + "/owner.lock")
    filename = work.path(fixture.root, name); filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((work.parent.TransportRecoveryError, work.parent.previous.TransportRecoveryError)):
            work.write_amendment(fixture.root)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_provider_white_status_retries_full_charge_same_original_range(fixture, monkeypatch, status):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    bad, good = response(status), response(206)
    base, calls = session(monkeypatch, [bad, good])
    item = fixture.records[0]["raw_files"][0]
    result = work.core._range(work.OFFICIAL_URL, 100, 10, 1000,
        session=work.RangeSession(base, ledger, item, retry), ledger=ledger)
    assert result == b"0123456789" and len(calls) == 2 and calls[0] == calls[1]
    assert bad.closed and good.closed
    assert ledger.data["charged"]["provider_body_bytes"] == 20
    assert [r["status"] for r in ledger.data["history"]] == ["FAILURE_FULL_RESERVATION_CONSUMED", "SUCCESS_FULL_BODY_VERIFIED"]
    work.verify_retry_proof(fixture.root, work.ref(fixture.root, retry.filename), ledger.data)
    ledger.checkpoint(close=True)


def test_each_range_has_at_most_four_logical_attempts(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    base, calls = session(monkeypatch, [response(503) for _ in range(5)])
    with pytest.raises(work.core.PreparationError, match="exact_range"):
        work.core._range(work.OFFICIAL_URL, 100, 10, 1000,
            session=work.RangeSession(base, ledger, fixture.records[0]["raw_files"][0], retry), ledger=ledger)
    assert len(calls) == 4 and retry.data["extra_logical_retries"] == 3
    assert ledger.data["charged"]["provider_body_bytes"] == 40
    ledger.checkpoint(close=True)


def test_sixtyfour_extra_logical_retries_are_shared_and_durable(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    for ordinal in range(64):
        key = f"fixture-{ordinal}"
        retry.begin(key); ledger.reserve("provider_body_bytes" if ordinal % 2 else "r2_get_body_bytes", 1, key)
        reservation = dict(ledger.data["active_transfer"])
        retry.retry(key, ledger=ledger, failure={"http_status": 503}, reservation=reservation)
        ledger.finish(verified_bytes=1)
    assert retry.data["extra_logical_retries"] == 64
    retry.begin("last"); ledger.reserve("provider_body_bytes", 1, "last")
    assert not retry.available("last")
    with pytest.raises(work.SupplementError, match="allowance_exhausted"):
        retry.retry("last", ledger=ledger, failure={"http_status": 503}, reservation=dict(ledger.data["active_transfer"]))
    ledger.finish(); ledger.checkpoint(close=True)
    work.verify_retry_proof(fixture.root, work.ref(fixture.root, retry.filename), work.read(work.ledger_file(fixture.root)))
    with pytest.raises(work.SupplementError, match="never_reset"):
        work.RetryBudget(fixture.root, retry.amendment_ref)


@pytest.mark.parametrize("outcome", [response(206, **{"Content-Range": "WRONG"}), response(200), response(403),
    requests.exceptions.SSLError("PRIVATE"), requests.exceptions.ProxyError("PRIVATE"),
    requests.exceptions.ReadTimeout("PRIVATE"), ValueError("PRIVATE")])
def test_nonwhite_wrong206_SSL_Proxy_unknown_are_not_retried(fixture, monkeypatch, outcome):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    base, calls = session(monkeypatch, [outcome])
    with pytest.raises(Exception):
        work.core._range(work.OFFICIAL_URL, 100, 10, 1000,
            session=work.RangeSession(base, ledger, fixture.records[0]["raw_files"][0], retry), ledger=ledger)
    assert len(calls) == 1 and retry.data["extra_logical_retries"] == 0
    assert ledger.data["charged"]["provider_body_bytes"] == 10 and "active_transfer" not in ledger.data
    ledger.checkpoint(close=True)


@pytest.mark.parametrize("changed", ["URL", "Range", "encoding", "trust_env", "NoCache", "adapter", "identity"])
def test_proxy_rejects_changed_request_or_reservation_before_HTTP(fixture, monkeypatch, changed):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    base, calls = session(monkeypatch, [response(206)])
    ledger.reserve("provider_body_bytes", 10, "range:100:10")
    url = work.OFFICIAL_URL; kwargs = {"headers": {"Range": "bytes=100-109", "Accept-Encoding": "identity"},
                                     "stream": True, "timeout": (20, 120)}
    if changed == "URL": url += "?no"
    elif changed == "Range": kwargs["headers"]["Range"] = "bytes=101-110"
    elif changed == "encoding": kwargs["headers"]["Accept-Encoding"] = "gzip"
    elif changed == "trust_env": base.trust_env = False
    elif changed == "NoCache": base.headers["Pragma"] = "cache"
    elif changed == "adapter": base.adapters["https://"].max_retries.total = 1
    else: ledger.data["active_transfer"]["identity"] = "wrong"
    with pytest.raises(work.SupplementError):
        work.RangeSession(base, ledger, fixture.records[0]["raw_files"][0], retry).get(url, **kwargs)
    assert not calls
    ledger.finish(); ledger.checkpoint(close=True)


def test_retry_budget_and_backoff_failure_cannot_refund_or_replay(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    retry.begin("fixture"); ledger.reserve("provider_body_bytes", 5, "range:0:5")
    def interrupt(delay): raise KeyboardInterrupt()
    retry.sleep = interrupt
    with pytest.raises(KeyboardInterrupt):
        retry.retry("fixture", ledger=ledger, failure={"http_status": 503}, reservation=dict(ledger.data["active_transfer"]))
    assert retry.data["extra_logical_retries"] == 1 and retry.data["active_retry"] == 1
    assert ledger.data["charged"]["provider_body_bytes"] == 5 and "active_transfer" not in ledger.data
    ledger.checkpoint(close=True)
    with pytest.raises(work.SupplementError): work.verify_retry_proof(fixture.root, work.ref(fixture.root, retry.filename), ledger.data)


def test_successful_original_zip_inflate_CRC_SHA_is_untouched(fixture, monkeypatch):
    native = b"SYNTHETIC NATIVE BYTES" * 30
    compressor = zlib.compressobj(wbits=-15); compressed = compressor.compress(native) + compressor.flush()
    member = b"synthetic/data.bdf"; crc = zlib.crc32(native) & 0xffffffff
    header = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, crc, len(compressed), len(native), len(member), 0)
    archive = header + member + compressed
    item = {"path": "data/source_extensions_20261019/Yang2025/synthetic/data.bdf", "bytes": len(native),
        "sha256": hashlib.sha256(native).hexdigest(), "crc32": f"{crc:08x}", "download": {"kind": "zip_range",
        "url": work.OFFICIAL_URL, "archive_bytes": len(archive), "header_offset": 0, "member": member.decode(),
        "compression": 8, "compressed_bytes": len(compressed)}}
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    base = work.parent.previous.session(); calls = []
    def get(url, **kwargs):
        start, end = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
        calls.append((start, end))
        if len(calls) == 1: return response(503)
        return response(206, archive[start:end+1], start=start, total=len(archive))
    monkeypatch.setattr(base, "get", get)
    target = fixture.root / "synthetic.bdf"
    result = work.download_with_retries(item, target, session=base, ledger=ledger, budget=retry)
    assert target.read_bytes() == native and result["sha256"] == item["sha256"] and len(calls) == 4
    assert ledger.data["charged"]["provider_body_bytes"] == len(archive) + 30
    ledger.checkpoint(close=True)


def test_R2_client_disables_internal_retries_without_recording_environment(monkeypatch):
    import boto3
    captured = {}
    monkeypatch.setenv("R2_ENDPOINT", "https://synthetic.invalid")
    def client(service, **kwargs): captured.update(kwargs); return "synthetic-client"
    monkeypatch.setattr(boto3, "client", client)
    assert work.create_r2_client() == "synthetic-client"
    assert captured["config"].retries["total_max_attempts"] == 1
    assert captured["config"].connect_timeout == 15 and captured["config"].read_timeout == 120
    assert captured["region_name"] == "auto"


def backup_case(fixture):
    root = fixture.root; folder = work.audit_dir(root) / "fixture-1"
    artifact = work.path(root, f"cache/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/fixture-1/metadata.json")
    artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_bytes(b"305")
    entry = {**work.ref(root, artifact), "dataset": "Yang2025", "role": "result"}
    manifest = folder / "cache_backup_manifest.json"; work.core.atomic_json(manifest, {"schema_version": 1, "files": [entry]})
    receipt = folder / "cache_backup_receipt.json"
    prefix = f"paper3/source-extensions-20261019/Yang2025/{work.PLAN_SHA}/fixture-1/cache/"
    base = {"operation": "paper3_explicit_manifest_backup", "status": "failed", "execute": True,
        "source_extension_profile": work.storage.profile(), "manifest_sha256": work.core.sha_file(manifest)[0],
        "prefix": prefix, "error_type": "BackupError", "error_code": "object_backup_not_verified",
        "objects": [{**entry, "status": "failed", "put": {"status": "failed", "http_status": 503, "error_type": "ClientError"}}]}
    return manifest, receipt, prefix, entry, base


def test_R2_retry_preserves_prior_failed_receipts_full_charge_and_same_manifest(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    manifest, receipt, prefix, entry, failed = backup_case(fixture)
    work.core.atomic_json(receipt, failed); initial = receipt.read_bytes()
    successful = copy.deepcopy(failed); successful.update(status="verified")
    successful["objects"][0].update(status="verified", put={"status": "created", "http_status": 200},
        get={"status": "verified", "http_status": 200, "complete_body_read": True, "bytes_read": 3, "sha256": entry["sha256"]})
    outcomes = [failed, successful]; calls = []
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(work, "create_r2_client", lambda: client)
    def backup(**kwargs):
        assert kwargs["client"] is client and kwargs["prefix"] == prefix
        calls.append(kwargs); value = outcomes.pop(0); work.core.atomic_json(receipt, value); return value
    monkeypatch.setattr(work.storage, "run_backup", backup)
    ledger.reserve("r2_get_body_bytes", 3, prefix)
    wrapper = work.BackupWithRetries(fixture.root, ledger, retry)
    assert wrapper(manifest_path=manifest, root=fixture.root, prefix=prefix, receipt_path=receipt, execute=True)["status"] == "verified"
    ledger.finish(verified_bytes=3)
    copies = list((fixture.root / work.BASE / "backup-history").rglob("*.txt"))
    assert len(calls) == 2 and any(p.read_bytes() == initial for p in copies)
    assert len(retry.data["IO_receipt_snapshots"]) == 4
    from tools.build_data_plan_deploy_manifest import Closure
    closure = Closure(fixture.root)
    # Walk just the opaque archive inventory; the synthetic amendment in this
    # low-level fixture deliberately has no scientific authority descriptor.
    closure.walk(retry.data["IO_receipt_snapshots"])
    assert all(item["opaque_snapshot"]["path"] in closure.files for item in retry.data["IO_receipt_snapshots"])
    assert ledger.data["charged"]["r2_get_body_bytes"] == 6
    work.verify_retry_proof(fixture.root, work.ref(fixture.root, retry.filename), ledger.data)
    ledger.checkpoint(close=True)


@pytest.mark.parametrize("mutation", ["generic", "SSL", "Proxy", "integrity", "206", "unrelated", "missing_status"])
def test_R2_unknown_nonwhite_and_integrity_failures_do_not_retry(fixture, mutation):
    manifest, receipt, prefix, entry, failed = backup_case(fixture)
    if mutation == "generic": failed["objects"] = []
    elif mutation == "unrelated": failed["objects"][0]["path"] = "other"
    elif mutation == "missing_status": failed["objects"][0]["put"].pop("http_status")
    elif mutation == "206": failed["objects"][0]["put"]["http_status"] = 206
    else:
        failed["objects"][0]["put"]["error_type"] = {"SSL": "SSLError", "Proxy": "ProxyError", "integrity": "BackupError"}[mutation]
        failed["objects"][0]["put"]["error_code"] = "remote_integrity_mismatch"
    assert work.retryable_backup_failure(failed, manifest_sha=work.core.sha_file(manifest)[0], prefix=prefix, files=[entry]) is None


def fake_science(monkeypatch, fixture, *, error=None):
    calls = []
    monkeypatch.setattr(work.core, "prepare", lambda *a, **k: pytest.fail("must not call prepare"))
    monkeypatch.setattr(work.core, "assemble_manifest", lambda *a, **k: pytest.fail("must not assemble/finalize"))
    monkeypatch.setattr(work.core, "validate_plan", lambda *a: {"synthetic_native_profile": True})
    real_import = work.importlib.import_module
    monkeypatch.setattr(work.importlib, "import_module", lambda name: SimpleNamespace(materialize=lambda *a: pytest.fail("fake science"))
        if name == work.core.ADAPTERS["Yang2025"] else real_import(name))
    def process(root, plan, record, **kwargs):
        calls.append(record["record_id"])
        assert kwargs["audit_native"] is None and callable(kwargs["download_fn"])
        if error: raise error
        return fake_complete(root, record)
    monkeypatch.setattr(work.core, "process_record", process)
    return calls


def test_single_supplement_complete_proof_does_not_finalize_or_change_V2_counter(fixture, monkeypatch):
    authorize(fixture); calls = fake_science(monkeypatch, fixture)
    prior = fixture.parent_state.read_bytes()
    result = work.run(fixture.root, execute=True)
    assert result["verified_records"] == 3 and result["original_V2_finalization_still_required"] is True
    assert calls == [r["record_id"] for r in fixture.records] and fixture.parent_state.read_bytes() == prior
    assert not (work.audit_dir(fixture.root) / "coordinator_status.json").exists()
    ledger = work.read(work.ledger_file(fixture.root))
    assert "wall_intent" not in ledger and "active_transfer" not in ledger
    assert "explicit_range_transport_completion" in ledger
    completed = work.read(work.path(fixture.root, work.BASE + "/completion.json"))
    assert completed["original_V2_call06_state_opaque"]["path"].endswith(".txt")
    assert "explicit_range_transport_completion" not in work.read(work.checked(fixture.root, completed["after_closed_ledger_before_completion_pointer"]))
    monkeypatch.setattr(work.core, "Ledger", lambda *a, **k: pytest.fail("offline Ledger"))
    monkeypatch.setattr(work.parent.previous, "session", lambda: pytest.fail("offline Session"))
    assert work.run(fixture.root, execute=True)["new_HTTP_or_R2_requests"] == 0


def test_failed_or_unknown_supplement_never_replays_or_claims_completion(fixture, monkeypatch):
    authorize(fixture); calls = fake_science(monkeypatch, fixture, error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt): work.run(fixture.root, execute=True)
    assert (fixture.root / work.BASE / "execution_claim.json").exists()
    assert not (fixture.root / work.BASE / "completion.json").exists()
    failure = work.read(fixture.root / work.BASE / "failure.json")
    assert failure["status"] == "FAILED_SINGLE_SUPPLEMENT_CLOSED_NO_REPLAY"
    with pytest.raises(work.SupplementError): work.run(fixture.root, execute=True)
    assert len(calls) == 1 and "wall_intent" not in work.read(work.ledger_file(fixture.root))


def test_completion_roster_and_retry_counter_tampering_are_rejected(fixture, monkeypatch):
    authorize(fixture); fake_science(monkeypatch, fixture); work.run(fixture.root, execute=True)
    filename = state_file(fixture.root, "fixture-1"); state = work.read(filename); state["status"] = "FAILED_CLOSED"
    work.core.atomic_json(filename, state)
    with pytest.raises(work.SupplementError): work.verify_completion(fixture.root)


def test_cumulative_cache_gate_remains_original(fixture, monkeypatch):
    authorize(fixture); fake_science(monkeypatch, fixture); fixture.plan["budgets"]["cache_bytes"] = 10
    with pytest.raises(work.SupplementError, match="cache_budget"):
        work.run(fixture.root, execute=True)
    assert not (fixture.root / work.BASE / "completion.json").exists()


def test_malformed_existing_backup_receipt_is_preserved_and_fails_closed(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    manifest, receipt, prefix, _, _ = backup_case(fixture)
    receipt.write_bytes(b"{TRUNCATED HISTORICAL FAILURE")
    monkeypatch.setattr(work, "create_r2_client", lambda: SimpleNamespace(close=lambda: None))
    # Original profiled IO returns a sanitized preflight error, no real client operation.
    ledger.reserve("r2_get_body_bytes", 3, prefix)
    wrapper = work.BackupWithRetries(fixture.root, ledger, retry)
    result = wrapper(manifest_path=manifest, root=fixture.root, prefix=prefix, receipt_path=receipt, execute=True)
    assert result["status"] == "failed" and retry.data["extra_logical_retries"] == 0
    assert receipt.read_bytes() == b"{TRUNCATED HISTORICAL FAILURE"
    assert any(p.read_bytes() == receipt.read_bytes() for p in (fixture.root / work.BASE / "backup-history").rglob("*.txt"))
    ledger.finish(); ledger.checkpoint(close=True)


def test_original_restore_gets_same_retry_off_client_without_retry_loop(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    client = SimpleNamespace(close=lambda: None); calls = []
    monkeypatch.setattr(work, "create_r2_client", lambda: client)
    report = work.audit_dir(fixture.root) / "fixture-1/raw_restore_report.json"
    def restore(**kwargs):
        calls.append(kwargs)
        assert kwargs["client"] is client and kwargs["require_fresh"] is True
        work.core.atomic_json(report, {"status": "failed", "http_status": 503})
        return {"status": "failed"}
    monkeypatch.setattr(work.storage, "run_restore", restore)
    wrapper = work.BackupWithRetries(fixture.root, ledger, retry)
    assert wrapper.restore(root=fixture.root, report_path=report, execute=True, require_fresh=True,
        backup_receipt_path=work.audit_dir(fixture.root) / "fixture-1/raw_backup_receipt.json", paths=["synthetic"])["status"] == "failed"
    assert len(calls) == 1 and retry.data["extra_logical_retries"] == 0
    ledger.checkpoint(close=True)


def test_global_COMPLETE_write_is_rejected_even_if_all_records_exist(fixture, monkeypatch):
    authorize(fixture); fake_science(monkeypatch, fixture)
    original = work.core.process_record
    def bad_process(*args, **kwargs):
        value = original(*args, **kwargs)
        work.core.atomic_json(work.audit_dir(fixture.root) / "coordinator_status.json", {"status": "SOURCE_EXTENSION_CACHE_COMPLETE"})
        return value
    monkeypatch.setattr(work.core, "process_record", bad_process)
    with pytest.raises(work.SupplementError, match="global_aggregation"):
        work.run(fixture.root, execute=True)
    assert not (fixture.root / work.BASE / "completion.json").exists()


def test_offline_supplement_proof_survives_later_original_V2_call07_and_closure(fixture, monkeypatch):
    authorize(fixture); fake_science(monkeypatch, fixture); work.run(fixture.root, execute=True)
    state = work.read(fixture.parent_state); state["calls_used"] = 7
    state["terminal"] = "SOURCE_EXTENSION_CACHE_COMPLETE"
    work.core.atomic_json(fixture.parent_state, state)
    assert work.verify_completion(fixture.root)["verified_records"] == 3
    from tools.build_data_plan_deploy_manifest import Closure
    closure = Closure(fixture.root)
    closure.path(work.ledger_file(fixture.root).relative_to(fixture.root).as_posix())
    assert not closure.errors
    assert work.BASE + "/completion.json" in closure.files
    assert work.parent.HISTORY + "/recovery_ledger.json" not in closure.files


def test_original_budget_exhaustion_stops_retry_before_another_GET(fixture, monkeypatch):
    fixture.plan["budgets"]["provider_body_bytes"] = 10
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    base, calls = session(monkeypatch, [response(503), response(206)])
    with pytest.raises(work.core.PreparationError, match="body_budget_exhausted"):
        work.core._range(work.OFFICIAL_URL, 100, 10, 1000,
            session=work.RangeSession(base, ledger, fixture.records[0]["raw_files"][0], retry), ledger=ledger)
    assert len(calls) == 1 and ledger.data["charged"]["provider_body_bytes"] == 10
    ledger.checkpoint(close=True)


def test_forged_logical_counter_with_same_total_is_rejected(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    retry.begin("real"); ledger.reserve("provider_body_bytes", 1, "real")
    retry.retry("real", ledger=ledger, failure={"http_status": 503}, reservation=dict(ledger.data["active_transfer"]))
    ledger.finish(verified_bytes=1); ledger.checkpoint(close=True)
    retry.data["logical_attempts"] = {"real": 1, "forged": 2}; retry.save()
    with pytest.raises(work.SupplementError):
        work.verify_retry_proof(fixture.root, work.ref(fixture.root, retry.filename), ledger.data)


def test_R2_same_manifest_has_at_most_four_logical_backup_calls(fixture, monkeypatch):
    ledger = actual_ledger(fixture); retry = budget(fixture, monkeypatch)
    manifest, receipt, prefix, _, failed = backup_case(fixture)
    monkeypatch.setattr(work, "create_r2_client", lambda: SimpleNamespace(close=lambda: None))
    calls = []
    def backup(**kwargs):
        calls.append(1); work.core.atomic_json(receipt, failed); return failed
    monkeypatch.setattr(work.storage, "run_backup", backup)
    ledger.reserve("r2_get_body_bytes", 3, prefix)
    result = work.BackupWithRetries(fixture.root, ledger, retry)(manifest_path=manifest, root=fixture.root,
        prefix=prefix, receipt_path=receipt, execute=True)
    assert result["status"] == "failed" and len(calls) == 4 and retry.data["extra_logical_retries"] == 3
    ledger.finish(); ledger.checkpoint(close=True)
    assert ledger.data["charged"]["r2_get_body_bytes"] == 12


def test_strict_closed_proof_rejects_intent_keys_even_when_None(fixture):
    value = work.read(work.ledger_file(fixture.root)); value["wall_intent"] = None
    work.core.atomic_json(work.ledger_file(fixture.root), value)
    with pytest.raises(work.SupplementError, match="no_intent_keys"):
        work.closed(fixture.root, fixture.plan)


def test_unchanged_V2_actual_history_accepts_additional_closed_record_preparation_before_call07(tmp_path, monkeypatch):
    from tests import test_resume_yang_source_transport_v2 as originals
    root, plan, filename = originals.fixture_root.__wrapped__(tmp_path, monkeypatch)
    originals.authorize((root, plan, filename))
    originals.single_prepare(monkeypatch, (root, plan, filename),
        error=work.core.PreparationError("provider_did_not_honor_exact_range"))
    for _ in range(6):
        work.parent.run(root, execute=True)
    original_state = work.path(root, work.parent.HISTORY + "/recovery_ledger.json").read_bytes()
    ledger = work.core.Ledger(filename, plan)
    ledger.reserve("provider_body_bytes", 5, "SYNTHETIC_SUPPLEMENT_RECORD_RANGE")
    ledger.finish(verified_bytes=5); ledger.checkpoint(close=True)
    value = work.read(filename); value["explicit_range_transport_amendment"] = {"fixture_scope": "NO_REAL_AUTHORITY"}
    work.core.atomic_json(filename, value)
    assert work.path(root, work.parent.HISTORY + "/recovery_ledger.json").read_bytes() == original_state
    assert work.parent.run(root, execute=True)["calls_used"] == 7
    _, amendment_ref = work.parent.verify_static_authority(root, plan)
    assert work.parent._history(root, plan, amendment_ref)[2]["calls_used"] == 7


def test_global_fingerprint_detects_real_original_final_backup_names(fixture):
    ledger = actual_ledger(fixture)
    record = fixture.records[0]
    state = work.read(state_file(fixture.root, record["record_id"]))
    artifact = work.checked(fixture.root, state["artifacts"][0])
    before = work.global_aggregation_fingerprints(fixture.root)
    captured = []
    def no_network_backup(**kwargs):
        captured.append(kwargs)
        assert kwargs["manifest_path"] == work.audit_dir(fixture.root) / "final_backup_manifest.json"
        assert kwargs["receipt_path"] == work.audit_dir(fixture.root) / "final_backup_receipt.json"
        work.core.atomic_json(kwargs["receipt_path"], {"status": "failed", "fixture_scope": "NO_ACTUAL_R2"})
        return {"status": "failed"}
    with pytest.raises(work.core.PreparationError, match="full_get_sha_not_verified"):
        work.core.verified_backup(fixture.root, work.audit_dir(fixture.root) / "final_backup",
            dataset="Yang2025", files=[(artifact, "result")],
            prefix=f"paper3/source-extensions-20261019/Yang2025/{work.PLAN_SHA}/final/",
            ledger=ledger, backup_fn=no_network_backup)
    after = work.global_aggregation_fingerprints(fixture.root)
    changed = [row["source_at_snapshot"] for old, row in zip(before, after)
               if old["sha256_at_snapshot"] != row["sha256_at_snapshot"]]
    assert len(captured) == 1
    assert changed == [f"audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/final_backup_manifest.json",
                       f"audit/source_extensions_20261019/Yang2025/{work.PLAN_SHA}/final_backup_receipt.json"]
    ledger.checkpoint(close=True)
