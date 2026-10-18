"""Non-EEG mathematical software fixtures only; never real source caches or fits.

Temporary source identities/HTTP receipts below are test doubles. They do not
attest that real W1, a real source inventory, or a scientific W2 run completed.
"""
import importlib.util
import fcntl
import io
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from p3work.adapters.source_native import CHO_EEG_CHANNELS, _physionet, audit_native_source, iter_native_contexts
from p3work.contracts import CHANNELS, digest, file_digest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper3_prepare_source_cache", ROOT / "tools/prepare_source_cache.py")
cache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache
spec.loader.exec_module(cache)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def frozen_contract():
    return {"status": "window_frozen", "channels": list(CHANNELS), "sfreq_hz": 128,
            "selected_profile": "long", "context_s": [0, 3], "crop_s": [.5, 2.5], "n_times": 256,
            "window_audit_receipt": {"decision_pool_role": "meta_train", "split_sha256": "a" * 64,
                "raw_metadata_manifest_sha256": "b" * 64,
                "long": {"participant_mean_coverage": .9, "participant_mean_abs_class_exclusion_gap": .02},
                "short": {"participant_mean_coverage": .95, "participant_mean_abs_class_exclusion_gap": .01}}}


def mathematical_record(path, *, intertrial_value=None):
    """Simple deterministic sinusoids packaged as a native-structure test double."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(4000, dtype=np.float64) / 250
    x = np.stack([np.sin(2 * np.pi * (9 + ch / 3) * t) + .2 * np.cos(2 * np.pi * 17 * t + ch)
                  for ch in range(25)], axis=1)
    if intertrial_value is not None:
        x[1500:1700] = intertrial_value
    savemat(path, {"data": {"X": x, "trial": np.array([1, 1701]), "y": np.array([1, 2]),
                          "fs": 250, "classes": np.array(["left hand", "right hand", "feet", "tongue"], dtype=object),
                          "artifacts": np.array([0, 1]), "gender": "fixture", "age": 0}})
    return audit_native_source("BNCI2014_001", path, provider_subject="1", session_id="T",
                               source_version="STRUCTURE_FIXTURE_NOT_EEG", expected_sha256=file_digest(path))


@pytest.fixture
def partial_fixture(tmp_path):
    root = tmp_path
    write(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "NOT_STARTED"}})
    write(root / "contracts/input_contract.json", frozen_contract())
    for relative in ("p3work/adapters/source_native.py", "p3work/preprocess.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    raw = root / "data/math_fixture/A01T.mat"
    result = mathematical_record(raw)
    metadata = root / "manifests/math_fixture_native.jsonl"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("".join(json.dumps(row) + "\n" for row in result["trials"]))
    write(root / "manifests/terms_fixture.json", {"scope": "TEST_DOUBLE_ONLY_NO_LICENSE_ATTESTATION"})
    descriptor = lambda p: {"path": str(p.relative_to(root)), "sha256": file_digest(p)}
    item = {**descriptor(raw), "dataset": "BNCI2014_001", "provider_subject": "1", "session_id": "T",
            "file": raw.name, "version": "STRUCTURE_FIXTURE_NOT_EEG", "bytes": raw.stat().st_size,
            "response_content_length": raw.stat().st_size, "complete_download": True,
            "http_status": 200, "native_schema_audited": True,
            "url": cache.expected_source_objects()["BNCI2014_001", "1", "T", raw.name],
            "native_trials": descriptor(metadata)}
    inventory = {"schema_version": "paper3-source-native-inventory-v2", "role": "source_train",
                 "status": "SOURCE_RAW_INVENTORY_PARTIAL", "synthetic": False, "old_npz_reused": False,
                 "provider_roster_verified": False, "files": [item],
                 "provider_terms_evidence": descriptor(root / "manifests/terms_fixture.json")}
    write(root / "manifests/source_fixture.json", inventory)
    return root, inventory, result


def test_w1_block_happens_before_inventory_or_native_access(tmp_path, monkeypatch, capsys):
    write(tmp_path / "status/tasks.json", {"W1": {"status": "IN_PROGRESS"}})
    monkeypatch.setattr(cache, "iter_native_contexts", lambda *a, **k: pytest.fail("native signal accessed before W1"))
    assert cache.main(["--root", str(tmp_path), "--execute", "--partial-scope"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert "W1 must be COMPLETE" in result["reason"]
    assert result["numerical_cache_created"] is False and not (tmp_path / "cache").exists()


def test_declaring_partial_source_inventory_complete_cannot_complete_w2(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    inventory.update(status="SOURCE_RAW_INVENTORY_COMPLETE", provider_roster_verified=True)
    write(root / "manifests/source_fixture.json", inventory)
    monkeypatch.setattr(cache, "verify_frozen_window", lambda root: {"SOFTWARE_TEST_DOUBLE": True})
    with pytest.raises(cache.SourceCacheGateError, match="1/505"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True)
    assert not (root / "cache").exists()


def test_unfrozen_window_blocks_even_explicit_partial_cache(partial_fixture):
    root, _, _ = partial_fixture
    c = frozen_contract(); c["status"] = "pending"
    write(root / "contracts/input_contract.json", c)
    with pytest.raises(ValueError, match="not frozen"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    assert not (root / "cache").exists()


def test_partial_dry_run_never_calls_native_context_reader(partial_fixture, monkeypatch):
    root, _, _ = partial_fixture
    monkeypatch.setattr(cache, "iter_native_contexts", lambda *a, **k: pytest.fail("dry run generated numeric cache"))
    result = cache.prepare_cache(root, "manifests/source_fixture.json", partial_scope=True)
    assert result["would_emit_status"] == "PILOT_CACHE_PARTIAL" and result["numerical_cache_created"] is False
    assert not (root / "cache").exists()


def test_mathematical_partial_shards_bind_raw_code_contract_and_artifacts(partial_fixture):
    root, _, _ = partial_fixture
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    manifest_path = root / result["cache_manifest"]["path"]
    assert file_digest(manifest_path) == result["cache_manifest"]["sha256"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "PILOT_CACHE_PARTIAL" and result["formal_w2_complete"] is False
    assert manifest["full_source_inventory_verified"] is False and manifest["training_started"] is False
    shard = manifest["shards"][0]
    for key in ("X", "y", "events", "exclusions", "adapter_code", "processing_code"):
        assert file_digest(root / shard[key]["path"]) == shard[key]["sha256"]
    x = np.load(root / shard["X"]["path"], allow_pickle=False)
    y = np.load(root / shard["y"]["path"], allow_pickle=False)
    assert x.shape == (2, 11, 256) and x.dtype == np.float32
    np.testing.assert_array_equal(y, [0, 1])
    np.testing.assert_allclose(x.mean(1), 0, atol=2e-13)
    np.testing.assert_allclose(x.mean(2), 0, atol=2e-13)
    rows = json.loads((root / shard["events"]["path"]).read_text())
    assert all(row["role"] == "source_train" and row["task_end_verified"] for row in rows)
    assert all(row["preprocessing_qc"]["normalization_applied"] is False for row in rows)
    assert all(row["native_unit_string"] is None and row["effective_numeric_unit"] == "uV" for row in rows)
    assert all(row["unit_evidence_kind"] == "fixed_adapter_numeric_unit_declaration_not_native_header" for row in rows)
    assert rows[0]["original_context_samples"] == [500, 1250]
    assert json.loads((root / "status/tasks.json").read_text())["W2"]["status"] == "NOT_STARTED"


def test_original_task_boundaries_exclude_intertrial_future_values(tmp_path):
    first_path, second_path = tmp_path / "a/A01T.mat", tmp_path / "b/A01T.mat"
    first, second = mathematical_record(first_path), mathematical_record(second_path, intertrial_value=np.inf)
    first_contexts = list(iter_native_contexts("BNCI2014_001", first_path, rows=first["trials"],
        contract=frozen_contract(), expected_sha256=file_digest(first_path)))
    second_contexts = list(iter_native_contexts("BNCI2014_001", second_path, rows=second["trials"],
        contract=frozen_contract(), expected_sha256=file_digest(second_path)))
    for a, b in zip(first_contexts, second_contexts):
        np.testing.assert_array_equal(a["signal"], b["signal"])
        assert a["signal"].shape == (11, 750)


def test_native_metadata_cannot_extend_task_or_relabel_original_trial(partial_fixture):
    root, inventory, result = partial_fixture
    raw = root / inventory["files"][0]["path"]
    rows = result["trials"]
    rows[0]["task_end_sample"] += 100000
    with pytest.raises(ValueError, match="metadata differs"):
        list(iter_native_contexts("BNCI2014_001", raw, rows=rows, contract=frozen_contract(),
                                  expected_sha256=file_digest(raw)))


def test_legacy_npz_and_unregistered_dataset_inventory_rejected(partial_fixture):
    root, inventory, _ = partial_fixture
    inventory["files"][0]["path"] = "data/old_cache.npz"
    write(root / "manifests/source_fixture.json", inventory)
    with pytest.raises(cache.SourceCacheGateError, match="Old NPZ"):
        cache.prepare_cache(root, "manifests/source_fixture.json", partial_scope=True)
    inventory["files"][0]["dataset"] = "Wang2026"
    write(root / "manifests/source_fixture.json", inventory)
    with pytest.raises(cache.SourceCacheGateError, match="Unregistered"):
        cache.prepare_cache(root, "manifests/source_fixture.json", partial_scope=True)


def test_completed_participant_resume_uses_hashes_without_rereading_signals(partial_fixture, monkeypatch):
    root, _, _ = partial_fixture
    first = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    monkeypatch.setattr(cache, "_prepare_shard", lambda *a, **k: pytest.fail("completed participant regenerated"))
    second = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    assert first["cache_manifest"] == second["cache_manifest"]
    manifest = json.loads((root / first["cache_manifest"]["path"]).read_text())
    x = root / manifest["shards"][0]["X"]["path"]
    x.write_bytes(b"CORRUPTED_TEST_DOUBLE")
    with pytest.raises(cache.SourceCacheGateError, match="SHA256 receipt"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)


def make_r2_only_fixture(root, inventory):
    """A fully explicit mocked backup receipt; never contacts R2."""
    item = inventory["files"][0]
    old = root / item["path"]
    body = old.read_bytes()
    item["object_uid"] = digest([item["dataset"], str(item["provider_subject"]), str(item["session_id"]), item["file"]])
    item["path"] = "data/source_preparation/staging/" + item["object_uid"] + "/A01T.mat"
    item.update(raw_storage="verified_r2_only", temporary_raw_owned=True)
    old.unlink()
    prefix = "paper3/source-test-fixture/"
    entry = {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"],
             "role": "source_development_raw", "dataset": item["dataset"],
             "key": prefix + item["path"], "status": "verified",
             "put": {"status": "created", "http_status": 200, "request_body_bytes": item["bytes"],
                     "request_body_sha256": item["sha256"]},
             "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                     "bytes_read": item["bytes"], "sha256": item["sha256"], "declared_content_length": item["bytes"]}}
    receipt = root / "audit/source_full/mock_backup_receipt.json"
    write(receipt, {"schema_version": 1, "operation": "paper3_explicit_manifest_backup",
                    "status": "verified", "execute": True, "verification": "full_get_body_sha256",
                    "prefix": prefix, "objects": [entry]})
    item["backup_receipt"] = {"path": str(receipt.relative_to(root)), "sha256": file_digest(receipt)}
    write(root / "manifests/source_fixture.json", inventory)
    return body, item


def test_r2_raw_restore_is_one_exact_object_and_removed_only_after_hash_check(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    calls = []
    def restore(**kwargs):
        calls.append(kwargs)
        assert kwargs["paths"] == [item["path"]] and kwargs["execute"] is True
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return {"status": "restored_verified"}
    monkeypatch.setattr(cache, "run_restore", restore)
    with pytest.raises(cache.SourceCacheGateError, match="explicit --restore-backed-raw"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True,
                                 restore_backed_raw=True)
    assert result["status"] == "PILOT_CACHE_PARTIAL" and len(calls) == 1
    assert not (root / item["path"]).exists()


def test_r2_failure_keeps_w2_incomplete_and_has_no_cache_manifest(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    _, item = make_r2_only_fixture(root, inventory)
    monkeypatch.setattr(cache, "run_restore", lambda **kwargs: {"status": "failed"})
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True,
                                 restore_backed_raw=True)
    assert result["status"] == "SOURCE_CACHE_INCOMPLETE" and result["formal_w2_complete"] is False
    assert not (root / "cache/source_v2/source_cache.json").exists()
    assert not (root / item["path"]).exists()


def test_R2_restore_has_independent_inventory_derived_durable_reservation_before_call(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    def restore(**kwargs):
        ledger = json.loads((root / "audit/source_cache/resource_progress.json").read_text())
        assert set(ledger["byte_scopes"]) == {"R2_cache_raw_restore_GET_body"}
        budget = ledger["byte_scopes"]["R2_cache_raw_restore_GET_body"]
        assert budget["charged_bytes"] == len(body) + 1 and budget["observed_bytes"] == 0
        assert budget["limit_bytes"] == 2 * (len(body) + 1)
        assert ledger["limit_derivation"]["official_provider_GET_bytes_included"] is False
        path = root / item["path"]; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
        return {"status": "restored_verified", "objects": [{"bytes_read": len(body), "complete_body_read": True}]}
    monkeypatch.setattr(cache, "run_restore", restore)
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True,
                                 partial_scope=True, restore_backed_raw=True)
    assert result["status"] == "PILOT_CACHE_PARTIAL"
    budget = json.loads((root / result["resource_progress_path"]).read_text())["byte_scopes"]["R2_cache_raw_restore_GET_body"]
    assert budget["observed_bytes"] == budget["charged_bytes"] == len(body)
    assert budget["attempts"][0]["status"] == "COMPLETE_OBSERVED_BODY"
    assert not (root / item["path"]).exists()


def test_failed_R2_retries_stop_at_independent_cap_and_need_recorded_amendment(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    calls = []
    def failed_restore(**kwargs):
        calls.append("fixture_failed_GET")
        return {"status": "failed", "objects": [{"bytes_read": 7, "complete_body_read": False}]}
    monkeypatch.setattr(cache, "run_restore", failed_restore)
    for _ in range(3):
        result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True,
                                     partial_scope=True, restore_backed_raw=True)
        assert result["status"] == "SOURCE_CACHE_INCOMPLETE"
    assert len(calls) == 2  # A third network operation is refused before run_restore.
    journal = json.loads((root / "audit/source_cache/resource_progress.json").read_text())
    ledger = journal["byte_scopes"]["R2_cache_raw_restore_GET_body"]
    assert ledger["charged_bytes"] == 2 * (len(body) + 1) and ledger["observed_bytes"] == 14
    old_identity = journal["resource_identity_sha256"]
    def completed_restore(**kwargs):
        path = root / item["path"]; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
        return {"status": "restored_verified", "objects": [{"bytes_read": len(body), "complete_body_read": True}]}
    monkeypatch.setattr(cache, "run_restore", completed_restore)
    with pytest.raises(cache.SourceCacheGateError, match="budget is unavailable"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True,
                            restore_backed_raw=True, max_r2_restore_bytes=3 * (len(body) + 1))
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True,
        restore_backed_raw=True, max_r2_restore_bytes=3 * (len(body) + 1),
        operational_amendment_note="Explicit operational-only fixture retry extension")
    assert result["status"] == "PILOT_CACHE_PARTIAL"
    journal = json.loads((root / "audit/source_cache/resource_progress.json").read_text())
    assert journal["resource_identity_sha256"] == old_identity
    assert len(journal["byte_scopes"]["R2_cache_raw_restore_GET_body"]["amendments"]) == 1


def test_real_SIGKILL_after_R2_commit_preserves_budget_and_raw_ownership_on_replay(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    control = root / "SOFTWARE_TEST_DOUBLE_BODY.bin"; control.write_bytes(body)
    script = '''
import os, signal, sys
from pathlib import Path
from tools import prepare_source_cache as cache
root, relative, control = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
def interrupted_restore(**kwargs):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(control.read_bytes())
    os.kill(os.getpid(), signal.SIGKILL)
cache.run_restore = interrupted_restore
cache.prepare_cache(root, 'manifests/source_fixture.json', execute=True,
    partial_scope=True, restore_backed_raw=True)
'''
    child = subprocess.run([sys.executable, "-c", script, str(root), item["path"], str(control)],
                           cwd=ROOT, capture_output=True, timeout=20)
    assert child.returncode == -signal.SIGKILL
    journal = json.loads((root / "audit/source_cache/resource_progress.json").read_text())
    ledger = journal["byte_scopes"]["R2_cache_raw_restore_GET_body"]
    assert ledger["charged_bytes"] == len(body) + 1 and ledger["observed_bytes"] == 0
    assert ledger["active"] is not None
    assert cache.recover_owned_cache_restore(root, item) is True
    assert control.read_bytes() == body  # Unmarked local files are preserved.
    def restore(**kwargs):
        target = root / item["path"]; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(body)
        return {"status": "restored_verified", "objects": [{"bytes_read": len(body), "complete_body_read": True}]}
    monkeypatch.setattr(cache, "run_restore", restore)
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True,
                                 partial_scope=True, restore_backed_raw=True)
    assert result["status"] == "PILOT_CACHE_PARTIAL" and not (root / item["path"]).exists()
    ledger = json.loads((root / "audit/source_cache/resource_progress.json").read_text())["byte_scopes"]["R2_cache_raw_restore_GET_body"]
    assert ledger["charged_bytes"] == 2 * len(body) + 1 and ledger["observed_bytes"] == len(body)
    assert ledger["attempts"][0]["status"] == "UNCLEAN_FULL_RESERVATION_REMAINS_CHARGED"


@pytest.mark.parametrize("expiry_point", ["last_raw", "final_manifest"])
def test_last_CPU_work_cannot_return_cache_completion_after_wall_cap(partial_fixture, monkeypatch, expiry_point):
    root, _, _ = partial_fixture
    from tools import source_preparation_batch as batch
    actual_budget = batch.ResourceBudget
    tick = [0.0]
    def budget_factory(*args, **kwargs):
        return actual_budget(*args, **kwargs, clock=lambda: tick[0], utc_clock=lambda: 1000 + tick[0], boot_id="fixture")
    budget_factory.R2_RESTORE = actual_budget.R2_RESTORE
    monkeypatch.setattr(batch, "ResourceBudget", budget_factory)
    if expiry_point == "last_raw":
        actual_iterator = cache.iter_native_contexts
        def last_raw_expiry(*args, **kwargs):
            yield from actual_iterator(*args, **kwargs)
            tick[0] = 2.0
        last_raw_expiry.__module__ = actual_iterator.__module__
        monkeypatch.setattr(cache, "iter_native_contexts", last_raw_expiry)
    else:
        actual_write = cache.write_receipt
        def late_manifest_expiry(path, value):
            actual_write(path, value)
            if Path(path).name == "source_cache.json": tick[0] = 2.0
        monkeypatch.setattr(cache, "write_receipt", late_manifest_expiry)
    with pytest.raises(cache.SourceCacheGateError, match="wall_budget_exhausted"):
        cache.prepare_cache(root, "manifests/source_fixture.json", execute=True,
                            partial_scope=True, max_wall_seconds=1)
    assert json.loads((root / "status/tasks.json").read_text())["W2"]["status"] == "NOT_STARTED"
    journal = json.loads((root / "audit/source_cache/resource_progress.json").read_text())
    assert journal["active_wall_elapsed_s"] == 2.0


def test_native_R2_restore_helper_uses_budgeted_body_proxy_without_network(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    class Client:
        def get_object(self, **kwargs):
            return {"ResponseMetadata": {"HTTPStatusCode": 200}, "ContentLength": len(body), "Body": io.BytesIO(body)}
    actual_restore = cache.run_restore
    def fixture_bucket_restore(**kwargs):
        return actual_restore(**kwargs, bucket="SOFTWARE_TEST_DOUBLE_BUCKET")
    monkeypatch.setattr(cache, "run_restore", fixture_bucket_restore)
    monkeypatch.setattr(cache, "_NATIVE_RESTORE_IMPLEMENTATION", fixture_bucket_restore)
    monkeypatch.setattr(cache, "create_client", lambda: Client())
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True,
                                 partial_scope=True, restore_backed_raw=True)
    assert result["status"] == "PILOT_CACHE_PARTIAL"
    ledger = json.loads((root / "audit/source_cache/resource_progress.json").read_text())["byte_scopes"]["R2_cache_raw_restore_GET_body"]
    assert ledger["observed_bytes"] == ledger["charged_bytes"] == len(body)
    assert ledger["attempts"][0]["reserved_bytes"] == len(body) + 1
    assert not (root / item["path"]).exists()


def test_killed_cache_restore_requires_durable_ownership_and_full_original_receipt(partial_fixture):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    marker = cache.begin_owned_cache_restore(root, item)
    restored = root / item["path"]
    restored.parent.mkdir(parents=True, exist_ok=True)
    restored.write_bytes(body)  # Mock complete GET commit, then no finally runs.
    assert json.loads(marker.read_text())["status"] == "RESTORE_IN_PROGRESS"
    assert cache.recover_owned_cache_restore(root, item) is True
    assert not restored.exists()
    assert json.loads(marker.read_text())["status"] == "RAW_REMOVED_AFTER_VERIFIED_BACKUP"


@pytest.mark.parametrize("proof_state", ["missing_marker", "old_completed_marker", "changed_raw"])
def test_matching_content_alone_never_authorizes_reappeared_raw_deletion(partial_fixture, proof_state):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    restored = root / item["path"]
    if proof_state != "missing_marker":
        marker = cache.begin_owned_cache_restore(root, item)
        if proof_state == "old_completed_marker":
            record = json.loads(marker.read_text())
            record["status"] = "RAW_REMOVED_AFTER_VERIFIED_BACKUP"
            write(marker, record)
    restored.parent.mkdir(parents=True, exist_ok=True)
    content = body + b"CHANGED_UNOWNED_BYTES" if proof_state == "changed_raw" else body
    restored.write_bytes(content)
    with pytest.raises(cache.SourceCacheGateError):
        cache.recover_owned_cache_restore(root, item)
    assert restored.read_bytes() == content


def test_active_cache_restore_holds_raw_worker_lock_until_context_finishes(partial_fixture, monkeypatch):
    root, inventory, _ = partial_fixture
    body, item = make_r2_only_fixture(root, inventory)
    path = root / item["path"]
    def restore(**kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {"status": "restored_verified"}
    monkeypatch.setattr(cache, "run_restore", restore)
    lock_path = root / "audit/source_preparation/.source_worker.lock"
    from tools.source_preparation_batch import ResourceBudget
    budget = ResourceBudget({"plan_sha256": "a" * 64, "max_download_bytes": 2 * len(body), "max_wall_seconds": 100},
        root / "audit/mock_restore_budget.json", scope=ResourceBudget.R2_RESTORE, identity_sha256="b" * 64)
    try:
        with cache._available_raw(root, item, path, restore_backed_raw=True, resource_budget=budget):
            assert path.exists()
            with lock_path.open("a") as competitor:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        budget.close()
    assert not path.exists()
    with lock_path.open("a") as competitor:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_partial_cache_cannot_enter_formal_pilot_even_with_forged_w2_status(partial_fixture):
    root, _, _ = partial_fixture
    result = cache.prepare_cache(root, "manifests/source_fixture.json", execute=True, partial_scope=True)
    write(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {
          "status": "COMPLETE", "cache_manifest": result["cache_manifest"]}})
    pilot_spec = importlib.util.spec_from_file_location("paper3_source_partial_pilot_check", ROOT / "tools/run_controlled_pilot.py")
    pilot = importlib.util.module_from_spec(pilot_spec)
    sys.modules[pilot_spec.name] = pilot
    pilot_spec.loader.exec_module(pilot)
    with pytest.raises(pilot.PilotGateError, match="New, real, source_train"):
        pilot.verify_inputs(root)


def test_documented_physionet_128hz_variant_uses_native_header_clock():
    class Annotations(list):
        @property
        def description(self):
            return np.asarray([row["description"] for row in self])
    class HeaderFixture:
        info = {"sfreq": 128.0}
        ch_names = [name + "." for name in CHO_EEG_CHANNELS]
        _orig_units = {name: "µV" for name in ch_names}
        n_times = 1280
        annotations = Annotations([{"description": "T1", "onset": 0.0, "duration": 3.0},
                                   {"description": "T2", "onset": 4.0, "duration": 3.0}])
    base = {"dataset": "PhysionetMI", "role": "source", "provider_subject": "88",
            "source_version": "HEADER_FIXTURE_NOT_EEG", "source_file_sha256": "d" * 64,
            "source_file_bytes": 0, "session_id": "provider_single_visit"}
    result = _physionet(Path("S088R04.edf"), base, HeaderFixture())
    assert result["summary"]["native_sampling_variant"] == "known_128Hz_provider_variant"
    assert [row["analysis_anchor_sample"] for row in result["trials"]] == [0, 512]
    assert [row["task_end_sample"] for row in result["trials"]] == [384, 896]
