"""SOFTWARE_TEST_DOUBLE reader/cache binding; no real EEG, GET, fit or GPU.

The full-cohort fixture fabricates metadata and historical receipt envelopes.
These tests exercise verification software only, never scientific readiness.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from p3work import source_cache_acceptance as acceptance
from p3work.adapters import source_dispatch
from p3work.contracts import CHANNELS, digest, file_digest
from tests.test_source_cache_acceptance import factory, write
from tools import prepare_source_cache as cache


ROOT = Path(__file__).resolve().parents[1]
CODE = ("p3work/adapters/source_native.py", "p3work/adapters/source_dispatch.py",
        "p3work/adapters/physionet_tal_v1.py", "p3work/preprocess.py")


def copy_code(root):
    for relative in CODE:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())


def stamp(revision):
    return {"native_reader_revision": deepcopy(revision),
        "physionet_tal_reader_sha256": revision["variant_reader"]["sha256"],
        "source_dispatch_sha256": revision["dispatcher"]["sha256"]}


def summary_stamp(revision):
    return {**stamp(revision), "adapter_role": "frozen_shared_metadata_helpers_component",
        "active_variant_file_sha256": revision["variant_reader"]["sha256"]}


def refresh_raw_receipts(f):
    """Re-seal test-double historical envelopes, without making any request."""
    for item in f.inventory["files"]:
        path = f.root / item["backup_receipt"]["path"]
        receipt = json.loads(path.read_text())
        for row in receipt["objects"]:
            if row["role"] != "audit":
                continue
            actual = f.root / row["path"]
            row.update(sha256=file_digest(actual), bytes=actual.stat().st_size)
            row["put"].update(request_body_sha256=row["sha256"], request_body_bytes=row["bytes"])
            row["get"].update(sha256=row["sha256"], bytes_read=row["bytes"], declared_content_length=row["bytes"])
        write(path, receipt)
    f.seal()


@pytest.fixture
def versioned(factory):
    f = factory
    copy_code(f.root)
    revision = source_dispatch.current_revision()
    selected = next(item for item in f.inventory["files"] if item["dataset"] == "PhysionetMI")
    selected_uids = set()
    for item in f.inventory["files"]:
        audit_path = f.root / item["adapter_audit"]["path"]
        audit = json.loads(audit_path.read_text())
        audit["adapter_file_sha256"] = revision["base_adapter"]["sha256"]
        if item is selected:
            native = f.root / item["native_trials"]["path"]
            rows = [json.loads(line) for line in native.read_text().splitlines()]
            for row in rows:
                row.update(stamp(revision)); selected_uids.add(row["trial_uid"])
            native.write_text("".join(json.dumps(row) + "\n" for row in rows))
            audit.update(summary_stamp(revision))
        write(audit_path, audit)
    person_shard = next(s for s in f.manifest["shards"] if
                        s["dataset"] == selected["dataset"] and s["subject_id"] == selected["provider_subject"])
    event_path = f.root / person_shard["events"]["path"]
    events = json.loads(event_path.read_text())
    for event in events:
        if event["event_uid"] in selected_uids:
            event.update(stamp(revision))
    write(event_path, events)
    person_shard["native_reader_revisions"] = [deepcopy(revision)]
    f.manifest.update(native_reader_revisions=[deepcopy(revision)],
                      reader_dispatcher_code=deepcopy(revision["dispatcher"]))
    refresh_raw_receipts(f)
    f.revision, f.selected, f.person_shard = revision, selected, person_shard
    return f


def test_mixed_historical_and_registered_source_metadata_accept_and_bind_all_code(versioned, monkeypatch):
    f = versioned
    def forbidden(*args, **kwargs):
        raise AssertionError("SOFTWARE_TEST_DOUBLE attempted raw/network/fit")
    monkeypatch.setattr(cache, "iter_native_contexts", forbidden)
    proof = acceptance.verify_source_cache_acceptance(f.cache)
    for key in ("base_adapter", "variant_reader", "dispatcher"):
        item = f.revision[key]
        assert proof["artifact_sha256"][item["path"]] == item["sha256"]
    assert proof["original_trials"] == proof["valid_trials"] == 1010
    assert not any((f.root / item["path"]).exists() for item in f.inventory["files"])


@pytest.mark.parametrize("remove_revision", [False, True])
def test_all_excluded_participant_still_requires_its_registered_revision(versioned, remove_revision):
    f = versioned
    previous = f.person_shard
    exclusions = []
    for item in f.inventory["files"]:
        if (item["dataset"], item["provider_subject"]) != (previous["dataset"], previous["subject_id"]):
            continue
        rows = [json.loads(line) for line in (f.root / item["native_trials"]["path"]).read_text().splitlines()]
        exclusions.extend({"event_uid": row["trial_uid"], "reason": "insufficient_original_task_context",
                           "original_metadata": row} for row in rows)
    write(f.root / previous["exclusions"]["path"], exclusions)
    f.manifest["shards"].remove(previous)
    f.manifest["participants_without_usable_trials"].append({
        "dataset": previous["dataset"], "subject_id": previous["subject_id"],
        "reason": "no_valid_trials_under_frozen_contract", "excluded_original_trials": len(exclusions),
        "exclusions": previous["exclusions"], "native_reader_revisions": [] if remove_revision else [f.revision]})
    f.seal()
    if remove_revision:
        with pytest.raises(acceptance.SourceCacheAcceptanceError, match="shard_reader"):
            acceptance.verify_source_cache_acceptance(f.cache)
    else:
        proof = acceptance.verify_source_cache_acceptance(f.cache)
        assert proof["excluded_trials"] == len(exclusions) and proof["participants"] == 224


@pytest.mark.parametrize("violation", ["unknown_revision", "revision_hash", "row_reader_sha", "audit_revision",
    "audit_variant", "audit_role", "missing_manifest", "extra_manifest_revision", "missing_shard", "code_bytes", "code_symlink"])
def test_resealed_unknown_or_inconsistent_reader_evidence_is_rejected(versioned, violation):
    f = versioned
    native_path = f.root / f.selected["native_trials"]["path"]
    rows = [json.loads(line) for line in native_path.read_text().splitlines()]
    audit_path = f.root / f.selected["adapter_audit"]["path"]
    audit = json.loads(audit_path.read_text())
    if violation == "unknown_revision":
        rows[0]["native_reader_revision"]["revision_id"] = "unregistered_variant"
    elif violation == "revision_hash":
        rows[0]["native_reader_revision"]["revision_sha256"] = "a" * 64
    elif violation == "row_reader_sha":
        rows[0]["physionet_tal_reader_sha256"] = "a" * 64
    elif violation == "audit_revision":
        audit.pop("native_reader_revision")
    elif violation == "audit_variant":
        audit["active_variant_file_sha256"] = "a" * 64
    elif violation == "audit_role":
        audit["adapter_role"] = "legacy_only_reader"
    elif violation == "missing_manifest":
        f.manifest.pop("native_reader_revisions")
    elif violation == "extra_manifest_revision":
        f.manifest["native_reader_revisions"].append(deepcopy(f.revision))
    elif violation == "missing_shard":
        f.person_shard.pop("native_reader_revisions")
    elif violation == "code_bytes":
        (f.root / f.revision["variant_reader"]["path"]).write_bytes(b"# tampered SOFTWARE_TEST_DOUBLE\n")
    else:
        code = f.root / f.revision["variant_reader"]["path"]
        held = code.with_name("held-reader.py"); code.rename(held); code.symlink_to(held)
    native_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(audit_path, audit)
    refresh_raw_receipts(f)
    with pytest.raises(acceptance.SourceCacheAcceptanceError, match="reader"):
        acceptance.verify_source_cache_acceptance(f.cache)


@pytest.mark.parametrize("marker", [None, {}, "unknown"])
def test_partial_or_null_marker_is_not_a_legacy_reader(tmp_path, marker):
    copy_code(tmp_path)
    with pytest.raises(cache.SourceCacheGateError, match="reader"):
        cache.verify_native_reader_provenance(tmp_path, "PhysionetMI", [{"native_reader_revision": marker}])


def test_unmarked_historical_metadata_remains_legal_without_new_code_payload(tmp_path):
    assert cache.verify_native_reader_provenance(tmp_path, "PhysionetMI", [{"trial_uid": "historical"}], {}) is None
    assert cache.verify_native_reader_provenance(tmp_path, "BNCI2014_001", [{"trial_uid": "historical"}], {}) is None


def test_audit_only_revision_marker_cannot_silently_select_legacy(tmp_path):
    copy_code(tmp_path)
    with pytest.raises(cache.SourceCacheGateError, match="reader"):
        cache.verify_native_reader_provenance(tmp_path, "PhysionetMI", [{"trial_uid": "historical"}],
                                            summary_stamp(source_dispatch.current_revision()))


def test_revision_cannot_be_added_to_a_legacy_event_only(factory):
    f = factory
    event_path = f.root / f.manifest["shards"][0]["events"]["path"]
    events = json.loads(event_path.read_text())
    events[0]["native_reader_revision"] = None
    write(event_path, events); f.seal()
    with pytest.raises(acceptance.SourceCacheAcceptanceError, match="identity"):
        acceptance.verify_source_cache_acceptance(f.cache)


@pytest.fixture
def partial_versioned(tmp_path):
    """A tiny fake EDF byte object; iterating it is replaced by numeric doubles."""
    root = tmp_path; copy_code(root)
    from tests.test_source_cache import frozen_contract
    contract = frozen_contract()
    write(root / "status/tasks.json", {"W1": {"status": "COMPLETE"}, "W2": {"status": "NOT_STARTED"}})
    write(root / "contracts/input_contract.json", contract)
    revision = source_dispatch.current_revision()
    raw = root / "data/SOFTWARE_TEST_DOUBLE/S001R04.edf"
    raw.parent.mkdir(parents=True); raw.write_bytes(b"SOFTWARE_TEST_DOUBLE_NOT_A_REAL_EDF")
    row = {"dataset": "PhysionetMI", "role": "source", "provider_subject": "1", "session_id": "provider_single_visit",
        "source_file_sha256": file_digest(raw), "source_file_bytes": raw.stat().st_size, "source_version": "SOFTWARE_TEST_DOUBLE",
        "source_record_id": raw.name, "physical_run_id": "4", "original_trial_index": 1,
        "raw_label_code": "T1", "canonical_label": "left_hand", "analysis_anchor_sample": 1000,
        "task_end_sample": 2000, "task_end_verified": True, **stamp(revision)}
    row["trial_uid"] = digest({key: row[key] for key in acceptance._UID_KEYS})
    metadata = root / "audit/trials.jsonl"; metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(row) + "\n")
    audit_path = root / "audit/adapter.json"
    write(audit_path, {**summary_stamp(revision), "adapter_file_sha256": revision["base_adapter"]["sha256"]})
    terms = root / "audit/terms.json"; write(terms, {"scope": "SOFTWARE_TEST_DOUBLE"})
    desc = lambda path: {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path), "bytes": path.stat().st_size}
    item = {**desc(raw), "dataset": "PhysionetMI", "provider_subject": "1", "session_id": "provider_single_visit",
        "file": raw.name, "version": row["source_version"], "complete_download": True, "http_status": 200,
        "native_schema_audited": True, "response_content_length": raw.stat().st_size,
        "url": cache.expected_source_objects()["PhysionetMI", "1", "provider_single_visit", raw.name],
        "native_trials": desc(metadata), "adapter_audit": desc(audit_path)}
    inventory = {"schema_version": "paper3-source-native-inventory-v2", "status": "SOURCE_RAW_INVENTORY_PARTIAL",
        "role": "source_train", "synthetic": False, "old_npz_reused": False, "provider_roster_verified": False,
        "provider_terms_evidence": desc(terms), "files": [item]}
    write(root / "manifests/inventory.json", inventory)
    return root, inventory, row, revision


def test_partial_builder_routes_revision_and_binds_resume_and_excluded_participants(partial_versioned, monkeypatch):
    root, _, row, revision = partial_versioned
    calls = []
    # Add one valid synthetic context to retain a numeric shard in this partial
    # engineering run; the signal is analytic, never a native EEG recording.
    def valid_contexts(dataset, path, **kwargs):
        calls.append(kwargs["revision"])
        t = np.arange(750) / 250
        yield {"row": row, "signal": np.stack([np.sin(2 * np.pi * (9 + c / 3) * t) for c in range(11)]),
            "channel_names": list(CHANNELS), "sfreq_hz": 250, "task_sample": 0, "task_end_sample": 750,
            "task_end_verified": True, "native_unit": "V", "unit_evidence_kind": "native_EDF_header_MNE_SI",
            "original_context_samples": [1000, 1750]}
    monkeypatch.setattr(cache, "iter_native_contexts", valid_contexts)
    result = cache._prepare_cache(root, "manifests/inventory.json", execute=True, partial_scope=True)
    manifest = json.loads((root / result["cache_manifest"]["path"]).read_text())
    state = json.loads((root / "cache/source_v2/build_state.json").read_text())
    assert calls == [revision]
    assert manifest["status"] == "PILOT_CACHE_PARTIAL"
    assert manifest["native_reader_revisions"] == [revision]
    assert manifest["reader_dispatcher_code"] == revision["dispatcher"]
    assert manifest["shards"][0]["native_reader_revisions"] == [revision]
    assert state["identity"]["native_reader_revisions"] == [revision]
    events = json.loads((root / manifest["shards"][0]["events"]["path"]).read_text())
    assert events[0]["native_reader_revision"] == revision
    assert events[0]["adapter_sha256"] == revision["base_adapter"]["sha256"]
    monkeypatch.setattr(cache, "iter_native_contexts", lambda *a, **k: pytest.fail("completed shard refit/read"))
    cache._prepare_cache(root, "manifests/inventory.json", execute=True, partial_scope=True)
    state["identity"]["native_reader_revisions"] = []
    write(root / "cache/source_v2/build_state.json", state)
    with pytest.raises(cache.SourceCacheGateError, match="identity differs"):
        cache._prepare_cache(root, "manifests/inventory.json", execute=True, partial_scope=True)


def test_all_excluded_shard_keeps_revision_evidence(partial_versioned):
    root, _, row, revision = partial_versioned
    temporary = root / "temporary"; temporary.mkdir()
    raw = root / "data/SOFTWARE_TEST_DOUBLE/S001R04.edf"
    item = {"sha256": file_digest(raw)}
    # Direct shard tests isolate the no-valid-trial branch without any I/O
    # beyond the test-double raw byte file and JSON exclusion output.
    from unittest.mock import patch
    with patch.object(cache, "iter_native_contexts", return_value=iter([
            {"row": row, "excluded": "insufficient_original_task_context"}])):
        shard = cache._prepare_shard(root, root / "cache", temporary, dataset="PhysionetMI", subject="1",
            records=[(item, raw, [row])], contract={}, contract_sha="a" * 64,
            adapter=revision["base_adapter"], processing={}, restore_backed_raw=False)
    assert shard["native_reader_revisions"] == [revision] and "X" not in shard


@pytest.mark.parametrize("violation", ["missing_audit", "mixed_rows", "different_dataset", "missing_code"])
def test_versioned_preflight_rejects_before_native_signal_access(partial_versioned, monkeypatch, violation):
    root, inventory, row, revision = partial_versioned
    item = inventory["files"][0]
    if violation == "missing_audit":
        item.pop("adapter_audit")
    elif violation == "mixed_rows":
        metadata = root / item["native_trials"]["path"]
        unmarked = {key: value for key, value in row.items() if key not in cache._READER_ROW_FIELDS}
        metadata.write_text(json.dumps(row) + "\n" + json.dumps(unmarked) + "\n")
        item["native_trials"]["sha256"] = file_digest(metadata)
    elif violation == "different_dataset":
        row["dataset"] = "BNCI2014_001"
        # A registered Physio revision on another dataset is rejected by the
        # shared helper, regardless of any claimed native object identity.
        with pytest.raises(cache.SourceCacheGateError, match="Physio"):
            cache.verify_native_reader_provenance(root, row["dataset"], [row])
        return
    else:
        (root / revision["dispatcher"]["path"]).unlink()
    write(root / "manifests/inventory.json", inventory)
    monkeypatch.setattr(cache, "iter_native_contexts", lambda *a, **k: pytest.fail("native access before gate"))
    with pytest.raises(cache.SourceCacheGateError, match="reader|audit"):
        cache._prepare_cache(root, "manifests/inventory.json", execute=True, partial_scope=True)
    assert not (root / "cache").exists()


def test_final_cache_R2_manifest_covers_versioned_events_without_raw_or_code_upload(versioned, monkeypatch):
    f = versioned
    from tools import prepare_all_sources
    monkeypatch.setattr(prepare_all_sources.batch, "execution_gate", lambda root: f.cache.input_contract_sha256)
    cache_path = f.root / f.cache.manifest_path
    inventory_path = f.root / f.manifest["raw_inventory"]["path"]
    catalog_path = f.root / f.tasks["W2"]["raw_r2_receipts_catalog"]
    receipts = [f.root / item["backup_receipt"]["path"] for item in f.inventory["files"]]
    _, manifest = prepare_all_sources.cache_backup_manifest(f.root, cache_path, inventory_path,
        catalog_path, receipts, f.root / "contracts/input_contract.json")
    by_path = {row["path"]: row for row in manifest["files"]}
    for descriptor in (f.person_shard["events"], f.person_shard["exclusions"], f.manifest["raw_inventory"]):
        assert by_path[descriptor["path"]]["sha256"] == descriptor["sha256"]
    assert by_path[f.cache.manifest_path]["sha256"] == f.cache.manifest_sha256
    assert not any(row["role"] == "source_development_raw" for row in manifest["files"])
    assert not any(row["path"].endswith(".py") for row in manifest["files"])
