"""Archive-reference semantics; fixtures never download, fit, or read raw BDFs."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from p3work import source_extension_bridge as bridge
from tools import resume_yang_source_transport_v2 as transport


REPOSITORY = Path(__file__).resolve().parents[1]


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.fixture
def member_fixture(tmp_path, monkeypatch):
    # The official 153-record plan is metadata, never a payload or fit fixture.
    plan_bytes = (REPOSITORY / bridge.YANG_MEMBER_PLAN).read_bytes()
    assert hashlib.sha256(plan_bytes).hexdigest() == bridge.YANG_MEMBER_PLAN_FILE_SHA
    plan = json.loads(plan_bytes)
    path = tmp_path / bridge.YANG_MEMBER_PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plan_bytes)
    code = tmp_path / bridge.YANG_MEMBER_TRANSPORT
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_bytes((REPOSITORY / bridge.YANG_MEMBER_TRANSPORT).read_bytes())
    record = plan["records"][0]
    native_path = (f"audit/source_extensions_20261019/Yang2025/{bridge.PARALLEL_YANG_PLAN_SHA}/"
                   f"{record['record_id']}/native_metadata.json")
    native = {"schema_version": "paper3-yang-native-inventory-record-v1",
        "dataset_id": "Yang2025_2C", "dataset_version": 5,
        "subject": 1, "session": 1, "participant_id": "Yang2025_2C:sub-001",
        "source_record_id": record["raw_files"][0]["download"]["member"],
        "raw_files": {name: {"path": raw["download"]["member"], "sha256": raw["sha256"],
            "bytes": raw["bytes"], "header_sha256": "a"*64, "start_date_time": "20.03.1913.53.10"}
            for name, raw in zip(("signal", "event"), record["raw_files"])}}
    native_ref = write(tmp_path, native_path, native)
    raw_receipt = write(tmp_path, "audit/software_member_fixture/raw_receipt.json", {"synthetic": True})
    final_receipt = write(tmp_path, "audit/software_member_fixture/final_receipt.json", {"synthetic": True})
    shards = []
    for row in plan["records"]:
        filename = (f"audit/source_extensions_20261019/Yang2025/{bridge.PARALLEL_YANG_PLAN_SHA}/"
                    f"{row['record_id']}/native_metadata.json")
        shards.append({"record_id": row["record_id"], "subject_id": row["subject_id"],
            "session_id": row["session_id"], "raw_files": copy.deepcopy(row["raw_files"]),
            "native_metadata": native_ref if row == record else {"path": filename, "sha256": "0"*64, "bytes": 0},
            "raw_backup_receipt": raw_receipt})
    manifest_path = f"cache/source_extensions_20261019/Yang2025/{bridge.PARALLEL_YANG_PLAN_SHA}/source_cache.json"
    manifest = {"dataset": "Yang2025", "role": "source_train", "source_only": True, "shards": shards}
    manifest_ref = write(tmp_path, manifest_path, manifest)
    history = {"calls_used": 8, "software_fixture_only": True}
    actual = {"status": "SOURCE_EXTENSION_CACHE_COMPLETE", "already_complete_offline_verified": True,
        "verified_records": 153, "manifest": manifest_ref, "backup_receipt": final_receipt,
        "transport_history": history}
    completed = {"mode": "already_complete", "plan_sha256": bridge.PARALLEL_YANG_PLAN_SHA,
        "manifest": manifest_ref, "backup_receipt": final_receipt,
        "transport_history": copy.deepcopy(history), "transport_completion_proof": copy.deepcopy(actual)}
    calls = []

    def verified(root, supplied_plan):
        calls.append((root, supplied_plan))
        return copy.deepcopy(actual)

    raw_rows = [{"path": r["path"], "sha256": r["sha256"], "bytes": r["bytes"],
        "role": "source_development_raw", "dataset": "Yang2025"} for r in record["raw_files"]]
    final_rows = [{"software_fixture_object": i} for i in range(5)]

    def receipts(root, item, closure):
        if item == raw_receipt:
            return copy.deepcopy(raw_rows)
        if item == final_receipt:
            return copy.deepcopy(final_rows)
        pytest.fail("unexpected receipt outside the exact fixture")

    monkeypatch.setattr(transport, "verify_complete_transport_history", verified)
    monkeypatch.setattr(transport, "run", lambda *a, **kw: pytest.fail("no coordinator dispatch"))
    monkeypatch.setattr(bridge, "_receipt_entries", receipts)
    original_open = Path.open
    bdf_attempts = []

    def no_bdf(path, *args, **kwargs):
        if path.suffix.lower() == ".bdf":
            bdf_attempts.append(str(path))
            pytest.fail("archive references must never open a BDF payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_bdf)
    return {"root": tmp_path, "plan": plan, "record": record, "native": native,
        "native_ref": native_ref, "manifest": manifest, "manifest_ref": manifest_ref,
        "actual": actual, "completed": completed, "calls": calls, "raw_rows": raw_rows,
        "final_rows": final_rows, "bdf_attempts": bdf_attempts}


def refresh_native(fixture):
    item = write(fixture["root"], fixture["native_ref"]["path"], fixture["native"])
    fixture["native_ref"] = item
    fixture["manifest"]["shards"][0]["native_metadata"] = item
    manifest = write(fixture["root"], fixture["manifest_ref"]["path"], fixture["manifest"])
    fixture["manifest_ref"] = manifest
    fixture["actual"]["manifest"] = manifest
    fixture["completed"]["manifest"] = manifest
    fixture["completed"]["transport_completion_proof"] = copy.deepcopy(fixture["actual"])
    return item


def traverse(fixture, item=None, *, completed=True):
    closure = {}
    bridge._provenance_closure(fixture["root"], item or fixture["native_ref"], closure,
        yang_completion=fixture["completed"] if completed else None)
    return closure


def test_exact_official_pair_is_classified_without_local_raw_payload(member_fixture):
    f = member_fixture
    closure = traverse(f)
    assert f["native_ref"]["path"] in closure
    assert bridge.YANG_MEMBER_PLAN in closure and bridge.YANG_MEMBER_TRANSPORT in closure
    assert len(f["calls"]) == 1 and f["calls"][0][1] == f["plan"]
    assert len(f["plan"]["records"]) == 153
    assert len({r["download"]["member"] for row in f["plan"]["records"] for r in row["raw_files"]}) == 306
    assert f["native"]["raw_files"]["signal"]["path"].endswith("/sub-001/ses-01/eeg/data.bdf")
    assert all(not (f["root"] / raw["path"]).exists() for raw in f["record"]["raw_files"])
    assert not f["bdf_attempts"]


@pytest.mark.parametrize("change", ["schema", "dataset", "version", "bool_version", "subject", "bool_subject",
    "session", "bool_session", "participant", "record", "source_record", "missing_event", "extra_component",
    "sha", "size", "bool_size", "unknown_member", "unsafe_member", "cross_record_signal", "cross_record_event"])
def test_wrong_native_identity_or_pair_never_gains_archive_permission(member_fixture, change):
    f = member_fixture; native = f["native"]
    if change == "schema": native["schema_version"] = "arbitrary-native-v1"
    elif change == "dataset": native["dataset_id"] = "Yang2025_3C"
    elif change == "version": native["dataset_version"] = 4
    elif change == "bool_version": native["dataset_version"] = True
    elif change == "subject": native["subject"] = 2
    elif change == "bool_subject": native["subject"] = True
    elif change == "session": native["session"] = 2
    elif change == "bool_session": native["session"] = True
    elif change == "participant": native["participant_id"] = "Yang2025_2C:sub-002"
    elif change == "record": native["record_id"] = "sub-002_ses-01"
    elif change == "source_record": native["source_record_id"] = "unknown/data.bdf"
    elif change == "missing_event": native["raw_files"].pop("event")
    elif change == "extra_component": native["raw_files"]["extra"] = {}
    elif change == "sha": native["raw_files"]["signal"]["sha256"] = "f"*64
    elif change == "size": native["raw_files"]["signal"]["bytes"] += 1
    elif change == "bool_size": native["raw_files"]["signal"]["bytes"] = True
    elif change == "unknown_member": native["raw_files"]["signal"]["path"] = "unknown/data.bdf"
    elif change == "unsafe_member": native["raw_files"]["signal"]["path"] = "../../outside.bdf"
    else:
        name, index = ("signal", 0) if change == "cross_record_signal" else ("event", 1)
        other = f["plan"]["records"][1]["raw_files"][index]
        native["raw_files"][name].update(path=other["download"]["member"], sha256=other["sha256"], bytes=other["bytes"])
    refresh_native(f)
    with pytest.raises(bridge.ExtensionGateError, match="Yang_native"):
        traverse(f)
    assert not f["bdf_attempts"]


@pytest.mark.parametrize("change", ["missing", "sha", "size", "role", "dataset", "duplicate"])
def test_both_members_require_exact_verified_raw_get_rows(member_fixture, change):
    f = member_fixture
    if change == "missing": f["raw_rows"].pop()
    elif change == "sha": f["raw_rows"][0]["sha256"] = "f"*64
    elif change == "size": f["raw_rows"][0]["bytes"] += 1
    elif change == "role": f["raw_rows"][0]["role"] = "result"
    elif change == "dataset": f["raw_rows"][0]["dataset"] = "Dreyer2023"
    else: f["raw_rows"][1] = copy.deepcopy(f["raw_rows"][0])
    with pytest.raises(bridge.ExtensionGateError, match="raw_full_GET"):
        traverse(f)
    assert not f["bdf_attempts"]


@pytest.mark.parametrize("change", ["not_complete", "unverified", "records", "float_records", "calls7", "calls9",
    "float_calls", "changed_proof", "missing_final_get"])
def test_archive_classification_cannot_replace_real153_five_get_eight_call_proof(member_fixture, change):
    f = member_fixture
    if change == "not_complete": f["actual"]["status"] = "RUNNING"
    elif change == "unverified": f["actual"]["already_complete_offline_verified"] = False
    elif change == "records": f["actual"]["verified_records"] = 152
    elif change == "float_records": f["actual"]["verified_records"] = 153.0
    elif change.startswith("calls"): f["actual"]["transport_history"]["calls_used"] = int(change[-1])
    elif change == "float_calls": f["actual"]["transport_history"]["calls_used"] = 8.0
    elif change == "changed_proof": f["completed"]["transport_completion_proof"]["extra"] = True
    else: f["final_rows"].pop()
    with pytest.raises(bridge.ExtensionGateError, match="Yang153|five_final"):
        traverse(f)
    assert not f["bdf_attempts"]


def test_plan_file_pin_and_semantic_pin_are_both_required(member_fixture, monkeypatch):
    f = member_fixture; path = f["root"] / bridge.YANG_MEMBER_PLAN
    changed = copy.deepcopy(f["plan"]); changed["role"] = "external_evaluation"
    path.write_text(json.dumps(changed))
    with pytest.raises(bridge.ExtensionGateError, match="SHA_or_size"):
        traverse(f)
    monkeypatch.setattr(bridge, "YANG_MEMBER_PLAN_FILE_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(bridge.ExtensionGateError, match="frozen_Yang153_member_plan"):
        traverse(f)
    assert not f["calls"]


def test_frozen_transport_code_cannot_be_replaced(member_fixture):
    f = member_fixture; (f["root"] / bridge.YANG_MEMBER_TRANSPORT).write_text("not the original verifier")
    with pytest.raises(bridge.ExtensionGateError, match="SHA_or_size"):
        traverse(f)
    assert not f["calls"]


@pytest.mark.parametrize("mode", ["no_complete_context", "ordinary_context", "ordinary_same_identity",
    "extra_native_dependency", "extra_member_dependency", "missing_native_metadata"])
def test_permission_does_not_escape_canonical_pair_or_hide_missing_metadata(member_fixture, mode):
    f = member_fixture; item = f["native_ref"]
    if mode == "no_complete_context":
        with pytest.raises(bridge.ExtensionGateError, match="artifact_missing"):
            traverse(f, completed=False)
        return
    if mode == "ordinary_context": item = write(f["root"], "audit/unbound_native.json", f["native"])
    elif mode == "ordinary_same_identity":
        f["native"]["ordinary_reference"] = copy.deepcopy(f["native"]["raw_files"]["signal"]); item = refresh_native(f)
    elif mode == "extra_native_dependency":
        f["native"]["required_dependency"] = {"path": "audit/missing_metadata.json", "sha256": "f"*64, "bytes": 1}; item = refresh_native(f)
    elif mode == "extra_member_dependency":
        f["native"]["raw_files"]["signal"]["required_dependency"] = {"path": "audit/missing_metadata.json", "sha256": "f"*64, "bytes": 1}; item = refresh_native(f)
    else: (f["root"] / item["path"]).unlink()
    with pytest.raises(bridge.ExtensionGateError, match="artifact_missing"):
        traverse(f, item)
    assert not f["bdf_attempts"]


def historical_fixture(fixture):
    relative = bridge.YANG_MEMBER_HISTORY
    raw = (REPOSITORY / relative).read_bytes()
    item = {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert item == fixture["plan"]["historical_get_evidence"][0]
    path = fixture["root"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    catalog = bridge._yang_member_catalog(fixture["root"], fixture["completed"], {})
    return item, json.loads(raw), catalog


def test_fixed_historical_provider_root_is_portable_without_raw_restoration(member_fixture):
    f = member_fixture; item, receipt, _ = historical_fixture(f)
    assert str(f["root"]) != bridge.YANG_MEMBER_ORIGINAL_ROOT
    assert receipt["objects"][0]["path"].startswith(bridge.YANG_MEMBER_ORIGINAL_ROOT + "/")
    assert item["path"] in traverse(f, item)
    assert not f["bdf_attempts"]


@pytest.mark.parametrize("change", ["schema", "status", "subject", "bool_subject", "session", "bool_session",
    "source_url", "archive_bytes", "target", "objects_missing", "objects_extra", "member", "sha", "size",
    "bool_size", "crc", "compressed_size", "object_status", "fresh_root", "other_root", "dotdot", "symlink_name",
    "other_record", "wrong_receipt_pin"])
def test_historical_raw_permission_is_exact_and_never_normalizes_paths(member_fixture, change):
    f = member_fixture; item, receipt, catalog = historical_fixture(f)
    if change == "schema": receipt["schema_version"] = True
    elif change == "status": receipt["status"] = "PARTIAL"
    elif change == "subject": receipt["subject"] = 2
    elif change == "bool_subject": receipt["subject"] = True
    elif change == "session": receipt["session"] = 2
    elif change == "bool_session": receipt["session"] = True
    elif change == "source_url": receipt["source_url"] = "https://example.invalid/archive"
    elif change == "archive_bytes": receipt["archive_bytes"] += 1
    elif change == "target": receipt["raw_target_directory"] = str(f["root"] / "data")
    elif change == "objects_missing": receipt["objects"].pop()
    elif change == "objects_extra": receipt["objects"].append(copy.deepcopy(receipt["objects"][0]))
    elif change == "member": receipt["objects"][0]["archive_member"] = "unknown/data.bdf"
    elif change == "sha": receipt["objects"][0]["sha256"] = "f"*64
    elif change == "size": receipt["objects"][0]["bytes"] += 1
    elif change == "bool_size": receipt["objects"][0]["bytes"] = True
    elif change == "crc": receipt["objects"][0]["crc32"] = "00000000"
    elif change == "compressed_size": receipt["objects"][0]["compressed_bytes"] += 1
    elif change == "object_status": receipt["objects"][0]["status"] = "PARTIAL"
    elif change == "fresh_root": receipt["objects"][0]["path"] = str(f["root"] / f["record"]["raw_files"][0]["path"])
    elif change == "other_root": receipt["objects"][0]["path"] = receipt["objects"][0]["path"].replace("/workspace/research3/", "/other/root/")
    elif change == "dotdot": receipt["objects"][0]["path"] = receipt["objects"][0]["path"].replace("/data/", "/data/../data/")
    elif change == "symlink_name": receipt["objects"][0]["path"] = receipt["objects"][0]["path"].replace("/data/", "/data_link/")
    elif change == "other_record":
        raw = f["plan"]["records"][1]["raw_files"][0]
        receipt["objects"][0].update(path=bridge.YANG_MEMBER_ORIGINAL_ROOT+"/"+raw["path"],
            archive_member=raw["download"]["member"], sha256=raw["sha256"], bytes=raw["bytes"])
    else: item["sha256"] = "f"*64
    with pytest.raises(bridge.ExtensionGateError, match="Yang_historical|original_Yang"):
        bridge._yang_historical_members(f["root"], item, receipt, catalog, {})
    assert not f["bdf_attempts"]


@pytest.mark.parametrize("change", ["other_receipt", "extra_object_dependency", "extra_receipt_dependency", "mutated_receipt"])
def test_historical_exception_cannot_hide_unknown_or_missing_typed_dependencies(member_fixture, monkeypatch, change):
    f = member_fixture; item, receipt, _ = historical_fixture(f)
    missing = {"path": "audit/still_required_missing.json", "sha256": "f"*64, "bytes": 1}
    if change == "other_receipt":
        other = write(f["root"], "audit/ordinary_receipt.json", receipt)
        with pytest.raises(ValueError, match="root-relative"):
            traverse(f, other)
    elif change == "mutated_receipt":
        receipt["objects"][0]["bytes"] += 1
        write(f["root"], item["path"], receipt)
        with pytest.raises(bridge.ExtensionGateError, match="SHA_or_size"):
            traverse(f, item)
    else:
        # Exercise recursion using a semantic fixture with an explicit authority
        # stub. The production path separately requires the whole receipt pin.
        target = receipt["objects"][0] if change == "extra_object_dependency" else receipt
        target["mandatory_extra"] = missing
        catalog = bridge._yang_member_catalog(f["root"], f["completed"], {})
        changed = write(f["root"], item["path"], receipt)
        catalog["historical"] = changed
        monkeypatch.setattr(bridge, "_yang_member_catalog", lambda *args: catalog)
        with pytest.raises(bridge.ExtensionGateError, match="artifact_missing"):
            traverse(f, changed)
    assert not f["bdf_attempts"]


def test_historical_members_still_require_both_verified_raw_gets(member_fixture):
    f = member_fixture; item, receipt, catalog = historical_fixture(f)
    f["raw_rows"].pop()
    with pytest.raises(bridge.ExtensionGateError, match="raw_full_GET"):
        bridge._yang_historical_members(f["root"], item, receipt, catalog, {})
    assert not f["bdf_attempts"]


@pytest.fixture
def prior_alias_fixture(tmp_path):
    original = "p3work/source_extensions/dreyer2023.py"
    active = write(tmp_path, original, {"current_V5_fixture": True})
    historical = write(tmp_path, "audit/qualified_native_snapshot.py.txt", {"historical_V4_fixture": True})
    aliases = {original: historical}
    prior = {"accounting_schema": "paper3-dreyer-six-ledger-budget-v1",
        "closed_history_sources": [{"source": name} for name in
            ("V1", "V2", "auxiliary_schema_review", "V3", "V4", "native_qualification")],
        "native_parser_evidence": {**historical, "path": original}}
    wrapper = SimpleNamespace(WrapperError=ValueError,
        prior_budget_evidence=lambda root, plan: copy.deepcopy(prior))
    return {"root": tmp_path, "original": original, "active": active, "historical": historical,
        "aliases": aliases, "prior": prior, "wrapper": wrapper}


def test_prior_budget_uses_existing_qualified_alias_without_changing_current_parser(prior_alias_fixture):
    f = prior_alias_fixture; closure = {}
    bridge._parallel_prior_budget(f["root"], f["prior"], {}, closure,
        wrapper=f["wrapper"], aliases=f["aliases"])
    assert closure == {f["historical"]["path"]: f["historical"]["sha256"]}
    assert (f["root"] / f["original"]).read_bytes() != (f["root"] / f["historical"]["path"]).read_bytes()
    assert bridge.descriptor(f["root"], f["root"] / f["original"]) == f["active"]


@pytest.mark.parametrize("change", ["no_alias", "wrong_SHA", "wrong_size", "unknown_reference", "tampered_archive", "missing_metadata"])
def test_prior_budget_alias_never_hides_unknown_or_mismatched_dependencies(prior_alias_fixture, change):
    f = prior_alias_fixture; aliases = f["aliases"]
    if change == "no_alias": aliases = None
    elif change == "wrong_SHA": f["prior"]["native_parser_evidence"]["sha256"] = "f"*64
    elif change == "wrong_size": f["prior"]["native_parser_evidence"]["bytes"] += 1
    elif change == "unknown_reference": f["prior"]["native_parser_evidence"]["path"] = "p3work/unknown_historical_parser.py"
    elif change == "tampered_archive": (f["root"] / f["historical"]["path"]).write_text("tampered")
    else: f["prior"]["mandatory_metadata"] = {"path": "audit/missing_metadata.json", "sha256": "f"*64, "bytes": 1}
    with pytest.raises(bridge.ExtensionGateError, match="SHA_or_size|artifact_missing"):
        bridge._parallel_prior_budget(f["root"], f["prior"], {}, {}, wrapper=f["wrapper"], aliases=aliases)


def test_prior_alias_permission_is_not_global_and_does_not_remap_active_parser(prior_alias_fixture):
    f = prior_alias_fixture
    with pytest.raises(bridge.ExtensionGateError, match="SHA_or_size"):
        bridge._provenance_closure(f["root"], f["prior"], {})
    f["prior"]["native_parser_evidence"] = f["active"]
    closure = {}
    bridge._parallel_prior_budget(f["root"], f["prior"], {}, closure,
        wrapper=f["wrapper"], aliases=f["aliases"])
    assert closure == {f["original"]: f["active"]["sha256"]}


@pytest.mark.parametrize("change", ["missing_history", "extra_history", "wrong_order", "wrong_schema"])
def test_existing_alias_cannot_replace_authentic_six_closed_budget_histories(prior_alias_fixture, change):
    f = prior_alias_fixture; value = copy.deepcopy(f["prior"])
    if change == "missing_history": value["closed_history_sources"].pop()
    elif change == "extra_history": value["closed_history_sources"].append({"source": "invented"})
    elif change == "wrong_order": value["closed_history_sources"].reverse()
    else: value["accounting_schema"] = "invented"
    with pytest.raises(bridge.ExtensionGateError, match="six_actual_conserved"):
        bridge._parallel_prior_budget(f["root"], value, {}, {}, wrapper=f["wrapper"], aliases=f["aliases"])
