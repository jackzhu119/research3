"""Software fixtures only: no real EEG fit, network, R2 operation or CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from p3work import source_extension_bridge as bridge
from p3work import source_extension_training as extension
from p3work import source_extension_io
from p3work import source_training as core
from p3work.contracts import CHANNELS, SOURCE_IDS, digest, file_digest
from p3work.f0_bridge import load_verified_source
from tools import run_controlled_pilot
from tools.train_source_extension import main

ROOT = Path(__file__).resolve().parents[1]
COMPLETE_PREPARATION = bridge._completed_preparation


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return bridge.descriptor(root, path)


def signal(seed):
    x = np.random.default_rng(seed).normal(size=(11, 128))*1e-5
    x -= x.mean(0, keepdims=True)
    x -= x.mean(1, keepdims=True)
    return x.astype(np.float32)


def refresh_software_cache_proof(root, manifest):
    """Refresh only a clearly mocked proof after an intentional fixture edit."""
    shard = manifest["shards"][0]
    keys = ("X", "y", "events", "exclusions", "metadata", "native_metadata", "unit_evidence")
    for key in keys:
        shard[key] = bridge.descriptor(root, root / shard[key]["path"])
    rows = [{**shard[k], "role": "result" if k in keys[:5] else "audit", "dataset": "Dreyer2023"}
            for k in keys]
    shard["cache_backup_receipt"] = write(root, "cache_receipt.json", rows)


@dataclass
class MathCache:
    root: Path
    arrays: list
    labels: np.ndarray
    rows: list
    locations: list
    artifact_sha256: dict
    input_contract_sha256: str = bridge.INPUT_SHA
    manifest_path: str = bridge.BASE_MANIFEST
    manifest_sha256: str = "a"*64
    upstream_acceptance_sha256: dict = None
    n_times: int = 128

    def __post_init__(self):
        self.upstream_acceptance_sha256 = {"software_fixture": "b"*64}

    def batch(self, indices):
        return np.stack([self.arrays[self.locations[int(i)][0]][self.locations[int(i)][1]] for i in indices])


@pytest.fixture
def fixture_cache(tmp_path, monkeypatch):
    root = tmp_path
    for relative in ("p3work/source_extension_io.py", "tools/r2_backup.py", "tools/r2_restore.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    baseline_descriptor = write(root, bridge.BASE_MANIFEST, {"software_fixture": True})
    monkeypatch.setattr(bridge, "BASE_SHA", baseline_descriptor["sha256"])
    arrays, rows, labels, locations = [], [], [], []
    for d, dataset in enumerate(sorted(SOURCE_IDS)):
        pair = np.stack([signal(d*4+i) for i in range(2)])
        arrays.append(pair)
        for i in range(2):
            rows.append({"dataset": dataset, "subject_id": "1", "role": "source_train",
                "event_uid": f"software-base-{d}-{i}", "processed_signal_sha256": hashlib.sha256(pair[i].tobytes()).hexdigest()})
            labels.append(i); locations.append((d, i))
    baseline = MathCache(root, arrays, np.asarray(labels), rows, locations,
        {bridge.BASE_MANIFEST: baseline_descriptor["sha256"]}, manifest_sha256=baseline_descriptor["sha256"])
    monkeypatch.setattr(run_controlled_pilot, "verify_inputs", lambda r: baseline)
    monkeypatch.setattr(bridge, "_frozen_roster", lambda *a: None)
    monkeypatch.setattr(bridge, "_completed_preparation", lambda *a: None)
    folder = root / "cache/extensions/Dreyer2023/session1"
    folder.mkdir(parents=True)
    x = np.stack([signal(100), signal(101)])
    np.save(folder / "X.npy", x, allow_pickle=False)
    np.save(folder / "y.npy", np.array([0, 1], dtype=np.int64), allow_pickle=False)
    adapter = write(root, "adapter.json", {"software_fixture": True})
    native = []
    for i in range(2):
        native.append({"dataset": "Dreyer2023", "trial_uid": f"software-extension-{i}",
            "raw_sha256": "c"*64, "original_trial_index": i, "label": i,
            "canonical_label": ["left_hand", "right_hand"][i], "analysis_anchor_sample": i*1000,
            "task_end_sample": i*1000+2000, "original_context_samples": [i*1000, i*1000+768],
            "native_sfreq_hz": 512, "task_end_verified": True, "acquisition_order_verified": True})
    ndesc = write(root, "native.json", {"trials": native, "waveform_QC_applied": False})
    profile = write(root, "profile.json", {"scope": "software-fixture"})
    unit = write(root, "unit.json", {"dataset": "Dreyer2023", "si_units_verified": True,
        "native_units": {name: "uV" for name in CHANNELS},
        "native_metadata": ndesc, "native_profile": profile})
    processed_metadata = write(root, "processed_metadata.json", {"scope": "software-fixture"})
    rawreceipt = write(root, "raw_receipt.json", {"software_fixture": True})
    rawfiles = [{"path": "data/source_extensions_20261019/Dreyer2023/example.edf",
                 "sha256": "c"*64, "bytes": 10}]
    events = [{**row, "event_uid": row["trial_uid"], "role": "source_train", "subject_id": "1",
        "session_id": "single_day", "input_contract_sha256": bridge.INPUT_SHA,
        "processed_signal_sha256": hashlib.sha256(x[i].tobytes()).hexdigest(), "unit_evidence": unit,
        "preprocessing_qc": {"reference": "CAR_exact_clinic11", "normalization_applied": False,
                              "stored_unit": "V", "si_units_verified": True}}
        for i, row in enumerate(native)]
    edesc = write(root, (folder / "events.json").relative_to(root).as_posix(), events)
    exc = write(root, (folder / "exclusions.json").relative_to(root).as_posix(), [])
    proof_entries = [{**item, "dataset": "Dreyer2023", "role": role}
        for item, role in [(bridge.descriptor(root, folder / "X.npy"), "result"),
        (bridge.descriptor(root, folder / "y.npy"), "result"), (edesc, "result"),
        (exc, "result"), (processed_metadata, "result"), (ndesc, "audit"), (unit, "audit")]]
    cache_receipt = write(root, "cache_receipt.json", proof_entries)
    rawproof = [{**r, "dataset": "Dreyer2023", "role": "source_development_raw"} for r in rawfiles]
    def read_proof(p):
        if p.name == "cache_receipt.json":
            return json.loads(p.read_text()), "software-proof-fixture"
        return copy.deepcopy(rawproof), "software-proof-fixture"
    restore = SimpleNamespace(load_verified_receipt=read_proof)
    monkeypatch.setattr(source_extension_io, "_modules", lambda: (None, restore))
    role = write(root, "source_role.json", {"dataset": "Dreyer2023", "role": "source_train",
        "source_only": True, "unique_role_within_branch": True})
    manifest = {"schema_version": "paper3-source-extension-cache-v1", "status": "SOURCE_EXTENSION_CACHE_COMPLETE",
        "dataset": "Dreyer2023", "role": "source_train", "source_only": True, "unique_role_within_branch": True,
        "channels": list(CHANNELS), "stage": bridge.STAGE, "input_contract_sha256": bridge.INPUT_SHA,
        "synthetic": False, "old_npz_reused": False, "normalization_applied": False, "role_assignment": role,
        "valid_trials": 2,
        "shards": [{"subject_id": "1", "session_id": "single_day", "raw_sha256": "c"*64,
            "valid_trials": 2, "original_events": 2,
            "X": bridge.descriptor(root, folder / "X.npy"), "y": bridge.descriptor(root, folder / "y.npy"),
            "events": edesc, "exclusions": exc, "native_metadata": ndesc,
            "metadata": processed_metadata, "native_profile": profile, "unit_evidence": unit,
            "raw_files": rawfiles, "cache_backup_receipt": cache_receipt,
            "raw_backup_receipt": rawreceipt, "adapter_code": adapter}]}
    mdesc = write(root, "cache/extensions/Dreyer2023/source_cache.json", manifest)
    def load():
        return bridge.load_extension_cache(root, arm="BASE4_PLUS_DREYER", manifest_path=mdesc["path"],
            manifest_sha256=file_digest(root / mdesc["path"]), output="runs/source_extensions/BASE4_PLUS_DREYER/seed-20261018")
    return root, manifest, mdesc, load, native, events


def test_composite_is_new_capability_preserves_base_and_exact_five_sources(fixture_cache):
    root, manifest, _, load, _, _ = fixture_cache
    before = set(SOURCE_IDS)
    cache = load()
    assert type(cache) is bridge.ExtensionCache and not isinstance(cache, run_controlled_pilot.VerifiedCache)
    assert len(cache.rows) == 10 and cache.batch(np.array([8, 9])).shape == (2, 11, 128)
    assert {r["dataset"] for r in cache.rows} == before | {"Dreyer2023"}
    assert SOURCE_IDS == before and "Dreyer2023" not in SOURCE_IDS
    assert not (root / "runs").exists()


@pytest.mark.parametrize("field,value", [("source_only", False), ("unique_role_within_branch", False),
    ("role", "meta_train"), ("dataset", "HSStroke2026_v6"), ("normalization_applied", True),
    ("synthetic", True), ("old_npz_reused", True), ("channels", ["C3", "Cz", "C4"])])
def test_manifest_role_shape_and_nonfixture_gates(fixture_cache, field, value):
    root, manifest, mdesc, load, _, _ = fixture_cache
    manifest[field] = value
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError):
        load()


def test_member_SHA_tamper_fails_before_data_is_used(fixture_cache):
    root, manifest, _, load, _, _ = fixture_cache
    with (root / manifest["shards"][0]["X"]["path"]).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(bridge.ExtensionGateError, match="SHA"):
        load()


def test_cache_full_GET_proof_must_cover_every_actual_cache_object(fixture_cache):
    root, manifest, mdesc, load, _, _ = fixture_cache
    rows = json.loads((root / "cache_receipt.json").read_text())
    manifest["shards"][0]["cache_backup_receipt"] = write(root, "cache_receipt.json", rows[1:])
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="exact_cache_objects_full_GET"):
        load()


def test_raw_GET_evidence_must_match_component_path_size_and_SHA(fixture_cache):
    root, manifest, mdesc, load, _, _ = fixture_cache
    manifest["shards"][0]["raw_files"][0]["bytes"] += 1
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="exact_frozen_record_components"):
        load()


def test_native_label_mismatch_is_not_relabelled(fixture_cache):
    root, manifest, mdesc, load, native, _ = fixture_cache
    native[0]["label"] = 1
    manifest["shards"][0]["native_metadata"] = write(root, "native.json", {"trials": native, "waveform_QC_applied": False})
    refresh_software_cache_proof(root, manifest)
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="original_native"):
        load()


def test_unaccounted_native_event_is_not_silently_discarded(fixture_cache):
    root, manifest, mdesc, load, native, events = fixture_cache
    native.append({**native[-1], "trial_uid": "software-unknown", "original_trial_index": 99})
    manifest["shards"][0]["native_metadata"] = write(root, "native.json", {"trials": native, "waveform_QC_applied": False})
    unit = json.loads((root / "unit.json").read_text())
    unit["native_metadata"] = manifest["shards"][0]["native_metadata"]
    udesc = write(root, "unit.json", unit)
    for event in events:
        event["unit_evidence"] = udesc
    manifest["shards"][0]["events"] = write(root, manifest["shards"][0]["events"]["path"], events)
    refresh_software_cache_proof(root, manifest)
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="valid_or_excluded"):
        load()


def test_unit_proof_from_a_different_native_record_is_not_reused(fixture_cache):
    root, manifest, mdesc, load, _, events = fixture_cache
    evidence = json.loads((root / "unit.json").read_text())
    evidence["native_metadata"]["sha256"] = "9"*64
    udesc = write(root, "unit.json", evidence)
    for event in events:
        event["unit_evidence"] = udesc
    manifest["shards"][0]["events"] = write(root, manifest["shards"][0]["events"]["path"], events)
    refresh_software_cache_proof(root, manifest)
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="this_native_record"):
        load()


@pytest.mark.parametrize("bad", [None, {"valid": 1}])
def test_malformed_or_non_boolean_event_validity_is_closed(fixture_cache, bad):
    root, manifest, mdesc, load, _, events = fixture_cache
    events[0] = bad
    manifest["shards"][0]["events"] = write(root, manifest["shards"][0]["events"]["path"], events)
    refresh_software_cache_proof(root, manifest)
    write(root, mdesc["path"], manifest)
    with pytest.raises(bridge.ExtensionGateError, match="boolean_validity"):
        load()


def yang_window_fixture():
    original = {"native_sfreq_hz": 1000, "original_marker_sample": 123000,
        "original_context_samples": [125250, 126750], "analysis_anchor_sample": 125250,
        "crop_start_sample": 125500, "crop_stop_sample_exclusive": 126500,
        "label_code": "1", "label": 0, "raw_event_sha256": "e"*64}
    event = {"raw_event_sha256": "e"*64,
             "MI_onset_verified": False, "physical_MI_end_verified": False}
    return event, original


def test_Yang_source_window_is_accepted_without_fake_MI_onset_or_task_end():
    event, original = yang_window_fixture()
    bridge._native_window("Yang2025", event, original)
    assert "task_end_sample" not in original and "task_end_verified" not in original


@pytest.mark.parametrize("scope,key,value", [("event", "MI_onset_verified", True),
    ("event", "physical_MI_end_verified", True), ("event", "raw_event_sha256", "f"*64),
    ("original", "original_marker_sample", 123001), ("original", "crop_start_sample", 125501),
    ("original", "label_code", "9"), ("original", "native_sfreq_hz", 512)])
def test_Yang_frozen_source_window_does_not_become_physical_MI_or_wrong_raw(scope, key, value):
    event, original = yang_window_fixture()
    (event if scope == "event" else original)[key] = value
    with pytest.raises(bridge.ExtensionGateError):
        bridge._native_window("Yang2025", event, original)


def test_new_source_receipt_requires_its_own_document_bound_profile(tmp_path):
    receipt = write(tmp_path, "receipt.json", {"schema_version": 1, "status": "verified"})
    _, restore = source_extension_io._modules()
    with pytest.raises(source_extension_io.SourceExtensionIOError, match="profile_mismatch"):
        restore.load_verified_receipt(tmp_path / receipt["path"])


@pytest.fixture
def preparation_proof_fixture(tmp_path, monkeypatch):
    """One-record SOFTWARE proof only; provider roster/parser are separately gated."""
    from p3work import source_extension_coordinator as coordinator
    root, dataset, sha = tmp_path, "Yang2025", "d"*64
    for rel in ("p3work/source_extension_io.py", "p3work/source_extension_coordinator.py",
                "p3work/preprocess.py", "tools/r2_backup.py", "tools/r2_restore.py"):
        target=root/rel; target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    inventory=write(root,"inventory.json",{"software_fixture":True})
    profile=write(root,"profile.json",{"software_fixture":True})
    audit=f"audit/source_extensions_20261019/{dataset}/{sha}"
    plan={"dataset":dataset,"plan_sha256":sha,"record_count":1,"inventory":inventory,"native_profile":profile,
        "code":{p:bridge.descriptor(root,root/p) for p in
                ("p3work/source_extension_coordinator.py","p3work/preprocess.py")},
        "budgets":{"wall_seconds":60,"provider_body_bytes":10,"r2_get_body_bytes":100000}}
    pdesc=write(root,audit+"/frozen_plan.json",plan)
    native=write(root,audit+"/native_inventory.json",{"software_fixture":True})
    unit=write(root,audit+"/unit_evidence.json",{"software_fixture":True})
    role=write(root,audit+"/role_assignment.json",{"software_fixture":True})
    manifest={"dataset":dataset,"plan_sha256":sha,"official_inventory":inventory,"native_profile":profile,
        "processing_code":plan["code"]["p3work/preprocess.py"],
        "coordinator_code":plan["code"]["p3work/source_extension_coordinator.py"],
        "native_inventory":native,"unit_evidence":unit,"role_assignment":role,"valid_trials":2,"shards":[]}
    mdesc=write(root,f"cache/source_extensions_20261019/{dataset}/{sha}/source_cache.json",manifest)
    rows=[{**item,"role":r,"dataset":dataset} for item,r in
          ((mdesc,"result"),(native,"audit"),(unit,"audit"),(pdesc,"audit"),(role,"audit"))]
    proof=write(root,audit+"/final_backup_receipt.json",rows)
    status={"status":"SOURCE_EXTENSION_CACHE_COMPLETE","dataset":dataset,"manifest":mdesc,
        "record_count":1,"valid_trials":2,"training_started":False,"GPU_actions":0,
        "external_payload_accessed":False,"backup_receipt":proof}
    write(root,audit+"/coordinator_status.json",status)
    amount=sum(row["bytes"] for row in rows)
    ledger={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":sha,
        "charged":{"provider_body_bytes":0,"r2_get_body_bytes":amount},"elapsed_seconds":1.2,
        "history":[{"kind":"r2_get_body_bytes","reserved_bytes":amount,"verified_bytes":amount,
                    "status":"SUCCESS_FULL_BODY_VERIFIED"}]}
    write(root,audit+"/budget_ledger.json",ledger)
    monkeypatch.setattr(coordinator,"validate_plan",lambda *a:None)
    restore=SimpleNamespace(load_verified_receipt=lambda p:(json.loads(p.read_text()),"software-profile-proof"))
    monkeypatch.setattr(source_extension_io,"_modules",lambda:(None,restore))
    def check(): return COMPLETE_PREPARATION(root,manifest,root/mdesc["path"],{})
    return root,audit,manifest,ledger,status,check


def test_closed_complete_preparation_proof_is_checked_read_only(preparation_proof_fixture):
    root, _, _, _, _, check=preparation_proof_fixture
    before={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    check()
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("key,value",[("wall_intent",{}),("active_transfer",{}),("elapsed_seconds",61)])
def test_unclosed_or_overrun_preparation_ledger_cannot_authorize_fit(preparation_proof_fixture,key,value):
    root,audit,_,ledger,_,check=preparation_proof_fixture
    ledger[key]=value;write(root,audit+"/budget_ledger.json",ledger)
    with pytest.raises(bridge.ExtensionGateError,match="closed_in_budget"):
        check()


def test_preparation_byte_charges_cannot_be_reset_below_history(preparation_proof_fixture):
    root,audit,_,ledger,_,check=preparation_proof_fixture
    ledger["charged"]["r2_get_body_bytes"]=0;write(root,audit+"/budget_ledger.json",ledger)
    with pytest.raises(bridge.ExtensionGateError,match="consumed_bytes_do_not_match"):
        check()


def test_final_R2_proof_must_include_the_completed_manifest(preparation_proof_fixture):
    root,audit,_,_,status,check=preparation_proof_fixture
    rows=json.loads((root/status["backup_receipt"]["path"]).read_text())
    status["backup_receipt"]=write(root,audit+"/final_backup_receipt.json",rows[1:])
    write(root,audit+"/coordinator_status.json",status)
    with pytest.raises(bridge.ExtensionGateError,match="complete_final_source_manifest_full_GET"):
        check()


def parallel_evidence_fixture(root, monkeypatch):
    """Bridge dispatch fixtures only; the frozen wrapper has its own real math tests."""
    wrapper_file=root/bridge.PARALLEL_WRAPPER
    wrapper_file.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/bridge.PARALLEL_WRAPPER,wrapper_file)
    wrapper_ref=bridge.descriptor(root,wrapper_file)
    records=[{"record_id":f"software-{i}"} for i in range(520)]
    plan={"schema_version":"SYNTHETIC_SOFTWARE_BRIDGE_ONLY","records":records}
    plan_sha=digest(plan);plan["plan_sha256"]=plan_sha
    monkeypatch.setattr(bridge,"PARALLEL_DREYER_PLAN_SHA",plan_sha)
    plan_ref=write(root,bridge.PARALLEL_DREYER_PLAN,plan)
    monkeypatch.setattr(bridge,"PARALLEL_DREYER_PLAN_FILE_SHA",plan_ref["sha256"])
    audit="audit/source_extensions_20261019/Dreyer2023/"+plan_sha
    write(root,audit+"/frozen_plan.json",plan)
    refs={}
    for i,r in enumerate(records):
        write(root,audit+"/"+r["record_id"]+"/state.json",{"software_fixture":True,"record_id":r["record_id"]})
        if i<128:
            refs[r["record_id"]]=write(root,"audit/old128/"+r["record_id"]+".json",{"software_fixture":True})
            write(root,audit+"/"+r["record_id"]+"/v4_numeric_regression.json",{"software_fixture":True})
    labels=["V1","V2","auxiliary_schema_review","V3","V4","native_qualification"]
    history=[]
    for label in labels:
        history.append({"source":label,"plan":write(root,"audit/history/"+label+"/plan.json",{"software_fixture":True}),
                        "closed_ledger":write(root,"audit/history/"+label+"/ledger.json",{"software_fixture":True})})
    prior={"accounting_schema":"paper3-dreyer-six-ledger-budget-v1","closed_history_sources":history}
    alias=write(root,"audit/old_code/adapter.json",{"historical_software_fixture":True})
    qualification={"historical_code_aliases":{"p3work/source_extensions/dreyer2023.py":alias},
                   "reference_states":refs,"completion_proof":write(root,"audit/qualification/proof.json",{"software_fixture":True})}
    declaration={"software_fixture":True,"prior_budget_evidence":prior,"qualification_evidence":qualification,
        "initial_yang_process":{"mode":"concurrent_no_cache_wrapper_v2",
            "plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,
            "closed_ledger":{"path":"audit/source_extensions_20261019/Yang2025/LIVE/budget_ledger.json",
                             "sha256":"a"*64,"bytes":1}}}
    decl=write(root,audit+"/parallel_root_declaration.json",declaration)
    comparisons=[{"record_id":records[i]["record_id"],"old_reference_state":refs[records[i]["record_id"]]}
                 for i in range(128)]
    regressions={"reference_records":128,"verified_records":128,"comparisons":comparisons,
                 "embedded_in_final_GET_backed_operational_evidence":True}
    limits={"software_fixture":True}
    evidence={"schema_version":"paper3-dreyer-parallel-cpu-execution-v5",
        "status":"RECORDS_VERIFIED_FINAL_BACKUP_PENDING","dataset":"Dreyer2023",
        "plan_sha256":plan_sha,"concurrent_yang_plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,
        "record_count":520,"record_ids_sha256":digest([r["record_id"]for r in records]),
        "wrapper_implementation":wrapper_ref,"root_declaration":decl,"scientific_kernel_unchanged":True,
        "qualification_evidence":qualification,"old128_numerical_regression":regressions,
        "yang_peer_coordination":{"schema_version":"paper3-dreyer-yang-peer-coordination-v1",
            "initial_yang_process":declaration["initial_yang_process"],
            "last_verified_live_peer":declaration["initial_yang_process"],
            "global_lock_takeover":False,"completed_yang_proof":None},
        "GPU_actions":0,"actual_fits":0,"external_payload_accessed":False,
        "carried_over_verified_R2_body_bytes":0,"server_actions":False,"limits":limits}
    item=write(root,audit+"/parallel_execution_receipt.json",evidence)
    manifest={"dataset":"Dreyer2023","plan_sha256":plan_sha,"operational_evidence":item,"shards":records}
    status={"operational_evidence":item}
    calls=[]
    class MockFrozenWrapper:
        class WrapperError(ValueError):pass
        LIMITS=limits
        def verify_declaration(self,r,path,p):
            calls.append("verify_declaration")
            actual=json.loads((r/path).read_text())
            if actual!=declaration:raise self.WrapperError("mock_original_frozen_declaration_changed")
            return copy.deepcopy(declaration),bridge.descriptor(r,r/path)
        def verified_completion(self,r,p,d):
            calls.append("verified_completion")
            return copy.deepcopy(status)
        def prior_budget_evidence(self,r,p):
            calls.append("prior_budget_evidence")
            return copy.deepcopy(prior)
    verifier=MockFrozenWrapper()
    monkeypatch.setattr(bridge,"_v5_wrapper",lambda r,c:verifier)
    return audit,manifest,status,evidence,verifier,calls


def test_parallel_v5_requires_actual_offline_verifier_and_two_extra_final_GET_objects(tmp_path,monkeypatch):
    audit,manifest,status,_,_,calls=parallel_evidence_fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    closure={};extra=bridge._parallel_evidence(tmp_path,manifest,status,audit,closure)
    assert len(extra)==2 and all(role=="audit" for _,role in extra)
    assert calls==["verify_declaration","verified_completion","prior_budget_evidence"]
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert all(f"audit/history/{label}/ledger.json" in closure for label in
        ("V1","V2","auxiliary_schema_review","V3","V4","native_qualification"))
    assert len([p for p in closure if p.startswith(audit+"/software-") and p.endswith("/state.json")])==520
    assert len([p for p in closure if p.endswith("/v4_numeric_regression.json")])==128
    assert not any("Yang2025" in p for p in closure)
    assert bridge._parallel_evidence(tmp_path,{"dataset":"Yang2025"},{},"unused",{})==[]
    with pytest.raises(bridge.ExtensionGateError,match="seven_GET_objects_required"):
        bridge._parallel_evidence(tmp_path,{"dataset":"Dreyer2023"},{},"unused",{})


@pytest.mark.parametrize("key,value",[("GPU_actions",1),("actual_fits",1),
    ("external_payload_accessed",True),("carried_over_verified_R2_body_bytes",1),
    ("concurrent_yang_plan_sha256","f"*64),("schema_version","paper3-dreyer-parallel-cpu-execution-v1"),
    ("record_count",519),("scientific_kernel_unchanged",1),("GPU_actions",False)])
def test_parallel_V5_scope_types_and_old_schema_cannot_expand_authority(tmp_path,monkeypatch,key,value):
    audit,manifest,status,evidence,_,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    evidence[key]=value
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="scope_or_budget_changed"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


@pytest.mark.parametrize("failure",["qualification","actual128numeric","final7GET","closedledger"])
def test_parallel_V5_forwards_real_proof_failure_without_permitting_fit(tmp_path,monkeypatch,failure):
    audit,manifest,status,_,verifier,calls=parallel_evidence_fixture(tmp_path,monkeypatch)
    def blocked(*a):raise verifier.WrapperError(failure)
    monkeypatch.setattr(verifier,"verified_completion",blocked)
    with pytest.raises(bridge.ExtensionGateError,match="actual_qualification_numeric_and_complete_GET"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})
    assert calls==["verify_declaration"]


def test_parallel_V5_missing_actual_complete_cache_cannot_adopt_metadata_qualification(tmp_path,monkeypatch):
    audit,manifest,status,_,verifier,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(verifier,"verified_completion",lambda *a:None)
    with pytest.raises(bridge.ExtensionGateError,match="actual_qualification_numeric_and_complete_GET"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


@pytest.mark.parametrize("filename",["state.json","v4_numeric_regression.json"])
def test_parallel_V5_state_and_numeric_report_snapshots_cannot_change_during_verification(tmp_path,monkeypatch,filename):
    audit,manifest,status,_,verifier,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    def changed(*a):
        write(tmp_path,audit+"/software-0/"+filename,{"changed_during_verification":True})
        return copy.deepcopy(status)
    monkeypatch.setattr(verifier,"verified_completion",changed)
    with pytest.raises(bridge.ExtensionGateError,match="SHA_or_size"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


@pytest.mark.parametrize("change",["short_list","unverified","not_embedded","float_count"])
def test_parallel_V5_all128_actual_regressions_required(tmp_path,monkeypatch,change):
    audit,manifest,status,evidence,_,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    regressions=evidence["old128_numerical_regression"]
    if change=="short_list":regressions["comparisons"].pop()
    elif change=="unverified":regressions["verified_records"]=127
    elif change=="float_count":regressions["verified_records"]=128.0
    else:regressions["embedded_in_final_GET_backed_operational_evidence"]=False
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="all128_actual_numeric"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def test_parallel_V5_old_plan_and_old_wrapper_never_impersonate_current(tmp_path,monkeypatch):
    audit,manifest,status,evidence,_,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    current=manifest["plan_sha256"];manifest["plan_sha256"]="971c1a1272fdb25b0f21e7d69c4e7349593d352341bdaa2eaa0725b600dbc1d1"
    with pytest.raises(bridge.ExtensionGateError,match="exact_declared_Dreyer_V5"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})
    manifest["plan_sha256"]=current
    evidence["wrapper_implementation"]["path"]=bridge.PARALLEL_PARENT_WRAPPER
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="declared_Dreyer_V5_CPU_wrapper"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def test_V5_frozen_verifier_loader_checks_all_six_release_bindings_without_live_Yang(tmp_path):
    refs=((bridge.PARALLEL_WRAPPER,bridge.PARALLEL_WRAPPER_SHA),
          (bridge.PARALLEL_PARENT_WRAPPER,bridge.PARALLEL_PARENT_WRAPPER_SHA),
          (bridge.PARALLEL_DREYER_PLAN,bridge.PARALLEL_DREYER_PLAN_FILE_SHA),
          (bridge.DREYER_NATIVE,bridge.DREYER_NATIVE_SHA),
          (bridge.DREYER_QUALIFICATION,bridge.DREYER_QUALIFICATION_SHA),
          (bridge.DREYER_ADMISSION,bridge.DREYER_ADMISSION_SHA))
    for path,_ in refs:
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    closure={};module=bridge._v5_wrapper(tmp_path,closure)
    assert module.DREYER_PLAN_SHA==bridge.PARALLEL_DREYER_PLAN_SHA
    assert set(closure)=={p for p,_ in refs}
    for path,_ in refs:
        target=tmp_path/path;original=target.read_bytes();target.write_bytes(original+b" ")
        with pytest.raises(bridge.ExtensionGateError,match="SHA_or_size"):
            bridge._v5_wrapper(tmp_path,{})
        target.write_bytes(original)


def test_provenance_closure_preserves_original_alias_and_never_demands_raw_payload(tmp_path):
    original="p3work/source_extensions/dreyer2023.py"
    old=write(tmp_path,"audit/historical/native.json",{"historical_adapter":True})
    write(tmp_path,original,{"current_new_adapter":True})
    raw={"path":"data/source_extensions_20261019/Dreyer2023/not_restored.edf","sha256":"b"*64,"bytes":42}
    dependent=write(tmp_path,"audit/dependent.json",{"old_adapter":{**old,"path":original},"raw_files":[raw]})
    closure={};bridge._provenance_closure(tmp_path,dependent,closure,aliases={original:old})
    assert set(closure)=={dependent["path"],old["path"]}
    assert original not in closure and not (tmp_path/raw["path"]).exists()
    (tmp_path/old["path"]).write_text("tamper")
    with pytest.raises(bridge.ExtensionGateError,match="SHA_or_size"):
        bridge._provenance_closure(tmp_path,dependent,{},aliases={original:old})


def peer_coordination_fixture(root, *, already_complete=False):
    """Small peer-schema fixture; no claim of real Yang qualification is made."""
    snapshot=write(root,"audit/software_peer/closed_snapshot.json",{"software_fixture":True})
    mapping={snapshot["path"]:snapshot["sha256"]}
    complete={"mode":"already_complete","plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,
        "transport_history":{"provenance_closure":dict(mapping)},
        "transport_completion_proof":{"status":"SOURCE_EXTENSION_CACHE_COMPLETE",
            "already_complete_offline_verified":True,"verified_records":153,
            "provenance_closure":dict(mapping)}}
    live={"mode":"concurrent_no_cache_wrapper_v2","plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,
          "pid":2026,"command_sha256":"d"*64}
    initial=complete if already_complete else live
    coordination={"schema_version":"paper3-dreyer-yang-peer-coordination-v1",
        "initial_yang_process":copy.deepcopy(initial),
        "last_verified_live_peer":None if already_complete else copy.deepcopy(initial),
        "global_lock_takeover":True,"completed_yang_proof":complete}
    return {"initial_yang_process":initial},coordination,snapshot


def test_still_live_Yang_never_dereferences_its_recorded_objects(tmp_path,monkeypatch):
    declaration,coordination,_=peer_coordination_fixture(tmp_path)
    initial=declaration["initial_yang_process"]
    initial["closed_ledger"]={"path":"audit/Yang2025/live/budget_ledger.json","sha256":"b"*64,"bytes":1}
    coordination.update(initial_yang_process=copy.deepcopy(initial),last_verified_live_peer=copy.deepcopy(initial),
                        global_lock_takeover=False,completed_yang_proof=None)
    monkeypatch.setattr(bridge,"artifact",lambda *a:pytest.fail("must not inspect any live Yang artifact"))
    closure={};bridge._peer_coordination_closure(tmp_path,declaration,coordination,closure)
    assert closure=={}


@pytest.mark.parametrize("already_complete",[False,True])
def test_Yang_closed_takeover_binds_both_actual_provenance_maps(tmp_path,already_complete):
    declaration,coordination,snapshot=peer_coordination_fixture(tmp_path,already_complete=already_complete)
    closure={};bridge._peer_coordination_closure(tmp_path,declaration,coordination,closure)
    assert closure=={snapshot["path"]:snapshot["sha256"]}
    (tmp_path/snapshot["path"]).write_text("tampered closed proof")
    with pytest.raises(bridge.ExtensionGateError,match="SHA_or_size"):
        bridge._peer_coordination_closure(tmp_path,declaration,coordination,{})


@pytest.mark.parametrize("change",["missing","integer_takeover","old_mode","changed_lastlive",
    "missing_completion","claim_while_live","missing_history_map","missing_full_map",
    "missing153","float153","unverified","changed_initial","extra_field"])
def test_peer_observation_cannot_be_upgraded_to_a_completed_Yang_claim(tmp_path,change):
    declaration,coordination,_=peer_coordination_fixture(tmp_path)
    if change=="missing":coordination=None
    elif change=="integer_takeover":coordination["global_lock_takeover"]=1
    elif change=="old_mode":coordination["last_verified_live_peer"]["mode"]="concurrent_no_cache_wrapper"
    elif change=="changed_lastlive":coordination["last_verified_live_peer"]["pid"]=7
    elif change=="missing_completion":coordination["completed_yang_proof"]=None
    elif change=="claim_while_live":coordination["global_lock_takeover"]=False
    elif change=="missing_history_map":coordination["completed_yang_proof"]["transport_history"]["provenance_closure"]={}
    elif change=="missing_full_map":coordination["completed_yang_proof"]["transport_completion_proof"]["provenance_closure"]={}
    elif change=="missing153":coordination["completed_yang_proof"]["transport_completion_proof"]["verified_records"]=152
    elif change=="float153":coordination["completed_yang_proof"]["transport_completion_proof"]["verified_records"]=153.0
    elif change=="unverified":coordination["completed_yang_proof"]["transport_completion_proof"]["already_complete_offline_verified"]=False
    elif change=="changed_initial":coordination["initial_yang_process"]["pid"]=7
    else:coordination["unexpected_permission"]=True
    with pytest.raises(bridge.ExtensionGateError):
        bridge._peer_coordination_closure(tmp_path,declaration,coordination,{})


def test_common_gain_unknown_SI_is_allowed_but_unknown_relative_gain_is_not(tmp_path):
    unit = write(tmp_path, "unit.json", {"common_positive_gain_verified": True,
        "calibration": {"common_positive_gain_verified": True, "SI_unit_verified": False,
                        "common_scale_rational": "17/3"}})
    row = {"unit_evidence": unit, "preprocessing_qc": {"reference": "CAR_exact_clinic11",
        "normalization_applied": False, "stored_unit": "calibrated_numeric_unknown_SI",
        "common_positive_gain_verified": True}}
    bridge._unit_qc(tmp_path, row, {})
    row["preprocessing_qc"]["per_channel_gain_unknown"] = True
    with pytest.raises(bridge.ExtensionGateError, match="relative_channel_gain"):
        bridge._unit_qc(tmp_path, row, {})
    row["preprocessing_qc"].pop("per_channel_gain_unknown")
    row["preprocessing_qc"]["common_positive_gain_verified"] = False
    with pytest.raises(bridge.ExtensionGateError, match="common_positive_gain"):
        bridge._unit_qc(tmp_path, row, {})


def test_actual_kernel_uses_gain_invariant_float64_RMS_before_cast():
    x = torch.from_numpy(np.stack([signal(200), signal(201)]))
    from p3work.spatial_geometry import joint_trial_rms
    assert joint_trial_rms(x).dtype == torch.float64
    torch.testing.assert_close(joint_trial_rms(x.double()*17.25), joint_trial_rms(x), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("path", ["../escape.json", "/tmp/absolute.json", ".env", ".aws/token.json",
    "data/external/query.json", "cache/legacy.npz", "cache/payload.pkl"])
def test_paths_do_not_reach_credentials_external_or_legacy(tmp_path, path):
    with pytest.raises((bridge.ExtensionGateError, ValueError)):
        bridge.local(tmp_path, path, exists=False)


def test_symlink_cannot_escape_or_alias_cache(tmp_path):
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bridge.local(tmp_path, "alias/member.json", exists=False)


def test_pending_plan_never_fabricates_epoch_or_completed_fit(capsys):
    assert main(["--arm", "BASE4_PLUS_YANG"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["epochs"] is None and result["optimizer_updates"] is None
    assert result["actual_fits"] == 0 and result["training_started"] is False
    assert result["old_meta_entry_compatible"] is False


def test_execute_missing_GPU_or_budget_is_closed_before_cache_read(monkeypatch, capsys):
    monkeypatch.setattr(extension, "build_plan", lambda *a, **k: pytest.fail("must not inspect cache"))
    assert main(["--arm", "BASE4_PLUS_DREYER", "--execute"]) == 2
    assert json.loads(capsys.readouterr().out)["training_started_this_invocation"] is False
    with pytest.raises(bridge.ExtensionGateError, match="manual_GPU"):
        extension.execute_extension(".", arm="BASE4_PLUS_YANG", seed=20261018,
            extension_manifest="absent", extension_sha256="a"*64, baseline_receipt="absent", budget_path="absent")


def budget_fixture(root):
    plan = {"arm": "BASE4_PLUS_DREYER", "seed": 20261018, "plan_sha256": "d"*64, "optimizer_updates": 64}
    budget = {"stage": "source_extension_training", "arm": plan["arm"], "seed": plan["seed"],
        "plan_sha256": plan["plan_sha256"], "max_jobs": 1, "max_download_bytes": 0,
        "max_wall_seconds": 60, "max_gpu_seconds": 60, "max_fit_invocations": 4,
        "max_observed_optimizer_calls": 68}
    write(root, "budget.json", budget)
    return plan, budget


@pytest.mark.parametrize("field,value", [("max_jobs", 2), ("max_download_bytes", 1),
    ("max_wall_seconds", 0), ("max_gpu_seconds", 59), ("max_fit_invocations", 0),
    ("max_observed_optimizer_calls", 63), ("seed", 20261019), ("stage", "meta_training")])
def test_exact_job_wall_GPU_and_replay_call_budgets(tmp_path, field, value):
    plan, budget = budget_fixture(tmp_path)
    budget[field] = value
    write(tmp_path, "budget.json", budget)
    with pytest.raises(bridge.ExtensionGateError, match="budget"):
        extension._read_budget(tmp_path, "budget.json", plan)


def test_unknown_budget_fields_rejected_not_copied_as_credentials(tmp_path):
    plan, budget = budget_fixture(tmp_path)
    budget["credential"] = "software-placeholder"
    write(tmp_path, "budget.json", budget)
    with pytest.raises(bridge.ExtensionGateError):
        extension._read_budget(tmp_path, "budget.json", plan)


def test_claim_prevents_second_output_or_revised_trajectory(tmp_path):
    cache = type("Cache", (), {"root": tmp_path})()
    plan, _ = budget_fixture(tmp_path)
    descriptor = bridge.descriptor(tmp_path, tmp_path / "budget.json")
    (tmp_path / "runs/source_extensions/claims").mkdir(parents=True)
    path, state = extension._claim_state(cache, plan, descriptor, {"device": "fixture"}, "runs/first")
    assert state["fit_invocations"] == 0
    with pytest.raises(bridge.ExtensionGateError, match="duplicate_fit"):
        extension._claim_state(cache, plan, descriptor, {"device": "fixture"}, "runs/second")


def dispatched_claim_fixture(tmp_path):
    cache = type("Cache", (), {"root": tmp_path})()
    plan, _ = budget_fixture(tmp_path)
    descriptor = bridge.descriptor(tmp_path, tmp_path / "budget.json")
    (tmp_path / "runs/source_extensions/claims").mkdir(parents=True)
    path, state = extension._claim_state(cache, plan, descriptor, {"device": "fixture"}, "runs/first")
    state.update(fit_invocations=1, active_invocation=True)
    core.atomic_json(path, state)
    return cache, plan, descriptor, path, state


def test_dispatched_claim_never_recreates_missing_wall_ledger(tmp_path):
    cache, plan, desc, path, state = dispatched_claim_fixture(tmp_path)
    with pytest.raises(bridge.ExtensionGateError, match="wall_ledger_missing"):
        extension._claim_state(cache, plan, desc, {"device": "fixture"}, "runs/first")
    assert not (tmp_path / "runs/first/operational_budget.json").exists()
    assert json.loads(path.read_text()) == state


def test_dispatched_claim_never_recreates_lost_fit_directory(tmp_path):
    cache, plan, desc, path, state = dispatched_claim_fixture(tmp_path)
    core.operational_budget(tmp_path / "runs/first", initial_seconds=60)
    with pytest.raises(bridge.ExtensionGateError, match="fit_recovery_artifact_missing"):
        extension._claim_state(cache, plan, desc, {"device": "fixture"}, "runs/first")
    assert not (tmp_path / "runs/first/fit").exists()
    assert json.loads(path.read_text()) == state


def test_dispatched_claim_refuses_existing_but_rolled_back_wall(tmp_path):
    cache, plan, desc, path, state = dispatched_claim_fixture(tmp_path)
    state["observed_wall_seconds_highwater"] = 2.0
    core.atomic_json(path, state)
    core.operational_budget(tmp_path / "runs/first", initial_seconds=60)
    with pytest.raises(bridge.ExtensionGateError, match="wall_budget_or_highwater_rollback"):
        extension._claim_state(cache, plan, desc, {"device": "fixture"}, "runs/first")


def test_dispatched_claim_resumes_only_matching_evidence_and_keeps_uncertain_call(tmp_path, monkeypatch):
    cache, plan, desc, path, state = dispatched_claim_fixture(tmp_path)
    plan.update(job={"software_fixture": True}, epochs=1)
    expected = {"software_math_fixture_identity": True}
    monkeypatch.setattr(core, "fit_identity", lambda *a, **k: expected)
    core.operational_budget(tmp_path / "runs/first", initial_seconds=60)
    fit = tmp_path / "runs/first/fit"
    write(tmp_path, "runs/first/fit/identity.json", expected)
    core.atomic_checkpoint(fit / "last_state.pt", {"identity": expected,
        "identity_sha256": digest(expected), "completed_updates": 1, "observed_optimizer_calls_total": 1})
    write(tmp_path, "runs/first/fit/execution_progress.json", {"identity_sha256": digest(expected),
        "observed_optimizer_calls_total": 1})
    _, recovered = extension._claim_state(cache, plan, desc, {"device": "fixture"}, "runs/first")
    assert recovered["uncertain_optimizer_calls"] == 1
    assert recovered["observed_optimizer_calls_highwater"] == 1
    # A later restored old journal cannot erase an already observed call.
    write(tmp_path, "runs/first/fit/execution_progress.json", {"identity_sha256": digest(expected),
        "observed_optimizer_calls_total": 0})
    with pytest.raises(bridge.ExtensionGateError, match="call_journal_rollback"):
        extension._claim_state(cache, plan, desc, {"device": "fixture"}, "runs/first")


def test_missing_claim_with_existing_checkpoint_is_not_a_new_fit(tmp_path):
    cache = type("Cache", (), {"root": tmp_path})()
    plan, _ = budget_fixture(tmp_path)
    descriptor = bridge.descriptor(tmp_path, tmp_path / "budget.json")
    (tmp_path / "runs/first/fit").mkdir(parents=True)
    with pytest.raises(bridge.ExtensionGateError, match="missing_claim"):
        extension._claim_state(cache, plan, descriptor, {}, "runs/first")


@pytest.mark.parametrize("artifact", ["operational_budget.json", "progress.json", "source_extension_plan.json"])
def test_lost_claim_with_any_remaining_trajectory_evidence_never_starts_again(tmp_path, artifact):
    cache = type("Cache", (), {"root": tmp_path})()
    plan, _ = budget_fixture(tmp_path)
    desc = bridge.descriptor(tmp_path, tmp_path / "budget.json")
    write(tmp_path, "runs/first/"+artifact, {"software_fixture": True})
    with pytest.raises(bridge.ExtensionGateError, match="missing_claim"):
        extension._claim_state(cache, plan, desc, {}, "runs/first")
    assert not (tmp_path / "runs/first/fit").exists()


def test_completed_receipt_cannot_be_constructed_from_flags_only(tmp_path):
    write(tmp_path, "false_receipt.json", {"schema_version": extension.SCHEMA,
        "status": "SOURCE_EXTENSION_F0_COMPLETE", "synthetic": True})
    with pytest.raises(bridge.ExtensionGateError, match="actual_independent"):
        extension.verify_completed_extension(tmp_path, "false_receipt.json")


def test_primary_bridge_refuses_extension_status_even_with_same_shape(tmp_path):
    write(tmp_path, "receipt.json", {"status": "SOURCE_EXTENSION_F0_COMPLETE", "synthetic": False})
    contract = json.loads((ROOT / "contracts/input_contract.json").read_text())
    with pytest.raises(ValueError, match="four-source"):
        load_verified_source(tmp_path, "receipt.json", contract)


def test_actual_resumable_kernel_replays_CPU_math_fixture_exactly(tmp_path):
    """Real source optimizer/checkpoint/RNG code on tiny synthetic inputs only."""
    for relative in set(core.POLICY_FILES) | set(core.CODE_FILES):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    arrays = [np.stack([signal(300), signal(301)])]
    rows = [{"dataset": "synthetic_math_only", "subject_id": "1", "event_uid": f"math-{i}"} for i in range(2)]
    cache = MathCache(tmp_path, arrays, np.array([0, 1], dtype=np.int64), rows, [(0, 0), (0, 1)], {})
    job = {"job_id": "synthetic-math", "phase": "final", "heldout_dataset": None,
           "seed": 20261018, "train_indices": [0, 1], "validation_indices": []}
    cfg = {**core.SOURCE_CONFIG, "epochs": 2, "steps_per_epoch": 2, "batch_size": 2,
           "checkpoint_every_updates": 1, "threads": 1}
    runtime = {"device": "cpu", "scope": "SOFTWARE_MATH_FIXTURE_NOT_SCIENTIFIC_FIT"}
    continuous = tmp_path / "runs/continuous"
    resumed = tmp_path / "runs/resumed"
    a = core.run_fit(cache, job, continuous, runtime=runtime, epochs=2, config=cfg, synthetic_fixture=True)
    paused = core.run_fit(cache, job, resumed, runtime=runtime, epochs=2, config=cfg,
                          max_new_updates=1, synthetic_fixture=True)
    assert not paused["complete"] and paused["committed_optimizer_updates"] == 1
    b = core.run_fit(cache, job, resumed, runtime=runtime, epochs=2, config=cfg, synthetic_fixture=True)
    assert a["complete"] and b["complete"] and a["committed_optimizer_updates"] == 4
    left, right = core.verified_checkpoint(continuous / "last_state.pt"), core.verified_checkpoint(resumed / "last_state.pt")
    for field in ("model", "optimizer", "rng", "completed_updates", "history"):
        assert run_controlled_pilot._same_tree(left[field], right[field])
    assert json.loads((resumed / "status.json").read_text())["status"] == "SYNTHETIC_FIXTURE_COMPLETE"
    with pytest.raises(core.SourceTrainingError, match="real completed"):
        core.verify_complete_fit(cache, job, resumed)
    with (resumed / "last_state.pt").open("ab") as stream:
        stream.write(b"torn")
    with pytest.raises(core.SourceTrainingError, match="SHA256"):
        core.verified_checkpoint(resumed / "last_state.pt")
