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
    audit="audit/source_extensions_20261019/Dreyer2023/"+bridge.PARALLEL_DREYER_PLAN_SHA
    for relative in (bridge.PARALLEL_WRAPPER, bridge.PARALLEL_PARENT_WRAPPER):
        target=root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    wrapper=bridge.descriptor(root,root/bridge.PARALLEL_WRAPPER)
    history_sources,rows=[],[]
    totals={"provider_body_bytes":0,"r2_get_body_bytes":0}
    walls=[]
    original={"provider_body_bytes":1000,"r2_get_body_bytes":2000,"wall_seconds":100}
    for i,(label,plan_path,ledger_path,_) in enumerate(bridge.PARALLEL_PRIOR_SOURCES,1):
        budgets=dict(original)
        if label == "V3":
            # A prior ceiling can retain fractional rounding reserve.
            budgets["wall_seconds"]=92
        plan={"schema_version":"SYNTHETIC_SOFTWARE_HISTORY_ONLY","budgets":budgets,"source":label}
        plan_sha=digest(plan);plan["plan_sha256"]=plan_sha
        pdesc=write(root,plan_path,plan)
        charged={"provider_body_bytes":i*10,"r2_get_body_bytes":i*20}
        wall=i+.25
        ledger={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":plan_sha,
            "charged":charged,"elapsed_seconds":wall,"history":[
                {"kind":kind,"reserved_bytes":value+5,"verified_bytes":value,"status":"SUCCESS_FULL_BODY_VERIFIED"}
                for kind,value in charged.items()]}
        ldesc=write(root,ledger_path,ledger)
        rows.append({"source":label,"plan":pdesc,"closed_ledger":ldesc,"plan_sha256":plan_sha,
                     "charged_body_bytes":charged,"elapsed_seconds":wall})
        history_sources.append((label,plan_path,ledger_path,plan_sha));walls.append(wall)
        for kind,value in charged.items():totals[kind]+=value
    # Only expected historical fixture hashes change. Production always uses
    # the four immutable release/audit plan hashes declared in the bridge.
    monkeypatch.setattr(bridge,"PARALLEL_PRIOR_SOURCES",tuple(history_sources))
    original_remaining={k:original[k]-v for k,v in totals.items()}
    original_remaining["wall_seconds"]=original["wall_seconds"]-math.ceil(math.fsum(walls))
    parent_remaining={k:budgets[k]-v for k,v in charged.items()}
    parent_remaining["wall_seconds"]=math.floor(budgets["wall_seconds"]-wall)
    remaining={k:min(v,parent_remaining[k]) for k,v in original_remaining.items()}
    parent={"source":"V3","authorized_caps":{k:budgets[k] for k in (*totals,"wall_seconds")},
        "actual_charged_body_bytes":charged,"actual_elapsed_seconds":wall,"remaining_caps":parent_remaining}
    prior={"accounting_schema":"paper3-dreyer-four-ledger-budget-v1","original_caps":original,
        "closed_history_sources":rows,"deducted_body_bytes":totals,"deducted_elapsed_seconds":math.fsum(walls),
        "deducted_wall_seconds":math.ceil(math.fsum(walls)),"wall_rounding":"ceil_sum_actual_elapsed_once",
        "original_caps_minus_all_actual_usage":original_remaining,"immediate_parent_conservation_bound":parent,
        "remaining_cap_policy":"minimum_of_initial_caps_minus_all_actual_usage_and_immediate_parent_caps_minus_its_actual_usage",
        "wall_remaining_rounding":"minimum_of_original_minus_ceil_cumulative_and_floor_parent_minus_actual",
        "retained_prior_wall_rounding_seconds":original_remaining["wall_seconds"]-remaining["wall_seconds"],
        "remaining_caps":remaining,"uses_original_caps_not_V2_or_V3_remaining_caps":True,
        "carried_over_verified_R2_body_bytes":0,"old_cache_or_receipts_adopted_as_new_records":False}
    write(root,audit+"/frozen_plan.json",{"plan_sha256":bridge.PARALLEL_DREYER_PLAN_SHA,"budgets":remaining})
    declaration={"schema_version":"paper3-dreyer-parallel-root-declaration-v1","dataset":"Dreyer2023",
        "plan_sha256":bridge.PARALLEL_DREYER_PLAN_SHA,"concurrent_yang_plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,
        "wrapper_implementation":wrapper,"only_dreyer_entrypoint":bridge.PARALLEL_WRAPPER,
        "ordinary_dreyer_prepare_forbidden":True,"GPU_actions":0,"actual_fits":0,
        "sealed_external_access_allowed":False,"limits":bridge.PARALLEL_LIMITS,
        "root_authorization":bridge.PARALLEL_AUTHORIZATION,"prior_budget_evidence":prior,
        "new_record_namespace":"cache/source_extensions_20261019/Dreyer2023/"+bridge.PARALLEL_DREYER_PLAN_SHA,
        "initial_yang_process":{"pid":2026,"starttime_ticks":42,"boot_id":"abbddfa5-6cc8-471e-8519-0a488f6ef2bb",
            "command_sha256":"d"*64,"plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA}}
    decl=write(root,audit+"/parallel_root_declaration.json",declaration)
    evidence={"schema_version":"paper3-dreyer-parallel-cpu-execution-v1",
        "status":"RECORDS_VERIFIED_FINAL_BACKUP_PENDING","dataset":"Dreyer2023",
        "plan_sha256":bridge.PARALLEL_DREYER_PLAN_SHA,
        "concurrent_yang_plan_sha256":bridge.PARALLEL_YANG_PLAN_SHA,"record_count":520,
        "record_ids_sha256":digest([f"software-{i}" for i in range(520)]),
        "scientific_kernel_unchanged":True,"GPU_actions":0,"actual_fits":0,
        "external_payload_accessed":False,"carried_over_verified_R2_body_bytes":0,
        "root_declaration":decl,"wrapper_implementation":wrapper,"server_actions":False,"limits":bridge.PARALLEL_LIMITS}
    item=write(root,audit+"/parallel_execution_receipt.json",evidence)
    manifest={"dataset":"Dreyer2023","plan_sha256":bridge.PARALLEL_DREYER_PLAN_SHA,"operational_evidence":item,
        "shards":[{"record_id":f"software-{i}"} for i in range(520)]}
    return audit,manifest,{"operational_evidence":item},evidence


def test_parallel_CPU_proofs_are_additional_GET_objects_not_extra_training_permission(tmp_path,monkeypatch):
    audit,manifest,status,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    extra=bridge._parallel_evidence(tmp_path,manifest,status,audit,{})
    assert len(extra)==2 and all(role=="audit" for _,role in extra)
    assert bridge._parallel_evidence(tmp_path,{"dataset":"Yang2025"},{},"unused",{})==[]
    with pytest.raises(bridge.ExtensionGateError,match="seven_GET_objects_required"):
        bridge._parallel_evidence(tmp_path,{"dataset":"Dreyer2023"},{},"unused",{})


@pytest.mark.parametrize("key,value",[("GPU_actions",1),("actual_fits",1),
    ("external_payload_accessed",True),("carried_over_verified_R2_body_bytes",1),
    ("concurrent_yang_plan_sha256","f"*64)])
def test_parallel_CPU_proof_never_expands_data_GPU_or_budget_authority(tmp_path,monkeypatch,key,value):
    audit,manifest,status,evidence=parallel_evidence_fixture(tmp_path,monkeypatch)
    evidence[key]=value
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="scope_or_budget_changed"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def rewrite_parallel_declaration(root,audit,manifest,status,evidence,declaration):
    evidence["root_declaration"]=write(root,audit+"/parallel_root_declaration.json",declaration)
    item=write(root,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item


def test_parallel_v4_binds_four_closed_histories_and_never_reads_live_process(tmp_path,monkeypatch):
    audit,manifest,status,_=parallel_evidence_fixture(tmp_path,monkeypatch)
    read=Path.read_text
    def no_live_process(path,*args,**kwargs):
        assert not str(path).startswith("/proc/")
        return read(path,*args,**kwargs)
    monkeypatch.setattr(Path,"read_text",no_live_process)
    before={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    closure={};bridge._parallel_evidence(tmp_path,manifest,status,audit,closure)
    assert before=={p:p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert all(p in closure and q in closure for _,p,q,_ in bridge.PARALLEL_PRIOR_SOURCES)
    assert bridge.PARALLEL_WRAPPER in closure and bridge.PARALLEL_PARENT_WRAPPER in closure


@pytest.mark.parametrize("change",["missing_aux","missing_v3","duplicate_parent","refund_failed_transfer",
    "unclosed_wall","open_transfer","NaN_wall","old_remaining_cap","round_each_wall",
    "refund_prior_rounding","omit_parent_bound","refund_parent_bytes"])
def test_parallel_v4_prior_usage_cannot_be_lost_or_refunded(tmp_path,monkeypatch,change):
    audit,manifest,status,evidence=parallel_evidence_fixture(tmp_path,monkeypatch)
    declaration=json.loads((tmp_path/evidence["root_declaration"]["path"]).read_text())
    prior=declaration["prior_budget_evidence"];rows=prior["closed_history_sources"]
    if change=="missing_aux":rows.pop(2)
    elif change=="missing_v3":rows.pop()
    elif change=="duplicate_parent":rows[2]=copy.deepcopy(rows[1])
    elif change=="omit_parent_bound":prior.pop("immediate_parent_conservation_bound")
    elif change=="refund_parent_bytes":
        prior["immediate_parent_conservation_bound"]["remaining_caps"]["provider_body_bytes"]+=1
    elif change=="refund_prior_rounding":
        prior["remaining_caps"]["wall_seconds"]=prior["original_caps_minus_all_actual_usage"]["wall_seconds"]
        prior["retained_prior_wall_rounding_seconds"]=0
        plan=json.loads((tmp_path/audit/"frozen_plan.json").read_text())
        plan["budgets"]["wall_seconds"]=prior["remaining_caps"]["wall_seconds"]
        write(tmp_path,audit+"/frozen_plan.json",plan)
    elif change=="old_remaining_cap":
        prior["remaining_caps"]["provider_body_bytes"]+=1
        plan=json.loads((tmp_path/audit/"frozen_plan.json").read_text())
        plan["budgets"]["provider_body_bytes"]+=1;write(tmp_path,audit+"/frozen_plan.json",plan)
    elif change=="round_each_wall":
        prior["deducted_wall_seconds"]=sum(math.ceil(row["elapsed_seconds"]) for row in rows)
        prior["remaining_caps"]["wall_seconds"]=prior["original_caps"]["wall_seconds"]-prior["deducted_wall_seconds"]
        plan=json.loads((tmp_path/audit/"frozen_plan.json").read_text())
        plan["budgets"]["wall_seconds"]=prior["remaining_caps"]["wall_seconds"]
        write(tmp_path,audit+"/frozen_plan.json",plan)
    else:
        row=rows[0];path=row["closed_ledger"]["path"];ledger=json.loads((tmp_path/path).read_text())
        if change=="refund_failed_transfer":ledger["history"][0].update(status="FAILURE_FULL_RESERVATION_CONSUMED",verified_bytes=0)
        elif change=="unclosed_wall":ledger["wall_intent"]={}
        elif change=="open_transfer":ledger["active_transfer"]={}
        elif change=="NaN_wall":ledger["elapsed_seconds"]=float("nan")
        row["closed_ledger"]=write(tmp_path,path,ledger)
    rewrite_parallel_declaration(tmp_path,audit,manifest,status,evidence,declaration)
    with pytest.raises(bridge.ExtensionGateError):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def test_parallel_v4_retains_immediate_parent_rounding_reserve(tmp_path,monkeypatch):
    audit,manifest,status,evidence=parallel_evidence_fixture(tmp_path,monkeypatch)
    declaration=json.loads((tmp_path/evidence["root_declaration"]["path"]).read_text())
    prior=declaration["prior_budget_evidence"]
    assert prior["original_caps_minus_all_actual_usage"]["wall_seconds"]==89
    assert prior["immediate_parent_conservation_bound"]["remaining_caps"]["wall_seconds"]==87
    assert prior["retained_prior_wall_rounding_seconds"]==2
    assert prior["remaining_caps"]["wall_seconds"]==87
    bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def test_parallel_v4_rejects_old_plan_and_old_wrapper_even_with_matching_fixture_descriptors(tmp_path,monkeypatch):
    audit,manifest,status,evidence=parallel_evidence_fixture(tmp_path,monkeypatch)
    old=bridge.PARALLEL_PRIOR_SOURCES[1][3]
    manifest["plan_sha256"]=old
    with pytest.raises(bridge.ExtensionGateError,match="exact_declared"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})
    manifest["plan_sha256"]=bridge.PARALLEL_DREYER_PLAN_SHA
    evidence["wrapper_implementation"]=bridge.descriptor(tmp_path,tmp_path/bridge.PARALLEL_PARENT_WRAPPER)
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="declared_Dreyer_CPU_wrapper"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


def test_parallel_v4_stale_ledger_SHA_and_resource_caps_are_rejected(tmp_path,monkeypatch):
    audit,manifest,status,evidence=parallel_evidence_fixture(tmp_path,monkeypatch)
    declaration=json.loads((tmp_path/evidence["root_declaration"]["path"]).read_text())
    row=declaration["prior_budget_evidence"]["closed_history_sources"][0]
    path=row["closed_ledger"]["path"];ledger=json.loads((tmp_path/path).read_text())
    ledger["elapsed_seconds"]+=.1;write(tmp_path,path,ledger)
    with pytest.raises(bridge.ExtensionGateError,match="SHA_or_size"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})
    evidence["limits"]={**bridge.PARALLEL_LIMITS,"cpu_threads":2}
    item=write(tmp_path,audit+"/parallel_execution_receipt.json",evidence)
    manifest["operational_evidence"]=status["operational_evidence"]=item
    with pytest.raises(bridge.ExtensionGateError,match="scope_or_budget"):
        bridge._parallel_evidence(tmp_path,manifest,status,audit,{})


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
