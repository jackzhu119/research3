"""Software fixtures only: no EEG fits, downloads, R2 or CUDA."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import audit_source_extension_completion as audit


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def valid_signal():
    x=np.random.default_rng(42).normal(size=(3,128))*1e-6
    x-=x.mean(axis=0,keepdims=True);x-=x.mean(axis=1,keepdims=True)
    return x.astype("float32")


def ledger_fixture(root):
    plan={"plan_sha256":"a"*64,"budgets":{"provider_body_bytes":100,"r2_get_body_bytes":200,"wall_seconds":10}}
    value={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":"a"*64,
        "charged":{"provider_body_bytes":40,"r2_get_body_bytes":80},"elapsed_seconds":2.5,
        "history":[{"kind":"provider_body_bytes","reserved_bytes":50,"verified_bytes":40,"status":"SUCCESS_FULL_BODY_VERIFIED"},
                   {"kind":"r2_get_body_bytes","reserved_bytes":80,"verified_bytes":80,"status":"SUCCESS_FULL_BODY_VERIFIED"}]}
    path=root/"audit/ledger.json";write(path,value)
    return plan,value,path


def test_closed_ledger_is_readonly_and_exact(tmp_path):
    plan,value,path=ledger_fixture(tmp_path);before=path.read_bytes();closure={}
    result=audit.closed_ledger(tmp_path,path,plan,80,closure)
    assert result["charged"]==value["charged"] and result["required_full_GET_bytes_lower_bound"]==80
    assert path.read_bytes()==before and "audit/ledger.json" in closure


@pytest.mark.parametrize("change",["wall_intent","active_transfer","bad_sha","wall_exhausted","wall_nan","bool_wall",
    "body_exhausted","byte_refund","unknown_transfer","negative_reservation","bool_reservation","missing_charge","insufficient_get"])
def test_bad_or_live_ledger_never_passes(tmp_path,change):
    plan,value,path=ledger_fixture(tmp_path);required=80
    if change in {"wall_intent","active_transfer"}:value[change]={}
    elif change=="bad_sha":value["plan_sha256"]="b"*64
    elif change=="wall_exhausted":value["elapsed_seconds"]=10.01
    elif change=="wall_nan":value["elapsed_seconds"]=float("nan")
    elif change=="bool_wall":value["elapsed_seconds"]=True
    elif change=="body_exhausted":value["charged"]["provider_body_bytes"]=101
    elif change=="byte_refund":value["history"][0]["status"]="FAILURE_FULL_RESERVATION_CONSUMED"
    elif change=="unknown_transfer":value["history"][0]["status"]="FIT_RUNNING"
    elif change=="negative_reservation":value["history"][0]["reserved_bytes"]=-1
    elif change=="bool_reservation":value["history"][0]["reserved_bytes"]=True
    elif change=="missing_charge":value["charged"].pop("provider_body_bytes")
    elif change=="insufficient_get":required=81
    write(path,value)
    with pytest.raises(audit.CompletionAuditError):audit.closed_ledger(tmp_path,path,plan,required,{})


@pytest.mark.parametrize("status",["FAILURE_FULL_RESERVATION_CONSUMED","HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"])
def test_failed_transfers_keep_full_reserved_charge(tmp_path,status):
    plan,value,path=ledger_fixture(tmp_path)
    value["history"][0]["status"]=status;value["charged"]["provider_body_bytes"]=50
    write(path,value)
    assert audit.closed_ledger(tmp_path,path,plan,80,{})["charged"]["provider_body_bytes"]==50


@pytest.mark.parametrize("dataset",sorted(audit.SPECS))
def test_incomplete_dataset_does_not_read_live_ledger_or_write_report(tmp_path,monkeypatch,capsys,dataset):
    spec=audit.SPECS[dataset];status=tmp_path/f"audit/source_extensions_20261019/{dataset}/{spec['sha']}/coordinator_status.json"
    write(status,{"status":"RUNNING"})
    original=audit.read
    def no_partial_reads(path):
        assert Path(path)==status
        return original(path)
    monkeypatch.setattr(audit,"read",no_partial_reads)
    report="audit/new_completion_receipt.json"
    assert audit.main(["--root",str(tmp_path),"--dataset",dataset,"--report",report])==3
    result=json.loads(capsys.readouterr().out)
    assert result["status"]=="PENDING_COMPLETE_DATA_GATE" and not (tmp_path/report).exists()


def test_missing_completion_status_does_not_read_plan_or_ledger(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,"read",lambda _:pytest.fail("no file may be read"))
    assert audit.audit_completion(tmp_path,"Ma2022")["status"]=="PENDING_COMPLETE_DATA_GATE"


def test_unregistered_dataset_is_rejected_before_access(tmp_path):
    with pytest.raises(audit.CompletionAuditError,match="registered_permitted"):
        audit.audit_completion(tmp_path,"SEALED_EXTERNAL")


def test_complete_status_without_real_bound_plan_cannot_pass(tmp_path,capsys):
    spec=audit.SPECS["Ma2022"]
    write(tmp_path/f"audit/source_extensions_20261019/Ma2022/{spec['sha']}/coordinator_status.json",{"status":spec["completion"]})
    write(tmp_path/spec["plan"],{"plan_sha256":spec["sha"],"dataset":"Ma2022","role":spec["role"]})
    assert audit.main(["--root",str(tmp_path),"--dataset","Ma2022","--report","audit/no.json"])==1
    assert json.loads(capsys.readouterr().out)["error_code"]=="exact_current_frozen_dataset_plan_required"
    assert not (tmp_path/"audit/no.json").exists()


def test_cli_only_complete_pass_can_write_new_immutable_receipt(tmp_path,monkeypatch,capsys):
    result={"status":"PASS_COMPLETE_NATIVE_CACHE_AND_FULL_GET_EVIDENCE","dataset":"Ma2022","actual_scientific_fits":0}
    monkeypatch.setattr(audit,"audit_completion",lambda *a:deepcopy(result))
    args=["--root",str(tmp_path),"--dataset","Ma2022","--report","audit/pass.json"]
    assert audit.main(args)==0
    path=tmp_path/"audit/pass.json";before=path.read_bytes()
    assert json.loads(before)==result
    capsys.readouterr()
    assert audit.main(args)==1 and path.read_bytes()==before


@pytest.mark.parametrize("path",["results/no.json","audit/no.txt","audit/../sealed/query.json",".env","audit/.env.json"])
def test_report_path_never_targets_unsafe_or_non_audit_files(tmp_path,monkeypatch,path):
    monkeypatch.setattr(audit,"audit_completion",lambda *a:{"status":"PASS_COMPLETE_NATIVE_CACHE_AND_FULL_GET_EVIDENCE"})
    assert audit.main(["--root",str(tmp_path),"--dataset","Ma2022","--report",path])==1


@pytest.mark.parametrize("change",["missing","duplicate","wrong_count","extra","unknown"])
def test_real_roster_must_be_complete_and_unique(change):
    records=[{"record_id":"one"},{"record_id":"two"}];shards=deepcopy(records)
    plan={"records":records,"record_count":2}
    if change=="missing":shards.pop()
    elif change=="duplicate":shards[1]=deepcopy(shards[0])
    elif change=="wrong_count":plan["record_count"]=3
    elif change=="extra":shards.append({"record_id":"three"})
    elif change=="unknown":shards[0]["record_id"]="unknown"
    with pytest.raises(audit.CompletionAuditError):audit.exact_roster(plan,shards,2)


def test_native_uids_are_unique_across_all_records():
    seen=set();audit.add_unique_uids(seen,["one","two"])
    with pytest.raises(audit.CompletionAuditError):audit.add_unique_uids(seen,["three","one"])


@pytest.mark.parametrize("change",["nan","infinite","zero","wrong_shape","CAR_drift","temporal_drift"])
def test_numeric_qc_detects_actual_signal_defects(change):
    signal=valid_signal()
    if change=="nan":signal[0,0]=np.nan
    elif change=="infinite":signal[0,0]=np.inf
    elif change=="zero":signal[:]=0
    elif change=="wrong_shape":signal=signal[:,:127]
    elif change=="CAR_drift":signal[:,0]+=1e-6
    elif change=="temporal_drift":signal[0]+=1e-6
    with pytest.raises(audit.CompletionAuditError):audit.numeric_qc(signal)


def ma_fixture():
    signal=valid_signal();cal=[{"name":name,"unit":"uV","physical_min":-1.,"physical_max":1.,"digital_min":-32768,"digital_max":32767} for name in ("C3","Cz","C4")]
    native={"dataset":"Ma2022","provider_version":1,"event_clock":"observed_1_based_native_samples_not_generic_sidecar_milliseconds",
        "reference_evidence":"official_v1_sidecar_unipolar_M1","header":{"sfreq_hz":250,"stored_trial_samples":1000,"n_samples":1000,"selected_calibration":cal}}
    unit={"dataset":"Ma2022","role":audit.SPECS["Ma2022"]["role"],"native_unit":"uV","stored_unit":"V","conversion_to_V":1e-6,
        "selected_calibration":deepcopy(cal),"reference_evidence":native["reference_evidence"],"original_B20_prefix_recoverable":False}
    event={"retained_trial_index":0,"analysis_anchor_sample":0,"task_end_sample":1000,"original_context_samples":[0,375],"native_sfreq_hz":250,
        "original_acquisition_ordinal":None,"physical_run_id":None,"original_paid_B20_prefix_recoverable":False,"data_after_context_accessed":False,
        "reference":"CAR_exact_C3_Cz_C4","normalization_applied":False,"native_unit":"uV","stored_unit":"V","pre_RMS_joint_rms_volts":audit.numeric_qc(signal)}
    return native,unit,[event],signal[None]


def test_Ma_native_bounds_unit_and_joint_energy_are_checked():
    audit.ma_units_and_bounds(*ma_fixture())


@pytest.mark.parametrize("change",["clock","conversion","calibration","invented_prefix","context","run","future","normalization","energy"])
def test_Ma_rejects_unverified_units_recovered_ordinals_and_future_access(change):
    native,unit,events,x=ma_fixture();event=events[0]
    if change=="clock":native["event_clock"]="guessed_milliseconds"
    elif change=="conversion":unit["conversion_to_V"]=1
    elif change=="calibration":unit["selected_calibration"][0]["physical_max"]=2
    elif change=="invented_prefix":event["original_acquisition_ordinal"]=0
    elif change=="context":event["original_context_samples"]=[0,1001]
    elif change=="run":event["physical_run_id"]="invented_run_01"
    elif change=="future":event["data_after_context_accessed"]=True
    elif change=="normalization":event["normalization_applied"]=True
    elif change=="energy":event["pre_RMS_joint_rms_volts"]*=2
    with pytest.raises(audit.CompletionAuditError):audit.ma_units_and_bounds(native,unit,events,x)


def source_summary_fixture(root, monkeypatch, dataset="Dreyer2023"):
    """Software control-flow fixture; capability/state gates are explicitly mocked.

    The real bridge native-row parser and this auditor's artifact/ledger readers
    run on small local JSON files. No real completion or scientific fit is made.
    """
    spec=deepcopy(audit.SPECS[dataset]);spec["records"]=1
    count=32 if dataset=="Dreyer2023" else 4
    spec["native_trials"]=count
    audit_dir=root/"audit/software_source_fixture"
    record={"record_id":"one","subject_id":"person1","session_id":"session1"}
    raw_paths=[root/"raw/software_source_fixture/signal.edf"]
    if dataset=="Yang2025":raw_paths.append(root/"raw/software_source_fixture/events.bdf")
    for i,path in enumerate(raw_paths):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(f"software fixture {i} only; not EEG".encode())
    raw_files=[audit.bridge.descriptor(root,path) for path in raw_paths]
    originals=[{"trial_uid":f"trial{i}","dataset":dataset,"source_file_sha256":raw_files[0]["sha256"],
        "label":0 if i<14 else 1} for i in range(count)]
    prefix={"reason":"terminal_start_fixation_prefix_without_native_MI_cue_or_paired800",
        "cue_present":False,"paired800_present":False,"canonical_label_available":False,
        "original_annotation_indices":[193,194]}
    if dataset=="Dreyer2023":
        metadata={"dataset":dataset,"source_file_sha256":raw_files[0]["sha256"],
            "waveform_QC_applied":False,"trials":originals,
            "acquisition_control":{"terminal_incomplete_prefixes":[prefix],
                "incomplete_prefixes_are_not_MI_trials":True}}
    else:
        for i,row in enumerate(originals):
            row.update(event_uid=row.pop("trial_uid"),original_event_index=i,
                label=[0,1,0,None][i],context_start_sample=i*1000,
                context_stop_sample_exclusive=i*1000+1500)
        metadata={"dataset_id":"Yang2025_2C","derived_before_waveform_preprocessing":True,
            "events":originals,"raw_files":{"signal":raw_files[0],"event":raw_files[1]}}
    native_path=root/"cache/software_source_fixture/native.json";write(native_path,metadata)
    excluded_indices={0,count-1} if dataset=="Dreyer2023" else {3}
    retained=[{"event_uid":row.get("trial_uid",row.get("event_uid")),"dataset":dataset,
        "label":row["label"]} for i,row in enumerate(originals) if i not in excluded_indices]
    exclusions=[{"event_uid":row.get("trial_uid",row.get("event_uid")),
        "exclusion_reason":"software_QC_fixture"} for i,row in enumerate(originals) if i in excluded_indices]
    exclusion_path=root/"cache/software_source_fixture/exclusions.json";write(exclusion_path,exclusions)
    shard={**record,"native_metadata":audit.bridge.descriptor(root,native_path),
        "exclusions":audit.bridge.descriptor(root,exclusion_path),"raw_files":raw_files,
        "raw_sha256":raw_files[0]["sha256"],"raw_backup_receipt":"raw_receipt_fixture"}
    if dataset=="Yang2025":
        profile={"schema_version":"paper3-yang-source-profile-v1","dataset_id":"Yang2025_2C",
            "source_only":True,"MI_onset_verified":False,"context_from_marker_s":[2.25,3.75],
            "crop_from_marker_s":[2.5,3.5],"input_contract_sha256":audit.bridge.INPUT_SHA,
            "native_unit_policy":"common_gain_numeric_unresolved_no_absolute_QC"}
        profile_path=root/"cache/software_source_fixture/profile.json";write(profile_path,profile)
        shard["native_profile"]=audit.bridge.descriptor(root,profile_path)
    artifacts=[shard["native_metadata"],shard["exclusions"]]
    write(audit_dir/"one/state.json",{"shard":shard,"artifacts":artifacts})
    manifest={"shards":[shard],"valid_trials":len(retained)}
    manifest_path=root/"cache/software_source_fixture/source_cache.json";write(manifest_path,manifest)
    final_rows=[{"path":f"audit/software_source_fixture/final{i}.json","sha256":"f"*64,
        "bytes":1,"role":"audit","dataset":dataset} for i in range(spec["final_full_GET_objects"])]
    raw_rows=[{**item,"role":"source_development_raw","dataset":dataset} for item in raw_files]
    def receipt_entries(_root,item,_closure):
        assert _root==root
        if item=="raw_receipt_fixture":return deepcopy(raw_rows)
        assert item=="final_receipt_fixture"
        return deepcopy(final_rows)
    monkeypatch.setattr(audit.bridge,"_receipt_entries",receipt_entries)
    monkeypatch.setattr(audit.bridge,"load_extension_cache",lambda *a,**k:
        SimpleNamespace(rows=retained,artifact_sha256={}))
    monkeypatch.setattr(audit.coordinator,"_state_verified",lambda *a:True)
    plan={"plan_sha256":"a"*64,"records":[record],"record_count":1,
        "budgets":{"provider_body_bytes":1000000,"r2_get_body_bytes":1000000,"wall_seconds":10}}
    required=sum(item["bytes"] for item in raw_files+artifacts)+len(final_rows)
    ledger={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":plan["plan_sha256"],
        "charged":{"provider_body_bytes":0,"r2_get_body_bytes":required},"elapsed_seconds":1,
        "history":[{"kind":"r2_get_body_bytes","reserved_bytes":required,"verified_bytes":required,
            "status":"SUCCESS_FULL_BODY_VERIFIED"}]}
    write(audit_dir/"budget_ledger.json",ledger)
    status={"backup_receipt":"final_receipt_fixture"}
    return SimpleNamespace(root=root,dataset=dataset,spec=spec,plan=plan,manifest=manifest,
        manifest_path=manifest_path,audit_dir=audit_dir,status=status,retained=retained,
        native_path=native_path,exclusion_path=exclusion_path,final_rows=final_rows)


def summarize_source(fixture):
    return audit.source_summary(fixture.root,fixture.dataset,fixture.spec,fixture.plan,
        fixture.manifest,fixture.manifest_path,fixture.audit_dir,fixture.status,{})


def test_Dreyer_dynamic_32_native_trials_14_18_before_QC_prefix_has_no_UID(tmp_path,monkeypatch):
    fixture=source_summary_fixture(tmp_path,monkeypatch)
    result=summarize_source(fixture)
    assert result["native_trials"]==result["unique_native_trial_UIDs"]==32
    assert result["native_binary_labels"]=={"0":14,"1":18}
    assert result["retained_labels"]=={"0":13,"1":17}
    assert result["retained_trials"]==30 and result["excluded_trials"]==2
    assert result["native_non_binary_events"]==0 and result["final_full_GET_objects"]==7
    assert result["native_UIDs_sorted_sha256"]==audit.digest(sorted(f"trial{i}" for i in range(32)))
    prefix=audit.read(fixture.native_path)["acquisition_control"]["terminal_incomplete_prefixes"][0]
    assert not {"trial_uid","event_uid","label","cache_index"}&prefix.keys()


def test_Yang_keeps_153_record_registration_and_five_final_objects(tmp_path,monkeypatch):
    assert audit.SPECS["Yang2025"]["records"]==153
    assert audit.SPECS["Yang2025"]["sha"]=="ab9a075ff1fa84548bfe958d3c4b6d6897d6b9dbc29d0ed49d807ec3d4b2b609"
    result=summarize_source(source_summary_fixture(tmp_path,monkeypatch,"Yang2025"))
    assert result["native_trials"]==4 and result["native_binary_labels"]=={"0":2,"1":1}
    assert result["native_non_binary_events"]==1 and result["final_full_GET_objects"]==5


@pytest.mark.parametrize("dataset,count",[("Dreyer2023",5),("Dreyer2023",6),("Dreyer2023",8),
    ("Yang2025",4),("Yang2025",6),("Yang2025",7)])
def test_current_source_final_GET_count_is_exact(tmp_path,monkeypatch,dataset,count):
    fixture=source_summary_fixture(tmp_path,monkeypatch,dataset)
    while len(fixture.final_rows)<count:fixture.final_rows.append(deepcopy(fixture.final_rows[-1]))
    del fixture.final_rows[count:]
    with pytest.raises(audit.CompletionAuditError,match="exact_current_final_full_GET"):
        summarize_source(fixture)


@pytest.mark.parametrize("change",["invented_prefix_exclusion","duplicate_exclusion","retained_exclusion",
    "missing_exclusion","missing_reason","duplicate_retained","unknown_retained","native_total","nonbinary_Dreyer"])
def test_native_pre_QC_UID_denominator_cannot_be_replaced_by_counts(tmp_path,monkeypatch,change):
    fixture=source_summary_fixture(tmp_path,monkeypatch)
    exclusions=audit.read(fixture.exclusion_path)
    if change=="invented_prefix_exclusion":exclusions[0]["event_uid"]="invented_terminal_prefix_UID"
    elif change=="duplicate_exclusion":exclusions[1]=deepcopy(exclusions[0])
    elif change=="retained_exclusion":exclusions[0]["event_uid"]=fixture.retained[0]["event_uid"]
    elif change=="missing_exclusion":exclusions.pop()
    elif change=="missing_reason":exclusions[0].pop("exclusion_reason")
    elif change=="duplicate_retained":fixture.retained[1]["event_uid"]=fixture.retained[0]["event_uid"]
    elif change=="unknown_retained":fixture.retained[0]["event_uid"]="unknown"
    elif change=="native_total":fixture.spec["native_trials"]=40
    elif change=="nonbinary_Dreyer":
        value=audit.read(fixture.native_path);value["trials"][0]["label"]=None
        write(fixture.native_path,value)
        fixture.manifest["shards"][0]["native_metadata"]=audit.bridge.descriptor(tmp_path,fixture.native_path)
    if change in {"invented_prefix_exclusion","duplicate_exclusion","retained_exclusion","missing_exclusion","missing_reason"}:
        write(fixture.exclusion_path,exclusions)
        fixture.manifest["shards"][0]["exclusions"]=audit.bridge.descriptor(tmp_path,fixture.exclusion_path)
    # Keep synthetic descriptors/state consistent to reach the semantic check.
    shard=fixture.manifest["shards"][0]
    write(fixture.audit_dir/"one/state.json",{"shard":shard,"artifacts":[shard["native_metadata"],shard["exclusions"]]})
    with pytest.raises(audit.CompletionAuditError):summarize_source(fixture)


def test_Dreyer_only_current_V5_status_is_considered_not_old_V4(tmp_path,monkeypatch):
    spec=audit.SPECS["Dreyer2023"]
    assert spec["sha"]=="afc8dd0d2d873f455e901c6f702fc2d5649fde15f76873c40a24424c10d48e5a"
    assert spec["plan"].endswith("plan_after_all520_native_review_v5.json")
    assert spec["native_trials"]==20792 and spec["final_full_GET_objects"]==7
    old="971c1a1272fdb25b0f21e7d69c4e7349593d352341bdaa2eaa0725b600dbc1d1"
    write(tmp_path/f"audit/source_extensions_20261019/Dreyer2023/{old}/coordinator_status.json",
        {"status":spec["completion"]})
    monkeypatch.setattr(audit,"read",lambda _:pytest.fail("old completion must not be opened"))
    assert audit.audit_completion(tmp_path,"Dreyer2023")["status"]=="PENDING_COMPLETE_DATA_GATE"


def test_old_V4_plan_cannot_be_substituted_at_current_V5_path(tmp_path,monkeypatch):
    spec=audit.SPECS["Dreyer2023"]
    status=tmp_path/f"audit/source_extensions_20261019/Dreyer2023/{spec['sha']}/coordinator_status.json"
    write(status,{"status":spec["completion"]})
    write(tmp_path/spec["plan"],{"plan_sha256":"971c1a1272fdb25b0f21e7d69c4e7349593d352341bdaa2eaa0725b600dbc1d1",
        "dataset":"Dreyer2023","role":"source_train"})
    original=audit.read;seen=[]
    def record_reads(path):
        seen.append(Path(path));return original(path)
    monkeypatch.setattr(audit,"read",record_reads)
    with pytest.raises(audit.CompletionAuditError,match="exact_current_frozen_dataset_plan"):
        audit.audit_completion(tmp_path,"Dreyer2023")
    assert seen==[status,tmp_path/spec["plan"]]
