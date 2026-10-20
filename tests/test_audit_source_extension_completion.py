"""Software fixtures only: no EEG fits, downloads, R2 or CUDA."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

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
