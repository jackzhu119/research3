"""Synthetic operational fixtures only; no native data, network, fit or GPU."""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import qualify_dreyer_native_structures as q
from tools import prepare_dreyer_source_cpu_v4 as parent_worker
from p3work.source_extensions import dreyer2023 as frozen_native


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+"\n")
    return path


def redigest(plan):
    plan["plan_sha256"]=q.core.digest({k:v for k,v in plan.items() if k!="plan_sha256"})
    return plan


@pytest.fixture
def parent_fixture(tmp_path,monkeypatch):
    records=[{"record_id":f"software-{i:03}","subject_id":f"software-{i//6}","session_id":"source-day",
        "raw_files":[{"path":f"data/source_extensions_20261019/Dreyer2023/software-{i}.edf","sha256":f"{i:064x}","bytes":100}]} for i in range(520)]
    parent={"dataset":"Dreyer2023","record_count":520,"records":records,
        "budgets":{"provider_body_bytes":10000,"r2_get_body_bytes":20000,"wall_seconds":20000}}
    redigest(parent);monkeypatch.setattr(q,"PARENT",parent["plan_sha256"])
    write(tmp_path/q.PARENT_PLAN,parent)
    audit=tmp_path/f"audit/source_extensions_20261019/Dreyer2023/{q.PARENT}"
    claim=write(audit/"parallel_worker_claim.json",{"active_invocation":False})
    ledger=write(audit/"budget_ledger.json",{"schema_version":"paper3-source-extension-budget-ledger-v1",
        "plan_sha256":q.PARENT,"charged":{"provider_body_bytes":100,"r2_get_body_bytes":200},
        "elapsed_seconds":2.25,"history":[{"kind":"provider_body_bytes","reserved_bytes":100,"verified_bytes":100,"status":"SUCCESS_FULL_BODY_VERIFIED"},
            {"kind":"r2_get_body_bytes","reserved_bytes":200,"verified_bytes":200,"status":"SUCCESS_FULL_BODY_VERIFIED"}]})
    for relative in q.CODE:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text("# SYNTHETIC OPERATIONAL TEST ONLY\n")
    wrapper=SimpleNamespace(fixed_plans=lambda root:(parent,{},{}),closed_history=parent_worker.closed_history)
    monkeypatch.setattr(q,"private_module",lambda *a:wrapper)
    monkeypatch.setattr(q.core,"_state_verified",lambda root,state,p,record:
        state.get("status")=="COMPLETE" and state.get("record_id")==record["record_id"])
    return tmp_path,parent,audit,claim,ledger


def test_make_plan_retains_prior_actual_charges_and_floor_wall(parent_fixture):
    root,parent,_,_,_=parent_fixture
    plan=q.make_plan(root)
    assert plan["budgets"]=={"provider_body_bytes":9900,"r2_get_body_bytes":19800,"wall_seconds":10800}
    assert plan["parent_actual_elapsed_seconds"]==2.25 and plan["parent_actual_charged"]=={"provider_body_bytes":100,"r2_get_body_bytes":200}
    assert plan["raw_cleanup_allowed"] is False and plan["scientific_admission_granted"] is False
    assert q.validate(root,plan)==parent


def test_plan_reuses_only_proven_complete_parent_states(parent_fixture):
    root,parent,audit,_,_=parent_fixture
    write(audit/parent["records"][0]["record_id"]/"state.json",{"status":"COMPLETE","record_id":parent["records"][0]["record_id"]})
    write(audit/parent["records"][1]["record_id"]/"state.json",{"status":"RAW_VERIFIED","record_id":parent["records"][1]["record_id"]})
    plan=q.make_plan(root)
    assert list(plan["reused_complete_parent_states"])==[parent["records"][0]["record_id"]]


@pytest.mark.parametrize("change",["active","wall_open","transfer_open","charge_refund","nan_wall"])
def test_parent_must_be_stopped_with_actual_closed_nonrefunded_history(parent_fixture,change):
    root,_,_,claim_path,ledger_path=parent_fixture
    claim=q.read(claim_path);ledger=q.read(ledger_path)
    if change=="active":claim["active_invocation"]=True
    elif change=="wall_open":ledger["wall_intent"]={}
    elif change=="transfer_open":ledger["active_transfer"]={}
    elif change=="charge_refund":ledger["charged"]["provider_body_bytes"]-=1
    elif change=="nan_wall":ledger["elapsed_seconds"]=float("nan")
    write(claim_path,claim);write(ledger_path,ledger)
    with pytest.raises((q.core.PreparationError,parent_worker.WrapperError)):q.make_plan(root)


@pytest.mark.parametrize("key,value",[("record_count",519),("scientific_admission_granted",True),
    ("raw_cleanup_allowed",True),("EEG_array_samples_read",True),("fits",1),("GPU_actions",1),
    ("sealed_external_payload_access_allowed",True),("dataset","SEALED_EXTERNAL"),
    ("role","source_train"),("maximum_owned_raw_bytes",7*1024**3),("minimum_free_disk_bytes",1)])
def test_rehashed_plan_cannot_expand_frozen_qualification_authority(parent_fixture,key,value):
    root,_,_,_,_=parent_fixture;plan=q.make_plan(root);plan[key]=value;redigest(plan)
    with pytest.raises(q.core.PreparationError):q.validate(root,plan)


@pytest.mark.parametrize("change",["body_reset","wall_reset","negative_body","zero_wall","large_aux_wall","missing_code","extra_code","wrong_code_path","partial_roster"])
def test_rehashed_plan_cannot_reset_budget_or_code_scope(parent_fixture,change):
    root,parent,_,_,_=parent_fixture;plan=q.make_plan(root)
    if change=="body_reset":plan["budgets"]["provider_body_bytes"]=parent["budgets"]["provider_body_bytes"]
    elif change=="wall_reset":plan["budgets"]["wall_seconds"]=parent["budgets"]["wall_seconds"]
    elif change=="negative_body":plan["budgets"]["provider_body_bytes"]=-1
    elif change=="zero_wall":plan["budgets"]["wall_seconds"]=0
    elif change=="large_aux_wall":plan["budgets"]["wall_seconds"]=10801
    elif change=="missing_code":plan["code"].pop(q.ENTRY)
    elif change=="extra_code":plan["code"]["extra.py"]=deepcopy(plan["code"][q.ENTRY])
    elif change=="wrong_code_path":plan["code"][q.ENTRY]=deepcopy(plan["code"][q.CODE[1]])
    elif change=="partial_roster":plan["records"].pop()
    redigest(plan)
    with pytest.raises(q.core.PreparationError):q.validate(root,plan)


def fake_native(unknown=False,outside=False):
    annotations=[{"original_annotation_index":i,"data_record":0,"onset_token":f"+{i}","duration_token":"0",
        "description":code,"onset":Fraction(i),"duration":Fraction(0)} for i,code in enumerate(["32769","12345" if unknown else "768","32770"])]
    if outside:annotations[-1]["onset"]=Fraction(100)
    return {"annotations":annotations,"n_times":5120,"record_count":10,"signal_labels":["C3"],"signal_units":{"C3":"uV"}}


@pytest.mark.parametrize("token,allowed",[("0",True),("65535",True),("65536",False),("-1",False),("x",False),(None,False),(42,False)])
def test_private_numeric_metadata_vocabulary_never_grants_scientific_marker(token,allowed):
    assert (token in q.MetadataNumericMarkers()) is allowed


def test_private_capture_never_patches_public_scientific_marker_set(monkeypatch,tmp_path):
    original=frozen_native.ALLOWED_MARKERS;native=fake_native();before=deepcopy(native)
    private=SimpleNamespace(ALLOWED_MARKERS=frozenset({"32769","768","32770"}),
        _read_native_metadata=lambda _:native,_trials=lambda _:list(range(40)))
    monkeypatch.setattr(q,"private_module",lambda *a:private)
    raw=tmp_path/"software.edf";raw.write_bytes(b"SYNTHETIC_METADATA_TEST_ONLY")
    result=q.native_structure(tmp_path,raw)
    assert frozen_native.ALLOWED_MARKERS is original and native==before
    assert result["annotations_modified_or_synthesized"] is False
    assert result["scientific_admission_granted"] is False and result["EEG_array_samples_read"] is False
    assert [r["description"] for r in result["native_annotations"]]==["32769","768","32770"]


@pytest.mark.parametrize("change",["unknown","outside_file","missing_stop"])
def test_unknown_or_non_V4_structures_are_captured_without_false_original_gate_pass(monkeypatch,tmp_path,change):
    native=fake_native(unknown=change=="unknown",outside=change=="outside_file")
    if change=="missing_stop":native["annotations"].pop()
    def trial_gate(_):
        if change=="missing_stop":raise ValueError("Original run markers and forty balanced native MI trials required")
        return list(range(40))
    private=SimpleNamespace(ALLOWED_MARKERS=frozenset({"32769","768","32770"}),_read_native_metadata=lambda _:native,_trials=trial_gate)
    monkeypatch.setattr(q,"private_module",lambda *a:private)
    raw=tmp_path/"software.edf";raw.write_bytes(b"SYNTHETIC_METADATA_TEST_ONLY")
    result=q.native_structure(tmp_path,raw)
    assert result["status"]=="NATIVE_TAL_STRUCTURE_CAPTURED_FOR_REVIEW"
    assert result["existing_V4_gate"]["status"]=="EXISTING_V4_GATE_REQUIRES_REVIEW"
    assert result["scientific_admission_granted"] is False
    assert result["native_marker_counts"].get("32770",0)==(0 if change=="missing_stop" else 1)


def test_malformed_native_metadata_is_recorded_for_review_without_fabrication(monkeypatch,tmp_path):
    def malformed(_):raise ValueError("Discontinuous original clock")
    private=SimpleNamespace(ALLOWED_MARKERS=frozenset(),_read_native_metadata=malformed)
    monkeypatch.setattr(q,"private_module",lambda *a:private)
    raw=tmp_path/"software.edf";raw.write_bytes(b"SYNTHETIC_METADATA_TEST_ONLY")
    result=q.native_structure(tmp_path,raw)
    assert result["status"]=="NATIVE_METADATA_REVIEW_REQUIRED" and result["EEG_array_samples_read"] is False
    assert result["scientific_admission_granted"] is False and "native_trial_count" not in result


def test_execute_cannot_make_its_own_unfrozen_plan(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(q,"make_plan",lambda _:pytest.fail("execute must require preexisting frozen plan"))
    monkeypatch.setattr(q,"execute",lambda *a,**k:pytest.fail("must not dispatch"))
    assert q.main(["--root",str(tmp_path),"--execute"])==1
    assert "frozen_qualification_plan_required" in json.loads(capsys.readouterr().out)["error_code"]


def test_freeze_and_execute_are_separate_invocations(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(q,"make_plan",lambda _:pytest.fail("must reject before making plan"))
    assert q.main(["--root",str(tmp_path),"--execute","--write-plan"])==1
    assert "separately" in json.loads(capsys.readouterr().out)["error_code"]


def test_default_is_readonly_without_any_execution(tmp_path,monkeypatch,capsys):
    value={"plan_sha256":"f"*64,"reused_complete_parent_states":{},"budgets":{"wall_seconds":100}}
    monkeypatch.setattr(q,"make_plan",lambda _:value);monkeypatch.setattr(q,"validate",lambda *a:None)
    monkeypatch.setattr(q,"execute",lambda *a,**k:pytest.fail("default must never execute"))
    assert q.main(["--root",str(tmp_path)])==0
    assert json.loads(capsys.readouterr().out)["status"]=="DRY_RUN_SOURCE_NATIVE_QUALIFICATION_ONLY"
    assert list(tmp_path.iterdir())==[]


def test_execute_holds_parent_Dreyer_dataset_lock(parent_fixture,monkeypatch):
    root,parent,_,_,_=parent_fixture;plan=q.make_plan(root);active=[]
    @contextmanager
    def lock(_):
        active.append(True)
        try:yield
        finally:active.pop()
    monkeypatch.setattr(q,"validate",lambda *a:parent)
    monkeypatch.setattr(q,"private_module",lambda *a:SimpleNamespace(dataset_lock=lock))
    def owned(*a,**kwargs):
        assert active==[True]
        return {"status":"SYNTHETIC_LOCK_SCOPE_ONLY"}
    monkeypatch.setattr(q,"_execute_dataset_owned",owned)
    assert q.execute(root,plan)["status"]=="SYNTHETIC_LOCK_SCOPE_ONLY" and active==[]


def reused_fixture(parent_fixture):
    root,parent,audit,_,_=parent_fixture;record=parent["records"][0]
    native_path=write(audit/record["record_id"]/"native_metadata.json",{"native_marker_counts":{"769":20,"770":20}})
    ndesc=q.core.descriptor(root,native_path)
    original={"status":"COMPLETE","record_id":record["record_id"],"artifacts":[ndesc]}
    original_path=write(audit/record["record_id"]/"state.json",original)
    plan=q.make_plan(root)
    state={"status":"STRUCTURE_QUALIFICATION_RECORDED","plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],
        "reused_parent_state":q.core.descriptor(root,original_path),"native_structure":ndesc,
        "native_marker_counts":{"769":20,"770":20},"existing_V4_gate":{"status":"EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS"},
        "scientific_admission_granted":False,"EEG_array_samples_read":False}
    return root,parent,record,plan,state


def test_reused_native_structure_is_exact_parent_cache_backed_evidence(parent_fixture):
    root,parent,record,plan,state=reused_fixture(parent_fixture)
    assert q.verified_state(root,state,plan,record,parent) is True


@pytest.mark.parametrize("change",["wrong_parent_descriptor","wrong_native_descriptor","wrong_counts","wrong_gate","claim_scientific","claim_EEG"])
def test_reused_native_proof_cannot_lose_binding_or_expand_authority(parent_fixture,change):
    root,parent,record,plan,state=reused_fixture(parent_fixture)
    if change=="wrong_parent_descriptor":state["reused_parent_state"]["sha256"]="f"*64
    elif change=="wrong_native_descriptor":state["native_structure"]["sha256"]="f"*64
    elif change=="wrong_counts":state["native_marker_counts"]["769"]=19
    elif change=="wrong_gate":state["existing_V4_gate"]={"status":"TRAINING_READY"}
    elif change=="claim_scientific":state["scientific_admission_granted"]=True
    elif change=="claim_EEG":state["EEG_array_samples_read"]=True
    with pytest.raises(q.core.PreparationError):q.verified_state(root,state,plan,record,parent)


def fresh_fixture(parent_fixture,monkeypatch):
    root,parent,_,_,_=parent_fixture;plan=q.make_plan(root);record=parent["records"][0]
    native={"source_file":{k:record["raw_files"][0][k] for k in ("path","sha256","bytes")},
        "status":"NATIVE_TAL_STRUCTURE_CAPTURED_FOR_REVIEW","EEG_array_samples_read":False,
        "scientific_admission_granted":False,"annotations_modified_or_synthesized":False,
        "native_marker_counts":{"769":20,"770":20},"existing_V4_gate":{"status":"EXISTING_V4_GATE_REQUIRES_REVIEW","error_code":"stop absent"}}
    npath=write(root/"audit/software/native_structure.json",native);ndesc=q.core.descriptor(root,npath)
    state={"status":"STRUCTURE_QUALIFICATION_RECORDED","plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],
        "raw_backup_receipt":{"software":True},"native_structure":ndesc,"metadata_backup_receipt":{"software":True},
        "scientific_admission_granted":False,"EEG_array_samples_read":False,
        "native_marker_counts":native["native_marker_counts"],"existing_V4_gate":native["existing_V4_gate"]}
    rows=[dict(ndesc,dataset="Dreyer2023",role="audit")]
    monkeypatch.setattr(q.core,"verify_raw_receipt",lambda *a:[dict(record["raw_files"][0],role="source_development_raw",dataset="Dreyer2023")])
    monkeypatch.setattr(q.core,"receipt_rows",lambda *a:rows)
    return root,parent,record,plan,state,native,npath,rows


def test_new_native_structure_receipt_is_exact_metadata_only_full_GET(parent_fixture,monkeypatch):
    root,parent,record,plan,state,_,_,_=fresh_fixture(parent_fixture,monkeypatch)
    assert q.verified_state(root,state,plan,record,parent) is True


@pytest.mark.parametrize("change",["wrong_dataset","wrong_role","wrong_bytes","duplicate_row","wrong_source","EEG","admission","synthesized","state_counts","state_gate"])
def test_new_native_structure_proofs_are_exact_and_do_not_grant_training(parent_fixture,monkeypatch,change):
    root,parent,record,plan,state,native,npath,rows=fresh_fixture(parent_fixture,monkeypatch)
    if change=="wrong_dataset":rows[0]["dataset"]="SEALED_EXTERNAL"
    elif change=="wrong_role":rows[0]["role"]="result"
    elif change=="wrong_bytes":rows[0]["bytes"]+=1
    elif change=="duplicate_row":rows.append(deepcopy(rows[0]))
    elif change=="state_counts":state["native_marker_counts"]={"769":19,"770":20}
    elif change=="state_gate":state["existing_V4_gate"]={"status":"TRAINING_READY"}
    else:
        if change=="wrong_source":native["source_file"]["sha256"]="f"*64
        elif change=="EEG":native["EEG_array_samples_read"]=True
        elif change=="admission":native["scientific_admission_granted"]=True
        elif change=="synthesized":native["annotations_modified_or_synthesized"]=True
        write(npath,native);state["native_structure"]=q.core.descriptor(root,npath)
        rows[0].update(state["native_structure"])
    with pytest.raises(q.core.PreparationError):q.verified_state(root,state,plan,record,parent)


class FakeLedger:
    def __init__(self,path,plan):
        self.path,self.plan=path,plan
        self.data={"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":plan["plan_sha256"],
            "charged":{"provider_body_bytes":0,"r2_get_body_bytes":0},"elapsed_seconds":0.,"history":[]}
    def checkpoint(self,**kwargs):pass


@pytest.mark.parametrize("change",["path","sha256","bytes","created_by"])
def test_resume_owned_raw_marker_must_match_exact_claim_before_partial_touch(parent_fixture,monkeypatch,change):
    root,parent,_,_,_=parent_fixture;plan=q.make_plan(root);plan=deepcopy(plan);plan["records"]=plan["records"][:1]
    record=plan["records"][0];rawitem=record["raw_files"][0]
    owned={k:rawitem[k] for k in ("path","sha256","bytes")};owned["created_by"]=q.ENTRY
    if change=="path":owned[change]="data/unowned.edf"
    elif change=="sha256":owned[change]="f"*64
    elif change=="bytes":owned[change]+=1
    elif change=="created_by":owned[change]="someone_else"
    path=root/q.BASE/plan["plan_sha256"]/record["record_id"]/"state.json"
    write(path,{"plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],"owned_raw":owned})
    raw=root/rawitem["path"];raw.parent.mkdir(parents=True,exist_ok=True)
    partial=raw.with_name(raw.name+".owned.partial");partial.write_bytes(b"UNOWNED_PARTIAL_MUST_BE_PRESERVED")
    monkeypatch.setattr(q.core,"Ledger",FakeLedger)
    monkeypatch.setattr(q.core,"download_component",lambda *a,**k:pytest.fail("bad ownership must stop before download/partial mutation"))
    with pytest.raises(q.core.PreparationError):q._execute_dataset_owned(root,plan,parent,max_new_records=1)
    assert partial.read_bytes()==b"UNOWNED_PARTIAL_MUST_BE_PRESERVED"


def test_resume_complete_record_reuses_proofs_without_any_rebackup(parent_fixture,monkeypatch):
    root,parent,_,_,_=parent_fixture;plan=q.make_plan(root);plan=deepcopy(plan);plan["records"]=plan["records"][:1]
    record=plan["records"][0]
    path=root/q.BASE/plan["plan_sha256"]/record["record_id"]/"state.json"
    write(path,{"plan_sha256":plan["plan_sha256"],"record_id":record["record_id"],
        "status":"STRUCTURE_QUALIFICATION_RECORDED","scientific_admission_granted":False,"EEG_array_samples_read":False})
    before=path.read_bytes()
    monkeypatch.setattr(q.core,"Ledger",FakeLedger)
    monkeypatch.setattr(q,"verified_state",lambda *a:True)
    monkeypatch.setattr(q.core,"verified_backup",lambda *a,**k:pytest.fail("completed record must not rebackup"))
    monkeypatch.setattr(q,"native_structure",lambda *a,**k:pytest.fail("completed record must not reread native payload"))
    result=q._execute_dataset_owned(root,plan,parent,max_new_records=1)
    assert result["status"]=="PAUSED_STRUCTURE_QUALIFICATION_ONLY" and path.read_bytes()==before


@pytest.fixture
def complete_fixture(tmp_path,monkeypatch):
    records=[{"record_id":f"software-complete-{i:03}"} for i in range(520)]
    plan={"plan_sha256":"f"*64,"records":records,
        "budgets":{"provider_body_bytes":10000000,"r2_get_body_bytes":10000000,"wall_seconds":1000}}
    write(tmp_path/q.BASE/"plan.json",plan);folder=tmp_path/q.BASE/plan["plan_sha256"]
    states=[]
    for record in records:
        path=write(folder/record["record_id"]/"state.json",{"record_id":record["record_id"],"plan_sha256":plan["plan_sha256"],
            "status":"STRUCTURE_QUALIFICATION_RECORDED","scientific_admission_granted":False,"EEG_array_samples_read":False,
            "reused_parent_state":{"software_fixture":True}})
        states.append(q.core.descriptor(tmp_path,path))
    summary={"schema_version":q.SCHEMA,"status":"ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY",
        "plan":q.core.descriptor(tmp_path,tmp_path/q.BASE/"plan.json"),"record_states":states,"records":520,
        "scientific_admission_granted":False,"EEG_array_samples_read":False,"fits":0,"GPU_actions":0}
    summary_path=write(folder/"all_structure_summary.json",summary);summary_item=q.core.descriptor(tmp_path,summary_path)
    rows=[dict(summary["plan"],dataset="Dreyer2023",role="audit"),dict(summary_item,dataset="Dreyer2023",role="audit"),
        *[dict(item,dataset="Dreyer2023",role="audit") for item in states]]
    receipt_path=write(folder/"final_backup_receipt.json",{"software_fixture":True})
    result={"status":summary["status"],"summary":summary_item,"backup_receipt":q.core.descriptor(tmp_path,receipt_path),"scientific_admission_granted":False}
    status_path=write(folder/"qualification_status.json",result)
    (folder/"owner.lock").write_bytes(b"")
    total=sum(r["bytes"] for r in rows)
    ledger=write(folder/"budget_ledger.json",{"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":plan["plan_sha256"],
        "charged":{"provider_body_bytes":0,"r2_get_body_bytes":total},"elapsed_seconds":42.,
        "history":[{"kind":"r2_get_body_bytes","reserved_bytes":total,"verified_bytes":total,"status":"SUCCESS_FULL_BODY_VERIFIED"}]})
    monkeypatch.setattr(q,"verified_state",lambda root,state,p,record,parent:
        state.get("record_id")==record["record_id"] and state.get("plan_sha256")==p["plan_sha256"])
    monkeypatch.setattr(q.core,"receipt_rows",lambda *a:rows)
    monkeypatch.setattr(q,"private_module",lambda *a:SimpleNamespace(closed_history=parent_worker.closed_history))
    return tmp_path,plan,folder,rows,summary,summary_path,result,status_path,ledger


def test_all520_complete_reentry_is_readonly_and_does_not_reinitialize_or_rebackup(complete_fixture,monkeypatch):
    root,plan,folder,_,_,_,expected,_,_=complete_fixture
    before={p:p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    monkeypatch.setattr(q.core,"Ledger",lambda *a:pytest.fail("closed completion cannot initialize mutable ledger"))
    monkeypatch.setattr(q.core,"verified_backup",lambda *a,**k:pytest.fail("closed completion cannot GET or overwrite receipt"))
    assert q._execute_dataset_owned(root,plan,{})==expected
    assert before=={p:p.read_bytes() for p in folder.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change",["missing_final_row","duplicate_final_row","wrong_final_role","wrong_final_bytes",
    "unclosed_ledger","refunded_ledger","plan_ledger_mismatch","wall_exhausted","partial_summary","summary_EEG","summary_fit"])
def test_completed_qualification_requires_exact_all520_final_GET_and_closed_budget(complete_fixture,change):
    root,plan,folder,rows,summary,summary_path,result,status_path,ledger_path=complete_fixture
    if change=="missing_final_row":rows.pop()
    elif change=="duplicate_final_row":rows[-1]=deepcopy(rows[-2])
    elif change=="wrong_final_role":rows[-1]["role"]="result"
    elif change=="wrong_final_bytes":rows[-1]["bytes"]+=1
    elif change in {"unclosed_ledger","refunded_ledger","plan_ledger_mismatch","wall_exhausted"}:
        ledger=q.read(ledger_path)
        if change=="unclosed_ledger":ledger["wall_intent"]={}
        elif change=="refunded_ledger":
            ledger["charged"]["r2_get_body_bytes"]=0;ledger["history"]=[]
        elif change=="plan_ledger_mismatch":ledger["plan_sha256"]="e"*64
        elif change=="wall_exhausted":ledger["elapsed_seconds"]=1001
        write(ledger_path,ledger)
    else:
        if change=="partial_summary":summary["record_states"].pop()
        elif change=="summary_EEG":summary["EEG_array_samples_read"]=True
        elif change=="summary_fit":summary["fits"]=1
        write(summary_path,summary);result["summary"]=q.core.descriptor(root,summary_path);write(status_path,result)
        rows[1].update(result["summary"])
    with pytest.raises((q.core.PreparationError,parent_worker.WrapperError)):
        q.completed_qualification(root,plan,{},folder)
