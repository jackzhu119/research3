"""No network/GPU/fit: cross-boot accounting and transactional recovery gates."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from tools import recover_cpu_preparation_20261020 as work


OLD = {"utc":"2026-10-19T12:00:00+00:00","monotonic_ns":1000000000,"boot_id":"a"*8+"-"+"a"*4+"-"+"a"*4+"-"+"a"*4+"-"+"a"*12}
NOW = {"utc":"2026-10-20T06:00:00.100000+00:00","monotonic_ns":2000000000,"boot_id":"b"*8+"-"+"b"*4+"-"+"b"*4+"-"+"b"*4+"-"+"b"*12}


def inputs():
    progress = {"status":"RUNNING_LOCAL_CPU","plan_sha256":work.STIEGER_SHA,"active_wall_elapsed_s":3000.,
        "completed_sessions":109,"wall_intent":deepcopy(OLD),"official_provider_body_bytes":7,"r2_restore_body_bytes":800}
    ledger = {"schema_version":"paper3-source-extension-budget-ledger-v1","plan_sha256":work.YANG_SHA,
        "charged":{"provider_body_bytes":12,"r2_get_body_bytes":150},"elapsed_seconds":1800.,"wall_intent":deepcopy(OLD),
        "history":[{"kind":"provider_body_bytes","reserved_bytes":20,"verified_bytes":12,"status":"SUCCESS_FULL_BODY_VERIFIED"},
                   {"kind":"r2_get_body_bytes","reserved_bytes":50,"verified_bytes":50,"status":"SUCCESS_FULL_BODY_VERIFIED"}],
        "active_transfer":{"kind":"r2_get_body_bytes","reserved_bytes":100,"identity":"raw-full-get"}}
    return progress, ledger


def test_entire_UTC_gap_rounds_up_and_original_values_untouched():
    p,l = inputs(); oldp,oldl=deepcopy(p),deepcopy(l)
    x,y,proof=work.reconcile_values(p,l,NOW)
    assert p==oldp and l==oldl
    assert proof["entire_UTC_gap_billed_seconds"]=={"Stieger2021":64801,"Yang2025":64801}
    assert x["active_wall_elapsed_s"]==67801 and y["elapsed_seconds"]==66601
    assert "wall_intent" not in x and "wall_intent" not in y
    assert x["completed_sessions"]==109 and x["r2_restore_body_bytes"]==800
    assert y["charged"]==oldl["charged"] and "active_transfer" not in y
    assert y["history"][-1]["status"]=="HARD_INTERRUPTION_FULL_RESERVATION_CONSUMED"
    assert y["history"][-1]["reserved_bytes"]==100
    assert proof["body_reservation_refunded_bytes"]==0


@pytest.mark.parametrize("change", [
    lambda v: v.update(boot_id=OLD["boot_id"]),
    lambda v: v.update(utc="2026-10-18T06:00:00+00:00"),
    lambda v: v.update(utc="2026-10-30T06:00:00+00:00"),
    lambda v: v.update(utc="2026-10-20T06:00:00"),
    lambda v: v.update(utc="not_a_clock"),
])
def test_ambiguous_or_unbounded_clock_history_refused(change):
    value=deepcopy(NOW);change(value)
    with pytest.raises(work.RecoveryError):work.conservative_gap(OLD,value)


@pytest.mark.parametrize("change", [
    lambda p,l:p.update(plan_sha256="0"*64),
    lambda p,l:l.update(plan_sha256="0"*64),
    lambda p,l:p.update(active_wall_elapsed_s=float("nan")),
    lambda p,l:l["charged"].update(r2_get_body_bytes=149),
    lambda p,l:l["history"][0].update(verified_bytes=21),
    lambda p,l:l["history"][1].update(status="unclosed"),
    lambda p,l:l["active_transfer"].update(reserved_bytes=-1),
])
def test_identity_wall_or_transfer_evidence_cannot_be_invented(change):
    p,l=inputs();change(p,l)
    with pytest.raises(work.RecoveryError):work.reconcile_values(p,l,NOW)


def fixture_root(tmp_path,monkeypatch):
    root=tmp_path
    p,l=inputs()
    for relative,value in [(work.PROGRESS,p),(work.YANG_LEDGER,l),("status/tasks.json",{"W3":{"status":"IN_PROGRESS_LOCAL_CPU"}})]:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    entry=root/work.ENTRY;entry.parent.mkdir(parents=True);entry.write_bytes(Path(work.__file__).read_bytes())
    plans=({"plan_sha256":work.STIEGER_SHA,"files":[],"output":"cache/stieger_v2","max_wall_seconds":86400},
           {"plan_sha256":work.YANG_SHA,"budgets":{"wall_seconds":172800}})
    for relative,value in [(work.STIEGER_PLAN,plans[0]),(work.YANG_PLAN,plans[1])]:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    monkeypatch.setattr(work,"load_plans",lambda r:plans)
    monkeypatch.setattr(work,"snapshot",lambda r:deepcopy(NOW))
    return root,plans


def test_preview_has_no_mutation_and_execute_needs_operator_attestation(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    before=(root/work.PROGRESS).read_bytes(),(root/work.YANG_LEDGER).read_bytes()
    result=work.reconcile(root)
    assert result["status"]=="PLANNED_EXPLICIT_RUNTIME_RECONCILIATION_NO_MUTATION"
    assert before==((root/work.PROGRESS).read_bytes(),(root/work.YANG_LEDGER).read_bytes())
    assert not (root/work.BASE).exists()
    with pytest.raises(work.RecoveryError,match="attestation"):
        work.reconcile(root,execute=True)
    assert before==((root/work.PROGRESS).read_bytes(),(root/work.YANG_LEDGER).read_bytes())


def test_transaction_preserves_originals_and_retry_never_recharges_or_refills(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    before={r:(root/r).read_bytes() for r in (work.PROGRESS,work.YANG_LEDGER,"status/tasks.json")}
    result=work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    amendment=work.read(root/result["amendment"]["path"])
    for row in amendment["original_records"]:
        assert (root/row["snapshot"]["path"]).read_bytes()==before[row["original_path"]]
    assert set(amendment["legacy_guards_at_reconciliation"])=={"Stieger2021","Yang2025"}
    assert all(v["automatic_resume_allowed"] is False for v in amendment["legacy_guards_at_reconciliation"].values())
    p,l=work.read(root/work.PROGRESS),work.read(root/work.YANG_LEDGER)
    assert p["active_wall_elapsed_s"]==67801 and l["elapsed_seconds"]==66601
    assert p["explicit_runtime_reconciliation"]==result["amendment"]
    prior=(root/work.PROGRESS).read_bytes(),(root/work.YANG_LEDGER).read_bytes()
    repeat=work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    assert repeat["status"]=="RECONCILIATION_ALREADY_COMMITTED_NO_RUNTIME_REWRITE"
    assert prior==((root/work.PROGRESS).read_bytes(),(root/work.YANG_LEDGER).read_bytes())


def test_committed_reconciliation_cannot_restore_smaller_published_wall(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    p=work.read(root/work.PROGRESS);p["active_wall_elapsed_s"]=0
    (root/work.PROGRESS).write_text(json.dumps(p))
    with pytest.raises(work.RecoveryError,match="floor_rolled_back"):
        work.reconcile(root,execute=True,confirmed_runtime_replacement=True)


def test_existing_worker_lock_refuses_all_live_mutation(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    before=(root/work.PROGRESS).read_bytes()
    import fcntl
    lock=root/'cache/stieger_v2/.prepare.lock'
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(work.RecoveryError,match="owns_lock"):
            work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    assert (root/work.PROGRESS).read_bytes()==before


def test_crash_between_target_commits_resumes_same_prepared_transaction(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    original=work.source.atomic_json
    armed={"failed":False}
    def partial(path,value,**kwargs):
        if Path(path)==root/work.YANG_LEDGER and not armed["failed"]:
            armed["failed"]=True
            raise OSError("SYNTHETIC_TRANSACT_WRITE_FAILURE")
        return original(path,value,**kwargs)
    monkeypatch.setattr(work.source,"atomic_json",partial)
    with pytest.raises(OSError):work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    marker=work.read(work.recovery_marker(root,NOW["boot_id"]))
    assert marker["status"]=="RECONCILIATION_PREPARED"
    assert work.read(root/work.PROGRESS)["active_wall_elapsed_s"]==67801
    assert work.read(root/work.YANG_LEDGER)["elapsed_seconds"]==1800
    result=work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    assert result["status"]=="RECONCILIATION_COMMITTED"
    assert work.read(root/work.PROGRESS)["active_wall_elapsed_s"]==67801
    assert work.read(root/work.YANG_LEDGER)["elapsed_seconds"]==66601


@pytest.mark.parametrize("cap",[0,86400,259201,True])
def test_operational_wall_amendment_bounded_to_explicit_CPU_scope(tmp_path,monkeypatch,cap):
    root,_=fixture_root(tmp_path,monkeypatch)
    with pytest.raises(work.RecoveryError,match="48_to_72"):
        work.reconcile(root,cpu_wall_cap_seconds=cap)


def test_changed_tool_after_amendment_cannot_resume_authority(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    (root/work.ENTRY).write_text("CHANGED_SOFTWARE_FIXTURE")
    with pytest.raises(work.source.PreparationError,match="artifact_sha"):
        work.load_stieger_authorization(root)


def resume_fixture(tmp_path,monkeypatch):
    root,_=fixture_root(tmp_path,monkeypatch)
    result=work.reconcile(root,execute=True,confirmed_runtime_replacement=True)
    entries=[{"id":i,"name":f"S1_Session_{i}.mat","partition":"meta_train","provider_subject":"S1","session_ordinal":i} for i in (1,2,3)]
    plan={"plan_sha256":work.STIEGER_SHA,"files":entries,"output":"cache/stieger_v2","max_wall_seconds":86400,
          "max_download_bytes":1000,"max_r2_restore_bytes":2000,"max_cache_bytes":10000}
    monkeypatch.setattr(work,"load_stieger_authorization",lambda r:(plan,result["amendment"],172800))
    monkeypatch.setattr(work.stieger,"validate_plan",lambda *a,**k:({},{}))
    monkeypatch.setattr(work.stieger,"clock_snapshot",lambda:deepcopy(NOW))
    monkeypatch.setattr(work.stieger,"raw_receipt_catalog",lambda *a:{})
    owner={"schema_version":work.stieger.SCHEMA,"created_by":"tools/prepare_stieger_cache.py","plan_sha256":work.STIEGER_SHA}
    work.source.atomic_json(root/'cache/stieger_v2/ownership.json',owner)
    calls=[];updates=[]
    monkeypatch.setattr(work.stieger,"update_w3_task",lambda root,plan,status,**kwargs:updates.append(status))
    monkeypatch.setattr(work.stieger,"recover_open_transfer",lambda *a,**k:None)
    monkeypatch.setattr(work.stieger,"verify_session_state",lambda state,*a:state.get('status')=='COMPLETE')
    def process(entry,*,root,output,plan,remaining_source_bytes,remaining_restore_bytes,**kwargs):
        calls.append((entry['id'],remaining_source_bytes,remaining_restore_bytes))
        p=output/entry['partition']/entry['provider_subject']/f"session-{entry['session_ordinal']}"/'state.json'
        old=work.read(p) if p.exists() else {}
        if old.get('status')=='COMPLETE':return old
        state={"status":"COMPLETE","plan_sha256":work.STIEGER_SHA,"cache_bytes":50,
               "official_provider_body_bytes":old.get('official_provider_body_bytes',0),"r2_restore_body_bytes":old.get('r2_restore_body_bytes',0)+80}
        work.source.atomic_json(p,state)
        return state
    monkeypatch.setattr(work.stieger,"process_session",process)
    return root,plan,calls,updates


def test_resume_original_kernel_uses_original_body_caps_and_no_false_W3_on_pause(tmp_path,monkeypatch):
    root,plan,calls,updates=resume_fixture(tmp_path,monkeypatch)
    prior=root/'cache/stieger_v2/meta_train/S1/session-1/state.json'
    work.source.atomic_json(prior,{"status":"COMPLETE","plan_sha256":work.STIEGER_SHA,"cache_bytes":50,
          "official_provider_body_bytes":20,"r2_restore_body_bytes":400})
    result=work.resume_stieger(root,execute=True,max_new_files=1)
    assert result['status']=='PARTIAL_BATCH_NO_W3_COMPLETE'
    assert calls==[(1,980,1600),(2,980,1600)]
    assert result['official_provider_body_bytes']==20 and result['r2_restore_body_bytes']==480
    assert result['active_wall_elapsed_s']==67801 and 'wall_intent' not in result
    assert updates==['PARTIAL_CACHE_NOT_COMPLETE'] and result['completed_sessions']==109
    assert plan['max_wall_seconds']==86400  # operational cap never mutates science plan


def test_cpu_resume_kernel_failure_has_closed_clock_and_no_complete(tmp_path,monkeypatch):
    root,plan,calls,updates=resume_fixture(tmp_path,monkeypatch)
    def fail(*a,**k):raise work.stieger.CacheError('SYNTHETIC_FAILED_BEFORE_BODY')
    monkeypatch.setattr(work.stieger,'process_session',fail)
    with pytest.raises(work.stieger.CacheError):work.resume_stieger(root,execute=True)
    value=work.read(root/work.PROGRESS)
    assert value['status']=='FAILED_CLOSED_NO_W3_COMPLETE' and 'wall_intent' not in value
    assert updates==['FAILED_CLOSED_NOT_COMPLETE'] and 'COMPLETE' not in updates


def test_final_clock_overrun_never_retains_W3_complete(tmp_path,monkeypatch):
    root,plan,calls,updates=resume_fixture(tmp_path,monkeypatch)
    original=work.stieger.wall_checkpoint
    def cutoff(path,progress,*,reopen):
        original(path,progress,reopen=reopen)
        if not reopen:
            progress['active_wall_elapsed_s']=172801
            work.source.atomic_json(path,progress)
    monkeypatch.setattr(work.stieger,'wall_checkpoint',cutoff)
    with pytest.raises(work.RecoveryError,match='final_closed'):
        work.resume_stieger(root,execute=True,max_new_files=1)
    assert work.read(root/work.PROGRESS)['status']=='RESOURCE_BUDGET_STOP_NO_W3_COMPLETE'
    assert updates[-1]=='RESOURCE_BUDGET_STOP_NOT_COMPLETE' and 'COMPLETE' not in updates


def test_default_CPU_resume_is_read_only_and_lock_prevents_duplicate(tmp_path,monkeypatch):
    root,_,calls,updates=resume_fixture(tmp_path,monkeypatch)
    before=(root/work.PROGRESS).read_bytes()
    result=work.resume_stieger(root)
    assert result['status']=='RECOVERED_STIEGER_RESUME_PLAN_NO_NETWORK'
    assert calls==[] and updates==[] and before==(root/work.PROGRESS).read_bytes()
    import fcntl
    with (root/'cache/stieger_v2/.prepare.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(work.RecoveryError,match='owns_lock'):
            work.resume_stieger(root,execute=True)
    assert calls==[] and updates==[]
