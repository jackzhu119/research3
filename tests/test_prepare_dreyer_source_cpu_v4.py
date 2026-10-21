"""Operational software fixtures only: no real EEG, fit, GPU, network or R2."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from p3work import source_extension_coordinator as real
from tools import prepare_dreyer_source_cpu_v4 as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")
    return path


def ledger(charged=None, wall=0., history=None):
    return {"schema_version":"paper3-source-extension-budget-ledger-v1",
        "plan_sha256":worker.DREYER_PLAN_SHA,
        "charged":charged or {"provider_body_bytes":0,"r2_get_body_bytes":0},
        "elapsed_seconds":wall,"history":history or []}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    if worker.DREYER_PLAN_SHA is None:
        monkeypatch.setattr(worker,"DREYER_PLAN_SHA","d"*64)
    target = tmp_path/worker.ENTRY
    target.parent.mkdir(parents=True)
    shutil.copyfile(worker.__file__,target)
    parent=tmp_path/worker.PARENT_WRAPPER_PATH
    shutil.copyfile(worker.ROOT/worker.PARENT_WRAPPER_PATH,parent)
    core = SimpleNamespace(**{k:getattr(real,k) for k in
        ("local","descriptor","atomic_json","digest","checked","PreparationError","Ledger","storage")})
    monkeypatch.setattr(worker,"kernel",lambda:core)
    return tmp_path,core


def test_default_checks_remain_readonly_and_explicit_execute_requires_declaration(monkeypatch,capsys,tmp_path):
    monkeypatch.setattr(worker,"DREYER_PLAN_SHA","d"*64)
    monkeypatch.setattr(worker,"fixed_plans",lambda *a:({},{},{}))
    monkeypatch.setattr(worker,"build_declaration",lambda *a:{"software_fixture":True})
    assert worker.main(["--root",str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DRY_RUN_NO_NETWORK_R2_OR_FIT"
    assert list(tmp_path.iterdir()) == []
    assert worker.main(["--root",str(tmp_path),"--execute"]) == 1
    assert "declaration" in json.loads(capsys.readouterr().out)["error_code"]


def test_declaration_freeze_and_execute_are_separate(monkeypatch,capsys,tmp_path):
    assert worker.main(["--root",str(tmp_path),"--execute","--write-declaration"]) == 1
    assert "separately" in json.loads(capsys.readouterr().out)["error_code"]


@pytest.mark.parametrize("change",["wrong_Dreyer","wrong_Yang","partial_roster","wrong_native_adapter"])
def test_exact_two_plans_and_entire520_roster_required(isolated,change):
    root,core = isolated
    dreyer={"dataset":"Dreyer2023","plan_sha256":worker.DREYER_PLAN_SHA,
        "record_count":520,"records":[{} for _ in range(520)],
        "code":{"p3work/source_extensions/dreyer2023.py":{"sha256":worker.NATIVE_ADAPTER_SHA}}}
    yang={"dataset":"Yang2025","plan_sha256":worker.YANG_PLAN_SHA}
    if change=="wrong_Dreyer":dreyer["plan_sha256"]="f"*64
    if change=="wrong_Yang":yang["plan_sha256"]="f"*64
    if change=="partial_roster":dreyer["records"].pop()
    if change=="wrong_native_adapter":dreyer["code"]["p3work/source_extensions/dreyer2023.py"]["sha256"]="c"*64
    write(root/worker.PLAN_PATH,dreyer);write(root/worker.YANG_PATH,yang)
    core.validate_plan=lambda *a:pytest.fail("must reject before any adapter")
    with pytest.raises(worker.WrapperError,match="exact"):
        worker.fixed_plans(root)


@pytest.mark.parametrize("old_sha",[worker.OLD_DREYER_SHA,worker.V2_DREYER_SHA,worker.V3_DREYER_SHA])
def test_version4_never_executes_original_or_stopped_V2_or_V3_plan(isolated,old_sha):
    root,core=isolated
    write(root/worker.PLAN_PATH,{"dataset":"Dreyer2023","plan_sha256":old_sha,
        "record_count":520,"records":[{}for _ in range(520)],
        "code":{"p3work/source_extensions/dreyer2023.py":{"sha256":worker.NATIVE_ADAPTER_SHA}}})
    write(root/worker.YANG_PATH,{"dataset":"Yang2025","plan_sha256":worker.YANG_PLAN_SHA})
    core.validate_plan=lambda *a:pytest.fail("old plan must close before adapter/IO")
    with pytest.raises(worker.WrapperError,match="exact"):
        worker.fixed_plans(root)


@pytest.fixture
def four_history(isolated,monkeypatch):
    root,core=isolated
    sources=[]
    for label,provider,r2,wall,caps in [
        ("V1",10,30,12.4,(1000,2000,100)),
        ("V2",20,40,19.6,(990,1970,87)),
        ("auxiliary_schema_review",50,0,6.2,(100,1,1800)),
        ("V3",30,50,1.2,(920,1930,61))]:
        old={"source":label,"budgets":dict(zip(
            ("provider_body_bytes","r2_get_body_bytes","wall_seconds"),caps))}
        sha=core.digest(old);old["plan_sha256"]=sha
        base=root/"history"/label
        plan_file=write(base/"plan.json",old)
        history=[{"kind":"provider_body_bytes","reserved_bytes":provider+3,"verified_bytes":provider,
            "status":"SUCCESS_FULL_BODY_VERIFIED"}]
        if r2:history.append({"kind":"r2_get_body_bytes","reserved_bytes":r2,
            "status":"FAILURE_FULL_RESERVATION_CONSUMED"})
        previous=ledger({"provider_body_bytes":provider,"r2_get_body_bytes":r2},wall,history)
        previous["plan_sha256"]=sha
        ledger_file=write(base/"ledger.json",previous)
        sources.append((label,plan_file.relative_to(root).as_posix(),ledger_file.relative_to(root).as_posix(),sha))
    monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(sources))
    return root,core,sources


def test_all_four_prior_actual_bytes_and_one_rounded_wall_are_deducted_not_reset(four_history):
    root,_,_=four_history
    plan={"budgets":{"provider_body_bytes":890,"r2_get_body_bytes":1880,"wall_seconds":59}}
    proof=worker.prior_budget_evidence(root,plan)
    assert proof["deducted_wall_seconds"]==40 and proof["deducted_body_bytes"]=={
        "provider_body_bytes":110,"r2_get_body_bytes":120}
    assert [x["source"]for x in proof["closed_history_sources"]]==["V1","V2","auxiliary_schema_review","V3"]
    assert proof["uses_original_caps_not_V2_or_V3_remaining_caps"]is True
    assert proof["carried_over_verified_R2_body_bytes"]==0
    assert proof["original_caps_minus_all_actual_usage"]["wall_seconds"]==60
    assert proof["immediate_parent_conservation_bound"]["remaining_caps"]["wall_seconds"]==59
    assert proof["remaining_caps"]["wall_seconds"]==59
    assert proof["retained_prior_wall_rounding_seconds"]==1
    for invalid_wall in [60,100]:
        plan["budgets"]["wall_seconds"]=invalid_wall
        with pytest.raises(worker.WrapperError,match="deducted"):
            worker.prior_budget_evidence(root,plan)


@pytest.mark.parametrize("change",["missing_auxiliary","missing_V3","duplicate","open_auxiliary","open_V3","tampered_plan","over_budget"])
def test_history_source_missing_duplicates_tampering_or_unclosed_usage_rejected(four_history,monkeypatch,change):
    root,_,sources=four_history
    if change=="missing_auxiliary":monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(sources[:2]+sources[3:]))
    elif change=="missing_V3":monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(sources[:3]))
    elif change=="duplicate":
        altered=list(sources);altered[2]=(sources[2][0],sources[2][1],sources[1][2],sources[2][3])
        monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(altered))
    elif change=="tampered_plan":
        path=root/sources[2][1];value=worker.read_json(path);value["unexpected"]="changed";write(path,value)
    else:
        path=root/sources[3 if change=="open_V3" else 2][2];value=worker.read_json(path)
        if change in {"open_auxiliary","open_V3"}:value["active_transfer"]={}
        else:
            value["charged"]["provider_body_bytes"]=200
            value["history"][0].update(reserved_bytes=200,verified_bytes=200)
        write(path,value)
    with pytest.raises(worker.WrapperError):worker.remaining_budget_evidence(root)


def test_version4_without_root_frozen_plan_is_unexecutable(isolated,monkeypatch):
    root,_=isolated
    monkeypatch.setattr(worker,"DREYER_PLAN_SHA",None)
    with pytest.raises(worker.WrapperError,match="not_frozen"):
        worker.fixed_plans(root)


def test_frozen_78a349_parent_may_not_be_patched_or_replaced(isolated):
    root,_=isolated
    path=root/worker.PARENT_WRAPPER_PATH
    path.write_bytes(path.read_bytes()+b"\n# software fixture tamper\n")
    with pytest.raises(worker.WrapperError,match="parent_wrapper"):
        worker.implementation(root)


@pytest.mark.parametrize("bad",["active_transfer","wall_intent","charges","unclosed_history","nan"])
def test_prior_failed_interrupted_or_inconsistent_budgets_cannot_disappear(bad):
    previous=ledger()
    if bad in {"active_transfer","wall_intent"}:previous[bad]={}
    elif bad=="charges":previous["charged"]["provider_body_bytes"]=1
    elif bad=="nan":previous["elapsed_seconds"]=float("nan")
    else:previous["history"]=[{"kind":"provider_body_bytes","reserved_bytes":1,"status":"RUNNING"}]
    with pytest.raises(worker.WrapperError):worker.closed_history(previous)


def test_dataset_lock_rejects_second_worker_before_io(isolated):
    root,_=isolated
    with worker.dataset_lock(root):
        with pytest.raises(worker.WrapperError,match="already_running"):
            with worker.dataset_lock(root):pytest.fail("duplicate worker entered")


def test_actual_flock_owner_parser_uses_inode_not_pid_constant(tmp_path):
    path=tmp_path/"lease.lock";path.touch()
    with path.open("a+")as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert worker.os.getpid() in worker.lock_owners(path)


def fingerprint(root,pid=4201,start=100):
    return {"pid":pid,"starttime_ticks":start,"boot_id":"fixture-boot-id",
        "cwd":str(root),"command":["python","-u","-B","tools/prepare_source_extensions.py",
            "--plan",worker.YANG_PATH,"--execute"]}


def test_peer_identity_binds_actual_command_plan_boot_and_starttime(isolated):
    root,_=isolated
    write(root/worker.YANG_PATH,{"plan_sha256":worker.YANG_PLAN_SHA})
    one=worker.yang_identity(root,fingerprint(root))
    two=worker.yang_identity(root,fingerprint(root,start=101))
    assert one["pid"]==two["pid"] and one!=two
    bad=fingerprint(root);bad["command"][-2]="other_plan.json"
    assert worker.yang_identity(root,bad) is None
    bad=fingerprint(root);bad["command"][3]="tools/prepare_dreyer_source_cpu.py"
    assert worker.yang_identity(root,bad) is None


@pytest.fixture
def peer(isolated,monkeypatch):
    root,core=isolated
    write(root/worker.YANG_PATH,{"plan_sha256":worker.YANG_PLAN_SHA})
    lock=root/worker.AUDIT_ROOT/".source-extension-singleton.lock";lock.parent.mkdir(parents=True,exist_ok=True);lock.touch()
    state={"owners":[4201],"fingerprint":fingerprint(root)}
    monkeypatch.setattr(worker,"lock_owners",lambda path:state["owners"])
    monkeypatch.setattr(worker,"process_fingerprint",lambda pid:state["fingerprint"])
    core.receipt_rows=lambda *a:[{"software_fixture_only":True}]
    guard=worker.PeerGuard(root)
    yield root,core,state,guard
    guard.close()


def test_peer_pid_reuse_or_replacement_stops_new_records(peer):
    _,_,state,guard=peer
    guard.check()
    state["fingerprint"]["starttime_ticks"]+=1
    with pytest.raises(worker.WrapperError,match="replaced"):
        guard.check()


def test_peer_failure_does_not_unlock_or_takeover_global(peer):
    _,_,state,guard=peer
    guard.check();state["owners"]=[]
    with pytest.raises(worker.WrapperError,match="not_complete"):
        guard.check()
    assert guard.owned is False


def test_unknown_global_owner_is_never_accepted(peer):
    _,_,state,guard=peer
    state["fingerprint"]=None
    with pytest.raises(worker.WrapperError,match="unknown_owner"):
        guard.check()


def test_real_complete_peer_allows_global_takeover(peer):
    root,core,state,guard=peer
    guard.check();state["owners"]=[]
    manifest=root/"complete_yang.json"
    write(manifest,{"plan_sha256":worker.YANG_PLAN_SHA,"record_count":153,"shards":[{} for _ in range(153)]})
    status=root/worker.AUDIT_ROOT/"Yang2025"/worker.YANG_PLAN_SHA/"coordinator_status.json"
    write(status,{"status":"SOURCE_EXTENSION_CACHE_COMPLETE","dataset":"Yang2025","record_count":153,
        "manifest":core.descriptor(root,manifest),"backup_receipt":{"software_fixture":True}})
    guard.check()
    assert guard.owned is True
    # The wrapper's retained fd now rejects the ordinary coordinator's EX lock.
    with guard.path.open("a+")as other:
        with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize("change",["disk","memory","cache"])
def test_shared_resource_caps_pause_before_next_io(isolated,monkeypatch,change):
    root,_=isolated;output=root/"cache"
    snapshot={"free_disk_bytes":20*1024**3,"anon_plus_kernel_bytes":1024**3}
    if change=="disk":snapshot["free_disk_bytes"]=worker.LIMITS["minimum_free_disk_bytes"]
    elif change=="memory":snapshot["anon_plus_kernel_bytes"]=worker.LIMITS["maximum_anon_plus_kernel_bytes"]
    else:
        output.mkdir();(output/"large").write_bytes(b"XX")
        monkeypatch.setitem(worker.LIMITS,"cache_bytes",1)
    monkeypatch.setattr(worker,"resource_snapshot",lambda _:snapshot)
    with pytest.raises(worker.WrapperError):worker.resource_guard(root,output)


def test_cpu_limits_use_one_allowed_cpu_and_only_positive_nice(monkeypatch):
    calls=[]
    monkeypatch.setattr(worker.os,"sched_getaffinity",lambda _:{2,4})
    monkeypatch.setattr(worker.os,"sched_setaffinity",lambda pid,value:calls.append(("affinity",value)))
    monkeypatch.setattr(worker.os,"getpriority",lambda *a:12)
    monkeypatch.setattr(worker.os,"setpriority",lambda *a:calls.append(("nice",a[-1])))
    worker.configure_cpu()
    assert calls==[("affinity",{4}),("nice",12)]
    assert worker.os.environ["OPENBLAS_NUM_THREADS"]=="1"


@pytest.fixture
def execution(isolated,monkeypatch):
    root,core=isolated
    plan={"plan_sha256":worker.DREYER_PLAN_SHA,"dataset":"Dreyer2023","record_count":520,
        "records":[{"record_id":f"record-{i}","raw_files":[{"bytes":1}]}for i in range(520)],
        "budgets":{"provider_body_bytes":100000,"r2_get_body_bytes":100000,"wall_seconds":10000}}
    declpath=worker.audit_path(root)/"parallel_root_declaration.json";write(declpath,{"software_fixture":True})
    desc=core.descriptor(root,declpath)
    monkeypatch.setattr(worker,"fixed_plans",lambda *a:(plan,{},{}))
    monkeypatch.setattr(worker,"verify_declaration",lambda *a:({},desc))
    monkeypatch.setattr(worker,"configure_cpu",lambda:None)
    monkeypatch.setattr(worker,"resource_guard",lambda *a:None)
    monkeypatch.setattr(worker,"verified_completion",lambda *a:None)
    peer=SimpleNamespace(peer={"pid":42},owned=False,check=lambda:None,close=lambda:None)
    monkeypatch.setattr(worker,"PeerGuard",lambda _:peer)
    core.ADAPTERS={"Dreyer2023":"fixture_dreyer_adapter"}
    original_import=worker.importlib.import_module
    monkeypatch.setattr(worker.importlib,"import_module",lambda name:SimpleNamespace(materialize=None,audit_native=None)
        if name=="fixture_dreyer_adapter"else original_import(name))
    import requests
    monkeypatch.setattr(requests,"Session",lambda:SimpleNamespace(
        __enter__=lambda self:self,__exit__=lambda *a:None))
    class Session:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(requests,"Session",Session)
    calls=[]
    def process(root,plan,record,**kwargs):
        calls.append(record["record_id"])
        kwargs["ledger"].reserve("provider_body_bytes",1,record["record_id"])
        kwargs["ledger"].finish(verified_bytes=1)
        value={"status":"COMPLETE","record_id":record["record_id"]}
        write(worker.audit_path(root)/record["record_id"]/"state.json",value)
        return value
    core.process_record=process
    return root,core,plan,desc,calls


def test_one_new_record_then_resume_same_ledger_never_repeats_completed_record(execution):
    root,core,plan,desc,calls=execution
    # The frozen kernel itself revalidates COMPLETE and returns without IO.
    real_process=core.process_record
    def reuse(root,plan,record,**kwargs):
        path=worker.audit_path(root)/record["record_id"]/"state.json"
        return read(path)if path.exists()else real_process(root,plan,record,**kwargs)
    def read(path):return json.loads(path.read_bytes())
    core.process_record=reuse
    relative=desc["path"]
    assert worker.execute(root,declaration_path=relative,max_new_records=1)["verified_records"]==1
    assert worker.execute(root,declaration_path=relative,max_new_records=1)["verified_records"]==2
    journal=read(worker.audit_path(root)/"budget_ledger.json")
    assert journal["charged"]["provider_body_bytes"]==2 and calls==["record-0","record-1"]
    assert "wall_intent"not in journal and "active_transfer"not in journal


def test_dispatched_missing_ledger_refuses_before_second_record(execution):
    root,_,_,desc,calls=execution
    worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    (worker.audit_path(root)/"budget_ledger.json").unlink()
    with pytest.raises(worker.WrapperError,match="ledger_missing"):
        worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    assert calls==["record-0"]


def test_peer_failure_preserves_records_budget_and_resumes_same_namespace(execution,monkeypatch):
    root,_,_,desc,calls=execution
    counter={"n":0,"fail":True}
    def check():
        counter["n"]+=1
        if counter["fail"]and counter["n"]>6:
            raise worker.WrapperError("Yang_failed_stop_new_records")
    monkeypatch.setattr(worker,"PeerGuard",lambda _:SimpleNamespace(peer={},check=check,close=lambda:None))
    with pytest.raises(worker.WrapperError,match="Yang_failed"):
        worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    journal=worker.read_json(worker.audit_path(root)/"budget_ledger.json")
    assert "wall_intent"not in journal
    assert (worker.audit_path(root)/"parallel_worker_claim.json").is_file()
    counter["fail"]=False
    worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    assert worker.read_json(worker.audit_path(root)/"parallel_worker_claim.json")["invocations"]==2


def test_missing_claim_with_existing_record_or_ledger_never_creates_new_work(execution):
    root,_,_,desc,calls=execution
    worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    (worker.audit_path(root)/"parallel_worker_claim.json").unlink()
    with pytest.raises(worker.WrapperError,match="missing_claim"):
        worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    assert calls==["record-0"]


def test_verified_complete_reuse_does_not_open_a_ledger_peer_or_network(execution,monkeypatch):
    root,core,_,desc,calls=execution
    worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    monkeypatch.setattr(worker,"verified_completion",lambda *a:{"status":"SOURCE_EXTENSION_CACHE_COMPLETE",
        "new_provider_GETs_this_invocation":0,"new_R2_operations_this_invocation":0})
    monkeypatch.setattr(worker,"PeerGuard",lambda _:pytest.fail("complete reuse touched peer"))
    core.Ledger=lambda *a:pytest.fail("complete reuse opened mutable ledger")
    result=worker.execute(root,declaration_path=desc["path"])
    assert result["new_R2_operations_this_invocation"]==0 and calls==["record-0"]


def test_final_backup_includes_seven_exact_objects_and_required_original_IO(isolated,monkeypatch):
    root,core=isolated;audit=worker.audit_path(root);audit.mkdir(parents=True)
    decl=write(audit/"parallel_root_declaration.json",{"software_fixture":True})
    write(audit/"frozen_plan.json",{"software_fixture":True})
    plan={"records":[{"record_id":str(i)}for i in range(520)]}
    shards=[{"record_id":str(i),"subject_id":"fixture","session_id":str(i),
        "raw_files":[{"bytes":1}],"raw_sha256":"0"*64,"native_metadata":{},
        "raw_backup_receipt":{},"unit_evidence":{}}for i in range(520)]
    core.assemble_manifest=lambda *a:{"shards":shards,"valid_trials":520}
    captured={}
    def backup(*args,**kwargs):
        captured.update(kwargs)
        assert kwargs["backup_fn"]is core.storage.run_backup
        assert len(kwargs["files"])==7
        return {"software_fixture_proof":True}
    core.verified_backup=backup
    output=root/f"cache/source_extensions_20261019/Dreyer2023/{worker.DREYER_PLAN_SHA}"
    output.mkdir(parents=True)
    close=[]
    result=worker._finalize(root,plan,[],SimpleNamespace(checkpoint=lambda **k:close.append(k)),core.descriptor(root,decl))
    assert close==[{"close":True}]
    assert result["status"]=="SOURCE_EXTENSION_CACHE_COMPLETE"
    names={p.name for p,_ in captured["files"]}
    assert {"parallel_execution_receipt.json","parallel_root_declaration.json"}<=names
    proof=worker.read_json(audit/"parallel_execution_receipt.json")
    assert proof["actual_fits"]==proof["carried_over_verified_R2_body_bytes"]==0
    assert proof["record_count"]==520 and proof["scientific_kernel_unchanged"]is True


@pytest.fixture
def completed(isolated):
    root,core=isolated;audit=worker.audit_path(root);audit.mkdir(parents=True)
    decl=write(audit/"parallel_root_declaration.json",{"software_fixture":True})
    dummy=write(audit/"dummy_source_fixture.json",{"software_fixture_only":True})
    artifact=core.descriptor(root,dummy)
    keys=("X","y","events","exclusions","metadata","native_metadata","unit_evidence")
    records=[{"record_id":str(i)}for i in range(520)]
    plan={"records":records,"budgets":{"wall_seconds":1000,"r2_get_body_bytes":100000000,"provider_body_bytes":100000000}}
    write(audit/"frozen_plan.json",plan)
    shards=[{"record_id":str(i),"subject_id":"fixture","session_id":str(i),
        "raw_files":[{"bytes":1}],"raw_sha256":"0"*64,"raw_backup_receipt":{},
        **{k:artifact for k in keys}}for i in range(520)]
    core.assemble_manifest=lambda *a:{"shards":shards,"valid_trials":520}
    seen=[]
    def verify_state(root,state,plan,record):
        seen.append(record["record_id"])
        return state.get("status")=="COMPLETE"
    core._state_verified=verify_state
    for record in records:write(audit/record["record_id"]/"state.json",{"status":"COMPLETE"})
    def backup(*args,**kwargs):
        rows=[{**core.descriptor(root,p),"role":role,"dataset":"Dreyer2023"}for p,role in kwargs["files"]]
        receipt=write(audit/"final_backup_receipt.json",{"objects":rows})
        return core.descriptor(root,receipt)
    core.verified_backup=backup
    core.receipt_rows=lambda root,item:worker.read_json(core.checked(root,item))["objects"]
    output=root/f"cache/source_extensions_20261019/Dreyer2023/{worker.DREYER_PLAN_SHA}";output.mkdir(parents=True)
    desc=core.descriptor(root,decl)
    status=worker._finalize(root,plan,[],SimpleNamespace(checkpoint=lambda **k:None),desc)
    manifest=worker.read_json(root/status["manifest"]["path"])
    amount=sum(r["bytes"]for r in core.receipt_rows(root,status["backup_receipt"]))
    amount+=sum(sum(f["bytes"]for f in s["raw_files"])+sum(s[k]["bytes"]for k in keys)for s in shards)
    write(audit/"budget_ledger.json",ledger({"provider_body_bytes":0,"r2_get_body_bytes":amount},2.,
        [{"kind":"r2_get_body_bytes","reserved_bytes":amount,"verified_bytes":amount,"status":"SUCCESS_FULL_BODY_VERIFIED"}]))
    return root,core,plan,desc,status,seen


def test_complete_reuse_rechecks_all520_records_final7_closed_budget_without_io(completed):
    root,_,plan,desc,_,seen=completed
    value=worker.verified_completion(root,plan,desc)
    assert value["completed_local_proofs_reverified"]is True
    assert len(seen)==520 and value["new_provider_GETs_this_invocation"]==value["new_R2_operations_this_invocation"]==0


@pytest.mark.parametrize("change",["missing_final_proof","open_ledger","low_GET_charge","bad_record"])
def test_flags_alone_do_not_make_complete_reuse_valid(completed,change):
    root,core,plan,desc,status,_=completed;audit=worker.audit_path(root)
    if change=="missing_final_proof":
        receipt=root/status["backup_receipt"]["path"];value=worker.read_json(receipt);value["objects"].pop()
        write(receipt,value)
        status["backup_receipt"]=core.descriptor(root,receipt);write(audit/"coordinator_status.json",status)
    elif change=="open_ledger":
        value=worker.read_json(audit/"budget_ledger.json");value["wall_intent"]={};write(audit/"budget_ledger.json",value)
    elif change=="low_GET_charge":
        write(audit/"budget_ledger.json",ledger())
    else:write(audit/"3/state.json",{"status":"FAILED_CLOSED"})
    with pytest.raises(worker.WrapperError):worker.verified_completion(root,plan,desc)


def test_pending_root_plan_default_dryrun_is_explicit_and_creates_no_files(monkeypatch,capsys,tmp_path):
    monkeypatch.setattr(worker,"DREYER_PLAN_SHA",None)
    monkeypatch.setattr(worker,"fixed_plans",lambda *a:pytest.fail("pending plan must never read or validate a plan"))
    assert worker.main(["--root",str(tmp_path)])==0
    assert json.loads(capsys.readouterr().out)["status"]=="PENDING_ROOT_FROZEN_PLAN_NO_NETWORK_R2_OR_FIT"
    assert list(tmp_path.iterdir())==[]
    assert worker.main(["--root",str(tmp_path),"--write-declaration"])==1
    assert "not_frozen" in json.loads(capsys.readouterr().out)["error_code"]
    assert list(tmp_path.iterdir())==[]
