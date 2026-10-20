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
from tools import prepare_dreyer_source_cpu_v5 as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")
    return path


def ledger(charged=None, wall=0., history=None):
    return {"schema_version":"paper3-source-extension-budget-ledger-v1",
        "plan_sha256":worker.DREYER_PLAN_SHA,
        "charged":charged or {"provider_body_bytes":0,"r2_get_body_bytes":0},
        "elapsed_seconds":wall,"history":history or []}


def freeze_fixture_identifiers(monkeypatch):
    for index,name in enumerate(("DREYER_PLAN_SHA","NATIVE_ADAPTER_SHA",
            "QUALIFICATION_COMPLETION_SHA","ADMISSION_REVIEW_SHA")):
        if getattr(worker,name) is None:
            monkeypatch.setattr(worker,name,format(index+10,"x")*64)
    if worker.YANG_WRAPPER_SHA is None:
        # Fixture-only binding to the actual candidate bytes; production remains blocked.
        monkeypatch.setattr(worker,"YANG_WRAPPER_SHA",hashlib.sha256(
            (worker.ROOT/worker.YANG_ENTRY).read_bytes()).hexdigest())


@pytest.fixture
def v2_verifier(monkeypatch):
    """V5 delegates to frozen V2; its full historical verifier is tested separately."""
    verifier=SimpleNamespace(static_calls=[],complete_calls=[],completion_result=None)
    def static(root,plan):
        verifier.static_calls.append((root,plan));return {"software_fixture_static_authority":True}
    def complete(root,plan):
        verifier.complete_calls.append((root,plan));return verifier.completion_result
    verifier.verify_static_authority=static
    verifier.verify_complete_transport_history=complete
    original=worker.importlib.import_module
    monkeypatch.setattr(worker.importlib,"import_module",lambda name:verifier
        if name==worker.YANG_TRANSPORT_MODULE else original(name))
    return verifier


@pytest.fixture
def isolated(tmp_path, monkeypatch,v2_verifier):
    freeze_fixture_identifiers(monkeypatch)
    target = tmp_path/worker.ENTRY
    target.parent.mkdir(parents=True)
    shutil.copyfile(worker.__file__,target)
    parent=tmp_path/worker.PARENT_WRAPPER_PATH
    shutil.copyfile(worker.ROOT/worker.PARENT_WRAPPER_PATH,parent)
    peer=tmp_path/worker.YANG_ENTRY
    shutil.copyfile(worker.ROOT/worker.YANG_ENTRY,peer)
    core = SimpleNamespace(**{k:getattr(real,k) for k in
        ("local","descriptor","atomic_json","digest","checked","PreparationError","Ledger","storage","verify_raw_file")})
    core.peer_verifier=v2_verifier
    monkeypatch.setattr(worker,"kernel",lambda:core)
    return tmp_path,core


def test_default_checks_remain_readonly_and_explicit_execute_requires_declaration(monkeypatch,capsys,tmp_path):
    freeze_fixture_identifiers(monkeypatch)
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


@pytest.mark.parametrize("old_sha",[worker.OLD_DREYER_SHA,worker.V2_DREYER_SHA,worker.V3_DREYER_SHA,worker.V4_DREYER_SHA,worker.QUALIFICATION_PLAN_SHA])
def test_version5_never_executes_original_or_closed_ancestor_or_qualification_plan(isolated,old_sha):
    root,core=isolated
    write(root/worker.PLAN_PATH,{"dataset":"Dreyer2023","plan_sha256":old_sha,
        "record_count":520,"records":[{}for _ in range(520)],
        "code":{"p3work/source_extensions/dreyer2023.py":{"sha256":worker.NATIVE_ADAPTER_SHA}}})
    write(root/worker.YANG_PATH,{"dataset":"Yang2025","plan_sha256":worker.YANG_PLAN_SHA})
    core.validate_plan=lambda *a:pytest.fail("old plan must close before adapter/IO")
    with pytest.raises(worker.WrapperError,match="exact"):
        worker.fixed_plans(root)


@pytest.fixture
def six_history(isolated,monkeypatch):
    root,core=isolated
    sources=[]
    for label,provider,r2,wall,caps in [
        ("V1",10,30,12.4,(1000,2000,100)),
        ("V2",20,40,19.6,(990,1970,87)),
        ("auxiliary_schema_review",50,0,6.2,(100,1,1800)),
        ("V3",30,50,1.2,(920,1930,61)),
        ("V4",40,60,2.2,(890,1880,59)),
        ("native_qualification",50,70,3.2,(850,1820,10))]:
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


def test_all_six_actual_usages_and_every_ancestor_wall_reservation_are_conserved(six_history,monkeypatch):
    monkeypatch.setattr(worker,"INTERSECT_QUALIFICATION_STAGE_CAP",True)
    root,_,_=six_history
    # Initial minus all actual is 100-ceil(44.8)=55; V2/V3/V4 continuation
    # bounds are 54/54/53; the qualification phase reservation leaves only 6.
    plan={"budgets":{"provider_body_bytes":800,"r2_get_body_bytes":1750,"wall_seconds":6}}
    proof=worker.prior_budget_evidence(root,plan)
    assert proof["accounting_schema"]=="paper3-dreyer-six-ledger-budget-v1"
    assert proof["deducted_wall_seconds"]==45 and proof["deducted_body_bytes"]=={
        "provider_body_bytes":200,"r2_get_body_bytes":250}
    assert [x["source"]for x in proof["closed_history_sources"]]==[
        "V1","V2","auxiliary_schema_review","V3","V4","native_qualification"]
    assert proof["carried_over_verified_R2_body_bytes"]==0
    assert proof["original_caps_minus_all_actual_usage"]=={
        "provider_body_bytes":800,"r2_get_body_bytes":1750,"wall_seconds":55}
    assert proof["remaining_caps"]==plan["budgets"]
    for invalid_wall in [7,53,55,100]:
        plan["budgets"]["wall_seconds"]=invalid_wall
        with pytest.raises(worker.WrapperError,match="deducted|conserv"):
            worker.prior_budget_evidence(root,plan)


def replace_history_cap(root,core,sources,label,key,value,monkeypatch):
    changed=list(sources);index=[r[0]for r in sources].index(label)
    old=root/changed[index][1];body=worker.read_json(old);body.pop("plan_sha256")
    body["budgets"][key]=value;sha=core.digest(body);body["plan_sha256"]=sha;write(old,body)
    journal=root/changed[index][2];previous=worker.read_json(journal);previous["plan_sha256"]=sha;write(journal,previous)
    changed[index]=(*changed[index][:3],sha)
    monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(changed))


@pytest.mark.parametrize("label,cap,expected",[("V2",35,2),("V3",10,3),("V4",7,1)])
def test_each_ancestor_ceiling_intersects_all_subsequent_actual_wall_usage(six_history,monkeypatch,label,cap,expected):
    root,core,sources=six_history
    replace_history_cap(root,core,sources,label,"wall_seconds",cap,monkeypatch)
    assert worker.remaining_budget_evidence(root)["remaining_caps"]["wall_seconds"]==expected


@pytest.mark.parametrize("label,cap",[("V2",195),("V3",125),("V4",95),("native_qualification",55)])
def test_each_ancestor_body_ceiling_is_conserved_without_double_deducting(six_history,monkeypatch,label,cap):
    monkeypatch.setattr(worker,"INTERSECT_QUALIFICATION_STAGE_CAP",True)
    root,core,sources=six_history
    replace_history_cap(root,core,sources,label,"provider_body_bytes",cap,monkeypatch)
    assert worker.remaining_budget_evidence(root)["remaining_caps"]["provider_body_bytes"]==5


@pytest.mark.parametrize("missing",["V1","V2","auxiliary_schema_review","V3","V4","native_qualification"])
def test_every_actual_history_source_required_once(six_history,monkeypatch,missing):
    root,_,sources=six_history
    monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(s for s in sources if s[0]!=missing))
    with pytest.raises(worker.WrapperError):worker.remaining_budget_evidence(root)


@pytest.mark.parametrize("change",["duplicate","open_auxiliary","open_V3","open_V4","open_qualification","tampered_plan","over_budget"])
def test_history_duplicate_tampered_or_unclosed_usage_rejected(six_history,monkeypatch,change):
    root,_,sources=six_history
    if change=="duplicate":
        altered=list(sources);altered[2]=(sources[2][0],sources[2][1],sources[1][2],sources[2][3])
        monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(altered))
    elif change=="tampered_plan":
        path=root/sources[4][1];value=worker.read_json(path);value["unexpected"]="changed";write(path,value)
    else:
        index={"open_auxiliary":2,"open_V3":3,"open_V4":4,"open_qualification":5}.get(change,2)
        path=root/sources[index][2];value=worker.read_json(path)
        if change.startswith("open_"):value["active_transfer"]={}
        else:
            value["charged"]["provider_body_bytes"]=200
            value["history"][0].update(reserved_bytes=200,verified_bytes=200)
        write(path,value)
    with pytest.raises(worker.WrapperError):worker.remaining_budget_evidence(root)


def test_version5_without_root_frozen_plan_is_unexecutable(isolated,monkeypatch):
    root,_=isolated
    monkeypatch.setattr(worker,"DREYER_PLAN_SHA",None)
    with pytest.raises(worker.WrapperError,match="not_frozen"):
        worker.fixed_plans(root)


def test_frozen_V4_parent_may_not_be_patched_or_replaced(isolated):
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


def fingerprint(root,pid=4201,start=100,explicit_plan=False):
    command=["python","-u","-B",worker.YANG_ENTRY,"--execute"]
    if explicit_plan:command.extend(["--plan",worker.YANG_PATH])
    return {"pid":pid,"starttime_ticks":start,"boot_id":"fixture-boot-id",
        "cwd":str(root),"command":command}


def transport_amendment_fixture(root,core,planpath,beforepath=None):
    value={"schema_version":worker.YANG_SCHEMA,"status":"AUTHORIZED_SINGLE_CALL_SAMEPLAN_TRANSPORT_V2",
        "frozen_plan":core.descriptor(root,planpath),"wrapper_code":core.descriptor(root,root/worker.YANG_ENTRY),
        "session_headers":{"Cache-Control":"no-cache","Pragma":"no-cache"},
        "requests_adapter_retries":0,"max_independent_execute_calls":worker.YANG_MAX_CALLS,
        "no_tight_retry":True,"original_kernel_is_only_transfer_and_wall_budget_authority":True}
    if beforepath is not None:value["parent_closed_ledger_snapshot"]=core.descriptor(root,beforepath)
    return value


def no_cache_peer_files(root,core):
    plan=write(root/worker.YANG_PATH,{"dataset":"Yang2025","plan_sha256":worker.YANG_PLAN_SHA})
    amendment=write(root/worker.YANG_AMENDMENT_PATH,transport_amendment_fixture(root,core,plan))
    history={"identity":{"schema_version":worker.YANG_SCHEMA,
        "amendment":core.descriptor(root,amendment),"plan_sha256":worker.YANG_PLAN_SHA},
        "calls_used":1,"active_call":1,"terminal":None,"call_receipts":[]}
    write(root/worker.YANG_HISTORY_PATH,history)
    return history


def test_peer_identity_binds_real_NoCache_code_default_plan_boot_and_starttime(isolated):
    root,core=isolated;no_cache_peer_files(root,core)
    one=worker.yang_identity(root,fingerprint(root))
    two=worker.yang_identity(root,fingerprint(root,start=101))
    assert one["pid"]==two["pid"] and one!=two
    assert one["mode"]=="concurrent_no_cache_wrapper_v2"
    assert worker.yang_identity(root,fingerprint(root,explicit_plan=True))["plan_sha256"]==worker.YANG_PLAN_SHA
    boot=fingerprint(root);boot["boot_id"]="new-fixture-boot"
    assert worker.yang_identity(root,boot)!=one


@pytest.mark.parametrize("bad",["old_cli","wrong_plan","dryrun","write_amendment","unknown_argument",
    "wrong_root","duplicate_plan","invalid_limit","changed_cwd","changed_code"])
def test_disguised_or_unbounded_peer_is_never_accepted(isolated,bad):
    root,core=isolated;no_cache_peer_files(root,core);fp=fingerprint(root)
    if bad=="old_cli":fp["command"][3]="tools/prepare_source_extensions.py"
    elif bad=="wrong_plan":fp["command"].extend(["--plan","other_plan.json"])
    elif bad=="dryrun":fp["command"].remove("--execute")
    elif bad=="write_amendment":fp["command"].append("--write-amendment")
    elif bad=="unknown_argument":fp["command"].append("--unbounded")
    elif bad=="wrong_root":fp["command"].extend(["--root",str(root/"other")])
    elif bad=="duplicate_plan":fp["command"].extend(["--plan",worker.YANG_PATH,"--plan",worker.YANG_PATH])
    elif bad=="invalid_limit":fp["command"].extend(["--max-new-records","0"])
    elif bad=="changed_cwd":fp["cwd"]=str(root/"other")
    else:(root/worker.YANG_ENTRY).write_text("# replaced software fixture peer\n")
    assert worker.yang_identity(root,fp)is None


@pytest.mark.parametrize("bad",["terminal_failure","over_limit","wrong_amendment","wrong_plan","bad_authority"])
def test_peer_transport_history_and_amendment_must_bind_reviewed_bounded_wrapper(isolated,bad):
    root,core=isolated;history=no_cache_peer_files(root,core)
    if bad=="terminal_failure":history["terminal"]="STOPPED_NON_TRANSPORT_ERROR_NO_RETRY"
    elif bad=="over_limit":history["calls_used"]=worker.YANG_MAX_CALLS+1
    elif bad=="wrong_amendment":history["identity"]["amendment"]["sha256"]="f"*64
    elif bad=="wrong_plan":history["identity"]["plan_sha256"]="f"*64
    else:
        filename=root/worker.YANG_AMENDMENT_PATH;amendment=worker.read_json(filename)
        amendment["original_kernel_is_only_transfer_and_wall_budget_authority"]=False;write(filename,amendment)
    write(root/worker.YANG_HISTORY_PATH,history)
    assert worker.yang_identity(root,fingerprint(root))is None


@pytest.fixture
def peer(isolated,monkeypatch):
    root,core=isolated;no_cache_peer_files(root,core)
    lock=root/worker.AUDIT_ROOT/".source-extension-singleton.lock";lock.parent.mkdir(parents=True,exist_ok=True);lock.touch()
    owner=root/worker.YANG_OWNER_PATH;owner.parent.mkdir(parents=True,exist_ok=True);owner.touch()
    state={"persistent":[4201],"global":[4201],"fingerprint":fingerprint(root)}
    monkeypatch.setattr(worker,"lock_owners",lambda path:state["persistent"if path==owner else"global"])
    monkeypatch.setattr(worker,"process_fingerprint",lambda pid:state["fingerprint"])
    def incomplete(_):raise worker.WrapperError("Yang_not_complete_stop_new_records_preserve_ledger")
    monkeypatch.setattr(worker,"completed_yang",incomplete)
    guard=worker.PeerGuard(root,expected_peer=worker.yang_identity(root,state["fingerprint"]))
    yield root,core,state,guard
    guard.close()


def test_peer_pid_reuse_or_replacement_stops_new_records(peer):
    _,_,state,guard=peer
    guard.check();state["fingerprint"]["starttime_ticks"]+=1
    with pytest.raises(worker.WrapperError,match="replaced"):guard.check()


def test_same_persistent_wrapper_spans_global_lock_gap_without_taking_over(peer):
    root,_,state,guard=peer
    first=worker.current_yang(root);guard.check();state["global"]=[]
    guard.check()
    assert worker.current_yang(root)==first and guard.peer==first and guard.owned is False
    state["global"]=[4201];guard.check()


def test_peer_disappearance_does_not_unlock_or_takeover_global(peer):
    _,_,state,guard=peer
    guard.check();state.update(persistent=[],**{"global":[]})
    with pytest.raises(worker.WrapperError,match="not_complete"):guard.check()
    assert guard.owned is False


@pytest.mark.parametrize("bad",["unknown_global","multiple_persistent","invalid_process","global_without_persistent"])
def test_unknown_owner_is_never_accepted_even_during_global_lock_gap(peer,bad):
    _,_,state,guard=peer
    if bad=="unknown_global":state["global"]=[4300]
    elif bad=="multiple_persistent":state["persistent"]=[4201,4300]
    elif bad=="invalid_process":state["fingerprint"]=None
    else:state["persistent"]=[]
    with pytest.raises(worker.WrapperError):guard.check()


def test_real_complete_peer_allows_global_takeover_only_after_full_offline_verification(peer,monkeypatch):
    _,_,state,guard=peer
    guard.check();state.update(persistent=[],**{"global":[]});checked=[]
    monkeypatch.setattr(worker,"completed_yang",lambda root:checked.append(True)or{
        "mode":"already_complete","plan_sha256":worker.YANG_PLAN_SHA,"whole_source_proof":"software_fixture"})
    guard.check();assert guard.owned is True and checked==[True]
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
    monkeypatch.setattr(worker,"verify_declaration",lambda *a:({"initial_yang_process":{"pid":42}},desc))
    monkeypatch.setattr(worker,"qualification_evidence",lambda *a:{"reference_states":{},"qualifier_owned_raw":[]})
    monkeypatch.setattr(worker,"protect_qualified_raw",lambda *a:None)
    monkeypatch.setattr(worker,"new_record_provenance",lambda *a:None)
    monkeypatch.setattr(worker,"configure_cpu",lambda:None)
    monkeypatch.setattr(worker,"resource_guard",lambda *a:None)
    monkeypatch.setattr(worker,"verified_completion",lambda *a:None)
    peer=SimpleNamespace(peer={"pid":42},owned=False,check=lambda:None,close=lambda:None,
        coordination_evidence=lambda:coordination_fixture())
    monkeypatch.setattr(worker,"PeerGuard",lambda *a,**k:peer)
    core.ADAPTERS={"Dreyer2023":"fixture_dreyer_adapter"}
    original_import=worker.importlib.import_module
    monkeypatch.setattr(worker.importlib,"import_module",lambda name:SimpleNamespace(materialize=None,audit_native=None)
        if name=="fixture_dreyer_adapter"else original_import(name))
    class Session:
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(worker,"provider_session",Session)
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
    monkeypatch.setattr(worker,"PeerGuard",lambda *a,**k:SimpleNamespace(peer={},check=check,close=lambda:None))
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
    monkeypatch.setattr(worker,"PeerGuard",lambda *a,**k:pytest.fail("complete reuse touched peer"))
    core.Ledger=lambda *a:pytest.fail("complete reuse opened mutable ledger")
    result=worker.execute(root,declaration_path=desc["path"])
    assert result["new_R2_operations_this_invocation"]==0 and calls==["record-0"]


def coordination_fixture(initial=None,*,takeover=False,completion=None):
    import copy
    if initial is None:initial={"mode":"concurrent_no_cache_wrapper_v2","plan_sha256":worker.YANG_PLAN_SHA,
        "pid":4201,"starttime_ticks":100,"boot_id":"software-fixture-boot"}
    return {"schema_version":"paper3-dreyer-yang-peer-coordination-v1",
        "initial_yang_process":copy.deepcopy(initial),
        "last_verified_live_peer":copy.deepcopy(initial)if initial.get("mode")=="concurrent_no_cache_wrapper_v2"else None,
        "global_lock_takeover":takeover,"completed_yang_proof":copy.deepcopy(completion)}


def completion_fixture():
    return {"mode":"already_complete","plan_sha256":worker.YANG_PLAN_SHA,
        "transport_completion_proof":{"software_fixture_full_V1_plus_V2_provenance":{
            "source_manifest":{"sha256":"1"*64},"closed_call_receipts":[{"sha256":"2"*64}]}}}


@pytest.mark.parametrize("takeover",[False,True])

def test_final_backup_includes_seven_exact_objects_and_required_original_IO(isolated,monkeypatch,takeover):
    monkeypatch.setattr(worker,"qualification_evidence",lambda *a:{"software_fixture":True})
    monkeypatch.setattr(worker,"all_numeric_regressions",lambda *a:{"software_fixture_128_regressions":True})
    monkeypatch.setattr(worker,"new_record_provenance",lambda *a:None)
    root,core=isolated;audit=worker.audit_path(root);audit.mkdir(parents=True)
    completion=completion_fixture()if takeover else None
    coordination=coordination_fixture(takeover=takeover,completion=completion)
    monkeypatch.setattr(worker,"completed_yang",lambda root:completion if takeover else pytest.fail("live finish read completion/ledger"))
    decl=write(audit/"parallel_root_declaration.json",{"initial_yang_process":coordination["initial_yang_process"]})
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
    result=worker._finalize(root,plan,[],SimpleNamespace(checkpoint=lambda **k:close.append(k)),core.descriptor(root,decl),coordination)
    assert close==[{"close":True}]
    assert result["status"]=="SOURCE_EXTENSION_CACHE_COMPLETE"
    names={p.name for p,_ in captured["files"]}
    assert {"parallel_execution_receipt.json","parallel_root_declaration.json"}<=names
    proof=worker.read_json(audit/"parallel_execution_receipt.json")
    assert proof["actual_fits"]==proof["carried_over_verified_R2_body_bytes"]==0
    assert proof["record_count"]==520 and proof["scientific_kernel_unchanged"]is True


    assert proof["yang_peer_coordination"]==coordination
    if takeover:
        assert proof["yang_peer_coordination"]["completed_yang_proof"]["transport_completion_proof"]==completion["transport_completion_proof"]

@pytest.fixture(params=[False,True])
def completed(isolated,monkeypatch,request):
    monkeypatch.setattr(worker,"qualification_evidence",lambda *a:{"software_fixture":True})
    monkeypatch.setattr(worker,"all_numeric_regressions",lambda *a:{"software_fixture_128_regressions":True})
    monkeypatch.setattr(worker,"new_record_provenance",lambda *a:None)
    root,core=isolated;audit=worker.audit_path(root);audit.mkdir(parents=True)
    takeover=request.param;completion=completion_fixture()if takeover else None
    coordination=coordination_fixture(takeover=takeover,completion=completion)
    core.coordination_calls=[];core.fixture_coordination=coordination
    def completed_peer(root):
        if not takeover:pytest.fail("live finish read completion/ledger")
        core.coordination_calls.append(True);return completion
    monkeypatch.setattr(worker,"completed_yang",completed_peer)
    decl=write(audit/"parallel_root_declaration.json",{"initial_yang_process":coordination["initial_yang_process"]})
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
    status=worker._finalize(root,plan,[],SimpleNamespace(checkpoint=lambda **k:None),desc,coordination)
    manifest=worker.read_json(root/status["manifest"]["path"])
    amount=sum(r["bytes"]for r in core.receipt_rows(root,status["backup_receipt"]))
    amount+=sum(sum(f["bytes"]for f in s["raw_files"])+sum(s[k]["bytes"]for k in keys)for s in shards)
    write(audit/"budget_ledger.json",ledger({"provider_body_bytes":0,"r2_get_body_bytes":amount},2.,
        [{"kind":"r2_get_body_bytes","reserved_bytes":amount,"verified_bytes":amount,"status":"SUCCESS_FULL_BODY_VERIFIED"}]))
    return root,core,plan,desc,status,seen


def test_complete_reuse_rechecks_all520_records_final7_closed_budget_without_io(completed):
    root,core,plan,desc,_,seen=completed
    value=worker.verified_completion(root,plan,desc)
    assert value["completed_local_proofs_reverified"]is True
    assert len(seen)==520 and value["new_provider_GETs_this_invocation"]==value["new_R2_operations_this_invocation"]==0
    assert len(core.coordination_calls)==(2 if core.fixture_coordination["global_lock_takeover"]else 0)


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
    assert json.loads(capsys.readouterr().out)["status"]=="PENDING_ALL520_REVIEW_AND_ROOT_FROZEN_V5_NO_NETWORK_R2_OR_FIT"
    assert list(tmp_path.iterdir())==[]
    assert worker.main(["--root",str(tmp_path),"--write-declaration"])==1
    assert "not_frozen" in json.loads(capsys.readouterr().out)["error_code"]
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize("missing",["DREYER_PLAN_SHA","NATIVE_ADAPTER_SHA",
    "QUALIFICATION_COMPLETION_SHA","ADMISSION_REVIEW_SHA","YANG_WRAPPER_SHA"])
def test_every_unfrozen_scientific_or_qualification_identifier_blocks_before_plan_read(isolated,monkeypatch,missing):
    root,_=isolated
    monkeypatch.setattr(worker,missing,None)
    monkeypatch.setattr(worker,"read_json",lambda *a:pytest.fail("unfrozen entry reached a plan or model"))
    with pytest.raises(worker.WrapperError,match="not_frozen"):
        worker.fixed_plans(root)


@pytest.fixture
def qualified520(isolated,monkeypatch):
    """Tiny JSON proofs test qualification gates; no original EEG or remote IO."""
    root,core=isolated
    records=[{"record_id":f"fixture-{i}","raw_files":[{
        "path":f"data/fixture-{i}.edf","sha256":format(i,"064x"),"bytes":1}]} for i in range(520)]
    parent={"records":records};parent["plan_sha256"]=core.digest(parent)
    monkeypatch.setattr(worker,"V4_DREYER_SHA",parent["plan_sha256"])
    parentpath=write(root/"qualification/parent.json",parent)
    parent_journal=ledger();parent_journal["plan_sha256"]=parent["plan_sha256"]
    parent_journalpath=write(root/"qualification/parent_ledger.json",parent_journal)
    original_code=root/"qualification/original.py";original_code.parent.mkdir(exist_ok=True)
    original_code.write_text("# software fixture historical code\n")
    alias=root/"qualification/immutable_alias.py";shutil.copyfile(original_code,alias)
    native_refs={};old_refs={}
    for i,record in enumerate(records):
        raw=record["raw_files"][0]
        native=({"source_file_sha256":raw["sha256"],"source_file_bytes":1} if i<128 else {
            "source_file":raw,"annotations_modified_or_synthesized":False,
            "scientific_admission_granted":False,"EEG_array_samples_read":False})
        nativepath=write(root/f"qualification/{i}/native.json",native)
        native_refs[i]=core.descriptor(root,nativepath)
        if i<128:
            oldpath=write(root/f"qualification/{i}/old_state.json",{
                "status":"COMPLETE","artifacts":[native_refs[i]]})
            old_refs[record["record_id"]]=core.descriptor(root,oldpath)
    qplan={"record_count":520,"records":records,"scientific_admission_granted":False,
        "code":{"old_adapter.py":core.descriptor(root,original_code)},
        "parent_plan":core.descriptor(root,parentpath),"parent_closed_ledger":core.descriptor(root,parent_journalpath),
        "reused_complete_parent_states":old_refs,
        "budgets":{"provider_body_bytes":100000000,"r2_get_body_bytes":100000000,"wall_seconds":10000}}
    qplan["plan_sha256"]=core.digest(qplan)
    monkeypatch.setattr(worker,"QUALIFICATION_PLAN_SHA",qplan["plan_sha256"])
    qplanpath=write(root/"qualification/plan.json",qplan)
    refs=[]
    for i,record in enumerate(records):
        state={"status":"STRUCTURE_QUALIFICATION_RECORDED","record_id":record["record_id"],
            "plan_sha256":qplan["plan_sha256"],"scientific_admission_granted":False,
            "EEG_array_samples_read":False,"native_structure":native_refs[i]}
        if i<128:state.update(reused_parent_state=old_refs[record["record_id"]],
            existing_V4_gate={"status":"EXISTING_V4_NATIVE_STRUCTURE_GATE_PASS"})
        else:
            receipt=write(root/f"qualification/{i}/metadata_receipt.json",{"objects":[{
                **native_refs[i],"dataset":"Dreyer2023","role":"audit"}]})
            state.update(raw_backup_receipt={"software_fixture":True},
                metadata_backup_receipt=core.descriptor(root,receipt))
        refs.append(core.descriptor(root,write(root/f"qualification/{i}/state.json",state)))
    summary={"plan":core.descriptor(root,qplanpath),"records":520,"record_states":refs,
        "scientific_admission_granted":False,"EEG_array_samples_read":False,"fits":0,"GPU_actions":0,
        "status":"ALL520_STRUCTURES_CAPTURED_REVIEW_REQUIRED_NOT_TRAINING_READY"}
    summarypath=write(root/"qualification/summary.json",summary)
    finalreceipt=write(root/"qualification/final_receipt.json",{"objects":[
        {**core.descriptor(root,qplanpath),"dataset":"Dreyer2023","role":"audit"},
        {**core.descriptor(root,summarypath),"dataset":"Dreyer2023","role":"audit"},
        *[{**r,"dataset":"Dreyer2023","role":"audit"}for r in refs]]})
    status={"status":summary["status"],"scientific_admission_granted":False,
        "summary":core.descriptor(root,summarypath),"backup_receipt":core.descriptor(root,finalreceipt)}
    statuspath=write(root/"qualification/status.json",status)
    journal=ledger({"provider_body_bytes":0,"r2_get_body_bytes":10000000},2.,[{
        "kind":"r2_get_body_bytes","reserved_bytes":10000000,"verified_bytes":10000000,
        "status":"SUCCESS_FULL_BODY_VERIFIED"}])
    journal["plan_sha256"]=qplan["plan_sha256"]
    journalpath=write(root/"qualification/ledger.json",journal)
    sources=list(worker.HISTORY_SOURCES)
    sources[4]=("V4",parentpath.relative_to(root).as_posix(),parent_journalpath.relative_to(root).as_posix(),parent["plan_sha256"])
    sources[5]=("native_qualification",qplanpath.relative_to(root).as_posix(),journalpath.relative_to(root).as_posix(),qplan["plan_sha256"])
    monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(sources))
    reviewpath=write(root/"qualification/review.json",{"software_fixture_all520_review":True})
    proof={"schema_version":"paper3-dreyer-all520-closed-qualification-proof-v1",
        "offline_all520_verified_before_canonical_change":True,"scientific_admission_granted":False,
        "qualification_plan":core.descriptor(root,qplanpath),"qualification_status":core.descriptor(root,statuspath),
        "qualification_summary":core.descriptor(root,summarypath),"closed_ledger":core.descriptor(root,journalpath),
        "historical_code_aliases":{"old_adapter.py":core.descriptor(root,alias)}}
    proofpath=root/worker.QUALIFICATION_COMPLETION_PATH
    admission={"schema_version":"paper3-dreyer-v5-native-admission-v1",
        "status":"ALL520_NATIVE_ADMISSION_FROZEN","adapter_sha256":worker.NATIVE_ADAPTER_SHA,
        "record_count":520,"record_ids_sha256":core.digest([r["record_id"]for r in records]),
        "fits":0,"GPU_actions":0,"external_payload_access_allowed":False,
        "all520_review":core.descriptor(root,reviewpath)}
    admissionpath=root/worker.ADMISSION_REVIEW_PATH
    def bind():
        write(proofpath,proof);proofref=core.descriptor(root,proofpath)
        monkeypatch.setattr(worker,"QUALIFICATION_COMPLETION_SHA",proofref["sha256"])
        admission["qualification_completion_proof"]=proofref;write(admissionpath,admission)
        monkeypatch.setattr(worker,"ADMISSION_REVIEW_SHA",core.descriptor(root,admissionpath)["sha256"])
    bind()
    checked_old=[];raw_checks=[]
    core._state_verified=lambda root,state,parent,record:checked_old.append(record["record_id"])or True
    core.verify_raw_receipt=lambda *a:raw_checks.append(True)
    core.receipt_rows=lambda root,item:worker.read_json(core.checked(root,item))["objects"]
    return SimpleNamespace(root=root,core=core,plan={"records":records},proof=proof,admission=admission,
        bind=bind,status=status,statuspath=statuspath,summary=summary,summarypath=summarypath,
        finalreceipt=finalreceipt,journal=journal,journalpath=journalpath,qplanpath=qplanpath,alias=alias,
        checked_old=checked_old,raw_checks=raw_checks)


def test_qualification_520_historical_metadata_proofs_require_128_original_references(qualified520):
    f=qualified520;proof=worker.qualification_evidence(f.root,f.plan)
    assert proof["record_count"]==520 and proof["reference_record_count"]==128
    assert proof["historical_metadata_only_not_new_cache"]is True
    assert len(f.checked_old)==128 and len(f.raw_checks)==392


@pytest.mark.parametrize("bad",["flags","missing_alias","partial_summary","missing_final_GET","open_ledger","alternate_ledger","alternate_plan","alternate_parent_ledger","admission_adapter"])
def test_flags_partial_history_unclosed_ledger_or_missing_full_GET_never_grant_V5_admission(qualified520,monkeypatch,bad):
    f=qualified520
    if bad=="flags":f.proof["scientific_admission_granted"]=True
    elif bad=="missing_alias":f.proof["historical_code_aliases"]={}
    elif bad=="partial_summary":
        f.summary["record_states"].pop();write(f.summarypath,f.summary)
        f.proof["qualification_summary"]=f.core.descriptor(f.root,f.summarypath)
        f.status["summary"]=f.proof["qualification_summary"];write(f.statuspath,f.status)
        f.proof["qualification_status"]=f.core.descriptor(f.root,f.statuspath)
    elif bad=="missing_final_GET":
        receipt=worker.read_json(f.finalreceipt);receipt["objects"].pop();write(f.finalreceipt,receipt)
        f.status["backup_receipt"]=f.core.descriptor(f.root,f.finalreceipt);write(f.statuspath,f.status)
        f.proof["qualification_status"]=f.core.descriptor(f.root,f.statuspath)
    elif bad=="open_ledger":
        f.journal["active_transfer"]={};write(f.journalpath,f.journal)
        f.proof["closed_ledger"]=f.core.descriptor(f.root,f.journalpath)
    elif bad=="alternate_ledger":
        alias=f.root/"qualification/alternate_ledger.json";shutil.copyfile(f.journalpath,alias)
        f.proof["closed_ledger"]=f.core.descriptor(f.root,alias)
    elif bad=="alternate_plan":
        alias=f.root/"qualification/alternate_plan.json";shutil.copyfile(f.qplanpath,alias)
        f.proof["qualification_plan"]=f.core.descriptor(f.root,alias)
    elif bad=="alternate_parent_ledger":
        sources=list(worker.HISTORY_SOURCES);alias=f.root/"qualification/alternate_parent_ledger.json"
        shutil.copyfile(f.root/sources[4][2],alias)
        sources[4]=(sources[4][0],sources[4][1],alias.relative_to(f.root).as_posix(),sources[4][3])
        monkeypatch.setattr(worker,"HISTORY_SOURCES",tuple(sources))
    else:f.admission["adapter_sha256"]="f"*64
    f.bind()
    with pytest.raises(worker.WrapperError):worker.qualification_evidence(f.root,f.plan)


@pytest.fixture
def completed_yang153(isolated):
    """V2's pure verifier proves all caches; V5 independently binds its result."""
    root,core=isolated;audit=root/worker.AUDIT_ROOT/"Yang2025"/worker.YANG_PLAN_SHA
    records=[{"record_id":f"fixture-{i}"}for i in range(153)]
    plan={"dataset":"Yang2025","plan_sha256":worker.YANG_PLAN_SHA,"records":records,
        "record_count":153,"budgets":{"provider_body_bytes":1000000000,
            "r2_get_body_bytes":1000000000,"wall_seconds":1000}}
    planpath=write(root/worker.YANG_PATH,plan);core.validate_plan=lambda *a:None
    manifestpath=write(audit/"source_cache.json",{"software_fixture_all153_cache_proof":True})
    receiptpath=write(audit/"final_receipt.json",{"software_fixture_full_GET_proof":True})
    ledgerpath=write(audit/"budget_ledger.json",{"software_fixture_original_closed_ledger":True})
    status={"status":"SOURCE_EXTENSION_CACHE_COMPLETE","dataset":"Yang2025","record_count":153,
        "training_started":False,"GPU_actions":0,"external_payload_accessed":False,
        "manifest":core.descriptor(root,manifestpath),"backup_receipt":core.descriptor(root,receiptpath)}
    statuspath=write(audit/"coordinator_status.json",status)
    amendment=transport_amendment_fixture(root,core,planpath)
    amendmentpath=write(root/worker.YANG_AMENDMENT_PATH,amendment);amendmentref=core.descriptor(root,amendmentpath)
    history={"identity":{"schema_version":worker.YANG_SCHEMA,"amendment":amendmentref,"plan_sha256":worker.YANG_PLAN_SHA},
        "terminal":"SOURCE_EXTENSION_CACHE_COMPLETE","active_call":None,"calls_used":1,
        "call_receipts":[{"software_fixture_verified_closed_call":True}]}
    historypath=write(root/worker.YANG_HISTORY_PATH,history)
    result={"status":"SOURCE_EXTENSION_CACHE_COMPLETE","already_complete_offline_verified":True,
        "verified_records":153,"manifest":status["manifest"],"backup_receipt":status["backup_receipt"],
        "original_closed_ledger":core.descriptor(root,ledgerpath),
        "transport_history":{"amendment":amendmentref,"recovery_ledger":core.descriptor(root,historypath),"calls_used":1,
            "provenance_closure":{"software_fixture_V1_plus_V2_full_historical_chain":True}},
        "complete_V2_provenance_receipts":{"software_fixture_opaque_full_proof":True}}
    core.peer_verifier.completion_result=result
    return SimpleNamespace(root=root,core=core,status=status,statuspath=statuspath,receiptpath=receiptpath,
        ledgerpath=ledgerpath,history=history,historypath=historypath,plan=plan,planpath=planpath,
        amendment=amendment,amendmentpath=amendmentpath,result=result,verifier=core.peer_verifier)


def test_completed_V2_peer_calls_pure_verifier_and_preserves_entire_transport_proof(completed_yang153):
    f=completed_yang153;value=worker.completed_yang(f.root)
    assert value["mode"]=="already_complete" and value["plan_sha256"]==worker.YANG_PLAN_SHA
    assert f.verifier.static_calls==[(f.root,f.plan)]and f.verifier.complete_calls==[(f.root,f.plan)]
    assert value["transport_completion_proof"]==f.result
    assert value["transport_history"]==f.result["transport_history"]


@pytest.mark.parametrize("bad",["none","status","not_verified","partial_records","manifest","backup_receipt",
    "original_ledger","history_amendment","history_ledger","empty_closure","non_dict_closure","history_none","training_flag"])
def test_incomplete_or_unbound_V2_verifier_result_never_allows_takeover(completed_yang153,bad):
    f=completed_yang153
    if bad=="none":f.verifier.completion_result=None
    elif bad=="status":f.result["status"]="PARTIAL"
    elif bad=="not_verified":f.result["already_complete_offline_verified"]=False
    elif bad=="partial_records":f.result["verified_records"]=152
    elif bad in {"manifest","backup_receipt"}:f.result[bad]={"other":"proof"}
    elif bad=="original_ledger":f.result["original_closed_ledger"]={"other":"ledger"}
    elif bad=="history_amendment":f.result["transport_history"]["amendment"]={"other":"amendment"}
    elif bad=="history_ledger":f.result["transport_history"]["recovery_ledger"]={"other":"history"}
    elif bad=="empty_closure":f.result["transport_history"]["provenance_closure"]={}
    elif bad=="non_dict_closure":f.result["transport_history"]["provenance_closure"]=["not-a-proof-dict"]
    elif bad=="history_none":f.result["transport_history"]=None
    else:f.status["training_started"]=True;write(f.statuspath,f.status)
    with pytest.raises(worker.WrapperError):worker.completed_yang(f.root)


@pytest.mark.parametrize("stage",["static","complete"])
def test_V2_pure_verifier_errors_are_propagated_without_takeover(completed_yang153,stage):
    f=completed_yang153
    def blocked(*a):raise ValueError("software_fixture_V2_provenance_rejected")
    if stage=="static":f.verifier.verify_static_authority=blocked
    else:f.verifier.verify_complete_transport_history=blocked
    with pytest.raises(ValueError,match="V2_provenance_rejected"):worker.completed_yang(f.root)


@pytest.fixture
def numeric_pair(isolated):
    import copy
    import numpy as np
    root,core=isolated
    record={"record_id":"fixture-reference","raw_files":[{"path":"data/software-fixture.edf","sha256":"0"*64,"bytes":1}]}
    parentpath=write(root/"numeric/parent.json",{"records":[record]})
    events=[{"trial_uid":f"fixture-uid-{i}","label":i,"original_annotation_index":4+6*i,
        "original_context_samples":[1536+5120*i,2304+5120*i],"adapter_sha256":"old",
        "unit_evidence":{"old_plan":"old"},"preprocessing":{"processed_signal_sha256":str(i)*64,"QC":"valid"}}
        for i in range(2)]
    exclusions=[{"trial_uid":"fixture-excluded-uid","label":0,"adapter_sha256":"old",
        "unit_evidence":{"old_plan":"old"},"exclusion_reason":"software fixture QC exclusion"}]
    native={"trials":copy.deepcopy(events)}
    metadata={"original_trials":3,"valid_trials":2,"excluded_trials":1,
        "original_class_counts":{"left_hand":2,"right_hand":1},"valid_class_counts":{"left_hand":1,"right_hand":1},
        "native_sfreq_hz":512,"native_n_times":10240,"native_record_count":20,
        "native_signal_order":["C3","C4"],"native_signal_units":{"C3":"uV","C4":"uV"},
        "selected_channels":["C3","C4"],"MNE_version":"software_fixture_version",
        "SI_conversion":"software_fixture_SI_once","signal_contract_sha256":"1"*64,
        "native_profile_sha256":"2"*64,"preprocess_file_sha256":"3"*64,
        "stage":"pre_rms_car11_temporal_demeaned","normalization_applied":False}
    adapter={"path":"p3work/source_extensions/dreyer2023.py","sha256":worker.NATIVE_ADAPTER_SHA,"bytes":1}
    plan={"records":[record],"code":{"p3work/source_extensions/dreyer2023.py":adapter}}
    states=[]
    for mode in ["old","new"]:
        if mode=="new":
            folder=root/f"cache/source_extensions_20261019/Dreyer2023/{worker.DREYER_PLAN_SHA}/{record['record_id']}"
            audit=worker.audit_path(root)/record["record_id"]
        else:folder=audit=root/"numeric"/mode
        folder.mkdir(parents=True,exist_ok=True);audit.mkdir(parents=True,exist_ok=True)
        np.save(folder/"X.npy",np.arange(2*11*128,dtype=np.float32).reshape(2,11,128)/1000)
        np.save(folder/"y.npy",np.array([0,1],dtype=np.int64))
        rows=copy.deepcopy(events);excluded=copy.deepcopy(exclusions);n=copy.deepcopy(native)
        for row in [*rows,*excluded,*n["trials"]]:
            row["adapter_sha256"]=mode
            row["unit_evidence"]={"provenance_plan":mode}
        for name,value in [("events",rows),("exclusions",excluded),("metadata",metadata)]:
            write(folder/(name+".json"),value)
        write(audit/"native_metadata.json",n);write(audit/"unit_evidence.json",{"software_fixture":mode})
        paths={name:folder/filename for name,filename in [
            ("X","X.npy"),("y","y.npy"),("events","events.json"),("exclusions","exclusions.json"),("metadata","metadata.json")]}
        paths.update({k:audit/(k+".json")for k in ("native_metadata","unit_evidence")})
        shard={name:core.descriptor(root,path)for name,path in paths.items()}
        artifacts=list(shard.values())
        shard.update(record_id=record["record_id"],raw_files=record["raw_files"],
            adapter_code=copy.deepcopy(adapter)if mode=="new"else{"sha256":"0"*64})
        states.append({"status":"COMPLETE","record_id":record["record_id"],"shard":shard,
            "raw_backup_receipt":{"software_fixture":True},"backup_receipt":{"software_fixture_full_body_GET":True},
            "artifacts":artifacts,"plan_sha256":worker.DREYER_PLAN_SHA if mode=="new"else worker.V4_DREYER_SHA})
    oldpath=write(root/"numeric/old_state.json",states[0])
    qualification={"parent_plan":core.descriptor(root,parentpath),
        "reference_states":{record["record_id"]:core.descriptor(root,oldpath)}}
    core._state_verified=lambda root,state,plan,record:state.get("status")=="COMPLETE"
    return SimpleNamespace(root=root,core=core,record=record,old=states[0],new=states[1],qualification=qualification,
        plan=plan,np=np)


def rebind_numeric_artifact(f,key):
    path=f.root/f.new["shard"][key]["path"];ref=f.core.descriptor(f.root,path)
    f.new["shard"][key]=ref
    f.new["artifacts"]=[ref if r["path"]==ref["path"]else r for r in f.new["artifacts"]]


def test_new_cache_numeric_regression_checks_actual_bits_UID_order_and_QC_and_writes_only_on_request(numeric_pair):
    f=numeric_pair
    report=worker.numeric_regression(f.root,f.plan,f.record,f.new,f.qualification)
    assert report["arrays"]["X"]["bitwise_identical"]is True
    assert report["arrays"]["y"]["bitwise_identical"]is True
    assert report["native_trial_bindings_identical"]is True and report["scientific_metadata_identical"]is True
    path=worker.audit_path(f.root)/f.record["record_id"]/"v4_numeric_regression.json"
    assert not path.exists()
    assert worker.numeric_regression(f.root,f.plan,f.record,f.new,f.qualification,write=True)==report
    assert worker.read_json(path)==report


@pytest.mark.parametrize("bad",["X_bit0","y_value","dtype","shape","UID","order","QC","native_UID","metadata","wrong_adapter"])
def test_changed_single_bit_label_UID_order_QC_or_native_clock_never_passes_regression(numeric_pair,bad):
    f=numeric_pair;key="X"if bad in {"X_bit0","dtype","shape"}else"y"if bad=="y_value"else"events"
    if bad in {"X_bit0","y_value","dtype","shape"}:
        path=f.root/f.new["shard"][key]["path"];array=f.np.load(path,allow_pickle=False)
        if bad=="X_bit0":array.view(f.np.uint32).flat[0]^=1
        elif bad=="y_value":array[0]=1
        elif bad=="dtype":array=array.astype(f.np.float64)
        else:array=array.reshape(1,22,128)
        f.np.save(path,array);rebind_numeric_artifact(f,key)
    elif bad=="wrong_adapter":f.new["shard"]["adapter_code"]["sha256"]="f"*64
    else:
        key="native_metadata"if bad=="native_UID"else"metadata"if bad=="metadata"else"events"
        path=f.root/f.new["shard"][key]["path"];value=worker.read_json(path)
        if bad=="UID":value[0]["trial_uid"]="changed"
        elif bad=="order":value.reverse()
        elif bad=="QC":value[0]["preprocessing"]["processed_signal_sha256"]="f"*64
        elif bad=="native_UID":value["trials"][0]["trial_uid"]="changed"
        else:value["native_sfreq_hz"]=256
        write(path,value);rebind_numeric_artifact(f,key)
    with pytest.raises(worker.WrapperError):worker.numeric_regression(f.root,f.plan,f.record,f.new,f.qualification)


def test_old_references_are_checked_but_never_silently_count_as_128_new_regressions(numeric_pair):
    f=numeric_pair
    worker.numeric_regression(f.root,f.plan,f.record,f.new,f.qualification,write=True)
    with pytest.raises(worker.WrapperError,match="exact128"):
        worker.all_numeric_regressions(f.root,f.plan,[f.new],f.qualification)


@pytest.fixture
def qualifier_raw(isolated):
    root,core=isolated
    path=root/"data/software-fixture.edf";path.parent.mkdir();path.write_bytes(b"software fixture raw bytes only")
    raw=core.descriptor(root,path);record={"record_id":"fixture-raw","raw_files":[raw]}
    context={"qualifier_owned_raw":[{**raw,"created_by":"tools/qualify_dreyer_native_structures.py"}]}
    return root,core,path,record,context


def test_qualified_raw_remains_readable_without_creating_new_ownership_or_deleting(qualifier_raw):
    root,_,path,record,context=qualifier_raw
    before=path.read_bytes();worker.protect_qualified_raw(root,record,context)
    assert path.read_bytes()==before
    assert not worker.audit_path(root).exists()


@pytest.mark.parametrize("bad",["state_ownership","ownership_marker","missing_raw","changed_SHA"])
def test_qualifier_owned_raw_never_adopted_deleted_or_implicitly_refetched(qualifier_raw,bad):
    root,_,path,record,context=qualifier_raw;audit=worker.audit_path(root)/record["record_id"]
    if bad=="state_ownership":write(audit/"state.json",{"owned_raw":[record["raw_files"][0]["path"]]})
    elif bad=="ownership_marker":write(audit/(path.name+".ownership.json"),{"software_fixture":True})
    elif bad=="missing_raw":path.unlink()
    else:path.write_bytes(b"changed software fixture bytes")
    with pytest.raises(ValueError):worker.protect_qualified_raw(root,record,context)


def test_additional_qualification_recovery_provenance_is_descriptor_checked_and_retained(qualified520):
    f=qualified520
    extra=write(f.root/"qualification/recovery_provenance.json",{"software_fixture_only":True})
    refs=[f.core.descriptor(f.root,extra)];f.proof["additional_historical_proofs"]=refs;f.bind()
    assert worker.qualification_evidence(f.root,f.plan)["additional_historical_proofs"]==refs


def test_invalid_additional_qualification_recovery_provenance_never_passes(qualified520):
    f=qualified520
    ref=f.core.descriptor(f.root,f.alias);ref["sha256"]="f"*64
    f.proof["additional_historical_proofs"]=[ref];f.bind()
    with pytest.raises(ValueError):worker.qualification_evidence(f.root,f.plan)


def test_all128_regression_inventory_is_exact_and_embedded_in_final_proof(isolated,monkeypatch):
    root,core=isolated
    records=[{"record_id":f"fixture-{i}"}for i in range(128)]
    states=[{"record_id":r["record_id"]}for r in records]
    qualification={"reference_states":{r["record_id"]:{"software_fixture":True}for r in records}}
    monkeypatch.setattr(worker,"numeric_regression",lambda root,plan,record,state,qualification:{
        "record_id":record["record_id"],"software_fixture_numeric_report":True})
    for record in records:write(worker.audit_path(root)/record["record_id"]/"v4_numeric_regression.json",{
        "record_id":record["record_id"],"software_fixture_numeric_report":True})
    value=worker.all_numeric_regressions(root,{"records":records},states,qualification)
    assert value["verified_records"]==value["reference_records"]==128
    assert value["embedded_in_final_GET_backed_operational_evidence"]is True
    path=worker.audit_path(root)/records[-1]["record_id"]/"v4_numeric_regression.json"
    write(path,{"record_id":records[-1]["record_id"],"software_fixture_numeric_report":False})
    with pytest.raises(worker.WrapperError,match="all128"):
        worker.all_numeric_regressions(root,{"records":records},states,qualification)


@pytest.mark.parametrize("field,value",[
    ("schema_version","paper3-yang-sameplan-no-cache-transport-recovery-v1"),("status","AUTHORIZED_SAMEPLAN_TRANSPORT_AMENDMENT"),
    ("requests_adapter_retries",1),("requests_adapter_retries",False),
    ("session_headers",{"Cache-Control":"max-age=999"}),
    ("max_independent_execute_calls",16),("max_independent_execute_calls",True),
    ("no_tight_retry",False),("original_kernel_is_only_transfer_and_wall_budget_authority",False),
    ("frozen_plan",{}),("wrapper_code",{})])
def test_V2_peer_self_checks_static_authority_after_pure_verifier(isolated,field,value):
    root,core=isolated;no_cache_peer_files(root,core)
    path=root/worker.YANG_AMENDMENT_PATH;amendment=worker.read_json(path);amendment[field]=value;write(path,amendment)
    with pytest.raises(worker.WrapperError,match="authority"):worker.yang_authority(root)
    assert core.peer_verifier.static_calls
    assert worker.yang_identity(root,fingerprint(root))is None


@pytest.mark.parametrize("count,active,terminal",[(0,None,None),(1,1,None),(1,None,None),(8,8,None),(8,None,None),
    (1,None,"SOURCE_EXTENSION_CACHE_COMPLETE"),(8,None,"SOURCE_EXTENSION_CACHE_COMPLETE")])
def test_V2_live_state_accepts_only_bounded_single_call_identity(isolated,count,active,terminal):
    root,core=isolated;state=no_cache_peer_files(root,core)
    state.update(calls_used=count,active_call=active,terminal=terminal,
        call_receipts=[{"software_fixture_closed_call":i}for i in range(count-int(active is not None))])
    write(root/worker.YANG_HISTORY_PATH,state)
    assert worker.yang_identity(root,fingerprint(root))["mode"]=="concurrent_no_cache_wrapper_v2"


@pytest.mark.parametrize("bad",["V1_schema","V1_attempt_state","16_calls","bool_count","active_mismatch",
    "active_zero","active_bool","receipt_count","receipts_not_list","complete_active","failure_terminal"])
def test_V1_disguise_or_inconsistent_V2_active_terminal_count_is_rejected(isolated,bad):
    root,core=isolated;state=no_cache_peer_files(root,core)
    if bad=="V1_schema":state["identity"]["schema_version"]="paper3-yang-sameplan-no-cache-transport-recovery-v1"
    elif bad=="V1_attempt_state":
        identity=state["identity"];state={"identity":identity,"attempts_used_including_inherited":16,
            "active_attempt":16,"terminal":None,"attempt_receipts":[{}for _ in range(14)]}
    elif bad=="16_calls":state.update(calls_used=16,active_call=None,call_receipts=[{}for _ in range(16)])
    elif bad=="bool_count":state["calls_used"]=True
    elif bad=="active_mismatch":state["active_call"]=2
    elif bad=="active_zero":state.update(calls_used=0,active_call=0)
    elif bad=="active_bool":state["active_call"]=True
    elif bad=="receipt_count":state["call_receipts"]=[{}]
    elif bad=="receipts_not_list":state["call_receipts"]={}
    elif bad=="complete_active":state["terminal"]="SOURCE_EXTENSION_CACHE_COMPLETE"
    else:state["terminal"]="STOPPED_NON_TRANSPORT_ERROR_NO_RETRY"
    write(root/worker.YANG_HISTORY_PATH,state)
    assert worker.yang_identity(root,fingerprint(root))is None


@pytest.mark.parametrize("bad",["old_V1_entry","old_V1_code","old_V1_owner"])
def test_old_V1_entry_code_or_persistent_lease_cannot_impersonate_frozen_V2(peer,monkeypatch,bad):
    root,core,state,guard=peer
    if bad=="old_V1_entry":
        state["fingerprint"]["command"][3]="tools/resume_yang_source_transport.py"
        with pytest.raises(worker.WrapperError):guard.check()
    elif bad=="old_V1_code":
        shutil.copyfile(worker.ROOT/"tools/resume_yang_source_transport.py",root/worker.YANG_ENTRY)
        with pytest.raises(worker.WrapperError):guard.check()
    else:
        old=root/"audit/data_plan_20261020/yang_transport_recovery/no_cache_wrapper_v1/owner.lock"
        old.parent.mkdir(parents=True,exist_ok=True);(root/worker.YANG_OWNER_PATH).rename(old)
        state["global"]=[]
        with pytest.raises(worker.WrapperError,match="not_complete"):guard.check()
    assert guard.owned is False


@pytest.mark.parametrize("bad",["wrong_dataset","partial_plan","wrong_manifest","wrong_closed_ledger","authority_policy"])
def test_completed_V2_peer_requires_original_153_plan_and_identical_source_proofs(completed_yang153,bad):
    f=completed_yang153
    if bad in {"wrong_dataset","partial_plan"}:
        if bad=="wrong_dataset":f.plan["dataset"]="Dreyer2023"
        else:f.plan["records"].pop()
        write(f.planpath,f.plan)
    elif bad=="wrong_manifest":
        f.status["manifest"]={"different":"source_cache"};write(f.statuspath,f.status)
    elif bad=="wrong_closed_ledger":f.ledgerpath.write_text("changed dummy closed ledger")
    else:f.amendment["no_tight_retry"]=False;write(f.amendmentpath,f.amendment)
    with pytest.raises(worker.WrapperError):worker.completed_yang(f.root)


def test_unfrozen_V2_SHA_blocks_before_pure_verifier_or_history_read(isolated,monkeypatch):
    root,core=isolated;no_cache_peer_files(root,core)
    monkeypatch.setattr(worker,"YANG_WRAPPER_SHA",None)
    with pytest.raises(worker.WrapperError,match="not_frozen"):worker.yang_authority(root)
    assert core.peer_verifier.static_calls==[]and core.peer_verifier.complete_calls==[]


def test_new_record_provenance_requires_seven_actual_V5_namespace_artifacts(numeric_pair):
    f=numeric_pair
    assert worker.new_record_provenance(f.root,f.plan,f.record,f.new)is None
    assert f.new["raw_backup_receipt"]and f.new["backup_receipt"]
    assert len(f.new["artifacts"])==7


@pytest.mark.parametrize("bad",["status","plan","record","shard_record","raw","adapter_descriptor",
    "missing_artifact","duplicate_artifact","old_X_with_new_proofs","old_X_adopted_as_artifact",
    "cache_alias","native_in_cache","descriptor_not_in_artifacts"])
def test_old_cache_or_mismatched_shard_cannot_be_admitted_with_fresh_complete_artifacts_and_proofs(numeric_pair,bad):
    import copy
    f=numeric_pair
    if bad=="status":f.new["status"]="FAILED"
    elif bad=="plan":f.new["plan_sha256"]=worker.V4_DREYER_SHA
    elif bad=="record":f.new["record_id"]="other"
    elif bad=="shard_record":f.new["shard"]["record_id"]="other"
    elif bad=="raw":f.new["shard"]["raw_files"]=[]
    elif bad=="adapter_descriptor":f.new["shard"]["adapter_code"]["path"]="old_adapter.py"
    elif bad=="missing_artifact":f.new["artifacts"].pop()
    elif bad=="duplicate_artifact":f.new["artifacts"][-1]=f.new["artifacts"][0]
    elif bad in {"old_X_with_new_proofs","old_X_adopted_as_artifact"}:
        original=f.new["shard"]["X"];f.new["shard"]["X"]=copy.deepcopy(f.old["shard"]["X"])
        if bad=="old_X_adopted_as_artifact":
            f.new["artifacts"]=[f.new["shard"]["X"]if r==original else r for r in f.new["artifacts"]]
        assert f.new["raw_backup_receipt"]and f.new["backup_receipt"]and len(f.new["artifacts"])==7
    elif bad in {"cache_alias","native_in_cache"}:
        key="X"if bad=="cache_alias"else"native_metadata";original=f.new["shard"][key]
        alias=f.root/f"cache/alias_{key}.fixture";alias.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f.root/original["path"],alias);ref=f.core.descriptor(f.root,alias)
        f.new["shard"][key]=ref;f.new["artifacts"]=[ref if r==original else r for r in f.new["artifacts"]]
    else:f.new["shard"]["X"]={**f.new["shard"]["X"],"unverified_extra":True}
    with pytest.raises(worker.WrapperError):worker.new_record_provenance(f.root,f.plan,f.record,f.new)
    with pytest.raises(worker.WrapperError):worker.numeric_regression(f.root,f.plan,f.record,f.new,f.qualification)


def test_provider_NoCache_session_headers_zero_retries_and_default_transport_settings(monkeypatch):
    import requests
    original=requests.Session
    monkeypatch.setattr(requests.Session,"request",lambda *a,**k:pytest.fail("session configuration must issue no request"))
    with worker.provider_session() as session:
        assert requests.Session is original
        assert {k:session.headers[k]for k in worker.PROVIDER_HEADERS}=={
            "Cache-Control":"no-cache","Pragma":"no-cache"}
        assert session.trust_env is True and session.verify is True
        for scheme in ("https://","http://"):
            assert session.adapters[scheme].max_retries.total==0
            assert session.adapters[scheme].max_retries.read is False


def test_provider_session_configuration_failure_closes_its_session(monkeypatch):
    import requests
    observed=[]
    class Session:
        headers={}
        def mount(self,*a):raise RuntimeError("software_fixture_mount_failure")
        def close(self):observed.append("closed")
    monkeypatch.setattr(requests,"Session",Session)
    with pytest.raises(RuntimeError,match="software_fixture_mount_failure"):
        worker.provider_session()
    assert observed==["closed"]


def test_execute_passes_provider_factory_session_and_original_raw_spec_unchanged(execution,monkeypatch):
    from copy import deepcopy
    root,core,plan,desc,calls=execution
    raw=plan["records"][0]["raw_files"][0]
    raw.update(sha256="e"*64,download={"kind":"direct","url":"https://provider.invalid/original.edf"})
    expected=deepcopy(plan["records"][0]);observed=[]
    class Session:
        def __enter__(self):observed.append("entered");return self
        def __exit__(self,*a):observed.append("closed")
    session=Session();monkeypatch.setattr(worker,"provider_session",lambda:session)
    original=core.process_record
    def process(root,plan,record,**kwargs):
        assert kwargs["session"]is session and record==expected
        observed.append("original_spec_passed_to_frozen_kernel")
        return original(root,plan,record,**kwargs)
    core.process_record=process
    result=worker.execute(root,declaration_path=desc["path"],max_new_records=1)
    assert result["verified_records"]==1
    assert observed==["entered","original_spec_passed_to_frozen_kernel","closed"]


def test_V2_no_completion_proof_never_reads_original_source_ledger(completed_yang153,monkeypatch):
    f=completed_yang153;f.verifier.completion_result=None
    descriptor=f.core.descriptor
    def guarded(root,path):
        if Path(path).name=="budget_ledger.json":pytest.fail("None completion proof reached original source ledger")
        return descriptor(root,path)
    monkeypatch.setattr(f.core,"descriptor",guarded)
    with pytest.raises(worker.WrapperError,match="not_complete"):worker.completed_yang(f.root)
    assert f.verifier.complete_calls==[(f.root,f.plan)]


@pytest.mark.parametrize("bad",["schema","amendment","plan","zero_calls","nine_calls","bool_count",
    "active","terminal","missing_receipt","receipts_not_list","returned_count"])
def test_completed_V2_rechecks_actual_history_even_after_verifier_descriptor_rebinding(completed_yang153,bad):
    f=completed_yang153
    if bad=="schema":f.history["identity"]["schema_version"]="paper3-yang-sameplan-no-cache-transport-recovery-v1"
    elif bad=="amendment":f.history["identity"]["amendment"]={"wrong":"authority"}
    elif bad=="plan":f.history["identity"]["plan_sha256"]="f"*64
    elif bad=="zero_calls":f.history.update(calls_used=0,call_receipts=[])
    elif bad=="nine_calls":f.history.update(calls_used=9,call_receipts=[{}for _ in range(9)])
    elif bad=="bool_count":f.history["calls_used"]=True
    elif bad=="active":f.history["active_call"]=1
    elif bad=="terminal":f.history["terminal"]=None
    elif bad=="missing_receipt":f.history["call_receipts"]=[]
    elif bad=="receipts_not_list":f.history["call_receipts"]={}
    else:f.result["transport_history"]["calls_used"]=2
    write(f.historypath,f.history)
    f.result["transport_history"]["recovery_ledger"]=f.core.descriptor(f.root,f.historypath)
    with pytest.raises(worker.WrapperError,match="actual_Yang_V2"):worker.completed_yang(f.root)


def test_None_V2_history_cannot_reach_real_history_path(completed_yang153,monkeypatch):
    f=completed_yang153;f.result["transport_history"]=None;descriptor=f.core.descriptor
    def guarded(root,path):
        if Path(path)==root/worker.YANG_HISTORY_PATH:pytest.fail("None V2 history reached actual history path")
        return descriptor(root,path)
    monkeypatch.setattr(f.core,"descriptor",guarded)
    with pytest.raises(worker.WrapperError,match="actual_completed_Yang_V2_history"):worker.completed_yang(f.root)


def test_live_peer_coordination_preserves_original_observation_without_claiming_completion(peer):
    _,_,_,guard=peer;guard.check();value=guard.coordination_evidence()
    assert value==coordination_fixture(guard.initial_peer)
    assert value["last_verified_live_peer"]==guard.peer
    assert value["completed_yang_proof"]is None and value["global_lock_takeover"]is False


def test_peer_guard_persists_entire_completed_proof_after_global_takeover(peer,monkeypatch):
    _,_,state,guard=peer;guard.check();initial=guard.initial_peer;proof=completion_fixture()
    state.update(persistent=[],**{"global":[]})
    monkeypatch.setattr(worker,"completed_yang",lambda root:proof)
    guard.check();value=guard.coordination_evidence()
    assert guard.completion==proof and value==coordination_fixture(initial,takeover=True,completion=proof)
    assert value["completed_yang_proof"]["transport_completion_proof"]==proof["transport_completion_proof"]
    monkeypatch.setattr(worker,"completed_yang",lambda root:pytest.fail("owned guard re-read mutable peer state"))
    guard.check()


def test_live_finish_coordination_verification_does_not_read_completion_or_live_ledger(isolated,monkeypatch):
    root,core=isolated;value=coordination_fixture()
    decl=write(root/"coordination/declaration.json",{"initial_yang_process":value["initial_yang_process"]})
    monkeypatch.setattr(worker,"completed_yang",lambda root:pytest.fail("live finish read completion or live ledger"))
    read_json=worker.read_json
    def read(path):
        if Path(path).name=="budget_ledger.json":pytest.fail("live finish read budget ledger")
        return read_json(path)
    monkeypatch.setattr(worker,"read_json",read)
    assert worker.verify_peer_coordination(root,core.descriptor(root,decl),value)==value


@pytest.mark.parametrize("initial_complete",[False,True])
def test_takeover_coordination_rechecks_full_completion_offline_and_binds_initial_state(isolated,monkeypatch,initial_complete):
    root,core=isolated;proof=completion_fixture();initial=proof if initial_complete else None
    value=coordination_fixture(initial,takeover=True,completion=proof)
    decl=write(root/"coordination/declaration.json",{"initial_yang_process":value["initial_yang_process"]});calls=[]
    monkeypatch.setattr(worker,"completed_yang",lambda root:calls.append(root)or proof)
    assert worker.verify_peer_coordination(root,core.descriptor(root,decl),value)==value
    assert calls==[root]
    assert value["last_verified_live_peer"]is None if initial_complete else value["last_verified_live_peer"]==value["initial_yang_process"]


@pytest.mark.parametrize("bad",["missing_key","extra_key","schema","initial","last_live","takeover_type",
    "no_takeover_with_proof","takeover_no_proof","changed_completion","unknown_initial_mode",
    "already_complete_no_takeover","already_complete_last_live","already_complete_changed_completion"])
def test_forged_initial_live_observation_or_takeover_completion_cannot_enter_final_proof(isolated,monkeypatch,bad):
    root,core=isolated;proof=completion_fixture()
    initial=proof if bad.startswith("already_complete")else None
    if bad=="unknown_initial_mode":initial={"mode":"old_NoCache_V1_disguise"}
    value=coordination_fixture(initial,takeover=bad.startswith("already_complete"),
        completion=proof if bad.startswith("already_complete")else None)
    decl=write(root/"coordination/declaration.json",{"initial_yang_process":value["initial_yang_process"]})
    if bad=="missing_key":value.pop("completed_yang_proof")
    elif bad=="extra_key":value["unverified_extra"]=True
    elif bad=="schema":value["schema_version"]="unreviewed"
    elif bad=="initial":value["initial_yang_process"]={"mode":"different_initial"}
    elif bad=="last_live":value["last_verified_live_peer"]={"mode":"replaced_peer"}
    elif bad=="takeover_type":value["global_lock_takeover"]=1
    elif bad=="no_takeover_with_proof":value["completed_yang_proof"]=proof
    elif bad=="takeover_no_proof":value["global_lock_takeover"]=True
    elif bad=="changed_completion":value.update(global_lock_takeover=True,completed_yang_proof={"mode":"different_completion"})
    elif bad=="already_complete_no_takeover":value["global_lock_takeover"]=False
    elif bad=="already_complete_last_live":value["last_verified_live_peer"]={"mode":"concurrent_no_cache_wrapper_v2"}
    elif bad=="already_complete_changed_completion":value["completed_yang_proof"]={"mode":"different_completion"}
    monkeypatch.setattr(worker,"completed_yang",lambda root:proof)
    with pytest.raises(worker.WrapperError):worker.verify_peer_coordination(root,core.descriptor(root,decl),value)


def test_takeover_proof_uses_full_V2_offline_verifier_and_rejects_later_provenance_change(completed_yang153):
    f=completed_yang153;completion=json.loads(json.dumps(worker.completed_yang(f.root)))
    value=coordination_fixture(takeover=True,completion=completion)
    decl=write(f.root/"coordination/declaration.json",{"initial_yang_process":value["initial_yang_process"]})
    assert worker.verify_peer_coordination(f.root,f.core.descriptor(f.root,decl),value)==value
    assert len(f.verifier.complete_calls)==2
    f.result["complete_V2_provenance_receipts"]["changed_after_frozen_final_proof"]=True
    with pytest.raises(worker.WrapperError,match="takeover_completion"):
        worker.verify_peer_coordination(f.root,f.core.descriptor(f.root,decl),value)
    assert len(f.verifier.complete_calls)==3


def test_execute_passes_peer_coordination_evidence_to_final_GET_backed_operation(execution,monkeypatch):
    root,core,_,desc,_=execution;value=coordination_fixture(takeover=True,completion=completion_fixture());captured=[]
    peer=SimpleNamespace(peer=value["initial_yang_process"],check=lambda:None,close=lambda:None,
        coordination_evidence=lambda:value)
    monkeypatch.setattr(worker,"PeerGuard",lambda *a,**k:peer)
    class SoftwareLedger:
        def __init__(self,*a):pass
        def checkpoint(self,**k):pass
    core.Ledger=SoftwareLedger
    core.process_record=lambda root,plan,record,**k:{"record_id":record["record_id"],"status":"COMPLETE"}
    def finalize(root,plan,states,journal,declaration,evidence):
        captured.append(evidence);assert len(states)==520
        return {"status":"SOURCE_EXTENSION_CACHE_COMPLETE"}
    monkeypatch.setattr(worker,"_finalize",finalize)
    assert worker.execute(root,declaration_path=desc["path"])["status"]=="SOURCE_EXTENSION_CACHE_COMPLETE"
    assert captured==[value]


def test_completed_operation_rejects_forged_coordination_even_with_all_file_GET_descriptors_rebound(completed):
    root,core,plan,decl,status,_=completed
    manifestpath=root/status["manifest"]["path"];manifest=worker.read_json(manifestpath)
    oppath=root/manifest["operational_evidence"]["path"];operation=worker.read_json(oppath)
    operation["yang_peer_coordination"]["last_verified_live_peer"]={"mode":"replaced_peer"};write(oppath,operation)
    opref=core.descriptor(root,oppath);manifest["operational_evidence"]=opref;write(manifestpath,manifest)
    status["manifest"]=core.descriptor(root,manifestpath);status["operational_evidence"]=opref
    receiptpath=root/status["backup_receipt"]["path"];receipt=worker.read_json(receiptpath)
    for row in receipt["objects"]:
        if row["path"]==opref["path"]:row.update(opref)
        elif row["path"]==status["manifest"]["path"]:row.update(status["manifest"])
    write(receiptpath,receipt);status["backup_receipt"]=core.descriptor(root,receiptpath)
    write(worker.audit_path(root)/"coordinator_status.json",status)
    with pytest.raises(worker.WrapperError,match="live_peer_observation"):
        worker.verified_completion(root,plan,decl)


@pytest.mark.parametrize("scientific_change",[False,True])
def test_sealed_live_observation_survives_failed_backup_and_later_verified_takeover_without_rewriting(scientific_change,isolated,monkeypatch):
    root,core=isolated;audit=worker.audit_path(root);audit.mkdir(parents=True)
    initial=coordination_fixture()["initial_yang_process"]
    declaration=write(audit/"parallel_root_declaration.json",{"initial_yang_process":initial})
    declaration_ref=core.descriptor(root,declaration)
    write(audit/"frozen_plan.json",{"software_fixture_frozen_plan":True})
    plan={"records":[{"record_id":str(i)}for i in range(520)]}
    shards=[{"record_id":str(i),"subject_id":"fixture","session_id":str(i),
        "raw_files":[{"bytes":1}],"raw_sha256":"0"*64,"native_metadata":{},
        "raw_backup_receipt":{},"unit_evidence":{}}for i in range(520)]
    core.assemble_manifest=lambda *a:{"shards":shards,"valid_trials":520}
    monkeypatch.setattr(worker,"qualification_evidence",lambda *a:{"software_fixture_qualification":True})
    numerical={"verified_records":128,"scientific_QC_digest":"original-software-fixture-digest"}
    monkeypatch.setattr(worker,"all_numeric_regressions",lambda *a:dict(numerical))
    monkeypatch.setattr(worker,"new_record_provenance",lambda *a:None)
    output=root/f"cache/source_extensions_20261019/Dreyer2023/{worker.DREYER_PLAN_SHA}";output.mkdir(parents=True)
    backups=[]
    def backup(*a,**k):
        backups.append([p.name for p,_ in k["files"]])
        assert len(k["files"])==7
        if len(backups)==1:raise core.PreparationError("software_fixture_final_backup_interrupted")
        return {"software_fixture_full_GET_backup":True}
    core.verified_backup=backup
    closes=[];journal=SimpleNamespace(checkpoint=lambda **k:closes.append(k))
    completion=completion_fixture();offline_checks=[]
    monkeypatch.setattr(worker,"completed_yang",lambda root:offline_checks.append(root)or completion)
    original_observation=coordination_fixture(initial)
    with pytest.raises(core.PreparationError,match="backup_interrupted"):
        worker._finalize(root,plan,[],journal,declaration_ref,original_observation)
    operation_path=audit/"parallel_execution_receipt.json";manifest_path=output/"source_cache.json"
    sealed_operation=operation_path.read_bytes();sealed_manifest=manifest_path.read_bytes()
    assert worker.read_json(operation_path)["yang_peer_coordination"]==original_observation
    assert offline_checks==[]and closes==[]
    # The next invocation independently verifies actual takeover; it must retain
    # the earlier immutable observation when retrying the same final backup.
    global_lock=root/worker.AUDIT_ROOT/".source-extension-singleton.lock"
    global_lock.parent.mkdir(parents=True,exist_ok=True);global_lock.touch()
    guard=worker.PeerGuard(root,expected_peer=initial)
    try:
        guard.check();current=guard.coordination_evidence()
        assert current["global_lock_takeover"]is True and current["completed_yang_proof"]==completion
        assert offline_checks==[root]
        if scientific_change:
            numerical["scientific_QC_digest"]="changed-software-fixture-digest"
            with pytest.raises(core.PreparationError,match="immutable"):
                worker._finalize(root,plan,[],journal,declaration_ref,current)
            assert len(backups)==1 and closes==[]
        else:
            result=worker._finalize(root,plan,[],journal,declaration_ref,current)
            assert result["status"]=="SOURCE_EXTENSION_CACHE_COMPLETE"
            assert len(backups)==2 and backups[0]==backups[1]and closes==[{"close":True}]
        assert operation_path.read_bytes()==sealed_operation
        assert manifest_path.read_bytes()==sealed_manifest
        assert offline_checks==[root]
    finally:guard.close()
