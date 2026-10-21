"""Offline synthetic coordinator tests: actual IO policies, no external requests."""
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from p3work import source_extension_io as storage
from p3work.source_extension_coordinator import Ledger, PreparationError, descriptor, digest
from p3work.source_extensions import ma2022


SPEC = importlib.util.spec_from_file_location("ma_development_tool",Path(__file__).parents[1]/"tools/prepare_ma2022_development.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def _field(value,width):
    data=str(value).encode("ascii")
    assert len(data)<=width
    return data.ljust(width,b" ")


def native_bytes(identifier,*,headerless=False):
    names="Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 PO7 PO3 POz PO4 PO8 O1 O2".split()
    ns,records=32,8
    fixed=b"".join(_field(v,w) for v,w in [("0",8),("SYNTHETIC_"+identifier,80),("SOFTWARE_FIXTURE",80),
        ("01.01.26",8),("00.00.00",8),(256*(ns+1),8),("",44),(records,8),("1",8),(ns,4)])
    arrays=[(names,16),([""]*ns,80),(["uV"]*ns,8),(["-100"]*ns,8),(["100"]*ns,8),
        (["-32768"]*ns,8),(["32767"]*ns,8),(["NaN"]*ns,80),([250]*ns,8),([""]*ns,32)]
    header=b"".join(_field(v,w)for values,w in arrays for v in values)
    data=(9000*np.sin(2*np.pi*10*np.arange(250)[None,:]/250+np.arange(32)[:,None]*.13)).astype("<i2")
    edf=fixed+header+data.tobytes()*records
    events=(("" if headerless else "onset\tduration\ttrial_type\tresponse_time\tsample\tvalue\n")+
            "1\t1000\tleft\tn/a\t1\t1\n1001\t1000\tright\tn/a\t1001\t2\n").encode()
    return edf,events


class Response:
    def __init__(self,body,status=200,headers=None):
        self.body=body;self.status_code=status
        self.headers=headers or {"Content-Length":str(len(body))}
        self.closed=False
    def iter_content(self,size):
        for offset in range(0,len(self.body),size):yield self.body[offset:offset+size]
    def close(self):self.closed=True


class Provider:
    def __init__(self,bodies):self.bodies=bodies;self.calls=[];self.bad=False;self.forbid=False
    def get(self,url,**kwargs):
        if self.forbid:pytest.fail("unexpected provider request")
        self.calls.append(url)
        body=self.bodies[url]
        if self.bad:body=body[:-1]+bytes([body[-1]^1])
        return Response(body)


class R2:
    def __init__(self):self.objects={};self.puts=[];self.gets=[]
    def put_object(self,**kwargs):
        key=kwargs["Key"];body=kwargs["Body"].read()
        assert kwargs["IfNoneMatch"]=="*" and kwargs["ContentLength"]==len(body)
        assert key not in self.objects
        self.objects[key]=body;self.puts.append(key)
        return {"ResponseMetadata":{"HTTPStatusCode":200}}
    def get_object(self,**kwargs):
        key=kwargs["Key"];body=self.objects[key];self.gets.append(key)
        return {"Body":io.BytesIO(body),"ContentLength":len(body),"ResponseMetadata":{"HTTPStatusCode":200}}


@pytest.fixture
def scenario(tmp_path,monkeypatch):
    root=tmp_path/"root";root.mkdir()
    real=Path(__file__).parents[1]
    for name in [tool.CREATOR,"p3work/source_extension_io.py","p3work/source_extension_coordinator.py",
                 "tools/r2_backup.py","tools/r2_restore.py","p3work/source_extensions/ma2022.py"]:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(real/name,target)
    files,bodies=[],{}
    for subject in range(1,26):
        for day in range(1,6):
            identifier=f"{subject:03d}_{day:02d}"
            for kind,body in zip(["eeg.edf","events.tsv"],native_bytes(identifier,headerless=day==4)):
                number=100000+len(files)
                url=f"https://ndownloader.figshare.com/files/{number}"
                md5=hashlib.md5(body).hexdigest()
                files.append({"id":number,"name":f"sub-{subject:03d}_ses-{day:02d}_task_motorimagery_{kind}",
                    "size":len(body),"download_url":url,"computed_md5":md5,"supplied_md5":md5,"is_link_only":False})
                bodies[url]=body
    catalog={"version":1,"license":{"name":"CC BY 4.0"},"download_disabled":False,
        "is_confidential":False,"is_embargoed":False,"files":files}
    p=root/"audit/source_extensions_20261019/Ma2022/catalog.json"
    p.parent.mkdir(parents=True);p.write_text(json.dumps(catalog))
    monkeypatch.setattr(tool,"CATALOG_SHA256",hashlib.sha256(p.read_bytes()).hexdigest())
    monkeypatch.setattr(tool,"RAW_BYTES",sum(x["size"]for x in files))
    plan=tool.build_plan(root,p,disk_reserve_bytes=1,max_wall_seconds=600)
    provider,remote=Provider(bodies),R2()
    def backup(**kwargs):return storage.run_backup(client=remote,bucket="SYNTHETIC_TEST_BUCKET",**kwargs)
    def restore(**kwargs):return storage.run_restore(client=remote,bucket="SYNTHETIC_TEST_BUCKET",**kwargs)
    return root,p,catalog,plan,provider,remote,backup,restore


def paths(root,plan,record):
    audit=root/f"audit/source_extensions_20261019/Ma2022/{plan['plan_sha256']}"
    return audit,audit/record["record_id"],audit/record["record_id"]/"state.json"


def execute_one(scenario,**overrides):
    root,_,_,plan,provider,_,backup,restore=scenario
    return tool.prepare(root,plan,execute=True,max_new_records=1,session=provider,
        backup_fn=overrides.get("backup_fn",backup),restore_fn=restore,
        create_cache=overrides.get("create_cache"))


def resume_first(scenario):
    root,_,_,plan,provider,_,backup,restore=scenario
    audit,_,_=paths(root,plan,plan["records"][0]);ledger=Ledger(audit/"budget_ledger.json",plan)
    try:return tool.process_record(root,plan,plan["records"][0],ledger=ledger,session=provider,
        backup_fn=backup,restore_fn=restore,create_cache=ma2022.create_development_cache)
    finally:ledger.checkpoint(close=True)


def test_complete_plan_is_exact250_native_files_and_keeps_development_gates(scenario):
    _,_,_,plan,_,_,_,_=scenario
    assert len(plan["records"])==125 and sum(len(x["raw_files"])for x in plan["records"])==250
    assert plan["role"]=="independent_development_unassigned"
    assert plan["training_ready"] is False and plan["source3_training_pool_member"] is False
    assert plan["native_profile"]["original_B20_prefix_recoverable"] is False
    assert all(x["path"].startswith("data/source_extensions_20261019/Ma2022/") for r in plan["records"]for x in r["raw_files"])
    assert all(not x["name"].endswith((".zip",".mat",".npz"))for r in plan["records"]for x in r["raw_files"])
    assert all("/development_car3_native_v2/" in r["output"]for r in plan["records"])


@pytest.mark.parametrize("bad",["version","license","missing","duplicate","MD5","URL","oversize"])
def test_bad_public_catalog_rejected(scenario,monkeypatch,bad):
    root,p,catalog,_,_,_,_,_=scenario
    if bad=="version":catalog["version"]=2
    elif bad=="license":catalog["license"]["name"]="Restricted"
    elif bad=="missing":catalog["files"].pop()
    elif bad=="duplicate":catalog["files"].append(catalog["files"][0])
    elif bad=="MD5":catalog["files"][0]["computed_md5"]="0"*32
    elif bad=="URL":catalog["files"][0]["download_url"]="https://example.com/private"
    else:catalog["files"][0]["size"]=64*1024**2
    p.write_text(json.dumps(catalog));monkeypatch.setattr(tool,"CATALOG_SHA256",hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(PreparationError):tool.build_plan(root,p)


def test_default_dry_run_never_calls_network_R2_or_adapter(scenario):
    root,_,_,plan,provider,_,_,_=scenario;provider.forbid=True
    def forbidden(**kwargs):pytest.fail("dry run called a mutation/IO")
    result=tool.prepare(root,plan,session=provider,backup_fn=forbidden,restore_fn=forbidden,create_cache=forbidden)
    assert result["status"]=="DRY_RUN_NO_NETWORK_R2_OR_FIT" and not provider.calls
    assert not(root/"cache").exists()


def test_one_record_requires_real_profiled_raw_and_cache_full_GET_proofs(scenario):
    root,_,_,plan,provider,remote,_,_=scenario
    result=execute_one(scenario);record=plan["records"][0]
    _,_,p=paths(root,plan,record);state=json.loads(p.read_bytes())
    assert result["verified_records"]==1 and state["status"]=="COMPLETE"
    assert len(provider.calls)==2 and len(remote.puts)==len(remote.gets)==9
    assert all(not(root/x["path"]).exists()for x in record["raw_files"])
    assert state["retained_trials"]==2
    raw=json.loads((root/state["raw_backup_receipt"]["path"]).read_bytes())
    assert len(raw["objects"])==2 and all(x["role"]=="source_development_raw"for x in raw["objects"])
    assert raw["source_extension_profile"]==storage.profile()
    for row in raw["objects"]:assert row["get"]["complete_body_read"] is True and row["get"]["sha256"]==row["sha256"]
    provider.forbid=True;before=len(remote.gets)
    resumed=resume_first(scenario)
    assert resumed["status"]=="COMPLETE" and len(remote.gets)==before


def test_all125_final_closure_and_resume_require_every_profiled_proof(scenario):
    root,_,_,plan,provider,remote,backup,restore=scenario
    result=tool.prepare(root,plan,execute=True,session=provider,backup_fn=backup,restore_fn=restore)
    assert result["status"]=="MA2022_DEVELOPMENT_CACHE_COMPLETE" and result["verified_records"]==125
    manifest=json.loads((root/result["manifest"]["path"]).read_bytes())
    inventory=json.loads((root/manifest["native_inventory"]["path"]).read_bytes())
    assert manifest["retained_trials"]==250 and manifest["sessions"]==125
    assert len(manifest["shards"])==len(inventory["records"])==125
    assert manifest["source_training_allowed"] is False and manifest["original_B20_prefix_recoverable"] is False
    assert len(provider.calls)==250 and len(remote.puts)==len(remote.gets)==125*9+3
    provider.forbid=True;before=len(remote.gets)
    again=tool.prepare(root,plan,execute=True,session=provider,backup_fn=backup,restore_fn=restore)
    assert again==result and len(remote.gets)==before


def test_headerless_first_actual_trial_is_retained_with_identical_numeric_preprocessing(tmp_path):
    edf,headed=native_bytes("HEADERLESS_TEST")
    p=tmp_path/"native.edf";p.write_bytes(edf)
    q=tmp_path/"headered.tsv";q.write_bytes(headed)
    r=tmp_path/"headerless.tsv";r.write_bytes(headed.split(b"\n",1)[1])
    sha=hashlib.sha256(edf).hexdigest()
    native=ma2022.audit_pair(p,r,edf_sha256=sha,events_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
    assert native["native_event_table_layout"]["header_present"] is False
    assert native["native_event_table_layout"]["data_rows"]==2
    assert [e["native_event_data_line_1based"]for e in native["events"]]==[1,2]
    assert [e["analysis_anchor_sample"]for e in native["events"]]==[0,1000]
    assert [e["label"]for e in native["events"]]==[0,1]
    for events,name in [(q,"headed_cache"),(r,"headerless_cache")]:
        ma2022.create_development_cache(p,events,tmp_path/name,edf_sha256=sha,
            events_sha256=hashlib.sha256(events.read_bytes()).hexdigest())
    for name in ["X.npy","y.npy"]:
        assert np.array_equal(np.load(tmp_path/"headed_cache"/name),np.load(tmp_path/"headerless_cache"/name))


@pytest.mark.parametrize("bad",["first_clock","sample_clock","wrong_column_order","five_columns","wrong_hand","extra_trial"])
def test_headerless_wrong_columns_labels_or_clock_never_guessed(tmp_path,bad):
    edf,body=native_bytes("BAD_HEADERLESS",headerless=True)
    rows=[line.split("\t")for line in body.decode().splitlines()]
    if bad=="first_clock":rows[0][0]="1001"
    elif bad=="sample_clock":rows[0][4]="0"
    elif bad=="wrong_column_order":rows[0][2],rows[0][3]=rows[0][3],rows[0][2]
    elif bad=="five_columns":rows[0].pop()
    elif bad=="wrong_hand":rows[0][2]="right"
    else:rows.append(rows[-1])
    p=tmp_path/"native.edf";p.write_bytes(edf)
    q=tmp_path/"native.tsv";q.write_text("\n".join("\t".join(x)for x in rows)+"\n")
    with pytest.raises(ma2022.Ma2022Error):
        ma2022.audit_pair(p,q,edf_sha256=hashlib.sha256(edf).hexdigest(),
            events_sha256=hashlib.sha256(q.read_bytes()).hexdigest())


def test_old_v1_cache_namespace_is_preserved_and_never_reused(scenario):
    root,_,_,plan,_,_,_,_=scenario
    old=root/"cache/source_extensions_20261019/Ma2022/development_car3/sub-001/ses-01"
    old.mkdir(parents=True);sentinel=old/"manifest.json";sentinel.write_text("OLD_V1_PROOF_MUST_REMAIN")
    execute_one(scenario)
    assert sentinel.read_text()=="OLD_V1_PROOF_MUST_REMAIN"
    assert (root/plan["records"][0]["output"]/"manifest.json").is_file()


def test_first_preexisting_pilot_raw_and_cache_stay_unowned_and_unmodified(scenario):
    root,_,_,plan,provider,_,_,_=scenario;record=plan["records"][0]
    files=[]
    for item in record["raw_files"]:
        p=root/item["path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(provider.bodies[item["url"]]);files.append(p)
    before=[hashlib.sha256(x.read_bytes()).hexdigest()for x in files]
    output=root/record["output"]
    ma2022.create_development_cache(*files,output,edf_sha256=before[0],events_sha256=before[1])
    cache_before=hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()
    provider.forbid=True;execute_one(scenario)
    _,_,p=paths(root,plan,record);state=json.loads(p.read_bytes())
    assert state["owned_raw"]==[] and all(p.exists()for p in files)
    assert [hashlib.sha256(x.read_bytes()).hexdigest()for x in files]==before
    assert hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()==cache_before


def test_corrupt_COMPLETE_cache_is_detected_without_redownload(scenario):
    root,_,_,plan,provider,remote,_,_=scenario;execute_one(scenario)
    p=root/plan["records"][0]["output"]/"X.npy"
    b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
    provider.forbid=True;before=len(remote.gets)
    with pytest.raises(PreparationError):resume_first(scenario)
    assert len(remote.gets)==before


def test_missing_receipt_member_rejected_even_after_receipt_descriptor_rehashed(scenario):
    root,_,_,plan,provider,_,_,_=scenario;execute_one(scenario)
    _,_,p=paths(root,plan,plan["records"][0]);state=json.loads(p.read_bytes())
    receipt=root/state["cache_backup_receipt"]["path"];body=json.loads(receipt.read_bytes());body["objects"].pop();receipt.write_text(json.dumps(body))
    state["cache_backup_receipt"]=descriptor(root,receipt);p.write_text(json.dumps(state));provider.forbid=True
    with pytest.raises(PreparationError):resume_first(scenario)


def test_failed_GET_reservation_is_conserved_across_resume(scenario):
    root,_,_,plan,provider,_,_,_=scenario;provider.bad=True
    with pytest.raises(PreparationError):execute_one(scenario)
    audit,_,p=paths(root,plan,plan["records"][0]);ledger=json.loads((audit/"budget_ledger.json").read_bytes())
    expected=plan["records"][0]["raw_files"][0]["bytes"]+tool.REDIRECT_ALLOWANCE+tool.CHUNK
    assert ledger["charged"]["provider_body_bytes"]==expected
    assert ledger["history"][-1]["status"]=="FAILURE_FULL_RESERVATION_CONSUMED"
    assert json.loads(p.read_bytes())["status"]=="FAILED_CLOSED"
    provider.bad=False;resumed=resume_first(scenario)
    final=json.loads((audit/"budget_ledger.json").read_bytes())
    assert resumed["status"]=="COMPLETE"
    assert final["charged"]["provider_body_bytes"]==expected+sum(x["bytes"]for x in plan["records"][0]["raw_files"])


def test_failed_raw_R2_never_removes_owned_native_pair(scenario):
    root,_,_,plan,provider,_,_,_=scenario
    def failed(**kwargs):return {"status":"failed","execute":True}
    with pytest.raises(PreparationError):execute_one(scenario,backup_fn=failed)
    record=plan["records"][0]
    assert all((root/x["path"]).exists()for x in record["raw_files"])
    provider.forbid=True;assert resume_first(scenario)["status"]=="COMPLETE"


def test_failed_cache_can_restore_verified_raw_without_provider_redownload(scenario):
    root,_,_,plan,provider,remote,_,_=scenario
    def failed(*args,**kwargs):raise ValueError("synthetic interruption before cache")
    with pytest.raises(PreparationError):execute_one(scenario,create_cache=failed)
    for item in plan["records"][0]["raw_files"]:(root/item["path"]).unlink()
    provider.forbid=True;before=len(remote.gets)
    assert resume_first(scenario)["status"]=="COMPLETE"
    assert len(remote.gets)>before  # actual mocked full R2 GETs, no provider GET


def test_singleton_prevents_second_executor_before_any_request(scenario):
    root,_,_,plan,provider,_,_,_=scenario;provider.forbid=True
    p=root/"audit/source_extensions_20261019/Ma2022/.development-singleton.lock";p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a+")as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(PreparationError,match="already_running"):execute_one(scenario)


def test_changed_bound_adapter_or_budget_fails_closed(scenario):
    root,_,_,plan,provider,_,_,_=scenario;provider.forbid=True
    changed=deepcopy(plan);changed["budgets"]["wall_seconds"]+=1
    with pytest.raises(PreparationError):tool.prepare(root,changed,execute=True,session=provider)
    p=root/"p3work/source_extensions/ma2022.py";p.write_bytes(p.read_bytes()+b"\n# synthetic modification\n")
    with pytest.raises(PreparationError):tool.prepare(root,plan,execute=True,session=provider)


def test_symlink_raw_is_preserved_and_never_adopted(scenario,tmp_path):
    root,_,_,plan,_,_,_,_=scenario;item=plan["records"][0]["raw_files"][0]
    target=tmp_path/"unrelated";target.write_bytes(b"preserve me")
    p=root/item["path"];p.parent.mkdir(parents=True);p.symlink_to(target)
    with pytest.raises(PreparationError):execute_one(scenario)
    assert target.read_bytes()==b"preserve me" and p.is_symlink()
