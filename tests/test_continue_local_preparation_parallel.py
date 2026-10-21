"""SOFTWARE_TEST_DOUBLE: threads/local metadata only, no EEG/network/GPU."""
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import weakref

import pytest

from tools import continue_local_preparation_parallel as parallel
from p3work.recovery_budget import WallBudgetBlocked


def write(path, value):
    parallel.base.atomic_json(path, value)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    raw = b"SOFTWARE_TEST_DOUBLE"
    entries = [{"id": i, "name": f"S1_Session_{i}.mat", "provider_subject": "S1", "session_ordinal": i,
                "size": len(raw), "computed_md5": hashlib.md5(raw).hexdigest(),
                "download_url": f"https://ndownloader.figshare.com/files/{i}"} for i in range(1, 7)]
    selection = {"scope": "SOFTWARE_TEST_DOUBLE_NOT_SCIENTIFIC", "stage": "meta-duration", "plan_sha256": "a" * 64,
        "files": entries, "max_jobs": len(entries), "max_download_bytes": len(entries) * len(raw), "split_sha256": "d" * 64}
    amendment = {"kind": "concurrency_operational_only", "plan_sha256": "c" * 64}
    qualification = {"plan_sha256": "b" * 64, "files": []}
    batch = tmp_path / "audit/stieger_w1/batches" / selection["plan_sha256"]
    batch.mkdir(parents=True)
    write(batch / "batch_progress.json", {"plan_sha256": selection["plan_sha256"],
        "active_wall_elapsed_s": 123., "status": "INTERRUPTED_OR_FAILED_CLOSED"})
    monkeypatch.setattr(parallel.base, "build_plan", lambda *a, **k: qualification)

    def complete(entry, plan=None, *, status="COMPLETE"):
        plan = selection if plan is None else plan
        obj = tmp_path / "audit/stieger_w1/batches" / plan["plan_sha256"] / "objects" / str(entry["id"])
        obj.mkdir(parents=True, exist_ok=True)
        write(obj / "adapter_audit.json", {"scope": "SOFTWARE_TEST_DOUBLE", "source_file_sha256": hashlib.sha256(raw).hexdigest()})
        write(obj / "trial_metadata.json", [{"scope": "SOFTWARE_TEST_DOUBLE"}])
        raw_path = f"data/stieger2021/w1-batch-temp/{entry['id']}/{entry['name']}"
        rows = [{"path": raw_path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                 "role": "source_development_raw", "dataset": "Stieger2021"}]
        for name in ("adapter_audit.json", "trial_metadata.json"):
            path = obj / name
            rows.append({"path": path.relative_to(tmp_path).as_posix(), "sha256": parallel.base.sha256_file(path)[0],
                         "bytes": path.stat().st_size, "role": "audit", "dataset": "Stieger2021"})
        write(obj / "backup_manifest.json", {"schema_version": 1, "files": rows})
        write(obj / "backup_receipt.json", {"status": "verified", "execute": True, "objects": [
            {**row, "status": "verified", "get": {"status": "verified", "http_status": 200, "complete_body_read": True,
                "sha256": row["sha256"], "bytes_read": row["bytes"], "declared_content_length": row["bytes"]}} for row in rows]})
        names = ("adapter_audit.json", "trial_metadata.json", "backup_manifest.json", "backup_receipt.json")
        state = {"file_id": entry["id"], "name": entry["name"], "plan_sha256": plan["plan_sha256"], "status": status,
            "schema_blocked": status != "COMPLETE", "raw_path": raw_path,
            "raw_integrity": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "md5": entry["computed_md5"]},
            "backup_receipt": (obj / "backup_receipt.json").relative_to(tmp_path).as_posix(),
            "retained_metadata_sha256": {(obj / name).relative_to(tmp_path).as_posix():
                parallel.base.sha256_file(obj / name)[0] for name in names}}
        write(obj / "state.json", state)
        return state

    class Client:
        closed = False
        def close(self):
            assert threading.current_thread() is threading.main_thread()
            self.closed = True
    client = Client()
    backup_threads = []
    def factory():
        assert threading.current_thread() is threading.main_thread()
        return (lambda **kwargs: backup_threads.append(threading.get_ident())), client

    def run(process=None, **kwargs):
        return parallel._run_locked(root=tmp_path, selection=selection, amendment=amendment,
            qualification=qualification, inventory=tmp_path / "inventory.json", batch_dir=batch,
            progress_path=tmp_path / "status/local_preparation.json", progress={}, backup_factory=factory,
            process_fn=process or (lambda entry, **kw: complete(entry)), preflight=lambda *a: {}, **kwargs)
    return SimpleNamespace(root=tmp_path, selection=selection, amendment=amendment, qualification=qualification,
        entries=entries, batch=batch, complete=complete, run=run, client=client, factory=factory)


def test_two_complete_file_lifecycles_are_bounded_and_no_entry_is_scheduled_twice(fixture):
    mutex = threading.Lock(); barrier = threading.Barrier(2)
    counts = {"active": 0, "peak": 0}; observed = []
    def process(entry, **kwargs):
        with mutex:
            counts["active"] += 1; counts["peak"] = max(counts["peak"], counts["active"]); observed.append(entry["id"])
        if entry["id"] <= 2:
            barrier.wait(timeout=3)
        time.sleep(.01)
        state = fixture.complete(entry)
        with mutex:
            counts["active"] -= 1
        return state
    result = fixture.run(process)
    assert result["status"] == "ALL_DURATION_OBJECTS_VERIFIED"
    assert counts["peak"] == 2 and counts["active"] == 0
    assert sorted(observed) == [1, 2, 3, 4, 5, 6] and len(set(observed)) == len(observed)
    progress = parallel.serial.read_json(fixture.batch / "batch_progress.json")
    assert progress["active_wall_elapsed_s"] >= 123 and progress["max_wall_seconds"] == 64800
    assert progress["plan_sha256"] == fixture.selection["plan_sha256"]
    assert fixture.client.closed


def test_completed_and_qualification_reuse_skip_processing_with_original_provenance(fixture):
    fixture.complete(fixture.entries[0])
    fixture.qualification["files"] = [fixture.entries[1]]
    fixture.complete(fixture.entries[1], fixture.qualification)
    called = []
    def process(entry, **kwargs):
        called.append(entry["id"]); return fixture.complete(entry)
    fixture.run(process)
    assert sorted(called) == [3, 4, 5, 6]
    copied = parallel.serial.object_state(fixture.entries[1], fixture.batch)
    assert copied["reused_qualification_object"] is True
    assert copied["plan_sha256"] == fixture.selection["plan_sha256"]
    assert parallel.verify_object(fixture.entries[1], copied, root=fixture.root,
                                 plan=fixture.selection, qualification=fixture.qualification)


def test_first_failure_stops_dispatch_and_drains_existing_second_object(fixture):
    barrier = threading.Barrier(2); called = []; finished = []
    def process(entry, **kwargs):
        called.append(entry["id"]); barrier.wait(timeout=3)
        if entry["id"] == 1:
            raise parallel.base.BatchError("network_failure_fixed_no_details")
        time.sleep(.05)
        state = fixture.complete(entry); finished.append(entry["id"]); return state
    with pytest.raises(parallel.base.BatchError, match="network_failure_fixed"):
        fixture.run(process)
    assert sorted(called) == [1, 2] and finished == [2]
    assert parallel.serial.object_state(fixture.entries[1], fixture.batch)["status"] == "COMPLETE"
    assert parallel.serial.read_json(fixture.batch / "batch_progress.json")["status"] == "INTERRUPTED_OR_FAILED_CLOSED"
    assert fixture.client.closed


def test_unknown_schema_blocks_before_any_new_client_or_file_dispatch(fixture, monkeypatch):
    fixture.complete(fixture.entries[0], status="SCHEMA_BLOCKED")
    monkeypatch.setattr(parallel, "_shared_backup", lambda: pytest.fail("unknown schema contacted R2"))
    with pytest.raises(parallel.PreparationError, match="unknown_native_schema"):
        fixture.run(lambda *a, **k: pytest.fail("unknown schema dispatched"))


def test_three_object_GET_proof_cannot_be_replaced_by_raw_only_success(fixture):
    state = fixture.complete(fixture.entries[0])
    path = fixture.root / state["backup_receipt"]
    receipt = parallel.serial.read_json(path)
    receipt["objects"][1]["get"]["complete_body_read"] = False
    write(path, receipt)
    state["retained_metadata_sha256"][state["backup_receipt"]] = parallel.base.sha256_file(path)[0]
    write(path.with_name("state.json"), state)
    with pytest.raises(Exception, match="three_full_get"):
        parallel.verify_object(fixture.entries[0], state, root=fixture.root,
                               plan=fixture.selection, qualification=fixture.qualification)


@pytest.mark.parametrize("mode", ["unclosed_serial", "missing_parallel_journal", "different_selection", "missing_progress"])
def test_resume_never_discards_unknown_or_missing_wall_evidence(fixture, mode):
    path = fixture.batch / "batch_progress.json"; prior = parallel.serial.read_json(path)
    if mode == "unclosed_serial": prior["status"] = "RUNNING_SERIAL_CPU"
    elif mode == "missing_parallel_journal": prior["parallel_operational_plan_sha256"] = fixture.amendment["plan_sha256"]
    elif mode == "different_selection": prior["plan_sha256"] = "e" * 64
    if mode == "missing_progress": path.unlink()
    else: write(path, prior)
    with pytest.raises(parallel.PreparationError):
        fixture.run(lambda *a, **k: pytest.fail("uncertain history dispatched"))


def test_wall_budget_cap_is_not_reset_or_extended(fixture):
    path = fixture.batch / "batch_progress.json"; prior = parallel.serial.read_json(path)
    prior["active_wall_elapsed_s"] = 64800.; write(path, prior)
    with pytest.raises(Exception, match="exhausted|budget"):
        fixture.run(lambda *a, **k: pytest.fail("exhausted budget dispatched"))


def test_completed_resume_preserves_wall_high_water_without_reprocessing(fixture):
    fixture.run()
    first = parallel.serial.read_json(fixture.batch / "batch_progress.json")["active_wall_elapsed_s"]
    result = fixture.run(lambda *a, **k: pytest.fail("completed resume dispatched"))
    second = parallel.serial.read_json(fixture.batch / "batch_progress.json")["active_wall_elapsed_s"]
    assert result["verified_complete_files"] == 6 and second >= first >= 123


def test_native_parser_serializes_and_releases_large_sessions_before_unlock():
    refs = []; lock = threading.Lock(); active = 0; peak = 0
    class Native:
        def __init__(self): self.eeg = bytearray(65536); self.rows = [{"metadata": "SOFTWARE_TEST_DOUBLE"}]
        def schema_audit(self): return {"status": "SOFTWARE_TEST_DOUBLE"}
    class Guard:
        def __enter__(self): lock.acquire()
        def __exit__(self, *args):
            assert refs[-1]() is None
            lock.release()
    def native_load(*args, **kwargs):
        nonlocal active, peak
        active += 1; peak = max(peak, active)
        value = Native(); refs.append(weakref.ref(value)); time.sleep(.01); active -= 1
        return value
    load = parallel.serialized_native_loader(Guard(), native_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        proxies = list(pool.map(lambda _: load(Path("fixture"), expected_sha256="f" * 64), range(4)))
    assert peak == 1 and all(ref() is None for ref in refs)
    assert all(type(proxy) is parallel.TinyMetadataSession for proxy in proxies)
    assert all(proxy.schema_audit() == {"status": "SOFTWARE_TEST_DOUBLE"} for proxy in proxies)


@pytest.mark.parametrize("failure_type", [ValueError, MemoryError])
def test_parser_exception_tracebacks_release_native_arrays_before_unlock(failure_type):
    refs = []
    class Big: pass
    class Guard:
        def __enter__(self): pass
        def __exit__(self, *args): assert refs[0]() is None
    def native_load(*args, **kwargs):
        big = Big(); big.eeg = bytearray(65536); refs.append(weakref.ref(big))
        raise failure_type(big)
    load = parallel.serialized_native_loader(Guard(), native_load)
    with pytest.raises(failure_type):
        load(Path("fixture"), expected_sha256="f" * 64)


def test_resource_preflight_uses_clean_cgroup_cache_not_host_or_shmem_as_free(monkeypatch, tmp_path):
    gb = 1024**3
    def observation():
        return {"cgroup_memory_limit_bytes": 16*gb, "cgroup_memory_current_bytes": 13*gb,
            "cgroup_memory_stat": {"file": 10*gb, "shmem": 2*gb, "file_dirty": 0, "file_writeback": 0,
                "unevictable": 0, "active_file": 7*gb, "inactive_file": gb},
            "host_available_bytes": 30*gb, "disk_free_bytes": 20*gb}
    monkeypatch.setattr(parallel, "_resource_observation", lambda root: observation())
    result = parallel.resource_preflight(tmp_path, 935203397)
    assert result["effective_available_memory_bytes"] == 11*gb and result["effective_memory_cap_bytes"] == 16*gb
    low = observation(); low["cgroup_memory_stat"]["shmem"] = 10*gb
    monkeypatch.setattr(parallel, "_resource_observation", lambda root: low)
    with pytest.raises(parallel.PreparationError, match="cgroup_headroom"):
        parallel.resource_preflight(tmp_path, 935203397)
    low = observation(); low["disk_free_bytes"] = 1
    monkeypatch.setattr(parallel, "_resource_observation", lambda root: low)
    with pytest.raises(parallel.PreparationError, match="disk_headroom"):
        parallel.resource_preflight(tmp_path, 935203397)


def test_shared_SDK_client_and_fixed_bucket_are_initialized_once_on_main_thread(monkeypatch):
    client = object(); created = []; calls = []
    def create():
        assert threading.current_thread() is threading.main_thread()
        created.append(True); return client
    monkeypatch.setattr(parallel.r2, "create_client", create)
    monkeypatch.setenv("R2_BUCKET", "SOFTWARE_TEST_DOUBLE_BUCKET")
    monkeypatch.setattr(parallel.r2, "run_backup", lambda **kwargs: calls.append(kwargs))
    backup, actual = parallel._shared_backup()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: backup(prefix=f"paper3/fixture/{i}/"), range(2)))
    assert actual is client and created == [True]
    assert all(call["client"] is client and call["bucket"] == "SOFTWARE_TEST_DOUBLE_BUCKET" for call in calls)
    assert len({call["prefix"] for call in calls}) == 2


def test_CLI_dryrun_has_no_lock_client_or_state_and_execute_respects_original_two_locks(fixture, monkeypatch):
    split = fixture.root / "split.json"; write(split, {"scope": "SOFTWARE_TEST_DOUBLE"})
    qbatch = fixture.root / "qualification"
    monkeypatch.setattr(parallel, "_inputs", lambda root: (fixture.selection, {}, fixture.qualification,
        fixture.root / "inventory.json", split, qbatch))
    proposal = {"plan_sha256": fixture.amendment["plan_sha256"]}
    monkeypatch.setattr(parallel, "build_operational_amendment", lambda *a, **k: proposal)
    monkeypatch.setattr(parallel, "_shared_backup", lambda: pytest.fail("lock failure initialized client"))
    lock_path = fixture.root / "audit/local_preparation/.coordinator.lock"
    dry = parallel.coordinate(root=fixture.root)
    assert dry["status"] == "DRY_RUN_NO_NETWORK_NO_NEW_STATE" and not lock_path.exists()
    with pytest.raises(parallel.PreparationError, match="explicit_reviewed"):
        parallel.coordinate(root=fixture.root, execute=True)
    for lock in (lock_path, fixture.batch / ".batch.lock"):
        with parallel.serial.exclusive_lock(lock):
            with pytest.raises(parallel.PreparationError, match="already_active"):
                parallel.coordinate(root=fixture.root, execute=True, expected_operational_sha256=proposal["plan_sha256"])


def test_amendment_does_not_rewrite_original_scientific_or_serial_plan(fixture):
    prior = {"plan_sha256": parallel.PINNED_PRIOR_AMENDMENT_SHA, "scope": "SOFTWARE_TEST_DOUBLE"}
    for name, value in (("plans/stieger_meta_duration_selection.json", fixture.selection),
                        ("plans/stieger_meta_duration_execution_amendment.json", prior), ("split.json", {"split": "SOFTWARE_TEST_DOUBLE"})):
        write(fixture.root / name, value)
    for name in (*parallel.POLICY_FILES, *parallel.CODE_FILES, "tools/source_preparation_parallel.py"):
        path = fixture.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("SOFTWARE_TEST_DOUBLE_PROTOCOL")
    before = copy.deepcopy(fixture.selection), copy.deepcopy(prior)
    amendment = parallel.build_operational_amendment(fixture.root, fixture.selection, prior, split_path=fixture.root / "split.json")
    assert before == (fixture.selection, prior)
    assert amendment["kind"] == "concurrency_operational_only"
    assert amendment["prior_operational_execution_plan_sha256"] == parallel.PINNED_PRIOR_AMENDMENT_SHA
    assert amendment["max_jobs"] == 6 and amendment["max_download_bytes"] == fixture.selection["max_download_bytes"]
    assert amendment["max_wall_seconds"] == 64800 and amendment["final_protocol_freeze_sha256"] is None
    assert amendment["source_continuation"]["path"] == "tools/source_preparation_parallel.py"


def test_clock_rollback_with_inflight_worker_drains_and_persists_block_without_reset(fixture, monkeypatch):
    entered = threading.Event(); finished = []
    class StartedExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            assert entered.wait(timeout=2)
            return future
    monkeypatch.setattr(parallel, "ThreadPoolExecutor", StartedExecutor)
    def process(entry, **kwargs):
        entered.set(); time.sleep(.03)
        state = fixture.complete(entry); finished.append(entry["id"]); return state
    with pytest.raises(WallBudgetBlocked):
        fixture.run(process, monotonic=lambda: 99. if entered.is_set() else 100.)
    assert finished == [1] and fixture.client.closed
    progress = parallel.serial.read_json(fixture.batch / "batch_progress.json")
    assert progress["status"] == "INTERRUPTED_OR_FAILED_CLOSED"
    assert progress["operational_wall_budget"]["blocked"] is True
    assert progress["active_wall_elapsed_s"] >= 123
    journal = parallel.serial.read_json(fixture.batch / "parallel_wall_budget.json")
    assert journal["clock_block"] is not None
    with pytest.raises(WallBudgetBlocked):
        fixture.run(lambda *a, **k: pytest.fail("blocked clock resumed work"))
    assert parallel.serial.read_json(fixture.batch / "parallel_wall_budget.json")["clock_block"] == journal["clock_block"]


def test_real_SIGKILL_of_pure_metadata_fixture_charges_open_interval_on_resume(fixture):
    config = {"root": str(fixture.root), "selection": fixture.selection, "amendment": fixture.amendment,
              "qualification": fixture.qualification, "batch": str(fixture.batch)}
    config_path = fixture.root / "SOFTWARE_TEST_DOUBLE_child.json"; write(config_path, config)
    script = r'''
import json, sys, threading
from pathlib import Path
from tools import continue_local_preparation_parallel as p
cfg=json.loads(Path(sys.argv[1]).read_text());root=Path(cfg['root'])
p.base.reuse_qualification_object=lambda *a,**k:False
def process(entry,**kwargs):
    (root/'SOFTWARE_TEST_DOUBLE_started').write_text('pure metadata fixture, no EEG or network')
    threading.Event().wait(10)
p._run_locked(root=root,selection=cfg['selection'],amendment=cfg['amendment'],qualification=cfg['qualification'],
    inventory=root/'unused.json',batch_dir=Path(cfg['batch']),progress_path=root/'status/local_preparation.json',progress={},
    backup_factory=lambda:(lambda **k:None,None),process_fn=process,preflight=lambda *a:{})
'''
    child = subprocess.Popen([sys.executable, "-c", script, str(config_path)], cwd=parallel.REPO,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        limit = time.monotonic() + 5
        while not (fixture.root / "SOFTWARE_TEST_DOUBLE_started").exists() and time.monotonic() < limit:
            time.sleep(.01)
        assert (fixture.root / "SOFTWARE_TEST_DOUBLE_started").exists()
        child.kill(); child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
    assert child.returncode == -9
    before = parallel.serial.read_json(fixture.batch / "batch_progress.json")["active_wall_elapsed_s"]
    time.sleep(.05)
    fixture.run()
    journal = parallel.serial.read_json(fixture.batch / "parallel_wall_budget.json")
    assert journal["recovery_events"] and journal["durable_wall_seconds"] > before + .04


def test_tiny_proxy_native_metadata_is_byte_identical_to_serial_base_output(fixture, monkeypatch):
    entry = fixture.entries[0]
    raw = fixture.root / "data/stieger2021/raw" / entry["name"]
    raw.parent.mkdir(parents=True); raw.write_bytes(b"SOFTWARE_TEST_DOUBLE")
    class Native:
        rows = [{"trial_uid": "SOFTWARE_TEST_DOUBLE", "label": 0}]
        def __init__(self): self.eeg = bytearray(65536)
        def schema_audit(self):
            return {"participant_id": "S1", "session_id": "1", "pure_lr_trials": 1, "endpoint_errors": 0,
                    "source_file_sha256": hashlib.sha256(b"SOFTWARE_TEST_DOUBLE").hexdigest()}
    def backup(**kwargs):
        manifest = parallel.serial.read_json(kwargs["manifest_path"])
        result = {"status": "verified", "execute": True, "objects": [
            {**row, "status": "verified", "get": {"status": "verified", "complete_body_read": True,
                 "sha256": row["sha256"], "bytes_read": row["bytes"]}} for row in manifest["files"]]}
        write(kwargs["receipt_path"], result); return result
    monkeypatch.setattr(parallel.base, "check_space", lambda *a: None)
    native = lambda *a, **k: Native()
    for name, loader in (("serial", native), ("proxy", parallel.serialized_native_loader(threading.Lock(), native))):
        parallel.base.process_file(entry, root=fixture.root, batch_dir=fixture.root / name,
            plan=fixture.selection, backup_fn=backup, load_fn=loader)
    for name in ("adapter_audit.json", "trial_metadata.json"):
        assert (fixture.root / "serial/objects/1" / name).read_bytes() == (fixture.root / "proxy/objects/1" / name).read_bytes()


@pytest.mark.parametrize("failed_stage", ["window", "source", "delivery"])
def test_post_handover_window_or_child_failure_persists_closed_coordinator_status(fixture, monkeypatch, failed_stage):
    split = fixture.root / "split.json"; write(split, {"scope": "SOFTWARE_TEST_DOUBLE"})
    proposal = {"plan_sha256": fixture.amendment["plan_sha256"]}
    monkeypatch.setattr(parallel, "_inputs", lambda root: (fixture.selection, {}, fixture.qualification,
        fixture.root / "inventory.json", split, fixture.root / "qualification"))
    monkeypatch.setattr(parallel, "build_operational_amendment", lambda *a, **k: proposal)
    monkeypatch.setattr(parallel.serial, "count_verified", lambda *a: {"verified_complete_files": 62,
        "verified_schema_unknown_files": 0, "incomplete_files": 0})
    monkeypatch.setattr(parallel, "_run_locked", lambda **k: {"scope": "SOFTWARE_TEST_DOUBLE"})
    monkeypatch.setattr(parallel.base, "validate_plan", lambda *a: None)
    monkeypatch.setattr(parallel.base, "finish_durations", lambda *a: {"status": "WINDOW_INSUFFICIENT" if failed_stage == "window" else "WINDOW_FROZEN"})
    monkeypatch.setattr(parallel.serial, "freeze_input_contract", lambda **k: "f"*64)
    write(fixture.root / "status/tasks.json", {"W2": {"status": "COMPLETE", "scope": "SOFTWARE_TEST_DOUBLE"}})
    def invoke(*args, **kwargs):
        if kwargs["stage"].lower().startswith(failed_stage):
            raise parallel.PreparationError("fixture_child_failed")
    monkeypatch.setattr(parallel.serial, "invoke_cpu_script", invoke)
    with pytest.raises(parallel.PreparationError):
        parallel.coordinate(root=fixture.root, execute=True, expected_operational_sha256=proposal["plan_sha256"])
    saved = parallel.serial.read_json(fixture.root / "status/local_preparation.json")
    assert saved["status"] == "STOPPED_CLOSED_MINIMUM_GAP" and saved["final_protocol_freeze_sha256"] is None


def retain_partial(fixture, entry, *, owned=True):
    raw_path = f"data/stieger2021/w1-batch-temp/{entry['id']}/{entry['name']}"
    partial = fixture.root / (raw_path + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True); partial.write_bytes(b"partial")
    write(fixture.batch / "objects" / str(entry["id"]) / "state.json", {
        "plan_sha256": fixture.selection["plan_sha256"], "file_id": entry["id"], "name": entry["name"],
        "status": "FAILED", "raw_path": raw_path, "temporary_raw_owned": owned})
    return partial


def test_two_retained_partials_are_admitted_before_new_raw_without_third_slot(fixture):
    partials = {entry["id"]: retain_partial(fixture, entry) for entry in fixture.entries[-2:]}
    barrier = threading.Barrier(2); called = []
    def process(entry, **kwargs):
        called.append(entry["id"])
        if entry["id"] in partials:
            barrier.wait(timeout=2); partials[entry["id"]].unlink()
        return fixture.complete(entry)
    fixture.run(process)
    assert set(called[:2]) == {5, 6} and len(called) == 6


@pytest.mark.parametrize("violation", ["third_object", "unowned", "unknown_object"])
def test_unknown_unowned_or_third_legacy_temp_is_preserved_and_blocks_new_work(fixture, violation):
    if violation == "third_object":
        paths = [retain_partial(fixture, entry) for entry in fixture.entries[:3]]
    elif violation == "unowned":
        paths = [retain_partial(fixture, fixture.entries[0], owned=False)]
    else:
        path = fixture.root / "data/stieger2021/w1-batch-temp/999/unknown.partial"
        path.parent.mkdir(parents=True); path.write_bytes(b"unknown"); paths = [path]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(parallel.PreparationError, match="temporary|two_file"):
        fixture.run(lambda *a, **k: pytest.fail("invalid staging dispatched"))
    assert all(path.read_bytes() == body for path, body in before.items())
    assert not fixture.client.closed  # Client factory was never invoked.
